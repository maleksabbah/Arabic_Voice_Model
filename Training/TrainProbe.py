"""
w2v-BERT 2.0 Frozen Encoder Probe
==================================
Freezes the entire w2v-BERT 2.0 encoder (600M params).
Trains only a CTC head (single linear layer) on Arabic dialect data.
Evaluates on Casablanca (per-dialect WER), MGB-2 test, FLEURS.

This is a diagnostic: how much Arabic dialect knowledge is
already in the pretrained representations?

Usage:
    python Training/TrainProbe.py \
        --train_data /workspace/datasets/train_combined \
        --eval_casablanca /workspace/datasets/eval_casablanca_test \
        --eval_mgb2 /workspace/datasets/eval_mgb2_test \
        --eval_fleurs /workspace/datasets/eval_fleurs \
        --output_dir /workspace/probe_output \
        --epochs 10 \
        --batch_size 8 \
        --lr 1e-3
"""

import os
import argparse
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datasets import load_from_disk, Audio
from transformers import (
    Wav2Vec2BertModel,
    Wav2Vec2BertConfig,
    Wav2Vec2CTCTokenizer,
    SeamlessM4TFeatureExtractor,
    Trainer,
    TrainingArguments,
)
from torch import nn
import evaluate


# ============================================================
# MODEL: w2v-BERT 2.0 + CTC head
# ============================================================
class Wav2VecBertCTCProbe(nn.Module):
    """
    Frozen w2v-BERT 2.0 encoder with a trainable CTC linear head.
    Only the lm_head is trained — the encoder is completely frozen.
    """
    def __init__(self, model_name, vocab_size, pad_token_id=0, blank_id=0):
        super().__init__()
        self.encoder = Wav2Vec2BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Freeze the entire encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Trainable CTC head: single linear layer
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.pad_token_id = pad_token_id
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.4f}%)")

    def forward(self, input_features, attention_mask=None, labels=None):
        # Forward through frozen encoder
        with torch.no_grad():
            outputs = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
            )
        hidden_states = outputs.last_hidden_state

        # CTC head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # CTC loss
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # (T, B, C)

            # Input lengths = sequence lengths from encoder
            input_lengths = torch.full(
                (logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device
            )
            if attention_mask is not None:
                # Approximate output lengths from attention mask
                # w2v-BERT 2.0 downsamples by adapter, need to compute actual lengths
                input_lengths = self._get_output_lengths(attention_mask)

            # Label lengths (non-padding)
            label_mask = labels != -100
            label_lengths = label_mask.sum(dim=-1)

            # Replace -100 with 0 for CTC (it ignores via label_lengths)
            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = 0

            loss = self.ctc_loss(log_probs, labels_clean, input_lengths, label_lengths)

        return {"loss": loss, "logits": logits}

    def _get_output_lengths(self, attention_mask):
        """Compute output sequence lengths after encoder downsampling."""
        # facebook/w2v-bert-2.0 ships with add_adapter=False, so the encoder
        # keeps one output frame per input_features frame (the feature
        # extractor already did the 2x stacking). Only the adapter downsamples.
        input_lengths = attention_mask.sum(dim=-1)
        if self.encoder.config.add_adapter:
            for _ in range(self.encoder.config.num_adapter_layers):
                input_lengths = (input_lengths + 2 * (self.encoder.config.adapter_stride // 2) - self.encoder.config.adapter_kernel_size) // self.encoder.config.adapter_stride + 1
        return input_lengths.long()


# ============================================================
# VOCABULARY
# ============================================================
def build_vocab(dataset, text_col="normalized_text"):
    """Build character vocabulary from training data."""
    chars = set()
    # Read only the text column; iterating rows would decode every audio clip
    for text in dataset[text_col]:
        if text:
            chars.update(text)

    # Sort for reproducibility
    chars = sorted(chars)

    # Build vocab: 0 = <blank>/<pad>, then characters, then <unk>
    vocab = {"<pad>": 0}
    for i, c in enumerate(chars):
        vocab[c] = i + 1
    vocab["<unk>"] = len(vocab)
    vocab["|"] = len(vocab)  # word boundary / space

    # Map space to |
    if " " in vocab:
        space_id = vocab.pop(" ")
        # Remap | to space's id
        vocab["|"] = space_id

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Characters: {''.join(chars)}")
    return vocab


def save_vocab(vocab, path):
    """Save vocabulary to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab to {path}")


def text_to_labels(text, vocab):
    """Convert text to label sequence using vocab."""
    labels = []
    for c in text:
        if c == " ":
            labels.append(vocab.get("|", vocab.get("<unk>")))
        elif c in vocab:
            labels.append(vocab[c])
        else:
            labels.append(vocab.get("<unk>", 0))
    return labels


# ============================================================
# DATA COLLATOR
# ============================================================
@dataclass
class CTCDataCollator:
    """
    Collator for CTC training with w2v-BERT 2.0.
    Pads audio features and label sequences.
    """
    feature_extractor: SeamlessM4TFeatureExtractor
    vocab: dict
    max_audio_len: int = 160000  # 10 seconds at 16kHz

    def __call__(self, features):
        # Extract audio arrays
        audio_arrays = []
        texts = []
        for f in features:
            audio = f["audio"]
            if isinstance(audio, dict):
                array = audio["array"]
            else:
                array = audio
            # Truncate if too long
            if len(array) > self.max_audio_len:
                array = array[:self.max_audio_len]
            audio_arrays.append(array)
            texts.append(f.get("normalized_text", ""))

        # Process audio with feature extractor
        batch = self.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        # Encode labels
        label_sequences = [text_to_labels(t, self.vocab) for t in texts]
        max_label_len = max(len(l) for l in label_sequences)
        padded_labels = []
        for l in label_sequences:
            padded = l + [-100] * (max_label_len - len(l))
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ============================================================
# METRICS
# ============================================================
def compute_wer(pred_ids, label_ids, vocab):
    """Compute WER from predicted and label IDs."""
    wer_metric = evaluate.load("wer")

    # Reverse vocab
    id_to_char = {v: k for k, v in vocab.items()}

    def decode(ids):
        # CTC decode: collapse repeated, remove blanks
        decoded = []
        prev = None
        for i in ids:
            if i == 0:  # blank
                prev = None
                continue
            if i == prev:
                continue
            if i in id_to_char:
                c = id_to_char[i]
                if c == "|":
                    c = " "
                decoded.append(c)
            prev = i
        return "".join(decoded).strip()

    predictions = [decode(p) for p in pred_ids]
    references = [decode(l[l != -100].tolist()) if isinstance(l, torch.Tensor)
                   else decode([x for x in l if x != -100])
                   for l in label_ids]

    # Filter empty
    pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not pairs:
        return {"wer": 1.0}

    preds, refs = zip(*pairs)
    wer = wer_metric.compute(predictions=list(preds), references=list(refs))
    return {"wer": wer}


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, feature_extractor, dataset, vocab, batch_size=8, name="eval", device="cuda"):
    """Run evaluation and return WER."""
    model.eval()
    collator = CTCDataCollator(feature_extractor=feature_extractor, vocab=vocab)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"]

            outputs = model(input_features=input_features, attention_mask=attention_mask)
            logits = outputs["logits"]
            pred_ids = torch.argmax(logits, dim=-1).cpu()

            all_preds.extend(pred_ids.tolist())
            all_labels.extend(labels.tolist())

    results = compute_wer(all_preds, all_labels, vocab)
    print(f"  {name}: WER = {results['wer']:.4f} ({results['wer']*100:.1f}%)")
    return results


def evaluate_casablanca_per_dialect(model, feature_extractor, dataset, vocab, batch_size=8, device="cuda"):
    """Evaluate Casablanca with per-dialect WER breakdown."""
    if "dialect" not in dataset.column_names:
        return evaluate_model(model, feature_extractor, dataset, vocab, batch_size, "Casablanca", device)

    dialects = set(dataset["dialect"])
    results = {}

    for dialect in sorted(dialects):
        subset = dataset.filter(lambda x: x["dialect"] == dialect)
        r = evaluate_model(model, feature_extractor, subset, vocab, batch_size, f"Casablanca/{dialect}", device)
        results[dialect] = r["wer"]

    avg_wer = np.mean(list(results.values()))
    print(f"  Casablanca average: WER = {avg_wer:.4f} ({avg_wer*100:.1f}%)")
    results["average"] = avg_wer
    return results


# ============================================================
# CUSTOM TRAINER
# ============================================================
class CTCTrainer(Trainer):
    """Custom trainer that handles our model's forward signature."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        input_features = inputs.get("input_features")
        attention_mask = inputs.get("attention_mask")

        outputs = model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="w2v-BERT 2.0 Frozen Probe")
    parser.add_argument("--model_name", default="facebook/w2v-bert-2.0")
    parser.add_argument("--train_data", default="/workspace/datasets/train_combined")
    parser.add_argument("--eval_casablanca", default="/workspace/datasets/eval_casablanca_test")
    parser.add_argument("--eval_mgb2", default="/workspace/datasets/eval_mgb2_test")
    parser.add_argument("--eval_fleurs", default="/workspace/datasets/eval_fleurs")
    parser.add_argument("--output_dir", default="/workspace/probe_output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_audio_len", type=int, default=160000, help="Max audio samples (10s=160000)")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")

    # ---- Load training data ----
    print("\nLoading training data...")
    train_dataset = load_from_disk(args.train_data)
    print(f"Training samples: {len(train_dataset)}")

    # Ensure audio is cast to 16kHz
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

    # ---- Build vocabulary ----
    print("\nBuilding vocabulary...")
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        print(f"Loaded existing vocab: {len(vocab)} tokens")
    else:
        vocab = build_vocab(train_dataset)
        os.makedirs(args.output_dir, exist_ok=True)
        save_vocab(vocab, vocab_path)

    # ---- Load feature extractor ----
    print("\nLoading feature extractor...")
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.model_name)

    # ---- Build model ----
    print("\nLoading model...")
    model = Wav2VecBertCTCProbe(
        model_name=args.model_name,
        vocab_size=len(vocab),
        pad_token_id=vocab["<pad>"],
        blank_id=vocab["<pad>"],
    )
    model = model.to(device)

    # ---- Data collator ----
    collator = CTCDataCollator(
        feature_extractor=feature_extractor,
        vocab=vocab,
        max_audio_len=args.max_audio_len,
    )

    if not args.eval_only:
        # ---- Training ----
        print("\nStarting training...")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=50,
            save_steps=1000,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="none",
            gradient_accumulation_steps=1,
        )

        trainer = CTCTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )

        if args.checkpoint:
            trainer.train(resume_from_checkpoint=args.checkpoint)
        else:
            trainer.train()

        # Save final model
        torch.save(model.lm_head.state_dict(), os.path.join(args.output_dir, "ctc_head.pt"))
        save_vocab(vocab, vocab_path)
        print(f"\nSaved CTC head to {args.output_dir}/ctc_head.pt")

    # ---- Evaluation ----
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    all_results = {}

    # Casablanca (per-dialect)
    if os.path.exists(args.eval_casablanca):
        print("\nCasablanca (per-dialect):")
        casa_dataset = load_from_disk(args.eval_casablanca)
        casa_dataset = casa_dataset.cast_column("audio", Audio(sampling_rate=16000))
        casa_results = evaluate_casablanca_per_dialect(
            model, feature_extractor, casa_dataset, vocab, args.batch_size, device
        )
        all_results["casablanca"] = casa_results
    else:
        print(f"\nCasablanca not found at {args.eval_casablanca}")

    # MGB-2 test
    if os.path.exists(args.eval_mgb2):
        print("\nMGB-2 test:")
        mgb2_dataset = load_from_disk(args.eval_mgb2)
        mgb2_dataset = mgb2_dataset.cast_column("audio", Audio(sampling_rate=16000))
        mgb2_results = evaluate_model(
            model, feature_extractor, mgb2_dataset, vocab, args.batch_size, "MGB-2", device
        )
        all_results["mgb2"] = mgb2_results
    else:
        print(f"\nMGB-2 test not found at {args.eval_mgb2}")

    # FLEURS
    if os.path.exists(args.eval_fleurs):
        print("\nFLEURS Arabic:")
        fleurs_dataset = load_from_disk(args.eval_fleurs)
        fleurs_dataset = fleurs_dataset.cast_column("audio", Audio(sampling_rate=16000))
        fleurs_results = evaluate_model(
            model, feature_extractor, fleurs_dataset, vocab, args.batch_size, "FLEURS", device
        )
        all_results["fleurs"] = fleurs_results
    else:
        print(f"\nFLEURS not found at {args.eval_fleurs}")

    # ---- Summary ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    results_path = os.path.join(args.output_dir, "probe_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    for benchmark, result in all_results.items():
        if isinstance(result, dict) and "wer" in result:
            print(f"  {benchmark}: {result['wer']*100:.1f}% WER")
        elif isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, float):
                    print(f"  {benchmark}/{k}: {v*100:.1f}% WER")


if __name__ == "__main__":
    main()
