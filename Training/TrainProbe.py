"""
w2v-BERT 2.0 Frozen Encoder Probe v3 — Weighted Layer Sum
Learns a weighted combination of ALL encoder layers, not just the last one.
CTC linear head on frozen encoder representations.
"""

import os, argparse, json, re, time, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from datasets import load_from_disk, Audio
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from tqdm import tqdm
import evaluate

# ============================================================
# UTILS
# ============================================================

ARABIC_DIACRITICS = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]')

def normalize_arabic(text):
    if not text or not isinstance(text, str):
        return ""
    text = ARABIC_DIACRITICS.sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = text.replace('ة', 'ه').replace('ى', 'ي').replace('ـ', '')
    text = re.sub(r'[^ء-ي٠-٩\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_to_labels(text, vocab):
    labels = []
    for c in text:
        if c == " ":
            labels.append(vocab.get("|", vocab.get("<unk>")))
        elif c in vocab:
            labels.append(vocab[c])
        else:
            labels.append(vocab.get("<unk>", 0))
    return labels


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================

def build_vocab(dataset, text_col="normalized_text"):
    chars = set()
    for item in dataset:
        text = item.get(text_col, "")
        if text:
            chars.update(text)
    chars = sorted(chars)
    vocab = {"<pad>": 0}
    for i, c in enumerate(chars):
        vocab[c] = i + 1
    vocab["<unk>"] = len(vocab)
    vocab["|"] = len(vocab)
    if " " in vocab:
        space_id = vocab.pop(" ")
        vocab["|"] = space_id
    return vocab


def filter_dataset(dataset, min_seconds=0.5, max_seconds=30.0, samples_per_frame=320):
    min_samples = int(min_seconds * 16000)
    max_samples = int(max_seconds * 16000)

    def is_valid(sample):
        audio = sample["audio"]
        array = audio["array"] if isinstance(audio, dict) else audio
        if len(array) < min_samples:
            return False
        if len(array) > max_samples:
            return False
        text = sample.get("normalized_text", "")
        if not text or not text.strip():
            return False
        if np.any(np.isnan(array)) or np.any(np.isinf(array)):
            return False
        encoder_frames = len(array) // samples_per_frame
        if len(text) > encoder_frames:
            return False
        return True

    before = len(dataset)
    dataset = dataset.filter(is_valid)
    after = len(dataset)
    removed = before - after
    if removed > 0:
        print(f"  Filtered: {before} -> {after} ({removed} removed, {removed/before*100:.1f}%)")
    return dataset


@dataclass
class CTCDataCollator:
    feature_extractor: SeamlessM4TFeatureExtractor
    vocab: dict
    max_audio_len: int = 480000

    def __call__(self, features):
        audio_arrays = []
        texts = []
        for f in features:
            audio = f["audio"]
            array = audio["array"] if isinstance(audio, dict) else audio
            if len(array) > self.max_audio_len:
                array = array[:self.max_audio_len]
            audio_arrays.append(np.array(array, dtype=np.float32))
            texts.append(f.get("normalized_text", ""))

        batch = self.feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True,
        )

        label_sequences = [text_to_labels(t, self.vocab) for t in texts]
        max_label_len = max(len(l) for l in label_sequences) if label_sequences else 0
        padded_labels = [l + [-100] * (max_label_len - len(l)) for l in label_sequences]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ============================================================
# MODEL
# ============================================================

class Wav2VecBertCTCProbe(nn.Module):
    def __init__(self, model_name, vocab_size, pad_token_id=0):
        super().__init__()
        self.encoder = Wav2Vec2BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        num_layers = self.encoder.config.num_hidden_layers  # 24 for w2v-BERT 2.0

        for param in self.encoder.parameters():
            param.requires_grad = False

        # Force encoder into eval mode permanently — no dropout, no SpecAugment
        self.encoder.eval()
        self.encoder.config.apply_spec_augment = False

        # Learnable layer weights — one scalar per encoder layer
        # Initialized to zero so softmax gives uniform 1/num_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=pad_token_id, reduction="mean", zero_infinity=True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Params: {total:,} total, {trainable:,} trainable ({100*trainable/total:.4f}%)")
        print(f"  Layer weights: {num_layers} layers")

    def train(self, mode=True):
        # Override train() so encoder ALWAYS stays in eval mode
        super().train(mode)
        self.encoder.eval()
        return self

    def get_layer_weight_distribution(self):
        """Return the softmax-normalized layer weights for inspection."""
        return torch.softmax(self.layer_weights, dim=0).detach().cpu().numpy()

    def forward(self, input_features, attention_mask=None, labels=None):
        with torch.no_grad():
            outputs = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,  # return all layer outputs
            )

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is the embedding layer output, 1..24 are the transformer layers
        # We skip index 0 (raw embedding) and use only transformer layer outputs
        hidden_states = torch.stack(outputs.hidden_states[1:], dim=0)  # (num_layers, batch, frames, 1024)

        # Weighted sum across layers
        weights = torch.softmax(self.layer_weights, dim=0)  # (num_layers,)
        # Reshape for broadcasting: (num_layers, 1, 1, 1)
        weights = weights.view(-1, 1, 1, 1)
        # Weighted sum: (batch, frames, 1024)
        blended = (weights * hidden_states).sum(dim=0)

        logits = self.lm_head(blended)

        loss = None
        if labels is not None:
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            if attention_mask is not None:
                input_lengths = attention_mask.sum(dim=-1).long()
            else:
                input_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device)

            label_mask = labels != -100
            label_lengths = label_mask.sum(dim=-1)

            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = 0

            loss = self.ctc_loss(log_probs, labels_clean, input_lengths, label_lengths)

        return {"loss": loss, "logits": logits}


# ============================================================
# TRAIN
# ============================================================

def train(model, train_loader, eval_fn, args, device):
    # Optimize BOTH lm_head AND layer_weights
    trainable_params = list(model.lm_head.parameters()) + [model.layer_weights]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # Warmup + cosine decay schedule
    total_steps = min(args.max_steps, len(train_loader) * args.epochs)
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_wer = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()  # encoder stays in eval via override
        epoch_losses = []
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in bar:
            input_features = batch["input_features"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # FORWARD
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_features=input_features, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"]
            else:
                outputs = model(input_features=input_features, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]

            # BACKWARD
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            scheduler.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

            # LOG
            if global_step % args.log_every == 0:
                avg_loss = np.mean(epoch_losses[-100:])
                lr = scheduler.get_last_lr()[0]
                bar.set_postfix({"loss": f"{loss_val:.3f}", "avg": f"{avg_loss:.3f}", "lr": f"{lr:.1e}", "step": global_step})

            # PRINT LAYER WEIGHTS periodically
            if global_step % (args.save_every // 2) == 0:
                w = model.get_layer_weight_distribution()
                top_3 = np.argsort(w)[-3:][::-1]
                top_str = ", ".join([f"L{i+1}={w[i]:.3f}" for i in top_3])
                print(f"  Layer weights (step {global_step}): top 3 = {top_str}")

            # SAVE + MID-TRAINING EVAL
            if global_step % args.save_every == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    "lm_head": model.lm_head.state_dict(),
                    "layer_weights": model.layer_weights.data,
                }, os.path.join(save_dir, "probe_head.pt"))

                # Quick eval to track progress
                if eval_fn:
                    mid_wer = eval_fn()
                    if mid_wer < best_wer:
                        best_wer = mid_wer
                        best_dir = os.path.join(args.output_dir, "best")
                        os.makedirs(best_dir, exist_ok=True)
                        torch.save({
                            "lm_head": model.lm_head.state_dict(),
                            "layer_weights": model.layer_weights.data,
                        }, os.path.join(best_dir, "probe_head.pt"))
                        print(f"  *** New best WER: {best_wer*100:.1f}% ***")

            # STOP
            if global_step >= args.max_steps:
                break

        print(f"  Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}")

        if global_step >= args.max_steps:
            break

    # Save final
    torch.save({
        "lm_head": model.lm_head.state_dict(),
        "layer_weights": model.layer_weights.data,
    }, os.path.join(args.output_dir, "probe_head.pt"))

    # Print final layer weight distribution
    w = model.get_layer_weight_distribution()
    print(f"\n  Final layer weights:")
    for i, wi in enumerate(w):
        bar_str = "█" * int(wi * 100)
        print(f"    Layer {i+1:2d}: {wi:.4f} {bar_str}")
    top_3 = np.argsort(w)[-3:][::-1]
    print(f"  Top layers: {', '.join([f'L{i+1} ({w[i]:.3f})' for i in top_3])}")

    elapsed = time.time() - start_time
    print(f"  Training done: {global_step} steps, {elapsed/60:.1f} min, best WER: {best_wer*100:.1f}%")
    return global_step, best_wer


# ============================================================
# EVAL
# ============================================================

def ctc_decode(ids, id_to_char):
    decoded = []
    prev = None
    for i in ids:
        if i == 0:
            prev = None
            continue
        if i == prev:
            continue
        c = id_to_char.get(i, "")
        if c == "|":
            c = " "
        if c:
            decoded.append(c)
        prev = i
    return "".join(decoded).strip()


def evaluate_dataset(model, feature_extractor, dataset, vocab, batch_size=8, name="eval", device="cuda", print_samples=3):
    model.eval()  # encoder stays in eval via override
    wer_metric = evaluate.load("wer")
    id_to_char = {v: k for k, v in vocab.items()}
    collator = CTCDataCollator(feature_extractor=feature_extractor, vocab=vocab)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)

    all_preds, all_refs = [], []

    with torch.no_grad():
        for batch in loader:
            input_features = batch["input_features"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_features=input_features, attention_mask=attention_mask)
            pred_ids = torch.argmax(outputs["logits"], dim=-1).cpu()

            for pred, label in zip(pred_ids.tolist(), batch["labels"].tolist()):
                pred_text = ctc_decode(pred, id_to_char)
                ref_ids = [x for x in label if x != -100]
                ref_text = ctc_decode(ref_ids, id_to_char)
                if ref_text.strip():
                    all_preds.append(pred_text)
                    all_refs.append(ref_text)

    wer_score = wer_metric.compute(predictions=all_preds, references=all_refs) if all_refs else 1.0
    print(f"  {name}: WER={wer_score*100:.1f}% ({len(all_refs)} samples)")

    # Print sample predictions so we can see what's actually happening
    if print_samples > 0 and all_refs:
        print(f"  Sample predictions ({name}):")
        indices = list(range(len(all_refs)))
        random.shuffle(indices)
        for i in indices[:print_samples]:
            print(f"    REF: {all_refs[i][:80]}")
            print(f"    PRD: {all_preds[i][:80]}")
            print()

    return {"wer": wer_score, "samples": len(all_refs)}


def evaluate_casablanca(model, feature_extractor, dataset, vocab, batch_size=8, device="cuda"):
    if "dialect" not in dataset.column_names:
        return evaluate_dataset(model, feature_extractor, dataset, vocab, batch_size, "Casablanca", device)

    results = {}
    for dialect in sorted(set(dataset["dialect"])):
        subset = dataset.filter(lambda x: x["dialect"] == dialect)
        r = evaluate_dataset(model, feature_extractor, subset, vocab, batch_size, f"Casablanca/{dialect}", device, print_samples=1)
        results[dialect] = r["wer"]

    avg = np.mean(list(results.values()))
    print(f"  Casablanca avg: WER={avg*100:.1f}%")
    results["average"] = avg
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/w2v-bert-2.0")
    parser.add_argument("--train_data", default="/workspace/datasets/train_combined")
    parser.add_argument("--eval_casablanca", default="/workspace/datasets/eval_casablanca_test")
    parser.add_argument("--eval_mgb2", default="/workspace/datasets/eval_mgb2_test")
    parser.add_argument("--eval_fleurs", default="/workspace/datasets/eval_fleurs")
    parser.add_argument("--output_dir", default="/workspace/probe_output_v3")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. DATA ----
    print("Loading data...")
    train_dataset = load_from_disk(args.train_data)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  Raw: {len(train_dataset)} samples")

    print("Filtering...")
    train_dataset = filter_dataset(train_dataset)

    print("Building vocab...")
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
    else:
        vocab = build_vocab(train_dataset)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  Vocab: {len(vocab)} tokens")

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.model_name)
    collator = CTCDataCollator(feature_extractor=feature_extractor, vocab=vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=2, pin_memory=True,
    )

    # ---- 2. MODEL ----
    print("Loading model...")
    model = Wav2VecBertCTCProbe(model_name=args.model_name, vocab_size=len(vocab)).to(device)

    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(os.path.join(args.checkpoint, "probe_head.pt"), map_location=device)
        model.lm_head.load_state_dict(ckpt["lm_head"])
        model.layer_weights.data = ckpt["layer_weights"].to(device)

    # ---- 3. TRAIN ----
    if not args.eval_only:
        # Create a quick eval function for mid-training checks using FLEURS (small, fast)
        eval_fn = None
        if os.path.exists(args.eval_fleurs):
            fleurs_ds = load_from_disk(args.eval_fleurs).cast_column("audio", Audio(sampling_rate=16000))
            fleurs_ds = filter_dataset(fleurs_ds)
            def quick_eval():
                r = evaluate_dataset(model, feature_extractor, fleurs_ds, vocab, args.batch_size, "FLEURS (mid-train)", device, print_samples=2)
                return r["wer"]
            eval_fn = quick_eval

        train(model, train_loader, eval_fn, args, device)

    # ---- 4. EVAL ----
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    results = {}

    if os.path.exists(args.eval_casablanca):
        ds = load_from_disk(args.eval_casablanca).cast_column("audio", Audio(sampling_rate=16000))
        ds = filter_dataset(ds)
        results["casablanca"] = evaluate_casablanca(model, feature_extractor, ds, vocab, args.batch_size, device)

    if os.path.exists(args.eval_mgb2):
        ds = load_from_disk(args.eval_mgb2).cast_column("audio", Audio(sampling_rate=16000))
        ds = filter_dataset(ds)
        results["mgb2"] = evaluate_dataset(model, feature_extractor, ds, vocab, args.batch_size, "MGB-2", device)

    if os.path.exists(args.eval_fleurs):
        ds = load_from_disk(args.eval_fleurs).cast_column("audio", Audio(sampling_rate=16000))
        ds = filter_dataset(ds)
        results["fleurs"] = evaluate_dataset(model, feature_extractor, ds, vocab, args.batch_size, "FLEURS", device)

    # Save layer weights to results
    w = model.get_layer_weight_distribution()
    results["layer_weights"] = {f"layer_{i+1}": float(wi) for i, wi in enumerate(w)}

    # ---- 5. SAVE ----
    with open(os.path.join(args.output_dir, "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, r in results.items():
        if name == "layer_weights":
            continue
        if isinstance(r, dict) and "wer" in r:
            print(f"  {name}: {r['wer']*100:.1f}% WER")
        elif isinstance(r, dict):
            for k, v in r.items():
                if isinstance(v, float):
                    print(f"  {name}/{k}: {v*100:.1f}% WER")
    print("="*60)


if __name__ == "__main__":
    main()
