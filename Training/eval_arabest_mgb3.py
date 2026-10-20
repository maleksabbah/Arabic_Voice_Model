"""
Evaluate AraBEST-RQ fine-tuned model on MGB-3 (Egyptian Arabic).
Run on RunPod where the model checkpoint lives.

Usage:
  python eval_arabest_mgb3.py \
    --model_dir /workspace/arabest_model \
    --yaml_config /workspace/AraBEST-RQ/model/hparams/hybrid_300M.yaml \
    --checkpoint /workspace/finetune_output_arabest/best \
    --max_samples 500
"""

import os, argparse, re, json, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Audio
from tqdm import tqdm
import evaluate
from hyperpyyaml import load_hyperpyyaml


# ============================================================
# UTILS (same as finetune_arabest.py)
# ============================================================

ARABIC_DIACRITICS = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]')

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


# ============================================================
# MODEL (identical to finetune_arabest.py)
# ============================================================

class ArabEstCTCFinetune(nn.Module):
    def __init__(self, model_dir, yaml_config, vocab_size, pad_token_id=0,
                 device="cpu", unfreeze_layers=None, use_grad_checkpoint=False):
        super().__init__()

        with open(yaml_config, encoding="utf-8") as f:
            hparams = load_hyperpyyaml(f)

        self.compute_features = hparams["compute_features"]
        self.normalize = hparams["normalize"]

        model_list = hparams["model"]
        self.cnn = model_list[0]
        self.wrapper = model_list[1]
        self.transformer = hparams["Transformer"]

        # Load pretrained weights
        model_ckpt_path = os.path.join(model_dir, "model.ckpt")
        norm_ckpt_path = os.path.join(model_dir, "normalizer.ckpt")

        if not os.path.exists(model_ckpt_path):
            for root, dirs, files in os.walk(model_dir):
                if "model.ckpt" in files:
                    model_ckpt_path = os.path.join(root, "model.ckpt")
                    norm_ckpt_path = os.path.join(root, "normalizer.ckpt")
                    break

        state_dict = torch.load(model_ckpt_path, map_location="cpu", weights_only=True)
        model_list.load_state_dict(state_dict)

        if os.path.exists(norm_ckpt_path):
            norm_state = torch.load(norm_ckpt_path, map_location="cpu", weights_only=True)
            self.normalize.load_state_dict(norm_state)
            self.normalize.eval()

        self.transformer.encoder.output_hidden_states = True
        self.transformer.output_hidden_states = True

        self.unfreeze_layers = unfreeze_layers or [17, 18, 19]
        self.unfreeze_indices = [l - 1 for l in self.unfreeze_layers]

        # Freeze everything (eval mode)
        for param in self.parameters():
            param.requires_grad = False

        # CTC head
        total_layers = hparams["num_encoder_layers"]
        hidden_size = hparams["d_model"]
        self.layer_weights = nn.Parameter(torch.zeros(total_layers))
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=pad_token_id, reduction="mean", zero_infinity=True)

    def forward(self, audio, audio_lengths, labels=None):
        with torch.no_grad():
            feats = self.compute_features(audio)
            feats = self.normalize(feats, audio_lengths, epoch=0)

            B, T, C = feats.shape
            pad_needed = (4 - T % 4) % 4
            if pad_needed > 0:
                feats = torch.nn.functional.pad(feats, [0, 0, 0, pad_needed])

            src = self.cnn(feats)

        result = self.wrapper(src, audio_lengths)
        enc_out, hidden_states = result

        layer_outputs = torch.stack(hidden_states[1:], dim=0)
        weights = torch.softmax(self.layer_weights, dim=0)
        weights = weights.view(-1, 1, 1, 1)
        blended = (weights * layer_outputs).sum(dim=0)

        logits = self.lm_head(blended)

        loss = None
        if labels is not None:
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            output_T = logits.shape[1]
            input_lengths = (audio_lengths * output_T).long().clamp(min=1, max=output_T)
            label_mask = labels != -100
            label_lengths = label_mask.sum(dim=-1)
            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = 0
            loss = self.ctc_loss(log_probs, labels_clean, input_lengths, label_lengths)

        return {"loss": loss, "logits": logits}


# ============================================================
# MGB-3 DATASET
# ============================================================

class MGB3Dataset(Dataset):
    def __init__(self, hf_dataset, vocab, max_audio_len=480000):
        self.dataset = hf_dataset
        self.vocab = vocab
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]
        array = audio["array"] if isinstance(audio, dict) else audio
        array = np.array(array, dtype=np.float32)
        if len(array) > self.max_audio_len:
            array = array[:self.max_audio_len]
        text = normalize_arabic(item["sentence"])
        labels = text_to_labels(text, self.vocab)
        return {"audio": array, "labels": labels, "reference": text}


def collate_fn(batch):
    audios = [torch.from_numpy(item["audio"]) for item in batch]
    labels_list = [item["labels"] for item in batch]
    refs = [item["reference"] for item in batch]

    max_audio = max(a.shape[0] for a in audios)
    padded_audio = torch.zeros(len(audios), max_audio)
    audio_lengths = torch.zeros(len(audios))
    for i, a in enumerate(audios):
        padded_audio[i, :a.shape[0]] = a
        audio_lengths[i] = a.shape[0] / max_audio

    max_label = max(len(l) for l in labels_list)
    padded_labels = torch.full((len(labels_list), max_label), -100, dtype=torch.long)
    for i, l in enumerate(labels_list):
        padded_labels[i, :len(l)] = torch.tensor(l, dtype=torch.long)

    return {
        "audio": padded_audio,
        "audio_lengths": audio_lengths,
        "labels": padded_labels,
        "references": refs,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="AraBEST pretrained model dir")
    parser.add_argument("--yaml_config", required=True, help="SpeechBrain YAML config")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint dir (with vocab.json, layer_weights.pt, lm_head.pt, encoder_layers.pt)")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load vocab from checkpoint
    vocab_path = os.path.join(args.checkpoint, "vocab.json")
    print(f"Loading vocab: {vocab_path}")
    with open(vocab_path) as f:
        vocab = json.load(f)
    print(f"  Vocab size: {len(vocab)}")
    id_to_char = {v: k for k, v in vocab.items()}

    # Build model
    print("Loading AraBEST model...")
    model = ArabEstCTCFinetune(
        model_dir=args.model_dir,
        yaml_config=args.yaml_config,
        vocab_size=len(vocab),
        device=device,
    )

    # Load fine-tuned weights
    print(f"Loading checkpoint: {args.checkpoint}")
    layer_weights = torch.load(os.path.join(args.checkpoint, "layer_weights.pt"), map_location="cpu")
    lm_head = torch.load(os.path.join(args.checkpoint, "lm_head.pt"), map_location="cpu")
    model.layer_weights.data = layer_weights
    model.lm_head.load_state_dict(lm_head)

    # Load unfrozen encoder layers if they exist
    encoder_path = os.path.join(args.checkpoint, "encoder_layers.pt")
    if os.path.exists(encoder_path):
        print("  Loading fine-tuned encoder layers...")
        encoder_state = torch.load(encoder_path, map_location="cpu")
        encoder_layers = model.transformer.encoder.layers if hasattr(model.transformer.encoder, 'layers') else None
        if encoder_layers is None:
            for name, module in model.transformer.encoder.named_children():
                if isinstance(module, nn.ModuleList):
                    encoder_layers = module
                    break
        if encoder_layers is not None:
            for idx_str, state in encoder_state.items():
                idx = int(idx_str)
                if idx < len(encoder_layers):
                    encoder_layers[idx].load_state_dict(state)
                    print(f"    Loaded encoder layer {idx + 1}")

    model = model.to(device)
    model.eval()

    # Print layer weight distribution
    weights = torch.softmax(model.layer_weights, dim=0).detach().cpu().numpy()
    top5 = np.argsort(weights)[::-1][:5]
    print("\n  Layer weight distribution (top 5):")
    for i in top5:
        print(f"    L{i+1}: {weights[i]:.4f}")

    # Load MGB-3
    print("\nLoading MGB-3 dataset...")
    mgb3 = load_dataset("MightyStudent/Egyptian-ASR-MGB-3", split="train", trust_remote_code=True)
    mgb3 = mgb3.cast_column("audio", Audio(sampling_rate=16000))

    # Shuffle and limit
    indices = list(range(len(mgb3)))
    np.random.seed(42)
    np.random.shuffle(indices)
    n_samples = min(args.max_samples, len(mgb3))
    mgb3 = mgb3.select(indices[:n_samples])
    print(f"  Using {n_samples} samples")

    # Filter empty refs
    mgb3 = mgb3.filter(lambda x: bool(normalize_arabic(x["sentence"]).strip()))
    print(f"  After filtering empty: {len(mgb3)} samples")

    # Build dataloader
    ds = MGB3Dataset(mgb3, vocab)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    # Evaluate
    wer_metric = evaluate.load("wer")
    all_preds, all_refs = [], []
    per_sample_wers = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="MGB-3"):
            audio = batch["audio"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)

            outputs = model(audio=audio, audio_lengths=audio_lengths)
            pred_ids = torch.argmax(outputs["logits"], dim=-1).cpu()

            for pred, ref in zip(pred_ids.tolist(), batch["references"]):
                pred_text = ctc_decode(pred, id_to_char)
                if ref.strip():
                    all_preds.append(pred_text)
                    all_refs.append(ref)

    # Compute WER
    corpus_wer = wer_metric.compute(predictions=all_preds, references=all_refs)

    # Also compute per-sample average (like your Whisper benchmarks)
    from jiwer import wer as jiwer_wer
    per_sample = []
    for p, r in zip(all_preds, all_refs):
        try:
            per_sample.append(jiwer_wer(r, p))
        except:
            per_sample.append(1.0)
    avg_wer = np.mean(per_sample)

    print(f"\n{'='*60}")
    print(f"  MGB-3 (Egyptian Arabic) Results")
    print(f"{'='*60}")
    print(f"  Corpus WER:      {corpus_wer * 100:.1f}%")
    print(f"  Per-sample avg:  {avg_wer * 100:.1f}%")
    print(f"  Samples:         {len(all_refs)}")
    print(f"{'='*60}")

    # Show some examples
    print("\n  Sample predictions:")
    sample_indices = list(range(len(all_refs)))
    random.seed(42)
    random.shuffle(sample_indices)
    for i in sample_indices[:10]:
        print(f"    REF: {all_refs[i][:80]}")
        print(f"    PRD: {all_preds[i][:80]}")
        w = per_sample[i]
        print(f"    WER: {w:.2f}")
        print()

    # Save results
    results = {
        "benchmark": "MGB-3",
        "corpus_wer": float(corpus_wer),
        "per_sample_avg_wer": float(avg_wer),
        "num_samples": len(all_refs),
        "checkpoint": args.checkpoint,
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), "mgb3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
