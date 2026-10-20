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

# Model + text helpers come from the trainer so the checkpoint layout always matches
from TrainingAraBEST import ArabEstCTCFinetune, normalize_arabic, text_to_labels, ctc_decode


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
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint dir (with finetune_ckpt.pt)")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # vocab.json lives in the training output_dir, one level above the checkpoint
    vocab_path = os.path.join(args.checkpoint, "vocab.json")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(os.path.dirname(os.path.normpath(args.checkpoint)), "vocab.json")
    print(f"Loading vocab: {vocab_path}")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"  Vocab size: {len(vocab)}")
    id_to_char = {v: k for k, v in vocab.items()}

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(os.path.join(args.checkpoint, "finetune_ckpt.pt"), map_location="cpu")

    print("Loading AraBEST model...")
    model = ArabEstCTCFinetune(
        model_dir=args.model_dir,
        yaml_config=args.yaml_config,
        vocab_size=len(vocab),
        device=device,
        unfreeze_layers=ckpt.get("unfreeze_layers"),
        use_grad_checkpoint=False,
    )
    model.lm_head.load_state_dict(ckpt["lm_head"])
    model.layer_weights.data = ckpt["layer_weights"]

    # Same "layer_{idx}.{param}" layout _save_checkpoint writes
    encoder_layers = model._get_encoder_layers()
    for key, value in ckpt.get("encoder_layers", {}).items():
        layer_name, param_name = key.split(".", 1)
        layer_idx = int(layer_name.replace("layer_", ""))
        layer_state = encoder_layers[layer_idx].state_dict()
        layer_state[param_name] = value
        encoder_layers[layer_idx].load_state_dict(layer_state)
    print(f"  Restored {len(ckpt.get('encoder_layers', {}))} encoder params "
          f"(L{','.join(str(l) for l in model.unfreeze_layers)})")

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
