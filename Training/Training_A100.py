"""
Manual LoRA training for Whisper Large V3 with MGB-3, FLEURS, Casablanca benchmarks.
Standalone script for RunPod — no project imports.

Usage:
  python Training/Training_A100.py --db /workspace/asr.db --series 1 2 3 --epochs 3
"""
import argparse
import os
import time
import sqlite3
import random
import gc
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from jiwer import wer as compute_wer

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


# ============================================================
# DATASET
# ============================================================
class ChunkDataset(Dataset):
    def __init__(self, chunks, processor, max_duration=30):
        self.chunks = chunks
        self.processor = processor
        self.max_duration = max_duration

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        c = self.chunks[idx]
        try:
            audio, sr = sf.read(c["file_path"], dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            max_samples = int(self.max_duration * sr)
            audio = audio[:max_samples]
        except:
            audio = np.zeros(16000, dtype=np.float32)

        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)

        labels = self.processor.tokenizer(
            c["transcription"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=448,
        ).input_ids.squeeze(0)

        return {"input_features": input_features, "labels": labels}


def collate_fn(batch):
    input_features = torch.stack([b["input_features"] for b in batch])
    label_lengths = [len(b["labels"]) for b in batch]
    max_len = max(label_lengths)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b["labels"])] = b["labels"]
    return {"input_features": input_features, "labels": labels}


# ============================================================
# DATA LOADING
# ============================================================
def get_labeled_chunks(db_path, series_ids):
    conn = sqlite3.connect(db_path)
    ph = ",".join(str(s) for s in series_ids)
    rows = conn.execute(f"""
        SELECT c.id, c.file_path, c.transcription, e.series_id
        FROM chunks c JOIN episodes e ON c.episode_id = e.id
        WHERE e.series_id IN ({ph})
        AND c.transcription IS NOT NULL AND c.transcription != ''
    """).fetchall()
    conn.close()
    chunks = [{"id": r[0], "file_path": r[1], "transcription": r[2], "series_id": r[3]} for r in rows]
    chunks = [c for c in chunks if os.path.exists(c["file_path"])]
    return chunks


# ============================================================
# BENCHMARKS — uses soundfile for audio, no torchcodec needed
# ============================================================
def load_audio_from_sample(sample):
    """Extract audio array from a HuggingFace dataset sample."""
    audio = np.array(sample["audio"]["array"], dtype=np.float32)
    sr = sample["audio"]["sampling_rate"]
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def run_inference(model, processor, audio, device):
    """Run inference on a single audio array. Bypasses peft wrapper."""
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=torch.float16)
    with torch.no_grad():
        predicted_ids = model.base_model.model.generate(
            inputs,
            language="ar",
            task="transcribe",
            max_new_tokens=448,
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def benchmark_fleurs(model, processor, device, max_samples=200):
    """Benchmark on FLEURS Arabic (MSA)."""
    print("  [BENCH] FLEURS (MSA)...", end=" ", flush=True)
    model.eval()
    try:
        ds = load_dataset("google/fleurs", "ar_eg", split="test", streaming=True)
        refs, hyps = [], []
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            audio = load_audio_from_sample(sample)
            ref = sample["transcription"]
            hyp = run_inference(model, processor, audio, device)
            refs.append(ref)
            hyps.append(hyp)
        w = compute_wer(refs, hyps) if refs else 1.0
        print(f"{w:.2%} ({len(refs)} samples)")
        return w
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def benchmark_mgb3(model, processor, device, max_samples=200):
    """Benchmark on MGB-3 (Egyptian dialect)."""
    print("  [BENCH] MGB-3 (Egyptian)...", end=" ", flush=True)
    model.eval()
    try:
        ds = load_dataset("MightyStudent/Egyptian-ASR-MGB-3", split="train", streaming=True)
        refs, hyps = [], []
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            audio = load_audio_from_sample(sample)
            ref = sample["sentence"]
            hyp = run_inference(model, processor, audio, device)
            refs.append(ref)
            hyps.append(hyp)
        w = compute_wer(refs, hyps) if refs else 1.0
        print(f"{w:.2%} ({len(refs)} samples)")
        return w
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def benchmark_casablanca(model, processor, device, max_samples=200):
    """Benchmark on Casablanca (multi-dialect Arabic)."""
    print("  [BENCH] Casablanca (multi-dialect)...", end=" ", flush=True)
    model.eval()
    try:
        ds = load_dataset("UBC-NLP/Casablanca", "Egypt", split="test", streaming=True)
        refs, hyps = [], []
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            audio = load_audio_from_sample(sample)
            ref = sample["transcription"]
            hyp = run_inference(model, processor, audio, device)
            refs.append(ref)
            hyps.append(hyp)
        w = compute_wer(refs, hyps) if refs else 1.0
        print(f"{w:.2%} ({len(refs)} samples)")
        return w
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def run_all_benchmarks(model, processor, device, max_samples=200):
    """Run all benchmarks and return results dict."""
    results = {}
    results["fleurs"] = benchmark_fleurs(model, processor, device, max_samples)
    results["mgb3"] = benchmark_mgb3(model, processor, device, max_samples)
    results["casablanca"] = benchmark_casablanca(model, processor, device, max_samples)
    return results


# ============================================================
# TRAINING
# ============================================================
def train(args):
    device = "cuda"

    print("=" * 60)
    print("MANUAL LORA TRAINING — WHISPER LARGE V3")
    print("=" * 60)
    print(f"DB: {args.db}")
    print(f"Series: {args.series}")
    print(f"Epochs: {args.epochs}")
    print(f"Rank: {args.rank}, Alpha: {args.alpha}")
    print(f"LR: {args.lr}")
    print(f"Batch: {args.batch_size} x {args.accumulation}")
    print(f"Benchmark samples: {args.benchmark_samples}")
    print()

    # Load data
    print("[DATA] Loading...")
    chunks = get_labeled_chunks(args.db, args.series)
    print(f"[DATA] {len(chunks)} labeled chunks")
    if not chunks:
        print("No data!")
        return

    random.seed(42)
    random.shuffle(chunks)
    val_size = min(1000, int(len(chunks) * 0.05))
    val_chunks = chunks[:val_size]
    train_chunks = chunks[val_size:]
    print(f"[DATA] Train: {len(train_chunks)}, Val: {len(val_chunks)}")

    # Load model
    print("[MODEL] Loading whisper-large-v3...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16,
    )

    # Apply LoRA
    print(f"[MODEL] LoRA rank={args.rank}, alpha={args.alpha}")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable input grads for peft compatibility
    model.enable_input_require_grads()

    # Cast trainable params to float32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.to(device)

    # Dataloader
    train_dataset = ChunkDataset(train_chunks, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.accumulation
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
    )

    scaler = torch.amp.GradScaler('cuda')

    # Run baseline benchmarks before training
    print("\n[BASELINE] Running benchmarks before training...")
    baseline = run_all_benchmarks(model, processor, device, args.benchmark_samples)
    print()

    # Training loop
    best_val_wer = 1.0
    print(f"[TRAIN] {total_steps} total steps, {len(train_loader)} steps/epoch")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(device, dtype=torch.float16)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda'):
                # Bypass peft wrapper — call base model directly
                outputs = model.base_model.model(input_features=input_features, labels=labels)
                loss = outputs.loss / args.accumulation

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * args.accumulation

            if (step + 1) % 100 == 0:
                avg = epoch_loss / (step + 1)
                el = time.time() - t0
                rate = (step + 1) / el
                eta = (len(train_loader) - step - 1) / rate
                lr = scheduler.get_last_lr()[0]
                print(f"  [E{epoch+1} {step+1}/{len(train_loader)}] loss={avg:.4f} lr={lr:.2e} {rate:.1f}s/s ETA {eta/60:.0f}m")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # Val WER on subset
        print(f"  [E{epoch+1}] Evaluating val set...")
        model.eval()
        val_refs, val_hyps = [], []
        val_sample = random.sample(val_chunks, min(200, len(val_chunks)))
        for c in val_sample:
            try:
                audio, sr = sf.read(c["file_path"], dtype="float32")
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                hyp = run_inference(model, processor, audio, device)
                val_refs.append(c["transcription"])
                val_hyps.append(hyp)
            except:
                pass
        val_wer = compute_wer(val_refs, val_hyps) if val_refs else 1.0

        print(f"  [E{epoch+1}] Loss={avg_loss:.4f} Val_WER={val_wer:.2%} Time={elapsed/60:.1f}m")

        # Run benchmarks
        print(f"  [E{epoch+1}] Running benchmarks...")
        bench = run_all_benchmarks(model, processor, device, args.benchmark_samples)

        # Save checkpoint
        ckpt_dir = f"/workspace/checkpoints/lora_epoch_{epoch+1}"
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"  [E{epoch+1}] Saved to {ckpt_dir}")

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_dir = "/workspace/checkpoints/lora_best"
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  [E{epoch+1}] New best! WER={val_wer:.2%}")

        model.train()

    # Final summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Best Val WER: {best_val_wer:.2%}")
    print(f"Baseline: FLEURS={baseline.get('fleurs','N/A')} MGB3={baseline.get('mgb3','N/A')} Casa={baseline.get('casablanca','N/A')}")
    if bench:
        print(f"Final:   FLEURS={bench.get('fleurs','N/A')} MGB3={bench.get('mgb3','N/A')} Casa={bench.get('casablanca','N/A')}")
    print(f"Checkpoints: /workspace/checkpoints/")
    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--series", type=int, nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--benchmark-samples", type=int, default=200)
    a = p.parse_args()
    train(a)


if __name__ == "__main__":
    main()