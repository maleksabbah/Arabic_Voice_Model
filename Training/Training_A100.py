"""
Manual LoRA training for Whisper Large V3.
CRITICAL: No task_type in LoraConfig — that's what caused the input_ids bug.

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
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from jiwer import wer as compute_wer

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


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
            audio = audio[:int(self.max_duration * sr)]
        except:
            audio = np.zeros(16000, dtype=np.float32)

        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)

        labels = self.processor.tokenizer(
            c["transcription"], return_tensors="pt",
            padding=False, truncation=True, max_length=440,
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
    return [c for c in chunks if os.path.exists(c["file_path"])]


def transcribe(model, processor, audio, device):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=torch.float16)
    with torch.no_grad():
        ids = model.generate(input_features=inputs, language="ar", task="transcribe", max_new_tokens=440)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def benchmark(model, processor, device, name, dataset_id, config, split, ref_field, max_samples=200):
    print(f"  [BENCH] {name}...", end=" ", flush=True)
    model.eval()
    try:
        ds = load_dataset(dataset_id, config, split=split, streaming=True) if config else load_dataset(dataset_id, split=split, streaming=True)
        refs, hyps = [], []
        for i, s in enumerate(ds):
            if i >= max_samples: break
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            sr = s["audio"]["sampling_rate"]
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            hyps.append(transcribe(model, processor, audio, device))
            refs.append(s[ref_field])
        w = compute_wer(refs, hyps) if refs else 1.0
        print(f"{w:.2%} ({len(refs)} samples)")
        return w
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def run_all_benchmarks(model, processor, device, max_samples=200):
    r = {}
    r["fleurs"] = benchmark(model, processor, device, "FLEURS (MSA)", "google/fleurs", "ar_eg", "test", "transcription", max_samples)
    r["mgb3"] = benchmark(model, processor, device, "MGB-3 (Egyptian)", "MightyStudent/Egyptian-ASR-MGB-3", None, "train", "sentence", max_samples)
    r["casablanca"] = benchmark(model, processor, device, "Casablanca (Egyptian)", "UBC-NLP/Casablanca", "Egypt", "test", "transcription", max_samples)
    return r


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
    print(flush=True)

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
    print(f"[DATA] Train: {len(train_chunks)}, Val: {len(val_chunks)}", flush=True)

    print("[MODEL] Loading whisper-large-v3...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16)

    # NO task_type — that's what caused the input_ids bug
    print(f"[MODEL] LoRA rank={args.rank}, alpha={args.alpha}")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.to(device)

    train_dataset = ChunkDataset(train_chunks, processor)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs // args.accumulation
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1,
    )

    scaler = torch.amp.GradScaler('cuda')

    print("\n[BASELINE] Running benchmarks...", flush=True)
    baseline = run_all_benchmarks(model, processor, device, args.benchmark_samples)

    best_val_wer = 1.0
    print(f"\n[TRAIN] {total_steps} total steps, {len(train_loader)} steps/epoch", flush=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(device, dtype=torch.float16)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(input_features=input_features, labels=labels)
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
                print(f"  [E{epoch+1} {step+1}/{len(train_loader)}] loss={avg:.4f} lr={lr:.2e} {rate:.1f}s/s ETA {eta/60:.0f}m", flush=True)

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # Val WER
        print(f"  [E{epoch+1}] Evaluating val set...", flush=True)
        model.eval()
        val_refs, val_hyps = [], []
        for c in random.sample(val_chunks, min(200, len(val_chunks))):
            try:
                audio, sr = sf.read(c["file_path"], dtype="float32")
                if len(audio.shape) > 1: audio = audio.mean(axis=1)
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                val_hyps.append(transcribe(model, processor, audio, device))
                val_refs.append(c["transcription"])
            except: pass
        val_wer = compute_wer(val_refs, val_hyps) if val_refs else 1.0
        print(f"  [E{epoch+1}] Loss={avg_loss:.4f} Val_WER={val_wer:.2%} Time={elapsed/60:.1f}m", flush=True)

        # Benchmarks
        print(f"  [E{epoch+1}] Running benchmarks...", flush=True)
        bench = run_all_benchmarks(model, processor, device, args.benchmark_samples)

        # Save checkpoint
        ckpt_dir = f"/workspace/checkpoints/lora_epoch_{epoch+1}"
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"  [E{epoch+1}] Saved to {ckpt_dir}", flush=True)

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_dir = "/workspace/checkpoints/lora_best"
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  [E{epoch+1}] New best! WER={val_wer:.2%}", flush=True)

        model.train()

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