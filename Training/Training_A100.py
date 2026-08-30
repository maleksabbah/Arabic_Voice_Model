"""
Standalone LoRA training script for RunPod.
Trains fresh LoRA on base Whisper Large V3 using consensus-labeled data from asr.db.

Usage:
  python train_lora.py --db /workspace/asr.db --series 1 2 3 4 5 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 --epochs 3
  python train_lora.py --db /workspace/asr.db --series 1 2 3 --epochs 3 --rank 64
"""
import argparse
import os
import time
import sqlite3
import random
import gc

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
import soundfile as sf
from jiwer import wer


class ChunkDataset(Dataset):
    """Loads consensus-labeled chunks from SQLite."""
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
            # Resample if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            # Truncate
            max_samples = int(self.max_duration * sr)
            audio = audio[:max_samples]
        except Exception as e:
            # Return silence on error
            audio = np.zeros(16000, dtype=np.float32)

        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features.squeeze(0)

        labels = self.processor.tokenizer(
            c["transcription"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=448,
        ).input_ids.squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels,
        }


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
    # Filter to existing files
    chunks = [c for c in chunks if os.path.exists(c["file_path"])]
    return chunks


def benchmark_sample(model, processor, chunks, n=200, device="cuda"):
    """Quick WER benchmark on a random sample."""
    model.eval()
    sample = random.sample(chunks, min(n, len(chunks)))
    refs = []
    hyps = []
    for c in sample:
        try:
            audio, sr = sf.read(c["file_path"], dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                predicted_ids = model.generate(inputs, language="ar", task="transcribe", max_new_tokens=448)
            hyp = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            refs.append(c["transcription"])
            hyps.append(hyp)
        except:
            pass
    if not refs:
        return 1.0
    model.train()
    return wer(refs, hyps)


def train(args):
    print("=" * 60)
    print("LORA TRAINING — WHISPER LARGE V3")
    print("=" * 60)
    print(f"DB: {args.db}")
    print(f"Series: {args.series}")
    print(f"Epochs: {args.epochs}")
    print(f"Rank: {args.rank}, Alpha: {args.alpha}")
    print(f"LR: {args.lr}")
    print(f"Batch size: {args.batch_size} x {args.accumulation} accumulation")
    print(f"Effective batch: {args.batch_size * args.accumulation}")
    print()

    # Load data
    print("[DATA] Loading labeled chunks...")
    chunks = get_labeled_chunks(args.db, args.series)
    print(f"[DATA] {len(chunks)} labeled chunks with audio files")

    if not chunks:
        print("No data! Exiting.")
        return

    # Split train/val
    random.shuffle(chunks)
    val_size = min(1000, int(len(chunks) * 0.05))
    val_chunks = chunks[:val_size]
    train_chunks = chunks[val_size:]
    print(f"[DATA] Train: {len(train_chunks)}, Val: {len(val_chunks)}")

    # Load model and processor
    print("[MODEL] Loading openai/whisper-large-v3...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16,
    )

    # Apply LoRA
    print(f"[MODEL] Applying LoRA (rank={args.rank}, alpha={args.alpha})...")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to("cuda")


    # Dataset and dataloader
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
    warmup_steps = int(total_steps * 0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
    )

    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    print(f"\n[TRAIN] Starting training — {total_steps} total steps")
    best_val_wer = 1.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        step_count = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_features = batch["input_features"].to("cuda", dtype=torch.float16)
            labels = batch["labels"].to("cuda")

            with torch.amp.autocast('cuda', dtype=torch.float16):
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
                step_count += 1

            epoch_loss += loss.item() * args.accumulation

            if (step + 1) % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed
                eta = (len(train_loader) - step - 1) / steps_per_sec
                lr_now = scheduler.get_last_lr()[0]
                print(f"  [E{epoch+1} {step+1}/{len(train_loader)}] loss={avg_loss:.4f} lr={lr_now:.2e} {steps_per_sec:.1f} steps/s ETA {eta/60:.0f}m")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # Validation WER
        print(f"  [E{epoch+1}] Benchmarking on val set...")
        val_wer = benchmark_sample(model, processor, val_chunks, n=min(200, len(val_chunks)))
        print(f"  [E{epoch+1}] Loss={avg_loss:.4f} Val_WER={val_wer:.2%} Time={elapsed/60:.1f}m")

        # Save checkpoint
        checkpoint_dir = f"/workspace/checkpoints/lora_epoch_{epoch+1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        print(f"  [E{epoch+1}] Saved to {checkpoint_dir}")

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_dir = "/workspace/checkpoints/lora_best"
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  [E{epoch+1}] New best! WER={val_wer:.2%}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Best Val WER: {best_val_wer:.2%}")
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
    a = p.parse_args()
    train(a)


if __name__ == "__main__":
    main()