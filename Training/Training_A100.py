"""
Whisper Large V3 + LoRA Training on A100
No torchcodec, no datasets Audio — loads WAV files directly with soundfile.
"""
import os, re, json, random, sqlite3, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from jiwer import wer, cer

SOURCE_MODEL = "openai/whisper-large-v3"
CHUNKS_DIR = "/workspace/chunks"
DB_PATH = "/workspace/asr.db"
CHECKPOINTS_DIR = "/workspace/checkpoints"
SERIES_IDS = None  # Set via --series flag, or loads all series with transcriptions

EPOCHS = 3
LEARNING_RATE = 1e-4
GRADIENT_CLIP = 1.0
ACCUMULATION_STEPS = 1
BATCH_SIZE = 16
NUM_WORKERS = 4
WARMUP_STEPS = 200
VALIDATION_SPLIT = 0.15
VAL_SAMPLES = 500
SEED = 42

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"]

ARABIC_DIACRITICS = set("ًٌٍَُِّْٰٕ")

def normalize_arabic(text):
    if not text: return ""
    text = "".join(c for c in text if c not in ARABIC_DIACRITICS)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = text.replace('ـ', '')
    text = re.sub(r'[.,،؟!?:;؛\-\"\'()\[\]{}]', '', text)
    text = " ".join(text.split()).strip()
    return text

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio

def load_data_from_db(db_path, series_ids, chunks_dir):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if series_ids:
        placeholders = ','.join('?' * len(series_ids))
        c.execute(f"""
            SELECT c.file_path, c.transcription, c.transcription_cleaned,
                   c.episode_id, c.was_filtered, c.duration,
                   s.name as series_name
            FROM chunks c
            JOIN episodes e ON c.episode_id = e.id
            JOIN series s ON e.series_id = s.id
            WHERE e.series_id IN ({placeholders})
        """, series_ids)
    else:
        c.execute("""
            SELECT c.file_path, c.transcription, c.transcription_cleaned,
                   c.episode_id, c.was_filtered, c.duration,
                   s.name as series_name
            FROM chunks c
            JOIN episodes e ON c.episode_id = e.id
            JOIN series s ON e.series_id = s.id
        """)

    samples = []
    skipped = {"no_text": 0, "filtered": 0, "no_file": 0, "too_short": 0, "too_long": 0}

    for row in c.fetchall():
        file_path, transcription, transcription_cleaned, episode_id, was_filtered, duration, series_name = row
        if not transcription:
            skipped["no_text"] += 1
            continue
        if was_filtered:
            skipped["filtered"] += 1
            continue
        if duration and duration < 0.5:
            skipped["too_short"] += 1
            continue
        if duration and duration > 30.0:
            skipped["too_long"] += 1
            continue
        filename = file_path.split('\\')[-1]
        linux_path = f"{chunks_dir}/episode_{episode_id}/{filename}"
        if not os.path.exists(linux_path):
            skipped["no_file"] += 1
            continue
        text = transcription_cleaned or transcription
        samples.append({"audio_path": linux_path, "text": text, "series": series_name})

    conn.close()
    print(f"Loaded {len(samples)} chunks")
    if any(skipped.values()):
        print(f"Skipped: {skipped}")
    return samples

class ChunkDataset(Dataset):
    """Same filtering as the old prepare_sample, but CPU-only so it can run in workers."""
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = normalize_arabic(sample["text"])
        if not text or len(text.strip()) < 2:
            return None
        try:
            audio = load_audio(sample["audio_path"])
        except Exception:
            return None
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids[0]
        if labels.shape[0] > 448:
            return None
        return input_features, labels


def collate_batch(batch):
    n_bad = sum(1 for b in batch if b is None)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, 0, n_bad
    input_features = torch.stack([b[0] for b in batch])
    max_len = max(b[1].shape[0] for b in batch)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (_, l) in enumerate(batch):
        padded_labels[i, :l.shape[0]] = l
    return input_features, padded_labels, len(batch), n_bad

def evaluate(model, processor, samples, device, gen_config):
    model.eval()
    wer_scores, cer_scores, results = [], [], []
    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating", leave=False, ncols=100):
            try:
                reference_norm = normalize_arabic(sample["text"])
                if not reference_norm: continue
                audio = load_audio(sample["audio_path"])
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
                pred_ids = model.generate(inputs.input_features, **gen_config)
                prediction_norm = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                w = wer(reference_norm, prediction_norm) if prediction_norm else 1.0
                c2 = cer(reference_norm, prediction_norm) if prediction_norm else 1.0
                wer_scores.append(w)
                cer_scores.append(c2)
                results.append({'series': sample.get("series", "unknown"), 'reference': reference_norm, 'prediction': prediction_norm, 'wer': w, 'cer': c2})
            except:
                continue

    mean_wer = float(np.mean(wer_scores)) if wer_scores else 1.0
    mean_cer = float(np.mean(cer_scores)) if cer_scores else 1.0

    series_metrics = {}
    for sn in set(r['series'] for r in results):
        sr = [r for r in results if r['series'] == sn]
        series_metrics[sn] = {'count': len(sr), 'wer': float(np.mean([r['wer'] for r in sr]))}

    print(f"\n   Overall: WER={mean_wer:.4f}, CER={mean_cer:.4f} ({len(wer_scores)} samples)")
    for name, sm in series_metrics.items():
        print(f"   {name}: WER={sm['wer']:.4f} (n={sm['count']})")

    if results:
        sorted_r = sorted(results, key=lambda x: x['wer'], reverse=True)
        print(f"\n   WORST 10:")
        for r in sorted_r[:10]:
            print(f"   [{r['series']}] WER={r['wer']:.3f} REF: {r['reference'][:60]} PRD: {r['prediction'][:60]}")
        print(f"\n   BEST 10:")
        for r in sorted_r[-10:]:
            print(f"   [{r['series']}] WER={r['wer']:.3f} REF: {r['reference'][:60]} PRD: {r['prediction'][:60]}")

    model.train()
    return {'wer': mean_wer, 'cer': mean_cer, 'samples': len(wer_scores), 'per_series': series_metrics}

def main():
    global DB_PATH, CHUNKS_DIR, EPOCHS, LEARNING_RATE, LORA_RANK, LORA_ALPHA
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=DB_PATH)
    parser.add_argument("--chunks-dir", type=str, default=CHUNKS_DIR)
    parser.add_argument("--series", type=int, nargs="+", default=None)
    parser.add_argument("--exclude", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--alpha", type=int, default=LORA_ALPHA)
    args = parser.parse_args()

    DB_PATH = args.db
    CHUNKS_DIR = args.chunks_dir
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    LORA_RANK = args.rank
    LORA_ALPHA = args.alpha

    if args.series:
        series_ids = args.series
    else:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT id FROM series").fetchall()
        series_ids = [r[0] for r in rows if r[0] not in args.exclude]
        conn.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(CHECKPOINTS_DIR) / f"lora_largev3_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print(f"\nLoading {SOURCE_MODEL}...")
    processor = WhisperProcessor.from_pretrained(SOURCE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(SOURCE_MODEL).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nApplying LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}")
    for param in model.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\nLoading data...")
    all_samples = load_data_from_db(DB_PATH, series_ids, CHUNKS_DIR)
    if len(all_samples) == 0:
        print("ERROR: No data loaded! Check paths.")
        return

    random.seed(SEED)
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * VALIDATION_SPLIT)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    # Held-out set stays at VALIDATION_SPLIT; only a fixed random subset is scored each epoch
    # so that per-epoch WER stays comparable and best-checkpoint selection is not sampling noise.
    val_eval_samples = random.sample(val_samples, min(VAL_SAMPLES, len(val_samples)))
    print(f"Train: {len(train_samples):,}, Val: {len(val_samples):,} (evaluating {len(val_eval_samples):,} of them)")

    train_loader = DataLoader(
        ChunkDataset(train_samples, processor),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=collate_batch, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(train_samples) * EPOCHS) // (ACCUMULATION_STEPS * BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    scaler = torch.amp.GradScaler('cuda')
    gen_config = {"language": "ar", "task": "transcribe"}

    print(f"\n{'='*80}")
    print(f"TRAINING: {SOURCE_MODEL} + LoRA")
    print(f"  Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}, Accum: {ACCUMULATION_STEPS}")
    print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}, modules: {TARGET_MODULES}")
    print(f"  Total steps: {total_steps}, Warmup: {WARMUP_STEPS}")
    print(f"{'='*80}")

    metrics_log = {"config": {"source_model": SOURCE_MODEL, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "target_modules": TARGET_MODULES, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "train_samples": len(train_samples), "val_samples": len(val_samples)}, "epochs": []}
    best_wer = float("inf")

    for epoch in range(EPOCHS):
        print(f"\n{'='*80}\nEPOCH {epoch+1}/{EPOCHS}\n{'='*80}")
        torch.manual_seed(SEED + epoch)
        epoch_losses, skipped, processed = [], 0, 0
        model.train()
        optimizer.zero_grad()
        num_batches = len(train_loader)
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=120, total=num_batches)

        for batch_idx, batch in enumerate(bar):
            try:
                input_features, padded_labels, n_ok, n_bad = batch
                skipped += n_bad
                if n_ok == 0: continue
                input_features = input_features.to(device, non_blocking=True)
                padded_labels = padded_labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out = model(input_features=input_features, labels=padded_labels)
                    loss = out.loss / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                loss_val = out.loss.item()
                epoch_losses.append(loss_val)
                processed += n_ok
                if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx == num_batches - 1):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                if batch_idx % 25 == 0:
                    bar.set_postfix({'loss': f"{loss_val:.3f}", 'avg': f"{np.mean(epoch_losses[-50:]):.3f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f"\n   OOM at batch {batch_idx}!")
                skipped += BATCH_SIZE; continue
            except:
                skipped += BATCH_SIZE; continue

        print(f"\nLoss: Mean={np.mean(epoch_losses):.4f}, Median={np.median(epoch_losses):.4f}")
        print(f"Processed: {processed}, Skipped: {skipped}")

        print("\nEvaluating...")
        val_metrics = evaluate(model, processor, val_eval_samples, device, gen_config)

        if val_metrics['wer'] < best_wer:
            best_wer = val_metrics['wer']
            best_dir = run_dir / "best"
            best_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            print(f"   *** New best WER: {best_wer:.4f} ***")

        epoch_dir = run_dir / f"epoch_{epoch+1:02d}"
        epoch_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        processor.save_pretrained(str(epoch_dir))

        metrics_log["epochs"].append({"epoch": epoch+1, "loss_mean": float(np.mean(epoch_losses)), "val_wer": val_metrics['wer'], "val_cer": val_metrics['cer'], "processed": processed, "skipped": skipped})
        with open(run_dir / "training_log.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

    print("\nMerging LoRA into base model...")
    merged_dir = run_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE! Best WER: {best_wer:.4f}")
    print(f"Merged model: {merged_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
