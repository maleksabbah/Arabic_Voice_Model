"""
Eval-only: runs the March Training_A100.py evaluate() function against a checkpoint.
Same normalize_arabic, same per-sample WER, same 15% val split, same seed 42.

Usage:
  # Against the old March checkpoint:
  python Training/Evaluate.py --lora-path /workspace/checkpoints/largev3_best --db /workspace/asr.db

  # Against the current run's merged model:
  python Training/Evaluate.py --merged-path /workspace/checkpoints/lora_largev3_20260916_083556/merged --db /workspace/asr.db

  # Against the current run's best adapter:
  python Training/Evaluate.py --lora-path /workspace/checkpoints/lora_largev3_20260916_083556/best --db /workspace/asr.db

  # Specific series (March used 1,3,4,10):
  python Training/Evaluate.py --lora-path /workspace/checkpoints/largev3_best --db /workspace/asr.db --series 1 3 4 10

  # All series except 5-8 (current run):
  python Training/Evaluate.py --lora-path /workspace/checkpoints/largev3_best --db /workspace/asr.db
"""
import argparse
import os
import re
import random
import sqlite3
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import wer, cer

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

def evaluate(model, processor, samples, device):
    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    wer_scores, cer_scores, results = [], [], []

    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating", ncols=100):
            try:
                reference_norm = normalize_arabic(sample["text"])
                if not reference_norm: continue
                audio = load_audio(sample["audio_path"])
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
                pred_ids = model.generate(inputs.input_features, **gen_config)
                prediction_norm = normalize_arabic(
                    processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                )
                w = wer(reference_norm, prediction_norm) if prediction_norm else 1.0
                c2 = cer(reference_norm, prediction_norm) if prediction_norm else 1.0
                wer_scores.append(w)
                cer_scores.append(c2)
                results.append({
                    'series': sample.get("series", "unknown"),
                    'reference': reference_norm,
                    'prediction': prediction_norm,
                    'wer': w, 'cer': c2,
                })
            except:
                continue

    mean_wer = float(np.mean(wer_scores)) if wer_scores else 1.0
    mean_cer = float(np.mean(cer_scores)) if cer_scores else 1.0

    series_metrics = {}
    for sn in set(r['series'] for r in results):
        sr = [r for r in results if r['series'] == sn]
        series_metrics[sn] = {
            'count': len(sr),
            'wer': float(np.mean([r['wer'] for r in sr])),
            'cer': float(np.mean([r['cer'] for r in sr])),
        }

    print(f"\n{'='*60}")
    print(f"VAL WER RESULTS")
    print(f"{'='*60}")
    print(f"Overall: WER={mean_wer:.4f}, CER={mean_cer:.4f} ({len(wer_scores)} samples)")
    print(f"\nPer-series:")
    for name, sm in sorted(series_metrics.items(), key=lambda x: x[1]['wer']):
        print(f"  {name}: WER={sm['wer']:.4f}, CER={sm['cer']:.4f} (n={sm['count']})")

    if results:
        sorted_r = sorted(results, key=lambda x: x['wer'], reverse=True)
        print(f"\nWORST 10:")
        for r in sorted_r[:10]:
            print(f"  [{r['series']}] WER={r['wer']:.3f}")
            print(f"    REF: {r['reference'][:80]}")
            print(f"    PRD: {r['prediction'][:80]}")
        print(f"\nBEST 10:")
        for r in sorted_r[-10:]:
            print(f"  [{r['series']}] WER={r['wer']:.3f}")
            print(f"    REF: {r['reference'][:80]}")
            print(f"    PRD: {r['prediction'][:80]}")

    print(f"{'='*60}")
    return mean_wer, mean_cer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="/workspace/asr.db")
    parser.add_argument("--chunks-dir", type=str, default="/workspace/chunks")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--merged-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--series", type=int, nargs="+", default=None)
    parser.add_argument("--exclude", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--max-val", type=int, default=None, help="Cap val eval to N samples (default: all)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Determine series
    if args.series:
        series_ids = args.series
    else:
        conn = sqlite3.connect(args.db)
        rows = conn.execute("SELECT id FROM series").fetchall()
        series_ids = [r[0] for r in rows if r[0] not in args.exclude]
        conn.close()
    print(f"Series: {series_ids}")

    # Load data and split — same as March
    all_samples = load_data_from_db(args.db, series_ids, args.chunks_dir)
    if not all_samples:
        print("No data!")
        return

    random.seed(42)
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * 0.15)
    val_samples = all_samples[:val_size]
    print(f"Total: {len(all_samples)}, Val: {len(val_samples)}")

    if args.max_val and len(val_samples) > args.max_val:
        val_samples = random.sample(val_samples, args.max_val)
        print(f"Subsampled val to {len(val_samples)}")

    # Load model
    print(f"\nLoading {args.base_model}...")
    processor = WhisperProcessor.from_pretrained(args.base_model)

    if args.merged_path:
        print(f"Loading merged model from {args.merged_path}")
        model = WhisperForConditionalGeneration.from_pretrained(args.merged_path).to(device)
    elif args.lora_path:
        print(f"Loading base + LoRA from {args.lora_path}")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
        model = PeftModel.from_pretrained(model, args.lora_path).to(device)
    else:
        print("Loading base model only (no LoRA)")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)

    model.eval()

    # Run eval
    val_wer, val_cer = evaluate(model, processor, val_samples, device)
    print(f"\nFinal Val WER: {val_wer:.4f}")
    print(f"Final Val CER: {val_cer:.4f}")

if __name__ == "__main__":
    main()
