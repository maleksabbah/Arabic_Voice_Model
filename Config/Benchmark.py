"""
Arabic ASR Benchmark Suite — A100 Edition
==========================================
All benchmarks from the Open Universal Arabic ASR Leaderboard:
  1. FLEURS        — MSA clean read speech
  2. MGB-3         — Egyptian dialect
  3. Casablanca    — 8 Arabic dialects
  4. MGB-2         — Multi-dialect broadcast (MSA + Egyptian + Gulf + Levantine + North African)
  5. MASC clean    — Multi-dialect clean test
  6. MASC noisy    — Multi-dialect noisy test
  7. Common Voice  — MSA crowd-sourced
  8. SADA          — Saudi dialects (10 dialects)

Usage:
  # Full suite with LoRA adapter vs base:
  python benchmark_a100.py --lora-path checkpoints/largev3_best --base-model openai/whisper-large-v3 --samples 200

  # Skip base (already know the numbers):
  python benchmark_a100.py --lora-path checkpoints/largev3_best --base-model openai/whisper-large-v3 --skip-base --samples 200

  # Specific benchmarks:
  python benchmark_a100.py --lora-path checkpoints/largev3_best --base-model openai/whisper-large-v3 --benchmarks fleurs mgb3 casablanca
"""
import argparse
import re
import os
import torch
import soundfile as sf
import numpy as np
from jiwer import wer, cer
from tqdm import tqdm
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")

def normalize_arabic(text):
    if not text: return ""
    text = "".join(c for c in text if c not in DIACRITICS)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = text.replace('ـ', '')
    text = re.sub(r'[.,،؟!?:;؛\-\"\'()\[\]{}]', '', text)
    text = " ".join(text.split()).strip()
    return text

def safe_generate(model, inputs, gen_config, device):
    """Generate with autocast to handle float16/float32 mismatch."""
    with torch.amp.autocast('cuda'):
        pred_ids = model.generate(inputs.input_features.to(device), **gen_config)
    return pred_ids

def process_audio(sample):
    """Extract audio array and sample rate from a dataset sample."""
    audio = sample["audio"]
    wav = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    return wav, sr

# =============================================================================
# 1. FLEURS — MSA Clean
# =============================================================================
def benchmark_fleurs(model, processor, device, max_samples=500):
    print("\n  [BENCHMARK] FLEURS (MSA)...")
    dataset = load_dataset("google/fleurs", "ar_eg", split="test", trust_remote_code=True)
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    n = min(max_samples, len(dataset))

    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    wer_scores, cer_scores = [], []

    with torch.no_grad():
        for idx in tqdm(indices[:n], desc="  FLEURS", ncols=100, leave=False):
            try:
                sample = dataset[idx]
                reference = normalize_arabic(sample["transcription"].strip())
                if not reference: continue
                wav, sr = process_audio(sample)
                inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                pred_ids = safe_generate(model, inputs, gen_config, device)
                prediction = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                w = wer(reference, prediction) if prediction else 1.0
                c = cer(reference, prediction) if prediction else 1.0
                wer_scores.append(w); cer_scores.append(c)
            except: continue

    results = {"wer": float(np.mean(wer_scores)) if wer_scores else 1.0, "cer": float(np.mean(cer_scores)) if cer_scores else 1.0, "total": len(wer_scores)}
    print(f"  FLEURS: WER={results['wer']:.4f}, CER={results['cer']:.4f} (n={results['total']})")
    model.train()
    return results

# =============================================================================
# 2. MGB-3 — Egyptian Arabic
# =============================================================================
def benchmark_mgb3(model, processor, device, max_samples=500):
    print("\n  [BENCHMARK] MGB-3 (Egyptian)...")
    dataset = load_dataset("MightyStudent/Egyptian-ASR-MGB-3", split="train", trust_remote_code=True)
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    n = min(max_samples, len(dataset))

    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    wer_scores, cer_scores = [], []

    with torch.no_grad():
        for idx in tqdm(indices[:n], desc="  MGB-3", ncols=100, leave=False):
            try:
                sample = dataset[idx]
                reference = normalize_arabic(sample["sentence"].strip())
                if not reference: continue
                wav, sr = process_audio(sample)
                inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                pred_ids = safe_generate(model, inputs, gen_config, device)
                prediction = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                w = wer(reference, prediction) if prediction else 1.0
                c = cer(reference, prediction) if prediction else 1.0
                wer_scores.append(w); cer_scores.append(c)
            except: continue

    results = {"wer": float(np.mean(wer_scores)) if wer_scores else 1.0, "cer": float(np.mean(cer_scores)) if cer_scores else 1.0, "total": len(wer_scores)}
    print(f"  MGB-3: WER={results['wer']:.4f}, CER={results['cer']:.4f} (n={results['total']})")
    model.train()
    return results

# =============================================================================
# 3. Casablanca — 8 Arabic Dialects
# =============================================================================
CASABLANCA_DIALECTS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]

def benchmark_casablanca(model, processor, device, max_samples=500):
    print("\n  [BENCHMARK] Casablanca (8 dialects)...")
    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    per_dialect = {}
    all_wer, all_cer = [], []
    samples_per_dialect = max(max_samples // len(CASABLANCA_DIALECTS), 10)

    for dialect in CASABLANCA_DIALECTS:
        try:
            ds = load_dataset("UBC-NLP/Casablanca", dialect, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Skipping {dialect}: {e}")
            continue

        indices = list(range(len(ds)))
        np.random.seed(42)
        np.random.shuffle(indices)
        n = min(samples_per_dialect, len(ds))
        wer_scores, cer_scores = [], []

        with torch.no_grad():
            for idx in tqdm(indices[:n], desc=f"  Casa-{dialect[:3]}", ncols=100, leave=False):
                try:
                    sample = ds[idx]
                    reference = normalize_arabic(sample.get("transcription", sample.get("sentence", "")).strip())
                    if not reference: continue
                    wav, sr = process_audio(sample)
                    inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                    pred_ids = safe_generate(model, inputs, gen_config, device)
                    prediction = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                    w = wer(reference, prediction) if prediction else 1.0
                    c = cer(reference, prediction) if prediction else 1.0
                    wer_scores.append(w); cer_scores.append(c)
                except: continue

        if wer_scores:
            d_wer = float(np.mean(wer_scores))
            per_dialect[dialect] = {"wer": d_wer, "cer": float(np.mean(cer_scores)), "total": len(wer_scores)}
            all_wer.extend(wer_scores); all_cer.extend(cer_scores)
            print(f"  Casablanca-{dialect}: WER={d_wer:.4f} (n={len(wer_scores)})")

    results = {"wer": float(np.mean(all_wer)) if all_wer else 1.0, "cer": float(np.mean(all_cer)) if all_cer else 1.0, "total": len(all_wer), "per_dialect": per_dialect}
    print(f"  Casablanca AVG: WER={results['wer']:.4f} ({results['total']} samples)")
    model.train()
    return results

# =============================================================================
# 4. MGB-2 — Multi-dialect Broadcast
# =============================================================================
def benchmark_mgb2(model, processor, device, max_samples=500):
    print("\n  [BENCHMARK] MGB-2 (Multi-dialect broadcast)...")
    try:
        dataset = load_dataset("QCRI/mgb2", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  Failed to load MGB-2: {e}")
        return {"wer": 1.0, "cer": 1.0, "total": 0}

    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    n = min(max_samples, len(dataset))

    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    wer_scores, cer_scores = [], []

    with torch.no_grad():
        for idx in tqdm(indices[:n], desc="  MGB-2", ncols=100, leave=False):
            try:
                sample = dataset[idx]
                reference = normalize_arabic(sample.get("text", sample.get("sentence", sample.get("transcription", ""))).strip())
                if not reference: continue
                wav, sr = process_audio(sample)
                inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                pred_ids = safe_generate(model, inputs, gen_config, device)
                prediction = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                w = wer(reference, prediction) if prediction else 1.0
                c = cer(reference, prediction) if prediction else 1.0
                wer_scores.append(w); cer_scores.append(c)
            except: continue

    results = {"wer": float(np.mean(wer_scores)) if wer_scores else 1.0, "cer": float(np.mean(cer_scores)) if cer_scores else 1.0, "total": len(wer_scores)}
    print(f"  MGB-2: WER={results['wer']:.4f}, CER={results['cer']:.4f} (n={results['total']})")
    model.train()
    return results

# =============================================================================
# 5. MASC — Multi-dialect Arabic (clean / noisy)
# =============================================================================
def benchmark_masc(model, processor, device, max_samples=500, split="test_clean"):
    label = f"MASC-{split.replace('test_', '')}"
    print(f"\n  [BENCHMARK] {label}...")
    try:
        ds = load_dataset("pain/MASC", split=split, trust_remote_code=True)
    except:
        try:
            ds_list = []
            for i, sample in enumerate(load_dataset("pain/MASC", split=split, streaming=True, trust_remote_code=True)):
                if i >= max_samples: break
                ds_list.append(sample)
            ds = ds_list
        except Exception as e:
            print(f"  Failed to load MASC {split}: {e}")
            return {"wer": 1.0, "cer": 1.0, "total": 0}

    total = len(ds)
    indices = list(range(total))
    np.random.seed(42)
    np.random.shuffle(indices)
    n = min(max_samples, total)

    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    wer_scores, cer_scores = [], []
    per_dialect = {}

    with torch.no_grad():
        for idx in tqdm(indices[:n], desc=f"  {label}", ncols=100, leave=False):
            try:
                sample = ds[idx]
                reference = normalize_arabic(sample.get("sentence", sample.get("text", "")).strip())
                if not reference: continue
                wav, sr = process_audio(sample)
                inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                pred_ids = safe_generate(model, inputs, gen_config, device)
                prediction = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                w = wer(reference, prediction) if prediction else 1.0
                c = cer(reference, prediction) if prediction else 1.0
                wer_scores.append(w); cer_scores.append(c)

                dialect = sample.get("dialect", "unknown")
                if dialect not in per_dialect: per_dialect[dialect] = {"wer": [], "cer": []}
                per_dialect[dialect]["wer"].append(w)
                per_dialect[dialect]["cer"].append(c)
            except: continue

    per_dialect_summary = {}
    for d, scores in per_dialect.items():
        per_dialect_summary[d] = {"wer": float(np.mean(scores["wer"])), "total": len(scores["wer"])}

    results = {"wer": float(np.mean(wer_scores)) if wer_scores else 1.0, "cer": float(np.mean(cer_scores)) if cer_scores else 1.0, "total": len(wer_scores), "per_dialect": per_dialect_summary}
    print(f"  {label}: WER={results['wer']:.4f} ({results['total']} samples)")
    for d, s in sorted(per_dialect_summary.items()):
        if s["total"] >= 5: print(f"    {d}: WER={s['wer']:.4f} (n={s['total']})")
    model.train()
    return results

# =============================================================================
# 6. Common Voice Arabic
# =============================================================================
def benchmark_common_voice(model, processor, device, max_samples=500):
    print("\n  [BENCHMARK] Common Voice Arabic...")
    try:
        ds = load_dataset("MohamedRashad/common-voice-18-arabic", split="test", trust_remote_code=True)
    except:
        try:
            ds = load_dataset("mozilla-foundation/common_voice_17_0", "ar", split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Failed to load Common Voice: {e}")
            return {"wer": 1.0, "cer": 1.0, "total": 0}

    indices = list(range(len(ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    n = min(max_samples, len(ds))

    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    wer_scores, cer_scores = [], []

    with torch.no_grad():
        for idx in tqdm(indices[:n], desc="  CV-Arabic", ncols=100, leave=False):
            try:
                sample = ds[idx]
                reference = normalize_arabic(sample.get("sentence", sample.get("text", "")).strip())
                if not reference: continue
                wav, sr = process_audio(sample)
                inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                pred_ids = safe_generate(model, inputs, gen_config, device)
                prediction = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                w = wer(reference, prediction) if prediction else 1.0
                c = cer(reference, prediction) if prediction else 1.0
                wer_scores.append(w); cer_scores.append(c)
            except: continue

    results = {"wer": float(np.mean(wer_scores)) if wer_scores else 1.0, "cer": float(np.mean(cer_scores)) if cer_scores else 1.0, "total": len(wer_scores)}
    print(f"  Common Voice: WER={results['wer']:.4f} (n={results['total']})")
    model.train()
    return results

# =============================================================================
# 7. SADA — Saudi Dialects
# =============================================================================
def benchmark_sada(model, processor, device, max_samples=500):
    print("\n  [BENCHMARK] SADA (Saudi dialects)...")
    import glob

    sada_dir = "/workspace/datasets/sada"
    if not os.path.isdir(sada_dir):
        print(f"  SADA not found at {sada_dir}. Download from Kaggle first:")
        print(f"    kaggle datasets download -d sdaiancai/sada2022 -p /workspace/datasets/sada")
        return {"wer": 1.0, "cer": 1.0, "total": 0}

    wav_files = sorted(glob.glob(os.path.join(sada_dir, "**", "*.wav"), recursive=True))
    if not wav_files:
        print(f"  No WAV files found in {sada_dir}")
        return {"wer": 1.0, "cer": 1.0, "total": 0}

    transcript_map = {}
    for tsv_path in glob.glob(os.path.join(sada_dir, "**", "*.tsv"), recursive=True):
        try:
            import csv
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    fpath = row.get("path", row.get("file", row.get("wav", "")))
                    text = row.get("sentence", row.get("text", row.get("transcription", "")))
                    if fpath and text:
                        transcript_map[os.path.basename(fpath)] = text
        except: pass

    for txt_path in glob.glob(os.path.join(sada_dir, "**", "*.txt"), recursive=True):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        transcript_map[os.path.basename(parts[0])] = parts[1]
                    elif len(parts) == 1:
                        parts2 = line.strip().split('|')
                        if len(parts2) >= 2:
                            transcript_map[os.path.basename(parts2[0])] = parts2[1]
        except: pass

    if not transcript_map:
        print(f"  No transcripts found. Trying JSON...")
        for json_path in glob.glob(os.path.join(sada_dir, "**", "*.json"), recursive=True):
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        fpath = item.get("path", item.get("file", item.get("wav", "")))
                        text = item.get("sentence", item.get("text", item.get("transcription", "")))
                        if fpath and text:
                            transcript_map[os.path.basename(fpath)] = text
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, str):
                            transcript_map[os.path.basename(k)] = v
            except: pass

    print(f"  Found {len(wav_files)} WAV files, {len(transcript_map)} transcripts")

    if not transcript_map:
        print(f"  WARNING: No transcripts found. Listing files in SADA dir:")
        for f in sorted(os.listdir(sada_dir))[:20]:
            print(f"    {f}")
        return {"wer": 1.0, "cer": 1.0, "total": 0}

    paired = [(w, transcript_map[os.path.basename(w)]) for w in wav_files if os.path.basename(w) in transcript_map]
    np.random.seed(42)
    np.random.shuffle(paired)
    n = min(max_samples, len(paired))
    print(f"  Paired samples: {len(paired)}, evaluating {n}")

    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    wer_scores, cer_scores = [], []

    with torch.no_grad():
        for wav_path, text in tqdm(paired[:n], desc="  SADA", ncols=100, leave=False):
            try:
                reference = normalize_arabic(text.strip())
                if not reference: continue
                wav, sr = sf.read(wav_path)
                wav = np.array(wav, dtype=np.float32)
                if sr != 16000:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                pred_ids = safe_generate(model, inputs, gen_config, device)
                prediction = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                w = wer(reference, prediction) if prediction else 1.0
                c = cer(reference, prediction) if prediction else 1.0
                wer_scores.append(w); cer_scores.append(c)
            except: continue

    results = {"wer": float(np.mean(wer_scores)) if wer_scores else 1.0, "cer": float(np.mean(cer_scores)) if cer_scores else 1.0, "total": len(wer_scores)}
    print(f"  SADA: WER={results['wer']:.4f} (n={results['total']})")
    model.train()
    return results

# =============================================================================
# Registry + runner
# =============================================================================
BENCHMARKS = {
    "fleurs":       benchmark_fleurs,
    "mgb3":         benchmark_mgb3,
    "casablanca":   benchmark_casablanca,
    "mgb2":         benchmark_mgb2,
    "masc_clean":   lambda m, p, d, n: benchmark_masc(m, p, d, n, split="test_clean"),
    "masc_noisy":   lambda m, p, d, n: benchmark_masc(m, p, d, n, split="test_noisy"),
    "common_voice": benchmark_common_voice,
    "sada":         benchmark_sada,
}

def run_benchmarks(model, processor, device, max_samples, names):
    results = {}
    for name in names:
        if name in BENCHMARKS:
            results[name] = BENCHMARKS[name](model, processor, device, max_samples)

    print(f"\n  {'='*70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"  {'='*70}")
    print(f"  {'Benchmark':<25} {'WER':<10} {'CER':<10} {'Samples'}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"  {name:<25} {r.get('wer',1.0):<10.4f} {r.get('cer',1.0):<10.4f} {r.get('total',0)}")

    valid = [r["wer"] for r in results.values() if isinstance(r, dict) and r.get("total", 0) > 0]
    avg = float(np.mean(valid)) if valid else 1.0
    print(f"  {'-'*55}")
    print(f"  {'AVERAGE':<25} {avg:<10.4f}")
    print(f"  {'='*70}")
    results["average_wer"] = avg
    return results

# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark Arabic ASR models on A100")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--merged-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Only FLEURS + MGB-3 + Casablanca")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.benchmarks:
        names = args.benchmarks
    elif args.quick:
        names = ["fleurs", "mgb3", "casablanca"]
    else:
        names = list(BENCHMARKS.keys())

    print(f"\n{'='*70}")
    print(f"ARABIC ASR BENCHMARK SUITE")
    print(f"{'='*70}")
    print(f"Benchmarks: {names}")
    print(f"Samples per benchmark: {args.samples}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    processor = WhisperProcessor.from_pretrained(args.base_model)

    # --- Base model ---
    base_results = None
    if not args.skip_base:
        print(f"\n{'='*70}")
        print(f"BASELINE: {args.base_model}")
        print(f"{'='*70}")
        base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
        base_model.eval()
        base_results = run_benchmarks(base_model, processor, device, args.samples, names)
        del base_model
        torch.cuda.empty_cache()

    if args.base_only:
        return

    # --- Trained model ---
    if args.merged_path:
        print(f"\n{'='*70}")
        print(f"TRAINED (MERGED): {args.merged_path}")
        print(f"{'='*70}")
        t_processor = WhisperProcessor.from_pretrained(args.merged_path)
        t_model = WhisperForConditionalGeneration.from_pretrained(args.merged_path).to(device)
        t_model.eval()
    elif args.lora_path:
        print(f"\n{'='*70}")
        print(f"TRAINED (LoRA): {args.lora_path}")
        print(f"{'='*70}")
        t_processor = processor
        base = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
        t_model = PeftModel.from_pretrained(base, args.lora_path).to(device)
        t_model.eval()
    else:
        print("No --lora-path or --merged-path provided.")
        return

    trained_results = run_benchmarks(t_model, t_processor, device, args.samples, names)
    del t_model
    torch.cuda.empty_cache()

    # --- Comparison ---
    if base_results:
        print(f"\n  {'='*70}")
        print(f"  COMPARISON: Whisper Large V3 (base) vs Your LoRA")
        print(f"  {'='*70}")
        print(f"  {'Benchmark':<25} {'Base WER':>10} {'LoRA WER':>10} {'Delta':>10} {'Result':>10}")
        print(f"  {'-'*65}")
        for name in names:
            b = base_results.get(name, {})
            t = trained_results.get(name, {})
            if isinstance(b, dict) and isinstance(t, dict):
                bw = b.get("wer")
                tw = t.get("wer")
                if bw is not None and tw is not None:
                    delta = bw - tw
                    tag = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
                    pct = (delta / bw * 100) if bw > 0 else 0
                    print(f"  {name:<25} {bw:>10.4f} {tw:>10.4f} {delta:>+10.4f} {tag:>7} ({pct:+.1f}%)")
        b_avg = base_results.get("average_wer")
        t_avg = trained_results.get("average_wer")
        if b_avg and t_avg:
            delta = b_avg - t_avg
            tag = "BETTER" if delta > 0 else "WORSE"
            pct = (delta / b_avg * 100) if b_avg > 0 else 0
            print(f"  {'-'*65}")
            print(f"  {'AVERAGE':<25} {b_avg:>10.4f} {t_avg:>10.4f} {delta:>+10.4f} {tag:>7} ({pct:+.1f}%)")
        print(f"  {'='*70}")

        # Leaderboard context
        print(f"\n  {'='*70}")
        print(f"  LEADERBOARD CONTEXT (Open Universal Arabic ASR Leaderboard)")
        print(f"  {'='*70}")
        print(f"  #1 Nvidia Conformer-CTC + 4gram LM:  25.71% avg WER")
        print(f"  #2 Whisper Large V3:                  ~35% avg WER")
        print(f"  Your model:                           {t_avg*100:.2f}% avg WER")
        print(f"  {'='*70}")

if __name__ == "__main__":
    main()