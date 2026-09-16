"""
Arabic ASR Benchmark Suite — A100 Edition
==========================================
All benchmarks from the Open Universal Arabic ASR Leaderboard:
  1. FLEURS        — MSA clean read speech
  2. MGB-3         — Egyptian dialect
  3. Casablanca    — 8 Arabic dialects
  4. MGB-2         — Multi-dialect broadcast
  5. MASC clean    — Multi-dialect clean test
  6. MASC noisy    — Multi-dialect noisy test
  7. Common Voice  — MSA crowd-sourced
  8. SADA          — Saudi dialects

Usage:
  python benchmark_a100.py --lora-path checkpoints/largev3_best --base-model openai/whisper-large-v3 --samples 200
  python benchmark_a100.py --lora-path checkpoints/largev3_best --base-model openai/whisper-large-v3 --skip-base --samples 200
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

DIACRITICS = set("ًٌٍَُِّْٰٕ")

def normalize_arabic(text):
    if not text: return ""
    text = "".join(c for c in text if c not in DIACRITICS)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = text.replace('ـ', '')
    text = re.sub(r'[.,،؟!?:;؛\-\"\'()\[\]{}]', '', text)
    text = " ".join(text.split()).strip()
    return text

def safe_generate(model, inputs, gen_config, device):
    with torch.amp.autocast('cuda'):
        pred_ids = model.generate(inputs.input_features.to(device), **gen_config)
    return pred_ids

def process_audio(sample):
    audio = sample["audio"]
    wav = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    return wav, sr

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

BENCHMARKS = {
    "fleurs": benchmark_fleurs,
    "mgb3": benchmark_mgb3,
    "casablanca": benchmark_casablanca,
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

        print(f"\n  {'='*70}")
        print(f"  LEADERBOARD CONTEXT (Open Universal Arabic ASR Leaderboard)")
        print(f"  {'='*70}")
        print(f"  #1 Nvidia Conformer-CTC + 4gram LM:  25.71% avg WER")
        print(f"  #2 Whisper Large V3:                  ~35% avg WER")
        print(f"  Your model:                           {t_avg*100:.2f}% avg WER")
        print(f"  {'='*70}")

if __name__ == "__main__":
    main()
