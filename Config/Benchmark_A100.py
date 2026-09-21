"""
Arabic ASR Benchmark Suite — All HuggingFace benchmarks
========================================================
Runs 5 of 8 Open Universal Arabic ASR Leaderboard benchmarks:
  1. FLEURS        — MSA clean read speech
  2. MGB-3         — Egyptian dialect
  3. MGB-2         — Multi-dialect broadcast (gated, needs HF login)
  4. Casablanca    — 8 Arabic dialects
  5. Common Voice  — MSA crowd-sourced (gated, needs HF login)

Usage:
  # Full suite against merged model:
  python benchmark_full.py --merged-path /workspace/checkpoints/lora_largev3_*/merged

  # Full suite against LoRA adapter:
  python benchmark_full.py --lora-path /workspace/checkpoints/lora_largev3_*/best

  # Skip base model (only test trained):
  python benchmark_full.py --merged-path /workspace/checkpoints/lora_largev3_*/merged --skip-base

  # Specific benchmarks only:
  python benchmark_full.py --merged-path /workspace/checkpoints/lora_largev3_*/merged --benchmarks fleurs mgb3

  # Login first:
  huggingface-cli login --token YOUR_TOKEN
"""
import argparse
import re
import os
import torch
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

def process_audio(sample):
    audio = sample["audio"]
    wav = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav, 16000

def run_single_benchmark(model, processor, device, name, dataset_id, config, split, ref_field, max_samples, trust_remote=True):
    print(f"\n  [BENCH] {name}...")
    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    try:
        if config:
            ds = load_dataset(dataset_id, config, split=split, trust_remote_code=trust_remote)
        else:
            ds = load_dataset(dataset_id, split=split, trust_remote_code=trust_remote)
        indices = list(range(len(ds)))
        np.random.seed(42)
        np.random.shuffle(indices)
        n = min(max_samples, len(ds))
        wer_scores, cer_scores = [], []
        all_refs, all_preds = [], []
        with torch.no_grad():
            for idx in tqdm(indices[:n], desc=f"  {name}", ncols=100, leave=False):
                try:
                    sample = ds[idx]
                    reference = normalize_arabic(sample[ref_field].strip())
                    if not reference: continue
                    wav, sr = process_audio(sample)
                    inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                    with torch.amp.autocast('cuda'):
                        pred_ids = model.generate(inputs.input_features.to(device), **gen_config)
                    prediction = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                    w = wer(reference, prediction) if prediction else 1.0
                    c = cer(reference, prediction) if prediction else 1.0
                    wer_scores.append(w)
                    cer_scores.append(c)
                    all_refs.append(reference)
                    all_preds.append(prediction)
                except Exception as e:
                    continue
        mean_wer = float(np.mean(wer_scores)) if wer_scores else 1.0
        mean_cer = float(np.mean(cer_scores)) if cer_scores else 1.0
        corpus_wer_val = wer(all_refs, all_preds) if all_refs else 1.0
        corpus_cer_val = cer(all_refs, all_preds) if all_refs else 1.0
        r = {
            "mean_wer": mean_wer, "mean_cer": mean_cer,
            "corpus_wer": corpus_wer_val, "corpus_cer": corpus_cer_val,
            "total": len(wer_scores)
        }
        print(f"  {name}: Mean WER={mean_wer:.4f} | Corpus WER={corpus_wer_val:.4f} | CER={mean_cer:.4f} (n={r['total']})")
        return r
    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        return None

CASABLANCA_DIALECTS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]

def run_casablanca(model, processor, device, max_samples=200):
    print(f"\n  [BENCH] Casablanca (8 dialects)...")
    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    per_dialect = {}
    all_wer, all_cer = [], []
    all_refs, all_preds = [], []
    samples_per = max(max_samples // len(CASABLANCA_DIALECTS), 10)
    for dialect in CASABLANCA_DIALECTS:
        try:
            ds = load_dataset("UBC-NLP/Casablanca", dialect, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Skipping {dialect}: {e}")
            continue
        indices = list(range(len(ds)))
        np.random.seed(42)
        np.random.shuffle(indices)
        n = min(samples_per, len(ds))
        wer_scores, cer_scores = [], []
        d_refs, d_preds = [], []
        with torch.no_grad():
            for idx in tqdm(indices[:n], desc=f"  Casa-{dialect[:3]}", ncols=100, leave=False):
                try:
                    sample = ds[idx]
                    reference = normalize_arabic(sample.get("transcription", sample.get("sentence", "")).strip())
                    if not reference: continue
                    wav, sr = process_audio(sample)
                    inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                    with torch.amp.autocast('cuda'):
                        pred_ids = model.generate(inputs.input_features.to(device), **gen_config)
                    prediction = normalize_arabic(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                    w = wer(reference, prediction) if prediction else 1.0
                    c = cer(reference, prediction) if prediction else 1.0
                    wer_scores.append(w)
                    cer_scores.append(c)
                    d_refs.append(reference)
                    d_preds.append(prediction)
                except:
                    continue
        if wer_scores:
            d_mean_wer = float(np.mean(wer_scores))
            d_corpus_wer = wer(d_refs, d_preds) if d_refs else 1.0
            per_dialect[dialect] = {"mean_wer": d_mean_wer, "corpus_wer": d_corpus_wer, "cer": float(np.mean(cer_scores)), "total": len(wer_scores)}
            all_wer.extend(wer_scores)
            all_cer.extend(cer_scores)
            all_refs.extend(d_refs)
            all_preds.extend(d_preds)
            print(f"  Casablanca-{dialect}: Mean WER={d_mean_wer:.4f} | Corpus WER={d_corpus_wer:.4f} (n={len(wer_scores)})")
    mean_wer = float(np.mean(all_wer)) if all_wer else 1.0
    corpus_wer_val = wer(all_refs, all_preds) if all_refs else 1.0
    r = {
        "mean_wer": mean_wer, "corpus_wer": corpus_wer_val,
        "mean_cer": float(np.mean(all_cer)) if all_cer else 1.0,
        "total": len(all_wer), "per_dialect": per_dialect
    }
    print(f"  Casablanca AVG: Mean WER={mean_wer:.4f} | Corpus WER={corpus_wer_val:.4f} ({r['total']} samples)")
    return r

ALL_BENCHMARKS = {
    "fleurs": {"name": "FLEURS (MSA)", "dataset": "google/fleurs", "config": "ar_eg", "split": "test", "ref": "transcription"},
    "mgb3": {"name": "MGB-3 (Egyptian)", "dataset": "MightyStudent/Egyptian-ASR-MGB-3", "config": None, "split": "train", "ref": "sentence"},
    "mgb2": {"name": "MGB-2 (Multi-dialect)", "dataset": "QCRI/mgb2", "config": None, "split": "test", "ref": "text"},
    "cv18": {"name": "Common Voice 18 (MSA)", "dataset": "mozilla-foundation/common_voice_18_0", "config": "ar", "split": "test", "ref": "sentence"},
}

def run_all(model, processor, device, max_samples, benchmark_names):
    results = {}
    for key in benchmark_names:
        if key == "casablanca":
            results["casablanca"] = run_casablanca(model, processor, device, max_samples)
        elif key in ALL_BENCHMARKS:
            b = ALL_BENCHMARKS[key]
            results[key] = run_single_benchmark(model, processor, device, b["name"], b["dataset"], b["config"], b["split"], b["ref"], max_samples)
        else:
            print(f"  Unknown benchmark: {key}")
    return results

def print_results(results, label):
    print(f"\n  {'='*80}")
    print(f"  {label}")
    print(f"  {'='*80}")
    print(f"  {'Benchmark':<30} {'Mean WER':<12} {'Corpus WER':<12} {'CER':<12} {'Samples'}")
    print(f"  {'-'*75}")
    for name, r in results.items():
        if r:
            mw = r.get('mean_wer', r.get('wer', 0))
            cw = r.get('corpus_wer', '-')
            mc = r.get('mean_cer', r.get('cer', 0))
            n = r.get('total', 0)
            cw_str = f"{cw:.4f}" if isinstance(cw, float) else cw
            print(f"  {name:<30} {mw:<12.4f} {cw_str:<12} {mc:<12.4f} {n}")
    valid_mean = [r["mean_wer"] for r in results.values() if r and r.get("total", 0) > 0]
    valid_corpus = [r["corpus_wer"] for r in results.values() if r and r.get("total", 0) > 0 and isinstance(r.get("corpus_wer"), float)]
    if valid_mean:
        print(f"  {'-'*75}")
        print(f"  {'AVERAGE':<30} {np.mean(valid_mean):<12.4f} {np.mean(valid_corpus) if valid_corpus else '-':<12} ")
    print(f"  {'='*80}")

def print_comparison(base_results, trained_results):
    print(f"\n  {'='*80}")
    print(f"  COMPARISON: Base vs Trained")
    print(f"  {'='*80}")
    print(f"  {'Benchmark':<25} {'Base MeanWER':>12} {'Trained MeanWER':>15} {'Delta':>10} {'Corpus Base':>12} {'Corpus Trained':>15}")
    print(f"  {'-'*90}")
    for name in trained_results:
        b = base_results.get(name, {})
        t = trained_results.get(name, {})
        if b and t:
            bw = b.get("mean_wer", 0)
            tw = t.get("mean_wer", 0)
            delta = bw - tw
            tag = "BETTER" if delta > 0 else "WORSE"
            bcw = b.get("corpus_wer", "-")
            tcw = t.get("corpus_wer", "-")
            bcw_str = f"{bcw:.4f}" if isinstance(bcw, float) else str(bcw)
            tcw_str = f"{tcw:.4f}" if isinstance(tcw, float) else str(tcw)
            print(f"  {name:<25} {bw:>12.4f} {tw:>15.4f} {delta:>+10.4f} {tag:>7}  {bcw_str:>12} {tcw_str:>15}")
    print(f"  {'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Arabic ASR Benchmark Suite")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--merged-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["fleurs", "mgb3", "mgb2", "casablanca", "cv18"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"ARABIC ASR BENCHMARK SUITE")
    print(f"{'='*80}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Samples per benchmark: {args.samples}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    processor = WhisperProcessor.from_pretrained(args.base_model)

    # Base model
    base_results = None
    if not args.skip_base:
        print(f"\n{'='*80}")
        print(f"BASELINE: {args.base_model}")
        print(f"{'='*80}")
        base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
        base_model.eval()
        base_results = run_all(base_model, processor, device, args.samples, args.benchmarks)
        print_results(base_results, f"BASELINE: {args.base_model}")
        del base_model
        torch.cuda.empty_cache()

    # Trained model
    if args.merged_path:
        print(f"\n{'='*80}")
        print(f"TRAINED (MERGED): {args.merged_path}")
        print(f"{'='*80}")
        t_processor = WhisperProcessor.from_pretrained(args.merged_path)
        t_model = WhisperForConditionalGeneration.from_pretrained(args.merged_path).to(device)
        t_model.eval()
    elif args.lora_path:
        print(f"\n{'='*80}")
        print(f"TRAINED (LoRA): {args.lora_path}")
        print(f"{'='*80}")
        t_processor = processor
        base = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
        t_model = PeftModel.from_pretrained(base, args.lora_path).to(device)
        t_model.eval()
    else:
        print("No --lora-path or --merged-path provided. Only base model benchmarked.")
        return

    trained_results = run_all(t_model, t_processor, device, args.samples, args.benchmarks)
    print_results(trained_results, "TRAINED MODEL")
    del t_model
    torch.cuda.empty_cache()

    if base_results:
        print_comparison(base_results, trained_results)

    # Leaderboard context
    print(f"\n  {'='*80}")
    print(f"  LEADERBOARD CONTEXT")
    print(f"  {'='*80}")
    print(f"  #1 Nvidia Conformer-CTC + 4gram LM:  25.71% avg WER")
    print(f"  #2 Whisper Large V3:                  ~35% avg WER")
    valid = [r["mean_wer"] for r in trained_results.values() if r and r.get("total", 0) > 0]
    if valid:
        print(f"  Your model:                           {np.mean(valid)*100:.2f}% avg WER")
    print(f"  {'='*80}")

if __name__ == "__main__":
    main()
