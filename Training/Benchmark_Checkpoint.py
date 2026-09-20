"""
Benchmark-only: runs the Training_A100.py benchmark suite against a checkpoint.

The benchmark functions in Training_A100.py are nested inside main() and only
run after epoch 3, so a run that dies early never produces numbers. This does
the same three benchmarks against any saved adapter or merged model.

Usage:
  python Training/Benchmark_Checkpoint.py --lora-path /workspace/checkpoints/<run>/step_2_2500
  python Training/Benchmark_Checkpoint.py --merged-path /workspace/checkpoints/<run>/merged
  python Training/Benchmark_Checkpoint.py --base-only          # base Whisper, for the before/after column
"""
import argparse, json, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import wer, cer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Training.Training_A100 import normalize_arabic

CASABLANCA_DIALECTS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]


def process_audio(sample):
    audio = sample["audio"]
    wav = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    return wav, sr


def safe_generate(m, inputs, gen_config, device):
    with torch.amp.autocast('cuda'):
        return m.generate(inputs.input_features.to(device), **gen_config)


def score_dataset(m, proc, device, ds, ref_field, n, desc):
    gen_config = {"language": "ar", "task": "transcribe"}
    indices = list(range(len(ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    n = min(n, len(ds))
    wer_scores, cer_scores = [], []
    with torch.no_grad():
        for idx in tqdm(indices[:n], desc=desc, ncols=100, leave=False):
            try:
                sample = ds[idx]
                if callable(ref_field):
                    raw = ref_field(sample)
                else:
                    raw = sample[ref_field]
                reference = normalize_arabic((raw or "").strip())
                if not reference:
                    continue
                wav, sr = process_audio(sample)
                inputs = proc(wav, sampling_rate=sr, return_tensors="pt").to(device)
                pred_ids = safe_generate(m, inputs, gen_config, device)
                prediction = normalize_arabic(proc.batch_decode(pred_ids, skip_special_tokens=True)[0].strip())
                wer_scores.append(wer(reference, prediction) if prediction else 1.0)
                cer_scores.append(cer(reference, prediction) if prediction else 1.0)
            except Exception:
                continue
    return wer_scores, cer_scores


def run_benchmark(m, proc, device, name, dataset_id, config, split, ref_field, max_samples=200):
    print(f"\n  [BENCH] {name}...")
    m.eval()
    try:
        ds = (load_dataset(dataset_id, config, split=split, trust_remote_code=True) if config
              else load_dataset(dataset_id, split=split, trust_remote_code=True))
        w, c = score_dataset(m, proc, device, ds, ref_field, max_samples, f"  {name}")
        r = {"wer": float(np.mean(w)) if w else 1.0, "cer": float(np.mean(c)) if c else 1.0, "total": len(w)}
        print(f"  {name}: WER={r['wer']:.4f}, CER={r['cer']:.4f} (n={r['total']})")
        return r
    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        return None


def run_casablanca(m, proc, device, max_samples=200):
    print(f"\n  [BENCH] Casablanca (8 dialects)...")
    m.eval()
    per_dialect, all_wer, all_cer = {}, [], []
    samples_per = max(max_samples // len(CASABLANCA_DIALECTS), 10)
    ref = lambda s: s.get("transcription", s.get("sentence", ""))
    for dialect in CASABLANCA_DIALECTS:
        try:
            ds = load_dataset("UBC-NLP/Casablanca", dialect, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Skipping {dialect}: {e}")
            continue
        w, c = score_dataset(m, proc, device, ds, ref, samples_per, f"  Casa-{dialect[:3]}")
        if w:
            per_dialect[dialect] = {"wer": float(np.mean(w)), "cer": float(np.mean(c)), "total": len(w)}
            all_wer.extend(w)
            all_cer.extend(c)
            print(f"  Casablanca-{dialect}: WER={per_dialect[dialect]['wer']:.4f} (n={len(w)})")
    r = {"wer": float(np.mean(all_wer)) if all_wer else 1.0,
         "cer": float(np.mean(all_cer)) if all_cer else 1.0,
         "total": len(all_wer), "per_dialect": per_dialect}
    print(f"  Casablanca AVG: WER={r['wer']:.4f} ({r['total']} samples)")
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--merged-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--base-only", action="store_true", help="Score the base model with no adapter")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--out", type=str, default=None, help="Write results JSON here")
    args = parser.parse_args()

    if not (args.lora_path or args.merged_path or args.base_only):
        parser.error("pass one of --lora-path, --merged-path, --base-only")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    processor = WhisperProcessor.from_pretrained(args.base_model)
    if args.merged_path:
        print(f"Loading merged model: {args.merged_path}")
        model = WhisperForConditionalGeneration.from_pretrained(args.merged_path).to(device)
        label = args.merged_path
    else:
        print(f"Loading base model: {args.base_model}")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
        label = args.base_model
        if args.lora_path:
            print(f"Applying adapter: {args.lora_path}")
            model = PeftModel.from_pretrained(model, args.lora_path).to(device)
            print("Merging adapter into base weights...")
            model = model.merge_and_unload()
            label = args.lora_path
    model.eval()

    results = {"checkpoint": label, "max_samples": args.max_samples, "timestamp": datetime.now().isoformat()}
    results["fleurs"] = run_benchmark(model, processor, device, "FLEURS (MSA)", "google/fleurs", "ar_eg", "test", "transcription", args.max_samples)
    results["mgb3"] = run_benchmark(model, processor, device, "MGB-3 (Egyptian)", "MightyStudent/Egyptian-ASR-MGB-3", None, "train", "sentence", args.max_samples)
    results["casablanca"] = run_casablanca(model, processor, device, args.max_samples)

    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS — {label}")
    print(f"{'='*80}")
    for name in ("fleurs", "mgb3", "casablanca"):
        r = results.get(name)
        if r:
            print(f"  {name}: WER={r['wer']:.4f}, CER={r['cer']:.4f} (n={r['total']})")
    valid = [results[n]["wer"] for n in ("fleurs", "mgb3", "casablanca") if results.get(n) and results[n].get("total", 0) > 0]
    results["average_wer"] = float(np.mean(valid)) if valid else 1.0
    print(f"  AVERAGE: {results['average_wer']:.4f}")
    print(f"{'='*80}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
