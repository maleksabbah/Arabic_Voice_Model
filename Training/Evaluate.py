"""
Evaluate a trained LoRA checkpoint on MGB-3, FLEURS, and Casablanca benchmarks.

Usage:
  python Training/evaluate.py --checkpoint /workspace/checkpoints/lora_epoch_3 --max-samples 200
  python Training/evaluate.py --checkpoint /workspace/checkpoints/lora_epoch_3 --benchmark mgb3
  python Training/evaluate.py --checkpoint /workspace/checkpoints/lora_epoch_3 --benchmark all
"""
import argparse
import time
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset
from jiwer import wer as compute_wer


def load_model(checkpoint_path):
    print(f"[MODEL] Loading base whisper-large-v3...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16,
    )
    if checkpoint_path:
        print(f"[MODEL] Loading LoRA from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to("cuda")
    model.eval()
    print("[MODEL] Ready")
    return model, processor


def transcribe(model, processor, audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
    with torch.no_grad():
        ids = model.generate(input_features=inputs, language="ar", task="transcribe", max_new_tokens=440)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def run_benchmark(model, processor, name, dataset_id, config, split, ref_field, max_samples):
    print(f"\n[BENCH] {name}...", flush=True)
    t0 = time.time()
    try:
        if config:
            ds = load_dataset(dataset_id, config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_id, split=split, streaming=True)

        refs, hyps = [], []
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            hyp = transcribe(model, processor, audio)
            refs.append(sample[ref_field])
            hyps.append(hyp)

            if (i + 1) % 50 == 0:
                current_wer = compute_wer(refs, hyps)
                print(f"  [{i+1}/{max_samples}] WER so far: {current_wer:.2%}", flush=True)

        w = compute_wer(refs, hyps) if refs else 1.0
        elapsed = time.time() - t0
        print(f"  {name}: WER = {w:.2%} ({len(refs)} samples, {elapsed/60:.1f}m)")
        return w
    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="/workspace/checkpoints/lora_epoch_3")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--benchmark", choices=["all", "mgb3", "fleurs", "casablanca", "baseline"], default="all")
    a = p.parse_args()

    if a.benchmark == "baseline":
        model, processor = load_model(None)
        print("\n=== BASELINE (no LoRA) ===")
    else:
        model, processor = load_model(a.checkpoint)
        print(f"\n=== EVALUATING {a.checkpoint} ===")

    benchmarks = {
        "mgb3": ("MGB-3 (Egyptian)", "MightyStudent/Egyptian-ASR-MGB-3", None, "train", "sentence"),
        "fleurs": ("FLEURS (MSA)", "google/fleurs", "ar_eg", "test", "transcription"),
        "casablanca": ("Casablanca (Egyptian)", "UBC-NLP/Casablanca", "Egypt", "test", "transcription"),
    }

    results = {}
    if a.benchmark == "all" or a.benchmark == "baseline":
        for key, (name, ds_id, config, split, ref_field) in benchmarks.items():
            results[key] = run_benchmark(model, processor, name, ds_id, config, split, ref_field, a.max_samples)
    else:
        name, ds_id, config, split, ref_field = benchmarks[a.benchmark]
        results[a.benchmark] = run_benchmark(model, processor, name, ds_id, config, split, ref_field, a.max_samples)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for key, val in results.items():
        print(f"  {key}: {val:.2%}" if val is not None else f"  {key}: FAILED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()