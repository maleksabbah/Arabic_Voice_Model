import argparse
import torch
import torchaudio
import numpy as np
from jiwer import wer, cer
from tqdm import tqdm
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")

def strip_diacritics(text):
    return "".join(c for c in text if c not in DIACRITICS)


def benchmark(model, processor, device, max_samples=100):
    dataset = load_dataset("MightyStudent/Egyptian-ASR-MGB-3", split="train", trust_remote_code=True)

    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    samples = min(max_samples, len(dataset))

    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}

    wer_scores = []
    cer_scores = []

    with torch.no_grad():
        for idx in tqdm(indices[:samples], desc="Benchmarking", ncols=100):
            try:
                sample = dataset[idx]
                audio = sample["audio"]
                wav = torch.tensor(audio["array"]).float()
                sr = audio["sampling_rate"]

                reference = strip_diacritics(sample["sentence"].strip())
                if not reference:
                    continue

                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav.unsqueeze(0)).squeeze(0)
                    sr = 16000

                inputs = processor(wav.numpy(), sampling_rate=sr, return_tensors="pt").to(device)
                pred_ids = model.generate(inputs.input_features, **gen_config)
                prediction = strip_diacritics(
                    processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                )

                try:
                    wer_scores.append(wer(reference, prediction))
                    cer_scores.append(cer(reference, prediction))
                except:
                    wer_scores.append(1.0)
                    cer_scores.append(1.0)
            except:
                continue

    return {
        "wer_mean": float(np.mean(wer_scores)) if wer_scores else 1.0,
        "cer_mean": float(np.mean(cer_scores)) if cer_scores else 1.0,
        "perfect": sum(1 for w in wer_scores if w == 0.0),
        "total": len(wer_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark base vs LoRA on Egyptian MGB-3")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA adapter (e.g. checkpoints/lora_011_.../best)")
    parser.add_argument("--base-model", type=str, default="openai/whisper-small", help="Base model name")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Benchmark base Whisper Small
    print("=" * 70)
    print("BENCHMARKING BASE WHISPER SMALL")
    print("=" * 70)
    processor = WhisperProcessor.from_pretrained(args.base_model)
    base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
    base_results = benchmark(base_model, processor, device, args.samples)
    print(f"  WER={base_results['wer_mean']:.4f}, CER={base_results['cer_mean']:.4f}, Perfect={base_results['perfect']}/{base_results['total']}")
    del base_model
    torch.cuda.empty_cache()

    # 2. Benchmark LoRA model
    print("\n" + "=" * 70)
    print(f"BENCHMARKING LORA: {args.lora_path}")
    print("=" * 70)
    lora_base = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, args.lora_path).to(device)
    lora_results = benchmark(lora_model, processor, device, args.samples)
    print(f"  WER={lora_results['wer_mean']:.4f}, CER={lora_results['cer_mean']:.4f}, Perfect={lora_results['perfect']}/{lora_results['total']}")

    # 3. Compare
    delta = base_results["wer_mean"] - lora_results["wer_mean"]
    status = "BETTER" if delta > 0 else "WORSE"
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Base Whisper Small:  WER={base_results['wer_mean']:.4f}  CER={base_results['cer_mean']:.4f}")
    print(f"  LoRA model:          WER={lora_results['wer_mean']:.4f}  CER={lora_results['cer_mean']:.4f}")
    print(f"  Delta:               {delta:+.4f} ({status})")
    print("=" * 70)


if __name__ == "__main__":
    main()