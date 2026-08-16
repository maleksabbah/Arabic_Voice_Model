"""
Decoder Swap Experiment

Takes your fine-tuned encoder (good at Arabic dialects) and pairs it
with base Whisper Small's decoder (general Arabic knowledge intact).
Then benchmarks against both the full fine-tuned model and base Whisper Small.
"""

import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from jiwer import wer, cer
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Buckwalter to Arabic mapping
BW2AR = {
    "'": "\u0621", "|": "\u0622", ">": "\u0623", "&": "\u0624",
    "<": "\u0625", "}": "\u0626", "A": "\u0627", "b": "\u0628",
    "p": "\u0629", "t": "\u062A", "v": "\u062B", "j": "\u062C",
    "H": "\u062D", "x": "\u062E", "d": "\u062F", "*": "\u0630",
    "r": "\u0631", "z": "\u0632", "s": "\u0633", "$": "\u0634",
    "S": "\u0635", "D": "\u0636", "T": "\u0637", "Z": "\u0638",
    "E": "\u0639", "g": "\u063A", "_": "\u0640", "f": "\u0641",
    "q": "\u0642", "k": "\u0643", "l": "\u0644", "m": "\u0645",
    "n": "\u0646", "h": "\u0647", "w": "\u0648", "Y": "\u0649",
    "y": "\u064A", "F": "\u064B", "N": "\u064C", "K": "\u064D",
    "a": "\u064E", "u": "\u064F", "i": "\u0650", "~": "\u0651",
    "o": "\u0652", "`": "\u0670", "{": "\u0671", "^": "\u0655",
}

def buckwalter_to_arabic(text):
    return "".join(BW2AR.get(c, c) for c in text)

def strip_diacritics(text):
    diacritics = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")
    return "".join(c for c in text if c not in diacritics)


def evaluate(model, processor, dataset, device, max_samples, label):
    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}
    results = []

    with torch.no_grad():
        for i in tqdm(range(min(max_samples, len(dataset))), desc=f"  {label}", ncols=100):
            try:
                sample = dataset[i]
                audio = sample["audio"]
                wav = torch.tensor(audio["array"]).float()
                sr = audio["sampling_rate"]

                reference = strip_diacritics(buckwalter_to_arabic(sample["orthographic"]))

                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav.unsqueeze(0)).squeeze(0)
                    sr = 16000

                inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                pred_ids = model.generate(inputs.input_features, **gen_config)
                prediction = strip_diacritics(
                    processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                )

                try:
                    w = wer(reference, prediction)
                    c = cer(reference, prediction)
                except:
                    w, c = 1.0, 1.0

                results.append({"reference": reference, "prediction": prediction, "wer": w, "cer": c})
            except Exception as e:
                continue

    wer_scores = [r["wer"] for r in results]
    return {
        "results": results,
        "wer_mean": float(np.mean(wer_scores)),
        "wer_median": float(np.median(wer_scores)),
        "cer_mean": float(np.mean([r["cer"] for r in results])),
        "perfect": sum(1 for w in wer_scores if w == 0.0),
        "total": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description="Decoder swap experiment")
    parser.add_argument("--finetuned", type=str, required=True, help="Your fine-tuned model path")
    parser.add_argument("--base", type=str, default="openai/whisper-small", help="Base Whisper model")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    dataset = load_dataset("halabi2016/arabic_speech_corpus", split="test", trust_remote_code=True)
    print(f"  {len(dataset)} samples\n")

    # ========================================
    # Model 1: Your fine-tuned model (as-is)
    # ========================================
    print("=" * 80)
    print("MODEL 1: Fine-tuned (your encoder + your decoder)")
    print("=" * 80)
    processor_ft = WhisperProcessor.from_pretrained(args.finetuned)
    model_ft = WhisperForConditionalGeneration.from_pretrained(args.finetuned).to(device)
    results_ft = evaluate(model_ft, processor_ft, dataset, device, args.max_samples, "Fine-tuned")
    print(f"  WER: {results_ft['wer_mean']:.4f}")
    del model_ft
    torch.cuda.empty_cache()

    # ========================================
    # Model 2: Base Whisper Small (as-is)
    # ========================================
    print("\n" + "=" * 80)
    print("MODEL 2: Base Whisper Small (base encoder + base decoder)")
    print("=" * 80)
    processor_base = WhisperProcessor.from_pretrained(args.base)
    model_base = WhisperForConditionalGeneration.from_pretrained(args.base).to(device)
    results_base = evaluate(model_base, processor_base, dataset, device, args.max_samples, "Base")
    print(f"  WER: {results_base['wer_mean']:.4f}")

    # ========================================
    # Model 3: Your encoder + base decoder (HYBRID)
    # ========================================
    print("\n" + "=" * 80)
    print("MODEL 3: HYBRID (your encoder + base decoder)")
    print("=" * 80)
    model_hybrid = WhisperForConditionalGeneration.from_pretrained(args.finetuned).to(device)

    # Swap decoder from base model
    model_hybrid.model.decoder.load_state_dict(model_base.model.decoder.state_dict())
    # Also swap lm_head (the final output projection)
    model_hybrid.proj_out.load_state_dict(model_base.proj_out.state_dict())

    results_hybrid = evaluate(model_hybrid, processor_base, dataset, device, args.max_samples, "Hybrid")
    print(f"  WER: {results_hybrid['wer_mean']:.4f}")

    del model_base, model_hybrid
    torch.cuda.empty_cache()

    # ========================================
    # Model 4: Base encoder + your decoder
    # ========================================
    print("\n" + "=" * 80)
    print("MODEL 4: REVERSE HYBRID (base encoder + your decoder)")
    print("=" * 80)
    model_reverse = WhisperForConditionalGeneration.from_pretrained(args.base).to(device)

    # Swap encoder from fine-tuned
    model_ft_reload = WhisperForConditionalGeneration.from_pretrained(args.finetuned)
    model_reverse.model.encoder.load_state_dict(model_ft_reload.model.encoder.state_dict())
    del model_ft_reload

    model_reverse = model_reverse.to(device)
    results_reverse = evaluate(model_reverse, processor_base, dataset, device, args.max_samples, "Reverse")
    print(f"  WER: {results_reverse['wer_mean']:.4f}")

    del model_reverse
    torch.cuda.empty_cache()

    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'=' * 80}")
    print("DECODER SWAP EXPERIMENT RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Configuration':<45} {'WER':<10} {'CER':<10} {'Perfect'}")
    print(f"{'-' * 75}")
    print(f"{'Your encoder + your decoder (fine-tuned)':<45} {results_ft['wer_mean']:<10.4f} {results_ft['cer_mean']:<10.4f} {results_ft['perfect']}/{results_ft['total']}")
    print(f"{'Base encoder + base decoder (whisper-small)':<45} {results_base['wer_mean']:<10.4f} {results_base['cer_mean']:<10.4f} {results_base['perfect']}/{results_base['total']}")
    print(f"{'Your encoder + base decoder (HYBRID)':<45} {results_hybrid['wer_mean']:<10.4f} {results_hybrid['cer_mean']:<10.4f} {results_hybrid['perfect']}/{results_hybrid['total']}")
    print(f"{'Base encoder + your decoder (REVERSE)':<45} {results_reverse['wer_mean']:<10.4f} {results_reverse['cer_mean']:<10.4f} {results_reverse['perfect']}/{results_reverse['total']}")

    print(f"\nINTERPRETATION:")
    best = min(
        [("Fine-tuned", results_ft), ("Base", results_base), ("Hybrid", results_hybrid), ("Reverse", results_reverse)],
        key=lambda x: x[1]["wer_mean"]
    )
    print(f"  Best config: {best[0]} ({best[1]['wer_mean']:.4f} WER)")

    if results_hybrid["wer_mean"] < results_base["wer_mean"]:
        improvement = results_base["wer_mean"] - results_hybrid["wer_mean"]
        print(f"  Your encoder IMPROVES over base by {improvement:.4f} WER")
        print(f"  -> Fine-tuning DID teach the encoder useful Arabic features")
    else:
        print(f"  Your encoder does NOT improve over base")
        print(f"  -> Fine-tuning may have damaged the encoder too")


if __name__ == "__main__":
    main()