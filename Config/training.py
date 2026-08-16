import argparse
import torch
import torchaudio
import numpy as np
from jiwer import wer, cer
from tqdm import tqdm
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from Config.Database import get_db
from Training.Model import Episode
from Training.ModelTraining import TrainingService


# ============================================================================
# Diacritics stripping
# ============================================================================
DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")


def strip_diacritics(text):
    return "".join(c for c in text if c not in DIACRITICS)


# ============================================================================
# Unseen Egyptian benchmark (MightyStudent/Egyptian-ASR-MGB-3)
# ============================================================================
def benchmark_unseen_egyptian(model, processor, device, max_samples=100):
    """Benchmark on completely unseen Egyptian Arabic audio (MGB-3).
    Returns WER, CER, and number of perfect predictions."""

    print("\n  [BENCHMARK] Loading unseen Egyptian test set (MGB-3)...")
    dataset = load_dataset("MightyStudent/Egyptian-ASR-MGB-3", split="train", trust_remote_code=True)

    # Shuffle and take a subset
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    samples = min(max_samples, len(dataset))

    model.eval()
    gen_config = {"language": "ar", "task": "transcribe"}

    wer_scores = []
    cer_scores = []

    with torch.no_grad():
        for idx in tqdm(indices[:samples], desc="  [BENCHMARK] Egyptian", ncols=100, leave=False):
            try:
                sample = dataset[idx]
                audio = sample["audio"]
                wav = torch.tensor(audio["array"]).float()
                sr = audio["sampling_rate"]

                reference = strip_diacritics(sample["sentence"].strip())
                if not reference:
                    continue

                # Resample if needed
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

    results = {
        "wer_mean": float(np.mean(wer_scores)) if wer_scores else 1.0,
        "wer_median": float(np.median(wer_scores)) if wer_scores else 1.0,
        "cer_mean": float(np.mean(cer_scores)) if cer_scores else 1.0,
        "perfect": sum(1 for w in wer_scores if w == 0.0),
        "total": len(wer_scores),
    }

    model.train()
    return results


# ============================================================================
# Base Whisper Small benchmark (run once at start for comparison)
# ============================================================================
def benchmark_base_whisper(device, max_samples=100):
    """Get base Whisper Small WER on unseen Egyptian for comparison."""
    print("\n" + "=" * 80)
    print("BASELINE: Benchmarking base Whisper Small on unseen Egyptian (MGB-3)...")
    print("=" * 80)

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    model.eval()

    results = benchmark_unseen_egyptian(model, processor, device, max_samples)

    print(f"  Base Whisper Small: WER={results['wer_mean']:.4f}, CER={results['cer_mean']:.4f}, Perfect={results['perfect']}/{results['total']}")

    del model
    torch.cuda.empty_cache()

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Whisper model on Arabic dialects")

    # Series arguments
    parser.add_argument("--series", type=str, action="append",
                        help="Series config as 'id:ratio' (e.g., --series 1:1.0 --series 2:0.5)")

    # Episode filters
    parser.add_argument("--episodes", type=str, nargs="+",
                        help="Filter by episode names")
    parser.add_argument("--episode-ids", type=int, nargs="+",
                        help="Filter by episode IDs")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")

    # Model arguments
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--source-model", type=str,
                        default="openai/whisper-small",
                        help="Source model (default: openai/whisper-small)")

    # LoRA arguments
    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA training instead of full fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    parser.add_argument("--lora-lr", type=float, default=1e-4,
                        help="Learning rate for LoRA (default: 1e-4)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max total training samples (default: use all)")
    parser.add_argument("--target-modules", type=str, nargs="+",
                        default=["q_proj", "v_proj", "k_proj", "o_proj"],
                        help="LoRA target modules (default: q_proj v_proj k_proj o_proj)")

    # Benchmark arguments
    parser.add_argument("--benchmark-samples", type=int, default=100,
                        help="Number of unseen Egyptian samples for benchmark (default: 100)")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip unseen Egyptian benchmark after each epoch")
    parser.add_argument("--skip-base-benchmark", action="store_true",
                        help="Skip base Whisper Small benchmark at start")

    args = parser.parse_args()

    # Parse series configs
    if args.series:
        series_configs = []
        for s in args.series:
            parts = s.split(":")
            series_id = int(parts[0])
            ratio = float(parts[1]) if len(parts) > 1 else 1.0
            series_configs.append({"series_id": series_id, "ratio": ratio})
    else:
        # Default: all Lebanese/Syrian data (Series 1-4 + MASC Leb/Syr from 5)
        series_configs = [
            {"series_id": 1, "ratio": 1.0},
            {"series_id": 2, "ratio": 1.0},
            {"series_id": 3, "ratio": 1.0},
            {"series_id": 4, "ratio": 1.0},
        ]

    db = next(get_db())

    # Attach episode name filter to series configs
    if args.episodes:
        episodes = db.query(Episode).filter(Episode.name.in_(args.episodes)).all()
        if not episodes:
            print(f"No episodes found matching: {args.episodes}")
            return
        print(f"  Episodes by name: {[(e.id, e.name) for e in episodes]}")
        for config in series_configs:
            config["episode_names"] = args.episodes

    # Resolve episode ID filter
    episode_ids = []
    if args.episode_ids:
        episode_ids.extend(args.episode_ids)

    # =========================================================================
    # Print config
    # =========================================================================
    print("=" * 80)
    print("LEVANTINE LoRA TRAINING")
    print("=" * 80)
    print(f"  Mode:           {'LoRA' if args.lora else 'Full fine-tuning'}")
    print(f"  Source model:    {args.source_model}")
    print(f"  Series:          {series_configs}")
    print(f"  Epochs:          {args.epochs}")
    if args.episodes:
        print(f"  Episode filter:  {args.episodes}")
    if episode_ids:
        print(f"  Episode IDs:     {episode_ids}")
    if args.lora:
        print(f"  LoRA rank:       {args.lora_rank}")
        print(f"  LoRA alpha:      {args.lora_alpha}")
        print(f"  LoRA dropout:    {args.lora_dropout}")
        print(f"  LoRA LR:         {args.lora_lr}")
        print(f"  Target modules:  {args.target_modules}")
    print(f"  Benchmark:       {'ON' if not args.skip_benchmark else 'OFF'} ({args.benchmark_samples} samples)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Step 1: Benchmark base Whisper Small (our target to beat)
    # =========================================================================
    base_results = None
    if not args.skip_base_benchmark:
        base_results = benchmark_base_whisper(device, args.benchmark_samples)

    # =========================================================================
    # Step 2: Create trainer and run training
    # =========================================================================
    trainer = TrainingService(
        db=db,
        checkpoints_dir=args.checkpoints_dir,
        source_model=args.source_model
    )

    # Store benchmark config for use during training
    trainer._benchmark_config = {
        "enabled": not args.skip_benchmark,
        "max_samples": args.benchmark_samples,
        "base_results": base_results,
        "epoch_benchmarks": [],
    }

    if args.lora:
        # Monkey-patch _evaluate_comprehensive to add Egyptian benchmark
        original_eval = trainer._evaluate_comprehensive

        def eval_with_benchmark(dataset):
            # Run original validation on training data
            val_metrics, detailed_results = original_eval(dataset)

            # Run unseen Egyptian benchmark
            if trainer._benchmark_config["enabled"]:
                print("\n  Running unseen Egyptian benchmark...")
                bench_results = benchmark_unseen_egyptian(
                    trainer.model, trainer.processor, trainer.device,
                    trainer._benchmark_config["max_samples"]
                )

                # Compare to base
                base = trainer._benchmark_config["base_results"]
                if base:
                    delta = base["wer_mean"] - bench_results["wer_mean"]
                    status = "BETTER" if delta > 0 else "WORSE"
                    print(f"\n  {'=' * 60}")
                    print(f"  UNSEEN EGYPTIAN BENCHMARK")
                    print(f"  {'=' * 60}")
                    print(f"  Base Whisper Small:  WER={base['wer_mean']:.4f}")
                    print(f"  Current LoRA model:  WER={bench_results['wer_mean']:.4f}")
                    print(f"  Delta:               {delta:+.4f} ({status} than base)")
                    print(f"  {'=' * 60}")
                else:
                    print(f"\n  Unseen Egyptian WER: {bench_results['wer_mean']:.4f}")

                trainer._benchmark_config["epoch_benchmarks"].append(bench_results)
                val_metrics["unseen_egyptian"] = bench_results

            return val_metrics, detailed_results

        trainer._evaluate_comprehensive = eval_with_benchmark

        # Run LoRA training
        result = trainer.train_lora(
            series_configs=series_configs,
            epochs=args.epochs,
            learning_rate=args.lora_lr,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            episode_ids=episode_ids if episode_ids else None,
            max_samples=args.max_samples,
        )

        # Print final benchmark summary
        benchmarks = trainer._benchmark_config["epoch_benchmarks"]
        if benchmarks:
            print("\n" + "=" * 80)
            print("UNSEEN EGYPTIAN BENCHMARK PROGRESSION")
            print("=" * 80)
            base = trainer._benchmark_config["base_results"]
            if base:
                print(f"  {'Epoch':<10} {'WER':<12} {'CER':<12} {'vs Base':<15} {'Status'}")
                print(f"  {'-' * 60}")
                print(f"  {'Base':<10} {base['wer_mean']:<12.4f} {base['cer_mean']:<12.4f} {'---':<15} {'BASELINE'}")
                for i, b in enumerate(benchmarks):
                    delta = base["wer_mean"] - b["wer_mean"]
                    status = "BETTER" if delta > 0 else "WORSE"
                    print(f"  {i+1:<10} {b['wer_mean']:<12.4f} {b['cer_mean']:<12.4f} {delta:+.4f}{'':>7} {status}")

            # Best epoch for unseen data
            best_epoch = min(range(len(benchmarks)), key=lambda i: benchmarks[i]["wer_mean"])
            print(f"\n  Best unseen Egyptian WER: Epoch {best_epoch+1} ({benchmarks[best_epoch]['wer_mean']:.4f})")

            if base:
                best_delta = base["wer_mean"] - benchmarks[best_epoch]["wer_mean"]
                if best_delta > 0:
                    print(f"  RESULT: LoRA BEATS base Whisper Small by {best_delta:.4f} WER")
                else:
                    print(f"  RESULT: LoRA is {abs(best_delta):.4f} WER WORSE than base Whisper Small")
                    print(f"  Consider: fewer epochs, lower LR, or more training data")

            print("=" * 80)

    else:
        trainer.train(
            series_configs=series_configs,
            epochs=args.epochs,
            episode_ids=episode_ids if episode_ids else None,
            max_samples = args.max_samples,
        )


if __name__ == "__main__":
    main()