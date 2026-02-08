import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from tqdm import tqdm

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

from sqlalchemy.orm import Session

from DataLoader import DataLoader

# Try to load BERTScore
try:
    import evaluate

    BERTSCORE_AVAILABLE = True
except:
    BERTSCORE_AVAILABLE = False


class TrainingService:

    def __init__(
            self,
            db: Session,
            checkpoints_dir: str = "./checkpoints",
            source_model: str = "openai/whisper-small",
    ):

        self.db = db
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.source_model = source_model

        # Current run directory (created when training starts)
        self.run_dir = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.bertscore = None

        # Training config
        # Training config
        self.config = {
            "epochs": 3,
            "learning_rates": [1e-4, 5e-5, 2e-5],
            "gradient_clips": [1.0, 0.5, 0.3],
            "freeze_encoder": [False, False, False],
            "validation_split": 0.15,
            "random_seed": 42,
            "gradient_log_frequency": 100,
            "gradient_detailed_log": 500,
            "save_detailed_results": True,
            "show_examples": 50,
        }

        # Generation config
        self.gen_config = {
            "language": "ar",
            "task": "transcribe",
            "temperature": 0.3,
            "num_beams": 3,
            "top_p": 0.9,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.4,
        }

    def _get_next_run_number(self) -> int:
        """Find the next run number."""
        existing_runs = list(self.checkpoints_dir.glob("run_*"))
        if not existing_runs:
            return 1

        numbers = []
        for run in existing_runs:
            try:
                num = int(run.name.split("_")[1])
                numbers.append(num)
            except:
                continue

        return max(numbers) + 1 if numbers else 1

    def _create_run_dir(self) -> Path:
        """Create a new run directory."""
        run_num = self._get_next_run_number()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"run_{run_num:03d}_{timestamp}"

        run_dir = self.checkpoints_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"TRAINING RUN: {run_name}")
        print(f"{'=' * 80}")
        print(f"   Source model: {self.source_model}")
        print(f"   Saving to: {run_dir}")

        return run_dir

    def list_runs(self) -> List[Dict]:
        """List all available runs."""
        runs = []
        for run_path in sorted(self.checkpoints_dir.glob("run_*")):
            info = {"name": run_path.name, "path": str(run_path)}

            # Try to load training log for more info
            log_file = run_path / "training_log.json"
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        log = json.load(f)
                    info["source_model"] = log.get("config", {}).get("source_model", "unknown")
                    info["epochs"] = len(log.get("epochs", []))
                    if log.get("epochs"):
                        last_epoch = log["epochs"][-1]
                        info["final_wer"] = last_epoch.get("validation_metrics", {}).get("wer", {}).get("mean")
                except:
                    pass

            runs.append(info)

        return runs

    def print_runs(self):
        """Print all available runs nicely."""
        runs = self.list_runs()

        print(f"\n{'=' * 80}")
        print("AVAILABLE CHECKPOINTS")
        print(f"{'=' * 80}")

        if not runs:
            print("   No runs found.")
        else:
            for run in runs:
                wer_str = f", WER: {run['final_wer']:.4f}" if run.get('final_wer') else ""
                epochs_str = f", Epochs: {run['epochs']}" if run.get('epochs') else ""
                print(f"   {run['name']}{epochs_str}{wer_str}")
                print(f"      Path: {run['path']}")
                if run.get('source_model'):
                    print(f"      Source: {run['source_model']}")

        print(f"{'=' * 80}\n")

    def load_model(self):
        """Load model from source."""
        source = Path(self.source_model)

        # Check if it's a local path
        if source.exists():
            print(f"Loading from local path: {self.source_model}")
            self.processor = WhisperProcessor.from_pretrained(str(source))
            self.model = WhisperForConditionalGeneration.from_pretrained(str(source)).to(self.device)
        else:
            # Assume it's a HuggingFace model
            print(f"Loading from HuggingFace: {self.source_model}")
            self.processor = WhisperProcessor.from_pretrained(self.source_model)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.source_model).to(self.device)

        print(f"   Loaded on {self.device.upper()}")

        # Load BERTScore
        if BERTSCORE_AVAILABLE:
            try:
                self.bertscore = evaluate.load("bertscore")
                print("   BERTScore loaded")
            except Exception as e:
                print(f"   BERTScore failed: {e}")

        return self.model, self.processor

    def train(
            self,
            series_configs: List[Dict],
            epochs: Optional[int] = None,
            learning_rates: Optional[List[float]] = None,
            gradient_clips: Optional[List[float]] = None,
            freeze_encoder: Optional[List[bool]] = None,
    ) -> dict:

        epochs = epochs or self.config["epochs"]
        learning_rates = learning_rates or self.config["learning_rates"]
        gradient_clips = gradient_clips or self.config["gradient_clips"]
        freeze_encoder = freeze_encoder or self.config["freeze_encoder"]

        # Create new run directory
        self.run_dir = self._create_run_dir()

        # Load model from source
        self.load_model()

        # Load data
        print("\nLoading data...")
        loader = DataLoader(self.db)
        dataset = loader.load_multiple_series(series_configs)

        # Split train/val
        random.seed(self.config["random_seed"])
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        val_size = int(len(indices) * self.config["validation_split"])
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_data = dataset.select(train_indices)
        val_data = dataset.select(val_indices)

        print(f"\nDataset split:")
        print(f"   Training:   {len(train_data):,}")
        print(f"   Validation: {len(val_data):,}")

        # Metrics log
        metrics_log = {
            "config": {
                "series_configs": series_configs,
                "source_model": self.source_model,
                "run_dir": str(self.run_dir),
                "epochs": epochs,
                "learning_rates": learning_rates[:epochs],
                "gradient_clips": gradient_clips[:epochs],
                "freeze_encoder": freeze_encoder[:epochs],
                "train_samples": len(train_data),
                "val_samples": len(val_data),
            },
            "epochs": [],
            "gradient_stats": []
        }

        best_wer = float("inf")
        best_series_wer = {}

        # Gradient log
        gradient_log_file = self.run_dir / "gradient_flow.log"
        with open(gradient_log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Source model: {self.source_model}\n")
            f.write(f"Run directory: {self.run_dir}\n")

        global_step = 0

        # Print config
        print("\n" + "=" * 80)
        print("TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Source model: {self.source_model}")
        print(f"Run directory: {self.run_dir}")
        print()
        for ep in range(epochs):
            lr = learning_rates[ep] if ep < len(learning_rates) else learning_rates[-1]
            clip = gradient_clips[ep] if ep < len(gradient_clips) else gradient_clips[-1]
            freeze = freeze_encoder[ep] if ep < len(freeze_encoder) else freeze_encoder[-1]
            print(f"Epoch {ep + 1}: LR={lr}, Clip={clip}, Encoder={'FROZEN' if freeze else 'TRAINABLE'}")
        print("=" * 80)

        # Training loop
        for epoch in range(epochs):
            lr = learning_rates[epoch] if epoch < len(learning_rates) else learning_rates[-1]
            clip = gradient_clips[epoch] if epoch < len(gradient_clips) else gradient_clips[-1]
            freeze = freeze_encoder[epoch] if epoch < len(freeze_encoder) else freeze_encoder[-1]

            print(f"\n{'=' * 80}")
            print(f"EPOCH {epoch + 1}/{epochs}")
            print(f"{'=' * 80}")
            print(f"   LR = {lr}")
            print(f"   Clip = {clip}")
            print(f"   Encoder frozen = {freeze}")

            # Freeze/unfreeze encoder
            for p in self.model.model.encoder.parameters():
                p.requires_grad = not freeze
            for p in self.model.model.decoder.parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr
            )
            scaler = torch.cuda.amp.GradScaler()

            # Shuffle per epoch
            random.seed(self.config["random_seed"] + epoch)
            epoch_indices = list(range(len(train_data)))
            random.shuffle(epoch_indices)

            epoch_losses = []
            epoch_gradient_stats = []
            skipped = {"long": 0, "error": 0}
            processed = 0

            self.model.train()
            bar = tqdm(epoch_indices, desc=f"Epoch {epoch + 1}/{epochs}", ncols=130)

            for idx in bar:
                sample = train_data[idx]

                try:
                    audio = sample["audio"]
                    wav = torch.tensor(audio["array"]).float()
                    sr = audio["sampling_rate"]
                    text = sample["text"]

                    inputs = self.processor(wav, sampling_rate=sr, return_tensors="pt").to(self.device)
                    labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

                    if labels.shape[1] > self.model.config.max_target_positions:
                        skipped["long"] += 1
                        continue

                    with torch.cuda.amp.autocast():
                        out = self.model(input_features=inputs.input_features, labels=labels)
                        loss = out.loss

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    grad_stats_before = self._compute_gradient_stats()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    grad_stats_after = self._compute_gradient_stats()

                    scaler.step(optimizer)
                    scaler.update()

                    loss_val = loss.item()
                    epoch_losses.append(loss_val)
                    processed += 1
                    global_step += 1

                    if global_step % self.config["gradient_log_frequency"] == 0:
                        grad_entry = {
                            'epoch': epoch + 1,
                            'step': global_step,
                            'loss': loss_val,
                            'before_clip': grad_stats_before,
                            'after_clip': grad_stats_after,
                            'clipped': grad_stats_before['total_norm'] > clip
                        }
                        epoch_gradient_stats.append(grad_entry)

                        bar.set_postfix({
                            'loss': f"{loss_val:.3f}",
                            'avg': f"{np.mean(epoch_losses):.3f}",
                            'grad': f"{grad_stats_after['total_norm']:.3f}",
                        })
                    else:
                        bar.set_postfix({
                            'loss': f"{loss_val:.3f}",
                            'avg': f"{np.mean(epoch_losses):.3f}"
                        })

                    if global_step % self.config["gradient_detailed_log"] == 0:
                        self._log_gradient_flow(global_step, gradient_log_file)

                except Exception as e:
                    skipped["error"] += 1
                    continue

            metrics_log["gradient_stats"].extend(epoch_gradient_stats)

            if epoch_gradient_stats:
                avg_grad = np.mean([g['after_clip']['total_norm'] for g in epoch_gradient_stats])
                clipped_count = sum(1 for g in epoch_gradient_stats if g['clipped'])
                print(f"\nGradient: Avg={avg_grad:.4f}, Clipped={clipped_count}/{len(epoch_gradient_stats)}")

            print(f"\nLoss: Mean={np.mean(epoch_losses):.4f}, Median={np.median(epoch_losses):.4f}")
            print(f"Processed: {processed}, Skipped: {sum(skipped.values())}")

            # Validation
            print("\nEvaluating...")
            val_metrics, detailed_results = self._evaluate_comprehensive(val_data)

            # Save best overall
            if val_metrics['wer']['mean'] < best_wer:
                best_wer = val_metrics['wer']['mean']
                best_dir = self.run_dir / "best"
                best_dir.mkdir(exist_ok=True)
                self.model.save_pretrained(str(best_dir))
                self.processor.save_pretrained(str(best_dir))
                print(f"   Saved BEST model (WER: {best_wer:.4f})")

            # Per-series best
            if 'per_series' in val_metrics:
                for series_name, series_metrics in val_metrics['per_series'].items():
                    series_wer = series_metrics['wer_mean']

                    if series_name not in best_series_wer or series_wer < best_series_wer[series_name]:
                        best_series_wer[series_name] = series_wer
                        series_checkpoint = self.run_dir / f"best_{series_name}"
                        series_checkpoint.mkdir(exist_ok=True)
                        self.model.save_pretrained(str(series_checkpoint))
                        self.processor.save_pretrained(str(series_checkpoint))
                        print(f"   Saved BEST {series_name} model (WER: {series_wer:.4f})")
                    elif series_wer > best_series_wer[series_name] * 1.1:
                        print(f"   WARNING: {series_name} WER increased >10%!")
                        print(f"   Was {best_series_wer[series_name]:.4f}, now {series_wer:.4f}")

            # Save epoch checkpoint
            epoch_dir = self.run_dir / f"epoch_{epoch + 1:02d}"
            epoch_dir.mkdir(exist_ok=True)
            self.model.save_pretrained(str(epoch_dir))
            self.processor.save_pretrained(str(epoch_dir))
            print(f"   Saved epoch {epoch + 1} checkpoint")

            metrics_log["epochs"].append({
                "epoch": epoch + 1,
                "loss": {"mean": float(np.mean(epoch_losses)), "median": float(np.median(epoch_losses))},
                "validation_metrics": val_metrics,
                "samples": {"processed": processed, "skipped": sum(skipped.values())}
            })

            # Save metrics after each epoch
            with open(self.run_dir / "training_log.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

        # Final save
        final_dir = self.run_dir / "final"
        final_dir.mkdir(exist_ok=True)
        self.model.save_pretrained(str(final_dir))
        self.processor.save_pretrained(str(final_dir))

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Run directory: {self.run_dir}")
        print(f"Best overall WER: {best_wer:.4f}")
        for series_name, s_wer in best_series_wer.items():
            print(f"Best {series_name} WER: {s_wer:.4f}")
        print(f"\nCheckpoints saved:")
        print(f"   {self.run_dir}/best          <- Best overall")
        print(f"   {self.run_dir}/final         <- Final state")
        print(f"   {self.run_dir}/epoch_XX      <- Each epoch")
        print("=" * 80)

        return metrics_log

    def _compute_gradient_stats(self) -> dict:
        """Compute gradient statistics."""
        total_norm = 0.0
        encoder_norm = 0.0
        decoder_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        grad_norms = {'encoder': [], 'decoder': []}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                if 'encoder' in name:
                    encoder_norm += param_norm ** 2
                    grad_norms['encoder'].append(param_norm)
                elif 'decoder' in name:
                    decoder_norm += param_norm ** 2
                    grad_norms['decoder'].append(param_norm)

                max_grad = max(max_grad, param.grad.abs().max().item())
                min_grad = min(min_grad, param.grad.abs().min().item())

        return {
            'total_norm': total_norm ** 0.5,
            'encoder_norm': encoder_norm ** 0.5 if grad_norms['encoder'] else 0.0,
            'decoder_norm': decoder_norm ** 0.5 if grad_norms['decoder'] else 0.0,
            'max_grad': max_grad,
            'min_grad': min_grad if min_grad != float('inf') else 0.0,
            'mean_encoder': float(np.mean(grad_norms['encoder'])) if grad_norms['encoder'] else 0.0,
            'mean_decoder': float(np.mean(grad_norms['decoder'])) if grad_norms['decoder'] else 0.0
        }

    def _log_gradient_flow(self, step: int, log_file: Path):
        """Log detailed gradient flow."""
        grad_info = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_info.append({
                    'name': name,
                    'grad_norm': param.grad.data.norm(2).item(),
                    'grad_mean': param.grad.data.mean().item(),
                    'grad_std': param.grad.data.std().item(),
                })

        with open(log_file, 'a') as f:
            f.write(f"\n{'=' * 80}\nStep: {step}\n{'=' * 80}\n")
            for info in grad_info[:10]:
                f.write(f"Layer: {info['name']}\n")
                f.write(f"  Grad Norm: {info['grad_norm']:.6f}\n")
                f.write(f"  Grad Mean: {info['grad_mean']:.6f}\n")
                f.write(f"  Grad Std: {info['grad_std']:.6f}\n")

    def _evaluate_comprehensive(self, dataset) -> tuple:
        """Comprehensive evaluation."""
        self.model.eval()

        results = []
        wer_scores = []
        cer_scores = []
        losses = []
        all_references = []
        all_predictions = []

        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 80)

        with torch.no_grad():
            for sample in tqdm(dataset, desc="Evaluating", leave=False):
                try:
                    audio = sample["audio"]
                    wav = torch.tensor(audio["array"]).float()
                    sr = audio["sampling_rate"]
                    reference = sample["text"]
                    series = sample.get("series", "unknown")

                    inputs = self.processor(wav, sampling_rate=sr, return_tensors="pt").to(self.device)

                    # Loss
                    try:
                        labels = self.processor.tokenizer(reference, return_tensors="pt").input_ids.to(self.device)
                        if labels.shape[1] <= self.model.config.max_target_positions:
                            with torch.cuda.amp.autocast():
                                out = self.model(input_features=inputs.input_features, labels=labels)
                                loss = out.loss.item()
                            losses.append(loss)
                        else:
                            loss = None
                    except:
                        loss = None

                    # Generate
                    pred_ids = self.model.generate(inputs.input_features, **self.gen_config)
                    prediction = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

                    # WER/CER
                    try:
                        wer_score = wer(reference, prediction)
                        cer_score = cer(reference, prediction)
                        wer_scores.append(wer_score)
                        cer_scores.append(cer_score)
                    except:
                        wer_score = None
                        cer_score = None

                    all_references.append(reference)
                    all_predictions.append(prediction)

                    results.append({
                        'series': series,
                        'reference': reference,
                        'prediction': prediction,
                        'wer': float(wer_score) if wer_score is not None else None,
                        'cer': float(cer_score) if cer_score is not None else None,
                        'loss': float(loss) if loss is not None else None,
                    })
                except:
                    continue

        # BERTScore
        bert_scores = None
        if self.bertscore and all_references:
            try:
                print("\nComputing BERTScore...")
                bert_results = self.bertscore.compute(
                    predictions=all_predictions,
                    references=all_references,
                    lang="ar",
                    model_type="bert-base-multilingual-cased"
                )
                bert_scores = {
                    'precision': float(np.mean(bert_results['precision'])),
                    'recall': float(np.mean(bert_results['recall'])),
                    'f1': float(np.mean(bert_results['f1']))
                }
            except Exception as e:
                print(f"   BERTScore failed: {e}")

        def safe_mean(vals):
            return float(np.mean(vals)) if vals else 0.0

        def safe_median(vals):
            return float(np.median(vals)) if vals else 0.0

        def safe_std(vals):
            return float(np.std(vals)) if vals else 0.0

        metrics = {
            'wer': {'mean': safe_mean(wer_scores), 'median': safe_median(wer_scores), 'std': safe_std(wer_scores)},
            'cer': {'mean': safe_mean(cer_scores), 'median': safe_median(cer_scores), 'std': safe_std(cer_scores)},
            'loss': {'mean': safe_mean(losses), 'median': safe_median(losses)},
            'bertscore': bert_scores,
            'total_samples': len(results)
        }

        # Per-series
        series_metrics = {}
        for series_name in set(r['series'] for r in results):
            series_results = [r for r in results if r['series'] == series_name]
            series_wer = [r['wer'] for r in series_results if r['wer'] is not None]
            series_cer = [r['cer'] for r in series_results if r['cer'] is not None]
            series_metrics[series_name] = {
                'count': len(series_results),
                'wer_mean': safe_mean(series_wer),
                'cer_mean': safe_mean(series_cer)
            }
        metrics['per_series'] = series_metrics

        # Print
        print(f"\nOverall WER: {metrics['wer']['mean']:.4f}")
        print(f"Overall CER: {metrics['cer']['mean']:.4f}")
        print(f"Overall Loss: {metrics['loss']['mean']:.4f}")
        if bert_scores:
            print(f"BERTScore F1: {bert_scores['f1']:.4f}")
        print(f"\nPer-series:")
        for series_name, sm in series_metrics.items():
            print(f"   {series_name}: WER={sm['wer_mean']:.4f}, CER={sm['cer_mean']:.4f}, Count={sm['count']}")
        print(f"\nTotal samples: {metrics['total_samples']}")
        print("=" * 80)

        # Examples
        if self.config["save_detailed_results"] and results:
            sorted_results = sorted([r for r in results if r['wer'] is not None], key=lambda x: x['wer'], reverse=True)
            n = self.config['show_examples']

            print(f"\nWORST {n} PREDICTIONS:")
            print("-" * 80)
            for i, r in enumerate(sorted_results[:n], 1):
                print(f"\n{i}. {r['series']} | WER:{r['wer']:.3f}")
                print(f"   REF: {r['reference']}")
                print(f"   PRD: {r['prediction']}")

            print(f"\nBEST {n} PREDICTIONS:")
            print("-" * 80)
            for i, r in enumerate(sorted_results[-n:], 1):
                print(f"\n{i}. {r['series']} | WER:{r['wer']:.3f}")
                print(f"   REF: {r['reference']}")
                print(f"   PRD: {r['prediction']}")

        self.model.train()
        return metrics, results


def get_training_service(
        db: Session,
        checkpoints_dir: str = "./checkpoints",
        source_model: str = "openai/whisper-small",
) -> TrainingService:

    return TrainingService(db, checkpoints_dir, source_model)