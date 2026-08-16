"""
Decoder Layer Probing Script

Probes the decoder layers to see where the dialect signal gets lost.
The encoder clearly distinguishes Egyptian from Lebanese (99-100% accuracy).
This script checks if the decoder receives and maintains that signal.
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sqlalchemy.orm import Session
from Config.Config import get_db
from Config.Config import DataLoader


class DecoderProbe:

    def __init__(self, model_path: str, db: Session):
        self.db = db
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model from {model_path}...")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.hidden_states = {}
        self.hooks = []

        print(f"Model loaded on {self.device.upper()}")
        print(f"Encoder layers: {self.model.config.encoder_layers}")
        print(f"Decoder layers: {self.model.config.decoder_layers}")

    def _register_hooks(self):
        """Register hooks on encoder layers, decoder layers, and cross-attention."""
        self.hooks = []
        self.hidden_states = {}

        # Encoder layers
        for i, layer in enumerate(self.model.model.encoder.layers):
            def hook_fn(module, input, output, name=f"encoder_{i}"):
                h = output[0] if isinstance(output, tuple) else output
                self.hidden_states[name] = h.detach().cpu()
            self.hooks.append(layer.register_forward_hook(hook_fn))

        # Decoder layers - self attention output
        for i, layer in enumerate(self.model.model.decoder.layers):
            # Hook on the full decoder layer output
            def layer_hook(module, input, output, name=f"decoder_{i}"):
                h = output[0] if isinstance(output, tuple) else output
                self.hidden_states[name] = h.detach().cpu()
            self.hooks.append(layer.register_forward_hook(layer_hook))

            # Hook on cross-attention specifically
            def cross_attn_hook(module, input, output, name=f"decoder_{i}_cross_attn"):
                h = output[0] if isinstance(output, tuple) else output
                self.hidden_states[name] = h.detach().cpu()
            self.hooks.append(layer.encoder_attn.register_forward_hook(cross_attn_hook))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_representations(self, dataset, dialect_label: str, max_samples: int = 100):
        """Extract hidden states from both encoder and decoder."""
        representations = defaultdict(list)
        metadata = []

        self._register_hooks()

        samples = min(max_samples, len(dataset))
        print(f"\nExtracting {dialect_label} representations ({samples} samples)...")

        with torch.no_grad():
            for i in tqdm(range(samples), desc=f"  {dialect_label}"):
                sample = dataset[i]

                try:
                    audio = sample["audio"]
                    wav = torch.tensor(audio["array"]).float()
                    sr = audio["sampling_rate"]
                    text = sample["text"]

                    inputs = self.processor(wav, sampling_rate=sr, return_tensors="pt").to(self.device)

                    # We need to run the full model (encoder + decoder) to get decoder states
                    # Use the reference text as decoder input (teacher forcing)
                    labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

                    # Forward pass through full model
                    outputs = self.model(
                        input_features=inputs.input_features,
                        decoder_input_ids=labels[:, :-1],  # shift right
                    )

                    # Collect hidden states
                    for layer_name, hidden_state in self.hidden_states.items():
                        # Mean pool across time/sequence dimension
                        pooled = hidden_state.squeeze(0).mean(dim=0).numpy()
                        representations[layer_name].append(pooled)

                    metadata.append({
                        "dialect": dialect_label,
                        "text": text,
                        "index": i,
                    })

                    self.hidden_states = {}

                except Exception as e:
                    self.hidden_states = {}
                    continue

        self._remove_hooks()

        for key in representations:
            representations[key] = np.array(representations[key])

        print(f"  Extracted {len(metadata)} samples across {len(representations)} layers")

        return dict(representations), metadata

    def probe_all_layers(self, repr_a, repr_b, label_a, label_b):
        """Run dialect separation probe on all layers."""

        # Separate encoder, decoder, and cross-attention layers
        encoder_layers = sorted([k for k in repr_a if k.startswith("encoder_")])
        decoder_layers = sorted([k for k in repr_a if k.startswith("decoder_") and "cross" not in k])
        cross_attn_layers = sorted([k for k in repr_a if "cross_attn" in k])

        all_results = {}

        for section_name, layers in [
            ("ENCODER", encoder_layers),
            ("DECODER (self-attention + FFN output)", decoder_layers),
            ("DECODER CROSS-ATTENTION", cross_attn_layers),
        ]:
            if not layers:
                continue

            print(f"\n{'=' * 80}")
            print(f"DIALECT SEPARATION: {section_name}")
            print(f"{'=' * 80}")
            print(f"{'Layer':<30} {'Accuracy':<12} {'Interpretation'}")
            print(f"{'-' * 70}")

            for layer in layers:
                if layer not in repr_a or layer not in repr_b:
                    continue

                X = np.vstack([repr_a[layer], repr_b[layer]])
                y = np.array([0] * len(repr_a[layer]) + [1] * len(repr_b[layer]))

                idx = np.random.permutation(len(X))
                X, y = X[idx], y[idx]

                clf = LogisticRegression(max_iter=1000, random_state=42)
                try:
                    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                    mean_acc = scores.mean()
                    std_acc = scores.std()
                except:
                    mean_acc = 0.5
                    std_acc = 0.0

                if mean_acc > 0.9:
                    interp = "STRONG - clearly distinguishes"
                elif mean_acc > 0.75:
                    interp = "MODERATE - partially distinguishes"
                elif mean_acc > 0.6:
                    interp = "WEAK - barely distinguishes"
                else:
                    interp = "NONE - cannot tell apart"

                print(f"{layer:<30} {mean_acc:.3f} ±{std_acc:.3f}  {interp}")

                all_results[layer] = {
                    "accuracy": float(mean_acc),
                    "std": float(std_acc),
                    "section": section_name,
                }

        return all_results

    def compute_similarity(self, repr_a, repr_b, label_a, label_b):
        """Cosine similarity at each layer."""

        encoder_layers = sorted([k for k in repr_a if k.startswith("encoder_")])
        decoder_layers = sorted([k for k in repr_a if k.startswith("decoder_") and "cross" not in k])
        cross_attn_layers = sorted([k for k in repr_a if "cross_attn" in k])

        all_results = {}

        for section_name, layers in [
            ("ENCODER", encoder_layers),
            ("DECODER", decoder_layers),
            ("CROSS-ATTENTION", cross_attn_layers),
        ]:
            if not layers:
                continue

            print(f"\n{'=' * 80}")
            print(f"COSINE SIMILARITY: {section_name}")
            print(f"{'=' * 80}")
            print(f"{'Layer':<30} {'Cross-Sim':<12} {'Within-A':<12} {'Within-B':<12} {'Ratio'}")
            print(f"{'-' * 80}")

            for layer in layers:
                if layer not in repr_a or layer not in repr_b:
                    continue

                A = repr_a[layer]
                B = repr_b[layer]

                A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
                B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)

                cross_sim = np.mean(A_norm @ B_norm.T)
                within_a = np.mean(A_norm @ A_norm.T)
                within_b = np.mean(B_norm @ B_norm.T)
                ratio = cross_sim / max(within_a, within_b)

                print(f"{layer:<30} {cross_sim:.4f}      {within_a:.4f}      {within_b:.4f}      {ratio:.4f}")

                all_results[layer] = {
                    "cross_similarity": float(cross_sim),
                    "within_a": float(within_a),
                    "within_b": float(within_b),
                    "ratio": float(ratio),
                }

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Probe decoder layers for dialect analysis")

    parser.add_argument("--source-model", type=str,
                        default=r"./checkpoints/run_004_20260208_121850/best")
    parser.add_argument("--series-a", type=int, required=True)
    parser.add_argument("--series-b", type=int, required=True)
    parser.add_argument("--episodes-a", type=str, nargs="*")
    parser.add_argument("--episodes-b", type=str, nargs="*")
    parser.add_argument("--label-a", type=str, default="Dialect_A")
    parser.add_argument("--label-b", type=str, default="Dialect_B")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./probing_results")

    args = parser.parse_args()

    db = next(get_db())
    probe = DecoderProbe(args.source_model, db)

    loader = DataLoader(db)

    print("\nLoading Dialect A data...")
    dataset_a = loader.load_datasets(
        series_id=args.series_a,
        episode_names=args.episodes_a,
    )
    print(f"  Loaded {len(dataset_a)} samples")

    print("Loading Dialect B data...")
    dataset_b = loader.load_datasets(
        series_id=args.series_b,
        episode_names=args.episodes_b,
    )
    print(f"  Loaded {len(dataset_b)} samples")

    # Extract from both encoder and decoder
    repr_a, meta_a = probe.extract_representations(dataset_a, args.label_a, args.max_samples)
    repr_b, meta_b = probe.extract_representations(dataset_b, args.label_b, args.max_samples)

    # Run probes
    separation = probe.probe_all_layers(repr_a, repr_b, args.label_a, args.label_b)
    similarity = probe.compute_similarity(repr_a, repr_b, args.label_a, args.label_b)

    # Summary
    print(f"\n{'=' * 80}")
    print("FULL MODEL SUMMARY: Encoder → Decoder")
    print(f"{'=' * 80}")

    # Track accuracy through the full pipeline
    print(f"\n{'Layer':<30} {'Accuracy':<12} {'Signal Status'}")
    print(f"{'-' * 60}")

    ordered_layers = (
        sorted([k for k in separation if k.startswith("encoder_")]) +
        sorted([k for k in separation if k.startswith("decoder_") and "cross" not in k]) +
        sorted([k for k in separation if "cross_attn" in k])
    )

    prev_acc = None
    signal_lost_at = None

    for layer in ordered_layers:
        acc = separation[layer]["accuracy"]

        if prev_acc is not None:
            delta = acc - prev_acc
            if delta < -0.1 and signal_lost_at is None:
                signal_lost_at = layer
                status = f"<-- SIGNAL DROPS HERE (delta: {delta:.3f})"
            elif acc < 0.7:
                status = "SIGNAL WEAK"
            elif acc < 0.5:
                status = "SIGNAL LOST"
            else:
                status = "OK"
        else:
            status = "OK"

        print(f"{layer:<30} {acc:.3f}        {status}")
        prev_acc = acc

    if signal_lost_at:
        print(f"\nDIAGNOSIS: Dialect signal drops at {signal_lost_at}")
        print(f"  -> Apply LoRA starting from this layer onward")
    else:
        print(f"\nDIAGNOSIS: Signal maintained throughout")
        print(f"  -> Problem may be in the final output projection (lm_head)")

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "separation": separation,
        "similarity": similarity,
    }
    with open(output_path / "decoder_probe_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {args.output_dir}/decoder_probe_report.json")


if __name__ == "__main__":
    main()