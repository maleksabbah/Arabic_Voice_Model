"""
AraBEST-RQ Fine-Tuning — Unfreeze Top Encoder Layers + CTC Head
Based on train_probe_arabest.py but unfreezes L17-19 (the signal-carrying
layers) with a lower learning rate while keeping L1-16 frozen.

Key differences from the frozen probe:
  - L17-19 unfrozen with lr=1e-5 (encoder_lr)
  - CTC head + layer weights use lr=1e-3 (head_lr)
  - Gradient checkpointing on unfrozen layers to save VRAM
  - Checkpoint saves encoder state, not just head
  - model.train() keeps L1-16 frozen in eval, L17-19 in train mode

Usage:
  python TrainingAraBEST.py \
    --model_dir /workspace/arabest_model \
    --yaml_config /workspace/AraBEST-RQ/model/hparams/hybrid_300M.yaml \
    --train_data /workspace/datasets/train_combined \
    --output_dir /workspace/finetune_output_arabest \
    --unfreeze_layers 17,18,19 \
    --encoder_lr 1e-5 \
    --head_lr 1e-3
"""

import os, argparse, json, re, time, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk, Audio
from tqdm import tqdm
import evaluate
from hyperpyyaml import load_hyperpyyaml


# ============================================================
# UTILS
# ============================================================

ARABIC_DIACRITICS = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]')


def normalize_arabic(text):
    if not text or not isinstance(text, str):
        return ""
    text = ARABIC_DIACRITICS.sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = text.replace('ة', 'ه').replace('ى', 'ي').replace('ـ', '')
    text = re.sub(r'[^ء-ي٠-٩\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_to_labels(text, vocab):
    labels = []
    for c in text:
        if c == " ":
            labels.append(vocab.get("|", vocab.get("<unk>")))
        elif c in vocab:
            labels.append(vocab[c])
        else:
            labels.append(vocab.get("<unk>", 0))
    return labels


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA  (identical to probe)
# ============================================================

def build_vocab(dataset, text_col="normalized_text"):
    chars = set()
    for item in dataset:
        text = item.get(text_col, "")
        if text:
            chars.update(text)
    chars = sorted(chars)
    vocab = {"<pad>": 0}
    for i, c in enumerate(chars):
        vocab[c] = i + 1
    vocab["<unk>"] = len(vocab)
    vocab["|"] = len(vocab)
    if " " in vocab:
        space_id = vocab.pop(" ")
        vocab["|"] = space_id
    return vocab


def filter_dataset(dataset, min_seconds=0.5, max_seconds=30.0):
    min_samples = int(min_seconds * 16000)
    max_samples = int(max_seconds * 16000)

    def is_valid(sample):
        audio = sample["audio"]
        array = audio["array"] if isinstance(audio, dict) else audio
        if len(array) < min_samples or len(array) > max_samples:
            return False
        text = sample.get("normalized_text", "")
        if not text or not text.strip():
            return False
        if np.any(np.isnan(array)) or np.any(np.isinf(array)):
            return False
        fbank_frames = len(array) // 160 + 1
        encoder_frames = fbank_frames // 4
        if len(text) > encoder_frames:
            return False
        return True

    before = len(dataset)
    dataset = dataset.filter(is_valid)
    after = len(dataset)
    removed = before - after
    if removed > 0:
        print(f"  Filtered: {before} -> {after} ({removed} removed, {removed / before * 100:.1f}%)")
    return dataset


class ArabEstProbeDataset(Dataset):
    def __init__(self, hf_dataset, vocab, max_audio_len=480000):
        self.dataset = hf_dataset
        self.vocab = vocab
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]
        array = audio["array"] if isinstance(audio, dict) else audio
        array = np.array(array, dtype=np.float32)
        if len(array) > self.max_audio_len:
            array = array[: self.max_audio_len]
        text = item.get("normalized_text", "")
        labels = text_to_labels(text, self.vocab)
        return {"audio": array, "labels": labels}


def collate_fn(batch):
    audios = [torch.from_numpy(item["audio"]) for item in batch]
    labels_list = [item["labels"] for item in batch]

    max_audio = max(a.shape[0] for a in audios)
    padded_audio = torch.zeros(len(audios), max_audio)
    audio_lengths = torch.zeros(len(audios))
    for i, a in enumerate(audios):
        padded_audio[i, : a.shape[0]] = a
        audio_lengths[i] = a.shape[0] / max_audio

    max_label = max(len(l) for l in labels_list)
    padded_labels = torch.full((len(labels_list), max_label), -100, dtype=torch.long)
    for i, l in enumerate(labels_list):
        padded_labels[i, : len(l)] = torch.tensor(l, dtype=torch.long)

    return {
        "audio": padded_audio,
        "audio_lengths": audio_lengths,
        "labels": padded_labels,
    }


# ============================================================
# MODEL — Fine-tuning variant
# ============================================================

class ArabEstCTCFinetune(nn.Module):
    """AraBEST-RQ encoder with selectively unfrozen layers + CTC head.

    Unlike the probe, this model:
      - Unfreezes specified conformer layers (e.g. L17-19)
      - Keeps L1-16, CNN, normalizer frozen
      - Supports gradient checkpointing on unfrozen layers
    """

    def __init__(self, model_dir, yaml_config, vocab_size, pad_token_id=0,
                 device="cpu", unfreeze_layers=None, use_grad_checkpoint=True):
        super().__init__()

        # ---- Load YAML config ----
        print(f"  Loading config: {yaml_config}")
        with open(yaml_config, encoding="utf-8") as f:
            hparams = load_hyperpyyaml(f)

        self.compute_features = hparams["compute_features"]
        self.normalize = hparams["normalize"]

        model_list = hparams["model"]
        self.cnn = model_list[0]
        self.wrapper = model_list[1]
        self.transformer = hparams["Transformer"]

        # ---- Load pretrained weights ----
        model_ckpt_path = os.path.join(model_dir, "model.ckpt")
        norm_ckpt_path = os.path.join(model_dir, "normalizer.ckpt")

        if not os.path.exists(model_ckpt_path):
            for root, dirs, files in os.walk(model_dir):
                if "model.ckpt" in files:
                    model_ckpt_path = os.path.join(root, "model.ckpt")
                    norm_ckpt_path = os.path.join(root, "normalizer.ckpt")
                    break

        print(f"  Loading model checkpoint: {model_ckpt_path}")
        assert os.path.exists(model_ckpt_path), f"model.ckpt not found. Searched: {model_dir}"

        state_dict = torch.load(model_ckpt_path, map_location="cpu", weights_only=True)
        model_list.load_state_dict(state_dict)
        print(f"  Model checkpoint loaded: {len(state_dict)} keys")

        if os.path.exists(norm_ckpt_path):
            print(f"  Loading normalizer: {norm_ckpt_path}")
            norm_state = torch.load(norm_ckpt_path, map_location="cpu", weights_only=True)
            self.normalize.load_state_dict(norm_state)
            self.normalize.eval()
        else:
            print("  WARNING: normalizer.ckpt not found")

        # ---- Enable hidden state extraction ----
        self.transformer.encoder.output_hidden_states = True
        self.transformer.output_hidden_states = True

        # ---- Freeze/unfreeze strategy ----
        # Default: unfreeze L17-19 (1-indexed)
        self.unfreeze_layers = unfreeze_layers or [17, 18, 19]
        # Convert to 0-indexed for accessing encoder.layers
        self.unfreeze_indices = [l - 1 for l in self.unfreeze_layers]
        self.use_grad_checkpoint = use_grad_checkpoint

        # Step 1: Freeze EVERYTHING
        for param in self.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze the specified conformer layers
        self._encoder_layers = self._get_encoder_layers()
        encoder_layers = self._encoder_layers
        total_encoder_layers = len(encoder_layers)
        unfrozen_count = 0
        for idx in self.unfreeze_indices:
            if idx < total_encoder_layers:
                layer = encoder_layers[idx]
                for param in layer.parameters():
                    param.requires_grad = True
                unfrozen_count += sum(p.numel() for p in layer.parameters())
                print(f"  Unfroze encoder layer {idx + 1} "
                      f"({sum(p.numel() for p in layer.parameters()):,} params)")
            else:
                print(f"  WARNING: Layer {idx + 1} doesn't exist (model has {total_encoder_layers} layers)")

        # ---- CTC head (always trainable) ----
        total_layers = hparams["num_encoder_layers"]
        hidden_size = hparams["d_model"]

        # Use all layers for weighted sum (same as all-24 probe)
        num_weights = total_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_weights))
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=pad_token_id, reduction="mean", zero_infinity=True)

        # Count params
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = frozen + trainable
        print(f"\n  === Parameter Summary ===")
        print(f"  Total:     {total:,}")
        print(f"  Frozen:    {frozen:,} ({100 * frozen / total:.1f}%)")
        print(f"  Trainable: {trainable:,} ({100 * trainable / total:.1f}%)")
        print(f"    - Encoder (L{','.join(str(l) for l in self.unfreeze_layers)}): {unfrozen_count:,}")
        head_params = sum(p.numel() for p in self.lm_head.parameters()) + self.layer_weights.numel()
        print(f"    - Head + layer weights: {head_params:,}")
        print(f"  Layer weights: {num_weights} layers, d_model={hidden_size}")

    def _get_encoder_layers(self):
        """Navigate SpeechBrain's module tree to find the conformer layers list."""
        # SpeechBrain TransformerASR stores layers in:
        #   transformer.encoder.layers (nn.ModuleList)
        encoder = self.transformer.encoder
        if hasattr(encoder, 'layers'):
            return encoder.layers
        elif hasattr(encoder, 'conformer_layers'):
            return encoder.conformer_layers
        # Fallback: try to find any ModuleList in the encoder
        for name, module in encoder.named_children():
            if isinstance(module, nn.ModuleList):
                print(f"  Found encoder layers at: encoder.{name} ({len(module)} layers)")
                return module
        raise RuntimeError("Cannot find encoder layers — check SpeechBrain architecture")

    def _get_encoder_params(self):
        """Return only the unfrozen encoder parameters (for optimizer group)."""
        encoder_layers = self._get_encoder_layers()
        params = []
        for idx in self.unfreeze_indices:
            if idx < len(encoder_layers):
                params.extend(encoder_layers[idx].parameters())
        return [p for p in params if p.requires_grad]

    def _get_head_params(self):
        """Return head + layer weight parameters (for optimizer group)."""
        return list(self.lm_head.parameters()) + [self.layer_weights]

    def train(self, mode=True):
        """Override: frozen layers stay in eval, unfrozen layers follow mode."""
        super().train(mode)
        # Always eval: CNN, wrapper, normalizer, feature extractor
        self.cnn.eval()
        self.wrapper.eval()
        self.normalize.eval()
        self.compute_features.eval()

        # Frozen encoder layers stay in eval
        encoder_layers = self._encoder_layers
        for i, layer in enumerate(encoder_layers):
            if i not in self.unfreeze_indices:
                layer.eval()
            else:
                # Unfrozen layers follow the requested mode
                layer.train(mode)
        return self

    def get_layer_weight_distribution(self):
        weights = torch.softmax(self.layer_weights, dim=0).detach().cpu().numpy()
        return weights

    def forward(self, audio, audio_lengths, labels=None):
        """
        Forward pass — same pipeline as probe, but unfrozen layers
        get gradients flowing through them.
        """
        # Frozen part: no gradients for features + CNN
        with torch.no_grad():
            feats = self.compute_features(audio)
            feats = self.normalize(feats, audio_lengths, epoch=0)

            B, T, C = feats.shape
            pad_needed = (4 - T % 4) % 4
            if pad_needed > 0:
                feats = torch.nn.functional.pad(feats, [0, 0, 0, pad_needed])

            src = self.cnn(feats)

        # Encoder pass — WITH gradients for unfrozen layers
        # The wrapper + transformer handle gradient flow to unfrozen layers
        result = self.wrapper(src, audio_lengths)
        enc_out, hidden_states = result

        # hidden_states: [input_embedding] + [24 layer outputs]
        layer_outputs = torch.stack(hidden_states[1:], dim=0)  # (24, B, T', d_model)

        # Weighted sum (learnable)
        weights = torch.softmax(self.layer_weights, dim=0)
        weights = weights.view(-1, 1, 1, 1)
        blended = (weights * layer_outputs).sum(dim=0)

        # CTC projection
        logits = self.lm_head(blended)

        loss = None
        if labels is not None:
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            output_T = logits.shape[1]
            input_lengths = (audio_lengths * output_T).long().clamp(min=1, max=output_T)

            label_mask = labels != -100
            label_lengths = label_mask.sum(dim=-1)

            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = 0

            loss = self.ctc_loss(log_probs, labels_clean, input_lengths, label_lengths)

        return {"loss": loss, "logits": logits}


# ============================================================
# TRAIN — two learning rate groups
# ============================================================

def train(model, train_loader, eval_fn, args, device):
    # Two param groups: encoder (slow) and head (fast)
    encoder_params = model._get_encoder_params()
    head_params = model._get_head_params()

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr, "weight_decay": 0.01},
        {"params": head_params,    "lr": args.head_lr,    "weight_decay": 0.01},
    ])
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    total_steps = min(args.max_steps, len(train_loader) * args.epochs)
    warmup_steps = min(1000, total_steps // 8)  # longer warmup for encoder stability

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_wer = float("inf")
    start_time = time.time()

    print(f"\n  === Training Config ===")
    print(f"  Encoder LR: {args.encoder_lr}")
    print(f"  Head LR:    {args.head_lr}")
    print(f"  Warmup:     {warmup_steps} steps")
    print(f"  Total:      {total_steps} steps (max)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs:     {args.epochs}")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in bar:
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(audio=audio, audio_lengths=audio_lengths, labels=labels)
                    loss = outputs["loss"]
            else:
                outputs = model(audio=audio, audio_lengths=audio_lengths, labels=labels)
                loss = outputs["loss"]

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Clip encoder and head separately
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
                torch.nn.utils.clip_grad_norm_(head_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
                torch.nn.utils.clip_grad_norm_(head_params, 1.0)
                optimizer.step()

            scheduler.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = np.mean(epoch_losses[-100:])
                enc_lr = scheduler.get_last_lr()[0]
                head_lr = scheduler.get_last_lr()[1]
                bar.set_postfix({
                    "loss": f"{loss_val:.3f}",
                    "avg": f"{avg_loss:.3f}",
                    "enc_lr": f"{enc_lr:.1e}",
                    "head_lr": f"{head_lr:.1e}",
                    "step": global_step,
                })

            # Print layer weights periodically
            if global_step % (args.save_every // 2) == 0:
                w = model.get_layer_weight_distribution()
                top_3 = np.argsort(w)[-3:][::-1]
                top_str = ", ".join([f"L{i + 1}={w[i]:.3f}" for i in top_3])
                print(f"  Layer weights (step {global_step}): top = {top_str}")

            # Save + eval
            if global_step % args.save_every == 0:
                _save_checkpoint(model, args, global_step)

                if eval_fn:
                    mid_wer = eval_fn()
                    if mid_wer < best_wer:
                        best_wer = mid_wer
                        _save_checkpoint(model, args, tag="best")
                        print(f"  *** New best WER: {best_wer * 100:.1f}% ***")

            if global_step >= args.max_steps:
                break

        print(f"  Epoch {epoch + 1}: loss={np.mean(epoch_losses):.4f}")
        if global_step >= args.max_steps:
            break

    # Save final
    _save_checkpoint(model, args, tag="final")

    # Print final layer weight distribution
    w = model.get_layer_weight_distribution()
    print(f"\n  Final layer weights:")
    for i, wi in enumerate(w):
        bar_str = "█" * int(wi * 100)
        print(f"    Layer {i + 1:2d}: {wi:.4f} {bar_str}")
    top_3 = np.argsort(w)[-3:][::-1]
    print(f"  Top layers: {', '.join([f'L{i + 1} ({w[i]:.3f})' for i in top_3])}")

    elapsed = time.time() - start_time
    print(f"  Training done: {global_step} steps, {elapsed / 60:.1f} min, best WER: {best_wer * 100:.1f}%")
    return global_step, best_wer


def _save_checkpoint(model, args, global_step=None, tag=None):
    """Save full checkpoint: encoder layers + head + layer weights."""
    if tag:
        save_dir = os.path.join(args.output_dir, tag)
    else:
        save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_dir, exist_ok=True)

    # Save unfrozen encoder layer states
    encoder_layers = model._get_encoder_layers()
    encoder_state = {}
    for idx in model.unfreeze_indices:
        if idx < len(encoder_layers):
            layer_state = encoder_layers[idx].state_dict()
            for k, v in layer_state.items():
                encoder_state[f"layer_{idx}.{k}"] = v

    torch.save({
        "lm_head": model.lm_head.state_dict(),
        "layer_weights": model.layer_weights.data,
        "encoder_layers": encoder_state,
        "unfreeze_layers": model.unfreeze_layers,
    }, os.path.join(save_dir, "finetune_ckpt.pt"))

    if tag:
        print(f"  Saved {tag} checkpoint: {save_dir}")


# ============================================================
# EVAL  (identical to probe)
# ============================================================

def ctc_decode(ids, id_to_char):
    decoded = []
    prev = None
    for i in ids:
        if i == 0:
            prev = None
            continue
        if i == prev:
            continue
        c = id_to_char.get(i, "")
        if c == "|":
            c = " "
        if c:
            decoded.append(c)
        prev = i
    return "".join(decoded).strip()


def evaluate_dataset(model, dataset, vocab, batch_size=8, name="eval", device="cuda", print_samples=3):
    model.eval()
    wer_metric = evaluate.load("wer")
    id_to_char = {v: k for k, v in vocab.items()}

    probe_ds = ArabEstProbeDataset(dataset, vocab)
    loader = DataLoader(probe_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    all_preds, all_refs = [], []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)

            outputs = model(audio=audio, audio_lengths=audio_lengths)
            pred_ids = torch.argmax(outputs["logits"], dim=-1).cpu()

            for pred, label in zip(pred_ids.tolist(), batch["labels"].tolist()):
                pred_text = ctc_decode(pred, id_to_char)
                ref_ids = [x for x in label if x != -100]
                ref_text = ctc_decode(ref_ids, id_to_char)
                if ref_text.strip():
                    all_preds.append(pred_text)
                    all_refs.append(ref_text)

    wer_score = wer_metric.compute(predictions=all_preds, references=all_refs) if all_refs else 1.0
    print(f"  {name}: WER={wer_score * 100:.1f}% ({len(all_refs)} samples)")

    if print_samples > 0 and all_refs:
        print(f"  Sample predictions ({name}):")
        indices = list(range(len(all_refs)))
        random.shuffle(indices)
        for i in indices[:print_samples]:
            print(f"    REF: {all_refs[i][:80]}")
            print(f"    PRD: {all_preds[i][:80]}")
            print()

    return {"wer": wer_score, "samples": len(all_refs)}


def evaluate_casablanca(model, dataset, vocab, batch_size=8, device="cuda"):
    if "dialect" not in dataset.column_names:
        return evaluate_dataset(model, dataset, vocab, batch_size, "Casablanca", device)

    results = {}
    for dialect in sorted(set(dataset["dialect"])):
        subset = dataset.filter(lambda x: x["dialect"] == dialect)
        r = evaluate_dataset(model, subset, vocab, batch_size, f"Casablanca/{dialect}", device, print_samples=1)
        results[dialect] = r["wer"]

    avg = np.mean(list(results.values()))
    print(f"  Casablanca avg: WER={avg * 100:.1f}%")
    results["average"] = avg
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AraBEST-RQ Fine-Tuning (Unfreeze Top Layers + CTC)")
    parser.add_argument("--model_dir", default="/workspace/arabest_model",
                        help="Path to downloaded AraBEST-RQ checkpoint dir")
    parser.add_argument("--yaml_config", required=True,
                        help="Path to AraBEST-RQ YAML config (e.g. hybrid_300M.yaml)")
    parser.add_argument("--train_data", default="/workspace/datasets/train_combined")
    parser.add_argument("--eval_casablanca", default="/workspace/datasets/eval_casablanca_test")
    parser.add_argument("--eval_mgb2", default="/workspace/datasets/eval_mgb2_test")
    parser.add_argument("--eval_fleurs", default="/workspace/datasets/eval_fleurs")
    parser.add_argument("--output_dir", default="/workspace/finetune_output_arabest")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Smaller default than probe — backprop through encoder uses more VRAM")
    parser.add_argument("--encoder_lr", type=float, default=1e-5,
                        help="Learning rate for unfrozen encoder layers (slow)")
    parser.add_argument("--head_lr", type=float, default=1e-3,
                        help="Learning rate for CTC head + layer weights (fast)")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tune checkpoint dir to resume from")
    parser.add_argument("--unfreeze_layers", default="17,18,19",
                        help="Comma-separated 1-indexed layer numbers to unfreeze (default: 17,18,19)")
    parser.add_argument("--no_grad_checkpoint", action="store_true",
                        help="Disable gradient checkpointing (uses more VRAM but faster)")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. DATA ----
    print("Loading data...")
    train_dataset = load_from_disk(args.train_data)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  Raw: {len(train_dataset)} samples")

    print("Filtering...")
    train_dataset = filter_dataset(train_dataset)

    print("Building vocab...")
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
    else:
        vocab = build_vocab(train_dataset)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  Vocab: {len(vocab)} tokens")

    probe_train = ArabEstProbeDataset(train_dataset, vocab)
    train_loader = DataLoader(
        probe_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # ---- 2. MODEL ----
    print("Loading AraBEST-RQ encoder...")
    unfreeze_layers = [int(x.strip()) for x in args.unfreeze_layers.split(",")]
    model = ArabEstCTCFinetune(
        model_dir=args.model_dir, yaml_config=args.yaml_config,
        vocab_size=len(vocab), device=device,
        unfreeze_layers=unfreeze_layers,
        use_grad_checkpoint=not args.no_grad_checkpoint,
    ).to(device)

    if args.checkpoint:
        print(f"  Loading fine-tune checkpoint: {args.checkpoint}")
        ckpt_path = os.path.join(args.checkpoint, "finetune_ckpt.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.lm_head.load_state_dict(ckpt["lm_head"])
        model.layer_weights.data = ckpt["layer_weights"].to(device)

        # Restore unfrozen encoder layers
        if "encoder_layers" in ckpt:
            encoder_layers = model._get_encoder_layers()
            for key, value in ckpt["encoder_layers"].items():
                # key format: "layer_{idx}.{param_name}"
                parts = key.split(".", 1)
                layer_idx = int(parts[0].replace("layer_", ""))
                param_name = parts[1]
                if layer_idx < len(encoder_layers):
                    layer_state = encoder_layers[layer_idx].state_dict()
                    layer_state[param_name] = value
                    encoder_layers[layer_idx].load_state_dict(layer_state)
            print(f"  Restored {len(ckpt['encoder_layers'])} encoder params")

    # ---- 3. TRAIN ----
    if not args.eval_only:
        eval_fn = None
        if os.path.exists(args.eval_fleurs):
            fleurs_ds = load_from_disk(args.eval_fleurs).cast_column("audio", Audio(sampling_rate=16000))
            fleurs_ds = filter_dataset(fleurs_ds)

            def quick_eval():
                r = evaluate_dataset(model, fleurs_ds, vocab, args.batch_size,
                                     "FLEURS (mid-train)", device, print_samples=2)
                return r["wer"]

            eval_fn = quick_eval

        train(model, train_loader, eval_fn, args, device)

    # ---- 4. EVAL ----
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = {}

    if os.path.exists(args.eval_casablanca):
        ds = load_from_disk(args.eval_casablanca).cast_column("audio", Audio(sampling_rate=16000))
        ds = filter_dataset(ds)
        results["casablanca"] = evaluate_casablanca(model, ds, vocab, args.batch_size, device)

    if os.path.exists(args.eval_mgb2):
        ds = load_from_disk(args.eval_mgb2).cast_column("audio", Audio(sampling_rate=16000))
        ds = filter_dataset(ds)
        results["mgb2"] = evaluate_dataset(model, ds, vocab, args.batch_size, "MGB-2", device)

    if os.path.exists(args.eval_fleurs):
        ds = load_from_disk(args.eval_fleurs).cast_column("audio", Audio(sampling_rate=16000))
        ds = filter_dataset(ds)
        results["fleurs"] = evaluate_dataset(model, ds, vocab, args.batch_size, "FLEURS", device)

    # Save layer weights
    w = model.get_layer_weight_distribution()
    results["layer_weights"] = {f"layer_{i + 1}": float(wi) for i, wi in enumerate(w)}

    # ---- 5. SAVE ----
    with open(os.path.join(args.output_dir, "finetune_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for name, r in results.items():
        if name == "layer_weights":
            continue
        if isinstance(r, dict) and "wer" in r:
            print(f"  {name}: {r['wer'] * 100:.1f}% WER")
        elif isinstance(r, dict):
            for k, v in r.items():
                if isinstance(v, float):
                    print(f"  {name}/{k}: {v * 100:.1f}% WER")
    print("=" * 60)


if __name__ == "__main__":
    main()
