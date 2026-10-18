"""
AraBEST-RQ Frozen Encoder Probe — Weighted Layer Sum
Learns a weighted combination of ALL conformer layers from any AraBEST-RQ
checkpoint (300M, 600M, etc.). Architecture is inferred from the checkpoint
and YAML config — no hardcoded dimensions.

Usage:
  # First download the model on your pod:
  pip install speechbrain huggingface_hub hyperpyyaml
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Elyadata/AraBEST-RQ-300M-6k', local_dir='/workspace/arabest_model')"

  # Then run (point --yaml_config at the matching YAML):
  python train_probe_arabest.py \
    --model_dir /workspace/arabest_model \
    --yaml_config /workspace/AraBEST-RQ/model/hparams/hybrid_300M.yaml \
    --train_data /workspace/datasets/train_combined \
    --output_dir /workspace/probe_output_arabest
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
# DATA
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
    """Filter out invalid samples.

    AraBEST-RQ uses Fbank (hop=10ms) + CNN (stride 2x2 = 4x downsample).
    So encoder frames ≈ (audio_seconds * 100) / 4 = 25 frames/sec.
    """
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
        # Encoder frames after Fbank + CNN 4x downsample
        fbank_frames = len(array) // 160 + 1  # hop_length=10ms at 16kHz
        encoder_frames = fbank_frames // 4     # CNN stride 2x2
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
    """Wraps a HuggingFace dataset for raw waveform access."""

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
    """Pad waveforms and labels to max length in batch."""
    audios = [torch.from_numpy(item["audio"]) for item in batch]
    labels_list = [item["labels"] for item in batch]

    # Pad waveforms
    max_audio = max(a.shape[0] for a in audios)
    padded_audio = torch.zeros(len(audios), max_audio)
    audio_lengths = torch.zeros(len(audios))  # relative lengths for SpeechBrain
    for i, a in enumerate(audios):
        padded_audio[i, : a.shape[0]] = a
        audio_lengths[i] = a.shape[0] / max_audio  # SpeechBrain uses relative lengths

    # Pad labels
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
# MODEL
# ============================================================

class ArabEstCTCProbe(nn.Module):
    """Frozen AraBEST-RQ encoder with learnable weighted-layer-sum + CTC head.

    Architecture is built from the YAML config — works with any AraBEST-RQ
    model size (300M, 600M, etc.) without code changes.
    """

    def __init__(self, model_dir, yaml_config, vocab_size, pad_token_id=0, device="cpu"):
        super().__init__()

        # ---- Load YAML config — builds all model objects automatically ----
        print(f"  Loading config: {yaml_config}")
        with open(yaml_config, encoding="utf-8") as f:
            hparams = load_hyperpyyaml(f)

        # Extract the objects hyperpyyaml instantiated
        self.compute_features = hparams["compute_features"]
        self.normalize = hparams["normalize"]

        # model is nn.ModuleList([CNN, wrapper])
        model_list = hparams["model"]
        self.cnn = model_list[0]
        self.wrapper = model_list[1]
        self.transformer = hparams["Transformer"]

        # ---- Load pretrained weights ----
        model_ckpt_path = os.path.join(model_dir, "model.ckpt")
        norm_ckpt_path = os.path.join(model_dir, "normalizer.ckpt")

        if not os.path.exists(model_ckpt_path):
            # Search for it (snapshot_download may nest files)
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

        # Load normalizer
        if os.path.exists(norm_ckpt_path):
            print(f"  Loading normalizer: {norm_ckpt_path}")
            # A SpeechBrain stats dict (count/glob_mean/glob_std), not a state_dict
            self.normalize._load(norm_ckpt_path)
            self.normalize.eval()
        else:
            print("  WARNING: normalizer.ckpt not found — using unnormalized features")

        # ---- Enable hidden state extraction ----
        self.transformer.encoder.output_hidden_states = True
        self.transformer.output_hidden_states = True

        # ---- Freeze everything ----
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.wrapper.parameters():
            param.requires_grad = False
        for param in self.compute_features.parameters():
            param.requires_grad = False
        for param in self.normalize.parameters():
            param.requires_grad = False
        self.cnn.eval()
        self.wrapper.eval()

        # ---- Probe head — dimensions read from the config, not hardcoded ----
        num_layers = hparams["num_encoder_layers"]
        hidden_size = hparams["d_model"]

        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=pad_token_id, reduction="mean", zero_infinity=True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Params: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.4f}%)")
        print(f"  Layer weights: {num_layers} layers, d_model={hidden_size}")

    def train(self, mode=True):
        """Override: encoder/CNN/normalizer always stay in eval mode."""
        super().train(mode)
        self.cnn.eval()
        self.wrapper.eval()
        self.normalize.eval()
        return self

    def get_layer_weight_distribution(self):
        """Return softmax-normalized layer weights for inspection."""
        return torch.softmax(self.layer_weights, dim=0).detach().cpu().numpy()

    def forward(self, audio, audio_lengths, labels=None):
        """
        Forward pass:
          raw waveform → Fbank → normalize → CNN → Conformer (all layers) →
          weighted sum → lm_head → CTC loss
        """
        with torch.no_grad():
            # 1. Extract Fbank features (80-dim log mel)
            feats = self.compute_features(audio)  # (B, T_fbank, 80)

            # 2. Sentence-level normalization
            feats = self.normalize(feats, audio_lengths, epoch=0)

            # 3. Pad time dimension to be divisible by 4 (for CNN stride alignment)
            B, T, C = feats.shape
            pad_needed = (4 - T % 4) % 4
            if pad_needed > 0:
                feats = torch.nn.functional.pad(feats, [0, 0, 0, pad_needed])

            # 4. CNN frontend
            src = self.cnn(feats)  # (B, T/4, 20, 64) → flattened to (B, T/4, 1280) by TransformerASR

            # 5. Conformer encoder — returns (enc_out, hidden_states) since output_hidden_states=True
            result = self.wrapper(src, audio_lengths)
            enc_out, hidden_states = result

            # hidden_states has 25 elements: [input_embedding] + [24 layer outputs]
            # Skip index 0 (input projection), use 1..24 (conformer layer outputs)
            layer_outputs = torch.stack(hidden_states[1:], dim=0)  # (24, B, T', 1024)

        # 6. Weighted sum across layers (learnable, softmax-normalized)
        weights = torch.softmax(self.layer_weights, dim=0)  # (24,)
        weights = weights.view(-1, 1, 1, 1)  # (24, 1, 1, 1) for broadcasting
        blended = (weights * layer_outputs).sum(dim=0)  # (B, T', 1024)

        # 7. CTC projection
        logits = self.lm_head(blended)  # (B, T', vocab_size)

        loss = None
        if labels is not None:
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)  # (T', B, vocab)

            # Compute actual output lengths from relative audio lengths
            # After Fbank + CNN (4x downsample), output frames ≈ (fbank_frames + pad) / 4
            output_T = logits.shape[1]
            input_lengths = (audio_lengths * output_T).long().clamp(min=1, max=output_T)

            label_mask = labels != -100
            label_lengths = label_mask.sum(dim=-1)

            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = 0

            loss = self.ctc_loss(log_probs, labels_clean, input_lengths, label_lengths)

        return {"loss": loss, "logits": logits}


# ============================================================
# TRAIN
# ============================================================

def train(model, train_loader, eval_fn, args, device):
    # Optimize layer weights + lm_head
    trainable_params = list(model.lm_head.parameters()) + [model.layer_weights]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # Warmup + cosine decay
    total_steps = min(args.max_steps, len(train_loader) * args.epochs)
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_wer = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()  # encoder stays eval via override
        epoch_losses = []
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in bar:
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Forward
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(audio=audio, audio_lengths=audio_lengths, labels=labels)
                    loss = outputs["loss"]
            else:
                outputs = model(audio=audio, audio_lengths=audio_lengths, labels=labels)
                loss = outputs["loss"]

            # Backward
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            scheduler.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

            # Log
            if global_step % args.log_every == 0:
                avg_loss = np.mean(epoch_losses[-100:])
                lr = scheduler.get_last_lr()[0]
                bar.set_postfix({
                    "loss": f"{loss_val:.3f}",
                    "avg": f"{avg_loss:.3f}",
                    "lr": f"{lr:.1e}",
                    "step": global_step,
                })

            # Print layer weights periodically
            if global_step % (args.save_every // 2) == 0:
                w = model.get_layer_weight_distribution()
                top_3 = np.argsort(w)[-3:][::-1]
                top_str = ", ".join([f"L{i + 1}={w[i]:.3f}" for i in top_3])
                print(f"  Layer weights (step {global_step}): top 3 = {top_str}")

            # Save + mid-training eval
            if global_step % args.save_every == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    "lm_head": model.lm_head.state_dict(),
                    "layer_weights": model.layer_weights.data,
                }, os.path.join(save_dir, "probe_head.pt"))

                if eval_fn:
                    mid_wer = eval_fn()
                    if mid_wer < best_wer:
                        best_wer = mid_wer
                        best_dir = os.path.join(args.output_dir, "best")
                        os.makedirs(best_dir, exist_ok=True)
                        torch.save({
                            "lm_head": model.lm_head.state_dict(),
                            "layer_weights": model.layer_weights.data,
                        }, os.path.join(best_dir, "probe_head.pt"))
                        print(f"  *** New best WER: {best_wer * 100:.1f}% ***")

            if global_step >= args.max_steps:
                break

        print(f"  Epoch {epoch + 1}: loss={np.mean(epoch_losses):.4f}")
        if global_step >= args.max_steps:
            break

    # Save final
    torch.save({
        "lm_head": model.lm_head.state_dict(),
        "layer_weights": model.layer_weights.data,
    }, os.path.join(args.output_dir, "probe_head.pt"))

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


# ============================================================
# EVAL
# ============================================================

def ctc_decode(ids, id_to_char):
    decoded = []
    prev = None
    for i in ids:
        if i == 0:  # blank
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
    parser = argparse.ArgumentParser(description="AraBEST-RQ Probe (Weighted Layer Sum + CTC)")
    parser.add_argument("--model_dir", default="/workspace/arabest_model",
                        help="Path to downloaded AraBEST-RQ checkpoint dir")
    parser.add_argument("--yaml_config", required=True,
                        help="Path to AraBEST-RQ YAML config (e.g. hybrid_300M.yaml)")
    parser.add_argument("--train_data", default="/workspace/datasets/train_combined")
    parser.add_argument("--eval_casablanca", default="/workspace/datasets/eval_casablanca_test")
    parser.add_argument("--eval_mgb2", default="/workspace/datasets/eval_mgb2_test")
    parser.add_argument("--eval_fleurs", default="/workspace/datasets/eval_fleurs")
    parser.add_argument("--output_dir", default="/workspace/probe_output_arabest")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to probe checkpoint dir to resume from")
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

    # Wrap in probe dataset + dataloader
    probe_train = ArabEstProbeDataset(train_dataset, vocab)
    train_loader = DataLoader(
        probe_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # ---- 2. MODEL ----
    print("Loading AraBEST-RQ encoder...")
    model = ArabEstCTCProbe(
        model_dir=args.model_dir, yaml_config=args.yaml_config,
        vocab_size=len(vocab), device=device,
    ).to(device)

    if args.checkpoint:
        print(f"  Loading probe checkpoint: {args.checkpoint}")
        ckpt = torch.load(os.path.join(args.checkpoint, "probe_head.pt"), map_location=device)
        model.lm_head.load_state_dict(ckpt["lm_head"])
        model.layer_weights.data = ckpt["layer_weights"].to(device)

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
    with open(os.path.join(args.output_dir, "probe_results.json"), "w") as f:
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
