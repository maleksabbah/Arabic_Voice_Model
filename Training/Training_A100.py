"""
LoRA training for Whisper Large V3 using Seq2SeqTrainer.
Uses HuggingFace Trainer which handles peft/Whisper compatibility correctly.

Usage:
  python Training/Training_A100.py --db /workspace/asr.db --series 1 2 3 --epochs 3
"""
import argparse
import os
import sqlite3
import random
import numpy as np
import torch
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import evaluate


# ============================================================
# DATASET
# ============================================================
class ChunkDataset(Dataset):
    def __init__(self, chunks, processor):
        self.chunks = chunks
        self.processor = processor

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        c = self.chunks[idx]
        try:
            audio, sr = sf.read(c["file_path"], dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            max_samples = int(30 * sr)
            audio = audio[:max_samples]
        except:
            audio = np.zeros(16000, dtype=np.float32)

        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(
            c["transcription"],
            return_tensors="pt",
            truncation=True,
            max_length=448,
        ).input_ids[0]

        return {"input_features": input_features, "labels": labels}


# ============================================================
# DATA COLLATOR
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ============================================================
# METRICS
# ============================================================
def make_compute_metrics(processor):
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer_val = metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer_val}

    return compute_metrics


# ============================================================
# DATA LOADING
# ============================================================
def get_labeled_chunks(db_path, series_ids):
    conn = sqlite3.connect(db_path)
    ph = ",".join(str(s) for s in series_ids)
    rows = conn.execute(f"""
        SELECT c.id, c.file_path, c.transcription, e.series_id
        FROM chunks c JOIN episodes e ON c.episode_id = e.id
        WHERE e.series_id IN ({ph})
        AND c.transcription IS NOT NULL AND c.transcription != ''
    """).fetchall()
    conn.close()
    chunks = [{"id": r[0], "file_path": r[1], "transcription": r[2], "series_id": r[3]} for r in rows]
    chunks = [c for c in chunks if os.path.exists(c["file_path"])]
    return chunks


# ============================================================
# MAIN
# ============================================================
def train(args):
    print("=" * 60)
    print("LORA TRAINING — WHISPER LARGE V3 (Seq2SeqTrainer)")
    print("=" * 60)
    print(f"DB: {args.db}")
    print(f"Series: {args.series}")
    print(f"Epochs: {args.epochs}")
    print(f"Rank: {args.rank}, Alpha: {args.alpha}")
    print(f"LR: {args.lr}")
    print(f"Batch: {args.batch_size} x {args.accumulation}")
    print()

    # Load data
    print("[DATA] Loading...")
    chunks = get_labeled_chunks(args.db, args.series)
    print(f"[DATA] {len(chunks)} labeled chunks")

    if not chunks:
        print("No data!")
        return

    random.seed(42)
    random.shuffle(chunks)
    val_size = min(1000, int(len(chunks) * 0.05))
    val_chunks = chunks[:val_size]
    train_chunks = chunks[val_size:]
    print(f"[DATA] Train: {len(train_chunks)}, Val: {len(val_chunks)}")

    # Load model
    print("[MODEL] Loading whisper-large-v3...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16,
    )

    # Set language and task
    model.generation_config.language = "ar"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Apply LoRA
    print(f"[MODEL] LoRA rank={args.rank}, alpha={args.alpha}")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = ChunkDataset(train_chunks, processor)
    val_dataset = ChunkDataset(val_chunks, processor)

    # Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="/workspace/checkpoints",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=448,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=4,
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    print(f"\n[TRAIN] Starting...")
    trainer.train()

    # Save best
    print("[SAVE] Saving best model...")
    model.save_pretrained("/workspace/checkpoints/lora_best")
    processor.save_pretrained("/workspace/checkpoints/lora_best")

    # Final eval
    metrics = trainer.evaluate()
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final WER: {metrics.get('eval_wer', 'N/A')}")
    print(f"Checkpoints: /workspace/checkpoints/")
    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--series", type=int, nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    a = p.parse_args()
    train(a)


if __name__ == "__main__":
    main()