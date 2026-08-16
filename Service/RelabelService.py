# RelabelService.py
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from sqlalchemy.orm import Session

from Training.Model import Episode, Chunk


class RelabelService:
    def __init__(self, db: Session, model_path: str = "./checkpoints/run_004_20260208_121850/best"):
        self.db = db
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.model is None:
            print(f"Loading model from {self.model_path}...")
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            print(f"Loaded on {self.device.upper()}")
        return self.model

    def relabel_series(self, series_id: int) -> dict:
        self.load_model()

        chunks = self.db.query(Chunk).join(Episode).filter(
            Episode.series_id == series_id,
            Chunk.was_filtered == False
        ).all()

        stats = {"total": 0, "relabeled": 0, "failed": 0}

        print(f"\nRelabeling series {series_id}: {len(chunks)} chunks")

        import librosa
        with torch.no_grad():
            for chunk in tqdm(chunks, desc="Relabeling"):
                stats["total"] += 1

                if not Path(chunk.file_path).exists():
                    stats["failed"] += 1
                    continue

                try:
                    audio, sr = librosa.load(chunk.file_path, sr=16000)
                    inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
                    pred_ids = self.model.generate(
                        inputs.input_features,
                        language="ar",
                        task="transcribe"
                    )
                    new_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

                    # Save old, update new
                    chunk.original_transcription = chunk.transcription
                    chunk.transcription = new_text
                    stats["relabeled"] += 1

                except Exception as e:
                    print(f"Error {chunk.id}: {e}")
                    stats["failed"] += 1

        self.db.commit()
        print(f"Done: {stats}")
        return stats


def get_relabel_service(db: Session, model_path: str = "./checkpoints/run_004_20260208_121850/best") -> RelabelService:
    return RelabelService(db, model_path)