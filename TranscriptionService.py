import torch
import whisper
from pathlib import Path
from tqdm import tqdm

from sqlalchemy.orm import Session

from Model import Episode, Chunk, ProcessingStatus


class TranscriptionService:
    def __init__(self, db: Session, model_name: str = "large-v3"):
        self.db = db
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.model is None:
            print(f"Loading Whisper {self.model_name}...")
            self.model = whisper.load_model(self.model_name)
            print(f"Loaded on {self.device.upper()}")
        return self.model

    def transcribe_episode(
            self,
            episode_id: int,
            language: str = "ar",
            overwrite: bool = False,
    ) -> dict:
        self.load_model()

        episode = self.db.query(Episode).filter(Episode.id == episode_id).first()
        if not episode:
            raise ValueError(f"Episode {episode_id} not found")

        chunks = self.db.query(Chunk).filter(Chunk.episode_id == episode_id).order_by(Chunk.chunk_index).all()

        stats = {"total": 0, "transcribed": 0, "skipped": 0, "failed": 0}

        print(f"\nTranscribing Episode {episode_id}: {episode.name}")
        print(f"   {len(chunks)} chunks")

        for chunk in tqdm(chunks, desc=f"Ep {episode_id}"):
            stats["total"] += 1

            # Skip if already has transcription
            if chunk.transcription and not overwrite:
                stats["skipped"] += 1
                continue

            # Check file exists
            if not Path(chunk.file_path).exists():
                stats["failed"] += 1
                continue

            try:
                self._transcribe_chunk(chunk, language)
                stats["transcribed"] += 1
            except Exception as e:
                print(f"   Failed {chunk.filename}: {e}")
                stats["failed"] += 1

        self.db.commit()
        print(f"   Done: {stats}")
        return stats

    def transcribe_series(
            self,
            series_id: int,
            language: str = "ar",
            overwrite: bool = False,
    ) -> dict:
        episodes = self.db.query(Episode).filter(Episode.series_id == series_id).all()

        total_stats = {"total": 0, "transcribed": 0, "skipped": 0, "failed": 0, "episodes": 0}

        for episode in episodes:
            stats = self.transcribe_episode(episode.id, language, overwrite)
            total_stats["total"] += stats["total"]
            total_stats["transcribed"] += stats["transcribed"]
            total_stats["skipped"] += stats["skipped"]
            total_stats["failed"] += stats["failed"]
            total_stats["episodes"] += 1

        return total_stats

    def _transcribe_chunk(self, chunk: Chunk, language: str):
        result = self.model.transcribe(
            chunk.file_path,
            language=language,
            word_timestamps=True
        )

        # Update chunk directly (no separate Transcription table)
        chunk.transcription = result["text"].strip()
        chunk.is_cleaned = False  # Needs cleaning
        chunk.was_filtered = False
        chunk.source = "whisper"

        self.db.flush()


def get_transcription_service(db: Session, model_name: str = "large-v3") -> TranscriptionService:
    return TranscriptionService(db, model_name)