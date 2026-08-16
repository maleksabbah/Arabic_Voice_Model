from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Whisper ASR Pipeline"
    debug: bool = True

    # Database
    database_url: str = "sqlite:///./storage/asr.db"

    # Storage paths
    storage_root: Path = Path("D:/tarjma_storage")
    uploads_dir: Path = Path("D:/tarjma_storage/uploads")
    audio_dir: Path = Path("D:/tarjma_storage/audio")
    chunks_dir: Path = Path("D:/tarjma_storage/chunks")

    # Chunking defaults
    max_chunk_seconds: int = 25
    min_silence_duration: float = 0.5
    silence_threshold_db: int = -40

    # Whisper (for later)
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    default_language: str = "ar"

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist (handle symlinks)
for dir_path in [settings.uploads_dir, settings.audio_dir, settings.chunks_dir]:
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)