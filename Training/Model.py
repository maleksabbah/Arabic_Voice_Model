from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, String, Float, Text,
    ForeignKey, Enum, Boolean, UniqueConstraint
)
from sqlalchemy.orm import relationship

from Config.Database import Base


class ProcessingStatus(str, PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Series(Base):
    __tablename__ = "series"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    language = Column(String(10), default="ar")
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())

    episodes = relationship("Episode", back_populates="series", cascade="all, delete-orphan")


class Episode(Base):
    __tablename__ = "episodes"
    __table_args__ = (
        UniqueConstraint('series_id', 'episode_number', name='unique_series_episode'),
    )

    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(Integer, ForeignKey("series.id"), nullable=False)
    name = Column(String(255), nullable=False)
    episode_number = Column(Integer, nullable=True)
    original_file = Column(String(500), nullable=True)
    audio_file = Column(String(500), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    status_message = Column(Text, nullable=True)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at = Column(String, default=lambda: datetime.utcnow().isoformat(), onupdate=lambda: datetime.utcnow().isoformat())

    series = relationship("Series", back_populates="episodes")
    chunks = relationship("Chunk", back_populates="episode", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())

    # Transcription fields
    transcription = Column(Text, nullable=True)
    transcription_cleaned = Column(Text, nullable=True)
    is_cleaned = Column(Boolean, default=False)
    was_filtered = Column(Boolean, default=False)
    filter_reason = Column(String(255), nullable=True)
    source = Column(String(50), nullable=True)

    episode = relationship("Episode", back_populates="chunks")


class ScrapingJob(Base):
    __tablename__ = "scraping_jobs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    total_channels = Column(Integer, default=0)
    channels_completed = Column(Integer, default=0)
    total_videos_found = Column(Integer, default=0)
    total_duration_hours = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at = Column(String, default=lambda: datetime.utcnow().isoformat(),
                        onupdate=lambda: datetime.utcnow().isoformat())

    videos = relationship("VideoQueue", back_populates="job", cascade="all, delete-orphan")


class VideoQueue(Base):
    __tablename__ = "video_queue"
    __table_args__ = (
        UniqueConstraint('video_id', name='unique_video_id'),
    )

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("scraping_jobs.id"), nullable=True)
    video_id = Column(String(20), nullable=False, index=True)
    channel_handle = Column(String(255), nullable=False)
    dialect = Column(String(50), nullable=False)
    sub_dialect = Column(String(50), nullable=True)
    genre = Column(String(50), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    title = Column(String(500), nullable=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    error_message = Column(Text, nullable=True)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())

    job = relationship("ScrapingJob", back_populates="videos")
