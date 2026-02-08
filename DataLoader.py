import json
from pathlib import Path
from typing import List, Optional, Union

from datasets import Dataset, Audio, concatenate_datasets
from sqlalchemy.orm import Session

from Model import Series, Episode, Chunk


class DataLoader:
    def __init__(self, db: Session):
        self.db = db

    def load_datasets(
            self,
            series_id: Optional[int] = None,
            episode_ids: Optional[List[int]] = None,
            episode_numbers: Optional[List[int]] = None,
            skip_filtered: bool = True,
            min_duration: float = 0.5,
            max_duration: float = 30.0,
    ) -> Dataset:
        """
        Load chunks as a Dataset.

        Args:
            series_id: Filter by series ID
            episode_ids: Filter by specific episode IDs (database IDs)
            episode_numbers: Filter by episode numbers (e.g., [1, 2, 3])
            skip_filtered: Skip chunks marked as filtered
            min_duration: Minimum chunk duration
            max_duration: Maximum chunk duration
        """
        query = self.db.query(Chunk).join(Episode)

        if series_id:
            query = query.filter(Episode.series_id == series_id)
        if episode_ids:
            query = query.filter(Chunk.episode_id.in_(episode_ids))
        if episode_numbers and series_id:
            query = query.filter(Episode.episode_number.in_(episode_numbers))

        chunks = query.all()

        data = {"audio": [], "text": [], "series": [], "episode": []}
        skipped = {"no_text": 0, "filtered": 0, "too_short": 0, "too_long": 0, "no_file": 0}

        for chunk in chunks:
            # Skip if no transcription
            if not chunk.transcription:
                skipped["no_text"] += 1
                continue

            # Skip if filtered
            if skip_filtered and chunk.was_filtered:
                skipped["filtered"] += 1
                continue

            # Skip wrong duration
            if chunk.duration and chunk.duration < min_duration:
                skipped["too_short"] += 1
                continue
            if chunk.duration and chunk.duration > max_duration:
                skipped["too_long"] += 1
                continue

            # Check file exists
            if not Path(chunk.file_path).exists():
                skipped["no_file"] += 1
                continue

            # Use cleaned text if available
            text = chunk.transcription_cleaned or chunk.transcription

            data["audio"].append(chunk.file_path)
            data["text"].append(text)
            data["series"].append(chunk.episode.series.name if chunk.episode.series else "unknown")
            data["episode"].append(chunk.episode.episode_number)

        print(f"Loaded {len(data['audio'])} chunks")
        if any(skipped.values()):
            print(f"Skipped: {skipped}")

        dataset = Dataset.from_dict(data)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        return dataset

    def load_multiple_series(
            self,
            series_config: List[dict],
            skip_filtered: bool = True,
    ) -> Dataset:
        """
        Load multiple series/episodes with ratios.

        Config format:
            [
                {"series_id": 1, "ratio": 1.0},  # All of series 1
                {"series_id": 2, "ratio": 0.5},  # 50% of series 2
                {"series_id": 3, "episodes": [1, 2, 3]},  # Only eps 1-3 of series 3
                {"series_id": 4, "episodes": [1, 2], "ratio": 0.5},  # 50% of eps 1-2
                {
                    "series_id": 5,
                    "episodes": {
                        1: 1.0,   # Episode 1 at 100%
                        2: 0.5,   # Episode 2 at 50%
                        3: 0.25,  # Episode 3 at 25%
                    }
                },
            ]
        """
        datasets = []

        for config in series_config:
            series_id = config["series_id"]
            ratio = config.get("ratio", 1.0)
            episodes = config.get("episodes", None)

            # Get series name for logging
            series = self.db.query(Series).filter(Series.id == series_id).first()
            series_name = series.name if series else f"Series {series_id}"

            # Handle episode-level ratios
            if isinstance(episodes, dict):
                # episodes = {ep_num: ratio, ...}
                for ep_num, ep_ratio in episodes.items():
                    ds = self.load_datasets(
                        series_id=series_id,
                        episode_numbers=[int(ep_num)],
                        skip_filtered=skip_filtered
                    )

                    if len(ds) == 0:
                        print(f"  {series_name} Ep {ep_num}: 0 samples (empty)")
                        continue

                    # Apply ratio
                    if ep_ratio < 1.0:
                        n_samples = max(1, int(len(ds) * ep_ratio))
                        ds = ds.shuffle(seed=42).select(range(n_samples))

                    datasets.append(ds)
                    print(f"  {series_name} Ep {ep_num}: {len(ds)} samples (ratio={ep_ratio})")

            # Handle episode list (all with same ratio)
            elif isinstance(episodes, list):
                ds = self.load_datasets(
                    series_id=series_id,
                    episode_numbers=episodes,
                    skip_filtered=skip_filtered
                )

                if len(ds) == 0:
                    print(f"  {series_name} Eps {episodes}: 0 samples (empty)")
                    continue

                # Apply ratio
                if ratio < 1.0:
                    n_samples = max(1, int(len(ds) * ratio))
                    ds = ds.shuffle(seed=42).select(range(n_samples))

                datasets.append(ds)
                print(f"  {series_name} Eps {episodes}: {len(ds)} samples (ratio={ratio})")

            # Handle full series
            else:
                ds = self.load_datasets(
                    series_id=series_id,
                    skip_filtered=skip_filtered
                )

                if len(ds) == 0:
                    print(f"  {series_name}: 0 samples (empty)")
                    continue

                # Apply ratio
                if ratio < 1.0:
                    n_samples = max(1, int(len(ds) * ratio))
                    ds = ds.shuffle(seed=42).select(range(n_samples))

                datasets.append(ds)
                print(f"  {series_name}: {len(ds)} samples (ratio={ratio})")

        if not datasets:
            raise ValueError("No data loaded! Check your series configs.")

        combined = concatenate_datasets(datasets)
        combined = combined.shuffle(seed=42)

        print(f"\nTotal: {len(combined)} samples")
        return combined

    def export_json(
            self,
            output_path: Path,
            series_id: Optional[int] = None,
            episode_numbers: Optional[List[int]] = None,
            skip_filtered: bool = True,
    ) -> str:
        """Export chunks to JSON lines file."""
        query = self.db.query(Chunk).join(Episode)

        if series_id:
            query = query.filter(Episode.series_id == series_id)
        if episode_numbers and series_id:
            query = query.filter(Episode.episode_number.in_(episode_numbers))

        chunks = query.all()
        count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                if not chunk.transcription:
                    continue
                if skip_filtered and chunk.was_filtered:
                    continue

                entry = {
                    "audio_path": chunk.file_path,
                    "text": chunk.transcription_cleaned or chunk.transcription,
                    "duration": chunk.duration,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "episode": chunk.episode.episode_number,
                    "series": chunk.episode.series.name if chunk.episode.series else None,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        print(f"Exported {count} chunks to {output_path}")
        return str(output_path)

    def get_stats(
            self,
            series_id: Optional[int] = None,
            episode_numbers: Optional[List[int]] = None,
    ) -> dict:
        """Get statistics for chunks."""
        query = self.db.query(Chunk).join(Episode)

        if series_id:
            query = query.filter(Episode.series_id == series_id)
        if episode_numbers and series_id:
            query = query.filter(Episode.episode_number.in_(episode_numbers))

        chunks = query.all()

        total = len(chunks)
        with_text = sum(1 for c in chunks if c.transcription)
        filtered = sum(1 for c in chunks if c.was_filtered)
        durations = [c.duration for c in chunks if c.duration]

        return {
            "total_chunks": total,
            "with_transcription": with_text,
            "filtered_out": filtered,
            "usable": with_text - filtered,
            "total_duration_hours": sum(durations) / 3600 if durations else 0,
            "avg_duration_sec": sum(durations) / len(durations) if durations else 0,
            "min_duration_sec": min(durations) if durations else 0,
            "max_duration_sec": max(durations) if durations else 0,
        }

    def list_episodes(self, series_id: int) -> List[dict]:
        """List all episodes in a series with chunk counts."""
        episodes = self.db.query(Episode).filter(Episode.series_id == series_id).all()

        result = []
        for ep in episodes:
            chunk_count = self.db.query(Chunk).filter(Chunk.episode_id == ep.id).count()
            transcribed = self.db.query(Chunk).filter(
                Chunk.episode_id == ep.id,
                Chunk.transcription.isnot(None)
            ).count()

            result.append({
                "episode_id": ep.id,
                "episode_number": ep.episode_number,
                "name": ep.name,
                "chunks": chunk_count,
                "transcribed": transcribed,
            })

        return sorted(result, key=lambda x: x["episode_number"] or 0)


def get_data_loader(db: Session) -> DataLoader:
    return DataLoader(db)