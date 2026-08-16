"""
CLI for consensus transcription.

Usage:
    # Transcribe all untranscribed chunks in an episode
    python transcribe_consensus.py --episode 42

    # Transcribe all episodes in a series
    python transcribe_consensus.py --series 10

    # Overwrite existing transcriptions
    python transcribe_consensus.py --episode 42 --overwrite

    # With your LoRA checkpoint as one of the models
    python transcribe_consensus.py --series 10 --lora checkpoints/lora_017_.../epoch_01

    # Include SeamlessM4T as third model (needs more VRAM)
    python transcribe_consensus.py --episode 42 --seamless

    # Tune consensus thresholds
    python transcribe_consensus.py --episode 42 --max-distance 0.3 --min-agreement 0.7 --min-models 2

    # Dry run — show what would happen without saving
    python transcribe_consensus.py --episode 42 --dry-run
"""

import argparse
import sys

from Config.Database import get_db
from Training.Model import Episode, Chunk
from TranscriptionService import (
    get_consensus_service,
    get_single_model_service,
)


def main():
    parser = argparse.ArgumentParser(description="Consensus transcription CLI")

    # Target selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode", type=int, help="Episode ID to transcribe")
    group.add_argument("--series", type=int, help="Series ID (all episodes)")

    # Model config
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--seamless", action="store_true", help="Include SeamlessM4T as third model")
    parser.add_argument("--single", action="store_true", help="Single model mode (no consensus, just Whisper Large)")

    # Consensus tuning
    parser.add_argument("--max-distance", type=float, default=0.4, help="Max Levenshtein distance ratio to count as agreement (default: 0.4)")
    parser.add_argument("--min-agreement", type=float, default=0.6, help="Min fraction of model pairs that must agree (default: 0.6)")
    parser.add_argument("--min-models", type=int, default=2, help="Min models that must be close to best (default: 2)")

    # Behavior
    parser.add_argument("--language", type=str, default="ar", help="Language code (default: ar)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing transcriptions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be transcribed without saving")

    args = parser.parse_args()

    db = next(get_db())

    # Validate target exists
    if args.episode:
        episode = db.query(Episode).filter(Episode.id == args.episode).first()
        if not episode:
            print(f"Episode {args.episode} not found")
            sys.exit(1)
        chunk_count = db.query(Chunk).filter(Chunk.episode_id == args.episode).count()
        untranscribed = db.query(Chunk).filter(
            Chunk.episode_id == args.episode,
            Chunk.transcription.is_(None),
        ).count()
        print(f"Episode {args.episode}: {episode.name}")
        print(f"  {chunk_count} chunks, {untranscribed} untranscribed")

    elif args.series:
        episodes = db.query(Episode).filter(Episode.series_id == args.series).all()
        if not episodes:
            print(f"No episodes found in series {args.series}")
            sys.exit(1)
        total_chunks = 0
        total_untranscribed = 0
        for ep in episodes:
            c = db.query(Chunk).filter(Chunk.episode_id == ep.id).count()
            u = db.query(Chunk).filter(
                Chunk.episode_id == ep.id,
                Chunk.transcription.is_(None),
            ).count()
            total_chunks += c
            total_untranscribed += u
        print(f"Series {args.series}: {len(episodes)} episodes")
        print(f"  {total_chunks} chunks, {total_untranscribed} untranscribed")

    if args.dry_run:
        print("\n[DRY RUN] Would transcribe the above. Exiting.")
        sys.exit(0)

    # Build transcription service
    if args.single:
        print("\nMode: Single model (Whisper Large V3, no consensus)")
        service = get_single_model_service(db)
    else:
        print(f"\nMode: Consensus transcription")
        print(f"  LoRA: {args.lora or 'none'}")
        print(f"  SeamlessM4T: {'yes' if args.seamless else 'no'}")
        print(f"  Max distance: {args.max_distance}")
        print(f"  Min agreement: {args.min_agreement}")
        print(f"  Min models: {args.min_models}")

        service = get_consensus_service(
            db=db,
            lora_path=args.lora,
            include_seamless=args.seamless,
            max_char_distance_ratio=args.max_distance,
            min_agreement_ratio=args.min_agreement,
            min_models_agree=args.min_models,
        )

    # Run
    if args.episode:
        result = service.transcribe_episode(
            episode_id=args.episode,
            language=args.language,
            overwrite=args.overwrite,
        )
        print(f"\nResult: {result}")

    elif args.series:
        result = service.transcribe_series(
            series_id=args.series,
            language=args.language,
            overwrite=args.overwrite,
        )
        print(f"\nResult: {result}")


if __name__ == "__main__":
    main()