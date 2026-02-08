import argparse
from Database import get_db
from ModelTraining import TrainingService


def main():
    parser = argparse.ArgumentParser(description="Train Whisper model on Arabic dialects")

    # Series arguments (can specify multiple)
    parser.add_argument("--series", type=str, action="append",
                        help="Series config as 'id:ratio' (e.g., --series 1:1.0 --series 2:0.5)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")

    # Model arguments
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--source-model", type=str,
                        default=r"C:\Users\Ali\Downloads\whisper_cedars-20260126T153241Z-3-001",
                        help="Source model (HuggingFace name or local path)")

    args = parser.parse_args()

    # Parse series configs
    if args.series:
        series_configs = []
        for s in args.series:
            parts = s.split(":")
            series_id = int(parts[0])
            ratio = float(parts[1]) if len(parts) > 1 else 1.0
            series_configs.append({"series_id": series_id, "ratio": ratio})
    else:
        series_configs = [
            {"series_id": 1, "ratio": 1.0},
            {"series_id": 2, "ratio": 1.0},
        ]

    print(f"Training config:")
    print(f"  Series: {series_configs}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Source model: {args.source_model}")
    print(f"  Checkpoints dir: {args.checkpoints_dir}")
    print()

    db = next(get_db())
    trainer = TrainingService(
        db=db,
        checkpoints_dir=args.checkpoints_dir,
        source_model=args.source_model
    )
    trainer.train(
        series_configs=series_configs,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()