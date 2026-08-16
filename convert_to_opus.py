"""
Convert all WAV chunks to Opus and update the database paths.
Run from your project root:
    python convert_to_opus.py

This will:
1. Find all .wav files in D:\tarjma_storage\chunks\
2. Convert each to .opus using ffmpeg
3. Update the file_path in asr.db to point to the .opus file
4. Delete the original .wav file
"""
import os
import subprocess
from pathlib import Path
from sqlalchemy import create_engine, text

# Config
CHUNKS_DIR = Path("D:/tarjma_storage/chunks")
DB_PATH = "storage/asr.db"


def convert_wav_to_opus(wav_path: Path) -> Path:
    """Convert a WAV file to Opus. Returns the opus path."""
    opus_path = wav_path.with_suffix(".opus")
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-c:a", "libopus", "-b:a", "32k",  # 32kbps is plenty for speech
        "-ar", "16000", "-ac", "1",
        str(opus_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR converting {wav_path.name}: {result.stderr[:100]}")
        return None
    return opus_path


def main():
    # Find all WAV files
    wav_files = list(CHUNKS_DIR.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files in {CHUNKS_DIR}")

    if not wav_files:
        print("No WAV files found!")
        return

    # Calculate current size
    total_wav_size = sum(f.stat().st_size for f in wav_files)
    print(f"Total WAV size: {total_wav_size / (1024 ** 3):.1f} GB")

    # Convert files
    converted = 0
    failed = 0
    total_opus_size = 0

    for i, wav_path in enumerate(wav_files):
        opus_path = convert_wav_to_opus(wav_path)
        if opus_path and opus_path.exists():
            total_opus_size += opus_path.stat().st_size
            wav_path.unlink()  # Delete original WAV
            converted += 1
        else:
            failed += 1

        if (i + 1) % 500 == 0 or i == len(wav_files) - 1:
            pct = (i + 1) / len(wav_files) * 100
            print(f"  [{i + 1}/{len(wav_files)}] ({pct:.0f}%) converted={converted} failed={failed}")

    print(f"\nConversion done: {converted} converted, {failed} failed")
    print(f"WAV size: {total_wav_size / (1024 ** 3):.1f} GB")
    print(f"Opus size: {total_opus_size / (1024 ** 3):.1f} GB")
    print(
        f"Saved: {(total_wav_size - total_opus_size) / (1024 ** 3):.1f} GB ({(1 - total_opus_size / total_wav_size) * 100:.0f}%)")

    # Update database paths
    print(f"\nUpdating database paths...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM chunks WHERE file_path LIKE '%.wav'"))
        count = result.scalar()
        print(f"  {count} chunks to update")

        conn.execute(text("""
            UPDATE chunks 
            SET file_path = REPLACE(file_path, '.wav', '.opus'),
                filename = REPLACE(filename, '.wav', '.opus')
            WHERE file_path LIKE '%.wav'
        """))
        conn.commit()
        print(f"  Updated {count} paths from .wav to .opus")

    print("\nDone! Ready to upload.")


if __name__ == "__main__":
    main()