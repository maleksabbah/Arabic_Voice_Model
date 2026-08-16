"""
Production consensus pipeline.
Runs 3-model consensus on ALL chunks with original_transcription,
writes accepted transcriptions to the `transcription` column.

Models:
  1. Whisper Large V3 — from DB (original_transcription)
  2. CodeSwitching — MohamedRashad/Arabic-Whisper-CodeSwitching-Edition
  3. NVIDIA FastConformer — nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0

Usage:
  # Set HF cache to D: drive first (one-time):
  #   $env:HF_HOME = "D:\hf_cache"

  # Run all series except 5-8:
  python Consensus.py

  # Run specific series:
  python Consensus.py --series 9 10 11 12

  # Dry run (no DB writes):
  python Consensus.py --dry-run

  # Resume from where you left off (skips chunks with transcription already set):
  python Consensus.py --resume
"""
import argparse
import gc
import os
import time
import torch
import re
import Levenshtein

import sys

sys.path.insert(0, ".")

from Config.Database import get_db
from Training.Model import Episode, Chunk, Series

ARABIC_DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")


def normalize_arabic(text):
    if not text:
        return ""
    text = "".join(c for c in text if c not in ARABIC_DIACRITICS)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = text.replace('ـ', '')
    text = " ".join(text.split()).strip()
    return text


def transcribe_whisper_sequential(model_name, chunks, label="model"):
    """Transcribe chunks sequentially with Whisper pipeline. For 4GB GPU."""
    from transformers import pipeline as hf_pipeline

    print(f"\n{'=' * 60}")
    print(f"Loading: {model_name} [{label}] ({len(chunks)} chunks)")
    print(f"{'=' * 60}")

    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device="cuda",
        torch_dtype=torch.float16,
    )

    results = {}
    start = time.time()
    errors = 0

    for i, chunk in enumerate(chunks):
        try:
            out = pipe(chunk.file_path, generate_kwargs={"language": "ar", "task": "transcribe"})
            results[chunk.id] = out["text"]
        except Exception as e:
            if "3000 mel" in str(e):
                try:
                    out = pipe(chunk.file_path, generate_kwargs={"language": "ar", "task": "transcribe"},
                               return_timestamps=True)
                    results[chunk.id] = out["text"]
                except Exception as e2:
                    results[chunk.id] = ""
                    errors += 1
            else:
                results[chunk.id] = ""
                errors += 1

        if i < 3 or (i + 1) % 100 == 0 or i == len(chunks) - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
            text_preview = results.get(chunk.id, "")[:50]
            print(
                f"  [{i + 1}/{len(chunks)}] ({rate:.1f} chunks/s, ETA {eta / 60:.0f}m) ep{chunk.episode_id} {chunk.filename}: {text_preview}")

    elapsed = time.time() - start
    print(f"Completed {len(chunks)} chunks in {elapsed / 60:.1f}m ({errors} errors)")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return results


def transcribe_fastconformer_batch(model_name, chunks, label="conformer", batch_size=8):
    """Transcribe chunks with FastConformer using native batch support."""
    import nemo.collections.asr as nemo_asr

    print(f"\n{'=' * 60}")
    print(f"Loading: {model_name} [{label}] ({len(chunks)} chunks)")
    print(f"{'=' * 60}")

    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name)
    asr_model.change_decoding_strategy(decoder_type="ctc")
    asr_model.eval()
    asr_model.cuda()

    file_paths = [chunk.file_path for chunk in chunks]
    results = {}
    start = time.time()

    try:
        outputs = asr_model.transcribe(file_paths, batch_size=batch_size)
        if hasattr(outputs, 'text'):
            texts = outputs.text
        elif isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], str):
                texts = outputs
            else:
                texts = [o.text if hasattr(o, 'text') else str(o) for o in outputs]
        else:
            texts = outputs

        for i, (chunk, text) in enumerate(zip(chunks, texts)):
            results[chunk.id] = text
            if i < 3 or (i + 1) % 100 == 0 or i == len(chunks) - 1:
                print(f"  [{i + 1}/{len(chunks)}] ep{chunk.episode_id} {chunk.filename}: {text[:50]}")

    except Exception as e:
        print(f"  Batch failed: {e}")
        print(f"  Falling back to sequential (batch_size=1)")
        for i, chunk in enumerate(chunks):
            try:
                out = asr_model.transcribe([chunk.file_path])
                if hasattr(out, 'text'):
                    text = out.text[0]
                elif isinstance(out, list):
                    text = out[0] if isinstance(out[0], str) else out[0].text
                else:
                    text = str(out)
                results[chunk.id] = text
            except Exception as e2:
                results[chunk.id] = ""
            if i < 3 or (i + 1) % 100 == 0:
                text_preview = results.get(chunk.id, "")[:50]
                print(f"  [{i + 1}/{len(chunks)}] ep{chunk.episode_id} {chunk.filename}: {text_preview}")

    elapsed = time.time() - start
    print(f"Completed {len(chunks)} chunks in {elapsed / 60:.1f}m")

    del asr_model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_consensus(chunks, cs_results, fc_results, max_distance=0.4):
    """Run 3-model consensus and return list of result dicts."""
    results = []

    for chunk in chunks:
        large_v3 = normalize_arabic(chunk.original_transcription)
        cs = normalize_arabic(cs_results.get(chunk.id, ""))
        fc = normalize_arabic(fc_results.get(chunk.id, ""))

        texts = {"large_v3": large_v3, "codeswitching": cs, "conformer": fc}
        model_names = list(texts.keys())

        distances = {}
        agreements = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a, b = model_names[i], model_names[j]
                ta, tb = texts[a], texts[b]
                if not ta or not tb:
                    distances[f"{a}_vs_{b}"] = 1.0
                    continue
                dist = Levenshtein.distance(ta, tb)
                max_len = max(len(ta), len(tb))
                ratio = dist / max_len if max_len > 0 else 1.0
                distances[f"{a}_vs_{b}"] = ratio
                if ratio <= max_distance:
                    agreements.append((a, b))

        is_accepted = len(agreements) >= 1
        all_agree = len(agreements) == 3

        best_model = None
        best_score = float("inf")
        for mn in texts:
            if not texts[mn]:
                continue
            score = sum(v for k, v in distances.items() if mn in k)
            count = sum(1 for k in distances if mn in k)
            avg = score / count if count else 1.0
            if avg < best_score:
                best_score = avg
                best_model = mn

        # Use original (non-normalized) text for the best model
        if best_model == "large_v3":
            best_text = chunk.original_transcription
        elif best_model == "codeswitching":
            best_text = cs_results.get(chunk.id, "")
        else:
            best_text = fc_results.get(chunk.id, "")

        results.append({
            "chunk": chunk,
            "accepted": is_accepted,
            "all_agree": all_agree,
            "best_model": best_model,
            "best_text": best_text,
            "num_agreements": len(agreements),
            "distances": distances,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Production consensus pipeline")
    parser.add_argument("--series", type=int, nargs="+", default=None)
    parser.add_argument("--exclude", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--max-distance", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--resume", action="store_true", help="Skip chunks already transcribed")
    parser.add_argument("--batch-size", type=int, default=8, help="FastConformer batch size")
    args = parser.parse_args()

    db = next(get_db())

    # Determine series
    if args.series:
        series_ids = args.series
    else:
        all_series = db.query(Series).all()
        series_ids = [s.id for s in all_series if s.id not in args.exclude]

    # Print series info
    print("=" * 60)
    print("PRODUCTION CONSENSUS PIPELINE")
    print("=" * 60)
    for sid in series_ids:
        s = db.query(Series).filter(Series.id == sid).first()
        if s:
            count = (
                db.query(Chunk)
                .join(Episode)
                .filter(Episode.series_id == sid, Chunk.original_transcription.isnot(None))
                .count()
            )
            print(f"  Series {sid}: {s.name} — {count} chunks")

    # Get all chunks
    query = (
        db.query(Chunk)
        .join(Episode)
        .filter(
            Episode.series_id.in_(series_ids),
            Chunk.original_transcription.isnot(None),
        )
    )
    if args.resume:
        query = query.filter(Chunk.transcription.is_(None))

    all_chunks = query.all()

    if not all_chunks:
        print("\nNo chunks to process!")
        return

    print(f"\nTotal chunks to process: {len(all_chunks)}")
    print(f"Threshold: {args.max_distance}")
    print(f"Dry run: {args.dry_run}")
    print(f"Resume mode: {args.resume}")

    total_start = time.time()

    # Model 2: CodeSwitching (sequential, ~3min/100 chunks)
    cs_results = transcribe_whisper_sequential(
        "MohamedRashad/Arabic-Whisper-CodeSwitching-Edition",
        all_chunks,
        "codeswitching",
    )

    # Model 3: FastConformer (batched, fast)
    fc_results = transcribe_fastconformer_batch(
        "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0",
        all_chunks,
        "conformer",
        batch_size=args.batch_size,
    )

    # Run consensus
    print(f"\n{'=' * 60}")
    print(f"3-MODEL CONSENSUS (threshold={args.max_distance})")
    print(f"{'=' * 60}\n")

    consensus_results = run_consensus(all_chunks, cs_results, fc_results, args.max_distance)

    # Stats
    accepted = [r for r in consensus_results if r["accepted"]]
    rejected = [r for r in consensus_results if not r["accepted"]]
    all_agree = [r for r in consensus_results if r["all_agree"]]

    # Per-series breakdown
    series_stats = {}
    for r in consensus_results:
        sid = r["chunk"].episode.series_id if hasattr(r["chunk"], 'episode') else "?"
        if sid not in series_stats:
            series_stats[sid] = {"total": 0, "accepted": 0, "all_agree": 0}
        series_stats[sid]["total"] += 1
        if r["accepted"]:
            series_stats[sid]["accepted"] += 1
        if r["all_agree"]:
            series_stats[sid]["all_agree"] += 1

    for sid, stats in sorted(series_stats.items()):
        s = db.query(Series).filter(Series.id == sid).first()
        name = s.name if s else f"Series {sid}"
        pct = 100 * stats["accepted"] / stats["total"] if stats["total"] else 0
        print(f"  {name}: {stats['accepted']}/{stats['total']} accepted ({pct:.0f}%)")

    # Write to DB
    if not args.dry_run:
        for r in accepted:
            r["chunk"].transcription = r["best_text"]
        for r in rejected:
            r["chunk"].transcription = ""
        db.commit()
        print(f"\nWrote {len(accepted)} accepted, {len(rejected)} rejected (set to empty).")
    else:
        print(f"\n[DRY RUN] Would write {len(accepted)} transcriptions to DB.")

    # Summary
    total_elapsed = time.time() - total_start
    total = len(consensus_results)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total chunks:     {total}")
    print(f"Accepted:         {len(accepted)} ({100 * len(accepted) / total:.0f}%)")
    print(f"All 3 agree:      {len(all_agree)} ({100 * len(all_agree) / total:.0f}%)")
    print(f"Rejected:         {len(rejected)} ({100 * len(rejected) / total:.0f}%)")
    print(f"Total time:       {total_elapsed / 60:.1f} minutes")
    print(f"Rate:             {total / (total_elapsed / 60):.0f} chunks/minute")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()