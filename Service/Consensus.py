"""
Production consensus pipeline (RunPod version — raw sqlite3, no project imports).
Runs 3-model consensus, writes accepted transcriptions to the `transcription` column.

Models:
  1. Whisper Large V3 Turbo — whisper_text column
  2. CodeSwitching — MohamedRashad/Arabic-Whisper-CodeSwitching-Edition → codeswitching_text column
  3. NVIDIA FastConformer — nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0 → conformer_text column

Usage:
  # Run everything (fills missing columns, then consensus):
  python Consensus.py --db /workspace/asr.db

  # Run individual phases:
  python Consensus.py --db /workspace/asr.db --phase whisper
  python Consensus.py --db /workspace/asr.db --phase codeswitching
  python Consensus.py --db /workspace/asr.db --phase conformer
  python Consensus.py --db /workspace/asr.db --phase consensus

  # Run specific series:
  python Consensus.py --db /workspace/asr.db --series 1 2 3

  # Dry run (no DB writes on consensus):
  python Consensus.py --db /workspace/asr.db --phase consensus --dry-run

  # Resume (skip chunks already transcribed):
  python Consensus.py --db /workspace/asr.db --resume
"""
import argparse
import gc
import os
import time
import torch
import re
import Levenshtein

import sqlite3

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

    print(f"\n{'='*60}")
    print(f"Loading: {model_name} [{label}] ({len(chunks)} chunks)")
    print(f"{'='*60}")

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
        cid, file_path, episode_id, filename = chunk["id"], chunk["file_path"], chunk["episode_id"], chunk["filename"]
        try:
            out = pipe(file_path, generate_kwargs={"language": "ar", "task": "transcribe"})
            results[cid] = out["text"]
        except Exception as e:
            if "3000 mel" in str(e):
                try:
                    out = pipe(file_path, generate_kwargs={"language": "ar", "task": "transcribe"}, return_timestamps=True)
                    results[cid] = out["text"]
                except Exception as e2:
                    results[cid] = ""
                    errors += 1
            else:
                results[cid] = ""
                errors += 1

        if i < 3 or (i + 1) % 100 == 0 or i == len(chunks) - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
            text_preview = results.get(cid, "")[:50]
            print(f"  [{i+1}/{len(chunks)}] ({rate:.1f} chunks/s, ETA {eta/60:.0f}m) ep{episode_id} {filename}: {text_preview}")

    elapsed = time.time() - start
    print(f"Completed {len(chunks)} chunks in {elapsed/60:.1f}m ({errors} errors)")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return results


def transcribe_fastconformer_batch(model_name, chunks, label="conformer", batch_size=8):
    """Transcribe chunks with FastConformer using native batch support."""
    import nemo.collections.asr as nemo_asr

    print(f"\n{'='*60}")
    print(f"Loading: {model_name} [{label}] ({len(chunks)} chunks)")
    print(f"{'='*60}")

    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name)
    asr_model.change_decoding_strategy(decoder_type="ctc")
    asr_model.eval()
    asr_model.cuda()

    file_paths = [chunk["file_path"] for chunk in chunks]
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
            results[chunk["id"]] = text
            if i < 3 or (i + 1) % 100 == 0 or i == len(chunks) - 1:
                print(f"  [{i+1}/{len(chunks)}] ep{chunk['episode_id']} {chunk['filename']}: {text[:50]}")

    except Exception as e:
        print(f"  Batch failed: {e}")
        print(f"  Falling back to sequential (batch_size=1)")
        for i, chunk in enumerate(chunks):
            try:
                out = asr_model.transcribe([chunk["file_path"]])
                if hasattr(out, 'text'):
                    text = out.text[0]
                elif isinstance(out, list):
                    text = out[0] if isinstance(out[0], str) else out[0].text
                else:
                    text = str(out)
                results[chunk["id"]] = text
            except Exception as e2:
                results[chunk["id"]] = ""
            if i < 3 or (i + 1) % 100 == 0:
                text_preview = results.get(chunk["id"], "")[:50]
                print(f"  [{i+1}/{len(chunks)}] ep{chunk['episode_id']} {chunk['filename']}: {text_preview}")

    elapsed = time.time() - start
    print(f"Completed {len(chunks)} chunks in {elapsed/60:.1f}m")

    del asr_model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_consensus(chunks, whisper_results, cs_results, fc_results, max_distance=0.4):
    """Run 3-model consensus and return list of result dicts."""
    results = []

    for chunk in chunks:
        cid = chunk["id"]
        whisper_raw = whisper_results.get(cid, "")
        cs_raw = cs_results.get(cid, "")
        fc_raw = fc_results.get(cid, "")

        whisper_norm = normalize_arabic(whisper_raw)
        cs_norm = normalize_arabic(cs_raw)
        fc_norm = normalize_arabic(fc_raw)

        texts = {"whisper": whisper_norm, "codeswitching": cs_norm, "conformer": fc_norm}
        raw_texts = {"whisper": whisper_raw, "codeswitching": cs_raw, "conformer": fc_raw}
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

        best_text = raw_texts.get(best_model, "") if best_model else ""

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
    parser.add_argument("--db", type=str, default="/workspace/asr.db")
    parser.add_argument("--series", type=int, nargs="+", default=None)
    parser.add_argument("--exclude", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--max-distance", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--resume", action="store_true", help="Skip chunks already transcribed")
    parser.add_argument("--batch-size", type=int, default=8, help="FastConformer batch size")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "whisper", "codeswitching", "conformer", "consensus"],
                        help="Run a specific phase only")
    parser.add_argument("--save-every", type=int, default=500, help="Save to DB every N chunks per model")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    # Ensure columns exist
    for col in ["whisper_text", "conformer_text", "codeswitching_text"]:
        try:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} TEXT")
        except:
            pass
    conn.commit()

    # Determine series
    if args.series:
        series_ids = args.series
    else:
        rows = conn.execute("SELECT id FROM series").fetchall()
        series_ids = [r[0] for r in rows if r[0] not in args.exclude]

    # Print series info
    print("=" * 60)
    print("PRODUCTION CONSENSUS PIPELINE (RunPod)")
    print("=" * 60)
    ph = ",".join(str(s) for s in series_ids)
    for sid in series_ids:
        row = conn.execute("SELECT name FROM series WHERE id=?", (sid,)).fetchone()
        name = row[0] if row else f"Series {sid}"
        count = conn.execute(f"""
            SELECT COUNT(*) FROM chunks c JOIN episodes e ON c.episode_id=e.id
            WHERE e.series_id=?
        """, (sid,)).fetchone()[0]
        print(f"  Series {sid}: {name} — {count} chunks")

    # Get all chunks
    ph = ",".join(str(s) for s in series_ids)
    all_chunks = conn.execute(f"""
        SELECT c.id, c.file_path, c.filename, c.episode_id, e.series_id,
               c.whisper_text, c.conformer_text, c.codeswitching_text, c.transcription
        FROM chunks c
        JOIN episodes e ON c.episode_id=e.id
        WHERE e.series_id IN ({ph})
    """).fetchall()

    all_chunks = [
        {"id": r[0], "file_path": r[1], "filename": r[2], "episode_id": r[3],
         "series_id": r[4], "whisper_text": r[5], "conformer_text": r[6],
         "codeswitching_text": r[7], "transcription": r[8]}
        for r in all_chunks
    ]

    if not all_chunks:
        print("\nNo chunks found!")
        conn.close()
        return

    print(f"\nTotal chunks in DB: {len(all_chunks)}")
    print(f"Threshold: {args.max_distance}")
    print(f"Phase: {args.phase}")

    # Show current column status
    w_done = sum(1 for c in all_chunks if c["whisper_text"] and c["whisper_text"].strip())
    cf_done = sum(1 for c in all_chunks if c["conformer_text"] and c["conformer_text"].strip())
    cs_done = sum(1 for c in all_chunks if c["codeswitching_text"] and c["codeswitching_text"].strip())
    print(f"Whisper:       {w_done}/{len(all_chunks)}")
    print(f"Conformer:     {cf_done}/{len(all_chunks)}")
    print(f"CodeSwitching: {cs_done}/{len(all_chunks)}")

    total_start = time.time()

    # ── PHASE: WHISPER ──
    if args.phase in ("all", "whisper"):
        need_whisper = [c for c in all_chunks if not c["whisper_text"] or not c["whisper_text"].strip()]
        if need_whisper:
            print(f"\n[WHISPER] {len(need_whisper)} chunks need transcription")
            w_results = transcribe_whisper_sequential(
                "openai/whisper-large-v3-turbo", need_whisper, "whisper-turbo",
            )
            # Save to DB in batches
            batch = []
            for cid, text in w_results.items():
                batch.append((text, cid))
                if len(batch) >= args.save_every:
                    conn.executemany("UPDATE chunks SET whisper_text=? WHERE id=?", batch)
                    conn.commit()
                    print(f"  [SAVE] {len(batch)} whisper results committed")
                    batch = []
            if batch:
                conn.executemany("UPDATE chunks SET whisper_text=? WHERE id=?", batch)
                conn.commit()
                print(f"  [SAVE] {len(batch)} whisper results committed (final)")
        else:
            print(f"\n[WHISPER] All {len(all_chunks)} chunks already done, skipping")

    # ── PHASE: CODESWITCHING ──
    if args.phase in ("all", "codeswitching"):
        need_cs = [c for c in all_chunks if not c["codeswitching_text"] or not c["codeswitching_text"].strip()]
        if need_cs:
            print(f"\n[CODESWITCHING] {len(need_cs)} chunks need transcription")
            cs_results = transcribe_whisper_sequential(
                "MohamedRashad/Arabic-Whisper-CodeSwitching-Edition", need_cs, "codeswitching",
            )
            batch = []
            for cid, text in cs_results.items():
                batch.append((text, cid))
                if len(batch) >= args.save_every:
                    conn.executemany("UPDATE chunks SET codeswitching_text=? WHERE id=?", batch)
                    conn.commit()
                    print(f"  [SAVE] {len(batch)} codeswitching results committed")
                    batch = []
            if batch:
                conn.executemany("UPDATE chunks SET codeswitching_text=? WHERE id=?", batch)
                conn.commit()
                print(f"  [SAVE] {len(batch)} codeswitching results committed (final)")
        else:
            print(f"\n[CODESWITCHING] All {len(all_chunks)} chunks already done, skipping")

    # ── PHASE: CONFORMER ──
    if args.phase in ("all", "conformer"):
        need_cf = [c for c in all_chunks if not c["conformer_text"] or not c["conformer_text"].strip()]
        if need_cf:
            print(f"\n[CONFORMER] {len(need_cf)} chunks need transcription")
            fc_results = transcribe_fastconformer_batch(
                "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0", need_cf, "conformer",
                batch_size=args.batch_size,
            )
            batch = []
            for cid, text in fc_results.items():
                batch.append((text, cid))
                if len(batch) >= args.save_every:
                    conn.executemany("UPDATE chunks SET conformer_text=? WHERE id=?", batch)
                    conn.commit()
                    print(f"  [SAVE] {len(batch)} conformer results committed")
                    batch = []
            if batch:
                conn.executemany("UPDATE chunks SET conformer_text=? WHERE id=?", batch)
                conn.commit()
                print(f"  [SAVE] {len(batch)} conformer results committed (final)")
        else:
            print(f"\n[CONFORMER] All {len(all_chunks)} chunks already done, skipping")

    # ── PHASE: CONSENSUS ──
    if args.phase in ("all", "consensus"):
        # Re-read from DB to get latest values
        all_chunks = conn.execute(f"""
            SELECT c.id, c.file_path, c.filename, c.episode_id, e.series_id,
                   c.whisper_text, c.conformer_text, c.codeswitching_text
            FROM chunks c
            JOIN episodes e ON c.episode_id=e.id
            WHERE e.series_id IN ({ph})
        """).fetchall()
        all_chunks = [
            {"id": r[0], "file_path": r[1], "filename": r[2], "episode_id": r[3],
             "series_id": r[4], "whisper_text": r[5], "conformer_text": r[6],
             "codeswitching_text": r[7]}
            for r in all_chunks
        ]

        # Check readiness
        ready = [c for c in all_chunks
                 if c["whisper_text"] and c["whisper_text"].strip()
                 and c["conformer_text"] and c["conformer_text"].strip()
                 and c["codeswitching_text"] and c["codeswitching_text"].strip()]

        not_ready = len(all_chunks) - len(ready)
        if not_ready > 0:
            print(f"\n[CONSENSUS] WARNING: {not_ready} chunks missing one or more model outputs, skipping those")

        if not ready:
            print("[CONSENSUS] No chunks ready for consensus!")
            conn.close()
            return

        if args.resume:
            ready = [c for c in ready if not c.get("transcription") or not c.get("transcription", "").strip()]
            print(f"[CONSENSUS] Resume mode: {len(ready)} chunks without transcription")

        # Build result dicts from DB columns
        whisper_results = {c["id"]: c["whisper_text"] for c in ready}
        cs_results = {c["id"]: c["codeswitching_text"] for c in ready}
        fc_results = {c["id"]: c["conformer_text"] for c in ready}

        print(f"\n{'='*60}")
        print(f"3-MODEL CONSENSUS (threshold={args.max_distance})")
        print(f"Chunks: {len(ready)}")
        print(f"{'='*60}\n")

        consensus_results = run_consensus(ready, whisper_results, cs_results, fc_results, args.max_distance)

        # Per-series breakdown
        series_names = {}
        for row in conn.execute("SELECT id, name FROM series").fetchall():
            series_names[row[0]] = row[1]

        series_stats = {}
        for r in consensus_results:
            sid = r["chunk"]["series_id"]
            if sid not in series_stats:
                series_stats[sid] = {"total": 0, "accepted": 0, "all_agree": 0}
            series_stats[sid]["total"] += 1
            if r["accepted"]:
                series_stats[sid]["accepted"] += 1
            if r["all_agree"]:
                series_stats[sid]["all_agree"] += 1

        print(f"\n{'='*60}")
        print(f"PER-SERIES BREAKDOWN")
        print(f"{'='*60}")
        for sid in sorted(series_stats.keys()):
            stats = series_stats[sid]
            name = series_names.get(sid, f"Series {sid}")
            pct = 100 * stats["accepted"] / stats["total"] if stats["total"] > 0 else 0
            agree_pct = 100 * stats["all_agree"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {name}: {stats['accepted']}/{stats['total']} accepted ({pct:.0f}%), all-3-agree: {stats['all_agree']} ({agree_pct:.0f}%)")

        # Write to DB
        accepted = [r for r in consensus_results if r["accepted"]]
        rejected = [r for r in consensus_results if not r["accepted"]]
        all_agree_list = [r for r in consensus_results if r["all_agree"]]

        if not args.dry_run:
            print(f"\nWriting {len(accepted)} accepted transcriptions to DB...")
            batch = [(r["best_text"], r["chunk"]["id"]) for r in accepted]
            conn.executemany("UPDATE chunks SET transcription=? WHERE id=?", batch)
            batch_rej = [("", r["chunk"]["id"]) for r in rejected]
            conn.executemany("UPDATE chunks SET transcription=? WHERE id=?", batch_rej)
            conn.commit()
            print(f"Written {len(accepted)} transcriptions. {len(rejected)} rejected (set to empty).")
        else:
            print(f"\n[DRY RUN] Would write {len(accepted)} transcriptions to DB.")

        # Summary
        total_elapsed = time.time() - total_start
        total = len(consensus_results)

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total chunks:     {total}")
        print(f"Accepted:         {len(accepted)} ({100*len(accepted)/total:.0f}%)")
        print(f"All 3 agree:      {len(all_agree_list)} ({100*len(all_agree_list)/total:.0f}%)")
        print(f"Rejected:         {len(rejected)} ({100*len(rejected)/total:.0f}%)")
        print(f"Total time:       {total_elapsed/60:.1f} minutes")
        print(f"Rate:             {total/(total_elapsed/60):.0f} chunks/minute")
        print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()