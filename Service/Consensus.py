"""
Parallel two-model consensus pseudo-labeling pipeline.
Runs Whisper Large V3 Turbo + NVIDIA FastConformer in parallel using threading,
processes consensus per-chunk and writes to DB immediately.

Usage:
  python consensus.py --db /workspace/asr.db --series 1 2 3 4 5
  python consensus.py --db /workspace/asr.db --series 1 --dry-run
  python consensus.py --db /workspace/asr.db --series 1 --resume
"""
import argparse
import gc
import os
import re
import time
import threading
import queue
import sqlite3

import torch
import Levenshtein

ARABIC_DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")


def normalize_arabic(t):
    if not t:
        return ""
    t = "".join(c for c in t if c not in ARABIC_DIACRITICS)
    t = re.sub(r'[أإآٱ]', 'ا', t)
    t = t.replace('ـ', '')
    return " ".join(t.split()).strip()


def get_chunks(db_path, series_ids, resume=False):
    conn = sqlite3.connect(db_path)
    ph = ",".join(str(s) for s in series_ids)
    q = f"""SELECT c.id, c.file_path, c.filename, e.series_id, e.name
            FROM chunks c JOIN episodes e ON c.episode_id = e.id
            WHERE e.series_id IN ({ph})"""
    if resume:
        q += " AND (c.transcription IS NULL OR c.transcription = '')"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [{"id": r[0], "file_path": r[1], "filename": r[2],
             "series_id": r[3], "episode_name": r[4]} for r in rows]


def whisper_worker(chunks, result_dict, model_size="large-v3-turbo", done_event=None):
    """Run Whisper Turbo inference. Puts results in result_dict[chunk_id] = text."""
    from faster_whisper import WhisperModel

    print(f"[WHISPER] Loading {model_size} (int8_float16)...")
    model = WhisperModel(model_size, device="cuda:0", compute_type="int8_float16")
    print(f"[WHISPER] Loaded. Processing {len(chunks)} chunks...")

    start = time.time()
    errors = 0

    for i, c in enumerate(chunks):
        try:
            segs, _ = model.transcribe(
                c["file_path"],
                language="ar",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            result_dict[c["id"]] = " ".join([s.text for s in segs]).strip()
        except Exception as e:
            result_dict[c["id"]] = ""
            errors += 1

        if (i + 1) % 1000 == 0 or i == len(chunks) - 1:
            el = time.time() - start
            rate = (i + 1) / el
            eta = (len(chunks) - i - 1) / rate
            print(f"  [WHISPER {i+1}/{len(chunks)}] {rate:.1f}/s ETA {eta/60:.0f}m err={errors}")

    print(f"[WHISPER] Done in {(time.time()-start)/60:.1f}m ({errors} errors)")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if done_event:
        done_event.set()


def conformer_worker(chunks, result_dict, batch_size=64, done_event=None):
    """Run FastConformer inference. Puts results in result_dict[chunk_id] = text."""
    import nemo.collections.asr as nemo_asr

    print(f"[CONFORMER] Loading model...")
    m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
        "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0"
    )
    m.change_decoding_strategy(decoder_type="ctc")
    m.eval()
    m.cuda()
    print(f"[CONFORMER] Loaded. Processing {len(chunks)} chunks in batches of {batch_size}...")

    start = time.time()
    errors = 0

    for bs in range(0, len(chunks), batch_size):
        be = min(bs + batch_size, len(chunks))
        batch_paths = [chunks[i]["file_path"] for i in range(bs, be)]
        batch_chunks = chunks[bs:be]

        try:
            out = m.transcribe(batch_paths, batch_size=batch_size)
            if hasattr(out, 'text'):
                texts = out.text
            elif isinstance(out, list) and len(out) > 0:
                if isinstance(out[0], str):
                    texts = out
                else:
                    texts = [o.text if hasattr(o, 'text') else str(o) for o in out]
            else:
                texts = out
            for c, t in zip(batch_chunks, texts):
                result_dict[c["id"]] = t
        except Exception as e:
            errors += 1
            print(f"  [CONFORMER] Batch fail at {bs}: {e}")
            for c in batch_chunks:
                try:
                    o = m.transcribe([c["file_path"]])
                    t = o.text[0] if hasattr(o, 'text') else (o[0] if isinstance(o[0], str) else str(o[0]))
                    result_dict[c["id"]] = t
                except:
                    result_dict[c["id"]] = ""

        if be % 2000 == 0 or be == len(chunks):
            el = time.time() - start
            rate = be / el if el > 0 else 0
            eta = (len(chunks) - be) / rate if rate > 0 else 0
            print(f"  [CONFORMER {be}/{len(chunks)}] {rate:.1f}/s ETA {eta/60:.0f}m err={errors}")

    print(f"[CONFORMER] Done in {(time.time()-start)/60:.1f}m ({errors} errors)")
    del m
    gc.collect()
    torch.cuda.empty_cache()
    if done_event:
        done_event.set()


def consensus_writer(chunks, whisper_results, conformer_results, db_path,
                     max_distance=0.4, dry_run=False):
    """
    Wait for both models to finish each chunk, run consensus, write to DB immediately.
    Polls results dicts until both have an entry for each chunk.
    """
    conn = None if dry_run else sqlite3.connect(db_path)

    accepted = 0
    rejected = 0
    total = len(chunks)
    start = time.time()
    last_print = 0

    for i, c in enumerate(chunks):
        cid = c["id"]

        # Wait for both models to have this chunk's result
        while cid not in whisper_results or cid not in conformer_results:
            time.sleep(0.01)

        # Run consensus
        wt = normalize_arabic(whisper_results[cid])
        ct = normalize_arabic(conformer_results[cid])

        if not wt or not ct:
            is_accepted = False
            best_text = ""
            dist = 1.0
        else:
            d = Levenshtein.distance(wt, ct)
            ml = max(len(wt), len(ct))
            dist = d / ml if ml > 0 else 1.0
            is_accepted = dist <= max_distance
            best_text = whisper_results[cid] if is_accepted else ""

        # Write immediately to DB
        if not dry_run:
            if is_accepted:
                conn.execute(
                    "UPDATE chunks SET transcription = ? WHERE id = ?",
                    (best_text, cid)
                )
                accepted += 1
            else:
                conn.execute(
                    "UPDATE chunks SET transcription = '', was_filtered = 1, filter_reason = ? WHERE id = ?",
                    (f"dist>{max_distance}", cid)
                )
                rejected += 1

            # Commit every 500 chunks
            if (i + 1) % 500 == 0:
                conn.commit()
        else:
            if is_accepted:
                accepted += 1
            else:
                rejected += 1

        # Free memory — remove processed results
        del whisper_results[cid]
        del conformer_results[cid]

        # Print progress
        if (i + 1) % 1000 == 0 or i == total - 1:
            el = time.time() - start
            rate = (i + 1) / el
            eta = (total - i - 1) / rate if rate > 0 else 0
            pct_acc = 100 * accepted / (i + 1)
            print(f"  [CONSENSUS {i+1}/{total}] acc={accepted} rej={rejected} ({pct_acc:.0f}%) {rate:.1f}/s ETA {eta/60:.0f}m")

    # Final commit
    if not dry_run and conn:
        conn.commit()
        conn.close()

    return accepted, rejected


def main():
    p = argparse.ArgumentParser(description="Parallel two-model consensus pipeline")
    p.add_argument("--db", required=True)
    p.add_argument("--series", type=int, nargs="+", required=True)
    p.add_argument("--max-distance", type=float, default=0.4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--whisper-model", default="large-v3-turbo")
    a = p.parse_args()

    print("=" * 60)
    print("PARALLEL TWO-MODEL CONSENSUS PIPELINE")
    print("=" * 60)
    print(f"DB: {a.db}")
    print(f"Models: Whisper {a.whisper_model} (int8_float16) + FastConformer")
    print(f"Series: {a.series}")
    print(f"Threshold: {a.max_distance}")
    print(f"Batch size (conformer): {a.batch_size}")
    print(f"Dry run: {a.dry_run}")
    print(f"Resume: {a.resume}")
    print()

    # Get chunks
    chunks = get_chunks(a.db, a.series, a.resume)
    if not chunks:
        print("No chunks to process!")
        return

    # Check files exist
    missing = [c for c in chunks if not os.path.exists(c["file_path"])]
    if missing:
        print(f"WARNING: {len(missing)} missing files! Example: {missing[0]['file_path']}")
        chunks = [c for c in chunks if os.path.exists(c["file_path"])]

    # Per-series stats
    sc = {}
    for c in chunks:
        sc[c["series_id"]] = sc.get(c["series_id"], 0) + 1
    for sid, cnt in sorted(sc.items()):
        print(f"  Series {sid}: {cnt}")
    print(f"  Total: {len(chunks)}")
    print()

    # Shared result dicts (thread-safe for single-writer-per-key)
    whisper_results = {}
    conformer_results = {}

    t0 = time.time()

    # Launch both models in parallel threads
    print("Launching both models in parallel...")
    w_thread = threading.Thread(
        target=whisper_worker,
        args=(chunks, whisper_results, a.whisper_model)
    )
    c_thread = threading.Thread(
        target=conformer_worker,
        args=(chunks, conformer_results, a.batch_size)
    )

    w_thread.start()
    c_thread.start()

    # Run consensus writer in main thread — processes chunks as results come in
    accepted, rejected = consensus_writer(
        chunks, whisper_results, conformer_results,
        a.db, a.max_distance, a.dry_run
    )

    # Wait for model threads to fully finish
    w_thread.join()
    c_thread.join()

    # Summary
    total = len(chunks)
    el = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total chunks:  {total}")
    print(f"Accepted:      {accepted} ({100*accepted/total:.0f}%)")
    print(f"Rejected:      {rejected} ({100*rejected/total:.0f}%)")
    print(f"Total time:    {el/60:.1f} minutes ({el/3600:.1f} hours)")
    print(f"Rate:          {total/el:.1f} chunks/second")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()