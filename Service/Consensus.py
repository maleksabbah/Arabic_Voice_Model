"""
Consensus v5 — Reliable saves. Each model saves independently after every batch.
Conformer processes in small batches (not one giant call).
Three phases: whisper, conformer, consensus. Run individually or all at once.

Usage:
  python Service/Consensus.py --db /workspace/asr.db --series 1 2 3 --phase all
  python Service/Consensus.py --db /workspace/asr.db --series 1 2 3 --phase whisper
  python Service/Consensus.py --db /workspace/asr.db --series 1 2 3 --phase conformer
  python Service/Consensus.py --db /workspace/asr.db --series 1 2 3 --phase consensus
"""
import argparse, gc, os, re, time, sqlite3, torch, Levenshtein

DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")

def norm(t):
    if not t: return ""
    t = "".join(c for c in t if c not in DIACRITICS)
    t = re.sub(r'[أإآٱ]', 'ا', t)
    t = t.replace('ـ', '')
    return " ".join(t.split()).strip()

def ensure_columns(db_path):
    conn = sqlite3.connect(db_path)
    for col in ["whisper_text", "conformer_text"]:
        try:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} TEXT")
        except:
            pass
    conn.commit()
    conn.close()
    print("[DB] Columns ensured: whisper_text, conformer_text")

def get_chunks(db_path, series_ids, phase="all"):
    conn = sqlite3.connect(db_path)
    ph = ",".join(str(s) for s in series_ids)
    q = f"SELECT c.id, c.file_path, c.filename, e.series_id FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph})"
    if phase == "whisper":
        q += " AND (c.whisper_text IS NULL)"
    elif phase == "conformer":
        q += " AND (c.conformer_text IS NULL)"
    elif phase == "consensus":
        q += " AND c.whisper_text IS NOT NULL AND c.conformer_text IS NOT NULL AND (c.transcription IS NULL OR c.transcription = '')"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [{"id":r[0], "file_path":r[1], "filename":r[2], "series_id":r[3]} for r in rows]

def run_whisper(chunks, db_path, batch_size=32):
    from transformers import pipeline as hf_pipeline
    print(f"[WHISPER] Loading openai/whisper-large-v3-turbo...")
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch.float16,
        device="cuda",
    )
    print(f"[WHISPER] Loaded. {len(chunks)} chunks, batch_size={batch_size}")

    t0 = time.time()
    errs = 0
    done = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch = chunks[batch_start:batch_end]
        paths = [c["file_path"] for c in batch]

        conn = sqlite3.connect(db_path)
        try:
            results = pipe(paths, batch_size=len(paths), generate_kwargs={"language": "ar", "task": "transcribe"})
            for c, r in zip(batch, results):
                text = r.get("text", "").strip() if isinstance(r, dict) else ""
                conn.execute("UPDATE chunks SET whisper_text=? WHERE id=?", (text, c["id"]))
                done += 1
        except Exception as e:
            print(f"  [WHISPER] Batch fail at {batch_start}: {e}")
            for c in batch:
                try:
                    r = pipe(c["file_path"], generate_kwargs={"language": "ar", "task": "transcribe"})
                    text = r.get("text", "").strip() if isinstance(r, dict) else ""
                    conn.execute("UPDATE chunks SET whisper_text=? WHERE id=?", (text, c["id"]))
                except:
                    conn.execute("UPDATE chunks SET whisper_text='' WHERE id=?", (c["id"],))
                    errs += 1
                done += 1

        conn.commit()
        conn.close()

        el = time.time() - t0
        r = done / el if el > 0 else 0
        eta = (len(chunks) - done) / r if r > 0 else 0
        print(f"  [WHISPER {done}/{len(chunks)}] {r:.1f}/s ETA {eta/60:.0f}m err={errs}")

    el = time.time() - t0
    print(f"[WHISPER] Done in {el/60:.1f}m. {done} chunks, {errs} errors")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

def run_conformer(chunks, db_path, batch_size=64):
    import nemo.collections.asr as nemo_asr
    print(f"[CONFORMER] Loading...")
    m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained("nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0")
    m.change_decoding_strategy(decoder_type="ctc")
    m.eval()
    m.cuda()
    print(f"[CONFORMER] Loaded. {len(chunks)} chunks, batch_size={batch_size}")

    t0 = time.time()
    errs = 0
    done = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch = chunks[batch_start:batch_end]
        paths = [c["file_path"] for c in batch]

        conn = sqlite3.connect(db_path)
        try:
            out = m.transcribe(paths, batch_size=len(paths))
            if hasattr(out, 'text'):
                texts = out.text
            elif isinstance(out, list) and len(out) > 0:
                texts = out if isinstance(out[0], str) else [o.text if hasattr(o, 'text') else str(o) for o in out]
            else:
                texts = out

            for c, t in zip(batch, texts):
                conn.execute("UPDATE chunks SET conformer_text=? WHERE id=?", (t, c["id"]))
                done += 1
        except Exception as e:
            print(f"  [CONFORMER] Batch fail at {batch_start}: {e}")
            for c in batch:
                try:
                    o = m.transcribe([c["file_path"]])
                    t = o.text[0] if hasattr(o, 'text') else (o[0] if isinstance(o[0], str) else str(o[0]))
                    conn.execute("UPDATE chunks SET conformer_text=? WHERE id=?", (t, c["id"]))
                except:
                    conn.execute("UPDATE chunks SET conformer_text='' WHERE id=?", (c["id"],))
                    errs += 1
                done += 1

        conn.commit()
        conn.close()

        el = time.time() - t0
        r = done / el if el > 0 else 0
        eta = (len(chunks) - done) / r if r > 0 else 0
        print(f"  [CONFORMER {done}/{len(chunks)}] {r:.1f}/s ETA {eta/60:.0f}m err={errs}")

    el = time.time() - t0
    print(f"[CONFORMER] Done in {el/60:.1f}m. {done} chunks, {errs} errors")
    del m
    gc.collect()
    torch.cuda.empty_cache()

def run_consensus(db_path, series_ids, max_distance=0.4):
    conn = sqlite3.connect(db_path)
    ph = ",".join(str(s) for s in series_ids)
    rows = conn.execute(f"""
        SELECT c.id, c.whisper_text, c.conformer_text
        FROM chunks c JOIN episodes e ON c.episode_id=e.id
        WHERE e.series_id IN ({ph})
        AND c.whisper_text IS NOT NULL AND c.conformer_text IS NOT NULL
        AND (c.transcription IS NULL OR c.transcription = '')
    """).fetchall()

    if not rows:
        print("[CONSENSUS] No chunks ready")
        conn.close()
        return

    print(f"[CONSENSUS] Processing {len(rows)} chunks, threshold={max_distance}")
    t0 = time.time()
    acc = rej = 0

    for i, r in enumerate(rows):
        cid, wt_raw, ct_raw = r
        wt = norm(wt_raw or "")
        ct = norm(ct_raw or "")

        if not wt or not ct:
            conn.execute("UPDATE chunks SET transcription='', was_filtered=1, filter_reason='empty' WHERE id=?", (cid,))
            rej += 1
        else:
            d = Levenshtein.distance(wt, ct)
            ml = max(len(wt), len(ct))
            dist = d / ml if ml > 0 else 1.0
            if dist <= max_distance:
                conn.execute("UPDATE chunks SET transcription=? WHERE id=?", (wt_raw, cid))
                acc += 1
            else:
                conn.execute("UPDATE chunks SET transcription='', was_filtered=1, filter_reason=? WHERE id=?",
                            (f"dist>{max_distance}:{dist:.3f}", cid))
                rej += 1

        if (i + 1) % 5000 == 0:
            conn.commit()
            pct = 100 * acc / (acc + rej) if (acc + rej) > 0 else 0
            print(f"  [CONSENSUS {i+1}/{len(rows)}] acc={acc} rej={rej} ({pct:.0f}%)")

    conn.commit()
    conn.close()
    tot = acc + rej
    pct = 100 * acc / tot if tot > 0 else 0
    print(f"[CONSENSUS] Done in {(time.time()-t0)/60:.1f}m | Accepted: {acc} ({pct:.0f}%) | Rejected: {rej}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--series", type=int, nargs="+", required=True)
    p.add_argument("--max-distance", type=float, default=0.4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--whisper-batch", type=int, default=32)
    p.add_argument("--phase", choices=["all", "whisper", "conformer", "consensus"], default="all")
    a = p.parse_args()

    print("=" * 60)
    print("CONSENSUS v5 — RELIABLE SAVES")
    print("=" * 60)
    print(f"DB: {a.db}")
    print(f"Series: {a.series}")
    print(f"Phase: {a.phase}")
    print(f"Threshold: {a.max_distance}")
    print(f"Conformer batch: {a.batch_size}")
    print(f"Whisper batch: {a.whisper_batch}")
    print()

    ensure_columns(a.db)

    if a.phase in ("all", "whisper"):
        chunks = get_chunks(a.db, a.series, phase="whisper")
        chunks = [c for c in chunks if os.path.exists(c["file_path"])]
        if chunks:
            print(f"\n=== PHASE 1: WHISPER ({len(chunks)} chunks) ===")
            run_whisper(chunks, a.db, a.whisper_batch)
        else:
            print("Whisper: all done")

    if a.phase in ("all", "conformer"):
        chunks = get_chunks(a.db, a.series, phase="conformer")
        chunks = [c for c in chunks if os.path.exists(c["file_path"])]
        if chunks:
            print(f"\n=== PHASE 2: CONFORMER ({len(chunks)} chunks) ===")
            run_conformer(chunks, a.db, a.batch_size)
        else:
            print("Conformer: all done")

    if a.phase in ("all", "consensus"):
        print(f"\n=== PHASE 3: CONSENSUS ===")
        run_consensus(a.db, a.series, a.max_distance)

    # Summary
    conn = sqlite3.connect(a.db)
    ph = ",".join(str(s) for s in a.series)
    total = conn.execute(f"SELECT COUNT(*) FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph})").fetchone()[0]
    w = conn.execute(f"SELECT COUNT(*) FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph}) AND whisper_text IS NOT NULL").fetchone()[0]
    c = conn.execute(f"SELECT COUNT(*) FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph}) AND conformer_text IS NOT NULL").fetchone()[0]
    labeled = conn.execute(f"SELECT COUNT(*) FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph}) AND transcription IS NOT NULL AND transcription != ''").fetchone()[0]
    conn.close()
    print(f"\n{'='*60}")
    print(f"Total: {total} | Whisper: {w} | Conformer: {c} | Labeled: {labeled}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()