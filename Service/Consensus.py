"""
Consensus v4 — Batched Whisper + NeMo Conformer, intermediate saves per model.
Three phases: 1) Whisper batched inference → save to DB, 2) Conformer batched inference → save to DB, 3) Consensus pass → final labels.
Each phase saves independently so no work is lost.

Usage:
  python Service/Consensus.py --db /workspace/asr.db --series 1 2 3 --batch-size 64
  python Service/Consensus.py --db /workspace/asr.db --series 1 2 3 --phase whisper
  python Service/Consensus.py --db /workspace/asr.db --series 1 2 3 --phase conformer
  python Service/Consensus.py --db /workspace/asr.db --series 1 2 3 --phase consensus
"""
import argparse, gc, os, re, time, sqlite3, torch, Levenshtein, threading

DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")

def norm(t):
    if not t: return ""
    t = "".join(c for c in t if c not in DIACRITICS)
    t = re.sub(r'[أإآٱ]', 'ا', t)
    t = t.replace('ـ', '')
    return " ".join(t.split()).strip()

def ensure_columns(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN whisper_text TEXT")
    except: pass
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN conformer_text TEXT")
    except: pass
    conn.commit()
    conn.close()

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
    return [{"id":r[0],"file_path":r[1],"filename":r[2],"series_id":r[3]} for r in rows]

def run_whisper(chunks, db_path, batch_size=32):
    from transformers import pipeline
    print(f"[WHISPER] Loading large-v3-turbo via HuggingFace pipeline...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch.float16,
        device="cuda",
    )
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="ar", task="transcribe")
    print(f"[WHISPER] Loaded. Processing {len(chunks)} chunks, batch_size={batch_size}")

    conn = sqlite3.connect(db_path)
    t0 = time.time()
    errs = 0
    done = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch = chunks[batch_start:batch_end]
        paths = [c["file_path"] for c in batch]

        try:
            results = pipe(paths, batch_size=batch_size, generate_kwargs={"language": "ar", "task": "transcribe"})
            for c, r in zip(batch, results):
                text = r["text"].strip() if r and "text" in r else ""
                conn.execute("UPDATE chunks SET whisper_text=? WHERE id=?", (text, c["id"]))
                done += 1
        except Exception as e:
            print(f"  [WHISPER] Batch fail at {batch_start}: {e}")
            for c in batch:
                try:
                    r = pipe(c["file_path"], generate_kwargs={"language": "ar", "task": "transcribe"})
                    text = r["text"].strip() if r and "text" in r else ""
                    conn.execute("UPDATE chunks SET whisper_text=? WHERE id=?", (text, c["id"]))
                except:
                    conn.execute("UPDATE chunks SET whisper_text='' WHERE id=?", (c["id"],))
                    errs += 1
                done += 1

        conn.commit()
        el = time.time()-t0
        r = done/el if el > 0 else 0
        eta = (len(chunks)-done)/r if r > 0 else 0
        if done % 200 == 0 or done >= len(chunks):
            print(f"  [WHISPER {done}/{len(chunks)}] {r:.1f}/s ETA {eta/60:.0f}m err={errs}")

    conn.commit()
    conn.close()
    el = time.time()-t0
    print(f"[WHISPER] Done in {el/60:.1f}m. {len(chunks)} chunks, {errs} errors")
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
    print(f"[CONFORMER] Loaded. Processing {len(chunks)} chunks, batch={batch_size}")

    conn = sqlite3.connect(db_path)
    all_paths = [c["file_path"] for c in chunks]
    t0 = time.time()

    try:
        out = m.transcribe(all_paths, batch_size=batch_size)
        if hasattr(out, 'text'):
            texts = out.text
        elif isinstance(out, list) and len(out) > 0:
            texts = out if isinstance(out[0], str) else [o.text if hasattr(o, 'text') else str(o) for o in out]
        else:
            texts = out

        for i, (c, t) in enumerate(zip(chunks, texts)):
            conn.execute("UPDATE chunks SET conformer_text=? WHERE id=?", (t, c["id"]))
            if (i+1) % 1000 == 0:
                conn.commit()
                el = time.time()-t0
                r = (i+1)/el
                print(f"  [CONFORMER {i+1}/{len(chunks)}] {r:.1f}/s ETA {(len(chunks)-i-1)/r/60:.0f}m saved={i+1}")

    except Exception as e:
        print(f"[CONFORMER] Failed: {e}")
        for c in chunks:
            conn.execute("UPDATE chunks SET conformer_text='' WHERE id=?", (c["id"],))

    conn.commit()
    conn.close()
    el = time.time()-t0
    print(f"[CONFORMER] Done in {el/60:.1f}m")
    del m
    gc.collect()
    torch.cuda.empty_cache()

def run_consensus(chunks, db_path, max_distance=0.4):
    print(f"[CONSENSUS] Processing {len(chunks)} chunks, threshold={max_distance}")
    conn = sqlite3.connect(db_path)
    t0 = time.time()
    acc = rej = 0

    rows = conn.execute(
        f"SELECT id, whisper_text, conformer_text FROM chunks WHERE id IN ({','.join(str(c['id']) for c in chunks)})"
    ).fetchall()

    for r in rows:
        cid, wt, ct = r[0], norm(r[1] or ""), norm(r[2] or "")
        if not wt or not ct:
            conn.execute("UPDATE chunks SET transcription='', was_filtered=1, filter_reason=? WHERE id=?",
                        (f"empty_model_output", cid))
            rej += 1
            continue

        d = Levenshtein.distance(wt, ct)
        ml = max(len(wt), len(ct))
        dist = d/ml if ml > 0 else 1.0

        if dist <= max_distance:
            conn.execute("UPDATE chunks SET transcription=? WHERE id=?", (r[1], cid))
            acc += 1
        else:
            conn.execute("UPDATE chunks SET transcription='', was_filtered=1, filter_reason=? WHERE id=?",
                        (f"dist>{max_distance}:{dist:.3f}", cid))
            rej += 1

        if (acc+rej) % 1000 == 0:
            conn.commit()
            pct = 100*acc/(acc+rej) if (acc+rej) > 0 else 0
            print(f"  [CONSENSUS {acc+rej}/{len(rows)}] acc={acc} rej={rej} ({pct:.0f}%)")

    conn.commit()
    conn.close()
    el = time.time()-t0
    tot = acc+rej
    pct = 100*acc/tot if tot > 0 else 0
    print(f"[CONSENSUS] Done in {el/60:.1f}m | Accepted: {acc} ({pct:.0f}%) | Rejected: {rej}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--series", type=int, nargs="+", required=True)
    p.add_argument("--max-distance", type=float, default=0.4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--phase", choices=["all","whisper","conformer","consensus"], default="all")
    a = p.parse_args()

    print(f"DB: {a.db} | Series: {a.series} | Phase: {a.phase} | Threshold: {a.max_distance}")
    ensure_columns(a.db)

    if a.phase == "all":
        w_chunks = get_chunks(a.db, a.series, phase="whisper")
        c_chunks = get_chunks(a.db, a.series, phase="conformer")
        w_chunks = [c for c in w_chunks if os.path.exists(c["file_path"])]
        c_chunks = [c for c in c_chunks if os.path.exists(c["file_path"])]

        threads = []
        if w_chunks:
            print(f"\n=== WHISPER ({len(w_chunks)} chunks) ===")
            wt = threading.Thread(target=run_whisper, args=(w_chunks, a.db))
            threads.append(wt)
        else:
            print("Whisper: all chunks already processed")

        if c_chunks:
            print(f"=== CONFORMER ({len(c_chunks)} chunks) ===")
            ct = threading.Thread(target=run_conformer, args=(c_chunks, a.db, a.batch_size))
            threads.append(ct)
        else:
            print("Conformer: all chunks already processed")

        for t in threads: t.start()
        for t in threads: t.join()

    elif a.phase == "whisper":
        chunks = get_chunks(a.db, a.series, phase="whisper")
        chunks = [c for c in chunks if os.path.exists(c["file_path"])]
        if chunks:
            print(f"\n=== PHASE 1: WHISPER ({len(chunks)} chunks) ===")
            run_whisper(chunks, a.db)
        else:
            print("Whisper: all chunks already processed")

    elif a.phase == "conformer":
        chunks = get_chunks(a.db, a.series, phase="conformer")
        chunks = [c for c in chunks if os.path.exists(c["file_path"])]
        if chunks:
            print(f"\n=== PHASE 2: CONFORMER ({len(chunks)} chunks) ===")
            run_conformer(chunks, a.db, a.batch_size)
        else:
            print("Conformer: all chunks already processed")

    if a.phase in ("all", "consensus"):
        chunks = get_chunks(a.db, a.series, phase="consensus")
        if chunks:
            print(f"\n=== PHASE 3: CONSENSUS ({len(chunks)} chunks) ===")
            run_consensus(chunks, a.db, a.max_distance)
        else:
            print("Consensus: no chunks ready (need both whisper + conformer results)")

    # Summary
    conn = sqlite3.connect(a.db)
    ph = ",".join(str(s) for s in a.series)
    total = conn.execute(f"SELECT COUNT(*) FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph})").fetchone()[0]
    w_done = conn.execute(f"SELECT COUNT(*) FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph}) AND c.whisper_text IS NOT NULL").fetchone()[0]
    c_done = conn.execute(f"SELECT COUNT(*) FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph}) AND c.conformer_text IS NOT NULL").fetchone()[0]
    labeled = conn.execute(f"SELECT COUNT(*) FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph}) AND c.transcription IS NOT NULL AND c.transcription != ''").fetchone()[0]
    conn.close()
    print(f"\n{'='*60}")
    print(f"PROGRESS: Total={total} | Whisper={w_done} | Conformer={c_done} | Labeled={labeled}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()