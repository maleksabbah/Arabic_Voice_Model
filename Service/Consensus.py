import argparse, gc, os, re, time, threading, sqlite3, torch, Levenshtein

DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")

def norm(t):
    if not t: return ""
    t = "".join(c for c in t if c not in DIACRITICS)
    t = re.sub(r'[أإآٱ]', 'ا', t)
    t = t.replace('ـ', '')
    return " ".join(t.split()).strip()

def get_chunks(db, sids, resume=False):
    conn = sqlite3.connect(db)
    ph = ",".join(str(s) for s in sids)
    q = f"SELECT c.id,c.file_path,c.filename,e.series_id FROM chunks c JOIN episodes e ON c.episode_id=e.id WHERE e.series_id IN ({ph})"
    if resume: q += " AND (c.transcription IS NULL OR c.transcription='')"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [{"id":r[0],"file_path":r[1],"filename":r[2],"series_id":r[3]} for r in rows]

def whisper_worker(chunks, res):
    from faster_whisper import WhisperModel
    print("[WHISPER] Loading...")
    m = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    print(f"[WHISPER] Loaded. {len(chunks)} chunks")
    t0 = time.time(); errs = 0
    for i, c in enumerate(chunks):
        try:
            segs, _ = m.transcribe(c["file_path"], language="ar", beam_size=5, vad_filter=True)
            res[c["id"]] = " ".join([s.text for s in segs]).strip()
        except:
            res[c["id"]] = ""; errs += 1
        if (i+1) % 1000 == 0 or i == len(chunks)-1:
            el = time.time()-t0; r = (i+1)/el; eta = (len(chunks)-i-1)/r
            print(f"  [WHISPER {i+1}/{len(chunks)}] {r:.1f}/s ETA {eta/60:.0f}m err={errs}")
    print(f"[WHISPER] Done {(time.time()-t0)/60:.1f}m")
    del m; gc.collect(); torch.cuda.empty_cache()

def conformer_worker(chunks, res, bs=64):
    import nemo.collections.asr as nemo_asr
    print("[CONFORMER] Loading...")
    m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained("nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0")
    m.change_decoding_strategy(decoder_type="ctc")
    m.eval()
    m.cuda()
    print(f"[CONFORMER] Loaded. {len(chunks)} chunks, batch={bs}")
    t0 = time.time()
    all_paths = [c["file_path"] for c in chunks]
    try:
        out = m.transcribe(all_paths, batch_size=bs)
        if hasattr(out, 'text'):
            texts = out.text
        elif isinstance(out, list) and len(out) > 0:
            texts = out if isinstance(out[0], str) else [o.text if hasattr(o, 'text') else str(o) for o in out]
        else:
            texts = out
        for c, t in zip(chunks, texts):
            res[c["id"]] = t
            if len(res) % 2000 == 0:
                el = time.time()-t0; r = len(res)/el
                print(f"  [CONFORMER {len(res)}/{len(chunks)}] {r:.1f}/s ETA {(len(chunks)-len(res))/r/60:.0f}m")
    except Exception as e:
        print(f"[CONFORMER] Failed: {e}")
        for c in chunks: res[c["id"]] = ""
    print(f"[CONFORMER] Done {(time.time()-t0)/60:.1f}m")
    del m; gc.collect(); torch.cuda.empty_cache()

def consensus_writer(chunks, wr, cr, db_path, max_d=0.4, dry=False):
    conn = None if dry else sqlite3.connect(db_path)
    acc = rej = 0; t0 = time.time()
    for i, c in enumerate(chunks):
        cid = c["id"]
        while cid not in wr or cid not in cr: time.sleep(0.01)
        wt = norm(wr.get(cid,"")); ct = norm(cr.get(cid,""))
        if not wt or not ct:
            ok = False; best = ""
        else:
            d = Levenshtein.distance(wt, ct); ml = max(len(wt),len(ct))
            dist = d/ml if ml>0 else 1.0; ok = dist <= max_d
            best = wr[cid] if ok else ""
        if not dry:
            if ok:
                conn.execute("UPDATE chunks SET transcription=? WHERE id=?", (best, cid)); acc += 1
            else:
                conn.execute("UPDATE chunks SET transcription='',was_filtered=1,filter_reason=? WHERE id=?", (f"dist>{max_d}", cid)); rej += 1
            if (i+1) % 500 == 0: conn.commit()
        else:
            acc += 1 if ok else 0; rej += 0 if ok else 1
        if cid in wr: del wr[cid]
        if cid in cr: del cr[cid]
        if (i+1) % 1000 == 0 or i == len(chunks)-1:
            el = time.time()-t0; r = (i+1)/el; pct = 100*acc/(i+1)
            print(f"  [CONSENSUS {i+1}/{len(chunks)}] acc={acc} rej={rej} ({pct:.0f}%) {r:.1f}/s ETA {(len(chunks)-i-1)/r/60:.0f}m")
    if not dry and conn: conn.commit(); conn.close()
    return acc, rej

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--series", type=int, nargs="+", required=True)
    p.add_argument("--max-distance", type=float, default=0.4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()
    print(f"DB: {a.db} | Series: {a.series} | Threshold: {a.max_distance} | Resume: {a.resume}")
    chunks = get_chunks(a.db, a.series, a.resume)
    if not chunks: print("No chunks!"); return
    missing = [c for c in chunks if not os.path.exists(c["file_path"])]
    if missing:
        print(f"WARNING: {len(missing)} missing files")
        chunks = [c for c in chunks if os.path.exists(c["file_path"])]
    sc = {}
    for c in chunks: sc[c["series_id"]] = sc.get(c["series_id"],0)+1
    for sid,cnt in sorted(sc.items()): print(f"  Series {sid}: {cnt}")
    print(f"  Total: {len(chunks)}")
    t0 = time.time()
    wr = {}; cr = {}
    wt = threading.Thread(target=whisper_worker, args=(chunks, wr))
    ct = threading.Thread(target=conformer_worker, args=(chunks, cr, a.batch_size))
    wt.start(); ct.start()
    acc, rej = consensus_writer(chunks, wr, cr, a.db, a.max_distance, a.dry_run)
    wt.join(); ct.join()
    el = time.time()-t0; tot = len(chunks)
    print(f"\nTotal: {tot} | Accepted: {acc} ({100*acc/tot:.0f}%) | Rejected: {rej} ({100*rej/tot:.0f}%) | Time: {el/60:.1f}m ({el/3600:.1f}h)")

if __name__ == "__main__":
    main()