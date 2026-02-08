import re
from typing import List, Optional, Dict, Set
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from Database import get_db
from Model import Series, Episode, Chunk
from ASRService import IngestionService, IngestionError

router = APIRouter()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_episode_number(filename: str) -> Optional[int]:
    """
    Extract episode number from filename.
    Priority:
    1. ep/eps pattern: "Ep 27", "eps30", "ep_5"
    2. Underscore pattern: "_27_", "_27.zip" (takes last match to avoid years)
    """
    match = re.search(r'(?:^|[\s_\-])eps?[\s_]?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    matches = re.findall(r'_(\d+)(?:_|\.)', filename)
    if matches:
        return int(matches[-1])

    return None


def parse_and_validate_files(
        files: List[UploadFile],
        extensions: List[str]
) -> Dict[int, UploadFile]:
    """
    Parse episode numbers from filenames and validate.
    Returns dict of {episode_number: file}
    Skips duplicates (keeps first), rejects unparsed.
    """
    filtered = [f for f in files if any(f.filename.endswith(ext) for ext in extensions)]

    unparsed = []
    episodes = {}

    for f in filtered:
        ep_num = extract_episode_number(f.filename)

        if ep_num is None:
            unparsed.append(f.filename)
        elif ep_num in episodes:
            print(f"[SKIP] Duplicate episode {ep_num}: {f.filename} (keeping {episodes[ep_num].filename})")
        else:
            episodes[ep_num] = f

    if unparsed:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract episode number from: {unparsed}"
        )

    return episodes


def validate_episode_pairs(
        set1: Dict[int, UploadFile],
        set2: Dict[int, UploadFile],
        name1: str,
        name2: str
) -> Set[int]:
    """
    Find matching episode numbers between two sets.
    Skips unmatched, returns set of matched episode numbers.
    """
    eps1 = set(set1.keys())
    eps2 = set(set2.keys())
    matched = eps1 & eps2

    skipped_1 = eps1 - eps2
    skipped_2 = eps2 - eps1

    for ep in sorted(skipped_1):
        print(f"[SKIP] {name1} episode {ep} has no matching {name2}")
    for ep in sorted(skipped_2):
        print(f"[SKIP] {name2} episode {ep} has no matching {name1}")

    if not matched:
        raise HTTPException(
            status_code=400,
            detail=f"No matching pairs found between {name1} and {name2}"
        )

    return matched


# ============================================================================
# SERIES ENDPOINTS
# ============================================================================

@router.post("/series")
def create_series(
        name: str,
        language: str = "ar",
        description: Optional[str] = None,
        db: Session = Depends(get_db)
):
    existing = db.query(Series).filter(Series.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Series '{name}' already exists (id={existing.id})")
    series = Series(name=name, language=language, description=description)
    db.add(series)
    db.commit()
    db.refresh(series)
    return {"id": series.id, "name": series.name}


@router.get("/series")
def list_series(db: Session = Depends(get_db)):
    series = db.query(Series).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "language": s.language,
            "episode_count": db.query(Episode).filter(Episode.series_id == s.id).count()
        }
        for s in series
    ]


@router.get("/series/{series_id}")
def get_series(series_id: int, db: Session = Depends(get_db)):
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    return {
        "id": series.id,
        "name": series.name,
        "language": series.language,
        "description": series.description
    }


@router.get("/series/{series_id}/stats")
def get_series_stats(series_id: int, db: Session = Depends(get_db)):
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    episodes = db.query(Episode).filter(Episode.series_id == series_id).all()
    chunks = db.query(Chunk).join(Episode).filter(Episode.series_id == series_id).all()

    transcribed = sum(1 for c in chunks if c.transcription)
    total_duration = sum(c.duration or 0 for c in chunks)

    return {
        "series_id": series_id,
        "name": series.name,
        "episodes": len(episodes),
        "chunks": len(chunks),
        "transcribed": transcribed,
        "total_duration_hours": round(total_duration / 3600, 2)
    }


@router.get("/series/{series_id}/episodes")
def list_episodes(series_id: int, db: Session = Depends(get_db)):
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    episodes = db.query(Episode).filter(Episode.series_id == series_id).all()
    return [
        {
            "id": ep.id,
            "name": ep.name,
            "episode_number": ep.episode_number,
            "status": ep.status.value if ep.status else None,
            "duration_seconds": ep.duration_seconds,
            "chunk_count": db.query(Chunk).filter(Chunk.episode_id == ep.id).count()
        }
        for ep in episodes
    ]


@router.delete("/series/{series_id}")
def delete_series(series_id: int, db: Session = Depends(get_db)):
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    service = IngestionService(db)
    service.delete_series_files(series_id)

    db.query(Series).filter(Series.id == series_id).delete()
    db.commit()

    return {"message": f"Series {series_id} deleted"}


@router.delete("/episodes/{episode_id}")
def delete_episode(episode_id: int, db: Session = Depends(get_db)):
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    service = IngestionService(db)
    service.delete_episode_files(episode_id)

    db.query(Episode).filter(Episode.id == episode_id).delete()
    db.commit()

    return {"message": f"Episode {episode_id} deleted"}


# ============================================================================
# INGESTION ENDPOINTS
# ============================================================================

@router.post("/series/{series_id}/ingest/video")
async def ingest_video(
        series_id: int,
        video: UploadFile = File(...),
        episode_name: Optional[str] = Form(None),
        episode_number: Optional[int] = Form(None),
        max_chunk_sec: int = Form(45),
        db: Session = Depends(get_db)
):
    """Ingest a single video file - extracts audio, chunks, ready for transcription."""
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    if episode_number is None:
        episode_number = extract_episode_number(video.filename)
        if episode_number is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract episode number from '{video.filename}'. "
                       f"Use format like 'Ep 5', 'eps10', '_3_' in filename, or provide episode_number manually."
            )

    try:
        service = IngestionService(db)
        video.file.seek(0)
        result = service.process_episode(
            video_file=video.file,
            video_filename=video.filename,
            series_id=series.id,
            episode_name=episode_name,
            episode_number=episode_number,
            max_chunk_sec=max_chunk_sec
        )
        return result
    except IngestionError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/series/{series_id}/ingest/video-batch")
async def ingest_video_batch(
        series_id: int,
        files: List[UploadFile] = File(...),
        max_chunk_sec: int = Form(45),
        db: Session = Depends(get_db)
):
    """Batch ingest multiple video files."""
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    video_episodes = parse_and_validate_files(files, [".mp4", ".mkv", ".avi", ".mov"])

    service = IngestionService(db)
    results = []

    for ep_num in sorted(video_episodes.keys()):
        video_file = video_episodes[ep_num]
        print(f"\n[Episode {ep_num}] Processing {video_file.filename}")

        try:
            video_file.file.seek(0)
            result = service.process_episode(
                video_file=video_file.file,
                video_filename=video_file.filename,
                series_id=series.id,
                episode_name=f"Episode {ep_num}",
                episode_number=ep_num,
                max_chunk_sec=max_chunk_sec
            )
            results.append(result)
            print(f"Done: {result.get('chunk_count', 0)} chunks")
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "episode_number": ep_num,
                "status": "failed",
                "error": str(e)
            })

    return {"episodes": results, "total": len(results)}


@router.post("/series/{series_id}/ingest/chunks-with-transcripts")
async def ingest_chunks_with_transcripts(
        series_id: int,
        chunks_zip: UploadFile = File(...),
        transcripts_json: UploadFile = File(...),
        episode_number: Optional[int] = Form(None),
        db: Session = Depends(get_db)
):
    """Ingest a single episode with pre-chunked audio and transcripts."""
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    if episode_number is None:
        episode_number = extract_episode_number(chunks_zip.filename)
        if episode_number is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract episode number from '{chunks_zip.filename}'. "
                       f"Use format like 'Ep 5', 'eps10', '_3_' in filename, or provide episode_number manually."
            )

    try:
        service = IngestionService(db)
        chunks_zip.file.seek(0)
        transcripts_json.file.seek(0)
        result = service.process_episode_with_transcripts(
            chunks_zip=chunks_zip.file,
            transcripts_json=transcripts_json.file,
            series_id=series.id,
            series_name=series.name,
            episode_name=f"Episode {episode_number}",
            episode_number=episode_number
        )
        return result
    except IngestionError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/series/{series_id}/ingest/chunks-batch")
async def ingest_chunks_batch(
        series_id: int,
        files: List[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    """Batch ingest multiple episodes with pre-chunked audio and transcripts."""
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    zip_episodes = parse_and_validate_files(files, [".zip"])
    json_episodes = parse_and_validate_files(files, [".json"])
    matched = validate_episode_pairs(zip_episodes, json_episodes, "ZIPs", "JSONs")

    service = IngestionService(db)
    results = []

    for ep_num in sorted(matched):
        zip_file = zip_episodes[ep_num]
        json_file = json_episodes[ep_num]
        print(f"\n[Episode {ep_num}] Processing {zip_file.filename}")

        try:
            zip_file.file.seek(0)
            json_file.file.seek(0)
            result = service.process_episode_with_transcripts(
                chunks_zip=zip_file.file,
                transcripts_json=json_file.file,
                series_id=series.id,
                series_name=series.name,
                episode_name=f"Episode {ep_num}",
                episode_number=ep_num
            )
            results.append(result)
            print(f"Done: {result.get('chunk_count', 0)} chunks")
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "episode_number": ep_num,
                "status": "failed",
                "error": str(e)
            })

    return {"episodes": results, "total": len(results)}


@router.post("/series/{series_id}/ingest/srt")
async def ingest_srt(
        series_id: int,
        media_file: UploadFile = File(...),
        srt_file: UploadFile = File(...),
        episode_number: Optional[int] = Form(None),
        db: Session = Depends(get_db)
):
    """Ingest a single episode with media file and SRT subtitles."""
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    if episode_number is None:
        episode_number = extract_episode_number(media_file.filename)
        if episode_number is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract episode number from '{media_file.filename}'. "
                       f"Use format like 'Ep 5', 'eps10', '_3_' in filename, or provide episode_number manually."
            )

    try:
        service = IngestionService(db)
        media_file.file.seek(0)
        srt_file.file.seek(0)
        result = service.process_episode_with_srt(
            video_file=media_file.file,
            srt_file=srt_file.file,
            video_filename=media_file.filename,
            srt_filename=srt_file.filename,
            series_id=series.id,
            episode_name=f"Episode {episode_number}",
            episode_number=episode_number
        )
        return result
    except IngestionError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/series/{series_id}/ingest/srt-batch")
async def ingest_srt_batch(
        series_id: int,
        files: List[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    """Batch ingest multiple episodes with media files and SRT subtitles."""
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    media_episodes = parse_and_validate_files(files, [".mp4", ".mkv", ".mp3", ".wav", ".m4a"])
    srt_episodes = parse_and_validate_files(files, [".srt"])
    matched = validate_episode_pairs(media_episodes, srt_episodes, "Media files", "SRT files")

    service = IngestionService(db)
    results = []

    for ep_num in sorted(matched):
        media_file = media_episodes[ep_num]
        srt_file = srt_episodes[ep_num]
        print(f"\n[Episode {ep_num}] Processing {media_file.filename} + {srt_file.filename}")

        try:
            media_file.file.seek(0)
            srt_file.file.seek(0)
            result = service.process_episode_with_srt(
                video_file=media_file.file,
                srt_file=srt_file.file,
                video_filename=media_file.filename,
                srt_filename=srt_file.filename,
                series_id=series.id,
                episode_name=f"Episode {ep_num}",
                episode_number=ep_num
            )
            results.append(result)
            print(f"Done: {result.get('chunk_count', 0)} chunks")
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "episode_number": ep_num,
                "status": "failed",
                "error": str(e)
            })

    return {"episodes": results, "total": len(results)}


@router.post("/series/{series_id}/ingest/masc")
async def ingest_masc_dataset(
        series_id: int,
        tar_path: str = Form(...),
        transcripts_csv: str = Form(...),
        metadata_csv: str = Form(...),
        dialect: Optional[str] = Form(None),
        max_samples: Optional[int] = Form(None),
        db: Session = Depends(get_db)
):
    """Ingest MASC dataset from local paths."""
    series = db.query(Series).filter(Series.id == series_id).first()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    if not Path(tar_path).exists():
        raise HTTPException(status_code=404, detail="Tar file not found")
    if not Path(transcripts_csv).exists():
        raise HTTPException(status_code=404, detail="Transcription file not found")
    try:
        service = IngestionService(db)
        result = service.process_masc_dataset(
            tar_path=tar_path,
            transcripts_csv=transcripts_csv,
            metadata_csv=metadata_csv,
            series_id=series_id,
            dialect=dialect,
            max_samples=max_samples
        )
        return result
    except IngestionError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UPLOAD PAGE
# ============================================================================

@router.get("/upload-page", response_class=HTMLResponse)
def upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Batch Upload</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h2 { color: #333; }
            h3 { color: #666; margin-top: 30px; }
            .form-group { margin: 20px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="file"] { padding: 10px; border: 2px dashed #ccc; width: 100%; box-sizing: border-box; }
            input[type="number"] { padding: 8px; width: 100px; }
            button { background: #4CAF50; color: white; padding: 12px 24px; border: none; cursor: pointer; font-size: 16px; margin-top: 20px; }
            button:hover { background: #45a049; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            .note { color: #666; font-size: 14px; margin-top: 5px; }
            .section { background: #f9f9f9; padding: 15px; border-radius: 8px; margin: 15px 0; }
            .tab { display: none; }
            .tab.active { display: block; }
            .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
            .tabs button { background: #ddd; color: #333; margin-top: 0; }
            .tabs button.active { background: #4CAF50; color: white; }
            #status { display: none; margin-top: 20px; padding: 15px; border-radius: 8px; }
            pre { white-space: pre-wrap; word-wrap: break-word; margin-top: 8px; }
        </style>
    </head>
    <body>
        <h2>Batch Upload</h2>
        <p class="note">Filenames must contain episode number: "Ep 5", "eps10", "ep_3", or "_5_". Duplicates and unmatched pairs are skipped automatically.</p>

        <div class="tabs">
            <button class="active" onclick="showTab('chunks')">Chunks + JSON</button>
            <button onclick="showTab('srt')">SRT + Media</button>
            <button onclick="showTab('video')">Videos Only</button>
        </div>

        <div id="chunks" class="tab active">
            <h3>Upload Chunks (ZIP) + Transcripts (JSON)</h3>
            <form action="/api/series/1/ingest/chunks-batch" method="post" enctype="multipart/form-data" id="chunksForm">
                <div class="form-group">
                    <label>Series ID:</label>
                    <input type="number" id="chunksSeriesId" value="1" min="1">
                </div>
                <div class="section">
                    <div class="form-group">
                        <label>ZIP Files (audio chunks):</label>
                        <input type="file" name="files" multiple accept=".zip">
                        <p class="note">Select all episode ZIPs (Ctrl+A)</p>
                    </div>
                </div>
                <div class="section">
                    <div class="form-group">
                        <label>JSON Files (transcripts):</label>
                        <input type="file" name="files" multiple accept=".json">
                        <p class="note">Select all transcript JSONs (Ctrl+A)</p>
                    </div>
                </div>
                <button type="submit">Upload All</button>
            </form>
        </div>

        <div id="srt" class="tab">
            <h3>Upload SRT + Media Files</h3>
            <form action="/api/series/1/ingest/srt-batch" method="post" enctype="multipart/form-data" id="srtForm">
                <div class="form-group">
                    <label>Series ID:</label>
                    <input type="number" id="srtSeriesId" value="1" min="1">
                </div>
                <div class="section">
                    <div class="form-group">
                        <label>Media Files (mp4, mkv, mp3, wav, m4a):</label>
                        <input type="file" name="files" multiple accept=".mp4,.mkv,.mp3,.wav,.m4a">
                        <p class="note">Select all media files (Ctrl+A)</p>
                    </div>
                </div>
                <div class="section">
                    <div class="form-group">
                        <label>SRT Files:</label>
                        <input type="file" name="files" multiple accept=".srt">
                        <p class="note">Select all SRT files (Ctrl+A)</p>
                    </div>
                </div>
                <button type="submit">Upload All</button>
            </form>
        </div>

        <div id="video" class="tab">
            <h3>Upload Videos (for chunking, no transcripts)</h3>
            <form action="/api/series/1/ingest/video-batch" method="post" enctype="multipart/form-data" id="videoForm">
                <div class="form-group">
                    <label>Series ID:</label>
                    <input type="number" id="videoSeriesId" value="1" min="1">
                </div>
                <div class="section">
                    <div class="form-group">
                        <label>Video Files (mp4, mkv, avi, mov):</label>
                        <input type="file" name="files" multiple accept=".mp4,.mkv,.avi,.mov">
                        <p class="note">Select all video files (Ctrl+A)</p>
                    </div>
                </div>
                <div class="form-group">
                    <label>Max Chunk Seconds:</label>
                    <input type="number" name="max_chunk_sec" value="45" min="10" max="120">
                </div>
                <button type="submit">Upload All</button>
            </form>
        </div>

        <div id="status"></div>

        <script>
            function showTab(name) {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
                document.getElementById(name).classList.add('active');
                event.target.classList.add('active');
            }

            document.getElementById('chunksSeriesId').addEventListener('change', function() {
                document.getElementById('chunksForm').action = '/api/series/' + this.value + '/ingest/chunks-batch';
            });
            document.getElementById('srtSeriesId').addEventListener('change', function() {
                document.getElementById('srtForm').action = '/api/series/' + this.value + '/ingest/srt-batch';
            });
            document.getElementById('videoSeriesId').addEventListener('change', function() {
                document.getElementById('videoForm').action = '/api/series/' + this.value + '/ingest/video-batch';
            });

            document.querySelectorAll('form').forEach(form => {
                form.addEventListener('submit', async function(e) {
                    e.preventDefault();
                    const status = document.getElementById('status');
                    const btn = form.querySelector('button[type="submit"]');
                    const originalText = btn.textContent;

                    btn.disabled = true;
                    btn.textContent = 'Uploading...';
                    status.style.display = 'block';
                    status.style.background = '#fff3cd';
                    status.style.color = '#856404';
                    status.innerHTML = '<b>Uploading files... Please wait.</b>';

                    try {
                        const formData = new FormData(form);
                        const resp = await fetch(form.action, { method: 'POST', body: formData });
                        const data = await resp.json();

                        if (!resp.ok) {
                            status.style.background = '#f8d7da';
                            status.style.color = '#721c24';
                            status.innerHTML = '<b>Error:</b><pre>' +
                                (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail, null, 2)) + '</pre>';
                        } else {
                            status.style.background = '#d4edda';
                            status.style.color = '#155724';
                            let html = '<b>Success!</b><br>';
                            if (data.episodes) {
                                html += '<br>Processed ' + data.total + ' episodes:<br><br>';
                                data.episodes.forEach(ep => {
                                    const icon = ep.status === 'failed' ? '&#10060;' : '&#9989;';
                                    html += icon + ' Episode ' + (ep.episode_number || '?') +
                                        (ep.chunk_count ? ' - ' + ep.chunk_count + ' chunks' : '') +
                                        (ep.error ? ' - ERROR: ' + ep.error : '') + '<br>';
                                });
                            } else {
                                html += '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                            }
                            status.innerHTML = html;
                        }
                    } catch(err) {
                        status.style.background = '#f8d7da';
                        status.style.color = '#721c24';
                        status.innerHTML = '<b>Network Error:</b> ' + err.message;
                    }
                    btn.disabled = false;
                    btn.textContent = originalText;
                });
            });
        </script>
    </body>
    </html>
    """