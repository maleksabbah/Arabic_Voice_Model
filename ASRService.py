import os
import shutil
import subprocess
import zipfile
import re
import json
import tarfile
import pandas as pd
import soundfile as sf
import io
from pathlib import Path
from typing import List, Tuple, Optional, BinaryIO

from sqlalchemy.orm import Session

from Config import settings
from Model import Series, Episode, Chunk, ProcessingStatus


class IngestionError(Exception):
    pass


class IngestionService:
    def __init__(self, db: Session):
        self.db = db

        # Paths from config
        self.uploads_dir = settings.uploads_dir
        self.audio_dir = settings.audio_dir
        self.chunks_dir = settings.chunks_dir

        # Chunking settings
        self.max_chunk_sec = settings.max_chunk_seconds  # 45
        self.min_silence_duration = settings.min_silence_duration  # 0.5
        self.silence_threshold_db = settings.silence_threshold_db  # -35

    def process_episode(
            self,
            video_file: BinaryIO,
            video_filename: str,
            series_id: int,
            episode_name: Optional[str] = None,
            episode_number: Optional[int] = None,
            max_chunk_sec: Optional[int] = None
    ) -> dict:
        name = episode_name or Path(video_filename).stem
        episode = Episode(
            series_id=series_id,
            name=name,
            episode_number=episode_number,
            status=ProcessingStatus.PENDING,
            status_message="Processing episode",
        )
        self.db.add(episode)
        self.db.commit()
        self.db.refresh(episode)

        try:
            chunks = self._process_video_to_chunks(
                video_file=video_file,
                video_filename=video_filename,
                episode=episode,
                max_chunk_sec=max_chunk_sec
            )
            return {
                "episode_id": episode.id,
                "episode_number": episode_number,
                "name": episode.name,
                "duration": episode.duration_seconds,
                "chunk_count": len(chunks),
                "status": "completed"
            }
        except Exception as e:
            episode.status = ProcessingStatus.FAILED
            episode.status_message = str(e)
            self.db.commit()
            return {
                "episode_id": episode.id,
                "episode_number": episode_number,
                "name": episode.name,
                "status": "failed",
                "error": str(e)
            }

    def process_zip(
            self,
            zip_file: BinaryIO,
            series_id: int,
            max_chunk_sec: Optional[int] = None,
    ) -> List[dict]:
        results = []

        temp_zip_path = self.uploads_dir / f"temp_series_{series_id}.zip"
        with open(temp_zip_path, "wb") as temp_zip:
            shutil.copyfileobj(zip_file, temp_zip)

        temp_extract_dir = self.uploads_dir / f"temp_extract_{series_id}"
        temp_extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as z:
                video_extensions = ('.mp4', '.webm', '.avi', '.mp3')
                video_files = [
                    f for f in z.namelist()
                    if f.lower().endswith(video_extensions)
                       and not os.path.basename(f).startswith('.')
                       and not f.startswith('__MACOSX')
                ]

                print(f"Found {len(video_files)} video files")

                for idx, video_path in enumerate(video_files, 1):
                    video_name = os.path.basename(video_path)
                    episode_name = Path(video_path).stem

                    print(f"\n[{idx}/{len(video_files)}] Processing: {video_name}")

                    try:
                        z.extract(video_path, temp_extract_dir)
                        extracted_path = temp_extract_dir / video_path

                        episode = Episode(
                            series_id=series_id,
                            name=episode_name,
                            episode_number=idx,
                            status=ProcessingStatus.PENDING
                        )
                        self.db.add(episode)
                        self.db.commit()
                        self.db.refresh(episode)

                        with open(extracted_path, "rb") as f:
                            chunks = self._process_video_to_chunks(
                                video_file=f,
                                video_filename=video_name,
                                episode=episode,
                                max_chunk_sec=max_chunk_sec
                            )
                        results.append({
                            "episode_id": episode.id,
                            "episode_number": idx,
                            "name": episode.name,
                            "duration": episode.duration_seconds,
                            "chunk_count": len(chunks),
                            "status": "completed"
                        })

                        if extracted_path.exists():
                            extracted_path.unlink()
                        print(f" Done: {len(chunks)} chunks")
                    except Exception as e:
                        print(f"Failed: {e}")
                        results.append({
                            "name": episode_name,
                            "episode_number": idx,
                            "status": "failed",
                            "error": str(e)
                        })
                        continue
            return results
        finally:
            if temp_zip_path.exists():
                temp_zip_path.unlink()
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)

    def process_episode_with_transcripts(
            self,
            chunks_zip: BinaryIO,
            transcripts_json: BinaryIO,
            series_id: int,
            series_name: str,
            episode_name: str,
            episode_number: int,
    ) -> dict:
        temp_dir = self.uploads_dir / f"temp_series_{series_name}_{episode_number}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save and extract zip
            zip_path = temp_dir / "chunks.zip"
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(chunks_zip, f)

            chunks_extract_dir = temp_dir / "chunks"
            chunks_extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(chunks_extract_dir)

            # Save and load JSON
            json_path = temp_dir / "transcripts.json"
            with open(json_path, "wb") as f:
                shutil.copyfileobj(transcripts_json, f)
            with open(json_path, "r", encoding='utf-8') as f:
                data = json.load(f)
            transcripts = {
                item["file"]: item["text"]
                for item in data
                if item.get("text") and item.get("file")
            }

            # Find WAV files (recursive)
            wav_files = list(chunks_extract_dir.glob("**/*.wav"))
            if not wav_files:
                raise IngestionError("No wav files found in ZIP")

            print(f"Found {len(wav_files)} wav files")

            # Create episode and commit to get ID
            episode = Episode(
                series_id=series_id,
                name=episode_name,
                episode_number=episode_number,
                status=ProcessingStatus.PENDING,
                status_message="Processing episode with transcripts",
            )
            self.db.add(episode)
            self.db.commit()
            self.db.refresh(episode)

            # Storage directory (now episode.id exists)
            storage_dir = self.chunks_dir / f"episode_{episode.id}"
            storage_dir.mkdir(parents=True, exist_ok=True)

            # Process each WAV file
            chunks = []
            total_duration = 0.0
            no_transcript = 0

            for idx, wav_path in enumerate(sorted(wav_files)):
                filename = wav_path.name

                # Look at transcript by filename
                text = transcripts.get(filename, "")

                if not text:
                    no_transcript += 1
                    continue

                # Copy WAV to storage
                dest_path = storage_dir / filename
                shutil.copy2(wav_path, dest_path)

                # Get duration
                duration = self._get_duration(dest_path)
                total_duration += duration

                # Create chunk with transcripts
                chunk = Chunk(
                    episode_id=episode.id,
                    chunk_index=idx,
                    filename=filename,
                    file_path=str(dest_path),
                    start_time=0,
                    end_time=duration,
                    duration=round(duration, 3),
                    is_cleaned=True,
                    transcription=text
                )
                self.db.add(chunk)
                chunks.append(chunk)

            self.db.commit()

            episode.duration_seconds = total_duration
            episode.status = ProcessingStatus.COMPLETED
            episode.status_message = f"Processed {len(chunks)} chunks"
            self.db.commit()

            print(f"Done: {len(chunks)} chunks, {no_transcript} skipped (no transcript)")

            return {
                "episode_id": episode.id,
                "episode_number": episode_number,
                "name": episode.name,
                "duration": round(total_duration, 2),
                "chunk_count": len(chunks),
                "skipped": no_transcript,
                "status": "completed"
            }
        except Exception as e:
            return {
                "episode_number": episode_number,
                "name": episode_name,
                "status": "failed",
                "error": str(e)
            }
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def process_batch_with_transcripts(
            self,
            pairs: List[Tuple[BinaryIO, BinaryIO]],
            series_id: int,
            series_name: str,
            start_episode: int = 1
    ) -> List[dict]:
        results = []
        for idx, (zip_file, json_file) in enumerate(pairs):
            ep_num = start_episode + idx
            print(f"\n[Episode {ep_num}] Processing")

            result = self.process_episode_with_transcripts(
                chunks_zip=zip_file,
                transcripts_json=json_file,
                series_id=series_id,
                series_name=series_name,
                episode_name=f"Episode {ep_num}",
                episode_number=ep_num,
            )
            results.append(result)

            print(f"  Done: {result.get('chunk_count', 0)} chunks")
        return results

    def process_masc_dataset(
            self,
            tar_path: str,
            transcripts_csv: str,
            metadata_csv: str,
            series_id: int,
            dialect: Optional[str] = None,
            max_samples: Optional[int] = None
    ) -> dict:
        print("Loading CSVs")

        transcripts_df = pd.read_csv(transcripts_csv, encoding="utf-8")
        metadata_df = pd.read_csv(metadata_csv, encoding="utf-8")

        df = transcripts_df.merge(metadata_df[['video_id', 'dialect', 'country', 'gender']], on='video_id', how='left')

        print(f"Total samples: {len(df)}")

        if dialect:
            df = df[df['dialect'] == dialect]
            print(f"Filtered to {dialect}: {len(df)} samples")

        if max_samples:
            df = df.head(max_samples)
            print(f"Limited to {max_samples} samples")

        episode_name = f"MASC_{dialect}" if dialect else "MASC_all"
        episode = Episode(
            series_id=series_id,
            name=episode_name,
            status=ProcessingStatus.PROCESSING,
            status_message="Processing MASC dataset",
        )
        self.db.add(episode)
        self.db.commit()
        self.db.refresh(episode)

        storage_dir = self.chunks_dir / f"episode_{episode.id}"
        storage_dir.mkdir(parents=True, exist_ok=True)

        video_ids = set(df["video_id"].unique())
        print(f"Need {len(video_ids)} videos to process")

        print("Streaming from tar")

        chunks = []
        total_duration = 0.0
        processed = 0
        skipped = 0

        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if not member.name.endswith(".wav"):
                    continue
                video_id = member.name.split("/")[-1].replace(".wav", "")

                if video_id not in video_ids:
                    continue

                video_segments = df[df["video_id"] == video_id]

                if len(video_segments) == 0:
                    continue
                print(f"\n[{processed + 1}] Processing: {video_id} ({len(video_segments)} segments)")

                try:
                    f = tar.extractfile(member)
                    audio_data, sample_rate = sf.read(io.BytesIO(f.read()))

                    for idx, row in video_segments.iterrows():
                        start_sec = row["start"]
                        end_sec = row["end"]
                        text = row["text"]

                        start_samples = int(start_sec * sample_rate)
                        end_samples = int(end_sec * sample_rate)

                        segment_audio = audio_data[start_samples:end_samples]
                        duration = len(segment_audio) / sample_rate

                        if duration < 0.5:
                            skipped += 1
                            continue

                        chunk_idx = len(chunks)
                        filename = f"chunk_{chunk_idx:05d}.wav"
                        chunk_path = storage_dir / filename

                        sf.write(str(chunk_path), segment_audio, sample_rate)

                        chunk = Chunk(
                            episode_id=episode.id,
                            chunk_index=chunk_idx,
                            filename=filename,
                            file_path=str(chunk_path),
                            start_time=round(start_sec, 3),
                            end_time=round(end_sec, 3),
                            duration=round(duration, 3),
                            is_cleaned=True,
                            transcription=text
                        )
                        self.db.add(chunk)
                        chunks.append(chunk)
                        total_duration += duration

                    processed += 1

                    if processed % 100 == 0:
                        self.db.commit()
                        print(f" Progress: {processed} videos, {len(chunks)} chunks")

                except Exception as e:
                    print(f"Error: {e}")
                    skipped += 1
                    continue

        self.db.commit()

        episode.duration_seconds = total_duration
        episode.status = ProcessingStatus.COMPLETED
        episode.status_message = f"Processed {len(chunks)} chunks from {processed} videos"
        self.db.commit()

        print(f"\nDone!")
        print(f"  Videos processed: {processed}")
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Skipped: {skipped}")
        print(f"  Total duration: {total_duration / 3600:.2f} hours")

        return {
            "episode_id": episode.id,
            "episode_number": None,
            "episode_name": episode_name,
            "videos_processed": processed,
            "chunks_created": len(chunks),
            "chunk_count": len(chunks),
            "skipped": skipped,
            "total_duration_hours": round(total_duration / 3600, 2),
            "status": "completed"
        }

    def process_episode_with_srt(
            self,
            video_file: BinaryIO,
            srt_file: BinaryIO,
            video_filename: str,
            srt_filename: str,
            series_id: int,
            episode_name: Optional[str] = None,
            episode_number: Optional[int] = None
    ) -> dict:
        temp_dir = self.uploads_dir / f"temp_srt_{series_id}_{episode_number or 'ep'}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save video
            video_path = temp_dir / video_filename
            with open(video_path, 'wb') as f:
                shutil.copyfileobj(video_file, f)

            # Save SRT
            srt_path = temp_dir / srt_filename
            with open(srt_path, 'wb') as f:
                shutil.copyfileobj(srt_file, f)

            # Extract audio from video
            audio_path = temp_dir / "audio.wav"
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise IngestionError(f"FFmpeg failed: {result.stderr}")

            # Delete video, keep audio
            video_path.unlink()

            # Parse SRT
            subtitles = self._parse_srt(srt_path)
            if not subtitles:
                raise IngestionError("No subtitles found in SRT file")

            print(f"Parsed {len(subtitles)} subtitles")

            # Create episode
            name = episode_name or f"Episode {episode_number}" if episode_number else video_filename
            episode = Episode(
                series_id=series_id,
                name=name,
                episode_number=episode_number,
                status=ProcessingStatus.PROCESSING,
                status_message="Processing video with SRT"
            )
            self.db.add(episode)
            self.db.commit()
            self.db.refresh(episode)

            # Storage directories
            audio_storage = self.audio_dir / f"episode_{episode.id}"
            audio_storage.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_path, audio_storage / "audio.wav")

            chunks_storage = self.chunks_dir / f"episode_{episode.id}"
            chunks_storage.mkdir(parents=True, exist_ok=True)

            # Load audio
            audio_data, sample_rate = sf.read(str(audio_path))

            # Create chunks from subtitles
            chunks = []
            total_duration = 0.0

            for idx, sub in enumerate(subtitles):
                start_sec = sub['start']
                end_sec = sub['end']
                text = sub['text']

                # Skip empty subtitles
                if not text.strip():
                    continue

                # Convert to samples
                start_sample = int(start_sec * sample_rate)
                end_sample = int(end_sec * sample_rate)

                # Extract segment
                segment_audio = audio_data[start_sample:end_sample]
                duration = len(segment_audio) / sample_rate

                # Skip very short segments
                if duration < 0.3:
                    continue

                # Save chunk
                filename = f"chunk_{idx:05d}.wav"
                chunk_path = chunks_storage / filename
                sf.write(str(chunk_path), segment_audio, sample_rate)

                # Create chunk record
                chunk = Chunk(
                    episode_id=episode.id,
                    chunk_index=idx,
                    filename=filename,
                    file_path=str(chunk_path),
                    start_time=round(start_sec, 3),
                    end_time=round(end_sec, 3),
                    duration=round(duration, 3),
                    is_cleaned=True,
                    transcription=text
                )
                self.db.add(chunk)
                chunks.append(chunk)
                total_duration += duration

            self.db.commit()

            # Update episode
            episode.duration_seconds = total_duration
            episode.status = ProcessingStatus.COMPLETED
            episode.status_message = f"Processed {len(chunks)} chunks from SRT"
            self.db.commit()

            print(f"Done: {len(chunks)} chunks, {total_duration / 60:.1f} minutes")

            return {
                "episode_id": episode.id,
                "episode_number": episode_number,
                "name": name,
                "chunk_count": len(chunks),
                "duration_minutes": round(total_duration / 60, 2),
                "status": "completed"
            }

        except Exception as e:
            return {
                "episode_name": episode_name,
                "episode_number": episode_number,
                "status": "failed",
                "error": str(e)
            }

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _parse_srt(self, srt_path: Path) -> List[dict]:
        subtitles = []

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        blocks = content.strip().split('\n\n')

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            timestamp_line = lines[1]
            text_lines = lines[2:]

            match = re.match(
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                timestamp_line
            )

            if not match:
                continue

            start_sec = (
                    int(match.group(1)) * 3600 +
                    int(match.group(2)) * 60 +
                    int(match.group(3)) +
                    int(match.group(4)) / 1000
            )

            end_sec = (
                    int(match.group(5)) * 3600 +
                    int(match.group(6)) * 60 +
                    int(match.group(7)) +
                    int(match.group(8)) / 1000
            )

            text = ' '.join(text_lines).strip()

            subtitles.append({
                'start': start_sec,
                'end': end_sec,
                'text': text
            })

        return subtitles

    def process_zip_with_srt(
            self,
            zip_file: BinaryIO,
            series_id: int
    ) -> List[dict]:
        temp_dir = self.uploads_dir / f"temp_srt_zip_{series_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            zip_path = temp_dir / "upload.zip"
            with open(zip_path, 'wb') as f:
                shutil.copyfileobj(zip_file, f)

            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)

            video_extensions = {'.mp4', '.mkv', '.avi', '.webm', '.mov', '.mp3', '.wav'}
            video_files = {}
            srt_files = {}

            for f in extract_dir.rglob('*'):
                if f.name.startswith('.') or '__MACOSX' in str(f):
                    continue

                name_without_ext = f.stem
                ext = f.suffix.lower()

                if ext in video_extensions:
                    video_files[name_without_ext] = f
                elif ext == '.srt':
                    srt_files[name_without_ext] = f

            print(f"Found {len(video_files)} videos, {len(srt_files)} SRTs")

            results = []

            for name, video_path in sorted(video_files.items()):
                srt_path = srt_files.get(name)

                if not srt_path:
                    print(f"[{name}] No matching SRT, skipping")
                    results.append({
                        "name": name,
                        "status": "skipped",
                        "error": "No matching SRT file"
                    })
                    continue

                match = re.search(r'(\d+)', name)
                ep_num = int(match.group(1)) if match else None

                print(f"\n[{name}] Processing (Episode {ep_num})...")

                with open(video_path, 'rb') as vf, open(srt_path, 'rb') as sf_file:
                    result = self.process_episode_with_srt(
                        video_file=vf,
                        srt_file=sf_file,
                        video_filename=video_path.name,
                        srt_filename=srt_path.name,
                        series_id=series_id,
                        episode_name=name,
                        episode_number=ep_num
                    )
                    results.append(result)

                print(f"  Done: {result.get('chunk_count', 0)} chunks")

            return results

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _process_video_to_chunks(
            self,
            video_file: BinaryIO,
            video_filename: str,
            episode: Episode,
            max_chunk_sec: Optional[int] = None,
    ) -> List[Chunk]:
        max_sec = max_chunk_sec or self.max_chunk_sec

        episode.status_message = "Saving video"
        self.db.commit()

        temp_video_path = self.uploads_dir / f"temp_{episode.id}_{video_filename}"
        with open(temp_video_path, 'wb') as f:
            shutil.copyfileobj(video_file, f)

        episode.status_message = "Processing audio"
        self.db.commit()

        audio_dir = self.audio_dir / f"episode_{episode.id}"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / "audio.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path)
        ]

        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            temp_video_path.unlink(missing_ok=True)
            raise IngestionError(f"Audio extraction failed: {result.stderr.decode()}")

        temp_video_path.unlink(missing_ok=True)

        episode.audio_file = str(audio_path)
        episode.duration_seconds = self._get_duration(audio_path)
        self.db.commit()

        episode.status_message = "Detecting silence"
        self.db.commit()

        silence_starts, silence_ends = self._detect_silence(audio_path)

        episode.status_message = f"Found {len(silence_starts)} silence points"
        self.db.commit()

        total_duration = episode.duration_seconds
        boundaries = self._calc_chunk_boundaries(silence_ends, total_duration, max_sec)

        episode.status_message = f"Creating {len(boundaries)} chunks"
        self.db.commit()

        self.db.query(Chunk).filter(Chunk.episode_id == episode.id).delete()
        self.db.commit()

        chunks = []
        chunks_dir = self.chunks_dir / f"episode_{episode.id}"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for idx, (start, end) in enumerate(boundaries):
            duration = end - start
            filename = f"chunk_{idx:04d}.wav"
            chunk_path = chunks_dir / filename

            if self._extract_chunks(audio_path, chunk_path, start, duration):
                chunk = Chunk(
                    episode_id=episode.id,
                    chunk_index=idx,
                    filename=filename,
                    file_path=str(chunk_path),
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    duration=round(duration, 3),
                    is_cleaned=False,
                    transcription=None
                )
                self.db.add(chunk)
                chunks.append(chunk)

        self.db.commit()

        episode.status = ProcessingStatus.COMPLETED
        episode.status_message = "Processing completed"
        self.db.commit()

        return chunks

    def _get_duration(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except:
            return 0.0

    def _detect_silence(self, audio_path: Path) -> Tuple[List[float], List[float]]:
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-af", f"silencedetect=noise={self.silence_threshold_db}dB:d={self.min_silence_duration}",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr

        starts, ends = [], []
        for line in output.split('\n'):
            if 'silence_start' in line:
                try:
                    t = float(line.split('silence_start: ')[1].split()[0])
                    starts.append(t)
                except:
                    pass
            elif 'silence_end' in line:
                try:
                    t = float(line.split('silence_end: ')[1].split()[0])
                    ends.append(t)
                except:
                    pass

        return starts, ends

    def _calc_chunk_boundaries(
            self,
            silence_ends: List[float],
            total_duration: float,
            max_sec: int,
    ) -> List[Tuple[float, float]]:
        chunks = []
        current_start = 0.0

        for silence_end in silence_ends:
            gap = silence_end - current_start
            if gap <= max_sec and gap > 0.5:
                chunks.append((current_start, silence_end))
                current_start = silence_end
            elif gap > max_sec:
                while silence_end - current_start > max_sec:
                    chunks.append((current_start, current_start + max_sec))
                    current_start += max_sec

                if silence_end - current_start > 0.5:
                    chunks.append((current_start, silence_end))
                    current_start = silence_end

        remaining = total_duration - current_start
        if remaining <= max_sec and remaining > 0.5:
            chunks.append((current_start, total_duration))
        elif remaining > max_sec:
            while total_duration - current_start > max_sec:
                chunks.append((current_start, current_start + max_sec))
                current_start += max_sec
            if total_duration - current_start > 0.5:
                chunks.append((current_start, total_duration))

        return chunks

    def _extract_chunks(self, audio_path: Path, output_path: Path, start: float, duration: float) -> bool:
        cmd = [
            "ffmpeg",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(audio_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def delete_series_files(self, series_id: int):
        """Delete all files for a series."""
        episodes = self.db.query(Episode).filter(Episode.series_id == series_id).all()

        for episode in episodes:
            chunks_dir = self.chunks_dir / f"episode_{episode.id}"
            if chunks_dir.exists():
                shutil.rmtree(chunks_dir)

            audio_dir = self.audio_dir / f"episode_{episode.id}"
            if audio_dir.exists():
                shutil.rmtree(audio_dir)

        print(f"Deleted files for series {series_id}")

    def delete_episode_files(self, episode_id: int):
        """Delete all files for an episode."""
        chunks_dir = self.chunks_dir / f"episode_{episode_id}"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)

        audio_dir = self.audio_dir / f"episode_{episode_id}"
        if audio_dir.exists():
            shutil.rmtree(audio_dir)

        print(f"Deleted files for episode {episode_id}")







































