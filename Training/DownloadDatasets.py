"""
Dataset Download & Preprocessing Pipeline
for w2v-BERT 2.0 Arabic Dialect ASR Probe

Training: MGB-2 (train), MGB-3 (Egyptian), Casablanca (val split)
Evaluation: Casablanca (test), MGB-2 (test), FLEURS Arabic

MGB-2 is Kaldi format (tar.gz with segments/text/wav files).
Other datasets load via HuggingFace datasets library.
All audio resampled to 16kHz mono.
"""

import os
import tarfile
import soundfile as sf
import numpy as np
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from huggingface_hub import login, hf_hub_download
from datasets import load_dataset, Audio, concatenate_datasets, Dataset, DatasetDict

# ============================================================
# CONFIG
# ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN", None)
OUTPUT_DIR = "/workspace/datasets"
SAMPLE_RATE = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Logged into HuggingFace")
else:
    print("WARNING: No HF_TOKEN set — MGB-2 will fail")
    print("Set with: export HF_TOKEN=your_token_here")


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_arabic(text):
    """Normalize Arabic text for CTC training."""
    if not text or not isinstance(text, str):
        return ""
    # Remove diacritics (tashkeel)
    text = re.sub(r'[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]', '', text)
    # Normalize alef variants to bare alef
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize teh marbuta to heh
    text = text.replace('ة', 'ه')
    # Normalize alef maksura to yeh
    text = text.replace('ى', 'ي')
    # Remove tatweel
    text = text.replace('ـ', '')
    # Remove non-Arabic, non-space characters (keep Arabic letters + spaces)
    text = re.sub(r'[^ء-ي٠-٩\s]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# MGB-2: Kaldi format via hf_hub_download
# ============================================================
MGB2_REPO = "QCRI/mgb2"

def download_mgb2_split(split_name):
    """
    Download and extract an MGB-2 split (train or test).
    Returns the extracted directory path.
    """
    extract_dir = os.path.join(OUTPUT_DIR, "mgb2_raw")
    split_dir = os.path.join(extract_dir, split_name)

    # Check if already extracted (train ships XML alignments, not Kaldi segments)
    marker = os.path.join(split_dir, "segments.non_overlap_speech") if split_name == "test" else os.path.join(split_dir, "xml", "utf8")
    if os.path.exists(marker):
        print(f"  MGB-2 {split_name} already extracted at {split_dir}")
        return split_dir

    os.makedirs(extract_dir, exist_ok=True)

    print(f"  Downloading MGB-2 {split_name}.tar.gz ...")
    try:
        tar_path = hf_hub_download(
            MGB2_REPO,
            f"{split_name}/{split_name}.tar.gz",
            repo_type="dataset",
            cache_dir=os.path.join(extract_dir, "hf_cache"),
            token=HF_TOKEN,
        )
    except Exception as e:
        print(f"  Failed to download {split_name}.tar.gz: {e}")
        # Try without subdirectory
        try:
            tar_path = hf_hub_download(
                MGB2_REPO,
                f"{split_name}.tar.gz",
                repo_type="dataset",
                cache_dir=os.path.join(extract_dir, "hf_cache"),
                token=HF_TOKEN,
            )
        except Exception as e2:
            print(f"  Also failed: {e2}")
            return None

    print(f"  Extracting to {extract_dir}/ ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            tf.extract(member, extract_dir, set_attrs=False)

    return split_dir


def load_mgb2_segments(mgb2_dir, split_type="test"):
    """
    Load MGB-2 segments from Kaldi-format files.
    Returns list of (utt_id, recording_id, start_s, end_s, transcript).
    """
    # Determine file names based on split
    if split_type == "test":
        text_file = os.path.join(mgb2_dir, "text.non_overlap_speech")
        segments_file = os.path.join(mgb2_dir, "segments.non_overlap_speech")
    else:
        text_file = os.path.join(mgb2_dir, "text")
        segments_file = os.path.join(mgb2_dir, "segments")

    if not os.path.exists(text_file) or not os.path.exists(segments_file):
        print(f"  Missing files: text={os.path.exists(text_file)}, segments={os.path.exists(segments_file)}")
        # List what's actually in the directory
        if os.path.exists(mgb2_dir):
            print(f"  Contents of {mgb2_dir}: {os.listdir(mgb2_dir)}")
        return []

    # Load texts
    texts = {}
    with open(text_file, encoding="utf-8") as f:
        for line in f:
            utt_id, _, transcript = line.partition(" ")
            transcript = transcript.strip()
            if transcript:
                texts[utt_id] = transcript

    # Load segments
    segments = []
    with open(segments_file, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            utt_id, rec_id, start, end = parts
            if utt_id in texts:
                segments.append((utt_id, rec_id, float(start), float(end), texts[utt_id]))

    print(f"  Loaded {len(segments)} segments from {mgb2_dir}")
    return segments


# Same cut as the Kaldi MGB-2 recipe's train_mer80: drops only badly aligned segments
MGB2_MAX_WMER = 80.0

def load_mgb2_xml_segments(mgb2_dir, max_wmer=MGB2_MAX_WMER):
    """
    Load MGB-2 train segments from the aligned XML transcripts (xml/utf8/*.xml).
    Returns list of (utt_id, recording_id, start_s, end_s, transcript).
    """
    xml_dir = os.path.join(mgb2_dir, "xml", "utf8")
    if not os.path.isdir(xml_dir):
        print(f"  Missing {xml_dir}")
        return []

    segments = []
    dropped = 0
    for xml_file in sorted(Path(xml_dir).glob("*.xml")):
        rec_id = xml_file.stem
        root = ET.parse(xml_file).getroot()
        for seg in root.iter("segment"):
            if float(seg.get("WMER", 0)) > max_wmer:
                dropped += 1
                continue
            words = [el.text for el in seg.iter("element") if el.text]
            if not words:
                continue
            segments.append((
                seg.get("id"), rec_id,
                float(seg.get("starttime")), float(seg.get("endtime")),
                " ".join(words),
            ))

    print(f"  Loaded {len(segments)} segments from {xml_dir} (dropped {dropped} with WMER > {max_wmer})")
    return segments


def mgb2_segments_to_audio_files(mgb2_dir, segments, output_path, max_segments=None):
    """
    Extract audio segments from MGB-2 wav files and save as individual files.
    Creates a manifest JSON for later loading.
    """
    os.makedirs(output_path, exist_ok=True)
    manifest = []

    # Find wav.scp or wav files
    wav_scp = os.path.join(mgb2_dir, "wav.scp")
    wav_paths = {}

    if os.path.exists(wav_scp):
        with open(wav_scp) as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    # wav.scp paths are relative to the split dir, not the cwd
                    wav_paths[parts[0]] = os.path.join(mgb2_dir, parts[1])
    else:
        # Look for wav files directly
        for f in Path(mgb2_dir).rglob("*.wav"):
            wav_paths[f.stem] = str(f)

    if not wav_paths:
        print(f"  No wav files found in {mgb2_dir}")
        print(f"  Contents: {os.listdir(mgb2_dir)[:20]}")
        return []

    print(f"  Found {len(wav_paths)} recordings")

    if max_segments:
        segments = segments[:max_segments]

    loaded_recordings = {}
    skipped = 0

    for i, (utt_id, rec_id, start, end, transcript) in enumerate(segments):
        if rec_id not in wav_paths:
            skipped += 1
            continue

        # Test utt ids contain ':', which the RunPod network volume rejects in filenames
        out_file = os.path.join(output_path, f"{utt_id.replace(':', '-')}.wav")
        normalized = normalize_arabic(transcript)

        # Resume: segment already written by an earlier run
        if os.path.exists(out_file) and end - start >= 0.1:
            if normalized:
                manifest.append({"audio_path": out_file, "text": normalized, "utt_id": utt_id, "duration": end - start})
            continue

        # Load recording (cache it)
        if rec_id not in loaded_recordings:
            wav_path = wav_paths[rec_id]
            try:
                audio, sr = sf.read(wav_path, dtype="float32")
                if sr != SAMPLE_RATE:
                    import torchaudio
                    import torch
                    audio_t = torch.tensor(audio).float()
                    if audio_t.dim() == 1:
                        audio_t = audio_t.unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                    audio_t = resampler(audio_t)
                    audio = audio_t.squeeze().numpy()
                    sr = SAMPLE_RATE
                loaded_recordings[rec_id] = (audio, sr)
            except Exception as e:
                print(f"  Error loading {wav_path}: {e}")
                skipped += 1
                continue

        audio, sr = loaded_recordings[rec_id]

        # Extract segment
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) < sr * 0.1:  # Skip segments < 0.1s
            skipped += 1
            continue

        # Save segment
        sf.write(out_file, segment_audio, sr)

        if normalized:
            manifest.append({
                "audio_path": out_file,
                "text": normalized,
                "utt_id": utt_id,
                "duration": end - start,
            })

        if (i + 1) % 5000 == 0:
            print(f"    Processed {i+1}/{len(segments)} segments...")

        # Segments come grouped by recording, so a tiny cache suffices; the pod is capped at 8 GB RAM
        if len(loaded_recordings) > 2:
            oldest = list(loaded_recordings.keys())[0]
            del loaded_recordings[oldest]

    # Save manifest
    manifest_path = os.path.join(output_path, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(manifest)} segments to {output_path} (skipped {skipped})")
    return manifest


def manifest_to_dataset(manifest_path):
    """Convert a manifest JSON to a HuggingFace Dataset."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    ds = Dataset.from_dict({
        "audio": [item["audio_path"] for item in manifest],
        "normalized_text": [item["text"] for item in manifest],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return ds


# ============================================================
# OTHER DATASETS: HuggingFace
# ============================================================

def download_mgb3():
    """Download MGB-3 Egyptian dialect (16h)."""
    print("\n" + "="*60)
    print("Downloading MGB-3 (Egyptian)...")
    print("="*60)
    try:
        ds = load_dataset("MightyStudent/Egyptian-ASR-MGB-3")
        print(f"MGB-3 loaded: {ds}")
        return ds
    except Exception as e:
        print(f"MGB-3 download failed: {e}")
        return None


def download_casablanca():
    """Download Casablanca multi-dialect dataset (8 dialects)."""
    print("\n" + "="*60)
    print("Downloading Casablanca (8 dialects)...")
    print("="*60)

    dialects = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]
    all_val = []
    all_test = []

    for dialect in dialects:
        try:
            ds = load_dataset("UBC-NLP/Casablanca", dialect)
            print(f"  {dialect}: {ds}")

            if "validation" in ds:
                val = ds["validation"].map(lambda x, d=dialect: {"dialect": d})
                all_val.append(val)
            if "test" in ds:
                test = ds["test"].map(lambda x, d=dialect: {"dialect": d})
                all_test.append(test)
        except Exception as e:
            print(f"  {dialect} failed: {e}")

    result = DatasetDict()
    if all_val:
        result["validation"] = concatenate_datasets(all_val)
    if all_test:
        result["test"] = concatenate_datasets(all_test)

    print(f"Casablanca combined: {result}")
    return result


def download_fleurs():
    """Download FLEURS Arabic for MSA evaluation."""
    print("\n" + "="*60)
    print("Downloading FLEURS Arabic...")
    print("="*60)
    try:
        ds = load_dataset("google/fleurs", "ar_eg", split="test")
        print(f"FLEURS loaded: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"FLEURS download failed: {e}")
        return None


# ============================================================
# PREPROCESSING (for HuggingFace datasets)
# ============================================================

def preprocess_dataset(dataset, audio_col="audio", text_col="sentence", name="unknown"):
    """Standardize a HF dataset: resample, normalize text, filter empty."""
    print(f"\nPreprocessing {name}...")

    cols = dataset.column_names if hasattr(dataset, 'column_names') else []

    # Find audio column
    if audio_col not in cols:
        for candidate in ["audio", "Audio", "speech", "input_values"]:
            if candidate in cols:
                audio_col = candidate
                break

    # Find text column
    if text_col not in cols:
        for candidate in ["sentence", "transcription", "text", "transcript"]:
            if candidate in cols:
                text_col = candidate
                break

    print(f"  Using columns: audio='{audio_col}', text='{text_col}'")

    # Cast audio to 16kHz
    dataset = dataset.cast_column(audio_col, Audio(sampling_rate=SAMPLE_RATE))

    # Normalize text
    def process(batch):
        batch["normalized_text"] = normalize_arabic(batch.get(text_col, ""))
        return batch

    dataset = dataset.map(process)

    # Filter empty
    before = len(dataset)
    dataset = dataset.filter(lambda x: len(x["normalized_text"].strip()) > 0)
    after = len(dataset)
    print(f"  Filtered: {before} -> {after} ({before - after} removed)")

    if audio_col != "audio":
        dataset = dataset.rename_column(audio_col, "audio")

    return dataset


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("Arabic Dialect ASR Dataset Pipeline")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Sample rate: {SAMPLE_RATE}Hz")
    print("="*60)

    train_datasets = []

    # ---- 1. MGB-2 (Kaldi format) ----
    print("\n" + "="*60)
    print("Processing MGB-2...")
    print("="*60)

    # Train split
    mgb2_train_dir = download_mgb2_split("train")
    if mgb2_train_dir:
        segments = load_mgb2_xml_segments(mgb2_train_dir)
        if segments:
            train_audio_dir = os.path.join(OUTPUT_DIR, "mgb2_train_audio")
            manifest = mgb2_segments_to_audio_files(
                mgb2_train_dir, segments, train_audio_dir
            )
            if manifest:
                mgb2_train_ds = manifest_to_dataset(
                    os.path.join(train_audio_dir, "manifest.json")
                )
                train_datasets.append(("mgb2", mgb2_train_ds))
                print(f"  MGB-2 train: {len(mgb2_train_ds)} samples")

    # Test split (for eval)
    mgb2_test_dir = download_mgb2_split("test")
    if mgb2_test_dir:
        segments = load_mgb2_segments(mgb2_test_dir, split_type="test")
        if segments:
            test_audio_dir = os.path.join(OUTPUT_DIR, "mgb2_test_audio")
            manifest = mgb2_segments_to_audio_files(
                mgb2_test_dir, segments, test_audio_dir
            )
            if manifest:
                mgb2_test_ds = manifest_to_dataset(
                    os.path.join(test_audio_dir, "manifest.json")
                )
                mgb2_test_ds.save_to_disk(os.path.join(OUTPUT_DIR, "eval_mgb2_test"))
                print(f"  MGB-2 test saved: {len(mgb2_test_ds)} samples")

    # ---- 2. MGB-3 Egyptian ----
    mgb3 = download_mgb3()
    if mgb3 is not None:
        split_name = list(mgb3.keys())[0] if isinstance(mgb3, DatasetDict) else "train"
        data = mgb3[split_name] if isinstance(mgb3, DatasetDict) else mgb3
        mgb3_processed = preprocess_dataset(data, name="MGB-3")
        train_datasets.append(("mgb3", mgb3_processed))

    # ---- 3. Casablanca ----
    casablanca = download_casablanca()
    if casablanca is not None:
        if "validation" in casablanca:
            casa_train = preprocess_dataset(
                casablanca["validation"],
                text_col="transcription",
                name="Casablanca train (val split)"
            )
            train_datasets.append(("casablanca", casa_train))

        if "test" in casablanca:
            casa_test = preprocess_dataset(
                casablanca["test"],
                text_col="transcription",
                name="Casablanca test"
            )
            casa_test.save_to_disk(os.path.join(OUTPUT_DIR, "eval_casablanca_test"))
            print(f"  Casablanca test saved")

    # ---- 4. FLEURS (eval only) ----
    fleurs = download_fleurs()
    if fleurs is not None:
        fleurs_processed = preprocess_dataset(fleurs, text_col="transcription", name="FLEURS")
        fleurs_processed.save_to_disk(os.path.join(OUTPUT_DIR, "eval_fleurs"))
        print(f"  FLEURS saved")

    # ---- COMBINE TRAINING DATA ----
    if train_datasets:
        print("\n" + "="*60)
        print("Combining training datasets...")
        print("="*60)

        standardized = []
        for name, ds in train_datasets:
            keep_cols = ["audio", "normalized_text"]
            if "dialect" in ds.column_names:
                keep_cols.append("dialect")

            remove_cols = [c for c in ds.column_names if c not in keep_cols]
            if remove_cols:
                ds = ds.remove_columns(remove_cols)

            # add_column, not map: map would decode every row's audio just to tag it
            ds = ds.add_column("source", [name] * len(ds))
            print(f"  {name}: {len(ds)} samples")
            standardized.append(ds)

        combined = concatenate_datasets(standardized)
        combined = combined.shuffle(seed=42)

        print(f"\nTotal training samples: {len(combined)}")
        combined.save_to_disk(os.path.join(OUTPUT_DIR, "train_combined"))
        print(f"Saved to {OUTPUT_DIR}/train_combined")
    else:
        print("\nNo training datasets loaded!")

    # ---- SUMMARY ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for folder in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, folder)
        if os.path.isdir(path):
            print(f"  {folder}/")
    print("\nDone. Next: python train_probe.py")


if __name__ == "__main__":
    main()
