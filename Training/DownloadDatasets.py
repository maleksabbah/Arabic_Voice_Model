"""
Dataset Download & Preprocessing Pipeline
for w2v-BERT 2.0 Arabic Dialect ASR Probe

Training: MGB-2, MGB-3, Casablanca train
Evaluation: Casablanca test, MGB-2 test, FLEURS Arabic

All audio resampled to 16kHz mono.
Output: HuggingFace Dataset format on disk.
"""

import os
import torch
import torchaudio
from datasets import load_dataset, Audio, concatenate_datasets, DatasetDict
from huggingface_hub import login
import re
import unicodedata

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
    print("No HF_TOKEN set — will only work for non-gated datasets")
    print("Set it with: export HF_TOKEN=your_token_here")


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
# DOWNLOAD FUNCTIONS
# ============================================================

def download_mgb2():
    """Download MGB-2 dataset (1,200h broadcast Arabic). Requires HF access."""
    print("\n" + "="*60)
    print("Downloading MGB-2...")
    print("="*60)

    try:
        ds = load_dataset("QCRI/mgb2", trust_remote_code=True, token=HF_TOKEN)
        print(f"MGB-2 loaded: {ds}")
        return ds
    except Exception as e:
        print(f"MGB-2 download failed: {e}")
        print("Make sure you have access at https://huggingface.co/datasets/QCRI/mgb2")
        return None


def download_mgb3():
    """Download MGB-3 Egyptian dialect (16h)."""
    print("\n" + "="*60)
    print("Downloading MGB-3 (Egyptian)...")
    print("="*60)

    try:
        ds = load_dataset("MightyStudent/Egyptian-ASR-MGB-3", trust_remote_code=True)
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

    dialects = ["Algeria", "Egypt", "Emirate", "Jordan", "Mauritania", "Morocco", "Palestine", "Yemen"]
    all_val = []
    all_test = []

    for dialect in dialects:
        try:
            ds = load_dataset("UBC-NLP/Casablanca", dialect, trust_remote_code=True)
            print(f"  {dialect}: {ds}")

            # Add dialect column
            if "validation" in ds:
                val = ds["validation"].map(lambda x: {"dialect": dialect})
                all_val.append(val)
            if "test" in ds:
                test = ds["test"].map(lambda x: {"dialect": dialect})
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
        ds = load_dataset("google/fleurs", "ar_eg", split="test", trust_remote_code=True)
        print(f"FLEURS loaded: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"FLEURS download failed: {e}")
        return None


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_dataset(dataset, audio_col="audio", text_col="sentence", name="unknown"):
    """
    Standardize a dataset:
    - Resample audio to 16kHz
    - Normalize Arabic text
    - Remove empty/too-short samples
    """
    print(f"\nPreprocessing {name}...")

    # Try to find the right column names
    cols = dataset.column_names if hasattr(dataset, 'column_names') else []

    # Find audio column
    if audio_col not in cols:
        for candidate in ["audio", "Audio", "speech", "input_values"]:
            if candidate in cols:
                audio_col = candidate
                break

    # Find text column
    if text_col not in cols:
        for candidate in ["sentence", "transcription", "text", "transcript", "Transcription"]:
            if candidate in cols:
                text_col = candidate
                break

    print(f"  Using columns: audio='{audio_col}', text='{text_col}'")
    print(f"  Available columns: {cols}")

    # Cast audio to 16kHz
    dataset = dataset.cast_column(audio_col, Audio(sampling_rate=SAMPLE_RATE))

    # Normalize text and filter
    def process(batch):
        text = batch.get(text_col, "")
        normalized = normalize_arabic(text)
        batch["normalized_text"] = normalized
        return batch

    dataset = dataset.map(process)

    # Filter out empty transcriptions
    before = len(dataset)
    dataset = dataset.filter(lambda x: len(x["normalized_text"].strip()) > 0)
    after = len(dataset)
    print(f"  Filtered: {before} -> {after} samples ({before - after} removed)")

    # Rename columns for consistency
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

    # ---- TRAINING DATA ----
    train_datasets = []

    # 1. MGB-2
    mgb2 = download_mgb2()
    if mgb2 is not None:
        if "train" in mgb2:
            mgb2_train = preprocess_dataset(mgb2["train"], name="MGB-2 train")
            train_datasets.append(("mgb2", mgb2_train))
            # Save MGB-2 test separately for evaluation
            if "test" in mgb2:
                mgb2_test = preprocess_dataset(mgb2["test"], name="MGB-2 test")
                mgb2_test.save_to_disk(os.path.join(OUTPUT_DIR, "eval_mgb2_test"))
                print(f"  Saved MGB-2 test to {OUTPUT_DIR}/eval_mgb2_test")
        else:
            # Some versions have different split names
            print(f"  MGB-2 splits available: {list(mgb2.keys())}")
            for split_name in mgb2.keys():
                print(f"    {split_name}: {len(mgb2[split_name])} samples")

    # 2. MGB-3 Egyptian
    mgb3 = download_mgb3()
    if mgb3 is not None:
        # MGB-3 might have different split structure
        for split in mgb3.keys() if isinstance(mgb3, dict) else ["train"]:
            data = mgb3[split] if isinstance(mgb3, dict) else mgb3
            mgb3_processed = preprocess_dataset(data, name=f"MGB-3 {split}")
            train_datasets.append(("mgb3", mgb3_processed))
            break  # Just get the training data

    # 3. Casablanca (validation split for training, test for eval)
    casablanca = download_casablanca()
    if casablanca is not None:
        if "validation" in casablanca:
            casa_train = preprocess_dataset(
                casablanca["validation"],
                text_col="transcription",
                name="Casablanca train (validation split)"
            )
            train_datasets.append(("casablanca", casa_train))

        if "test" in casablanca:
            casa_test = preprocess_dataset(
                casablanca["test"],
                text_col="transcription",
                name="Casablanca test"
            )
            casa_test.save_to_disk(os.path.join(OUTPUT_DIR, "eval_casablanca_test"))
            print(f"  Saved Casablanca test to {OUTPUT_DIR}/eval_casablanca_test")

    # 4. FLEURS Arabic (eval only)
    fleurs = download_fleurs()
    if fleurs is not None:
        fleurs_processed = preprocess_dataset(fleurs, text_col="transcription", name="FLEURS Arabic")
        fleurs_processed.save_to_disk(os.path.join(OUTPUT_DIR, "eval_fleurs"))
        print(f"  Saved FLEURS to {OUTPUT_DIR}/eval_fleurs")

    # ---- COMBINE TRAINING DATA ----
    if train_datasets:
        print("\n" + "="*60)
        print("Combining training datasets...")
        print("="*60)

        # Standardize columns before concatenation
        standardized = []
        for name, ds in train_datasets:
            # Keep only audio and normalized_text
            keep_cols = ["audio", "normalized_text"]
            if "dialect" in ds.column_names:
                keep_cols.append("dialect")

            remove_cols = [c for c in ds.column_names if c not in keep_cols]
            if remove_cols:
                ds = ds.remove_columns(remove_cols)

            # Add source tag
            ds = ds.map(lambda x: {"source": name})

            print(f"  {name}: {len(ds)} samples")
            standardized.append(ds)

        combined = concatenate_datasets(standardized)
        combined = combined.shuffle(seed=42)

        print(f"\nTotal training samples: {len(combined)}")
        combined.save_to_disk(os.path.join(OUTPUT_DIR, "train_combined"))
        print(f"Saved combined training data to {OUTPUT_DIR}/train_combined")
    else:
        print("\nNo training datasets were loaded successfully!")

    # ---- SUMMARY ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for folder in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, folder)
        if os.path.isdir(path):
            print(f"  {folder}/")

    print("\nDone! Next step: run the probe training script.")
    print("  python train_probe.py")


if __name__ == "__main__":
    main()
