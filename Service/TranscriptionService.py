import torch
import whisper
import Levenshtein
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

from sqlalchemy.orm import Session
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoProcessor,
    SeamlessM4Tv2ForSpeechToText,
)
from peft import PeftModel
import torchaudio

from Training.Model import Episode, Chunk

# =============================================================================
# Arabic text normalization (for fair comparison between model outputs)
# =============================================================================
DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison purposes.
    Strips diacritics, normalizes alef variants, removes tatweel."""
    if not text:
        return ""
    # Strip diacritics
    text = "".join(c for c in text if c not in DIACRITICS)
    # Normalize alef variants -> bare alef
    for alef in ("أ", "إ", "آ", "ٱ"):
        text = text.replace(alef, "ا")
    # Remove tatweel (kashida)
    text = text.replace("ـ", "")
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


# =============================================================================
# Individual model transcribers
# =============================================================================
class WhisperTranscriber:
    """OpenAI Whisper (any size) via the openai-whisper package."""

    def __init__(self, model_name: str = "large-v3", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.name = f"whisper-{model_name}"

    def load(self):
        if self.model is None:
            print(f"  Loading {self.name}...")
            self.model = whisper.load_model(self.model_name)
        return self

    def transcribe(self, audio_path: str, language: str = "ar") -> str:
        result = self.model.transcribe(audio_path, language=language)
        return result["text"].strip()

    def unload(self):
        del self.model
        self.model = None
        torch.cuda.empty_cache()


class HFWhisperTranscriber:
    """HuggingFace Whisper (for LoRA checkpoints or different variants)."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        lora_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.lora_path = lora_path
        self.device = device
        self.model = None
        self.processor = None
        self.name = f"hf-whisper-{model_name.split('/')[-1]}"
        if lora_path:
            self.name += f"+lora"

    def load(self):
        if self.model is None:
            print(f"  Loading {self.name}...")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            base = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(base, self.lora_path)
            else:
                self.model = base
            self.model = self.model.to(self.device)
            self.model.eval()
        return self

    def transcribe(self, audio_path: str, language: str = "ar") -> str:
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            sr = 16000
        # Mix to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)

        inputs = self.processor(
            wav.numpy(), sampling_rate=sr, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            pred_ids = self.model.generate(
                inputs.input_features, language=language, task="transcribe"
            )
        text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        return text.strip()

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()


class SeamlessTranscriber:
    """Meta SeamlessM4T v2 for ASR."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.name = "seamless-m4t-v2"

    def load(self):
        if self.model is None:
            print(f"  Loading {self.name}...")
            self.processor = AutoProcessor.from_pretrained(
                "facebook/seamless-m4t-v2-large"
            )
            self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                "facebook/seamless-m4t-v2-large"
            ).to(self.device)
            self.model.eval()
        return self

    def transcribe(self, audio_path: str, language: str = "ar") -> str:
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            sr = 16000
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)

        inputs = self.processor(
            audios=wav.numpy(), sampling_rate=sr, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs, tgt_lang="arb"
            )
        text = self.processor.batch_decode(
            output_tokens[0], skip_special_tokens=True
        )[0]
        return text.strip()

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()


# =============================================================================
# Consensus engine
# =============================================================================
class ConsensusEngine:
    """Compare multiple transcriptions and decide if they agree enough."""

    def __init__(
        self,
        min_agreement_ratio: float = 0.6,
        max_char_distance_ratio: float = 0.4,
        min_models_agree: int = 2,
    ):
        # Levenshtein distance / max_len must be BELOW this to count as "agreeing"
        self.max_char_distance_ratio = max_char_distance_ratio
        # Fraction of model pairs that must agree
        self.min_agreement_ratio = min_agreement_ratio
        # Minimum number of models whose outputs are close to the best
        self.min_models_agree = min_models_agree

    def evaluate(
        self, transcriptions: Dict[str, str]
    ) -> Dict:
        """
        Given {model_name: raw_transcription}, compute consensus.

        Returns dict with:
          - accepted: bool
          - best_transcription: str (the one closest to all others)
          - confidence: float (0-1, how much agreement)
          - distances: dict of pairwise distances
          - reject_reason: str or None
        """
        names = list(transcriptions.keys())
        texts = {k: normalize_arabic(v) for k, v in transcriptions.items()}
        n = len(names)

        # Need at least 2 models
        if n < 2:
            return {
                "accepted": False,
                "best_transcription": list(transcriptions.values())[0] if n == 1 else "",
                "confidence": 0.0,
                "distances": {},
                "reject_reason": "fewer_than_2_models",
            }

        # Filter out empty transcriptions (hallucination to nothing)
        non_empty = {k: v for k, v in texts.items() if len(v) > 0}
        if len(non_empty) < 2:
            return {
                "accepted": False,
                "best_transcription": "",
                "confidence": 0.0,
                "distances": {},
                "reject_reason": "too_many_empty_transcriptions",
            }

        # Compute pairwise Levenshtein distance ratios
        pairwise = {}
        agreement_count = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                t1 = texts[names[i]]
                t2 = texts[names[j]]
                if not t1 and not t2:
                    ratio = 0.0  # Both empty = agree
                elif not t1 or not t2:
                    ratio = 1.0  # One empty, one not = disagree
                else:
                    dist = Levenshtein.distance(t1, t2)
                    max_len = max(len(t1), len(t2))
                    ratio = dist / max_len

                pair_key = f"{names[i]}_vs_{names[j]}"
                pairwise[pair_key] = round(ratio, 4)

                if ratio <= self.max_char_distance_ratio:
                    agreement_count += 1
                total_pairs += 1

        agreement_ratio = agreement_count / total_pairs if total_pairs > 0 else 0

        # Find the transcription closest to all others (lowest average distance)
        avg_distances = {}
        for i, name_i in enumerate(names):
            dists = []
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                t1 = texts[name_i]
                t2 = texts[name_j]
                if not t1 and not t2:
                    dists.append(0.0)
                elif not t1 or not t2:
                    dists.append(1.0)
                else:
                    d = Levenshtein.distance(t1, t2)
                    dists.append(d / max(len(t1), len(t2)))
            avg_distances[name_i] = np.mean(dists) if dists else 1.0

        best_model = min(avg_distances, key=avg_distances.get)
        best_text = transcriptions[best_model]  # Return raw (not normalized)

        # Count how many models are "close" to the best
        close_count = 1  # The best model itself
        best_norm = texts[best_model]
        for name in names:
            if name == best_model:
                continue
            t = texts[name]
            if not best_norm and not t:
                close_count += 1
            elif not best_norm or not t:
                continue
            else:
                d = Levenshtein.distance(best_norm, t)
                ratio = d / max(len(best_norm), len(t))
                if ratio <= self.max_char_distance_ratio:
                    close_count += 1

        # Decision
        accepted = (
            agreement_ratio >= self.min_agreement_ratio
            and close_count >= self.min_models_agree
        )

        reject_reason = None
        if not accepted:
            if agreement_ratio < self.min_agreement_ratio:
                reject_reason = f"low_agreement({agreement_ratio:.2f}<{self.min_agreement_ratio})"
            elif close_count < self.min_models_agree:
                reject_reason = f"insufficient_model_agreement({close_count}<{self.min_models_agree})"

        confidence = agreement_ratio * (close_count / n)

        return {
            "accepted": accepted,
            "best_transcription": best_text,
            "best_model": best_model,
            "confidence": round(confidence, 4),
            "agreement_ratio": round(agreement_ratio, 4),
            "close_count": close_count,
            "distances": pairwise,
            "reject_reason": reject_reason,
        }


# =============================================================================
# Main transcription service with consensus
# =============================================================================
class TranscriptionService:
    def __init__(
        self,
        db: Session,
        models: Optional[List] = None,
        consensus_config: Optional[Dict] = None,
        use_consensus: bool = True,
    ):
        self.db = db
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_consensus = use_consensus

        # Default: just Whisper Large V3 (backward compatible)
        if models is None:
            self.transcribers = [WhisperTranscriber("large-v3", self.device)]
        else:
            self.transcribers = models

        # Consensus engine
        if consensus_config is None:
            consensus_config = {}
        self.consensus = ConsensusEngine(**consensus_config)

    def load_models(self):
        """Load all models into memory."""
        print(f"\nLoading {len(self.transcribers)} transcription models...")
        for t in self.transcribers:
            t.load()
        print("All models loaded.\n")

    def unload_models(self):
        """Free all model memory."""
        for t in self.transcribers:
            t.unload()

    def transcribe_episode(
        self,
        episode_id: int,
        language: str = "ar",
        overwrite: bool = False,
    ) -> dict:
        self.load_models()

        episode = self.db.query(Episode).filter(Episode.id == episode_id).first()
        if not episode:
            raise ValueError(f"Episode {episode_id} not found")

        chunks = (
            self.db.query(Chunk)
            .filter(Chunk.episode_id == episode_id)
            .order_by(Chunk.chunk_index)
            .all()
        )

        stats = {
            "total": 0,
            "transcribed": 0,
            "skipped": 0,
            "failed": 0,
            "filtered": 0,
        }

        print(f"\nTranscribing Episode {episode_id}: {episode.name}")
        print(f"  {len(chunks)} chunks, {len(self.transcribers)} models")
        print(f"  Consensus: {'ON' if self.use_consensus else 'OFF'}")

        for chunk in tqdm(chunks, desc=f"Ep {episode_id}"):
            stats["total"] += 1

            if chunk.transcription and not overwrite:
                stats["skipped"] += 1
                continue

            if not Path(chunk.file_path).exists():
                stats["failed"] += 1
                continue

            try:
                result = self._transcribe_chunk(chunk, language)
                if result["accepted"]:
                    stats["transcribed"] += 1
                else:
                    stats["filtered"] += 1
            except Exception as e:
                print(f"  Failed {chunk.filename}: {e}")
                stats["failed"] += 1

        self.db.commit()
        print(f"  Done: {stats}")
        return stats

    def transcribe_series(
        self,
        series_id: int,
        language: str = "ar",
        overwrite: bool = False,
    ) -> dict:
        episodes = self.db.query(Episode).filter(Episode.series_id == series_id).all()

        total_stats = {
            "total": 0,
            "transcribed": 0,
            "skipped": 0,
            "failed": 0,
            "filtered": 0,
            "episodes": 0,
        }

        for episode in episodes:
            stats = self.transcribe_episode(episode.id, language, overwrite)
            for k in ["total", "transcribed", "skipped", "failed", "filtered"]:
                total_stats[k] += stats[k]
            total_stats["episodes"] += 1

        return total_stats

    def _transcribe_chunk(self, chunk: Chunk, language: str) -> Dict:
        """
        Core method: run all models, apply consensus, update DB.
        """
        if not self.use_consensus or len(self.transcribers) == 1:
            # Single model mode (backward compatible)
            text = self.transcribers[0].transcribe(chunk.file_path, language)
            chunk.transcription = text
            chunk.is_cleaned = False
            chunk.was_filtered = False
            chunk.source = self.transcribers[0].name
            self.db.flush()
            return {"accepted": True, "transcription": text}

        # --- Multi-model consensus mode ---
        transcriptions = {}
        for transcriber in self.transcribers:
            try:
                text = transcriber.transcribe(chunk.file_path, language)
                transcriptions[transcriber.name] = text
            except Exception as e:
                print(f"    {transcriber.name} failed on {chunk.filename}: {e}")
                continue

        if not transcriptions:
            chunk.was_filtered = True
            chunk.filter_reason = "all_models_failed"
            self.db.flush()
            return {"accepted": False, "transcription": ""}

        # Run consensus
        result = self.consensus.evaluate(transcriptions)

        if result["accepted"]:
            chunk.transcription = result["best_transcription"]
            chunk.is_cleaned = False
            chunk.was_filtered = False
            chunk.source = f"consensus:{result['best_model']}"
            # Store confidence as part of original_transcription for debugging
            chunk.original_transcription = (
                f"confidence={result['confidence']} "
                f"agreement={result['agreement_ratio']} "
                f"models={result['close_count']}/{len(transcriptions)}"
            )
        else:
            # Rejected — mark as filtered, save all outputs for review
            chunk.was_filtered = True
            chunk.filter_reason = result["reject_reason"]
            # Store all model outputs so you can review later
            all_outputs = " | ".join(
                f"[{k}]: {v}" for k, v in transcriptions.items()
            )
            chunk.original_transcription = all_outputs
            chunk.transcription = None  # Don't save bad transcription

        self.db.flush()
        return {
            "accepted": result["accepted"],
            "transcription": result.get("best_transcription", ""),
            **result,
        }


# =============================================================================
# Factory functions for common configurations
# =============================================================================
def get_single_model_service(
    db: Session, model_name: str = "large-v3"
) -> TranscriptionService:
    """Backward compatible: single Whisper model, no consensus."""
    return TranscriptionService(
        db=db,
        models=[WhisperTranscriber(model_name)],
        use_consensus=False,
    )


def get_consensus_service(
    db: Session,
    lora_path: Optional[str] = None,
    include_seamless: bool = False,
    max_char_distance_ratio: float = 0.4,
    min_agreement_ratio: float = 0.6,
    min_models_agree: int = 2,
) -> TranscriptionService:
    """
    Multi-model consensus transcription.

    Default setup (2 models, works on RTX 3050 4GB):
      - Whisper Large V3 (via openai-whisper, sequential)
      - HF Whisper Small + your LoRA

    With include_seamless=True (needs more VRAM):
      - + SeamlessM4T v2

    Models are loaded one at a time to fit in VRAM.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = [
        WhisperTranscriber("large-v3", device),
    ]

    if lora_path:
        models.append(
            HFWhisperTranscriber(
                model_name="openai/whisper-small",
                lora_path=lora_path,
                device=device,
            )
        )
    else:
        # If no LoRA, use base Whisper Small as second opinion
        models.append(
            HFWhisperTranscriber(
                model_name="openai/whisper-small",
                device=device,
            )
        )

    if include_seamless:
        models.append(SeamlessTranscriber(device))

    return TranscriptionService(
        db=db,
        models=models,
        use_consensus=True,
        consensus_config={
            "max_char_distance_ratio": max_char_distance_ratio,
            "min_agreement_ratio": min_agreement_ratio,
            "min_models_agree": min_models_agree,
        },
    )