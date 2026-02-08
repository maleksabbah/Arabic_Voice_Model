
import re
from typing import Optional, List

from sqlalchemy.orm import Session

from Model import Episode, Chunk, ProcessingStatus


class CleaningService:

    def __init__(self, db: Session):
        self.db = db

        # Arabic Unicode pattern
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')

        # Common Whisper hallucinations
        self.hallucinations = [
            r'^thank you\.?$',
            r'^i don\'?t know\.?$',
            r'^продолжение следует\.{3}$',
            r'^а я умер\.?$',
            r'^salam\.?$',
            r'^oughta accept that\.?$',
            r'^get off!?$',
            r'^yol\.?$',
            r'^ben alayim\.?$',
            r'^\.\.\.[a-z\s]+\.?$',
            r'^ishtariko? fil[- ]?canat?\.?$',
            r'^ishtar[ie]ko? fil[- ]?qanat?\.?$',
            r'^subscribe\.?$',
            r'^like and subscribe\.?$',
            r'^اشترك في القناة\.?$',
            r'^اشتركوا في القناة\.?$',
        ]

        # Music/sound placeholders
        self.music_placeholders = [
            r'^موسيقى$',
            r'^موسيقا$',
            r'^مو+سيقى$',
        ]

        # Translator credits
        self.translator_patterns = [
            r'ترجمة\s+\w+',
            r'نانا\s+محمد',
            r'ترجمة:?\s*',
        ]

        # Short interjections
        self.short_interjections = [
            r'^اه+$',
            r'^ها+$',
            r'^ايه$',
            r'^اي+$',
            r'^اوه+$',
            r'^يا+$',
        ]

        # Banned words (from your training script)
        self.banned_words = ["شكرا", "شكراً", "شكرًا"]

    def clean_episode(self, episode_id: int) -> dict:
        chunks = self.db.query(Chunk).filter(Chunk.episode_id == episode_id).all()

        stats = {"total": 0, "cleaned": 0, "filtered": 0}

        for chunk in chunks:
            if not chunk:
                continue

            stats["total"] += 1
            result = self._clean_transcription(chunk)

            if result["filtered"]:
                stats["filtered"] += 1
            else:
                stats["cleaned"] += 1

        self.db.commit()
        return stats

    def clean_series(self, series_id: int) -> dict:
        episodes = self.db.query(Episode).filter(Episode.series_id == series_id).all()

        total_stats = {"total": 0, "cleaned": 0, "filtered": 0, "episodes": 0}

        for episode in episodes:
            stats = self.clean_episode(episode.id)
            total_stats["total"] += stats["total"]
            total_stats["cleaned"] += stats["cleaned"]
            total_stats["filtered"] += stats["filtered"]
            total_stats["episodes"] += 1
            print(f"Episode {episode.id}: {stats}")

        return total_stats

    def clean_all(self) -> dict:
        transcriptions = self.db.query(Chunk).filter(Chunk.transcription != None).all()

        stats = {"total": 0, "cleaned": 0, "filtered": 0}

        for t in transcriptions:
            stats["total"] += 1
            result = self._clean_transcription(t)

            if result["filtered"]:
                stats["filtered"] += 1
            else:
                stats["cleaned"] += 1

        self.db.commit()
        return stats

    def _clean_transcription(self, chunk: Chunk) -> dict:
        text = chunk.transcription or ""
        text = text.strip()

        # Check if should be filtered
        filter_reason = self._get_filter_reason(text)

        if filter_reason:
            chunk.was_filtered = True
            chunk.filter_reason = filter_reason
            chunk.is_cleaned = True
            return {"filtered": True, "reason": filter_reason}

        # Clean the text
        cleaned = self._clean_text(text)

        #Check cleaned text
        if not cleaned or len(cleaned) < 2:
            chunk.was_filtered = True
            chunk.filter_reason = "too_short_after_cleaning"
            chunk.is_cleaned = True
            return {"filtered": True, "reason": "too_short_after_cleaning"}
        chunk.transcription = cleaned
        chunk.was_filtered = False
        chunk.is_cleaned = True
        return {"filtered": False, "text": cleaned}

    def _get_filter_reason(self, text: str) -> Optional[str]:
        if not text:
            return "empty"

        if self._is_hallucination(text):
            return "hallucination"

        if self._is_music_placeholder(text):
            return "music_placeholder"

        if self._has_translator_credit(text):
            return "translator_credit"

        if self._is_short_interjection(text):
            return "short_interjection"

        if self._is_repetitive(text):
            return "repetitive"

        if not self._is_arabic_text(text):
            return "not_arabic"

        if self._has_banned_word(text):
            return "banned_word"

        if not self._is_valid_arabic(text):
            return "invalid_arabic"

        return None

    def _clean_text(self, text: str) -> str:
        # Remove translator credits
        for pattern in self.translator_patterns:
            text = re.sub(pattern, '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _is_arabic_text(self, text: str) -> bool:
        return bool(self.arabic_pattern.search(text))

    def _is_hallucination(self, text: str) -> bool:
        text_lower = text.lower().strip()
        for pattern in self.hallucinations:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _is_music_placeholder(self, text: str) -> bool:
        text_clean = text.strip()
        for pattern in self.music_placeholders:
            if re.match(pattern, text_clean):
                return True
        return False

    def _has_translator_credit(self, text: str) -> bool:
        for pattern in self.translator_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _is_short_interjection(self, text: str) -> bool:
        """Check if text is just a short interjection."""
        text_clean = text.strip()
        if len(text_clean) <= 3:
            for pattern in self.short_interjections:
                if re.match(pattern, text_clean):
                    return True
        return False

    def _has_banned_word(self, text: str) -> bool:
        """Check if text contains banned words."""
        for word in self.banned_words:
            if word in text:
                return True
        return False

    def _is_repetitive(self, text: str, threshold: int = 2) -> bool:
        """Check if text is overly repetitive."""
        words = text.split()

        if len(words) < 2:
            return False

        # Check 2-3 word texts
        if len(words) <= 3:
            unique_words = set(words)
            if len(unique_words) == 1:
                return True

        # Check exactly 2 identical words
        if len(words) == 2 and words[0] == words[1]:
            return True

        if len(words) < 4:
            return False

        # Check word repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            if count > threshold and len(word) > 1:
                return True

        # Check phrase repetition
        for phrase_len in [2, 3, 4]:
            if len(words) < phrase_len * 2:
                continue

            phrases = {}
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i + phrase_len])
                phrases[phrase] = phrases.get(phrase, 0) + 1

            for phrase, count in phrases.items():
                if count > max(2, threshold // 2):
                    return True

        # Check alternating pattern
        if len(words) >= 6:
            for i in range(len(words) - 5):
                if (words[i] == words[i + 2] == words[i + 4] and
                        words[i + 1] == words[i + 3] == words[i + 5] and
                        words[i] != words[i + 1]):
                    return True

        return False

    def _is_valid_arabic(self, text: str) -> bool:
        allowed_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s،؛؟!.\-]'

        allowed_chars = len([c for c in text if re.match(allowed_pattern, c)])
        allowed_percentage = allowed_chars / len(text) if text else 0

        if allowed_percentage < 0.95:
            return False

        # Must have at least 3 Arabic letters
        arabic_letters = len([c for c in text if
                              re.match(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', c)])

        return arabic_letters >= 3

    def get_stats(self, series_id: Optional[int] = None) -> dict:
        """Get cleaning statistics."""
        query = self.db.query(Chunk)

        if series_id:
            query = query.join(Episode).filter(Episode.series_id == series_id)

        chunks = query.all()

        total = len(chunks)
        cleaned = sum(1 for c in chunks if c.is_cleaned and not c.was_filtered)
        filtered = sum(1 for c in chunks if c.was_filtered)
        not_cleaned = sum(1 for c in chunks if not c.is_cleaned)

        # Filter reasons breakdown
        reasons = {}
        for c in chunks:
            if c.was_filtered and c.filter_reason:
                reasons[c.filter_reason] = reasons.get(c.filter_reason, 0) + 1

        return {
            "total": total,
            "cleaned": cleaned,
            "filtered": filtered,
            "not_cleaned": not_cleaned,
            "filter_reasons": reasons
        }


def get_cleaning_service(db: Session) -> CleaningService:
    return CleaningService(db)