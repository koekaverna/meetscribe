"""Hallucination filter for Whisper segments.

Whisper invents text on silence and noise: subtitle credits, sign-offs, sound
captions. ``no_speech_prob`` alone is not a usable signal — turbo models barely
raise it — so the filter combines a phrase blocklist, a hard ``avg_logprob``
floor, and the legacy two-threshold rule restricted to short segments.
"""

import re
from dataclasses import dataclass, field

# Drop reasons, as written to the log and stored with dropped segments.
REASON_BLOCKLIST = "blocklist"
REASON_CAPTION = "caption"
REASON_LOGPROB = "logprob"
REASON_LEGACY = "legacy"
REASON_DUPLICATE = "duplicate"
REASON_FAILED = "failed"

DEFAULT_HALLUCINATION_PHRASES = [
    "Редактор субтитров",
    "Корректор",
    "Субтитры подготовлены",
    "Субтитры сделал",
    "Субтитры сделаны",
    "Субтитры добавил",
    "Продолжение следует",
    "С вами был",
    "Спасибо за просмотр",
    "Спасибо за внимание",
    "Благодарю за внимание",
    "Подписывайтесь",
    "Увидимся в следующем видео",
    "Увидимся в следующих видео",
    "До новых встреч",
    "Добро пожаловать",
    "Фондю любит тебя",
    "КОНЕЦ",
    "СПОКОЙНАЯ МУЗЫКА",
    "СМЕХ",
]

_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_CYRILLIC_RE = re.compile(r"[А-ЯЁ]")
# Shorter all-caps texts are as likely an abbreviation as a caption
_CAPTION_MIN_LETTERS = 5


def normalize_words(text: str) -> list[str]:
    """Lowercase, fold ё→е, strip punctuation and split into words."""
    cleaned = _NON_WORD_RE.sub(" ", text.lower().replace("ё", "е"))
    return cleaned.split()


def is_caption(text: str) -> bool:
    """Whether the text is a sound caption: Cyrillic, written in capitals only.

    Whisper renders non-speech sounds the way subtitles do ("ДИНАМИЧНАЯ
    МУЗЫКА", "СМЕХ"); it does not write speech in capitals.
    """
    letters = [c for c in text if c.isalpha()]
    return (
        len(letters) >= _CAPTION_MIN_LETTERS
        and all(c.isupper() for c in letters)
        and _CYRILLIC_RE.search(text) is not None
    )


def _contains_phrase(words: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return any(words[i : i + n] == phrase for i in range(len(words) - n + 1))


@dataclass
class HallucinationFilter:
    """Decides whether a transcribed segment is a hallucination.

    A segment is dropped when any rule matches:

    1. ``blocklist``: the text contains a known phrase and is at most
       ``phrase_max_extra_words`` words longer than it.
    2. ``caption``: the text is a sound caption in capitals (see ``is_caption``).
    3. ``logprob``: ``avg_logprob <= logprob_floor``.
    4. ``legacy``: ``no_speech_prob >= no_speech_prob_threshold`` and
       ``avg_logprob <= avg_logprob_threshold``, only for segments of up to
       ``legacy_max_words`` words.
    """

    no_speech_prob_threshold: float = 0.5
    avg_logprob_threshold: float = -0.25
    legacy_max_words: int = 5
    logprob_floor: float = -1.0
    phrases: list[str] = field(default_factory=lambda: list(DEFAULT_HALLUCINATION_PHRASES))
    phrase_max_extra_words: int = 8
    drop_captions: bool = True

    def __post_init__(self) -> None:
        normalized = (normalize_words(p) for p in self.phrases)
        self._phrases = [p for p in normalized if p]

    def check(self, text: str, no_speech_prob: float, avg_logprob: float) -> str | None:
        """Return the drop reason, or None if the segment should be kept."""
        words = normalize_words(text)

        for phrase in self._phrases:
            if len(words) <= len(phrase) + self.phrase_max_extra_words and _contains_phrase(
                words, phrase
            ):
                return REASON_BLOCKLIST

        if self.drop_captions and is_caption(text):
            return REASON_CAPTION

        if avg_logprob <= self.logprob_floor:
            return REASON_LOGPROB

        if (
            len(words) <= self.legacy_max_words
            and no_speech_prob >= self.no_speech_prob_threshold
            and avg_logprob <= self.avg_logprob_threshold
        ):
            return REASON_LEGACY

        return None
