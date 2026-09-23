"""Drop Whisper's non-speech hallucinations. Pure: no model, ffmpeg or settings.

With VAD off (see config.py), Whisper runs over music and effects too, and on
film audio it produces two kinds of fake lines there (measured on the Fargo and
Interstellar intros, 2026-09-23):

- stock phrases from its training data, above all "Thank you." (four times in
  two film intros, each alone in 30-80 s of score);
- SDH-style descriptors and music marks ("PIANO PLAYS", "THE END", "¶¶"),
  learned from hearing-impaired subtitles. We never write SDH, so these go.

Whisper's own confidence signals don't separate them: no_speech_prob was 0.00
for every segment, avg_logprob and compression_ratio are per 30 s window, and
the fakes' word probabilities (0.36-0.78) overlap real short lines ("Yeah."
0.44). Isolation does: a stock phrase is dropped only when it sits alone with
no speech for `isolation_s` on both sides, so a real "Thank you." inside a
conversation survives.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .cues import Word

# Normalised (lowercase, letters and spaces only) texts Whisper emits over
# music/silence. Keep this list short and specific: every entry can also be
# a real line, which isolation protects.
STOCK_PHRASES = frozenset(
    {
        "thank you",
        "thank you very much",
        "thanks for watching",
        "thank you for watching",
        "thanks for watching and see you next time",
        "please subscribe",
        "subscribe to my channel",
        "bye",
        "bye bye",
        "you",
        "the end",
        "subtitles by the amaraorg community",
        "subtitles by",
        "transcription by",
    }
)

# Mean word probability gates. Measured fakes scored 0.36-0.78 ("THE END" 0.36,
# "Thank you." 0.54-0.66, "PIANO PLAYS" 0.73); real short lines in the same
# films mostly scored higher. Isolated lines (no speech for isolation_s either
# side) are dropped below ISOLATED_MAX_PROB; a caps line inside a scene only
# below CAPS_IN_SCENE_MAX_PROB, since shouted lines there are usually real.
ISOLATED_MAX_PROB = 0.85
CAPS_IN_SCENE_MAX_PROB = 0.5

_MUSIC = re.compile(r"^[\s♪♫¶#*~.…-]*$")
_LETTERS = re.compile(r"[^a-z ]+")


class _Seg(Protocol):
    start: float
    end: float
    words: Sequence[Word]


def segment_text(seg: _Seg) -> str:
    return " ".join(w.text.strip() for w in seg.words).strip()


@dataclass(frozen=True)
class Segment:
    """Minimal segment for callers without one (the transcriber passes
    align.AlignSegment, which has the same fields)."""

    start: float
    end: float
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return segment_text(self)


def _norm(text: str) -> str:
    return " ".join(_LETTERS.sub("", text.lower().replace(".", "")).split())


_CAPS_IGNORE = frozenset({"OK", "I"})


def is_caps_line(text: str) -> bool:
    """All-caps with 2+ words or 5+ letters ("PIANO PLAYS", "EXPLOSION"). Words of
    two letters or fewer, "I" and "OK" don't count, so "NO!", "OK. OK." and
    "I... I..." aren't caps lines. It can still be a real shout ("STOP IT!"),
    so the caller only drops it with low confidence as well."""
    letters = [c for c in text if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return False
    words = ["".join(c for c in w if c.isalpha()) for w in text.split()]
    words = [w for w in words if len(w) >= 3 and w not in _CAPS_IGNORE]
    if not words:
        return False
    return len(text.split()) >= 2 or sum(len(w) for w in words) >= 5


def is_descriptor(text: str) -> bool:
    """Unambiguous non-speech: bare music marks ("♪♪", "¶¶") or a bracketed
    description ("[MUSIC]", "(GUNSHOT)"). Always dropped."""
    t = text.strip()
    if not t or _MUSIC.match(t):
        return True
    if (t.startswith("[") and t.endswith("]")) or (t.startswith("(") and t.endswith(")")):
        return True
    return False


def filter_segments[S: _Seg](
    segments: Sequence[S], isolation_s: float = 4.0
) -> tuple[list[S], list[str]]:
    """Return (kept segments, texts of the dropped ones), order preserved."""
    kept: list[S] = []
    dropped: list[str] = []
    for i, seg in enumerate(segments):
        text = segment_text(seg)
        if not seg.words or is_descriptor(text):
            if text:
                dropped.append(text)
            continue
        prev_end = segments[i - 1].end if i > 0 else float("-inf")
        next_start = segments[i + 1].start if i + 1 < len(segments) else float("inf")
        isolated = seg.start - prev_end >= isolation_s and next_start - seg.end >= isolation_s
        mean_prob = sum(w.prob for w in seg.words) / len(seg.words)
        if is_caps_line(text):
            # "PIANO PLAYS" (0.73, alone in the score) vs a shouted "HELP! HELP!"
            # mid-scene.
            if (isolated and mean_prob < ISOLATED_MAX_PROB) or mean_prob < CAPS_IN_SCENE_MAX_PROB:
                dropped.append(text)
                continue
        if _norm(text) in STOCK_PHRASES:
            if isolated and mean_prob < ISOLATED_MAX_PROB:
                dropped.append(text)
                continue
        kept.append(seg)
    return kept, dropped
