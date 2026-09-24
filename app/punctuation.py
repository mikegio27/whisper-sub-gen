"""Repair whisper's no-punctuation mode by re-decoding just those stretches.

Without text conditioning (config.py), some 30 s windows come out as run-on
lowercase with no punctuation at all ("what's your news baby i'm pregnant i'm
sorry i'm pregnant"): 0.5-2.5% of cues per film on the eval set. Re-decoding
the same audio with a punctuated initial_prompt steers whisper back into
subtitle style ("What's your news, baby? I'm pregnant. I'm sorry? I'm
pregnant."). Measured on The Rock, 2026-09-24.

The re-decode window has to be a little wider than the segment, so it can
pick up words from the neighbours ("Goodies, which had to be...") or hear a
different sentence. So only re-decoded words timed inside the original span
are kept, and the result is only accepted if it has the same words as before,
now punctuated. Pure apart from the injected `decode` callable.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .cues import Word

# Punctuated, capitalised, neutral: the style we want whisper to continue in.
PROMPT = "Hello, everyone. I'm glad you're here! Where were we? Let's begin, then."

MIN_WORDS = 4  # shorter unpunctuated segments are usually fine ("yeah okay")
MAX_WINDOW_S = 28.0  # one whisper window, so initial_prompt applies to all of it
PAD_S = 0.3
MIN_SIMILARITY = 0.8  # same words, now punctuated; anything less is a different reading

_WORD = re.compile(r"[a-z0-9']+")


def is_unpunctuated(text: str) -> bool:
    """Run-on lowercase with no punctuation, the signature of the failure."""
    if len(re.findall(r"[A-Za-z']+", text)) < MIN_WORDS:
        return False
    return not re.search(r"[A-Z.,?!;:]", text)


@dataclass
class _Seg:
    start: float
    end: float
    words: list[Word]


@dataclass
class RepairStats:
    candidates: int = 0
    repaired: int = 0
    rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, why: str) -> None:
        self.rejected += 1
        self.reasons[why] = self.reasons.get(why, 0) + 1

    def as_dict(self) -> dict:
        return {
            "candidates": self.candidates,
            "repaired": self.repaired,
            "rejected": self.rejected,
            "reasons": dict(self.reasons),
        }


def _text(words: Sequence[Word]) -> str:
    return " ".join(w.text.strip() for w in words)


def _tokens(words: Sequence[Word]) -> list[str]:
    return _WORD.findall(_text(words).lower())


def groups(segments: Sequence, max_window_s: float = MAX_WINDOW_S) -> list[list[int]]:
    """Indices of unpunctuated segments, adjacent ones merged while the merged
    span fits one whisper window."""
    out: list[list[int]] = []
    for i, seg in enumerate(segments):
        if not is_unpunctuated(_text(seg.words)):
            continue
        if out and out[-1][-1] == i - 1 and seg.end - segments[out[-1][0]].start <= max_window_s:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _norm(w: Word) -> str:
    return "".join(_WORD.findall(w.text.lower()))


def trim_edges(old: Sequence[Word], new: Sequence[Word]) -> list[Word]:
    """Drop re-decoded words before the first / after the last word that matches
    the original run: at the edges they were borrowed from a neighbour
    ("Goodies, which had to be..." when the run starts at "which")."""
    a, b = [_norm(w) for w in old], [_norm(w) for w in new]
    blocks = [
        m
        for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
        if m.size
    ]
    if not blocks:
        return list(new)
    return list(new[blocks[0].b : blocks[-1].b + blocks[-1].size])


def accept(old: Sequence[Word], new: Sequence[Word]) -> str | None:
    """None if `new` is an acceptable repair of `old`, else the reason not."""
    if not new:
        return "empty"
    if is_unpunctuated(_text(new)):
        return "still unpunctuated"
    a, b = _tokens(old), _tokens(new)
    sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    if sim < MIN_SIMILARITY:
        return "different words"
    return None


def repair(
    segments: Sequence,
    decode: Callable[[float, float], list[Word]],
    make_segment: Callable[[float, float, list[Word]], object] | None = None,
) -> tuple[list, RepairStats]:
    """Return segments with unpunctuated runs re-decoded where that helps.

    `decode(start_s, end_s)` re-transcribes that span of the film with PROMPT and
    returns words with film-relative times. `make_segment(start, end, words)`
    builds the replacement segment (default: the input's type via _Seg).
    """
    stats = RepairStats()
    make = make_segment or (lambda s, e, w: _Seg(s, e, w))
    out = list(segments)
    # Right to left, so replacing a run doesn't shift the indices still to do.
    for run in reversed(groups(segments)):
        stats.candidates += 1
        first, last = segments[run[0]], segments[run[-1]]
        old = [w for i in run for w in segments[i].words]
        lo, hi = first.start, last.end
        prev_end = segments[run[0] - 1].end if run[0] > 0 else 0.0
        a = max(prev_end, lo - PAD_S, 0.0)
        b = hi + PAD_S
        if run[-1] + 1 < len(segments):
            b = min(b, segments[run[-1] + 1].start)
        try:
            new = decode(a, b)
        except Exception:  # noqa: BLE001 - a failed repair keeps the original text
            stats.reject("decode error")
            continue
        # Only words that start inside the original span: the padding may have
        # picked up a neighbour's words.
        new = [w for w in new if lo - 0.15 <= w.start <= hi + 0.05]
        new = trim_edges(old, new)
        why = accept(old, new)
        if why:
            stats.reject(why)
            continue
        out[run[0] : run[-1] + 1] = [make(lo, hi, new)]
        stats.repaired += 1
    return out, stats
