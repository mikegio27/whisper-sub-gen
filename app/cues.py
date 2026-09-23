"""Word timings -> subtitle cues that follow `standards.CueRules`.

Pure (no model, ffmpeg or settings). The input is a word stream from faster-whisper
(`word_timestamps=True`) or, later, a forced aligner; the output is a list of `srt.Cue`.

Pipeline, in `compose()`:

1. `_sanitize`: drop empty words, sort by onset, glue fragments ("'s", "n't", "...", ",") onto
   their word, clamp stretched words and word overlaps, collapse whisper word loops.
2. Split into runs at every pause >= `split_pause`; a pause always starts a new cue.
3. `_segment`: a DP over each run picks the cue boundaries with the lowest total cost
   (break quality, size, reading speed, duration). Lookahead is bounded by what fits in
   `max_lines x max_line_chars`, so the whole thing is O(n * k).
4. `_drop_repeats`: drop a short cue that repeats the previous one within 1 s.
5. `_timed`: timing in integer milliseconds (min/max duration, linger, cps, min_gap), and
   `_layout` for the line breaks.

Cleanup that removes words (documented, and the only places text is lost):
- runs of more than `_MAX_REPEATS` identical words keep only the first `_MAX_REPEATS`;
- a cue whose normalized text equals the previous cue's, both short and < 1 s apart, is dropped.
Both are whisper decoding loops far more often than real speech.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.srt import Cue
from app.standards import DEFAULT_RULES, CueRules


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str  # as the ASR gives it, possibly with a leading space
    prob: float = 1.0  # ASR confidence; carried through for later phases


# --- word classes -------------------------------------------------------------------------

# Ending a cue or a line on one of these leaves the reader hanging ("...the / door").
_WEAK = frozenset(
    "a an the of to in on at by for with from into onto upon about over under after before "
    "and or but nor if as than because while my your his its our their".split()
)
# A line (or cue) that *starts* with one of these reads as a natural clause start.
_BREAK_BEFORE = frozenset(
    "and or but nor so because although though when while where which who whom whose that "
    "if unless until than to of in on at for with from about into after before".split()
)
# "Mr." is not the end of a sentence.
_ABBREV = frozenset("mr mrs ms dr st prof sr jr vs etc mt lt col gen sgt capt".split())

_CLOSERS = "\"'”’)]}»"
_OPENERS = set("([{“‘¿¡«")
_PUNCT_ONLY = re.compile(r"^[.,!?;:…%)\]}\"'”’»\-–—]+$")
_CONTRACTION = re.compile(r"^(?:['’](?:s|t|d|m|re|ll|ve)|n['’]t)$", re.IGNORECASE)
_NORM = re.compile(r"[^\w']+")

# Whisper loops ("no no no no no no") are cut to this many repeats. Real speech rarely goes
# past three ("no, no, no!"), so three keeps the emphasis.
_MAX_REPEATS = 3
# Only short cues are candidates for the repeat drop, since a long line said twice in a row
# is more likely real (a character repeating a question) than a decoding loop.
_REPEAT_MAX_CHARS = 32
_REPEAT_MAX_GAP = 1.0
# Upper bound on words per cue for the DP lookahead. The char limit (2 x 42) normally ends
# the lookahead first; this only guards against a run of zero-length tokens.
_MAX_CUE_WORDS = 40
# How far the next cue's start may be pushed later so this one reaches min_duration.
_NUDGE_MS = 100


def _norm(text: str) -> str:
    return _NORM.sub("", text.lower()).strip("'")


def _is_sentence_end(text: str) -> bool:
    t = text.rstrip(_CLOSERS)
    if t.endswith(("?", "!", "…", "♪")):
        return True
    return t.endswith(".") and _norm(t[:-1]) not in _ABBREV


def _is_clause_end(text: str) -> bool:
    return text.rstrip(_CLOSERS).endswith((",", ";", ":", "-", "–", "—"))


def _is_weak_end(text: str) -> bool:
    # Punctuation after the word ("to," / "for.") means it closes something, so it's fine.
    return text[-1:].isalnum() and _norm(text) in _WEAK


# --- sanitize -----------------------------------------------------------------------------


def _glues_back(raw: str, text: str, spaced_stream: bool) -> bool:
    if text == "♪":
        return False
    if _CONTRACTION.match(text) or _PUNCT_ONLY.match(text):
        return True
    # Whisper words carry a leading space; one without is a sub-word continuation ("-known",
    # "ley"). Aligner output may have no spaces at all, so only trust this in spaced streams.
    return spaced_stream and not raw[:1].isspace()


def _sanitize(words: Sequence[Word], rules: CueRules) -> list[Word]:
    items = [w for w in words if w.text and w.text.strip()]
    if not items:
        return []
    # Onsets are the reliable half of an ASR timestamp, so a word with end < start keeps
    # its start and is treated as instantaneous.
    items = [w if w.end >= w.start else Word(w.start, w.start, w.text, w.prob) for w in items]
    items.sort(key=lambda w: w.start)  # stable: equal onsets keep ASR order
    spaced = sum(w.text[:1].isspace() for w in items[1:]) >= (len(items) - 1) / 2

    glued: list[Word] = []
    pending_open = ""
    for w in items:
        text = w.text.strip()
        if all(c in _OPENERS for c in text):
            pending_open += text
            continue
        if pending_open:
            text, pending_open = pending_open + text, ""
        elif glued and _glues_back(w.text, text, spaced):
            p = glued[-1]
            glued[-1] = Word(p.start, max(p.end, w.end), p.text + text, min(p.prob, w.prob))
            continue
        glued.append(Word(w.start, w.end, text, w.prob))
    if pending_open and glued:
        p = glued[-1]
        glued[-1] = Word(p.start, p.end, p.text + pending_open, p.prob)

    out: list[Word] = []
    for i, w in enumerate(glued):
        # A word stretched over music or silence: keep the onset, drop the tail.
        end = min(w.end, w.start + rules.max_word_duration)
        if i + 1 < len(glued):
            end = min(end, glued[i + 1].start)  # overlapping words: next onset wins
        out.append(Word(w.start, max(end, w.start), w.text, w.prob))
    return _collapse_repeats(out)


def _collapse_repeats(words: list[Word]) -> list[Word]:
    out: list[Word] = []
    run_key, run_len = None, 0
    for w in words:
        key = _norm(w.text)
        if key and key == run_key:
            run_len += 1
        else:
            run_key, run_len = key, 1
        if run_len <= _MAX_REPEATS:
            out.append(w)
    return out


# --- segmentation -------------------------------------------------------------------------


def _runs(ws: list[Word], rules: CueRules) -> list[tuple[int, int]]:
    runs, lo = [], 0
    for i in range(1, len(ws)):
        if ws[i].start - ws[i - 1].end >= rules.split_pause:
            runs.append((lo, i))
            lo = i
    runs.append((lo, len(ws)))
    return runs


def _break_cost(last: Word, nxt: Word) -> float:
    """Cost of ending a cue after `last` when `nxt` follows in the same run."""
    if _is_sentence_end(last.text):
        cost = 0.0
    elif _is_clause_end(last.text):
        cost = 2.0
    else:
        cost = 6.0
        if _is_weak_end(last.text):
            cost += 8.0
        elif _norm(nxt.text) in _BREAK_BEFORE:
            cost -= 3.0
    # A pause under split_pause is still a natural seam in the speech.
    return cost - 6.0 * min(max(nxt.start - last.end, 0.0), 0.6)


def _segment(
    ws: list[Word], lo: int, hi: int, after: float | None, rules: CueRules
) -> list[tuple[int, int]]:
    """Split ws[lo:hi] (one pause-free run) into cues; returns [(i, j)] word index ranges.

    best[b] is the cheapest segmentation of the first b words; each cue's cost is local, so a
    plain shortest-path DP is exact. Summing a per-cue constant with a convex size term makes
    balanced cues cheaper than one full cue plus a stub.
    """
    n = hi - lo
    capacity = rules.max_lines * rules.max_line_chars
    best = [math.inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for a in range(n):
        if best[a] == math.inf:
            continue
        first = ws[lo + a]
        lines = line_len = chars = internal = 0
        for b in range(a + 1, min(n, a + _MAX_CUE_WORDS) + 1):
            w = ws[lo + b - 1]
            size = len(w.text)
            # Greedy line filling gives the fewest lines, so it is an exact fit test.
            if line_len == 0:
                lines, line_len = 1, size
            elif line_len + 1 + size <= rules.max_line_chars:
                line_len += 1 + size
            else:
                lines, line_len = lines + 1, size
            if lines > rules.max_lines and b > a + 1:
                break
            if b > a + 1:
                chars += 1
                internal += _is_sentence_end(ws[lo + b - 2].text)
            chars += size
            span = w.end - first.start
            if span > 2 * rules.max_duration and b > a + 1:
                break

            if b < n:
                avail = ws[lo + b].start - rules.min_gap - first.start
            else:
                tail = rules.max_linger
                if after is not None:
                    tail = min(tail, after - rules.min_gap - w.end)
                avail = span + tail
            avail = max(min(avail, rules.max_duration), 1e-3)

            cost = 4.0 + 10.0 * (chars / capacity) ** 2 + 2.5 * internal
            if span > rules.max_duration:
                cost += 50.0 + 20.0 * (span - rules.max_duration)
            cps = chars / avail
            if cps > rules.max_cps:
                cost += 3.0 * (cps - rules.max_cps)
            elif cps > rules.target_cps:
                cost += 0.3 * (cps - rules.target_cps)
            if avail < rules.min_duration:
                cost += 4.0 * (rules.min_duration - avail)
            if b - a == 1 and n > 1:
                cost += 6.0  # orphan: one word alone when a neighbour could take it
            if b < n:
                cost += _break_cost(w, ws[lo + b])
            if lines == 2 and rules.max_lines == 2:
                # A cue that can only be broken badly ("topsy-turvy. I / don't ...") should
                # lose to one split at the sentence instead.
                split = _best_split([x.text for x in ws[lo + a : lo + b]], rules)
                if split is not None:
                    cost += 0.4 * max(split[0], 0.0)
            total = best[a] + cost
            if total < best[b]:
                best[b], back[b] = total, a

    cuts, b = [], n
    while b > 0:
        cuts.append((lo + back[b], lo + b))
        b = back[b]
    return cuts[::-1]


def _drop_repeats(ws: list[Word], groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for g in groups:
        if out:
            p = out[-1]
            gt, pt = _group_text(ws, g), _group_text(ws, p)
            if (
                _norm(gt)
                and _norm(gt) == _norm(pt)
                and len(gt) <= _REPEAT_MAX_CHARS
                and len(pt) <= _REPEAT_MAX_CHARS
                and ws[g[0]].start - ws[p[1] - 1].end < _REPEAT_MAX_GAP
            ):
                continue
        out.append(g)
    return out


def _group_text(ws: list[Word], g: tuple[int, int]) -> str:
    return " ".join(w.text for w in ws[g[0] : g[1]])


# --- layout and timing --------------------------------------------------------------------


def _best_split(tokens: list[str], rules: CueRules) -> tuple[float, int] | None:
    """Best 2-line break as (badness, index of the first bottom-line token), or None if no
    break fits. Badness: imbalance in chars, plus linguistic bonuses and penalties."""
    limit = rules.max_line_chars
    total = sum(len(t) for t in tokens) + len(tokens) - 1
    best: tuple[float, int] | None = None
    top = -1
    for k in range(1, len(tokens)):
        top += len(tokens[k - 1]) + 1
        bottom = total - top - 1
        # A line holding a single over-long word is allowed; there's nothing else to do.
        if (top > limit and k > 1) or (bottom > limit and len(tokens) - k > 1):
            continue
        # Balance matters less than breaking at a natural seam, hence the 0.5.
        cost = 0.5 * abs(top - bottom)
        if top > bottom:
            cost += 1.0  # slight preference for a bottom-heavy pyramid
        last, nxt = tokens[k - 1], tokens[k]
        if _is_sentence_end(last):
            cost -= 14.0
        elif _is_clause_end(last):
            cost -= 8.0
        elif _is_weak_end(last):
            cost += 12.0  # "the / door": article/preposition cut off from its noun
        elif k > 1 and _is_sentence_end(tokens[k - 2]):
            cost += 8.0  # "...topsy-turvy. I / don't": a new sentence's first word left behind
        elif _norm(nxt) in _BREAK_BEFORE:
            cost -= 3.0
        if best is None or cost < best[0]:
            best = (cost, k)
    return best


def _layout(tokens: list[str], rules: CueRules) -> str:
    """Line-break one cue: one line if it fits, else the best-balanced linguistic 2-line split."""
    text = " ".join(tokens)
    if len(text) <= rules.max_line_chars or len(tokens) == 1 or rules.max_lines < 2:
        return text
    split = _best_split(tokens, rules)
    if split is not None:
        k = split[1]
        return " ".join(tokens[:k]) + "\n" + " ".join(tokens[k:])
    # More than two lines' worth (only with max_lines > 2 or over-long words): fill greedily.
    lines: list[str] = []
    for t in tokens:
        if lines and len(lines[-1]) + 1 + len(t) <= rules.max_line_chars:
            lines[-1] += " " + t
        else:
            lines.append(t)
    return "\n".join(lines)


def _timed(ws: list[Word], groups: list[tuple[int, int]], rules: CueRules) -> list[Cue]:
    """Cue timing in integer ms, so rounding can never create an overlap or a short gap."""
    gap = round(rules.min_gap * 1000)
    min_d = round(rules.min_duration * 1000)
    max_d = round(rules.max_duration * 1000)
    linger = round(rules.max_linger * 1000)
    starts = [round(ws[i].start * 1000) for i, _ in groups]
    spoken = [round(ws[j - 1].end * 1000) for _, j in groups]

    cues: list[Cue] = []
    prev_end: int | None = None
    for k, (i, j) in enumerate(groups):
        text = _layout([w.text for w in ws[i:j]], rules)
        s = starts[k]
        if prev_end is not None and s < prev_end + gap:
            s = prev_end + gap  # only with degenerate input (onsets closer than min_gap)
        e_spoken = max(spoken[k], s)
        chars = len(text.replace("\n", ""))
        want = max(
            e_spoken + round(rules.min_linger * 1000),
            s + min_d,
            s + math.ceil(chars / rules.target_cps * 1000),
        )
        # Linger is capped, but min_duration beats the cap.
        want = min(want, max(e_spoken + linger, s + min_d))
        if k + 1 < len(groups):
            limit = starts[k + 1] - gap
            if want > limit and limit < s + min_d:
                # Squeezed below min_duration by the next cue: steal up to 0.1 s of its
                # onset rather than flash this one. min_gap still wins after that.
                shift = min(_NUDGE_MS, s + min_d - limit)
                starts[k + 1] += shift
                limit += shift
            want = min(want, limit)
        want = min(want, s + max_d)
        end = max(want, s + 1)
        nxt = starts[k + 1] if k + 1 < len(groups) else None
        end = _float_safe(s, end, nxt, rules)
        cues.append(Cue(s / 1000, end / 1000, text))
        prev_end = end
    return cues


def _float_safe(s: int, end: int, nxt: int | None, rules: CueRules) -> int:
    """Shift `end` by a millisecond where float seconds would misjudge a limit.

    0.833 s of ms is not always >= 0.833 once both ends are floats (157.973 - 157.14 <
    0.833), and consumers compare floats, so the checks are redone in the output domain.
    """

    def dur(e: int) -> float:
        return e / 1000 - s / 1000

    def gap_ok(e: int) -> bool:
        return nxt is None or nxt / 1000 - e / 1000 >= rules.min_gap

    while end > s + 1 and (dur(end) > rules.max_duration or not gap_ok(end)):
        end -= 1
    if dur(end) < rules.min_duration and gap_ok(end + 1) and dur(end + 1) <= rules.max_duration:
        end += 1
    return end


def compose(words: Sequence[Word], rules: CueRules = DEFAULT_RULES) -> list[Cue]:
    """Build standards-following cues from word timings. See the module docstring."""
    ws = _sanitize(words, rules)
    if not ws:
        return []
    groups: list[tuple[int, int]] = []
    for lo, hi in _runs(ws, rules):
        after = ws[hi].start if hi < len(ws) else None
        groups += _segment(ws, lo, hi, after, rules)
    return _timed(ws, _drop_repeats(ws, groups), rules)
