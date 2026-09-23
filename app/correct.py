"""Local-LLM word correction (ROADMAP Phase 3): fix misheard words, keep the timings.

The LLM never rewrites text. Code picks the suspicious words (`flag_words`), the LLM may
propose a replacement for some of them, and code decides whether each proposal is plausible
(`judge_edit`). Anything unexpected (Ollama down, timeout, bad JSON, an edit that doesn't
sound like the original) keeps the ASR text: the pass can only fail open.

Stages, in `correct()`:

1. `flag_words`: whisper prob < threshold, a capitalised word mid-sentence that isn't a known
   name (a misheard name: "No, no, Patrona!"), or a word said 3+ times in a row (a loop).
2. `_lines` + `plan_windows`: split the words into cue-sized lines, then windows of ~12-20
   lines around the flags (merged when they overlap), each with 2 read-only lines either side,
   kept under ~1,500 input tokens.
3. `build_request`: numbered lines with the flags marked `[k:word]`, a JSON schema in Ollama's
   `format` (`{"edits": [{"i": k, "text": "..."}]}`, maxItems = number of flags).
4. `judge_edit`: accept only if the index is a flag, the replacement is 1-3 plain words, it
   changes more than case/punctuation, and it is an exact context name OR phonetically /
   orthographically close to the original (same Metaphone key, or letter Levenshtein <= 0.5).
5. `apply_edits`: the replacement takes the original word's start/end (split by characters
   across a multi-word replacement) and prob=1.0, so an accepted word never looks
   low-confidence to a later stage. The ASR's leading space and edge punctuation are kept.

Pure apart from `_post` (stdlib urllib to Ollama `/api/chat`); no settings are read here. The
transcriber passes them in (`correct_video`).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.context import MediaContext
from app.cues import Word

log = logging.getLogger(__name__)

# Default for LLM_FLAG_PROB. On the OotP clip (212 words, large-v3-turbo int8_float16 beam 5,
# 2026-09-22) 0.6 flags 9.9% of words by probability alone; the known errors sit at
# "whingey" 0.68, "Patrona" 0.63-0.67, "Fig?" 0.47, "Excto" 0.34, "road?" 0.28. Patrona is
# caught by the capitalisation rule instead, so the prob threshold can stay under ~10%.
DEFAULT_FLAG_PROB = 0.6

# Windows: lines of editable text per window (a line ~ one cue), read-only context either side.
WINDOW_LINES = 12
WINDOW_MAX_LINES = 20
CONTEXT_LINES = 2
# ~1,500 input tokens at ~4 chars/token, system prompt and header included.
TOKEN_BUDGET = 1500
CHARS_PER_TOKEN = 4
MAX_EDITS_PER_WINDOW = 6
MAX_REPLACEMENT_WORDS = 3
MAX_LEV = 0.5
# Looser bound when the replacement is a known name ("Fig" -> "Figg" 0.25,
# "Patrona" -> "Patronum" 0.38), but still a bound.
NAME_MAX_LEV = 0.6
# Line splitting (words -> cue-sized lines for the prompt).
_LINE_MAX_CHARS = 84
_LINE_PAUSE_S = 1.0
# Stop contacting Ollama for this film after this many failed requests in a row.
_MAX_CONSECUTIVE_ERRORS = 2

_SENT_END = re.compile(r"[.!?…]['\"”’)\]]*$")
_LETTERS = re.compile(r"[^a-z0-9]")
_EDGE = re.compile(r"^(?P<lead>\s*[\"'“‘(\[¿¡«-]*)(?P<core>.*?)(?P<trail>[.,!?;:…\"'”’)\]»-]*)$")
_OK_CHARS = re.compile(r"^[\w' ’.-]+$")
# Capitalised mid-sentence but normal English, not a misheard name.
_ALWAYS_CAPS = frozenset(
    "i i'm i'll i've i'd god lord mum mom dad sir madam mr mrs ms dr ok okay tv "
    "monday tuesday wednesday thursday friday saturday sunday january february march april "
    "june july august september october november december christmas english".split()
)


_ABBREV = frozenset("mr mrs ms dr st prof sr jr vs mt lt col gen sgt capt".split())


def _ends_sentence(text: str) -> bool:
    """True after "run!" or "it." but not after "Mrs." (the name follows)."""
    text = text.strip()
    return bool(_SENT_END.search(text)) and _letters(text) not in _ABBREV


def _core(text: str) -> str:
    m = _EDGE.match(text)
    return m["core"] if m else text.strip()


def _letters(text: str) -> str:
    return _LETTERS.sub("", text.lower())


# --- flagging -----------------------------------------------------------------------------


def _name_tokens(names: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for n in names:
        for tok in re.split(r"[\s/,]+", n):
            t = _letters(tok)
            if len(t) > 1 and tok[:1].isupper():  # "Head of the Order": not "of", "the"
                out.add(t)
    return out


def _learned_names(words: Sequence[Word]) -> set[str]:
    """Capitalised words the ASR is sure of and says 2+ times ("Dudley" x5 at ~0.95): real
    names even when no metadata is available. Sentence starters ("What") get in too, which
    is harmless: this set only ever suppresses the mid-sentence capital flag."""
    seen: dict[str, list[float]] = {}
    for w in words:
        core = _core(w.text)
        if core[:1].isupper():
            seen.setdefault(_letters(core), []).append(w.prob)
    return {k for k, ps in seen.items() if k and len(ps) >= 2 and sum(ps) / len(ps) >= 0.9}


def flag_words(
    words: Sequence[Word],
    threshold: float = DEFAULT_FLAG_PROB,
    names: Sequence[str] = (),
) -> set[int]:
    """Indices of words worth a second look. See the module docstring for the rules."""
    known = _name_tokens(names) | _learned_names(words)
    flags: set[int] = set()
    run_start = 0
    for i, w in enumerate(words):
        core = _core(w.text)
        key = _letters(core)
        if not key:  # punctuation-only fragment
            continue
        if w.prob < threshold:
            flags.add(i)
        prev = words[i - 1].text.strip() if i else ""
        if (
            i
            and core[:1].isupper()
            and len(key) > 1
            and not core.isupper()  # "D.", "NASA": initials/acronyms
            and prev
            and not _ends_sentence(prev)
            and key not in known
            and key not in _ALWAYS_CAPS
            and core.lower() not in _ALWAYS_CAPS
        ):
            flags.add(i)
        # Runs of the same word ("no no no").
        if i and key == _letters(_core(words[i - 1].text)):
            if i - run_start + 1 >= 3:
                flags.update(range(run_start, i + 1))
        else:
            run_start = i
    return flags


# --- windows ------------------------------------------------------------------------------


def _lines(words: Sequence[Word]) -> list[tuple[int, int]]:
    """Cue-sized [start, end) word ranges: break after sentence ends, at pauses, or at the
    two-line cue size."""
    out: list[tuple[int, int]] = []
    start, chars = 0, 0
    for i, w in enumerate(words):
        if i > start and (
            w.start - words[i - 1].end >= _LINE_PAUSE_S or chars + len(w.text) > _LINE_MAX_CHARS
        ):
            out.append((start, i))
            start, chars = i, 0
        chars += len(w.text)
        if _ends_sentence(w.text):
            out.append((start, i + 1))
            start, chars = i + 1, 0
    if start < len(words):
        out.append((start, len(words)))
    return out


@dataclass
class Window:
    lines: list[tuple[int, int]]  # word ranges of the editable lines
    before: list[tuple[int, int]] = field(default_factory=list)  # read-only context
    after: list[tuple[int, int]] = field(default_factory=list)
    flags: list[int] = field(default_factory=list)  # word indices, prompt index = position + 1


def _line_chars(words: Sequence[Word], rng: tuple[int, int]) -> int:
    return sum(len(w.text) for w in words[rng[0] : rng[1]]) + 8  # + numbering + markers


def plan_windows(
    words: Sequence[Word],
    flags: set[int],
    *,
    budget_chars: int = TOKEN_BUDGET * CHARS_PER_TOKEN,
    target: int = WINDOW_LINES,
    max_lines: int = WINDOW_MAX_LINES,
    context: int = CONTEXT_LINES,
) -> list[Window]:
    """Windows of `target` lines centred on flagged lines, merged while <= `max_lines`, each
    trimmed (then split) to fit `budget_chars` including the context lines."""
    lines = _lines(words)
    line_of = {}
    for li, (a, b) in enumerate(lines):
        for wi in range(a, b):
            line_of[wi] = li
    flagged_lines = sorted({line_of[f] for f in flags if f in line_of})
    n = len(lines)
    spans: list[list[int]] = []  # [start, end) in line numbers
    for fl in flagged_lines:
        s = max(0, min(fl - target // 2, n - target))
        e = min(n, s + target)
        if spans and s < spans[-1][1]:
            if e - spans[-1][0] <= max_lines:
                spans[-1][1] = max(spans[-1][1], e)
                continue
            s = spans[-1][1]
            if fl >= s:
                spans.append([s, max(e, fl + 1)])
            continue
        spans.append([s, e])

    windows: list[Window] = []
    flagged_set = set(flagged_lines)

    def emit(s: int, e: int) -> None:
        # Trim unflagged edge lines, then split, until the window fits the budget.
        def size(a: int, b: int) -> int:
            lo, hi = max(0, a - context), min(n, b + context)
            return sum(_line_chars(words, lines[k]) for k in range(lo, hi))

        while size(s, e) > budget_chars and e - s > 1:
            if s not in flagged_set:
                s += 1
            elif e - 1 not in flagged_set:
                e -= 1
            else:
                mid = (s + e) // 2
                emit(s, mid)
                emit(mid, e)
                return
        if not any(k in flagged_set for k in range(s, e)):
            return
        rng = lines[s:e]
        win = Window(
            lines=rng,
            before=lines[max(0, s - context) : s],
            after=lines[e : min(n, e + context)],
        )
        win.flags = sorted(f for f in flags if rng[0][0] <= f < rng[-1][1])
        windows.append(win)

    for s, e in spans:
        emit(s, e)
    return windows


# --- prompt -------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You proofread automatic subtitles made by speech recognition. Some words, marked "
    "[number:word], may be misheard: usually a character name, a made-up term from the film, "
    "or a sound-alike word. For a marked word you are confident was misheard, give what was "
    "actually said (1-3 words that sound like the original). Leave correct words alone; when "
    "unsure, give no edit. Never rewrite unmarked text. Reply with JSON only."
)


def _render_word(w: Word, k: int | None) -> str:
    if k is None:
        return w.text
    m = _EDGE.match(w.text)
    if not m or not m["core"]:
        return w.text
    return f"{m['lead']}[{k}:{m['core']}]{m['trail']}"


def _render(words: Sequence[Word], rng: tuple[int, int], marks: dict[int, int]) -> str:
    return "".join(_render_word(words[i], marks.get(i)) for i in range(*rng)).strip()


def _header(ctx: MediaContext | None) -> str:
    if ctx is None or not ctx.title:
        return ""
    title = ctx.title + (f" ({ctx.year})" if ctx.year else "")
    if ctx.is_episode:
        title += f", S{ctx.season:02d}E{ctx.episode:02d}"
        if ctx.episode_title:
            title += f' "{ctx.episode_title}"'
    parts = [f"Title: {title}"]
    names = ", ".join(ctx.names)[:700]
    if names:
        parts.append(f"Names: {names.rsplit(', ', 1)[0] if len(names) == 700 else names}")
    terms = ", ".join(ctx.terms)[:300]
    if terms:
        parts.append(f"Terms: {terms.rsplit(', ', 1)[0] if len(terms) == 300 else terms}")
    return "\n".join(parts) + "\n\n"


def build_request(
    words: Sequence[Word],
    win: Window,
    ctx: MediaContext | None,
    *,
    model: str,
    last: bool = False,
) -> dict:
    """The Ollama /api/chat body for one window."""
    marks = {wi: k + 1 for k, wi in enumerate(win.flags)}
    body: list[str] = []
    for rng in win.before:
        body.append("  (before) " + _render(words, rng, {}))
    for n, rng in enumerate(win.lines, 1):
        body.append(f"{n:>3} " + _render(words, rng, marks))
    for rng in win.after:
        body.append("  (after) " + _render(words, rng, {}))
    k = len(win.flags)
    user = (
        _header(ctx)
        + "Subtitle lines (context lines are not editable):\n"
        + "\n".join(body)
        + "\n\n"
        f"Marked words: 1-{k}. Return edits only for marked words that were misheard, as "
        '{"edits": [{"i": <marked number>, "text": "<replacement>"}]}; '
        '{"edits": []} if all are right.'
    )
    schema = {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "maxItems": k,
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer", "minimum": 1, "maximum": k},
                        "text": {"type": "string"},
                    },
                    "required": ["i", "text"],
                },
            }
        },
        "required": ["edits"],
    }
    req = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "format": schema,
        # Bounded output: ~25 tokens per edit is plenty for {"i": 7, "text": "Patronum"}.
        "options": {"temperature": 0, "num_predict": 32 + 25 * k},
    }
    if last:
        req["keep_alive"] = 0  # free the VRAM once the film is done
    return req


def parse_edits(content: object) -> list[dict] | None:
    """The `edits` list from a model reply, or None when the reply is unusable."""
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError:
            return None
    if not isinstance(content, dict) or not isinstance(content.get("edits"), list):
        return None
    return [e for e in content["edits"] if isinstance(e, dict)]


# --- acceptance ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_lev(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


_VOWELS = frozenset("AEIOU")


def metaphone(word: str) -> str:
    """Lawrence Philips' original Metaphone (1990), compact. Letters only; an initial vowel
    is coded 'A'. Good enough to say "sounds alike" for English-ish names and words."""
    w = re.sub(r"[^A-Z]", "", word.upper())
    if not w:
        return ""
    if w[:2] in ("AE", "GN", "KN", "PN", "WR"):
        w = w[1:]
    elif w[0] == "X":
        w = "S" + w[1:]
    elif w[:2] == "WH":
        w = "W" + w[2:]
    out: list[str] = []
    n = len(w)

    def at(k: int) -> str:
        return w[k] if 0 <= k < n else ""

    for i, c in enumerate(w):
        if c == at(i - 1) and c != "C":
            continue
        nxt, prv = at(i + 1), at(i - 1)
        if c in _VOWELS:
            if i == 0:
                out.append("A")
        elif c == "B":
            if not (prv == "M" and i == n - 1):
                out.append("B")
        elif c == "C":
            if nxt == "I" and at(i + 2) == "A" or nxt == "H":
                out.append("K" if prv == "S" else "X")
            elif nxt in ("I", "E", "Y"):
                if prv != "S":
                    out.append("S")
            else:
                out.append("K")
        elif c == "D":
            out.append("J" if nxt == "G" and at(i + 2) in ("E", "I", "Y") else "T")
        elif c == "G":
            if nxt == "H" and not (i + 2 >= n or at(i + 2) in _VOWELS):
                continue
            if nxt == "N" and (i + 2 == n or w[i + 1 :] == "NED"):
                continue
            if prv == "D" and nxt in ("E", "I", "Y"):
                continue
            out.append("J" if nxt in ("I", "E", "Y") and prv != "G" else "K")
        elif c == "H":
            if nxt in _VOWELS and prv not in ("C", "S", "P", "T", "G"):
                out.append("H")
        elif c == "K":
            if prv != "C":
                out.append("K")
        elif c == "P":
            out.append("F" if nxt == "H" else "P")
        elif c == "Q":
            out.append("K")
        elif c == "S":
            if nxt == "H" or (nxt == "I" and at(i + 2) in ("O", "A")):
                out.append("X")
            else:
                out.append("S")
        elif c == "T":
            if nxt == "I" and at(i + 2) in ("O", "A"):
                out.append("X")
            elif nxt == "H":
                out.append("0")
            elif not (nxt == "C" and at(i + 2) == "H"):
                out.append("T")
        elif c == "V":
            out.append("F")
        elif c == "W" or c == "Y":
            if nxt in _VOWELS:
                out.append(c)
        elif c == "X":
            out.append("KS")
        elif c == "Z":
            out.append("S")
        else:  # F J L M N R
            out.append(c)
    return "".join(out)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    text: str = ""  # cleaned replacement (no edge punctuation) when ok


def judge_edit(
    original: str,
    replacement: object,
    *,
    names: Sequence[str] = (),
) -> Verdict:
    """Is `replacement` a plausible correction of the ASR word `original`? Pure."""
    if not isinstance(replacement, str):
        return Verdict(False, "not a string")
    if "\n" in replacement or "\r" in replacement:
        return Verdict(False, "newline")
    rep = _core(replacement.strip())
    toks = rep.split()
    if not toks:
        return Verdict(False, "empty")
    if len(toks) > MAX_REPLACEMENT_WORDS:
        return Verdict(False, f"{len(toks)} words")
    if not _OK_CHARS.match(rep):
        return Verdict(False, "odd characters")
    orig_core = _core(original.strip())
    a, b = _letters(orig_core), _letters(rep)
    if not b:
        return Verdict(False, "no letters")
    if a == b:
        return Verdict(False, "casing/punctuation only")
    name_set = {n.casefold() for n in names} | {t for t in _name_tokens(names)}
    rep_is_name = rep.casefold() in name_set or b in name_set
    if a in _name_tokens(names) and not rep_is_name:
        return Verdict(False, "original is a known name")
    dist = norm_lev(a, b)
    ka, kb = metaphone(orig_core), metaphone(rep)
    if rep_is_name:
        # A context name gets a looser test, never a free pass: live on
        # qwen3.5:4b (2026-09-23) an unconditional accept let the model
        # replace "dead", "What", "We're"... with "Dementors".
        if dist <= NAME_MAX_LEV or (len(ka) >= 2 and kb.startswith(ka[:2]) and dist <= 0.75):
            return Verdict(True, f"context name (lev {dist:.2f})", rep)
        return Verdict(False, f"context name but not close (lev {dist:.2f}, {ka} vs {kb})")
    if dist <= MAX_LEV:
        return Verdict(True, f"close spelling (lev {dist:.2f})", rep)
    if len(ka) >= 2 and ka == kb:
        return Verdict(True, f"same sound ({ka})", rep)
    return Verdict(False, f"not close (lev {dist:.2f}, {ka or '-'} vs {kb or '-'})")


# --- applying -----------------------------------------------------------------------------


def _replacement_words(orig: Word, rep: str, initial: bool = False) -> list[Word]:
    """`rep` in place of `orig`: same span (split by characters), same leading space and edge
    punctuation, prob 1.0 (an accepted edit is not low-confidence any more). `initial`:
    the word starts a sentence, so the replacement is capitalised too."""
    m = _EDGE.match(orig.text)
    lead, trail = (m["lead"], m["trail"]) if m else (" ", "")
    toks = rep.split()
    if initial:
        toks[0] = toks[0][:1].upper() + toks[0][1:]
    total = sum(len(t) for t in toks)
    span = max(orig.end - orig.start, 0.0)
    out, t = [], orig.start
    for n, tok in enumerate(toks):
        end = orig.end if n == len(toks) - 1 else t + span * len(tok) / total
        text = (lead if n == 0 else " ") + tok + (trail if n == len(toks) - 1 else "")
        out.append(Word(t, end, text, 1.0))
        t = end
    return out


def apply_edits(words: Sequence[Word], edits: dict[int, str]) -> list[Word]:
    """Words with `edits` (word index -> accepted replacement) applied."""
    out: list[Word] = []
    for i, w in enumerate(words):
        if i in edits:
            initial = i == 0 or _ends_sentence(words[i - 1].text)
            out.extend(_replacement_words(w, edits[i], initial))
        else:
            out.append(w)
    return out


# --- driver -------------------------------------------------------------------------------


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    if not isinstance(data, dict):
        raise ValueError("reply is not an object")
    return data


def _unload(url: str, model: str) -> None:
    """Best effort: ask Ollama to drop the model now (a request with no messages)."""
    try:
        _post(url, {"model": model, "messages": [], "keep_alive": 0}, 10.0)
    except Exception as exc:  # noqa: BLE001
        log.debug("ollama unload failed: %s", exc)


def correct(
    words: Sequence[Word],
    context: MediaContext | None,
    *,
    url: str,
    model: str,
    threshold: float = DEFAULT_FLAG_PROB,
    timeout_s: float = 120.0,
    budget_s: float = 900.0,
    cancel: threading.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[Word], dict]:
    """Proofread `words` with the LLM at `url`. Returns (words, stats); never raises for
    LLM trouble, it just keeps the ASR words."""
    t0 = clock()
    names = list(context.names) + list(context.terms) if context else []
    flags = flag_words(words, threshold, names)
    # The lines get whatever the fixed parts of the prompt leave of the token budget.
    fixed = len(SYSTEM_PROMPT) + len(_header(context)) + 300
    windows = plan_windows(words, flags, budget_chars=TOKEN_BUDGET * CHARS_PER_TOKEN - fixed)
    stats: dict = {
        "model": model,
        "flagged": len(flags),
        "windows": len(windows),
        "requests": 0,
        "accepted": 0,
        "rejected": 0,
        "errors": 0,
        "stopped": None,
        "context": context.source if context else "none",
        "reasons": {},
    }
    reasons: Counter[str] = Counter()
    accepted: dict[int, str] = {}
    unload_needed = False
    consecutive_errors = 0
    for wn, win in enumerate(windows):
        if cancel is not None and cancel.is_set():
            stats["stopped"] = "cancelled"
            break
        if clock() - t0 > budget_s:
            stats["stopped"] = "budget"
            break
        if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
            stats["stopped"] = "unreachable"
            break
        last = wn == len(windows) - 1
        body = build_request(words, win, context, model=model, last=last)
        stats["requests"] += 1
        unload_needed = not last
        try:
            reply = _post(url, body, timeout_s)
            edits = parse_edits((reply.get("message") or {}).get("content"))
        except Exception as exc:  # noqa: BLE001 - fail open on anything
            log.debug("window %d: ollama request failed: %s", wn, exc)
            edits, reply = None, None
            if last:
                unload_needed = True  # the keep_alive:0 may not have landed
        if edits is None:
            stats["errors"] += 1
            consecutive_errors += 1
            continue
        consecutive_errors = 0
        taken: set[int] = set()
        n_ok = 0
        for e in edits:
            k = e.get("i")
            if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(win.flags):
                v = Verdict(False, "index not flagged")
            elif k in taken:
                v = Verdict(False, "duplicate index")
            elif n_ok >= MAX_EDITS_PER_WINDOW:
                v = Verdict(False, "window edit cap")
            else:
                v = judge_edit(words[win.flags[k - 1]].text, e.get("text"), names=names)
            orig = words[win.flags[k - 1]].text.strip() if v.reason != "index not flagged" else "?"
            log.debug(
                "%s edit [%s] %r -> %r: %s",
                "accept" if v.ok else "reject",
                k,
                orig,
                e.get("text"),
                v.reason,
            )
            if v.ok:
                taken.add(k)
                n_ok += 1
                accepted[win.flags[k - 1]] = v.text
            else:
                if isinstance(k, int) and not isinstance(k, bool):
                    taken.add(k)
                reasons[v.reason.split(" (")[0]] += 1
    if unload_needed and stats["requests"]:
        _unload(url, model)
    stats["accepted"] = len(accepted)
    stats["rejected"] = sum(reasons.values())
    stats["reasons"] = dict(reasons)
    stats["elapsed_s"] = round(clock() - t0, 1)
    return apply_edits(words, accepted), stats


def correct_video(
    words: Sequence[Word], video, cancel: threading.Event | None = None
) -> tuple[list[Word], dict]:
    """`correct()` with the settings and the video's context. For the transcriber."""
    from app.config import settings
    from app.context import media_context

    ctx = media_context(
        video, jellyfin_url=settings.jellyfin_url, jellyfin_api_key=settings.jellyfin_api_key
    )
    new, stats = correct(
        words,
        ctx,
        url=settings.ollama_url,
        model=settings.ollama_model,
        threshold=settings.llm_flag_prob,
        timeout_s=settings.llm_timeout_s,
        budget_s=settings.llm_budget_s,
        cancel=cancel,
    )
    log.info(
        "llm correction for %s: %d flagged, %d windows, %d requests, %d accepted, "
        "%d rejected %s, %d errors, %.0fs%s",
        getattr(video, "name", video),
        stats["flagged"],
        stats["windows"],
        stats["requests"],
        stats["accepted"],
        stats["rejected"],
        stats["reasons"],
        stats["errors"],
        stats["elapsed_s"],
        f", stopped: {stats['stopped']}" if stats["stopped"] else "",
    )
    return new, stats
