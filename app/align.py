"""CTC forced alignment of whisper's words (docs/ROADMAP.md Phase 2).

Whisper's word timestamps come from cross-attention and are often a few hundred ms off. Here
each whisper segment is re-timed against the audio with the MMS forced-aligner acoustic model
(`MahmoudAshraf/mms-300m-1130-forced-aligner`, a wav2vec2 CTC model over 26 letters + "'"):

1. `plan_windows`: one audio window per segment, its core span +-0.5 s, clamped to the audio
   and kept from reaching far into a neighbour's core span.
2. `Aligner._emissions`: a batched forward pass (windows sorted by length, padded, and the
   padded frames trimmed off) -> per-frame log-probs, one frame per 20 ms.
3. `normalize_word`: whisper's text -> the model's alphabet (lowercase, digits spelled out with
   num2words, non-Latin scripts through uroman, punctuation dropped). Each word keeps its own
   token list, so spans map back to the original `Word` text.
4. `viterbi`: log-space CTC forced alignment over [blank, c1, blank, c2, ..., blank], with a
   `<star>` wildcard at both edges that soaks up speech whisper didn't transcribe.
5. `align_window`: token spans -> word spans -> seconds, or a fallback reason
   (no alignable chars, window too short, mean token log-prob below `min_score`), in which
   case the segment keeps whisper's timings.

Everything except `Aligner` is pure numpy and unit-tested without torch; torch, transformers,
uroman and num2words are imported lazily so `import app.align` stays cheap.
"""

from __future__ import annotations

import functools
import logging
import re
import threading
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .cues import Word

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
# wav2vec2's conv stack strides 5*2^6 = 320 samples = 20 ms per frame. The Aligner recomputes
# it from the model config; this is the default for the pure helpers.
FRAME_S = 0.02
# Context around each whisper segment. Whisper's segment edges are often a few hundred ms off,
# so the window has to be wider than the segment for the words to land where they are said.
WINDOW_PAD_S = 0.5
# How far a window may reach into a neighbouring segment's core span. The <star> token absorbs
# the neighbour's speech; the cap keeps the window from pulling this segment's first/last word
# onto the neighbour's audio.
NEIGHBOUR_OVERLAP_S = 0.25
# Log-prob of the <star> wildcard on every frame. The model has no such token, so a column is
# appended after log_softmax, as torchaudio's MMS tutorial and ctc-forced-aligner do: 0 = it
# matches anything for free, so only edge audio the text can't explain ends up in it.
STAR_LOGP = 0.0
# Mean per-token log-prob below which a segment's alignment is rejected. Calibrated 2026-09-22
# on The Big Lebowski (1,850 segments): whisper's own text scores median -1.5 (1st pct -4.7),
# another segment's text forced into the window median -4.7, text forced onto no-speech gaps
# -3.1..-6.5. Film noise makes the score a weak separator, so the cut only drops the clear
# failures: -5 rejects 0.5% (9/1850, incl. the "Transcription by CastingWords" credits
# hallucination at -7.5); onset error vs the human sub was flat for any cut in -4..-6.
DEFAULT_MIN_SCORE = -5.0

_NEG = -np.inf


# --- data -----------------------------------------------------------------------------------


@dataclass
class AlignSegment:
    """One whisper segment: its (predicted) span and its words."""

    start: float
    end: float
    words: list[Word]


@dataclass
class AlignStats:
    segments: int = 0
    aligned: int = 0
    fallback: int = 0
    reasons: Counter = field(default_factory=Counter)
    words: int = 0
    elapsed_s: float = 0.0
    emissions_s: float = 0.0
    viterbi_s: float = 0.0
    mean_score: float | None = None

    def as_dict(self) -> dict:
        return {
            "segments": self.segments,
            "aligned": self.aligned,
            "fallback": self.fallback,
            "reasons": dict(self.reasons),
            "words": self.words,
            "elapsed_s": round(self.elapsed_s, 2),
            "emissions_s": round(self.emissions_s, 2),
            "viterbi_s": round(self.viterbi_s, 2),
            "mean_score": None if self.mean_score is None else round(self.mean_score, 3),
        }


@dataclass
class AlignResult:
    words: list[Word]
    stats: AlignStats


# --- windows --------------------------------------------------------------------------------


def _core(seg: AlignSegment) -> tuple[float, float]:
    """The span the segment claims: its own edges widened to cover its words."""
    lo, hi = seg.start, seg.end
    if seg.words:
        lo = min(lo, min(w.start for w in seg.words))
        hi = max(hi, max(w.end for w in seg.words))
    return lo, hi


def plan_windows(
    segments: Sequence[AlignSegment],
    total_s: float,
    *,
    pad: float = WINDOW_PAD_S,
    overlap: float = NEIGHBOUR_OVERLAP_S,
) -> list[tuple[float, float]]:
    """The audio window (seconds) to align each segment in.

    [core start - pad, core end + pad], clamped to [0, total_s], and reaching at most `overlap`
    into the previous/next segment's core span (never cutting into this segment's own core)."""
    cores = [_core(s) for s in segments]
    out = []
    for i, (lo, hi) in enumerate(cores):
        w_lo, w_hi = lo - pad, hi + pad
        if i > 0:
            w_lo = max(w_lo, min(lo, cores[i - 1][1] - overlap))
        if i + 1 < len(cores):
            w_hi = min(w_hi, max(hi, cores[i + 1][0] + overlap))
        w_lo = max(0.0, w_lo)
        w_hi = min(total_s, w_hi)
        out.append((w_lo, max(w_lo, w_hi)))
    return out


# --- text normalisation ---------------------------------------------------------------------

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})
# 1,000 / 3.5 / 21st / 1984
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:st|nd|rd|th)?")
_ORDINAL = re.compile(r"(?:st|nd|rd|th)$")


@functools.cache
def _num2words():
    try:
        from num2words import num2words
    except ImportError:
        log.warning("num2words not installed; words with digits won't be aligned")
        return None
    return num2words


@functools.cache
def _uroman():
    try:
        import uroman
    except ImportError:
        log.warning("uroman not installed; non-Latin words won't be aligned")
        return None
    return uroman.Uroman()


# Languages whose num2words converter was checked (2026-09-23) to return, without
# raising, for cardinals, ordinals and decimals up to 9 digits. Others can raise
# TypeError/IndexError, or **hang forever** (am: 7-digit numbers), inside the
# aligner lock where cancel can't reach. Anything else keeps its digits, which
# the aligner places in the gap between aligned neighbours.
_NUM2WORDS_LANGS = frozenset("en es fr de it pt nl sv da no fi pl ru uk ja tr he hu ro id".split())
_MAX_NUMBER_DIGITS = 9


def _spell_number(m: re.Match, language: str) -> str:
    raw = m.group(0)
    if language not in _NUM2WORDS_LANGS:
        return raw
    n2w = _num2words()
    if n2w is None:
        return raw
    ordinal = bool(_ORDINAL.search(raw))
    digits = _ORDINAL.sub("", raw).replace(",", "")
    if sum(c.isdigit() for c in digits) > _MAX_NUMBER_DIGITS:
        return raw
    try:
        if ordinal:
            spoken = n2w(int(digits), lang=language, to="ordinal")
        elif "." in digits:
            spoken = n2w(float(digits), lang=language)
        elif len(digits) == 4 and 1100 <= int(digits) <= 2099 and language == "en":
            spoken = n2w(int(digits), lang=language, to="year")  # "nineteen eighty-four"
        else:
            spoken = n2w(int(digits), lang=language)
    except Exception:  # noqa: BLE001 - a third-party converter; never let it sink alignment
        return raw
    return f" {spoken} "


def _strip_marks(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_word(
    text: str, language: str = "en", alphabet: str = "abcdefghijklmnopqrstuvwxyz'"
) -> str:
    """Whisper word text -> the characters the aligner can emit ("" = unalignable).

    Lowercase, digits spelled out (num2words, in `language`), accents stripped, other scripts
    romanized (uroman), everything outside `alphabet` dropped. Apostrophes survive only inside a
    word ("don't"), since at the edges they are almost always quote marks."""
    t = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).lower().strip()
    if not t:
        return ""
    if any(c.isdigit() for c in t):
        t = _NUMBER.sub(lambda m: _spell_number(m, language or "en"), t).lower()
    t = _strip_marks(t)
    allowed = set(alphabet)
    if any(c.isalpha() and c not in allowed for c in t):
        ur = _uroman()
        if ur is not None:
            t = _strip_marks(ur.romanize_string(t).lower())
    out = "".join(c for c in t if c in allowed)
    out = re.sub(r"'{2,}", "'", out).strip("'")
    return out


# --- CTC forced alignment -------------------------------------------------------------------


def min_frames(tokens: Sequence[int]) -> int:
    """Fewest frames a CTC path needs to emit `tokens`: one per token, plus a blank between two
    identical neighbours ("ll" can't be told from "l" otherwise)."""
    return len(tokens) + sum(1 for a, b in zip(tokens, tokens[1:], strict=False) if a == b)


def viterbi(logp: np.ndarray, tokens: Sequence[int], blank: int = 0) -> np.ndarray | None:
    """Best CTC path emitting `tokens` through the (T, V) log-prob matrix `logp`.

    Returns, per frame, the index into `tokens` that frame emits, or -1 for blank; None when no
    path exists (T < min_frames). Standard trellis over the extended sequence
    [blank, t0, blank, t1, ..., blank]: stay, advance one, or skip a blank between two
    different tokens; the path starts in state 0 or 1 and ends in the last blank or last token.
    """
    n_frames = logp.shape[0]
    n_tok = len(tokens)
    if n_tok == 0 or n_frames < min_frames(tokens):
        return None
    n_states = 2 * n_tok + 1
    ext = np.full(n_states, blank, dtype=np.int64)
    ext[1::2] = tokens
    em = logp[:, ext].astype(np.float64)  # (T, S)
    # A skip (s-2 -> s) is allowed only into a token state whose token differs from the
    # previous token; blanks are never skipped over into.
    no_skip = np.ones(n_states, dtype=bool)
    no_skip[3::2] = ext[3::2] == ext[1:-2:2]

    alpha = np.full(n_states, _NEG)
    alpha[0] = em[0, 0]
    alpha[1] = em[0, 1]
    back = np.zeros((n_frames, n_states), dtype=np.int8)
    prev1 = np.full(n_states, _NEG)
    prev2 = np.full(n_states, _NEG)
    for t in range(1, n_frames):
        prev1[1:] = alpha[:-1]
        prev2[2:] = alpha[:-2]
        prev2[no_skip] = _NEG
        choice = (prev1 > alpha).astype(np.int8)
        best = np.where(choice == 1, prev1, alpha)
        skip = prev2 > best
        choice[skip] = 2
        best = np.where(skip, prev2, best)
        back[t] = choice
        alpha = best + em[t]

    end = n_states - 1 if alpha[-1] >= alpha[-2] else n_states - 2
    if not np.isfinite(alpha[end]):
        return None
    path = np.empty(n_frames, dtype=np.int64)
    s = end
    for t in range(n_frames - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])  # int(): int8 arithmetic would overflow past 127 states
    return np.where(path % 2 == 1, (path - 1) // 2, -1)


def token_spans(path: np.ndarray, n_tokens: int) -> list[tuple[int, int]]:
    """Per token: (first frame, last frame) it is emitted on."""
    frames = np.arange(len(path))
    spans = []
    for k in range(n_tokens):
        idx = frames[path == k]
        spans.append((int(idx[0]), int(idx[-1])))
    return spans


def with_star(logp: np.ndarray, star_logp: float = STAR_LOGP) -> tuple[np.ndarray, int]:
    """Append the <star> wildcard column; returns (matrix, star id)."""
    col = np.full((logp.shape[0], 1), star_logp, dtype=logp.dtype)
    return np.concatenate([logp, col], axis=1), logp.shape[1]


@dataclass
class WindowAlignment:
    spans: list[tuple[int, int] | None] | None  # per word, frames; None = word unalignable
    score: float | None  # mean per-token log-prob of the text tokens
    reason: str | None  # why the window fell back, or None


def align_window(
    logp: np.ndarray,
    word_tokens: Sequence[Sequence[int]],
    *,
    blank: int = 0,
    star: bool = True,
    min_score: float = DEFAULT_MIN_SCORE,
) -> WindowAlignment:
    """Align one window's words. `word_tokens[i]` are word i's token ids (may be empty)."""
    flat = [t for toks in word_tokens for t in toks]
    if not flat:
        return WindowAlignment(None, None, "no_tokens")
    offset = 0
    if star:
        logp, star_id = with_star(logp)
        seq = [star_id, *flat, star_id]
        offset = 1
    else:
        seq = flat
    if logp.shape[0] < min_frames(seq):
        return WindowAlignment(None, None, "too_short")
    path = viterbi(logp, seq, blank)
    if path is None:
        return WindowAlignment(None, None, "too_short")
    spans = token_spans(path, len(seq))[offset : offset + len(flat)]
    frames = np.arange(len(path))
    scores = []
    for k, tok in enumerate(flat):
        sel = frames[path == k + offset]
        scores.append(float(logp[sel, tok].mean()))
    score = float(np.mean(scores))
    if score < min_score:
        return WindowAlignment(None, score, "low_score")
    per_word: list[tuple[int, int] | None] = []
    i = 0
    for toks in word_tokens:
        if not toks:
            per_word.append(None)
            continue
        per_word.append((spans[i][0], spans[i + len(toks) - 1][1]))
        i += len(toks)
    return WindowAlignment(per_word, score, None)


def spans_to_words(
    words: Sequence[Word],
    spans: Sequence[tuple[int, int] | None],
    window_start: float,
    *,
    frame_s: float = FRAME_S,
    window_end: float | None = None,
) -> list[Word]:
    """Frame spans -> the original words with new start/end (seconds).

    A word spans [first frame, last frame + 1) of its tokens. A word with no span (only
    digits/punctuation) sits in the gap between its aligned neighbours."""
    times: list[tuple[float, float] | None] = []
    for sp in spans:
        if sp is None:
            times.append(None)
            continue
        start = window_start + sp[0] * frame_s
        end = window_start + (sp[1] + 1) * frame_s
        if window_end is not None:
            end = min(end, window_end)
        times.append((start, end))
    out = []
    for i, (w, tm) in enumerate(zip(words, times, strict=True)):
        if tm is None:
            prev_end = next((t[1] for t in reversed(times[:i]) if t), None)
            next_start = next((t[0] for t in times[i + 1 :] if t), None)
            lo = prev_end if prev_end is not None else next_start
            hi = next_start if next_start is not None else prev_end
            tm = (lo, max(lo, hi))
        out.append(Word(tm[0], tm[1], w.text, w.prob))
    return out


def enforce_order(words: Sequence[Word]) -> list[Word]:
    """Neighbouring windows overlap a little, so the first word of one segment can land before
    the last of the previous; clamp so starts never go back in time."""
    out: list[Word] = []
    prev_end = -np.inf
    for w in words:
        if w.start < prev_end:
            start = prev_end
            w = Word(start, max(w.end, start), w.text, w.prob)
        out.append(w)
        prev_end = max(prev_end, w.end)
    return out


def frames_for_samples(n: int, kernels: Sequence[int], strides: Sequence[int]) -> int:
    """Output frames of a conv stack (no padding) for an input of n samples."""
    for k, s in zip(kernels, strides, strict=True):
        n = (n - k) // s + 1
    return max(n, 0)


def plan_batches(lengths: Sequence[int], budget: int, max_items: int = 32) -> list[list[int]]:
    """Group window indices into batches, longest first, so that padded size
    (count x longest) stays within `budget` samples. A window longer than the budget runs alone."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        longest = lengths[cur[0]] if cur else lengths[i]
        if cur and ((len(cur) + 1) * longest > budget or len(cur) >= max_items):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


# --- model ----------------------------------------------------------------------------------


class Aligner:
    """Owns the (lazily loaded) MMS aligner model. `free()` drops it between jobs."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        model_dir: str | None = None,
        batch_seconds: float = 120.0,
        min_score: float = DEFAULT_MIN_SCORE,
        cpu_threads: int = 0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model_dir = model_dir
        self.batch_seconds = batch_seconds
        self.min_score = min_score
        self.cpu_threads = cpu_threads
        self._lock = threading.Lock()
        self._model = None
        self._feature_extractor = None
        self._vocab: dict[str, int] = {}
        self._blank = 0
        self._frame_s = FRAME_S
        self._conv: tuple[list[int], list[int]] = ([], [])

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        t0 = time.time()
        if self.device == "cpu" and self.cpu_threads:
            torch.set_num_threads(self.cpu_threads)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        kw = {"cache_dir": self.model_dir}
        model = Wav2Vec2ForCTC.from_pretrained(self.model_name, dtype=dtype, **kw)
        model.to(self.device).eval()
        tok = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name, **kw)
        fe = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name, **kw)
        vocab = tok.get_vocab()
        self._blank = vocab.get("<blank>", model.config.pad_token_id)
        self._vocab = {k: v for k, v in vocab.items() if len(k) == 1}
        cfg = model.config
        self._conv = (list(cfg.conv_kernel), list(cfg.conv_stride))
        self._frame_s = float(np.prod(cfg.conv_stride)) / fe.sampling_rate
        self._feature_extractor = fe
        self._model = model
        log.info(
            "aligner %s loaded on %s (%s) in %.1fs; %d chars, frame %.0f ms",
            self.model_name,
            self.device,
            dtype,
            time.time() - t0,
            len(self._vocab),
            self._frame_s * 1000,
        )

    def free(self) -> None:
        """Release the model (and its VRAM on a shared card)."""
        with self._lock:
            if self._model is None:
                return
            self._model = None
            self._feature_extractor = None
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    # The cuBLAS workspace pins a cached block that
                    # empty_cache() can't release: ~0.4 GB stayed reserved on
                    # the shared card between jobs until this was added.
                    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
                    if clear is not None:
                        clear()
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            log.info("aligner model freed")

    def _emissions(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        """Log-probs (T_i, V) for each window, from one padded forward pass."""
        import torch

        fe, model = self._feature_extractor, self._model
        batch = fe(
            windows,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        values = batch["input_values"].to(self.device, dtype=model.dtype)
        mask = batch["attention_mask"].to(self.device)
        with torch.inference_mode():
            logits = model(values, attention_mask=mask).logits
            logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        out = []
        for i, w in enumerate(windows):
            n = frames_for_samples(len(w), *self._conv)
            out.append(logp[i, :n])  # drop the frames that only saw padding
        return out

    def _tokens(self, words: Sequence[Word], language: str) -> list[list[int]]:
        alphabet = "".join(self._vocab)
        return [[self._vocab[c] for c in normalize_word(w.text, language, alphabet)] for w in words]

    def align(
        self,
        audio: np.ndarray,
        segments: Sequence[AlignSegment],
        language: str,
        *,
        check_cancel: Callable[[], None] | None = None,
    ) -> AlignResult:
        """Re-time every segment's words. A segment that can't be aligned keeps whisper's
        timings (counted in stats). `check_cancel` is called between batches and may raise."""
        t_start = time.time()
        with self._lock:
            self._load()
            stats = AlignStats(segments=len(segments))
            total_s = len(audio) / SAMPLE_RATE
            windows = plan_windows(segments, total_s)
            bounds = [(int(lo * SAMPLE_RATE), int(hi * SAMPLE_RATE)) for lo, hi in windows]
            tokens = [self._tokens(s.words, language) for s in segments]
            result: list[list[Word] | None] = [None] * len(segments)
            # Segments with nothing to align skip the forward pass entirely.
            todo = []
            for i, seg in enumerate(segments):
                stats.words += len(seg.words)
                if not any(tokens[i]):
                    stats.reasons["no_tokens"] += 1
                elif frames_for_samples(bounds[i][1] - bounds[i][0], *self._conv) < 1:
                    stats.reasons["too_short"] += 1  # shorter than the conv receptive field
                else:
                    todo.append(i)
            lengths = [bounds[i][1] - bounds[i][0] for i in todo]
            budget = int(self.batch_seconds * SAMPLE_RATE)
            scores = []
            for batch in plan_batches(lengths, budget):
                if check_cancel is not None:
                    check_cancel()
                idx = [todo[j] for j in batch]
                t0 = time.time()
                ems = self._emissions([audio[bounds[i][0] : bounds[i][1]] for i in idx])
                stats.emissions_s += time.time() - t0
                t0 = time.time()
                for i, em in zip(idx, ems, strict=True):
                    wa = align_window(em, tokens[i], blank=self._blank, min_score=self.min_score)
                    if wa.reason is not None:
                        stats.reasons[wa.reason] += 1
                        if wa.score is not None:
                            log.debug("segment %d: %s (%.2f)", i, wa.reason, wa.score)
                        continue
                    scores.append(wa.score)
                    result[i] = spans_to_words(
                        segments[i].words,
                        wa.spans,
                        windows[i][0],
                        frame_s=self._frame_s,
                        window_end=windows[i][1],
                    )
                stats.viterbi_s += time.time() - t0

        words: list[Word] = []
        for i, seg in enumerate(segments):
            if result[i] is None:
                stats.fallback += 1
                words.extend(seg.words)
            else:
                stats.aligned += 1
                words.extend(result[i])
        stats.mean_score = float(np.mean(scores)) if scores else None
        stats.elapsed_s = time.time() - t_start
        return AlignResult(enforce_order(words), stats)
