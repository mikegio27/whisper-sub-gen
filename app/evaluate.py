"""Compare a generated (hypothesis) SRT with a human (reference) SRT of the same video.

The two subs segment the speech differently, may be shifted against each other (a different
release or cut) and the human one carries things we never transcribe (SDH tags such as
"[DOOR SLAMS]" or "MAN:", song lyrics, dialogue dashes). So nothing is compared cue by cue.
Instead both files become one normalised word stream each, the streams are aligned with
difflib, and the metrics come from that alignment:

- `wer_approx`: (substitutions + deletions + insertions) / reference words, counted from the
  difflib opcodes. Approximate because SequenceMatcher greedily takes the longest common runs
  rather than computing a minimum edit distance, so it can report slightly more edits than true
  Levenshtein WER; a "replace" of n ref vs m hyp words counts min(n, m) substitutions plus the
  surplus as deletions/insertions. The normalisation (cheap number/contraction folding, SDH
  stripping) also isn't a full WER normaliser, and human subs paraphrase and condense speech,
  so against a human reference this measures "agreement", with a floor well above 0.
- onset/offset error: for cues where *both* subs start (end) on the same aligned word, the
  difference between the two cue starts (ends). The median signed onset error is the global
  offset between the releases; the residual errors after removing it are the timing quality.
- drift: that offset over the first vs the last third of the film. A framerate mismatch
  (23.976 vs 25 fps) shows up here; it is reported, not corrected.

`compare()` is pure. CLI: `python -m app.evaluate HYP.srt REF.srt [--json]`, which also prints
`qa.score` for both files side by side. This module must not import app.worker/app.api/app.main
(they construct the Worker at import) or need settings.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from dataclasses import dataclass

from app import qa
from app.srt import Cue

# An aligned word only anchors a timing measurement if it sits in a matched run at least this
# long. A lone matched "you" between two mismatches is as likely coincidence as a real anchor.
MIN_ANCHOR_RUN = 2

WITHIN_MS = (100, 250, 500)

# SDH / non-speech markup that human subs carry and ASR never produces.
_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)", re.S)
# "MAN:", "HARRY:", "MAN 2:", "MRS. WEASLEY:" at the start of a line. Uppercase only, because
# "Look: ..." in normal mixed-case dialogue is speech, not a label.
_SPEAKER = re.compile(r"^[A-Z][A-Z0-9 .'\-]{1,30}:\s*")
_LEAD_DASH = re.compile(r"^[\-–—]+\s*")
_MUSIC = ("♪", "♫", "#")
_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_APOS = re.compile(r"['’]")

# Hesitations: humans usually leave them out, whisper sometimes writes them. Either way they
# are not the words we are trying to get right.
_FILLERS = frozenset({"uh", "um", "umm", "er", "erm", "ah", "hmm", "mm", "mhm", "huh", "eh"})
# Cheap spelling equivalences. Contractions are folded by dropping the apostrophe
# ("don't" == "dont"), not by expanding them, which would need a real normaliser.
_EQUIV = {"ok": "okay", "mr": "mister", "mrs": "missus"}

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n: int) -> list[str]:
    """0..999_999 as words ("21" -> twenty one), so "21" in one sub matches "twenty-one" in
    the other. Larger numbers are left as digits: rare in dialogue and not worth a library."""
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + (_num_words(n % 10) if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (_num_words(n % 100) if n % 100 else [])
    return _num_words(n // 1000) + ["thousand"] + (_num_words(n % 1000) if n % 1000 else [])


def normalize_word(word: str) -> list[str]:
    """One surface word -> zero or more comparison tokens."""
    w = _APOS.sub("", word.lower())
    if w.isdigit():
        n = int(w)
        return _num_words(n) if n < 1_000_000 else [w]
    if w in _FILLERS:
        return []
    return [_EQUIV.get(w, w)]


def clean_text(text: str) -> str:
    """Strip what a human sub has and speech recognition doesn't: [..]/(..) tags, lyric lines,
    speaker labels and dialogue dashes. Lines are joined with spaces."""
    text = _BRACKETED.sub(" ", text)
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if any(m in line for m in _MUSIC):
            continue
        line = _LEAD_DASH.sub("", line)
        line = _SPEAKER.sub("", line)
        line = _LEAD_DASH.sub("", line)  # "- MAN: -Hey" style doubles
        if line:
            out.append(line)
    return _DIGIT_COMMA.sub("", " ".join(out))


@dataclass(frozen=True)
class Token:
    word: str
    time: float  # estimated: cue start, interpolated by character offset within the cue
    cue: int  # index into the (sorted) cue list
    first: bool  # first token of its cue
    last: bool  # last token of its cue


def tokenize(cues: list[Cue]) -> list[Token]:
    """Cue list (sorted by start) -> timed, normalised word stream. Only cue-level times
    exist in an SRT, so a word's time is its cue start plus its share of the cue by
    character offset; the first word of a cue gets exactly the cue start."""
    tokens: list[Token] = []
    for ci, cue in enumerate(cues):
        text = clean_text(cue.text)
        pieces: list[tuple[str, float]] = []
        for m in _WORD.finditer(text):
            t = cue.start + cue.duration * m.start() / max(len(text), 1)
            pieces += [(w, t) for w in normalize_word(m.group())]
        for k, (w, t) in enumerate(pieces):
            tokens.append(Token(w, t, ci, k == 0, k == len(pieces) - 1))
    return tokens


def _err_stats(diffs: list[float], offset: float | None) -> dict:
    """Signed diffs (s) -> median signed plus abs-error stats, raw and with `offset` removed."""

    def block(xs: list[float]) -> dict:
        ab = [abs(x) for x in xs]
        d = {
            "median_signed": statistics.median(xs) if xs else None,
            "median_abs": statistics.median(ab) if ab else None,
            "p90_abs": qa.percentile(ab, 90) if ab else None,
        }
        for ms in WITHIN_MS:
            # Whole-ms compare: SRT times are ms-quantised, float noise shouldn't flip a bucket.
            d[f"within_{ms}"] = (
                100 * sum(round(a * 1000) <= ms for a in ab) / len(ab) if ab else None
            )
        return d

    corrected = [x - offset for x in diffs] if offset is not None else []
    return {"n": len(diffs), "raw": block(diffs), "corrected": block(corrected)}


def compare(hyp: list[Cue], ref: list[Cue]) -> dict:
    """Align hyp against ref and return WER-style and timing metrics (see module docstring).
    Timing values are in seconds, `within_*` and ratios in percent / 0..1 as named."""
    hyp = sorted(hyp, key=lambda c: c.start)
    ref = sorted(ref, key=lambda c: c.start)
    rt, ht = tokenize(ref), tokenize(hyp)

    # autojunk=False: with the default, any word in >1% of a 10k-word stream ("the", "you")
    # is treated as junk and never matched, which wrecks the alignment. Measured at <1 s for
    # 20k words vs 20k words, so no chunking is needed.
    sm = difflib.SequenceMatcher(None, [t.word for t in rt], [t.word for t in ht], autojunk=False)
    subs = dels = ins = matched = 0
    onsets: list[tuple[float, float]] = []  # (ref cue start, hyp start - ref start)
    ends: list[float] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
            if i2 - i1 < MIN_ANCHOR_RUN:
                continue
            for r, h in zip(rt[i1:i2], ht[j1:j2], strict=True):
                rc, hc = ref[r.cue], hyp[h.cue]
                if r.first and h.first:
                    onsets.append((rc.start, hc.start - rc.start))
                if r.last and h.last:
                    ends.append(hc.end - rc.end)
        elif tag == "replace":
            n_ref, n_hyp = i2 - i1, j2 - j1
            subs += min(n_ref, n_hyp)
            dels += max(0, n_ref - n_hyp)
            ins += max(0, n_hyp - n_ref)
        elif tag == "delete":
            dels += i2 - i1
        elif tag == "insert":
            ins += j2 - j1

    onset_diffs = [d for _, d in onsets]
    # One global offset for both onset and end errors: it is a property of the two releases,
    # so a systematic "our cues linger too long" bias stays visible in the end errors instead
    # of being subtracted away by their own median.
    if onset_diffs:
        offset = statistics.median(onset_diffs)
    elif ends:
        offset = statistics.median(ends)
    else:
        offset = None

    n_ref = len(rt)
    return {
        "ref_cues": len(ref),
        "hyp_cues": len(hyp),
        "ref_words": n_ref,
        "hyp_words": len(ht),
        "matched_ratio": matched / n_ref if n_ref else 0.0,
        "wer_approx": (subs + dels + ins) / n_ref if n_ref else 0.0,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "global_offset": offset,
        "onset": _err_stats(onset_diffs, offset),
        "offset": _err_stats(ends, offset),
        "drift": _drift(onsets, ref),
    }


def _drift(onsets: list[tuple[float, float]], ref: list[Cue]) -> dict:
    """Median onset diff over the first vs the last third of the reference's time span."""
    out: dict = {"first_third": None, "last_third": None, "drift": None, "n_first": 0, "n_last": 0}
    if not onsets or not ref:
        return out
    t0 = ref[0].start
    span = max(c.end for c in ref) - t0
    first = [d for t, d in onsets if t < t0 + span / 3]
    last = [d for t, d in onsets if t >= t0 + 2 * span / 3]
    out["n_first"], out["n_last"] = len(first), len(last)
    if first:
        out["first_third"] = statistics.median(first)
    if last:
        out["last_third"] = statistics.median(last)
    if first and last:
        out["drift"] = out["last_third"] - out["first_third"]
    return out


def _fmt_s(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 1000:+.0f} ms"


def _fmt_abs(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 1000:.0f} ms"


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def format_comparison(r: dict) -> str:
    lines = [
        f"words        ref {r['ref_words']}  hyp {r['hyp_words']}  "
        f"(cues ref {r['ref_cues']}  hyp {r['hyp_cues']})",
        f"matched      {100 * r['matched_ratio']:.1f}% of ref words",
        f"wer_approx   {100 * r['wer_approx']:.1f}%  (S {r['substitutions']}  "
        f"D {r['deletions']}  I {r['insertions']})",
        f"global off.  {_fmt_s(r['global_offset'])}  (median onset diff, hyp - ref)",
    ]
    d = r["drift"]
    lines.append(
        f"drift        first third {_fmt_s(d['first_third'])} (n {d['n_first']})  "
        f"last third {_fmt_s(d['last_third'])} (n {d['n_last']})  -> {_fmt_s(d['drift'])}"
    )
    for name in ("onset", "offset"):
        s = r[name]
        lines.append(f"{name} error (n {s['n']})")
        for label, key in (("after offset removal", "corrected"), ("raw", "raw")):
            b = s[key]
            within = "  ".join(f"<={ms}ms {_fmt_pct(b[f'within_{ms}'])}" for ms in WITHIN_MS)
            lines.append(
                f"  {label:<21} median {_fmt_s(b['median_signed'])}  "
                f"|median| {_fmt_abs(b['median_abs'])}  p90 {_fmt_abs(b['p90_abs'])}  {within}"
            )
    return "\n".join(lines)


def format_qa_side_by_side(hyp_score: dict, ref_score: dict) -> str:
    h, r = qa.report_rows(hyp_score), qa.report_rows(ref_score)
    width = max(len(label) for label, _ in h)
    vw = max(len(v) for _, v in h + [("", "hyp")])
    out = [f"{'':<{width}}  {'hyp':<{vw}}  ref"]
    out += [f"{a:<{width}}  {hv:<{vw}}  {rv}" for (a, hv), (_, rv) in zip(h, r, strict=True)]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m app.evaluate", description="Compare a generated SRT with a human one."
    )
    ap.add_argument("hyp", help="generated subtitle file")
    ap.add_argument("ref", help="reference (human) subtitle file of the same video")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    args = ap.parse_args(argv)

    hyp, ref = qa.load_srt(args.hyp), qa.load_srt(args.ref)
    result = compare(hyp, ref)
    scores = {"hyp": qa.score(hyp), "ref": qa.score(ref)}
    if args.json:
        print(json.dumps({"compare": result, "qa": scores}, indent=2))
    else:
        print(format_comparison(result))
        print()
        print(format_qa_side_by_side(scores["hyp"], scores["ref"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
