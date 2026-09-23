"""Standards score for a cue list: how often a sub breaks the rules in `standards.CueRules`.

Pure (no model, ffmpeg or settings), so it can score our output, a human download from the
library, or a cue list in a test the same way. `python -m app.qa FILE.srt [--json]` prints the
report for one file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

from app.srt import Cue, parse_srt
from app.standards import DEFAULT_RULES, CueRules

# Counts that go into `violations_per_100`. These are the hard-rule breaks a viewer notices.
# Left out on purpose: `cps_over_target` (a soft goal, and a superset of `cps_over_max`),
# `very_long` (a subset of `too_long`), `overlap` (a subset of `touching_or_overlap`) and
# `whole_second_durations` (a symptom of the old pipeline, not a viewer-facing rule). Adding
# the subsets would count the same cue twice for one problem.
VIOLATION_KEYS = (
    "too_short",
    "too_long",
    "cps_over_max",
    "touching_or_overlap",
    "line_too_long",
    "too_many_lines",
    "consecutive_duplicate_text",
)

COUNT_KEYS = (
    "too_short",
    "too_long",
    "very_long",
    "cps_over_target",
    "cps_over_max",
    "touching_or_overlap",
    "overlap",
    "line_too_long",
    "too_many_lines",
    "consecutive_duplicate_text",
    "whole_second_durations",
)

_NON_WORD = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def load_srt(path: str | Path) -> list[Cue]:
    """Read an SRT from disk. Human downloads are often cp1252 rather than UTF-8, so fall back
    instead of failing; latin-1 decodes any byte, so the last attempt never raises."""
    data = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return parse_srt(data.decode(enc))
        except UnicodeDecodeError:
            continue
    raise AssertionError("unreachable: latin-1 decodes any byte string")


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100). 0.0 for empty input, so reports on an
    empty file stay printable and JSON-safe."""
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo = math.floor(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _norm_text(text: str) -> str:
    return _SPACE.sub(" ", _NON_WORD.sub("", text.lower())).strip()


def score(cues: list[Cue], rules: CueRules = DEFAULT_RULES) -> dict:
    """Count rule violations. Returns {"cues", "counts", "rates", "violations_per_100",
    "stats"}; rates are count / cues (0.0 when there are no cues)."""
    cues = sorted(cues, key=lambda c: c.start)
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for i, cue in enumerate(cues):
        dur = cue.duration
        if dur < rules.min_duration:
            counts["too_short"] += 1
        if dur > rules.max_duration:
            counts["too_long"] += 1
        if dur > rules.flag_duration:
            counts["very_long"] += 1
        # Cue.cps is inf for a zero/negative duration, which correctly counts as unreadable.
        if cue.cps > rules.target_cps:
            counts["cps_over_target"] += 1
        if cue.cps > rules.max_cps:
            counts["cps_over_max"] += 1
        if i + 1 < len(cues):
            gap = cues[i + 1].start - cue.end
            # Compare in whole ms: SRT times are ms-quantised, and float noise must not turn
            # an exact 83 ms gap into a violation.
            if round(gap * 1000) < round(rules.min_gap * 1000):
                counts["touching_or_overlap"] += 1
            if round(gap * 1000) < 0:
                counts["overlap"] += 1
            if _norm_text(cue.text) and _norm_text(cue.text) == _norm_text(cues[i + 1].text):
                counts["consecutive_duplicate_text"] += 1
        if any(len(line) > rules.max_line_chars for line in cue.lines):
            counts["line_too_long"] += 1
        if len(cue.lines) > rules.max_lines:
            counts["too_many_lines"] += 1
        # Whisper's segment timestamps come out on a coarse grid, so an exact 1.000/2.000 s
        # duration is a tell that the timing was predicted rather than measured.
        if dur > 0 and abs(dur - round(dur)) <= 0.001:
            counts["whole_second_durations"] += 1

    n = len(cues)
    durations = [c.duration for c in cues]
    cps = [c.cps for c in cues if c.duration > 0]
    return {
        "cues": n,
        "counts": counts,
        "rates": {k: (v / n if n else 0.0) for k, v in counts.items()},
        "violations_per_100": (100 * sum(counts[k] for k in VIOLATION_KEYS) / n if n else 0.0),
        "stats": {
            "median_duration": statistics.median(durations) if durations else 0.0,
            "p95_duration": percentile(durations, 95),
            "max_duration": max(durations, default=0.0),
            "median_cps": statistics.median(cps) if cps else 0.0,
        },
    }


def report_rows(s: dict) -> list[tuple[str, str]]:
    """(label, value) pairs for a score, shared by format_report and evaluate's side-by-side."""
    rows = [("cues", str(s["cues"]))]
    for k in COUNT_KEYS:
        rows.append((k, f"{s['counts'][k]} ({100 * s['rates'][k]:.1f}%)"))
    rows.append(("violations_per_100", f"{s['violations_per_100']:.1f}"))
    st = s["stats"]
    rows += [
        ("median_duration", f"{st['median_duration']:.2f} s"),
        ("p95_duration", f"{st['p95_duration']:.2f} s"),
        ("max_duration", f"{st['max_duration']:.2f} s"),
        ("median_cps", f"{st['median_cps']:.1f}"),
    ]
    return rows


def format_report(s: dict) -> str:
    rows = report_rows(s)
    width = max(len(label) for label, _ in rows)
    lines = [f"{label:<{width}}  {value}" for label, value in rows]
    lines.append(f"(violations_per_100 = {' + '.join(VIOLATION_KEYS)})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.qa", description=__doc__)
    ap.add_argument("srt", help="subtitle file to score")
    ap.add_argument("--json", action="store_true", help="print the score as JSON")
    args = ap.parse_args(argv)
    s = score(load_srt(args.srt))
    print(json.dumps(s, indent=2) if args.json else format_report(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
