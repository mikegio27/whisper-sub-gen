"""Subtitle timing/readability rules shared by the cue composer and the QA scorer.

Numbers follow https://subtitling.net/standards/subtitle-timing and
/subtitle-reading-speed (Netflix / Karamitroglou): 5/6 s minimum, 7 s maximum,
~2 frames between cues, 17 cps target (20 hard ceiling for adults), and the usual
2 lines x 42 characters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CueRules:
    max_line_chars: int = 42
    max_lines: int = 2
    min_duration: float = 0.833  # 5/6 s
    max_duration: float = 7.0
    target_cps: float = 17.0  # aim for this when there is room to linger
    max_cps: float = 20.0  # above this most viewers can't keep up
    min_gap: float = 0.083  # ~2 frames at 24 fps
    # How long a cue may stay up after the last word ends, when nothing follows
    # soon. Viewers finish reading just after the speech stops.
    max_linger: float = 1.0
    # ...and at least this long, unless the next cue needs the room. With
    # forced-aligned word ends a cue otherwise vanishes the instant speech
    # stops. Eval set, median |end error| vs human subs: 0 -> 422-571 ms,
    # 0.4 -> 277-467, 0.7 -> 234-420, 1.0 -> 157-387 ms (2026-09-24, better on
    # every film, signed bias ~0).
    min_linger: float = 1.0
    # Show a cue this long before the first word, as pro subs do ("a frame or
    # two early"; CTC word onsets also land slightly late). Never into the
    # previous cue's min_gap. Same-release refs, raw median |onset error|,
    # 0 -> 0.1 s: Hereditary 230->170, Hot Fuzz 96->75, John Wick 194->111,
    # Die Hard 111->80, Moon 57->94 ms (2026-09-24).
    lead_in: float = 0.1
    # A silence at least this long between two words always starts a new cue.
    split_pause: float = 0.6
    # A single word longer than this is a timestamp artefact (whisper/VAD
    # stretching a word across music or silence), not speech. Clamp it.
    max_word_duration: float = 2.0
    # Cues longer than this (after composition) are flagged by QA regardless.
    flag_duration: float = 10.0


DEFAULT_RULES = CueRules()
