"""SRT cue model plus reading and writing. Pure: no model, ffmpeg or settings."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Cue:
    """One subtitle event. `text` holds at most a few lines joined by "\\n"."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")

    @property
    def chars(self) -> int:
        """Reading-speed character count: visible characters, line breaks excluded."""
        return len(self.text.replace("\n", ""))

    @property
    def cps(self) -> float:
        return self.chars / self.duration if self.duration > 0 else float("inf")


def format_ts(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_srt(cues: Iterable[Cue]) -> str:
    out = []
    for i, cue in enumerate(cues, 1):
        out.append(f"{i}\n{format_ts(cue.start)} --> {format_ts(cue.end)}\n{cue.text}\n")
    return "\n".join(out)


_TS_LINE = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)
_TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def _secs(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str, *, strip_tags: bool = True) -> list[Cue]:
    """Lenient parser for real-world SRTs (human downloads included): BOM, CRLF,
    missing index lines, '.' millisecond separators, <i>/{\\an8} tags."""
    cues: list[Cue] = []
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            m = _TS_LINE.search(line)
            if not m:
                continue
            g = m.groups()
            body = "\n".join(t.strip() for t in lines[i + 1 :] if t.strip())
            if strip_tags:
                body = _TAG.sub("", body).strip()
            if body:
                cues.append(Cue(_secs(*g[:4]), _secs(*g[4:]), body))
            break
    cues.sort(key=lambda c: c.start)
    return cues
