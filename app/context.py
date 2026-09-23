"""What is this video? Title, year and the names people say in it, for the LLM pass (P3).

Best effort and never raises: `media_context()` always returns a `MediaContext`, at worst one
with only the title parsed from the path.

1. The path, Jellyfin naming: movies `Title (Year)/Title (Year).mkv`, episodes
   `Show (Year)/Season N/Show - S01E02 - Episode title.mkv` (scene-style `Show.S01E02.x264` too).
2. If a Jellyfin URL and API key are given, the item's `People` (character names from the
   actors' roles) and `Overview` (capitalised words -> extra terms). Jellyfin's `/Items` has no
   path filter, so we search by name (+ year) and pick the result whose `Path` is ours; Jellyfin
   and this service mount the same NFS export at `/media`, so paths compare equal.

Results are cached per path (and series lookups per series) for the life of the process, so the
episodes of one show cost one series query.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import PurePath

log = logging.getLogger(__name__)

_TIMEOUT_S = 10.0
_MAX_NAMES = 60
_MAX_TERMS = 30
_CACHE_MAX = 128


@dataclass(frozen=True)
class MediaContext:
    title: str = ""  # film title, or the show name for an episode
    year: int | None = None
    season: int | None = None
    episode: int | None = None
    episode_title: str = ""
    names: tuple[str, ...] = ()  # character names (what gets said), most important first
    terms: tuple[str, ...] = ()  # other proper nouns (from the overview)
    overview: str = ""
    source: str = "path"  # "path" | "jellyfin"

    @property
    def is_episode(self) -> bool:
        return self.season is not None and self.episode is not None


# --- path parsing ------------------------------------------------------------------------

_YEAR = re.compile(r"^(?P<title>.+?)\s*[(\[](?P<year>(?:19|20)\d{2})[)\]]")
_BARE_YEAR = re.compile(r"^(?P<title>.+?)[\s.]+(?P<year>(?:19|20)\d{2})(?:[\s.]|$)")
_SXXEYY = re.compile(
    r"^(?P<show>.*?)[\s._-]*\b[Ss](?P<s>\d{1,2})[\s._-]?[Ee](?P<e>\d{1,3})(?:-?[Ee]\d{1,3})*"
    r"(?:[\s._]*-[\s._]*(?P<ep>.+)|[\s._].*)?$"
)
_SEASON_DIR = re.compile(r"^(?:season|series|staffel|saison)[\s._-]*(\d{1,2})$", re.IGNORECASE)
# Jellyfin/Sonarr tags and scene junk after the title: [imdbid-tt..], {tmdb-..}, " - 1080p".
_TAGS = re.compile(r"\s*[\[{][^\]}]*[\]}]")
_QUALITY = re.compile(
    r"[\s._-]+(?:\d{3,4}p|web[\s.-]?dl|webrip|bluray|brrip|hdtv|x26[45]|h\.?26[45]|hevc|remux)\b.*$",
    re.IGNORECASE,
)


def _clean(s: str) -> str:
    s = _TAGS.sub("", s)
    s = _QUALITY.sub("", s)
    if " " not in s and s.count(".") >= 2:  # Scene.Style.Name
        s = s.replace(".", " ")
    return re.sub(r"\s+", " ", s.replace("_", " ")).strip(" -.")


def _title_year(s: str) -> tuple[str, int | None]:
    s = _TAGS.sub("", s).strip()
    for pat in (_YEAR, _BARE_YEAR):
        m = pat.match(s)
        if m and _clean(m["title"]):
            return _clean(m["title"]), int(m["year"])
    return _clean(s), None


def parse_path(path: str | PurePath) -> MediaContext:
    """Title/year (+ season/episode) from a Jellyfin-style path. Pure; never raises."""
    try:
        p = PurePath(path)
        stem, parent = p.stem, p.parent
        m = _SXXEYY.match(_TAGS.sub("", stem).strip())
        if m:
            show = _clean(m["show"] or "")
            year = None
            # The show dir is more reliable (and carries the year): Show (Year)/Season N/file.
            show_dir = parent.parent if _SEASON_DIR.match(parent.name) else parent
            if show_dir.name:
                dir_title, year = _title_year(show_dir.name)
                if dir_title and (not show or _SEASON_DIR.match(parent.name)):
                    show = dir_title
            if show:
                show, y2 = _title_year(show)
                year = year or y2
            return MediaContext(
                title=show,
                year=year,
                season=int(m["s"]),
                episode=int(m["e"]),
                episode_title=_clean(m["ep"] or ""),
            )
        title, year = _title_year(stem)
        if year is None and parent.name:
            dtitle, dyear = _title_year(parent.name)
            if dyear is not None:
                title, year = dtitle, dyear
        return MediaContext(title=title, year=year)
    except Exception:  # noqa: BLE001 - best effort by contract
        return MediaContext()


# --- Jellyfin ----------------------------------------------------------------------------

_ROLE_JUNK = re.compile(r"\((?:voice|uncredited|archive[^)]*)\)|\bvoice\b", re.IGNORECASE)
_SKIP_ROLES = re.compile(
    r"^(?:himself|herself|themselves|self|narrator|additional voices?)$", re.IGNORECASE
)
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*)\b")


def _characters(people: list) -> list[str]:
    """Character names from Jellyfin People (actor roles): 'Harry Potter', 'Mrs. Figg'.
    Drops 'Death Eater #2', 'Himself', and splits 'Vernon / Dudley' style multi-roles."""
    out: list[str] = []
    for person in people or ():
        if not isinstance(person, dict) or person.get("Type") not in (None, "Actor", "GuestStar"):
            continue
        role = person.get("Role") or ""
        for part in re.split(r"\s*/\s*|\s*;\s*", role):
            part = _ROLE_JUNK.sub("", part).strip(" ,")
            if not part or _SKIP_ROLES.match(part) or re.search(r"\d|#", part):
                continue
            if not any(ch.isupper() for ch in part):  # "young boy", "man in bar"
                continue
            if part not in out:
                out.append(part)
    return out


def _overview_terms(overview: str) -> list[str]:
    seen: list[str] = []
    for m in _PROPER.finditer(overview or ""):
        w = m.group(1)
        if len(w) > 2 and w not in seen:
            seen.append(w)
    return seen


class _Jellyfin:
    def __init__(self, url: str, api_key: str, timeout: float = _TIMEOUT_S) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def get(self, path: str, **params) -> dict:
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        req = urllib.request.Request(
            f"{self.url}{path}?{qs}",
            headers={
                "Authorization": f'MediaBrowser Token="{self.api_key}"',
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return data if isinstance(data, dict) else {}


def _pick(items: list, video: str, name: str, year: int | None) -> dict | None:
    items = [i for i in items or () if isinstance(i, dict)]
    for it in items:  # same file
        if it.get("Path") == video:
            return it
    base = PurePath(video).name
    for it in items:
        if it.get("Path") and PurePath(it["Path"]).name == base:
            return it
    named = [i for i in items if str(i.get("Name", "")).casefold() == name.casefold()]
    if year is not None:
        dated = [i for i in named or items if i.get("ProductionYear") == year]
        if len(dated) == 1:
            return dated[0]
    if len(named) == 1:
        return named[0]
    return items[0] if len(items) == 1 else None


_FIELDS = "Path,People,Overview,ProductionYear"
_series_cache: dict[tuple[str, str, int | None], dict | None] = {}


def _series_item(jf: _Jellyfin, video: str, ctx: MediaContext) -> dict | None:
    key = (jf.url, ctx.title.casefold(), ctx.year)
    if key not in _series_cache:
        data = jf.get(
            "/Items",
            searchTerm=ctx.title,
            includeItemTypes="Series",
            recursive="true",
            fields=_FIELDS,
            limit=10,
        )
        items = data.get("Items") or []
        # A series Path is its folder, so "is the video under it" beats a name match.
        hit = next(
            (
                i
                for i in items
                if isinstance(i, dict) and i.get("Path") and video.startswith(i["Path"] + "/")
            ),
            None,
        )
        if len(_series_cache) >= _CACHE_MAX:
            _series_cache.clear()
        _series_cache[key] = hit or _pick(items, video, ctx.title, ctx.year)
    return _series_cache[key]


def _from_jellyfin(jf: _Jellyfin, video: str, ctx: MediaContext) -> MediaContext:
    if ctx.is_episode:
        series = _series_item(jf, video, ctx)
        if not series or not series.get("Id"):
            return ctx
        eps = jf.get(f"/Shows/{series['Id']}/Episodes", season=ctx.season, fields=_FIELDS)
        items = [i for i in eps.get("Items") or () if isinstance(i, dict)]
        ep = next((i for i in items if i.get("Path") == video), None) or next(
            (i for i in items if i.get("IndexNumber") == ctx.episode), None
        )
        # Episode guest stars first (specific), then the series regulars.
        people = ((ep or {}).get("People") or []) + (series.get("People") or [])
        overview = (ep or {}).get("Overview") or series.get("Overview") or ""
        return replace(
            ctx,
            year=ctx.year or series.get("ProductionYear"),
            episode_title=ctx.episode_title or (ep or {}).get("Name", "") or "",
            names=tuple(_characters(people)[:_MAX_NAMES]),
            terms=tuple(_overview_terms(overview)[:_MAX_TERMS]),
            overview=overview,
            source="jellyfin",
        )
    if not ctx.title:
        return ctx
    data = jf.get(
        "/Items",
        searchTerm=ctx.title,
        includeItemTypes="Movie",
        recursive="true",
        years=ctx.year,
        fields=_FIELDS,
        limit=10,
    )
    item = _pick(data.get("Items") or [], video, ctx.title, ctx.year)
    if item is None and ctx.year is not None:  # the year in the folder name can be off by one
        data = jf.get(
            "/Items",
            searchTerm=ctx.title,
            includeItemTypes="Movie",
            recursive="true",
            fields=_FIELDS,
            limit=10,
        )
        item = _pick(data.get("Items") or [], video, ctx.title, ctx.year)
    if item is None:
        return ctx
    overview = item.get("Overview") or ""
    return replace(
        ctx,
        title=item.get("Name") or ctx.title,
        year=item.get("ProductionYear") or ctx.year,
        names=tuple(_characters(item.get("People") or [])[:_MAX_NAMES]),
        terms=tuple(_overview_terms(overview)[:_MAX_TERMS]),
        overview=overview,
        source="jellyfin",
    )


_cache: dict[tuple[str, str], MediaContext] = {}


def media_context(
    video: str | PurePath, *, jellyfin_url: str = "", jellyfin_api_key: str = ""
) -> MediaContext:
    """Everything we can cheaply learn about `video`. Never raises."""
    video = str(video)
    key = (video, jellyfin_url)
    if key in _cache:
        return _cache[key]
    ctx = parse_path(video)
    if jellyfin_url and jellyfin_api_key:
        try:
            ctx = _from_jellyfin(_Jellyfin(jellyfin_url, jellyfin_api_key), video, ctx)
        except Exception as exc:  # noqa: BLE001 - network, JSON, shapes: all best effort
            log.warning("jellyfin lookup failed for %s: %s", PurePath(video).name, exc)
    log.info(
        "context for %s: %r (%s) %s, %d names, %d terms, from %s",
        PurePath(video).name,
        ctx.title,
        ctx.year,
        f"S{ctx.season:02d}E{ctx.episode:02d}" if ctx.is_episode else "film",
        len(ctx.names),
        len(ctx.terms),
        ctx.source,
    )
    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[key] = ctx
    return ctx
