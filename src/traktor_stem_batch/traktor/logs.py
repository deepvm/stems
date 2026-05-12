from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..models import Track

STEM_STORED_RE = re.compile(
    r'Finished stem separation job for track "(?P<title>.*?)" - result: successful: '
    r'stem file stored to "(?P<path>.*?)", with offset (?P<offset>[^,]+)'
)


@dataclass(frozen=True)
class StemLogEntry:
    title: str
    path: Path
    offset: str


def default_log_path(collection_path: Path) -> Path:
    return collection_path.parent / "Logs" / "Traktor.log"


def parse_stem_log(path: Path) -> list[StemLogEntry]:
    if not path.exists():
        return []
    entries: list[StemLogEntry] = []
    for line in path.read_text(errors="replace").splitlines():
        match = STEM_STORED_RE.search(line)
        if not match:
            continue
        entries.append(
            StemLogEntry(
                title=match.group("title"),
                path=Path(match.group("path")),
                offset=match.group("offset"),
            )
        )
    return entries


def logged_native_stem_path(
    *,
    track: Track,
    collection_path: Path,
    stems_dir: Path,
) -> Path | None:
    entries = parse_stem_log(default_log_path(collection_path))
    stems_root = stems_dir.expanduser().resolve()
    for entry in reversed(entries):
        if entry.title != track.title:
            continue
        try:
            entry.path.resolve().relative_to(stems_root)
        except ValueError:
            continue
        return entry.path
    return None
