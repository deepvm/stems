from __future__ import annotations

import re
from pathlib import Path

from .models import Track
from .paths import iter_audio_files

PREFIX_RE = re.compile(r"^\s*\d+\s*-\s*")


def title_artist_from_filename(path: Path) -> tuple[str, str | None]:
    stem = PREFIX_RE.sub("", path.stem).strip()
    parts = [part.strip() for part in stem.split(" - ") if part.strip()]
    if len(parts) >= 2:
        return " - ".join(parts[1:]), parts[0]
    return stem or path.stem, None


def scan_music_dir(music_dir: Path) -> list[Track]:
    tracks: list[Track] = []
    for path in iter_audio_files(music_dir):
        title, artist = title_artist_from_filename(path)
        tracks.append(Track(path=path, title=title, artist=artist))
    return tracks
