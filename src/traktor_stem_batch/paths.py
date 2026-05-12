from __future__ import annotations

import re
from pathlib import Path

AUDIO_EXTENSIONS = (".flac", ".wav", ".aiff", ".aif", ".mp3", ".m4a", ".alac")
DEFAULT_MUSIC_DIR = Path("/Users/user/Music/DJ")
DEFAULT_TRAKTOR_STEMS_DIR = Path("/Users/user/Music/Traktor/Stems")
DEFAULT_STATE_DIR = Path(".stembatch")


def iter_audio_files(root: Path, extensions: tuple[str, ...] = AUDIO_EXTENSIONS):
    root = root.expanduser()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _version_key(path: Path) -> tuple[int, ...]:
    match = re.search(r"Traktor\s+([0-9][0-9.]*)", str(path))
    if not match:
        return (0,)
    return tuple(int(part) for part in match.group(1).split(".") if part.isdigit())


def find_default_collection(home: Path | None = None) -> Path | None:
    home = home or Path.home()
    root = home / "Documents" / "Native Instruments"
    candidates = list(root.glob("Traktor */collection.nml"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (_version_key(p), p.stat().st_mtime))[-1]
