from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from ..errors import CollectionError


def nml_dir_to_path(dir_value: str, file_value: str) -> Path:
    directory = dir_value.replace("/:", "/").replace(":", "")
    return Path(directory) / file_value


@dataclass(frozen=True)
class CollectionEntry:
    path: Path
    title: str | None
    artist: str | None
    audio_id: str | None


class TraktorCollection:
    def __init__(self, path: Path):
        self.path = path
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self._entries_cache = None
        self._by_path_cache = None

    def entries(self) -> list[CollectionEntry]:
        if self._entries_cache is not None:
            return self._entries_cache
        items: list[CollectionEntry] = []
        for entry in self.root.findall(".//ENTRY"):
            loc = entry.find("LOCATION")
            if loc is None:
                continue
            dir_value = loc.get("DIR")
            file_value = loc.get("FILE")
            if not dir_value or not file_value:
                continue
            items.append(
                CollectionEntry(
                    path=nml_dir_to_path(dir_value, file_value),
                    title=entry.get("TITLE"),
                    artist=entry.get("ARTIST"),
                    audio_id=entry.get("AUDIO_ID"),
                )
            )
        self._entries_cache = items
        return items

    def by_path(self) -> dict[Path, CollectionEntry]:
        if self._by_path_cache is not None:
            return self._by_path_cache
        self._by_path_cache = {entry.path: entry for entry in self.entries()}
        return self._by_path_cache

    def find(self, audio_path: Path) -> CollectionEntry | None:
        target = audio_path.expanduser()
        direct = self.by_path().get(target)
        if direct:
            return direct
        for entry in self.entries():
            try:
                if entry.path.samefile(target):
                    return entry
            except OSError:
                continue
        return None

    def entry_element(self, audio_path: Path) -> ET.Element | None:
        target = audio_path.expanduser()
        for entry in self.root.findall(".//ENTRY"):
            loc = entry.find("LOCATION")
            if loc is None:
                continue
            dir_value = loc.get("DIR")
            file_value = loc.get("FILE")
            if not dir_value or not file_value:
                continue
            entry_path = nml_dir_to_path(dir_value, file_value)
            if entry_path == target:
                return entry
            try:
                if entry_path.samefile(target):
                    return entry
            except OSError:
                continue
        return None

    def mark_generated_stem(self, audio_path: Path) -> bool:
        entry = self.entry_element(audio_path)
        if entry is None:
            return False
        info = entry.find("INFO")
        if info is None:
            info = ET.SubElement(entry, "INFO")
        flags = int(info.get("FLAGS", "0"))
        updated = flags | 64
        if updated == flags:
            return False
        info.set("FLAGS", str(updated))
        return True

    def has_generated_stem(self, audio_path: Path) -> bool:
        entry = self.entry_element(audio_path)
        if entry is None:
            return False
        info = entry.find("INFO")
        if info is None:
            return False
        return (int(info.get("FLAGS", "0")) & 64) != 0

    def backup(self) -> Path:
        backup_path = self.path.with_suffix(self.path.suffix + ".bak")
        if backup_path.exists():
            index = 1
            while True:
                candidate = self.path.with_suffix(self.path.suffix + f".bak{index}")
                if not candidate.exists():
                    backup_path = candidate
                    break
                index += 1
        shutil.copy2(self.path, backup_path)
        return backup_path

    def write_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=self.path.parent) as fh:
            tmp_path = Path(fh.name)
            self.tree.write(fh, encoding="utf-8", xml_declaration=True)
        try:
            ET.parse(tmp_path)
        except ET.ParseError as exc:
            tmp_path.unlink(missing_ok=True)
            raise CollectionError(f"refusing to write invalid NML: {exc}") from exc
        tmp_path.replace(self.path)
