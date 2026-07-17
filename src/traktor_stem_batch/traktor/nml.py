from __future__ import annotations

import shutil
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from ..errors import CollectionError


def normalize_path(path: Path) -> Path:
    return Path(unicodedata.normalize("NFC", str(path)))


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
        self._element_by_path_cache = None

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
        cache = {}
        for entry in self.entries():
            norm_path = normalize_path(entry.path)
            cache[norm_path] = entry
            try:
                resolved = normalize_path(entry.path.expanduser().resolve())
                cache[resolved] = entry
            except OSError:
                pass
        self._by_path_cache = cache
        return self._by_path_cache

    def find(self, audio_path: Path) -> CollectionEntry | None:
        target = normalize_path(audio_path.expanduser())
        entry = self.by_path().get(target)
        if entry is not None:
            return entry
        try:
            resolved = normalize_path(target.resolve())
            entry = self.by_path().get(resolved)
            if entry is not None:
                return entry
        except OSError:
            pass
        return None

    def _element_by_path(self) -> dict[Path, ET.Element]:
        if self._element_by_path_cache is not None:
            return self._element_by_path_cache
        cache = {}
        for entry in self.root.findall(".//ENTRY"):
            loc = entry.find("LOCATION")
            if loc is None:
                continue
            dir_value = loc.get("DIR")
            file_value = loc.get("FILE")
            if not dir_value or not file_value:
                continue
            entry_path = nml_dir_to_path(dir_value, file_value)
            norm_path = normalize_path(entry_path)
            cache[norm_path] = entry
            try:
                resolved = normalize_path(entry_path.expanduser().resolve())
                cache[resolved] = entry
            except OSError:
                pass
        self._element_by_path_cache = cache
        return self._element_by_path_cache

    def entry_element(self, audio_path: Path) -> ET.Element | None:
        target = normalize_path(audio_path.expanduser())
        el = self._element_by_path().get(target)
        if el is not None:
            return el
        try:
            resolved = normalize_path(target.resolve())
            el = self._element_by_path().get(resolved)
            if el is not None:
                return el
        except OSError:
            pass
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
