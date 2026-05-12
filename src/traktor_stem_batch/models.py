from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STEM_ORDER = ("drums", "bass", "other", "vocals")
CONTAINER_ORDER = ("master", "drums", "bass", "other", "vocals")


@dataclass(frozen=True)
class Track:
    path: Path
    title: str
    artist: str | None = None
    audio_id: str | None = None

    @property
    def display_name(self) -> str:
        if self.artist:
            return f"{self.artist} - {self.title}"
        return self.title


@dataclass(frozen=True)
class StemSet:
    master: Path
    drums: Path
    bass: Path
    other: Path
    vocals: Path

    def as_ordered_paths(self) -> list[Path]:
        return [self.master, self.drums, self.bass, self.other, self.vocals]

    def as_dict(self) -> dict[str, Path]:
        return {
            "master": self.master,
            "drums": self.drums,
            "bass": self.bass,
            "other": self.other,
            "vocals": self.vocals,
        }


@dataclass(frozen=True)
class PackagePlan:
    output: Path
    stems: StemSet
    codec: str
    bitrate: int
    sample_rate: int
    native: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "output": str(self.output),
            "codec": self.codec,
            "bitrate": self.bitrate,
            "sample_rate": self.sample_rate,
            "native": self.native,
            "streams": [
                {"slot": name, "path": str(path)}
                for name, path in self.stems.as_dict().items()
            ],
        }
