from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


def write_test_wav(path: Path, frequency: float, seconds: float = 0.25, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    amplitude = 0.15
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        for i in range(frames):
            value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            frame = struct.pack("<hh", value, value)
            fh.writeframesraw(frame)


def wav_info(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as fh:
        return {
            "channels": fh.getnchannels(),
            "sample_width": fh.getsampwidth(),
            "sample_rate": fh.getframerate(),
            "frames": fh.getnframes(),
        }
