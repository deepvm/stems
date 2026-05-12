from __future__ import annotations

import base64
import math
from pathlib import Path

_A = 0x67452301
_B = 0xEFCDAB89
_C = 0x98BADCFE
_D = 0x10325476

_S = (
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
)

_K = tuple(int(abs(math.sin(i + 1)) * (1 << 32)) & 0xFFFFFFFF for i in range(64))
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"


def _rotate_left(value: int, bits: int) -> int:
    value &= 0xFFFFFFFF
    return ((value << bits) | (value >> (32 - bits))) & 0xFFFFFFFF


def _md5_round(state: list[int], block: bytes) -> None:
    words = [int.from_bytes(block[i : i + 4], "little") for i in range(0, 64, 4)]
    a, b, c, d = state

    for i in range(64):
        if i < 16:
            f = (b & c) | (~b & d)
            g = i
        elif i < 32:
            f = (d & b) | (~d & c)
            g = (5 * i + 1) % 16
        elif i < 48:
            f = b ^ c ^ d
            g = (3 * i + 5) % 16
        else:
            f = c ^ (b | ~d)
            g = (7 * i) % 16

        a, d, c, b = (
            d,
            c,
            b,
            (b + _rotate_left(a + f + _K[i] + words[g], _S[i])) & 0xFFFFFFFF,
        )

    state[0] = (state[0] + a) & 0xFFFFFFFF
    state[1] = (state[1] + b) & 0xFFFFFFFF
    state[2] = (state[2] + c) & 0xFFFFFFFF
    state[3] = (state[3] + d) & 0xFFFFFFFF


def traktor_hash_words(audio_id: str) -> list[int]:
    audio_id_bytes = base64.b64decode(audio_id)
    if len(audio_id_bytes) != 256:
        raise ValueError(f"expected decoded AUDIO_ID to be 256 bytes, got {len(audio_id_bytes)}")
    state = [_A, _B, _C, _D]
    for offset in range(0, 256, 64):
        _md5_round(state, audio_id_bytes[offset : offset + 64])
    _md5_round(state, bytes(64))
    return state


def stem_bucket_from_audio_id(audio_id: str, algorithm: str = "traktor-md5-audio-id") -> str:
    if algorithm != "traktor-md5-audio-id":
        raise ValueError(f"unknown native filename algorithm: {algorithm}")
    return f"{traktor_hash_words(audio_id)[0] & 0x7F:03d}"


def stem_filename_from_audio_id(audio_id: str, algorithm: str = "traktor-md5-audio-id") -> str:
    if algorithm != "traktor-md5-audio-id":
        raise ValueError(f"unknown native filename algorithm: {algorithm}")
    chars: list[str] = []
    for word in traktor_hash_words(audio_id):
        for shift in range(0, 35, 5):
            chars.append(_ALPHABET[(word >> shift) & 0x1F])
    return "".join(chars) + ".stem.mp4"


def native_stem_path(
    stems_dir: Path,
    audio_id: str,
    algorithm: str = "traktor-md5-audio-id",
) -> Path:
    return (
        stems_dir
        / stem_bucket_from_audio_id(audio_id, algorithm=algorithm)
        / stem_filename_from_audio_id(audio_id, algorithm=algorithm)
    )


def candidate_stem_names(audio_id: str) -> dict[str, str]:
    return {"traktor-md5-audio-id": stem_filename_from_audio_id(audio_id)}


def calibration_matches(audio_id: str, existing_filename: str) -> dict[str, bool]:
    candidates = candidate_stem_names(audio_id)
    return {name: filename == existing_filename for name, filename in candidates.items()}
