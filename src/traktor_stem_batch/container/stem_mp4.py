from __future__ import annotations

import base64
import gc
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..errors import MissingDependencyError, StemBatchError
from ..models import PackagePlan, StemSet

STEM_NAMES = ("master", "drums", "bass", "other", "vocals")
TRAKTOR_STEM_METADATA = [
    {"color": "#FD6C38", "name": "Drums"},
    {"color": "#D232F4", "name": "Bass"},
    {"color": "#00FFAC", "name": "Other"},
    {"color": "#45DAFD", "name": "Vocals"},
]
TRAKTOR_NATIVE_METADATA = {
    "mastering_dsp": {
        "compressor": {
            "attack": 0.003000000026077032,
            "dry_wet": 50,
            "enabled": False,
            "hp_cutoff": 300,
            "input_gain": 6,
            "output_gain": 0,
            "ratio": 3,
            "release": 0.300000011920929,
            "threshold": 0,
        },
        "limiter": {"ceiling": 0, "enabled": False, "release": 0.001000000047497451, "threshold": 0},
    },
    "offset": 0,
    "stems": TRAKTOR_STEM_METADATA,
    "version": 2,
}


def dependency_report() -> dict[str, bool]:
    report = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "MP4Box": shutil.which("MP4Box") is not None,
        "numpy": False,
    }
    try:
        __import__("numpy")
        report["numpy"] = True
    except Exception:
        pass
    return report


def build_package_plan(
    stems: StemSet,
    output: Path,
    codec: str = "aac",
    bitrate: int = 256000,
    sample_rate: int = 44100,
    native: bool = False,
) -> PackagePlan:
    return PackagePlan(output=output, stems=stems, codec=codec, bitrate=bitrate, sample_rate=sample_rate, native=native)


def native_metadata_json() -> str:
    return json.dumps(TRAKTOR_NATIVE_METADATA, separators=(",", ":"), ensure_ascii=False)


def _audio_tc(audio):
    import numpy as np

    array = np.asarray(audio, dtype="float32")
    if array.ndim != 2:
        raise StemBatchError("audio array must be 2-dimensional")
    if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
        array = array.T
    if array.shape[1] == 1:
        array = np.repeat(array, 2, axis=1)
    if array.shape[1] > 2:
        array = array[:, :2]
    if array.shape[1] != 2:
        raise StemBatchError("audio array must be mono or stereo")
    return np.ascontiguousarray(np.clip(np.nan_to_num(array, copy=False), -0.99, 0.99))


def _stack_master_and_stems(master, stems: dict[str, object]):
    import numpy as np

    missing = [name for name in ("drums", "bass", "other", "vocals") if name not in stems]
    if missing:
        raise StemBatchError("missing separated stems: " + ", ".join(missing))
    tracks = [_audio_tc(master)] + [_audio_tc(stems[name]) for name in ("drums", "bass", "other", "vocals")]
    length = max(track.shape[0] for track in tracks)
    data = np.zeros((len(tracks), length, 2), dtype="float32")
    for index, track in enumerate(tracks):
        data[index, : track.shape[0], :] = track
    return data


def _write_temp_m4a_files(
    temp_dir: Path,
    tracks: list[object],
    sample_rate: int,
    codec: str,
    bitrate: int,
    output_sample_rate: int,
) -> None:
    import numpy as np

    length = max(track.shape[0] for track in tracks)
    
    # Pre-allocate stacked array for all 10 audio channels (5 tracks * 2 channels)
    stacked = np.zeros((length, len(tracks) * 2), dtype="float32")
    for index, track in enumerate(tracks):
        count = min(length, track.shape[0])
        if count:
            stacked[:count, index * 2 : index * 2 + 2] = track[:count]

    filters = ";".join(
        f"[0:a]pan=stereo|c0=c{index * 2}|c1=c{index * 2 + 1}[a{index}]"
        for index in range(len(tracks))
    )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(len(tracks) * 2),
        "-i",
        "pipe:0",
        "-filter_complex",
        filters,
    ]
    for index in range(len(tracks)):
        cmd.extend(
            [
                "-map",
                f"[a{index}]",
                "-c:a",
                codec,
                "-b:a",
                str(bitrate),
                "-ar",
                str(output_sample_rate),
                str(temp_dir / f"{index}.m4a"),
            ]
        )
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise StemBatchError("ffmpeg stdin is not available")
        # Write the entire stacked array in a single call to minimize Python overhead
        process.stdin.write(stacked.tobytes())
        process.stdin.close()
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", "ignore") if process.stderr else ""
        raise StemBatchError(stderr.strip() or "ffmpeg failed while encoding stems") from exc
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", "ignore") if process.stderr else ""
    if process.wait() != 0:
        raise StemBatchError(stderr.strip() or "ffmpeg failed while encoding stems")


def write_native_stem_arrays(
    *,
    master,
    stems: dict[str, object],
    output: Path,
    sample_rate: int,
    codec: str = "aac",
    bitrate: int = 256000,
    output_sample_rate: int = 44100,
) -> None:
    report = dependency_report()
    missing = [name for name in ("ffmpeg", "ffprobe", "MP4Box", "numpy") if not report[name]]
    if missing:
        raise MissingDependencyError("missing package dependencies: " + ", ".join(missing))
    tracks = None
    try:
        tracks = [_audio_tc(master)] + [_audio_tc(stems[name]) for name in ("drums", "bass", "other", "vocals")]
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            _write_temp_m4a_files(temp_dir, tracks, sample_rate, codec, bitrate, output_sample_rate)
            if output.exists():
                output.unlink()
            metadata_payload = base64.b64encode(native_metadata_json().encode("utf-8")).decode("ascii")
            cmd = ["MP4Box", "-new", "-timescale", str(output_sample_rate)]
            for index in range(5):
                cmd.extend(["-add", str(temp_dir / f"{index}.m4a#ID=Z")])
            cmd.extend(
                [
                    "-no-iod",
                    "-group-clean",
                    "-brand",
                    "mp42:1",
                    "-ab",
                    "isom",
                    "-ab",
                    "mp41",
                    "-ab",
                    "mp42",
                    "-udta",
                    "0:type=stem:src=base64," + metadata_payload,
                    "-quiet",
                    str(output),
                ]
            )
            subprocess.run(cmd, check=True)
    finally:
        del tracks
        gc.collect()


def verify_with_ffprobe(path: Path) -> tuple[bool, str]:
    if shutil.which("ffprobe") is None:
        return False, "ffprobe not found"
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-print_format", "json", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffprobe failed"
    streams = [stream for stream in json.loads(result.stdout).get("streams", []) if stream.get("codec_type") == "audio"]
    if len(streams) != 5:
        return False, f"expected 5 audio streams, found {len(streams)}"
    return True, "5 audio streams"


def verify_native_metadata(path: Path) -> tuple[bool, str]:
    if shutil.which("MP4Box") is None:
        return False, "MP4Box not found"
    result = subprocess.run(["MP4Box", "-info", str(path)], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip() or "MP4Box failed"
    info = result.stdout + result.stderr
    checks = {
        "major brand mp42": "Major Brand mp42" in info,
        "stem metadata": "UDTA types:" in info and "\tstem:" in info,
        "metadata version 2": '"version":2' in info,
        "vocals slot": '"name":"Vocals"' in info,
        "all tracks enabled": "Disabled In Movie" not in info,
        "no alternate track group": "Alternate Group ID" not in info,
        "no root IOD": "File has root IOD" not in info,
    }
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        return False, "missing native metadata: " + ", ".join(missing)
    return True, "native metadata v2"
