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


def inject_stem_metadata(mp4_path: Path, json_metadata: str) -> None:
    data = bytearray(mp4_path.read_bytes())
    
    # 1. Find the 'moov' box
    offset = 0
    moov_offset = -1
    moov_size = -1
    while offset < len(data):
        box_size = int.from_bytes(data[offset:offset+4], "big")
        box_type = data[offset+4:offset+8].decode("ascii", errors="ignore")
        if box_type == "moov":
            moov_offset = offset
            moov_size = box_size
            break
        offset += box_size
        
    if moov_offset == -1:
        raise ValueError("Could not find 'moov' box")
        
    # 2. Build the custom 'stem' box and 'udta' box
    json_bytes = json_metadata.encode("utf-8")
    stem_box_size = 8 + len(json_bytes)
    stem_box = stem_box_size.to_bytes(4, "big") + b"stem" + json_bytes
    
    udta_box_size = 8 + len(stem_box)
    udta_box = udta_box_size.to_bytes(4, "big") + b"udta" + stem_box
    
    # 3. Scan children of 'moov' to find if 'udta' already exists, and slice it out
    child_offset = moov_offset + 8
    moov_end = moov_offset + moov_size
    udta_offset = -1
    udta_size = -1
    
    while child_offset < moov_end:
        child_size = int.from_bytes(data[child_offset:child_offset+4], "big")
        child_type = data[child_offset+4:child_offset+8].decode("ascii", errors="ignore")
        if child_type == "udta":
            udta_offset = child_offset
            udta_size = child_size
            break
        child_offset += child_size
        
    if udta_offset != -1:
        # Slice out the old 'udta' box
        data[udta_offset : udta_offset + udta_size] = b""
        moov_size -= udta_size
        
    # 4. Insert our clean 'udta_box' at the end of 'moov'
    new_moov_size = moov_size + len(udta_box)
    data[moov_offset:moov_offset+4] = new_moov_size.to_bytes(4, "big")
    
    insert_pos = moov_offset + moov_size
    data[insert_pos:insert_pos] = udta_box
    
    # 5. Clear alternate groups in all 'tkhd' boxes to ensure standard playback
    idx = data.find(b"tkhd")
    while idx != -1:
        box_start = idx - 4
        version = data[box_start + 8]
        if version == 1:
            alt_group_offset = box_start + 54
        else:
            alt_group_offset = box_start + 42
            
        data[alt_group_offset : alt_group_offset + 2] = b"\x00\x00"
        idx = data.find(b"tkhd", idx + 4)
        
    mp4_path.write_bytes(data)


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
    missing = [name for name in ("ffmpeg", "ffprobe", "numpy") if not report[name]]
    if missing:
        raise MissingDependencyError("missing package dependencies: " + ", ".join(missing))
    tracks = None
    try:
        import numpy as np
        
        tracks = [_audio_tc(master)] + [_audio_tc(stems[name]) for name in ("drums", "bass", "other", "vocals")]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()

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
            "-y",
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
            cmd.extend([
                "-map", f"[a{index}]",
                "-c:a", codec,
                "-b:a", str(bitrate),
                "-ar", str(output_sample_rate),
                f"-disposition:a:{index}", "default"
            ])
            
        # Set major brand and skip_iods for Traktor compatibility
        cmd.extend([
            "-brand", "mp42",
            "-skip_iods", "1",
            str(output)
        ])
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            if process.stdin is None:
                raise StemBatchError("ffmpeg stdin is not available")
            process.stdin.write(stacked.tobytes())
            process.stdin.close()
        except Exception as exc:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            if isinstance(exc, BrokenPipeError):
                stderr = process.stderr.read().decode("utf-8", "ignore") if process.stderr else ""
                raise StemBatchError(stderr.strip() or "ffmpeg failed while encoding stems") from exc
            raise exc
        finally:
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        stderr = process.stderr.read().decode("utf-8", "ignore") if process.stderr else ""
        try:
            if process.wait(timeout=120) != 0:
                raise StemBatchError(stderr.strip() or "ffmpeg failed while encoding stems")
        except subprocess.TimeoutExpired:
            process.kill()
            raise StemBatchError("ffmpeg encoding timed out")
            
        # Inject Traktor metadata cleanly
        metadata = native_metadata_json()
        inject_stem_metadata(output, metadata)

    finally:
        del tracks


def verify_with_ffprobe(path: Path) -> tuple[bool, str]:
    if shutil.which("ffprobe") is None:
        return False, "ffprobe not found"
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-print_format", "json", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False, "ffprobe verification timed out"
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffprobe failed"
    streams = [stream for stream in json.loads(result.stdout).get("streams", []) if stream.get("codec_type") == "audio"]
    if len(streams) != 5:
        return False, f"expected 5 audio streams, found {len(streams)}"
    return True, "5 audio streams"


def verify_native_metadata(path: Path) -> tuple[bool, str]:
    if shutil.which("MP4Box") is None:
        return False, "MP4Box not found"
    try:
        result = subprocess.run(
            ["MP4Box", "-info", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False, "MP4Box metadata verification timed out"
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
