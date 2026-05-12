from __future__ import annotations

import base64
import gc
import json
import shutil
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path

from ..errors import MissingDependencyError, StemBatchError
from ..models import PackagePlan, StemSet

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
        "limiter": {
            "ceiling": 0,
            "enabled": False,
            "release": 0.001000000047497451,
            "threshold": 0,
        },
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
        "stempeg": False,
        "numpy": False,
        "soundfile": False,
    }
    for module in ("numpy", "soundfile"):
        try:
            __import__(module)
        except Exception:
            report[module] = False
        else:
            report[module] = True
    try:
        metadata.version("stempeg")
    except metadata.PackageNotFoundError:
        report["stempeg"] = False
    else:
        report["stempeg"] = True
    return report


def _resample_to_temp(path: Path, target_sample_rate: int, temp_dir: Path, index: int) -> Path:
    if shutil.which("ffmpeg") is None:
        raise MissingDependencyError("ffmpeg command not found")
    output = temp_dir / f"{index:02d}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "2",
        "-ar",
        str(target_sample_rate),
        "-c:a",
        "pcm_f32le",
        str(output),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise StemBatchError(result.stderr.strip() or f"failed to resample {path}")
    return output


def _load_audio_tensor(paths: list[Path], target_sample_rate: int | None = None):
    import numpy as np
    import soundfile as sf

    arrays = []
    sample_rate = target_sample_rate
    max_len = 0
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        for index, path in enumerate(paths):
            load_path = path
            try:
                source_rate = sf.info(str(path)).samplerate
            except Exception:
                if target_sample_rate is None:
                    raise StemBatchError(f"could not read audio info: {path}") from None
                load_path = _resample_to_temp(path, target_sample_rate, temp_dir, index)
                source_rate = target_sample_rate
            if target_sample_rate is not None and source_rate != target_sample_rate:
                load_path = _resample_to_temp(path, target_sample_rate, temp_dir, index)

            data, rate = sf.read(str(load_path), always_2d=True, dtype="float32")
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            if data.shape[1] != 2:
                raise StemBatchError(f"{path} must be mono or stereo")
            if sample_rate is None:
                sample_rate = rate
            elif sample_rate != rate:
                raise StemBatchError(f"sample rate mismatch: {path} has {rate}, expected {sample_rate}")
            arrays.append(data)
            max_len = max(max_len, data.shape[0])

    padded = []
    for data in arrays:
        if data.shape[0] < max_len:
            pad = np.zeros((max_len - data.shape[0], 2), dtype=data.dtype)
            data = np.vstack([data, pad])
        padded.append(data)
    return np.stack(padded, axis=0), int(sample_rate or 44100)


def build_package_plan(
    stems: StemSet,
    output: Path,
    codec: str = "aac",
    bitrate: int = 256000,
    sample_rate: int = 44100,
    native: bool = False,
) -> PackagePlan:
    return PackagePlan(
        output=output,
        stems=stems,
        codec=codec,
        bitrate=bitrate,
        sample_rate=sample_rate,
        native=native,
    )


def write_stem_file(plan: PackagePlan, dry_run: bool = False) -> None:
    missing_inputs = [path for path in plan.stems.as_ordered_paths() if not path.exists()]
    if dry_run:
        payload = plan.to_dict()
        payload["missing_inputs"] = [str(path) for path in missing_inputs]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if missing_inputs:
        raise StemBatchError("missing input stems: " + ", ".join(str(path) for path in missing_inputs))

    report = dependency_report()
    missing = [
        name
        for name in ("ffmpeg", "ffprobe", "MP4Box", "stempeg", "numpy", "soundfile")
        if not report[name]
    ]
    if missing:
        raise MissingDependencyError(
            "missing package dependencies: "
            + ", ".join(missing)
            + ". Install system tools with `brew install ffmpeg gpac` "
            + "and Python packages with `uv sync`."
        )

    data = None
    try:
        data, detected_rate = _load_audio_tensor(
            plan.stems.as_ordered_paths(),
            target_sample_rate=plan.sample_rate,
        )
        plan.output.parent.mkdir(parents=True, exist_ok=True)
        if plan.native:
            _write_traktor_native_stem(
                output=plan.output,
                data=data,
                sample_rate=detected_rate,
                codec=plan.codec,
                bitrate=plan.bitrate,
                output_sample_rate=plan.sample_rate,
            )
        else:
            _write_legacy_stempeg_stem(
                output=plan.output,
                data=data,
                sample_rate=detected_rate,
                codec=plan.codec,
                bitrate=plan.bitrate,
                output_sample_rate=plan.sample_rate,
            )
    finally:
        del data
        gc.collect()


def native_metadata_json() -> str:
    return json.dumps(TRAKTOR_NATIVE_METADATA, separators=(",", ":"), ensure_ascii=False)


def _stempeg_codec(codec: str) -> str:
    if codec != "aac":
        return codec
    from stempeg.cmds import get_aac_codec

    return get_aac_codec()


def _write_temp_m4a_files(
    *,
    temp_dir: Path,
    data,
    sample_rate: int,
    codec: str,
    bitrate: int,
    output_sample_rate: int,
) -> None:
    import stempeg

    writer = stempeg.FilesWriter(
        codec=_stempeg_codec(codec),
        bitrate=bitrate,
        output_sample_rate=output_sample_rate,
        stem_names=["0", "1", "2", "3", "4"],
        multiprocess=True,
    )
    stempeg.write_stems(
        path=str(temp_dir / "tmp.m4a"),
        data=data,
        sample_rate=sample_rate,
        writer=writer,
    )


def _write_traktor_native_stem(
    *,
    output: Path,
    data,
    sample_rate: int,
    codec: str,
    bitrate: int,
    output_sample_rate: int,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        _write_temp_m4a_files(
            temp_dir=temp_dir,
            data=data,
            sample_rate=sample_rate,
            codec=codec,
            bitrate=bitrate,
            output_sample_rate=output_sample_rate,
        )
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


def _write_legacy_stempeg_stem(
    *,
    output: Path,
    data,
    sample_rate: int,
    codec: str,
    bitrate: int,
    output_sample_rate: int,
) -> None:
    import stempeg

    writer = stempeg.NIStemsWriter(
        default_metadata=TRAKTOR_NATIVE_METADATA,
        stems_metadata=TRAKTOR_STEM_METADATA,
        codec=codec,
        bitrate=bitrate,
        output_sample_rate=output_sample_rate,
    )
    stempeg.write_stems(
        path=str(output),
        data=data,
        sample_rate=sample_rate,
        writer=writer,
    )


def verify_with_ffprobe(path: Path) -> tuple[bool, str]:
    if shutil.which("ffprobe") is None:
        return False, "ffprobe not found"
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffprobe failed"
    payload = json.loads(result.stdout)
    streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"]
    if len(streams) != 5:
        return False, f"expected 5 audio streams, found {len(streams)}"
    return True, "5 audio streams"


def verify_native_metadata(path: Path) -> tuple[bool, str]:
    if shutil.which("MP4Box") is None:
        return False, "MP4Box not found"
    result = subprocess.run(
        ["MP4Box", "-info", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
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
