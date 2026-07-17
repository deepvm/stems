from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .container.stem_mp4 import (
    dependency_report,
    verify_native_metadata,
    verify_with_ffprobe,
    write_native_stem_arrays,
)
from .errors import StemBatchError
from .models import Track
from .paths import DEFAULT_MUSIC_DIR, DEFAULT_STATE_DIR, DEFAULT_TRAKTOR_STEMS_DIR, find_default_collection
from .scanner import scan_music_dir, title_artist_from_filename
from .separation.backends import SUPPORTED_BACKENDS, SUPPORTED_MLX_MODELS, build_backend
from .state import JobState
from .traktor.native import calibration_matches, candidate_stem_names, native_stem_path
from .traktor.logs import logged_native_stem_path
from .traktor.nml import TraktorCollection
from .util import human_bool, sanitize_filename

def _detect_best_aac_codec() -> str:
    if sys.platform != "darwin":
        return "aac"
    try:
        output = subprocess.check_output(["ffmpeg", "-encoders"], stderr=subprocess.DEVNULL, encoding="utf-8")
        if "aac_at" in output:
            return "aac_at"
    except Exception:
        pass
    return "aac"


@dataclass(frozen=True)
class ProcessItem:
    index: int
    total: int
    track: Track
    output: Path


@dataclass(frozen=True)
class ProcessResult:
    item: ProcessItem
    elapsed: float
    work_dir: Path


def _collection(path: str | None) -> TraktorCollection | None:
    if path:
        return TraktorCollection(Path(path).expanduser())
    found = find_default_collection()
    return TraktorCollection(found) if found else None


def _enrich_tracks(tracks: list[Track], collection: TraktorCollection | None) -> list[Track]:
    if collection is None:
        return tracks
    enriched: list[Track] = []
    for track in tracks:
        entry = collection.find(track.path)
        if entry:
            enriched.append(
                Track(
                    path=track.path,
                    title=entry.title or track.title,
                    artist=entry.artist or track.artist,
                    audio_id=entry.audio_id,
                )
            )
        else:
            enriched.append(track)
    return enriched


def _single_track(audio_path: Path, collection: TraktorCollection | None) -> Track:
    audio_path = audio_path.expanduser()
    if not audio_path.exists():
        raise StemBatchError(f"audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise StemBatchError(f"audio path is not a file: {audio_path}")
    if collection is not None:
        entry = collection.find(audio_path)
        if entry is not None:
            return Track(
                path=entry.path,
                title=entry.title or audio_path.stem,
                artist=entry.artist,
                audio_id=entry.audio_id,
            )
    title, artist = title_artist_from_filename(audio_path)
    return Track(path=audio_path, title=title, artist=artist)


def _process_tracks(args: argparse.Namespace, collection: TraktorCollection | None) -> list[Track]:
    if args.audio:
        return [_single_track(Path(args.audio), collection)]
    tracks = _enrich_tracks(scan_music_dir(Path(args.music_dir)), collection)
    if args.limit:
        tracks = tracks[: args.limit]
    return tracks


def _traktor_is_running() -> bool:
    if sys.platform != "darwin":
        return False
    result = subprocess.run(["ps", "-ax"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return "/Traktor Pro 4.app/Contents/MacOS/Traktor Pro 4" in result.stdout


def _output_path(
    *,
    track: Track,
    stems_dir: Path,
    collection: TraktorCollection,
    native_algorithm: str,
) -> Path:
    logged_path = logged_native_stem_path(
        track=track,
        collection_path=collection.path,
        stems_dir=stems_dir,
    )
    if logged_path is not None:
        return logged_path
    if not track.audio_id:
        raise StemBatchError(f"missing AUDIO_ID for native output: {track.path}")
    try:
        return native_stem_path(stems_dir, track.audio_id, algorithm=native_algorithm)
    except ValueError as exc:
        raise StemBatchError(str(exc)) from exc


def _duration(value: float) -> str:
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}m {seconds:02d}s"


def _status(message: str) -> None:
    print(message, flush=True)


def _native_stem_ready(output: Path) -> tuple[bool, str]:
    if not output.exists():
        return False, "missing"
    ok, message = verify_with_ffprobe(output)
    if not ok:
        return False, message
    ok, message = verify_native_metadata(output)
    if not ok:
        return False, message
    return True, "linked native stem exists"


def _mark_collection_stem(
    *,
    collection: TraktorCollection,
    track: Track,
    collection_backup: Path | None,
) -> tuple[Path | None, bool]:
    changed = collection.mark_generated_stem(track.path)
    if not changed:
        if not collection.has_generated_stem(track.path):
            raise StemBatchError(f"could not set generated-stem flag in collection: {track.path}")
        return collection_backup, False
    if collection_backup is None:
        collection_backup = collection.backup()
        _status(f"collection backup: {collection_backup}")
    return collection_backup, True


def _cleanup_work_dir(path: Path, root: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        _status(f"warning: could not clean work dir {path}: {exc}")
        return
    try:
        root.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        return


def _validate_process_args(args: argparse.Namespace) -> None:
    if args.backend not in SUPPORTED_BACKENDS:
        raise StemBatchError(f"only --backend in {SUPPORTED_BACKENDS} is supported")
    if args.model not in SUPPORTED_MLX_MODELS:
        raise StemBatchError(f"only --model in {SUPPORTED_MLX_MODELS} is supported")
    if args.shifts < 0:
        raise StemBatchError("--shifts must be 0 or greater")
    if args.track_workers != 1:
        raise StemBatchError("only --track-workers 1 is supported")
    if args.batch_size < 0:
        raise StemBatchError("--batch-size must be 0 (for auto-detect) or greater")
    if args.mlx_cache_limit_mb is not None and args.mlx_cache_limit_mb < 0:
        raise StemBatchError("--mlx-cache-limit-mb must be 0 or greater")
    if args.mlx_memory_limit_mb is not None and args.mlx_memory_limit_mb < 0:
        raise StemBatchError("--mlx-memory-limit-mb must be 0 or greater")
    if args.cooling_pause is not None and args.cooling_pause < 0:
        raise StemBatchError("--cooling-pause must be 0 or greater")
    if not (0 <= args.overlap < 1):
        raise StemBatchError("--overlap must be between 0 (inclusive) and 1 (exclusive)")


# Replaced by pipeline encode_and_finalize


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"traktor-stem-batch {__version__}")
    report = dependency_report()
    for key in ("ffmpeg", "ffprobe", "MP4Box", "numpy"):
        print(f"{key}: {human_bool(report[key])}")
    git_available = shutil.which("git") is not None
    print(f"git: {human_bool(git_available)}")
    try:
        __import__("demucs_mlx")
        demucs_mlx_ok = True
    except Exception:
        demucs_mlx_ok = False
    print(f"demucs-mlx: {human_bool(demucs_mlx_ok)}")

    found = find_default_collection()
    print(f"default collection: {found if found else 'not found'}")
    warnings: list[str] = []
    if not report["ffmpeg"] or not report["ffprobe"]:
        warnings.append("ffmpeg/ffprobe missing: run `brew install ffmpeg`")
    if not report["MP4Box"]:
        warnings.append("MP4Box missing: run `brew install gpac`")
    if not report["numpy"]:
        warnings.append("audio Python deps missing: run `uv sync`")
    if not demucs_mlx_ok:
        warnings.append("demucs-mlx missing: run `uv sync`")
    if not git_available:
        warnings.append("git missing: run `brew install git`; uv uses it to install demucs-mlx")
    if warnings:
        print("")
        for warning in warnings:
            print(f"warning: {warning}")
        print("hint: install everything with `brew install uv ffmpeg gpac git` then `uv sync`")
    return 0


def _assert_apple_silicon() -> None:
    import platform
    import sys
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise StemBatchError(
            "This application is optimized specifically for Apple Silicon (M-series) Macs running macOS.\n"
            "Intel Macs, Windows, and Linux are not supported by the MLX Metal backend."
        )


def cmd_scan(args: argparse.Namespace) -> int:
    _assert_apple_silicon()
    tracks = scan_music_dir(Path(args.music_dir))
    collection = _collection(args.collection)
    tracks = _enrich_tracks(tracks, collection)
    for track in tracks[: args.limit or len(tracks)]:
        marker = "matched" if track.audio_id else "no-audio-id"
        print(f"{marker}\t{track.display_name}\t{track.path}")
    print(f"total: {len(tracks)}")
    print("note: scan does not create stems; run `process` to write linked native stem files")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    _assert_apple_silicon()
    _validate_process_args(args)
    music_dir = Path(args.music_dir).expanduser()
    stems_dir = Path(args.stems_dir).expanduser()
    collection = _collection(args.collection)
    if args.mode == "native" and collection is None:
        raise StemBatchError("native mode requires Traktor collection.nml")
    if not args.dry_run and not args.allow_running_traktor and _traktor_is_running():
        raise StemBatchError("Traktor Pro 4 is running. Close Traktor before writing stems or collection flags.")

    state = None if args.dry_run else JobState(Path(args.state_db))
    
    # ── Resource Profile & Dynamic Scaling ──
    import os
    cpu_cores = os.cpu_count() or 4
    gpu_cores = 16
    try:
        gpu_cores = detect_gpu_cores()
    except Exception:
        pass
    total_ram_mb = detect_physical_ram_mb()

    # Apply profile defaults
    if args.profile == "silent":
        profile_batch_size = 4
        profile_workers = 1
        profile_cooling = 1.0
        profile_memory_limit = max(4096, int(total_ram_mb * 0.30))
        profile_cache_limit = 256
    elif args.profile == "ultra":
        profile_batch_size = max(1, gpu_cores)
        profile_workers = max(1, (cpu_cores - 1) // 3)
        profile_cooling = 0.0
        profile_memory_limit = max(4096, int(total_ram_mb * 0.85))
        profile_cache_limit = 1024
    elif args.profile == "extreme":
        profile_batch_size = max(1, gpu_cores // 2)
        profile_workers = max(1, (cpu_cores - 1) // 5)
        profile_cooling = 0.0
        profile_memory_limit = max(4096, int(total_ram_mb * 0.70))
        profile_cache_limit = 512
    else:  # balanced
        profile_batch_size = max(1, gpu_cores // 2)
        profile_workers = max(1, (cpu_cores - 2) // 5)
        profile_cooling = 0.5
        profile_memory_limit = max(4096, int(total_ram_mb * 0.50))
        profile_cache_limit = 512

    batch_size = args.batch_size if args.batch_size > 0 else profile_batch_size
    max_workers = profile_workers
    cooling_pause = args.cooling_pause if args.cooling_pause is not None else profile_cooling
    memory_limit_mb = args.mlx_memory_limit_mb if args.mlx_memory_limit_mb is not None else profile_memory_limit
    cache_limit_mb = args.mlx_cache_limit_mb if args.mlx_cache_limit_mb is not None else profile_cache_limit

    if args.verbose_backend or args.dry_run:
        _status(f"Profile: {args.profile}. GPU cores: {gpu_cores}, CPU cores: {cpu_cores}, RAM: {total_ram_mb}MB.")
        _status(f"Settings: batch-size={batch_size}, workers={max_workers}, cooling-pause={cooling_pause}s, mlx-mem={memory_limit_mb}MB, mlx-cache={cache_limit_mb}MB.")

    backend = build_backend(
        name=args.backend,
        model=args.model,
        shifts=args.shifts,
        verbose=args.verbose_backend,
        cache_limit_mb=cache_limit_mb,
        memory_limit_mb=memory_limit_mb,
        batch_size=batch_size,
        overlap=args.overlap,
    )
    tracks = _process_tracks(args, collection)

    collection_backup: Path | None = None
    pending: list[ProcessItem] = []
    try:
        for index, track in enumerate(tracks, start=1):
            output = _output_path(
                track=track,
                stems_dir=stems_dir,
                collection=collection,
                native_algorithm=args.native_algorithm,
            )
            item = ProcessItem(index=index, total=len(tracks), track=track, output=output)
            if not args.dry_run and not args.reprocess_existing and not args.force:
                if state is not None and state.is_done_current(track.path) and output.exists():
                    ready, reason = True, "state db shows done"
                else:
                    ready, reason = _native_stem_ready(output)
                if ready:
                    if args.update_collection:
                        collection_backup, changed = _mark_collection_stem(
                            collection=collection,
                            track=track,
                            collection_backup=collection_backup,
                        )
                        if changed:
                            _status(f"[{index}/{len(tracks)}] collection: stem flag set")
                    if state is not None:
                        state.set(track.path, "done", output_path=output)
                    _status(f"[{index}/{len(tracks)}] skip existing: {track.display_name}")
                    continue
            if state is not None:
                state.set(track.path, "running", output_path=output)
            pending.append(item)

        if not pending:
            return 0

        # Perform compilation warm-up on the GPU natively in Metal
        if not args.dry_run:
            model = backend._load_model()
            _status("Compiling model for GPU acceleration (warm-up)...")
            import mlx.core as mx
            warmup_start = time.monotonic()
            segment_len = int(model.samplerate * float(model.segment))
            
            # Warm up for batch_size = 1
            dummy_input_1 = mx.zeros((1, 2, segment_len), dtype=mx.float32)
            mx.eval(model.forward_compiled(dummy_input_1))
            
            # Warm up for batch_size = batch_size
            if batch_size > 1:
                dummy_input_b = mx.zeros((batch_size, 2, segment_len), dtype=mx.float32)
                mx.eval(model.forward_compiled(dummy_input_b))
                
            _status(f"Model compiled successfully in {time.monotonic() - warmup_start:.1f}s.")

        work_dir_root = Path(args.work_dir).expanduser()

        def finish_result(result: ProcessResult) -> None:
            nonlocal collection_backup
            item = result.item
            if not args.dry_run:
                _status(f"[{item.index}/{item.total}] done: {item.track.display_name} ({_duration(result.elapsed)})")
                if args.update_collection:
                    collection_backup, changed = _mark_collection_stem(
                        collection=collection,
                        track=item.track,
                        collection_backup=collection_backup,
                    )
                    if changed:
                        _status(f"[{item.index}/{item.total}] collection: stem flag set")
                if state is not None:
                    state.set(item.track.path, "done", output_path=item.output)
                _cleanup_work_dir(result.work_dir, work_dir_root)

        # ── Pipeline Setup ──
        import queue
        import threading
        import gc
        from concurrent.futures import ThreadPoolExecutor
        from .separation.backends import SeparatedAudio, STEM_NAMES, apply_model_batched
        import mlx.core as mx
        import numpy as np

        decode_queue = queue.Queue(maxsize=1)
        decode_error = None

        def decode_worker():
            nonlocal decode_error
            try:
                model = backend._load_model()
                sample_rate = int(model.samplerate)
                for item in pending:
                    if args.verbose_backend:
                        _status(f"[{item.index}/{item.total}] pre-decode: {item.track.display_name}")
                    master = backend._load_audio(item.track.path, sample_rate)
                    decode_queue.put((item, master, sample_rate))
            except Exception as e:
                decode_error = e
                decode_queue.put(None)

        decode_thread = threading.Thread(target=decode_worker, daemon=True)
        decode_thread.start()

        def encode_and_finalize(item, separated_audio, codec, bitrate, output_sample_rate, native, dry_run):
            started = time.monotonic()
            work_dir = work_dir_root / sanitize_filename(item.track.path.stem)
            
            if dry_run:
                print(
                    json.dumps(
                        {
                            "output": str(item.output),
                            "codec": codec,
                            "bitrate": bitrate,
                            "sample_rate": output_sample_rate,
                            "native": native,
                            "streams": [
                                {"slot": "master", "path": str(item.track.path)},
                                {"slot": "drums", "source": f"{backend.name}:drums"},
                                {"slot": "bass", "source": f"{backend.name}:bass"},
                                {"slot": "other", "source": f"{backend.name}:other"},
                                {"slot": "vocals", "source": f"{backend.name}:vocals"},
                            ],
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                write_native_stem_arrays(
                    master=separated_audio.master,
                    stems=separated_audio.stems,
                    output=item.output,
                    sample_rate=separated_audio.sample_rate,
                    codec=codec,
                    bitrate=bitrate,
                    output_sample_rate=output_sample_rate,
                )
                ok, msg = verify_with_ffprobe(item.output)
                if not ok:
                    raise StemBatchError(msg)
                ok, msg = verify_native_metadata(item.output)
                if not ok:
                    raise StemBatchError(msg)
                    
            return ProcessResult(item=item, elapsed=time.monotonic() - started, work_dir=work_dir)

        # Thread pool for encoding/MP4Box packaging, dynamically sized to prevent CPU starvation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for item in pending:
                if decode_error:
                    raise decode_error
                
                # Retrieve pre-decoded audio
                decode_res = decode_queue.get()
                if decode_res is None:
                    if decode_error:
                        raise decode_error
                    raise StemBatchError("Pre-decode worker stopped unexpectedly")
                
                item_q, master, sample_rate = decode_res
                assert item_q == item
                
                _status(f"[{item.index}/{item.total}] separate: {item.track.display_name}")
                
                # Main GPU separation stage (runs on the main thread for GIL and KeyboardInterrupt stability)
                model = backend._load_model()
                started_gpu = time.monotonic()
                try:
                    out = apply_model_batched(
                        model,
                        mx.array(master[None]),
                        batch_size=backend.batch_size,
                        shifts=backend.shifts,
                        split=True,
                        overlap=backend.overlap,
                        progress=backend.verbose,
                        segment=None,
                    )
                    mx.eval(out)
                except Exception as exc:
                    mx.clear_cache()
                    gc.collect()
                    raise exc
                
                separated = np.array(out[0]).astype("float32", copy=False)
                stems = {name: separated[index] for index, name in enumerate(STEM_NAMES)}
                backend._validate_stem_sum(master, stems)
                
                separated_audio = SeparatedAudio(master=master, stems=stems, sample_rate=sample_rate)
                
                # Dispatch encoding task to CPU thread pool
                future = executor.submit(
                    encode_and_finalize,
                    item=item,
                    separated_audio=separated_audio,
                    codec=args.codec,
                    bitrate=args.bitrate,
                    output_sample_rate=args.sample_rate,
                    native=(args.mode == "native"),
                    dry_run=args.dry_run
                )
                futures.append((item, future))
                
                # Release array references immediately in the main thread
                del out, separated, stems, separated_audio
                
                # Dynamic resource throttling based on active memory usage and system thermals
                try:
                    avail_ram = _get_available_ram_mb()
                    if avail_ram < 2048:
                        _status(f"Warning: Low available memory ({avail_ram}MB). Throttling resource usage...")
                        import mlx.core as mx
                        mx.clear_cache()
                        import gc
                        gc.collect()
                        time.sleep(2.0)
                        
                    if _check_thermal_warning():
                        _status("Warning: Thermal/performance warning level detected. Throttling execution for cooling...")
                        time.sleep(2.0)
                except Exception:
                    pass
                
                # Optional cooling pause between tracks to prevent thermal throttling
                if cooling_pause > 0 and not args.dry_run:
                    time.sleep(cooling_pause)
                
            # Process results as they complete
            for item, future in futures:
                try:
                    result = future.result()
                    finish_result(result)
                except Exception as exc:
                    if state is not None:
                        state.set(item.track.path, "error", output_path=item.output, error=str(exc))
                    if not args.continue_on_error:
                        raise
                    _status(f"[{item.index}/{item.total}] error: {item.track.display_name}: {exc}")

    except Exception as exc:
        if state is not None and "item" in locals():
            state.set(item.track.path, "error", output_path=item.output, error=str(exc))
        raise
    finally:
        # Clear GPU allocator caches and run full GC on shutdown
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        import gc
        gc.collect()

        if args.update_collection and collection_backup is not None:
            _status("Saving updated Traktor collection to disk...")
            try:
                collection.write_atomic()
                _status("Traktor collection saved successfully.")
            except Exception as e:
                _status(f"Error saving Traktor collection: {e}")
        if state is not None:
            state.close()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    ok, message = verify_with_ffprobe(Path(args.path))
    if not ok:
        print(message)
        return 2
    native_ok, native_message = verify_native_metadata(Path(args.path))
    if native_ok:
        print(f"{message}, {native_message}")
    else:
        print(f"{message}; warning: {native_message}")
    return 0


def cmd_calibrate_native(args: argparse.Namespace) -> int:
    _assert_apple_silicon()
    collection = _collection(args.collection)
    if collection is None:
        raise StemBatchError("collection not found")
    entry = collection.find(Path(args.audio))
    if entry is None or not entry.audio_id:
        raise StemBatchError("track not found in collection or missing AUDIO_ID")

    logged_path = logged_native_stem_path(
        track=Track(
            path=entry.path,
            title=entry.title or Path(args.audio).stem,
            artist=entry.artist,
            audio_id=entry.audio_id,
        ),
        collection_path=collection.path,
        stems_dir=Path(args.stems_dir),
    )
    if logged_path:
        print(f"logged-native-path\t{logged_path}")

    existing = Path(args.stem_file).name if args.stem_file else ""
    computed_path = native_stem_path(Path(args.stems_dir), entry.audio_id)
    print(f"computed-native-path\t{computed_path}")
    candidates = candidate_stem_names(entry.audio_id)
    if existing:
        matches = calibration_matches(entry.audio_id, existing)
        for name, matched in matches.items():
            print(f"{name}\t{candidates[name]}\t{'match' if matched else 'no'}")
    else:
        for name, filename in candidates.items():
            print(f"{name}\t{filename}")
    return 0


def detect_physical_ram_mb() -> int:
    try:
        import subprocess
        output = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], encoding='utf-8')
        return int(output.strip()) // (1024 * 1024)
    except Exception:
        return 8192


def detect_gpu_cores() -> int | None:
    import subprocess
    import re
    try:
        result = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'], encoding='utf-8')
        match = re.search(r'Total Number of Cores:\s*(\d+)', result)
        if match:
            return int(match.group(1))
        match = re.search(r'GPU Cores:\s*(\d+)', result)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


def _get_available_ram_mb() -> int:
    import subprocess
    import re
    try:
        out = subprocess.check_output(["vm_stat"], encoding="utf-8")
        page_size = 4096
        try:
            page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], encoding="utf-8").strip())
        except Exception:
            pass
        free_pages = 0
        speculative_pages = 0
        for line in out.splitlines():
            if "Pages free:" in line:
                free_pages = int(re.search(r'Pages free:\s*(\d+)', line).group(1))
            elif "Pages speculative:" in line:
                speculative_pages = int(re.search(r'Pages speculative:\s*(\d+)', line).group(1))
        return (free_pages + speculative_pages) * page_size // (1024 * 1024)
    except Exception:
        return 4096


def _check_thermal_warning() -> bool:
    import subprocess
    try:
        out = subprocess.check_output(['pmset', '-g', 'therm'], encoding='utf-8')
        if "warning" in out.lower() and "no thermal warning" not in out.lower():
            return True
        if "performance warning" in out.lower() and "no performance warning" not in out.lower():
            return True
    except Exception:
        pass
    return False


def build_parser() -> argparse.ArgumentParser:
    # Calculate a safe default memory limit: 70% of total physical RAM
    total_ram_mb = detect_physical_ram_mb()
    safe_mem_limit = max(4096, int(total_ram_mb * 0.70))

    parser = argparse.ArgumentParser(
        prog="traktor-stem-batch",
        description="Batch music source separation and Traktor Pro 4 stem file builder."
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    doctor = sub.add_parser(
        "doctor",
        help="Run dependency diagnostics (checks ffmpeg, ffprobe, MP4Box, demucs-mlx, Traktor collection)."
    )
    doctor.set_defaults(func=cmd_doctor)

    scan = sub.add_parser(
        "scan",
        help="Scan a directory and show which files match tracks in your Traktor library."
    )
    scan.add_argument(
        "--music-dir",
        default=str(DEFAULT_MUSIC_DIR),
        help="Directory containing audio files to scan (default: %(default)s)"
    )
    scan.add_argument(
        "--collection",
        help="Path to Traktor's collection.nml file (auto-detected if omitted)"
    )
    scan.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit scan to the first N files (0 for no limit, default: %(default)s)"
    )
    scan.set_defaults(func=cmd_scan)

    process = sub.add_parser(
        "process",
        help="Separate audio into stems and package them into Traktor-compatible stem files."
    )
    process.add_argument(
        "--music-dir",
        default=str(DEFAULT_MUSIC_DIR),
        help="Directory containing audio files for batch processing (default: %(default)s)"
    )
    process.add_argument(
        "--audio",
        help="Path to a single audio track to process (ignores --music-dir)"
    )
    process.add_argument(
        "--collection",
        help="Path to Traktor's collection.nml file (auto-detected if omitted)"
    )
    process.add_argument(
        "--stems-dir",
        default=str(DEFAULT_TRAKTOR_STEMS_DIR),
        help="Target directory where Traktor linked stems will be saved (default: %(default)s)"
    )
    process.add_argument(
        "--mode",
        choices=("native",),
        default="native",
        help="Stem generation/linking mode (default: %(default)s)"
    )
    process.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default="demucs-mlx",
        help="Separation backend to use (choices: %(choices)s, default: %(default)s)"
    )
    process.add_argument(
        "--model",
        choices=SUPPORTED_MLX_MODELS,
        default="htdemucs",
        help="MLX model to download/run from huggingface mlx-community/demucs-mlx-fp16 (choices: %(choices)s, default: %(default)s)"
    )
    process.add_argument(
        "--shifts",
        type=int,
        default=0,
        help="Number of random shifts for separation. 0 (default) is fastest, 1 is recommended for quality (default: %(default)s)"
    )
    process.add_argument(
        "--track-workers",
        type=int,
        default=1,
        help="Number of parallel track workers (default: %(default)s)"
    )
    process.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="MLX batch size for chunk separation (0 for auto-detect based on half of GPU cores, default: %(default)s)"
    )
    process.add_argument(
        "--mlx-cache-limit-mb",
        type=int,
        default=None,
        help="MLX compilation cache limit in MB (default: derived from profile)"
    )
    process.add_argument(
        "--mlx-memory-limit-mb",
        type=int,
        default=None,
        help="MLX memory limit in MB (default: derived from profile)"
    )
    process.add_argument(
        "--profile",
        choices=["silent", "balanced", "extreme", "ultra"],
        default="balanced",
        help="Resource profile adjusting batch sizes, CPU thread workers, and cooling pauses (choices: %(choices)s, default: %(default)s)"
    )
    process.add_argument(
        "--cooling-pause",
        type=float,
        default=None,
        help="Cooling pause in seconds between tracks to allow CPU/GPU to cool down (default: derived from profile)"
    )
    process.add_argument(
        "--overlap",
        type=float,
        default=0.1,
        help="Overlap between chunks for separation (default: %(default)s)"
    )
    process.add_argument(
        "--work-dir",
        default=str(DEFAULT_STATE_DIR / "work"),
        help="Directory for temporary processing files (default: %(default)s)"
    )
    process.add_argument(
        "--state-db",
        default=str(DEFAULT_STATE_DIR / "jobs.sqlite3"),
        help="Path to SQLite database tracking job states (default: %(default)s)"
    )
    process.add_argument(
        "--native-algorithm",
        default="traktor-md5-audio-id",
        help="Algorithm for calculating Traktor stem file name hash (default: %(default)s)"
    )
    process.add_argument(
        "--codec",
        default=_detect_best_aac_codec(),
        help="Audio codec for stem streams (default: %(default)s)"
    )
    process.add_argument(
        "--bitrate",
        type=int,
        default=256000,
        help="Bitrate per stem channel in bps (default: %(default)s)"
    )
    process.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Output sample rate in Hz (default: %(default)s)"
    )
    process.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit processing to N files (0 for no limit, default: %(default)s)"
    )
    process.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing and overwriting of existing stem files"
    )
    process.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Reprocess tracks even if a valid stem already exists"
    )
    process.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned output paths without executing separation"
    )
    process.add_argument(
        "--no-update-collection",
        dest="update_collection",
        action="store_false",
        help="Do not update the Traktor collection.nml with generated-stem flags"
    )
    process.add_argument(
        "--allow-running-traktor",
        action="store_true",
        help="Allow running while Traktor Pro 4 is active"
    )
    process.add_argument(
        "--verbose-backend",
        action="store_true",
        help="Show MLX separation progress bar and detailed logs"
    )
    process.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other files if a separation error occurs"
    )
    process.set_defaults(update_collection=True)
    process.set_defaults(func=cmd_process)

    verify = sub.add_parser(
        "verify",
        help="Validate that a generated .stem.mp4 file has exactly five streams and correct Traktor metadata."
    )
    verify.add_argument("path", help="Path to the stem file to verify")
    verify.set_defaults(func=cmd_verify)

    calibrate = sub.add_parser(
        "calibrate-native",
        help="Calibrate and verify the expected native filename for Traktor's MD5 linked stem path."
    )
    calibrate.add_argument(
        "--collection",
        help="Path to Traktor's collection.nml file (auto-detected if omitted)"
    )
    calibrate.add_argument(
        "--audio",
        required=True,
        help="Path to the source audio file"
    )
    calibrate.add_argument(
        "--stems-dir",
        default=str(DEFAULT_TRAKTOR_STEMS_DIR),
        help="Target directory where Traktor linked stems are saved (default: %(default)s)"
    )
    calibrate.add_argument(
        "--stem-file",
        help="Optional stem filename to check calibration matches"
    )
    calibrate.set_defaults(func=cmd_calibrate_native)

    return parser


def main(argv: list[str] | None = None) -> int:
    from .container.cleanup import init_cleanup_handlers
    init_cleanup_handlers()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except StemBatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
