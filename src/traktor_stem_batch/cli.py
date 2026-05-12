from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .container.stem_mp4 import (
    build_package_plan,
    dependency_report,
    verify_native_metadata,
    verify_with_ffprobe,
    write_stem_file,
)
from .errors import StemBatchError
from .models import Track
from .paths import DEFAULT_MUSIC_DIR, DEFAULT_STATE_DIR, DEFAULT_TRAKTOR_STEMS_DIR, find_default_collection
from .scanner import scan_music_dir
from .separation.backends import build_backend
from .state import JobState
from .traktor.native import calibration_matches, candidate_stem_names, native_stem_path
from .traktor.logs import logged_native_stem_path
from .traktor.nml import TraktorCollection
from .util import human_bool, sanitize_filename


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
    collection.write_atomic()
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
    if args.backend != "demucs-mlx":
        raise StemBatchError("only --backend demucs-mlx is supported")
    if args.model != "htdemucs":
        raise StemBatchError("only --model htdemucs is supported")
    if args.shifts != 1:
        raise StemBatchError("only --shifts 1 is supported")
    if args.track_workers != 1:
        raise StemBatchError("only --track-workers 1 is supported")
    if args.mlx_cache_limit_mb < 0:
        raise StemBatchError("--mlx-cache-limit-mb must be 0 or greater")
    if args.mlx_memory_limit_mb < 0:
        raise StemBatchError("--mlx-memory-limit-mb must be 0 or greater")


def _process_item(
    *,
    item: ProcessItem,
    backend,
    work_dir_root: Path,
    codec: str,
    bitrate: int,
    sample_rate: int,
    native: bool,
    dry_run: bool,
) -> ProcessResult:
    started = time.monotonic()
    work_dir = work_dir_root / sanitize_filename(item.track.path.stem)
    stem_set = backend.separate(item.track.path, work_dir=work_dir, dry_run=dry_run)
    plan = build_package_plan(
        stems=stem_set,
        output=item.output,
        codec=codec,
        bitrate=bitrate,
        sample_rate=sample_rate,
        native=native,
    )
    write_stem_file(plan, dry_run=dry_run)
    if not dry_run:
        ok, message = verify_with_ffprobe(item.output)
        if not ok:
            raise StemBatchError(message)
        ok, message = verify_native_metadata(item.output)
        if not ok:
            raise StemBatchError(message)
    return ProcessResult(item=item, elapsed=time.monotonic() - started, work_dir=work_dir)


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"traktor-stem-batch {__version__}")
    report = dependency_report()
    for key in ("ffmpeg", "ffprobe", "MP4Box", "stempeg", "numpy", "soundfile"):
        print(f"{key}: {human_bool(report[key])}")
    git_available = shutil.which("git") is not None
    print(f"git: {human_bool(git_available)}")
    try:
        import demucs  # noqa: F401
        import demucs_mlx  # noqa: F401
        import mlx  # noqa: F401
    except Exception:
        demucs_mlx_available = False
    else:
        demucs_mlx_available = True
    print(f"demucs-mlx: {human_bool(demucs_mlx_available)}")

    found = find_default_collection()
    print(f"default collection: {found if found else 'not found'}")
    warnings: list[str] = []
    if not report["ffmpeg"] or not report["ffprobe"]:
        warnings.append("ffmpeg/ffprobe missing: run `brew install ffmpeg`")
    if not report["MP4Box"]:
        warnings.append("MP4Box missing: run `brew install gpac`")
    if not report["stempeg"]:
        warnings.append("stempeg missing: run `uv sync`")
    if not report["numpy"] or not report["soundfile"]:
        warnings.append("audio Python deps missing: run `uv sync`")
    if not demucs_mlx_available:
        warnings.append("demucs-mlx missing: run `uv sync`")
    if not git_available:
        warnings.append("git missing: run `brew install git`; uv uses it to install demucs-mlx")
    if warnings:
        print("")
        for warning in warnings:
            print(f"warning: {warning}")
        print("hint: install everything with `brew install uv ffmpeg gpac git` then `uv sync`")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
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
    _validate_process_args(args)
    music_dir = Path(args.music_dir).expanduser()
    stems_dir = Path(args.stems_dir).expanduser()
    collection = _collection(args.collection)
    if args.mode == "native" and collection is None:
        raise StemBatchError("native mode requires Traktor collection.nml")
    if not args.dry_run and not args.allow_running_traktor and _traktor_is_running():
        raise StemBatchError("Traktor Pro 4 is running. Close Traktor before writing stems or collection flags.")

    state = None if args.dry_run else JobState(Path(args.state_db))
    backend = build_backend(
        name=args.backend,
        model=args.model,
        shifts=args.shifts,
        verbose=args.verbose_backend,
        cache_limit_mb=args.mlx_cache_limit_mb,
        memory_limit_mb=args.mlx_memory_limit_mb,
    )
    tracks = _enrich_tracks(scan_music_dir(music_dir), collection)
    if args.limit:
        tracks = tracks[: args.limit]

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
            if not args.dry_run and not args.reprocess_existing:
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

        for item in pending:
            _status(f"[{item.index}/{item.total}] separate: {item.track.display_name}")
            try:
                result = _process_item(
                    item=item,
                    backend=backend,
                    work_dir_root=work_dir_root,
                    codec=args.codec,
                    bitrate=args.bitrate,
                    sample_rate=args.sample_rate,
                    native=args.mode == "native",
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                if state is not None:
                    state.set(item.track.path, "error", output_path=item.output, error=str(exc))
                if not args.continue_on_error:
                    raise
                _status(f"[{item.index}/{item.total}] error: {item.track.display_name}: {exc}")
            else:
                finish_result(result)
    except Exception as exc:
        if state is not None and "item" in locals():
            state.set(item.track.path, "error", output_path=item.output, error=str(exc))
        raise
    finally:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traktor-stem-batch")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor")
    doctor.set_defaults(func=cmd_doctor)

    scan = sub.add_parser("scan")
    scan.add_argument("--music-dir", default=str(DEFAULT_MUSIC_DIR))
    scan.add_argument("--collection")
    scan.add_argument("--limit", type=int, default=0)
    scan.set_defaults(func=cmd_scan)

    process = sub.add_parser("process")
    process.add_argument("--music-dir", default=str(DEFAULT_MUSIC_DIR))
    process.add_argument("--collection")
    process.add_argument("--stems-dir", default=str(DEFAULT_TRAKTOR_STEMS_DIR))
    process.add_argument("--mode", choices=("native",), default="native")
    process.add_argument(
        "--backend",
        choices=("demucs-mlx",),
        default="demucs-mlx",
    )
    process.add_argument("--model", default="htdemucs")
    process.add_argument("--shifts", type=int, default=1)
    process.add_argument("--track-workers", type=int, default=1)
    process.add_argument("--mlx-cache-limit-mb", type=int, default=512)
    process.add_argument("--mlx-memory-limit-mb", type=int, default=8192)
    process.add_argument("--work-dir", default=str(DEFAULT_STATE_DIR / "work"))
    process.add_argument("--state-db", default=str(DEFAULT_STATE_DIR / "jobs.sqlite3"))
    process.add_argument("--native-algorithm", default="traktor-md5-audio-id")
    process.add_argument("--codec", default="aac")
    process.add_argument("--bitrate", type=int, default=256000)
    process.add_argument("--sample-rate", type=int, default=44100)
    process.add_argument("--limit", type=int, default=0)
    process.add_argument("--force", action="store_true")
    process.add_argument("--reprocess-existing", action="store_true")
    process.add_argument("--dry-run", action="store_true")
    process.add_argument("--no-update-collection", dest="update_collection", action="store_false")
    process.add_argument("--allow-running-traktor", action="store_true")
    process.add_argument("--verbose-backend", action="store_true")
    process.add_argument("--continue-on-error", action="store_true")
    process.set_defaults(update_collection=True)
    process.set_defaults(func=cmd_process)

    verify = sub.add_parser("verify")
    verify.add_argument("path")
    verify.set_defaults(func=cmd_verify)

    calibrate = sub.add_parser("calibrate-native")
    calibrate.add_argument("--collection")
    calibrate.add_argument("--audio", required=True)
    calibrate.add_argument("--stems-dir", default=str(DEFAULT_TRAKTOR_STEMS_DIR))
    calibrate.add_argument("--stem-file")
    calibrate.set_defaults(func=cmd_calibrate_native)

    return parser


def main(argv: list[str] | None = None) -> int:
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
