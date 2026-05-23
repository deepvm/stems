# Traktor Stem Batch

Fast linked Stem generation for Traktor Pro 4 on Apple Silicon.

The tool keeps the original music files in place and writes native Traktor
linked stems to:

```text
/Users/user/Music/Traktor/Stems/{bucket}/{hash}.stem.mp4
```

Only one separation path is supported:

```text
backend: demucs-mlx
model: htdemucs
track-workers: 1
```

The backend uses the Python MLX port from `andrade0/demucs-mlx` through its API.

## Install

```bash
brew install uv ffmpeg gpac git
uv python install 3.14
uv sync
```

Check the environment:

```bash
uv run traktor-stem-batch doctor
```

The first real run downloads the HTDemucs weights. Close Traktor before writing
linked stems or collection flags.

## Run

Batch folder:

```bash
uv run traktor-stem-batch process \
  --music-dir /Users/user/Music/DJ \
  --collection "/Users/user/Documents/Native Instruments/Traktor 4.5.0/collection.nml" \
  --stems-dir /Users/user/Music/Traktor/Stems \
  --mode native \
  --backend demucs-mlx \
  --model htdemucs \
  --shifts 1 \
  --track-workers 1 \
  --force \
  --continue-on-error
```

Single file:

```bash
uv run traktor-stem-batch process \
  --audio "/Users/user/Music/DJ/t0ni, Antonia XM, Broosnica - Keepsake (Ultimate Vip).mp3" \
  --collection "/Users/user/Documents/Native Instruments/Traktor 4.5.0/collection.nml" \
  --stems-dir /Users/user/Music/Traktor/Stems \
  --mode native \
  --backend demucs-mlx \
  --model htdemucs \
  --shifts 1 \
  --track-workers 1 \
  --force \
  --continue-on-error
```

Existing valid linked stems are skipped automatically. Use
`--reprocess-existing` only when you intentionally want to replace a valid stem.

## Notes

- Original audio files are not moved, deleted, or renamed.
- The output is a Traktor native `.stem.mp4` with five AAC streams:
  `master, drums, bass, other, vocals`.
- The native filename is Traktor's hash path, so the stem stays linked to the
  original track in `collection.nml`.
- Linked stems require the track to exist in `collection.nml` with `AUDIO_ID`.
- The tool sets Traktor's generated-stem flag in `collection.nml` and creates a
  `.bak` copy before the first collection write.
- Temporary packaging files are created in system temp folders and deleted after
  each track. `.stembatch/work` is also cleaned after successful tracks.
- The local state database is `.stembatch/jobs.sqlite3`; it is small and safe to
  keep.
- `--shifts 1` is the default quality setting. `--shifts 0` is allowed and can be
  tested for speed, but `1` is the recommended command.
- `--track-workers 1` is intentional. Running multiple MLX separations at once
  usually slows down Apple Silicon and raises memory pressure.
- `--force` is kept for command compatibility. It does not rebuild an existing
  valid linked stem; use `--reprocess-existing` for that.

## Commands

### `doctor`

Checks system tools, Python dependencies, and the default Traktor collection
path.

```bash
uv run traktor-stem-batch doctor
```

### `scan`

Scans a music folder and shows which files are matched to Traktor
`collection.nml`.

```bash
uv run traktor-stem-batch scan \
  --music-dir /Users/user/Music/DJ \
  --collection "/Users/user/Documents/Native Instruments/Traktor 4.5.0/collection.nml" \
  --limit 10
```

Flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--music-dir PATH` | `/Users/user/Music/DJ` | Folder with source audio files. |
| `--collection PATH` | auto-detected when possible | Traktor `collection.nml`. |
| `--limit N` | `0` | Show only first `N` tracks. `0` means no limit. |

### `process`

Creates linked native Traktor stems.

```bash
uv run traktor-stem-batch process [flags]
```

Flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--music-dir PATH` | `/Users/user/Music/DJ` | Batch-process audio files from this folder. Ignored when `--audio` is set. |
| `--audio PATH` | unset | Process one exact audio file. The file must be in Traktor collection for linked stems. |
| `--collection PATH` | auto-detected when possible | Traktor `collection.nml`. Required for native linked stems. |
| `--stems-dir PATH` | `/Users/user/Music/Traktor/Stems` | Traktor native stems folder. |
| `--mode native` | `native` | Output mode. Only `native` is supported. |
| `--backend demucs-mlx` | `demucs-mlx` | Separation backend. Only `demucs-mlx` is supported. |
| `--model htdemucs` | `htdemucs` | Separation model. Only `htdemucs` is supported. |
| `--shifts N` | `1` | Extra separation shifts. Higher is slower. |
| `--track-workers 1` | `1` | Track worker count. Only `1` is supported. |
| `--batch-size N` | `1` | Compatibility flag; the API backend keeps one track in memory. |
| `--mlx-cache-limit-mb N` | `512` | MLX cache limit in MB. |
| `--mlx-memory-limit-mb N` | `8192` | MLX memory limit in MB. Use `0` to keep the MLX default. |
| `--work-dir PATH` | `.stembatch/work` | Temporary per-track work folder. Successful tracks clean this automatically. |
| `--state-db PATH` | `.stembatch/jobs.sqlite3` | Local processing state database. |
| `--native-algorithm NAME` | `traktor-md5-audio-id` | Hash algorithm for Traktor native stem path. |
| `--codec CODEC` | `aac` | Stem audio codec. |
| `--bitrate BPS` | `256000` | AAC bitrate for each stream. |
| `--sample-rate HZ` | `44100` | Output sample rate. |
| `--limit N` | `0` | Batch mode only: process first `N` tracks. `0` means no limit. |
| `--force` | off | Compatibility flag. Existing valid stems are still skipped. |
| `--reprocess-existing` | off | Rebuild even when a valid linked stem already exists. |
| `--dry-run` | off | Print planned output paths without writing stems or collection flags. |
| `--no-update-collection` | off | Do not write generated-stem flags to `collection.nml`. |
| `--allow-running-traktor` | off | Allow writes while Traktor is running. Use only if you know why. |
| `--verbose-backend` | off | Show backend progress/log output. |
| `--continue-on-error` | off | Continue batch processing after a failed track. |

### `verify`

Checks that a `.stem.mp4` has five audio streams and Traktor native metadata.

```bash
uv run traktor-stem-batch verify /path/to/file.stem.mp4
```

### `calibrate-native`

Shows the expected native Traktor stem path for a collection track.

```bash
uv run traktor-stem-batch calibrate-native \
  --audio "/path/to/audio.flac" \
  --collection "/path/to/collection.nml" \
  --stems-dir /Users/user/Music/Traktor/Stems
```

Flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--collection PATH` | auto-detected when possible | Traktor `collection.nml`. |
| `--audio PATH` | required | Audio file to find in the collection. |
| `--stems-dir PATH` | `/Users/user/Music/Traktor/Stems` | Traktor native stems folder. |
| `--stem-file PATH` | unset | Existing stem filename to compare against computed candidates. |
