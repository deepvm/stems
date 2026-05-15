# Traktor Stem Batch

Fast batch Stem generation for Traktor Pro 4 using only the MLX HTDemucs backend.

This project keeps the original music files in place and writes linked native
Traktor stems to:

```text
/Users/user/Music/Traktor/Stems/{bucket}/{hash}.stem.mp4
```

Only one separation mode is supported:

```text
backend: demucs-mlx
models: htdemucs, htdemucs_ft
shifts: 1
track-workers: 1
```

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

## Run

Close Traktor before writing stems or collection flags.

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

Existing valid linked stems are skipped automatically. Use
`--reprocess-existing` only when you intentionally want to replace a valid stem.

For the fine-tuned model, replace the model flag:

```bash
--model htdemucs_ft
```

`htdemucs_ft` uses the four fine-tuned HTDemucs checkpoints from Demucs'
`htdemucs_ft.yaml`. It is slower than `htdemucs`, but usually gives cleaner
separation.

## Notes

- Original FLAC files are not moved, deleted, or renamed.
- The first run downloads the selected HTDemucs checkpoint to the local model
  cache. `htdemucs_ft` downloads four fine-tuned checkpoints on first use.
- Per-track temporary files in `.stembatch/work` are removed after the stem is
  written, verified, and collection state is updated. The empty `.stembatch/work`
  folder is removed too. Failed tracks keep their work folder for debugging.
- The downloaded HTDemucs model cache is kept between runs so the selected model
  is not downloaded again for every track.
- MLX free-cache is limited to 512 MB by default and cleared after each track.
  MLX memory is limited to 8192 MB by default. Use
  `--mlx-memory-limit-mb 0` to keep the MLX default, or raise the value if a
  very long track needs it.
- The output is a Traktor native `.stem.mp4` with five AAC streams:
  `master, drums, bass, other, vocals`.
- Linked stems require the track to exist in `collection.nml` with `AUDIO_ID`.
- The tool sets Traktor's generated-stem flag in `collection.nml` and creates a
  `.bak` copy before the first collection write.

## Commands

```bash
uv run traktor-stem-batch scan \
  --music-dir /Users/user/Music/DJ \
  --collection "/Users/user/Documents/Native Instruments/Traktor 4.5.0/collection.nml" \
  --limit 10

uv run traktor-stem-batch verify /path/to/file.stem.mp4
```

### `doctor`

Checks system tools, Python dependencies, `demucs-mlx`, and the default Traktor
collection path.

```bash
uv run traktor-stem-batch doctor
```

### `scan`

Scans a music folder and shows which files are matched to Traktor
`collection.nml`.

```bash
uv run traktor-stem-batch scan [flags]
```

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

| Flag | Default | Description |
| --- | --- | --- |
| `--music-dir PATH` | `/Users/user/Music/DJ` | Batch-process audio files from this folder. Ignored when `--audio` is set. |
| `--audio PATH` | unset | Process one exact audio file. The file must be in Traktor collection for native linked stems. |
| `--collection PATH` | auto-detected when possible | Traktor `collection.nml`. Required for native linked stems. |
| `--stems-dir PATH` | `/Users/user/Music/Traktor/Stems` | Traktor native stems folder. |
| `--mode native` | `native` | Output mode. Only `native` is supported. |
| `--backend demucs-mlx` | `demucs-mlx` | Separation backend. Only `demucs-mlx` is supported. |
| `--model MODEL` | `htdemucs` | Separation model. Supported: `htdemucs`, `htdemucs_ft`. |
| `--shifts 1` | `1` | Demucs shift count. Only `1` is supported. |
| `--track-workers 1` | `1` | Track worker count. Only `1` is supported. |
| `--mlx-cache-limit-mb N` | `512` | MLX free-cache limit in MB. Use `0` to disable free-cache. |
| `--mlx-memory-limit-mb N` | `8192` | MLX memory guideline in MB. Use `0` to keep MLX default. |
| `--work-dir PATH` | `.stembatch/work` | Temporary per-track work folder. Successful tracks clean this automatically. |
| `--state-db PATH` | `.stembatch/jobs.sqlite3` | Local processing state database. |
| `--native-algorithm NAME` | `traktor-md5-audio-id` | Hash algorithm for Traktor native stem path. |
| `--codec CODEC` | `aac` | Stem audio codec. |
| `--bitrate BPS` | `256000` | AAC bitrate for each stream. |
| `--sample-rate HZ` | `44100` | Output sample rate. |
| `--limit N` | `0` | Batch mode only: process first `N` tracks. `0` means no limit. |
| `--force` | off | Compatibility flag. Kept for command consistency. |
| `--reprocess-existing` | off | Rebuild even when a valid linked stem already exists. |
| `--dry-run` | off | Print planned commands and output paths without writing stems or collection flags. |
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

Shows the expected native Traktor stem path for a collection track. This is
mostly useful when checking hash-path behavior against an existing stem file.

```bash
uv run traktor-stem-batch calibrate-native \
  --audio "/path/to/audio.flac" \
  --collection "/path/to/collection.nml" \
  --stems-dir /Users/user/Music/Traktor/Stems
```

| Flag | Default | Description |
| --- | --- | --- |
| `--collection PATH` | auto-detected when possible | Traktor `collection.nml`. |
| `--audio PATH` | required | Audio file to find in the collection. |
| `--stems-dir PATH` | `/Users/user/Music/Traktor/Stems` | Traktor native stems folder. |
| `--stem-file PATH` | unset | Existing stem filename to compare against computed candidates. |
