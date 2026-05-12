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
model: htdemucs
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

## Notes

- Original FLAC files are not moved, deleted, or renamed.
- The first run downloads the HTDemucs checkpoint to the local model cache.
- Per-track temporary files in `.stembatch/work` are removed after the stem is
  written, verified, and collection state is updated. The empty `.stembatch/work`
  folder is removed too. Failed tracks keep their work folder for debugging.
- The downloaded HTDemucs model cache is kept between runs so the model is not
  downloaded again for every track.
- MLX free-cache is limited to 512 MB by default and cleared after each track.
  MLX memory is limited to 8192 MB by default. Use
  `--mlx-memory-limit-mb 0` to keep the MLX default, or raise the value if a
  very long track needs it.
- The output is a Traktor native `.stem.mp4` with five AAC streams:
  `master, drums, bass, other, vocals`.
- Linked stems require the track to exist in `collection.nml` with `AUDIO_ID`.
- The tool sets Traktor's generated-stem flag in `collection.nml` and creates a
  `.bak` copy before the first collection write.

Useful checks:

```bash
uv run traktor-stem-batch scan \
  --music-dir /Users/user/Music/DJ \
  --collection "/Users/user/Documents/Native Instruments/Traktor 4.5.0/collection.nml" \
  --limit 10

uv run traktor-stem-batch verify /path/to/file.stem.mp4
```
