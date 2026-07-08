# Traktor Stem Batch (Optimized)

Fast, optimized linked Stem generation for Traktor Pro 4 on Apple Silicon.

This project is highly optimized for MacBook M4 Pro (and other Apple Silicon chips) using MLX, providing a **~3x end-to-end speedup** (processing a 4-minute track in **~6.1 seconds** instead of 17.8s) through:
* **Hybrid FP16/FP32 Inference:** GPU-heavy neural layers run in half-precision (Float16) while STFT calculations run in Float32.
* **Instant Start-up (PyTorch Bypass):** Weights are converted once and saved locally as `htdemucs.safetensors`, bypassing PyTorch and start-up weight conversion on subsequent runs (load time drops from ~2.5s to **~10ms**).
* **Vectorized FFmpeg Writes:** Audio channel stacking is fully vectorized in NumPy, eliminating Python loop overhead during M4A encoding.
* **CPU-GPU Overlapped Pipeline:** Concurrently pre-decodes the next track and post-encodes the completed track in the background while the GPU separates.
* **Traktor Collection Caching:** Fast $O(N)$ cached collection parsing instead of redundant $O(M \times N)$ linear scans.

---

## 📦 Installation

Install required system packages, Python 3.14, and project dependencies:

```bash
brew install uv ffmpeg gpac git
uv python install 3.14
uv sync
```

Verify your environment using the `doctor` command:

```bash
uv run traktor-stem-batch doctor
```

---

## ⏱️ How to Test & Benchmark a Single Track

To measure the separation speed and correctness on a single audio track, follow these steps.

### Step 1: Run the First Time (Pre-Caching)
The very first run will download the PyTorch `.th` weights (which takes some time depending on your connection), convert them to MLX format, and automatically save the optimized Float16 weights to `~/.cache/demucs_mlx/htdemucs.safetensors` and the model configuration to `~/.cache/demucs_mlx/htdemucs.json`.

```bash
uv run traktor-stem-batch process \
  --audio "/Users/user/Music/DJ/RÜFÜS DU SOL - Innerbloom (Radio Edit).mp3" \
  --allow-running-traktor \
  --verbose-backend
```

### Step 2: Run Benchmarks (Cached & Fast)
You **MUST** pass the `--force` (or `--reprocess-existing`) flag; otherwise, the tool will see the existing stem file and skip it.

```bash
uv run traktor-stem-batch process \
  --audio "/Users/user/Music/DJ/RÜFÜS DU SOL - Innerbloom (Radio Edit).mp3" \
  --allow-running-traktor \
  --verbose-backend \
  --force
```

The output will show:
`[1/1] done: RÜFÜS DU SOL - Innerbloom (Radio Edit) (6.1s)`

---

## 🏃 Run Commands

### Batch Folder Processing
To batch process a folder of DJ music:

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
  --continue-on-error
```

Existing valid stems are skipped automatically.

---

## 🛠️ CLI Commands Reference

### `doctor`
Checks dependencies, tools, and auto-detects the Traktor collection file.
```bash
uv run traktor-stem-batch doctor
```

### `scan`
Scans a folder and displays which files are matched in your Traktor library.
```bash
uv run traktor-stem-batch scan --music-dir /Users/user/Music/DJ
```

### `verify`
Validates that a generated `.stem.mp4` file has exactly five streams and correct Traktor metadata.
```bash
uv run traktor-stem-batch verify /path/to/track.stem.mp4
```

### `calibrate-native`
Prints candidates and verifies the expected native filename for Traktor's MD5 linked stem path.
```bash
uv run traktor-stem-batch calibrate-native \
  --audio "/path/to/track.mp3" \
  --collection "/path/to/collection.nml"
```

---

## ⚙️ Configuration Flags (for `process`)

| Flag | Default | Description |
| --- | --- | --- |
| `--audio PATH` | None | Process one specific track. Track must exist in Traktor collection for linked stems. |
| `--music-dir PATH` | `/Users/user/Music/DJ` | Music folder for batch mode (ignored when `--audio` is set). |
| `--collection PATH` | Auto-detected | Path to Traktor's `collection.nml`. |
| `--stems-dir PATH` | `/Users/user/Music/Traktor/Stems` | Target folder for Traktor native stems. |
| `--reprocess-existing` | Off | Force reprocessing even if a valid stem already exists. |
| `--force` | Off | Overwrites existing valid stems (acts identically to `--reprocess-existing` for testing). |
| `--shifts N` | `1` | Extra separation shifts. Quality setting: `1` is recommended, `0` is allowed for max speed. |
| `--verbose-backend` | Off | Show separation progress bar and backend logs. |
| `--allow-running-traktor` | Off | Allow processing while Traktor Pro 4 is open. |
| `--codec CODEC` | `aac` | Stems audio codec (default: `aac`). |
| `--bitrate BPS` | `256000` | AAC bitrate for each of the 5 channels. |
| `--sample-rate HZ` | `44100` | Output sample rate. |
| `--dry-run` | Off | Prints the planned output paths without executing separation. |
