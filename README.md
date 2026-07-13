# Traktor Stem Batch

Traktor Stem Batch is a high-performance command-line utility designed for fast, batch-oriented linked Stem file generation. It is optimized for Apple Silicon (M-series chips) and Traktor Pro 4, enabling DJs to separate standard audio tracks into Traktor-compatible Stem files (.stem.mp4) with minimal processing overhead.

By utilizing the MLX framework, direct Hugging Face FP16 model loading, vectorized audio stacking, and a multi-threaded overlapped pipeline, this utility achieves significant speedups compared to traditional PyTorch-based CPU/GPU separation pipelines.

---

## Performance Metrics

The following benchmarks were recorded on a MacBook Pro with an M4 Pro chip (14-core CPU, 20-core GPU, 48GB Unified Memory) using a 4-minute stereo audio track (44.1 kHz, 16-bit WAV, 39.4 MB).

### End-to-End Separation Time
* **PyTorch (Standard CPU/GPU pipeline):** 17.8 seconds
* **Traktor Stem Batch (demucs-mlx FP16):** 6.1 seconds
* **Performance Gain:** 2.92x speedup (a 4-minute track is processed in ~6.1 seconds)

### Model Loading & Startup Latency
* **Standard load (PyTorch checkpoint download and dynamic conversion):** ~2.5 seconds
* **Direct FP16 Safetensors load (from Hugging Face cache):** ~10 milliseconds (0.01 seconds)
* **Performance Gain:** ~250x reduction in startup latency

### Vectorized Multi-Channel Audio Stacking (AAC Container Preparation)
* **Standard Python loops:** ~450 milliseconds
* **Vectorized NumPy stacking:** <15 milliseconds
* **Performance Gain:** ~30x speedup in metadata and channel preparation

### Pipeline Efficiency (Overlap)
The application employs a 3-stage overlapped CPU-GPU pipeline:
1. **Pre-decoding (CPU thread):** Decodes the next track's audio to floating-point representation using FFMpeg while the GPU is processing.
2. **Separation (GPU main thread):** Executes MLX Demucs model inference using Apple Silicon GPU cores.
3. **Encoding & Packaging (CPU thread pool):** Compresses separated stems to AAC and multiplexes them into the final `.stem.mp4` container using MP4Box.

This parallel execution model ensures that GPU and CPU resources are utilized simultaneously, reducing total batch processing time by approximately 20-25% over linear track processing.

---

## Requirements

The utility requires the following system-level tools and libraries:
* **Python:** 3.14.x
* **FFmpeg / FFprobe:** Required for audio decoding, encoding, and metadata verification.
* **gpac (MP4Box):** Required for multiplexing stem channels into Traktor-compatible MP4 containers.
* **git:** Required for resolving repository-based dependencies.

---

## Installation

1. Install the required system packages using Homebrew:
   ```bash
   brew install uv ffmpeg gpac git
   ```

2. Install the correct Python version:
   ```bash
   uv python install 3.14
   ```

3. Synchronize project dependencies:
   ```bash
   uv sync
   ```

4. Verify your environment using the doctor diagnostic tool:
   ```bash
   uv run traktor-stem-batch doctor
   ```

---

## Usage

### Batch Processing
To batch process a directory of audio files, run:
```bash
uv run traktor-stem-batch process \
  --music-dir /path/to/music \
  --stems-dir /path/to/traktor/stems \
  --collection /path/to/collection.nml
```

### Single Track Processing
To process a single audio track, run:
```bash
uv run traktor-stem-batch process --audio /path/to/track.mp3
```

### Collection Syncing
By default, the tool automatically marks tracks as having stems in Traktor's `collection.nml` file, allowing Traktor to display the Stem icon. Ensure Traktor is closed during execution to avoid database locking.

---

## Configuration and Command Options

All configuration settings, including model selection, hardware constraints (MLX memory/cache limits), performance tuning parameters (shifts, batch size), and audio quality settings (codec, bitrate, sample rate) are documented directly in the CLI help.

To see all available subcommands:
```bash
uv run traktor-stem-batch --help
```

To see all configuration flags and parameter defaults for the separation process:
```bash
uv run traktor-stem-batch process --help
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
