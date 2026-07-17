# Traktor Stem Batch

Traktor Stem Batch is a high-performance command-line utility designed for fast, batch-oriented linked Stem file generation. It is optimized for Apple Silicon (M-series chips) and Traktor Pro 4, enabling DJs to separate standard audio tracks into Traktor-compatible Stem files (.stem.mp4) with minimal processing overhead.

By utilizing the MLX framework, direct Hugging Face FP16 model loading, vectorized audio stacking, and a multi-threaded overlapped pipeline, this utility achieves significant speedups compared to traditional PyTorch-based CPU/GPU separation pipelines.

---

## Performance Metrics

The following benchmarks were recorded on a MacBook Pro with an M4 Pro chip (12-core CPU, 16-core GPU, 24GB Unified Memory) using a 4-minute stereo audio track (44.1 kHz, 16-bit WAV, 39.4 MB) under `--profile extreme`.

### End-to-End Separation Time
* **PyTorch (Standard CPU/GPU pipeline):** 17.8 seconds
* **MLX Uncompiled GPU pipeline:** 6.1 seconds
* **Metal Compiled GPU pipeline (demucs-mlx FP16 + GPU STFT/iSTFT):** 3.0 seconds (1.8 seconds for a 2m21s track)
* **Performance Gain:** ~6.0x overall speedup over PyTorch; runs at ~80x real-time on GPU.

### Model Loading & Startup Latency
* **Standard load (PyTorch checkpoint download and dynamic conversion):** ~2.5 seconds
* **Direct FP16 Safetensors load (from Hugging Face cache):** ~10 milliseconds (0.01 seconds)
* **Performance Gain:** ~250x reduction in startup latency

### Hardware-Accelerated Audio Encoding
* **Standard software AAC encoder (FFmpeg aac):** 2.2 seconds (5 channels)
* **Apple AudioToolbox AAC encoder (FFmpeg aac_at):** 0.8 seconds (5 channels)
* **Performance Gain:** ~2.7x faster audio compression utilizing macOS system encoders.

### Vectorized Multi-Channel Audio Stacking (AAC Container Preparation)
* **Standard Python loops:** ~450 milliseconds
* **Vectorized NumPy stacking:** <15 milliseconds
* **Performance Gain:** ~30x speedup in metadata and channel preparation

### Pipeline Efficiency & Compilation
The application employs a fully integrated Metal GPU execution graph combined with a multi-threaded CPU pipeline:
1. **Model JIT Graph Compilation (`mx.compile`):** Fuses the entire separation pipeline (reflect padding, strided STFT, HTDemucs neural net, and OLA iSTFT) into a single Metal kernel. The GPU executes all operations in a single fused run with zero CPU-GPU transfer overhead.
2. **Pre-decoding (CPU thread):** Decodes the next track's audio to floating-point representation in-process while the GPU separates the current track.
3. **Hardware Encoding & Packaging (CPU thread pool):** Compresses the 5 separated tracks concurrently using macOS AudioToolbox hardware encoders, then multiplexes them into the final `.stem.mp4` container.

This parallel execution model ensures that GPU separation and CPU encoding run concurrently, completely hiding the packaging latency behind GPU execution.

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
