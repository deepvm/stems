from __future__ import annotations

import os
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import contextlib
import gc
import io
import json
import shutil
import subprocess
import inspect
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import mlx.core as mx
import mlx.utils as utils
import numpy as np

from demucs_mlx.htdemucs import HTDemucs, apply_conv1d
from demucs_mlx.pretrained import _load_weights
from demucs_mlx.utils import center_trim

from ..errors import BackendError, MissingDependencyError

SUPPORTED_BACKENDS = ("demucs-mlx",)
SUPPORTED_MLX_MODELS = ("htdemucs", "htdemucs_ft", "htdemucs_6s")
STEM_NAMES = ("drums", "bass", "other", "vocals")


@dataclass(frozen=True)
class SeparatedAudio:
    master: object
    stems: dict[str, object]
    sample_rate: int


class HybridHTDemucs(HTDemucs):
    """Subclass of HTDemucs to perform inference in hybrid Float16 precision.
    STFT and iSTFT are computed in Float32 (required by PyTorch STFT on CPU),
    while all heavy encoder/decoder/transformer layers run in Float16.
    """
    def __call__(self, mix: mx.array) -> mx.array:
        length = mix.shape[-1]
        length_pre_pad = None

        if self.use_train_segment:
            training_length = int(self.segment * self.samplerate)
            if mix.shape[-1] < training_length:
                length_pre_pad = mix.shape[-1]
                pad_widths = [(0, 0)] * (mix.ndim - 1) + [
                    (0, training_length - length_pre_pad)]
                mix = mx.pad(mix, pad_widths)

        # STFT (via PyTorch bridge) runs in float32
        z = self._spec(mix)
        mag = self._magnitude(z)
        x = mag

        B, C, Fq, T = x.shape

        # Normalize freq branch
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.sqrt(mx.mean((x - mean) ** 2, axis=(1, 2, 3), keepdims=True))
        x = (x - mean) / (1e-5 + std)

        # Normalize time branch
        xt = mix
        meant = mx.mean(xt, axis=(1, 2), keepdims=True)
        stdt = mx.sqrt(mx.mean((xt - meant) ** 2, axis=(1, 2), keepdims=True))
        xt = (xt - meant) / (1e-5 + stdt)

        # Cast inputs to float16 for neural network layers
        x = x.astype(mx.float16)
        xt = xt.astype(mx.float16)

        # Encoder
        saved = []      # freq skip connections
        saved_t = []    # time skip connections
        lengths = []    # freq branch lengths
        lengths_t = []  # time branch lengths

        for idx in range(len(self.encoder)):
            encode = self.encoder[idx]
            lengths.append(x.shape[-1])
            inject = None

            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt

            x = encode(x, inject)

            if idx == 0 and self.freq_emb is not None:
                frs = mx.arange(x.shape[-2])
                emb = self.freq_emb(frs)  # [Fr, C]
                emb = emb.T[None, :, :, None]  # [1, C, Fr, 1]
                emb = mx.broadcast_to(emb, x.shape)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        # CrossTransformer bottleneck
        if self.crosstransformer is not None:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = apply_conv1d(self.channel_upsampler, x)
                x = x.reshape(b, -1, f, t)
                xt = apply_conv1d(self.channel_upsampler_t, xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = x.reshape(b, -1, f * t)
                x = apply_conv1d(self.channel_downsampler, x)
                x = x.reshape(b, -1, f, t)
                xt = apply_conv1d(self.channel_downsampler_t, xt)

        # Decoder
        for idx in range(len(self.decoder)):
            decode = self.decoder[idx]
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0, :]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip_t = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip_t, length_t)

        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        # Cast outputs back to float32 before inverse STFT
        x = x.astype(mx.float32)
        xt = xt.astype(mx.float32)

        # Reconstruct freq output
        S = len(self.sources)
        x = x.reshape(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # Inverse STFT (via PyTorch bridge)
        x_np = np.array(x)
        z_out = self._mask(z, x)

        if self.use_train_segment:
            x_audio = self._ispec(z_out, training_length)
        else:
            x_audio = self._ispec(z_out, length)

        # Reconstruct time output
        if self.use_train_segment:
            xt = xt.reshape(B, S, -1, training_length)
        else:
            xt = xt.reshape(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]

        # Combine
        result = xt + x_audio

        if length_pre_pad is not None:
            result = result[..., :length_pre_pad]
        return result


def _add_weighted(out: mx.array, chunk: mx.array, weight: mx.array,
                  offset: int, length: int) -> mx.array:
    weighted = weight[None, None, None, :] * chunk
    before = out[..., :offset]
    middle = out[..., offset:offset + length] + weighted
    after = out[..., offset + length:]
    return mx.concatenate([before, middle, after], axis=-1)


def _add_weight(sum_w: mx.array, weight: mx.array,
                offset: int, length: int) -> mx.array:
    before = sum_w[:offset]
    middle = sum_w[offset:offset + length] + weight
    after = sum_w[offset + length:]
    return mx.concatenate([before, middle, after], axis=0)


def apply_model_batched(
    model: HybridHTDemucs,
    mix: mx.array,
    batch_size: int = 1,
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.,
    progress: bool = False,
    segment: float | None = None
) -> mx.array:
    """Apply model to input mixture using batched chunk processing and shifts."""
    if shifts:
        max_shift = int(0.5 * model.samplerate)
        out = mx.zeros((mix.shape[0], len(model.sources), mix.shape[1], mix.shape[2]))
        import random
        for shift_idx in range(shifts):
            offset = random.randint(0, max_shift)
            padded = mx.pad(mix, [(0, 0), (0, 0), (max_shift, max_shift)])
            shifted = padded[..., offset:offset + mix.shape[2] + max_shift - offset]
            res = apply_model_batched(
                model, shifted, batch_size=batch_size, shifts=0, split=split,
                overlap=overlap, transition_power=transition_power,
                progress=progress, segment=segment)
            out = out + res[..., max_shift - offset:max_shift - offset + mix.shape[2]]
        return out / shifts

    if not split:
        if segment is not None:
            valid_length = int(segment * model.samplerate)
        elif hasattr(model, 'valid_length'):
            valid_length = model.valid_length(mix.shape[-1])
        else:
            valid_length = mix.shape[-1]

        if mix.shape[-1] < valid_length:
            padded = mx.pad(mix, [(0, 0), (0, 0), (0, valid_length - mix.shape[-1])])
        else:
            padded = mix

        out = model(padded)
        mx.eval(out)
        return center_trim(out, mix.shape[-1])

    if segment is None:
        segment = float(model.segment)
    assert segment > 0.

    batch, channels, length = mix.shape
    segment_length = int(model.samplerate * segment)
    stride = int((1 - overlap) * segment_length)

    w_left = mx.arange(1, segment_length // 2 + 1, dtype=mx.float32)
    w_right = mx.arange(segment_length - segment_length // 2, 0, -1, dtype=mx.float32)
    weight = mx.concatenate([w_left, w_right])
    weight = (weight / mx.max(weight)) ** transition_power

    offsets = list(range(0, length, stride))
    if progress:
        try:
            import tqdm
            offsets = tqdm.tqdm(offsets, unit_scale=stride / model.samplerate, ncols=120, unit='seconds')
        except ImportError:
            pass

    # 1. Extract all chunks
    chunks = []
    chunk_info = []
    for offset in offsets:
        chunk_length = min(segment_length, length - offset)
        chunk = mix[..., offset:offset + chunk_length]
        if chunk.shape[-1] < segment_length:
            pad_r = segment_length - chunk.shape[-1]
            chunk = mx.pad(chunk, [(0, 0), (0, 0), (0, pad_r)])
        chunks.append(chunk)
        chunk_info.append((offset, chunk_length))

    # 2. Stack chunks: shape [N_chunks, C, T_segment]
    chunks_stacked = mx.concatenate(chunks, axis=0)

    # 3. Process stacked chunks in batches
    chunk_outs_list = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks_stacked[i:i + batch_size]
        batch_out = model(batch_chunks)
        mx.eval(batch_out)
        chunk_outs_list.append(batch_out)

    # 4. Concatenate batch outputs
    chunk_outs = mx.concatenate(chunk_outs_list, axis=0)

    # 5. Accumulate
    out = mx.zeros((batch, len(model.sources), channels, length))
    sum_weight = mx.zeros((length,))

    for idx, (offset, actual_len) in enumerate(chunk_info):
        chunk_out = chunk_outs[idx] # shape: [S, C, T_segment]
        chunk_out = chunk_out[None] # shape: [1, S, C, T_segment]
        
        chunk_out = chunk_out[..., :actual_len]
        w = weight[:actual_len]

        out = _add_weighted(out, chunk_out, w, offset, actual_len)
        sum_weight = _add_weight(sum_weight, w, offset, actual_len)

    out = out / mx.maximum(sum_weight[None, None, None, :], mx.array(1e-8))
    return out


class MlxDemucsBackend:
    name = "demucs-mlx"

    def __init__(
        self,
        model: str = "htdemucs",
        shifts: int = 1,
        verbose: bool = False,
        cache_limit_mb: int = 512,
        memory_limit_mb: int = 8192,
        batch_size: int = 1,
    ):
        if model not in SUPPORTED_MLX_MODELS:
            raise BackendError(f"only --model {', '.join(SUPPORTED_MLX_MODELS)} are supported")
        if shifts < 0:
            raise BackendError("--shifts must be 0 or greater")
        if batch_size <= 0:
            raise BackendError("--batch-size must be greater than 0")
        self.model_name = model
        self.shifts = shifts
        self.verbose = verbose
        self.cache_limit_mb = cache_limit_mb
        self.memory_limit_mb = memory_limit_mb
        self.batch_size = batch_size
        self._model = None

    def _load_model(self) -> HybridHTDemucs:
        if self._model is not None:
            return self._model
        self._configure_mlx()
        
        cache_dir = Path.home() / '.cache' / 'demucs_mlx'
        config_path = cache_dir / f"{self.model_name}_config.json"
        safetensors_path = cache_dir / f"{self.model_name}.safetensors"

        if not config_path.exists() or not safetensors_path.exists():
            import urllib.request
            import tempfile
            
            cache_dir.mkdir(parents=True, exist_ok=True)
            endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
            
            config_url = f"{endpoint}/mlx-community/demucs-mlx-fp16/resolve/main/{self.model_name}_config.json"
            safetensors_url = f"{endpoint}/mlx-community/demucs-mlx-fp16/resolve/main/{self.model_name}.safetensors"

            def download_file(url: str, dest: Path, desc: str):
                if self.verbose:
                    print(f"Downloading {desc} from {url}...")
                # Create a temporary file in the destination directory to avoid partial writes
                fd, temp_dest_str = tempfile.mkstemp(dir=str(dest.parent))
                temp_dest = Path(temp_dest_str)
                os.close(fd)
                try:
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req) as response, open(temp_dest, "wb") as out_file:
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                        while True:
                            chunk = response.read(1024 * 64)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            downloaded += len(chunk)
                            if self.verbose and total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownloading {desc}: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB/{total_size / 1024 / 1024:.1f}MB)", end="", flush=True)
                        if self.verbose:
                            print("", flush=True)
                    temp_dest.replace(dest)
                except Exception as exc:
                    if temp_dest.exists():
                        temp_dest.unlink()
                    raise BackendError(f"Failed to download {desc} from {url}: {exc}") from exc

            try:
                if not config_path.exists():
                    download_file(config_url, config_path, f"{self.model_name} config")
                if not safetensors_path.exists():
                    download_file(safetensors_url, safetensors_path, f"{self.model_name} weights")
            except Exception as exc:
                raise BackendError(f"Model download failed: {exc}") from exc

        # Load from downloaded files
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Support both Hugging Face nested 'kwargs' format and flat format
            params = config.get('kwargs', config)
            
            sig = inspect.signature(HTDemucs.__init__)
            valid_kwargs = {k: v for k, v in params.items() if k in sig.parameters}
            
            # Handle fraction strings like '39/5' for segment parameter
            if 'segment' in valid_kwargs and isinstance(valid_kwargs['segment'], str):
                from fractions import Fraction
                try:
                    valid_kwargs['segment'] = float(Fraction(valid_kwargs['segment']))
                except Exception:
                    pass
            
            model = HybridHTDemucs(**valid_kwargs)
            flat_state = mx.load(str(safetensors_path))
            
            # Ensure all values are mx.array
            flat_state_mx = {k: mx.array(v) if isinstance(v, np.ndarray) else v for k, v in flat_state.items()}
            _load_weights(model, flat_state_mx)
            
            # Cast model parameters to float16
            model.update(utils.tree_map(lambda x: x.astype(mx.float16), model.parameters()))
            mx.eval(model.parameters())
            
            self._model = model
        except Exception as exc:
            raise BackendError(f"Error loading model from downloaded files: {exc}") from exc
        return self._model

    def _configure_mlx(self) -> None:
        if self.cache_limit_mb >= 0:
            os.environ["DEMUCS_MLX_CACHE_LIMIT"] = str(self.cache_limit_mb * 1024 * 1024)
        if self.memory_limit_mb > 0:
            os.environ["MLX_MEMORY_LIMIT"] = str(self.memory_limit_mb * 1024 * 1024)
        
        # Setup MLX cache and memory limits
        if self.cache_limit_mb >= 0:
            mx.set_cache_limit(self.cache_limit_mb * 1024 * 1024)
        if self.memory_limit_mb > 0:
            mx.set_memory_limit(self.memory_limit_mb * 1024 * 1024)

    @staticmethod
    def _load_audio(path: Path, sample_rate: int):
        if shutil.which("ffmpeg") is None:
            raise MissingDependencyError("ffmpeg command not found. Run `brew install ffmpeg`.")

        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "2",
                "-ar",
                str(sample_rate),
                "-",
            ],
            check=False,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "ignore").strip()
            raise BackendError(message or f"could not decode {path}")
        return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2).T.copy()

    @staticmethod
    def _validate_stem_sum(master, stems: dict[str, object]) -> str:
        mix = np.asarray(master, dtype="float32")
        stem_sum = sum(np.asarray(stems[name], dtype="float32") for name in STEM_NAMES)
        length = min(mix.shape[1], stem_sum.shape[1])
        mix = mix[:, :length]
        stem_sum = stem_sum[:, :length]
        master_rms = float(np.sqrt(np.mean(np.asarray(mix, dtype="float64") ** 2)))
        if master_rms < 1e-5:
            return "silent master"
        ratio = float(np.sqrt(np.mean(np.asarray(stem_sum, dtype="float64") ** 2)) / master_rms)
        a = mix.reshape(-1)[::100].astype("float64")
        b = stem_sum.reshape(-1)[::100].astype("float64")
        corr = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
        if not np.isfinite(corr) or corr < 0.95 or ratio < 0.55 or ratio > 1.8:
            raise BackendError(f"stem sum sanity failed: corr={corr:.3f}, rms_ratio={ratio:.3f}")
        return f"corr={corr:.3f}, rms_ratio={ratio:.3f}"

    def separate(self, input_path: Path, dry_run: bool = False) -> SeparatedAudio | None:
        if dry_run:
            print(f"demucs-mlx {self.model_name} --shifts {self.shifts} {input_path}")
            return None

        model = self._load_model()
        sample_rate = int(model.samplerate)
        master = self._load_audio(input_path, sample_rate)
        try:
            out = apply_model_batched(
                model,
                mx.array(master[None]),
                batch_size=self.batch_size,
                shifts=self.shifts,
                split=True,
                overlap=0.25,
                progress=self.verbose,
                segment=None,
            )
            mx.eval(out)
            separated = np.array(out[0]).astype("float32", copy=False)
            stems = {name: separated[index] for index, name in enumerate(STEM_NAMES)}
            self._validate_stem_sum(master, stems)
            return SeparatedAudio(master=master, stems=stems, sample_rate=sample_rate)
        finally:
            mx.clear_cache()
            gc.collect()


def build_backend(
    name: str,
    model: str,
    shifts: int = 1,
    verbose: bool = False,
    cache_limit_mb: int = 512,
    memory_limit_mb: int = 8192,
    batch_size: int = 1,
) -> MlxDemucsBackend:
    if name not in SUPPORTED_BACKENDS:
        raise BackendError("only --backend demucs-mlx is supported")
    return MlxDemucsBackend(
        model=model,
        shifts=shifts,
        verbose=verbose,
        cache_limit_mb=cache_limit_mb,
        memory_limit_mb=memory_limit_mb,
        batch_size=batch_size,
    )
