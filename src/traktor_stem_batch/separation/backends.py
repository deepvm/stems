from __future__ import annotations

import contextlib
import gc
import io
import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..errors import BackendError, MissingDependencyError

SUPPORTED_BACKENDS = ("demucs-mlx",)
SUPPORTED_MLX_MODELS = ("htdemucs",)
STEM_NAMES = ("drums", "bass", "other", "vocals")


@dataclass(frozen=True)
class SeparatedAudio:
    master: object
    stems: dict[str, object]
    sample_rate: int


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
        if model != "htdemucs":
            raise BackendError("only --model htdemucs is supported")
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

    def _load_model(self):
        if self._model is not None:
            return self._model
        self._configure_mlx()
        try:
            from demucs_mlx import pretrained
        except Exception as exc:
            raise MissingDependencyError("demucs-mlx is not installed. Run `uv sync`.") from exc
        try:
            pretrained.REMOTE_ROOT = Path(str(resources.files("demucs").joinpath("remote")))
            if self.verbose:
                self._model = pretrained.load_model(self.model_name)
            else:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self._model = pretrained.load_model(self.model_name)
        except Exception as exc:
            raise BackendError(f"could not load {self.model_name}: {exc}") from exc
        return self._model

    def _configure_mlx(self) -> None:
        if self.cache_limit_mb >= 0:
            os.environ["DEMUCS_MLX_CACHE_LIMIT"] = str(self.cache_limit_mb * 1024 * 1024)
        if self.memory_limit_mb > 0:
            os.environ["MLX_MEMORY_LIMIT"] = str(self.memory_limit_mb * 1024 * 1024)
        import mlx.core as mx

        if self.cache_limit_mb >= 0:
            mx.set_cache_limit(self.cache_limit_mb * 1024 * 1024)
        if self.memory_limit_mb > 0:
            mx.set_memory_limit(self.memory_limit_mb * 1024 * 1024)

    @staticmethod
    def _load_audio(path: Path, sample_rate: int):
        if shutil.which("ffmpeg") is None:
            raise MissingDependencyError("ffmpeg command not found. Run `brew install ffmpeg`.")
        import numpy as np

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
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "ignore").strip()
            raise BackendError(message or f"could not decode {path}")
        return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2).T.copy()

    @staticmethod
    def _validate_stem_sum(master, stems: dict[str, object]) -> str:
        import numpy as np

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
        import mlx.core as mx
        import numpy as np
        from demucs_mlx.apply import apply_model

        model = self._load_model()
        sample_rate = int(model.samplerate)
        master = self._load_audio(input_path, sample_rate)
        try:
            out = apply_model(
                model,
                mx.array(master[None]),
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
