from __future__ import annotations

import gc
import importlib.util
import logging
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..errors import BackendError, MissingDependencyError
from ..models import StemSet

SUPPORTED_MLX_MODELS = ("htdemucs", "htdemucs_ft")


@dataclass(frozen=True)
class MlxModelBag:
    models: list[object]
    weights: list[list[float]]
    sources: list[str]
    samplerate: int


def _run_command(cmd: list[str], verbose: bool = False) -> None:
    if verbose:
        subprocess.run(cmd, check=True)
        return
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return
    output = (result.stdout + "\n" + result.stderr).replace("\r", "\n")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    tail = "\n".join(lines[-30:])
    raise BackendError(
        "backend command failed: "
        + " ".join(shlex.quote(part) for part in cmd)
        + (f"\n{tail}" if tail else "")
    )


class MlxDemucsBackend:
    name = "demucs-mlx"

    def __init__(
        self,
        model: str = "htdemucs",
        shifts: int = 1,
        verbose: bool = False,
        cache_limit_mb: int = 512,
        memory_limit_mb: int = 8192,
    ):
        if model not in SUPPORTED_MLX_MODELS:
            raise BackendError("only --model htdemucs or --model htdemucs_ft is supported")
        if shifts != 1:
            raise BackendError("only --shifts 1 is supported")
        self.model_name = model
        self.shifts = shifts
        self.overlap = 0.25
        self.verbose = verbose
        self.cache_limit_mb = cache_limit_mb
        self.memory_limit_mb = memory_limit_mb
        self._model = None
        self._model_lock = threading.Lock()
        self._memory_configured = False

    @staticmethod
    def available() -> bool:
        return (
            importlib.util.find_spec("demucs") is not None
            and importlib.util.find_spec("demucs_mlx") is not None
            and importlib.util.find_spec("mlx") is not None
        )

    def _load_model(self):
        if not self.available():
            raise MissingDependencyError("demucs-mlx is not installed. Run `uv sync`.")
        with self._model_lock:
            self._configure_memory()
            if self._model is None:
                from demucs_mlx import pretrained

                pretrained.REMOTE_ROOT = Path(str(resources.files("demucs").joinpath("remote")))
                if not self.verbose:
                    logging.getLogger("demucs_mlx.pretrained").setLevel(logging.ERROR)
                if self.model_name == "htdemucs_ft":
                    self._model = self._load_fine_tuned_bag(pretrained.REMOTE_ROOT, pretrained)
                else:
                    self._model = pretrained.load_model(self.model_name)
        return self._model

    @staticmethod
    def _load_fine_tuned_bag(remote_root: Path, pretrained) -> MlxModelBag:
        import yaml

        with open(remote_root / "htdemucs_ft.yaml") as handle:
            config = yaml.safe_load(handle)
        signatures = config["models"]
        models = [pretrained.load_model(signature) for signature in signatures]
        weights = config.get("weights")
        if weights is None:
            weights = [[1.0 for _ in models[0].sources] for _ in models]
        first = models[0]
        for model in models[1:]:
            if model.sources != first.sources:
                raise BackendError("htdemucs_ft checkpoints have different source layouts")
            if model.samplerate != first.samplerate:
                raise BackendError("htdemucs_ft checkpoints have different sample rates")
        return MlxModelBag(
            models=models,
            weights=weights,
            sources=list(first.sources),
            samplerate=int(first.samplerate),
        )

    def _configure_memory(self) -> None:
        if self._memory_configured:
            return
        import mlx.core as mx

        if self.cache_limit_mb >= 0:
            mx.set_cache_limit(self.cache_limit_mb * 1024 * 1024)
        if self.memory_limit_mb > 0:
            mx.set_memory_limit(self.memory_limit_mb * 1024 * 1024)
        self._memory_configured = True

    @staticmethod
    def _release_transient_memory() -> None:
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
        gc.collect()

    def _prepare_wav(self, input_path: Path, out_dir: Path, sample_rate: int) -> Path:
        if shutil.which("ffmpeg") is None:
            raise MissingDependencyError("ffmpeg command not found")
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = out_dir / "input.wav"
        _run_command(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(input_path),
                "-map",
                "0:a:0",
                "-ac",
                "2",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_f32le",
                str(wav_path),
            ],
            verbose=self.verbose,
        )
        return wav_path

    @staticmethod
    def _mix(arrays):
        import numpy as np

        max_len = max(array.shape[0] for array in arrays)
        channels = arrays[0].shape[1]
        mixed = np.zeros((max_len, channels), dtype="float32")
        for array in arrays:
            mixed[: array.shape[0], : array.shape[1]] += array.astype("float32", copy=False)
        return mixed

    def _apply_single_model(self, model, mix, apply_model):
        return apply_model(
            model,
            mix,
            shifts=1,
            split=True,
            overlap=self.overlap,
            progress=self.verbose,
        )

    def _separate_to_arrays(self, loaded_model, mix, apply_model) -> dict[str, object]:
        import mlx.core as mx
        import numpy as np

        if isinstance(loaded_model, MlxModelBag):
            source_arrays = {}
            totals = {source: 0.0 for source in loaded_model.sources}
            for model, model_weights in zip(loaded_model.models, loaded_model.weights):
                sources = self._apply_single_model(model, mix, apply_model)
                mx.eval(sources)
                for index, (source, weight) in enumerate(zip(model.sources, model_weights)):
                    if not weight:
                        continue
                    array = np.array(sources[0, index]).T * float(weight)
                    if source in source_arrays:
                        source_arrays[source] = source_arrays[source] + array
                    else:
                        source_arrays[source] = array
                    totals[source] += float(weight)
                del sources
                self._release_transient_memory()
            for source, total in totals.items():
                if total:
                    source_arrays[source] = source_arrays[source] / total
            return source_arrays

        sources = self._apply_single_model(loaded_model, mix, apply_model)
        mx.eval(sources)
        try:
            return {
                source: np.array(sources[0, index]).T
                for index, source in enumerate(loaded_model.sources)
            }
        finally:
            del sources
            self._release_transient_memory()

    def separate(self, input_path: Path, work_dir: Path, dry_run: bool = False) -> StemSet:
        out_dir = work_dir / "separated" / self.model_name / input_path.stem
        if dry_run:
            print(
                "demucs-mlx "
                + " ".join(
                    shlex.quote(part)
                    for part in [
                        "-n",
                        self.model_name,
                        "--shifts",
                        "1",
                        str(input_path),
                    ]
                )
            )
            return StemSet(
                master=input_path,
                drums=out_dir / "drums.wav",
                bass=out_dir / "bass.wav",
                other=out_dir / "other.wav",
                vocals=out_dir / "vocals.wav",
            )

        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from demucs_mlx.apply import apply_model

        model = self._load_model()
        wav_path = self._prepare_wav(input_path, work_dir / "mlx_input", int(model.samplerate))
        wav = None
        source_arrays = None
        other_parts = None
        try:
            wav, sample_rate = sf.read(str(wav_path), always_2d=True, dtype="float32")
            source_arrays = self._separate_to_arrays(model, mx.array(wav.T[None, :, :]), apply_model)

            missing = [name for name in ("drums", "bass", "vocals") if name not in source_arrays]
            if missing:
                raise BackendError("demucs-mlx did not create expected stems: " + ", ".join(missing))

            other_parts = [
                array
                for source, array in source_arrays.items()
                if source not in {"drums", "bass", "vocals"}
            ]
            if not other_parts:
                raise BackendError("demucs-mlx did not create other stem")

            out_dir.mkdir(parents=True, exist_ok=True)
            paths = {
                "drums": out_dir / "drums.wav",
                "bass": out_dir / "bass.wav",
                "other": out_dir / "other.wav",
                "vocals": out_dir / "vocals.wav",
            }
            sf.write(str(paths["drums"]), source_arrays["drums"], sample_rate, subtype="FLOAT")
            sf.write(str(paths["bass"]), source_arrays["bass"], sample_rate, subtype="FLOAT")
            sf.write(str(paths["other"]), self._mix(other_parts), sample_rate, subtype="FLOAT")
            sf.write(str(paths["vocals"]), source_arrays["vocals"], sample_rate, subtype="FLOAT")
            return StemSet(
                master=input_path,
                drums=paths["drums"],
                bass=paths["bass"],
                other=paths["other"],
                vocals=paths["vocals"],
            )
        finally:
            del source_arrays, other_parts, wav
            self._release_transient_memory()


def build_backend(
    name: str,
    model: str,
    shifts: int = 1,
    verbose: bool = False,
    cache_limit_mb: int = 512,
    memory_limit_mb: int = 8192,
) -> MlxDemucsBackend:
    if name != "demucs-mlx":
        raise BackendError("only --backend demucs-mlx is supported")
    return MlxDemucsBackend(
        model=model,
        shifts=shifts,
        verbose=verbose,
        cache_limit_mb=cache_limit_mb,
        memory_limit_mb=memory_limit_mb,
    )
