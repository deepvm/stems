from pathlib import Path
import shutil
import tempfile
import unittest

from traktor_stem_batch.audio.wavtools import write_test_wav, wav_info
from traktor_stem_batch.container.stem_mp4 import (
    _load_audio_tensor,
    build_package_plan,
    native_metadata_json,
)
from traktor_stem_batch.models import StemSet


class PackagePlanTests(unittest.TestCase):
    def test_plan_order_and_test_wavs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for index, name in enumerate(("master", "drums", "bass", "other", "vocals"), start=1):
                path = root / f"{name}.wav"
                write_test_wav(path, frequency=100.0 * index)
                paths[name] = path
                info = wav_info(path)
                self.assertEqual(info["channels"], 2)
                self.assertEqual(info["sample_rate"], 44100)

            stems = StemSet(**paths)
            plan = build_package_plan(stems, root / "out.stem.mp4")
            ordered_names = [item["slot"] for item in plan.to_dict()["streams"]]
            self.assertEqual(ordered_names, ["master", "drums", "bass", "other", "vocals"])

    def test_native_metadata_is_traktor_4_shape(self):
        payload = native_metadata_json()
        self.assertIn('"version":2', payload)
        self.assertIn('"offset":0', payload)
        self.assertIn('"name":"Vocals"', payload)
        self.assertNotIn('"Vox"', payload)

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not installed")
    def test_load_audio_tensor_resamples_to_target_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, name in enumerate(("master", "drums", "bass", "other", "vocals"), start=1):
                path = root / f"{name}.wav"
                write_test_wav(
                    path,
                    frequency=100.0 * index,
                    sample_rate=48000 if name == "master" else 44100,
                )
                paths.append(path)

            data, rate = _load_audio_tensor(paths, target_sample_rate=44100)

            self.assertEqual(rate, 44100)
            self.assertEqual(data.shape[0], 5)
            self.assertEqual(data.shape[2], 2)


if __name__ == "__main__":
    unittest.main()
