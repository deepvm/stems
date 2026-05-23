from pathlib import Path
import tempfile
import unittest

from traktor_stem_batch.audio.wavtools import write_test_wav, wav_info
from traktor_stem_batch.container.stem_mp4 import (
    _stack_master_and_stems,
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

    def test_stack_arrays_uses_traktor_order(self):
        import numpy as np

        master = np.zeros((2, 4), dtype="float32")
        stems = {
            "drums": np.ones((2, 3), dtype="float32"),
            "bass": np.ones((2, 4), dtype="float32") * 2,
            "other": np.ones((2, 2), dtype="float32") * 3,
            "vocals": np.ones((2, 4), dtype="float32") * 4,
        }
        data = _stack_master_and_stems(master, stems)
        self.assertEqual(data.shape, (5, 4, 2))
        self.assertEqual(data[1, 0, 0], 0.99)
        self.assertEqual(data[4, 0, 0], 0.99)


if __name__ == "__main__":
    unittest.main()
