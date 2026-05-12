import unittest

from pathlib import Path

from traktor_stem_batch.traktor.native import (
    candidate_stem_names,
    native_stem_path,
    stem_bucket_from_audio_id,
    stem_filename_from_audio_id,
)


GOOD_VIBRATIONS_AUDIO_ID = (
    "ALsd7/94mt9nd3ZDRmaIZTNIdWiGQ0ZmiHVDTnd3dkNWdoh1M1+FaPdUX4ea///////////////////////////"
    "N3v///////////////////////6zu/5iYh1RWd5l2Q1mneadUVnaal0NNZDM0Q0RDNEQzTXM01kNNdEXf////"
    "////////////////////////////////////////////////////hTM0MjNDIzQyNEQiNDIzRCM0MjaZMzRDNpl"
    "ERUNImUN5lDeaqr3v/Hp4ZUSP//////////////////////////////////////////////////3d3YcgAA=="
)


class NativeTests(unittest.TestCase):
    def test_candidate_names_are_native_stem_mp4(self):
        candidates = candidate_stem_names(GOOD_VIBRATIONS_AUDIO_ID)
        self.assertEqual(
            candidates["traktor-md5-audio-id"],
            "YNB5YZACIWLCQDMCDFGUDOYEB45D.stem.mp4",
        )

    def test_selected_algorithm(self):
        filename = stem_filename_from_audio_id(GOOD_VIBRATIONS_AUDIO_ID)
        self.assertEqual(filename, "YNB5YZACIWLCQDMCDFGUDOYEB45D.stem.mp4")

    def test_native_path_uses_bucket_and_hash(self):
        path = native_stem_path(Path("/Stems"), GOOD_VIBRATIONS_AUDIO_ID)
        self.assertEqual(path, Path("/Stems/056/YNB5YZACIWLCQDMCDFGUDOYEB45D.stem.mp4"))
        self.assertEqual(path.parent.name, stem_bucket_from_audio_id(GOOD_VIBRATIONS_AUDIO_ID))


if __name__ == "__main__":
    unittest.main()
