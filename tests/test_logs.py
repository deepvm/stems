from pathlib import Path
import tempfile
import unittest

from traktor_stem_batch.models import Track
from traktor_stem_batch.traktor.logs import logged_native_stem_path, parse_stem_log


LOG = '''2026-05-12 01:01:38 - Finished stem separation job for track "Good Vibrations" - result: successful: stem file stored to "/Users/user/Music/Traktor/Stems/056/YNB5YZACIWLCQDMCDFGUDOYEB45D.stem.mp4", with offset 0us, elapsed time: 0m 45s
'''


class LogTests(unittest.TestCase):
    def test_parse_stored_native_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Traktor.log"
            path.write_text(LOG)
            entries = parse_stem_log(path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].title, "Good Vibrations")
            self.assertEqual(entries[0].path.parent.name, "056")

    def test_logged_native_path_uses_collection_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir = root / "Logs"
            logs_dir.mkdir()
            (logs_dir / "Traktor.log").write_text(
                '2026-05-12 01:01:38 - Finished stem separation job for track "Title" - '
                f'result: successful: stem file stored to "{root}/Stems/001/ABC.stem.mp4", '
                "with offset 0us, elapsed time: 0m 1s"
            )
            path = logged_native_stem_path(
                track=Track(root / "song.flac", "Title", audio_id="AUDIO"),
                collection_path=root / "collection.nml",
                stems_dir=root / "Stems",
            )
            self.assertEqual(path, root / "Stems/001/ABC.stem.mp4")


if __name__ == "__main__":
    unittest.main()
