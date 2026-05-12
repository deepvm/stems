from pathlib import Path
import tempfile
import unittest

from traktor_stem_batch.traktor.nml import TraktorCollection, nml_dir_to_path


NML = """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<NML VERSION="20"><HEAD COMPANY="www.native-instruments.com" PROGRAM="Traktor Pro 4"></HEAD>
<COLLECTION ENTRIES="1"><ENTRY AUDIO_ID="ABC" TITLE="Title" ARTIST="Artist">
<LOCATION DIR="/:Users/:user/:Music/:DJ/:" FILE="01 - Artist - Title.flac" VOLUME="Macintosh HD"></LOCATION>
</ENTRY></COLLECTION></NML>
"""


class NmlTests(unittest.TestCase):
    def test_location_conversion(self):
        path = nml_dir_to_path("/:Users/:user/:Music/:DJ/:", "song.flac")
        self.assertEqual(path, Path("/Users/user/Music/DJ/song.flac"))

    def test_parse_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "collection.nml"
            path.write_text(NML)
            collection = TraktorCollection(path)
            entry = collection.entries()[0]
            self.assertEqual(entry.audio_id, "ABC")
            self.assertEqual(entry.title, "Title")
            self.assertEqual(entry.path, Path("/Users/user/Music/DJ/01 - Artist - Title.flac"))

    def test_mark_generated_stem_adds_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "collection.nml"
            path.write_text(
                """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<NML VERSION="20"><COLLECTION ENTRIES="1"><ENTRY AUDIO_ID="ABC" TITLE="Title">
<LOCATION DIR="/:Users/:user/:Music/:DJ/:" FILE="song.flac" VOLUME="Macintosh HD"></LOCATION>
<INFO FLAGS="12"></INFO>
</ENTRY></COLLECTION></NML>
"""
            )
            collection = TraktorCollection(path)
            changed = collection.mark_generated_stem(Path("/Users/user/Music/DJ/song.flac"))
            info = collection.entry_element(Path("/Users/user/Music/DJ/song.flac")).find("INFO")
            self.assertTrue(changed)
            self.assertEqual(info.get("FLAGS"), "76")
            self.assertTrue(collection.has_generated_stem(Path("/Users/user/Music/DJ/song.flac")))


if __name__ == "__main__":
    unittest.main()
