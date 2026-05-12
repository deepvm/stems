from pathlib import Path
import unittest

from traktor_stem_batch.scanner import title_artist_from_filename


class ScannerTests(unittest.TestCase):
    def test_title_artist_from_common_filename(self):
        title, artist = title_artist_from_filename(Path("01 - Artist Name - Track Name.flac"))
        self.assertEqual(title, "Track Name")
        self.assertEqual(artist, "Artist Name")

    def test_title_without_artist(self):
        title, artist = title_artist_from_filename(Path("Track Name.flac"))
        self.assertEqual(title, "Track Name")
        self.assertIsNone(artist)


if __name__ == "__main__":
    unittest.main()
