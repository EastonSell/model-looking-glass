"""Failure checks for the public saved-logit subset rebuilder."""
import io
from pathlib import Path
import unittest

import rebuild_subset


class RebuildTests(unittest.TestCase):
    def test_rejects_short_output_write(self):
        class ShortWriter:
            def write(self, block):
                return len(block) // 2

        with self.assertRaises(OSError):
            rebuild_subset.checked_write(ShortWriter(), b'abcdef')
        output = io.BytesIO()
        rebuild_subset.checked_write(output, b'abcdef')
        self.assertEqual(output.getvalue(), b'abcdef')

    def test_refuses_to_overwrite_existing_output(self):
        # Validation of the published parts precedes the exclusive create.
        directory = Path(__file__).resolve().parent
        with self.assertRaises(FileExistsError):
            rebuild_subset.rebuild(directory, directory / 'parts.json')


if __name__ == '__main__':
    unittest.main()
