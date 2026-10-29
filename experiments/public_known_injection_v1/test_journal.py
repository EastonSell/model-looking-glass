import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from journal import Journal


class JournalTests(unittest.TestCase):
    def test_complete_record_contains_contiguous_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'attempt'
            writer = Journal(root, raw_cap=64, metadata_cap=10000)
            writer.start(0, 'known-injection-01', 'clean', {})
            first = writer.record('logits', b'\x00' * 8, (2,))
            second = writer.record('tap', b'\x01' * 4, (1,))
            writer.complete([first, second], [], {'valid': True})
            writer.close()
            data = (root / 'known-injection-01.f32').read_bytes()
            self.assertEqual((first['offset'], second['offset']), (0, 8))
            self.assertEqual(hashlib.sha256(data[8:12]).hexdigest(), second['sha256'])
            self.assertEqual([json.loads(s)['type'] for s in (root / 'events.jsonl').read_text().splitlines()],
                             ['start', 'tensor', 'tensor', 'raw-recorded', 'complete'])

    def test_overflow_does_not_discard_prior_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'attempt'
            writer = Journal(root, raw_cap=4, metadata_cap=10000)
            writer.start(0, 'known-injection-01', 'clean', {})
            writer.record('logits', b'\x01' * 4, (1,))
            with self.assertRaises(ValueError):
                writer.record('excess', b'\x02' * 4, (1,))
            writer.fail(ValueError('cap'))
            writer.close()
            self.assertEqual((root / 'known-injection-01.f32').read_bytes(), b'\x01' * 4)
            self.assertTrue((root / 'failure.json').exists())


if __name__ == '__main__':
    unittest.main()
