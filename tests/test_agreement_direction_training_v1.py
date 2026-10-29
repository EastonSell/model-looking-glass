"""Model-free construction tests for the prospective agreement directions."""
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import io
import json

from looking_glass import agreement_direction_training_v1 as direction
from looking_glass.agreement_direction_training_v1 import decode_f32_bits, derive_directions, pair_differences


class DirectionTests(unittest.TestCase):
    def test_binary32_decode_preserves_subnormal_zero_and_rejects_nonfinite(self):
        self.assertEqual(decode_f32_bits(1), 2.0**-149)
        self.assertEqual(decode_f32_bits(0x80800000), -(2.0**-126))
        self.assertEqual(math.copysign(1.0, decode_f32_bits(0x80000000)), -1.0)
        for bits in (0x7f800000, 0xff800000, 0x7fc00000):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                decode_f32_bits(bits)

    def test_oriented_pair_differences_and_three_distinct_directions(self):
        # Each pair has singular-minus-plural (2, +/-1, 0, 0), so the
        # expected first two axes are known independently of the code.
        vectors = {}
        for i in range(8):
            delta = (2.0, 1.0 if i % 2 == 0 else -1.0, 0.0, 0.0)
            vectors[f's{i}'] = delta
            vectors[f'p{i}'] = (0.0, 0.0, 0.0, 0.0)
        pairs = [(f's{i}', f'p{i}') for i in range(8)]
        ds = pair_differences(vectors, pairs)
        u, v1, v2, metadata = derive_directions(ds)
        self.assertEqual(u, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(v1, (0.0, 1.0, 0.0, 0.0))
        self.assertAlmostEqual(math.fsum(x*x for x in v2), 1.0, places=13)
        self.assertAlmostEqual(v2[0], 0.0, places=13)
        self.assertAlmostEqual(v2[1], 0.0, places=13)
        self.assertEqual(metadata['mean_difference_norm'], 2.0)

    def test_rejects_degenerate_or_nonfinite_training(self):
        zero = [(0.0, 0.0, 0.0, 0.0)] * 8
        with self.assertRaises(ValueError):
            derive_directions(zero)
        ds = [(1.0, 0.0, 0.0, 0.0)] * 8
        with self.assertRaises(ValueError):
            derive_directions(ds)  # lexical control degenerates
        ds[0] = (math.nan, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            derive_directions(ds)

    def test_rejects_missing_or_duplicate_pair_identity(self):
        with self.assertRaises(ValueError):
            pair_differences({'s': (1.0,), 'p': (0.0,)}, [('s', 'p')] * 8)
        with self.assertRaises(ValueError):
            pair_differences({'s': (1.0,)}, [('s', 'p')])

    def test_archive_is_read_from_the_hashed_snapshot(self):
        # Replace the pathname immediately after its initial read.  A later
        # pathname reopen must not affect parsing of the already hashed bytes.
        cases, events = [], []
        payload = struct.pack('<768f', *([1.0] * 768))
        payload_hash = direction.digest(payload)
        for q, lemma, singular, plural in direction.PAIRS:
            for case_id, number in ((singular, 'singular'), (plural, 'plural')):
                cases.append({'case_id': case_id, 'quartet': q, 'lemma': lemma, 'number': number})
                step = {'name': 'capture_4', 'module_path': 'transformer.h.3.attn',
                        'output_selector': [0], 'batch_index': 0, 'invocation': 0, 'token_index': 4,
                        'output_shape': [1, 5, 768], 'selected_shape': [768]}
                events.append({'event': {'kind': 'clean_capture', 'case': case_id,
                                         'plan_metadata': {'steps': [step]}},
                               'tensors': {'capture_4': {'shape': [768], 'bytes': 3072,
                                                          'dtype': 'float32-le', 'sha256': payload_hash}}})
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            lines = [{'event': {'cases': cases}}] + events
            archive.writestr('attempt/events.jsonl', '\n'.join(json.dumps(x) for x in lines))
            archive.writestr('attempt/tensors/' + payload_hash + '.f32le', payload)
        saved = buffer.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'archive.zip'
            path.write_bytes(saved)
            original_read = direction.read_archive_snapshot

            def replace_after_read(source):
                raw = original_read(source)
                if Path(source) == path:
                    Path(source).write_bytes(b'other bytes')
                return raw

            with patch.object(direction, 'ARCHIVE_SHA256', direction.digest(saved)), \
                 patch.object(direction, 'read_archive_snapshot', replace_after_read):
                vectors, hashes = direction.read_archive(path)
            self.assertEqual(len(vectors), 16)
            self.assertEqual(set(hashes.values()), {payload_hash})

    def test_oversized_archive_read_is_bounded(self):
        class Oversized:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size=-1):
                self.requested = size
                if size != 16 * 1024 * 1024 + 1:
                    raise AssertionError('archive read is not bounded')
                return b'x' * size

        stream = Oversized()
        with patch.object(Path, 'open', return_value=stream), self.assertRaises(ValueError):
            direction.read_archive('unused-oversized-archive')
        self.assertEqual(stream.requested, 16 * 1024 * 1024 + 1)

    def test_vector_write_must_pass_readback_before_manifest(self):
        vector = (1.0,) + (0.0,) * 767
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            original_read = Path.read_bytes

            def corrupt_readback(self):
                return b'corrupt' if self.suffix == '.f64le' else original_read(self)

            with patch.object(Path, 'read_bytes', corrupt_readback), self.assertRaises(ValueError):
                direction.write_vectors(target, (vector, vector, vector))
            self.assertFalse((target / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
