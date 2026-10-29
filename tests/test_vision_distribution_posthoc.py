"""Analytic and resource checks for the saved ViT distribution reader."""
import io
import hashlib
import lzma
import math
from pathlib import Path
import shutil
import struct
import tarfile
import tempfile
import unittest

from looking_glass import vision_distribution_posthoc as subject


class DistributionMetricTests(unittest.TestCase):
    def test_comparison_api_is_available(self):
        self.assertTrue(hasattr(subject, 'compare_distribution'))

    def test_animal_margin_can_change_without_a_change_in_animal_mass(self):
        clean = [math.log(x) for x in (0.5, 0.25, 0.15, 0.10)]
        patched = [math.log(x) for x in (0.25, 0.5, 0.15, 0.10)]
        result = subject.compare_distribution(clean, patched, cat_ids=(1,), dog_ids=(0,), side='cat')
        self.assertAlmostEqual(result['total_variation'], 0.25)
        self.assertAlmostEqual(result['animal_mass_change'], 0.0)
        self.assertAlmostEqual(result['offgroup_mass_change'], 0.0)
        self.assertAlmostEqual(result['offgroup_absolute_change_share'], 0.0)
        self.assertAlmostEqual(result['annotation_oriented_margin_change'], -math.log(4.0))
        self.assertTrue(result['top1_changed'])

    def test_offgroup_change_does_not_by_itself_change_the_group_log_odds(self):
        clean = [math.log(x) for x in (0.5, 0.25, 0.15, 0.10)]
        patched = [math.log(x) for x in (0.4, 0.2, 0.3, 0.10)]
        result = subject.compare_distribution(clean, patched, cat_ids=(1,), dog_ids=(0,), side='dog')
        self.assertAlmostEqual(result['total_variation'], 0.15)
        self.assertAlmostEqual(result['animal_mass_change'], -0.15)
        self.assertAlmostEqual(result['offgroup_mass_change'], 0.15)
        self.assertAlmostEqual(result['offgroup_absolute_change_share'], 0.5)
        self.assertAlmostEqual(result['annotation_oriented_margin_change'], 0.0)

    def test_zero_probability_displacement_has_no_cosine_or_share(self):
        p = [0.5, 0.25, 0.15, 0.10]
        result = subject.compare_distribution([math.log(x) for x in p], [math.log(x) for x in p],
                                              cat_ids=(1,), dog_ids=(0,), side='cat')
        self.assertEqual(result['total_variation'], 0.0)
        self.assertIsNone(result['offgroup_absolute_change_share'])
        self.assertIsNone(subject.displacement_cosine(p, p, p))

    def test_opposite_probability_displacements_have_negative_alignment(self):
        p = [0.3, 0.3, 0.2, 0.2]
        a = [0.4, 0.2, 0.2, 0.2]
        b = [0.2, 0.4, 0.2, 0.2]
        self.assertAlmostEqual(subject.displacement_cosine(p, a, b), -1.0)

    def test_group_margin_is_stable_under_a_large_common_logit_offset(self):
        result = subject.compare_distribution([1e20, 1e20, 1e20], [0.0, 0.0, 0.0],
                                              cat_ids=(2,), dog_ids=(0, 1), side='cat')
        self.assertAlmostEqual(result['annotation_oriented_margin_change'], 0.0)
        self.assertAlmostEqual(result['total_variation'], 0.0)

    def test_margin_change_keeps_small_component_after_large_group_difference(self):
        result = subject.compare_distribution([1e20, 1e20, 0.0], [1e20, 0.0, 0.0],
                                              cat_ids=(2,), dog_ids=(0, 1), side='cat')
        self.assertAlmostEqual(result['annotation_oriented_margin_change'], -math.log(2.0))

    def test_top1_is_selected_from_logits_before_softmax_rounding(self):
        logits = [0.0, 2.0**-149, 0.0]
        result = subject.compare_distribution(logits, logits,
                                              cat_ids=(0,), dog_ids=(1,), side='cat')
        self.assertEqual(result['clean_top1_id'], 1)
        self.assertEqual(result['patched_top1_id'], 1)

    def test_cosine_handles_nonzero_subnormal_squared_norms(self):
        self.assertAlmostEqual(subject.displacement_cosine(
            [1.0, 1e-200], [1.0, 2e-200], [1.0, 3e-200]), 1.0)

    def test_nonfinite_logits_and_invalid_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            subject.compare_distribution([0.0, float('nan')], [0.0, 1.0],
                                         cat_ids=(0,), dog_ids=(1,), side='cat')
        with self.assertRaises(ValueError):
            subject.compare_distribution([0.0, 1.0], [0.0, 1.0],
                                         cat_ids=(0,), dog_ids=(0,), side='cat')

    def test_summary_keeps_ineligible_sides_out_of_evaluable_counts(self):
        self.assertTrue(hasattr(subject, 'summarize_rows'))
        row = lambda pair, eligible, inside, outside, random: {
            'pair_id': pair, 'side': 'cat', 'eligible': eligible,
            'inside_vs_outside_probability_displacement_cosine': 0.3,
            'inside_vs_random_probability_displacement_cosine': 0.4,
            'conditions': {
                'matched_inside': {'total_variation': inside, 'annotation_oriented_margin_change': 0.2,
                                   'offgroup_absolute_change_share': 0.6},
                'matched_outside': {'total_variation': outside, 'annotation_oriented_margin_change': 0.1,
                                    'offgroup_absolute_change_share': 0.4},
                'matched_random': {'total_variation': random, 'annotation_oriented_margin_change': -0.3,
                                   'offgroup_absolute_change_share': 0.7},
            },
        }
        result = subject.summarize_rows([row('pair-04', True, 0.3, 0.2, 0.1),
                                         row('pair-08', False, 0.1, 0.4, 0.6)])
        self.assertEqual(result['evaluable']['sides'], 1)
        self.assertEqual(result['baseline_ineligible']['sides'], 1)
        self.assertEqual(result['evaluable']['inside_tv_exceeds_outside'], 1)
        self.assertEqual(result['baseline_ineligible']['inside_tv_exceeds_outside'], 0)
        self.assertEqual(result['evaluable']['pair_count'], 1)
        self.assertAlmostEqual(result['evaluable']['inside_vs_random_cosine']['median'], 0.4)
        self.assertEqual(result['evaluable']['inside_vs_random_cosine']['undefined_count'], 0)

    def test_saved_dose_must_match_the_planned_value_and_receiver_ratio(self):
        dose = {'status': 'matchable', 'q': 10.0, 'tolerance': 0.001}
        condition = {'actual_dose': 10.0001, 'receiver_norm': 5.0,
                     'q_over_receiver_norm': 2.0}
        actual = subject._validated_dose(dose, condition)
        self.assertEqual(actual['planned_q'], 10.0)
        self.assertEqual(actual['actual_dose'], 10.0001)
        for invalid in ({**condition, 'actual_dose': 10.1},
                        {**condition, 'q_over_receiver_norm': 3.0},
                        {**condition, 'receiver_norm': float('nan')}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                subject._validated_dose(dose, invalid)
        with self.assertRaises(ValueError):
            subject._validated_dose({**dose, 'status': 'unmatchable'}, condition)


class BoundedXZTests(unittest.TestCase):
    def test_archive_scanner_api_is_available(self):
        self.assertTrue(hasattr(subject, 'scan_verified_tar'))

    def test_incremental_decoder_respects_small_reads_and_exact_output(self):
        source = b'abcdefghij' * 10000
        reader = subject.BoundedXZReader(io.BytesIO(lzma.compress(source)),
                                         decompressed_limit=len(source), memory_limit=32 * 1024 * 1024)
        blocks = []
        while True:
            block = reader.read(137)
            if not block:
                break
            blocks.append(block)
        self.assertEqual(b''.join(blocks), source)
        self.assertEqual(reader.decompressed_bytes, len(source))

    def test_decoder_rejects_expansion_over_limit_and_truncation(self):
        compressed = lzma.compress(b'x' * 10000)
        limited = subject.BoundedXZReader(io.BytesIO(compressed), decompressed_limit=9999,
                                          memory_limit=32 * 1024 * 1024)
        with self.assertRaises(ValueError):
            while limited.read(1024):
                pass
        truncated = subject.BoundedXZReader(io.BytesIO(compressed[:-8]), decompressed_limit=20000,
                                            memory_limit=32 * 1024 * 1024)
        with self.assertRaises(ValueError):
            while truncated.read(1024):
                pass

    def test_tar_reader_verifies_exact_members_before_yielding(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            for name, data in [('a.json', b'{}'), ('b.bin', b'abc')]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        expected = {'a.json': (2, hashlib.sha256(b'{}').hexdigest()),
                    'b.bin': (3, hashlib.sha256(b'abc').hexdigest())}
        reader = subject.BoundedXZReader(io.BytesIO(lzma.compress(buffer.getvalue())),
                                         decompressed_limit=100000, memory_limit=32 * 1024 * 1024)
        self.assertEqual(dict(subject.scan_verified_tar(reader, expected)),
                         {'a.json': b'{}', 'b.bin': b'abc'})

    def test_tar_reader_rejects_hash_mismatch_and_unlisted_member(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            info = tarfile.TarInfo('data')
            info.size = 3
            archive.addfile(info, io.BytesIO(b'abc'))
        packed = lzma.compress(buffer.getvalue())
        for expected in ({'data': (3, '0' * 64)}, {}):
            reader = subject.BoundedXZReader(io.BytesIO(packed), decompressed_limit=100000,
                                             memory_limit=32 * 1024 * 1024)
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                list(subject.scan_verified_tar(reader, expected))


class SavedEvidenceTests(unittest.TestCase):
    def test_binary32_decode_preserves_subnormals_and_rejects_nonfinite(self):
        data = struct.pack('<1000I', 1, 0x80000001, *([0] * 998))
        decoded = subject._decode_f32_logits(data)
        self.assertEqual(decoded[0], 2.0**-149)
        self.assertEqual(decoded[1], -(2.0**-149))
        for bits in (0x7f800000, 0x7fc00000):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                subject._decode_f32_logits(struct.pack('<1000I', bits, *([0] * 999)))

    def test_complete_published_archive_replay_reproduces_specific_failure_mode(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / 'results' / 'vision-coco-vit-v1-20261029'
        parts = sorted(evidence.glob('raw-evidence.tar.xz.part-*'))
        self.assertEqual(len(parts), 16)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / 'original.tar.xz'
            with archive_path.open('xb') as out:
                for part in parts:
                    with part.open('rb') as stream:
                        shutil.copyfileobj(stream, out, length=65536)
            report = subject.analyze_saved_evidence(
                archive_path, evidence / 'evidence-index.json',
                root / 'experiments' / 'vision_coco_spatial_v1.json')
        self.assertEqual(report['verification']['indexed_members'], 709)
        self.assertEqual(report['verification']['heldout_forward_records'], 228)
        self.assertEqual(report['verification']['selected_forward_records'], 64)
        self.assertEqual(report['summary']['evaluable']['pair_count'], 4)
        self.assertEqual(report['summary']['evaluable']['sides'], 8)
        self.assertEqual(report['summary']['baseline_ineligible']['sides'], 8)
        row = next(r for r in report['rows'] if (r['pair_id'], r['side']) == ('pair-04', 'cat'))
        vectors = row['probabilities']
        self.assertEqual(set(vectors), {'clean', 'matched_inside', 'matched_outside', 'matched_random'})
        for vector in vectors.values():
            self.assertEqual(len(vector), 1000)
            self.assertAlmostEqual(math.fsum(vector), 1.0)
        self.assertAlmostEqual(subject.displacement_cosine(vectors['clean'], vectors['matched_inside'],
                                                          vectors['matched_random']),
                               row['inside_vs_random_probability_displacement_cosine'])
        self.assertAlmostEqual(row['inside_vs_random_probability_displacement_cosine'],
                               0.9377, places=4)
        self.assertAlmostEqual(row['conditions']['matched_inside']['annotation_oriented_margin_change'],
                               0.25398507725, places=8)
        self.assertAlmostEqual(row['conditions']['matched_random']['annotation_oriented_margin_change'],
                               -0.51304668927, places=8)


if __name__ == '__main__':
    unittest.main()
