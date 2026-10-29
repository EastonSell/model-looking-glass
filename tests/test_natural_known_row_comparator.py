"""Fabricated-only checks for the protected-row comparator."""
import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SOURCE = Path(__file__).resolve().parent.parent / 'tools' / 'natural_known_row_comparator.py'


class ComparisonTests(unittest.TestCase):
    def module(self):
        self.assertTrue(SOURCE.is_file(), 'comparator implementation is required')
        spec = importlib.util.spec_from_file_location('tested_known_row_comparator', SOURCE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_equality_and_each_complete_prefix_direction_collide(self):
        mod = self.module()
        private = [[11, 22, 33], [50, 60]]
        self.assertEqual(mod.compare_rows(private, [[11, 22, 33]]), 'collision')
        self.assertEqual(mod.compare_rows(private, [[11, 22]]), 'collision')
        self.assertEqual(mod.compare_rows(private, [[11, 22, 33, 44]]), 'collision')
        self.assertEqual(mod.compare_rows(private, [[11, 22, 99], [50, 61]]), 'clear')

    def test_empty_or_non_integer_rows_are_rejected(self):
        mod = self.module()
        for private, public in (([], [[1]]), ([[1]], [[]]), ([[True]], [[1]]), ([[1]], [[-1]])):
            with self.subTest(private=private, public=public), self.assertRaises(ValueError):
                mod.compare_rows(private, public)

    def test_evidence_is_bound_to_the_sealed_record_and_audited_supervisor(self):
        mod = self.module()
        canon = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        digest = lambda raw: hashlib.sha256(raw).hexdigest()
        nonce = bytes(range(32))
        record = {'schema': 'mlg-number-heldout-v1', 'prediction_freeze_sha256': 'a' * 64,
                  'exposure_inventory_sha256': 'b' * 64, 'source_commits': {'natural': 'c' * 40},
                  'families': [{'cases': [{'input_ids': [1, 2, 3]}, {'input_ids': [8, 9]}]}]}
        record_raw = canon(record)
        commitment = {'schema': 'mlg-natural-heldout-public-commitment-v1',
                      'prediction_freeze_sha256': 'a' * 64, 'exposure_inventory_sha256': 'b' * 64,
                      'heldout_commitment': digest(b'MLG-number-v1-heldout-commit\0' + nonce + b'\0' + record_raw)}
        commitment_raw = canon(commitment)
        proposal_raw = canon({'cases': [{'input_ids': [1, 2, 4]}, {'input_ids': [7]}]})
        supervisor_raw = canon({'audit': {'preparation': {'nonce.bin': digest(nonce),
            'heldout-record.json': digest(record_raw), 'commitment.json': digest(commitment_raw)}}})
        observation_raw = canon({'status': 'completed', 'exit_code': 0,
            'supervisor_record_sha256': digest(supervisor_raw)})
        pins = {'freeze': 'a' * 64, 'inventory': 'b' * 64, 'source_commit': 'c' * 40,
                'proposal': digest(proposal_raw), 'commitment': digest(commitment_raw),
                'supervisor': digest(supervisor_raw), 'observation': digest(observation_raw),
                'private_count': 2, 'public_count': 2}
        evidence = dict(nonce=nonce, record_raw=record_raw, commitment_raw=commitment_raw,
                        proposal_raw=proposal_raw, supervisor_raw=supervisor_raw,
                        observation_raw=observation_raw, pins=pins)
        result = mod.compare_evidence(**evidence)
        self.assertEqual(result['status'], 'clear')
        self.assertEqual(result['natural_heldout_commitment'], commitment['heldout_commitment'])
        self.assertNotIn('record_sha256', result)
        evidence['nonce'] = b'x' * 32
        with self.assertRaises(ValueError):
            mod.compare_evidence(**evidence)

    def test_cli_writes_only_bound_result_and_rejects_modified_private_bytes(self):
        canon = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        digest = lambda raw: hashlib.sha256(raw).hexdigest()
        nonce = bytes(range(32))
        record = {'schema': 'mlg-number-heldout-v1', 'prediction_freeze_sha256': 'a' * 64,
                  'exposure_inventory_sha256': 'b' * 64, 'source_commits': {'natural': 'c' * 40},
                  'families': [{'cases': [{'input_ids': [1, 2]}, {'input_ids': [8, 9]}]}]}
        record_raw = canon(record)
        commitment_raw = canon({'schema': 'mlg-natural-heldout-public-commitment-v1',
            'prediction_freeze_sha256': 'a' * 64, 'exposure_inventory_sha256': 'b' * 64,
            'heldout_commitment': digest(b'MLG-number-v1-heldout-commit\0' + nonce + b'\0' + record_raw)})
        proposal_raw = canon({'cases': [{'input_ids': [1, 3]}, {'input_ids': [7]}]})
        supervisor_raw = canon({'audit': {'preparation': {'nonce.bin': digest(nonce),
            'heldout-record.json': digest(record_raw), 'commitment.json': digest(commitment_raw)}}})
        observation_raw = canon({'status': 'completed', 'exit_code': 0,
            'supervisor_record_sha256': digest(supervisor_raw)})
        pins = {'freeze': 'a' * 64, 'inventory': 'b' * 64, 'source_commit': 'c' * 40,
                'proposal': digest(proposal_raw), 'commitment': digest(commitment_raw),
                'supervisor': digest(supervisor_raw), 'observation': digest(observation_raw),
                'private_count': 2, 'public_count': 2, 'comparator_source': digest(SOURCE.read_bytes())}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); phase = root / 'phase'; phase.mkdir()
            supervisor = root / 'supervisor'; supervisor.mkdir()
            for name, raw in [('nonce.bin', nonce), ('heldout-record.json', record_raw),
                              ('commitment.json', commitment_raw)]:
                (phase / name).write_bytes(raw)
            (supervisor / 'supervisor.json').write_bytes(supervisor_raw)
            (supervisor / 'supervisor-observation.json').write_bytes(observation_raw)
            proposal = root / 'proposal.json'; proposal.write_bytes(proposal_raw)
            pin_file = root / 'pins.json'; pin_file.write_bytes(canon(pins))
            receipt = root / 'receipt.json'
            cmd = [sys.executable, '-B', str(SOURCE), '--phase', str(phase), '--supervisor', str(supervisor),
                   '--proposal', str(proposal), '--pins', str(pin_file), '--pins-sha256', digest(pin_file.read_bytes()),
                   '--receipt', str(receipt)]
            first = subprocess.run(cmd, capture_output=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            result = json.loads(receipt.read_bytes())
            self.assertEqual(result['status'], 'clear')
            self.assertEqual(result['comparator_source_sha256'], pins['comparator_source'])
            self.assertNotIn('record_sha256', result)
            (phase / 'nonce.bin').write_bytes(b'x' * 32)
            receipt.unlink()
            second = subprocess.run(cmd, capture_output=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertFalse(receipt.exists())

    def test_short_receipt_write_is_an_error_and_preserves_partial(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'receipt.json'
            original = Path.open
            class ShortWriter:
                def __enter__(self):
                    self.stream = original(path, 'xb')
                    return self
                def __exit__(self, *args):
                    self.stream.close()
                def write(self, data):
                    return self.stream.write(data[:1])
                def flush(self):
                    self.stream.flush()
                def fileno(self):
                    return self.stream.fileno()
            with mock.patch.object(Path, 'open', return_value=ShortWriter()):
                with self.assertRaises(OSError):
                    mod._write_new(path, b'abcdef')
            self.assertEqual(path.read_bytes(), b'a')


if __name__ == '__main__':
    unittest.main()
