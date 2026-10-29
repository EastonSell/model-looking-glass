"""Synthetic transitions against the real cooperative gate and lock."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'tools' / 'agreement_split_gate_transition.py'
RELEASE = (ROOT if (ROOT / 'tools' / 'agreement_split_gate_allocation.py').is_file()
           else ROOT.parent.parent / 'gate-allocation-release')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), 'transition implementation is required')
        self.transition = module(SOURCE, 'test_transition')
        self.helper = module(RELEASE / 'tools' / 'agreement_split_gate_allocation.py', 'test_gate_helper')
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name).resolve(); self.root = base / 'gate'; self.root.mkdir()
        (self.root / 'gate.lock').write_bytes(b'')
        self.sources = base / 'sources'; (self.sources / 'looking_glass').mkdir(parents=True)
        for name in self.helper.SOURCE_HASHES:
            shutil.copyfile(RELEASE / 'looking_glass' / (name + '.py'), self.sources / 'looking_glass' / (name + '.py'))
        rs = self.root.lstat(); ls = (self.root / 'gate.lock').lstat()
        path_hash = hashlib.sha256(os.path.normcase(os.path.abspath(str(self.root))).encode()).hexdigest()
        patcher = mock.patch.multiple(self.helper, GATE_ROOT_SHA256=path_hash,
            GATE_ROOT_IDENTITY=(rs.st_dev, rs.st_ino), GATE_LOCK_IDENTITY=(ls.st_dev, ls.st_ino))
        patcher.start(); self.addCleanup(patcher.stop); self.addCleanup(self.clear_modules)
        self.helper.operate('allocate', 'natural', self.root, self.sources, reader=self.reader)
        self.clear_modules()
        payload = {'schema': 'mlg-natural-heldout-public-commitment-v1',
            'prediction_freeze_sha256': self.helper.FREEZES['natural'],
            'exposure_inventory_sha256': 'a' * 64, 'heldout_commitment': 'b' * 64,
            'wrapper_sha256': 'c' * 64, 'deviation_sha256': 'd' * 64,
            'operational_binding_sha256': 'e' * 64}
        self.public = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()

    @staticmethod
    def clear_modules():
        for name in tuple(sys.modules):
            if name == 'looking_glass' or name.startswith('looking_glass.'):
                del sys.modules[name]

    @staticmethod
    def reader(name):
        return (RELEASE / 'looking_glass' / (name + '.py')).read_bytes()

    def test_public_commitment_then_unallocated_peer_is_explicitly_unchecked(self):
        value = self.transition.operate('commit', 'natural', self.root, self.sources,
            helper=self.helper, source_reader=self.reader, commitment_bytes=self.public,
            publication_check=lambda raw: raw == self.public)
        self.assertEqual(value['status'], 'commitment_announced')
        self.clear_modules()
        value = self.transition.operate('forward', 'natural', self.root, self.sources,
            helper=self.helper, source_reader=self.reader)
        self.assertEqual(value['comparison'], 'other_not_allocated_unchecked')
        marker = json.loads((self.root / 'natural-forward.json').read_bytes())
        self.assertEqual(marker['comparison'], 'other_not_allocated_unchecked')

    def test_unobserved_commitment_cannot_create_marker(self):
        with self.assertRaises(Exception):
            self.transition.operate('commit', 'natural', self.root, self.sources,
                helper=self.helper, source_reader=self.reader, commitment_bytes=self.public,
                publication_check=lambda raw: False)
        self.assertFalse((self.root / 'natural-commitment.json').exists())

    def test_mutated_helper_cannot_execute_before_hash_verification(self):
        trusted = (RELEASE / 'tools' / 'agreement_split_gate_allocation.py').read_bytes()
        marker = self.root.parent / 'executed'
        changed = self.root.parent / 'helper.py'
        changed.write_bytes(trusted + ("\nopen(%r,'w').write('bad')\n" % str(marker)).encode())
        with self.assertRaises(ValueError):
            self.transition.load_helper(changed, reader=lambda url: trusted)
        self.assertFalse(marker.exists())

    def test_commitment_publication_requires_immutable_exact_public_bytes(self):
        url = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'a' * 40 + '/results/x.json'
        source = self.root.parent / 'commitment.json'; source.write_bytes(self.public)
        digest = hashlib.sha256(self.public).hexdigest()
        self.assertEqual(self.transition.read_published(source, url, digest, reader=lambda u: self.public), self.public)
        with self.assertRaises(ValueError):
            self.transition.read_published(source, url.replace('a' * 40, 'main'), digest, reader=lambda u: self.public)
        with self.assertRaises(ValueError):
            self.transition.read_published(source, url, digest, reader=lambda u: b'changed')

    def test_cli_commit_uses_verified_publication_and_real_gate(self):
        commitment = self.root.parent / 'commitment.json'; commitment.write_bytes(self.public)
        url = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'a' * 40 + '/results/x.json'
        argv = ['transition', 'commit', '--track', 'natural', '--root', str(self.root),
                '--source-root', str(self.sources), '--helper-source', str(RELEASE / 'tools' / 'agreement_split_gate_allocation.py'),
                '--commitment-file', str(commitment), '--commitment-url', url,
                '--commitment-sha256', hashlib.sha256(self.public).hexdigest()]
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(sys, 'stdout', output), \
             mock.patch.object(self.transition, 'load_helper', return_value=self.helper), \
             mock.patch.object(self.transition, '_remote', return_value=self.public), \
             mock.patch.object(self.helper, '_remote', side_effect=self.reader):
            self.assertEqual(self.transition.main(), 0)
        self.assertEqual(output.getvalue().strip(), 'commitment_announced:natural')
        self.assertTrue((self.root / 'natural-commitment.json').is_file())

    def test_committed_peer_cannot_be_cleared_by_a_published_lookalike_receipt(self):
        self.transition.operate('commit', 'natural', self.root, self.sources, helper=self.helper,
            source_reader=self.reader, commitment_bytes=self.public, publication_check=lambda raw: raw == self.public)
        self.clear_modules()
        self.helper.operate('allocate', 'injection', self.root, self.sources, reader=self.reader)
        self.clear_modules()
        injected = {'schema': 'mlg-injection-heldout-public-commitment-v1',
            'prediction_freeze_sha256': self.helper.FREEZES['injection'],
            'exposure_inventory_sha256': '1' * 64, 'heldout_commitment': '2' * 64,
            'custody_source_sha256': '3' * 64, 'operational_binding_sha256': '4' * 64}
        injected_raw = json.dumps(injected, sort_keys=True, separators=(',', ':')).encode()
        self.transition.operate('commit', 'injection', self.root, self.sources, helper=self.helper,
            source_reader=self.reader, commitment_bytes=injected_raw, publication_check=lambda raw: raw == injected_raw)
        self.clear_modules()
        with self.assertRaises(Exception):
            self.transition.operate('forward', 'natural', self.root, self.sources,
                helper=self.helper, source_reader=self.reader)
        self.assertFalse((self.root / 'natural-forward.json').exists())
        self.clear_modules()
        receipt = {'schema': 'mlg-number-disjoint-split-check-v1', 'status': 'clear',
            'natural_prediction_freeze_sha256': self.helper.FREEZES['natural'],
            'natural_heldout_commitment': 'b' * 64, 'natural_exposure_inventory_sha256': 'a' * 64,
            'natural_commitment_file_sha256': hashlib.sha256(self.public).hexdigest(),
            'injection_prediction_freeze_sha256': self.helper.FREEZES['injection'],
            'injection_heldout_commitment': '2' * 64, 'injection_exposure_inventory_sha256': '1' * 64,
            'injection_commitment_file_sha256': hashlib.sha256(injected_raw).hexdigest(),
            'comparator_source_sha256': 'f' * 64, 'natural_case_count': 16, 'injection_case_count': 16}
        raw = json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode()
        with self.assertRaises(Exception):
            self.transition.operate('forward', 'natural', self.root, self.sources,
                helper=self.helper, source_reader=self.reader, receipt_bytes=raw, receipt_check=lambda data: data == raw)
        self.assertFalse((self.root / 'natural-forward.json').exists())


if __name__ == '__main__':
    unittest.main()
