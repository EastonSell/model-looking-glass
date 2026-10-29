"""Fabricated preparation evidence only; no real model or private split."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import unittest
from unittest import mock

from tests import test_injection_split_custody as fixtures
from looking_glass import injection_split_custody as custody
from looking_glass import pretrained_number_commitments as commitments

try:
    audit = importlib.import_module('looking_glass.injection_preparation_audit')
except ModuleNotFoundError as error:
    if error.name != 'looking_glass.injection_preparation_audit':
        raise
    audit = None


class PreparationAuditTests(unittest.TestCase):
    setUp = fixtures.InjectionCustodyTests.setUp
    prepare = fixtures.InjectionCustodyTests.prepare
    v2_public = fixtures.InjectionCustodyTests.v2_public

    def checked(self, phase, **overrides):
        config = self.root / 'preparation-config.json'
        if not config.exists():
            config.write_bytes(fixtures.canonical({'schema': 'fabricated-private-config-v1', 'value': 'fixture'}))
        kwargs = dict(public=self.inputs, config_path=config,
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            ledger_path=self.root / 'fake-ledger',
            site_nonce=fixtures.SITE_NONCE, site_record_bytes=self.site.record_bytes,
            tokenizer=fixtures.FakeTokenizer(), publication_check=lambda name, raw: True,
            source_check=lambda binding: True, prediction_check=lambda raw, sha: True,
            observe_inventory=lambda: self.inventory_sha,
            claim_check=lambda path, phase_name, output: self.claim.claimed)
        kwargs.update(overrides)
        return audit.audit_preparation(phase, **kwargs)

    def test_complete_fabricated_phase_binds_six_private_files_without_disclosure(self):
        phase = self.prepare().phase_root
        result = self.checked(phase)
        expected = {'claim.json', 'nonce.bin', 'heldout-record.json', 'inventory.json',
                    'commitment.json', 'preparation-complete.json'}
        self.assertEqual(result['status'], 'evidence_complete')
        self.assertEqual(result['inventory'], {'retained_tensor_bytes': 0})
        self.assertEqual(set(result['preparation']), expected)
        for name in expected:
            self.assertEqual(result['preparation'][name], hashlib.sha256((phase / name).read_bytes()).hexdigest())
        self.assertNotIn('alpha', json.dumps(result))
        self.assertNotIn(fixtures.SITE_NONCE.hex(), json.dumps(result))

    def test_mutated_or_extra_files_do_not_pass_audit(self):
        phase = self.prepare().phase_root
        record = phase / 'heldout-record.json'
        original = record.read_bytes()
        record.write_bytes(original + b' ')
        with self.assertRaises(ValueError):
            self.checked(phase)
        record.write_bytes(original)
        (phase / 'unexpected.bin').write_bytes(b'partial')
        with self.assertRaises(ValueError):
            self.checked(phase)

    def test_wrong_claim_config_or_current_inventory_is_rejected(self):
        phase = self.prepare().phase_root
        for overrides in ({'claim_check': lambda path, name, out: False},
                          {'config_sha256': '0' * 64},
                          {'observe_inventory': lambda: '0' * 64}):
            with self.subTest(overrides=tuple(overrides)), self.assertRaises(ValueError):
                self.checked(phase, **overrides)

    def test_hardlinked_private_file_is_rejected(self):
        phase = self.prepare().phase_root
        path = phase / 'nonce.bin'
        link = self.root / 'nonce-copy.bin'
        os.link(path, link)
        with self.assertRaises(ValueError):
            self.checked(phase)

    def test_alignment_is_independently_rechecked_after_preparation(self):
        phase = self.prepare().phase_root
        class WrongTokenizer(fixtures.FakeTokenizer):
            def encode(self, text, *, add_special_tokens):
                ids = super().encode(text, add_special_tokens=add_special_tokens)
                return ids + [123] if text.endswith(' is') else ids
        with self.assertRaises(ValueError):
            self.checked(phase, tokenizer=WrongTokenizer())

    def test_self_consistent_split_reusing_public_entropy_is_rejected(self):
        phase = self.prepare().phase_root
        nonce = bytes.fromhex('33' * 32)
        snapshot = commitments.load_inventory_snapshot(self.inventory, self.inventory_sha)
        allocation = commitments.sample_families(self.freeze, nonce, inventory_snapshot=snapshot,
            current_inventory_sha256=self.inventory_sha, site_nonce=fixtures.SITE_NONCE,
            site_record_bytes=self.site.record_bytes, site_commitment=self.site.commitment_sha256)
        cases = custody.tokenize_assignment(allocation, fixtures.FakeTokenizer())
        sealed = commitments.seal_heldout(self.freeze, nonce, cases, allocation=allocation,
            inventory_snapshot=snapshot, current_inventory_sha256=self.inventory_sha,
            site_nonce=fixtures.SITE_NONCE, site_record_bytes=self.site.record_bytes,
            site_commitment=self.site.commitment_sha256)
        (phase / 'nonce.bin').write_bytes(nonce)
        (phase / 'heldout-record.json').write_bytes(sealed.record_bytes)
        public = json.loads((phase / 'commitment.json').read_bytes())
        public['heldout_commitment'] = sealed.commitment_sha256
        public_bytes = fixtures.canonical(public)
        (phase / 'commitment.json').write_bytes(public_bytes)
        complete = json.loads((phase / 'preparation-complete.json').read_bytes())
        complete['commitment_file_sha256'] = hashlib.sha256(public_bytes).hexdigest()
        (phase / 'preparation-complete.json').write_bytes(fixtures.canonical(complete))
        with self.assertRaises(ValueError):
            self.checked(phase)

    def test_study_audit_rejects_an_alternate_freeze_before_generic_audit(self):
        with mock.patch.object(audit, 'audit_preparation') as generic:
            with self.assertRaises(ValueError):
                audit.audit_study_preparation(self.root, public=self.v2_public())
            generic.assert_not_called()

    def test_study_audit_requires_v2_even_with_matching_discovery_pins(self):
        pins = {'_STUDY_FREEZE': self.freeze.sha256, '_STUDY_SOURCE': self.source,
                '_STUDY_RUNTIME': fixtures.digest(self.runtime), '_STUDY_PREDICTION': fixtures.digest(self.prediction),
                '_STUDY_PLAN': fixtures.digest(self.plan), '_STUDY_SITE_COMMITMENT': self.site.commitment_sha256}
        with mock.patch.multiple(custody, **pins), mock.patch.object(audit, 'audit_preparation', return_value='verified') as generic:
            with self.assertRaises(ValueError):
                audit.audit_study_preparation(self.root, public=self.inputs)
            generic.assert_not_called()
            self.assertEqual(audit.audit_study_preparation(self.root, public=self.v2_public()), 'verified')


if __name__ == '__main__':
    unittest.main()
