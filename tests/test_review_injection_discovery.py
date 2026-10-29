"""Fabricated CPU-only checks for the independent public archive auditor."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from looking_glass.pretrained_number_commitments import seal_site
from looking_glass.pretrained_number_requests import create_family_request_plan
from looking_glass.pretrained_number_supervisor import run_discovery
from tests.test_number_injection_coordinate_oracle import Model


WORK = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('independent_injection_audit', WORK / 'review_injection_discovery.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        sentinel = torch.tensor([1], dtype=torch.int32).view(torch.float32)
        cls.old_flush = int((sentinel * 1.0).view(torch.int32).item()) == 0
        if not torch.set_flush_denormal(False):
            raise unittest.SkipTest('CPU denormal mode unavailable')
        cls.temp = tempfile.TemporaryDirectory()
        cls.attempt = Path(cls.temp.name).resolve() / 'attempt'
        source = 'a' * 40
        nonce = bytes(range(32, 64))  # Fabricated, unrelated to the real study.
        site = seal_site(nonce, source)
        plan = create_family_request_plan(bytes(range(32)), source)
        cls.plan_path = Path(cls.temp.name) / 'plan.json'
        cls.plan_path.write_bytes(plan)
        cls.source = source
        cls.result = run_discovery(model=Model(), attempt_dir=cls.attempt, plan_bytes=plan,
            source_commit=source, plan_sha256=hashlib.sha256(plan).hexdigest(), runtime_sha256='b' * 64,
            site_nonce=nonce, site_record_bytes=site.record_bytes, site_commitment=site.commitment_sha256,
            logical_limit_bytes=48 * 1024**2, disk_limit_bytes=128 * 1024**2)
        cls.public = cls.attempt / 'public'

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_flush_denormal(cls.old_flush)
        torch.set_num_threads(cls.old_threads)

    def test_complete_archive_recomputes_results(self):
        result = audit.audit(self.public, self.plan_path, self.source,
                             self.result.prediction_sha256, self.result.archive_sha256)
        self.assertEqual((result['status'], result['records'], len(result['cases'])), ('passed', 128, 8))

    def test_rehashed_prediction_cannot_forge_aggregate_votes(self):
        path = self.public / 'prediction.json'
        old = path.read_bytes()
        try:
            value = json.loads(old)
            value['votes']['1'] += 1
            path.write_bytes(audit.canonical(value))
            with self.assertRaisesRegex(ValueError, 'aggregate'):
                audit.audit(self.public, self.plan_path, self.source)
        finally:
            path.write_bytes(old)

    def test_rehashed_observation_cannot_forge_patch_donor(self):
        path = self.public / 'observation-00-forward_1.json'
        prediction = self.public / 'prediction.json'
        old_record, old_prediction = path.read_bytes(), prediction.read_bytes()
        try:
            record, selected = json.loads(old_record), json.loads(old_prediction)
            record['action']['donor']['request_id'] = 'f' * 32
            selected['cases'][0]['observations'][3]['action'] = record['action']
            path.write_bytes(audit.canonical(record))
            prediction.write_bytes(audit.canonical(selected))
            with self.assertRaisesRegex(ValueError, 'action or donor'):
                audit.audit(self.public, self.plan_path, self.source)
        finally:
            path.write_bytes(old_record)
            prediction.write_bytes(old_prediction)

    def test_rehashed_unselected_logit_changes_expected_archive_digest(self):
        path = self.public / 'observation-00-forward_1.json'
        prediction = self.public / 'prediction.json'
        old_record, old_prediction = path.read_bytes(), prediction.read_bytes()
        try:
            record, selected = json.loads(old_record), json.loads(old_prediction)
            raw = bytearray(base64.b64decode(record['payload']['logits']))
            raw[0] ^= 1  # Change one finite logit unrelated to selected margin IDs.
            record['payload']['logits'] = base64.b64encode(raw).decode()
            selected['cases'][0]['observations'][3]['logits_sha256'] = hashlib.sha256(raw).hexdigest()
            path.write_bytes(audit.canonical(record))
            prediction.write_bytes(audit.canonical(selected))
            with self.assertRaisesRegex(ValueError, 'archive digest'):
                audit.audit(self.public, self.plan_path, self.source, expected_archive=self.result.archive_sha256)
        finally:
            path.write_bytes(old_record)
            prediction.write_bytes(old_prediction)

    def test_public_plan_request_ids_are_independently_derived(self):
        old = self.plan_path.read_bytes()
        try:
            plan = json.loads(old)
            plan['requests'][0]['request_id'] = 'f' * 32
            self.plan_path.write_bytes(audit.canonical(plan))
            with self.assertRaisesRegex(ValueError, 'plan'):
                audit.audit(self.public, self.plan_path, self.source)
        finally:
            self.plan_path.write_bytes(old)

    def test_prediction_rejects_boolean_for_numeric_case_index(self):
        path = self.public / 'prediction.json'
        old = path.read_bytes()
        try:
            prediction = json.loads(old)
            prediction['cases'][0]['index'] = False
            path.write_bytes(audit.canonical(prediction))
            with self.assertRaisesRegex(ValueError, 'case'):
                audit.audit(self.public, self.plan_path, self.source)
        finally:
            path.write_bytes(old)

    def test_cli_refuses_unbound_archive_without_external_digests(self):
        command = [sys.executable, '-B', str(WORK / 'review_injection_discovery.py'), str(self.public),
                   '--request-plan', str(self.plan_path), '--source-commit', self.source,
                   '--output', str(Path(self.temp.name) / 'audit.json')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--plan-sha256', result.stderr)
        self.assertIn('--prediction-sha256', result.stderr)
        self.assertIn('--archive-sha256', result.stderr)


if __name__ == '__main__':
    unittest.main()
