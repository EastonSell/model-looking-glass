"""No-model v38 preservation, scored-reference and exclusion regressions."""
import copy
import hashlib
import importlib.util
import json
import os
import sys
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

module_path = os.environ.get('DISTRACTOR_V38_MODULE')
if module_path:
    spec = importlib.util.spec_from_file_location('looking_glass.pretrained_exposures_v38', module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
else:
    from looking_glass import pretrained_exposures_v38 as mod

case_path = Path(os.environ.get('DISTRACTOR_CASES_JSON',
    str(Path(__file__).resolve().parents[1] / 'experiments/public_distractor_repair_cases_v1.json')))

class ReservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = case_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != mod.CASE_SHA256:
            raise AssertionError('fixed case file differs')
        cls.cases = json.loads(raw)['cases']
        cls.prior = mod.prior.inventory_payload()
        with patch.object(mod.prior, 'inventory_payload', return_value=cls.prior):
            cls.payload = mod.inventory_payload()
        cls.digest = hashlib.sha256(mod._canonical(cls.payload)).hexdigest()
        if cls.digest != mod.INVENTORY_SHA256:
            raise AssertionError('canonical digest differs')

    def test_inventory_runs_without_importing_model_libraries(self):
        # Other tests legitimately import torch. Check this contract in a fresh
        # interpreter, blocking attempted imports as well as retained modules.
        code = r"""
import importlib.abc
import importlib.util
import pathlib
import sys

root, module_path = sys.argv[1:]
sys.path.insert(0, root)
blocked = {"torch", "transformers", "tensorflow", "jax"}
class RejectModelImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError("model library import attempted: " + fullname)
sys.meta_path.insert(0, RejectModelImport())
assert not blocked.intersection(sys.modules)
spec = importlib.util.spec_from_file_location(
    "looking_glass.pretrained_exposures_v38", module_path)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
assert pathlib.Path(subject.__file__).resolve() == pathlib.Path(module_path).resolve()
assert subject.inventory_digest() == subject.INVENTORY_SHA256
assert not blocked.intersection(sys.modules)
print("NO_MODEL_IMPORT_VERIFIED")
"""
        result = subprocess.run(
            [sys.executable, '-I', '-c', code,
             str(Path(__file__).resolve().parents[1]), str(Path(mod.__file__).resolve())],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                 'CUDA_VISIBLE_DEVICES': ''},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'NO_MODEL_IMPORT_VERIFIED')

    def test_preserves_predecessor_and_exact_rows(self):
        self.assertEqual(self.payload['records'][:len(self.prior['records'])], self.prior['records'])
        records = self.payload['records'][len(self.prior['records']):]
        self.assertEqual(len(records), 48)
        self.assertEqual(len(self.cases), 16)
        for case in self.cases:
            ident, row, prefix = case['case_id'], case['token_ids'], case['scored_prefix_token_ids']
            own = [r for r in records if r['record_id'] == 'planned-' + ident]
            self.assertEqual(len(own), 1)
            self.assertEqual(own[0]['category'], 'planned')
            self.assertEqual(own[0]['token_ids'], row)
            self.assertEqual(own[0]['scored_prefix_token_ids'], prefix)
            self.assertEqual(own[0]['scored_candidate_token_ids'], [318, 389])
            self.assertNotIn('candidate_token_ids', own[0])
            for token in (318, 389):
                refs = [r for r in records if r['record_id'] == 'reference-' + ident + '-' + str(token)]
                self.assertEqual(len(refs), 1)
                self.assertEqual(refs[0]['category'], 'reference-only')
                self.assertEqual(refs[0]['token_ids'], prefix + [token])
                self.assertIn('no separate model forward is asserted', refs[0]['provenance'])

    def test_public_api_rejects_reference_and_extensions(self):
        alternatives = [r['token_ids'] for r in self.payload['records']
                        if r['record_id'].startswith('reference-distractor-repair-')]
        self.assertEqual(len(alternatives), 32)
        with patch.object(mod, '_checked_inventory', return_value=(self.payload, mod.INVENTORY_SHA256)):
            for alt in alternatives:
                for rejected in (alt, alt + [42], alt[:-1]):
                    discovery = [[49900, i] for i in range(8)]
                    held = [[49901, i] for i in range(8)]
                    discovery[0] = rejected
                    with self.assertRaises(ValueError):
                        mod.check_known_splits(discovery, held,
                            expected_inventory_sha256=mod.INVENTORY_SHA256,
                            expected_model_identity=mod.v2.TARGET, cases_per_split=8)

    def test_builder_rejects_foreign_prefixes_in_both_directions(self):
        alt = self.cases[0]['scored_prefix_token_ids'] + [318]
        for row in (alt, alt[:-1], alt + [42]):
            old = copy.deepcopy(self.prior)
            old['records'].append({'record_id':'foreign-test', 'category':'planned', 'token_ids':row})
            with self.assertRaises(ValueError):
                mod._build_payload(old)

    def test_prefix_count_and_fixed_scope(self):
        self.assertEqual(len(mod._prefixes(self.payload['records'])), mod.FORBIDDEN_PREFIX_COUNT)
        self.assertEqual(self.payload['update'], {'planned_distractor_repair_inputs':16,
                          'reference_only_scored_prefix_alternatives':32})

if __name__ == '__main__':
    unittest.main()
