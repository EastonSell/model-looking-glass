import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import natural_transfer_phase as phase
from natural_transfer_check_executor import prepared


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def gitblob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


class PreflightTests(unittest.TestCase):
    def test_exact_public_reservations_and_tampering(self):
        cases = json.loads(Path(os.environ['NATURAL_CASES']).read_bytes())['cases']
        inventory = json.loads(Path(os.environ['NATURAL_INVENTORY']).read_bytes())
        self.assertEqual(phase.verify_reservations(inventory, cases)['own_cases'], 68)
        damaged = copy.deepcopy(inventory)
        name = 'planned-' + cases[0]['case_id']
        row = next(r for r in damaged['records'] if r['record_id'] == name)
        row['token_ids'][0] += 1
        with self.assertRaises(ValueError):
            phase.verify_reservations(damaged, cases)
        damaged = copy.deepcopy(inventory)
        damaged['records'].append(copy.deepcopy(damaged['records'][0]))
        with self.assertRaises(ValueError):
            phase.verify_reservations(damaged, cases)
        damaged = copy.deepcopy(inventory)
        row = next(r for r in damaged['records'] if r['record_id'] == name)
        row['scored_prefix_token_ids'] = cases[0]['token_ids']
        with self.assertRaises(ValueError):
            phase.verify_reservations(damaged, cases)
        damaged = copy.deepcopy(inventory)
        modified_cases = copy.deepcopy(cases)
        modified_cases[0]['token_ids'][0] = 1
        row = next(r for r in damaged['records'] if r['record_id'] == name)
        row['token_ids'][0] = True
        with self.assertRaises(ValueError):
            phase.verify_reservations(damaged, modified_cases)

    def test_phase_module_import_is_model_free(self):
        code = "import sys; import natural_transfer_phase; assert 'torch' not in sys.modules"
        proc = subprocess.run([sys.executable, '-B', '-c', code], cwd=str(Path(phase.__file__).parent),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_evaluation_recomputes_fixture_and_rejects_rehashed_freeze(self):
        fixture = Path(os.environ['NATURAL_FAKE_FIXTURE'])
        meta = json.loads((fixture/'fixture.json').read_bytes())
        mapping, prior, _ = prepared()
        freeze_raw = (fixture/'freeze.json').read_bytes()
        with tempfile.TemporaryDirectory(dir=os.environ['NATURAL_TEST_TEMP']) as name:
            tmp = Path(name)
            receipt = {'schema':'natural-invariant-transfer-direct-receipt-v2', 'phase':'discovery',
                       'accepted_complete_phase':True, 'exit_code':0, 'interrupted':False,
                       'prefix_empty_after':True, 'stage_empty_after':True, 'stderr':{'bytes':0},
                       'pre_run_sha256':'1'*64, 'prior_receipt_sha256':'1'*64,
                       'prior_audit_sha256':'2'*64, 'prior_audit_observation_sha256':'3'*64,
                       'prior_audit_observation_public_sha256':'4'*64,
                       'latest_exposure_sha256':'1'*64,
                       'run_files':{'started.json':{}, 'events.jsonl':{'sha256':meta['phases']['discovery']['events_sha256']},
                                    'terminal.json':{'sha256':meta['phases']['discovery']['terminal_sha256']}}}
            receipt_raw = json.dumps(receipt, sort_keys=True).encode()
            (tmp/'receipt.json').write_bytes(receipt_raw)
            observation = {'schema':'natural-invariant-transfer-freeze-observation-v2',
                           'discovery_receipt_sha256':digest(receipt_raw),
                           'discovery_events_sha256':meta['phases']['discovery']['events_sha256'],
                           'discovery_terminal_sha256':meta['phases']['discovery']['terminal_sha256'],
                           'freeze_sha256':digest(freeze_raw), 'public_commit':'a'*40,
                           'public_blob':gitblob(freeze_raw), 'latest_exposure_sha256':'1'*64}
            obs_raw = json.dumps(observation, sort_keys=True).encode()
            (tmp/'obs.json').write_bytes(obs_raw)
            private = {'freeze_observation':tmp/'obs.json', 'discovery_receipt':tmp/'receipt.json',
                       'mapping':fixture/'toy-mapping.json', 'discovery_dir':fixture/'discovery',
                       'discovery_freeze':fixture/'freeze.json'}
            pre = {'prior': {'receipt_sha256':'1'*64, 'audit_sha256':'2'*64,
                             'audit_observation_sha256':'3'*64, 'audit_observation_public_sha256':'4'*64,
                             'events_sha256':meta['prior_events_sha256']},
                   'latest_exposure_sha256':'1'*64}
            with mock.patch.object(phase, 'MAPPING_SHA', meta['mapping_sha256']):
                result = phase.verify_evaluation_freeze(private=private, pre=pre, pre_sha='1'*64,
                         prior=prior, observed_sha=digest(obs_raw), width=8, vocabulary=1500)
                self.assertEqual(result['freeze_sha256'], meta['freeze_sha256'])
                bad_receipt = dict(receipt, exit_code=1, interrupted=True)
                bad_receipt_raw = json.dumps(bad_receipt, sort_keys=True).encode()
                (tmp/'receipt.json').write_bytes(bad_receipt_raw)
                observation['discovery_receipt_sha256'] = digest(bad_receipt_raw)
                obs_raw = json.dumps(observation, sort_keys=True).encode()
                (tmp/'obs.json').write_bytes(obs_raw)
                with self.assertRaises(ValueError):
                    phase.verify_evaluation_freeze(private=private, pre=pre, pre_sha='1'*64,
                        prior=prior, observed_sha=digest(obs_raw), width=8, vocabulary=1500)
                (tmp/'receipt.json').write_bytes(receipt_raw)
                observation['discovery_receipt_sha256'] = digest(receipt_raw)
                bad = json.loads(freeze_raw)
                bad['selected'] = 'Q4' if bad['selected'] != 'Q4' else 'Q3'
                bad_raw = (json.dumps(bad, indent=2, sort_keys=True) + '\n').encode()
                (tmp/'forged.json').write_bytes(bad_raw)
                observation['freeze_sha256'] = digest(bad_raw)
                observation['public_blob'] = gitblob(bad_raw)
                obs_raw = json.dumps(observation, sort_keys=True).encode()
                (tmp/'obs.json').write_bytes(obs_raw)
                private['discovery_freeze'] = tmp/'forged.json'
                with self.assertRaises(ValueError):
                    phase.verify_evaluation_freeze(private=private, pre=pre, pre_sha='1'*64,
                        prior=prior, observed_sha=digest(obs_raw), width=8, vocabulary=1500)


if __name__ == '__main__':
    unittest.main()
