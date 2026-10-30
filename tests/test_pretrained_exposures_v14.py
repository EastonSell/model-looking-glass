"""Verify additional public injection use without model work or held-out access."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v2 as v2
from looking_glass import pretrained_exposures_v13 as v13

try:
    from looking_glass import pretrained_exposures_v14 as v14
except ImportError:
    v14 = None

ROOT = Path(v1.__file__).resolve().parents[1]
ROLES = ('clean', 'repeat', 'injected', *(f'forward_{i}' for i in range(1, 5)),
         *(f'reverse_{i}' for i in range(1, 5)), 'null', 'self', 'upstream',
         'downstream', 'verify_clean')


def committed_bytes(path):
    return subprocess.run(['git', 'show', v14.EVIDENCE_COMMIT + ':' + path], cwd=ROOT,
                          capture_output=True, check=True).stdout


def committed_many(paths):
    requests = ''.join(v14.EVIDENCE_COMMIT + ':' + path + '\n' for path in paths).encode('utf-8')
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT, input=requests,
                            capture_output=True, check=True).stdout
    values, cursor = [], 0
    for _ in paths:
        end = result.index(b'\n', cursor)
        header = result[cursor:end].split()
        if len(header) != 3 or header[1] != b'blob':
            raise ValueError('public observation Git blob is unavailable')
        size = int(header[2])
        cursor = end + 1
        values.append(result[cursor:cursor + size])
        cursor += size
        if result[cursor:cursor + 1] != b'\n':
            raise ValueError('invalid Git batch framing')
        cursor += 1
    if cursor != len(result):
        raise ValueError('unexpected trailing Git batch data')
    return values


def input_digest(token_ids):
    return hashlib.sha256(b'MLG-number-input-v1\0' + bytes([len(token_ids)]) +
                          b''.join(token.to_bytes(4, 'big') for token in token_ids)).hexdigest()


class ExposureV14Tests(unittest.TestCase):
    def setUp(self):
        if v14 is None:
            self.fail('the v14 exposure successor has not been implemented')

    def test_prior_records_are_preserved_and_completed_recipient_rows_are_added(self):
        old, new = v13.inventory_payload(), v14.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v13.INVENTORY_VERSION,
                                              'sha256': v13.INVENTORY_SHA256})
        self.assertEqual(new['update'], {'additional_public_injection_uses': 8,
                                         'observed_public_recipient_baseline_rows': 64})
        self.assertEqual(new['records'][:215], old['records'])
        self.assertEqual(len(new['records']), 279)
        self.assertEqual(Counter(row['category'] for row in new['records']), {
            'observed-saved': 232, 'observed-reported': 3, 'planned': 32,
            'reference-only': 4, 'synthetic-only': 8,
        })
        records = {row['record_id']: row for row in new['records']}
        indexes = [0, 1, 4, 5, 8, 9, 12, 13]
        expected_ids = [f'observed-agreement-discovery-{index + 1:02d}' for index in indexes]
        self.assertEqual(list(v14.INJECTION_RECORD_IDS), expected_ids)
        for index, record_id in zip(indexes, expected_ids):
            self.assertEqual(records[record_id]['category'], 'observed-saved')
            self.assertEqual(tuple(records[record_id]['token_ids']), v2.PUBLIC_DISCOVERY[index][1])
            self.assertEqual(records[record_id]['public_text'], v2.PUBLIC_DISCOVERY[index][0])
        use, = new['additional_observations']
        self.assertEqual(use['kind'], 'public-number-injection-discovery')
        self.assertEqual(use['record_ids'], expected_ids)
        self.assertEqual(use['evidence_commit'], v14.EVIDENCE_COMMIT)
        self.assertEqual(use['references'], list(v14.EVIDENCE_PATHS))
        self.assertEqual((use['plan_sha256'], use['archive_sha256']),
                         (v14.PLAN_SHA256, v14.ARCHIVE_SHA256))
        old_prefixes = {tuple(row['token_ids'][:n]) for row in old['records']
                        for n in range(1, len(row['token_ids']) + 1)}
        self.assertEqual(len(old_prefixes), 1709)
        added = new['records'][215:]
        self.assertEqual(len(added), len(v14.RECIPIENT_CASES))
        for row, (case_id, text, tokens, target) in zip(added, v14.RECIPIENT_CASES):
            self.assertEqual(row['record_id'], 'observed-' + case_id)
            self.assertEqual(row['category'], 'observed-saved')
            self.assertEqual((row['public_text'], tuple(row['token_ids']), row['target_token_id']),
                             (text, tokens, target))
            self.assertEqual(row['reference_commit'], v14.RECIPIENT_EVIDENCE_COMMIT)
            self.assertEqual(row['references'], list(v14.RECIPIENT_EVIDENCE_PATHS))
        all_prefixes = old_prefixes | {tuple(row['token_ids'][:n]) for row in added
                                       for n in range(1, len(row['token_ids']) + 1)}
        with mock.patch.object(v14, 'inventory_payload', return_value=new):
            self.assertEqual(v14._forbidden_rows(), all_prefixes)
            self.assertEqual(len(all_prefixes), 2287)
            self.assertEqual(hashlib.sha256(v14.canonical_inventory_bytes()).hexdigest(), v14.inventory_digest())

    def test_public_recipient_cases_and_completed_attempt_match_all_64_rows(self):
        raw = [subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT,
                              capture_output=True, check=True).stdout
               for commit, path in zip((v14.RECIPIENT_SOURCE_COMMIT,) +
                                       (v14.RECIPIENT_EVIDENCE_COMMIT,) * 5,
                                       v14.RECIPIENT_EVIDENCE_PATHS)]
        expected_hashes = (v14.RECIPIENT_CASES_SHA256, v14.RECIPIENT_MANIFEST_SHA256,
                           v14.RECIPIENT_RESULT_SHA256, v14.RECIPIENT_JOURNAL_SHA256,
                           v14.RECIPIENT_LOGITS_SHA256, v14.RECIPIENT_RECEIPT_SHA256)
        self.assertEqual([hashlib.sha256(data).hexdigest() for data in raw], list(expected_hashes))
        cases, manifest, result = map(json.loads, raw[:3])
        events = [json.loads(line) for line in raw[3].splitlines()]
        logits, receipt = raw[4], json.loads(raw[5])
        self.assertEqual(manifest['cases'], cases['cases'])
        self.assertEqual(manifest['cases_sha256'], v14.RECIPIENT_CASES_SHA256)
        self.assertEqual(manifest['source']['revision'], v14.RECIPIENT_SOURCE_COMMIT)
        self.assertEqual((manifest['expected_forwards'], manifest['raw_tensor_bytes']), (64, len(logits)))
        expected = [(case['case_id'], case['text'], tuple(case['input_ids']), case['recipient_id'])
                    for case in cases['cases']]
        self.assertEqual(expected, list(v14.RECIPIENT_CASES))
        self.assertEqual(len({case[0] for case in expected}), 64)
        self.assertEqual(len({case[2] for case in expected}), 64)
        self.assertEqual((result['status'], result['submitted_forwards'],
                          result['model_call_returns_observed'], result['recorded_forwards']),
                         ('complete', 64, 64, 64))
        self.assertEqual(result['manifest_sha256'], v14.RECIPIENT_MANIFEST_SHA256)
        self.assertEqual((result['checkpoint']['model_id'], result['checkpoint']['model_revision']),
                         (v14.TARGET.model_id, v14.TARGET.model_revision))
        self.assertEqual((receipt['process_exit_code'], receipt['failure_file_present']), (0, False))
        for filename, digest in (('manifest.json', v14.RECIPIENT_MANIFEST_SHA256),
                                 ('result.json', v14.RECIPIENT_RESULT_SHA256),
                                 ('attempt.jsonl', v14.RECIPIENT_JOURNAL_SHA256),
                                 ('last_logits.f32', v14.RECIPIENT_LOGITS_SHA256)):
            self.assertEqual(receipt['run_sha256'][filename], digest)
        self.assertEqual(Counter(event['event'] for event in events),
                         {'runtime_ready': 1, 'start': 64, 'raw_recorded': 64, 'complete': 64})
        expected_ids = [case[0] for case in expected]
        self.assertEqual([record['case_id'] for record in result['records']], expected_ids)
        records = {record['case_id']: record for record in result['records']}
        by_kind, positions = {}, {}
        for kind in ('start', 'raw_recorded', 'complete'):
            selected = [event for event in events if event['event'] == kind]
            by_kind[kind] = {event['case_id']: event for event in selected}
            positions[kind] = {event['case_id']: i for i, event in enumerate(events)
                               if event['event'] == kind}
            self.assertEqual(len(selected), len(by_kind[kind]))
            self.assertEqual(set(by_kind[kind]), set(expected_ids))
        next_offset = 0
        for case_id in expected_ids:
            record = records[case_id]
            self.assertLess(positions['start'][case_id], positions['raw_recorded'][case_id])
            self.assertLess(positions['raw_recorded'][case_id], positions['complete'][case_id])
            self.assertEqual(by_kind['raw_recorded'][case_id]['raw'], record['raw'])
            self.assertEqual({key: value for key, value in by_kind['complete'][case_id].items()
                              if key != 'event'}, record)
            segment = record['raw']
            self.assertEqual(segment['offset'], next_offset)
            self.assertEqual(segment['bytes'], 201028)
            data = logits[next_offset:next_offset + segment['bytes']]
            self.assertEqual(hashlib.sha256(data).hexdigest(), segment['sha256'])
            next_offset += segment['bytes']
        self.assertEqual(next_offset, len(logits))

    def test_published_plan_and_all_public_observations_bind_the_eight_known_rows(self):
        files = [committed_bytes(path) for path in v14.EVIDENCE_PATHS]
        for data, expected in zip(files, (v14.PLAN_SHA256, v14.AUDIT_SHA256,
                                          v14.PROVENANCE_SHA256, v14.OPERATOR_SHA256,
                                          v14.PREDICTION_SHA256)):
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
        plan, audit, provenance, operator, prediction = map(json.loads, files)
        self.assertEqual((plan['schema'], plan['checkpoint_revision'], plan['source_commit']),
                         ('mlg-number-injection-family-request-plan-v2',
                          v14.TARGET.model_revision, v14.STUDY_SOURCE_COMMIT))
        self.assertEqual((audit['status'], audit['records'], audit['plan_sha256'], audit['archive_sha256']),
                         ('passed', 128, v14.PLAN_SHA256, v14.ARCHIVE_SHA256))
        self.assertEqual((audit['prediction_sha256'], audit['source_commit']),
                         (v14.PREDICTION_SHA256, v14.STUDY_SOURCE_COMMIT))
        self.assertEqual((provenance['status'], provenance['public_observations'],
                          provenance['plan_sha256'], provenance['archive_sha256']),
                         ('consistent', 128, v14.PLAN_SHA256, v14.ARCHIVE_SHA256))
        self.assertEqual((operator['status'], operator['operator_observed_exit_code'],
                          operator['result_sha256']), ('passed', 0, v14.AUDIT_SHA256))
        self.assertEqual((prediction['status'], prediction['execution_device'], prediction['checkpoint_revision']),
                         ('complete', 'cpu', v14.TARGET.model_revision))
        requests = {(row['case_index'], row['role']): row for row in plan['requests']}
        self.assertEqual(len(requests), len(plan['requests']))
        self.assertEqual(set(requests), {(index, role) for index in range(8) for role in ROLES})
        self.assertEqual(len({row['request_id'] for row in plan['requests']}), 128)
        paths = [v14.PUBLIC_DIRECTORY + f'/observation-{index:02d}-{role}.json'
                 for index in range(8) for role in ROLES]
        saved = iter(committed_many(paths))
        hashes = []
        for index in range(8):
            input_ids = v2.PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2][1]
            for role in ROLES:
                request = requests[index, role]
                self.assertEqual(request['input_sha256'], input_digest(input_ids))
                raw = next(saved)
                observation = json.loads(raw)
                self.assertEqual((observation['schema'], observation['case_index'], observation['role'],
                                  observation['request_id'], observation['input_sha256']),
                                 ('mlg-number-injection-raw-observation-v1', index, role,
                                  request['request_id'], request['input_sha256']))
                hashes.append(hashlib.sha256(raw).digest())
        archive = hashlib.sha256(b'MLG-number-injection-archive-v1\0' + b''.join(hashes)).hexdigest()
        self.assertEqual(archive, v14.ARCHIVE_SHA256)

    def test_predecessor_or_agreement_row_drift_fails_closed(self):
        old = copy.deepcopy(v13.inventory_payload())
        row = next(row for row in old['records'] if row['record_id'] == v14.INJECTION_RECORD_IDS[0])
        row['token_ids'][1] += 1
        # Coherent synthetic pin reaches the downstream record-identity guard.
        synthetic_digest = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
        with mock.patch.object(v13, 'inventory_payload', return_value=old), \
                mock.patch.object(v13, "INVENTORY_SHA256", synthetic_digest), \
                mock.patch.object(v14, "PREDECESSOR_SHA256", synthetic_digest):
            with self.assertRaisesRegex(ValueError, "prior agreement discovery identity"):
                v14.inventory_payload()
        with mock.patch.object(v13, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v14.inventory_payload()

    def test_known_row_checks_still_fail_closed_and_do_not_authorize_a_seal(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        payload = v14.inventory_payload()
        options = {'expected_inventory_sha256': v14.INVENTORY_SHA256, 'expected_model_identity': v14.TARGET}
        with mock.patch.object(v14, 'inventory_payload', return_value=payload):
            check = v14.check_known_splits(discovery, heldout, cases_per_split=16, **options)
            self.assertEqual((check.inventory_version, check.candidate_count), (v14.INVENTORY_VERSION, 32))
            for candidate in (list(v2.PUBLIC_DISCOVERY[0][1][:-1]),
                              list(v2.PUBLIC_DISCOVERY[0][1]) + [123]):
                altered = copy.deepcopy(heldout)
                altered[0] = candidate
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        v14.check_known_splits(discovery, altered, cases_per_split=16, **options)
                    with self.assertRaises(ValueError):
                        v14.check_exposed_discovery(discovery, altered, **options)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v14.require_preseal_readiness(check)
            disclosure = v14.prepare_preseal_disclosure(
                check, protocol_sha256='a' * 64, acknowledge_prior_exposure_uncertainty=True)
            self.assertFalse(disclosure.execution_authorized)


if __name__ == '__main__':
    unittest.main()
