"""Verify completed greater-than and planned pairs/dose inputs without model work."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v6 as v6
from looking_glass import pretrained_exposures_v14 as v14

try:
    from looking_glass import pretrained_exposures_v15 as v15
except ImportError:
    v15 = None

ROOT = Path(v1.__file__).resolve().parents[1]


def committed_bytes(path):
    return subprocess.run(['git', 'show', v15.CASES_COMMIT + ':' + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


def digest_payload(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class ExposureV15Tests(unittest.TestCase):
    def setUp(self):
        if v15 is None:
            self.fail('the v15 public pairs/dose reservation is not implemented')

    def test_prior_history_is_preserved_and_public_cases_are_recorded(self):
        old, new = v14.inventory_payload(), v15.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v14.INVENTORY_VERSION,
                                              'sha256': v14.INVENTORY_SHA256})
        self.assertEqual(new['update'], {'observed_public_greater_than_rows': 48,
                                         'planned_public_pairs_dose_rows': 32})
        self.assertEqual(new['records'][:279], old['records'])
        self.assertEqual(new['additional_observations'], old['additional_observations'])
        self.assertEqual(len(new['records']), 359)
        observed, added = new['records'][279:327], new['records'][327:]
        self.assertEqual([(row['record_id'], row['public_text'], tuple(row['token_ids']))
                          for row in observed],
                         [('observed-' + case_id, text, ids) for case_id, text, ids in v15.GREATER_THAN_CASES])
        self.assertTrue(all(row['category'] == 'observed-saved'
                            and row['reference_commit'] == v15.GREATER_EVIDENCE_COMMIT
                            and row['references'] == list(v15.GREATER_EVIDENCE_PATHS) for row in observed))
        self.assertEqual([tuple(row['token_ids']) for row in added], list(v15.PAIRS_DOSE_ROWS))
        self.assertEqual([row['target_token_id'] for row in added],
                         [group[5] for group in v15.PAIRS_DOSE_GROUPS for _ in range(4)])
        self.assertEqual([row['record_id'] for row in added],
                         [f'planned-copy-pairs-dose-{index // 4 + 1:02d}-{index % 4}'
                          for index in range(32)])
        self.assertTrue(all(row['category'] == 'planned' and row['reference_commit'] == v15.CASES_COMMIT
                            and row['references'] == list(v15.CASES_PATHS) for row in added))
        self.assertTrue(all(row['category'] == 'planned' for row in new['records'][170:183]))
        old_prefixes = {tuple(row['token_ids'][:n]) for row in old['records']
                        for n in range(1, len(row['token_ids']) + 1)}
        with mock.patch.object(v15, 'inventory_payload', return_value=new):
            self.assertTrue(old_prefixes <= v15._forbidden_rows())
            self.assertEqual(len(old_prefixes), 2287)
            self.assertEqual(len(v15._forbidden_rows()), 2959)
            self.assertEqual(v15.FORBIDDEN_PREFIX_COUNT, 2959)
            self.assertEqual(hashlib.sha256(v15.canonical_inventory_bytes()).hexdigest(), v15.inventory_digest())

    def test_completed_public_greater_than_cases_match_the_saved_attempt(self):
        commits = (v15.GREATER_SOURCE_COMMIT,) + (v15.GREATER_EVIDENCE_COMMIT,) * 5
        raw = [subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT,
                              capture_output=True, check=True).stdout
               for commit, path in zip(commits, v15.GREATER_EVIDENCE_PATHS)]
        expected_hashes = (v15.GREATER_CASES_SHA256, v15.GREATER_MANIFEST_SHA256,
                           v15.GREATER_RESULT_SHA256, v15.GREATER_JOURNAL_SHA256,
                           v15.GREATER_LOGITS_SHA256, v15.GREATER_RECEIPT_SHA256)
        self.assertEqual([hashlib.sha256(data).hexdigest() for data in raw], list(expected_hashes))
        cases, manifest, result = map(json.loads, raw[:3])
        events = [json.loads(line) for line in raw[3].splitlines()]
        logits, receipt = raw[4], json.loads(raw[5])
        self.assertEqual(manifest['cases'], cases['cases'])
        self.assertEqual(manifest['cases_sha256'], v15.GREATER_CASES_SHA256)
        self.assertEqual(manifest['source']['revision'], v15.GREATER_SOURCE_COMMIT)
        expected = [(case['case_id'], case['text'], tuple(case['input_ids'])) for case in cases['cases']]
        self.assertEqual(expected, list(v15.GREATER_THAN_CASES))
        self.assertEqual((len(expected), len({case[0] for case in expected}),
                          len({case[2] for case in expected})), (48, 48, 48))
        self.assertEqual((manifest['expected_forwards'], manifest['raw_tensor_bytes']), (48, len(logits)))
        self.assertEqual((result['status'], result['submitted_forwards'],
                          result['model_call_returns_observed'], result['recorded_forwards']),
                         ('complete', 48, 48, 48))
        self.assertEqual(result['manifest_sha256'], v15.GREATER_MANIFEST_SHA256)
        self.assertEqual((result['checkpoint']['model_id'], result['checkpoint']['model_revision']),
                         (v15.TARGET.model_id, v15.TARGET.model_revision))
        self.assertEqual((receipt['process_exit_code'], receipt['failure_file_present']), (0, False))
        for name, digest in (('manifest.json', v15.GREATER_MANIFEST_SHA256),
                             ('result.json', v15.GREATER_RESULT_SHA256),
                             ('attempt.jsonl', v15.GREATER_JOURNAL_SHA256),
                             ('last_logits.f32', v15.GREATER_LOGITS_SHA256)):
            self.assertEqual(receipt['run_sha256'][name], digest)
        self.assertEqual(Counter(event['event'] for event in events),
                         {'runtime_ready': 1, 'start': 48, 'raw_recorded': 48, 'complete': 48})
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
            self.assertEqual((segment['offset'], segment['bytes']), (next_offset, 201028))
            data = logits[next_offset:next_offset + segment['bytes']]
            self.assertEqual(hashlib.sha256(data).hexdigest(), segment['sha256'])
            next_offset += segment['bytes']
        self.assertEqual(next_offset, len(logits))

    def test_public_cases_and_prior_protocol_bind_exact_inputs_and_targets(self):
        protocol, raw = [committed_bytes(path) for path in v15.CASES_PATHS]
        self.assertEqual(hashlib.sha256(protocol).hexdigest(), v15.PROTOCOL_SHA256)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), v15.CASES_SHA256)
        subprocess.run(['git', 'merge-base', '--is-ancestor', v15.PROTOCOL_COMMIT,
                        v15.CASES_COMMIT], cwd=ROOT, check=True)
        record = json.loads(raw)
        self.assertEqual((record['known_exposure_version'], record['known_exposure_sha256']),
                         ('distilgpt2-development-exposures-v13', v14.PREDECESSOR_SHA256))
        self.assertEqual((record['model_id'], record['model_revision']),
                         (v15.TARGET.model_id, v15.TARGET.model_revision))
        self.assertEqual((record['protocol_revision'], record['protocol_sha256']),
                         (v15.PROTOCOL_COMMIT, v15.PROTOCOL_SHA256))
        self.assertEqual(record['expected_forwards'], 576)
        self.assertEqual([tuple(group['token_ids']) for group in record['groups']], list(v15.PAIRS_DOSE_GROUPS))
        self.assertEqual([tuple(item) for item in record['permutations']], list(v6.PUBLIC_COPY_PERMUTATIONS))
        cases = record['cases']
        self.assertEqual([(case['case_id'], tuple(case['token_ids']), case['target_id'],
                           case['group'], case['variant'], case['text']) for case in cases],
                         [(f'copy-pairs-{i // 4 + 1:02d}-{i % 4}', row,
                           v15.PAIRS_DOSE_GROUPS[i // 4][5], i // 4 + 1, i % 4,
                           v15.PAIRS_DOSE_TEXTS[i]) for i, row in enumerate(v15.PAIRS_DOSE_ROWS)])
        complete = [case['token_ids'] + [case['target_id']] for case in cases]
        self.assertEqual(hashlib.sha256(json.dumps(complete, separators=(',', ':')).encode()).hexdigest(),
                         v15.COMPLETE_ROWS_SHA256)
        public_recipient, = record['additional_public_exposure']
        self.assertEqual((public_recipient['sha256'], public_recipient['input_rows_checked'],
                          public_recipient['collisions_found']), (v14.RECIPIENT_CASES_SHA256, 64, False))

    def test_predecessor_and_new_rows_reject_both_prefix_directions(self):
        candidate = list(v15.PAIRS_DOSE_ROWS[0])
        target = v15.PAIRS_DOSE_GROUPS[0][5]
        original = v14.inventory_payload()
        for old_row in (candidate[:-1], candidate + [target], candidate + [target, 123],
                        candidate[:2] + [50256]):
            changed = copy.deepcopy(original)
            changed['records'][0]['token_ids'] = old_row
            changed_digest = digest_payload(changed)
            with self.subTest(old_row=old_row), \
                    mock.patch.object(v14, 'inventory_payload', return_value=changed), \
                    mock.patch.object(v14, 'INVENTORY_SHA256', changed_digest), \
                    mock.patch.object(v15, 'PREDECESSOR_SHA256', changed_digest):
                with self.assertRaisesRegex(ValueError, 'prefix'):
                    v15.inventory_payload()
        rows = list(v15.PAIRS_DOSE_ROWS)
        rows[1] = rows[0][:-1]
        with mock.patch.object(v15, 'PAIRS_DOSE_ROWS', tuple(rows)):
            with self.assertRaises(ValueError):
                v15.inventory_payload()
        with mock.patch.object(v14, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v15.inventory_payload()

    def test_known_checks_exclude_planned_rows_without_authorizing_a_seal(self):
        discovery = [[1000 + i, 2000 + i] for i in range(16)]
        heldout = [[3000 + i, 4000 + i] for i in range(16)]
        payload = v15.inventory_payload()
        options = {'expected_inventory_sha256': v15.INVENTORY_SHA256, 'expected_model_identity': v15.TARGET}
        with mock.patch.object(v15, 'inventory_payload', return_value=payload):
            check = v15.check_known_splits(discovery, heldout, cases_per_split=16, **options)
            for candidate in (list(v15.PAIRS_DOSE_ROWS[0][:-1]), list(v15.PAIRS_DOSE_ROWS[0]) + [123]):
                altered = copy.deepcopy(heldout)
                altered[0] = candidate
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        v15.check_known_splits(discovery, altered, cases_per_split=16, **options)
                    with self.assertRaises(ValueError):
                        v15.check_exposed_discovery(discovery, altered, **options)
            with self.assertRaises(v1.ExposureInventoryIncomplete):
                v15.require_preseal_readiness(check)
            disclosure = v15.prepare_preseal_disclosure(
                check, protocol_sha256='a' * 64, acknowledge_prior_exposure_uncertainty=True)
            self.assertFalse(disclosure.execution_authorized)


if __name__ == '__main__':
    unittest.main()
