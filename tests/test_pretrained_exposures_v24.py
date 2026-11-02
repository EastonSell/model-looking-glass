"""Reconcile the public explicit-year run with its 48 reserved rows."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v23 as v23
from looking_glass import pretrained_exposures_v24 as v24

ROOT = Path(v1.__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def git_bytes(commit, path):
    return subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


class ExposureV24Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predecessor = v23.inventory_payload()
        if hashlib.sha256(canonical(cls.predecessor)).hexdigest() != v23.INVENTORY_SHA256:
            raise AssertionError('predecessor payload differs from its pin')

    def test_public_result_manifest_journal_and_receipt(self):
        raw = [git_bytes(v24.EVIDENCE_COMMIT, path) for path in v24.EVIDENCE_PATHS]
        self.assertEqual([hashlib.sha256(b).hexdigest() for b in raw], list(v24.EVIDENCE_SHA256S))
        case_bytes, manifest_bytes, journal_bytes, result_bytes, receipt_bytes = raw
        case_doc, manifest, result, receipt = map(json.loads,
                                                   (case_bytes, manifest_bytes, result_bytes, receipt_bytes))
        self.assertEqual(manifest['cases_sha256'], v24.CASE_SHA256)
        cases = [(c['case_id'], c['text'], tuple(c['input_ids']),
                  tuple(tuple(c[k]) for k in v24.CANDIDATE_GROUP_KEYS)) for c in case_doc['cases']]
        saved_cases = [(c['case_id'], c['text'], tuple(c['input_ids']),
                        tuple(tuple(c[k]) for k in v24.CANDIDATE_GROUP_KEYS)) for c in manifest['cases']]
        self.assertEqual(cases, list(v24.EXPLICIT_YEAR_DIRECTION_CASES))
        self.assertEqual(saved_cases, cases)
        self.assertEqual((len(cases), manifest['planned_forwards']), (48, 336))
        self.assertEqual((result['status'], result['submitted_forwards'], result['returned_forwards'],
                          result['model_call_returns_observed']), ('complete', 336, 336, 336))
        self.assertEqual(result['manifest_sha256'], v24.EVIDENCE_SHA256S[1])
        self.assertEqual((result['checkpoint']['model_id'], result['checkpoint']['model_revision']),
                         (v24.TARGET.model_id, v24.TARGET.model_revision))
        self.assertEqual((receipt['process_exit_code'], receipt['interrupted'], receipt['failure_file_present']),
                         (0, False, False))
        for name, digest in [('manifest.json', v24.EVIDENCE_SHA256S[1]),
                             ('attempt.jsonl', v24.EVIDENCE_SHA256S[2]),
                             ('result.json', v24.EVIDENCE_SHA256S[3])]:
            self.assertEqual(receipt['run_sha256'][name], digest)
        self.assertEqual(receipt['source_revision'], v24.SOURCE_COMMIT)
        expected_ids = {case[0] for case in cases}
        expected_actions = {'clean', 'noop', 'self', 'year_final', 'year_at_zero', 'rotated_final', 'clean_after'}
        expected_pairs = {(case_id, action) for case_id in expected_ids for action in expected_actions}
        journal = [json.loads(line) for line in journal_bytes.splitlines()]
        states, recorded_raw = {}, {}
        for event in journal:
            kind = event.get('event')
            if kind not in ('start', 'raw_recorded', 'complete'):
                continue
            pair = (event['case_id'], event['action'])
            self.assertIn(pair, expected_pairs)
            stage = states.get(pair, 0)
            self.assertEqual(kind, ('start', 'raw_recorded', 'complete')[stage])
            if kind == 'raw_recorded':
                recorded_raw[pair] = event['raw']
            if kind == 'complete':
                self.assertEqual(event['raw'], recorded_raw[pair])
            states[pair] = stage + 1
        self.assertEqual(set(states), expected_pairs)
        self.assertEqual(Counter(states.values()), {3: 336})
        result_records = {(r['case_id'], r['action']): r for r in result['records']}
        self.assertEqual((len(result['records']), set(result_records)), (336, expected_pairs))
        self.assertEqual({pair: r['raw'] for pair, r in result_records.items()}, recorded_raw)
        for case_id in sorted(expected_ids):
            path = v24.RAW_DIRECTORY + case_id + '.f32'
            case_raw = git_bytes(v24.EVIDENCE_COMMIT, path)
            self.assertEqual(hashlib.sha256(case_raw).hexdigest(), receipt['run_sha256'][case_id + '.f32'])
            slices = []
            for action in expected_actions:
                for item in recorded_raw[(case_id, action)].values():
                    start, length = item['offset'], item['bytes']
                    self.assertEqual(hashlib.sha256(case_raw[start:start + length]).hexdigest(), item['sha256'])
                    slices.append((start, start + length))
            slices.sort()
            self.assertEqual(slices[0][0], 0)
            self.assertEqual(slices[-1][1], len(case_raw))
            self.assertTrue(all(a[1] == b[0] for a, b in zip(slices, slices[1:])))

    def test_only_48_year_statuses_change(self):
        with mock.patch.object(v23, 'inventory_payload', return_value=self.predecessor):
            new = v24.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v23.INVENTORY_VERSION, 'sha256': v23.INVENTORY_SHA256})
        self.assertEqual(new['update'], {'observed_explicit_year_direction_rows': 48})
        old_by_id = {r['record_id']: r for r in self.predecessor['records']}
        new_by_id = {r['record_id']: r for r in new['records']}
        old_ids = {'planned-' + case_id for case_id, _, _, _ in v24.EXPLICIT_YEAR_DIRECTION_CASES}
        new_ids = {'observed-' + case_id for case_id, _, _, _ in v24.EXPLICIT_YEAR_DIRECTION_CASES}
        self.assertEqual((set(old_by_id) - set(new_by_id), set(new_by_id) - set(old_by_id)), (old_ids, new_ids))
        for key in set(old_by_id) - old_ids:
            self.assertEqual(old_by_id[key], new_by_id[key])
        for case_id, text, row, groups in v24.EXPLICIT_YEAR_DIRECTION_CASES:
            before, after = old_by_id['planned-' + case_id], new_by_id['observed-' + case_id]
            self.assertEqual((before['category'], after['category']), ('planned', 'observed-saved'))
            self.assertEqual((before['public_text'], after['public_text'], tuple(before['token_ids']),
                              tuple(after['token_ids'])), (text, text, row, row))
            self.assertEqual(after['candidate_token_groups'],
                             {name: list(group) for name, group in zip(v24.CANDIDATE_GROUP_KEYS, groups)})
            self.assertEqual((after['references'], after['reference_commit']),
                             (list(v24.EVIDENCE_PATHS), v24.EVIDENCE_COMMIT))
        self.assertEqual(len(new['records']), len(self.predecessor['records']))
        self.assertEqual(new['additional_observations'], self.predecessor['additional_observations'])
        self.assertEqual(v24._prefixes(new['records']), v24._prefixes(self.predecessor['records']))
        self.assertEqual(len(v24._prefixes(new['records'])), 3888)
        self.assertEqual(hashlib.sha256(canonical(new)).hexdigest(), v24.INVENTORY_SHA256)

    def test_changed_reservation_or_predecessor_fails_closed(self):
        changed = dict(self.predecessor)
        changed['records'] = [dict(r) for r in self.predecessor['records']]
        case_id = v24.EXPLICIT_YEAR_DIRECTION_CASES[0][0]
        target = next(r for r in changed['records'] if r['record_id'] == 'planned-' + case_id)
        target['token_ids'] = [1, 2, 3, 4]
        digest = hashlib.sha256(canonical(changed)).hexdigest()
        with mock.patch.object(v23, 'inventory_payload', return_value=changed), \
                mock.patch.object(v23, 'INVENTORY_SHA256', digest), \
                mock.patch.object(v24, 'PREDECESSOR_SHA256', digest):
            with self.assertRaises(ValueError):
                v24.inventory_payload()
        with mock.patch.object(v23, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v23, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v24.inventory_payload()


if __name__ == '__main__':
    unittest.main()
