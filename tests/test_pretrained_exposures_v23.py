"""Reconcile the public screened baseline against the fixed target rows."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from looking_glass import pretrained_exposures as v1
from looking_glass import pretrained_exposures_v22 as v22
from looking_glass import pretrained_exposures_v23 as v23

ROOT = Path(v1.__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def git_bytes(commit, path):
    return subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT,
                          check=True, capture_output=True).stdout


class ExposureV23Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predecessor = v22.inventory_payload()
        if hashlib.sha256(canonical(cls.predecessor)).hexdigest() != v22.INVENTORY_SHA256:
            raise AssertionError('predecessor payload differs from its pin')

    def test_public_screen_evidence_matches_32_target_rows_and_retained_logits(self):
        raw = [git_bytes(v23.EVIDENCE_COMMIT, p) for p in v23.EVIDENCE_PATHS]
        self.assertEqual([hashlib.sha256(b).hexdigest() for b in raw], list(v23.EVIDENCE_SHA256S))
        case_data, manifest, phase, child, journal, selection, logits = raw
        case_doc, manifest, phase, child, selection = map(json.loads,
                                                          (case_data, manifest, phase, child, selection))
        self.assertEqual(manifest['case_sha256'], v23.CASE_SHA256)
        self.assertEqual((manifest['checkpoint']['model_id'], manifest['checkpoint']['model_revision']),
                         (v23.TARGET.model_id, v23.TARGET.model_revision))
        self.assertEqual(manifest['source_commit'], v23.SOURCE_COMMIT)
        cases = case_doc['cases']
        targets = [(c['case_id'], c['prompt'], tuple(c['token_ids'])) for c in cases if c['role'] == 'target']
        locality = [(c['case_id'], c['prompt'], tuple(c['token_ids'])) for c in cases if c['role'] == 'locality']
        self.assertEqual((len(targets), len(locality)), (32, 12))
        self.assertEqual(tuple(targets), v23.SCREENED_OBSERVED_CASES)
        self.assertEqual(tuple(locality), v23.SCREENED_PLANNED_LOCALITY_CASES)
        self.assertEqual((phase['phase'], phase['status'], phase['source_commit'],
                          phase['job_drained'], phase['owner_reaped']),
                         ('screen', 'complete', v23.SOURCE_COMMIT, True, True))
        self.assertEqual((phase['supervisor']['clean'], phase['supervisor']['exit_code'],
                          phase['supervisor']['timed_out']), (True, 0, False))
        self.assertEqual((phase['manifest_sha256'], phase['child_result_sha256']),
                         (v23.EVIDENCE_SHA256S[1], v23.EVIDENCE_SHA256S[3]))
        self.assertEqual(phase['child'], child)
        self.assertEqual((child['phase'], child['manifest_sha256'], child['no_model_loaded']),
                         ('screen', v23.EVIDENCE_SHA256S[1], False))
        observations = [json.loads(line) for line in journal.splitlines()]
        self.assertEqual([o['case_id'] for o in observations], [c[0] for c in targets])
        self.assertEqual(len(observations), 32)
        self.assertEqual([o['case_id'] for o in selection['selection']['rows']], [c[0] for c in targets])
        self.assertEqual((selection['case_sha256'], selection['identity']['source_commit']),
                         (v23.CASE_SHA256, v23.SOURCE_COMMIT))
        for i, (row, selected) in enumerate(zip(observations, selection['selection']['rows'])):
            self.assertEqual((row['index'], row['offset'], row['bytes']), (i, i * 201028, 201028))
            self.assertEqual(hashlib.sha256(logits[row['offset']:row['offset'] + row['bytes']]).hexdigest(),
                             row['sha256'])
            self.assertEqual((selected['logit_byte_offset'], selected['logit_bytes'], selected['logit_sha256']),
                             (row['offset'], row['bytes'], row['sha256']))
        self.assertEqual(len(logits), 32 * 201028)
        self.assertEqual(selection['artifacts']['observations.jsonl']['sha256'], v23.EVIDENCE_SHA256S[4])
        self.assertEqual(selection['artifacts']['screen.f32']['sha256'], v23.EVIDENCE_SHA256S[6])

    def test_only_32_target_statuses_change(self):
        with mock.patch.object(v22, 'inventory_payload', return_value=self.predecessor):
            new = v23.inventory_payload()
        self.assertEqual(new['predecessor'], {'version': v22.INVENTORY_VERSION, 'sha256': v22.INVENTORY_SHA256})
        self.assertEqual(new['update'], {'observed_screened_target_rows': 32})
        old_by_id = {r['record_id']: r for r in self.predecessor['records']}
        new_by_id = {r['record_id']: r for r in new['records']}
        old_ids = {'planned-' + case_id for case_id, _, _ in v23.SCREENED_OBSERVED_CASES}
        new_ids = {'observed-' + case_id for case_id, _, _ in v23.SCREENED_OBSERVED_CASES}
        self.assertEqual((set(old_by_id) - set(new_by_id), set(new_by_id) - set(old_by_id)), (old_ids, new_ids))
        for key in set(old_by_id) - old_ids:
            self.assertEqual(old_by_id[key], new_by_id[key])
        for case_id, text, row in v23.SCREENED_OBSERVED_CASES:
            before, after = old_by_id['planned-' + case_id], new_by_id['observed-' + case_id]
            self.assertEqual((before['category'], after['category']), ('planned', 'observed-saved'))
            self.assertEqual((before['public_text'], after['public_text'], tuple(before['token_ids']),
                              tuple(after['token_ids'])), (text, text, row, row))
            self.assertEqual((after['references'], after['reference_commit']),
                             (list(v23.EVIDENCE_PATHS), v23.EVIDENCE_COMMIT))
        for case_id, _, _ in v23.SCREENED_PLANNED_LOCALITY_CASES:
            self.assertEqual(new_by_id['planned-' + case_id]['category'], 'planned')
        self.assertEqual(len(new['records']), len(self.predecessor['records']))
        self.assertEqual(new['additional_observations'], self.predecessor['additional_observations'])
        self.assertEqual(v23._prefixes(new['records']), v23._prefixes(self.predecessor['records']))
        self.assertEqual(len(v23._prefixes(new['records'])), 3888)
        self.assertEqual(hashlib.sha256(canonical(new)).hexdigest(), v23.INVENTORY_SHA256)

    def test_changed_prior_or_reserved_rows_fail_closed(self):
        changed = dict(self.predecessor)
        changed['records'] = [dict(r) for r in self.predecessor['records']]
        case_id = v23.SCREENED_OBSERVED_CASES[0][0]
        target = next(r for r in changed['records'] if r['record_id'] == 'planned-' + case_id)
        target['token_ids'] = [1, 2, 3, 4]
        digest = hashlib.sha256(canonical(changed)).hexdigest()
        with mock.patch.object(v22, 'inventory_payload', return_value=changed), \
                mock.patch.object(v22, 'INVENTORY_SHA256', digest), \
                mock.patch.object(v23, 'PREDECESSOR_SHA256', digest):
            with self.assertRaises(ValueError):
                v23.inventory_payload()
        with mock.patch.object(v22, 'inventory_payload', return_value=self.predecessor), \
                mock.patch.object(v22, 'INVENTORY_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                v23.inventory_payload()


if __name__ == '__main__':
    unittest.main()
