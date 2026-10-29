"""No-model checks for the public matched-study case and donor inventory."""
import json
from pathlib import Path
import unittest

from looking_glass import bert_agreement_cases as prior
from looking_glass import bert_matched_cases as cases


class MatchedCasesTest(unittest.TestCase):
    def test_manifest_is_pinned_and_disjoint_from_all_cited_rows(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/bert_matched_public_cases_v1.json'
        raw = path.read_bytes()
        result = cases.verify_manifest_bytes(raw)
        self.assertEqual((result['evaluation_rows'], result['targets'], result['exact_prior_row_overlap']), (64, 32, 0))
        self.assertFalse(result['cited_history_complete'])
        self.assertFalse(result['execution_authorized'])
        self.assertTrue(set(cases.SELECTED).isdisjoint(prior.LEXEMES))
        with self.assertRaises(ValueError):
            cases.verify_manifest_bytes(raw + b' ')

    def test_all_targets_bind_expected_same_and_opposite_number_donors(self):
        manifest = cases.build_manifest()
        rows = {row['row_id']: row for row in manifest['rows']}
        self.assertEqual(len(rows), 64)
        for target in manifest['targets']:
            own = rows[target['target_row_id']]
            self.assertEqual(own['subject_source'], 'A')
            self.assertEqual(target['donor_row_ids']['A_same'], own['row_id'])
            for name, donor_id in target['donor_row_ids'].items():
                donor = rows[donor_id]
                self.assertEqual(donor['family'], own['family'])
                self.assertEqual(donor['distractor_number'], own['distractor_number'])
                self.assertEqual(donor['distractor_lemma'], own['distractor_lemma'])
                self.assertEqual(donor['subject_source'], name[0])
                self.assertEqual(donor['subject_number'] == own['subject_number'], name.endswith('same'))
                self.assertEqual(donor['input_ids'][:2], own['input_ids'][:2])
                self.assertEqual(donor['input_ids'][3:], own['input_ids'][3:])
        self.assertEqual(len({tuple(row['input_ids']) for row in rows.values()}), 64)
        self.assertEqual(json.loads(cases.canonical_manifest_bytes()), manifest)

    def test_wrong_vocabulary_bytes_reject(self):
        with self.assertRaises(ValueError):
            cases.verify_vocab_bytes(b'not the pinned vocabulary')


if __name__ == '__main__':
    unittest.main()
