"""Verify the fixed public reservation without a model or protected rows."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from looking_glass import pretrained_exposures_v26 as prior
from looking_glass import pretrained_exposures_v27 as current


class ReservationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predecessor = prior.inventory_payload()
        assert hashlib.sha256(current._canonical(cls.predecessor)).hexdigest() == current.PREDECESSOR_SHA256
        raw = (Path(__file__).resolve().parents[1] / current.CASE_PATH).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == current.CASE_SHA256
        cls.cases = json.loads(raw)['cases']

    def test_exact_rows_roles_answers_and_predecessor(self):
        payload = current._build_payload(self.predecessor)
        n = len(self.predecessor['records'])
        self.assertEqual(payload['records'][:n], self.predecessor['records'])
        added = payload['records'][n:]
        self.assertEqual(len(added), 80)
        self.assertEqual(len(current.AGREEMENT_SUBJECT_FOLLOWUP_CASES), 80)
        for case, record, fixed in zip(self.cases, added, current.AGREEMENT_SUBJECT_FOLLOWUP_CASES):
            self.assertEqual(fixed, (case['case_id'], case['text'], tuple(case['token_ids']), case['role']))
            self.assertEqual(record['record_id'], 'planned-' + case['case_id'])
            self.assertEqual(record['category'], 'planned')
            self.assertEqual(record['token_ids'], case['token_ids'])
            self.assertEqual(record['public_text'], case['text'])
            self.assertEqual(record['case_role'], case['role'])
            self.assertEqual(record['candidate_token_ids'], list(current.AGREEMENT_SUBJECT_FOLLOWUP_ANSWER_IDS))
            self.assertEqual(record['reference_commit'], current.CASES_COMMIT)
        self.assertEqual(hashlib.sha256(current._canonical(payload)).hexdigest(), current.INVENTORY_SHA256)
        self.assertEqual(len(current._prefixes(payload['records'])), current.FORBIDDEN_PREFIX_COUNT)

    def test_foreign_full_row_and_prefix_relations_rejected(self):
        row = list(current.AGREEMENT_SUBJECT_FOLLOWUP_ROWS[0])
        for foreign in (row, row[:-1], row + [318]):
            changed = copy.deepcopy(self.predecessor)
            changed['records'].append({'record_id': 'foreign-collision', 'token_ids': foreign})
            with self.assertRaisesRegex(ValueError, 'prefix'):
                current._build_payload(changed)

    def test_preexisting_planned_identity_rejected(self):
        changed = copy.deepcopy(self.predecessor)
        changed['records'].append({'record_id': 'planned-' + self.cases[0]['case_id'], 'token_ids': [50256]})
        with self.assertRaisesRegex(ValueError, 'identities'):
            current._build_payload(changed)


if __name__ == '__main__':
    unittest.main()
