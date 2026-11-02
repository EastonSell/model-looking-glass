"""Public v34 reservation integrity tests; no checkpoint or model is loaded."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from looking_glass import pretrained_exposures_v34 as exposure


class ExposureV34Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = exposure.prior.inventory_payload()
        if hashlib.sha256(exposure._canonical(cls.previous)).hexdigest() != exposure.PREDECESSOR_SHA256:
            raise ValueError('actual v33 predecessor differs')

    def test_exact_public_cases_and_unchanged_predecessor(self):
        case_path = Path(__file__).parents[1] / exposure.CASE_PATH
        data = case_path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), exposure.CASE_SHA256)
        cases = json.loads(data)['cases']
        current = exposure._build_payload(self.previous)
        self.assertEqual(current['records'][:len(self.previous['records'])], self.previous['records'])
        added = current['records'][len(self.previous['records']):]
        self.assertEqual(len(added), 16)
        for actual, expected in zip(added, cases):
            self.assertEqual(actual['record_id'], 'planned-' + expected['case_id'])
            self.assertEqual(actual['category'], 'planned')
            self.assertEqual(actual['case_role'], expected['split'])
            self.assertEqual(actual['public_text'], expected['public_text'])
            self.assertEqual(actual['token_ids'], expected['token_ids'])
            self.assertEqual(actual['candidate_token_ids'], [318, 389])
        self.assertEqual(hashlib.sha256(exposure._canonical(current)).hexdigest(), exposure.INVENTORY_SHA256)
        self.assertEqual(len(exposure._prefixes(current['records'])), exposure.FORBIDDEN_PREFIX_COUNT)

    def test_equal_shorter_and_longer_foreign_rows_reject(self):
        row = exposure.KNOWN_INJECTION_ROWS[0]
        for foreign in (row, row[:-1], row + (318,)):
            altered = copy.deepcopy(self.previous)
            altered['records'].append({'record_id': 'fabricated-foreign', 'token_ids': list(foreign)})
            with self.subTest(row=foreign), self.assertRaises(ValueError):
                exposure._build_payload(altered)

    def test_mutated_own_identity_rejects(self):
        original = exposure.KNOWN_INJECTION_CASES
        try:
            first = original[0]
            exposure.KNOWN_INJECTION_CASES = ((original[1][0], *first[1:]), *original[1:])
            with self.assertRaises(ValueError):
                exposure._build_payload(self.previous)
        finally:
            exposure.KNOWN_INJECTION_CASES = original


if __name__ == '__main__':
    unittest.main()
