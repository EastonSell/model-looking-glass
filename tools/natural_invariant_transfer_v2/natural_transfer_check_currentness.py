import copy
import json
import os
import unittest
from pathlib import Path

import natural_transfer_currentness as c


class CurrentnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case_bytes = Path(os.environ['NATURAL_CASES']).read_bytes()
        cls.cases = json.loads(cls.case_bytes)['cases']
        cls.inventory_bytes = Path(os.environ['NATURAL_INVENTORY']).read_bytes()
        cls.inventory = json.loads(cls.inventory_bytes)
        ids = {'planned-' + row['case_id'] for row in cls.cases}
        cls.minimal = {'records': [r for r in cls.inventory['records'] if r['record_id'] in ids]}

    def compare(self, inventory):
        return c.compare(self.case_bytes, json.dumps(inventory, sort_keys=True).encode())

    def test_current_public_inventory_has_no_foreign_relation(self):
        report = c.compare(self.case_bytes, self.inventory_bytes)
        self.assertEqual(report['own_sequences'], 612)
        self.assertEqual(report['own_reservation_count'], 68)
        self.assertEqual(report['relations'], 0)
        self.assertEqual(report['inventory_sha256'], c.digest(self.inventory_bytes))

    def test_foreign_equal_shorter_and_longer_rows_all_reject(self):
        target = self.cases[0]['token_ids']
        for row in (target, target[:-1], target+[123]):
            inventory = copy.deepcopy(self.minimal)
            inventory['records'].append({'record_id': 'foreign', 'token_ids': row})
            self.assertGreater(self.compare(inventory)['relations'], 0)

    def test_masked_active_and_scored_continuations_are_compared(self):
        target = self.cases[0]['token_ids']
        inventory = copy.deepcopy(self.minimal)
        inventory['records'].append({'record_id': 'foreign-mask', 'token_ids': target+[1],
                                     'attention_mask': [1]*len(target)+[0]})
        self.assertGreater(self.compare(inventory)['relations'], 0)
        inventory = copy.deepcopy(self.minimal)
        inventory['records'].append({'record_id':'foreign-scored', 'token_ids':[100,101,102],
                                     'scored_prefix_token_ids':target[:-1],
                                     'scored_candidate_token_ids':[target[-1]]})
        self.assertGreater(self.compare(inventory)['relations'], 0)

    def test_fixed_pending_france_and_continuation_validation(self):
        raw = Path(os.environ['NATURAL_PENDING_FRANCE']).read_bytes()
        report = c.compare_pending(self.case_bytes, raw)
        self.assertEqual(report['own_sequences'], 612)
        self.assertEqual(report['foreign_sequences'], 73)
        self.assertEqual(report['relations'], 0)
        bundle = json.loads(raw)
        bundle['rows'][0]['continuations'][0]['appended_token_ids'].append(1)
        with self.assertRaises(ValueError):
            c.compare_pending(self.case_bytes, json.dumps(bundle).encode())
    def test_false_own_identity_and_boolean_token_reject(self):
        inventory = copy.deepcopy(self.minimal)
        inventory['records'][0]['record_id'] = 'missing-own'
        with self.assertRaises(ValueError):
            self.compare(inventory)
        inventory = copy.deepcopy(self.minimal)
        inventory['records'][0]['token_ids'][0] += 1
        with self.assertRaises(ValueError):
            self.compare(inventory)
        inventory = copy.deepcopy(self.minimal)
        inventory['records'][0]['scored_prefix_token_ids'] = inventory['records'][0]['token_ids']
        with self.assertRaises(ValueError):
            self.compare(inventory)
        inventory = copy.deepcopy(self.minimal)
        inventory['records'].append({'record_id':'foreign', 'token_ids':[True,42]})
        with self.assertRaises(ValueError):
            self.compare(inventory)


if __name__ == '__main__':
    unittest.main()
