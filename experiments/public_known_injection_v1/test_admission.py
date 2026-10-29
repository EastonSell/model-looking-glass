import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from admission import verify_inventory


class AdmissionTests(unittest.TestCase):
    def test_exact_planned_rows_and_tamper_rejections(self):
        cases = [{'case_id': f'known-injection-{i:02d}',
                  'split': 'discovery' if i <= 8 else 'held-out',
                  'token_ids': [464, 1000 + i]} for i in range(1, 17)]
        inventory = {'version': 'distilgpt2-development-exposures-v34',
                     'records': [{'record_id': 'planned-' + case['case_id'], 'category': 'planned',
                                  'case_role': case['split'], 'token_ids': case['token_ids'],
                                  'candidate_token_ids': [318, 389]} for case in cases]}
        raw = json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'inventory.json'
            path.write_bytes(raw)
            self.assertEqual(verify_inventory(path, hashlib.sha256(raw).hexdigest(), cases)['own_planned_rows'], 16)
            for mutation in ('observed', 'different-row', 'duplicate', 'foreign-prefix'):
                altered = json.loads(raw)
                own = next(x for x in altered['records'] if x['record_id'] == 'planned-' + cases[0]['case_id'])
                if mutation == 'observed':
                    own['category'] = 'observed-saved'
                elif mutation == 'different-row':
                    own['token_ids'][-1] += 1
                elif mutation == 'duplicate':
                    altered['records'].append(dict(own))
                else:
                    altered['records'].append({'record_id': 'foreign-prefix', 'token_ids': cases[0]['token_ids'][:-1]})
                data = json.dumps(altered, sort_keys=True, separators=(',', ':')).encode()
                path = Path(temp) / 'inventory.json'
                path.write_bytes(data)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    verify_inventory(path, hashlib.sha256(data).hexdigest(), cases)


if __name__ == '__main__':
    unittest.main()
