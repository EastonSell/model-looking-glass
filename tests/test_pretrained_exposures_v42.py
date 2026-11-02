"""Model-free v42 status-only checks against the exact v41 canonical input."""
import copy
import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

owner = Path(os.environ.get('EXPOSURE_V41_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(owner))
path = Path(os.environ.get('EXPOSURE_V42_SOURCE', owner / 'looking_glass/pretrained_exposures_v42.py'))
spec = importlib.util.spec_from_file_location('looking_glass.pretrained_exposures_v42_test', path)
v42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v42)


class StatusSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        canonical_path = os.environ.get('EXPOSURE_V41_CANONICAL')
        raw = Path(canonical_path).read_bytes() if canonical_path else v42.prior.canonical_inventory_bytes()
        if hashlib.sha256(raw).hexdigest() != v42.PREDECESSOR_SHA256:
            raise AssertionError('wrong predecessor bytes')
        cls.predecessor = json.loads(raw)
        cls.payload = v42._build_payload(cls.predecessor)

    def test_only_fixed_statuses_change(self):
        prior, result = self.predecessor['records'], self.payload['records']
        self.assertEqual(len(prior), len(result))
        changed=[]
        mutable={'record_id','category','provenance','reference_commit','references'}
        for old,new in zip(prior,result):
            if old==new: continue
            changed.append(old['record_id'])
            self.assertTrue(old['record_id'].startswith('planned-'))
            self.assertEqual(new['record_id'],'observed-'+old['record_id'][8:])
            self.assertEqual(new['category'],
                'observed-reported' if old['record_id'] in v42.DISCOVERY_IDS else 'observed-saved')
            self.assertEqual({k:v for k,v in old.items() if k not in mutable},
                             {k:v for k,v in new.items() if k not in mutable})
        self.assertEqual(set(changed),set(v42.DISCOVERY_IDS)|set(v42.COPY_IDS))
        self.assertEqual(len(changed),226)
        original={r['record_id']:r for r in prior}; resulting={r['record_id']:r for r in result}
        self.assertTrue(all(resulting[x]==original[x] for x in v42.EVALUATION_IDS))

    def test_prefixes_and_known_relation_are_preserved(self):
        prior=self.predecessor['records']; result=self.payload['records']
        self.assertEqual(v42._prefixes(prior),v42._prefixes(result))
        self.assertEqual(len(v42._prefixes(result)),9011)
        complete={tuple(row['token_ids']) for row in result}; forbidden=v42._prefixes(result)
        row=tuple(next(x for x in result if x['record_id']=='observed-copy-behavior-01-0')['token_ids'])
        self.assertTrue(v42._has_known_relation(row,forbidden,complete))
        self.assertTrue(v42._has_known_relation(row[:-1],forbidden,complete))
        self.assertTrue(v42._has_known_relation(row+(0,),forbidden,complete))

    def test_missing_or_duplicate_predecessor_rejects(self):
        predecessor=copy.deepcopy(self.predecessor)
        predecessor['records']=[x for x in predecessor['records'] if x['record_id']!=v42.DISCOVERY_IDS[0]]
        with self.assertRaises(ValueError):v42._build_payload(predecessor)
        predecessor=copy.deepcopy(self.predecessor)
        predecessor['records'].append(copy.deepcopy(predecessor['records'][0]))
        with self.assertRaises(ValueError):v42._build_payload(predecessor)


if __name__=='__main__':unittest.main()
