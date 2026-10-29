import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import natural_transfer_analyze as a
from natural_transfer_check_executor import prepared


def sha(data): return hashlib.sha256(data).hexdigest()


class TransferAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = Path(os.environ['NATURAL_FAKE_FIXTURE'])
        cls.mapping, cls.prior, _ = prepared()
        cls.meta = json.loads((cls.fixture/'fixture.json').read_bytes())
        cls.freeze = json.loads((cls.fixture/'freeze.json').read_bytes())
        cls.expected_result = json.loads((cls.fixture/'result.json').read_bytes())
        cls.tmp = tempfile.TemporaryDirectory(dir=os.environ['NATURAL_TEST_TEMP'])
        cls.discovery = Path(cls.tmp.name)/'discovery'
        shutil.copytree(cls.fixture/'discovery',cls.discovery)
        cls.original_events=(cls.discovery/'events.jsonl').read_bytes()
        cls.records=[json.loads(x) for x in cls.original_events.splitlines()]

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def setUp(self): (self.discovery/'events.jsonl').write_bytes(self.original_events)

    def binding(self, phase):
        value={x:'1'*64 for x in ('pre_run_sha256','prior_receipt_sha256','latest_exposure_sha256')}
        value['mapping_sha256']=self.meta['mapping_sha256']
        if phase=='evaluation': value['freeze_sha256']=self.meta['freeze_sha256']
        return value

    def load(self, phase, root=None, event_sha=None, binding=None):
        root = root or self.fixture/phase
        data=self.meta['phases'][phase]
        return a.load_phase(root,self.mapping,self.prior,phase,events_sha256=event_sha or data['events_sha256'],
                            terminal_sha256=data['terminal_sha256'],expected_binding=binding or self.binding(phase),
                            width=8,vocabulary=1500)

    def rehash(self, records):
        raw=b''.join((json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode() for x in records)
        (self.discovery/'events.jsonl').write_bytes(raw)
        return sha(raw)

    def test_exact_fixture_freeze_and_evaluation(self):
        discovery=self.load('discovery')
        frozen=a.make_freeze(discovery,self.mapping,mapping_sha256=self.meta['mapping_sha256'],
                            prior_events_sha256=self.meta['prior_events_sha256'],
                            discovery_events_sha256=self.meta['phases']['discovery']['events_sha256'])
        self.assertEqual(frozen,self.freeze)
        self.assertEqual(a.evaluate(self.load('evaluation'),frozen),self.expected_result)

    def test_rehashed_hook_metadata_mutations_fail(self):
        changes=[
            ('dtype','torch.float64'),('device','cuda:0'),('forward_index',1),
            ('preceding_same_site_actions',[]),('output_shape',[1,1,8]),
            ('selected_shape',[1,8]),('source_capture','before'),('module_aliases',[]),
            ('order',True),('invocation',False),('batch_index',False),
        ]
        for key,value in changes:
            records=copy.deepcopy(self.records)
            records[1]['event']['plan_metadata']['steps'][2][key]=value
            with self.subTest(field=key):
                with self.assertRaises(a.SavedEvidenceError):
                    self.load('discovery',self.discovery,self.rehash(records))

    def test_rehashed_dose_boolean_and_input_types_fail(self):
        for mutate in (lambda e: e[1]['event']['dose']['certificate']['checks'].__setitem__('delivered',1),
                       lambda e: e[1]['event']['input_ids'].__setitem__(2,False)):
            records=copy.deepcopy(self.records); mutate(records)
            with self.assertRaises(a.SavedEvidenceError):
                self.load('discovery',self.discovery,self.rehash(records))

    def test_wrong_or_missing_evaluation_freeze_binding_fails(self):
        binding=self.binding('evaluation'); binding['freeze_sha256']='2'*64
        with self.assertRaises(a.SavedEvidenceError): self.load('evaluation',binding=binding)
        del binding['freeze_sha256']
        with self.assertRaises(a.SavedEvidenceError): self.load('evaluation',binding=binding)

    def test_selector_accepts_only_anonymous_numeric_matrix(self):
        import inspect
        rows=self.load('discovery')
        matrix=[[row['candidates'][key]['oriented_delta_M'] for key in ('Q1','Q2','Q3','Q4')]
                for row in rows]
        selection=a.select_numeric(matrix)
        self.assertEqual(list(inspect.signature(a.select_numeric).parameters), ['oriented_rows'])
        self.assertEqual(selection,a.select_numeric(list(reversed(matrix))))
        frozen=a.make_freeze(rows,self.mapping,mapping_sha256='1'*64,prior_events_sha256='2'*64,
                            discovery_events_sha256='3'*64)
        self.assertEqual(selection['ranking'],frozen['ranking'])
        self.assertEqual(selection['selected'],frozen['selected'])
        for value in (True, float('nan'), {'case_id':'not numeric'}):
            malformed=copy.deepcopy(matrix); malformed[0][0]=value
            with self.assertRaises(a.SavedEvidenceError): a.select_numeric(malformed)

    def test_ties_and_zero_abstention(self):
        rows=self.load('discovery')
        for row in rows:
            for candidate in row['candidates'].values(): candidate['oriented_delta_M']=0.0
        frozen=a.make_freeze(rows,self.mapping,mapping_sha256='1'*64,prior_events_sha256='2'*64,
                             discovery_events_sha256='3'*64)
        self.assertIsNone(frozen['selected'])
        self.assertEqual(frozen['ranking'],['Q1','Q2','Q3','Q4'])
        for row in rows:
            for candidate in row['candidates'].values(): candidate['oriented_delta_M']=1.0
        frozen=a.make_freeze(rows,self.mapping,mapping_sha256='1'*64,prior_events_sha256='2'*64,
                             discovery_events_sha256='3'*64)
        self.assertEqual(frozen['selected'],'Q1')
        self.assertEqual(frozen['ties'],[['Q1','Q2','Q3','Q4']])

    def test_strict_primary_threshold_and_ineligible_baseline(self):
        rows=self.load('evaluation')
        for row in rows:
            row['receiver_M']=1.0 if row['determiner']=='this' else -1.0
            row['D0']=1.0
            for key,candidate in row['candidates'].items():
                candidate['Dq']=0.5 if key=='Q2' else 0.9
        result=a.evaluate(rows,self.freeze)
        self.assertTrue(result['eligible']); self.assertTrue(result['primary_pattern_supported'])
        for row in rows: row['candidates']['Q2']['Dq']=0.75
        self.assertFalse(a.evaluate(rows,self.freeze)['primary_pattern_supported'])
        for row in rows: row['receiver_M']=0.0
        result=a.evaluate(rows,self.freeze)
        self.assertFalse(result['eligible']); self.assertIsNone(result['primary_pattern_supported'])
        for row in rows:
            row['receiver_M']=1.0 if row['determiner']=='this' else -1.0
            row['D0']=1e-13
        self.assertFalse(a.evaluate(rows,self.freeze)['eligible'])


if __name__=='__main__': unittest.main()
