"""Fabricated schedule, fail-stop and capacity checks; no pretrained model."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from transformers import GPT2Config, GPT2LMHeadModel
try:
    import distractor_executor as executor
except ModuleNotFoundError:
    executor = None

ROOT = Path(tempfile.gettempdir())
CASE = {'case_id': 'distractor-repair-01', 'split': 'discovery', 'token_ids': [1, 7, 11, 21, 31]}

class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        cfg=GPT2Config(vocab_size=41, n_positions=16, n_embd=8, n_layer=5, n_head=2,
                       use_cache=False, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
        cfg._attn_implementation='eager'
        cls.model=GPT2LMHeadModel(cfg).eval()

    def test_complete_fixed_schedule(self):
        self.assertIsNotNone(executor, 'executor is not implemented')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='dist-fake-') as d:
            out=Path(d)/'run'
            result=executor.execute_phase(self.model,[CASE],out,phase='discovery',
                                          vocab_size=41,raw_cap=5_000_000,metadata_cap=2_000_000,
                                          minimum_free_bytes=0)
            self.assertEqual(result['completed_actions'],28)
            rows=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['type']=='complete' for r in rows),28)
            self.assertEqual(sum(r['type']=='start' for r in rows),28)
            self.assertEqual(rows[-1]['action'],'clean-repeat')
            vectors=[r['tensor'] for r in rows if r['type']=='tensor' and r['tensor']['name']=='realized_displacement']
            self.assertEqual(len(vectors),3)
            self.assertTrue(all(x['dtype']=='float64-le' and x['shape']==[8] for x in vectors))
            self.assertTrue((out/'distractor-repair-01.f64').is_file())

    def test_failed_control_is_retained_and_stops(self):
        self.assertIsNotNone(executor, 'executor is not implemented')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='dist-fake-') as d:
            out=Path(d)/'run'
            with mock.patch.object(executor, 'bitwise_equal', return_value=False):
                with self.assertRaises(executor.ControlFailure):
                    executor.execute_phase(self.model,[CASE],out,phase='discovery',vocab_size=41,
                                           raw_cap=5_000_000,metadata_cap=2_000_000,minimum_free_bytes=0)
            rows=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['type']=='start' for r in rows),2)
            self.assertEqual(sum(r['type']=='complete' for r in rows),1)
            self.assertTrue(any(r['type']=='failed-controls' for r in rows))
            self.assertTrue((out/'failure.json').is_file())

    def test_changed_downstream_base_fails_without_next_forward(self):
        self.assertIsNotNone(executor, 'executor is not implemented')
        original=executor.intervene
        def corrupt(model,tokens,clean,layer,kind,*args,**kwargs):
            result=original(model,tokens,clean,layer,kind,*args,**kwargs)
            if layer==1 and kind=='patch' and kwargs.get('q')=='Q1':
                result['captures']['patch_before']=result['captures']['patch_before'].clone()
                result['captures']['patch_before'][0,-2,0]+=1.0
            return result
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='dist-fake-') as d:
            out=Path(d)/'run'
            with mock.patch.object(executor,'intervene',side_effect=corrupt):
                with self.assertRaises(executor.ControlFailure):
                    executor.execute_phase(self.model,[CASE],out,phase='discovery',vocab_size=41,
                        raw_cap=5_000_000,metadata_cap=2_000_000,minimum_free_bytes=0)
            rows=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
            failed=next(x for x in rows if x['type']=='failed-controls')
            self.assertFalse(failed['checks']['downstream_base_equal_combined'])
            self.assertEqual(sum(x['type']=='start' for x in rows),6)

    def test_tiny_cap_stops_before_any_forward(self):
        self.assertIsNotNone(executor, 'executor is not implemented')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='dist-fake-') as d:
            out=Path(d)/'run'
            with mock.patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
                with self.assertRaises(ValueError):
                    executor.execute_phase(self.model,[CASE],out,phase='discovery',vocab_size=41,
                                           raw_cap=256,metadata_cap=2_000_000,minimum_free_bytes=0)
                self.assertEqual(forward.call_count,0)

if __name__=='__main__':
    unittest.main()
