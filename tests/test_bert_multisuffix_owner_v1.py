"""Fabricated CPU checks for the fixed multisuffix intervention schedule."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts import bert_multisuffix_agreement_v1 as owner


class Block(torch.nn.Module):
    def forward(self, hidden): return (hidden,)


class FakeModel(torch.nn.Module):
    def __init__(self, vocab=16, constant=False):
        super().__init__()
        self.block=Block()
        self.vocab=vocab
        self.constant=constant

    def forward(self, input_ids, attention_mask, return_dict):
        del attention_mask, return_dict
        basis=torch.arange(768,device=input_ids.device,dtype=torch.float32)/768+1
        context=input_ids.float().sum(dim=1)[:,None,None]/100000
        tokens=torch.ones_like(input_ids).float() if self.constant else input_ids.float()
        hidden=tokens.unsqueeze(-1)*basis/100 + context*basis.square()
        if self.constant:
            hidden[:,6,:]=basis
        hidden=self.block(hidden)[0]
        value=hidden.sum(dim=(1,2))/10000
        columns=torch.arange(self.vocab,dtype=torch.float32,device=input_ids.device)/20
        scores=value[:,None,None]*columns[None,None,:]
        return SimpleNamespace(logits=scores.expand(-1,input_ids.shape[1],-1))


def manifest():
    return json.loads((Path(__file__).resolve().parents[1]/'experiments/bert_multisuffix_agreement_cases_v1.json').read_bytes())


def execute(tmp_path,monkeypatch,*,vocab=16,constant=False):
    monkeypatch.setattr(owner,'require_policy',lambda device: owner.common.require_subnormals())
    monkeypatch.setattr(owner.common,'memory_sample',lambda device: None)
    assert torch.set_flush_denormal(False)
    model=FakeModel(vocab=vocab,constant=constant)
    path=tmp_path/'attempt';path.mkdir()
    journal=owner.Journal(path,vocab=vocab)
    try:
        eligible=owner.run(model,model.block,manifest(),torch.device('cpu'),journal)
    finally:
        journal.close()
    events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    starts=[json.loads(line) for line in (path/'starts.jsonl').read_text().splitlines()]
    return path,eligible,starts,events,json.loads((path/'plan.json').read_bytes())


def test_complete_fake_schedule_and_same_other_lemma_donors(tmp_path,monkeypatch):
    path,eligible,starts,events,plan=execute(tmp_path,monkeypatch)
    assert eligible==32 and len(starts)==len(events)==1088
    assert [e['arm'] for e in events[:160]]==['clean']*160
    assert all(row['eligible'] and row['requested_q']>0 for row in plan['cases'])
    for case in manifest()['cases']:
        relevant=[e for e in events[160:] if e['case_index']==case['index']]
        assert len(relevant)==29
        for event in relevant:
            if event['arm'] in ('opposite','fixed_terminal_opposite'):
                assert event['donor_index']==case['opposite_donor_index']
            if event['arm']=='same': assert event['donor_index']==case['same_donor_index']
            if event['arm']=='random': assert event['random_seed']==owner.SEED+case['index']
            if event['arm'] in ('opposite','same','random','fixed_terminal_opposite'):
                assert event['actual_l2']>0 and event['relative_direction_error']<=1e-4
    assert not (path/'failure.txt').exists()


def test_finite_zero_dose_keeps_clean_outcomes_and_skips_interventions(tmp_path,monkeypatch):
    path,eligible,starts,events,plan=execute(tmp_path,monkeypatch,constant=True)
    assert eligible==0 and len(starts)==len(events)==160
    assert all(row['eligible'] is False and row['reason']=='zero_donor_distance' for row in plan['cases'])
    assert all(event['arm']=='clean' for event in events)
    assert (path/'logits.f32').stat().st_size==160*16*4


def test_failed_float32_landing_is_base_ineligibility():
    base=np.ones(768,dtype=np.float32)
    direction=np.random.Generator(np.random.PCG64(owner.SEED)).standard_normal(768)
    with pytest.raises(owner.DoseIneligible):
        owner.landing(base,direction,float(np.finfo(np.float32).eps))
    with pytest.raises(ValueError,match='nonfinite'):
        owner.landing(base,direction,float('nan'))


def test_one_finite_landing_failure_does_not_discard_other_bases(tmp_path,monkeypatch):
    original=owner.landing
    count=0
    def first_landing_ineligible(*args):
        nonlocal count
        count+=1
        if count==1: raise owner.DoseIneligible('synthetic rounded no-op')
        return original(*args)
    monkeypatch.setattr(owner,'landing',first_landing_ineligible)
    path,eligible,starts,events,plan=execute(tmp_path,monkeypatch)
    assert eligible==31 and len(starts)==len(events)==160+29*31
    assert plan['cases'][0]['eligible'] is False
    assert plan['cases'][0]['reason']=='rounded_landing:terminal:opposite'
    assert all(row['eligible'] for row in plan['cases'][1:])
    assert not [event for event in events[160:] if event['case_index']==0]
    assert all(len([event for event in events[160:] if event['case_index']==i])==29 for i in range(1,32))


def test_direct_cli_rejects_unleased_cuda_before_attempt(tmp_path,monkeypatch):
    args=['probe','--site',str(tmp_path),'--git',str(tmp_path/'git'),'--model-dir',str(tmp_path),
          '--cases',str(tmp_path/'absent.json'),'--output',str(tmp_path/'attempt'),
          '--device','cuda','--source-commit','a'*40]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(RuntimeError,match='physical-lease owner'):
        owner.main()
    assert not (tmp_path/'attempt').exists()
