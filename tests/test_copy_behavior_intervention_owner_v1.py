import json
from pathlib import Path
import pytest
from scripts import copy_behavior_intervention_v1 as owner


def test_exact_published_screen_binding():
    cases=json.loads((owner.ROOT/owner.CASES).read_bytes())['cases']
    path=Path('E:/Codex/model-looking-glass/work/copy-behavior-screen-runs/attempt-ec54053bfedf489d89085c9168d6b843')
    audit=Path('E:/Codex/model-looking-glass/work/copy-behavior-screen-runs/attempt-ec54053bfedf489d89085c9168d6b843-independent.json')
    if not path.is_dir() or not audit.is_file():
        pytest.skip('the original closed screen and independent audit are not present locally')
    support, raw=owner.verify_screen(path,audit,cases)
    assert len(raw)==192*owner.base.LOGITS_BYTES
    assert [g for g in range(2,25,2) if support[g]=='supported']==[2,10,12,16,20]


def schedule_fixture(monkeypatch,tmp_path,eligible=True,bad_clean=False):
    cases=[{'case_id':f'fake-{v}','group':2,'variant':v} for v in range(8)]
    matrix=bytes(192*owner.base.LOGITS_BYTES)
    order=[]
    plan={'status':'eligible' if eligible else 'dose-ineligible','reason':None if eligible else 'dose floor',
          'q_out':1.0 if eligible else 0.0,'plans':[{h:{'delivered':bytes(256)} for h in owner.HEADS} for _ in range(8)]}
    monkeypatch.setattr(owner,'plan_family',lambda witnesses,weights: plan)
    monkeypatch.setattr(owner.old,'output_description',lambda x:{'status':x['status'],'q_out':x['q_out']})
    monkeypatch.setattr(owner,'inspect_actual',lambda noops,arms,p:{'status':'eligible','q_out':p['q_out']})
    def perform(case,kind,cache,head=None,delivery=None):
        order.append((case['variant'],kind))
        if kind=='clean':
            cache['clean']=b'x' if bad_clean else bytes(owner.base.LOGITS_BYTES)
            cache['clean_score']={'target_rank':1}
        elif kind=='noop':
            cache['before']=b'b';cache['noop_output']=b'o'
        elif kind.startswith('out_'):
            cache.setdefault('arm_outputs',{})[kind]=b'o'
            delta=({'out_h2':-1.0,'out_h10':-0.5,'out_h7':0.0}[kind] if case['variant']%4==0 else 0.0)
            cache.setdefault('comparisons',{})[kind]={'target_log_probability_delta':delta,'target_logit_delta':2*delta}
    with (tmp_path/'journal').open('wb') as log:
        if bad_clean:
            with pytest.raises(RuntimeError):
                owner.execute_family(cases,'unsupported',matrix,perform,lambda:None,log,{})
            return order,None
        family=owner.execute_family(cases,'unsupported',matrix,perform,lambda:None,log,{})
    return order,family


def test_all_eight_baselines_precede_fixed_rotated_arms(monkeypatch,tmp_path):
    order,family=schedule_fixture(monkeypatch,tmp_path)
    assert order[:16]==[(v,kind) for v in range(8) for kind in ('clean','noop')]
    for v in range(8):
        assert order[16+4*v:20+4*v]==[(v,kind) for kind in (*owner.action_order(2,v),'clean_after')]
    assert len(order)==48 and family['summary']['status']=='evaluable'
    assert family['summary']['D']=={'h2':-1.0,'h10':-0.5,'h7':0.0}
    assert family['summary']['contrast']==0.5
    assert family['baseline_status']=='unsupported'


def test_dose_floor_runs_clean_after_without_edits(monkeypatch,tmp_path):
    order,family=schedule_fixture(monkeypatch,tmp_path,eligible=False)
    assert len(order)==24 and all(kind in ('clean','noop','clean_after') for _,kind in order)
    assert family['summary']['status']=='dose-ineligible'


def test_screen_mismatch_stops_before_any_edit(monkeypatch,tmp_path):
    order,_=schedule_fixture(monkeypatch,tmp_path,bad_clean=True)
    assert order==[(0,'clean')]
