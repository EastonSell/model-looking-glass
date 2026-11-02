import math
import struct

import pytest

from looking_glass import copy_paired_continuation_v1 as m


def row(changes):
    raw=bytearray(m.ROW_BYTES)
    for i,bits in changes.items(): struct.pack_into('<I',raw,4*i,bits)
    return bytes(raw)


def test_float32_subnormal_unique_top_and_ties_are_not_flushed():
    tiny=row({13:1})
    assert m.score(tiny,13,14)['top_unique_id']==13
    assert m.score(tiny,13,14)['margin']==math.ldexp(1.0,-149)
    tied=m.score(row({}),13,14)
    assert tied['top_unique_id'] is None and tied['top_tie_count']==m.VOCAB
    assert 0 < tied['candidate_mass'] < 1


def test_baseline_odds_and_top_are_distinct_and_include_exact_boundary():
    a=m.score(row({1:0x3f800000,3:0x40000000}),1,2)
    b=m.score(row({2:0x3f800000,3:0x40000000}),2,1)
    gate=m.baseline([a,b,a,b])
    assert gate=={'odds_status':'supported','top_status':'unsupported'}
    tiny = m.score(row({1:0x3f800000, 2:0x24800000}),1,2)
    assert tiny['margin'] == 1.0  # Binary64 rounds 1 - 2**-54 to 1.
    assert m.baseline([tiny,b,tiny,b])['odds_status']=='unsupported'


def make_weights():
    out={}
    for h,scale in ((2,1.0),(10,2.0),(7,1.0)):
        values=[0.0]*(64*768)
        values[0]=scale
        values[5*768+3]=7.0  # Nonsymmetric rows: must use input rows, not columns.
        out[h]=struct.pack('<%df'%len(values),*values)
    return out


def make_noop(variant):
    raw=bytearray(m.WITNESS_BYTES)
    # A wrong position must not be mistaken for the final token.
    struct.pack_into('<f',raw,(10*m.WIDTH+2*64)*4,99.0+variant)
    if variant in (1,3):
        for head,value in ((2,4.0),(10,2.0),(7,4.0)):
            struct.pack_into('<f',raw,(11*m.WIDTH+head*64)*4,value)
    return bytes(raw)


def test_final_position_twelve_token_paired_plan_and_actual_witness():
    inputs=tuple(make_noop(v) for v in range(4))
    plan=m.plan_family(inputs,make_weights())
    assert plan['status']=='eligible' and plan['q_out']==4.0
    for variant in range(4):
        for head in m.HEADS:
            p=plan['plans'][variant][head]
            assert p['fraction']==1.0 and p['predicted_output_l2']==4.0
            assert p['donor_gap_input_l2']>0 and p['delivered_input_l2']>0
            assert p['source']==m.head_slice(inputs[variant],head)
    noops=tuple(bytes(m.WITNESS_BYTES) for _ in range(4))
    arms=[]
    for variant in range(4):
        d={}
        for head in m.HEADS:
            x=bytearray(noops[variant])
            struct.pack_into('<f',x,11*m.WIDTH*4,plan['plans'][variant][head]['predicted_vector'][0])
            d[head]=bytes(x)
        arms.append(d)
    actual=m.inspect_actual(noops,tuple(arms),plan)
    assert actual['status']=='eligible' and actual['actual_range']==0.0
    bad=dict(arms[0]); x=bytearray(bad[2]); x[10*m.WIDTH*4]=1; bad[2]=bytes(x)
    with pytest.raises(ValueError,match='untouched'):
        m.inspect_actual(noops,(bad,*arms[1:]),plan)
    mismatch=dict(arms[0]); x=bytearray(mismatch[2]); struct.pack_into('<f',x,11*m.WIDTH*4,4.5); mismatch[2]=bytes(x)
    assert m.inspect_actual(noops,(mismatch,*arms[1:]),plan)['status']=='dose-ineligible'


def test_zero_donor_difference_skips_all_plans_without_division():
    z=bytes(m.WITNESS_BYTES)
    plan=m.plan_family((z,z,z,z),make_weights())
    assert plan['status']=='dose-ineligible' and plan['q_out']==0.0 and plan['plans'] is None


def test_direct_counterbalanced_effect_and_exact_control_tie():
    clean=[{'correct_id':1,'other_id':2,'correct_logit':2.,'other_logit':0.},
           {'correct_id':2,'other_id':1,'correct_logit':2.,'other_logit':0.}]*2
    patches=[]
    for v,c in enumerate(clean):
        row={}
        for head,delta in ((2,1.0),(10,0.5),(7,0.1)):
            change=delta if v<2 else 0.0
            row[head]={**c,'correct_logit':c['correct_logit']-change}
        patches.append(row)
    s=m.summarize_effects(clean,patches)
    assert s['h2_control_rule'] is True and s['S']['2']==1.0
    assert s['S_h2_minus_control']['10']==0.5
    patches[0][10]=dict(patches[0][2]); patches[1][10]=dict(patches[1][2])
    s=m.summarize_effects(clean,patches)
    assert s['S_h2_minus_control_sign']['10']==0 and s['h2_control_rule'] is False


def test_primary_underfill_and_seventy_five_percent_boundary():
    f=lambda odds,dose,ok:{'odds_status':odds,'top_status':'unsupported','dose_status':dose,'h2_control_rule':ok}
    cases=[f('supported','eligible',True)]*5+[f('unsupported','eligible',True)]*11
    assert m.decide(cases)['status']=='ineligible'
    cases=[f('supported','eligible',True)]*6+[f('supported','eligible',False)]*2+[f('unsupported','eligible',False)]*8
    assert m.decide(cases)['status']=='criterion-met'
    assert m.decide(cases)['strata']['odds_unsupported']=={'families':8,'successes':0}
    assert m.decide(cases)['strata']['all_dose_eligible']=={'families':16,'successes':6}
    cases[0]=f('supported','eligible',False)
    assert m.decide(cases)['status']=='criterion-not-met'
