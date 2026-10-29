import math
import struct
import pytest
from looking_glass import copy_behavior_output_dose_v1 as dose
from looking_glass.copy_behavior_effects_v1 import summarize_family, decision


def comparisons():
    rows=[]
    for i in range(8):
        repeated=i%4==0
        rows.append({h:{'target_log_probability_delta':v if repeated else 0.0,
                        'target_logit_delta':2*v if repeated else 0.0}
                     for h,v in [('h2',-1.0),('h10',-0.5),('h7',0.0)]})
    return rows


def test_two_rotation_summary_and_fixed_underfill():
    supported={2,10,12,16,20}
    families=[summarize_family(g,'supported' if g in supported else 'unsupported',
                                'eligible',1.0,comparisons()) for g in range(2,25,2)]
    assert families[0]['D']=={'h2':-1.0,'h10':-0.5,'h7':0.0}
    assert len(families[0]['rotations'])==2
    result=decision(families)
    assert result['primary_status']=='ineligible'
    assert result['primary']=={'count':5,'positive':5,'mean_contrast':0.5,'mean_sign':1}
    assert result['all_dose_eligible']['count']==12
    assert result['baseline_unsupported_dose_eligible']['count']==7


def test_invalid_and_nonfinite_do_not_become_ineligible():
    rows=comparisons(); rows[1]['h2']['target_log_probability_delta']=float('nan')
    with pytest.raises(ValueError): summarize_family(2,'supported','eligible',1.0,rows)
    with pytest.raises(ValueError): summarize_family(2,'supported','invalid',1.0,comparisons())
    with pytest.raises(ValueError): decision([{'group':g,'status':'invalid','baseline_status':'supported'} for g in range(2,25,2)])


def test_eight_prompt_twentyfour_way_dose_and_actuals():
    witnesses=[]
    for v in range(8):
        raw=bytearray(11*768*4)
        for h in dose.HEADS:
            struct.pack_into('<f',raw,(10*768+64*h)*4,1+v/8+h/128)
        witnesses.append(bytes(raw))
    slices={}
    for h in dose.HEADS:
        raw=bytearray(dose.WEIGHT_BYTES)
        struct.pack_into('<f',raw,0,1+h/32)
        struct.pack_into('<f',raw,4*(768+3),0.25)
        slices[h]=bytes(raw)
    plan=dose.plan_family(tuple(witnesses),slices)
    assert plan['status']=='eligible' and len(plan['plans'])==8
    assert len(plan['source_output_norms'])==8
    q=plan['q_out']
    assert all(abs(plan['plans'][v][h]['predicted_output_l2']-q)<=1e-4*q
               for v in range(8) for h in dose.HEADS)
    noops=tuple(bytes(11*768*4) for _ in range(8)); arms=[]
    for v in range(8):
        row={}
        for h in dose.HEADS:
            raw=bytearray(noops[v])
            vec=plan['plans'][v][h]['predicted_vector']
            for j,x in enumerate(vec): struct.pack_into('<f',raw,(10*768+j)*4,x)
            row[h]=bytes(raw)
        arms.append(row)
    assert dose.inspect_actual(noops,tuple(arms),plan)['status']=='eligible'
    with pytest.raises(ValueError): dose.plan_family(tuple(witnesses[:4]),slices)

def test_fixed_six_family_and_strict_positive_boundaries():
    families=[summarize_family(g,'supported' if g<=12 else 'unsupported','eligible',1.0,comparisons())
              for g in range(2,25,2)]
    for i in (0,1):
        families[i].update({'contrast':0.0,'contrast_sign':0,'contrast_terms':[0.0]*24})
    result=decision(families)
    assert result['primary']['count']==6 and result['primary']['positive']==4
    assert result['primary_status']=='criterion-not-met'
    families[1].update({'contrast':0.5,'contrast_sign':1,'contrast_terms':[3.0]+[0.0]*23})
    result=decision(families)
    assert result['primary']['positive']==5 and result['primary_status']=='criterion-met'


def test_equal_repeat_and_shuffles_cancel_exactly():
    x=0.7609624449125756
    rows=[{h:{'target_log_probability_delta':x if h=='h10' else 0.0,
               'target_logit_delta':x if h=='h10' else 0.0} for h in ('h2','h10','h7')} for _ in range(8)]
    family=summarize_family(2,'supported','eligible',1.0,rows)
    assert family['D']=={'h2':0.0,'h10':0.0,'h7':0.0}
    assert family['contrast']==0.0


def test_contrast_keeps_small_difference_when_head_means_round_equal():
    large=2.0**60
    rows=[]
    for i in range(8):
        r2=large if i%4==0 else 0.0
        r10=-3.0 if i==1 else r2
        rows.append({'h2':{'target_log_probability_delta':r2,'target_logit_delta':r2},
                     'h10':{'target_log_probability_delta':r10,'target_logit_delta':r10},
                     'h7':{'target_log_probability_delta':0.0,'target_logit_delta':0.0}})
    family=summarize_family(2,'supported','eligible',1.0,rows)
    assert family['D']['h2']==family['D']['h10']==large
    assert family['contrast']==0.5


def test_exact_zero_from_two_rotations_is_not_positive():
    h2=[0.7,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    h10=[0.5,0.0,0.0,0.0,0.19999999999999996,0.0,0.0,0.0]
    rows=[]
    for a,b in zip(h2,h10):
        rows.append({h:{'target_log_probability_delta':v,'target_logit_delta':v}
                     for h,v in [('h2',a),('h10',b),('h7',0.0)]})
    family=summarize_family(2,'supported','eligible',1.0,rows)
    assert family['contrast']==0.0 and family['contrast_sign']==0


def test_aggregate_sign_uses_all_unrounded_family_terms():
    large=2.0**60
    families=[]
    for g in range(2,25,2):
        rows=[]
        for i in range(8):
            value=(large if i%4==0 else -3.0 if i==1 else 0.0) if g==2 else (-large if i%4==0 else 0.0) if g==4 else 0.0
            rows.append({h:{'target_log_probability_delta':v,'target_logit_delta':v}
                         for h,v in [('h2',0.0),('h10',value),('h7',0.0)]})
        families.append(summarize_family(g,'supported' if g in (2,4) else 'unsupported','eligible',1.0,rows))
    assert math.fsum(f['contrast'] for f in families[:2])==0.0
    result=decision(families)
    assert result['primary']['mean_contrast']==0.25 and result['primary']['mean_sign']==1
