"""Model-free paired-candidate scores and fixed donor-patch dose arithmetic."""
import math
import struct

from looking_glass.copy_head_factorial import _bits, _norm, _values, f64_to_f32_bits
from looking_glass.copy_head_output_dose import HEADS, FLOOR, TOLERANCE, WIDTH, HEAD_WIDTH, project, weights

VOCAB = 50257
ROW_BYTES = VOCAB * 4
LENGTH = 12
WITNESS_BYTES = LENGTH * WIDTH * 4
PAIR = (1, 0, 3, 2)


def score(raw, correct, other):
    if type(raw) is not bytes or len(raw) != ROW_BYTES:
        raise ValueError('complete float32 vocabulary row required')
    if any(type(i) is not int or not 0 <= i < VOCAB for i in (correct, other)) or correct == other:
        raise ValueError('two distinct candidate IDs required')
    values = _values(raw)
    maximum = max(values)
    logz = math.log(math.fsum(math.exp(x-maximum) for x in values))
    top_count = sum(x == maximum for x in values)
    top = values.index(maximum)
    lc, lo = values[correct], values[other]
    pc, po = math.exp((lc-maximum)-logz), math.exp((lo-maximum)-logz)
    return {'values':values, 'correct_id':correct, 'other_id':other,
            'correct_logit':lc, 'other_logit':lo, 'maximum':maximum, 'logz_shift':logz,
            'correct_log_probability':(lc-maximum)-logz, 'other_log_probability':(lo-maximum)-logz,
            'candidate_mass':math.fsum((pc,po)), 'margin':math.fsum((lc,-lo)),
            'correct_rank':1+sum(x>lc for x in values), 'other_rank':1+sum(x>lo for x in values),
            'top_unique_id':top if top_count == 1 else None,
            'top_representative_id':top, 'top_tie_count':top_count}


def public_score(s):
    return {k:v for k,v in s.items() if k != 'values'}


def compare(clean, patch):
    if len(clean['values']) != VOCAB or len(patch['values']) != VOCAB:
        raise ValueError('two complete scored rows required')
    kl, tv = [], []
    for a,b in zip(clean['values'],patch['values']):
        lp=(a-clean['maximum'])-clean['logz_shift']
        lq=(b-patch['maximum'])-patch['logz_shift']
        p,q=math.exp(lp),math.exp(lq)
        ratio=math.fsum((a,-b,-clean['maximum'],patch['maximum'],
                         -clean['logz_shift'],patch['logz_shift']))
        kl.append(p*ratio); tv.append(abs(p-q))
    if clean['top_unique_id'] is None or patch['top_unique_id'] is None:
        change='tied/indeterminate'
    else:
        change='changed' if clean['top_unique_id'] != patch['top_unique_id'] else 'unchanged'
    return {'kl_clean_to_patch':math.fsum(kl),'total_variation':0.5*math.fsum(tv),'top_change':change}


def baseline(scores):
    if len(scores) != 4:
        raise ValueError('four fixed variants required')
    # Compare the threshold in the same exact sum as the two source f32 logits.
    # A displayed binary64 margin can round up to 1 from just below it.
    odds=all(math.fsum((scores[i]['correct_logit'], -scores[i]['other_logit'], -1.0)) >= 0.0
             for i in (0,1))
    tops=all(scores[i]['top_unique_id'] == scores[i]['correct_id'] for i in (0,1))
    return {'odds_status':'supported' if odds else 'unsupported',
            'top_status':'supported' if tops else 'unsupported'}


def head_slice(raw, head):
    if type(raw) is not bytes or len(raw) != WITNESS_BYTES or head not in HEADS:
        raise ValueError('invalid 12-token head witness')
    offset=(11*WIDTH+head*HEAD_WIDTH)*4
    return raw[offset:offset+HEAD_WIDTH*4]


def plan_family(noop_inputs, weight_slices):
    if type(noop_inputs) is not tuple or len(noop_inputs) != 4:
        raise ValueError('four no-op input witnesses required')
    for raw in noop_inputs:
        if type(raw) is not bytes or len(raw) != WITNESS_BYTES:
            raise ValueError('wrong 12-token witness size')
        _values(raw)
    matrix=weights(weight_slices)
    original=[{h:head_slice(raw,h) for h in HEADS} for raw in noop_inputs]
    values=[{h:_values(row[h]) for h in HEADS} for row in original]
    gaps=[]
    for v in (0,2):
        gaps.append({h:_norm(project(tuple(b-a for a,b in zip(values[v][h],values[v+1][h])),matrix[h]))
                     for h in HEADS})
    q=min(gaps[s][h] for s in (0,1) for h in HEADS)
    minima=[(s,h) for s in (0,1) for h in HEADS if gaps[s][h] == q]
    result={'status':'dose-ineligible','reason':'dose floor','q_out':q,
            'source_output_norms':gaps,'minima':minima,'plans':None}
    if not math.isfinite(q):
        raise ValueError('nonfinite projected donor difference')
    if q <= FLOOR:
        return result
    plans,predicted=[],[]
    for variant in range(4):
        row={}
        for head in HEADS:
            source,donor=values[variant][head],values[PAIR[variant]][head]
            gap=tuple(b-a for a,b in zip(source,donor))
            fraction=q/gaps[variant//2][head]
            delivered=struct.pack('<64I',*(f64_to_f32_bits(float(a+fraction*d)) for a,d in zip(source,gap)))
            after=_values(delivered)
            delta=tuple(b-a for a,b in zip(source,after))
            vector=project(delta,matrix[head]); n=_norm(vector)
            row[head]={'source':original[variant][head], 'donor':original[PAIR[variant]][head],
                       'delivered':delivered,'fraction':fraction,'donor_gap_input_l2':_norm(gap),
                       'delivered_input_l2':_norm(delta),'predicted_output_l2':n,'predicted_vector':vector,
                       'changed_bits':sum((a^b).bit_count() for a,b in zip(_bits(original[variant][head]),_bits(delivered)))}
            predicted.append(n)
        plans.append(row)
    valid=(all(abs(n-q)<=TOLERANCE*q for n in predicted) and max(predicted)-min(predicted)<=TOLERANCE*q)
    result.update({'status':'eligible' if valid else 'dose-ineligible',
                   'reason':None if valid else 'predicted rounded dose mismatch','plans':plans})
    return result


def inspect_actual(noop_outputs, arms, plan):
    if plan.get('status') != 'eligible' or len(noop_outputs) != 4 or len(arms) != 4:
        raise ValueError('four planned output arms required')
    q=plan['q_out']; norms=[]; residuals=[]; descriptions=[]; vectors=[]
    token_start=11*WIDTH*4
    for variant,(noop,row) in enumerate(zip(noop_outputs,arms)):
        if type(noop) is not bytes or len(noop)!=WITNESS_BYTES or set(row)!=set(HEADS):
            raise ValueError('invalid output witness')
        before=_values(noop)[11*WIDTH:]; desc={}; vec={}
        for head in HEADS:
            raw=row[head]
            if type(raw) is not bytes or len(raw)!=WITNESS_BYTES or raw[:token_start]!=noop[:token_start]:
                raise ValueError('untouched output position changed')
            delta=tuple(b-a for a,b in zip(before,_values(raw)[11*WIDTH:]))
            n=_norm(delta); residual=_norm(a-b for a,b in zip(delta,plan['plans'][variant][head]['predicted_vector']))
            norms.append(n); residuals.append(residual); vec[head]=delta
            desc[str(head)]={'actual_output_l2':n,'relative_error':abs(n-q)/q,
                             'predicted_vector_residual_l2':residual}
        descriptions.append(desc); vectors.append(vec)
    cosines=[]
    for vec in vectors:
        row={}
        for h,k in ((2,10),(2,7),(10,7)):
            a,b=vec[h],vec[k]; na,nb=_norm(a),_norm(b)
            row[f'{h}-{k}']=None if na<=FLOOR or nb<=FLOOR else math.fsum(x*y for x,y in zip(a,b))/(na*nb)
        cosines.append(row)
    valid=(all(abs(n-q)<=TOLERANCE*q for n in norms) and all(r<=TOLERANCE*q for r in residuals)
           and max(norms)-min(norms)<=TOLERANCE*q)
    return {'status':'eligible' if valid else 'dose-ineligible',
            'reason':None if valid else 'actual output dose mismatch','q_out':q,
            'variants':descriptions,'pair_cosines':cosines,'actual_range':max(norms)-min(norms)}


def effect_terms(clean, patch):
    if clean['correct_id'] != patch['correct_id'] or clean['other_id'] != patch['other_id']:
        raise ValueError('candidate identity mismatch')
    return (clean['correct_logit'],-clean['other_logit'],-patch['correct_logit'],patch['other_logit'])


def summarize_effects(clean_scores, patch_scores):
    if len(clean_scores)!=4 or len(patch_scores)!=4 or any(set(row)!=set(HEADS) for row in patch_scores):
        raise ValueError('four variants and three fixed heads required')
    terms={h:[effect_terms(clean_scores[v],patch_scores[v][h]) for v in range(4)] for h in HEADS}
    numerators={h:tuple(x for v in range(4) for x in (terms[h][v] if v<2 else tuple(-z for z in terms[h][v])))
                for h in HEADS}
    d={str(h):[math.fsum(terms[h][v]) for v in range(4)] for h in HEADS}
    s={str(h):math.fsum(numerators[h])/2 for h in HEADS}
    s_sign={str(h):(math.fsum(numerators[h])>0)-(math.fsum(numerators[h])<0) for h in HEADS}
    controls={str(h):math.fsum((*numerators[2],*(-x for x in numerators[h]))) for h in (10,7)}
    control_sign={str(h):(controls[str(h)]>0)-(controls[str(h)]<0) for h in (10,7)}
    qualifies=(d['2'][0]>0 and d['2'][1]>0 and s_sign['2']>0 and all(v>0 for v in control_sign.values()))
    return {'d':d,'S':s,'S_sign':s_sign,'S_h2_minus_control':{str(h):controls[str(h)]/2 for h in (10,7)},
            'S_h2_minus_control_sign':control_sign,'h2_control_rule':qualifies}


def decide(families):
    if len(families)!=16 or any(f['odds_status'] not in ('supported','unsupported')
            or f['top_status'] not in ('supported','unsupported')
            or f['dose_status'] not in ('eligible','dose-ineligible') for f in families):
        raise ValueError('16 valid fixed family statuses required')
    dose_eligible=[f for f in families if f['dose_status']=='eligible']
    eligible=[f for f in dose_eligible if f['odds_status']=='supported']
    if any(type(f.get('h2_control_rule')) is not bool for f in dose_eligible):
        raise ValueError('invalid eligible rule')
    def stratum(rows):
        return {'families':len(rows), 'successes':sum(f['h2_control_rule'] for f in rows)}
    successes=sum(f['h2_control_rule'] for f in eligible); n=len(eligible)
    status='ineligible' if n<6 else 'criterion-met' if 4*successes>=3*n else 'criterion-not-met'
    return {'status':status,'eligible':n,'successes':successes,
            'top_supported_in_primary':sum(f['top_status']=='supported' for f in eligible),
            'strata':{'all_dose_eligible':stratum(dose_eligible),
                      'odds_unsupported':stratum([f for f in dose_eligible if f['odds_status']=='unsupported']),
                      'top_supported':stratum([f for f in dose_eligible if f['top_status']=='supported']),
                      'not_top_supported':stratum([f for f in dose_eligible if f['top_status']=='unsupported']),
                      'odds_and_top_supported':stratum([f for f in eligible if f['top_status']=='supported']),
                      'odds_supported_not_top':stratum([f for f in eligible if f['top_status']=='unsupported'])}}
