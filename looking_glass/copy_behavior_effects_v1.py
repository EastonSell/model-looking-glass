"""Fixed two-target family and eligibility summaries for the copying intervention."""
import math

HEADS = ('h2','h10','h7')


def summarize_family(group, baseline_status, dose_status, q, comparisons):
    if type(group) is not int or group not in range(2,25,2):
        raise ValueError('expected a fixed held-out group')
    if baseline_status not in ('supported','unsupported') or dose_status not in ('eligible','dose-ineligible'):
        raise ValueError('invalid or unknown family status')
    if not isinstance(q,(int,float)) or isinstance(q,bool) or not math.isfinite(q) or q < 0:
        raise ValueError('invalid family dose')
    if dose_status == 'dose-ineligible':
        return {'group':group,'baseline_status':baseline_status,'dose_status':dose_status,
                'status':'dose-ineligible','q_out':q,'rotations':None,'D':None,'contrast':None,
                'contrast_terms':None,'contrast_sign':None}
    if q <= 2**-8 or type(comparisons) is not list or len(comparisons) != 8:
        raise ValueError('eligible family needs a positive dose and eight comparisons')
    rotations=[]; d={}; logit_d={}; terms={}; logit_terms={}
    for head in HEADS:
        terms[head]=[]; logit_terms[head]=[]
    for rotation in range(2):
        current={}
        for head in HEADS:
            values=[comparisons[i][head]['target_log_probability_delta'] for i in range(4*rotation,4*rotation+4)]
            logits=[comparisons[i][head]['target_logit_delta'] for i in range(4*rotation,4*rotation+4)]
            if any(not math.isfinite(x) for x in values+logits):
                raise ValueError('nonfinite comparison')
            row=(values[0],values[0],values[0],-values[1],-values[2],-values[3])
            logit_row=(logits[0],logits[0],logits[0],-logits[1],-logits[2],-logits[3])
            terms[head].extend(row); logit_terms[head].extend(logit_row)
            current[head]={'target_log_probability_D':math.fsum(row)/3,
                           'target_logit_D':math.fsum(logit_row)/3}
        rotations.append(current)
    for head in HEADS:
        d[head]=math.fsum(terms[head])/6
        logit_d[head]=math.fsum(logit_terms[head])/6
    def difference(a,b):
        return math.fsum((*terms[a],*(-x for x in terms[b])))/6
    contrast_terms=[*terms['h10'],*(-x for x in terms['h2'])]
    numerator=math.fsum(contrast_terms)
    return {'group':group,'baseline_status':baseline_status,'dose_status':dose_status,
            'status':'evaluable','q_out':q,'rotations':rotations,'D':d,'target_logit_D':logit_d,
            'contrast':numerator/6,'contrast_terms':contrast_terms,
            'contrast_sign':1 if numerator>0 else -1 if numerator<0 else 0,
            'h10_minus_h7':difference('h10','h7'),
            'h2_minus_h7':difference('h2','h7')}



def describe(rows):
    n=len(rows)
    if not n:
        return {'count':0,'positive':0,'mean_contrast':None,'mean_sign':None}
    if any(type(row.get('contrast_terms')) is not list or len(row['contrast_terms']) != 24
           or any(not math.isfinite(x) for x in row['contrast_terms']) for row in rows):
        raise ValueError('invalid family contrast terms')
    numerators=[math.fsum(row['contrast_terms']) for row in rows]
    total=math.fsum(x for row in rows for x in row['contrast_terms'])
    if any(row.get('contrast_sign') != (1 if v>0 else -1 if v<0 else 0)
           or row.get('contrast') != v/6 for row,v in zip(rows,numerators)):
        raise ValueError('family contrast does not match its terms')
    return {'count':n,'positive':sum(v>0 for v in numerators),'mean_contrast':total/(6*n),
            'mean_sign':1 if total>0 else -1 if total<0 else 0}


def decision(families):
    if type(families) is not list or [x.get('group') for x in families] != list(range(2,25,2)):
        raise ValueError('expected all twelve fixed held-out families in order')
    for row in families:
        if row.get('status') not in ('evaluable','dose-ineligible') or row.get('baseline_status') not in ('supported','unsupported'):
            raise ValueError('invalid or unknown family status')
        if row['status']=='evaluable' and (not math.isfinite(row['contrast']) or row.get('dose_status')!='eligible'):
            raise ValueError('invalid evaluable family')
        if row['status']=='dose-ineligible' and (row.get('contrast') is not None or row.get('contrast_terms') is not None
                or row.get('contrast_sign') is not None or row.get('dose_status')!='dose-ineligible'):
            raise ValueError('invalid ineligible family')
    evaluable=[r for r in families if r['status']=='evaluable']
    supported=[r for r in evaluable if r['baseline_status']=='supported']
    unsupported=[r for r in evaluable if r['baseline_status']=='unsupported']
    primary=describe(supported)
    if primary['count'] < 6:
        status='ineligible'
    elif primary['mean_sign'] > 0 and 4*primary['positive'] >= 3*primary['count']:
        status='criterion-met'
    else:
        status='criterion-not-met'
    return {'primary_status':status,'primary':primary,'all_dose_eligible':describe(evaluable),
            'baseline_unsupported_dose_eligible':describe(unsupported),
            'dose_ineligible_groups':[r['group'] for r in families if r['status']=='dose-ineligible']}