"""Summarize closed v4 records already accepted by the independent replay.

Input evidence must be quiescent. This derives descriptive quantities from
audited pair/result JSON; it does not authenticate execution or infer effects
on unselected images. It excludes unaudited human-readable class labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    raw=path.read_bytes()
    return raw,json.loads(raw)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(audit_path,manifest_path,screen_path,data):
    audit_raw,audit=read(audit_path)
    manifest_raw,manifest=read(manifest_path)
    screen_raw,screen=read(screen_path)
    terminal_raw,terminal=read(data/'terminal.json')
    if (audit.get('schema')!='independent-vision-recognized-v4-audit-v1' or
            audit.get('intervention') is None or audit.get('pending_intervention') is not False or
            audit['manifest_sha256']!=digest(manifest_raw) or audit['screen_result_sha256']!=digest(screen_raw) or
            audit['intervention']['terminal_sha256']!=digest(terminal_raw) or
            audit['aggregate']!=terminal['aggregate']):
        raise ValueError('summary inputs do not match completed audit')
    screen_by_id={x['image_id']:x for x in screen['screen']}
    if len(screen_by_id)!=64 or len(manifest['images'])!=64:
        raise ValueError('screen inventory differs')
    recognition=[]
    for category in (17,18):
        for role in ('target','donor'):
            eligible=[x for x in manifest['images'] if x['category_id']==category and x['role']==role]
            observed=[screen_by_id[x['image_id']] for x in eligible]
            recognition.append({'category_id':category,'role':role,'count':len(observed),
                                'recognized':sum(x['recognized'] for x in observed),
                                'reason_counts':dict(Counter(reason for x in observed for reason in x['reasons']))})
    wanted=('margin','dog_mass','cat_mass','off_mass','top1_id','off_conditional_tv',
            'raw_margin_change','annotation_oriented_change','actual_dose',
            'receiver_norm','realized_over_receiver_norm','token_doses')
    def score(value):
        return {key:value[key] for key in wanted if key in value}
    q_ratios=[]; actual_ratios=[]; rows=[]
    selected={x['pair_id']:x for x in audit['selected_pairs']}
    pair_hashes=audit['intervention']['pair_hashes']
    for report in audit['intervention']['pairs']:
        pair_id=report['pair_id']; pair_raw,pair=read(data/f'{pair_id}.json')
        if pair_hashes[pair_id]!=digest(pair_raw) or pair['status']!=report['status']:
            raise ValueError('pair record differs from the completed audit')
        case={'pair_id':pair_id,'status':pair['status'],'site':selected[pair_id]['site'],
              'baseline_eligibility':pair['baseline_eligibility'],'dose_matchability':pair['dose_matchability'],
              'sides':{}}
        for side in ('cat','dog'):
            info=pair['sides'][side]; dose=info['dose']; conditions=info['conditions']
            relative_q=None
            if 'q' in dose and dose['receiver_norm']>0:
                relative_q=dose['q']/dose['receiver_norm'];q_ratios.append(relative_q)
            movements={k:conditions['matched_'+k]['annotation_oriented_change']
                       for k in ('opposite','same_species','random') if 'matched_'+k in conditions}
            side_pass=None
            if len(movements)==3:
                side_pass=(movements['opposite']>=.05 and movements['opposite']>=
                           max(abs(movements['same_species']),abs(movements['random']))+.025)
            for key in ('matched_opposite','matched_same_species','matched_random'):
                if key in conditions: actual_ratios.append(conditions[key]['realized_over_receiver_norm'])
            case['sides'][side]={'image_id':info['image_id'],'donor_id':info['same_species_donor_id'],
                'baseline':score(info['baseline']),'same_species_baseline':score(info['same_species_baseline']),
                'dose':dose,'q_over_receiver_norm':relative_q,
                'natural_opposite_donor_dose':dose['opposite_distance'],
                'matched_annotation_oriented_movements':movements,'side_rule_pass':side_pass,
                'conditions':{name:score(value) for name,value in conditions.items()}}
        rows.append(case)
    if len(rows)!=len(selected) or [row['pair_id'] for row in rows]!=[f'pair-{i:02d}' for i in range(len(rows))]:
        raise ValueError('selected pair order differs')
    def span(values):
        return {'count':len(values),'min':min(values) if values else None,'max':max(values) if values else None}
    return {'schema':'vision-recognized-v4-descriptive-summary-v1',
            'scope':'Conditional on a complete independent saved-byte replay and quiescent inputs; '
                    'no source, model, input-image, execution, chronology or pretraining-history attestation. '
                    'Screened cases are model-conditioned and reused donors are dependent.',
            'audit_sha256':digest(audit_raw),'manifest_sha256':digest(manifest_raw),
            'screen_result_sha256':digest(screen_raw),'intervention_terminal_sha256':digest(terminal_raw),
            'synthetic':audit['synthetic'],'screen_recognition':recognition,
            'screen_summary':audit['screen_summary'],'aggregate':audit['aggregate'],
            'intervention_forwards':audit['intervention']['forwards'],
            'q_over_receiver_norm':span(q_ratios),'matched_actual_over_receiver_norm':span(actual_ratios),
            'pairs':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','manifest','screen','data','output'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); result=run(args.audit,args.manifest,args.screen,args.data)
    raw=(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with args.output.open('xb') as f:
        if f.write(raw)!=len(raw): raise OSError('short report write')
    print(json.dumps({'sha256':digest(raw),'pairs':len(result['pairs']),'aggregate':result['aggregate']}))


if __name__=='__main__': main()
