"""Build case and family CSVs from a closed copying-screen result only."""
import argparse, csv, json, math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.cases.read_bytes())
    result = json.loads(args.result.read_bytes())
    cases, families = data['cases'], result['summary']['families']
    if result['status'] != 'complete' or len(cases) != 192 or len(families) != 24:
        raise ValueError('a complete, fixed 192-row result is required')
    if [f['group'] for f in families] != list(range(1,25)):
        raise ValueError('family order differs')
    groups = {g['group']:g for g in data['groups']}
    case_rows, family_rows = [], []
    for family in families:
        group, scores = family['group'], family['rows']
        group_cases = cases[(group-1)*8:group*8]
        if len(scores)!=8 or any(s['case_id']!=c['case_id'] for s,c in zip(scores,group_cases)):
            raise ValueError('case and score ordering differs')
        gap={(g['rotation'],g['shuffle']):g['gap'] for g in family['gaps']}
        if set(gap)!={(r,s) for r in range(2) for s in range(1,4)}:
            raise ValueError('six gaps per family are required')
        for case, score in zip(group_cases,scores):
            case_rows.append({'case_id':case['case_id'],'group':group,'split':case['split'],
                'rotation':case['rotation'],'shuffle':case['shuffle'],'prompt':case['text'],
                'target_id':case['target_id'],'target_word':groups[group]['words'][5 if case['rotation']==0 else 0],
                'target_log_probability':score['target_log_probability'],
                'target_probability':math.exp(score['target_log_probability']), 'target_rank':score['target_rank'],
                'top_unique_id':score['top_unique_id'],'top_unique_text':score.get('top_unique_text'),
                'top_representative_id':score['top_representative_id'],
                'top_representative_text':score.get('top_representative_text'),
                'top_representative_raw_token':score.get('top_representative_raw_token'),
                'top_tie_count':score['top_tie_count'],'family_baseline_status':family['baseline_status']})
        family_rows.append({'group':group,'split':family['split'],'pool':groups[group]['pool'],
            'words':' '.join(groups[group]['words']), 'baseline_status':family['baseline_status'],
            'rotation_0_target_id':group_cases[0]['target_id'],'rotation_1_target_id':group_cases[4]['target_id'],
            'rotation_0_target_unique_top':scores[0]['top_unique_id']==group_cases[0]['target_id'],
            'rotation_1_target_unique_top':scores[4]['top_unique_id']==group_cases[4]['target_id'],
            **{f'rotation_{r}_shuffle_{s}_gap':gap[r,s] for r in range(2) for s in range(1,4)},
            'minimum_gap':min(gap.values()), 'failure_reasons':json.dumps(family['reasons'],sort_keys=True,separators=(',',':'))})
    args.out.mkdir(parents=True,exist_ok=False)
    for name, rows in [('cases.csv',case_rows),('families.csv',family_rows)]:
        with (args.out/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'cases':len(case_rows),'families':len(family_rows),
        'development_supported':result['summary']['supported_development'],
        'heldout_supported':result['summary']['supported_held_out'],
        'synthetic_fixture':result.get('synthetic_fixture',False)}))

if __name__ == '__main__':
    main()