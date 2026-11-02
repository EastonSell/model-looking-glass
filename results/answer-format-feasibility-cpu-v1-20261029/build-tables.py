"""Build transparent CSV views of the independently scored saved-logit report."""
import argparse
import csv
import json
from pathlib import Path

LABELS = ('factual', 'counterfactual', 'madrid', 'stockholm')

def write(path, rows):
    if not rows:
        raise ValueError('empty table')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cases=json.loads(args.cases.read_bytes())['cases']
    report=json.loads(args.report.read_bytes())
    if report['schema']!='independent-answer-format-saved-logits-v1' or report['heldout'] is None:
        raise ValueError('expected complete two-phase independent report')
    by_id={row['case_id']:row for row in cases}
    rows=[];pairs=[];contrasts=[]
    for stage in ('development','heldout'):
        block=report[stage]
        if len(block['observations'])!=32 or len(block['pairs'])!=48 or len(block['contrasts'])!=8:
            raise ValueError('incomplete phase')
        for x in sorted(block['observations'],key=lambda y:y['canonical_index']):
            c=by_id[x['case_id']]
            r={'stage':stage,'canonical_index':x['canonical_index'],'execution_index':x['execution_index'],
               'case_id':x['case_id'],'fact':x['fact_id'],'family':x['family'],'prefix':x['prefix'],
               'prompt':c['prompt'],'unique_top_id':x['unique_top_id'],'top_token':x['representative_decoded_token'],
               'top_tie_count':x['top_tie_count'],'top_gap':x['top_gap'],'factual_top':x['factual_top'],
               'factual_answer':c['labels']['factual']['text'],'row_sha256':x['row_sha256']}
            for label in LABELS:
                for field in ('token_id','logit','logp'): r[label+'_'+field]=x['scores'][label][field]
            rows.append(r)
        for x in block['pairs']:
            r={k:x[k] for k in ('fact_id','family','reference','comparison','reference_case_id','comparison_case_id','top_relation','kl','total_variation')}
            r={'stage':stage,**r}
            for label in LABELS:r[label+'_logp_delta']=x['logp_delta'][label]
            pairs.append(r)
        for x in block['contrasts']:
            contrasts.append({'stage':stage,**x})
    if len(by_id)!=64 or len(rows)!=64 or len(pairs)!=96 or len(contrasts)!=16:
        raise ValueError('table counts differ')
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'cases.csv',rows)
    write(args.output/'prefix-pairs.csv',pairs)
    write(args.output/'example-contrasts.csv',contrasts)

if __name__=='__main__':main()
