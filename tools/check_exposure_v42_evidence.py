"""Reconcile the public v42 status evidence against the pinned v41 rows; no model."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

EXPECTED = {
 'inventory':'110ebcf674168e794544c257d158b3c3242f98bcf108375cdc073bb145e2bfad',
 'invariant_cases':'50ef3aea53a21a76ecf1bc4910882f78939a4d2fcca12528013ea56eaa586827',
 'invariant_events':'0be814ca696b4da0b90c99609098e926db3d489f2010da70c0bc9fd0adc3874d',
 'invariant_terminal':'32326e95b19b9632ed459f06eb781206427a3961a6f6f93df1a442a9277cc26b',
 'invariant_exit':'e401742df188267944a5c25fb7ff9fbb419413e43be97b7b3981da0dff8ea9f4',
 'invariant_prefix':'9bf1540c6b9466a16171c0242553db7b3df3d3c38aca5a58d2e47a40552bbd35',
 'invariant_inventory':'d81fb0d0add26e09974e3f5c5e68cca3279e34bc54593ccc745af68ecb7b137a',
 'copy_cases':'96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734',
 'copy_events':'f5a7857dec21cdc21d855ed4c07686c8b976162e41beae8256c6ab4f0877ade0',
}


def checked(path, key):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED[key]:
        raise ValueError('wrong '+key+' bytes')
    return raw


def require(value, reason):
    if not value: raise ValueError(reason)


def audit(args):
    records=json.loads(checked(args.inventory,'inventory'))['records']
    old={r['record_id']:r for r in records}
    require(len(old)==len(records)==1443,'predecessor record identity/count')
    case_list=json.loads(checked(args.invariant_cases,'invariant_cases'))['cases']
    cases={r['case_id']:r for r in case_list}
    require(len(cases)==len(case_list)==68,'invariant case identity/count')
    staged=Path(args.invariant_stage)
    event_bytes=checked(staged/'events.jsonl','invariant_events')
    terminal=json.loads(checked(staged/'terminal.json','invariant_terminal'))
    exit_record=json.loads(checked(staged/'owner-exit-observation.json','invariant_exit'))
    prefix=json.loads(checked(staged/'independent-prefix-review.json','invariant_prefix'))
    inventory=json.loads(checked(staged/'independent-inventory-review.json','invariant_inventory'))
    events=[json.loads(x) for x in event_bytes.splitlines()]
    require(len(events)==302 and events[0]['event']['kind']=='run_contract','invariant durable prefix length')
    forwards=events[1:]; seen=set()
    for i,row in enumerate(forwards):
        e=row['event']; case=cases.get(e.get('case_id'))
        require(type(row.get('index')) is int and row['index']==i+1 and e.get('kind')=='forward'
                and type(e.get('index')) is int and e['index']==i
                and case is not None and case['split']=='discovery'
                and e.get('split')=='discovery' and e.get('input_ids')==case['token_ids'],
                'invariant forward identity/input')
        seen.add(case['case_id'])
    discovery={r['case_id'] for r in case_list if r['split']=='discovery'}
    evaluation={r['case_id'] for r in case_list if r['split']=='evaluation'}
    require(len(discovery)==len(evaluation)==34 and seen==discovery
            and set(prefix['exposed_case_ids'])==discovery and prefix['checked_saved_forwards']==301,
            'invariant reached-case set')
    require(terminal.get('status')=='failed' and terminal.get('storage_accounting_complete') is False
            and terminal.get('detail',{}).get('forward_evidence_records')==301
            and terminal['detail'].get('model_returns')==302
            and exit_record.get('observer_exit_code')==1
            and exit_record.get('storage_accounting_complete') is False
            and inventory.get('orphan_tensor_files')==3 and inventory.get('saved_forward_records')==301,
            'invariant partial/failure scope')
    for case_id in discovery|evaluation:
        r=old.get('planned-'+case_id); c=cases[case_id]
        require(r is not None and r.get('category')=='planned' and r.get('token_ids')==c['token_ids']
                and r.get('public_text')==c['text'] and r.get('case_role')==c['role'],
                'invariant reservation identity')
    copy_cases=json.loads(checked(args.copy_cases,'copy_cases'))['cases']
    compressed=Path(args.copy_journal).read_bytes()
    raw=gzip.decompress(compressed)
    require(hashlib.sha256(raw).hexdigest()==EXPECTED['copy_events'],'copy journal hash')
    copy_events=[json.loads(x) for x in raw.splitlines()]
    starts=[x for x in copy_events if x.get('event')=='start']
    done=[x for x in copy_events if x.get('event')=='complete']
    require(len(copy_cases)==len(starts)==len(done)==192,'copy count')
    for i,(c,s,d) in enumerate(zip(copy_cases,starts,done)):
        require(c['case_id']==s.get('case_id')==d.get('case_id')
                and type(s.get('index')) is int and s['index']==i
                and type(d.get('index')) is int and d['index']==i
                and d.get('target_id')==c['target_id'] and d.get('group')==c['group']
                and d.get('variant')==c['variant'] and d.get('raw',{}).get('logits',{}).get('offset')==i*201028
                and d['raw']['logits'].get('bytes')==201028,'copy journal row')
        r=old.get('planned-'+c['case_id'])
        require(r is not None and r.get('category')=='planned' and r.get('token_ids')==c['token_ids']
                and r.get('public_text')==c['text'] and r.get('target_token_id')==c['target_id'],
                'copy reservation identity')
    return {'schema':'exposure-v42-evidence-reconciliation-v1','inventory_sha256':EXPECTED['inventory'],
            'invariant_case_sha256':EXPECTED['invariant_cases'],'invariant_events_sha256':EXPECTED['invariant_events'],
            'invariant_saved_forward_records':301,'invariant_reported_returns':302,
            'invariant_discovery_inputs':34,'invariant_evaluation_inputs_preserved_planned':34,
            'invariant_orphan_tensor_files':3,'invariant_storage_accounting_complete':False,
            'copy_case_sha256':EXPECTED['copy_cases'],'copy_events_sha256':EXPECTED['copy_events'],
            'copy_starts':192,'copy_completions':192,'copy_raw_bytes_indexed':192*201028,
            'scope':'Saved metadata and cooperative observations only; this check does not read raw logits, authenticate execution, or inspect protected/unknown exposure.'}


def main():
    ap=argparse.ArgumentParser()
    for option in ('inventory','invariant-cases','invariant-stage','copy-cases','copy-journal','output'):
        ap.add_argument('--'+option,required=True)
    args=ap.parse_args(); result=audit(args)
    raw=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
    with open(args.output,'xb') as out: out.write(raw)
    print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'invariant_inputs':34,'copy_inputs':192},sort_keys=True))


if __name__=='__main__': main()
