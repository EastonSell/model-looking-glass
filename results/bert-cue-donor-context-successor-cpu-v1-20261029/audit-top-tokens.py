"""Post-outcome descriptive top-token and dose scan of a closed journal."""
import argparse, array, base64, collections, csv, gzip, hashlib, json, math, sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def top(encoded, size):
    raw=base64.b64decode(encoded,validate=True)
    if len(raw)!=size*4: raise ValueError('logit width differs')
    values=array.array('f'); values.frombytes(raw)
    if sys.byteorder!='little': values.byteswap()
    if any(not math.isfinite(v) for v in values): raise ValueError('nonfinite logit')
    best=max(values); idx=values.index(best); ties=sum(v==best for v in values)
    return idx,ties


def main():
    p=argparse.ArgumentParser()
    for n in ('manifest','journal','result','vocab','output'): p.add_argument(n,type=Path)
    for n in ('manifest','journal','result','vocab'): p.add_argument('--expected-'+n+'-sha256',required=True)
    p.add_argument('--synthetic-vocab-size',type=int,default=None)
    a=p.parse_args()
    files={n:getattr(a,n) for n in ('manifest','journal','result','vocab')}
    expected={n:getattr(a,'expected_'+n+'_sha256') for n in files}
    if any(sha(v)!=expected[k] for k,v in files.items()): raise ValueError('input hash differs')
    manifest=json.loads(a.manifest.read_bytes()); result=json.loads(a.result.read_bytes())
    vocab=a.vocab.read_text(encoding='utf-8').splitlines()
    size=a.synthetic_vocab_size or 30522
    if len(vocab)!=size: raise ValueError('vocabulary size differs')
    split={p['id']:p['split'] for p in manifest['pairs']}
    clean={}; rows=[]; counts=collections.Counter(); norms=[]; fractions=[]; errors=[]
    with gzip.open(a.journal,'rt',encoding='utf-8') as stream:
        for line in stream:
            event=json.loads(line); kind=event['kind']; counts[kind]+=1
            if kind=='clean':
                key=(event['pair_id'],event['context'],event['cue'])
                if key in clean: raise ValueError('duplicate clean')
                token,ties=top(event['mask_logits_f32le'],size)
                clean[key]=(token,ties,event['input_ids'])
            elif kind=='transfer':
                key=(event['pair_id'],event['target_context'],event['target_cue'])
                if key not in clean or event['input_ids']!=clean[key][2]: raise ValueError('transfer target differs')
                token,ties=top(event['mask_logits_f32le'],size); before=clean[key][0]
                rows.append({'pair_id':key[0],'split':split[key[0]],'target_context':key[1],'target_cue':key[2],
                             'donor_context':event['donor_context'],'donor_cue':event['donor_cue'],'site':event['site'],
                             'clean_top_id':before,'clean_top_token':vocab[before],
                             'patched_top_id':token,'patched_top_token':vocab[token],
                             'clean_top_ties':clean[key][1],'patched_top_ties':ties,'top_changed':before!=token})
                norms.append(event['actual_norm']); fractions.append(event['requested_fraction']); errors.append(event['relative_error'])
    expected_transfers=sum(224 for p in result['pairs'] if p['dose_eligible'])
    if len(clean)!=32 or len(rows)!=expected_transfers: raise ValueError('saved schedule count differs')
    summaries=[]
    for split_name in ('discovery','evaluation'):
        for site in ('cue2','object2','mask2','cue5'):
            group=[r for r in rows if r['split']==split_name and r['site']==site]
            changed=[r for r in group if r['top_changed']]
            summaries.append({'split':split_name,'site':site,'transfers':len(group),'changed':len(changed),
                              'unique_recipients_changed':len({(r['pair_id'],r['target_context'],r['target_cue']) for r in changed})})
    report={'scope':'Post-outcome descriptive scan of saved full mask logits and recorded dose fields; not execution or landing attestation.',
            'manifest_sha256':expected['manifest'],'journal_sha256':expected['journal'],'result_sha256':expected['result'],
            'vocab_sha256':expected['vocab'],'event_counts':dict(sorted(counts.items())),
            'top_summaries':summaries,'examples':[r for r in rows if r['top_changed']][:20],
            'clean_top_ties':sum(v[1]>1 for v in clean.values()),'transfer_top_ties':sum(r['patched_top_ties']>1 for r in rows),
            'actual_norm_min':min(norms) if norms else None,'actual_norm_max':max(norms) if norms else None,
            'requested_fraction_min':min(fractions) if fractions else None,'requested_fraction_max':max(fractions) if fractions else None,
            'max_relative_error':max(errors) if errors else None}
    if any(sha(v)!=expected[k] for k,v in files.items()): raise ValueError('input changed during scan')
    a.output.mkdir(parents=False,exist_ok=False)
    with (a.output/'top-tokens.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [],lineterminator='\n'); w.writeheader(); w.writerows(rows)
    with (a.output/'top-audit.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    print(json.dumps({'transfers':len(rows),'top_summaries':summaries,'max_relative_error':report['max_relative_error']},sort_keys=True))


if __name__=='__main__': main()
