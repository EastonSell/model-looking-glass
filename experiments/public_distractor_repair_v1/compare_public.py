"""Compare only known public sequences; no protected or model access."""
import hashlib
import json
import sys
from pathlib import Path


def load(path, expected):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('public source digest mismatch')
    return json.loads(raw)


def row(values):
    if type(values) is not list or not values or any(type(v) is not int or v<0 for v in values):
        raise ValueError('invalid public token row')
    return tuple(values)


def related(a,b): return a[:min(len(a),len(b))]==b[:min(len(a),len(b))]


def main(case_path, inventory_path, original_path, successor_path, output):
    cases=load(case_path,'811a1f0251ff21ddaa1869ee96c80dfa47beb87a6b6c2f1e3b6505728fee1825')['cases']
    inventory=load(inventory_path,'c84d0b83ede1f526215898ce0c2f4f5bdd9c78c8cfeb3f2aced75faaa33fdf6b')
    original=load(original_path,'5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6')
    successor=load(successor_path,'6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4')
    if len(cases)!=16 or inventory.get('version')!='distilgpt2-development-exposures-v37':
        raise ValueError('unexpected cases or inventory version')
    proposed=[]
    for index,case in enumerate(cases):
        base=row(case['token_ids']); prefix=row(case['scored_prefix_token_ids'])
        if base!=prefix+(13,) or case['candidate_token_ids']!=[318,389]:
            raise ValueError('fixed scored-prefix metadata differs')
        proposed.extend((index,x) for x in (base,prefix,prefix+(318,),prefix+(389,)))
    if any(i!=j and related(a,b) for k,(i,a) in enumerate(proposed) for j,b in proposed[:k]):
        raise ValueError('different proposed cases share a full-row prefix')
    foreign=[]
    for entry in inventory['records']:
        base=row(entry['token_ids']); bases=[base]
        if 'attention_mask' in entry:
            mask=entry['attention_mask']
            if len(mask)!=len(base) or any(v not in (0,1) for v in mask): raise ValueError('invalid attention mask')
            active=tuple(v for v,m in zip(base,mask) if m)
            if active and active!=base:bases.append(active)
        tokens=[]
        if 'target_token_id' in entry: tokens.append(entry['target_token_id'])
        tokens.extend(entry.get('candidate_token_ids',[]))
        for group in entry.get('candidate_token_groups',{}).values():tokens.extend(group)
        if any(type(t) is not int or t<0 for t in tokens):raise ValueError('invalid continuation metadata')
        for b in bases:foreign.extend((b,*(b+(t,) for t in tokens)))
    inventory_count=len(foreign)
    for raw in original['rows']:
        b=row(raw);foreign.extend((b,*(b+(t,) for t in (318,389,373,547))))
    original_count=len(foreign)-inventory_count
    for entry in successor['cases']:
        b=row(entry['input_ids']);foreign.extend((b,*(b+(t,) for t in (318,389,373,547))))
    successor_count=len(foreign)-inventory_count-original_count
    collisions=[(i,j) for i,(_,a) in enumerate(proposed) for j,b in enumerate(foreign) if related(a,b)]
    if collisions:raise ValueError('public prefix relations: '+repr(collisions))
    report={'schema':'public-distractor-repair-public-comparison-v1',
       'case_sha256':'811a1f0251ff21ddaa1869ee96c80dfa47beb87a6b6c2f1e3b6505728fee1825',
       'inventory_version':'distilgpt2-development-exposures-v37',
       'inventory_sha256':'c84d0b83ede1f526215898ce0c2f4f5bdd9c78c8cfeb3f2aced75faaa33fdf6b',
       'original_support_sha256':'5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6',
       'successor_support_sha256':'6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4',
       'proposed_sequences':len(proposed),'inventory_sequences':inventory_count,
       'original_support_sequences':original_count,'successor_support_sequences':successor_count,
       'bidirectional_pairs':len(proposed)*len(foreign),'within_cross_case_pairs':1920,'conflicts':0,
       'scope':'public rows and continuations only; protected, unknown outside and pretraining exposure not checked'}
    data=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode('utf-8')
    with Path(output).open('xb') as stream:stream.write(data)
    print(json.dumps({'report_sha256':hashlib.sha256(data).hexdigest(),**report},sort_keys=True))


if __name__=='__main__':main(*sys.argv[1:])
