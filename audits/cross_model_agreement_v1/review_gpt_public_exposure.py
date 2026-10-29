"""Independently compare fixed GPT inputs with public records and supports only.

No model, sealed split, or private membership oracle is accessed.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer

if len(sys.argv) != 5:
    raise SystemExit('usage: review_gpt_public_exposure.py REPO CASE_MANIFEST TOKENIZER_JSON OUTPUT')
REPO = Path(sys.argv[1]).resolve()
SUPPORT_REPO = REPO
CASE = Path(sys.argv[2]).resolve()
TOKENIZER = Path(sys.argv[3]).resolve()
OUTPUT = Path(sys.argv[4]).resolve()
CASE_SHA = 'b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503'
TOKEN_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
OLD_SHA = '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6'
NEXT_SHA = '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
INVENTORY_SHA = '3a7ecf81eb7f04759ed5b8ffd0607c5955b634b938756e76eb52d2435aef9610'
INVENTORY_COMMIT = '21353f15818b5f38f74b5d555a63235d710916f7'
MODEL_REV = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
SUFFIXES = ((' is',318), (' are',389), (' was',373), (' were',547))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(repo, rev, path):
    return subprocess.check_output(['git', 'show', rev + ':' + path], cwd=repo)


def counts(a,b):
    return {'candidate_prefix_of_reference':sum(len(x)<=len(y) and x==y[:len(x)] for x in a for y in b),
            'reference_prefix_of_candidate':sum(len(y)<=len(x) and y==x[:len(y)] for x in a for y in b)}


def main():
    case_bytes = CASE.read_bytes()
    require(digest(case_bytes) == CASE_SHA, 'case bytes')
    obj = json.loads(case_bytes)
    cases = obj['cases']
    require(len(cases)==32 and obj['gpt']['revision']==MODEL_REV and obj['gpt']['tokenizer_sha256']==TOKEN_SHA, 'candidate identity')
    token_path = TOKENIZER
    require(digest(token_path.read_bytes())==TOKEN_SHA, 'tokenizer bytes')
    tokenizer = Tokenizer.from_file(str(token_path))
    require(tokenizer.padding is None and tokenizer.truncation is None, 'tokenizer settings')
    groups=[]
    for case in cases:
        item=case['gpt']
        row=item['input_ids']
        require(item['attention_mask']==[1]*len(row), 'attention mask')
        require(tokenizer.encode(item['text'],add_special_tokens=False).ids==row, 'base tokenization')
        group=[tuple(row)]
        for suffix,tid in SUFFIXES:
            require(tokenizer.encode(item['text']+suffix,add_special_tokens=False).ids==row+[tid], 'continuation tokenization')
            group.append(tuple(row+[tid]))
        groups.append(group)
    require(len({g[0] for g in groups})==32, 'unique inputs')
    sequences=[x for group in groups for x in group]
    internal=[counts(groups[i],groups[j]) for i in range(len(groups)) for j in range(i+1,len(groups))]
    require(not any(any(x.values()) for x in internal), 'candidate inputs overlap')
    old_bytes = git(SUPPORT_REPO,'8533660206f675b437ad4ea4fb4cef2116956ce1','experiments/natural_agreement_original_public_support_v1.json')
    next_bytes = git(SUPPORT_REPO,'afdbc15317983a3948116c28eb857c5ddb869783','experiments/natural_agreement_successor_inputs_v1.json')
    require(digest(old_bytes)==OLD_SHA and digest(next_bytes)==NEXT_SHA, 'public support identity')
    old=[tuple(x) for x in json.loads(old_bytes)['rows']]
    nxt=[tuple(c['input_ids']) for c in json.loads(next_bytes)['cases']]
    require(len(old)==128 and len(nxt)==256, 'public support counts')
    source_path=REPO/'looking_glass/pretrained_exposures_v35.py'
    source_bytes=source_path.read_bytes()
    require(source_bytes==git(REPO,INVENTORY_COMMIT,'looking_glass/pretrained_exposures_v35.py'), 'inventory source bytes')
    sys.path.insert(0,str(REPO))
    from looking_glass import pretrained_exposures_v35 as inventory
    payload=inventory.inventory_payload()
    canonical=json.dumps(payload,ensure_ascii=True,sort_keys=True,separators=(',', ':')).encode('ascii')
    require(digest(canonical)==INVENTORY_SHA and len(payload['records'])==1055, 'inventory payload')
    known=[]
    for record in payload['records']:
        row=tuple(record['token_ids'])
        known.append(row)
        if type(record.get('target_token_id')) is int:
            known.append(row+(record['target_token_id'],))
        known.extend(row+(x,) for x in record.get('candidate_token_ids', ()))
        known.extend(row+(x,) for group in record.get('candidate_token_groups',{}).values() for x in group)
    results={'inventory_v35':counts(sequences,known),'old_public_support':counts(sequences,old),'successor_public_pool':counts(sequences,nxt)}
    require(CASE.read_bytes()==case_bytes, 'candidate changed during observation')
    report={'schema':'cross-model-gpt-public-comparison-v1', 'case_sha256':CASE_SHA,
            'tokenizer_sha256':TOKEN_SHA, 'model_revision':MODEL_REV,
            'v35_commit':INVENTORY_COMMIT,'v35_source_sha256':digest(source_bytes),
            'v35_canonical_sha256':INVENTORY_SHA,'v35_records':len(payload['records']),
            'v35_input_and_extensions':len(known), 'old_public_support_sha256':OLD_SHA,
            'old_public_support_rows':len(old), 'successor_public_pool_sha256':NEXT_SHA,
            'successor_public_pool_rows':len(nxt), 'cases':len(cases),
            'inputs_plus_continuations':len(sequences), 'candidate_internal_relations':0,
            'prefix_counts':results,
            'scope':'Public sources only; original support covers sealed preparation conditionally. No private rows, model, or complete exposure history were checked.'}
    output=OUTPUT
    data=(json.dumps(report,sort_keys=True,separators=(',', ':'),ensure_ascii=True)+'\n').encode('ascii')
    output.write_bytes(data)
    print(json.dumps(report,sort_keys=True))
    print('receipt_sha256',digest(data))
    require(not any(any(x.values()) for x in results.values()), 'public prefix collision')


if __name__=='__main__':
    main()
