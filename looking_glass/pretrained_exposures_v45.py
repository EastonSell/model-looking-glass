"""Reserve 64 public paired-continuation prompts; no model forward is asserted."""
import copy
import hashlib
import json
from pathlib import Path

from . import pretrained_exposures_v41 as public_sequences
from . import pretrained_exposures_v44 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v45'
PREDECESSOR_SHA256 = '78d783dc37c1ace4c39300b62e67652d76f3323de79830628ecd64e46aa0bfa0'
INVENTORY_SHA256 = '8c0e09c2a1bf276e931f3cd7580cbb5076ad696b3371002307cf8cc76e4e4b37'
FORBIDDEN_PREFIX_COUNT = 9629
CASE_COMMIT = '09a85fddb97f4bad088c206d08fb1707ccae9a58'
CASE_PATH = 'experiments/distilgpt2_paired_continuation_cases_v1.json'
CASE_SHA256 = '68b567718209285a9adedd588d0c0d269efb4b7d8ed9d278a2fb8201d672d1dd'
REFERENCES = ('docs/DISTILGPT2_PAIRED_CONTINUATION_PATCH_V1.md',
              'docs/DISTILGPT2_PAIRED_CONTINUATION_POOL_AMENDMENT_V1.md', CASE_PATH)
for _name in dir(prior):
    if _name.isupper() and _name not in globals():
        globals()[_name] = getattr(prior, _name)
_canonical, _prefixes = prior._canonical, prior._prefixes


def _related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def _build_payload(predecessor):
    payload=copy.deepcopy(predecessor)
    records=payload['records']
    old_ids={r['record_id'] for r in records}
    if len(old_ids)!=len(records):
        raise ValueError('predecessor record identities repeat')
    raw=(Path(__file__).resolve().parents[1]/CASE_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CASE_SHA256:
        raise ValueError('fixed cases differ')
    case_data=json.loads(raw); cases=case_data['cases']
    if (case_data.get('schema')!='distilgpt2-paired-continuation-cases-v1'
            or case_data.get('model_forward_asserted') is not False
            or len(cases)!=64 or len({c['case_id'] for c in cases})!=64):
        raise ValueError('fixed case bundle identity differs')
    groups=[]
    for index,case in enumerate(cases):
        group, variant=index//4+1,index%4
        row=tuple(case['token_ids']); f,g=case['candidate_f_id'],case['candidate_g_id']
        identity='planned-'+case['case_id']
        if (identity in old_ids or case['case_id']!=f'copy-paired-{group:02d}-{variant}'
                or case['group']!=group or case['variant']!=variant or len(row)!=12
                or case['correct_id'] != (f if variant in (0,2) else g)
                or type(case['text']) is not str or not case['text']
                or f==g or any(type(x) is not int or not 0<=x<50257 for x in (*row,f,g))):
            raise ValueError('fixed case identity, prompt or candidates differ')
        groups.append((row,row+(f,),row+(g,)))
    old=public_sequences._public_sequences(records)
    if any(_related(row,previous) for group in groups for row in group for previous in old):
        raise ValueError('paired prompt or score extension has a public prefix relation')
    if any(_related(left,right) for i,group in enumerate(groups) for other in groups[i+1:]
           for left in group for right in other):
        raise ValueError('distinct paired prompts share a prefix relation')
    for case in cases:
        records.append({'record_id':'planned-'+case['case_id'],'category':'planned',
                        'token_ids':case['token_ids'],
                        'candidate_token_ids':[case['candidate_f_id'],case['candidate_g_id']],
                        'public_text':case['text'],'reference_commit':CASE_COMMIT,
                        'references':list(REFERENCES),
                        'provenance':'Public paired-continuation proposal; no model forward is asserted. '
                                     f'Case SHA-256 {CASE_SHA256}.'})
    payload['version']=INVENTORY_VERSION
    payload['predecessor']={'version':prior.INVENTORY_VERSION,'sha256':PREDECESSOR_SHA256}
    payload['update']={'planned_paired_continuation_rows':64}
    if FORBIDDEN_PREFIX_COUNT and len(_prefixes(records))!=FORBIDDEN_PREFIX_COUNT:
        raise ValueError('paired-continuation prefix count differs')
    return payload


def inventory_payload():
    predecessor=prior.inventory_payload()
    if (prior.INVENTORY_SHA256!=PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest()!=PREDECESSOR_SHA256):
        raise ValueError('predecessor inventory differs')
    return _build_payload(predecessor)


def canonical_inventory_bytes():
    return _canonical(inventory_payload())


def _checked_inventory():
    payload=inventory_payload()
    digest=hashlib.sha256(_canonical(payload)).hexdigest()
    if digest!=INVENTORY_SHA256:
        raise ValueError('paired inventory digest differs')
    return payload,digest


def inventory_digest():
    return _checked_inventory()[1]
