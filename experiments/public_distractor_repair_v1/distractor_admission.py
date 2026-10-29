"""Verify fixed planned/reference rows in the public v38 reservation."""
import hashlib
import json
from pathlib import Path

DIGEST='f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9'

def related(a,b):return a[:min(len(a),len(b))]==b[:min(len(a),len(b))]

def verify(path,cases):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DIGEST:raise ValueError('inventory digest differs')
    inventory=json.loads(raw);records=inventory['records']
    if inventory.get('version')!='distilgpt2-development-exposures-v38' or len(cases)!=16:
        raise ValueError('reservation version or case count differs')
    by_id={r['record_id']:r for r in records}
    if len(by_id)!=len(records):raise ValueError('duplicate inventory ID')
    own=set();proposals=[]
    for case in cases:
        cid=case['case_id'];full=case['token_ids'];prefix=case['scored_prefix_token_ids']
        if full!=prefix+[13] or case['candidate_token_ids']!=[318,389]:
            raise ValueError('fixed scored-prefix metadata differs')
        record_id='planned-'+cid;record=by_id.get(record_id,{})
        if (record.get('category')!='planned' or record.get('token_ids')!=full
            or record.get('public_text')!=case['public_text']
            or record.get('scored_prefix_token_ids')!=prefix
            or record.get('scored_candidate_token_ids')!=[318,389]):
            raise ValueError('own planned row differs')
        own.add(record_id);proposals.extend((tuple(full),tuple(prefix)))
        for token in (318,389):
            rid=f'reference-{cid}-{token}';record=by_id.get(rid,{})
            if record.get('category')!='reference-only' or record.get('token_ids')!=prefix+[token]:
                raise ValueError('own reference alternative differs')
            own.add(rid);proposals.append(tuple(prefix+[token]))
    if len(own)!=48:raise ValueError('duplicate own row')
    foreign=[]
    for record in records:
        if record['record_id'] in own:continue
        base=tuple(record['token_ids']);bases=[base]
        if 'attention_mask' in record:
            mask=record['attention_mask']
            if len(mask)!=len(base) or any(x not in (0,1) for x in mask):raise ValueError('invalid mask')
            active=tuple(x for x,m in zip(base,mask) if m)
            if active and active!=base:bases.append(active)
        continuations=list(record.get('candidate_token_ids',[]))
        if 'target_token_id' in record:continuations.append(record['target_token_id'])
        for group in record.get('candidate_token_groups',{}).values():continuations.extend(group)
        if any(type(x) is not int or x<0 for x in continuations):raise ValueError('invalid continuation')
        for b in bases:foreign.extend((b,*(b+(token,) for token in continuations)))
    if any(related(a,b) for a in proposals for b in foreign):
        raise ValueError('foreign public prefix relation')
    return {'reservation_sha256':DIGEST,'own_planned':16,'own_reference':32,
            'proposal_sequences':len(proposals),'foreign_sequences':len(foreign),
            'bidirectional_pairs':len(proposals)*len(foreign),'conflicts':0,
            'scope':'known public inventory only; protected and unknown exposure unchecked'}
