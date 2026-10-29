"""Model-free checks of the public invariant-successor reservation."""
import hashlib,json,os
from pathlib import Path
import pytest
from looking_glass import pretrained_exposures_v43 as candidate


def predecessor():
    fixture=os.environ.get('INVARIANT_V42_CANONICAL')
    if fixture:
        raw=Path(fixture).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=candidate.PREDECESSOR_SHA256: raise ValueError('fixture digest differs')
        return json.loads(raw)
    return candidate.prior.inventory_payload()


def test_append_only_and_public_case_mapping():
    before=predecessor(); after=candidate._build_payload(before)
    assert hashlib.sha256(candidate._canonical(after)).hexdigest()==candidate.INVENTORY_SHA256
    assert after['records'][:len(before['records'])]==before['records']
    assert len(after['records'])==len(before['records'])+34
    added=after['records'][len(before['records']):]
    assert [(x['record_id'],x['case_role'],x['public_text'],tuple(x['token_ids'])) for x in added]==[('planned-'+i,r,t,row) for i,r,t,row in candidate.INVARIANT_SUCCESSOR_CASES]
    assert all(x['category']=='planned' and x['candidate_token_ids']==list(candidate.INVARIANT_SUCCESSOR_ANSWER_IDS) for x in added)
    assert len(candidate._prefixes(after['records']))==candidate.FORBIDDEN_PREFIX_COUNT
    old_eval=[x for x in before['records'] if x['record_id'].startswith('planned-evaluation_')]
    assert len(old_eval)==34 and all(x['category']=='planned' for x in old_eval)


def test_known_input_collision_is_rejected(monkeypatch):
    before=predecessor(); changed=list(candidate.INVARIANT_SUCCESSOR_CASES)
    identity,role,text,_=changed[0]; changed[0]=(identity,role,text,tuple(before['records'][0]['token_ids']))
    monkeypatch.setattr(candidate,'INVARIANT_SUCCESSOR_CASES',tuple(changed))
    with pytest.raises(ValueError): candidate._build_payload(before)
