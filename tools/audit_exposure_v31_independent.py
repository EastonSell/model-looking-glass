"""Stdlib-only audit of proposed public v31 reservation against v30 bytes."""
import ast
import hashlib
import itertools
import json
from pathlib import Path


def read(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Unexpected digest for {path}")
    return raw


old_sha="921025a7f9c1246c5f10e97f2b987ca4971462241b2d05ebbc525d4aa0736c79"
new_sha="2d710a40a49a84b7d664cd851fccd2527ea318c5a8c911111fe7275b7567e565"
case_sha="b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69"
source_sha="729b2aa47d945b01e6aaf06ec2ef8e017aae6635a0c3f0237672a8272d8a4a68"
old=json.loads(read("work/year-label-v30-draft/inventory.canonical.json",old_sha))
new=json.loads(read("work/reviews/copy-prompt-v31-canonical-v2.json",new_sha))
earlier=json.loads(read("work/reviews/copy-prompt-v31-canonical.json","0e65cc8e3f3d1207cc5c903d928eba2d63065f45aea24c48fe964ad5a2757269"))
assert {k:v for k,v in new.items() if k!="records"}=={k:v for k,v in earlier.items() if k!="records"}
assert new["records"][:947] == earlier["records"][:947]
cases=json.loads(read("work/copy-head-prompt-dose-cases-proposal-v1.json",case_sha))["cases"]
source=read("work/teams/copy_head_prompt_dose_v1/looking_glass/pretrained_exposures_v31.py",source_sha)
assert len(old["records"])==915 and len(new["records"])==949 and len(cases)==32
assert new["records"][:915] == old["records"]
assert {k:v for k,v in new.items() if k not in ("version","predecessor","update","records")} == {k:v for k,v in old.items() if k not in ("version","predecessor","update","records")}
assert new["version"]=="distilgpt2-development-exposures-v31"
assert new["predecessor"]=={"version":"distilgpt2-development-exposures-v30","sha256":old_sha}
assert new["update"]=={"planned_copy_prompt_dose_rows":32,"planned_service_synthetic_rows":2,"model_forward_asserted":False}
assert new_sha == hashlib.sha256(json.dumps(new,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

module=ast.parse(source)
assignment=next(n for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="COPY_PROMPT_DOSE_CASES" for t in n.targets))
literal=ast.literal_eval(assignment.value)
assert [(i,t,list(row),target) for i,t,row,target in literal]==[(c["case_id"],c["text"],c["token_ids"],c["target_id"]) for c in cases]
for case,record in zip(cases,new["records"][915:947]):
    expected={"record_id":"planned-"+case["case_id"],"category":"planned","token_ids":case["token_ids"],
              "target_token_id":case["target_id"],"public_text":case["text"],
              "provenance":"Public per-prompt matched-dose copying input; no model forward asserted. Case SHA-256 "+case_sha+".",
              "references":["docs/DISTILGPT2_COPY_HEAD_PROMPT_DOSE_V1.md","experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json"],
              "reference_commit":"115d9bb42aadc481482de302d2df11c9c2d30789"}
    assert record==expected
for record,(name,tokens) in zip(new["records"][947:],(("planned-service-synthetic-mask-16",[50256,*range(1001,1016)]),
                                                     ("planned-service-synthetic-mask-32",[50256,*range(2001,2032)]))):
    assert record=={"record_id":name,"category":"planned","token_ids":tokens,"public_text":"",
                    "attention_mask":[1]*len(tokens),"provenance":"Public synthetic all-one-mask CPU service input; no model forward asserted. Public-scope receipt SHA-256 9ff6f36cb71960398a1113e5731e224d83e5483748cdd307ae719a6dc9dbbe39. Case SHA-256 47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c.",
                    "references":["experiments/short_cpu_service_reference_inputs_v1.json","docs/SHORT_SERVICE_SYNTHETIC_PUBLIC_SUPPORT_V1.json"],
                    "reference_commit":"63886558b85212ec59d73f04c78e5e722f1e36f6"}
all_ids=[r["record_id"] for r in new["records"]]
assert len(set(all_ids))==len(all_ids)
old_rows=[]
for r in old["records"]:
    base=tuple(r["token_ids"]);old_rows.append(base)
    if type(r.get("target_token_id")) is int: old_rows.append(base+(r["target_token_id"],))
    old_rows.extend(base+(x,) for x in r.get("candidate_token_ids",()))
    old_rows.extend(base+(x,) for group in r.get("candidate_token_groups",{}).values() for x in group)
added=[tuple(r["token_ids"]) for r in new["records"][915:]]
assert len(set(added))==34
def prefix_related(x,y):
    return x[:len(y)]==y or y[:len(x)]==x
assert not any(prefix_related(a,b) for a in added for b in old_rows)
assert not any(prefix_related(a,b) for a,b in itertools.combinations(added,2))
for file,key,expect in (("natural_agreement_original_public_support_v1.json","rows","5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6"),
                        ("natural_agreement_successor_inputs_v1.json","cases","6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4")):
    data=json.loads(read("work/teams/copy_head_prompt_dose_v1/experiments/"+file,expect))[key]
    rows=[tuple(r if key=="rows" else r["input_ids"]) for r in data]
    assert not any(prefix_related(a,b) for a in added for b in rows)
prefixes={tuple(r["token_ids"][:i]) for r in new["records"] for i in range(1,len(r["token_ids"])+1)}
assert len(prefixes)==5789
print(json.dumps({"status":"public_v31_reservation_consistent","v30_sha256":old_sha,"v31_sha256":new_sha,
                  "source_sha256":source_sha,"old_records_unchanged":915,"copying_rows":32,
                  "service_rows":2,"prefix_count":len(prefixes),"predecessor_input_and_continuation_rows":len(old_rows),
                  "scope":"public pinned predecessor and specified support sets; no execution or protected/unknown exposure attestation"},sort_keys=True))
