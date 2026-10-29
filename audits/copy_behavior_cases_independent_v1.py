"""Independently reconstruct the public copying screen cases; no model load."""
import hashlib
import json
from pathlib import Path
import re
import sys

root=Path(r"E:\Codex\model-looking-glass\work\copy-behavior-screen-v1")
proposal_path=Path(r"E:\Codex\model-looking-glass\work\copy-behavior-cases-proposal-v1.json")
tokenizer_path=Path("work/hf-cache/hub/models--distilbert--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa/tokenizer.json")
v38_path=Path(r"E:\Codex\model-looking-glass\work\public-distractor-localization-v1\inventory-v38.canonical.json")
v39_path=Path(r"E:\Codex\model-looking-glass\work\agreement-head-v2-v39.canonical.json")
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path,expected):
    raw=path.read_bytes();assert digest(raw)==expected,str(path);return raw
def index(rows):
    rows=set(rows);return rows,{r[:i] for r in rows for i in range(1,len(r)+1)}
def related(row,idx):
    rows,prefixes=idx
    return row in prefixes or any(row[:i] in rows for i in range(1,len(row)+1))
def public_sequences(records):
    rows=[]
    for record in records:
        base=tuple(record["token_ids"]);choices=[]
        if type(record.get("target_token_id")) is int:choices.append(record["target_token_id"])
        choices.extend(record.get("candidate_token_ids",()))
        for group in record.get("candidate_token_groups",{}).values():choices.extend(group)
        bases=[base]
        if "attention_mask" in record:
            mask=record["attention_mask"]
            assert len(mask)==len(base) and all(type(x) is int and x in (0,1) for x in mask)
            active=tuple(t for t,m in zip(base,mask) if m)
            if active and active!=base:bases.append(active)
        for b in bases:
            rows.append(b);rows.extend(b+(token,) for token in choices)
        if "scored_prefix_token_ids" in record:
            prefix=tuple(record["scored_prefix_token_ids"]);rows.append(prefix)
            rows.extend(prefix+(token,) for token in record.get("scored_candidate_token_ids",()))
    return rows

protocol=read(root/"docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_V1.md","78907b74bf3edb44a47c794508a9385da23b2bf2f26d07df4fefe74542b58875")
proposal_raw=read(proposal_path,"96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734")
proposal=json.loads(proposal_raw)
read(tokenizer_path,"8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6")
read(root/"looking_glass/pretrained_exposures_v38.py","91da6a6f7f6272136157226b6e859f6e3ee8acc44e58d390774f898358cafe70")
read(root/"looking_glass/pretrained_exposures_v39.py","889624dc5b93e44968c6affc9bbde37868fbc3928f2d762fa81a3db39e57d31f")
v38=json.loads(read(v38_path,"f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9"))
v39=json.loads(read(v39_path,"5e593d1b5a0ac35aba2791cd5eafe7114f7e93ba4bb1c90706e385d2fab0cadc"))
assert len(v38["records"])==1167 and len(v39["records"])==1183
assert v39["records"][:1167]==v38["records"]
support=json.loads(read(root/"experiments/natural_agreement_original_public_support_v1.json","5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6"))
successor=json.loads(read(root/"experiments/natural_agreement_successor_inputs_v1.json","6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4"))
assert len(support["rows"])==128 and len(successor["cases"])==256
known=public_sequences(v38["records"])+[tuple(x) for x in support["rows"]]+[tuple(x["input_ids"]) for x in successor["cases"]]
known38=index(known)
known39=index(public_sequences(v39["records"])+[tuple(x) for x in support["rows"]]+[tuple(x["input_ids"]) for x in successor["cases"]])
sys.path.insert(0,str(root))
from tokenizers import Tokenizer
from looking_glass import pretrained_exposures_v6 as v6,pretrained_exposures_v7 as v7
from looking_glass import pretrained_exposures_v9 as v9,pretrained_exposures_v12 as v12,pretrained_exposures_v15 as v15
prior={t for groups in (v6.PUBLIC_COPY_GROUPS,v7.PUBLIC_EVALUATION_GROUPS,v9.FOLLOWUP_GROUPS,v12.FACTORIAL_GROUPS,v15.PAIRS_DOSE_GROUPS) for g in groups for t in g}
for path,sha in (("experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json","b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69"),
                 ("experiments/distilgpt2_copy_output_dose_cases_v1.json","e45fe8cfe8b28a4f4308a4d3b2206f35cebe8154c7b4301ac1356c28258b135e")):
    data=json.loads(read(root/path,sha));prior.update(t for group in data["groups"] for t in group["token_ids"])
assert proposal["schema"]=="distilgpt2-copy-behavior-screen-cases-v1" and proposal["model_forward_asserted"] is False
assert proposal["protocol_sha256"]==digest(protocol) and proposal["selection_inventory_sha256"]=="f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9"
assert proposal["selection_inventory_record_count"]==1167 and proposal["public_support_rows"]==128 and proposal["public_successor_rows"]==256
pools=[m.group(1).split() for m in re.finditer(r"^\d+\. ([a-z ]+)$",protocol.decode(),re.M)]
assert len(pools)==8 and pools==proposal["candidate_word_pools"]
perms=((0,1,2,3,4,5),(4,2,0,3,1,5),(1,3,4,2,0,5),(3,0,4,1,2,5))
assert proposal["permutations"]==[list(p) for p in perms]
tokenizer=Tokenizer.from_file(str(tokenizer_path));selected=set();previous=[];word_entries=0;v39_relations=[]
for pool_index,words in enumerate(pools):
    for within in range(3):
        group=3*pool_index+within+1;seen=set();allowed=[];log=[]
        for pos,word in enumerate(words):
            ids=tokenizer.encode(" "+word,add_special_tokens=False).ids
            reason=("repeated-word" if word in seen else "not-single-token" if len(ids)!=1 else
                    "prior-copy-token" if ids[0] in prior else "earlier-selected-token" if ids[0] in selected else None)
            seen.add(word);log.append({"position":pos,"word":word,"token_ids":ids,"exclusion":reason})
            if reason is None:allowed.append((pos,word,ids[0]))
        a=proposal["selection_audit"][group-1]
        assert a["group"]==group and a["word_candidates"]==log and not a["rejected_combinations"] and len(allowed)>=6
        first=allowed[:6];positions=[x[0] for x in first];group_words=[x[1] for x in first];ids=[x[2] for x in first]
        assert a["chosen_positions"]==positions and len(set(ids))==6 and not (set(ids)&selected)
        split="development" if group%2 else "held-out"
        assert proposal["groups"][group-1]=={"group":group,"pool":pool_index+1,"split":split,"words":group_words,"token_ids":ids}
        pending=[]
        for rotation in range(2):
            order=tuple(range(6)) if rotation==0 else (1,2,3,4,5,0)
            for shuffle,permutation in enumerate(perms):
                sequence=tuple(order[p] for p in permutation)+order[:5]
                text="".join(" "+group_words[x] for x in sequence);row=tuple(ids[x] for x in sequence);target=ids[order[5]]
                ext=row+(target,)
                assert tokenizer.encode(text,add_special_tokens=False).ids==list(row)
                assert tokenizer.encode(text+" "+group_words[order[5]],add_special_tokens=False).ids==list(ext)
                assert not related(row,known38) and not related(ext,known38)
                assert not related(row,index(previous)) and not related(ext,index(previous))
                for before,after in pending:
                    assert not any(related(x,index((y,))) for x in (row,ext) for y in (before,after))
                if related(row,known39) or related(ext,known39):v39_relations.append((group,rotation,shuffle))
                variant=4*rotation+shuffle
                expected={"case_id":f"copy-behavior-{group:02d}-{variant}","group":group,"pool":pool_index+1,
                          "split":split,"variant":variant,"rotation":rotation,"shuffle":shuffle,
                          "text":text,"token_ids":list(row),"target_id":target}
                assert proposal["cases"][8*(group-1)+variant]==expected
                pending.append((row,ext))
        selected.update(ids);previous.extend(x for pair in pending for x in pair);word_entries+=len(log)
assert len(selected)==144 and len(previous)==384 and not v39_relations
assert len(proposal["groups"])==len(proposal["selection_audit"])==24 and len(proposal["cases"])==192
assert not any(x=="torch" or x.startswith("torch.") or x=="transformers" or x.startswith("transformers.") for x in sys.modules)
print(json.dumps({"status":"pass","groups":24,"cases":192,"distinct_token_ids":144,"candidate_log_entries":word_entries,
    "earlier_rejected_combinations":0,"v38_record_count":1167,"v39_record_count":1183,"v39_preserves_v38":True,
    "v38_public_sequence_count":len(known38[0]),"v39_public_sequence_count":len(known39[0]),"v39_related_cases":len(v39_relations),
    "proposal_sha256":digest(proposal_raw),"v38_sha256":digest(v38_path.read_bytes()),"v39_sha256":digest(v39_path.read_bytes()),
    "scope":"Tokenizer-only known public inputs, masks and continuations; no model, protected pool or unknown-pretraining attestation."},sort_keys=True))
