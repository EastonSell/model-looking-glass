"""Tokenizer-only independent reconstruction of the proposed copying cases."""
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

root = Path("work/teams/copy_head_prompt_dose_v1").resolve()
proposal_path = Path("work/copy-head-prompt-dose-cases-proposal-v1.json")
tokenizer_path = Path("work/hf-cache/hub/models--distilbert--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa/tokenizer.json")
protocol = root / "docs/DISTILGPT2_COPY_HEAD_PROMPT_DOSE_V1.md"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


assert sha(proposal_path.read_bytes()) == "b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69"
assert sha(tokenizer_path.read_bytes()) == "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
assert sha(protocol.read_bytes()) == "40d1aac13f0faa82c048db033f8a4152d8c6efb564274afa829b58042bac5d6d"
proposal = json.loads(proposal_path.read_bytes())
pools = [m.group(2).split() for m in re.finditer(r"^(\d+)\. `([^`]+)`$", protocol.read_text(encoding="utf-8"), re.M)]
assert len(pools) == 8 and pools == proposal["candidate_word_pools"]

sys.path.insert(0, str(root))
from tokenizers import Tokenizer
from looking_glass import pretrained_exposures_v6 as v6, pretrained_exposures_v7 as v7
from looking_glass import pretrained_exposures_v9 as v9, pretrained_exposures_v12 as v12
from looking_glass import pretrained_exposures_v15 as v15, pretrained_exposures_v29 as v29

payload, reported_digest = v29._checked_inventory()
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
inventory_digest = sha(canonical)
assert inventory_digest == reported_digest == proposal["selection_inventory_sha256"] == "f2dc3ff6cc8befc600cc62de68c9c804f8962c7261b87cad3c94ea7744bb7df6"
assert proposal["selection_inventory_version"] == "distilgpt2-development-exposures-v29"
prior = {value for groups in (v6.PUBLIC_COPY_GROUPS, v7.PUBLIC_EVALUATION_GROUPS, v9.FOLLOWUP_GROUPS,
                               v12.FACTORIAL_GROUPS, v15.PAIRS_DOSE_GROUPS)
         for group in groups for value in group}
known = {tuple(record["token_ids"]) for record in payload["records"]}
assert len(payload["records"]) == proposal["selection_inventory_record_count"] == 915


def prefix_index(rows):
    full = set(rows)
    prefixes = {row[:i] for row in full for i in range(1, len(row) + 1)}
    return full, prefixes


def intersects(row, index):
    full, prefixes = index
    return row in prefixes or any(row[:i] in full for i in range(1, len(row) + 1))


tokenizer = Tokenizer.from_file(str(tokenizer_path))
perms = ((0,1,2,3,4,5),(4,2,0,3,1,5),(1,3,4,2,0,5),(3,0,4,1,2,5))
assert proposal["permutations"] == [list(p) for p in perms]
old_ids, old_inputs, reconstructed = set(), [], []
all_candidates = all_rejected = 0
for gi, words in enumerate(pools):
    audit = proposal["selection_audit"][gi]
    allowed, checked_words, encountered = [], [], set()
    for pos, word in enumerate(words):
        ids = tokenizer.encode(" " + word, add_special_tokens=False).ids
        reason = ("repeated-word" if word in encountered else
                  "not-single-token" if len(ids) != 1 else
                  "prior-copy-token" if ids[0] in prior else
                  "earlier-selected-token" if ids[0] in old_ids else None)
        encountered.add(word)
        checked_words.append({"position":pos, "word":word, "token_ids":ids, "exclusion":reason})
        if reason is None:
            allowed.append((pos, word, ids[0]))
    assert checked_words == audit["word_candidates"]
    all_candidates += len(checked_words)
    first, rejected = None, []
    for six in itertools.combinations(allowed, 6):
        positions = [x[0] for x in six]
        ids = [x[2] for x in six]
        candidate = [tuple(ids[i] for i in p + (0,1,2,3,4)) for p in perms]
        reason = ("duplicate-token" if len(set(ids)) < 6 else
                  "within-family-relation" if any(intersects(row, prefix_index(candidate[i+1:])) for i,row in enumerate(candidate[:-1])) else
                  "earlier-family-relation" if any(intersects(row, prefix_index(old_inputs)) for row in candidate) else
                  "known-input-relation" if any(intersects(row, prefix_index(known)) for row in candidate) else None)
        if reason is None:
            first = six, candidate
            break
        rejected.append({"positions":positions,"reason":reason})
    assert first is not None and rejected == audit["rejected_combinations"]
    all_rejected += len(rejected)
    six, selected = first
    assert audit["chosen_positions"] == [x[0] for x in six]
    picked_words, ids = [x[1] for x in six], [x[2] for x in six]
    assert proposal["groups"][gi] == {"words":picked_words,"token_ids":ids}
    for variant,(perm,row) in enumerate(zip(perms,selected)):
        text = "".join(" " + picked_words[i] for i in perm + (0,1,2,3,4))
        assert tokenizer.encode(text, add_special_tokens=False).ids == list(row)
        assert tokenizer.encode(text + " " + picked_words[5], add_special_tokens=False).ids == [*row,ids[5]]
        expected = {"case_id":f"copy-prompt-dose-{gi+1:02d}-{variant}","group":gi+1,"variant":variant,
                    "text":text,"token_ids":list(row),"target_id":ids[5]}
        assert expected == proposal["cases"][gi*4+variant]
        reconstructed.append(row)
    old_ids.update(ids)
    old_inputs.extend(selected)

assert len(old_ids) == 48 and len(reconstructed) == 32
assert not any(intersects(row,prefix_index(known)) for row in reconstructed)
for field,path,expected_count in (
    ("public_support",root/"experiments/natural_agreement_original_public_support_v1.json",128),
    ("public_successor",root/"experiments/natural_agreement_successor_inputs_v1.json",256)):
    raw=path.read_bytes(); assert sha(raw)==proposal[field+"_sha256"]
    data=json.loads(raw)
    rs=([tuple(row) for row in data["rows"]] if field=="public_support" else
        [tuple(row["input_ids"]) for row in data["cases"]])
    assert len(rs)==expected_count==proposal[field+"_rows"]
    assert not any(intersects(row,prefix_index(rs)) for row in reconstructed)
assert proposal["model_forward_asserted"] is False
print(json.dumps({"status":"tokenizer_only_case_reconstruction_pass","groups":8,"cases":len(reconstructed),
                  "unique_selected_ids":len(old_ids),"word_candidates":all_candidates,
                  "prior_combination_rejections":all_rejected,"inventory_records":len(payload["records"]),
                  "public_support_rows":128,"public_successor_rows":256,
                  "case_sha256":sha(proposal_path.read_bytes()),"inventory_sha256":inventory_digest,
                  "scope":"public input-prefix relations at the pinned v29 inventory and the two specified public sets; no model forward, protected-row, unknown or pretraining exposure claim"},sort_keys=True))
