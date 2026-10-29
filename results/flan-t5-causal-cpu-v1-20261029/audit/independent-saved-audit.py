"""Independently replay saved T5 feasibility evidence; never execute a model."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_flush_denormal(False)
root = Path(sys.argv[1])
case_path = Path(sys.argv[2])
cases_raw = case_path.read_bytes()
cases = json.loads(cases_raw)
complete = json.loads((root / "completed.json").read_bytes())
started = json.loads((root / "started.json").read_bytes())

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def bits_equal(a, b):
    return a.shape == b.shape and np.array_equal(a.view("<u4"), b.view("<u4"))

def l2(a, b):
    delta = a.astype(np.float64).reshape(-1) - b.astype(np.float64).reshape(-1)
    return math.sqrt(math.fsum(float(value) ** 2 for value in delta))

def near(a, b, tol=1e-12):
    return math.isclose(a, b, abs_tol=tol, rel_tol=0.0)

assert complete["schema"] == "mlg-flant5-causal-feasibility-v1"
assert complete["status"] == "complete" and 0 <= complete["forwards"] <= 76
assert complete["case_manifest_sha256"] == started["case_manifest_sha256"] == sha(cases_raw)
assert complete["asset_manifest_sha256"] == started["asset_manifest_sha256"]
assert complete["source"] == started["source"]
assert complete["revision"] == cases["revision"] == "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
assert complete["model"] == cases["model"] == "google/flan-t5-small"
assert len(cases["pairs"]) == len(complete["pairs"]) == 2
assert not (root / "failure.json").exists()
assert len(list(root.glob("forward-*.bin"))) == len(list(root.glob("forward-*.json"))) == complete["forwards"]
assert sum(p.stat().st_size for p in root.iterdir() if p.is_file()) <= 128 * 1024 * 1024

by_name = {}
for index in range(complete["forwards"]):
    row = json.loads((root / f"forward-{index:03d}.json").read_bytes())
    assert row["index"] == index and row["file"] == f"forward-{index:03d}.bin"
    assert row["name"] not in by_name and row["dtype"] == "little-endian-float32"
    raw = (root / row["file"]).read_bytes()
    assert sha(raw) == row["sha256"] and len(raw) == row["logits_bytes"] + row["encoder_bytes"]
    assert row["logits_shape"] == [1, len(row["target_ids"]), 32128]
    assert row["encoder_shape"] == [1, len(row["source_ids"]), 512]
    assert row["logits_bytes"] == math.prod(row["logits_shape"]) * 4
    assert row["encoder_bytes"] == math.prod(row["encoder_shape"]) * 4
    assert row["target_ids"][-1] == 1 and row["decoder_ids"] == [0] + row["target_ids"][:-1]
    assert row["decoder_mask"] == [1] * len(row["decoder_ids"])
    logits = np.frombuffer(raw[:row["logits_bytes"]], dtype="<f4").copy().reshape(row["logits_shape"])
    state = np.frombuffer(raw[row["logits_bytes"]:], dtype="<f4").copy().reshape(row["encoder_shape"])
    assert np.isfinite(logits).all() and np.isfinite(state).all()
    log_probs = torch.log_softmax(torch.from_numpy(logits[0]), dim=-1)
    selected = [float(log_probs[i, token]) for i, token in enumerate(row["target_ids"])]
    assert selected == row["selected_logprobs"] and math.fsum(selected) == row["score"]
    by_name[row["name"]] = {"row": row, "logits": logits, "state": state}

def record(name, source, target):
    entry = by_name[name]
    row = entry["row"]
    assert row["source_ids"] == source["input_ids"] and row["source_mask"] == source["attention_mask"]
    assert row["target_ids"] == target["input_ids"]
    return entry

outcomes = []
for pair_index, (pair, saved) in enumerate(zip(cases["pairs"], complete["pairs"])):
    assert saved["pair_index"] == pair_index
    assert json.loads((root / f"pair-{pair_index}-decision.json").read_bytes()) == saved
    left, right = pair["sources"]
    eligible = (len(left["input_ids"]) == len(right["input_ids"]) <= 32
                and left["attention_mask"] == right["attention_mask"]
                and [i for i, (x, y) in enumerate(zip(left["input_ids"], right["input_ids"])) if x != y] == [6]
                and all(s["word_position"] == 6 and not s["special_tokens_mask"][0]
                        and not s["special_tokens_mask"][6]
                        and s["text"][s["offsets"][6][0]:s["offsets"][6][1]].strip() == s["text"].split()[-1]
                        for s in (left, right))
                and all(len(t["input_ids"]) <= 8 for t in pair["targets"]))
    if saved["status"] == "ineligible":
        assert not eligible
        assert not any(name.startswith(f"pair{pair_index}-") for name in by_name)
        outcomes.append(saved)
        continue
    assert eligible
    context = [record(f"pair{pair_index}-context-target{t}", cases["neutral"], pair["targets"][t])
               for t in range(2)]
    base = {(s, t): record(f"pair{pair_index}-base-{s}-{t}", pair["sources"][s], pair["targets"][t])
            for s in range(2) for t in range(2)}
    assert all(item["row"]["condition"] == "clean" for item in context + list(base.values()))
    assert all(bits_equal(base[s, 0]["state"], base[s, 1]["state"]) for s in range(2))
    margins = [math.fsum((base[s, 0]["row"]["score"], -base[s, 1]["row"]["score"])) for s in range(2)]
    delta = margins[0] - margins[1]
    context_scores = [x["row"]["score"] for x in context]
    context_margin = context_scores[0] - context_scores[1]
    supported = margins[0] > 0 and margins[1] < 0 and delta >= 1.0
    assert saved["baseline_margins"] == margins and saved["delta"] == delta
    assert saved["context_only_scores"] == context_scores and saved["context_only_margin"] == context_margin
    assert saved["context_adjusted_margins"] == [x - context_margin for x in margins]
    assert saved["baseline_supported"] is supported
    if not supported:
        assert saved["status"] == "baseline-unsupported"
        assert len([n for n in by_name if n.startswith(f"pair{pair_index}-")]) == 6
        outcomes.append(saved)
        continue
    conditions = ("repeat", "null", "token_self", "full_self", "full_swap",
                  "word_swap", "zero", "instruction_swap")
    controls = {}
    for condition in conditions:
        for side in range(2):
            own, donor = base[side, 0]["state"], base[1 - side, 0]["state"]
            position = 6 if condition in ("null", "token_self", "word_swap", "zero") else \
                       0 if condition == "instruction_swap" else None
            for target_index in range(2):
                run = record(f"pair{pair_index}-{condition}-{side}-{target_index}",
                             pair["sources"][side], pair["targets"][target_index])
                controls[condition, side, target_index] = run
                assert run["row"]["token_index"] == position
                action = "capture" if condition == "repeat" else "null" if condition == "null" else \
                         "zero" if condition == "zero" else "patch"
                assert run["row"]["condition"] == action
                if condition in ("repeat", "null", "token_self", "full_self"):
                    assert bits_equal(run["logits"], base[side, target_index]["logits"])
                if condition == "full_swap":
                    assert bits_equal(run["logits"], base[1 - side, target_index]["logits"])
                    assert bits_equal(run["state"], donor)
                elif condition in ("repeat", "full_self"):
                    assert bits_equal(run["state"], own)
                if position is not None:
                    expected = np.zeros_like(own[0, position]) if condition == "zero" else \
                               donor[0, position] if condition in ("word_swap", "instruction_swap") else own[0, position]
                    assert bits_equal(run["state"][0, position], expected)
                    assert all(bits_equal(run["state"][0, i], own[0, i]) for i in range(own.shape[1]) if i != position)
            if condition != "repeat":
                metadata = saved["patch_metadata"][f"{condition}-{side}"]
                assert metadata["destination_side"] == side and metadata["destination_position"] == position
                donor_side = None if condition == "zero" else \
                             1 - side if condition in ("full_swap", "word_swap", "instruction_swap") else side
                assert metadata["donor_side"] == donor_side
                assert metadata["donor_position"] == (None if donor_side is None else position)
                assert metadata["destination_special"] == (None if position is None else pair["sources"][side]["special_tokens_mask"][position])
                assert metadata["donor_special"] == (None if position is None or donor_side is None else pair["sources"][donor_side]["special_tokens_mask"][position])
                if position is None:
                    expected_distance = l2(own, donor if condition == "full_swap" else own)
                else:
                    expected_distance = l2(own[0, position], expected)
                assert near(saved["patch_distances_l2"][f"{condition}-{side}"], expected_distance)
                assert near(metadata["l2"], expected_distance)
    condition_margins = {c: [math.fsum((controls[c, s, 0]["row"]["score"],
                                        -controls[c, s, 1]["row"]["score"])) for s in range(2)]
                         for c in conditions}
    assert saved["condition_margins"] == condition_margins
    word = [margins[0] - condition_margins["word_swap"][0],
            condition_margins["word_swap"][1] - margins[1]]
    instruction = [margins[0] - condition_margins["instruction_swap"][0],
                   condition_margins["instruction_swap"][1] - margins[1]]
    positive = all(word[i] >= 0.20 * delta and word[i] >= instruction[i] + 0.05 * delta for i in range(2))
    assert saved["word_movements"] == word and saved["instruction_movements"] == instruction
    assert saved["status"] == ("positive" if positive else "negative")
    assert len([n for n in by_name if n.startswith(f"pair{pair_index}-")]) == 38
    outcomes.append(saved)

assert len(by_name) == complete["forwards"]
print(json.dumps({"status": "pass", "forwards": len(by_name), "pairs": outcomes,
                  "scope": "saved-evidence consistency only; no model execution"}, indent=2, allow_nan=False))
