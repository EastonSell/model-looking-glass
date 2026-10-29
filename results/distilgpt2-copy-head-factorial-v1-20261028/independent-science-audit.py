"""Independent saved-byte audit of the public matched-dose factorial; no model execution."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).parent
RUN = ROOT / "run"
ARMS = ("h10", "h2", "h7", "joint")
HEADS = (10, 2, 7)
SIZES = {"logits": 50257 * 4, "cproj_before": 11 * 768 * 4,
         "cproj_after": 11 * 768 * 4, "cproj_output": 11 * 768 * 4}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def near(left, right, tolerance=2e-12):
    return math.isclose(left, right, rel_tol=0, abs_tol=tolerance)


def decoded(raw):
    bits = np.frombuffer(raw, dtype="<u4")
    exponent = ((bits >> 23) & 255).astype(np.int32)
    assert not np.any(exponent == 255)
    fraction = bits & 0x7fffff
    significand = np.where(exponent == 0, fraction, fraction | 0x800000).astype(np.float64)
    values = np.ldexp(significand, np.where(exponent == 0, -149, exponent - 150))
    return np.copysign(values, np.where(bits >> 31, -1.0, 1.0))


def norm(values):
    return math.sqrt(math.fsum(float(value) ** 2 for value in values))


def logp(values, target):
    maximum = float(values.max())
    return (float(values[target]) - maximum) - math.log(
        math.fsum(math.exp(float(value) - maximum) for value in values))


def slice_start(head):
    return (10 * 768 + head * 64) * 4


receipt_raw = (ROOT / "supervisor.json").read_bytes()
result_raw = (RUN / "result.json").read_bytes()
manifest_raw = (RUN / "manifest.json").read_bytes()
journal_raw = (RUN / "attempt.jsonl").read_bytes()
reference_raw = (ROOT / "additive-reference.json").read_bytes()
receipt, result, manifest = map(json.loads, (receipt_raw, result_raw, manifest_raw))
reference = json.loads(reference_raw)
assert sha(receipt_raw) == "35626526739498d468c9e91d1c4f39e3f79603cd9015fc7fbd386ba3917f086d"
assert sha(result_raw) == "7f62ed7a014978f71c09ed2e5973d1ce38f544eff0f2a1652cc9fd75e2fb125b"
assert sha(reference_raw) == "be189d8ac0fae4b84f1971a8bcc0b81980a28942b158fa61c2450a88f18fb30f"
assert receipt["process_exit_code"] == 0 and receipt["interrupted"] is False
assert receipt["failure_file_present"] is False and receipt["stderr_bytes"] == 0
assert not (RUN / "failure.json").exists() and not (ROOT / "stderr.bin").read_bytes()
assert receipt["source_revision"] == manifest["source"]["revision"] == "5e4a82baed62016950a6f0fd0f6487b3f7d3b698"
assert manifest["source"]["tree"] == "1e270b3d87a405d862a7390a621dff799696317d"
assert sha((ROOT / "pre-run.json").read_bytes()) == receipt["pre_run_sha256"]
assert sha(manifest_raw) == receipt["run_sha256"]["manifest.json"] == result["manifest_sha256"]
assert sha(result_raw) == receipt["run_sha256"]["result.json"]
assert sha(journal_raw) == receipt["run_sha256"]["attempt.jsonl"]
assert reference["status"] == "pass" and reference["receipt_sha256"] == sha(receipt_raw)
assert reference["result_sha256"] == sha(result_raw) and reference["manifest_sha256"] == sha(manifest_raw)
assert reference["source_revision"] == manifest["source"]["revision"]
assert result["status"] == "complete"
assert result["submitted_forwards"] == result["model_call_returns_observed"] == result["returned_forwards"] == 224
assert result["raw_payload_bytes"] == 61_250_432 and len(result["cases"]) == 224
assert result["model_state_sha256_before"] == result["model_state_sha256_after"]
assert result["model_buffers_sha256_before"] == result["model_buffers_sha256_after"]
environment = {name: {"present": False, "value": None} for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY")}
assert manifest["cpu_backend_environment_at_torch_import"] == result["runtime"]["cpu_backend_environment"] == environment

events = [json.loads(line) for line in journal_raw.splitlines()]
assert Counter(event["event"] for event in events) == {"runtime_ready": 1, "start": 224,
    "raw_recorded": 224, "complete": 224, "family_dose": 8, "case_diagnostics": 32}
assert [{key: value for key, value in event.items() if key != "event"}
        for event in events if event["event"] == "complete"] == result["cases"]

records = {}
for group in range(8):
    family_records = result["cases"][28 * group:28 * (group + 1)]
    expected = [(variant, action) for variant in range(4) for action in ("clean", "noop")]
    for variant in range(4):
        rotation = (group + variant) % 4
        expected.extend((variant, action) for action in (*ARMS[rotation:], *ARMS[:rotation], "clean_after"))
    assert [(row["variant"], row["action"]) for row in family_records] == expected
    for row in family_records:
        key = (row["case_id"], row["action"])
        assert key not in records and row["group"] == group + 1
        records[key] = row

families = []
top1 = {action: {"repeated": 0, "shuffled": 0} for action in ("clean", *ARMS)}
file_bytes = 0
for group in range(8):
    family = []
    source_rows = []
    for variant in range(4):
        case = manifest["cases"][4 * group + variant]
        case_id = f"copy-factorial-{group + 1:02d}-{variant}"
        assert (case["case_id"], case["group"], case["variant"]) == (case_id, group + 1, variant)
        raw = (RUN / (case_id + ".f32")).read_bytes()
        file_bytes += len(raw)
        assert sha(raw) == receipt["run_sha256"][case_id + ".f32"]
        rotation = (group + variant) % 4
        actions = ("clean", "noop", *ARMS[rotation:], *ARMS[:rotation], "clean_after")
        offset, saved = 0, {}
        for action in actions:
            row = records[(case_id, action)]
            assert row["target_id"] == case["target_id"]
            payload = {}
            for name, info in sorted(row["raw"].items(), key=lambda item: item[1]["offset"]):
                assert info["offset"] == offset and info["bytes"] == SIZES[name]
                chunk = raw[offset:offset + info["bytes"]]
                assert len(chunk) == info["bytes"] and sha(chunk) == info["sha256"]
                payload[name] = chunk
                offset += len(chunk)
            wanted = {"logits"} if action in ("clean", "clean_after") else set(SIZES)
            assert set(payload) == wanted
            logits = decoded(payload["logits"])
            target = case["target_id"]
            score = row["score"]
            probability = logp(logits, target)
            assert near(probability, score["target_log_probability"])
            assert float(logits[target]) == score["target_logit"]
            assert 1 + int(np.count_nonzero(logits > logits[target])) == score["target_rank"]
            assert int(np.argmax(logits)) == score["top_token_id"]
            if action in top1:
                top1[action]["repeated" if variant == 0 else "shuffled"] += int(score["top_token_id"] == target)
            if action not in ("clean", "clean_after"):
                decoded(payload["cproj_before"])
                decoded(payload["cproj_after"])
                decoded(payload["cproj_output"])
            saved[action] = {"payload": payload, "score": probability, "logits": logits, "record": row}
        assert offset == len(raw)
        assert saved["clean"]["payload"]["logits"] == saved["noop"]["payload"]["logits"] == saved["clean_after"]["payload"]["logits"]
        assert saved["noop"]["payload"]["cproj_before"] == saved["noop"]["payload"]["cproj_after"]
        for action in ARMS:
            assert saved[action]["payload"]["cproj_before"] == saved["noop"]["payload"]["cproj_before"]
        source_rows.append(saved["noop"]["payload"]["cproj_before"])
        family.append(saved)

    original = [{head: decoded(row[slice_start(head):slice_start(head) + 256]) for head in HEADS}
                for row in source_rows]
    norms = [{head: norm(row[head]) for head in HEADS} for row in original]
    dose = min(item[head] for item in norms for head in HEADS)
    saved_dose = result["family_doses"][group]
    assert saved_dose["group"] == group + 1 and near(dose, saved_dose["requested_l2"])
    assert dose > 2.0 ** -20
    for variant, row in enumerate(family):
        delivered = {}
        for head in HEADS:
            values = original[variant][head]
            fraction = dose / norms[variant][head]
            expected = struct.pack("<64f", *((1.0 - fraction) * float(x) for x in values))
            delivered[head] = expected
            achieved = norm(decoded(expected) - values)
            before_bits = np.frombuffer(source_rows[variant][slice_start(head):slice_start(head) + 256], dtype="<u4")
            after_bits = np.frombuffer(expected, dtype="<u4")
            component = saved_dose["variants"][variant][str(head)]
            assert near(component["source_l2"], norms[variant][head])
            assert near(component["requested_fraction"], fraction)
            assert near(component["achieved_l2"], achieved)
            assert component["changed_float32_bit_patterns"] == int(np.count_nonzero(before_bits != after_bits))
            assert abs(achieved - dose) <= 1e-5 * dose
        for action, heads in (("h10", (10,)), ("h2", (2,)), ("h7", (7,)), ("joint", (2, 10))):
            expected = bytearray(source_rows[variant])
            intervention = row[action]["record"]["intervention"]
            assert intervention["heads"] == list(heads) and intervention["position"] == 10
            assert near(intervention["requested_l2_each"], dose)
            for head in heads:
                start = slice_start(head)
                expected[start:start + 256] = delivered[head]
                assert intervention["components"][str(head)] == saved_dose["variants"][variant][str(head)]
            assert row[action]["payload"]["cproj_after"] == bytes(expected)
            if action == "joint":
                combined = norm(np.concatenate([decoded(delivered[h]) - original[variant][h] for h in heads]))
                assert near(combined, intervention["joint_combined_achieved_l2"])
        for head in (2, 10):
            start = slice_start(head)
            assert (row["joint"]["payload"]["cproj_after"][start:start + 256]
                    == row[f"h{head}"]["payload"]["cproj_after"][start:start + 256])

        outputs = {name: decoded(row[name]["payload"]["cproj_output"])[10 * 768:11 * 768]
                   for name in ("noop", *ARMS)}
        diagnostic = result["case_diagnostics"][4 * group + variant]
        assert diagnostic["case_id"] == f"copy-factorial-{group + 1:02d}-{variant}"
        for name in ARMS:
            assert near(norm(outputs[name] - outputs["noop"]), diagnostic["final_output_delta_l2"][name])
        residual = (outputs["joint"] - outputs["h2"] - outputs["h10"]) + outputs["noop"]
        assert near(norm(residual), diagnostic["joint_minus_singles_output_residual_l2"])

    summary = result["summary"]["families"][group]
    reference_family = reference["families"][group]
    component, d = {}, {}
    for action in ARMS:
        changes = [row[action]["score"] - row["clean"]["score"] for row in family]
        shuffled = math.fsum(changes[1:]) / 3
        d[action] = changes[0] - shuffled
        saved = summary["arms"][action]
        assert all(near(a, b) for a, b in zip(changes, saved["case_deltas"]))
        assert near(shuffled, saved["shuffle_mean_delta"]) and near(d[action], saved["family_contrast"])
        component[action] = {"repeat_delta": changes[0], "shuffle_mean_delta": shuffled, "D": d[action]}
    additive_changes, prompt_interactions = [], []
    for variant, row in enumerate(family):
        additive_logits = (row["h2"]["logits"] - row["clean"]["logits"]) + row["h10"]["logits"]
        target = manifest["cases"][4 * group + variant]["target_id"]
        additive_changes.append(logp(additive_logits, target) - row["clean"]["score"])
        prompt = ((row["joint"]["score"] - row["h2"]["score"]) - row["h10"]["score"]) + row["clean"]["score"]
        assert near(prompt, summary["case_interactions"][variant])
        reference_case = reference_family["cases"][variant]
        assert reference_case["target_id"] == target
        assert near(reference_case["prompt_interaction"], prompt)
        for name in ("clean", "h10", "h2", "joint"):
            assert near(reference_case["log_probabilities"][name], row[name]["score"])
            if name != "clean":
                assert near(reference_case["deltas"][name], row[name]["score"] - row["clean"]["score"])
        assert near(reference_case["deltas"]["additive"], additive_changes[-1])
        assert near(reference_case["log_probabilities"]["additive"], additive_changes[-1] + row["clean"]["score"])
        assert near(reference_case["additive_prompt_interaction"],
                    (additive_changes[-1] - (row["h2"]["score"] - row["clean"]["score"]))
                    - (row["h10"]["score"] - row["clean"]["score"]))
        prompt_interactions.append(prompt)
    d_add = additive_changes[0] - math.fsum(additive_changes[1:]) / 3
    interaction = (d["joint"] - d["h2"]) - d["h10"]
    j_add = (d_add - d["h2"]) - d["h10"]
    assert near(interaction, summary["interaction"])
    for action in ("h10", "h2", "joint"):
        assert near(reference_family["contrasts"][action], d[action])
    assert near(reference_family["contrasts"]["additive"], d_add)
    assert near(reference_family["interaction"], interaction)
    assert near(reference_family["additive_interaction"], j_add)
    assert near(reference_family["joint_minus_additive"], d["joint"] - d_add)
    families.append({"group": group + 1, "dose": dose, "arms": component,
                     "prompt_interactions": prompt_interactions, "J": interaction,
                     "additive_case_deltas": additive_changes, "D_add": d_add,
                     "J_add": j_add, "joint_minus_additive": d["joint"] - d_add})

assert file_bytes == result["raw_payload_bytes"]
means = {action: math.fsum(row["arms"][action]["D"] for row in families) / 8 for action in ARMS}
for action in ARMS:
    assert near(means[action], result["summary"]["mean_contrast_by_arm"][action])
for action in ("clean", *ARMS):
    expected = result["summary"]["top1_clean"] if action == "clean" else result["summary"]["top1_by_arm"][action]
    assert top1[action] == expected
output = {"receipt_sha256": sha(receipt_raw), "result_sha256": sha(result_raw), "families": families,
          "additive_reference_sha256": sha(reference_raw),
          "mean_D": means, "mean_J": math.fsum(row["J"] for row in families) / 8,
          "mean_J_add": math.fsum(row["J_add"] for row in families) / 8,
          "mean_joint_minus_additive": math.fsum(row["joint_minus_additive"] for row in families) / 8,
          "top1": top1, "scope": "32 exposed public development rows; saved-byte replay, no model forward"}
assert near(output["mean_J"], reference["mean_interaction"])
assert near(output["mean_J_add"], reference["mean_additive_interaction"])
assert near(output["mean_joint_minus_additive"], reference["mean_joint_minus_additive"])
print(json.dumps(output, indent=2, allow_nan=False))
