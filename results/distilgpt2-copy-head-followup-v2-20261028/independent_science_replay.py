"""Offline replay of the six-arm public copying follow-up; no model forward."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RUN = ROOT / "run"
ARMS = (("h10_final", 10, 10), ("h2_final", 2, 10), ("h7_final", 7, 10),
        ("h3_final", 3, 10), ("h1_final", 1, 10), ("h10_target", 10, 5))
ARM_MAP = {name: (head, position) for name, head, position in ARMS}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def close(a, b):
    return math.isclose(a, b, abs_tol=2e-12, rel_tol=0)


receipt_raw = (ROOT / "supervisor.json").read_bytes()
receipt = json.loads(receipt_raw)
result_raw = (RUN / "result.json").read_bytes()
result = json.loads(result_raw)
manifest_raw = (RUN / "manifest.json").read_bytes()
manifest = json.loads(manifest_raw)
assert receipt["process_exit_code"] == 0 and receipt["interrupted"] is False
assert receipt["failure_file_present"] is False and receipt["stderr_bytes"] == 0
assert receipt["source_revision"] == "40e98bc5e851a7be95035f8a60b266042422fff9"
assert manifest["source"]["revision"] == receipt["source_revision"]
assert manifest["source"]["tree"] == "b179c1e0610b9559720881aa249cfe76fc525cd8"
cpu_env = {key: {"present": False, "value": None} for key in ("MKL_CBWR", "ATEN_CPU_CAPABILITY")}
assert manifest["cpu_backend_environment_at_torch_import"] == result["runtime"]["cpu_backend_environment"] == cpu_env
assert sha(result_raw) == receipt["run_sha256"]["result.json"]
assert sha(manifest_raw) == result["manifest_sha256"]
assert result["status"] == "complete" and result["submitted_forwards"] == result["returned_forwards"] == 288
assert result["raw_payload_bytes"] == 80_604_288
assert result["model_state_sha256_before"] == result["model_state_sha256_after"]
assert result["model_buffers_sha256_before"] == result["model_buffers_sha256_after"]
assert len(manifest["cases"]) == 32 and len(result["cases"]) == 288

rows = []
for i, case in enumerate(manifest["cases"]):
    group, variant = divmod(i, 4)
    case_id = f"copy-followup-{group + 1:02d}-{variant}"
    assert (case["case_id"], case["group"], case["variant"]) == (case_id, group + 1, variant)
    raw = (RUN / (case_id + ".f32")).read_bytes()
    assert sha(raw) == receipt["run_sha256"][case_id + ".f32"]
    offset, saved = 0, {}
    names = tuple(name for name, _, _ in ARMS)
    rotation = (group + 2 * variant) % 6
    actions = ("clean", "noop", *names[rotation:], *names[:rotation], "clean_after")
    for action, record in zip(actions, result["cases"][9 * i:9 * i + 9]):
        assert (record["case_id"], record["group"], record["variant"], record["target_id"], record["action"]) == (
            case_id, group + 1, variant, case["target_id"], action)
        payload = {}
        for name, info in sorted(record["raw"].items(), key=lambda item: item[1]["offset"]):
            assert info["offset"] == offset
            value = raw[offset:offset + info["bytes"]]
            assert len(value) == info["bytes"] and sha(value) == info["sha256"]
            payload[name] = value
            offset += len(value)
        expected = {"logits"} if action in ("clean", "clean_after") else {
            "logits", "cproj_before", "cproj_after", "cproj_output"}
        assert set(payload) == expected and len(payload["logits"]) == 50_257 * 4
        logits = np.frombuffer(payload["logits"], dtype="<f4").astype(np.float64)
        assert np.isfinite(logits).all()
        target, maximum = case["target_id"], float(logits.max())
        logp = float(logits[target]) - maximum - math.log(math.fsum(math.exp(float(x) - maximum) for x in logits))
        score = record["score"]
        assert close(logp, score["target_log_probability"])
        assert score["target_logit"] == float(logits[target])
        assert score["target_rank"] == 1 + int(np.count_nonzero(logits > logits[target]))
        assert score["top_token_id"] == int(np.argmax(logits))
        if action not in ("clean", "clean_after"):
            assert all(len(payload[name]) == 11 * 768 * 4 for name in
                       ("cproj_before", "cproj_after", "cproj_output"))
            before = np.frombuffer(payload["cproj_before"], dtype="<u4").reshape(1, 11, 768)
            after = np.frombuffer(payload["cproj_after"], dtype="<u4").reshape(1, 11, 768)
            expected_bits = before.copy()
            if action != "noop":
                head, position = ARM_MAP[action]
                start = 64 * head
                expected_bits[0, position, start:start + 64] = 0
                values = np.frombuffer(payload["cproj_before"], dtype="<f4").reshape(1, 11, 768)[0, position, start:start + 64]
                witness = record["intervention"]
                assert (witness["layer"], witness["head"], witness["position"], witness["channels"]) == (
                    3, head, position, [start, start + 64])
                assert witness["changed_float32_bit_patterns"] == int(np.count_nonzero(
                    before[0, position, start:start + 64] != after[0, position, start:start + 64]))
                assert close(witness["pre_head_l2"], math.sqrt(math.fsum(float(x) ** 2 for x in values)))
            assert np.array_equal(expected_bits, after)
            assert np.isfinite(np.frombuffer(payload["cproj_output"], dtype="<f4")).all()
        saved[action] = (payload, logp, score["top_token_id"] == target)
    assert offset == len(raw)
    assert saved["clean"][0]["logits"] == saved["noop"][0]["logits"] == saved["clean_after"][0]["logits"]
    assert all(saved["noop"][0]["cproj_before"] == saved[name][0]["cproj_before"] for name in names)
    rows.append(saved)

families = []
for group in range(8):
    family = rows[4 * group:4 * group + 4]
    entry = {"group": group + 1, "arms": {}, "candidate_minus_arm": {}}
    saved_family = result["summary"]["families"][group]
    for name, _, _ in ARMS:
        delta = [row[name][1] - row["clean"][1] for row in family]
        shuffle = math.fsum(delta[1:]) / 3
        contrast = delta[0] - shuffle
        assert all(close(a, b) for a, b in zip(delta, saved_family["arms"][name]["case_deltas"]))
        assert close(contrast, saved_family["arms"][name]["family_contrast"])
        entry["arms"][name] = {"repeat_delta": delta[0], "shuffle_mean_delta": shuffle, "contrast": contrast}
    for name, _, _ in ARMS[1:]:
        difference = entry["arms"]["h10_final"]["contrast"] - entry["arms"][name]["contrast"]
        assert close(difference, saved_family["candidate_minus_arm"][name])
        entry["candidate_minus_arm"][name] = difference
    families.append(entry)

means = {name: math.fsum(row["arms"][name]["contrast"] for row in families) / 8 for name, _, _ in ARMS}
paired = {name: math.fsum(row["candidate_minus_arm"][name] for row in families) / 8 for name, _, _ in ARMS[1:]}
for name, value in means.items():
    assert close(value, result["summary"]["mean_contrast_by_arm"][name])
for name, value in paired.items():
    assert close(value, result["summary"]["mean_candidate_minus_arm"][name])
top1 = {name: {"repeated": sum(row[name][2] for row in rows[::4]),
               "shuffled": sum(row[name][2] for i, row in enumerate(rows) if i % 4)}
        for name in ("clean", *(name for name, _, _ in ARMS))}
print(json.dumps({"receipt_sha256": sha(receipt_raw), "result_sha256": sha(result_raw),
                  "families": families, "means": means, "paired": paired, "top1": top1,
                  "scope": "32 fixed public inputs, 288 CPU forwards, saved-byte replay; not held out"},
                 indent=2, allow_nan=False))
