"""Independent saved-byte reconciliation for the public factorial attempt."""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

attempt = Path(sys.argv[1])
root = attempt / "run"
result = json.loads((root / "result.json").read_bytes())
manifest = json.loads((root / "manifest.json").read_bytes())
receipt = json.loads((attempt / "supervisor.json").read_bytes())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert receipt["process_exit_code"] == 0 and not receipt["interrupted"] and not receipt["failure_file_present"]
assert receipt["source_revision"] == "5e4a82baed62016950a6f0fd0f6487b3f7d3b698"
assert digest(attempt / "pre-run.json") == receipt["pre_run_sha256"]
assert all(digest(root / name) == value for name, value in receipt["run_sha256"].items())
assert result["status"] == "complete" and result["submitted_forwards"] == 224
assert result["model_call_returns_observed"] == result["returned_forwards"] == 224
assert result["raw_payload_bytes"] == 61_250_432 and len(result["cases"]) == 224
assert len(manifest["cases"]) == 32
assert struct.unpack("<I", struct.pack("<f", 3.0 * 2.0 ** -150))[0] == 2
assert struct.unpack("<I", struct.pack("<f", 1.0 + 2.0 ** -24))[0] == 0x3f800000
events = [json.loads(line) for line in (root / "attempt.jsonl").read_text().splitlines()]
assert len(events) == 713 and events[0]["event"] == "runtime_ready"
event_index, record_index = 1, 0


def check_triplet():
    global event_index, record_index
    record = result["cases"][record_index]
    start, raw, complete = events[event_index:event_index + 3]
    assert [start["event"], raw["event"], complete["event"]] == ["start", "raw_recorded", "complete"]
    assert (start["case_id"], start["action"]) == (record["case_id"], record["action"])
    assert (raw["case_id"], raw["action"], raw["raw"]) == (record["case_id"], record["action"], record["raw"])
    assert {key: value for key, value in complete.items() if key != "event"} == record
    event_index += 3
    record_index += 1


for family in range(8):
    for _ in range(8):
        check_triplet()
    dose_event = events[event_index]
    assert dose_event["event"] == "family_dose"
    assert {key: value for key, value in dose_event.items() if key != "event"} == result["family_doses"][family]
    event_index += 1
    for variant in range(4):
        for _ in range(5):
            check_triplet()
        diag = events[event_index]
        assert diag["event"] == "case_diagnostics"
        assert {key: value for key, value in diag.items() if key != "event"} == result["case_diagnostics"][4 * family + variant]
        event_index += 1
assert event_index == len(events) and record_index == 224


def values(raw):
    out = []
    for (bits,) in struct.iter_unpack("<I", raw):
        sign, exponent, fraction = bits >> 31, (bits >> 23) & 255, bits & 0x7fffff
        assert exponent != 255
        mantissa = fraction if exponent == 0 else fraction | (1 << 23)
        value = math.ldexp(float(mantissa), -149 if exponent == 0 else exponent - 150)
        out.append(-value if sign else value)
    return out


def norm(items):
    return math.sqrt(math.fsum(item ** 2 for item in items))


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


heads, arms = (10, 2, 7), ("h10", "h2", "h7", "joint")
family_rows, doses = [], []
for family in range(8):
    segment = result["cases"][28 * family:28 * (family + 1)]
    expected = [(v, a) for v in range(4) for a in ("clean", "noop")]
    for variant in range(4):
        k = (family + variant) % 4
        expected.extend((variant, a) for a in (*arms[k:], *arms[:k], "clean_after"))
    assert [(r["variant"], r["action"]) for r in segment] == expected
    parsed = []
    for variant in range(4):
        case = manifest["cases"][4 * family + variant]
        data = (root / (case["case_id"] + ".f32")).read_bytes()
        assert len(data) == 1_914_076
        offset, row = 0, {}
        for entry in (r for r in segment if r["variant"] == variant):
            assert entry["case_id"] == case["case_id"] and entry["target_id"] == case["target_id"]
            payload = {}
            for name, item in sorted(entry["raw"].items(), key=lambda pair: pair[1]["offset"]):
                assert item["offset"] == offset
                raw = data[offset:offset + item["bytes"]]
                assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
                payload[name], offset = raw, offset + item["bytes"]
            logits = values(payload["logits"])
            assert len(logits) == 50_257
            maximum, target = max(logits), logits[case["target_id"]]
            logp = (target - maximum) - math.log(math.fsum(math.exp(x - maximum) for x in logits))
            assert close(logp, entry["score"]["target_log_probability"])
            assert 1 + sum(x > target for x in logits) == entry["score"]["target_rank"]
            assert logits.index(maximum) == entry["score"]["top_token_id"]
            row[entry["action"]] = (payload, logp, entry["score"]["top_token_id"] == case["target_id"])
        assert offset == len(data)
        assert row["clean"][0]["logits"] == row["noop"][0]["logits"] == row["clean_after"][0]["logits"]
        assert row["noop"][0]["cproj_before"] == row["noop"][0]["cproj_after"]
        parsed.append(row)
    original = []
    for row in parsed:
        source = row["noop"][0]["cproj_before"]
        original.append({head: source[(10 * 768 + head * 64) * 4:(10 * 768 + (head + 1) * 64) * 4]
                         for head in heads})
    ns = [{head: norm(values(row[head])) for head in heads} for row in original]
    dose = min(n for row in ns for n in row.values())
    assert dose > 2.0 ** -20 and close(dose, result["family_doses"][family]["requested_l2"])
    doses.append(dose)
    for variant, row in enumerate(parsed):
        source = row["noop"][0]["cproj_before"]
        replacements = {}
        for head in heads:
            before = values(original[variant][head])
            fraction = dose / ns[variant][head]
            desired = b"".join(struct.pack("<f", (1.0 - fraction) * value) for value in before)
            after = values(desired)
            achieved = norm(new - old for new, old in zip(after, before))
            saved = result["family_doses"][family]["variants"][variant][str(head)]
            assert close(saved["source_l2"], ns[variant][head]) and close(saved["requested_fraction"], fraction)
            assert close(saved["achieved_l2"], achieved) and abs(achieved - dose) <= 1e-5 * dose
            replacements[head] = desired
        for action in arms:
            selected = {"h10": (10,), "h2": (2,), "h7": (7,), "joint": (2, 10)}[action]
            payload = row[action][0]
            assert payload["cproj_before"] == source
            expected_after = bytearray(source)
            for head in selected:
                start = (10 * 768 + head * 64) * 4
                expected_after[start:start + 256] = replacements[head]
            assert payload["cproj_after"] == bytes(expected_after)
        output = {name: values(row[name][0]["cproj_output"][-768 * 4:]) for name in ("noop", *arms)}
        saved_diag = result["case_diagnostics"][4 * family + variant]
        for action in arms:
            difference = norm(x - y for x, y in zip(output[action], output["noop"]))
            assert close(difference, saved_diag["final_output_delta_l2"][action])
        residual = norm(((j - h2) - h10) + clean for j, h2, h10, clean in
                        zip(output["joint"], output["h2"], output["h10"], output["noop"]))
        assert close(residual, saved_diag["joint_minus_singles_output_residual_l2"])
    family_rows.append(parsed)

families = []
for family, rows in enumerate(family_rows):
    contrasts, repeated, shuffled = {}, {}, {}
    for action in arms:
        delta = [row[action][1] - row["clean"][1] for row in rows]
        repeated[action], shuffled[action] = delta[0], math.fsum(delta[1:]) / 3
        contrasts[action] = repeated[action] - shuffled[action]
        saved = result["summary"]["families"][family]["arms"][action]
        assert all(close(a, b) for a, b in zip(delta, saved["case_deltas"]))
        assert close(contrasts[action], saved["family_contrast"])
    interaction = (contrasts["joint"] - contrasts["h2"]) - contrasts["h10"]
    assert close(interaction, result["summary"]["families"][family]["interaction"])
    families.append({"dose": doses[family], "contrast": contrasts, "repeat": repeated,
                     "shuffle_mean": shuffled, "interaction": interaction})
report = {"records_checked": 224, "families": families,
          "mean_contrast": {a: math.fsum(f["contrast"][a] for f in families) / 8 for a in arms},
          "mean_repeat_delta": {a: math.fsum(f["repeat"][a] for f in families) / 8 for a in arms},
          "mean_shuffle_delta": {a: math.fsum(f["shuffle_mean"][a] for f in families) / 8 for a in arms},
          "mean_interaction": math.fsum(f["interaction"] for f in families) / 8,
          "repeated_top1": {a: sum(rows[0][a][2] for rows in family_rows) for a in ("clean", *arms)}}
print(json.dumps(report, indent=2, allow_nan=False))
