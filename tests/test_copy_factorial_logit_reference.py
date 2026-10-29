"""Checks of the saved-logit reference; no checkpoint or model is used."""
import importlib
import hashlib
import json
import math
import struct
import subprocess
import sys

import pytest


def test_additive_logits_can_have_a_nonzero_log_probability_interaction():
    reference = importlib.import_module("looking_glass.copy_factorial_logit_reference")

    def pair(target):
        return struct.pack("<2f", target, 0.0)

    family = []
    for baseline in (0.0, 3.0, 3.0, 3.0):
        family.append({"target_id": 0, "clean": pair(baseline),
                       "h2": pair(baseline + 1.0), "h10": pair(baseline + 1.0),
                       "joint": pair(baseline + 2.0)})
    result = reference.analyze_family(family)
    assert result["interaction"] == pytest.approx(-0.1745489723392314, abs=1e-14)
    assert result["additive_interaction"] == pytest.approx(-0.1745489723392314, abs=1e-14)
    assert result["joint_minus_additive"] == 0.0


def test_eight_family_summary_separates_joint_and_additive_outcomes():
    reference = importlib.import_module("looking_glass.copy_factorial_logit_reference")
    pair = lambda target: struct.pack("<2f", target, 0.0)
    family = [{"target_id": 0, "clean": pair(b), "h2": pair(b + 1), "h10": pair(b + 1),
               "joint": pair(b + 3 if i == 0 else b + 2)}
              for i, b in enumerate((0, 3, 3, 3))]
    result = reference.analyze_study([family] * 8)
    assert len(result["families"]) == 8
    assert result["mean_additive_interaction"] == pytest.approx(-0.1745489723392314, abs=1e-14)
    assert result["mean_joint_minus_additive"] == pytest.approx(0.07834065946923044, abs=1e-14)
    assert result["mean_interaction"] == pytest.approx(-0.096208312870001, abs=1e-14)


def test_reference_does_not_round_the_constructed_logits_back_to_float32():
    reference = importlib.import_module("looking_glass.copy_factorial_logit_reference")
    clean = struct.pack("<2f", 0, 0)
    h2 = struct.pack("<2f", 1, 0)
    h10 = struct.pack("<2f", 2**24, 2**24)
    case = {"target_id": 0, "clean": clean, "h2": h2, "h10": h10, "joint": h10}
    result = reference.analyze_family([case] * 4)
    assert result["cases"][0]["log_probabilities"]["additive"] == pytest.approx(
        -0.31326168751822286, abs=1e-14
    )


def test_bit_decoder_preserves_signed_zero_and_float32_subnormals():
    reference = importlib.import_module("looking_glass.copy_factorial_logit_reference")
    values = reference._decode_float32(struct.pack("<4I", 1, 0x80000001, 0, 0x80000000))
    assert values[:2] == [2**-149, -(2**-149)]
    assert math.copysign(1.0, values[2]) == 1.0
    assert math.copysign(1.0, values[3]) == -1.0


def test_nonfinite_saved_logits_fail_instead_of_producing_a_reference():
    reference = importlib.import_module("looking_glass.copy_factorial_logit_reference")
    normal = struct.pack("<2f", 0, 1)
    bad = struct.pack("<2I", 0x7F800001, 0)
    case = {"target_id": 0, "clean": normal, "h2": bad, "h10": normal, "joint": normal}
    with pytest.raises(ValueError, match="finite"):
        reference.analyze_family([case] * 4)


def test_wrong_family_count_is_not_silently_summarized():
    reference = importlib.import_module("looking_glass.copy_factorial_logit_reference")
    with pytest.raises(ValueError, match="eight"):
        reference.analyze_study([])


def saved_fixture():
    cases, records, files = [], [], {}
    for group in range(8):
        family = [{"case_id": f"copy-factorial-{group + 1:02d}-{v}", "group": group + 1,
                   "variant": v, "target_id": 0} for v in range(4)]
        cases.extend(family)
        for case in family:
            files[case["case_id"]] = bytearray()
        schedule = [(v, action) for v in range(4) for action in ("clean", "noop")]
        for variant in range(4):
            arms = ("h10", "h2", "h7", "joint")
            rotation = (group + variant) % 4
            schedule.extend((variant, action) for action in (*arms[rotation:], *arms[:rotation], "clean_after"))
        for variant, action in schedule:
            case = family[variant]
            base = 0.0 if variant == 0 else 3.0
            shift = {"h2": 1.0, "h10": 1.0, "joint": 2.0}.get(action, 0.0)
            data = struct.pack("<2f", base + shift, 0.0)
            file = files[case["case_id"]]
            offset = len(file)
            file.extend(data)
            records.append({**case, "action": action,
                "score": {"target_log_probability": -math.log1p(math.exp(-(base + shift)))}, "raw": {"logits": {
                "offset": offset, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}})
    return cases, records, {key: bytes(value) for key, value in files.items()}


def test_extracts_the_fixed_schedule_and_rejects_a_corrupted_saved_vector():
    reference = importlib.import_module("looking_glass.copy_factorial_logit_reference")
    cases, records, files = saved_fixture()
    families = reference.extract_saved_families(cases, records, files, vocabulary_size=2)
    result = reference.analyze_study(families)
    assert result["mean_interaction"] == pytest.approx(-0.1745489723392314, abs=1e-14)
    assert result["mean_joint_minus_additive"] == 0.0
    corrupted = dict(files)
    name = "copy-factorial-01-0"
    corrupted[name] = bytes([files[name][0] ^ 1]) + files[name][1:]
    with pytest.raises(ValueError, match="digest"):
        reference.extract_saved_families(cases, records, corrupted, vocabulary_size=2)

    # Rehashing a changed no-op must not make it a valid clean control.
    changed = bytearray(files[name])
    item = records[1]["raw"]["logits"]
    changed[item["offset"]] ^= 1
    changed_files = {**files, name: bytes(changed)}
    changed_records = list(records)
    changed_records[1] = {**records[1], "raw": {"logits": {**item, "sha256": hashlib.sha256(
        changed[item["offset"]:item["offset"] + item["bytes"]]).hexdigest()}}}
    with pytest.raises(ValueError, match="no-op"):
        reference.extract_saved_families(cases, changed_records, changed_files, vocabulary_size=2)


def saved_artifact_fixture():
    cases, records, files = saved_fixture()
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    digest = lambda value: hashlib.sha256(value).hexdigest()
    source, protocol, clarification = "a" * 40, "b" * 64, "c" * 64
    case_raw = encode({"schema": "distilgpt2-copy-head-factorial-cases-v1", "cases": cases})
    manifest_raw = encode({"schema": "distilgpt2-copy-head-factorial-run-v1", "cases": cases,
        "cases_sha256": digest(case_raw), "protocol_sha256": protocol, "source": {"revision": source},
        "logit_reference_clarification": {"revision": source, "sha256": clarification},
        "exposure_inventories": {"v11": "d" * 64, "reservation": "e" * 64,
                                 "reservation_version": "distilgpt2-development-exposures-v12"}})
    logistic = lambda x: -math.log1p(math.exp(-x))
    arm_summary = {}
    for action, shift in (("h2", 1), ("h10", 1), ("joint", 2)):
        deltas = [logistic(b + shift) - logistic(b) for b in (0, 3, 3, 3)]
        arm_summary[action] = {"case_deltas": deltas,
                               "family_contrast": deltas[0] - math.fsum(deltas[1:]) / 3}
    prompt_interactions = [((logistic(b + 2) - logistic(b + 1)) - logistic(b + 1)) + logistic(b)
                           for b in (0, 3, 3, 3)]
    result_raw = encode({"schema": "distilgpt2-copy-head-factorial-result-v1", "status": "complete",
        "manifest_sha256": digest(manifest_raw), "submitted_forwards": 224,
        "model_call_returns_observed": 224, "returned_forwards": 224,
        "raw_payload_bytes": sum(map(len, files.values())), "cases": records,
        "summary": {"families": [{"group": i + 1, "interaction": -0.1745489723392314,
                                   "arms": arm_summary, "case_interactions": prompt_interactions}
                                  for i in range(8)]}})
    pre_raw = encode({"schema": "copying-head-factorial-launch-v1", "source_revision": source,
                      "case_sha256": digest(case_raw), "clarification_sha256": clarification})
    inventory = {"manifest.json": digest(manifest_raw), "result.json": digest(result_raw),
                 "attempt.jsonl": digest(b"journal")}
    inventory.update({name + ".f32": digest(raw) for name, raw in files.items()})
    receipt_raw = encode({"schema": "copying-head-factorial-process-observation-v1", "source_revision": source,
        "process_exit_code": 0, "interrupted": False, "failure_file_present": False,
        "pre_run_sha256": digest(pre_raw), "run_sha256": inventory})
    expected = {"source_revision": source, "case_sha256": digest(case_raw), "protocol_sha256": protocol,
                "clarification_sha256": clarification, "clarification_revision": source,
                "v11": "d" * 64, "v12": "e" * 64}
    return case_raw, manifest_raw, result_raw, pre_raw, receipt_raw, files, expected


def test_saved_artifact_binding_requires_receipt_and_matching_observed_interaction():
    reference = importlib.import_module("looking_glass.copy_factorial_logit_reference")
    case_raw, manifest_raw, result_raw, pre_raw, receipt_raw, files, expected = saved_artifact_fixture()
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    result = reference.analyze_saved(case_raw, manifest_raw, result_raw, pre_raw, receipt_raw,
                                     b"journal", files, expected=expected, vocabulary_size=2)
    assert result["mean_interaction"] == pytest.approx(-0.1745489723392314, abs=1e-14)
    assert result["mean_joint_minus_additive"] == 0.0
    bad_receipt = json.loads(receipt_raw)
    bad_receipt["process_exit_code"] = 1
    with pytest.raises(ValueError, match="receipt"):
        reference.analyze_saved(case_raw, manifest_raw, result_raw, pre_raw, encode(bad_receipt),
                                b"journal", files, expected=expected, vocabulary_size=2)
    bad_result = json.loads(result_raw)
    bad_result["summary"]["families"][0]["interaction"] += 0.25
    altered = encode(bad_result)
    rehashed_receipt = json.loads(receipt_raw)
    rehashed_receipt["run_sha256"]["result.json"] = hashlib.sha256(altered).hexdigest()
    with pytest.raises(ValueError, match="interaction"):
        reference.analyze_saved(case_raw, manifest_raw, altered, pre_raw, encode(rehashed_receipt),
                                b"journal", files, expected=expected, vocabulary_size=2)
    bad_result = json.loads(result_raw)
    bad_result["summary"]["families"][0]["arms"]["h2"]["family_contrast"] += 0.25
    altered = encode(bad_result)
    rehashed_receipt = json.loads(receipt_raw)
    rehashed_receipt["run_sha256"]["result.json"] = hashlib.sha256(altered).hexdigest()
    with pytest.raises(ValueError, match="arm"):
        reference.analyze_saved(case_raw, manifest_raw, altered, pre_raw, encode(rehashed_receipt),
                                b"journal", files, expected=expected, vocabulary_size=2)
    for invalid in (None, 223, True):
        bad_result = json.loads(result_raw)
        if invalid is None:
            del bad_result["model_call_returns_observed"]
        else:
            bad_result["model_call_returns_observed"] = invalid
        altered = encode(bad_result)
        rehashed_receipt = json.loads(receipt_raw)
        rehashed_receipt["run_sha256"]["result.json"] = hashlib.sha256(altered).hexdigest()
        with pytest.raises(ValueError, match="incomplete"):
            reference.analyze_saved(case_raw, manifest_raw, altered, pre_raw, encode(rehashed_receipt),
                                    b"journal", files, expected=expected, vocabulary_size=2)


def test_saved_directory_requires_the_complete_quiescent_receipt(tmp_path):
    reference = importlib.import_module("looking_glass.copy_factorial_logit_reference")
    case_raw, manifest_raw, result_raw, pre_raw, receipt_raw, files, expected = saved_artifact_fixture()
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "cases.json").write_bytes(case_raw)
    (tmp_path / "pre-run.json").write_bytes(pre_raw)
    (tmp_path / "supervisor.json").write_bytes(receipt_raw)
    (run / "manifest.json").write_bytes(manifest_raw)
    (run / "result.json").write_bytes(result_raw)
    (run / "attempt.jsonl").write_bytes(b"journal")
    for case_id, raw in files.items():
        (run / (case_id + ".f32")).write_bytes(raw)
    total = sum(map(len, files.values()))
    result = reference.analyze_directory(tmp_path, tmp_path / "cases.json", expected=expected, vocabulary_size=2,
                                         expected_total_bytes=total)
    assert result["mean_additive_interaction"] == pytest.approx(-0.1745489723392314, abs=1e-14)
    with pytest.raises(ValueError, match="payload total"):
        reference.analyze_directory(tmp_path, tmp_path / "cases.json", expected=expected, vocabulary_size=2,
                                    expected_total_bytes=total + 1)
    (run / "failure.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="file inventory"):
        reference.analyze_directory(tmp_path, tmp_path / "cases.json", expected=expected, vocabulary_size=2,
                                    expected_total_bytes=total)


def test_cli_does_not_report_success_for_a_missing_attempt(tmp_path):
    process = subprocess.run([sys.executable, "scripts/copy_factorial_logit_reference.py",
                              "--attempt", str(tmp_path / "missing"), "--case-file", str(tmp_path / "cases.json"),
                              "--source-revision", "a" * 40], capture_output=True, text=True, check=False)
    assert process.returncode != 0
    assert "saved run file inventory" in process.stderr
    assert not process.stdout
