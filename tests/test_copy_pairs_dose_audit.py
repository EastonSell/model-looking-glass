import math
import hashlib
from fractions import Fraction
import struct

import pytest


def _audit():
    try:
        from looking_glass import copy_pairs_dose_audit
    except ImportError:
        pytest.fail("independent pairwise saved-logit auditor is not implemented")
    return copy_pairs_dose_audit


def _logits(a, b, vocabulary_size=50257):
    return struct.pack("<2f", a, b) + struct.pack("<f", -1000.0) * (vocabulary_size - 2)


def test_full_vocabulary_additive_logits_can_have_nonzero_log_probability_interaction():
    audit = _audit()
    clean = _logits(0.0, 0.0)
    a = _logits(1.0, 0.0)
    b = _logits(0.0, 1.0)
    pair = _logits(1.0, 1.0)
    matching = _logits(-2.0, 0.0)
    cases = []
    for variant in range(4):
        actions = {name: clean for name in audit.ARMS}
        if variant == 0:
            actions.update(h2_low=a, h10_low=b, h2_h10_low=pair, h2_match=matching)
        cases.append({"target_id": 0, "clean": clean, "arms": actions})

    result = audit.analyze_family(cases, vocabulary_size=50257)
    logp = result["pairs"]["h2_h10_low"]["log_probability"]
    target = result["pairs"]["h2_h10_low"]["target_logit"]
    expected = 2 * math.log1p(math.exp(-1.0)) + 1.0 - 2 * math.log(2.0)
    assert logp["interaction"] == pytest.approx(expected, abs=1e-14)
    assert logp["additive_interaction"] == pytest.approx(expected, abs=1e-14)
    assert logp["pair_minus_additive"] == pytest.approx(0.0, abs=1e-14)
    assert target["interaction"] == 0.0
    assert target["pair_minus_additive"] == 0.0
    assert logp["pair_minus_match_a"] != pytest.approx(logp["interaction"])


def _dose_fixture():
    audit = _audit()
    source = bytearray(struct.pack("<f", 0.125) * (11 * 768))
    original = {2: 0.5, 7: 0.375, 10: 0.25}
    expected = {
        "low": {2: 0x3ec00000, 7: 0x3e800000, 10: 0x3e000000},
        "match": {2: 0x3ea57d86, 7: 0x3e4afb0d, 10: 0x3d95f61a},
        "high": {2: 0x3e800000, 7: 0x3e000000, 10: 0x00000000},
    }
    for head, value in original.items():
        start = (10 * 768 + 64 * head) * 4
        source[start:start + 256] = struct.pack("<f", value) * 64
    deliveries = {}
    for arm in audit.ARMS:
        components = arm.split("_")
        dose = components[-1]
        data = bytearray(source)
        for head_name in components[:-1]:
            head = int(head_name[1:])
            start = (10 * 768 + 64 * head) * 4
            data[start:start + 256] = struct.pack("<I", expected[dose][head]) * 64
        deliveries[arm] = bytes(data)
    return bytes(source), deliveries


def test_exact_dose_deliveries_and_untouched_coordinates():
    audit = _audit()
    source, deliveries = _dose_fixture()
    report = audit.validate_delivery_family([source] * 4, [deliveries] * 4)
    assert report["q_low"] == 1.0
    assert report["q_high"] == 2.0
    assert report["q_match"] == math.sqrt(2.0)
    assert report["variants"][0]["norms"] == {"h2": 4.0, "h7": 3.0, "h10": 2.0}
    assert report["variants"][0]["pairs"]["h2_h10_low"]["combined_norm"] == math.sqrt(2.0)
    tampered = dict(deliveries)
    data = bytearray(tampered["h2_h10_low"])
    data[0] ^= 1
    tampered["h2_h10_low"] = bytes(data)
    with pytest.raises(ValueError, match="delivery"):
        audit.validate_delivery_family([source] * 4, [tampered, deliveries, deliveries, deliveries])


@pytest.mark.parametrize("value,bits", [
    (0.0, 0x00000000), (-0.0, 0x80000000),
    (2.0 ** -150, 0x00000000), (-(2.0 ** -150), 0x80000000),
    (3.0 * 2.0 ** -150, 0x00000002),
    (1.0 + 2.0 ** -24, 0x3f800000),
    (1.0 + 3.0 * 2.0 ** -24, 0x3f800002),
])
def test_independent_float32_conversion_rounds_ties_to_even(value, bits):
    assert _audit()._nearest_float32_bits(value) == bits


def test_norm_uses_written_square_operation_instead_of_multiplication():
    # On the study runtime, Python's power and multiplication differ by one ULP.
    values = [float.fromhex("-0x1.8198ccc000000p+0"), float.fromhex("-0x1.d332fb951230cp-1")]
    expected = math.sqrt(math.fsum(value ** 2 for value in values))
    assert _audit()._norm(values) == expected


def test_documented_square_midpoint_has_an_even_and_odd_neighbor():
    value = float.fromhex("-0x1.8198ccc000000p+0")
    even = float.fromhex("0x1.22667986a3e14p+1")
    odd = float.fromhex("0x1.22667986a3e15p+1")
    exact = Fraction.from_float(value) ** 2
    assert exact - Fraction.from_float(even) == Fraction.from_float(odd) - exact
    assert struct.unpack("<Q", struct.pack("<d", even))[0] % 2 == 0
    assert struct.unpack("<Q", struct.pack("<d", odd))[0] % 2 == 1
    assert _audit()._norm([value, -value]) == math.sqrt(math.fsum(x ** 2 for x in (value, -value)))


def _saved_family_fixture(family_index=0):
    audit = _audit()
    source, deliveries = _dose_fixture()
    cases = [{"case_id": f"copy-pairs-{family_index + 1:02d}-{v}", "group": family_index + 1,
              "variant": v, "target_id": 0} for v in range(4)]
    buffers = {case["case_id"]: bytearray() for case in cases}
    records = []
    clean = struct.pack("<2f", 0.0, 0.0)
    def add(variant, action):
        case = cases[variant]
        buffer = buffers[case["case_id"]]
        parts = {"logits": clean}
        if action not in ("clean", "clean_after"):
            output_value = {"noop": 1.0, "h2_low": 2.0, "h10_low": 3.0,
                            "h2_h10_low": 5.0}.get(action, 0.0)
            output = struct.pack("<f", output_value) * (11 * 768)
            parts.update(cproj_before=source, cproj_after=source if action == "noop" else deliveries[action], cproj_output=output)
        raw = {}
        for name, payload in parts.items():
            raw[name] = {"offset": len(buffer), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            buffer.extend(payload)
        records.append({**case, "action": action, "raw": raw,
                        "score": {"target_logit": 0.0, "target_log_probability": -math.log(2.0),
                                  "target_rank": 1, "top_token_id": 0}})

    for variant in range(4):
        add(variant, "clean")
        add(variant, "noop")
    for variant in range(4):
        rotation = (4 * family_index + variant) % len(audit.ARMS)
        for action in audit.ARMS[rotation:] + audit.ARMS[:rotation]:
            add(variant, action)
        add(variant, "clean_after")
    return cases, records, {key: bytes(value) for key, value in buffers.items()}


def test_saved_family_rejects_mismatched_pair_witness_even_if_rehashed():
    audit = _audit()
    cases, records, files = _saved_family_fixture()
    report = audit.audit_saved_family(cases, records, files, family_index=0, vocabulary_size=2)
    assert report["delivery"]["q_low"] == 1.0
    assert report["numerical"]["pairs"]["h2_h10_low"]["log_probability"]["interaction"] == 0.0
    diagnostic = report["output_diagnostics"][0]
    assert diagnostic["output_delta_norms"]["h2_h10_low"] == math.sqrt(768 * 16)
    assert diagnostic["output_delta_norms"]["h2_match"] == math.sqrt(768)
    assert diagnostic["pair_residual_norms"]["h2_h10_low"] == math.sqrt(768)
    pair_record = next(row for row in records if row["variant"] == 0 and row["action"] == "h2_h10_low")
    item = pair_record["raw"]["cproj_after"]
    changed = bytearray(files[pair_record["case_id"]])
    changed[item["offset"]] ^= 1
    files[pair_record["case_id"]] = bytes(changed)
    item["sha256"] = hashlib.sha256(changed[item["offset"]:item["offset"] + item["bytes"]]).hexdigest()
    with pytest.raises(ValueError, match="delivery"):
        audit.audit_saved_family(cases, records, files, family_index=0, vocabulary_size=2)


def test_saved_family_rejects_missing_arms_and_overlapping_slices():
    audit = _audit()
    cases, records, files = _saved_family_fixture()
    with pytest.raises(ValueError, match="record"):
        audit.audit_saved_family(cases, records[:-1], files, family_index=0, vocabulary_size=2)
    records[1]["raw"]["cproj_before"]["offset"] -= 4
    with pytest.raises(ValueError, match="slice"):
        audit.audit_saved_family(cases, records, files, family_index=0, vocabulary_size=2)


def test_saved_family_rejects_forged_score_identity_and_rehashed_nonfinite_logits():
    audit = _audit()
    cases, records, files = _saved_family_fixture()
    records[2]["score"]["target_logit"] = 1.0
    with pytest.raises(ValueError, match="score"):
        audit.audit_saved_family(cases, records, files, family_index=0, vocabulary_size=2)
    records[2]["score"]["target_logit"] = 0.0
    records[2]["target_id"] = False
    with pytest.raises(ValueError, match="identity"):
        audit.audit_saved_family(cases, records, files, family_index=0, vocabulary_size=2)
    records[2]["target_id"] = 0
    item = records[2]["raw"]["logits"]
    changed = bytearray(files[records[2]["case_id"]])
    struct.pack_into("<I", changed, item["offset"], 0x7fc00000)
    files[records[2]["case_id"]] = bytes(changed)
    item["sha256"] = hashlib.sha256(changed[item["offset"]:item["offset"] + item["bytes"]]).hexdigest()
    with pytest.raises(ValueError, match="nonfinite"):
        audit.audit_saved_family(cases, records, files, family_index=0, vocabulary_size=2)


def test_nonadditive_observed_pair_is_distinct_from_the_additive_reference():
    audit = _audit()
    def row(a, b):
        return struct.pack("<2f", a, b)
    baseline = row(0, 0)
    cases = []
    for variant in range(4):
        arms = {name: baseline for name in audit.ARMS}
        if variant == 0:
            arms.update(h2_high=row(1, 0), h10_high=row(0, 1), h2_h10_high=row(2, 1))
        cases.append({"target_id": 0, "clean": baseline, "arms": arms})
    report = audit.analyze_family(cases, vocabulary_size=2)
    logp = report["pairs"]["h2_h10_high"]["log_probability"]
    raw = report["pairs"]["h2_h10_high"]["target_logit"]
    assert logp["interaction"] == pytest.approx(0.6201145069582775, abs=1e-15)
    assert logp["additive_interaction"] == pytest.approx(0.2402290139165551, abs=1e-15)
    assert logp["pair_minus_additive"] == pytest.approx(0.3798854930417224, abs=1e-15)
    assert logp["prompt_interactions"] == pytest.approx([0.6201145069582775, 0.0, 0.0, 0.0], abs=1e-15)
    assert logp["additive_prompt_interactions"] == pytest.approx([0.2402290139165551, 0.0, 0.0, 0.0], abs=1e-15)
    assert raw["interaction"] == 1.0
    assert raw["additive_interaction"] == 0.0
    assert raw["pair_minus_additive"] == 1.0


def test_study_requires_eight_ordered_complete_family_reports_and_uses_fsum_mean():
    audit = _audit()
    cases, records, files = _saved_family_fixture()
    sample = audit.audit_saved_family(cases, records, files, family_index=0, vocabulary_size=2)
    families = []
    for group in range(8):
        family = dict(sample)
        family["case_ids"] = [f"copy-pairs-{group + 1:02d}-{variant}" for variant in range(4)]
        families.append(family)
    report = audit.summarize_study(families)
    assert len(report["families"]) == 8
    assert report["mean_pairs"]["h2_h10_low"]["log_probability"]["interaction"] == 0.0
    assert report["top_one"]["clean"] == {"repeated": 8, "shuffled": 24}
    with pytest.raises(ValueError, match="famil"):
        audit.summarize_study(families[:-1])
    families[4] = families[3]
    with pytest.raises(ValueError, match="famil"):
        audit.summarize_study(families)


def test_complete_study_replay_requires_exact_counters_and_streams_families():
    audit = _audit()
    cases, records, files = [], [], {}
    for group in range(8):
        family_cases, family_records, family_files = _saved_family_fixture(group)
        cases.extend(family_cases)
        records.extend(family_records)
        files.update(family_files)
    result = {"schema": "distilgpt2-copy-head-pairs-dose-result-v1", "status": "complete",
              "submitted_forwards": 576, "model_call_returns_observed": 576, "returned_forwards": 576,
              "raw_payload_bytes": sum(len(value) for value in files.values()), "cases": records}
    reads = []
    def read_case(case_id):
        reads.append(case_id)
        return files[case_id]
    report = audit.replay_result(cases, result, read_case, vocabulary_size=2)
    assert len(report["families"]) == 8
    assert reads == [row["case_id"] for row in cases]
    for bad in (575, True, None):
        result["model_call_returns_observed"] = bad
        reads.clear()
        with pytest.raises(ValueError, match="counter"):
            audit.replay_result(cases, result, read_case, vocabulary_size=2)
        assert reads == []
