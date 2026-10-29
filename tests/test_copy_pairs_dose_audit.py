import math
import hashlib
import copy
import json
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


def test_recorded_binary64_metadata_does_not_equate_positive_and_negative_zero():
    audit = _audit()
    assert audit._typed_equal({"value": -0.0}, {"value": -0.0})
    assert not audit._typed_equal({"value": 0.0}, {"value": -0.0})


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


def _recorded_metadata_fixture(cases, records):
    source, deliveries = _dose_fixture()
    old = {2: 0.5, 7: 0.375, 10: 0.25}
    m, requested = 2.0, {"low": 1.0, "match": math.sqrt(2.0), "high": 2.0}
    norm = {2: 4.0, 7: 3.0, 10: 2.0}
    variants = []
    for _ in range(4):
        item = {}
        for dose in ("low", "match", "high"):
            item[dose] = {}
            for head in (2, 7, 10):
                start = (10 * 768 + 64 * head) * 4
                after_raw = deliveries[f"h{head}_{dose}"][start:start + 256]
                before_raw = source[start:start + 256]
                words_before = [v[0] for v in struct.iter_unpack("<I", before_raw)]
                words_after = [v[0] for v in struct.iter_unpack("<I", after_raw)]
                after = [v[0] for v in struct.iter_unpack("<f", after_raw)]
                item[dose][str(head)] = {
                    "source_l2": norm[head], "requested_fraction": requested[dose] / norm[head],
                    "achieved_l2": math.sqrt(math.fsum((value - old[head]) ** 2 for value in after)),
                    "changed_float32_elements": sum(a != b for a, b in zip(words_before, words_after)),
                    "changed_binary_bits": sum((a ^ b).bit_count() for a, b in zip(words_before, words_after)),
                }
        variants.append(item)
    family = {"group": cases[0]["group"], "m": m, "requested_l2": requested, "variants": variants}
    diagnostics = []
    for case in cases:
        delta = {name: math.sqrt(768 * (value - 1.0) ** 2) for name, value in
                 ((name, {"h2_low": 2.0, "h10_low": 3.0, "h2_h10_low": 5.0}.get(name, 0.0))
                  for name in _audit().ARMS)}
        residual = {dose: {pair: math.sqrt(768) if (dose, pair) == ("low", "2_10")
                            else math.sqrt(768 * (1.0 -
                                ({"2_7": 2.0, "2_10": 2.0, "7_10": 3.0}[pair] if dose == "low" else 0.0)) ** 2)
                            for pair in ("2_7", "2_10", "7_10")}
                    for dose in ("low", "high")}
        # Compute the residual directly from the simple fixture's four output scalars.
        for dose in ("low", "high"):
            for a, b in ((2, 7), (2, 10), (7, 10)):
                vals = {"noop": 1.0, "h2_low": 2.0, "h10_low": 3.0, "h2_h10_low": 5.0}
                p = vals.get(f"h{a}_h{b}_{dose}", 0.0)
                x = vals.get(f"h{a}_{dose}", 0.0)
                y = vals.get(f"h{b}_{dose}", 0.0)
                residual[dose][f"{a}_{b}"] = math.sqrt(768 * (((p - x) - y) + 1.0) ** 2)
        diagnostics.append({"case_id": case["case_id"], "final_output_delta_l2": delta,
                            "pair_minus_singles_output_residual_l2": residual})
    for record in records:
        action = record["action"]
        if action not in _audit().ARMS:
            record["intervention"] = None
            continue
        names = action.split("_")
        dose, heads = names[-1], [int(x[1:]) for x in names[:-1]]
        details = {"heads": heads, "position": 10, "dose": dose, "requested_l2_each": requested[dose],
                   "components": {str(head): variants[record["variant"]][dose][str(head)] for head in heads}}
        if len(heads) == 2:
            differences = []
            for head in heads:
                start = (10 * 768 + 64 * head) * 4
                raw = deliveries[f"h{head}_{dose}"][start:start + 256]
                differences.extend(value[0] - old[head] for value in struct.iter_unpack("<f", raw))
            combined = math.sqrt(math.fsum(value ** 2 for value in differences))
            details["joint_combined_achieved_l2"] = combined
            details["joint_combined_requested_l2"] = requested["match"] if dose == "low" else math.sqrt(2) * requested["high"]
            if dose == "low":
                details["joint_minus_matching_single_l2"] = {
                    str(head): combined - variants[record["variant"]]["match"][str(head)]["achieved_l2"] for head in heads}
        record["intervention"] = details
    return family, diagnostics


def test_saved_family_metadata_rejects_rehashed_landing_and_summary_claims():
    audit = _audit()
    cases, records, files = _saved_family_fixture()
    family, diagnostics = _recorded_metadata_fixture(cases, records)
    report = audit.audit_saved_family(cases, records, files, family_index=0, vocabulary_size=2)
    audit.validate_recorded_family_metadata(cases, records, report, family, diagnostics)
    bad_family = copy.deepcopy(family)
    bad_family["variants"][0]["low"]["2"]["changed_binary_bits"] += 1
    with pytest.raises(ValueError, match="metadata"):
        audit.validate_recorded_family_metadata(cases, records, report, bad_family, diagnostics)
    bad_records = copy.deepcopy(records)
    pair = next(row for row in bad_records if row["action"] == "h2_h10_low")
    pair["intervention"]["components"]["2"]["changed_binary_bits"] += 1
    with pytest.raises(ValueError, match="intervention metadata"):
        audit.validate_recorded_family_metadata(cases, bad_records, report, family, diagnostics)
    bad_records = copy.deepcopy(records)
    next(row for row in bad_records if row["action"] == "noop")["intervention"] = {"heads": []}
    with pytest.raises(ValueError, match="control intervention"):
        audit.validate_recorded_family_metadata(cases, bad_records, report, family, diagnostics)
    bad_diagnostics = copy.deepcopy(diagnostics)
    bad_diagnostics[0]["final_output_delta_l2"]["h2_match"] += 1.0
    with pytest.raises(ValueError, match="diagnostic metadata"):
        audit.validate_recorded_family_metadata(cases, records, report, family, bad_diagnostics)


def _zero_runner_summary(audit):
    metrics = ("target_log_probability", "target_logit")
    families = []
    for group in range(8):
        arms = {}
        for name in audit.ARMS:
            entry = {"target_ranks": [1] * 4, "target_top1": [True] * 4}
            for metric in metrics:
                entry[metric] = {"case_deltas": [0.0] * 4, "repeat_delta": 0.0,
                                 "shuffle_mean_delta": 0.0, "family_contrast": 0.0}
            arms[name] = entry
        interactions = {dose: {pair: {**{metric: 0.0 for metric in metrics},
                                      **{metric + "_case_interactions": [0.0] * 4 for metric in metrics}}
                               for pair in ("2_7", "2_10", "7_10")} for dose in ("low", "high")}
        match = {pair: {f"h{head}": 0.0 for head in heads} for pair, heads in
                 (("2_7", (2, 7)), ("2_10", (2, 10)), ("7_10", (7, 10)))}
        pair_comparisons = {dose: {metric: {"2_10_minus_2_7": 0.0, "2_10_minus_7_10": 0.0}
                                   for metric in metrics} for dose in ("low", "high")}
        families.append({"group": group + 1, "arms": arms, "interactions": interactions,
                         "total_match_comparisons": match, "total_match_logit_comparisons": match,
                         "pair_comparisons": pair_comparisons,
                         "dose_comparisons": {pair: {metric: 0.0 for metric in metrics}
                                              for pair in ("2_7", "2_10", "7_10")},
                         "clean_target_ranks": [1] * 4, "clean_target_top1": [True] * 4})
    by_metric = {metric: {name: 0.0 for name in audit.ARMS} for metric in metrics}
    return {"families": families, "mean_contrast_by_arm": by_metric,
            "mean_repeat_delta_by_arm": by_metric, "mean_shuffle_delta_by_arm": by_metric,
            "mean_interactions": {dose: {pair: {metric: 0.0 for metric in metrics}
                                          for pair in ("2_7", "2_10", "7_10")} for dose in ("low", "high")},
            "mean_pair_comparisons": {dose: {metric: {"2_10_minus_2_7": 0.0, "2_10_minus_7_10": 0.0}
                                              for metric in metrics} for dose in ("low", "high")},
            "mean_dose_comparisons": {pair: {metric: 0.0 for metric in metrics} for pair in ("2_7", "2_10", "7_10")},
            "mean_total_match_comparisons": {metric: {pair: {f"h{head}": 0.0 for head in heads}
                for pair, heads in (("2_7", (2, 7)), ("2_10", (2, 10)), ("7_10", (7, 10)))} for metric in metrics},
            "top1_clean": {"repeated": 8, "shuffled": 24},
            "top1_by_arm": {name: {"repeated": 8, "shuffled": 24} for name in audit.ARMS}}


def test_strict_replay_checks_every_recorded_metadata_and_summary():
    audit = _audit()
    cases, records, files, families, diagnostics = [], [], {}, [], []
    for group in range(8):
        family_cases, family_records, family_files = _saved_family_fixture(group)
        family, family_diagnostics = _recorded_metadata_fixture(family_cases, family_records)
        cases.extend(family_cases)
        records.extend(family_records)
        files.update(family_files)
        families.append(family)
        diagnostics.extend(family_diagnostics)
    result = {"schema": "distilgpt2-copy-head-pairs-dose-result-v1", "status": "complete",
              "submitted_forwards": 576, "model_call_returns_observed": 576, "returned_forwards": 576,
              "raw_payload_bytes": sum(len(value) for value in files.values()), "cases": records,
              "family_doses": families, "case_diagnostics": diagnostics, "summary": _zero_runner_summary(audit)}
    report = audit.replay_result(cases, result, files.__getitem__, vocabulary_size=2, verify_metadata=True)
    assert len(report["families"]) == 8
    result["summary"]["families"][0]["interactions"]["low"]["2_10"]["target_logit"] = 1.0
    with pytest.raises(ValueError, match="summary"):
        audit.replay_result(cases, result, files.__getitem__, vocabulary_size=2, verify_metadata=True)


def _journal_fixture():
    audit = _audit()
    events = [{"event": "runtime_ready", "cpu_rng_seed": 20261029, "actual_runtime": {"device": "cpu"}}]
    result = {"runtime": {"device": "cpu"}, "cases": [], "family_doses": [], "case_diagnostics": []}
    for group in range(8):
        def action(variant, name):
            case_id = f"copy-pairs-{group + 1:02d}-{variant}"
            record = {"case_id": case_id, "action": name, "raw": {"logits": {"offset": len(result["cases"])}}}
            result["cases"].append(record)
            events.extend(({"event": "start", "case_id": case_id, "action": name},
                           {"event": "raw_recorded", "case_id": case_id, "action": name, "raw": copy.deepcopy(record["raw"])},
                           {"event": "complete", **copy.deepcopy(record)}))
        for variant in range(4):
            action(variant, "clean")
            action(variant, "noop")
        family = {"group": group + 1, "m": 2.0}
        result["family_doses"].append(family)
        events.append({"event": "family_dose", **family})
        for variant in range(4):
            shift = (4 * group + variant) % len(audit.ARMS)
            for name in audit.ARMS[shift:] + audit.ARMS[:shift] + ("clean_after",):
                action(variant, name)
            diagnostic = {"case_id": f"copy-pairs-{group + 1:02d}-{variant}", "ok": True}
            result["case_diagnostics"].append(diagnostic)
            events.append({"event": "case_diagnostics", **diagnostic})
    def encoded():
        return b"".join((json.dumps(event, sort_keys=True) + "\n").encode() for event in events)
    return result, events, encoded


def test_journal_reconciles_every_start_raw_completion_and_diagnostic():
    audit = _audit()
    result, events, encoded = _journal_fixture()
    assert audit.audit_journal(result, encoded()) == 1769
    events[2]["raw"]["logits"]["offset"] = 999
    with pytest.raises(ValueError, match="journal"):
        audit.audit_journal(result, encoded())
    events[2]["raw"]["logits"]["offset"] = 0
    events.append({"event": "failed"})
    with pytest.raises(ValueError, match="journal"):
        audit.audit_journal(result, encoded())


def test_journal_rejects_a_result_field_that_overrides_the_completion_event():
    audit = _audit()
    result, events, encoded = _journal_fixture()
    result["cases"][0]["event"] = "failed"
    events[3]["event"] = "failed"
    with pytest.raises(ValueError, match="journal"):
        audit.audit_journal(result, encoded())
