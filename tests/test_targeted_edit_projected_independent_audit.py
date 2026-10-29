"""Model-free tests for independent projected-study evidence inspection."""
import importlib
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

import pytest


def load_auditor():
    name = "looking_glass.targeted_edit_projected_independent_audit"
    assert importlib.util.find_spec(name) is not None, "saved-output auditor has not been implemented"
    return importlib.import_module(name)


def test_float32_bit_decoder_preserves_signed_zero_and_subnormals():
    audit = load_auditor()
    assert audit.decode_float32_bits(0x00000001) == 2.0 ** -149
    assert audit.decode_float32_bits(0x80000001) == -(2.0 ** -149)
    assert math.copysign(1.0, audit.decode_float32_bits(0x80000000)) == -1.0
    assert audit.decode_float32_bits(0x7F7FFFFF) == (2.0 - 2.0 ** -23) * 2.0 ** 127
    for invalid in (0x7F800000, 0xFF800000, 0x7FC00001):
        with pytest.raises(audit.EvidenceError):
            audit.decode_float32_bits(invalid)


def test_shifted_log_softmax_keeps_normalizer_for_large_tied_logits():
    audit = load_auditor()
    raw = struct.pack("<III", 0x5D800000, 0x5D800000, 0x5D000000)
    values, logp, top = audit.score_row(raw)
    assert values[:2] == [2.0 ** 60, 2.0 ** 60]
    assert top == 0
    assert logp[0] == pytest.approx(-math.log(2.0), abs=1e-15)
    assert logp[1] == pytest.approx(-math.log(2.0), abs=1e-15)
    with pytest.raises(audit.EvidenceError):
        audit.score_row(b"\x00\x00\x00")


def make_journal_fixture():
    conditions = ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance")
    case_ids = [f"case-{number}" for number in range(13)]
    probe = {"subnormal_times_one_bits": 1, "normal_times_half_bits": 4194304,
             "numpy_cast_preserves_min_subnormal": True}
    matrices, records = {}, []
    for ordinal, condition in enumerate(conditions):
        rows = [struct.pack("<I", ordinal * 100 + index) for index in range(13)]
        matrices[condition] = b"".join(rows)
        for index, (case_id, row) in enumerate(zip(case_ids, rows)):
            records.append({"condition": condition, "case_id": case_id, "index": index,
                            "offset": index * 4, "bytes": 4, "sha256": hashlib.sha256(row).hexdigest(),
                            "denormal_probe_before": probe, "denormal_probe_after": probe})
    def encoded(items):
        return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                        for row in items)
    return matrices, case_ids, records, encoded


def test_journal_requires_exact_condition_order_row_bytes_and_probe_types():
    audit = load_auditor()
    matrices, case_ids, records, encoded = make_journal_fixture()
    assert audit.verify_journal(encoded(records), matrices, case_ids, row_bytes=4) == 78
    for changed in (
        [records[1], records[0], *records[2:]],
        [{**records[0], "offset": 4}, *records[1:]],
        [{**records[0], "sha256": "0" * 64}, *records[1:]],
        [{**records[0], "denormal_probe_after": {**records[0]["denormal_probe_after"],
                                                "numpy_cast_preserves_min_subnormal": 1}}, *records[1:]],
        records[:-1],
    ):
        with pytest.raises(audit.EvidenceError):
            audit.verify_journal(encoded(changed), matrices, case_ids, row_bytes=4)


def test_numerical_decision_obeys_strict_eligibility_and_success_boundaries():
    audit = load_auditor()
    projected = {"originally_negative_targets": 3, "reversed_targets": 2,
                 "mean_target_change": 0.6, "all_top_preserved": True,
                 "mean_kl": 0.0, "max_kl": 0.0, "mean_nll_deterioration": 0.0}
    summaries = {"projected": projected, "projected_random": {"mean_target_change": 0.1},
                 "unprojected": {"mean_kl": 0.001}}
    candidate = {"discovery_margin": -1.0, "calibration_shifts": {
        "projected": {"max_output_abs": 1e-6}, "unprojected": {"max_output_abs": 1e-4}}}
    result = audit.numerical_decision(summaries, candidate)
    assert result["decision"] == "success"
    assert result["criteria"]["random_control_advantage"] is True  # equality passes
    changed = {**summaries, "projected": {**projected, "mean_target_change": 0.5}}
    assert audit.numerical_decision(changed, candidate)["decision"] == "negative"
    changed_candidate = {**candidate, "calibration_shifts": {
        **candidate["calibration_shifts"], "unprojected": {"max_output_abs": 1e-6}}}
    assert audit.numerical_decision(summaries, changed_candidate)["decision"] == "ineligible"


def make_state_fixture():
    conditions = ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance")
    runtime = {"float32_matmul_precision": "highest", "mkldnn_deterministic": False,
               "deterministic_warn_only": False, "intraop_threads": 1, "interop_threads": 1,
               "deterministic_algorithms": True, "mkldnn_enabled": True, "cpu_autocast_enabled": False,
               "inference_mode_enabled": False, "cuda_visibility": "-1", "cuda_initialized": False,
               "hf_hub_offline": "1", "transformers_offline": "1", "omp_num_threads": None,
               "mkl_num_threads": None}
    probe = {"subnormal_times_one_bits": 1, "normal_times_half_bits": 4194304,
             "numpy_cast_preserves_min_subnormal": True}
    controls = {name: {"realized_weight_sha256": f"{i:064x}", "intended_delta_sha256": f"{i + 10:064x}",
                       "realized_delta_norm": 1.0, "realized_to_planned": 1.0,
                       "realized_to_projected": 1.0} for i, name in enumerate(conditions[1:], 1)}
    candidate = {"model_state_sha256": "a" * 64, "weight_sha256": "b" * 64,
                 "buffer_state_sha256": "c" * 64, "runtime_settings": runtime,
                 "construction": {"planned_delta_norm": 1.0, "weight_norm": 1000.0,
                                  "relative_dose": 0.001, "conditions": controls}}
    manifest = {"original_model_state_sha256": "a" * 64, "original_weight_sha256": "b" * 64}
    states = {}
    for name in conditions:
        applied = candidate["weight_sha256"] if name == "original" else controls[name]["realized_weight_sha256"]
        states[name] = {"condition": name, "applied_weight_sha256": applied,
                        "model_state_sha256": candidate["model_state_sha256"],
                        "weight_sha256": candidate["weight_sha256"],
                        "buffer_state_sha256": candidate["buffer_state_sha256"], "runtime_settings": runtime,
                        "addition_denormal_probe_before": probe, "addition_denormal_probe_after": probe,
                        "cleanup": "verified"}
    return states, candidate, manifest


def test_each_condition_requires_restoration_and_the_correct_control_weight():
    audit = load_auditor()
    states, candidate, manifest = make_state_fixture()
    assert audit.verify_condition_states(states, candidate, manifest) == 6
    for key, value in (("cleanup", "unverified"), ("applied_weight_sha256", "0" * 64),
                       ("buffer_state_sha256", "0" * 64),
                       ("addition_denormal_probe_after", {"numpy_cast_preserves_min_subnormal": True})):
        changed = {**states, "nuisance": {**states["nuisance"], key: value}}
        with pytest.raises(audit.EvidenceError):
            audit.verify_condition_states(changed, candidate, manifest)
    changed = {**states, "opposite": {**states["opposite"], "runtime_settings": {
        **candidate["runtime_settings"], "intraop_threads": True}}}
    with pytest.raises(audit.EvidenceError):
        audit.verify_condition_states(changed, candidate, manifest)


def test_full_vocabulary_rows_independently_reproduce_a_successful_synthetic_case():
    audit = load_auditor()
    case_file = Path(__file__).resolve().parents[1] / "experiments/targeted_edit_projected_public_cases_v1.json"
    cases = json.loads(case_file.read_bytes())["tokenizer_preflight"]["cases"]
    row_size = 50257 * 4
    base_rows = []
    for index in range(13):
        row = bytearray(row_size)
        struct.pack_into("<f", row, (6342 if index < 4 else 100) * 4, 1.0 if index < 4 else 2.0)
        base_rows.append(bytes(row))
    base = b"".join(base_rows)
    projected = list(base_rows)
    for index in range(4):
        row = bytearray(projected[index])
        struct.pack_into("<f", row, 3576 * 4, 2.0)
        projected[index] = bytes(row)
    unprojected = list(base_rows)
    for index in range(4, 13):
        row = bytearray(unprojected[index])
        struct.pack_into("<f", row, 101 * 4, 1.5)
        unprojected[index] = bytes(row)
    matrices = {name: base for name in ("original", "projected_random", "opposite", "nuisance")}
    matrices.update({"projected": b"".join(projected), "unprojected": b"".join(unprojected)})
    candidate = {"discovery_margin": -1.0, "calibration_shifts": {
        "projected": {"max_output_abs": 1e-7}, "unprojected": {"max_output_abs": 1e-4}}}
    derived = audit.score_matrices(matrices, cases, candidate)
    assert derived["decision"] == "success"
    assert derived["conditions"]["projected"]["mean_target_change"] == 2.0
    assert derived["conditions"]["projected"]["reversed_targets"] == 4
    assert derived["conditions"]["projected"]["all_top_preserved"] is True
    assert derived["conditions"]["projected"]["mean_kl"] == 0.0
    assert derived["criteria"]["locality_kl_advantage"] is True


def test_record_comparison_allows_only_small_numeric_rounding_not_type_or_structure_changes():
    audit = load_auditor()
    expected = {"row": [1.0, True, 2, None]}
    audit.compare_record({"row": [1.0 + 1e-13, True, 2, None]}, expected)
    for changed in ({"row": [1.00001, True, 2, None]}, {"row": [1.0, 1, 2, None]},
                    {"row": [1.0, True, 2]}, {"row": [float("nan"), True, 2, None]}):
        with pytest.raises(audit.EvidenceError):
            audit.compare_record(changed, expected)


def test_calibration_residuals_use_every_saved_token_and_float32_bit():
    audit = load_auditor()
    original = struct.pack("<6I", 0, 0, 0, 0, 0, 0)
    edited = struct.pack("<6I", 0x3E800000, 1, 0, 0, 0xBF000000, 0)
    assert audit.calibration_token_maxima(original, edited, width=3) == [0.25, 0.5]
    subnormal = struct.pack("<I", 1)
    assert audit.calibration_token_maxima(b"\0" * 4, subnormal, width=1) == [2.0 ** -149]
    with pytest.raises(audit.EvidenceError):
        audit.calibration_token_maxima(original, edited[:-4], width=3)
    with pytest.raises(audit.EvidenceError):
        audit.calibration_token_maxima(b"\0" * 4, struct.pack("<I", 0x7F800000), width=1)


def make_phase_fixture():
    def canonical(value):
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _states, candidate, manifest = make_state_fixture()
    source = "1" * 40
    manifest_sha, candidate_sha, development_sha = "2" * 64, "3" * 64, "4" * 64
    imported = {"looking_glass/" + name + ".py": "6" * 64 for name in (
        "__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity", "pretrained",
        "targeted_edit_cpu_v1", "targeted_edit_v2_edits", "targeted_edit_v2_owner",
        "targeted_edit_projected_capture", "targeted_edit_projected_development",
        "targeted_edit_projected_math", "targeted_edit_projected_execution", "targeted_edit_projected_score")}
    manifest.update({"source_commit": source, "created_utc": "2026-01-01T00:00:00+00:00",
                     "checkpoint": {"asset_sha256": {"config.json": "5" * 64}},
                     "source_sha256": imported})
    start = {"phase": "evaluation", "source_commit": source, "checkout_commit": "9" * 40, "manifest_sha256": manifest_sha,
             "candidate_sha256": candidate_sha, "development_record_sha256": development_sha,
             "started_utc": "2026-01-01T00:01:00+00:00"}
    child = {"phase": "evaluation", "owner_pid": 31, "owner_run_id": "a" * 32,
             "manifest_sha256": manifest_sha, "candidate_sha256": candidate_sha,
             "model_disposed": True, "no_model_loaded": False,
             "post_work_checkpoint_assets": manifest["checkpoint"]["asset_sha256"],
             "runtime_settings": candidate["runtime_settings"],
             "runtime": {"python_version": "3.11.15", "platform": "Windows-10-10.0.26200-SP0",
                 "machine": "AMD64", "processor": "AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD",
                 "torch_version": "2.8.0+cu128", "numpy_version": "2.4.6", "transformers_version": "4.57.3",
                 "torch_build_sha256": "9835a6eed581f616d1e294cc42162b701425ea24bf4723678b8233e78f30e8c6",
                 "torch_cpu_capability": "AVX2",
                 "gpt2_source_sha256": "b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80",
                 "intraop_threads": 1, "interop_threads": 1, "deterministic_algorithms": True,
                 "mkldnn_enabled": True, "omp_num_threads": None, "mkl_num_threads": None,
                 "cuda_visibility": "disabled", "cuda_initialized": False, "bytecode_attested": False,
                 "denormal_probe": {"subnormal_times_one_bits": 1, "normal_times_half_bits": 4194304,
                                    "numpy_cast_preserves_min_subnormal": True}},
             "imported_sources": manifest["source_sha256"],
             "finished_utc": "2026-01-01T00:02:00+00:00"}
    record = {"version": "targeted-edit-projected-phase-v1", "status": "complete", "phase": "evaluation",
              "source_commit": source, "checkout_commit": "9" * 40, "manifest_sha256": manifest_sha,
              "started_sha256": hashlib.sha256(canonical(start)).hexdigest(),
              "child_result_sha256": hashlib.sha256(canonical(child)).hexdigest(),
              "owner_reaped": True, "job_drained": True,
              "supervisor": {"pid": 31, "run_id": "a" * 32, "exit_code": 0,
                             "clean": True, "timed_out": False, "reason": "normal"},
              "child": child, "finished_utc": "2026-01-01T00:03:00+00:00"}
    return record, start, child, manifest, candidate, manifest_sha, candidate_sha, development_sha


def test_failed_or_uncertain_phase_overrides_favorable_numeric_evidence():
    audit = load_auditor()
    record, start, child, manifest, candidate, manifest_sha, candidate_sha, development_sha = make_phase_fixture()
    kwargs = {"phase": "evaluation", "manifest_sha256": manifest_sha,
              "candidate_sha256": candidate_sha, "development_record_sha256": development_sha}
    audit.verify_phase_record(record, start, child, manifest, candidate, **kwargs)
    for changed in ({**record, "job_drained": False}, {**record, "owner_reaped": 1},
                    {**record, "supervisor": {**record["supervisor"], "exit_code": True}},
                    {**record, "supervisor": {**record["supervisor"], "pid": 32}},
                    {**record, "status": "incomplete"}):
        with pytest.raises(audit.EvidenceError):
            audit.verify_phase_record(changed, start, child, manifest, candidate, **kwargs)


def test_rehashed_checkout_and_runtime_contradictions_are_rejected():
    audit = load_auditor()
    record, start, child, manifest, candidate, manifest_sha, candidate_sha, development_sha = make_phase_fixture()
    kwargs = {"phase": "evaluation", "manifest_sha256": manifest_sha,
              "candidate_sha256": candidate_sha, "development_record_sha256": development_sha}
    changed_start = {**start, "checkout_commit": "0" * 40}
    changed_record = {**record, "started_sha256": hashlib.sha256(audit._canonical(changed_start)).hexdigest()}
    with pytest.raises(audit.EvidenceError):
        audit.verify_phase_record(changed_record, changed_start, child, manifest, candidate, **kwargs)
    for changed_runtime in ({**child["runtime"], "cuda_initialized": True},
                            {**child["runtime"], "python_version": "3.10.0"}):
        changed_child = {**child, "runtime": changed_runtime}
        changed_record = {**record, "child": changed_child,
                          "child_result_sha256": hashlib.sha256(audit._canonical(changed_child)).hexdigest()}
        with pytest.raises(audit.EvidenceError):
            audit.verify_phase_record(changed_record, start, changed_child, manifest, candidate, **kwargs)


def test_rehashed_exposure_receipt_must_match_manifest_and_snapshot():
    audit = load_auditor()
    cases = [{"token_ids": [number]} for number in range(13)]
    inventory = {"version": "v15", "records": [{"token_ids": case["token_ids"]} for case in cases]}
    inventory_raw = audit._canonical(inventory)
    inventory_sha = hashlib.sha256(inventory_raw).hexdigest()
    manifest = {"exposure": {"canonical_sha256": inventory_sha, "version": "v15", "known_complete_rows": [],
                             "record_count": 13, "projected_rows": 13, "known_relations": 0},
                "source_sha256": {"looking_glass/pretrained_exposures_v15.py": "a" * 64}}
    receipt = {"inventory_sha256": inventory_sha, "source_path": "looking_glass/pretrained_exposures_v15.py",
               "source_sha256": "a" * 64, "reviewed_commit": "b" * 40, "observed_utc": "2026-01-01T00:00:00+00:00"}
    receipt_raw = audit._canonical(receipt)
    started = {"public_exposure_file_sha256": inventory_sha, "public_exposure_attestation_sha256": hashlib.sha256(receipt_raw).hexdigest(),
               "current_public_exposure": {key: value for key, value in manifest["exposure"].items() if key != "known_complete_rows"},
               "started_utc": "2026-01-01T00:01:00+00:00"}
    files = {"public-exposure.json": inventory_raw, "public-exposure-attestation.json": receipt_raw}
    audit.verify_public_exposure(manifest, started, files, cases)
    for mutation in ({**receipt, "inventory_sha256": "0" * 64}, {**receipt, "source_sha256": "0" * 64}):
        changed_raw = audit._canonical(mutation)
        changed_started = {**started, "public_exposure_attestation_sha256": hashlib.sha256(changed_raw).hexdigest()}
        with pytest.raises(audit.EvidenceError):
            audit.verify_public_exposure(manifest, changed_started,
                {**files, "public-exposure-attestation.json": changed_raw}, cases)


def test_recorded_norm_must_match_the_prespecified_point_one_percent_dose():
    audit = load_auditor()
    states, candidate, manifest = make_state_fixture()
    candidate["construction"]["relative_dose"] = 0.01
    with pytest.raises(audit.EvidenceError):
        audit.verify_condition_states(states, candidate, manifest)


def test_development_logits_and_all_calibration_shifts_bind_the_saved_rows():
    audit = load_auditor()
    conditions = ("projected", "unprojected", "projected_random", "opposite", "nuisance")
    original_logits = struct.pack("<3f", 0.0, -1.0, 1.0)
    changed_logits = struct.pack("<3f", 0.5, -1.0, 1.0)
    original_outputs = struct.pack("<6f", *([0.0] * 6))
    changed_outputs = struct.pack("<6f", 0.25, 0, 0, 0, -0.5, 0)
    artifacts = {"discovery-logits.f32": original_logits, "discovery-original-logits.f32": original_logits,
                 "discovery-grad-inputs.f32": b"same inputs", "discovery-original-inputs.f32": b"same inputs",
                 "discovery-grad-outputs.f32": b"same outputs", "discovery-original-outputs.f32": b"same outputs",
                 "calibration-original-outputs.f32": original_outputs}
    candidate = {"discovery_margin": 1.0, "discovery_changes": {}, "calibration_shifts": {}}
    for name in conditions:
        artifacts["discovery-" + name + "-logits.f32"] = changed_logits
        artifacts["discovery-" + name + "-inputs.f32"] = b"same inputs"
        artifacts["calibration-" + name + "-outputs.f32"] = changed_outputs
        candidate["discovery_changes"][name] = {"margin": 1.5, "change": 0.5}
        candidate["calibration_shifts"][name] = {"max_output_abs": 0.5,
                                                   "max_output_abs_by_token": [0.25, 0.5]}
    audit.verify_development_measurements(candidate, artifacts, target=0, competitor=1, width=3)
    broken = {**candidate, "calibration_shifts": {**candidate["calibration_shifts"],
        "opposite": {"max_output_abs": 0.5, "max_output_abs_by_token": [0.5, 0.25]}}}
    with pytest.raises(audit.EvidenceError):
        audit.verify_development_measurements(broken, artifacts, target=0, competitor=1, width=3)
    with pytest.raises(audit.EvidenceError):
        audit.verify_development_measurements(candidate, {**artifacts, "discovery-nuisance-inputs.f32": b"changed"},
                                              target=0, competitor=1, width=3)


def test_bounded_ordinary_reader_rejects_oversized_files(tmp_path):
    audit = load_auditor()
    path = tmp_path / "saved.json"
    path.write_bytes(b"abcdef")
    assert audit.read_ordinary(path, 6) == b"abcdef"
    with pytest.raises(audit.EvidenceError):
        audit.read_ordinary(path, 5)


@pytest.mark.skipif(os.name != "nt", reason="real Windows junction check")
def test_reader_rejects_a_junction_before_opening_its_target(tmp_path, monkeypatch):
    audit = load_auditor()
    target = tmp_path / "fake-private"
    target.mkdir()
    (target / "record.json").write_bytes(b"{}")
    junction = tmp_path / "redirect"
    subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    opened = []
    original_open = Path.open
    def observed(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", observed)
    try:
        with pytest.raises(audit.EvidenceError):
            audit.read_ordinary(junction / "record.json", 100)
        assert opened == []
    finally:
        junction.rmdir()


def test_snapshot_rejects_unlisted_failure_and_rehashed_file_mutation(tmp_path):
    audit = load_auditor()
    (tmp_path / "candidate").mkdir()
    (tmp_path / "candidate" / "record.json").write_bytes(b"abc")
    inventory = {"candidate/record.json": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}
    assert audit.snapshot_files(tmp_path, inventory) == {"candidate/record.json": b"abc"}
    (tmp_path / "failure.json").write_bytes(b"{}")
    with pytest.raises(audit.EvidenceError):
        audit.snapshot_files(tmp_path, inventory)
    (tmp_path / "failure.json").unlink()
    (tmp_path / "candidate" / "record.json").write_bytes(b"xyz")
    with pytest.raises(audit.EvidenceError):
        audit.snapshot_files(tmp_path, inventory)


def test_canonical_json_rejects_duplicate_keys_and_nonfinite_values():
    audit = load_auditor()
    assert audit.parse_document(b'{"a":1}\n') == {"a": 1}
    for raw in (b'{"a":1,"a":1}\n', b'{"a":NaN}\n', b'{ "a": 1 }\n', b'{"a":1}'):
        with pytest.raises(audit.EvidenceError):
            audit.parse_document(raw)


def test_development_states_bind_all_five_controls_and_every_forward_probe():
    audit = load_auditor()
    evaluation_states, candidate, _manifest = make_state_fixture()
    probe = {"subnormal_times_one_bits": 1, "normal_times_half_bits": 4194304,
             "numpy_cast_preserves_min_subnormal": True}
    candidate.update({"discovery_denormal_probe_before": probe, "discovery_denormal_probe_after": probe})
    development_states = {}
    for name, state in evaluation_states.items():
        copied = {key: value for key, value in state.items() if key != "runtime_settings"}
        copied["forward_denormal_probes"] = {"calibration": [{"before": probe, "after": probe} for _ in range(4)],
                                               "discovery": {"before": probe, "after": probe}}
        development_states[name] = copied
    assert audit.verify_development_states(development_states, candidate) == 6
    broken = {**development_states, "projected_random": {**development_states["projected_random"],
              "forward_denormal_probes": {"calibration": [{"before": probe, "after": probe}] * 3,
                                           "discovery": {"before": probe, "after": probe}}}}
    with pytest.raises(audit.EvidenceError):
        audit.verify_development_states(broken, candidate)


def test_manifest_rejects_rehashed_wrong_checkpoint_or_development_inputs():
    audit = load_auditor()
    case_file = Path(__file__).resolve().parents[1] / "experiments/targeted_edit_projected_public_cases_v1.json"
    case_bytes = case_file.read_bytes()
    cases = json.loads(case_bytes)["tokenizer_preflight"]["cases"]
    manifest = {"version": "targeted-edit-projected-run-manifest-v1",
        "source_commit": "fc2d8bc12cea349763a310956f503ae540cd3d81",
        "cases_sha256": "f7554a81ac7143c54a56d557fb04cb19419bac30e7dcd44c951eb63ddaff8ee8",
        "evaluation_cases": cases,
        "original_model_state_sha256": "6a69517e85aeecec31aa49914e78d95e1615a674216ca82ab6ea82f555cf2c9f",
        "original_weight_sha256": "26b478a67c89b984ecc5c039fe5de9a481d2638e52ff91769d77f1d16b069263",
        "reference_sha256": "cb666aa911db020b8904dab6f08f989e506d85e875d7cac65e72d3b94cef2fa5",
        "development": {"discovery_ids": [464, 3139, 1748, 286, 4881, 318],
            "calibration_ids": [[464, 4387, 9151, 318, 262], [7571, 5556, 734, 21767],
                                [464, 6697, 286, 3024, 318], [32, 1285, 468, 3598]],
            "target_id": 3576, "competitor_id": 6342},
        "checkpoint": {"model_id": "distilbert/distilgpt2", "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
            "tokenizer_id": "distilbert/distilgpt2", "tokenizer_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
            "model_class": "GPT2LMHeadModel", "tokenizer_class": "GPT2TokenizerFast", "device": "cpu", "dtype": "float32",
            "evaluation_mode": True, "use_cache": False, "attention_implementation": "eager", "parameter_count": 81912576,
            "torch_version": "2.8.0+cu128", "transformers_version": "4.57.3",
            "weights_sha256": "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
            "asset_sha256": {"config.json": "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a",
                "generation_config.json": "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805",
                "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
                "model.safetensors": "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
                "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
                "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
                "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"}}}
    assert len(audit.verify_manifest(manifest, case_bytes)) == 13
    for changed in ({**manifest, "checkpoint": {**manifest["checkpoint"], "device": "cuda"}},
                    {**manifest, "development": {**manifest["development"], "target_id": 6342}},
                    {**manifest, "original_weight_sha256": "0" * 64}):
        with pytest.raises(audit.EvidenceError):
            audit.verify_manifest(changed, case_bytes)


def test_incomplete_phase_record_cannot_authorize_a_numerical_result(tmp_path):
    audit = load_auditor()
    for status in ("incomplete", "complete"):
        document = {"version": "targeted-edit-projected-phase-v1", "status": status,
                    "phase": "evaluation", "artifacts": {}}
        raw = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (tmp_path / "phase-record.json").write_bytes(raw)
        with pytest.raises(audit.EvidenceError):
            audit.read_phase_snapshot(tmp_path, hashlib.sha256(raw).hexdigest(), phase="evaluation")


def test_failed_phase_stops_before_a_favorable_numerical_result(monkeypatch, tmp_path):
    audit = load_auditor()
    manifest = b"{}\n"
    marker = []
    monkeypatch.setattr(audit, "verify_manifest", lambda _manifest, _cases: [])
    def snapshot(_path, _digest, *, phase):
        if phase == "evaluation":
            raise audit.EvidenceError("cleanup was not verified")
        return {}, {}
    monkeypatch.setattr(audit, "read_phase_snapshot", snapshot)
    monkeypatch.setattr(audit, "score_matrices", lambda *_args: marker.append("scored"))
    with pytest.raises(audit.EvidenceError):
        audit.audit_saved(manifest, b"cases", tmp_path, tmp_path,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(), candidate_sha256="1" * 64,
            development_record_sha256="2" * 64, evaluation_record_sha256="3" * 64)
    assert marker == []


def test_cli_does_not_emit_a_decision_when_saved_audit_fails(monkeypatch, capsysbinary):
    audit = load_auditor()
    monkeypatch.setattr(audit, "read_ordinary", lambda _path, _maximum: b"{}\n")
    def fail(*_args, **_kwargs):
        raise audit.EvidenceError("private path or underlying detail")
    monkeypatch.setattr(audit, "audit_saved", fail)
    arguments = ["--manifest", "x", "--cases", "x", "--development", "x", "--evaluation", "x",
                 "--manifest-sha256", "1" * 64, "--candidate-sha256", "2" * 64,
                 "--development-sha256", "3" * 64, "--evaluation-sha256", "4" * 64]
    assert audit.main(arguments) != 0
    result = json.loads(capsysbinary.readouterr().out)
    assert result == {"status": "incomplete", "error": "saved evidence could not be verified"}
    assert "decision" not in result
