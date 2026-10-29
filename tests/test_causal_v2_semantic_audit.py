"""Fabricated saved-data cases only; no model or study inputs are generated."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest

try:
    from looking_glass import causal_v2_semantic_audit as audit
except ImportError:
    audit = None


def _test_file(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binary32_contrast_preserves_a_smallest_subnormal():
    assert audit is not None, "the independent semantic auditor is missing"
    # min normal - max subnormal = min positive subnormal, exactly.
    assert audit.binary32_contrast([0x007FFFFF, 0x00800000]) == math.ldexp(1, -149)


def test_binary32_contrast_rounds_halfway_to_even():
    assert audit is not None, "the independent semantic auditor is missing"
    # 1 + 2^-24 is halfway to the next float; an even significand stays at 1.
    assert audit.binary32_contrast([0xBF800000, 0x33800000]) == 1.0
    # The next significand is odd, so its halfway result rounds upward.
    assert audit.binary32_contrast([0xBF800001, 0x33800000]) == 1.0 + math.ldexp(2, -23)


def test_saved_bit_equality_ignores_signed_zero_but_not_subnormals():
    assert audit is not None, "the independent semantic auditor is missing"
    assert audit.same_float32_bits([0x80000000, 1], [0, 1])
    assert not audit.same_float32_bits([0, 1], [0, 0])


@pytest.mark.parametrize("bits", ([0, 0x7F800000], [0, 0x7FC00000], [0, True], [0]))
def test_binary32_contrast_rejects_nonfinite_or_malformed_bits(bits):
    assert audit is not None, "the independent semantic auditor is missing"
    with pytest.raises(ValueError):
        audit.binary32_contrast(bits)


def test_binary32_contrast_rejects_overflow():
    assert audit is not None, "the independent semantic auditor is missing"
    with pytest.raises(ValueError):
        audit.binary32_contrast([0xFF7FFFFF, 0x7F7FFFFF])


@pytest.fixture(scope="module")
def fabricated_condition():
    # Existing test-only builder uses out-of-protocol seeds and values and
    # never executes a model or generates reserved cases.
    return json.loads(_test_file("test_causal_v2_scoring").fixture()[-1][0])["records"][0]


def test_condition_reconstructs_patch_and_reverse_effects(fabricated_condition):
    checked = audit.inspect_condition(fabricated_condition, target="block.1.branch.0",
                                      strength=0.25, polarity=1)
    assert checked["first_unequal"] == "block.1.output"
    assert checked["forward_effects"] == (-0.25, -0.25, -0.25)
    assert checked["reverse_effects"] == (0.25, 0.25, 0.25)
    assert checked["target_repair"] == -0.25


@pytest.mark.parametrize("mutation", ("null", "source", "target", "comparison", "metadata"))
def test_condition_rejects_corrupt_saved_controls(fabricated_condition, mutation):
    record = copy.deepcopy(fabricated_condition)
    if mutation == "null":
        record["logits"]["null"][0]["bits"][1] = 0
    elif mutation == "source":
        record["plans"][0]["captures"]["post_action"]["bits"][0] = 0x3F800000
    elif mutation == "target":
        record["private_integrity"]["perturbed_target"]["bits"][1] = 0x3F800000
    elif mutation == "comparison":
        record["comparison"]["first_unequal"] = "block.2.output"
    else:
        record["plans"][0]["metadata"]["steps"][0]["invocation"] = 1
    with pytest.raises(ValueError):
        audit.inspect_condition(record, target="block.1.branch.0", strength=0.25, polarity=1)


@pytest.fixture(scope="module")
def fabricated_run():
    return _test_file("causal_v2_semantic_fixture").build_fixture()


def test_full_saved_run_reconstructs_all_chunks_and_reports_scope(fabricated_run):
    report = audit.audit_evidence(**fabricated_run)
    assert report["status"] == "saved_data_consistent"
    assert report["discovery_chunks"] == 54
    assert report["heldout_chunks"] == 54
    assert report["conditions_per_phase"] == 1728
    assert report["reported_exposure"] == "unblinded"
    assert report["execution_or_chronology_proven"] is False
    assert report["case_generation_or_exclusions_proven"] is False
    assert report["secrecy_proven"] is False
    assert report["prediction_commitment_sha256"] == audit._sha(fabricated_run["commitment"])
    assert report["reveal_sha256"] == audit._sha(fabricated_run["reveal"])


def test_changed_discovery_tensor_is_rejected_even_with_canonical_chunk(fabricated_run):
    from looking_glass import registry
    changed = dict(fabricated_run)
    chunks = list(changed["discovery_chunks"])
    chunk = json.loads(chunks[0])
    chunk["records"][0]["traces"]["perturbed"][1]["value"]["bits"][1] = 0x3F800000
    chunks[0] = registry._json_bytes(chunk)
    changed["discovery_chunks"] = chunks
    with pytest.raises(ValueError):
        audit.audit_evidence(**changed)


def test_rehashed_but_wrong_aggregate_is_rejected(fabricated_run):
    from looking_glass.causal_v2_custody import canonical_json
    changed = dict(fabricated_run)
    results = json.loads(changed["results"])
    results["equal_model_means"]["selection"]["first_changed"]["block_hit_rate"] = 0.123
    changed["results"] = canonical_json(results)
    with pytest.raises(ValueError):
        audit.audit_evidence(**changed)


def test_missing_last_heldout_chunk_is_not_partial_success(fabricated_run):
    changed = dict(fabricated_run)
    changed["heldout_chunks"] = changed["heldout_chunks"][:-1]
    with pytest.raises(ValueError):
        audit.audit_evidence(**changed)


def test_saved_reader_rejects_a_missing_or_incomplete_attempt(tmp_path):
    with pytest.raises((ValueError, OSError)):
        audit.audit_run(tmp_path / "missing")
    incomplete = tmp_path / "incomplete"
    (incomplete / "public").mkdir(parents=True)
    (incomplete / "private").mkdir()
    with pytest.raises((ValueError, OSError)):
        audit.audit_run(incomplete)


def test_saved_reader_rejects_hard_linked_registry_files(tmp_path):
    run = tmp_path / "hardlinked-run"
    public = run / "public"
    artifacts = public / "artifacts"
    artifacts.mkdir(parents=True)
    (run / "private").mkdir()
    for name in (".lock", "manifest.json", "outcome.json"):
        (public / name).write_bytes(b"x")
    os.link(public / ".lock", tmp_path / "outside-alias")
    with pytest.raises(ValueError, match="hard link"):
        audit.audit_run(run)


def test_saved_reader_rejects_uninventoried_root_files(tmp_path):
    run = tmp_path / "unexpected-root"
    run.mkdir()
    (run / "other.json").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="root"):
        audit.audit_run(run)


def test_saved_reader_rejects_rehashed_wrong_required_declarations(tmp_path):
    from looking_glass import registry
    root = tmp_path / "wrong-declarations"
    root.mkdir()
    (root / "private").mkdir()
    names = ["pre_manifest.json", "prediction_freeze.json", "prediction_commitment.json",
             "results.json", "reveal.json"]
    names += [f"source_inventory_{stage}.json" for stage in
              ("before_discovery", "after_discovery", "after_heldout", "before_success")]
    limits = {name: 1 for name in names}
    limits["failure.json"] = 16384
    run = registry.create_run(root / "public", protocol={}, inputs={}, model={}, code={}, metadata={},
        artifact_limits=limits, required_artifacts=names[:-1], max_total_bytes=sum(limits.values()))
    for name in names:
        run.write_artifact(name, b"x")
    run.finalize("succeeded", {})
    with pytest.raises(ValueError, match="declarations"):
        audit.audit_run(root)


def test_saved_reader_checks_registry_and_all_persisted_artifacts(fabricated_run, tmp_path):
    run = _test_file("causal_v2_semantic_fixture").save_fixture(tmp_path / "fabricated-run", fabricated_run)
    report = audit.audit_run(run)
    assert report["status"] == "saved_data_consistent"
    assert report["discovery_chunks"] == report["heldout_chunks"] == 54
    assert report["execution_or_chronology_proven"] is False
    assert len(report["public_outcome_sha256"]) == 64
    assert len(report["private_outcome_sha256"]) == 64

    chunk = run / "private" / "artifacts" / "cell-53-h.json"
    saved = chunk.read_bytes()
    try:
        chunk.write_bytes(saved + b" ")
        with pytest.raises((ValueError, OSError)):
            audit.audit_run(run)
    finally:
        chunk.write_bytes(saved)
    pending = run / "private" / ".pending"
    try:
        pending.write_bytes(b"incomplete")
        with pytest.raises((ValueError, OSError)):
            audit.audit_run(run)
    finally:
        pending.unlink()


def test_cli_rejects_incomplete_run_without_a_traceback(tmp_path):
    process = subprocess.run([sys.executable, "-m", "looking_glass.causal_v2_semantic_audit",
                              str(tmp_path / "missing")], capture_output=True, text=True, check=False)
    assert process.returncode != 0
    assert not process.stdout
    assert "incomplete" in process.stderr
    assert "Traceback" not in process.stderr


def test_all_abstentions_remain_missing_metrics_and_zero_coverage():
    cells = [{"selected": {"first_changed": None}, "true_block": 1,
              "site_metrics": [], "site_rows": []} for _ in range(9)]
    result = audit._policy_metrics(cells, "first_changed")
    assert result["coverage"] == 0
    assert result["conditional_on_selection"] is True
    assert result["polarity_mae"] is result["constant_mae"] is result["zero_mae"] is None
    assert result["sign"]["eligible"] == 0
    assert result["sign"]["polarity_accuracy"] is None


def test_sign_threshold_is_strict_and_near_zero_prediction_is_wrong():
    rows = [{"effect": effect, "polarity_prediction": prediction, "constant_prediction": 0.0}
            for effect, prediction in ((1e-7, 1.0), (1e-6, 1e-7), (-2e-6, -3e-6))]
    result = audit._sign_metrics(rows)
    assert result["eligible"] == 2
    assert result["excluded"] == 1
    assert result["sign_variation_tested"] is True
    assert result["polarity_accuracy"] == 0.5
    assert result["constant_accuracy"] == 0.0


def test_uniform_site_baseline_includes_all_sites_when_selectors_abstain():
    metrics = [{"polarity_mae": float(site), "constant_mae": float(site + 3),
                "zero_mae": float(site + 4)} for site in range(3)]
    cells = [{"model_seed": seed, "selected": {"first_changed": None, "magnitude": None},
              "true_block": 1, "site_metrics": metrics, "site_rows": [[], [], []],
              "observation_misses": 32} for seed in range(6) for _ in range(9)]
    freeze = {"cells": [{"selection": {"no_injection": {"cases": 16}}} for _ in cells]}
    models, equal = audit._summaries(cells, freeze)
    assert all(model["selection"]["first_changed"]["coverage"] == 0 for model in models)
    assert equal["selection"]["first_changed"]["polarity_mae"] is None
    assert equal["random"]["polarity_mae"] == 1.0
    assert equal["random"]["constant_mae"] == 4.0
    assert equal["random"]["zero_mae"] == 5.0
    assert equal["random"]["localization_hit_probability"] == 1 / 3
