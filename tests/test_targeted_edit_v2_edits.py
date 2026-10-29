"""Synthetic edit math and provenance checks; no model is loaded."""

import copy

import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_edits as edits


@pytest.fixture
def tensors():
    assert torch.set_flush_denormal(False)
    generator = torch.Generator(device="cpu").manual_seed(11)
    original = torch.randn(16, 12, generator=generator, dtype=torch.float32)
    left = torch.randn(16, generator=generator, dtype=torch.float32)
    right = torch.randn(12, generator=generator, dtype=torch.float32)
    gradient = v1._scale_outer(left, right, 0.01 * v1._norm(original))
    return original, gradient


def test_grid_doses_and_controls_have_declared_realized_norms_and_are_repeatable(tensors):
    original, gradient = tensors
    before = v1._tensor_sha(original)
    for _, dose in v2.GRID:
        delta = edits.construct_dose(original, gradient, dose)
        info = edits.attest(original, delta, dose)
        assert info["planned_norm"] == dose * v1._norm(original)
        assert 0.999 <= info["realized_norm"] / info["planned_norm"] <= 1.001
    updates, facts = edits.construct_controls(original, gradient, 0.001)
    repeated, again = edits.construct_controls(original, gradient, 0.001)
    assert torch.equal(updates["opposite"].view(torch.int32), (-updates["selected"]).view(torch.int32))
    assert all(torch.equal(updates[name].view(torch.int32), repeated[name].view(torch.int32))
               for name in updates)
    assert facts == again
    assert v1._tensor_sha(original) == before
    for name in ("random", "opposite"):
        assert 0.999 <= facts[name]["realized_norm"] / facts["selected"]["realized_norm"] <= 1.001


def test_new_random_draw_uses_frozen_distinct_cpu_seed(tensors):
    original, gradient = tensors
    updates, _ = edits.construct_controls(original, gradient, 0.001)
    generator = torch.Generator(device="cpu").manual_seed(20261030)
    left = torch.randn(16, generator=generator, dtype=torch.float32)
    right = torch.randn(12, generator=generator, dtype=torch.float32)
    expected = v1._scale_outer(left, right, 0.001 * v1._norm(original))
    assert torch.equal(updates["random"].view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("mutation", ["zero_direction", "nan_direction", "shape", "bad_dose", "requires_grad", "wrong_planned"])
def test_invalid_tensors_or_unrealized_dose_are_rejected(tensors, mutation):
    original, gradient = tensors
    dose = 0.001
    if mutation == "zero_direction":
        gradient = torch.zeros_like(gradient)
    elif mutation == "nan_direction":
        gradient = gradient.clone()
        gradient[0, 0] = float("nan")
    elif mutation == "shape":
        gradient = gradient[:-1]
    elif mutation == "bad_dose":
        dose = 0.003
    elif mutation == "requires_grad":
        gradient = gradient.clone().requires_grad_(True)
    else:
        delta = edits.construct_dose(original, gradient, dose)
        with pytest.raises(v1.StudyError):
            edits.attest(original, delta, 0.004)
        return
    with pytest.raises(v1.StudyError):
        edits.construct_dose(original, gradient, dose)


def compatibility_fixture(monkeypatch, tensors):
    original, gradient = tensors
    monkeypatch.setattr(v1, "WEIGHT_SHAPE", (16, 12))
    generator = torch.Generator(device="cpu").manual_seed(v1.SEED)
    left = torch.randn(original.shape[0], generator=generator, dtype=torch.float32)
    right = torch.randn(original.shape[1], generator=generator, dtype=torch.float32)
    random = v1._scale_outer(left, right, 0.01 * v1._norm(original))
    environment = {field: f"value-{field}" for field in edits.ENVIRONMENT_FIELDS}
    environment["python_executable_path_sha256"] = "old-path-fingerprint"
    model = {"model_id": "synthetic-only"}
    discovery = {"discovery_logit_sha256": "3" * 64, "gradient_outer_relative_residual": 0.0}
    state = "4" * 64
    reference = {"version": v1.VERSION, "site": v1.SITE, "site_shape": [16, 12],
                 "environment": environment, "model": model,
                 "discovery": discovery, "model_state_sha256": state,
                 "weight_sha256": v1._tensor_sha(original),
                 "candidates": v1._candidate_attestation(original, {"gradient": gradient, "random": random})}
    raw = v1._canonical(reference)
    monkeypatch.setattr(edits, "V1_PRECOMMIT_SHA256", v1._sha(raw))
    current_environment = copy.deepcopy(environment)
    current_environment["python_executable_path_sha256"] = "new-path-fingerprint"
    args = {"environment": current_environment, "model_metadata": model,
            "model_state_sha256": state, "original": original,
            "v1_gradient": gradient, "v1_random": random, "discovery": discovery}
    return raw, args


def test_v1_compatibility_ignores_only_path_fingerprints(monkeypatch, tensors):
    raw, args = compatibility_fixture(monkeypatch, tensors)
    result = edits.verify_v1_compatibility(raw, **args)
    assert result["status"] == "matching_recorded_v1"
    assert result["model_state_sha256"] == args["model_state_sha256"]


@pytest.mark.parametrize("mutation", ["digest", "runtime", "missing_runtime", "model", "discovery", "state", "weight", "gradient", "random"])
def test_v1_compatibility_fails_closed_on_reconstruction_change(monkeypatch, tensors, mutation):
    raw, args = compatibility_fixture(monkeypatch, tensors)
    args = copy.copy(args)
    if mutation == "digest":
        raw += b" "
    elif mutation == "runtime":
        args["environment"] = {**args["environment"], "torch_version": "different"}
    elif mutation == "missing_runtime":
        args["environment"] = {key: value for key, value in args["environment"].items() if key != "torch_version"}
    elif mutation == "model":
        args["model_metadata"] = {"model_id": "different"}
    elif mutation == "discovery":
        args["discovery"] = {**args["discovery"], "discovery_logit_sha256": "5" * 64}
    elif mutation == "state":
        args["model_state_sha256"] = "5" * 64
    else:
        key = {"weight": "original", "gradient": "v1_gradient", "random": "v1_random"}[mutation]
        args[key] = args[key].clone()
        args[key][0, 0] += 0.001
    with pytest.raises(v1.StudyError):
        edits.verify_v1_compatibility(raw, **args)


def test_v1_compatibility_rejects_reshaped_tensors_with_identical_bytes(monkeypatch, tensors):
    raw, args = compatibility_fixture(monkeypatch, tensors)
    for name in ("original", "v1_gradient", "v1_random"):
        reshaped = args[name].reshape(12, 16)
        assert v1._tensor_sha(reshaped) == v1._tensor_sha(args[name])
        args[name] = reshaped
    with pytest.raises(v1.StudyError):
        edits.verify_v1_compatibility(raw, **args)
