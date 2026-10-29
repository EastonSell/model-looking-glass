"""Private development-stage checks with a fabricated CPU model only."""
import hashlib
import json

import pytest
import torch

from test_targeted_edit_projected_capture import TinyModel
from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_projected_development as development
from looking_glass import targeted_edit_projected_math as projected
from looking_glass import targeted_edit_v2_owner as owner


def guarded_model():
    torch.set_flush_denormal(False)
    model = TinyModel()
    model.eval()
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[:6].copy_(torch.eye(6))
        model.embedding.weight[3, 1] = 1
    guard = owner._ModelGuard(model, model.projection, model.projection.weight, torch, v1)
    return model, guard


def test_development_uses_only_frozen_development_rows_and_preserves_raw_evidence(tmp_path):
    model, guard = guarded_model()
    original = guard.original.clone()
    calls = []
    forward = model.forward

    def record(**kwargs):
        calls.append(tuple(kwargs["input_ids"][0].tolist()))
        return forward(**kwargs)

    model.forward = record
    output = tmp_path / "attempt"
    candidate = development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, output)
    assert calls == [(3,), (1,), (2,), (3,)] + [(1,), (2,), (3,)] * len(projected.CONDITIONS)
    assert candidate["status"] == "private-candidate"
    expected_probe = {"subnormal_times_one_bits": 1, "normal_times_half_bits": 0x00400000,
                      "numpy_cast_preserves_min_subnormal": True}
    assert candidate["discovery_denormal_probe_before"] == expected_probe
    assert candidate["discovery_denormal_probe_after"] == expected_probe
    saved_candidate = json.loads((output / "pre-evaluation-candidate.json").read_bytes())
    assert saved_candidate["discovery_denormal_probe_before"] == expected_probe
    assert saved_candidate["discovery_denormal_probe_after"] == expected_probe
    assert candidate["discovery_token_ids"] == [3]
    assert candidate["calibration_token_ids"] == [[1], [2]]
    assert (candidate["target_token_id"], candidate["competitor_token_id"]) == (3, 1)
    assert (candidate["model_state_sha256"], candidate["weight_sha256"], candidate["buffer_state_sha256"]) == (
        guard.state_sha, guard.weight_sha, guard.buffer_sha
    )
    assert candidate["runtime_settings"] == guard.runtime_settings
    assert candidate["calibration_shifts"]["projected"]["max_output_abs"] < 1e-7
    assert candidate["calibration_shifts"]["unprojected"]["max_output_abs"] > 1e-6
    assert set(candidate["discovery_changes"]) == set(projected.CONDITIONS)
    for name in projected.CONDITIONS:
        assert len(candidate["calibration_shifts"][name]["max_output_abs_by_token"]) == 2
        assert len(candidate["calibration_shifts"][name]["max_algebraic_abs_by_token"]) == 2
        assert candidate["calibration_shifts"][name]["max_algebraic_abs"] == (
            candidate["construction"]["conditions"][name]["calibration_algebraic_max_abs"]
        )
    assert (output / "pre-evaluation-candidate.json").is_file()
    assert not (output / "terminal.json").exists()
    for name, record in candidate["artifacts"].items():
        data = (output / name).read_bytes()
        assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
    for condition in ("original", *projected.CONDITIONS):
        state = json.loads((output / ("state-" + condition + ".json")).read_bytes())
        assert state["condition"] == condition and state["cleanup"] == "verified"
        assert state["weight_sha256"] == guard.weight_sha
        assert state["addition_denormal_probe_before"] == state["addition_denormal_probe_after"]
        forwards = state["forward_denormal_probes"]
        assert len(forwards["calibration"]) == 2
        assert all(probe == {"before": expected_probe, "after": expected_probe}
                   for probe in (*forwards["calibration"], forwards["discovery"]))
    assert not any(name.startswith("weight") for name in candidate["artifacts"])
    assert not guard.poisoned and torch.equal(model.projection.weight, original)
    guard.verify()


def test_gradient_and_no_grad_baselines_must_match_before_edits(tmp_path):
    model, guard = guarded_model()
    original = guard.original.clone()
    forward = model.forward

    def mode_dependent(**kwargs):
        result = forward(**kwargs)
        if not torch.is_grad_enabled():
            offset = torch.zeros_like(result.logits)
            offset[..., 3] = 1
            result.logits = result.logits + offset
        return result

    model.forward = mode_dependent
    output = tmp_path / "incompatible"
    with pytest.raises(development.DevelopmentError, match="gradient and no-grad"):
        development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, output)
    assert guard.poisoned and torch.equal(model.projection.weight, original)
    assert (output / "discovery-logits.f32").is_file()
    assert not (output / "state-original.json").exists()
    assert not (output / "pre-evaluation-candidate.json").exists()


def test_changed_calibration_input_leaves_fenced_partial_attempt(monkeypatch, tmp_path):
    model, guard = guarded_model()
    original = guard.original.clone()
    actual_capture = development.capture.capture_calibration
    calls = []

    def tamper(*args):
        calls.append(True)
        result = actual_capture(*args)
        if len(calls) == 3:
            result["inputs"] = result["inputs"].clone()
            result["inputs"][0, 0] += 1
        return result

    monkeypatch.setattr(development.capture, "capture_calibration", tamper)
    output = tmp_path / "partial"
    with pytest.raises(development.DevelopmentError, match="inputs changed"):
        development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, output)
    assert len(calls) == 4 and guard.poisoned
    assert torch.equal(model.projection.weight, original)
    assert (output / "state-original.json").is_file()
    assert not (output / "state-projected.json").exists()
    assert not (output / "pre-evaluation-candidate.json").exists()


def test_interrupted_condition_restores_but_never_continues(monkeypatch, tmp_path):
    model, guard = guarded_model()
    original = guard.original.clone()
    actual_capture = development.capture.capture_calibration
    calls = []

    def interrupt(*args):
        calls.append(True)
        if len(calls) == 3:
            raise KeyboardInterrupt()
        return actual_capture(*args)

    monkeypatch.setattr(development.capture, "capture_calibration", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, output)
    assert len(calls) == 3 and guard.poisoned and torch.equal(model.projection.weight, original)
    assert not (output / "state-projected.json").exists()
    assert not (output / "pre-evaluation-candidate.json").exists()


def test_discovery_hook_removal_failure_fences_guard(monkeypatch, tmp_path):
    model, guard = guarded_model()
    register = model.projection.register_forward_pre_hook

    def register_leaking(*args, **kwargs):
        handle = register(*args, **kwargs)

        def fail_remove():
            raise OSError("injected hook removal failure")

        handle.remove = fail_remove
        return handle

    monkeypatch.setattr(model.projection, "register_forward_pre_hook", register_leaking)
    with pytest.raises(OSError, match="hook removal"):
        development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, tmp_path / "leaked")
    assert model.projection._forward_pre_hooks and guard.poisoned
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: None)


def test_construction_failure_after_baseline_commit_fences_guard(monkeypatch, tmp_path):
    model, guard = guarded_model()

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected construction failure")

    monkeypatch.setattr(projected, "prepare_projected_edits", fail)
    output = tmp_path / "construction-failure"
    with pytest.raises(RuntimeError, match="construction failure"):
        development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, output)
    assert (output / "state-original.json").is_file()
    assert guard.poisoned and torch.equal(model.projection.weight, guard.original)
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: None)


def test_interrupted_discovery_fences_guard(monkeypatch, tmp_path):
    _model, guard = guarded_model()

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(development.capture, "capture_discovery", interrupt)
    with pytest.raises(KeyboardInterrupt):
        development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, tmp_path / "interrupted-discovery")
    assert guard.poisoned
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: None)


@pytest.mark.parametrize("flip_at", (2, 3, 4, 5, 6, 7))
def test_each_nograd_forward_must_preserve_denormal_mode(tmp_path, flip_at):
    model, guard = guarded_model()
    forward = model.forward
    count = 0

    def flip(**kwargs):
        nonlocal count
        count += 1
        result = forward(**kwargs)
        if count == flip_at:
            torch.set_flush_denormal(True)
        return result

    model.forward = flip
    output = tmp_path / ("flush-at-" + str(flip_at))
    try:
        with pytest.raises(v1.StudyError, match="denormal"):
            development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, output)
        assert count == flip_at and guard.poisoned
        assert not (output / "pre-evaluation-candidate.json").exists()
    finally:
        torch.set_flush_denormal(False)
