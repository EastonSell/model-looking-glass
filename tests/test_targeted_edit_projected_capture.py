"""Fabricated CPU models exercise capture; no checkpoint or study cases."""
from types import SimpleNamespace
import sys

import pytest
import torch

from looking_glass import targeted_edit_projected_capture as capture
from looking_glass import targeted_edit_projected_math as projected


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-1, 1, 24).reshape(6, 4))

    def forward(self, inputs):
        return inputs @ self.weight


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 6)
        self.projection = Projection()
        self.head = torch.nn.Parameter(torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(48, dtype=torch.float32).reshape(8, 6) / 10)

    def forward(self, *, input_ids, attention_mask, use_cache):
        assert use_cache is False and torch.equal(attention_mask, torch.ones_like(input_ids))
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)) @ self.head)


def test_discovery_captures_full_gradient_and_restores_flags_and_hooks():
    model = TinyModel()
    model.embedding.weight.requires_grad_(False)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    weights = [parameter.detach().clone() for parameter in model.parameters()]
    result = capture.capture_discovery(model, model.projection, (1, 2, 3), 3, 1)
    h = model.embedding.weight[3].detach()
    g = (model.head[:, 3] - model.head[:, 1]).detach()
    assert torch.equal(result["input_factor"], h)
    assert torch.allclose(result["output_factor"], g, atol=1e-7, rtol=0)
    assert torch.allclose(result["gradient"], torch.outer(h, g), atol=1e-6, rtol=0)
    assert result["margin"] == pytest.approx(
        float(result["logits"][3].to(torch.float64) - result["logits"][1].to(torch.float64))
    )
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert all(parameter.grad is None and torch.equal(parameter, original)
               for parameter, original in zip(model.parameters(), weights))
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


def test_calibration_keeps_all_positions_and_their_order():
    model = TinyModel()
    rows = ((1, 2), (3, 4, 5), (6,), (7, 0))
    result = capture.capture_calibration(model, model.projection, rows)
    expected_inputs = model.embedding.weight[torch.tensor([1, 2, 3, 4, 5, 6, 7, 0])].detach()
    expected_outputs = expected_inputs @ model.projection.weight.detach()
    assert torch.equal(result["inputs"], expected_inputs)
    assert torch.equal(result["outputs"], expected_outputs)
    assert torch.equal(result["logits"], expected_outputs[[1, 4, 5, 7]] @ model.head.detach())
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


def test_fabricated_capture_drives_the_frozen_projection_construction():
    model = TinyModel()
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[:6].copy_(torch.eye(6))
        model.embedding.weight[3, 1] = 1
    discovery = capture.capture_discovery(model, model.projection, (3,), 3, 1)
    baseline = capture.capture_calibration(model, model.projection, ((1,), (2,)))
    weight = model.projection.weight.detach().clone()
    result = projected.prepare_projected_edits(
        weight, baseline["inputs"], discovery["input_factor"],
        discovery["output_factor"], discovery["gradient"],
    )
    measured = {}
    try:
        for name in ("projected", "unprojected"):
            with torch.no_grad():
                model.projection.weight.copy_(weight + result["updates"][name])
            observed = capture.capture_calibration(model, model.projection, ((1,), (2,)))
            assert torch.equal(observed["inputs"], baseline["inputs"])
            measured[name] = float(torch.max(torch.abs(observed["outputs"] - baseline["outputs"])))
    finally:
        with torch.no_grad():
            model.projection.weight.copy_(weight)
    assert measured["projected"] < 1e-7
    assert measured["unprojected"] > 1e-6
    assert torch.equal(model.projection.weight, weight)


def test_preexisting_gradients_are_preserved_and_rejected():
    model = TinyModel()
    model.head.grad = torch.ones_like(model.head)
    with pytest.raises(capture.CaptureError, match="preexisting"):
        capture.capture_discovery(model, model.projection, (1,), 1, 2)
    assert torch.equal(model.head.grad, torch.ones_like(model.head))
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


def test_repeated_site_invocation_fails_and_removes_hooks():
    model = TinyModel()
    original = model.forward

    def twice(**kwargs):
        first = original(**kwargs)
        original(**kwargs)
        return first

    model.forward = twice
    flags = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(capture.CaptureError, match="unexpected"):
        capture.capture_discovery(model, model.projection, (1, 2), 1, 2)
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


def test_failed_calibration_forward_removes_hooks():
    model = TinyModel()

    def fail(**_kwargs):
        raise RuntimeError("synthetic forward failure")

    model.forward = fail
    with pytest.raises(RuntimeError, match="synthetic"):
        capture.capture_calibration(model, model.projection, ((1,),))
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


def test_interrupted_discovery_propagates_after_best_effort_cleanup():
    model = TinyModel()
    flags = [parameter.requires_grad for parameter in model.parameters()]

    def interrupt(**kwargs):
        model.projection(model.embedding(kwargs["input_ids"]))
        raise KeyboardInterrupt()

    model.forward = interrupt
    with pytest.raises(KeyboardInterrupt):
        capture.capture_discovery(model, model.projection, (1, 2), 1, 2)
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


def test_failure_of_second_hook_registration_removes_the_first(monkeypatch):
    model = TinyModel()

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic registration failure")

    monkeypatch.setattr(model.projection, "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="registration"):
        capture.capture_discovery(model, model.projection, (1, 2), 1, 2)
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


def _invoke(model, phase):
    if phase == "discovery":
        return capture.capture_discovery(model, model.projection, (1, 2), 1, 2)
    return capture.capture_calibration(model, model.projection, ((1, 2),))


@pytest.mark.parametrize("phase", ["discovery", "calibration"])
def test_cleanup_does_not_depend_on_python_311_exception_api(monkeypatch, phase):
    model = TinyModel()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    with monkeypatch.context() as scoped:
        scoped.delattr(sys, "exception", raising=False)
        result = _invoke(model, phase)
    assert result and flags == [parameter.requires_grad for parameter in model.parameters()]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


@pytest.mark.parametrize("phase", ["discovery", "calibration"])
def test_outer_exception_cannot_hide_a_hook_removal_failure(monkeypatch, phase):
    model = TinyModel()
    register = model.projection.register_forward_pre_hook
    original_removers = []

    def failing_registration(*args, **kwargs):
        handle = register(*args, **kwargs)
        original_removers.append(handle.remove)

        def fail_without_remove():
            raise OSError("synthetic hook removal failure")

        handle.remove = fail_without_remove
        return handle

    monkeypatch.setattr(model.projection, "register_forward_pre_hook", failing_registration)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    try:
        raise LookupError("unrelated outer exception")
    except LookupError:
        with pytest.raises(OSError, match="hook removal"):
            _invoke(model, phase)
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert len(model.projection._forward_pre_hooks) == 1 and not model.projection._forward_hooks
    original_removers[0]()
    assert not model.projection._forward_pre_hooks


@pytest.mark.parametrize("phase", ["discovery", "calibration"])
def test_new_cleanup_cancellation_takes_precedence_over_forward_failure(monkeypatch, phase):
    model = TinyModel()
    register = model.projection.register_forward_pre_hook

    def cancelling_registration(*args, **kwargs):
        handle = register(*args, **kwargs)
        remove = handle.remove

        def cancel_after_remove():
            remove()
            raise KeyboardInterrupt()

        handle.remove = cancel_after_remove
        return handle

    def fail(**_kwargs):
        raise RuntimeError("synthetic forward failure")

    flags = [parameter.requires_grad for parameter in model.parameters()]
    monkeypatch.setattr(model.projection, "register_forward_pre_hook", cancelling_registration)
    model.forward = fail
    with pytest.raises(KeyboardInterrupt):
        _invoke(model, phase)
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


def test_finite_logits_with_overflowing_float32_objective_are_rejected():
    model = TinyModel()
    forward = model.forward

    def extreme(**kwargs):
        result = forward(**kwargs)
        offset = torch.tensor([3e38, -3e38, 0, 0, 0], dtype=torch.float32)
        return SimpleNamespace(logits=result.logits + offset)

    model.forward = extreme
    flags = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(capture.CaptureError, match="objective is nonfinite"):
        capture.capture_discovery(model, model.projection, (1, 2), 0, 1)
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks


@pytest.mark.parametrize("ids", [(), (-1,), (True,), (1.5,)])
def test_invalid_ids_fail_before_a_model_call(ids):
    model = TinyModel()
    with pytest.raises(capture.CaptureError):
        capture.capture_discovery(model, model.projection, ids, 1, 2)
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks
