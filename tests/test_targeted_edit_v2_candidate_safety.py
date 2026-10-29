"""Synthetic CPU checks for calling the unchanged historical candidate safely."""

from types import SimpleNamespace
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_v2_owner as owner


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)

    def forward(self, value):
        return value @ self.weight


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        self.projection = Projection()
        self.forwards = 0

    def forward(self, input_ids, attention_mask, use_cache):
        self.forwards += 1
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def candidate(model):
    return owner._candidate_v1_safely(v1, model, model.projection, (1, 2), 1, 2,
                                      expected_shape=(3, 4))


def assert_untouched(model, flags, weight):
    assert model.forwards == 0
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert torch.equal(model.projection.weight, weight)
    assert not model.projection._forward_hooks
    assert not model.projection._forward_pre_hooks


def test_missing_exception_api_rejects_before_model_mutation(monkeypatch):
    model = TinyModel()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    weight = model.projection.weight.detach().clone()
    with monkeypatch.context() as patch:
        patch.delattr(sys, "exception", raising=False)
        with pytest.raises(owner.OwnerError, match="exception"):
            candidate(model)
    assert_untouched(model, flags, weight)


@pytest.mark.parametrize("context", ["except", "finally"])
def test_ambient_exception_rejects_before_model_mutation(context):
    model = TinyModel()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    weight = model.projection.weight.detach().clone()
    if context == "except":
        try:
            raise ValueError("unrelated caller error")
        except ValueError:
            with pytest.raises(owner.OwnerError, match="exception"):
                candidate(model)
    else:
        with pytest.raises(ValueError, match="unrelated caller error"):
            try:
                raise ValueError("unrelated caller error")
            finally:
                with pytest.raises(owner.OwnerError, match="exception"):
                    candidate(model)
    assert_untouched(model, flags, weight)


def test_normal_context_preserves_v1_outputs_and_model_state():
    model = TinyModel()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    weight = model.projection.weight.detach().clone()
    expected_gradient, expected_random, expected_details = v1._candidate(
        model, model.projection, (1, 2), 1, 2, expected_shape=(3, 4))
    gradient, random, details = candidate(model)
    assert torch.equal(gradient, expected_gradient)
    assert torch.equal(random, expected_random)
    assert details == expected_details
    assert model.forwards == 2
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert torch.equal(model.projection.weight, weight)
    assert not model.projection._forward_hooks
    assert not model.projection._forward_pre_hooks


def test_hook_removal_failure_in_normal_context_propagates(monkeypatch):
    model = TinyModel()
    registered = []
    original_register = model.projection.register_forward_pre_hook

    class BrokenHandle:
        def remove(self):
            raise OSError("hook removal failed")

    def register(hook):
        registered.append(original_register(hook))
        return BrokenHandle()

    monkeypatch.setattr(model.projection, "register_forward_pre_hook", register)
    try:
        with pytest.raises(OSError, match="hook removal failed"):
            candidate(model)
        assert model.forwards == 1
        assert model.projection._forward_pre_hooks
    finally:
        for handle in registered:
            handle.remove()


def test_interrupted_candidate_restores_hooks_and_flags_before_propagating(monkeypatch):
    model = TinyModel()
    flags = [parameter.requires_grad for parameter in model.parameters()]

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("discovery interrupted")

    monkeypatch.setattr(model, "forward", interrupt)
    with pytest.raises(KeyboardInterrupt, match="discovery interrupted"):
        candidate(model)
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert not model.projection._forward_hooks
    assert not model.projection._forward_pre_hooks


def test_actual_python310_rejects_before_candidate_execution():
    launcher = shutil.which("py") if os.name == "nt" else None
    if launcher is None:
        pytest.skip("Windows Python launcher is not available")
    probe = subprocess.run([launcher, "-3.10", "-I", "-B", "-c", "pass"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if probe.returncode:
        pytest.skip("Python 3.10 is not installed")
    script = r"""
import importlib.util
import sys
from types import SimpleNamespace
assert sys.version_info[:2] == (3, 10)
assert not hasattr(sys, 'exception')
spec = importlib.util.spec_from_file_location('owner_under_test', sys.argv[1])
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
model = SimpleNamespace(calls=0, hooks=[], requires_grad=True)
def unsafe_candidate(model, *_args, **_kwargs):
    model.calls += 1
    model.hooks.append('installed')
    model.requires_grad = False
v1 = SimpleNamespace(_candidate=unsafe_candidate)
try:
    owner._candidate_v1_safely(v1, model, None, (1, 2), 1, 2)
except owner.OwnerError:
    pass
else:
    raise AssertionError('Python 3.10 was not rejected')
assert model.calls == 0 and model.hooks == [] and model.requires_grad is True
"""
    result = subprocess.run([launcher, "-3.10", "-I", "-B", "-c", script,
                             str(Path(owner.__file__).resolve())],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
