"""Synthetic regression checks for the v1-compatible candidate entry point."""

from types import SimpleNamespace
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as historical
from looking_glass import targeted_edit_candidate_v1_1 as successor


candidate = successor.derive_candidate


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)

    def forward(self, values):
        return values @ self.weight


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        self.projection = Projection()

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def _model():
    torch.random.default_generator.manual_seed(123)
    model = TinyModel().eval()
    model.embedding.weight.requires_grad_(False)
    return model


def _derive(model):
    return candidate(model, model.projection, (1, 2), 1, 2, expected_shape=(3, 4))


@pytest.mark.parametrize("ambient", [False, True])
def test_hook_removal_failure_is_not_suppressed_by_a_caller_exception(monkeypatch, ambient):
    model = _model()
    module = model.projection
    flags = [parameter.requires_grad for parameter in model.parameters()]
    original_register = module.register_forward_pre_hook
    real_handles = []

    class BrokenHandle:
        def remove(self):
            raise OSError("hook removal failed")

    def register(hook):
        real_handles.append(original_register(hook))
        return BrokenHandle()

    monkeypatch.setattr(module, "register_forward_pre_hook", register)
    try:
        if ambient:
            try:
                raise ValueError("unrelated caller exception")
            except ValueError:
                with pytest.raises(OSError, match="hook removal failed"):
                    _derive(model)
        else:
            with pytest.raises(OSError, match="hook removal failed"):
                _derive(model)
        assert not module._forward_hooks
        assert module._forward_pre_hooks  # failed removal remains explicitly unsafe
        assert [parameter.requires_grad for parameter in model.parameters()] == flags
        assert module.weight.grad is None
    finally:
        for handle in real_handles:
            handle.remove()


def test_cleanup_runs_when_sys_exception_is_unavailable(monkeypatch):
    model = _model()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    monkeypatch.delattr(sys, "exception", raising=False)
    _derive(model)
    assert not model.projection._forward_pre_hooks
    assert not model.projection._forward_hooks
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert model.projection.weight.grad is None


@pytest.mark.skipif(not hasattr(sys, "exception"), reason="historical v1 requires sys.exception")
def test_success_preserves_historical_candidate_bits_and_model_state():
    model = _model()
    module = model.projection
    original = module.weight.detach().clone()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    expected_gradient, expected_random, expected_details = historical._candidate(
        model, module, (1, 2), 1, 2, expected_shape=(3, 4))
    gradient, random, details = _derive(model)
    assert torch.equal(gradient.contiguous().view(torch.uint8),
                       expected_gradient.contiguous().view(torch.uint8))
    assert torch.equal(random.contiguous().view(torch.uint8),
                       expected_random.contiguous().view(torch.uint8))
    assert details == expected_details
    assert torch.equal(module.weight, original)
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert not module._forward_pre_hooks and not module._forward_hooks
    assert module.weight.grad is None


def test_registration_failure_removes_already_registered_hook(monkeypatch):
    model = _model()
    flags = [parameter.requires_grad for parameter in model.parameters()]

    def fail_registration(_hook):
        raise OSError("registration failed")

    monkeypatch.setattr(model.projection, "register_forward_hook", fail_registration)
    with pytest.raises(OSError, match="registration failed"):
        _derive(model)
    assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks
    assert [parameter.requires_grad for parameter in model.parameters()] == flags


@pytest.mark.parametrize("cleanup_exception", [KeyboardInterrupt("cleanup"), SystemExit(23)])
def test_fresh_cleanup_cancellation_takes_priority_over_body_error(monkeypatch, cleanup_exception):
    model = _model()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    registered = []
    original_register = model.projection.register_forward_pre_hook

    class BrokenHandle:
        def remove(self):
            raise cleanup_exception

    def register(hook):
        registered.append(original_register(hook))
        return BrokenHandle()

    def fail_forward(*_args, **_kwargs):
        raise OSError("body failed")

    monkeypatch.setattr(model.projection, "register_forward_pre_hook", register)
    monkeypatch.setattr(model, "forward", fail_forward)
    try:
        with pytest.raises(type(cleanup_exception)) as raised:
            _derive(model)
        assert raised.value is cleanup_exception
        assert isinstance(raised.value.__cause__, OSError)
        assert str(raised.value.__cause__) == "body failed"
        assert [parameter.requires_grad for parameter in model.parameters()] == flags
        assert not model.projection._forward_hooks
        assert model.projection.weight.grad is None
    finally:
        for handle in registered:
            handle.remove()


@pytest.mark.parametrize("body_exception", [KeyboardInterrupt("body"), SystemExit(17)])
def test_original_body_cancellation_survives_cleanup_error(monkeypatch, body_exception):
    model = _model()
    registered = []
    original_register = model.projection.register_forward_pre_hook

    class BrokenHandle:
        def remove(self):
            raise OSError("cleanup failed")

    def register(hook):
        registered.append(original_register(hook))
        return BrokenHandle()

    def fail_forward(*_args, **_kwargs):
        raise body_exception

    monkeypatch.setattr(model.projection, "register_forward_pre_hook", register)
    monkeypatch.setattr(model, "forward", fail_forward)
    try:
        with pytest.raises(type(body_exception)) as raised:
            _derive(model)
        assert raised.value is body_exception
        assert isinstance(raised.value.__cause__, OSError)
        assert str(raised.value.__cause__) == "cleanup failed"
        assert not model.projection._forward_hooks
        assert model.projection.weight.grad is None
    finally:
        for handle in registered:
            handle.remove()


def test_body_error_and_cleanup_error_remain_visible_inside_ambient_handler(monkeypatch):
    model = _model()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    registered = []
    original_register = model.projection.register_forward_pre_hook

    class BrokenHandle:
        def remove(self):
            raise RuntimeError("cleanup failed")

    def register(hook):
        registered.append(original_register(hook))
        return BrokenHandle()

    def fail_forward(*_args, **_kwargs):
        raise OSError("body failed")

    monkeypatch.setattr(model.projection, "register_forward_pre_hook", register)
    monkeypatch.setattr(model, "forward", fail_forward)
    try:
        try:
            raise ValueError("unrelated caller exception")
        except ValueError:
            with pytest.raises(OSError, match="body failed") as raised:
                _derive(model)
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert str(raised.value.__cause__) == "cleanup failed"
        assert not model.projection._forward_hooks
        assert [parameter.requires_grad for parameter in model.parameters()] == flags
    finally:
        for handle in registered:
            handle.remove()


def test_multiple_hook_failures_do_not_skip_other_cleanup(monkeypatch):
    model = _model()
    module = model.projection
    flags = [parameter.requires_grad for parameter in model.parameters()]
    registered = []
    attempted = []
    cancellation = SystemExit(29)
    pre_register = module.register_forward_pre_hook
    post_register = module.register_forward_hook

    class BrokenHandle:
        def __init__(self, name, error):
            self.name, self.error = name, error

        def remove(self):
            attempted.append(self.name)
            raise self.error

    def register_pre(hook):
        registered.append(pre_register(hook))
        return BrokenHandle("pre", OSError("pre cleanup failed"))

    def register_post(hook):
        registered.append(post_register(hook))
        return BrokenHandle("post", cancellation)

    monkeypatch.setattr(module, "register_forward_pre_hook", register_pre)
    monkeypatch.setattr(module, "register_forward_hook", register_post)
    try:
        with pytest.raises(SystemExit) as raised:
            _derive(model)
        assert raised.value is cancellation and raised.value.code == 29
        assert attempted == ["pre", "pre", "post", "post"]
        assert module.weight.grad is None
        assert [parameter.requires_grad for parameter in model.parameters()] == flags
    finally:
        for handle in registered:
            handle.remove()


def test_flag_restore_failure_does_not_skip_remaining_cleanup(monkeypatch):
    model = _model()
    first = model.embedding.weight
    first.requires_grad_(True)
    original = first.requires_grad_
    attempts = []

    def fail_restore(flag):
        attempts.append(flag)
        if flag:
            raise OSError("flag restoration failed")
        return original(flag)

    monkeypatch.setattr(first, "requires_grad_", fail_restore)
    try:
        with pytest.raises(OSError, match="flag restoration failed"):
            _derive(model)
        assert attempts == [False, True, True]
        assert first.requires_grad is False  # the failed model must be discarded
        assert model.projection.weight.requires_grad is True
        assert model.projection.weight.grad is None
        assert not model.projection._forward_pre_hooks and not model.projection._forward_hooks
    finally:
        original(True)


def test_actual_python310_runs_cleanup_without_sys_exception():
    launcher = shutil.which("py") if os.name == "nt" else None
    if launcher is None:
        pytest.skip("Windows Python launcher is unavailable")
    probe = subprocess.run([launcher, "-3.10", "-I", "-B", "-c", "pass"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if probe.returncode:
        pytest.skip("Python 3.10 is not installed")
    script = r'''
import importlib.util
import sys
from types import ModuleType
assert sys.version_info[:2] == (3, 10) and not hasattr(sys, 'exception')
package = ModuleType('looking_glass')
package.__path__ = []
sys.modules['looking_glass'] = package
v1 = ModuleType('looking_glass.targeted_edit_cpu_v1')
v1.StudyError = RuntimeError
sys.modules[v1.__name__] = v1
torch = ModuleType('torch')
torch.int64 = object()
def fail_tensor(*args, **kwargs):
    raise OSError('synthetic body failure')
torch.tensor = fail_tensor
sys.modules['torch'] = torch
spec = importlib.util.spec_from_file_location('looking_glass.targeted_edit_candidate_v1_1', sys.argv[1])
module_under_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_under_test)
class Parameter:
    def __init__(self, flag):
        self.requires_grad, self.grad, self.shape = flag, 'not cleared', (3, 4)
    def requires_grad_(self, flag):
        self.requires_grad = flag
        return self
class Handle:
    def __init__(self, hooks, kind):
        self.hooks, self.kind = hooks, kind
    def remove(self):
        del self.hooks[self.kind]
class Projection:
    def __init__(self):
        self.weight, self.hooks = Parameter(False), {}
    def register_forward_pre_hook(self, hook):
        self.hooks['pre'] = hook
        return Handle(self.hooks, 'pre')
    def register_forward_hook(self, hook):
        self.hooks['post'] = hook
        return Handle(self.hooks, 'post')
class Model:
    def __init__(self):
        self.projection, self.other = Projection(), Parameter(True)
    def parameters(self):
        return (self.projection.weight, self.other)
model = Model()
try:
    module_under_test.derive_candidate(model, model.projection, (1, 2), 1, 2, expected_shape=(3, 4))
except OSError as exc:
    assert str(exc) == 'synthetic body failure'
else:
    raise AssertionError('body failure was not propagated')
assert model.projection.hooks == {}
assert model.projection.weight.requires_grad is False and model.other.requires_grad is True
assert model.projection.weight.grad is None
'''
    result = subprocess.run([launcher, "-3.10", "-I", "-B", "-c", script,
                             str(Path(successor.__file__).resolve())],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
