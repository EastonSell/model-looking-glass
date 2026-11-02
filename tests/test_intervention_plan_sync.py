"""Synchronous CPU cancellation behavior for the shared intervention plan."""

import inspect
import sys
import threading
import weakref

import pytest
import torch
from torch import nn

from looking_glass.intervention_plan import ActivationPlan
from looking_glass.model_ownership import ModelLease


class InterruptedModel(nn.Module):
    def __init__(self, error):
        super().__init__()
        self.tap = nn.Identity()
        self.error = error

    def forward(self, value):
        self.tap(value)
        raise self.error


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_unwound_synchronous_cpu_cancellation_removes_owned_hooks_and_lease(error_type):
    original = error_type("cancelled after capture")
    model = InterruptedModel(original)
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])

    with pytest.raises(error_type) as caught:
        with plan:
            plan.run_synchronous_cpu(torch.ones(1, 2))

    assert caught.value is original
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    with pytest.raises(RuntimeError):
        _ = plan.captures
    with ModelLease(model):
        pass


def test_successful_synchronous_cpu_forward_publishes_normal_capture():
    model = nn.Sequential(nn.Identity())
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "0"}])
    with plan:
        result = plan.run_synchronous_cpu(torch.tensor([[3.0, 5.0]]))
    assert torch.equal(result, torch.tensor([[3.0, 5.0]]))
    assert torch.equal(plan.captures["seen"], torch.tensor([[3.0, 5.0]]))
    with ModelLease(model):
        pass


def test_cancellation_caught_inside_context_never_publishes_partial_capture():
    model = InterruptedModel(KeyboardInterrupt("cancelled after capture"))
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    with pytest.raises(RuntimeError):
        with plan:
            with pytest.raises(KeyboardInterrupt):
                plan.run_synchronous_cpu(torch.ones(1, 2))
    with pytest.raises(RuntimeError):
        _ = plan.captures
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    with ModelLease(model):
        pass


def test_known_non_cpu_input_rejected_before_forward():
    calls = []
    class Counting(nn.Module):
        def __init__(self):
            super().__init__()
            self.tap = nn.Identity()

        def forward(self, value):
            calls.append(True)
            return self.tap(value)

    model = Counting()
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    with pytest.raises(ValueError, match="CPU"):
        with plan:
            plan.run_synchronous_cpu({"nested": [torch.empty(1, device="meta")]})
    assert not calls
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    with ModelLease(model):
        pass


def test_known_non_cpu_model_state_is_rejected_before_forward():
    calls = []
    class Counting(nn.Module):
        def __init__(self):
            super().__init__()
            self.tap = nn.Identity()
            self.register_buffer("other_device", torch.empty(1, device="meta"))

        def forward(self, value):
            calls.append(True)
            return self.tap(value)

    model = Counting()
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    with pytest.raises(ValueError, match="CPU"):
        with plan:
            plan.run_synchronous_cpu(torch.ones(1, 2))
    assert not calls
    with ModelLease(model):
        pass


@pytest.mark.parametrize("site", ["selected_output", "root_output", "model_output"])
def test_known_non_cpu_output_quarantines_active_wrapper(site):
    roots = []
    class NonCpuOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.tap = nn.Identity()

        def forward(self, value):
            device_value = torch.empty(1, device="meta")
            roots.append(weakref.ref(device_value))
            if site == "selected_output":
                return self.tap(device_value)
            self.tap(value)
            if site == "model_output":
                transformers = pytest.importorskip("transformers")
                return transformers.modeling_outputs.Seq2SeqLMOutput(logits=device_value)
            return device_value

    model = NonCpuOutput()
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    with pytest.raises(ValueError, match="CPU"):
        with plan:
            plan.run_synchronous_cpu(torch.ones(1, 2))
    assert model.tap._forward_hooks
    assert len(roots) == 1 and roots[0]() is not None
    if site == "model_output":
        assert plan._sync_observed.logits is roots[0]()
    else:
        assert plan._sync_observed is roots[0]()
    with pytest.raises(RuntimeError):
        _ = plan.captures
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()


def test_cancellation_removes_only_the_plan_hooks():
    original = KeyboardInterrupt("cancelled")
    model = InterruptedModel(original)
    observed = []
    other = model.tap.register_forward_hook(lambda module, inputs, output: observed.append(output.clone()))
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    try:
        with pytest.raises(KeyboardInterrupt):
            with plan:
                plan.run_synchronous_cpu(torch.ones(1, 2))
        assert len(observed) == 1 and torch.equal(observed[0], torch.ones(1, 2))
        assert list(model.tap._forward_hooks) == [other.id]
        with ModelLease(model):
            pass
    finally:
        other.remove()


def test_retirement_error_preserves_root_error_and_quarantines():
    original = KeyboardInterrupt("root failure")
    secondary = RuntimeError("retirement failure")

    class FailedRetirementPlan(ActivationPlan):
        def _finish_synchronous(self, token, *, failed):
            raise secondary

    model = InterruptedModel(original)
    plan = FailedRetirementPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    with pytest.raises(KeyboardInterrupt) as caught:
        with plan:
            plan.run_synchronous_cpu(torch.ones(1, 2))
    assert caught.value is original and caught.value.__cause__ is secondary
    assert model.tap._forward_hooks
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()


def test_hook_removal_failure_preserves_cancellation_and_allows_known_retry():
    original = KeyboardInterrupt("root failure")
    removal = RuntimeError("hook removal failed")
    model = InterruptedModel(original)
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])

    class FailOnce:
        def __init__(self, handle):
            self.handle = handle
            self.failed = False

        def remove(self):
            if not self.failed:
                self.failed = True
                raise removal
            self.handle.remove()

    with pytest.raises(KeyboardInterrupt) as caught:
        with plan:
            plan._handles[1] = FailOnce(plan._handles[1])
            plan.run_synchronous_cpu(torch.ones(1, 2))
    assert caught.value is original and caught.value.__cause__ is removal
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()
    plan.close()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    with ModelLease(model):
        pass


def test_interruption_before_retirement_publication_keeps_lease_and_original():
    original = KeyboardInterrupt("root failure")
    secondary = SystemExit("retirement interrupted")
    model = InterruptedModel(original)
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    lines, start = inspect.getsourcelines(ActivationPlan._finish_synchronous)
    marker_line = start + next(i for i, line in enumerate(lines) if "self._sync_forward = None" in line)
    injected = []

    def trace(frame, event, arg):
        if frame.f_code is ActivationPlan._finish_synchronous.__code__ and event == "line" and frame.f_lineno == marker_line:
            injected.append(True)
            raise secondary
        return trace

    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt) as caught:
            with plan:
                plan.run_synchronous_cpu(torch.ones(1, 2))
    finally:
        sys.settrace(None)
    assert injected == [True]
    assert caught.value is original and caught.value.__cause__ is secondary
    assert model.tap._forward_hooks
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()


def test_interrupt_before_output_inspection_marker_quarantines():
    original = KeyboardInterrupt("before inspection publication")

    class NonCpuOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.tap = nn.Identity()

        def forward(self, value):
            return self.tap(torch.empty(1, device="meta"))

    model = NonCpuOutput()
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    lines, start = inspect.getsourcelines(ActivationPlan._check_synchronous_value)
    marker_line = start + next(i for i, line in enumerate(lines) if "self._sync_observation_pending = True" in line)
    injected = []

    def trace(frame, event, arg):
        if frame.f_code is ActivationPlan._check_synchronous_value.__code__ and event == "line" and frame.f_lineno == marker_line:
            injected.append(True)
            raise original
        return trace

    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt) as caught:
            with plan:
                plan.run_synchronous_cpu(torch.ones(1, 2))
    finally:
        sys.settrace(None)
    assert caught.value is original and injected == [True]
    assert model.tap._forward_hooks
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()


def test_interrupted_hook_failure_latch_preserves_error_and_lease():
    original = KeyboardInterrupt("target inspection interrupted")
    secondary = SystemExit("hook failure latch interrupted")

    class InterruptedLatchPlan(ActivationPlan):
        def _check_synchronous_value(self, value):
            raise original

        def _fail_synchronous_hook(self):
            raise secondary

    model = nn.Sequential(nn.Identity())
    plan = InterruptedLatchPlan(model, [{"name": "seen", "operation": "capture", "module_path": "0"}])
    with pytest.raises(KeyboardInterrupt) as caught:
        with plan:
            plan.run_synchronous_cpu(torch.ones(1, 2))
    assert caught.value is original
    chain = []
    error = caught.value
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__
    assert secondary in chain
    assert model[0]._forward_hooks
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()


def test_reentrant_exit_before_root_pre_hook_cannot_release_owned_hooks():
    class Reentrant(nn.Module):
        def __init__(self):
            super().__init__()
            self.tap = nn.Identity()
            self.attempted = False

        def _call_impl(self, *args, **kwargs):
            self.attempted = True
            with pytest.raises(RuntimeError):
                plan.__exit__(None, None, None)
            with pytest.raises(RuntimeError):
                ModelLease(self).acquire()
            assert self.tap._forward_hooks
            return super()._call_impl(*args, **kwargs)

        def forward(self, value):
            return self.tap(value)

    model = Reentrant()
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    with pytest.raises(RuntimeError):
        with plan:
            plan.run_synchronous_cpu(torch.ones(1, 2))
    assert model.attempted and model.tap._forward_hooks
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()
    with pytest.raises(RuntimeError):
        _ = plan.captures


def test_reentrant_wrapper_cannot_clear_outer_call_marker():
    attempts = []

    class Reentrant(nn.Module):
        def __init__(self):
            super().__init__()
            self.tap = nn.Identity()

        def forward(self, value):
            with pytest.raises(RuntimeError):
                plan.run_synchronous_cpu(value)
            attempts.append(True)
            with pytest.raises(RuntimeError):
                ModelLease(self).acquire()
            assert self.tap._forward_hooks
            return self.tap(value)

    model = Reentrant()
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    with pytest.raises(RuntimeError):
        with plan:
            plan.run_synchronous_cpu(torch.ones(1, 2))
    assert attempts == [True] and model.tap._forward_hooks
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()


@pytest.mark.parametrize("phase", ["before_root", "after_root"])
def test_interrupted_wrapper_boundary_retains_lease(phase):
    calls = []
    class Counting(nn.Module):
        def __init__(self):
            super().__init__()
            self.tap = nn.Identity()

        def forward(self, value):
            calls.append(True)
            return self.tap(value)

    model = Counting()
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    lines, start = inspect.getsourcelines(ActivationPlan.run_synchronous_cpu)
    if phase == "before_root":
        claim = next(i for i, line in enumerate(lines) if "self._sync_forward = token" in line)
        offset = next(i for i in range(claim + 1, len(lines)) if lines[i].strip() == "try:")
    else:
        offset = next(i for i, line in enumerate(lines) if "self._finish_synchronous(token, failed=False)" in line)
    boundary_line = start + offset
    injected = []
    original = KeyboardInterrupt("wrapper boundary")

    def trace(frame, event, arg):
        if frame.f_code is ActivationPlan.run_synchronous_cpu.__code__ and event == "line" and frame.f_lineno == boundary_line:
            injected.append(True)
            raise original
        return trace

    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt) as caught:
            with plan:
                plan.run_synchronous_cpu(torch.ones(1, 2))
    finally:
        sys.settrace(None)
    assert caught.value is original and injected == [True]
    assert len(calls) == (0 if phase == "before_root" else 1)
    assert model.tap._forward_hooks
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()


def test_foreign_wrapper_attempt_does_not_retire_active_owner():
    errors = []
    class WithForeignAttempt(nn.Module):
        def __init__(self):
            super().__init__()
            self.tap = nn.Identity()

        def forward(self, value):
            def foreign():
                try:
                    plan.run_synchronous_cpu(value)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=foreign)
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive()
            with pytest.raises(RuntimeError):
                ModelLease(self).acquire()
            assert self.tap._forward_hooks
            return self.tap(value)

    model = WithForeignAttempt()
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    with pytest.raises(RuntimeError):
        with plan:
            plan.run_synchronous_cpu(torch.ones(1, 2))
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert model.tap._forward_hooks
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()


@pytest.mark.parametrize("method", ["abort", "close", "exit"])
def test_foreign_lifecycle_attempt_then_cancellation_keeps_owner_quarantined(method):
    errors = []
    original = KeyboardInterrupt("root interrupted after foreign lifecycle call")

    class WithForeignLifecycle(nn.Module):
        def __init__(self):
            super().__init__()
            self.tap = nn.Identity()

        def forward(self, value):
            def foreign():
                try:
                    if method == "exit":
                        plan.__exit__(None, None, None)
                    else:
                        getattr(plan, method)()
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=foreign)
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive()
            raise original

    model = WithForeignLifecycle()
    plan = ActivationPlan(model, [{"name": "seen", "operation": "capture", "module_path": "tap"}])
    with pytest.raises(KeyboardInterrupt) as caught:
        with plan:
            plan.run_synchronous_cpu(torch.ones(1, 2))
    assert caught.value is original
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert model.tap._forward_hooks
    with pytest.raises(RuntimeError):
        ModelLease(model).acquire()


@pytest.mark.parametrize("stack", ["encoder", "decoder"])
def test_actual_t5_unwound_cancellation_cleans_and_accepts_a_fresh_plan(monkeypatch, stack):
    transformers = pytest.importorskip("transformers")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(313)
        config = transformers.T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32,
            num_layers=2, num_decoder_layers=2, num_heads=2, dropout_rate=0,
            decoder_start_token_id=0, pad_token_id=0)
        model = transformers.T5ForConditionalGeneration(config).eval()
    failing = model.get_submodule(f"{stack}.block.1")
    old = failing.forward
    original = KeyboardInterrupt("after first block")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(failing, "forward", fail)
    steps = [{"name": "seen", "operation": "capture", "module_path": f"{stack}.block.0",
              "batch_index": 0, "token_index": 1}]
    arguments = dict(input_ids=torch.tensor([[4, 5, 1]]), attention_mask=torch.tensor([[1, 1, 1]]),
        decoder_input_ids=torch.tensor([[0, 7]]), decoder_attention_mask=torch.tensor([[1, 1]]),
        use_cache=False, return_dict=True)
    plan = ActivationPlan(model, steps)
    with pytest.raises(KeyboardInterrupt) as caught:
        with torch.inference_mode(), plan:
            plan.run_synchronous_cpu(**arguments)
    assert caught.value is original
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    with pytest.raises(RuntimeError):
        _ = plan.captures
    monkeypatch.setattr(failing, "forward", old)
    fresh = ActivationPlan(model, steps)
    with torch.inference_mode(), fresh:
        output = fresh.run_synchronous_cpu(**arguments)
    assert output.logits.shape == (1, 2, 32)
    assert fresh.captures["seen"].shape == (16,)
