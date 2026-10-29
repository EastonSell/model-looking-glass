"""CPU-only tests for the fixed phase and bit-comparison logic."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass.pretrained_gpu_trace_pilot_core import TOKENS, run_fixed_trace


class CPUExecutor:
    def __init__(self, *, corrupt=None, fail=None):
        self.events, self.traces, self.reference = [], {}, None
        self.current = None
        self.corrupt, self.fail = corrupt, fail

    def bind_reference(self, value):
        self.reference = value

    def bind_reference_lease(self, lease, handles):
        self.lease, self.handles = lease, handles

    def bind_trace(self, mode, trace):
        self.traces[mode] = trace

    def begin(self, phase):
        assert self.current is None
        self.current = phase
        self.events.append(('begin', phase))

    def forward(self, phase, invoke):
        assert self.current == phase
        self.events.append(('forward', phase))
        if phase == self.fail:
            raise RuntimeError('fabricated forward failure')
        output = invoke()
        assert output.past_key_values is None
        value = output.logits.detach().to('cpu', copy=True)
        if phase == self.corrupt:
            value[0, 0, 0] += 1
        return value

    def end(self, phase, proof):
        assert self.current == phase and proof
        self.events.append(('end', phase))
        self.current = None


def model_and_inputs():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=6767, n_positions=16, n_embd=8, n_layer=6, n_head=2,
                        use_cache=False, _attn_implementation='eager')
    model = GPT2LMHeadModel(config).eval()
    ids = torch.tensor([TOKENS], dtype=torch.int64)
    return model, ids, torch.ones_like(ids)


def test_five_token_ten_forward_comparison_and_fixed_event_inventory():
    model, ids, mask = model_and_inputs()
    executor = CPUExecutor()
    result = run_fixed_trace(model, ids, mask, executor, study_id='cpu-fake')
    assert [phase for kind, phase in executor.events if kind == 'begin'] == [
        'baseline_cold', 'baseline_warm', 'reference', 'direct', 'gradient', 'id']
    assert [phase for kind, phase in executor.events if kind == 'forward'] == [
        'baseline_cold', 'baseline_warm', 'reference', 'reference', 'direct', 'direct',
        'gradient', 'gradient', 'id', 'id']
    assert result['direct_gradient_equal'] and len(executor.reference) == 18
    assert result['modes']['direct']['tensor_bytes'] == result['modes']['gradient']['tensor_bytes'] == 1152
    assert result['modes']['id']['tensor_bytes'] == 0
    assert all(result['modes'][mode]['event_count'] == 18 for mode in ('direct', 'gradient', 'id'))
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize('corrupt', ['baseline_warm', 'reference', 'direct', 'gradient', 'id'])
def test_logit_bit_discrepancy_never_reaches_later_phases(corrupt):
    model, ids, mask = model_and_inputs()
    executor = CPUExecutor(corrupt=corrupt)
    with pytest.raises(ValueError, match='logits differ'):
        run_fixed_trace(model, ids, mask, executor, study_id='cpu-fake')
    begun = [phase for kind, phase in executor.events if kind == 'begin']
    assert begun[-1] == corrupt


def test_failure_does_not_start_a_later_phase():
    model, ids, mask = model_and_inputs()
    executor = CPUExecutor(fail='gradient')
    with pytest.raises(RuntimeError, match='fabricated forward failure'):
        run_fixed_trace(model, ids, mask, executor, study_id='cpu-fake')
    assert ('begin', 'gradient') in executor.events and ('begin', 'id') not in executor.events


@pytest.mark.parametrize('bad', ['ids', 'mask'])
def test_changed_ids_or_boolean_mask_stop_before_any_forward(bad):
    model, ids, mask = model_and_inputs()
    executor = CPUExecutor()
    if bad == 'ids':
        ids[0, 0] += 1
    else:
        mask = mask.bool()
    with pytest.raises(ValueError, match='fixed'):
        run_fixed_trace(model, ids, mask, executor, study_id='cpu-fake')
    assert not executor.events


@pytest.mark.parametrize('body_cancel,second_cancel', [(True, False), (False, True)])
def test_trace_cleanup_keeps_cancellation_primary_and_retains_failed_handles(monkeypatch,
                                                                             body_cancel, second_cancel):
    model, ids, mask = model_and_inputs()
    executor = CPUExecutor()
    original_forward = executor.forward
    original_remove = torch.utils.hooks.RemovableHandle.remove
    count = 0
    def interrupted_forward(phase, invoke):
        if phase == 'direct':
            if body_cancel:
                raise KeyboardInterrupt('body cancellation')
            raise ValueError('body failure')
        return original_forward(phase, invoke)
    def interrupted_removal(handle):
        nonlocal count
        if executor.current == 'direct':
            count += 1
            if count == 1:
                raise OSError('cleanup failure')
            if count == 2 and second_cancel:
                raise KeyboardInterrupt('cleanup cancellation')
        return original_remove(handle)
    executor.forward = interrupted_forward
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(torch.utils.hooks.RemovableHandle, 'remove', interrupted_removal)
            with pytest.raises(KeyboardInterrupt, match='body cancellation' if body_cancel else 'cleanup cancellation'):
                run_fixed_trace(model, ids, mask, executor, study_id='cpu-fake')
        trace = executor.traces['direct']
        assert trace._handles and trace._lease._held and ('begin', 'gradient') not in executor.events
    finally:
        if 'direct' in executor.traces:
            executor.traces['direct'].close()
