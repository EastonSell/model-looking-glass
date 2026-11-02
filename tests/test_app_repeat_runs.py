"""Bounded same-process CPU lifecycle regressions, not a peak-RAM benchmark.

Repeated prompts may reuse a healthy model. Superseded unpinned trace tensor
objects must be released; genuine execution failure still quarantines the owner.
"""
import gc
import weakref

import pytest
import torch

from looking_glass.app import AppState
from looking_glass.app_runtime import LocalModelRuntime
from test_app_runtime import local_checkpoint
from test_app_llama import checkpoint as llama_checkpoint


A = 'hello world'
B = 'the small blue cat'
SETTINGS = {'max_new_tokens': 2}


def _bits(value):
    """Immutable independent bytes, never a retained view of a live capture."""
    return bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy())


def _model_snapshot(model):
    return {
        (kind, name): (tuple(value.shape), str(value.dtype), str(value.device), _bits(value))
        for kind, values in (
            ('parameter', model.named_parameters(remove_duplicate=False)),
            ('buffer', model.named_buffers(remove_duplicate=False)))
        for name, value in values
    }


def _hook_snapshot(model):
    fields = ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks',
              '_forward_hooks_with_kwargs', '_forward_hooks_always_called', '_forward_pre_hooks_with_kwargs')
    inventory = {
        (path, field): tuple((key, id(value)) for key, value in getattr(module, field, {}).items())
        for path, module in model.named_modules(remove_duplicate=False) for field in fields
    }
    global_hooks = torch.nn.modules.module
    for field in ('_global_forward_hooks', '_global_forward_pre_hooks', '_global_backward_hooks',
                  '_global_backward_pre_hooks', '_global_forward_hooks_always_called'):
        inventory[('global', field)] = tuple((key, id(value)) for key, value in getattr(global_hooks, field, {}).items())
    return inventory


def _trace_snapshot(state):
    # Run identities are intentionally fresh, but output IDs and every selected
    # vector/coordinate must be reproducible for the same fixed prompt/model.
    summary = state.store.summary()
    coordinate_keys = ('module_path', 'module_kind', 'layer_index', 'invocation', 'output_path',
                       'batch_positions', 'token_positions', 'shape', 'saved_shape', 'dtype',
                       'device', 'saved_dtype', 'saved_device', 'app_forward_index')
    return {
        'answer': summary['answer'],
        'generated_token_ids': summary['generated_token_ids'],
        'step_ids_masks': [(step['input_token_ids'], step['input_attention_mask'], step['selected_token_id'])
                          for step in summary['steps']],
        'capture_bytes': summary['total_bytes'],
        'captures': [(tuple(value.shape), _bits(value),
                      {key: row['metadata'].get(key) for key in coordinate_keys})
                     for value, row in zip(state.store._values, state.store._events, strict=True)],
    }


def _weak_captures(state):
    refs = [weakref.ref(value) for value in state.store._values]
    assert refs and all(ref() is not None for ref in refs), 'release assertions need actual live captures'
    return refs


def _assert_released(refs):
    gc.collect()
    survivors = sum(ref() is not None for ref in refs)
    assert survivors == 0, f'{survivors} superseded capture tensor objects remain retained'


def _loaded_state(tmp_path, family):
    path = tmp_path / family
    if family == 'llama':
        llama_checkpoint(path, kv=1)
    else:
        local_checkpoint(path, family)
    state = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / 'exports')
    state.post('load', {'path': str(path), 'device': 'cpu'})
    return state


@pytest.mark.parametrize('family', ['gpt2', 'llama'])
def test_ten_same_owner_a_b_a_cycles_release_superseded_captures(tmp_path, family):
    state = _loaded_state(tmp_path, family)
    model = state.runtime.model
    binding = state.runtime.capture_binding()
    model_before, hooks_before = _model_snapshot(model), _hook_snapshot(model)
    seen_capture_ids = set()

    def query(prompt):
        previous = _weak_captures(state) if state.store.summary() is not None else []
        summary = state.post('run', {'prompt': prompt, 'settings': dict(SETTINGS)})
        assert summary['capture_id'] not in seen_capture_ids
        seen_capture_ids.add(summary['capture_id'])
        _assert_released(previous)
        assert state.runtime.model is model and state.runtime.capture_binding() is binding
        assert state._live_binding is binding and state.runtime.info()['state'] == 'ready'
        assert _model_snapshot(model) == model_before
        assert _hook_snapshot(model) == hooks_before
        assert len(state.store._values) == 12  # 2 layers × 3 boundaries × 2 uncached forwards
        return _trace_snapshot(state)

    oracle = query(A)
    # Each family executes 30 cycle queries + oracle + one post-error recovery.
    for cycle in range(10):
        assert query(A) == oracle
        different = query(B)
        assert different['step_ids_masks'][0][0] != oracle['step_ids_masks'][0][0]
        assert different['captures'][0][0] != oracle['captures'][0][0], 'B must exercise a different prefix length'
        assert query(A) == oracle
        if cycle == 4:
            prior = _weak_captures(state)
            with pytest.raises(ValueError, match='budget'):
                state.post('run', {'prompt': A, 'settings': {'max_new_tokens': 2, 'max_trace_bytes': 1}})
            _assert_released(prior)
            assert state.store.summary() is None and state._capture_id is None and state._live_binding is None
            assert state.runtime.model is model and state.runtime.capture_binding() is binding
            assert state.runtime.info()['state'] == 'ready'
            assert _model_snapshot(model) == model_before and _hook_snapshot(model) == hooks_before
            assert query(A) == oracle
    assert len(seen_capture_ids) == 32
    remaining = _weak_captures(state)
    state.post('clear', {})
    _assert_released(remaining)
    assert state.runtime.model is model and state.runtime.capture_binding() is binding
    state.post('unload', {})
    assert state.runtime.model is None and state.runtime.capture_binding() is None
    assert not state.store.export_root.exists(), 'RAM-only repeated queries must not write exports'


@pytest.mark.parametrize('family', ['gpt2', 'llama'])
def test_real_execution_failure_is_not_recovered_as_preflight(tmp_path, monkeypatch, family):
    state = _loaded_state(tmp_path, family)
    state.post('run', {'prompt': A, 'settings': dict(SETTINGS)})
    refs = _weak_captures(state)
    model = state.runtime.model
    cls = type(model)
    original = cls.forward
    calls = []

    def fail_after_forward(self, *args, **kwargs):
        # A class-level test fault preserves the standard forward-identity guard;
        # this is a real executed forward, not a preflight rejection masquerade.
        original(self, *args, **kwargs)
        calls.append(True)
        raise RuntimeError('test-only executed-forward failure')

    with monkeypatch.context() as context:
        context.setattr(cls, 'forward', fail_after_forward)
        with pytest.raises(RuntimeError, match='executed-forward failure'):
            state.post('run', {'prompt': A, 'settings': dict(SETTINGS)})
    _assert_released(refs)
    assert calls == [True]
    assert state.runtime.info()['state'] == 'quarantined' and state.runtime.capture_binding() is None
    assert state.runtime.model is model  # No optimistic uncertain-owner disposal.
    assert state.store.summary() is None
    with pytest.raises(RuntimeError, match='quarantined'):
        state.post('run', {'prompt': A, 'settings': dict(SETTINGS)})
    assert calls == [True]


def test_release_assertion_detects_an_actual_retained_tensor():
    value = torch.ones(3)
    refs = [weakref.ref(value)]
    with pytest.raises(AssertionError, match='1 superseded'):
        _assert_released(refs)
    del value
    _assert_released(refs)


def test_bit_oracle_detects_signed_zero_not_just_numeric_equality():
    value = torch.tensor([0.0, 1.0])
    before = _bits(value)
    value.view(torch.int32)[0] = -2147483648
    assert torch.equal(value, torch.tensor([0.0, 1.0]))
    assert _bits(value) != before


def test_state_and_hook_oracles_detect_real_mutations():
    model = torch.nn.Linear(2, 2).eval()
    model.register_buffer('scalar', torch.tensor(0.0))
    before = _model_snapshot(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert _model_snapshot(model) != before
    before = _model_snapshot(model)
    model.scalar.reshape(-1).view(torch.int32)[0] = -2147483648
    assert _model_snapshot(model) != before, 'scalar signed-zero buffer mutation must be visible'
    hooks = _hook_snapshot(model)
    handle = model.register_forward_hook(lambda *_: None)
    try:
        assert _hook_snapshot(model) != hooks
    finally:
        handle.remove()
    assert _hook_snapshot(model) == hooks
