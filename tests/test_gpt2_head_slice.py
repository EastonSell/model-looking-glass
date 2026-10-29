"""Fabricated CPU checks for the pre-projection head-slice API."""

import pytest
import torch
import types

from looking_glass.copy_head_intervention import assert_no_forward_hooks
from looking_glass.gpt2_head_slice import GPT2HeadSlicePoint, run_gpt2_head_slice


@pytest.fixture
def model():
    from transformers import GPT2Config, GPT2LMHeadModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(847)
        config = GPT2Config(vocab_size=31, n_positions=8, n_embd=8, n_layer=2, n_head=2,
                            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, use_cache=False)
        config._attn_implementation = "eager"
        return GPT2LMHeadModel(config).eval()


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32),
                                               b.contiguous().view(torch.int32))


def call(model, ids, point, **kwargs):
    return run_gpt2_head_slice(model, ids, point=point, run_id="tiny", **kwargs)


def test_trace_and_patch_match_independent_projection_input_and_controls(model):
    ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.int64)
    point = GPT2HeadSlicePoint(layer=0, token=2, head=1)
    references = []
    handle = model.transformer.h[0].attn.c_proj.register_forward_pre_hook(
        lambda _module, args: references.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            clean = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    finally:
        handle.remove()
    observed = call(model, ids, point)
    self_patch = call(model, ids, point, replacement=observed.head_before.clone())
    replacement = torch.tensor([3.0, -2.0, 1.0, 5.0])
    patched = call(model, ids, point, replacement=replacement)
    repeated = call(model, ids, point)
    assert bits_equal(observed.before, references[0])
    assert bits_equal(observed.before, observed.after)
    assert bits_equal(observed.logits, clean)
    assert bits_equal(self_patch.before, self_patch.after)
    assert bits_equal(self_patch.logits, clean)
    assert bits_equal(patched.head_after, replacement)
    expected = patched.before.clone()
    expected[0, 2, 4:8] = replacement
    assert bits_equal(patched.after, expected)
    assert bits_equal(patched.logits[:, :2], clean[:, :2])
    assert not bits_equal(patched.logits[:, 2:], clean[:, 2:])
    assert bits_equal(repeated.logits, clean)
    assert observed.tensor_bytes == 3 * 4 * 8 * 4
    assert observed.metadata.to_dict() == {
        "run_id": "tiny", "module_path": "transformer.h.0.attn.c_proj", "layer": 0,
        "batch": 0, "token": 2, "invocation": 0, "head": 1, "heads": 2,
        "head_width": 4, "feature_start": 4, "feature_end": 8,
        "input_shape": [1, 4, 8], "input_ids": [1, 4, 7, 2], "attention_mask": [1, 1, 1, 1],
        "operation": "observe", "tensor_role": "projection_input", "input_argument": 0,
        "forward_index": 0, "use_cache": False, "dtype": "float32", "device": "cpu",
    }
    assert_no_forward_hooks(model)


def test_bit_exact_subnormal_and_signed_zero_delivery(model):
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    source = torch.tensor([1, -2147483648, 1065353216, -2147483647], dtype=torch.int32).view(torch.float32)
    point = GPT2HeadSlicePoint(0, 1, 0)
    probe = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    old_flush = int((probe + probe).view(torch.int32).item()) == 0
    supported = torch.set_flush_denormal(True)
    try:
        result = call(model, ids, point, replacement=source)
        assert bits_equal(result.head_after, source)
        mask = torch.ones_like(result.before, dtype=torch.bool)
        mask[0, 1, :4] = False
        assert torch.equal(result.before.view(torch.int32)[mask], result.after.view(torch.int32)[mask])
    finally:
        if supported:
            torch.set_flush_denormal(old_flush)


@pytest.mark.parametrize("layer,token,head", [(0, 0, 0), (0, 3, 1), (1, 1, 0), (1, 3, 1)])
def test_head_coordinates_match_separate_direct_hooks(model, layer, token, head):
    ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.int64)
    references = []
    handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
        lambda _module, args: references.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    point = GPT2HeadSlicePoint(layer, token, head)
    result = call(model, ids, point)
    assert bits_equal(result.before, references[0])
    assert bits_equal(result.head_before, references[0][0, token, 4 * head:4 * (head + 1)])
    assert (result.metadata.layer, result.metadata.token, result.metadata.head) == (layer, token, head)


def test_example_runs_with_tiny_model_and_reports_distinct_controls(model):
    from examples.gpt2_head_slice_cpu import run_demo
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    result = run_demo(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    assert all(result["checks"].values())
    assert result["target"]["metadata"]["head"] == 0
    assert result["comparison"]["metadata"]["head"] == 1
    assert result["target"]["realized_input_l2"] > 0
    assert result["comparison"]["realized_input_l2"] > 0


@pytest.mark.parametrize("kwargs", [
    {"max_input_bytes": 15}, {"max_tensor_bytes": 63}, {"max_capture_bytes": 191},
    {"max_logits_bytes": 247}, {"max_source_bytes": 15},
])
def test_limits_fail_before_any_forward(model, monkeypatch, kwargs):
    from transformers import GPT2LMHeadModel
    calls = []
    forward = GPT2LMHeadModel.forward
    def count(self, *args, **kw):
        calls.append(True)
        return forward(self, *args, **kw)
    monkeypatch.setattr(GPT2LMHeadModel, "forward", count)
    with pytest.raises(ValueError, match="budget"):
        call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 0, 0),
             replacement=torch.ones(4), **kwargs)
    assert calls == []
    assert_no_forward_hooks(model)


@pytest.mark.parametrize("point", [GPT2HeadSlicePoint(2, 0, 0), GPT2HeadSlicePoint(0, 2, 0),
                                    GPT2HeadSlicePoint(0, 0, 2)])
def test_invalid_coordinates_are_rejected(model, point):
    with pytest.raises(ValueError, match="out of range"):
        call(model, torch.tensor([[1, 2]]), point)


@pytest.mark.parametrize("replacement", [torch.ones(5), torch.tensor([1., float("nan"), 2., 3.]),
                                         torch.ones(4, dtype=torch.float64)])
def test_invalid_replacement_is_rejected(model, replacement):
    with pytest.raises(ValueError, match="replacement"):
        call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 0, 0), replacement=replacement)
    assert_no_forward_hooks(model)


def test_foreign_hook_is_rejected_and_preserved(model):
    handle = model.transformer.h[1].register_forward_hook(lambda *args: None)
    try:
        with pytest.raises(RuntimeError, match="already has a forward hook"):
            call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 0, 0))
        assert handle.id in model.transformer.h[1]._forward_hooks
    finally:
        handle.remove()


@pytest.mark.parametrize("changed", ["ids", "mask", "layout"])
def test_late_input_or_layout_mutation_discards_result(model, monkeypatch, changed):
    from transformers import GPT2LMHeadModel
    original = GPT2LMHeadModel.forward
    def mutate(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if changed == "ids":
            kwargs["input_ids"][0, 0] = 5
        elif changed == "mask":
            kwargs["attention_mask"][0, 0] = 0
        else:
            self.transformer.h[0].attn.num_heads += 1
        return output
    monkeypatch.setattr(GPT2LMHeadModel, "forward", mutate)
    with pytest.raises(RuntimeError, match="IDs, mask or model layout changed"):
        call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 0))
    assert_no_forward_hooks(model)


@pytest.mark.parametrize("site", ["block", "transformer"])
def test_instance_overridden_forward_is_rejected(model, monkeypatch, site):
    target = model.transformer.h[0] if site == "block" else model.transformer
    original = target.forward
    def overridden(self, *args, **kwargs):
        return original(*args, **kwargs)
    monkeypatch.setattr(target, "forward", types.MethodType(overridden, target))
    with pytest.raises(ValueError, match="unmodified"):
        call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 0))
    assert_no_forward_hooks(model)


def test_transient_changed_head_layout_at_projection_is_rejected(model, monkeypatch):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    selected = model.transformer.h[0]
    original = GPT2Block.forward
    def altered(self, *args, **kwargs):
        if self is not selected:
            return original(self, *args, **kwargs)
        attention = self.attn
        old_heads, old_width = attention.num_heads, attention.head_dim
        attention.num_heads, attention.head_dim = 4, 2
        try:
            return original(self, *args, **kwargs)
        finally:
            attention.num_heads, attention.head_dim = old_heads, old_width
    with monkeypatch.context() as patch:
        patch.setattr(GPT2Block, "forward", altered)
        with pytest.raises(RuntimeError, match="layout changed at projection input"):
            call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 1))
    assert_no_forward_hooks(model)
    call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 1))


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_unwound_projection_failure_removes_hooks_and_handles_lease(model, monkeypatch, failure):
    from transformers.pytorch_utils import Conv1D
    original = Conv1D.forward
    projection = model.transformer.h[0].attn.c_proj
    def fail(self, value):
        if self is projection:
            raise failure("synthetic projection failure")
        return original(self, value)
    with monkeypatch.context() as patch:
        patch.setattr(Conv1D, "forward", fail)
        with pytest.raises(failure, match="synthetic"):
            call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 0))
    assert_no_forward_hooks(model)
    if failure is KeyboardInterrupt:
        with pytest.raises(RuntimeError, match="already leased"):
            call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 0))
    else:
        assert call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 0)).metadata.token == 1


def test_failed_hook_removal_quarantines_model(model, monkeypatch):
    from torch.utils.hooks import RemovableHandle
    with monkeypatch.context() as patch:
        patch.setattr(RemovableHandle, "remove", lambda self: None)
        with pytest.raises(RuntimeError, match="cleanup is uncertain"):
            call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 0))
    with pytest.raises(RuntimeError, match="already leased"):
        call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 0))


@pytest.mark.parametrize("extra_calls", [0, 2])
def test_missing_or_repeated_projection_call_fails(model, monkeypatch, extra_calls):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    original = GPT2Attention.forward
    selected = model.transformer.h[0].attn
    def altered(self, hidden_states, *args, **kwargs):
        if self is not selected:
            return original(self, hidden_states, *args, **kwargs)
        if extra_calls == 0:
            return torch.zeros_like(hidden_states), None
        output = original(self, hidden_states, *args, **kwargs)
        self.c_proj(hidden_states.contiguous())
        return output
    with monkeypatch.context() as patch:
        patch.setattr(GPT2Attention, "forward", altered)
        with pytest.raises(RuntimeError, match="projection"):
            call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 0))
    assert_no_forward_hooks(model)
    call(model, torch.tensor([[1, 2]]), GPT2HeadSlicePoint(0, 1, 0))
