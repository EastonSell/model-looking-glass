"""Fabricated CPU models exercise the opt-in synchronous workflow boundary."""

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from looking_glass.flan_t5_intervention_workflow import compare_t5_donor_patch
from looking_glass.flan_t5_trace import T5TracePoint, trace_t5_forward
from looking_glass.model_ownership import ModelLease


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_t5():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51)
        config = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32,
                          num_layers=2, num_decoder_layers=2, num_heads=2,
                          dropout_rate=0.0, decoder_start_token_id=0,
                          pad_token_id=0, eos_token_id=1)
        return T5ForConditionalGeneration(config).eval()


def workflow_inputs(stack):
    return dict(receiver_input_ids=torch.tensor([[4, 5, 1]]),
                donor_input_ids=torch.tensor([[4, 6, 1]]),
                attention_mask=torch.ones((1, 3), dtype=torch.long),
                receiver_decoder_input_ids=torch.tensor([[0, 7, 1]]),
                donor_decoder_input_ids=torch.tensor([[0, 7, 1]]),
                decoder_attention_mask=torch.ones((1, 3), dtype=torch.long),
                target=T5TracePoint(stack, 1, "block", 0, 1),
                control=T5TracePoint(stack, 1, "block", 0, 0),
                readout_position=1, readout_token_id=7)


def assert_unleased_and_hook_free(model):
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    lease = ModelLease(model).acquire()
    lease.release()


@pytest.mark.parametrize("stack", ["encoder", "decoder"])
@pytest.mark.parametrize("module", ["block", "self_attention", "feed_forward"])
def test_sync_workflow_matches_direct_result_bits(stack, module):
    model = tiny_t5()
    inputs = workflow_inputs(stack)
    inputs["target"] = T5TracePoint(stack, 1, module, 0, 1)
    inputs["control"] = T5TracePoint(stack, 1, module, 0, 0)
    direct = compare_t5_donor_patch(model, **inputs)
    wrapped = compare_t5_donor_patch(model, **inputs, synchronous_cpu=True)
    assert wrapped.report == direct.report
    for name in ("receiver", "donor", "readout_logits"):
        assert set(wrapped.witnesses[name]) == set(direct.witnesses[name])
        for key, value in wrapped.witnesses[name].items():
            assert torch.equal(value.view(torch.int32), direct.witnesses[name][key].view(torch.int32))
    for arm, captures in wrapped.witnesses["plans"].items():
        assert set(captures) == set(direct.witnesses["plans"][arm])
        for key, value in captures.items():
            assert torch.equal(value.view(torch.int32), direct.witnesses["plans"][arm][key].view(torch.int32))
    assert_unleased_and_hook_free(model)


@pytest.mark.parametrize("stack", ["encoder", "decoder"])
@pytest.mark.parametrize("failed_forward", range(1, 7))
@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_sync_workflow_unwinds_each_root_call_and_preserves_interrupt(
        monkeypatch, stack, failed_forward, exception_type):
    model = tiny_t5()
    selected = getattr(model, stack).block[1]
    original = selected.forward
    interruption = exception_type("inside selected block")
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_forward:
            raise interruption
        return original(*args, **kwargs)

    monkeypatch.setattr(selected, "forward", interrupt)
    with pytest.raises(exception_type) as caught:
        compare_t5_donor_patch(model, **workflow_inputs(stack), synchronous_cpu=True)
    assert caught.value is interruption
    assert calls == failed_forward
    assert_unleased_and_hook_free(model)

    monkeypatch.setattr(selected, "forward", original)
    result = compare_t5_donor_patch(model, **workflow_inputs(stack), synchronous_cpu=True)
    assert result.report["forwards"] == 6
    assert all(result.report["controls"].values())
    assert_unleased_and_hook_free(model)


def test_sync_trace_can_unwind_a_supported_interrupt(monkeypatch):
    model = tiny_t5()
    selected = model.encoder.block[1]
    original = selected.forward
    interruption = KeyboardInterrupt("trace interrupt")

    def interrupt(*args, **kwargs):
        raise interruption

    monkeypatch.setattr(selected, "forward", interrupt)
    kwargs = dict(input_ids=torch.tensor([[4, 5, 1]]), attention_mask=torch.ones((1, 3), dtype=torch.long),
                  decoder_input_ids=torch.tensor([[0, 7, 1]]),
                  decoder_attention_mask=torch.ones((1, 3), dtype=torch.long),
                  points=[T5TracePoint("encoder", 1, "block", 0, 1)], run_id="sync-trace")
    with pytest.raises(KeyboardInterrupt) as caught:
        trace_t5_forward(model, **kwargs, synchronous_cpu=True)
    assert caught.value is interruption
    assert_unleased_and_hook_free(model)
    monkeypatch.setattr(selected, "forward", original)
    output, records = trace_t5_forward(model, **kwargs, synchronous_cpu=True)
    assert tuple(output.logits.shape) == (1, 3, 32)
    assert len(records) == 1
    assert_unleased_and_hook_free(model)


@pytest.mark.parametrize("stack", ["encoder", "decoder"])
@pytest.mark.parametrize("module", ["self_attention", "feed_forward"])
def test_sync_workflow_unwinds_selected_residual_boundary(monkeypatch, stack, module):
    model = tiny_t5()
    layer = 0 if module == "self_attention" else (1 if stack == "encoder" else 2)
    selected = getattr(model, stack).block[1].layer[layer]
    original = selected.forward
    interruption = KeyboardInterrupt("inside residual boundary")
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise interruption
        return original(*args, **kwargs)

    monkeypatch.setattr(selected, "forward", interrupt)
    inputs = workflow_inputs(stack)
    inputs["target"] = T5TracePoint(stack, 1, module, 0, 1)
    inputs["control"] = T5TracePoint(stack, 1, module, 0, 0)
    with pytest.raises(KeyboardInterrupt) as caught:
        compare_t5_donor_patch(model, **inputs, synchronous_cpu=True)
    assert caught.value is interruption
    assert calls == 3
    assert_unleased_and_hook_free(model)
    monkeypatch.setattr(selected, "forward", original)
    assert all(compare_t5_donor_patch(model, **inputs, synchronous_cpu=True).report["controls"].values())
    assert_unleased_and_hook_free(model)


@pytest.mark.parametrize("failed_forward", [1, 3])
def test_sync_workflow_quarantines_known_non_cpu_output(monkeypatch, failed_forward):
    model = tiny_t5()
    selected = model.encoder.block[1]
    original = selected.forward
    calls = 0

    def non_cpu_output(*args, **kwargs):
        nonlocal calls
        calls += 1
        output = original(*args, **kwargs)
        if calls == failed_forward:
            return (torch.empty_like(output[0], device="meta"), *output[1:])
        return output

    monkeypatch.setattr(selected, "forward", non_cpu_output)
    with pytest.raises(ValueError, match="non-CPU") as caught:
        compare_t5_donor_patch(model, **workflow_inputs("encoder"), synchronous_cpu=True)
    assert caught.value.__traceback__ is not None
    assert calls == failed_forward
    assert any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    with pytest.raises(RuntimeError, match="already leased"):
        ModelLease(model).acquire()


def test_sync_workflow_quarantines_reentrant_root_call(monkeypatch):
    model = tiny_t5()
    inputs = workflow_inputs("encoder")
    nested = dict(input_ids=inputs["receiver_input_ids"], attention_mask=inputs["attention_mask"],
                  decoder_input_ids=inputs["receiver_decoder_input_ids"],
                  decoder_attention_mask=inputs["decoder_attention_mask"], use_cache=False)

    def reenter(*args, **kwargs):
        return model(**nested)

    monkeypatch.setattr(model.encoder.block[1], "forward", reenter)
    with pytest.raises(RuntimeError, match="exactly one root model forward") as caught:
        compare_t5_donor_patch(model, **inputs, synchronous_cpu=True)
    assert caught.value.__traceback__ is not None
    assert any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    with pytest.raises(RuntimeError, match="already leased"):
        ModelLease(model).acquire()


@pytest.mark.parametrize("bad", [None, 0, 1, "true"])
def test_sync_flag_requires_an_explicit_boolean_before_forward(monkeypatch, bad):
    model = tiny_t5()
    selected = model.encoder.block[1]
    called = []
    original = selected.forward

    def count(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(selected, "forward", count)
    with pytest.raises((TypeError, ValueError), match="synchronous_cpu"):
        compare_t5_donor_patch(model, **workflow_inputs("encoder"), synchronous_cpu=bad)
    assert called == []
    assert_unleased_and_hook_free(model)


@pytest.mark.parametrize("bad", [None, 0, 1, "true"])
def test_trace_sync_flag_requires_an_explicit_boolean_before_forward(monkeypatch, bad):
    model = tiny_t5()
    called = []
    original = model.encoder.block[1].forward

    def count(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.encoder.block[1], "forward", count)
    with pytest.raises(TypeError, match="synchronous_cpu"):
        trace_t5_forward(model, input_ids=torch.tensor([[4, 5, 1]]),
                         attention_mask=torch.ones((1, 3), dtype=torch.long),
                         decoder_input_ids=torch.tensor([[0, 7, 1]]),
                         decoder_attention_mask=torch.ones((1, 3), dtype=torch.long),
                         points=[T5TracePoint("encoder", 1, "block", 0, 1)],
                         run_id="bad-sync", synchronous_cpu=bad)
    assert called == []
    assert_unleased_and_hook_free(model)
