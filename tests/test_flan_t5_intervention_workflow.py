"""Fabricated CPU checks for the reusable T5 donor-patch composition."""

import subprocess
import sys

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from looking_glass.flan_t5_intervention_workflow import compare_t5_donor_patch
from looking_glass.flan_t5_trace import T5TracePoint


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def tiny_t5():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        cfg = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=2,
                       num_decoder_layers=2, num_heads=2, dropout_rate=0.0,
                       decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        return T5ForConditionalGeneration(cfg).eval()


def args(stack="encoder", module="block"):
    return dict(receiver_input_ids=torch.tensor([[4, 5, 1, 0]]),
        donor_input_ids=torch.tensor([[4, 6, 1, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0]]),
        receiver_decoder_input_ids=torch.tensor([[0, 7, 1]]),
        donor_decoder_input_ids=torch.tensor([[0, 7, 1]]), decoder_attention_mask=torch.tensor([[1, 1, 1]]),
        target=T5TracePoint(stack, 1, module, 0, 1), control=T5TracePoint(stack, 1, module, 0, 0),
        readout_position=1, readout_token_id=7)


def bits_equal(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize("stack", ["encoder", "decoder"])
@pytest.mark.parametrize("module", ["block", "self_attention", "feed_forward"])
def test_same_forward_landing_and_controls(stack, module):
    model = tiny_t5()
    result = compare_t5_donor_patch(model, **args(stack, module))
    report, values = result.report, result.witnesses
    assert report["forwards"] == 6
    assert all(report["controls"].values())
    assert report["inputs"]["encoder_mask"] == [1, 1, 1, 0]
    assert report["readout"]["decoder_position"] == 1
    assert report["retained_tensor_bytes"] == report["retained_tensor_byte_bound"]
    assert report["sites"]["target"]["stack"] == stack
    assert report["sites"]["target"]["invocation"] == 0
    for arm, token in (("target", 1), ("off_target", 0)):
        observed = values["plans"][arm]
        assert bits_equal(observed["before"][0, token], values["receiver"][str(token)])
        expected = observed["before"].view(torch.int32).clone()
        expected[0, token] = values["donor"][str(token)].view(torch.int32)
        assert torch.equal(observed["after"].view(torch.int32), expected)
        assert report["realized_l2"][arm] >= 0
        assert report["plan_metadata"][arm]["steps"][1]["token_index"] == token
    assert bits_equal(values["readout_logits"]["receiver"], values["readout_logits"]["self"])
    assert not any(x._forward_hooks or x._forward_pre_hooks for x in model.modules())


def test_identical_input_retains_zero_dose_instead_of_claiming_active_control():
    model = tiny_t5()
    data = args()
    data["donor_input_ids"] = data["receiver_input_ids"].clone()
    result = compare_t5_donor_patch(model, **data)
    assert result.report["realized_l2"] == {"target": 0.0, "off_target": 0.0}
    assert result.report["target_status"] == "zero_displacement"
    assert result.report["off_target_status"] == "unavailable_zero_displacement"
    assert all(result.report["controls"].values())


@pytest.mark.parametrize("change", ["masked-target", "masked-control", "masked-readout", "shape",
                                     "different-boundary", "same-site", "readout-id", "budget"])
def test_invalid_requests_fail_before_forward(change):
    model, data = tiny_t5(), args()
    if change == "masked-target":
        data["target"] = T5TracePoint("encoder", 1, "block", 0, 3)
    elif change == "masked-control":
        data["control"] = T5TracePoint("encoder", 1, "block", 0, 3)
    elif change == "masked-readout":
        data["decoder_attention_mask"][0, 1] = 0
    elif change == "shape":
        data["donor_input_ids"] = torch.tensor([[4, 6, 1]])
    elif change == "different-boundary":
        data["control"] = T5TracePoint("decoder", 1, "block", 0, 0)
    elif change == "same-site":
        data["control"] = data["target"]
    elif change == "readout-id":
        data["readout_token_id"] = 32
    else:
        data["max_retained_bytes"] = 1
    with pytest.raises(ValueError):
        compare_t5_donor_patch(model, **data)
    assert not any(x._forward_hooks or x._forward_pre_hooks for x in model.modules())


def test_late_error_cleans_hooks(monkeypatch):
    model, data = tiny_t5(), args()
    module = model.decoder.block[1].layer[2]
    old = module.forward
    calls = 0
    def interrupted(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 3:  # two traces succeeded; fail inside the first intervention forward
            raise RuntimeError("fabricated failure")
        return old(*a, **kw)
    monkeypatch.setattr(module, "forward", interrupted)
    with pytest.raises(RuntimeError, match="fabricated failure"):
        compare_t5_donor_patch(model, **data)
    assert not any(x._forward_hooks or x._forward_pre_hooks for x in model.modules())
    monkeypatch.setattr(module, "forward", old)
    assert all(compare_t5_donor_patch(model, **data).report["controls"].values())


def test_compiling_is_rejected_before_forward(monkeypatch):
    model = tiny_t5()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(ValueError, match="compiled"):
        compare_t5_donor_patch(model, **args())


def test_big_endian_is_rejected_before_forward(monkeypatch):
    model = tiny_t5()
    monkeypatch.setattr(sys, "byteorder", "big")
    with pytest.raises(ValueError, match="little-endian"):
        compare_t5_donor_patch(model, **args())
    assert not any(x._forward_hooks or x._forward_pre_hooks for x in model.modules())


@pytest.mark.parametrize("stack", ["encoder", "decoder"])
def test_interrupt_inside_forward_propagates_and_does_not_reuse_dirty_model(stack):
    # The isolated process may deliberately retain a quarantined model/lease.
    script = f"""
import torch
from transformers import T5Config, T5ForConditionalGeneration
from looking_glass.flan_t5_trace import T5TracePoint
from looking_glass.flan_t5_intervention_workflow import compare_t5_donor_patch
torch.set_num_threads(1)
cfg = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=2,
    num_decoder_layers=2, num_heads=2, dropout_rate=0, decoder_start_token_id=0, pad_token_id=0)
model = T5ForConditionalGeneration(cfg).eval()
target_module = model.get_submodule({(stack + '.block.1')!r})
original = target_module.forward
calls = [0]
def interrupt(*args, **kwargs):
    calls[0] += 1
    if calls[0] == 3:
        raise KeyboardInterrupt('inside selected boundary')
    return original(*args, **kwargs)
target_module.forward = interrupt
ids = torch.tensor([[4, 5, 1]])
donor = torch.tensor([[4, 6, 1]])
decoder = torch.tensor([[0, 7, 1]])
kwargs = dict(receiver_input_ids=ids, donor_input_ids=donor, attention_mask=torch.ones_like(ids),
    receiver_decoder_input_ids=decoder, donor_decoder_input_ids=decoder,
    decoder_attention_mask=torch.ones_like(decoder), target=T5TracePoint({stack!r}, 1, 'block', 0, 1),
    control=T5TracePoint({stack!r}, 1, 'block', 0, 0), readout_position=1, readout_token_id=7)
try:
    compare_t5_donor_patch(model, **kwargs)
except KeyboardInterrupt as error:
    assert str(error) == 'inside selected boundary'
else:
    raise AssertionError('interruption was not propagated')
assert calls[0] == 3
dirty = any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
try:
    result = compare_t5_donor_patch(model, **kwargs)
except (ValueError, RuntimeError):
    assert calls[0] == 3  # quarantined model was rejected before another forward
else:
    # A future runtime may complete cleanup on BaseException; only accept reuse
    # if no hooks remained and the full controls still pass.
    assert not dirty and all(result.report['controls'].values())
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_displacement_decodes_binary32_subnormals_without_float32_casts():
    script = """
import torch
from looking_glass.flan_t5_intervention_workflow import _l2
a = torch.tensor([0], dtype=torch.int32).view(torch.float32)
b = torch.tensor([1], dtype=torch.int32).view(torch.float32)
assert torch.set_flush_denormal(True)
assert _l2(a, b) == 2.0 ** -149
assert torch.set_flush_denormal(False)
assert _l2(a, b) == 2.0 ** -149
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
