"""Synthetic CPU checks for a ViT donor-state intervention workflow."""

import hashlib
import json
import math
import subprocess
import sys

import pytest
import torch
from transformers import ViTConfig, ViTForImageClassification

from looking_glass.vit_intervention_workflow import compare_vit_donor_patch
from looking_glass.vit_trace import ViTTracePoint, patch_token_index


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_vit():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        config = ViTConfig(image_size=[32, 48], patch_size=[8, 16], num_channels=3,
            hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=24,
            num_labels=7, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        config._attn_implementation = "eager"
        return ViTForImageClassification(config).eval()


def test_cls_donor_patch_lands_and_preserves_other_tokens():
    model = tiny_vit()
    receiver = torch.linspace(-1, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
    donor = receiver.flip(-1).contiguous()
    patch = patch_token_index(1, 2, grid_rows=4, grid_columns=3)
    result = compare_vit_donor_patch(model, receiver_pixels=receiver, donor_pixels=donor,
        target=ViTTracePoint(0, "block", 0, 0), control=ViTTracePoint(0, "block", 0, patch),
        readout_class=3)
    before = result.witnesses["plans"]["target"]["before"]
    after = result.witnesses["plans"]["target"]["after"]
    expected = before.view(torch.int32).clone()
    expected[0, 0] = result.witnesses["donor"]["0"].view(torch.int32)
    assert torch.equal(after.view(torch.int32), expected)
    assert result.report["sites"]["target"]["token_kind"] == "cls"
    assert result.report["sites"]["off_target"]["processed_pixel_box_xyxy"] == [32, 8, 48, 16]
    assert all(result.report["controls"].values())
    assert result.report["retained_tensor_bytes"] <= result.report["retained_tensor_byte_bound"]
    assert result.report["inputs"]["receiver_sha256"] == hashlib.sha256(
        receiver.view(torch.int32).contiguous().numpy().tobytes()).hexdigest()
    assert result.report["inputs"]["donor_sha256"] == hashlib.sha256(
        donor.view(torch.int32).contiguous().numpy().tobytes()).hexdigest()
    json.dumps(result.report, allow_nan=False)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("boundary", ["attention", "intermediate", "block"])
@pytest.mark.parametrize("target_token,control_token", [(0, 6), (6, 0)])
def test_selected_state_reaches_real_downstream_boundary(monkeypatch, boundary, target_token, control_token):
    model = tiny_vit()
    layer = model.vit.encoder.layer[0]
    receiver = torch.linspace(-1, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
    donor = receiver.flip(-1).contiguous()
    destination = {"attention": layer.layernorm_after, "intermediate": layer.output,
                   "block": model.vit.encoder.layer[1]}[boundary]
    seen, residual = [], []

    def observe(module, store):
        original = module.forward

        def wrapper(*args, **kwargs):
            store.append(args[0].detach().clone())
            return original(*args, **kwargs)
        monkeypatch.setattr(module, "forward", wrapper)

    observe(destination, seen)
    if boundary == "attention":
        observe(layer.layernorm_before, residual)
    result = compare_vit_donor_patch(model, receiver_pixels=receiver, donor_pixels=donor,
        target=ViTTracePoint(0, boundary, 0, target_token),
        control=ViTTracePoint(0, boundary, 0, control_token), readout_class=2)
    # Execution order is receiver, donor, self, null, target, off-target, optional reflected, repeat.
    after = result.witnesses["plans"]["target"]["after"]
    expected = after if boundary != "attention" else residual[4] + after
    assert torch.equal(seen[4].view(torch.int32), expected.view(torch.int32))
    assert result.report["forwards"] == len(seen)
    assert result.report["sites"]["target"]["token_kind"] == ("cls" if target_token == 0 else "patch")
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_reported_doses_are_realized_float32_displacements():
    model = tiny_vit()
    receiver = torch.linspace(-1, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
    donor = receiver.flip(-1).contiguous()
    result = compare_vit_donor_patch(model, receiver_pixels=receiver, donor_pixels=donor,
        target=ViTTracePoint(0, "block", 0, 0), control=ViTTracePoint(0, "block", 0, 6),
        readout_class=3, relative_dose_tolerance=0.05)
    for name, token in (("target", 0), ("off_target", 6), ("reflected", 0)):
        state = result.witnesses["plans"][name]
        before = state["before"][0, token].tolist()
        after = state["after"][0, token].tolist()
        independent_l2 = math.sqrt(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(after, before)))
        assert result.report["doses"][name]["l2"] == pytest.approx(independent_l2, rel=1e-14)
    target_dose = result.report["doses"]["target"]["l2"]
    for name in ("off_target", "reflected"):
        dose = result.report["doses"][name]
        assert dose["ratio_to_target"] == pytest.approx(dose["l2"] / target_dose)
        assert dose["status"] == ("matched" if abs(dose["ratio_to_target"] - 1) <= 0.05 else "unmatched")
    assert result.report["doses"]["reflected"]["active"]


def test_identical_donor_has_no_active_reflection_or_false_dose_ratio():
    model = tiny_vit()
    pixels = torch.linspace(-1, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
    result = compare_vit_donor_patch(model, receiver_pixels=pixels, donor_pixels=pixels.clone(),
        target=ViTTracePoint(0, "block", 0, 0), control=ViTTracePoint(0, "block", 0, 6), readout_class=1)
    assert result.report["doses"]["target"] == {"l2": 0.0, "active": False}
    assert result.report["doses"]["off_target"] == {
        "l2": 0.0, "active": False, "ratio_to_target": None, "status": "zero_target"}
    assert result.report["doses"]["reflected"] == {
        "l2": None, "active": False, "ratio_to_target": None, "status": "zero_target"}
    assert result.witnesses["reflected_replacement"] is None
    assert "reflected" not in result.witnesses["plans"]
    assert result.report["forwards"] == 7


@pytest.mark.parametrize("mutation", ["shape", "same_site", "outside_token", "bad_class", "nonfinite",
                                      "input_limit", "tensor_limit", "retained_limit", "bad_tolerance", "meta"])
def test_invalid_requests_stop_before_forward(mutation):
    model = tiny_vit()
    receiver = torch.zeros(1, 3, 32, 48)
    donor = torch.ones_like(receiver)
    target, control = ViTTracePoint(0, "block", 0, 0), ViTTracePoint(0, "block", 0, 6)
    options = {"readout_class": 1}
    if mutation == "shape":
        donor = donor[:, :, :16]
    elif mutation == "same_site":
        control = target
    elif mutation == "outside_token":
        control = ViTTracePoint(0, "block", 0, 13)
    elif mutation == "bad_class":
        options["readout_class"] = True
    elif mutation == "nonfinite":
        donor[0, 0, 0, 0] = float("nan")
    elif mutation == "input_limit":
        options["max_input_bytes"] = 100
    elif mutation == "tensor_limit":
        options["max_tensor_bytes"] = 4
    elif mutation == "retained_limit":
        options["max_retained_bytes"] = 4
    elif mutation == "bad_tolerance":
        options["relative_dose_tolerance"] = float("nan")
    else:
        donor = torch.empty((1, 3, 32, 48), device="meta")
    calls = []
    handle = model.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises((TypeError, ValueError)):
            compare_vit_donor_patch(model, receiver_pixels=receiver, donor_pixels=donor,
                                    target=target, control=control, **options)
        assert calls == []
    finally:
        handle.remove()


def test_late_forward_exception_removes_hooks_and_allows_cpu_reuse(monkeypatch):
    model = tiny_vit()
    pixels = torch.randn(1, 3, 32, 48)
    module = model.vit.encoder.layer[0].output
    original, calls = module.forward, []

    def fail_during_target(*args, **kwargs):
        calls.append(True)
        if len(calls) == 5:
            raise RuntimeError("synthetic late failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "forward", fail_during_target)
    with pytest.raises(RuntimeError, match="synthetic late failure"):
        compare_vit_donor_patch(model, receiver_pixels=pixels, donor_pixels=pixels.flip(-1),
            target=ViTTracePoint(0, "block", 0, 0), control=ViTTracePoint(0, "block", 0, 6), readout_class=1)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    monkeypatch.setattr(module, "forward", original)
    result = compare_vit_donor_patch(model, receiver_pixels=pixels, donor_pixels=pixels.flip(-1),
        target=ViTTracePoint(0, "block", 0, 0), control=ViTTracePoint(0, "block", 0, 6), readout_class=1)
    assert all(result.report["controls"].values())


def test_repeated_invocation_rejects_and_cleans_hooks(monkeypatch):
    model = tiny_vit()
    encoder, pixels = model.vit.encoder, torch.zeros(1, 3, 32, 48)
    original = encoder.forward

    def repeat(*args, **kwargs):
        original(*args, **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(encoder, "forward", repeat)
    with pytest.raises(RuntimeError, match="max_invocations"):
        compare_vit_donor_patch(model, receiver_pixels=pixels, donor_pixels=pixels,
            target=ViTTracePoint(0, "block", 0, 0), control=ViTTracePoint(0, "block", 0, 6), readout_class=1)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_big_endian_is_rejected_before_any_forward(monkeypatch):
    model, pixels = tiny_vit(), torch.zeros(1, 3, 32, 48)
    calls = []
    handle = model.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        monkeypatch.setattr(sys, "byteorder", "big")
        with pytest.raises(ValueError, match="little-endian"):
            compare_vit_donor_patch(model, receiver_pixels=pixels, donor_pixels=pixels,
                target=ViTTracePoint(0, "block", 0, 0), control=ViTTracePoint(0, "block", 0, 6), readout_class=1)
        assert calls == []
    finally:
        handle.remove()


def test_target_arm_fatal_interruption_propagates_and_dirty_model_is_not_reused():
    # Run separately because the intentionally interrupted model may keep its lease.
    script = """
import torch
from transformers import ViTConfig, ViTForImageClassification
from looking_glass.vit_trace import ViTTracePoint
from looking_glass.vit_intervention_workflow import compare_vit_donor_patch
torch.set_num_threads(1)
cfg = ViTConfig(image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=24, num_labels=3)
cfg._attn_implementation = 'eager'
model = ViTForImageClassification(cfg).eval()
module = model.vit.encoder.layer[0].output
original = module.forward
calls = [0]
def fail(*args, **kwargs):
    calls[0] += 1
    if calls[0] == 5:
        raise KeyboardInterrupt('inside target arm')
    return original(*args, **kwargs)
module.forward = fail
pixels = torch.zeros(1, 3, 32, 32)
options = dict(receiver_pixels=pixels, donor_pixels=pixels.clone(),
               target=ViTTracePoint(0, 'block', 0, 0),
               control=ViTTracePoint(0, 'block', 0, 1), readout_class=1)
try:
    compare_vit_donor_patch(model, **options)
except KeyboardInterrupt as error:
    assert str(error) == 'inside target arm'
else:
    raise AssertionError('interruption did not propagate')
assert calls[0] == 5
dirty = any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
if dirty:
    try:
        compare_vit_donor_patch(model, **options)
    except (ValueError, RuntimeError):
        assert calls[0] == 5
    else:
        raise AssertionError('dirty model was reused')
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
