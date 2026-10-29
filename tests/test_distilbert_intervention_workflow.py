"""Fabricated CPU checks of the reusable known-intervention workflow."""

import json
import subprocess
import sys

import pytest
import torch

pytest.importorskip("transformers")
from transformers import DistilBertConfig, DistilBertForMaskedLM

from looking_glass.distilbert_intervention_workflow import _numerically_equal, run_known_distilbert_intervention


@pytest.fixture
def setup():
    config = DistilBertConfig(vocab_size=41, max_position_embeddings=16, dim=16,
                              hidden_dim=32, n_layers=2, n_heads=2, dropout=0,
                              attention_dropout=0)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = DistilBertForMaskedLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    opts = {"injection_layer": 0, "restoration_layer": 1, "target_token": 1,
            "control_token": 2, "readout_position": 3, "readout_token_id": 5,
            "coordinate": 0, "amount": 5.0}
    return model, ids, mask, opts


def same_bits(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32),
                                               b.contiguous().view(torch.int32))


def test_numerical_readout_equality_ignores_only_zero_sign():
    positive_zero = torch.tensor([0], dtype=torch.int32).view(torch.float32)
    negative_zero = torch.tensor([-2147483648], dtype=torch.int32).view(torch.float32)
    subnormal = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    assert _numerically_equal(positive_zero, negative_zero)
    assert not _numerically_equal(positive_zero, subnormal)


def test_float32_subnormal_measurements_survive_denormal_flush():
    code = r'''
import sys
import torch
from looking_glass.distilbert_intervention_workflow import _float32_values, _l2, _numerically_equal
zero = torch.tensor([0], dtype=torch.int32).view(torch.float32)
small = torch.tensor([1], dtype=torch.int32).view(torch.float32)
negative = torch.tensor([-2147483647], dtype=torch.int32).view(torch.float32)
for mode in (False, True):
    if not torch.set_flush_denormal(mode):
        sys.exit(77)
    assert _float32_values(small) == [2.0 ** -149]
    assert _float32_values(negative) == [-(2.0 ** -149)]
    assert _l2(small, zero) == 2.0 ** -149
    assert _l2(small, negative) == 2.0 ** -148
    assert not _numerically_equal(small, zero)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    if result.returncode == 77:
        pytest.skip("runtime cannot configure CPU denormal mode")
    assert result.returncode == 0, result.stdout + result.stderr


def test_injection_restore_off_target_and_full_witnesses(setup):
    model, ids, mask, opts = setup
    result = run_known_distilbert_intervention(model, ids, mask, **opts)
    report, w = result.report, result.witnesses
    json.dumps(report, allow_nan=False)
    assert report["input_ids"] == [1, 2, 3, 4, 0]
    assert report["attention_mask"] == [1, 1, 1, 1, 0]
    assert report["injection"]["status"] == "nonzero"
    assert report["restoration"]["realized_l2"] > 0
    assert report["off_target"]["status"] == "active"
    assert report["off_target"]["realized_l2"] > 0
    assert report["readout"]["injected_vector_changed"]
    assert report["restoration_comparison"]["status"] == "descriptive"
    assert all(report["controls"].values())
    assert report["retained_tensor_bytes"] <= report["retained_tensor_byte_bound"]
    assert len(report["clean_trace_events"]) == 4
    assert len(report["plan_metadata"]["target"]["steps"]) == 6
    for kind, position in (("target", 1), ("off_target", 2)):
        before = w["plans"][kind]["restoration_before"]
        after = w["plans"][kind]["restoration_after"]
        assert same_bits(after[0, position], w["clean_trace"][f"layer1-token{position}"][0, 0])
        expected = before.clone()
        expected[0, position] = after[0, position]
        assert same_bits(expected, after)
        assert same_bits(before, w["plans"]["injected"]["restoration_before"])
    assert same_bits(w["plans"]["undo"]["undo_after"], w["plans"]["injected"]["injection_before"])
    assert same_bits(w["readout_logits"]["self"], w["readout_logits"]["injected"])
    assert same_bits(w["readout_logits"]["undo"], w["readout_logits"]["clean"])
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_rounded_to_zero_injection_is_preserved_as_unavailable(setup):
    model, ids, mask, opts = setup
    opts["amount"] = 1e-300
    report = run_known_distilbert_intervention(model, ids, mask, **opts).report
    assert report["injection"]["status"] == "zero_displacement"
    assert report["off_target"]["status"] == "unavailable_zero_injection"
    assert report["restoration_comparison"]["status"] == "unavailable"
    assert not report["readout"]["injected_vector_changed"]


def test_unchanged_readout_does_not_claim_restoration(setup):
    model, ids, mask, opts = setup
    with torch.no_grad():
        model.vocab_projector.weight.zero_()
        model.vocab_projector.bias.zero_()
    report = run_known_distilbert_intervention(model, ids, mask, **opts).report
    assert report["injection"]["status"] == "nonzero"
    assert not report["readout"]["injected_vector_changed"]
    assert report["restoration_comparison"]["status"] == "unavailable"
    assert "injection did not change the selected readout vector" in report["restoration_comparison"]["reasons"]


def test_unchanged_selected_logit_is_reported_even_if_other_logits_move(setup):
    model, ids, mask, opts = setup
    with torch.no_grad():
        model.vocab_projector.weight[opts["readout_token_id"]].zero_()
        model.vocab_projector.bias[opts["readout_token_id"]] = 0
    report = run_known_distilbert_intervention(model, ids, mask, **opts).report
    assert report["readout"]["injected_vector_changed"]
    assert not report["readout"]["injected_selected_logit_changed"]
    assert report["restoration_comparison"]["status"] == "unavailable"
    assert "injection did not change the selected vocabulary logit" in report["restoration_comparison"]["reasons"]


def test_zero_dose_off_target_is_not_called_an_active_control(setup):
    model, ids, mask, opts = setup
    with torch.no_grad():
        model.distilbert.transformer.layer[1].attention.v_lin.weight.zero_()
        model.distilbert.transformer.layer[1].attention.v_lin.bias.zero_()
    report = run_known_distilbert_intervention(model, ids, mask, **opts).report
    assert report["injection"]["status"] == "nonzero"
    assert report["off_target"]["realized_l2"] == 0
    assert report["off_target"]["status"] == "unavailable_zero_displacement"
    assert report["restoration_comparison"]["status"] == "unavailable"


@pytest.mark.parametrize("change", [
    {"restoration_layer": 0}, {"control_token": 1}, {"control_token": 4},
    {"readout_position": 4}, {"coordinate": 16}, {"readout_token_id": 41},
    {"amount": 0}, {"amount": float("inf")}, {"amount": True},
])
def test_invalid_or_padded_coordinates_fail_before_forward(setup, change):
    model, ids, mask, opts = setup
    opts.update(change)
    with pytest.raises(ValueError):
        run_known_distilbert_intervention(model, ids, mask, **opts)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_later_forward_failure_removes_hooks_and_returns_no_result(setup, monkeypatch):
    model, ids, mask, opts = setup
    block = model.distilbert.transformer.layer[1]
    original = block.forward
    calls = 0
    def fail_on_injected(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("fabricated later failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(block, "forward", fail_on_injected)
    with pytest.raises(RuntimeError, match="fabricated later failure"):
        run_known_distilbert_intervention(model, ids, mask, **opts)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
