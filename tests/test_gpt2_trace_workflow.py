"""Synthetic CPU checks of the composition, with no pretrained model."""

import json

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass.gpt2_trace_workflow import compare_known_corruption
from looking_glass import gpt2_trace_workflow


def tiny_model():
    config = GPT2Config(vocab_size=32, n_positions=8, n_embd=16, n_layer=2,
                        n_head=2, use_cache=False, attn_pdrop=0, resid_pdrop=0,
                        embd_pdrop=0)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return GPT2LMHeadModel(config).eval()


def test_trace_compare_and_restore_with_same_site_and_comparison_controls():
    model = tiny_model()
    ids = torch.tensor([[3, 4]])
    report = compare_known_corruption(model, ids, coordinate=0, amount=5.0, readout_token=5)
    assert report["schema"] == "gpt2-known-corruption-workflow-v1"
    assert ids.tolist() == [[3, 4]]
    assert report["localization"]["status"] == "difference_observed"
    assert report["localization"]["causal_origin"] == "unknown"
    assert any(x["module_path"] == "transformer.h.0" for x in report["candidate_sites"])
    assert report["trace_limits"]["clean_tensor_bytes"] == 6 * 16 * 4
    result = report["interventions"]
    assert result["status"] == "complete"
    assert all(result["controls"].values())
    assert report["baselines"]["corrupt"]["max_absolute_logit_distance_to_clean"] > 0
    assert result["target_patch"]["measurement"]["max_absolute_logit_distance_to_clean"] == 0
    assert result["target_self_patch"]["witness"]["realized_displacement_l2"] == 0
    for arm in ("target_self_patch", "target_patch", "comparison_patch"):
        assert all(result[arm]["witness"]["checks"].values())
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in model.modules())
    json.dumps(report, allow_nan=False)


def test_optional_matched_active_control_uses_actual_same_forward_doses():
    model = tiny_model()
    report = compare_known_corruption(model, torch.tensor([[3, 4]]), coordinate=0,
        amount=5.0, readout_token=5, matched_control=True)
    assert report["schema"] == "gpt2-known-corruption-workflow-v2"
    assert report["interventions"]["target_patch"]["measurement"]["max_absolute_logit_distance_to_clean"] == 0
    matched = report["interventions"]["matched_control"]
    assert matched["status"] == "complete"
    q = matched["common_dose_l2"]
    assert q == min(matched["natural_target_l2"], matched["natural_comparison_l2"])
    doses = []
    for arm in ("target", "comparison"):
        result = matched[arm]
        assert all(result["witness"]["checks"].values())
        dose = result["witness"]["realized_displacement_l2"]
        assert dose > 0 and abs(dose - q) <= matched["absolute_dose_tolerance"]
        assert 0 < result["interpolation_fraction"] <= 1
        doses.append(dose)
    assert abs(doses[0] - doses[1]) <= 0.01 * doses[0]
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    json.dumps(report, allow_nan=False)


def test_matched_planner_handles_unequal_zero_and_rounding_collapsed_directions():
    from looking_glass.gpt2_trace_workflow import _plan_matched_patches
    target_corrupt = torch.tensor([0.0, 0.0])
    target_clean = torch.tensor([3.0, 4.0])
    control_corrupt = torch.tensor([0.0, 0.0])
    control_clean = torch.tensor([0.0, 10.0])
    result, replacements = _plan_matched_patches(target_clean, target_corrupt,
                                                  control_clean, control_corrupt)
    assert result["status"] == "ready" and result["common_dose_l2"] == 5.0
    assert torch.equal(replacements[0], target_clean)
    assert torch.equal(replacements[1], torch.tensor([0.0, 5.0]))
    assert result["target_fraction"] == 1.0 and result["comparison_fraction"] == 0.5
    result, replacements = _plan_matched_patches(target_clean, target_clean,
                                                  control_clean, control_corrupt)
    assert result["status"] == "dose_ineligible" and replacements is None
    result, replacements = _plan_matched_patches(torch.tensor([1e-7, 0.0]), target_corrupt,
                                                  control_clean, control_corrupt)
    assert result["status"] == "dose_ineligible" and replacements is None
    result, replacements = _plan_matched_patches(target_clean, target_corrupt,
                                                  target_clean, target_corrupt)
    assert result["status"] == "ready"
    assert result["target_fraction"] == result["comparison_fraction"] == 1.0
    with pytest.raises(RuntimeError, match="finite"):
        _plan_matched_patches(torch.tensor([float("nan"), 0.0]), target_corrupt,
                              control_clean, control_corrupt)
    tiny_target = torch.tensor([2e-6, 0.0])
    large_control = torch.tensor([100.0, 100.0])
    nearby_control = torch.tensor([100.00000762939453, 100.0])
    result, replacements = _plan_matched_patches(tiny_target, target_corrupt,
                                                  nearby_control, large_control)
    assert result["status"] == "dose_unmatchable" and replacements is None
    assert result["planned_comparison_l2"] == 0.0


def test_actual_dose_check_rejects_individually_near_q_but_unmatched_doses():
    from looking_glass.gpt2_trace_workflow import _dose_matches
    q = 0.0005
    tolerance = 0.01 * q
    assert _dose_matches(q, tolerance, 1.005 * q, 0.995 * q)
    assert not _dose_matches(q, tolerance, 1.009 * q, 0.991 * q)
    assert not _dose_matches(q, tolerance, q, 0.0)
    assert not _dose_matches(q, tolerance, q, float("nan"))


@pytest.mark.parametrize("status", ["dose_ineligible", "dose_unmatchable"])
def test_unavailable_matched_arm_preserves_full_restoration(monkeypatch, status):
    def unavailable(*_values):
        return {"status": status, "common_dose_l2": 0.0}, None
    monkeypatch.setattr(gpt2_trace_workflow, "_plan_matched_patches", unavailable)
    model = tiny_model()
    original_forward = model.forward
    calls = []
    def count_forward(*args, **kwargs):
        calls.append(True)
        return original_forward(*args, **kwargs)
    monkeypatch.setattr(model, "forward", count_forward)
    report = compare_known_corruption(model, torch.tensor([[3, 4]]), coordinate=0,
                                      amount=5.0, readout_token=5, matched_control=True)
    assert report["interventions"]["matched_control"]["status"] == status
    assert report["interventions"]["target_patch"]["measurement"]["max_absolute_logit_distance_to_clean"] == 0
    assert all(report["interventions"]["controls"].values())
    assert len(calls) == 8  # no additional forwards for an unavailable match


def test_equal_selected_site_is_not_presented_as_restoration():
    model = tiny_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    report = compare_known_corruption(model, torch.tensor([[3, 4]]), coordinate=0,
                                      amount=5.0, readout_token=5,
                                      target_site=(0, "attn"), comparison_site=(0, "block"))
    assert report["localization"]["status"] == "difference_observed"
    assert report["interventions"]["status"] == "not_run"
    assert report["baselines"]["input_ids_equal_clean_embeds"]
    assert "readout_logit" in report["baselines"]["corrupt"]


def test_invalid_or_unmatched_configuration_fails_before_model_run():
    model = tiny_model()
    ids = torch.tensor([[3, 4]])
    with pytest.raises(ValueError):
        compare_known_corruption(model, ids, coordinate=0, amount=0, readout_token=5)
    with pytest.raises(ValueError):
        compare_known_corruption(model, ids, coordinate=0, amount=5, readout_token=5,
                                 target_site=(0, "block"), comparison_site=(0, "block"))
    with pytest.raises(ValueError):
        compare_known_corruption(model, ids, coordinate=0, amount=float("nan"), readout_token=5)
    with pytest.raises(ValueError, match="matched_control"):
        compare_known_corruption(model, ids, coordinate=0, amount=5, readout_token=5, matched_control=1)
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(True))
    try:
        with pytest.raises(ValueError, match="hooks"):
            compare_known_corruption(model, ids, coordinate=0, amount=5, readout_token=5)
        assert calls == []
    finally:
        handle.remove()


@pytest.mark.parametrize("damage", ["missing", "reordered"])
def test_successful_but_incomplete_trace_does_not_attest_pairing(monkeypatch, damage):
    original_exit = gpt2_trace_workflow.GPT2TraceSession.__exit__
    original_compare = gpt2_trace_workflow.compare_transformer_traces
    comparisons = []

    def alter_events(session, *args):
        result = original_exit(session, *args)
        if session.run_id == "workflow-corrupt":
            assert session.successful
            if damage == "missing":
                session.events.pop()
            else:
                session.events[0], session.events[1] = session.events[1], session.events[0]
        return result

    def note_comparison(*args, **kwargs):
        comparisons.append(True)
        return original_compare(*args, **kwargs)

    monkeypatch.setattr(gpt2_trace_workflow.GPT2TraceSession, "__exit__", alter_events)
    monkeypatch.setattr(gpt2_trace_workflow, "compare_transformer_traces", note_comparison)
    with pytest.raises(RuntimeError, match="trace"):
        compare_known_corruption(tiny_model(), torch.tensor([[3, 4]]),
                                 coordinate=0, amount=5, readout_token=5)
    assert comparisons == []
