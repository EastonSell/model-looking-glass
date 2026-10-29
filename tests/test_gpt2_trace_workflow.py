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
