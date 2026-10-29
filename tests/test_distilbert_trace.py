"""Synthetic eager DistilBERT trace checks; no checkpoint is downloaded."""

import pytest
import torch

pytest.importorskip("transformers")
from transformers import DistilBertConfig, DistilBertForMaskedLM

from looking_glass.distilbert_trace import trace_distilbert_masked_lm
from looking_glass.model_ownership import ModelLease
from looking_glass.transformer_localization import compare_transformer_traces


def tiny_model():
    config = DistilBertConfig(vocab_size=41, max_position_embeddings=16, dim=16,
                              hidden_dim=32, n_layers=2, n_heads=2, dropout=0,
                              attention_dropout=0)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        return DistilBertForMaskedLM(config).eval()


def test_selected_outputs_match_independent_hooks_and_preserve_logits():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    reference, handles = {}, []
    try:
        for layer in range(2):
            for kind in ("attention", "ffn", "block"):
                path = f"distilbert.transformer.layer.{layer}" + ("" if kind == "block" else f".{kind}")
                def save(_module, _args, output, *, name=path):
                    reference[name] = (output[0] if type(output) is tuple else output).detach().clone()
                handles.append(model.get_submodule(path).register_forward_hook(save))
        with torch.no_grad():
            clean = model(input_ids=ids, attention_mask=mask, output_attentions=False,
                          output_hidden_states=False, return_dict=True).logits
    finally:
        for handle in handles:
            handle.remove()

    # The adapter overrides a config that would otherwise change block tuple order.
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    result = trace_distilbert_masked_lm(model, ids, mask, run_id="test",
                                       layers=(1, 0), samples=(1, 0), tokens=(3, 1),
                                       max_events=24, max_trace_bytes=24 * 16 * 4)
    assert torch.equal(result.output.logits, clean)
    assert result.output.attentions is None
    assert result.output.hidden_states is None
    assert result.attention_mask == ((1, 1, 1, 1), (1, 1, 1, 0))
    assert len(result.events) == 24
    assert result.tensor_bytes == 24 * 16 * 4
    for sequence, item in enumerate(result.events):
        event = item.event
        assert event.sequence == sequence and event.forward_index == event.invocation == 0
        assert event.output_path == (() if event.module_kind == "ffn" else (0,))
        assert event.generation.stage == "unknown" and event.generation.use_cache is None
        assert item.original_saved_shape == (16,)
        assert item.comparison_transform == "restore_singleton_batch_and_token_axes"
        expected = reference[event.module_path][event.batch_positions[0], event.token_positions[0]]
        assert torch.equal(item.value[0, 0], expected)
        assert event.saved_shape == (1, 1, 16)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_encoder_generation_context_remains_unknown_for_comparison():
    model = tiny_model()
    ids, mask = torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long)
    a = trace_distilbert_masked_lm(model, ids, mask, run_id="a")
    b = trace_distilbert_masked_lm(model, ids, mask, run_id="b")
    report = compare_transformer_traces(a.events, b.events, reference_complete=True,
                                       candidate_complete=True, pairing_verified=True,
                                       execution_order_verified=True)
    assert report["status"] == "inconclusive"
    assert any("generation" in issue["fields"] for issue in report["schema_coverage"]["reference_metadata_issues"])


def test_budget_and_hook_guards_fail_before_model_forward():
    model = tiny_model()
    ids, mask = torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="max_events"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="a", max_events=2)
    with pytest.raises(ValueError, match="max_trace_bytes"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="a", max_trace_bytes=10)
    with pytest.raises(ValueError, match="max_tensor_bytes"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="a", max_tensor_bytes=10)
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(ValueError, match="hooks"):
            trace_distilbert_masked_lm(model, ids, mask, run_id="a")
        assert calls == []
    finally:
        handle.remove()
    with ModelLease(model):
        with pytest.raises(RuntimeError, match="leased"):
            trace_distilbert_masked_lm(model, ids, mask, run_id="a")


def test_forward_failure_removes_hooks_and_does_not_publish_partial_result(monkeypatch):
    model = tiny_model()
    ids, mask = torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long)
    def fail(*args, **kwargs):
        raise RuntimeError("fabricated failure")
    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="fabricated failure"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="a")
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_later_target_failure_discards_partial_capture_and_cleans_up(monkeypatch):
    model = tiny_model()
    ids, mask = torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long)
    def fail(*args, **kwargs):
        raise RuntimeError("fabricated later target failure")
    monkeypatch.setattr(model.distilbert.transformer.layer[1].ffn, "forward", fail)
    with pytest.raises(RuntimeError, match="later target failure"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="a", layers=(0, 1))
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_aliased_selected_modules_are_rejected():
    model = tiny_model()
    model.distilbert.transformer.layer[1] = model.distilbert.transformer.layer[0]
    ids, mask = torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="aliases"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="a", layers=(0, 1))


def test_non_eager_attention_is_rejected():
    model = tiny_model()
    model.config._attn_implementation = "sdpa"
    ids, mask = torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="eager"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="a")


def test_cpu_autocast_is_rejected():
    model = tiny_model()
    ids, mask = torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="a")


def test_compiling_context_is_rejected_before_forward(monkeypatch):
    model = tiny_model()
    ids, mask = torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="compiled"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="a")
