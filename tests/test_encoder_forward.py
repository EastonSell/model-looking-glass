"""The opt-in encoder comparison contract, using only a tiny random model."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("transformers")
from transformers import DistilBertConfig, DistilBertForMaskedLM

from looking_glass.distilbert_trace import trace_distilbert_masked_lm
from looking_glass.trace_schema import GenerationContext, TraceEvent
from looking_glass.transformer_localization import compare_transformer_traces


@pytest.fixture
def paired():
    config = DistilBertConfig(vocab_size=41, max_position_embeddings=8, dim=8,
                              hidden_dim=16, n_layers=2, n_heads=2, dropout=0,
                              attention_dropout=0)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = DistilBertForMaskedLM(config).eval()
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    def run(run_id, *, input_ids=ids, attention_mask=mask, **kwargs):
        return trace_distilbert_masked_lm(model, input_ids, attention_mask, run_id=run_id,
                                         layers=kwargs.pop("layers", (1, 0)),
                                         samples=kwargs.pop("samples", (1, 0)),
                                         tokens=kwargs.pop("tokens", (1,)), **kwargs)
    return run, ids, mask


def compare(left, right, *, expected=(), contexts=None, **options):
    options.setdefault("reference_complete", True)
    options.setdefault("candidate_complete", True)
    options.setdefault("pairing_verified", True)
    options.setdefault("execution_order_verified", True)
    return compare_transformer_traces(
        left.events, right.events,
        encoder_contexts=(left.encoder_context, right.encoder_context) if contexts is None else contexts,
        expected_input_differences=expected, **options)


def test_matching_encoder_forward_is_comparable_without_inventing_generation(paired):
    run, ids, mask = paired
    left, right = run("left"), run("right")
    report = compare(left, right)
    assert report["status"] == "no_difference_observed"
    assert report["schema_coverage"]["encoder_forward_context"]["verified"]
    assert report["schema_coverage"]["encoder_forward_context"]["actual_input_differences"] == []
    assert all(item.event.generation == GenerationContext() for item in left.events + right.events)
    # Context owns immutable copies rather than later caller mutations.
    ids[0, 0] = 9
    mask[1, 2] = 1
    assert left.encoder_context.input_ids[0][0] == 1
    assert left.encoder_context.attention_mask[1][2] == 0
    assert "input_ids=" not in repr(left.encoder_context)


def test_intended_changed_id_must_match_the_exact_declared_coordinates(paired):
    run, ids, _ = paired
    changed = ids.clone()
    changed[0, 1] = 9
    left, right = run("left"), run("right", input_ids=changed)
    report = compare(left, right, expected=((0, 1),))
    assert report["status"] == "difference_observed"
    assert report["schema_coverage"]["encoder_forward_context"]["actual_input_differences"] == [[0, 1]]
    assert report["causal_origin"] == "unknown"
    for expected in ((), ((0, 1), (1, 0)), ((1, 0),), ((0, 1), (0, 1)), None):
        assert compare(left, right, expected=expected)["status"] == "inconclusive"


def test_full_mask_and_input_shapes_are_checked_even_outside_sampled_positions(paired):
    run, ids, mask = paired
    altered_mask = mask.clone()
    altered_mask[1, 2] = 1
    left, right = run("left"), run("right", attention_mask=altered_mask)
    report = compare(left, right)
    assert report["status"] == "inconclusive"
    assert "attention masks differ" in report["schema_coverage"]["encoder_forward_context"]["issues"]
    shorter = run("short", input_ids=ids[:, :2], attention_mask=mask[:, :2])
    assert compare(left, shorter)["status"] == "inconclusive"


def test_selected_layers_tokens_samples_and_complete_order_are_bound(paired):
    run, _, _ = paired
    left, right = run("left"), run("right")
    for alternate in (run("layers", layers=(0,)), run("tokens", tokens=(0,)),
                      run("samples", samples=(0,)), run("modules", modules=("block",))):
        assert compare(left, alternate)["status"] == "inconclusive"
    for events in (right.events[:-1], right.events[::-1]):
        altered = replace(right, events=events)
        assert compare(left, altered)["status"] == "inconclusive"
    altered_event = replace(right.events[0].event, layer_index=1,
                            module_path="distilbert.transformer.layer.1.attention")
    altered = replace(right, events=(replace(right.events[0], event=altered_event),) + right.events[1:])
    assert compare(left, altered)["status"] == "inconclusive"


def test_missing_untrusted_or_mixed_context_and_false_attestations_stay_inconclusive(paired):
    run, _, _ = paired
    left, right = run("left"), run("right")
    old = compare_transformer_traces(left.events, right.events, reference_complete=True,
                                     candidate_complete=True, pairing_verified=True,
                                     execution_order_verified=True)
    assert old["status"] == "inconclusive"
    assert "encoder_forward_context" not in old["schema_coverage"]
    assert compare(left, right, contexts=(left.encoder_context, None))["status"] == "inconclusive"
    assert compare(left, right, expected=None)["status"] == "inconclusive"
    assert compare(left, right, contexts=(right.encoder_context, left.encoder_context))["status"] == "inconclusive"
    same_id = replace(right.encoder_context, run_id="left")
    assert compare(left, right, contexts=(left.encoder_context, same_id))["status"] == "inconclusive"
    different_dimensions = replace(right.encoder_context, num_layers=3)
    assert compare(left, right, contexts=(left.encoder_context, different_dimensions))["status"] == "inconclusive"
    unsupported = replace(right.encoder_context, contract="unreviewed-encoder")
    assert compare(left, right, contexts=(left.encoder_context, unsupported))["status"] == "inconclusive"
    for option in ("reference_complete", "candidate_complete", "pairing_verified", "execution_order_verified"):
        assert compare(left, right, **{option: False})["status"] == "inconclusive"
    # A previously known generation context must not bypass an invalid encoder contract.
    mixed = tuple(replace(item, event=replace(item.event, generation=GenerationContext(
        stage="forward", use_cache=False))) for item in right.events)
    assert compare(left, replace(right, events=mixed))["status"] == "inconclusive"
    left_known = tuple(replace(item, event=replace(item.event, generation=GenerationContext(
        stage="forward", use_cache=False))) for item in left.events)
    assert compare(replace(left, events=left_known), replace(right, events=mixed))["status"] == "inconclusive"
    decoder = replace(right.events[0], event=replace(right.events[0].event,
        module_path="decoder.block.0", module_kind="decoder", generation=GenerationContext(
            stage="decode", generation_step=0, cache_position=0,
            cache_length_before=0, cache_length_after=1, use_cache=True)))
    assert compare(left, replace(right, events=(decoder,) + right.events[1:]))["status"] == "inconclusive"


def test_legacy_event_serialization_and_unknown_context_are_unchanged(paired):
    run, _, _ = paired
    left = run("left")
    original = left.events[0].event
    assert TraceEvent.from_json(original.to_json()) == original
    assert original.generation == GenerationContext()


@pytest.mark.parametrize("field", ("input_ids", "attention_mask"))
def test_model_mutating_its_input_clone_cannot_publish_context(paired, monkeypatch, field):
    _, ids, mask = paired
    # The real root forward runs first; mutation on return must still fail closed.
    # Construct a fresh model explicitly so the modified method remains local.
    config = DistilBertConfig(vocab_size=41, max_position_embeddings=8, dim=8,
                              hidden_dim=16, n_layers=1, n_heads=2, dropout=0,
                              attention_dropout=0)
    config._attn_implementation = "eager"
    model = DistilBertForMaskedLM(config).eval()
    original = model.forward
    def mutate(*args, **kwargs):
        output = original(*args, **kwargs)
        kwargs[field][0, 0] = 0
        return output
    monkeypatch.setattr(model, "forward", mutate)
    with pytest.raises(RuntimeError, match="inputs or model dimensions changed"):
        trace_distilbert_masked_lm(model, ids, mask, run_id="mutation")
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    assert ids[0, 0].item() == 1 and mask[0, 0].item() == 1
