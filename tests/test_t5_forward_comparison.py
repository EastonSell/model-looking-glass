"""Paired uncached T5 trace comparison on tiny, fabricated CPU inputs."""

import importlib.util
from dataclasses import replace

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from looking_glass.flan_t5_trace import T5TracePoint
from looking_glass.transformer_localization import compare_transformer_traces
from looking_glass.trace_schema import GenerationContext


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def paired():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(109)
        config = T5Config(vocab_size=41, d_model=16, d_kv=8, d_ff=32, num_layers=2,
                          num_decoder_layers=2, num_heads=2, dropout_rate=0,
                          decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        model = T5ForConditionalGeneration(config).eval()
    return model, torch.tensor([[4, 5, 1, 0]]), torch.tensor([[1, 1, 1, 0]]), \
        torch.tensor([[0, 7, 1]]), torch.tensor([[1, 1, 1]])


def test_paired_t5_events_compare_without_treating_cache_positions_as_known(paired):
    # Dropping either stack or treating selected [hidden] values as full slices must fail.
    assert importlib.util.find_spec("looking_glass.t5_forward") is not None
    from looking_glass.t5_forward import trace_t5_comparable_forward

    model, source, source_mask, decoder, decoder_mask = paired
    points = (T5TracePoint("encoder", 0, "block", 0, 1),
              T5TracePoint("decoder", 0, "block", 0, 1))
    def run(run_id):
        return trace_t5_comparable_forward(model, input_ids=source, attention_mask=source_mask,
            decoder_input_ids=decoder, decoder_attention_mask=decoder_mask, points=points, run_id=run_id)
    left, right = run("left"), run("right")
    report = compare_transformer_traces(left.events, right.events, reference_complete=True,
        candidate_complete=True, pairing_verified=True, execution_order_verified=True,
        t5_contexts=(left.t5_context, right.t5_context), expected_t5_encoder_differences=(),
        expected_t5_decoder_differences=())
    assert report["status"] == "no_difference_observed"
    assert report["schema_coverage"]["t5_forward_context"]["verified"] is True
    assert [item.event.module_path for item in left.events] == ["encoder.block.0", "decoder.block.0"]
    assert all(item.value.shape == (1, 1, 16) and item.original_saved_shape == (16,)
               and item.comparison_transform == "restore_singleton_batch_and_token_axes"
               and item.event.generation.cache_position is None for item in left.events)


@pytest.fixture
def traces(paired):
    from looking_glass.t5_forward import trace_t5_comparable_forward

    model, source, source_mask, decoder, decoder_mask = paired
    points = (T5TracePoint("encoder", 0, "block", 0, 1),
              T5TracePoint("decoder", 0, "block", 0, 1))
    def run(run_id, *, source_ids=source, source_attention=source_mask,
            decoder_ids=decoder, decoder_attention=decoder_mask, selected=points):
        return trace_t5_comparable_forward(model, input_ids=source_ids, attention_mask=source_attention,
                                           decoder_input_ids=decoder_ids,
                                           decoder_attention_mask=decoder_attention,
                                           points=selected, run_id=run_id)
    return run


def compare(left, right, *, encoder=(), decoder=(), **kwargs):
    flags = dict(reference_complete=True, candidate_complete=True, pairing_verified=True,
                 execution_order_verified=True)
    flags.update(kwargs)
    return compare_transformer_traces(left.events, right.events,
        t5_contexts=(left.t5_context, right.t5_context),
        expected_t5_encoder_differences=encoder, expected_t5_decoder_differences=decoder, **flags)


def test_declared_encoder_and_decoder_changes_are_checked_separately(traces):
    left = traces("left")
    encoder_changed = traces("encoder", source_ids=torch.tensor([[4, 6, 1, 0]]))
    decoder_changed = traces("decoder", decoder_ids=torch.tensor([[0, 8, 1]]))
    assert compare(left, encoder_changed, encoder=((0, 1),))["status"] == "difference_observed"
    assert compare(left, encoder_changed)["status"] == "inconclusive"
    assert compare(left, encoder_changed, decoder=((0, 1),))["status"] == "inconclusive"
    assert compare(left, decoder_changed, decoder=((0, 1),))["status"] == "difference_observed"
    assert compare(left, decoder_changed)["status"] == "inconclusive"
    assert compare(left, decoder_changed, decoder=None)["status"] == "inconclusive"


def test_an_equal_encoder_event_is_not_a_preceding_decoder_boundary(traces):
    left = traces("left")
    right = traces("right", decoder_ids=torch.tensor([[0, 8, 1]]))
    result = compare(left, right, decoder=((0, 1),))
    interval = result["first_observed_interval"]
    assert interval["first_unequal"]["module_path"] == "decoder.block.0"
    assert interval["last_equal"] is None


def test_mask_changes_and_masked_token_changes_are_not_paired(traces):
    left = traces("left")
    changed_mask = traces("mask", source_attention=torch.tensor([[1, 1, 1, 1]]))
    changed_masked_id = traces("masked", source_ids=torch.tensor([[4, 5, 1, 9]]))
    assert compare(left, changed_mask)["status"] == "inconclusive"
    assert compare(left, changed_masked_id, encoder=((0, 3),))["status"] == "inconclusive"
    assert compare(left, changed_mask)["schema_coverage"]["t5_forward_context"]["verified"] is False


def test_absent_t5_context_and_caller_attestations_are_inconclusive(traces):
    left, right = traces("left"), traces("right")
    result = compare_transformer_traces(left.events, right.events, reference_complete=True,
        candidate_complete=True, pairing_verified=True, execution_order_verified=True)
    assert result["status"] == "inconclusive"
    assert result["schema_coverage"]["t5_forward_context"]["verified"] is False
    for flag in ("pairing_verified", "execution_order_verified", "reference_complete", "candidate_complete"):
        assert compare(left, right, **{flag: False})["status"] == "inconclusive"


def test_different_lengths_and_selected_policy_are_inconclusive(traces):
    left = traces("left")
    longer_decoder = traces("long", decoder_ids=torch.tensor([[0, 7, 1, 0]]),
                            decoder_attention=torch.tensor([[1, 1, 1, 0]]))
    other_point = traces("other", selected=(T5TracePoint("encoder", 0, "block", 0, 1),
                                             T5TracePoint("decoder", 1, "block", 0, 1)))
    assert compare(left, longer_decoder)["status"] == "inconclusive"
    assert compare(left, other_point)["status"] == "inconclusive"


@pytest.mark.parametrize("field,value", [
    ("invocation", 1),
    ("output_path", (1,)),
    ("module_kind", "t5_decoder_feed_forward_residual"),
    ("generation", GenerationContext(stage="decode", use_cache=True, generation_step=1,
                                      cache_position=1, cache_length_before=1, cache_length_after=2)),
])
def test_event_coordinate_or_cache_mutations_cannot_validate(traces, field, value):
    left, right = traces("left"), traces("right")
    altered = replace(right.events[1], event=replace(right.events[1].event, **{field: value}))
    forged = replace(right, events=(right.events[0], altered))
    assert compare(left, forged)["status"] == "inconclusive"


def test_duplicate_or_missing_events_and_wrong_context_cannot_validate(traces):
    left, right = traces("left"), traces("right")
    for forged in (replace(right, events=right.events[:1]),
                   replace(right, events=(right.events[0], right.events[0])),
                   replace(right, t5_context=left.t5_context)):
        assert compare(left, forged)["status"] == "inconclusive"


def test_caller_inputs_are_cloned_and_persistent_model_input_mutation_rejected(paired):
    from looking_glass.t5_forward import trace_t5_comparable_forward
    model, source, source_mask, decoder, decoder_mask = paired
    original = model.encoder.forward
    def mutate_after_encoder(*args, **kwargs):
        result = original(*args, **kwargs)
        kwargs["input_ids"][0, 0] = 9
        return result
    model.encoder.forward = mutate_after_encoder
    with pytest.raises(ValueError, match="changed the recorded inputs"):
        trace_t5_comparable_forward(model, input_ids=source, attention_mask=source_mask,
            decoder_input_ids=decoder, decoder_attention_mask=decoder_mask,
            points=(T5TracePoint("encoder", 0, "block", 0, 1),), run_id="left")
    assert source.tolist() == [[4, 5, 1, 0]]
    assert all(not module._forward_hooks for module in model.modules())


def test_model_cannot_publish_an_int_to_float_mutated_input_clone(paired):
    from looking_glass.t5_forward import trace_t5_comparable_forward
    model, source, source_mask, decoder, decoder_mask = paired
    original = model.encoder.forward
    def change_dtype_after_encoder(*args, **kwargs):
        result = original(*args, **kwargs)
        ids = kwargs["input_ids"]
        ids.data = ids.data.to(torch.float32)
        return result
    model.encoder.forward = change_dtype_after_encoder
    with pytest.raises(ValueError, match="changed the recorded inputs"):
        trace_t5_comparable_forward(model, input_ids=source, attention_mask=source_mask,
            decoder_input_ids=decoder, decoder_attention_mask=decoder_mask,
            points=(T5TracePoint("encoder", 0, "block", 0, 1),), run_id="left")
    assert source.dtype == torch.int64 and source.tolist() == [[4, 5, 1, 0]]
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("index", range(4))
def test_every_cloned_input_requires_original_dtype_shape_and_cpu_device(paired, index):
    from looking_glass.t5_forward import _inputs_match
    _, source, source_mask, decoder, decoder_mask = paired
    inputs = (source, source_mask, decoder, decoder_mask)
    shapes = tuple(tuple(item.shape) for item in inputs)
    snapshot = tuple(tuple(tuple(row) for row in item.tolist()) for item in inputs)
    assert _inputs_match(inputs, shapes, snapshot)
    for mutated in (inputs[index].float(), inputs[index].reshape(-1),
                    torch.empty(shapes[index], dtype=torch.int64, device="meta")):
        changed = list(inputs)
        changed[index] = mutated
        assert not _inputs_match(tuple(changed), shapes, snapshot)


def test_multibatch_stack_local_masks_and_each_boundary(paired, monkeypatch):
    from looking_glass import t5_forward
    trace_t5_comparable_forward = t5_forward.trace_t5_comparable_forward
    original_trace = t5_forward.trace_t5_forward
    raw_records = []
    def keep_raw(*args, **kwargs):
        output, records = original_trace(*args, **kwargs)
        raw_records.append(records)
        return output, records
    monkeypatch.setattr(t5_forward, "trace_t5_forward", keep_raw)
    model, _, _, _, _ = paired
    source = torch.tensor([[4, 5, 1, 0], [6, 7, 8, 1]])
    source_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    decoder = torch.tensor([[0, 7, 1], [0, 9, 0]])
    decoder_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    points = tuple(T5TracePoint(stack, 0, role, batch, token)
                   for stack, batch, token in (("encoder", 1, 2), ("decoder", 1, 1))
                   for role in ("block", "self_attention", "feed_forward"))
    def run(name):
        return trace_t5_comparable_forward(model, input_ids=source, attention_mask=source_mask,
            decoder_input_ids=decoder, decoder_attention_mask=decoder_mask, points=points, run_id=name)
    left, right = run("left"), run("right")
    assert compare(left, right)["status"] == "no_difference_observed"
    assert [event.event.module_path for event in left.events] == [
        "encoder.block.0.layer.0", "encoder.block.0.layer.1", "encoder.block.0",
        "decoder.block.0.layer.0", "decoder.block.0.layer.2", "decoder.block.0"]
    # Inference tensors need not track ``_base``; compare the storage pointer.
    assert all(item.value.data_ptr() == raw.value.data_ptr()
               for item, raw in zip(left.events, raw_records[0], strict=True))
    source[1, 2] = 2
    decoder_mask[1, 2] = 1
    assert left.t5_context.input_ids[1][2] == 8
    assert left.t5_context.decoder_attention_mask[1][2] == 0
    assert "input_ids=" not in repr(left.t5_context)


def test_decoder_mask_change_and_invalid_difference_declarations_are_inconclusive(traces):
    left = traces("left")
    right = traces("right")
    mask_changed = replace(right, t5_context=replace(right.t5_context,
                           decoder_attention_mask=((1, 1, 0),)))
    assert compare(left, mask_changed)["status"] == "inconclusive"
    for invalid in (None, ((0, 1), (0, 1)), ((False, 1),), ((0, -1),), [(0, 1)]):
        assert compare(left, right, encoder=invalid)["status"] == "inconclusive"


def test_same_stack_equal_predecessor_is_retained(traces):
    points = (T5TracePoint("encoder", 0, "block", 0, 1),
              T5TracePoint("decoder", 0, "self_attention", 0, 1),
              T5TracePoint("decoder", 0, "block", 0, 1))
    left, right = traces("left", selected=points), traces("right", selected=points)
    altered = replace(right.events[-1], value=right.events[-1].value + 1)
    result = compare(left, replace(right, events=right.events[:-1] + (altered,)))
    assert result["status"] == "difference_observed"
    assert result["first_observed_interval"]["last_equal"]["module_path"] == "decoder.block.0.layer.0"


def test_inputs_are_rejected_before_model_on_unbounded_or_mismatched_mask(paired):
    from looking_glass.t5_forward import trace_t5_comparable_forward
    model, source, source_mask, decoder, decoder_mask = paired
    kwargs = dict(input_ids=source, attention_mask=source_mask, decoder_input_ids=decoder,
                  decoder_attention_mask=decoder_mask,
                  points=(T5TracePoint("encoder", 0, "block", 0, 1),), run_id="left")
    with pytest.raises(ValueError, match="bounded"):
        trace_t5_comparable_forward(model, **{**kwargs, "attention_mask": torch.ones(1, 5000, dtype=torch.long)})
    with pytest.raises(ValueError, match="budgets"):
        trace_t5_comparable_forward(model, **kwargs, max_input_tokens=4097)
