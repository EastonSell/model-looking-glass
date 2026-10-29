"""Fabricated paired ViT traces with explicit processed-grid policies."""

from dataclasses import replace
import math

import pytest
import torch
from transformers import ViTConfig, ViTForImageClassification

from looking_glass.trace_schema import RunIdentity
from looking_glass.vit_trace import ViTTracePoint, trace_vit_classifier
from looking_glass.vit_trace_comparison import compare_vit_traces


POINTS = (ViTTracePoint(1, "block", 0, 0), ViTTracePoint(0, "attention", 0, 1),
          ViTTracePoint(1, "intermediate", 0, 7), ViTTracePoint(0, "block", 0, 0))


@pytest.fixture(scope="module")
def traces():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(901)
        config = ViTConfig(image_size=[32, 48], patch_size=[8, 16], num_channels=3,
                           hidden_size=16, intermediate_size=24, num_attention_heads=2,
                           num_hidden_layers=2, num_labels=5, hidden_dropout_prob=0,
                           attention_probs_dropout_prob=0)
        config._attn_implementation = "eager"
        model = ViTForImageClassification(config).eval()
    reference = torch.zeros((1, 3, 32, 48))
    candidate = reference.clone()
    candidate[:, :, :8, :16] = 1.0
    a = trace_vit_classifier(model, reference, points=POINTS, run_id="reference")
    b = trace_vit_classifier(model, candidate, points=POINTS, run_id="candidate")
    same = trace_vit_classifier(model, reference, points=POINTS, run_id="same-image")
    yield a, b, same
    torch.set_num_threads(old_threads)


def compare(a, b, **kwargs):
    return compare_vit_traces(a, b, points=POINTS, pairing_verified=True, **kwargs)


def mutate_record(result, index, **kwargs):
    records = list(result.records)
    records[index] = replace(records[index], **kwargs)
    return replace(result, records=tuple(records))


def test_declared_alignment_and_quantitative_differences(traces):
    a, b, _ = traces
    without_attestation = compare_vit_traces(a, b, points=POINTS)
    assert without_attestation["status"] == "inconclusive"
    assert without_attestation["comparisons"] == []
    result = compare(a, b)
    assert result["status"] == "compared" and result["causal_origin"] == "unknown"
    assert [(x["layer"], x["module"], x["token"]) for x in result["comparisons"]] == [
        (0, "attention", 1), (0, "block", 0), (1, "intermediate", 7), (1, "block", 0)]
    assert result["comparisons"][0]["processed_pixel_box_xyxy"] == [0, 0, 16, 8]
    assert result["comparisons"][1]["token_kind"] == "cls"
    assert result["comparisons"][1]["processed_pixel_box_xyxy"] is None
    for row, left, right in zip(result["comparisons"], a.records, b.records, strict=True):
        difference = left.value.double() - right.value.double()
        assert row["l2_difference"] == pytest.approx(torch.linalg.vector_norm(difference).item())
        assert row["max_absolute_difference"] == pytest.approx(difference.abs().max().item())
        assert row["mean_absolute_difference"] == pytest.approx(difference.abs().mean().item())
        assert row["numeric_equality"] in ("equal", "different")
    assert any(row["l2_difference"] > 0 for row in result["comparisons"])


def test_same_input_is_a_no_difference_control(traces):
    a, _, same = traces
    result = compare(a, same)
    assert result["status"] == "compared"
    assert all(row["numeric_equality"] == "equal" and row["l2_difference"] == 0
               and row["bitwise_changed_elements"] == 0 for row in result["comparisons"])


@pytest.mark.parametrize("kind", ["missing", "extra", "reordered", "bytes", "run_id", "geometry", "row",
                                  "bool_row", "box", "bool_box", "token_kind", "module", "layer", "invocation", "token", "shape",
                                  "selector", "generation", "value"])
def test_bad_or_unpaired_metadata_returns_no_partial_metrics(traces, kind):
    a, b, _ = traces
    record = b.records[0]
    if kind == "missing":
        bad = replace(b, records=b.records[:-1])
    elif kind == "extra":
        bad = replace(b, records=b.records + (b.records[0],))
    elif kind == "reordered":
        bad = replace(b, records=tuple(reversed(b.records)))
    elif kind == "bytes":
        bad = replace(b, tensor_bytes=b.tensor_bytes + 4)
    elif kind == "run_id":
        bad = replace(b, records=tuple(replace(r, event=replace(r.event, identity=RunIdentity("reference")))
                                       for r in b.records))
    elif kind == "geometry":
        geometry = dict(b.image_metadata, grid_shape=[4, 4])
        bad = replace(b, image_metadata=geometry)
    elif kind == "row":
        bad = mutate_record(b, 0, patch_row=3)
    elif kind == "bool_row":
        bad = mutate_record(b, 0, patch_row=False)
    elif kind == "box":
        bad = mutate_record(b, 0, processed_pixel_box_xyxy=(0, 0, 1, 1))
    elif kind == "bool_box":
        bad = mutate_record(b, 0, processed_pixel_box_xyxy=(False, 0, 16, 8))
    elif kind == "token_kind":
        bad = mutate_record(b, 0, token_kind="cls")
    elif kind == "module":
        bad = mutate_record(b, 0, event=replace(record.event, module_kind="vit_block"))
    elif kind == "layer":
        bad = mutate_record(b, 0, event=replace(record.event, layer_index=1))
    elif kind == "invocation":
        bad = mutate_record(b, 0, event=replace(record.event, invocation=1))
    elif kind == "token":
        bad = mutate_record(b, 0, event=replace(record.event, token_positions=(2,)))
    elif kind == "shape":
        bad = mutate_record(b, 0, event=replace(record.event, shape=(1, 14, 16)))
    elif kind == "selector":
        bad = mutate_record(b, 0, event=replace(record.event, output_path=(0,)))
    elif kind == "generation":
        from looking_glass.trace_schema import GenerationContext
        bad = mutate_record(b, 0, event=replace(record.event, generation=GenerationContext(stage="forward")))
    else:
        bad = mutate_record(b, 0, value=record.value[:-1])
    result = compare(a, bad)
    assert result["status"] == "inconclusive"
    assert result["issues"] and result["comparisons"] == []


def test_distinct_valid_processed_grids_are_not_silently_paired(traces):
    a, _, _ = traces
    config = ViTConfig(image_size=[32, 48], patch_size=[4, 16], num_channels=3,
                       hidden_size=16, intermediate_size=24, num_attention_heads=2,
                       num_hidden_layers=2, num_labels=5)
    config._attn_implementation = "eager"
    model = ViTForImageClassification(config).eval()
    b = trace_vit_classifier(model, torch.zeros((1, 3, 32, 48)), points=POINTS, run_id="other-grid")
    result = compare(a, b)
    assert result["status"] == "inconclusive" and result["comparisons"] == []
    assert "processed-image geometries differ" in result["issues"]


def test_subnormal_and_signed_zero_comparison_under_flush(traces):
    a, _, same = traces
    left_words = torch.zeros(16, dtype=torch.int32)
    right_words = torch.zeros(16, dtype=torch.int32)
    left_words[1] = 1
    right_words[0] = -2147483648
    right_words[1] = 2
    left = mutate_record(a, 0, value=left_words.view(torch.float32))
    right = mutate_record(same, 0, value=right_words.view(torch.float32))
    probe = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    old_flush = int((probe + probe).view(torch.int32).item()) == 0
    supported = torch.set_flush_denormal(True)
    try:
        result = compare(left, right)
    finally:
        if supported:
            torch.set_flush_denormal(old_flush)
    row = result["comparisons"][0]
    assert row["numeric_equality"] == "different"
    assert row["numerically_changed_elements"] == 1 and row["bitwise_changed_elements"] == 2
    assert row["l2_difference"] == math.ldexp(1.0, -149)
    assert row["max_absolute_difference"] == math.ldexp(1.0, -149)
    assert row["mean_absolute_difference"] == math.ldexp(1.0, -153)


def test_nonfinite_and_budget_failure_do_not_return_partial_metrics(traces):
    a, _, same = traces
    words = torch.zeros(16, dtype=torch.int32)
    words[0] = 0x7f800000
    bad = mutate_record(same, 0, value=words.view(torch.float32))
    result = compare(a, bad)
    assert result["status"] == "inconclusive" and result["comparisons"] == []
    assert "nonfinite" in result["issues"][0]
    budget = compare(a, same, max_trace_bytes=a.tensor_bytes)
    assert budget["status"] == "inconclusive" and budget["comparisons"] == []
    assert "combined trace value byte budget exceeded" in budget["issues"]


def test_invalid_policy_or_attestation_type_is_rejected(traces):
    a, b, _ = traces
    with pytest.raises(ValueError, match="distinct"):
        compare_vit_traces(a, b, points=[POINTS[0], POINTS[0]], pairing_verified=True)
    with pytest.raises(ValueError, match="boolean"):
        compare_vit_traces(a, b, points=POINTS, pairing_verified=1)
    with pytest.raises(ValueError, match="budget"):
        compare_vit_traces(a, b, points=POINTS, pairing_verified=True, max_events=1025)
