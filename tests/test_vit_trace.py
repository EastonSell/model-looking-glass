"""Tiny CPU checks for CLS/patch ViT tracing and bounded capture."""

import subprocess
import sys

import pytest
import torch
from transformers import ViTConfig, ViTForImageClassification

from looking_glass.vit_trace import ViTTracePoint, patch_token_index, trace_vit_classifier


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


def test_direct_hooks_match_ordered_cls_and_nonsquare_patch_captures():
    model = tiny_vit()
    pixels = torch.linspace(-1, 1, 2 * 3 * 32 * 48).reshape(2, 3, 32, 48)
    paths = ("vit.encoder.layer.0.attention", "vit.encoder.layer.0.intermediate", "vit.encoder.layer.0",
             "vit.encoder.layer.1.attention", "vit.encoder.layer.1.intermediate", "vit.encoder.layer.1")
    observed, handles = {}, []
    for path in paths:
        def capture(_module, _args, output, *, path=path):
            observed[path] = output.detach().clone()
        handles.append(model.get_submodule(path).register_forward_hook(capture))
    try:
        with torch.inference_mode():
            baseline = model(pixel_values=pixels, interpolate_pos_encoding=False,
                             output_attentions=False, output_hidden_states=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    points = [ViTTracePoint(1, "block", 1, 12), ViTTracePoint(0, "intermediate", 0, 0),
              ViTTracePoint(1, "attention", 1, 4), ViTTracePoint(0, "attention", 0, 1),
              ViTTracePoint(1, "intermediate", 1, 4), ViTTracePoint(0, "block", 0, 0)]
    result = trace_vit_classifier(model, pixels, points=points, run_id="rectangular")
    assert torch.equal(result.output.logits, baseline.logits)
    assert [record.event.module_path for record in result.records] == list(paths)
    assert result.image_metadata == {"pixel_shape": [2, 3, 32, 48], "patch_shape": [8, 16],
        "grid_shape": [4, 3], "sequence_length": 13, "cls_token_index": 0, "patch_order": "row-major",
        "box_coordinates": "processed-pixel-xyxy-exclusive"}
    expected_widths = (16, 24, 16, 16, 24, 16)
    for record, width in zip(result.records, expected_widths, strict=True):
        event = record.event
        batch, token = event.batch_positions[0], event.token_positions[0]
        assert torch.equal(record.value, observed[event.module_path][batch, token])
        assert event.shape == (2, 13, width) and event.saved_shape == (width,)
        assert event.output_path == () and event.invocation == event.forward_index == 0
        assert event.generation.stage == "unknown" and event.generation.use_cache is None
    assert result.tensor_bytes == sum(x * 4 for x in expected_widths)
    cls, first, middle, last = result.records[1], result.records[0], result.records[3], result.records[5]
    assert (cls.token_kind, cls.patch_row, cls.patch_column, cls.processed_pixel_box_xyxy) == ("cls", None, None, None)
    assert (first.token_kind, first.patch_row, first.patch_column, first.processed_pixel_box_xyxy) == ("patch", 0, 0, (0, 0, 16, 8))
    assert (middle.patch_row, middle.patch_column, middle.processed_pixel_box_xyxy) == (1, 0, (0, 8, 16, 16))
    assert (last.patch_row, last.patch_column, last.processed_pixel_box_xyxy) == (3, 2, (32, 24, 48, 32))
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_patch_index_roundtrips_cls_excluded_and_rejects_invalid_positions():
    assert patch_token_index(0, 0, grid_rows=4, grid_columns=3) == 1
    assert patch_token_index(3, 2, grid_rows=4, grid_columns=3) == 12
    for args in ((-1, 0, 4, 3), (4, 0, 4, 3), (0, 3, 4, 3), (False, 0, 4, 3)):
        with pytest.raises(ValueError):
            patch_token_index(args[0], args[1], grid_rows=args[2], grid_columns=args[3])


@pytest.mark.parametrize("case", ["masked-size", "token", "layer", "batch", "duplicate", "event-budget",
                                  "trace-budget", "tensor-budget", "input-budget", "logit-budget", "nonfinite"])
def test_invalid_geometry_and_limits_fail_without_leaked_hooks(case):
    model = tiny_vit()
    pixels = torch.zeros((1, 3, 32, 48))
    points = [ViTTracePoint(0, "intermediate", 0, 1)]
    kw = {}
    if case == "masked-size":
        pixels = torch.zeros((1, 3, 32, 32))
    elif case == "token":
        points = [ViTTracePoint(0, "block", 0, 13)]
    elif case == "layer":
        points = [ViTTracePoint(2, "block", 0, 0)]
    elif case == "batch":
        points = [ViTTracePoint(0, "block", 1, 0)]
    elif case == "duplicate":
        points = points * 2
    elif case == "event-budget":
        points.append(ViTTracePoint(0, "block", 0, 0))
        kw["max_events"] = 1
    elif case == "trace-budget":
        kw["max_trace_bytes"] = 95  # one intermediate vector needs 96
    elif case == "tensor-budget":
        kw["max_tensor_bytes"] = 1247  # the full intermediate output needs 1248
    elif case == "input-budget":
        kw["max_input_bytes"] = 18431
    elif case == "logit-budget":
        kw["max_logits_bytes"] = 27
    else:
        pixels[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        trace_vit_classifier(model, pixels, points=points, run_id="bad", **kw)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_unbounded_run_id_is_rejected():
    with pytest.raises(ValueError, match="run_id"):
        trace_vit_classifier(tiny_vit(), torch.zeros((1, 3, 32, 48)), run_id="x" * 129,
                             points=[ViTTracePoint(0, "block", 0, 0)])


def test_model_defaults_for_attentions_and_hidden_states_are_overridden():
    model = tiny_vit()
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    result = trace_vit_classifier(model, torch.zeros((1, 3, 32, 48)), run_id="defaults",
                                  points=[ViTTracePoint(1, "block", 0, 0)])
    assert result.output.attentions is None and result.output.hidden_states is None


def test_noncontiguous_input_is_copied_without_changing_the_caller():
    model = tiny_vit()
    pixels = torch.linspace(-1, 1, 3 * 48 * 32).reshape(1, 3, 48, 32).transpose(-1, -2)
    original = pixels.clone()
    with torch.inference_mode():
        baseline = model(pixel_values=pixels, interpolate_pos_encoding=False, return_dict=True)
    result = trace_vit_classifier(model, pixels, run_id="noncontiguous",
                                  points=[ViTTracePoint(0, "block", 0, 0)])
    assert torch.equal(result.output.logits, baseline.logits)
    assert torch.equal(pixels, original)


def test_modified_projection_geometry_and_nonzero_invocation_are_rejected():
    model = tiny_vit()
    model.vit.embeddings.patch_embeddings.projection.stride = (8, 8)
    with pytest.raises(ValueError, match="row-major"):
        trace_vit_classifier(model, torch.zeros((1, 3, 32, 48)), run_id="stride",
                             points=[ViTTracePoint(0, "block", 0, 0)])
    with pytest.raises(ValueError, match="first invocation"):
        ViTTracePoint(0, "block", 0, 0, invocation=1)


def test_late_runtime_failure_cleans_hooks_and_allows_reuse(monkeypatch):
    model = tiny_vit()
    pixels = torch.zeros((1, 3, 32, 48))
    late = model.vit.encoder.layer[1].output
    old = late.forward
    def fail(*args, **kwargs):
        raise RuntimeError("fabricated late failure")
    monkeypatch.setattr(late, "forward", fail)
    with pytest.raises(RuntimeError, match="fabricated late failure"):
        trace_vit_classifier(model, pixels, run_id="fail", points=[ViTTracePoint(0, "block", 0, 0)])
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    monkeypatch.setattr(late, "forward", old)
    assert len(trace_vit_classifier(model, pixels, run_id="reuse", points=[ViTTracePoint(0, "block", 0, 0)]).records) == 1


def test_compiled_context_and_preexisting_hooks_are_rejected(monkeypatch):
    model = tiny_vit()
    point = [ViTTracePoint(0, "block", 0, 0)]
    pixels = torch.zeros((1, 3, 32, 48))
    with monkeypatch.context() as context:
        context.setattr(torch.compiler, "is_compiling", lambda: True)
        with pytest.raises(ValueError, match="compiled"):
            trace_vit_classifier(model, pixels, run_id="compiled", points=point)
    handle = model.vit.encoder.layer[1].register_forward_hook(lambda *_: None)
    try:
        with pytest.raises(ValueError, match="pre-existing"):
            trace_vit_classifier(model, pixels, run_id="hook", points=point)
    finally:
        handle.remove()


def test_repeated_selected_module_invocation_is_rejected_and_cleans_hooks(monkeypatch):
    model = tiny_vit()
    encoder = model.vit.encoder
    old = encoder.forward
    def repeat(*args, **kwargs):
        old(*args, **kwargs)
        return old(*args, **kwargs)
    monkeypatch.setattr(encoder, "forward", repeat)
    pixels = torch.zeros((1, 3, 32, 48))
    with pytest.raises(RuntimeError, match="max_invocations"):
        trace_vit_classifier(model, pixels, run_id="repeat", points=[ViTTracePoint(0, "block", 0, 0)])
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    monkeypatch.setattr(encoder, "forward", old)
    assert len(trace_vit_classifier(model, pixels, run_id="after-repeat",
                                    points=[ViTTracePoint(0, "block", 0, 0)]).records) == 1


def test_fatal_interruption_is_propagated_and_dirty_model_is_not_reused():
    # A fresh process may retain a quarantined model and lease.
    script = """
import torch
from transformers import ViTConfig, ViTForImageClassification
from looking_glass.vit_trace import ViTTracePoint, trace_vit_classifier
torch.set_num_threads(1)
cfg = ViTConfig(image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=24, num_labels=3)
cfg._attn_implementation = 'eager'
model = ViTForImageClassification(cfg).eval()
module = model.vit.encoder.layer[0].intermediate
calls = [0]
def fail(*args, **kwargs):
    calls[0] += 1
    raise KeyboardInterrupt('inside ViT forward')
module.forward = fail
pixels = torch.zeros((1, 3, 32, 32))
points = [ViTTracePoint(0, 'block', 0, 0)]
try:
    trace_vit_classifier(model, pixels, run_id='interrupted', points=points)
except KeyboardInterrupt as error:
    assert str(error) == 'inside ViT forward'
else:
    raise AssertionError('interruption did not propagate')
assert calls[0] == 1
dirty = any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
if dirty:
    try:
        trace_vit_classifier(model, pixels, run_id='dirty', points=points)
    except (ValueError, RuntimeError):
        assert calls[0] == 1
    else:
        raise AssertionError('dirty model was reused')
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
