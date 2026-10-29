"""Bounded eager ViT image-classifier traces with processed-grid coordinates."""

from dataclasses import dataclass
import hashlib

import torch

from .intervention_plan import ActivationPlan
from .trace_schema import GenerationContext, RunIdentity, TraceEvent


@dataclass(frozen=True, slots=True)
class ViTTracePoint:
    """One layer boundary and local batch/token coordinate, invocation zero."""

    layer: int
    module: str
    batch: int
    token: int
    invocation: int = 0

    def __post_init__(self):
        if type(self.module) is not str or self.module not in ("attention", "intermediate", "block"):
            raise ValueError("unsupported ViT boundary")
        for name in ("layer", "batch", "token", "invocation"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.invocation != 0:
            raise ValueError("only the first invocation is supported")


@dataclass(frozen=True, slots=True)
class ViTTraceRecord:
    """Trace metadata and a detached vector, plus its processed-grid mapping."""

    event: TraceEvent
    value: torch.Tensor
    token_kind: str
    patch_row: int | None
    patch_column: int | None
    processed_pixel_box_xyxy: tuple[int, int, int, int] | None


@dataclass(frozen=True, slots=True)
class ViTTraceResult:
    """Model output, selected records, processed-image geometry and saved bytes."""

    output: object
    records: tuple[ViTTraceRecord, ...]
    image_metadata: dict
    tensor_bytes: int


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _pair(value, name):
    if type(value) is int:
        value = (value, value)
    elif type(value) in (tuple, list):
        value = tuple(value)
    else:
        raise ValueError(f"{name} must be a positive integer or pair")
    if len(value) != 2 or any(type(x) is not int or x <= 0 for x in value):
        raise ValueError(f"{name} must be a positive integer or pair")
    return value


def patch_token_index(row, column, *, grid_rows, grid_columns):
    """Convert a zero-based processed patch-grid coordinate to its token index."""
    _positive(grid_rows, "grid_rows")
    _positive(grid_columns, "grid_columns")
    if (type(row) is not int or type(column) is not int or not 0 <= row < grid_rows
            or not 0 <= column < grid_columns):
        raise ValueError("patch row or column is outside the grid")
    return 1 + row * grid_columns + column


def trace_vit_classifier(model, pixel_values, *, points, run_id, max_events=128,
                         max_trace_bytes=4 * 1024 * 1024, max_tensor_bytes=16 * 1024 * 1024,
                         max_input_bytes=16 * 1024 * 1024, max_logits_bytes=4 * 1024 * 1024):
    """Run one eager CPU image-classifier forward and retain selected tokens.

    Images are already preprocessed; grid boxes refer to this input tensor, not
    an original image. The caller must own the model and pairing semantics.
    """
    from transformers import ViTForImageClassification
    from transformers.models.vit.modeling_vit import (
        ViTAttention, ViTEmbeddings, ViTIntermediate, ViTLayer, ViTPatchEmbeddings,
    )

    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        raise ValueError("compiled execution is unsupported")
    if type(model) is not ViTForImageClassification or model.config.model_type != "vit":
        raise TypeError("model must be a standard ViTForImageClassification")
    if (getattr(model.forward, "__func__", None) is not ViTForImageClassification.forward
            or model.config._attn_implementation != "eager" or model.vit.config._attn_implementation != "eager"):
        raise ValueError("only an unmodified eager root forward is supported")
    if (any(module.training for module in model.modules()) or model.is_gradient_checkpointing
            or torch.is_autocast_enabled("cpu")):
        raise ValueError("evaluation mode without checkpointing or autocast is required")
    if (any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
            or any(b.device.type != "cpu" for b in model.buffers())):
        raise ValueError("model parameters must be CPU float32 and buffers must be on CPU")
    if (torch.nn.modules.module._global_forward_pre_hooks or torch.nn.modules.module._global_forward_hooks
            or any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())):
        raise ValueError("pre-existing local or global forward hooks are unsupported")
    for name, value in (("max_events", max_events), ("max_trace_bytes", max_trace_bytes),
                        ("max_tensor_bytes", max_tensor_bytes), ("max_input_bytes", max_input_bytes),
                        ("max_logits_bytes", max_logits_bytes)):
        _positive(value, name)
    identity = RunIdentity(run_id)
    if len(run_id) > 128:
        raise ValueError("run_id must contain at most 128 characters")
    if (type(points) not in (tuple, list) or not points or len(points) > max_events
            or any(type(point) is not ViTTracePoint for point in points) or len(set(points)) != len(points)):
        raise ValueError("points must be a bounded nonempty list of distinct ViTTracePoint values")
    if (type(pixel_values) is not torch.Tensor or pixel_values.device.type != "cpu"
            or pixel_values.dtype != torch.float32 or pixel_values.ndim != 4 or not pixel_values.numel()
            or pixel_values.requires_grad):
        raise ValueError("pixel_values must be a nonempty CPU float32 [batch, channel, height, width] tensor")
    if pixel_values.numel() * pixel_values.element_size() > max_input_bytes:
        raise ValueError("input pixel byte budget exceeded")
    batch, channels, height, width = pixel_values.shape
    image_size = _pair(model.config.image_size, "image_size")
    patch_size = _pair(model.config.patch_size, "patch_size")
    hidden, intermediate = model.config.hidden_size, model.config.intermediate_size
    if (type(hidden) is not int or type(intermediate) is not int or min(hidden, intermediate) <= 0
            or (height, width) != image_size or channels != model.config.num_channels
            or height % patch_size[0] or width % patch_size[1]):
        raise ValueError("pixel shape or configured image/patch dimensions are unsupported")
    rows, columns = height // patch_size[0], width // patch_size[1]
    sequence = 1 + rows * columns
    embeddings = model.vit.embeddings
    patch = embeddings.patch_embeddings
    projection = patch.projection
    if (type(embeddings) is not ViTEmbeddings or type(patch) is not ViTPatchEmbeddings
            or getattr(embeddings.forward, "__func__", None) is not ViTEmbeddings.forward
            or getattr(patch.forward, "__func__", None) is not ViTPatchEmbeddings.forward
            or type(projection) is not torch.nn.Conv2d
            or getattr(projection.forward, "__func__", None) is not torch.nn.Conv2d.forward
            or tuple(projection.kernel_size) != patch_size or tuple(projection.stride) != patch_size
            or tuple(projection.padding) != (0, 0) or tuple(projection.dilation) != (1, 1)
            or projection.in_channels != channels or projection.out_channels != hidden
            or _pair(patch.image_size, "patch image size") != image_size
            or _pair(patch.patch_size, "patch size") != patch_size or patch.num_channels != channels
            or patch.num_patches != rows * columns
            or tuple(embeddings.cls_token.shape) != (1, 1, hidden)
            or tuple(embeddings.position_embeddings.shape) != (1, sequence, hidden)):
        raise ValueError("patch embedding geometry is not the supported row-major layout")
    layers = model.vit.encoder.layer
    if (not isinstance(layers, torch.nn.ModuleList) or not layers
            or any(type(layer) is not ViTLayer or type(layer.attention) is not ViTAttention
                   or type(layer.intermediate) is not ViTIntermediate
                   or layer.attention.attention.config._attn_implementation != "eager" for layer in layers)):
        raise ValueError("standard eager ViT layers are required")
    if type(model.classifier) is not torch.nn.Linear or model.classifier.out_features != model.config.num_labels:
        raise ValueError("standard ViT classification head is required")
    if batch * model.classifier.out_features * 4 > max_logits_bytes:
        raise ValueError("logit output byte budget exceeded")
    if any(point.layer >= len(layers) or point.batch >= batch or point.token >= sequence for point in points):
        raise ValueError("selected layer, image or token is out of range")
    widths = {"attention": hidden, "intermediate": intermediate, "block": hidden}
    if sum(widths[p.module] * 4 for p in points) > max_trace_bytes:
        raise ValueError("selected snapshots exceed max_trace_bytes")
    if any(batch * sequence * widths[p.module] * 4 > max_tensor_bytes for p in points):
        raise ValueError("selected module output exceeds max_tensor_bytes")

    # Clone integer encodings first, preserving the caller's exact float32 bits.
    pixel_bits = pixel_values.detach().view(torch.int32).contiguous().clone()
    pixels = pixel_bits.view(torch.float32)
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError("pixel_values must be finite")
    def pixel_digest():
        bits = pixels.detach().view(torch.int32).contiguous()
        return hashlib.sha256(memoryview(bits.numpy()).cast("B")).digest()
    before_digest = pixel_digest()
    dimensions = (len(layers), hidden, intermediate, model.config.num_labels, image_size, patch_size)
    order = {"attention": 0, "intermediate": 1, "block": 2}
    ordered = sorted(points, key=lambda p: (p.layer, order[p.module], p.batch, p.token))
    aliases = {}
    for path, module in model.named_modules(remove_duplicate=False):
        aliases.setdefault(id(module), []).append(path)
    steps = []
    for index, point in enumerate(ordered):
        path = f"vit.encoder.layer.{point.layer}"
        if point.module != "block":
            path += f".{point.module}"
        if aliases[id(model.get_submodule(path))] != [path]:
            raise ValueError("selected module must have a unique path")
        steps.append({"name": f"capture_{index}", "operation": "capture", "module_path": path,
                      "invocation": 0, "batch_index": point.batch, "token_index": point.token})
    plan = ActivationPlan(model, steps, max_steps=max_events, max_invocations=1,
                          max_tensor_bytes=max_tensor_bytes, max_capture_bytes=max_trace_bytes, max_source_bytes=1)
    with torch.inference_mode(), plan:
        output = model(pixel_values=pixels, interpolate_pos_encoding=False, output_attentions=False,
                       output_hidden_states=False, return_dict=True)
    if (pixels.dtype != torch.float32 or pixels.device.type != "cpu"
            or tuple(pixels.shape) != (batch, channels, height, width)
            or pixel_digest() != before_digest
            or (len(model.vit.encoder.layer), model.config.hidden_size, model.config.intermediate_size,
                model.config.num_labels, _pair(model.config.image_size, "image_size"),
                _pair(model.config.patch_size, "patch_size")) != dimensions):
        raise RuntimeError("pixel inputs or model geometry changed during the forward")
    if (type(output.logits) is not torch.Tensor or output.logits.dtype != torch.float32
            or output.logits.device.type != "cpu" or tuple(output.logits.shape) != (batch, model.config.num_labels)
            or not bool(torch.isfinite(output.logits).all())):
        raise RuntimeError("classifier did not return finite CPU float32 logits")
    captured, metadata = plan.captures, plan.metadata
    if len(metadata["steps"]) != len(ordered) or len(captured) != len(ordered):
        raise RuntimeError("plan omitted a requested ViT capture")
    records = []
    for index, (point, step) in enumerate(zip(ordered, metadata["steps"], strict=True)):
        name, path = f"capture_{index}", steps[index]["module_path"]
        value, feature_width = captured[name], widths[point.module]
        if (step["name"] != name or step["order"] != index or step["operation"] != "capture"
                or step["module_path"] != path or step["module_aliases"] != [path]
                or step["forward_index"] != 0 or step["invocation"] != 0
                or step["batch_index"] != point.batch or step["token_index"] != point.token
                or step["output_selector"] != [] or step["output_shape"] != [batch, sequence, feature_width]
                or step["selected_shape"] != [feature_width] or step["dtype"] != "torch.float32"
                or step["device"] != "cpu" or tuple(value.shape) != (feature_width,)
                or value.dtype != torch.float32 or value.device.type != "cpu"):
            raise RuntimeError("captured output differs from the declared ViT boundary")
        event = TraceEvent(identity=identity, sequence=index, forward_index=0, module_path=path,
            module_kind=f"vit_{point.module}", layer_index=point.layer, invocation=0, output_path=(),
            shape=(batch, sequence, feature_width), dtype=step["dtype"], device=step["device"],
            capture_mode="direct", batch_positions=(point.batch,), token_positions=(point.token,),
            batch_axis=0, token_axis=1, generation=GenerationContext(), saved_shape=tuple(value.shape),
            saved_dtype=str(value.dtype), saved_device=str(value.device))
        if point.token == 0:
            kind, row, col, box = "cls", None, None, None
        else:
            kind = "patch"
            row, col = divmod(point.token - 1, columns)
            box = (col * patch_size[1], row * patch_size[0], (col + 1) * patch_size[1], (row + 1) * patch_size[0])
        records.append(ViTTraceRecord(event, value, kind, row, col, box))
    size = sum(record.value.numel() * record.value.element_size() for record in records)
    if size != metadata["capture_bytes"] or size > max_trace_bytes:
        raise RuntimeError("captured ViT bytes differ from the declared budget")
    geometry = {"pixel_shape": [batch, channels, height, width], "patch_shape": list(patch_size),
                "grid_shape": [rows, columns], "sequence_length": sequence, "cls_token_index": 0,
                "patch_order": "row-major", "box_coordinates": "processed-pixel-xyxy-exclusive"}
    return ViTTraceResult(output, tuple(records), geometry, size)
