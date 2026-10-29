"""One bounded GPT-2 head-slice observation or patch on an eager CPU forward."""

from dataclasses import asdict, dataclass
import sys

import torch

from .copy_head_intervention import HeadProjectionWitness, ProjectionWitnessError, assert_no_forward_hooks
from .model_ownership import ModelLease
from .trace_schema import RunIdentity
from .transformer_trace import _note_cleanup


@dataclass(frozen=True, slots=True)
class GPT2HeadSlicePoint:
    """Zero-based local coordinates; this API supports batch and invocation zero."""

    layer: int
    token: int
    head: int
    batch: int = 0
    invocation: int = 0

    def __post_init__(self):
        for name in ("layer", "token", "head", "batch", "invocation"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.batch != 0 or self.invocation != 0:
            raise ValueError("only batch zero and invocation zero are supported")


@dataclass(frozen=True, slots=True)
class GPT2HeadSliceMetadata:
    """Coordinates of a slice of a module *input*, not an output trace event."""

    run_id: str
    module_path: str
    layer: int
    batch: int
    token: int
    invocation: int
    head: int
    heads: int
    head_width: int
    feature_start: int
    feature_end: int
    input_shape: tuple[int, int, int]
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    operation: str
    tensor_role: str = "projection_input"
    input_argument: int = 0
    forward_index: int = 0
    use_cache: bool = False
    dtype: str = "float32"
    device: str = "cpu"

    def to_dict(self):
        """Return independent JSON-compatible input-boundary metadata."""
        result = asdict(self)
        result["input_shape"] = list(self.input_shape)
        result["input_ids"] = list(self.input_ids)
        result["attention_mask"] = list(self.attention_mask)
        return result


@dataclass(frozen=True, slots=True)
class GPT2HeadSliceResult:
    """Detached full projection witnesses and logits from one completed call."""

    metadata: GPT2HeadSliceMetadata
    before: torch.Tensor
    after: torch.Tensor
    projection_output: torch.Tensor
    logits: torch.Tensor
    tensor_bytes: int

    @property
    def head_before(self):
        m = self.metadata
        return self.before[m.batch, m.token, m.feature_start:m.feature_end]

    @property
    def head_after(self):
        m = self.metadata
        return self.after[m.batch, m.token, m.feature_start:m.feature_end]


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _attention_layout(attention, projection):
    return (attention.embed_dim, attention.num_heads, attention.head_dim, attention.split_size,
            frozenset(attention.pruned_heads), attention.config._attn_implementation,
            attention.is_cross_attention, tuple(projection.weight.shape), tuple(projection.bias.shape))


class _LayoutCheckedWitness(HeadProjectionWitness):
    """Validate the head layout at the actual projection-input boundary."""

    def __init__(self, projection, *, attention, layout, **kwargs):
        super().__init__(projection, **kwargs)
        self.attention = attention
        self.layout = layout

    def _pre(self, module, arguments):
        if _attention_layout(self.attention, self.projection) != self.layout:
            raise ProjectionWitnessError("attention head layout changed at projection input")
        return super()._pre(module, arguments)


def run_gpt2_head_slice(model, input_ids, *, point, run_id, replacement=None,
                        max_input_bytes=16 * 1024, max_tensor_bytes=16 * 1024 * 1024,
                        max_capture_bytes=48 * 1024 * 1024,
                        max_source_bytes=64 * 1024, max_logits_bytes=16 * 1024 * 1024):
    """Observe or replace one head at the input of ``attn.c_proj``.

    The input is one unpadded CPU int64 token row. ``replacement=None`` observes;
    otherwise it must be a finite CPU float32 vector of exactly one head width.
    This performs one uncached forward. Its caller must own the model exclusively
    and establish any semantic correspondence between separate calls.
    """
    from transformers import GPT2LMHeadModel
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model
    from transformers.pytorch_utils import Conv1D

    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        raise ValueError("compiled execution is unsupported")
    if type(model) is not GPT2LMHeadModel or model.config.model_type != "gpt2":
        raise TypeError("model must be a standard GPT2LMHeadModel")
    if (getattr(model.forward, "__func__", None) is not GPT2LMHeadModel.forward
            or model.config._attn_implementation != "eager"
            or model.config.add_cross_attention):
        raise ValueError("only an unmodified eager self-attention root forward is supported")
    if (type(model.transformer) is not GPT2Model
            or getattr(model.transformer.forward, "__func__", None) is not GPT2Model.forward):
        raise ValueError("an unmodified GPT-2 transformer forward is required")
    if (any(module.training for module in model.modules()) or model.is_gradient_checkpointing
            or torch.is_autocast_enabled("cpu")):
        raise ValueError("evaluation mode without checkpointing or autocast is required")
    if (any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
            or any(b.device.type != "cpu" for b in model.buffers())):
        raise ValueError("model parameters must be CPU float32 and buffers on CPU")
    for name, value in (("max_input_bytes", max_input_bytes), ("max_tensor_bytes", max_tensor_bytes),
                        ("max_capture_bytes", max_capture_bytes), ("max_source_bytes", max_source_bytes),
                        ("max_logits_bytes", max_logits_bytes)):
        _positive(value, name)
    RunIdentity(run_id)
    if len(run_id) > 128:
        raise ValueError("run_id must contain at most 128 characters")
    if type(point) is not GPT2HeadSlicePoint:
        raise TypeError("point must be a GPT2HeadSlicePoint")
    if (type(input_ids) is not torch.Tensor or input_ids.device.type != "cpu"
            or input_ids.dtype != torch.int64 or input_ids.ndim != 2
            or input_ids.shape[0] != 1 or input_ids.shape[1] < 1):
        raise ValueError("input_ids must be one nonempty CPU int64 row")
    if input_ids.numel() * input_ids.element_size() > max_input_bytes:
        raise ValueError("input byte budget exceeded")
    ids = input_ids.detach().contiguous().clone()
    sequence = ids.shape[1]
    width, heads, vocab = model.config.n_embd, model.config.n_head, model.config.vocab_size
    layers = model.transformer.h
    if (type(width) is not int or type(heads) is not int or type(vocab) is not int
            or min(width, heads, vocab) <= 0 or width % heads
            or type(model.config.n_positions) is not int
            or sequence > model.config.n_positions or sequence > model.transformer.wpe.num_embeddings
            or not isinstance(layers, torch.nn.ModuleList) or len(layers) != model.config.n_layer):
        raise ValueError("unsupported GPT-2 dimensions")
    if point.layer >= len(layers) or point.token >= sequence or point.head >= heads:
        raise ValueError("selected layer, token or head is out of range")
    if bool((ids < 0).any() or (ids >= vocab).any()):
        raise ValueError("input token ID is outside the vocabulary")
    block = layers[point.layer]
    if type(block) is not GPT2Block or getattr(block.forward, "__func__", None) is not GPT2Block.forward:
        raise ValueError("an unmodified selected GPT-2 block is required")
    attention = block.attn
    projection = attention.c_proj
    if (type(attention) is not GPT2Attention
            or getattr(attention.forward, "__func__", None) is not GPT2Attention.forward
            or attention.is_cross_attention or attention.pruned_heads
            or attention.config._attn_implementation != "eager"
            or attention.embed_dim != width or attention.num_heads != heads
            or attention.head_dim != width // heads or attention.split_size != width
            or type(projection) is not Conv1D
            or getattr(projection.forward, "__func__", None) is not Conv1D.forward
            or tuple(projection.weight.shape) != (width, width)
            or tuple(projection.bias.shape) != (width,)):
        raise ValueError("selected attention projection is not the standard unpruned GPT-2 layout")
    full_bytes = sequence * width * 4
    slice_bytes = (width // heads) * 4
    if full_bytes > max_tensor_bytes or 3 * full_bytes > max_capture_bytes:
        raise ValueError("projection tensor or witness byte budget exceeded")
    if replacement is not None and slice_bytes > max_source_bytes:
        raise ValueError("replacement byte budget exceeded")
    if sequence * vocab * 4 > max_logits_bytes:
        raise ValueError("logits byte budget exceeded")
    path = f"transformer.h.{point.layer}.attn.c_proj"
    input_tuple = tuple(int(value) for value in ids[0].tolist())
    mask = torch.ones_like(ids)
    dimensions = (len(layers), width, heads, vocab, model.config.n_positions)
    layout = _attention_layout(attention, projection)
    witness = _LayoutCheckedWitness(projection, attention=attention, layout=layout,
        sequence_length=sequence, width=width, heads=heads,
        mode="observe" if replacement is None else "patch", head=None if replacement is None else point.head,
        position=None if replacement is None else point.token, replacement=replacement)
    lease = ModelLease(model)
    registration_started = False
    try:
        lease.acquire()
        lease.verify_tree()
        aliases = [name for name, module in lease.registered_paths if module is projection]
        if aliases != [path] or model.get_submodule(path) is not projection:
            raise ValueError("selected projection must have one stable module path")
        assert_no_forward_hooks(model)
        registration_started = True
        with torch.inference_mode(), witness:
            output = model(input_ids=ids, attention_mask=mask, use_cache=False,
                           output_attentions=False, output_hidden_states=False, return_dict=True)
        lease.verify_tree()
        assert_no_forward_hooks(model)
        if (tuple(int(value) for value in ids[0].tolist()) != input_tuple
                or mask.dtype != torch.int64 or tuple(mask.shape) != (1, sequence)
                or not bool(torch.all(mask == 1))
                or (len(model.transformer.h), model.config.n_embd, model.config.n_head,
                    model.config.vocab_size, model.config.n_positions) != dimensions
                or _attention_layout(attention, projection) != layout):
            raise RuntimeError("input IDs, mask or model layout changed during the forward")
        logits = output.logits
        if (type(logits) is not torch.Tensor or logits.device.type != "cpu" or logits.dtype != torch.float32
                or tuple(logits.shape) != (1, sequence, vocab) or not bool(torch.isfinite(logits).all())):
            raise RuntimeError("model did not return finite CPU float32 logits")
        before, after, projected = witness.before, witness.after, witness.output
        if any(type(tensor) is not torch.Tensor or tuple(tensor.shape) != (1, sequence, width)
               for tensor in (before, after, projected)):
            raise RuntimeError("incomplete projection witnesses")
        head_width = width // heads
        metadata = GPT2HeadSliceMetadata(run_id, path, point.layer, 0, point.token, 0, point.head,
            heads, head_width, point.head * head_width, (point.head + 1) * head_width,
            (1, sequence, width), input_tuple, (1,) * sequence,
            "observe" if replacement is None else "patch")
        return GPT2HeadSliceResult(metadata, before, after, projected, logits.detach(), 3 * full_bytes)
    finally:
        primary = sys.exc_info()[1]
        try:
            uncertain = False
            if registration_started:
                try:
                    assert_no_forward_hooks(model)
                except BaseException:
                    uncertain = True
                # Cancellation can interrupt the hook machinery between Python
                # frames even when the visible registries appear empty.
                uncertain = uncertain or (primary is not None and not isinstance(primary, Exception))
            if uncertain:
                lease._quarantine()
                if primary is None:
                    raise RuntimeError("hook cleanup is uncertain; discard the model or worker")
                _note_cleanup(primary, "Head-slice cleanup is uncertain; discard the model or worker.")
            else:
                lease.release()
        except BaseException as cleanup:
            if primary is None:
                raise
            _note_cleanup(primary, f"Head-slice cleanup also failed: {cleanup!r}; discard the model or worker.")
