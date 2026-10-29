"""Bounded eager observations of standard Hugging Face GPT-2 outputs.

The module does not load checkpoints. Each session is exclusive to one model
and entering thread; hooks are not suitable for multi-request dispatch.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import threading
import sys

import torch

from .probes import gradient_readout
from .model_ownership import ModelLease
from .trace_schema import GenerationContext, RunIdentity, TraceEvent


def _note_cleanup(primary, message):
    """Preserve a secondary cleanup diagnostic on Python 3.10 and later."""
    add_note = getattr(primary, 'add_note', None)
    if callable(add_note):
        add_note(message)
    else:
        diagnostic = RuntimeError(message)
        diagnostic.__cause__ = primary.__cause__
        primary.__cause__ = diagnostic
        primary.__suppress_context__ = True


def _indices(name, values):
    if values is None:
        return None
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f"{name} must be a nonempty tuple or list, or None")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{name} must contain nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate indices")
    return tuple(values)


@dataclass(frozen=True)
class GPT2TraceEvent:
    """One selected observation; coordinates describe the source tensor."""
    run_id: str
    sequence: int
    forward_index: int
    module_path: str
    module_kind: str
    layer_index: int
    invocation: int
    output_path: tuple
    batch_positions: tuple
    token_positions: tuple
    shape: tuple
    dtype: str
    device: str
    capture_mode: str
    value: torch.Tensor | None

    @property
    def saved_shape(self):
        return None if self.value is None else tuple(self.value.shape)

    @property
    def saved_device(self):
        return None if self.value is None else str(self.value.device)

    @property
    def tensor_bytes(self):
        return 0 if self.value is None else self.value.numel() * self.value.element_size()

    def metadata(self):
        """Return the schema-1.0 wire shape, excluding tensor contents."""
        return self.to_schema().to_dict()

    def to_schema(self):
        """Convert metadata to the shared immutable trace schema."""
        return TraceEvent(
            identity=RunIdentity(self.run_id), sequence=self.sequence,
            forward_index=self.forward_index, module_path=self.module_path,
            invocation=self.invocation, output_path=self.output_path,
            shape=self.shape, dtype=self.dtype, device=self.device,
            capture_mode=self.capture_mode, module_kind=self.module_kind,
            layer_index=self.layer_index, batch_positions=self.batch_positions,
            token_positions=self.token_positions, batch_axis=0, token_axis=1,
            generation=GenerationContext(), saved_shape=self.saved_shape,
            saved_dtype=None if self.value is None else str(self.value.dtype),
            saved_device=self.saved_device)


class GPT2TraceSession:
    """Capture named GPT-2 boundary outputs with explicit finite budgets.

    Byte limits count cumulative successful saved tensor bytes, not transient
    allocations or metadata. Failed sessions discard all partial captures.
    """
    def __init__(self, model, *, run_id=None, targets=("block", "attn", "mlp"),
                 layers=None, samples=None, tokens=None, mode="direct",
                 max_events=10000, max_trace_bytes=16 * 1024 * 1024,
                 capture_device="cpu"):
        if mode not in ("direct", "gradient", "id"):
            raise ValueError("mode must be direct, gradient, or id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        if type(max_trace_bytes) is not int or max_trace_bytes < 0:
            raise ValueError("max_trace_bytes must be a nonnegative integer")
        if capture_device not in ("cpu", "source"):
            raise ValueError("capture_device must be cpu or source")
        if (not isinstance(targets, (list, tuple)) or not targets
                or any(t not in ("block", "attn", "mlp") for t in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError("targets must contain distinct block, attn, or mlp roles")
        samples, tokens, layers = (_indices("samples", samples),
                                  _indices("tokens", tokens),
                                  _indices("layers", layers))
        if getattr(getattr(model, "config", None), "model_type", None) != "gpt2":
            raise TypeError("model must be a standard GPT-2 model")
        # Import lazily: legacy synthetic experiments require torch only.
        from transformers import GPT2LMHeadModel, GPT2Model
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP
        if type(model) is GPT2LMHeadModel:
            backbone, prefix = model.transformer, "transformer.h"
        elif type(model) is GPT2Model:
            backbone, prefix = model, "h"
        else:
            raise TypeError("model must be an unmodified GPT2Model or GPT2LMHeadModel")
        if getattr(model.config, "add_cross_attention", False):
            raise ValueError("cross-attention GPT-2 variants are unsupported")
        if model.is_gradient_checkpointing:
            raise ValueError("gradient checkpointing is unsupported")
        if not isinstance(backbone.h, torch.nn.ModuleList) or not backbone.h:
            raise ValueError("GPT-2 model has no standard block list")
        if any(type(block) is not GPT2Block or type(block.attn) is not GPT2Attention
               or type(block.mlp) is not GPT2MLP for block in backbone.h):
            raise TypeError("GPT-2 block, attention and MLP modules must be standard classes")
        if layers is None:
            layers = tuple(range(len(backbone.h)))
        if any(layer >= len(backbone.h) for layer in layers):
            raise ValueError("layer index outside model")

        self.model = model
        self.run_id, self.mode = run_id, mode
        self.max_events, self.max_trace_bytes = max_events, max_trace_bytes
        self.capture_device = capture_device
        self.samples, self.tokens = samples, tokens
        self.events = []
        self.successful = False
        self._state_lock = threading.Lock()
        self._used = False
        self._entering = False
        self._failed = False
        self._handles = []
        self._untracked_handle = None
        self._registration_uncertain = None
        self._registration_inflight = False
        self._accounted_bytes = 0
        self._sequence = 0
        self._next_forward = 0
        self._completed_forwards = 0
        self._active = False
        self._normal_completed = False
        self._owner_thread = None
        self._frames = defaultdict(list)
        self._site_frames = {}
        self._invocations = Counter()
        self._shape = None
        self._lease = ModelLease(model)
        self._selected_modules = []
        for layer in layers:
            block = backbone.h[layer]
            for kind in targets:
                module = block if kind == "block" else getattr(block, kind)
                path = f"{prefix}.{layer}" + ("" if kind == "block" else f".{kind}")
                self._selected_modules.append((module, path, kind, layer))
        # Hooks are object based: any alias can otherwise create false path labels,
        # even when the alias is an unselected path in the same model.
        counts = Counter(id(module) for _, module in model.named_modules(remove_duplicate=False))
        if any(counts[id(module)] != 1 for module, _, _, _ in self._selected_modules):
            raise ValueError("aliased selected modules cannot be attributed to unique paths")

    @property
    def tensor_bytes(self):
        return sum(event.tensor_bytes for event in self.events)

    def _invalidate(self):
        self._failed = True
        self.events.clear()

    def __enter__(self):
        with self._state_lock:
            if self._used:
                raise RuntimeError("GPT2TraceSession is single-use")
            self._used = True
            self._owner_thread = threading.current_thread()
        try:
            self._lease.acquire()
            self._entering = True
            self._lease.verify_tree()
            counts = Counter(id(module) for _, module in self._lease.registered_paths)
            for module, path, _, _ in self._selected_modules:
                try:
                    current = self.model.get_submodule(path)
                except AttributeError as error:
                    raise RuntimeError("selected model path changed before session entry") from error
                if current is not module or counts[id(module)] != 1:
                    raise RuntimeError("selected model path changed or became ambiguous before entry")
            self._register(self.model, "pre", self._begin_forward,
                           prepend=True, with_kwargs=True)
            self._register(self.model, "forward", self._mark_forward_complete)
            self._register(self.model, "forward", self._end_forward, always_call=True)
            for module, path, kind, layer in self._selected_modules:
                self._register(module, "pre", self._make_site_begin(path), prepend=True)
                self._register(module, "forward", self._make_hook(path, kind, layer))
                self._register(module, "forward", self._make_site_end(path), always_call=True)
            self._entering = False
            return self
        except BaseException as primary:
            self._entering = False
            try:
                self.close()
            except BaseException as cleanup:
                _note_cleanup(primary, f"Trace hook cleanup also failed: {cleanup!r}")
            raise

    def _register(self, module, kind, callback, **kwargs):
        register = module.register_forward_pre_hook if kind == 'pre' else module.register_forward_hook
        try:
            # Until the returned handle is recorded, any interruption may have
            # happened after PyTorch installed a hook. Keep the lease if unsure.
            self._registration_inflight = True
            handle = register(callback, **kwargs)
            self._untracked_handle = handle
            if not isinstance(handle, torch.utils.hooks.RemovableHandle):
                self._registration_uncertain = True
                raise RuntimeError("hook registration returned an invalid handle; model ownership retained")
            self._handles.append(handle)
            self._untracked_handle = None
            self._registration_inflight = False
        except BaseException as primary:
            if isinstance(self._untracked_handle, torch.utils.hooks.RemovableHandle):
                try:
                    self._untracked_handle.remove()
                    self._untracked_handle = None
                    self._registration_inflight = False
                except BaseException as cleanup:
                    _note_cleanup(primary, f"Untracked hook removal also failed: {cleanup!r}")
            elif self._registration_inflight:
                self._registration_uncertain = True
            raise

    def _begin_forward(self, module, args, kwargs):
        thread = threading.current_thread()
        frames = self._frames[thread]
        try:
            if (frames and not self._active) or self._normal_completed or self._site_frames:
                raise RuntimeError("a previous forward did not finish")
            frames.append(False)
            self._lease.verify_tree()
            if thread != self._owner_thread:
                raise RuntimeError("model forward must run on the session's owning thread")
            if self._failed:
                raise RuntimeError("trace session is already failed")
            if self._active:
                raise RuntimeError("nested or concurrent model forwards are unsupported")
            if self.model.is_gradient_checkpointing:
                raise RuntimeError("gradient checkpointing is unsupported")
            if hasattr(torch, "compiler") and torch.compiler.is_compiling():
                raise RuntimeError("compiled execution is unsupported")
            if len(args) > 1:
                raise ValueError("only input_ids may be passed positionally")
            ids = kwargs.get("input_ids", args[0] if args else None)
            embeds = kwargs.get("inputs_embeds")
            if (ids is None) == (embeds is None):
                raise ValueError("supply exactly one of input_ids or inputs_embeds")
            source = ids if ids is not None else embeds
            expected_dims = 2 if ids is not None else 3
            if not isinstance(source, torch.Tensor) or source.ndim != expected_dims:
                raise ValueError("input_ids must be [batch, token] or inputs_embeds [batch, token, hidden]")
            if source.shape[0] == 0 or source.shape[1] == 0:
                raise ValueError("batch and token dimensions must be nonempty")
            for name, selection, size in (("samples", self.samples, source.shape[0]),
                                          ("tokens", self.tokens, source.shape[1])):
                if selection is not None and any(index >= size for index in selection):
                    raise IndexError(f"{name} index outside current forward")
            self._active = True
            self._normal_completed = False
            frames[-1] = True
            self._shape = tuple(source.shape[:2])
            self._forward_index = self._next_forward
            self._next_forward += 1
            self._invocations.clear()
        except BaseException:
            self._invalidate()
            raise

    def _mark_forward_complete(self, module, args, output):
        if threading.current_thread() is self._owner_thread and self._active:
            self._normal_completed = True
        return None

    def _end_forward(self, module, args, output):
        thread = threading.current_thread()
        frames = self._frames.get(thread)
        if not frames:
            self._invalidate()
            return None
        entered = frames.pop()
        if not frames:
            self._frames.pop(thread, None)
        if not entered:
            self._invalidate()
            if thread is self._owner_thread and not frames:
                self._active = False
                self._shape = None
                self._normal_completed = False
        if entered:
            self._active = False
            self._shape = None
            if output is None or not self._normal_completed or self._site_frames:
                self._invalidate()
            else:
                try:
                    self._lease.verify_tree()
                except BaseException:
                    self._invalidate()
                    raise
                self._completed_forwards += 1
            self._normal_completed = False
        return None

    def _make_site_begin(self, path):
        def begin(_module, _args):
            try:
                self._lease.verify_tree()
                thread = threading.current_thread()
                if thread is not self._owner_thread or not self._active or self._failed:
                    raise RuntimeError("target entered outside the owning forward")
                key = (thread, path)
                if key in self._site_frames:
                    raise RuntimeError("reentrant target invocations are unsupported")
                invocation = self._invocations[path]
                self._invocations[path] += 1
                self._site_frames[key] = [invocation, False]
            except BaseException:
                self._invalidate()
                raise
        return begin

    def _make_site_end(self, path):
        def end(_module, _args, _output):
            frame = self._site_frames.pop((threading.current_thread(), path), None)
            # A normal capture hook serves as a completion marker. Merely
            # seeing a non-None output here does not prove earlier hooks ran.
            if frame is None or not frame[1]:
                self._invalidate()
        return end

    def _make_hook(self, path, kind, layer):
        def capture(_module, _args, output):
            try:
                self._lease.verify_tree()
                if threading.current_thread() is not self._owner_thread or not self._active or self._failed:
                    raise RuntimeError("capture occurred outside the owning forward")
                frame = self._site_frames.get((threading.current_thread(), path))
                if frame is None or frame[1]:
                    raise RuntimeError("target capture has no matching module entry")
                if self._sequence >= self.max_events:
                    raise RuntimeError("trace event capacity exceeded")
                if isinstance(output, torch.Tensor):
                    tensor, output_path = output, ()
                elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                    tensor, output_path = output[0], (0,)
                else:
                    raise TypeError("unsupported GPT-2 module output")
                if tensor.ndim != 3 or tuple(tensor.shape[:2]) != self._shape:
                    raise ValueError("GPT-2 boundary output must have [batch, token, hidden] shape")
                if not tensor.is_floating_point():
                    raise TypeError("GPT-2 boundary output must be floating point")
                batch = self.samples if self.samples is not None else tuple(range(tensor.shape[0]))
                token = self.tokens if self.tokens is not None else tuple(range(tensor.shape[1]))
                need = len(batch) * len(token) * tensor.shape[2] * tensor.element_size()
                if self.mode != "id" and self._accounted_bytes + need > self.max_trace_bytes:
                    raise RuntimeError("trace tensor byte capacity exceeded")
                value = None
                if self.mode != "id":
                    b = torch.tensor(batch, dtype=torch.long, device=tensor.device)
                    t = torch.tensor(token, dtype=torch.long, device=tensor.device)
                    # Advanced indexing creates an owned copy, avoiding a
                    # second clone for the direct reporting channel.
                    selected = tensor.detach()[b[:, None], t[None, :], :]
                    if self.mode == "gradient":
                        value = gradient_readout(selected)
                    else:
                        if not bool(torch.isfinite(selected).all()):
                            raise ValueError("capture requires finite selected activations")
                        value = selected
                    if self.capture_device == "cpu" and value.device.type != "cpu":
                        value = value.to("cpu")
                invocation = frame[0]
                event = GPT2TraceEvent(self.run_id, self._sequence, self._forward_index,
                                       path, kind, layer, invocation, output_path,
                                       batch, token, tuple(tensor.shape), str(tensor.dtype),
                                       str(tensor.device), self.mode, value)
                self.events.append(event)
                self._sequence += 1
                if value is not None:
                    self._accounted_bytes += need
                frame[1] = True
            except BaseException:
                self._invalidate()
                raise
            return None
        return capture

    def close(self):
        if self._owner_thread is not None and threading.current_thread() is not self._owner_thread:
            raise RuntimeError("trace session must be closed by its owning thread")
        if self._entering or self._model_forward_on_stack():
            raise RuntimeError("cannot close a trace while registering or during a forward")
        # PyTorch does not run always_call hooks on BaseException. Once the
        # owning model call has unwound, stale frames indicate an incomplete
        # forward, not one that is still running.
        if self._active or self._frames or self._site_frames or self._normal_completed:
            self._invalidate()
            self._active = False
            self._normal_completed = False
            self._shape = None
            self._frames.clear()
            self._site_frames.clear()
        failed = []
        first_error = None
        for handle in self._handles:
            try:
                handle.remove()
            except BaseException as error:
                failed.append(handle)
                if first_error is None:
                    first_error = error
        self._handles[:] = failed
        if isinstance(self._untracked_handle, torch.utils.hooks.RemovableHandle):
            try:
                self._untracked_handle.remove()
                self._untracked_handle = None
                self._registration_inflight = False
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if self._registration_uncertain is not None or self._registration_inflight:
            if first_error is None:
                first_error = RuntimeError("hook registration left uncertain model state; ownership retained")
        if first_error is not None:
            self._invalidate()
            raise first_error
        try:
            self._lease.release()
        except BaseException as primary:
            self._invalidate()
            try:
                self._lease.release()
            except BaseException as cleanup:
                _note_cleanup(primary, f"Model lease release retry also failed: {cleanup!r}")
            raise

    def _model_forward_on_stack(self):
        frame = sys._getframe()
        target_code = torch.nn.Module._call_impl.__code__
        while frame is not None:
            if frame.f_code is target_code and frame.f_locals.get('self') is self.model:
                return True
            frame = frame.f_back
        return False

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            try:
                self._lease.verify_tree()
            except BaseException as primary:
                self._invalidate()
                try:
                    self.close()
                except BaseException as cleanup:
                    _note_cleanup(primary, f"Trace hook cleanup also failed: {cleanup!r}")
                raise
        try:
            self.close()
        except BaseException as cleanup:
            self._invalidate()
            if exc is None:
                raise
            _note_cleanup(exc, f"Trace hook cleanup also failed: {cleanup!r}")
        self.successful = exc_type is None and not self._failed and self._completed_forwards > 0
        if not self.successful:
            self.events.clear()
        return False
