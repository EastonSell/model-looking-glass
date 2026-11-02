"""CPU-only fixed-bucket service returning last-active-token logits.

This composes the existing serial broker, worker and forward-only IPC. It is
not a GPU admission policy, batching service, tracing transport or generation.
"""
import re
import threading
from types import MappingProxyType

import torch

from .ipc import IPCInputRejected
from .resident_worker import ForwardRequest, ForwardResult, InvalidRequest, RequestCancelled, ResidentWorker, WorkerUnavailable


_BUCKETS = (16, 32)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z", re.ASCII)
_RESPONSE_CAP = 4 * 1024 * 1024
_FULL_OUTPUT_CAP = 8 * 1024 * 1024


def _validate_configuration(bucket_length, vocab_size, pad_token_id):
    if (type(bucket_length) is not int or bucket_length not in _BUCKETS
            or type(vocab_size) is not int or not 1 <= vocab_size <= 65536
            or type(pad_token_id) is not int or not 0 <= pad_token_id < vocab_size):
        raise ValueError("unsupported bucket, vocabulary or pad token")
    if bucket_length * vocab_size * 4 > _FULL_OUTPUT_CAP or vocab_size * 32 + 512 > _RESPONSE_CAP:
        raise ValueError("full worker output or last-token response exceeds its bound")


def make_short_input(token_ids, *, bucket_length, vocab_size, pad_token_id):
    """Copy a nonempty token list into one right-padded fixed bucket."""
    _validate_configuration(bucket_length, vocab_size, pad_token_id)
    if type(token_ids) is not list:
        raise ValueError("expected a bounded nonempty list of vocabulary IDs")
    values = token_ids[:bucket_length + 1]
    if (not 1 <= len(values) <= bucket_length
            or any(type(value) is not int or not 0 <= value < vocab_size for value in values)):
        raise ValueError("expected a bounded nonempty list of vocabulary IDs")
    count = len(values)
    return {"input_ids": [values + [pad_token_id] * (bucket_length - count)],
            "attention_mask": [[1] * count + [0] * (bucket_length - count)]}


def _snapshot_wire(inputs, bucket_length):
    # Bound both copies before semantic validation; a caller must still avoid
    # concurrent mutation because separate row copies are not an atomic view.
    if (type(inputs) is not dict or len(inputs) != 2
            or any(type(key) is not str for key in inputs)
            or set(inputs) != {"input_ids", "attention_mask"}):
        raise ValueError("expected exactly input IDs and attention mask")
    try:
        ids, masks = inputs["input_ids"], inputs["attention_mask"]
        if type(ids) is not list or len(ids) != 1 or type(masks) is not list or len(masks) != 1:
            raise ValueError("invalid fixed-bucket input matrices")
        id_row, mask_row = ids[0], masks[0]
        if type(id_row) is not list or type(mask_row) is not list:
            raise ValueError("invalid fixed-bucket input matrices")
        return {"input_ids": [id_row[:bucket_length + 1]],
                "attention_mask": [mask_row[:bucket_length + 1]]}
    except (KeyError, IndexError):
        raise ValueError("input matrices changed while being copied") from None


def _wire_rows(inputs, bucket_length, vocab_size, pad_token_id):
    if (type(inputs) is not dict or len(inputs) != 2
            or any(type(key) is not str for key in inputs)
            or set(inputs) != {"input_ids", "attention_mask"}):
        raise ValueError("expected exactly input IDs and attention mask")
    ids, masks = inputs["input_ids"], inputs["attention_mask"]
    if (type(ids) is not list or len(ids) != 1 or type(ids[0]) is not list
            or type(masks) is not list or len(masks) != 1 or type(masks[0]) is not list
            or len(ids[0]) != bucket_length or len(masks[0]) != bucket_length
            or any(type(x) is not int or not 0 <= x < vocab_size for x in ids[0])
            or any(type(x) is not int or x not in (0, 1) for x in masks[0])):
        raise ValueError("invalid fixed-bucket input matrices")
    active = sum(masks[0])
    if (not active or masks[0] != [1] * active + [0] * (bucket_length - active)
            or any(x != pad_token_id for x in ids[0][active:])):
        raise ValueError("mask must be a nonempty right-padded prefix with canonical pad IDs")
    return ids[0], masks[0], active


class ShortRequestFactory:
    """Prevalidate plain wire objects before calling the trusted tensor factory."""

    def __init__(self, tensor_factory, *, bucket_length, vocab_size, pad_token_id):
        if not callable(tensor_factory):
            raise TypeError("a trusted tensor factory is required")
        _validate_configuration(bucket_length, vocab_size, pad_token_id)
        self._tensor_factory = tensor_factory
        self._bucket, self._vocab, self._pad = bucket_length, vocab_size, pad_token_id

    def __call__(self, request_id, inputs, seed):
        if (type(request_id) is not str or _IDENTIFIER.fullmatch(request_id) is None
                or type(seed) is not int or not 0 <= seed < 2**63):
            raise IPCInputRejected("invalid short-input request")
        try:
            canonical = _snapshot_wire(inputs, self._bucket)
            _wire_rows(canonical, self._bucket, self._vocab, self._pad)
        except ValueError:
            raise IPCInputRejected("invalid short-input request") from None
        try:
            return self._tensor_factory(request_id, canonical, seed)
        except IPCInputRejected:
            raise RuntimeError("tensor factory failed after pre-validation") from None


class ShortLastTokenCPUWorker:
    """Project a CPU resident worker's complete output to one owned logit row."""

    cancellation_exception = RequestCancelled

    def __init__(self, worker, *, bucket_length, vocab_size, pad_token_id):
        _validate_configuration(bucket_length, vocab_size, pad_token_id)
        if (type(worker) is not ResidentWorker or worker.device.type != "cpu"
                or worker.max_batch_size != 1 or worker.max_sequence_length != bucket_length
                or worker.max_output_bytes < bucket_length * vocab_size * 4
                or worker._owner_completion is not None):
            raise ValueError("an unbound CPU resident worker with matching limits is required")
        self._worker = worker
        self._bucket, self._vocab, self._pad = bucket_length, vocab_size, pad_token_id
        self._failed = threading.Event()
        self._closed = False

    @property
    def poisoned(self):
        return self._closed or self._failed.is_set() or self._worker.poisoned

    @property
    def failed(self):
        return self._failed.is_set() or self._worker.poisoned

    @property
    def quarantined(self):
        return self._worker.quarantined

    def _active_position(self, request):
        if (type(request) is not ForwardRequest or type(request.inputs) not in (dict, MappingProxyType)
                or set(request.inputs) != {"input_ids", "attention_mask"}):
            raise InvalidRequest("expected IDs and attention mask")
        ids, mask = request.inputs["input_ids"], request.inputs["attention_mask"]
        if (any(type(value) is not torch.Tensor or value.device.type != "cpu"
                or value.dtype != torch.int64 or tuple(value.shape) != (1, self._bucket)
                for value in (ids, mask))):
            raise InvalidRequest("invalid fixed-bucket tensors")
        try:
            _, _, active = _wire_rows({"input_ids": ids.tolist(), "attention_mask": mask.tolist()},
                                      self._bucket, self._vocab, self._pad)
        except ValueError:
            raise InvalidRequest("invalid right-padded tensors") from None
        return active - 1

    def snapshot_request(self, request):
        if self.poisoned:
            raise WorkerUnavailable("short CPU worker is unavailable")
        try:
            snapshot = self._worker.snapshot_request(request)
            self._active_position(snapshot)
            return snapshot
        except InvalidRequest:
            raise
        except BaseException:
            self._failed.set()
            raise

    def execute(self, request, cancel_event=None):
        if self.poisoned:
            raise WorkerUnavailable("short CPU worker is unavailable")
        try:
            # Own the mask used for selecting the output, independent of caller mutation.
            snapshot = self.snapshot_request(request)
            position = self._active_position(snapshot)
            result = self._worker.execute(snapshot, cancel_event)
            full = result.logits
            if (type(result) is not ForwardResult or result.request_id != snapshot.request_id
                    or type(full) is not torch.Tensor or full.device.type != "cpu"
                    or full.dtype != torch.float32 or tuple(full.shape) != (1, self._bucket, self._vocab)):
                raise RuntimeError("resident output does not match the short-input profile")
            with torch.inference_mode(False), torch.no_grad():
                selected = full[:, position:position + 1, :].detach().clone(memory_format=torch.contiguous_format)
            if (selected.is_inference() or not selected.is_contiguous() or selected.storage_offset() != 0
                    or selected.untyped_storage().nbytes() != self._vocab * 4
                    or not bool(torch.isfinite(selected).all())):
                raise RuntimeError("selected logits are not finite, independently owned CPU float32")
            if cancel_event is not None and cancel_event.is_set():
                raise RequestCancelled("request cancelled after output selection")
            return ForwardResult(snapshot.request_id, selected)
        except (InvalidRequest, RequestCancelled):
            raise
        except BaseException:
            self._failed.set()
            raise

    def close(self):
        self._closed = True
        self._worker.close()


def serve_short_cpu_model(model, *, model_id, team_ids, bucket_length, stop_event, announce,
                          shutdown_timeout=10):
    """Serve up to two local teams serially; callers own the model process."""
    from .broker import Broker
    from .ipc import IPCServer
    from .service import make_gpt2_request_factory, serve_owner

    config = getattr(model, "config", None)
    vocab, pad = getattr(config, "vocab_size", None), getattr(config, "eos_token_id", None)
    _validate_configuration(bucket_length, vocab, pad)
    if (type(getattr(config, "n_positions", None)) is not int or config.n_positions < bucket_length
            or type(team_ids) not in (tuple, list) or not 1 <= len(team_ids) <= 2):
        raise ValueError("model positions or team count exceed this CPU service's contract")
    factory = ShortRequestFactory(make_gpt2_request_factory(
        vocab_size=vocab, max_positions=config.n_positions, max_batch=1, max_tokens=bucket_length),
        bucket_length=bucket_length, vocab_size=vocab, pad_token_id=pad)
    resident = ResidentWorker(model_id, model, device="cpu", max_batch_size=1,
        max_sequence_length=bucket_length, max_input_bytes=bucket_length * 16,
        max_output_bytes=bucket_length * vocab * 4)
    worker = ShortLastTokenCPUWorker(resident, bucket_length=bucket_length, vocab_size=vocab, pad_token_id=pad)
    broker = Broker(worker, max_pending=2, max_pending_per_team=1)
    try:
        server = IPCServer(broker, team_ids=team_ids, request_factory=factory,
            max_connections=2, max_request_bytes=4096, max_batch=1, max_tokens=bucket_length,
            max_response_bytes=_RESPONSE_CAP)
    except BaseException:
        broker.close(timeout=shutdown_timeout)
        raise
    return serve_owner(server, broker, worker, stop_event, announce=announce,
                       verify_drain=lambda: True, shutdown_timeout=shutdown_timeout)
