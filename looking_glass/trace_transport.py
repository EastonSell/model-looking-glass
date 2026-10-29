"""Trusted, single-sample trace custody prototype; not a network endpoint.

The owner must reserve before capture, route by trusted row ownership, and
ensure that collaborators do not retain aliases beyond their declared leases.
The current broker/IPC service intentionally does not expose this adapter.
"""
from concurrent.futures import Future
from dataclasses import dataclass
import inspect
import math
import sys
import threading
import time
import uuid

import torch

from .capture_dispatcher import CapturedSlice
from .host_budget import HostBudgetExceeded, InvalidHostHandle

_METADATA_ENVELOPE = 65536


class DeliveryError(RuntimeError):
    """A delivery is invalid, cancelled, failed, or not in the required state."""


class DeliveryConstructionError(DeliveryError):
    """Construction cleanup is uncertain; guard identifies the exact request."""

    def __init__(self, guard):
        super().__init__("constructor cleanup requires supervision")
        self.guard = guard


@dataclass(frozen=True, slots=True)
class DeliveryIdentity:
    broker_id: str
    job_id: int
    team_id: str
    run_id: str
    request_id: str
    sample_id: str
    batch_position: int
    delivery_id: str


@dataclass(frozen=True, slots=True)
class DeliveryManifest:
    identity: DeliveryIdentity
    event: object
    payload_bytes: int
    byte_order: str


def _text(value):
    return type(value) is str and 0 < len(value) <= 128 and bool(value.strip())


def _int(value):
    return type(value) is int and 0 <= value < 2**63


def _bounded_tuple(value, *, maximum=8, parts=False):
    if type(value) is not tuple or len(value) > maximum:
        return False
    return all(_int(item) or (parts and _text(item)) for item in value)


def _discard_traceback(error):
    # Callers may retain the new, sanitized DeliveryError. Do not chain a
    # callback/store traceback that could keep payloads alive after release.
    try:
        error.__traceback__ = None
        error.__cause__ = None
        error.__context__ = None
    except BaseException:
        pass


def _encode_payload(payload):
    try:
        return payload.numpy().tobytes(order="C")
    finally:
        payload = None


def _recover_constructor(budget, guard, request_id, metadata_bytes, owner):
    """Release only payload-free metadata owned by this exact generation."""
    try:
        generation = guard.generation
    except InvalidHostHandle:
        # The guard did not register, or its exact generation already closed.
        # A reused request ID is never touched by this guard.
        guard.cancel_and_close_empty()
        return
    # Stop any new ordinary reservations on this exact request before
    # inspecting and disposing its constructor-only metadata claim.
    guard.adopt().cancel()
    owned = []
    for claim in budget.active_claims():
        try:
            info = claim.info()
        except InvalidHostHandle:
            # A different request's claim may have been released concurrently.
            continue
        if info.request_id == request_id and info.request_generation == generation:
            if (info.stage != "metadata" or info.owners != (owner,)
                    or info.cleanup_transfers != () or info.active_operations != 0
                    or info.nbytes != metadata_bytes or info.items != 1):
                raise DeliveryError("unexpected constructor claim requires supervision")
            owned.append(claim)
    if len(owned) > 1:
        raise DeliveryError("multiple constructor claims require supervision")
    for claim in owned:
        claim.release()
    guard.cancel_and_close_empty()


def _validate_event(event, identity, max_bytes, source_bytes, metadata_limit):
    if type(event) is not CapturedSlice:
        raise ValueError("expected a CapturedSlice")
    if (event.run_id, event.job_id, event.request_id, event.sample_id,
            event.batch_position) != (identity.run_id,
                                      f"lg-{identity.broker_id}-{identity.job_id}",
                                      identity.request_id, identity.sample_id,
                                      identity.batch_position):
        raise ValueError("capture does not belong to the authorized sample")
    for value in (event.run_id, event.job_id, event.request_id, event.sample_id,
                  event.module_path, event.module_kind, event.source_dtype,
                  event.source_device, event.saved_dtype, event.saved_device):
        if not _text(value):
            raise ValueError("capture text exceeds the bounded schema")
    for value in (event.batch_position, event.forward_index, event.sequence,
                  event.invocation):
        if not _int(value):
            raise ValueError("capture index is invalid")
    for value in (event.layer_index, event.generation_step):
        if value is not None and not _int(value):
            raise ValueError("optional capture index is invalid")
    if (not _bounded_tuple(event.source_shape) or len(event.source_shape) < 3
            or any(size == 0 for size in event.source_shape)
            or not _bounded_tuple(event.saved_shape)
            or not _bounded_tuple(event.token_positions, maximum=128)
            or not event.token_positions
            or len(set(event.token_positions)) != len(event.token_positions)
            or (event.output_path is not None
                and not _bounded_tuple(event.output_path, parts=True))):
        raise ValueError("capture coordinates exceed the bounded schema")
    if event.saved_shape != (len(event.token_positions), *event.source_shape[2:]):
        raise ValueError("saved shape does not match selected tokens")
    if event.source_dtype != event.saved_dtype:
        raise ValueError("direct captures cannot change dtype")
    tensor = event.value
    if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu"
            or tensor.layout != torch.strided or tensor.is_nested or tensor.is_quantized
            or tensor.dtype not in (torch.float16, torch.float32, torch.float64)
            or tensor.requires_grad or not tensor.is_contiguous()
            or tensor.storage_offset() != 0):
        raise ValueError("payload must be a contiguous, detached CPU float tensor")
    nbytes = tensor.numel() * tensor.element_size()
    if (nbytes <= 0 or nbytes > max_bytes or nbytes > source_bytes
            or tensor.untyped_storage().nbytes() != nbytes):
        raise ValueError("payload exceeds its source reservation")
    if sys.byteorder != "little":
        raise ValueError("only little-endian payload encoding is supported")
    metadata = event.to_trace_event()
    # Bounds on every input field limit this allocation before the exact
    # encoded-size check. The temporary text is never returned or retained.
    encoded = metadata.to_json()
    if len(encoded) > metadata_limit:
        raise ValueError("trace metadata exceeds its reservation")
    return metadata, nbytes


class TraceDelivery:
    """One trusted owner's one-event delivery; all work is process-local.

    External producers and sinks are trusted to honor the ownership contract.
    A caller must perform model/batch postchecks before calling publish().
    """

    def __init__(self, budget, store, *, broker_id, job_id, team_id, run_id,
                 request_id, sample_id, batch_position, max_payload_bytes,
                 metadata_reserve_bytes=_METADATA_ENVELOPE):
        # An enclosing adoption slot may retire a failed construction only
        # when this is true. Set false before any ledger registration; a
        # guarded cleanup sets it true again only after it has completed.
        self._construction_clean = True
        if (type(broker_id) is not str or len(broker_id) != 32
                or any(char not in "0123456789abcdef" for char in broker_id)
                or not _int(job_id) or job_id == 0
                or any(not _text(item) for item in
                       (team_id, run_id, request_id, sample_id))
                or not _int(batch_position)
                or not _int(max_payload_bytes) or max_payload_bytes == 0
                or not _int(metadata_reserve_bytes)
                or metadata_reserve_bytes < _METADATA_ENVELOPE):
            raise ValueError("invalid bounded delivery configuration")
        delivery_id = uuid.uuid4().hex
        self._identity = DeliveryIdentity(broker_id, job_id, team_id, run_id,
                                          request_id, sample_id, batch_position,
                                          delivery_id)
        self._request_id = "delivery-" + delivery_id
        construction_owner = "constructor-" + uuid.uuid4().hex
        self._max_payload_bytes = max_payload_bytes
        self._metadata_limit = metadata_reserve_bytes
        self._condition = threading.Condition(threading.RLock())
        self._cancel_signal = threading.Event()
        self._cancel_signal_lock = threading.Lock()
        self._state = "new"
        self._cancelled = False
        self._tombstoning = False
        self._busy = False
        self._cleaning = False
        self._finalizing = False
        self._quarantined = False
        self._source_disposed = False
        self._source_claim = None
        self._destination_claim = None
        self._serialization_claim = None
        self._operation = None
        self._active_token = None
        self._entry = None
        self._payload = None
        self._pending = None
        self._metadata = None
        self._nbytes = None
        self._metadata_claim = None
        # The side-effect-free guard exists before registration. The ledger
        # retains it with the exact generation if adoption is interrupted.
        guard = budget.request_guard(self._request_id)
        self._construction_clean = False
        try:
            guard.open()
            self._request = guard.adopt()
            self._generation = guard.generation
            self._metadata_claim = self._request.reserve(
                construction_owner, "metadata", metadata_reserve_bytes)
            self._store_request = store.request(self._request_id)
        except BaseException:
            try:
                _recover_constructor(budget, guard, self._request_id,
                                     metadata_reserve_bytes, construction_owner)
            except BaseException:
                # The live guard remains discoverable for trusted recovery.
                raise DeliveryConstructionError(guard) from None
            self._construction_clean = True
            raise

    @property
    def identity(self):
        return self._identity

    @property
    def state(self):
        with self._condition:
            return "quarantined" if self._quarantined else self._state

    def _check_live(self, expected):
        if (self._is_cancelled() or self._quarantined or self._state != expected
                or self._active_token is not None):
            raise DeliveryError("delivery is not in the required live state")

    def _is_cancelled(self):
        return self._cancelled or self._cancel_signal.is_set()

    def _tombstone_locked(self, state):
        if self._cancelled:
            return
        self._cancel_signal.set()
        self._cancelled = True
        if not self._quarantined:
            self._state = state
        self._tombstoning = True
        try:
            threading.Thread(target=self._tombstone, name="trace-cancel",
                             daemon=True).start()
        except BaseException:
            self._tombstoning = False
            self._quarantined = True
            self._state = "quarantined"

    def _tombstone(self):
        failed = False
        try:
            self._request.cancel()
        except BaseException:
            failed = True
        with self._condition:
            self._tombstoning = False
            if failed:
                self._quarantined = True
                self._state = "quarantined"
            else:
                self._schedule_cleanup_locked()
            self._condition.notify_all()

    def _finish_invocation(self, token):
        with self._condition:
            if self._active_token is token and not self._quarantined:
                self._active_token = None
                self._release_source_locked()
                if self._cancelled or self._state == "releasing":
                    self._schedule_cleanup_locked()
                self._maybe_finalize_locked()
                self._condition.notify_all()

    def _interrupted_invocation(self, token):
        with self._condition:
            if self._active_token is token:
                self._quarantined = True
                self._state = "quarantined"
                self._condition.notify_all()

    def reserve_source(self, nbytes):
        """Reserve the producer's capture envelope before the capture exists."""
        if not _int(nbytes) or nbytes == 0 or nbytes > self._max_payload_bytes:
            raise ValueError("source reservation is outside the payload limit")
        token = object()
        try:
            with self._condition:
                self._check_live("new")
                self._active_token = token
                try:
                    self._source_claim = self._request.reserve("producer", "source", nbytes)
                except HostBudgetExceeded:
                    self._active_token = None
                    raise
                self._state = "reserved"
                if self._is_cancelled() or self._quarantined:
                    raise DeliveryError("source reservation was cancelled")
        except (DeliveryError, HostBudgetExceeded):
            raise
        except BaseException as exc:
            _discard_traceback(exc)
            self._interrupted_invocation(token)
            raise DeliveryError("source reservation ownership is uncertain") from None
        finally:
            self._finish_invocation(token)

    def _release_source_locked(self):
        if (self._source_disposed and not self._busy and self._active_token is None
                and self._source_claim is not None):
            try:
                self._source_claim.release()
                self._source_claim = None
            except BaseException:
                self._quarantined = True
                self._state = "quarantined"

    def confirm_source_disposed(self):
        """Producer declares all of its source and alias references disposed."""
        with self._condition:
            if self._source_disposed:
                return False
            if self._source_claim is None:
                raise DeliveryError("there is no source reservation")
            if self._state in ("new", "reserved") and not self._is_cancelled():
                raise DeliveryError("capture has not been staged or cancelled")
            self._source_disposed = True
            if not self._quarantined:
                self._release_source_locked()
                self._maybe_finalize_locked()
            self._condition.notify_all()
            return True

    def _schedule_cleanup_locked(self):
        """Start one bounded-per-delivery disposal without holding this lock."""
        if (self._busy or self._cleaning or self._quarantined or self._tombstoning
                or self._active_token is not None):
            return
        if (self._entry is None and self._payload is None
                and self._serialization_claim is None
                and self._destination_claim is None):
            self._release_source_locked()
            if self._state == "releasing" and not self._cancelled:
                self._state = "released"
            self._maybe_finalize_locked()
            self._condition.notify_all()
            return
        self._cleaning = True
        try:
            threading.Thread(target=self._cleanup, name="trace-disposal",
                             daemon=True).start()
        except BaseException:
            self._cleaning = False
            self._quarantined = True
            self._state = "quarantined"

    def _cleanup(self):
        # Store operations and tensor destructors may block or reenter. Move
        # references under the lock, dispose outside it, and retain claims.
        with self._condition:
            entry, payload = self._entry, self._payload
            self._entry = self._payload = None
        failed = False
        try:
            if entry is not None:
                entry.drop()
            entry = None
            payload = None
        except BaseException as exc:
            _discard_traceback(exc)
            failed = True
        with self._condition:
            try:
                if failed:
                    self._pending = (entry, payload)
                    raise RuntimeError("disposal did not complete")
                if self._serialization_claim is not None:
                    self._serialization_claim.release()
                    self._serialization_claim = None
                if self._destination_claim is not None:
                    self._destination_claim.release()
                    self._destination_claim = None
                self._release_source_locked()
                if self._state == "releasing" and not self._cancelled:
                    self._state = "released"
            except BaseException:
                # Unknown disposal/ledger state is not evidence of safe release.
                self._quarantined = True
                self._state = "quarantined"
            finally:
                self._cleaning = False
                self._maybe_finalize_locked()
                self._condition.notify_all()

    def _maybe_finalize_locked(self):
        if (self._quarantined or self._busy or self._cleaning or self._finalizing
                or self._active_token is not None
                or self._tombstoning
                or not (self._cancelled or self._state == "released")
                or self._source_claim is not None
                or self._destination_claim is not None
                or self._serialization_claim is not None):
            return
        self._metadata = None
        self._finalizing = True
        try:
            threading.Thread(target=self._finalize, name="trace-finalize",
                             daemon=True).start()
        except BaseException:
            self._finalizing = False
            self._quarantined = True
            self._state = "quarantined"

    def _finalize(self):
        # Ledger locks may be contended. drain() must only wait for this
        # worker, never perform a potentially blocked release itself.
        released = closed = False
        try:
            self._metadata_claim.release()
            released = True
            self._request.close()
            closed = True
        except BaseException:
            pass
        with self._condition:
            if released:
                self._metadata_claim = None
            self._finalizing = False
            if closed:
                self._state = "closed"
            else:
                self._quarantined = True
                self._state = "quarantined"
            self._condition.notify_all()

    def stage(self, event):
        """Clone an owned event into the store after both claims are reserved."""
        token = object()
        try:
            return self._stage_impl(event, token)
        except DeliveryError:
            raise
        except BaseException as exc:
            _discard_traceback(exc)
            self._interrupted_invocation(token)
            raise DeliveryError("trace staging ownership is uncertain") from None
        finally:
            event = None
            self._finish_invocation(token)

    def _stage_impl(self, event, token):
        failure = None
        metadata = None
        nbytes = None
        operation = None
        entry = None
        with self._condition:
            # Invalid or overlapping calls cannot cancel or finish the
            # operation already owned by a different staging invocation.
            if (self._is_cancelled() or self._quarantined or self._state != "reserved"
                    or self._active_token is not None):
                event = None
                raise DeliveryError("delivery is not in the required live state")
            self._active_token = token
            try:
                info = self._source_claim.info()
                if (info.request_id != self._request_id
                        or info.request_generation != self._generation):
                    raise ValueError("source claim belongs to another generation")
                metadata, nbytes = _validate_event(
                    event, self._identity, self._max_payload_bytes, info.nbytes,
                    self._metadata_limit)
                self._destination_claim = self._request.reserve(
                    "delivery", "store-copy", nbytes)
                self._operation = self._request.begin_operation(
                    "store-copy", (self._source_claim, self._destination_claim))
                operation = self._operation
                self._busy = True
                self._state = "staging"
            except BaseException as exc:
                failure = type(exc).__name__
                _discard_traceback(exc)
                # Source disposal can be acknowledged as soon as we release
                # the lifecycle lock, so remove this frame's reference first.
                event = None
                self._tombstone_locked("failed")
        if operation is not None:
            try:
                # Another request can hold the shared store lock. Admission is
                # pinned, but cancellation never waits for that lock.
                entry = self._store_request.reserve("event", nbytes)
                with self._condition:
                    self._entry = entry
                    cancelled = self._is_cancelled()
                if not cancelled:
                    entry.commit(event.value)
            except BaseException as exc:
                failure = type(exc).__name__
                _discard_traceback(exc)
        # This frame must not hold a producer value when source is released.
        event = None
        with self._condition:
            if operation is not None:
                try:
                    if self._operation is not operation:
                        raise RuntimeError("staging operation ownership changed")
                    live = operation.finish()
                    self._operation = None
                    self._busy = False
                    if not live:
                        self._cancelled = True
                except BaseException:
                    self._quarantined = True
                    self._state = "quarantined"
                    failure = "custody uncertain"
                if not self._quarantined:
                    if failure is not None or self._is_cancelled():
                        if not self._cancelled:
                            self._tombstone_locked("failed" if failure else "cancelled")
                        else:
                            self._schedule_cleanup_locked()
                    else:
                        self._metadata = metadata
                        self._nbytes = nbytes
                        self._state = "staged"
                        self._release_source_locked()
            self._condition.notify_all()
        if failure is not None or self._is_cancelled() or self._quarantined:
            raise DeliveryError("trace staging failed or was cancelled") from None

    def publish(self):
        """Publish metadata only after trusted worker/batch postchecks."""
        token = object()
        try:
            with self._condition:
                self._check_live("staged")
                self._active_token = token
                manifest = DeliveryManifest(self._identity, self._metadata, self._nbytes, "little")
                if self._is_cancelled():
                    raise DeliveryError("delivery was cancelled before publication")
                self._state = "ready"
                return manifest
        except DeliveryError:
            raise
        except BaseException as exc:
            _discard_traceback(exc)
            self._interrupted_invocation(token)
            raise DeliveryError("manifest publication is uncertain") from None
        finally:
            self._finish_invocation(token)

    def send(self, sink):
        """Synchronously send a complete bounded payload to a trusted sink."""
        if not callable(sink):
            raise ValueError("sink must be callable")
        token = object()
        try:
            return self._send_impl(sink, token)
        except (DeliveryError, HostBudgetExceeded):
            raise
        except BaseException as exc:
            _discard_traceback(exc)
            self._interrupted_invocation(token)
            raise DeliveryError("trace send ownership is uncertain") from None
        finally:
            sink = None
            self._finish_invocation(token)

    def _send_impl(self, sink, token):
        with self._condition:
            self._check_live("ready")
            self._active_token = token
            try:
                self._serialization_claim = self._request.reserve(
                    "delivery", "serialization", self._nbytes)
                self._operation = self._request.begin_operation(
                    "serialize-send", (self._destination_claim, self._serialization_claim))
            except HostBudgetExceeded:
                if self._serialization_claim is not None:
                    self._serialization_claim.release()
                    self._serialization_claim = None
                self._active_token = None
                raise
            self._busy = True
            self._state = "sending"
            entry = self._entry

        data = result = payload = None
        failure = None
        deferred = False
        try:
            # The delivery coordinator keeps the same destination claim
            # across take(): there is no ledger ownership gap. The store
            # handoff may block, so it must not hold the lifecycle lock.
            payload = entry.take()
        except BaseException as exc:
            _discard_traceback(exc)
            with self._condition:
                self._quarantined = True
                self._state = "quarantined"
                self._condition.notify_all()
            raise DeliveryError("store handoff ownership is uncertain") from None
        with self._condition:
            self._entry = None
            self._payload = payload
            payload = None
            if self._is_cancelled():
                failure = "cancelled"
        if failure is None:
            try:
                data = _encode_payload(self._payload)
                if len(data) != self._nbytes:
                    raise ValueError("encoded payload length changed")
                with self._condition:
                    if self._is_cancelled():
                        failure = "cancelled"
                    else:
                        # Cancellation after this point may interrupt a
                        # partial sink operation but cannot validate it.
                        self._state = "delivering"
                if failure is None:
                    result = sink(data)
                    deferred = (isinstance(result, Future) or inspect.isawaitable(result)
                                or inspect.isgenerator(result) or inspect.isasyncgen(result))
                    if deferred:
                        failure = "deferred sink"
                    elif result is not None:
                        raise ValueError("sink must return None")
            except BaseException as exc:
                failure = type(exc).__name__
                _discard_traceback(exc)
        sink = None
        with self._condition:
            if deferred:
                # A returned future/generator may still own or use the bytes.
                # Keep the operation and all associated objects quarantined.
                self._pending = (data, result)
                self._quarantined = True
                self._state = "quarantined"
            else:
                data = result = None
                try:
                    live = self._operation.finish()
                    self._operation = None
                    self._busy = False
                    if not live:
                        self._cancelled = True
                    if failure is not None or self._is_cancelled():
                        if not self._cancelled:
                            self._tombstone_locked("cancelled" if failure == "cancelled" else "failed")
                        else:
                            self._schedule_cleanup_locked()
                    else:
                        self._serialization_claim.release()
                        self._serialization_claim = None
                        self._state = "sent"
                        self._release_source_locked()
                except BaseException:
                    self._quarantined = True
                    self._state = "quarantined"
                    failure = "custody uncertain"
            self._condition.notify_all()
        if failure is not None or self._is_cancelled() or self._quarantined:
            raise DeliveryError("trace send failed or was cancelled") from None

    def ack_received(self, identity):
        with self._condition:
            if not self._matches_identity(identity):
                raise DeliveryError("acknowledgement identity does not match")
            if self._state == "received" and not self._is_cancelled():
                return False
            self._check_live("sent")
            self._state = "received"
            return True

    def ack_release(self, identity):
        with self._condition:
            if not self._matches_identity(identity):
                raise DeliveryError("acknowledgement identity does not match")
            if self._state in ("releasing", "released") and not self._is_cancelled():
                if self._state == "releasing":
                    self._schedule_cleanup_locked()
                return False
            self._check_live("received")
            self._state = "releasing"
            self._schedule_cleanup_locked()
            self._condition.notify_all()
            return True

    def _matches_identity(self, identity):
        if type(identity) is not DeliveryIdentity:
            return False
        for name in DeliveryIdentity.__dataclass_fields__:
            expected = getattr(self._identity, name)
            supplied = getattr(identity, name)
            if type(supplied) is not type(expected) or supplied != expected:
                return False
        return True

    def cancel(self):
        """Signal cancellation without waiting for the lifecycle or ledger."""
        with self._cancel_signal_lock:
            if self._cancel_signal.is_set() or self._finalizing or self._state == "closed":
                return False
            self._cancel_signal.set()
            try:
                threading.Thread(target=self._dispatch_cancel, name="trace-cancel-request",
                                 daemon=True).start()
            except BaseException:
                self._quarantined = True
                self._state = "quarantined"
        return True

    def _dispatch_cancel(self):
        with self._condition:
            if self._state != "closed" and not self._finalizing:
                self._tombstone_locked("cancelled")
            self._condition.notify_all()

    def drain(self, timeout):
        """Wait at most timeout seconds for confirmed local disposal."""
        if (type(timeout) not in (int, float) or timeout < 0 or timeout > 86400
                or not math.isfinite(timeout)):
            raise ValueError("timeout must be a nonnegative number")
        deadline = time.monotonic() + timeout
        if not self._condition.acquire(timeout=timeout):
            return False
        try:
            while True:
                if self._state == "closed":
                    return True
                if self._quarantined:
                    return False
                self._maybe_finalize_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
        finally:
            self._condition.release()

    def close(self, timeout):
        self.cancel()
        return self.drain(timeout)
