"""Bounded authenticated loopback transport for a single resident broker.

This module never accepts executable code or Python object deserialization.
The broker and adapter callables are trusted, server-side objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
import secrets
import select
import socket
import struct
import sys
import sysconfig
import threading
import time
from types import MappingProxyType, TracebackType

from .ipc_custody import IPCCustodyCapacity, IPCCustodyKey

_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_BROKER_ID = re.compile(r"[0-9a-f]{32}\Z")
_INPUT_NAMES = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
_MAX_REQUEST = 1024 * 1024
_MAX_RESPONSE = 4 * 1024 * 1024


class IPCError(RuntimeError):
    """A local transport or protocol error."""


class IPCRejected(IPCError):
    """The server refused admission before the job was submitted."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class IPCInputRejected(Exception):
    """Trusted factory denied this input before request-owned side effects.

    Raising the exact type asserts that this invocation has created no
    request-owned tensors or other retained aliases, external side effects,
    or asynchronous/native work. Subclasses are not this assertion.
    """


class _ProtocolError(Exception):
    pass


class _ResultTooLarge(Exception):
    pass


class _HandlerFaultSlot:
    """Preallocated, server-rooted custody for one IPC thread's escaped fault."""

    __slots__ = ("seen", "capture_complete", "exception", "traceback", "observer_failure")

    def __init__(self):
        self.seen = False
        self.capture_complete = False
        self.exception = None
        self.traceback = None
        self.observer_failure = None


@dataclass(frozen=True)
class IPCHandlerFault:
    """Trusted-local fault snapshot; incomplete capture requires recovery."""

    capture_complete: bool
    exception: BaseException | None = field(repr=False)
    traceback: TracebackType | None = field(repr=False)
    observer_failure: BaseException | None = field(default=None, repr=False)


@dataclass(frozen=True)
class IPCAcceptorFault:
    """Trusted-local acceptor fault snapshot; incomplete capture requires recovery."""

    capture_complete: bool
    exception: BaseException | None = field(repr=False)
    traceback: TracebackType | None = field(repr=False)
    observer_failure: BaseException | None = field(default=None, repr=False)


def _current_handler_exception():
    return sys.exc_info()[1]


def _current_handler_traceback(error):
    _, active_error, active_traceback = sys.exc_info()
    if active_error is not error:
        raise IPCError("handler exception context changed during capture")
    return active_traceback


def _supported_handler_runtime():
    bootstrap = getattr(threading.Thread, "_bootstrap_inner", None)
    code = getattr(bootstrap, "__code__", None)
    gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return (sys.implementation.name == "cpython" and (3, 10) <= sys.version_info[:2] <= (3, 13)
            and not sysconfig.get_config_var("Py_GIL_DISABLED")
            and (gil_enabled is None or gil_enabled()) and code is not None
            and "_invoke_excepthook" in code.co_names)


def _remaining(deadline):
    return max(0.0, deadline - time.monotonic())


def _read_exact(sock, count, deadline):
    data = bytearray(count)
    view = memoryview(data)
    offset = 0
    while offset < count:
        timeout = _remaining(deadline)
        if timeout <= 0 or not select.select([sock], [], [], timeout)[0]:
            raise TimeoutError("IPC read deadline elapsed")
        try:
            received = sock.recv_into(view[offset:], count - offset)
        except BlockingIOError:
            continue
        if not received:
            raise EOFError("IPC peer disconnected")
        offset += received
    return data


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _ProtocolError("duplicate JSON field")
        result[key] = value
    return result


def _bad_constant(_):
    raise _ProtocolError("non-finite JSON value")


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise _ProtocolError("non-finite JSON value")
    return value


def _receive(sock, limit, deadline):
    length = struct.unpack("!I", _read_exact(sock, 4, deadline))[0]
    if length == 0 or length > limit:
        raise _ProtocolError("invalid frame length")
    try:
        value = json.loads(_read_exact(sock, length, deadline).decode("utf-8"),
                           object_pairs_hook=_unique_object, parse_constant=_bad_constant,
                           parse_float=_finite_float)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise _ProtocolError("invalid JSON") from exc
    if type(value) is not dict:
        raise _ProtocolError("expected JSON object")
    return value


def _send(sock, message, limit, deadline):
    try:
        data = json.dumps(message, allow_nan=False, ensure_ascii=True,
                          separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise _ProtocolError("invalid response") from exc
    if len(data) > limit:
        raise _ResultTooLarge
    view = memoryview(struct.pack("!I", len(data)) + data)
    while view:
        timeout = _remaining(deadline)
        if timeout <= 0 or not select.select([], [sock], [], timeout)[1]:
            raise TimeoutError("IPC write deadline elapsed")
        try:
            sent = sock.send(view)
        except BlockingIOError:
            continue
        if not sent:
            raise EOFError("IPC peer disconnected")
        view = view[sent:]


def _identifier(value):
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _validate_request(message, token, team_tokens, invalid_token, max_batch, max_tokens):
    required = {"version", "op", "token", "team_id", "request_id", "job_type", "inputs", "seed"}
    if set(message) != required or type(message["version"]) is not int or message["version"] != 1:
        raise _ProtocolError("invalid envelope")
    supplied = message["token"]
    team_id = message["team_id"]
    if not _identifier(team_id) or not _identifier(message["request_id"]):
        raise _ProtocolError("invalid identifier")
    expected = team_tokens.get(team_id, invalid_token) if team_tokens else token
    if (type(supplied) is not str or not supplied.isascii() or
            not secrets.compare_digest(supplied, expected) or (team_tokens and team_id not in team_tokens)):
        raise IPCRejected("unauthorized")
    if message["op"] != "submit" or message["job_type"] != "forward":
        raise _ProtocolError("unsupported operation")
    seed = message["seed"]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise _ProtocolError("invalid seed")
    inputs = message["inputs"]
    if type(inputs) is not dict or "input_ids" not in inputs or not set(inputs) <= _INPUT_NAMES:
        raise _ProtocolError("invalid inputs")
    expected_shape = None
    for name, matrix in inputs.items():
        if type(matrix) is not list or not 1 <= len(matrix) <= max_batch:
            raise _ProtocolError("invalid batch")
        width = None
        for row in matrix:
            if type(row) is not list or not 1 <= len(row) <= max_tokens:
                raise _ProtocolError("invalid token sequence")
            if width is None:
                width = len(row)
            if len(row) != width:
                raise _ProtocolError("ragged input")
            for value in row:
                if type(value) is not int or not 0 <= value < 2**31:
                    raise _ProtocolError("invalid token value")
                if name == "attention_mask" and value not in (0, 1):
                    raise _ProtocolError("invalid mask")
        shape = len(matrix), width
        if expected_shape is None:
            expected_shape = shape
        if shape != expected_shape:
            raise _ProtocolError("input shapes differ")
    return message


def _default_request_factory(request_id, inputs, seed):
    import torch
    from .resident_worker import ForwardRequest
    return ForwardRequest(request_id=request_id,
                          inputs={name: torch.tensor(value, dtype=torch.long, device="cpu")
                                  for name, value in inputs.items()}, seed=seed)


def _float32_word_as_float(word):
    """Decode a finite binary32 word using only integer operations first.

    Every nonzero finite binary32 value is normal in binary64. Constructing its
    binary64 bits avoids a denormal-sensitive binary32-to-float conversion.
    """
    word &= 0xffffffff
    exponent = (word >> 23) & 0xff
    fraction = word & 0x7fffff
    if exponent == 0xff:
        raise _ProtocolError("non-finite worker result")
    if exponent:
        exponent64 = exponent + 896
        fraction64 = fraction << 29
    elif fraction:
        leading = fraction.bit_length() - 1
        exponent64 = leading + 874
        fraction64 = (fraction - (1 << leading)) << (52 - leading)
    else:
        exponent64, fraction64 = 0, 0
    bits = ((word & 0x80000000) << 32) | (exponent64 << 52) | fraction64
    return struct.unpack("!d", struct.pack("!Q", bits))[0]


def _float32_numpy_values(integer_tensor, numpy):
    """Decode binary32 fields in bulk without binary32 floating arithmetic."""
    try:
        try:
            bits = integer_tensor.numpy().view(numpy.uint32)
        except RuntimeError:
            # PyTorch may have been built without NumPy integration even when
            # NumPy itself is installed; integer conversion remains exact.
            bits = numpy.asarray(integer_tensor.tolist(), dtype=numpy.int32).view(numpy.uint32)
        exponent = (bits >> 23) & 0xff
        if bool(numpy.any(exponent == 0xff)):
            raise _ProtocolError("non-finite worker result")
        fraction = bits & 0x7fffff
        significand = numpy.where(exponent != 0, fraction | (1 << 23), fraction).astype(numpy.float64)
        power = numpy.where(exponent != 0, exponent.astype(numpy.int16) - 150, -149)
        values = numpy.ldexp(significand, power)
        values = numpy.copysign(values, numpy.where((bits & 0x80000000) != 0, -1.0, 1.0))
        return values.tolist()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise _ProtocolError("unsupported worker result representation") from exc


def _encode_tensor_result(result, max_response_bytes):
    import torch
    logits = result.logits
    if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or
            logits.ndim != 3 or not logits.is_floating_point()):
        raise _ProtocolError("invalid worker result")
    # Bound the estimated wire size before building Python objects. This is not
    # a bound on peak host memory during conversion or JSON serialization.
    if logits.numel() > (max_response_bytes - 512) // 32:
        raise _ResultTooLarge
    formats = {
        torch.float16: (torch.int16, 0xffff, 0x7c00, 0x03ff),
        torch.bfloat16: (torch.int16, 0xffff, 0x7f80, 0x007f),
        torch.float32: (torch.int32, 0xffffffff, 0x7f800000, 0x007fffff),
        torch.float64: (torch.int64, 0xffffffffffffffff, 0x7ff0000000000000,
                        0x000fffffffffffff),
    }
    if logits.dtype not in formats or logits.layout != torch.strided or logits.is_neg() or logits.is_conj():
        raise _ProtocolError("unsupported worker result representation")
    integer_dtype, mask, exponent_mask, fraction_mask = formats[logits.dtype]
    try:
        # Reinterpret before flattening: a floating-point copy can itself flush
        # subnormals. NumPy is optional and receives only this integer view.
        integer_tensor = logits.detach().view(integer_dtype).reshape(-1)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise _ProtocolError("unsupported worker result representation") from exc
    if logits.dtype == torch.float32:
        try:
            import numpy
        except ImportError:
            numpy = None
        if numpy is not None:
            values = _float32_numpy_values(integer_tensor, numpy)
            return {"shape": list(logits.shape), "dtype": str(logits.dtype), "logits": values}
    # The standard-library fallback reuses the integer list for float32 values.
    words = integer_tensor.tolist()
    for index, signed_word in enumerate(words):
        word = signed_word & mask
        exponent, fraction = word & exponent_mask, word & fraction_mask
        if exponent == exponent_mask:
            raise _ProtocolError("non-finite worker result")
        if logits.dtype == torch.float32:
            words[index] = _float32_word_as_float(word)
        elif exponent == 0 and fraction:
            raise _ProtocolError("subnormal worker result dtype is not supported")
    if logits.dtype == torch.float32:
        values = words
    else:
        del words
        values = logits.reshape(-1).tolist()
    return {"shape": list(logits.shape), "dtype": str(logits.dtype),
            "logits": values}


def _completion_binding(broker, guardian, authority):
    """Resolve only trusted, exact owner backends; legacy backends stay optional."""
    target = getattr(broker, "_target", None)
    candidates = (getattr(broker, "_completion_authority", None),
                  getattr(getattr(broker, "_target_broker", None), "_completion_authority", None),
                  getattr(getattr(target, "_broker", None), "_completion_authority", None))
    if authority is None and not any(candidate is not None for candidate in candidates):
        return None

    # Lazy imports retain the lightweight legacy transport for fake brokers.
    from .broker import Broker
    from .ipc_fair_bridge import FairIPCBridge
    from .owner_completion import OwnerCompletionAuthority

    if type(authority) is not OwnerCompletionAuthority:
        raise ValueError("the bound completion authority must be supplied exactly")
    if type(broker) is Broker:
        actual, fair_target = broker, None
        if guardian is not None or broker._fair_handoff is not None:
            raise ValueError("direct authority binding requires a direct broker")
    elif type(broker) is FairIPCBridge:
        actual, fair_target = broker._target_broker, broker._target
        if (guardian is not broker or type(actual) is not Broker or
                fair_target._broker is not actual or actual._fair_handoff is not fair_target):
            raise ValueError("fair authority binding must use the pinned broker and guardian")
    else:
        raise ValueError("authority binding requires an exact supported backend")
    if (actual._completion_authority is not authority or authority._broker is not actual
            or authority.unsafe):
        raise ValueError("completion authority does not match the live broker")
    if not _supported_handler_runtime():
        raise ValueError("authority-bound handler supervision requires supported CPython")
    return authority, actual, fair_target


class IPCServer:
    """One request per authenticated local connection, backed by a trusted broker.

    The server never constructs the model or closes the caller-owned broker.
    Factory/encoder overrides are trusted local configuration, not wire inputs.
    """

    def __init__(self, broker, *, request_factory=None, result_encoder=None,
                 max_connections=8, max_request_bytes=_MAX_REQUEST,
                 max_response_bytes=_MAX_RESPONSE, max_batch=8, max_tokens=256,
                 io_timeout=3.0, job_timeout=60.0, team_ids=None,
                 custody_guardian=None, custody_epoch=None, completion_authority=None):
        # These checks are performed before server construction acquires any
        # resource. Ordinary invalid arguments are not custody faults.
        for name, value, low, high in (
            ("max_connections", max_connections, 1, 128),
            ("max_request_bytes", max_request_bytes, 256, _MAX_REQUEST),
            ("max_response_bytes", max_response_bytes, 512, _MAX_RESPONSE),
            ("max_batch", max_batch, 1, 256), ("max_tokens", max_tokens, 1, 8192),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        for name, value in (("io_timeout", io_timeout), ("job_timeout", job_timeout)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
        broker_id = getattr(broker, "broker_id", None)
        if type(broker_id) is not str or not _BROKER_ID.fullmatch(broker_id):
            raise ValueError("broker must expose its 32-character hexadecimal lifetime ID")
        if team_ids is not None:
            if (type(team_ids) not in (list, tuple) or not 1 <= len(team_ids) <= 128 or
                    not all(_identifier(team) for team in team_ids) or len(set(team_ids)) != len(team_ids)):
                raise ValueError("team_ids must contain 1 to 128 distinct valid team identifiers")
        if custody_guardian is None:
            if custody_epoch is not None:
                raise ValueError("custody epoch requires a guardian")
        else:
            if team_ids is None:
                raise ValueError("custody notifications require per-team authentication")
            if (type(custody_epoch) is not str or not _BROKER_ID.fullmatch(custody_epoch)
                    or any(not callable(getattr(custody_guardian, name, None))
                           for name in ("reserve", "authenticate", "handler_closed"))
                    or getattr(broker, "custody_guardian", None) is not custody_guardian
                    or not callable(getattr(broker, "submit_registered", None))):
                raise ValueError("invalid trusted custody configuration")
        binding = _completion_binding(broker, custody_guardian, completion_authority)
        try:
            self._initialize(broker, broker_id, request_factory, result_encoder, max_connections,
                             max_request_bytes, max_response_bytes, max_batch, max_tokens,
                             io_timeout, job_timeout, team_ids, custody_guardian, custody_epoch, binding)
        except BaseException:
            if binding is not None:
                try:
                    binding[0].mark_unsafe()
                finally:
                    # The exact authority's signal is a sticky CPython
                    # reference assignment. A one-shot interruption at the
                    # method entry must not leave this owner apparently safe.
                    binding[0]._unsafe = True
            raise

    def _initialize(self, broker, broker_id, request_factory, result_encoder, max_connections,
                    max_request_bytes, max_response_bytes, max_batch, max_tokens,
                    io_timeout, job_timeout, team_ids, custody_guardian, custody_epoch, binding):
        self.broker = broker
        self._broker_id = broker_id
        self._completion_binding = binding
        self.request_factory = request_factory or _default_request_factory
        self.result_encoder = result_encoder
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.max_batch, self.max_tokens = max_batch, max_tokens
        self.io_timeout, self.job_timeout = io_timeout, job_timeout
        if team_ids is not None:
            generated = {}
            for team in team_ids:
                candidate = secrets.token_hex(32)
                while candidate in generated.values():
                    candidate = secrets.token_hex(32)
                generated[team] = candidate
            self._token = None
            self._team_tokens = MappingProxyType(generated)
        else:
            self._token = secrets.token_hex(32)
            self._team_tokens = MappingProxyType({})
        self._invalid_token = secrets.token_hex(32)
        self._custody_guardian = custody_guardian
        self._custody_broker = broker if custody_guardian is not None else None
        self._custody_epoch = custody_epoch
        self._custody_generation = 0
        self._custody_quarantine = threading.Event()
        self._max_connections = max_connections
        self._slots = threading.BoundedSemaphore(max_connections)
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._admission = threading.Condition()
        self._admissions_active = 0
        self._clients = set()
        self._handlers = set()
        # Completed handlers remain rooted until another thread has observed
        # their actual termination. A handler cannot prove its own exit.
        self._retiring_handlers = set()
        # Bound-mode slots are rooted before start. Escaped exceptions remain
        # here after the corresponding thread exits and consume capacity.
        self._handler_fault_slots = {}
        # There is exactly one acceptor per server. Root its slot before its
        # thread can start, independently of the bounded handler inventory.
        self._accept_fault_slot = _HandlerFaultSlot() if binding is not None else None
        self._stop = threading.Event()
        self._accepting_stopped = threading.Event()
        self._listener = None
        self._accept_thread = None
        self._accept_orphan = None
        self._accept_uncertain = False
        self._accept_failure = None
        self._startup_uncertain = False
        self._startup_failure = None
        self.address = None

    @property
    def broker_id(self):
        return self._broker_id

    @property
    def token(self):
        return self._token

    @property
    def team_tokens(self):
        return self._team_tokens

    @property
    def custody_epoch(self):
        return self._custody_epoch

    @property
    def custody_quarantined(self):
        """Report an uncertain custody transition without waiting for locks."""
        return self._custody_quarantine.is_set()

    @property
    def retained_handler_faults(self):
        """Return trusted-local fault snapshots, including incomplete captures.

        This is a nonblocking snapshot, not evidence of disposal. The server
        retains these roots until its owning process is replaced.
        """
        if self._completion_binding is None:
            raise IPCError("handler fault observation is not configured")
        if not self._lock.acquire(blocking=False):
            raise IPCError("handler fault inventory is temporarily unavailable")
        try:
            return tuple(IPCHandlerFault(slot.capture_complete, slot.exception, slot.traceback,
                                         slot.observer_failure)
                         for slot in self._handler_fault_slots.values() if slot.seen)
        finally:
            self._lock.release()

    @property
    def retained_acceptor_fault(self):
        """Return a trusted-local fault snapshot without taking the server lock.

        A snapshot read while the thread is alive can be incomplete. The
        server retains this single fault root until its process is replaced.
        """
        if self._completion_binding is None:
            raise IPCError("acceptor fault observation is not configured")
        slot = self._accept_fault_slot
        if not slot.seen:
            return None
        return IPCAcceptorFault(slot.capture_complete, slot.exception, slot.traceback,
                                slot.observer_failure)

    def _install_thread_fault_observer(self, thread, slot, *, acceptor=False):
        """Use the supported CPython per-thread hook without a global hook."""
        original = getattr(thread, "_invoke_excepthook", None)
        if not callable(original):
            raise IPCError("unsupported CPython thread exception machinery")
        authority = self._completion_binding[0]

        def observe(fault_thread):
            # No server lock: a failing thread may have stranded it. The slot
            # is already rooted by the server before the thread starts.
            try:
                try:
                    # The bound authority's sticky flag orders success decisions.
                    # Latch it before capture, event signaling or application code:
                    # any of those may pause while a worker is finishing.
                    try:
                        authority._unsafe = True
                    finally:
                        authority._unsafe = True
                    self._mark_custody_fault()
                finally:
                    try:
                        self._custody_quarantine.set()
                    finally:
                        # An interrupted signal leaves an explicit unknown slot.
                        try:
                            slot.seen = True
                        finally:
                            slot.seen = True
                            if acceptor:
                                self._accept_uncertain = True
                                self._accepting_stopped.set()
                error = _current_handler_exception()
                if not isinstance(error, BaseException):
                    raise IPCError("handler exception capture is unavailable")
                slot.exception = error
                captured_traceback = _current_handler_traceback(error)
                if type(captured_traceback) is not TracebackType:
                    raise IPCError("handler traceback capture is unavailable")
                slot.traceback = captured_traceback
                slot.capture_complete = True
            except BaseException as observer_failure:
                # Python normally chains the escaped target exception to this
                # failure. Keep that graph rooted even if capture never began.
                try:
                    slot.observer_failure = observer_failure
                finally:
                    slot.seen = True
                raise
            original(fault_thread)

        thread._invoke_excepthook = observe

    def _reap_retiring_handlers_locked(self):
        """Forget normal exits; preserve fault slots and exception roots."""
        for prior in tuple(self._retiring_handlers):
            if not prior.is_alive():
                slot = self._handler_fault_slots.get(prior)
                if slot is not None and not slot.seen:
                    del self._handler_fault_slots[prior]
                self._retiring_handlers.discard(prior)

    def _mark_custody_fault(self):
        # The owner decision and this signal use a shared authority. Mark it
        # before publishing the separate observational IPC quarantine flag.
        binding = self._completion_binding
        if binding is not None:
            try:
                binding[0].mark_unsafe()
            finally:
                binding[0]._unsafe = True
                self._custody_quarantine.set()
        elif self._custody_guardian is not None:
            self._custody_quarantine.set()

    def _check_completion_binding(self):
        binding = self._completion_binding
        if binding is None:
            return
        authority, actual, fair_target = binding
        if authority.unsafe:
            raise IPCRejected("unavailable")
        try:
            endpoint = self._custody_broker if fair_target is not None else actual
            if fair_target is not None:
                from .ipc_fair_bridge import _ServerConstruction
                construction = endpoint._server_construction
            valid = (_supported_handler_runtime() and self.broker is endpoint and actual._completion_authority is authority
                     and authority._broker is actual
                     and (actual._fair_handoff is None if fair_target is None else
                          (self._custody_guardian is endpoint and endpoint._target is fair_target
                           and endpoint._target_broker is actual and fair_target._broker is actual
                           and actual._fair_handoff is fair_target and endpoint._server is self
                           and type(construction) is _ServerConstruction
                           and construction.server is self)))
        except BaseException:
            self._mark_custody_fault()
            raise
        if not valid:
            self._mark_custody_fault()
            raise IPCRejected("unavailable")

    def _validate_completion_handle(self, handle, key, team_id, request_id):
        """Check the returned receipt before using it or sending acceptance."""
        authority, actual, fair_target = self._completion_binding
        if fair_target is None:
            from .broker import RequestHandle
            valid = (type(handle) is RequestHandle and handle._broker is actual
                     and type(handle.job_id) is int and handle.job_id > 0
                     and handle.team_id == team_id and handle.request_id == request_id)
        else:
            from .ipc_fair_bridge import _Proxy, _Record
            endpoint = self._custody_broker
            with endpoint._condition:
                record = endpoint._records.get(key)
                valid = (type(handle) is _Proxy and type(record) is _Record
                         and handle._owner is endpoint and handle._record is record
                         and record.proxy is handle and record.key is key
                         and record.team == team_id and record.request_id == request_id
                         and type(handle.job_id) is int and handle.job_id > 0)
        if not valid:
            raise IPCError("broker returned an invalid request handle")

    def _request_handle_cancel(self, handle):
        accepted = handle.cancel()
        if self._completion_binding is not None and type(accepted) is not bool:
            raise IPCError("broker returned an invalid cancellation result")
        return accepted

    def start(self):
        self._check_completion_binding()
        with self._lifecycle:
            if self._listener is not None or self._accepting_stopped.is_set():
                raise RuntimeError("server already started or closed")
            listener = None
            try:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                listener.setblocking(False)
                if self._accepting_stopped.is_set():
                    raise IPCError("server closed during startup")
                self._listener = listener
                self.address = listener.getsockname()
                self._accept_thread = threading.Thread(target=self._accept, daemon=True, name="looking-glass-ipc")
                if self._accept_fault_slot is not None:
                    self._install_thread_fault_observer(self._accept_thread, self._accept_fault_slot, acceptor=True)
                self._accept_thread.start()
            except BaseException as exc:
                monitored = self._completion_binding is not None or self._custody_guardian is not None
                try:
                    if monitored:
                        # The failed startup frame may still own the listener.
                        self._startup_failure = exc
                finally:
                    try:
                        self._mark_custody_fault()
                    finally:
                        if monitored:
                            self._startup_uncertain = True
                            self._stop.set()
                            self._accepting_stopped.set()
                if listener is not None:
                    listener.close()
                raise
        return self

    def _accept(self):
        try:
            self._accept_loop()
        except BaseException as exc:
            try:
                # Signal before retaining the exception: a blocked capture
                # must not let an owner finish a later success decision.
                binding = self._completion_binding
                if binding is not None:
                    try:
                        binding[0]._unsafe = True
                    finally:
                        binding[0]._unsafe = True
                self._mark_custody_fault()
            finally:
                try:
                    # The frame may own a socket between accept() and handoff.
                    self._accept_failure = exc
                finally:
                    try:
                        self._mark_custody_fault()
                    finally:
                        if self._completion_binding is not None or self._custody_guardian is not None:
                            self._accept_uncertain = True
                            self._accepting_stopped.set()
            raise

    def _accept_loop(self):
        while not self._accepting_stopped.is_set():
            try:
                if not select.select([self._listener], [], [], 0.1)[0]:
                    continue
                client, peer = self._listener.accept()
            except (OSError, ValueError):
                if self._accepting_stopped.is_set():
                    return
                if self._completion_binding is not None or self._custody_guardian is not None:
                    self._mark_custody_fault()
                    self._accept_uncertain = True
                    self._accepting_stopped.set()
                    return
                continue
            monitored = self._completion_binding is not None or self._custody_guardian is not None
            try:
                if monitored:
                    # Only one acceptor exists. Keep the just-accepted socket
                    # rooted until a handler owns it or rejection completes.
                    self._accept_orphan = client
                client.setblocking(False)
                if peer[0] != "127.0.0.1" or not self._slots.acquire(blocking=False):
                    try:
                        _send(client, {"version": 1, "status": "rejected", "error": "busy"},
                              self.max_response_bytes, time.monotonic() + min(self.io_timeout, 0.2))
                    except (OSError, EOFError, TimeoutError, ValueError, _ProtocolError, _ResultTooLarge):
                        pass
                    finally:
                        client.close()
                    self._accept_orphan = None
                    continue
                thread = threading.Thread(target=self._run_handler, args=(client,), daemon=True,
                                          name="looking-glass-ipc-client")
                fault_slot = _HandlerFaultSlot() if self._completion_binding is not None else None
                if fault_slot is not None:
                    self._install_thread_fault_observer(thread, fault_slot)
                full = False
                with self._lock:
                    if self._accepting_stopped.is_set():
                        client.close()
                        self._slots.release()
                        self._accept_orphan = None
                        return
                    # A socket slot can have been released just before its
                    # handler exits. Keep the combined live thread count
                    # bounded even in that interval.
                    self._reap_retiring_handlers_locked()
                    occupied = (len(self._handler_fault_slots) if self._completion_binding is not None
                                else len(self._handlers) + len(self._retiring_handlers))
                    if occupied >= self._max_connections:
                        full = True
                    else:
                        if fault_slot is not None:
                            self._handler_fault_slots[thread] = fault_slot
                        self._clients.add(client)
                        self._handlers.add(thread)
                        thread.start()
                        self._accept_orphan = None
                if full:
                    try:
                        _send(client, {"version": 1, "status": "rejected", "error": "busy"},
                              self.max_response_bytes, time.monotonic() + min(self.io_timeout, 0.2))
                    except (OSError, EOFError, TimeoutError, ValueError, _ProtocolError, _ResultTooLarge):
                        pass
                    finally:
                        client.close()
                        self._slots.release()
                    self._accept_orphan = None
            except BaseException:
                self._mark_custody_fault()
                if monitored:
                    self._accept_uncertain = True
                    self._accepting_stopped.set()
                raise

    def _send(self, client, message):
        _send(client, message, self.max_response_bytes, time.monotonic() + self.io_timeout)

    def _run_handler(self, client):
        # Retain a source-time fault signal even if an unexpected exception
        # escapes the final part of the handler's own cleanup.
        try:
            self._serve(client)
        except BaseException:
            self._mark_custody_fault()
            raise

    def _serve(self, client):
        handle = None
        handle_valid = False
        terminal = False
        request_id = None
        custody_key = None
        custody_registered = False
        custody_safe = True
        try:
            self._check_completion_binding()
            if self._custody_guardian is not None:
                if self._custody_quarantine.is_set():
                    raise IPCRejected("unavailable")
                with self._lock:
                    if self._custody_generation >= 2**63 - 1:
                        raise IPCError("custody generation exhausted")
                    self._custody_generation += 1
                    custody_key = IPCCustodyKey(self._custody_epoch, self._custody_generation)
                # A failed or interrupted reserve can still have committed.
                # The owner can recover by its pre-existing guardian record.
                custody_safe = False
                try:
                    self._custody_guardian.reserve(custody_key)
                except IPCCustodyCapacity as exc:
                    if type(exc) is not IPCCustodyCapacity:
                        raise
                    # Trusted guardian attests that nothing was registered,
                    # claimed, or left in flight for this key.
                    custody_safe = True
                    raise IPCRejected("busy") from None
                custody_registered = True
                custody_safe = True
            self._send(client, {"version": 1, "status": "ready"})
            message = _receive(client, self.max_request_bytes, time.monotonic() + self.io_timeout)
            _validate_request(message, self.token, self.team_tokens, self._invalid_token,
                              self.max_batch, self.max_tokens)
            request_id = message["request_id"]
            if self._custody_guardian is not None:
                custody_safe = False
                self._custody_guardian.authenticate(custody_key, message["team_id"], request_id)
                custody_safe = True
            # A trusted factory can allocate or retain input aliases before it
            # raises.  The guardian cannot infer cleanup from an exception.
            custody_safe = False
            try:
                request = self.request_factory(request_id, message["inputs"], message["seed"])
            except IPCInputRejected as exc:
                if type(exc) is not IPCInputRejected:
                    raise
                # This is a trusted, exact no-side-effects assertion made at
                # the factory boundary, not something IPC can verify.
                custody_safe = True
                raise IPCRejected("invalid_request") from None
            custody_safe = True
            if self._accepting_stopped.is_set():
                return
            with self._admission:
                if self._accepting_stopped.is_set():
                    raise IPCRejected("unavailable")
                self._admissions_active += 1
            try:
                try:
                    self._check_completion_binding()
                    if self._custody_guardian is None:
                        if self._completion_binding is not None:
                            # Submission may have committed even if its return
                            # never reaches this handler. Only the exact
                            # trusted broker can attest a clean rejection.
                            from .broker import Broker
                            custody_safe = False
                            try:
                                handle = self.broker.submit(message["team_id"], request)
                            except Exception:
                                authority, actual, _ = self._completion_binding
                                if (getattr(self.broker.submit, "__func__", None) is Broker.submit
                                        and not authority.unsafe and not actual.quarantined):
                                    custody_safe = True
                                raise
                            self._validate_completion_handle(handle, None, message["team_id"], request_id)
                            custody_safe = True
                        else:
                            handle = self.broker.submit(message["team_id"], request)
                        handle_valid = True
                    else:
                        custody_safe = False
                        if self._custody_quarantine.is_set():
                            raise IPCRejected("unavailable")
                        handle = self._custody_broker.submit_registered(custody_key, message["team_id"], request)
                        if self._completion_binding is not None:
                            self._validate_completion_handle(handle, custody_key, message["team_id"], request_id)
                        custody_safe = True
                        handle_valid = True
                except Exception as exc:
                    code = {"QueueFull": "queue_full", "BrokerUnavailable": "unavailable"}.get(
                        type(exc).__name__, "admission_failed")
                    raise IPCRejected(code) from exc
                if self._accepting_stopped.is_set():
                    custody_safe = False
                    self._request_handle_cancel(handle)
                    custody_safe = True
                    raise IPCRejected("unavailable")
            finally:
                with self._admission:
                    self._admissions_active -= 1
                    self._admission.notify_all()
            self._send(client, {"version": 1, "status": "accepted", "request_id": request_id,
                                "team_id": message["team_id"], "broker_id": self.broker_id,
                                "job_id": handle.job_id})
            deadline = time.monotonic() + self.job_timeout
            cancellation_seen = False
            cancellation_status = None
            cancellation_count = 0
            while not self._stop.is_set():
                custody_safe = False
                try:
                    outcome = handle.wait(timeout=0)
                except TimeoutError:
                    outcome = None
                custody_safe = True
                if outcome is not None:
                    custody_safe = False
                    custody_safe = self._send_outcome(client, outcome, handle, request_id, message["team_id"])
                    terminal = True
                    return
                if _remaining(deadline) == 0:
                    custody_safe = False
                    self._request_handle_cancel(handle)
                    custody_safe = True
                    self._send(client, {"version": 1, "status": "timeout", "request_id": request_id,
                                        "team_id": message["team_id"], "broker_id": self.broker_id,
                                        "job_id": handle.job_id})
                    terminal = True  # Delivery ended; the broker may still be running.
                    return
                ready = select.select([client], [], [], min(0.05, _remaining(deadline)))[0]
                if ready:
                    command = _receive(client, self.max_request_bytes,
                                       min(deadline, time.monotonic() + self.io_timeout))
                    if (set(command) != {"version", "op", "request_id"} or
                            type(command["version"]) is not int or command["version"] != 1 or
                            command["op"] != "cancel" or command["request_id"] != request_id):
                        raise _ProtocolError("invalid cancellation")
                    cancellation_count += 1
                    if cancellation_count > 16:
                        raise _ProtocolError("too many cancellations")
                    if cancellation_seen:
                        self._send(client, {"version": 1, "status": cancellation_status,
                                            "request_id": request_id, "team_id": message["team_id"],
                                            "broker_id": self.broker_id, "job_id": handle.job_id})
                        continue
                    cancellation_seen = True
                    custody_safe = False
                    cancellation_accepted = self._request_handle_cancel(handle)
                    custody_safe = True
                    if cancellation_accepted:
                        cancellation_status = "cancel_requested"
                        self._send(client, {"version": 1, "status": "cancel_requested",
                                            "request_id": request_id, "team_id": message["team_id"],
                                            "broker_id": self.broker_id, "job_id": handle.job_id})
                        continue
                    custody_safe = False
                    try:
                        outcome = handle.wait(timeout=0)
                    except TimeoutError:
                        custody_safe = True
                        cancellation_status = "cancel_not_accepted"
                        self._send(client, {"version": 1, "status": "cancel_not_accepted",
                                            "request_id": request_id, "team_id": message["team_id"],
                                            "broker_id": self.broker_id, "job_id": handle.job_id})
                        continue
                    custody_safe = self._send_outcome(client, outcome, handle, request_id, message["team_id"])
                    terminal = True
                    return
        except IPCRejected as exc:
            if not custody_safe:
                self._mark_custody_fault()
            self._try_reject(client, exc.code, request_id)
        except MemoryError:
            custody_safe = False
            self._mark_custody_fault()
            if handle is None:
                self._try_reject(client, "internal_error", request_id)
        except (_ProtocolError, _ResultTooLarge, ValueError, TypeError):
            if not custody_safe:
                self._mark_custody_fault()
            if handle is None:
                self._try_reject(client, "invalid_request", request_id)
        except (OSError, EOFError, TimeoutError):
            if not custody_safe:
                self._mark_custody_fault()
            pass
        except Exception:
            if not custody_safe:
                self._mark_custody_fault()
            if handle is None:
                self._try_reject(client, "internal_error", request_id)
        finally:
            pending_exception = sys.exc_info()[0] is not None
            cancellation_exception = None
            if pending_exception:
                custody_safe = False
            if not custody_safe:
                self._mark_custody_fault()
            if handle is not None and handle_valid and not terminal:
                try:
                    prior_safe = custody_safe
                    custody_safe = False
                    self._request_handle_cancel(handle)
                    custody_safe = prior_safe
                except BaseException as exc:
                    custody_safe = False
                    self._mark_custody_fault()
                    if not isinstance(exc, Exception):
                        cancellation_exception = exc
            # Drop potentially large result/input references before a slow peer
            # drains the socket. The broker retains any running work it owns.
            handle = request = message = outcome = None
            socket_closed = False
            try:
                if terminal:
                    self._graceful_finish(client)
                client.close()
                socket_closed = True
            finally:
                if not custody_safe or (custody_registered and not socket_closed):
                    self._mark_custody_fault()
                if custody_registered and custody_safe and socket_closed:
                    try:
                        # This reports a local event only. The guardian owns
                        # claims and the handle independently until disposal.
                        self._custody_guardian.handler_closed(custody_key)
                    except BaseException:
                        # The guardian must keep uncertain records charged.
                        self._mark_custody_fault()
                try:
                    with self._lock:
                        try:
                            self._clients.discard(client)
                            # Keep the handler discoverable until connection
                            # accounting has actually completed.
                            self._slots.release()
                            # Root first, then remove from active handlers;
                            # only another thread can observe actual exit.
                            self._retiring_handlers.add(threading.current_thread())
                            self._handlers.discard(threading.current_thread())
                        except BaseException:
                            # A notice may already have been observed. Latch
                            # before releasing the lock or the handler exits.
                            self._mark_custody_fault()
                            raise
                except BaseException:
                    self._mark_custody_fault()
                    raise
            if cancellation_exception is not None and not pending_exception:
                raise cancellation_exception

    def _graceful_finish(self, client):
        """Deliver the response even if a cancel raced with completion.

        On Windows, closing a TCP socket with unread client bytes can reset the
        connection and discard the just-sent response. Half-close and drain a
        bounded amount of pending input until the peer closes or the deadline.
        """
        try:
            client.shutdown(socket.SHUT_WR)
            deadline = time.monotonic() + self.io_timeout
            budget = self.max_request_bytes
            while budget > 0 and not self._stop.is_set():
                timeout = _remaining(deadline)
                if timeout <= 0 or not select.select([client], [], [], timeout)[0]:
                    return
                try:
                    data = client.recv(min(4096, budget))
                except BlockingIOError:
                    continue
                if not data:
                    return
                budget -= len(data)
        except (OSError, ValueError):
            pass

    def _try_reject(self, client, code, request_id):
        try:
            self._send(client, {"version": 1, "status": "rejected", "error": code,
                                "request_id": request_id})
        except (OSError, EOFError, TimeoutError, ValueError, _ProtocolError, _ResultTooLarge):
            pass

    def _send_outcome(self, client, outcome, handle, request_id, team_id):
        if (getattr(outcome, "broker_id", None) != self.broker_id or
                outcome.request_id != request_id or outcome.team_id != team_id or
                outcome.job_id != handle.job_id):
            uncertain = self._completion_binding is not None
            if uncertain:
                # Wrong-job results are not disclosed, but their custody is
                # also unknown. Do not issue a clean local-close notice.
                self._mark_custody_fault()
            self._send(client, {"version": 1, "status": "failed", "error": "invalid_result",
                                "request_id": request_id, "team_id": team_id,
                                "broker_id": self.broker_id, "job_id": handle.job_id})
            return not uncertain
        state = getattr(outcome.state, "value", outcome.state)
        response = {"version": 1, "status": state, "request_id": outcome.request_id,
                    "team_id": outcome.team_id, "broker_id": self.broker_id, "job_id": outcome.job_id,
                    "submitted_ns": outcome.submitted_ns,
                    "ready_ns": getattr(outcome, "ready_ns", None),
                    "started_ns": outcome.started_ns, "finished_ns": outcome.finished_ns}
        try:
            if state == "succeeded":
                if self.result_encoder is None and getattr(outcome.result, "request_id", None) != request_id:
                    raise _ProtocolError("worker result identity mismatch")
                response["result"] = (self.result_encoder(outcome.result) if self.result_encoder else
                                      _encode_tensor_result(outcome.result, self.max_response_bytes))
            elif state == "failed":
                response["error"] = str(outcome.error_type)[:128]
            elif state != "cancelled":
                raise _ProtocolError("invalid broker state")
            self._send(client, response)
            return True
        except (MemoryError, _ResultTooLarge, _ProtocolError, ValueError, TypeError) as exc:
            self._mark_custody_fault()
            response.pop("result", None)
            response["status"] = "failed"
            if isinstance(exc, MemoryError):
                response["error"] = "serialization_out_of_memory"
            elif isinstance(exc, _ResultTooLarge):
                response["error"] = "result_too_large"
            else:
                response["error"] = "invalid_result"
            self._send(client, response)
            return False

    def stop_accepting(self, timeout=5.0):
        """Stop new submissions without interrupting already accepted jobs."""
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        try:
            return self._stop_accepting(timeout)
        except BaseException:
            self._mark_custody_fault()
            raise

    def _stop_accepting(self, timeout):
        deadline = time.monotonic() + timeout
        with self._admission:
            self._accepting_stopped.set()
        if not self._lifecycle.acquire(timeout=_remaining(deadline)):
            return False
        try:
            if self._listener is not None:
                try:
                    self._listener.close()
                except BaseException:
                    self._mark_custody_fault()
                    raise
        finally:
            self._lifecycle.release()
        with self._admission:
            if not self._admission.wait_for(lambda: self._admissions_active == 0,
                                            timeout=_remaining(deadline)):
                return False
        if self._accept_thread is not None and self._accept_thread is not threading.current_thread():
            self._accept_thread.join(timeout=_remaining(deadline))
        return (not self._startup_uncertain and not self._accept_uncertain and self._accept_orphan is None and
                (self._accept_thread is None or not self._accept_thread.is_alive()))

    def drain(self, timeout=5.0):
        """Wait for client handlers without cancelling accepted work."""
        if not self._accepting_stopped.is_set():
            raise IPCError("stop accepting before draining")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        try:
            return self._drain(timeout)
        except BaseException:
            self._mark_custody_fault()
            raise

    def _drain(self, timeout):
        deadline = time.monotonic() + timeout
        if not self.stop_accepting(timeout=_remaining(deadline)):
            return False
        if self._accept_thread is not None and self._accept_thread is not threading.current_thread():
            self._accept_thread.join(timeout=_remaining(deadline))
        with self._lock:
            handlers = list(self._handlers | self._retiring_handlers)
        for thread in handlers:
            if thread is not threading.current_thread():
                thread.join(timeout=_remaining(deadline))
        with self._lock:
            self._reap_retiring_handlers_locked()
            return (not self._handlers and not self._retiring_handlers and
                    (self._accept_thread is None or not self._accept_thread.is_alive()))

    def close(self, timeout=5.0):
        """Stop admission and cancel clients; return False while handlers remain.

        A trusted factory or broker submit may be blocked. In that case this
        method does not claim shutdown succeeded, and can be called again.
        """
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        try:
            return self._close(timeout)
        except BaseException:
            self._mark_custody_fault()
            raise

    def _close(self, timeout):
        deadline = time.monotonic() + timeout
        self._stop.set()
        self._accepting_stopped.set()
        if not self._lifecycle.acquire(timeout=_remaining(deadline)):
            return False
        try:
            if self._listener is not None:
                try:
                    self._listener.close()
                except BaseException:
                    self._mark_custody_fault()
                    raise
        finally:
            self._lifecycle.release()
        with self._lock:
            clients = list(self._clients)
            orphan = self._accept_orphan
        if orphan is not None and not any(client is orphan for client in clients):
            clients.append(orphan)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except BaseException:
                self._mark_custody_fault()
                raise
        if self._accept_thread is not None and self._accept_thread is not threading.current_thread():
            self._accept_thread.join(timeout=_remaining(deadline))
        if self._startup_uncertain or self._accept_uncertain or self._accept_orphan is not None:
            return False
        with self._lock:
            handlers = list(self._handlers | self._retiring_handlers)
        for thread in handlers:
            if thread is not threading.current_thread():
                thread.join(timeout=_remaining(deadline))
        with self._lock:
            self._reap_retiring_handlers_locked()
            return (not self._handlers and not self._retiring_handlers and
                    (self._accept_thread is None or not self._accept_thread.is_alive()))

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        stopped = self.close()
        if not stopped and exc_type is None:
            raise IPCError("server shutdown incomplete")
        if not stopped and exc is not None and hasattr(exc, "add_note"):
            exc.add_note("IPC server shutdown incomplete; retry close()")
        return False


@dataclass(frozen=True)
class CancelAckObservation:
    """A validated client acknowledgement receipt, not a broker decision time."""

    accepted: bool
    received_ns: int


class IPCClient:
    """Connect to a known loopback server using an out-of-band secret."""

    def __init__(self, address, token, *, io_timeout=3.0, max_response_bytes=_MAX_RESPONSE,
                 observe_cancel_ack=False):
        if (type(address) is not tuple or len(address) != 2 or address[0] != "127.0.0.1" or
                type(address[1]) is not int or not 0 < address[1] < 65536):
            raise ValueError("only an IPv4 loopback endpoint is allowed")
        if type(token) is not str or not token.isascii() or not token:
            raise ValueError("invalid token")
        if type(io_timeout) not in (int, float) or not math.isfinite(io_timeout) or io_timeout <= 0:
            raise ValueError("invalid io_timeout")
        if type(max_response_bytes) is not int or not 512 <= max_response_bytes <= _MAX_RESPONSE:
            raise ValueError("invalid max_response_bytes")
        if type(observe_cancel_ack) is not bool:
            raise ValueError("observe_cancel_ack must be a boolean")
        self.address, self.token = address, token
        self.io_timeout, self.max_response_bytes = io_timeout, max_response_bytes
        self._observe_cancel_ack = observe_cancel_ack

    def submit(self, *, team_id, request_id, inputs, seed=0):
        sock = socket.create_connection(self.address, timeout=self.io_timeout)
        sock.setblocking(False)
        try:
            ready = _receive(sock, self.max_response_bytes, time.monotonic() + self.io_timeout)
            if ready.get("status") == "rejected":
                raise IPCRejected(ready.get("error", "rejected"))
            if (set(ready) != {"version", "status"} or type(ready["version"]) is not int or
                    ready["version"] != 1 or ready["status"] != "ready"):
                raise IPCError("invalid readiness response")
            request = {"version": 1, "op": "submit", "token": self.token, "team_id": team_id,
                       "request_id": request_id, "job_type": "forward", "inputs": inputs, "seed": seed}
            _send(sock, request, _MAX_REQUEST, time.monotonic() + self.io_timeout)
            response = _receive(sock, self.max_response_bytes, time.monotonic() + self.io_timeout)
            if response.get("status") == "rejected":
                raise IPCRejected(response.get("error", "rejected"))
            if (type(response.get("version")) is not int or response.get("version") != 1 or
                    response.get("status") != "accepted" or
                    response.get("request_id") != request_id or response.get("team_id") != team_id or
                    type(response.get("job_id")) is not int or response["job_id"] < 1 or
                    type(response.get("broker_id")) is not str or
                    not _BROKER_ID.fullmatch(response["broker_id"])):
                raise IPCError("invalid admission response")
            return IPCHandle(sock, request_id, team_id, response["broker_id"], response["job_id"], self.io_timeout,
                             self.max_response_bytes, observe_cancel_ack=self._observe_cancel_ack)
        except BaseException:
            sock.close()
            raise


class IPCHandle:
    def __init__(self, sock, request_id, team_id, broker_id, job_id, io_timeout, max_response_bytes,
                 *, observe_cancel_ack=False):
        if type(observe_cancel_ack) is not bool:
            raise ValueError("observe_cancel_ack must be a boolean")
        self._sock, self.request_id, self.team_id = sock, request_id, team_id
        self.broker_id, self.job_id = broker_id, job_id
        self._io_timeout, self._max_response_bytes = io_timeout, max_response_bytes
        self._send_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._closed = False
        self._cancel_sent = False
        self._cancel_attempted = False
        self.cancel_acknowledged = False
        self._observe_cancel_ack = observe_cancel_ack
        self._cancel_ack_observation = None
        self._cancel_ack_observation_failed = False
        self._cancel_ack_capture_pending = False

    @property
    def cancel_ack_observation(self):
        """First validated acknowledgement receipt, or None if not observed."""
        return self._cancel_ack_observation

    @property
    def cancel_ack_observation_failed(self):
        """Whether receipt capture failed or more than one acknowledgement arrived."""
        return self._cancel_ack_observation_failed or self._cancel_ack_capture_pending

    def _capture_cancel_ack(self, accepted):
        try:
            if (self._cancel_ack_observation_failed or self._cancel_ack_observation is not None
                    or not self._cancel_attempted):
                self._cancel_ack_observation_failed = True
                return
            stamp = time.perf_counter_ns()
            if type(stamp) is not int or not 0 <= stamp < 2 ** 63:
                raise ValueError("invalid acknowledgement observation timestamp")
            self._cancel_ack_observation = CancelAckObservation(accepted, stamp)
        except Exception:
            self._cancel_ack_observation_failed = True

    def cancel(self):
        """Request cooperative cancellation; call wait() for its acknowledgement."""
        with self._send_lock:
            if self._closed:
                raise IPCError("handle closed")
            if self._cancel_sent:
                return
            if self._observe_cancel_ack:
                # The peer may reply while _send is still returning; record
                # the local attempt before a frame can be acknowledged.
                self._cancel_attempted = True
            _send(self._sock, {"version": 1, "op": "cancel", "request_id": self.request_id},
                  _MAX_REQUEST, time.monotonic() + self._io_timeout)
            self._cancel_sent = True

    def wait(self, timeout=60.0):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        with self._read_lock:
            if self._closed:
                raise IPCError("handle closed")
            try:
                deadline = time.monotonic() + timeout
                while True:
                    response = _receive(self._sock, self._max_response_bytes, deadline)
                    if (type(response.get("version")) is not int or response.get("version") != 1 or
                            type(response.get("job_id")) is not int or response.get("request_id") != self.request_id or
                            response.get("team_id") != self.team_id or response.get("broker_id") != self.broker_id or
                            response.get("job_id") != self.job_id or
                            response.get("status") not in {"succeeded", "failed", "cancelled", "timeout",
                                                           "cancel_requested", "cancel_not_accepted"}):
                        raise IPCError("invalid result response")
                    if response["status"] in {"cancel_requested", "cancel_not_accepted"}:
                        if self._observe_cancel_ack:
                            # Set before entering the helper so interrupted
                            # capture cannot be mistaken for complete evidence.
                            self._cancel_ack_capture_pending = True
                            try:
                                self._capture_cancel_ack(response["status"] == "cancel_requested")
                            except Exception:
                                self._cancel_ack_observation_failed = True
                            self._cancel_ack_capture_pending = False
                        self.cancel_acknowledged = response["status"] == "cancel_requested"
                        continue
                    if (self._observe_cancel_ack and response["status"] == "succeeded"
                            and self._cancel_ack_observation is not None
                            and self._cancel_ack_observation.accepted):
                        self._cancel_ack_observation_failed = True
                    return response
            finally:
                self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
