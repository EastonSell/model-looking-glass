"""Bounded authenticated loopback transport for a single resident broker.

This module never accepts executable code or Python object deserialization.
The broker and adapter callables are trusted, server-side objects.
"""
from __future__ import annotations

import json
import math
import re
import secrets
import select
import socket
import struct
import threading
import time
from types import MappingProxyType

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


class _ProtocolError(Exception):
    pass


class _ResultTooLarge(Exception):
    pass


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


def _encode_tensor_result(result, max_response_bytes):
    import torch
    logits = result.logits
    if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or
            logits.ndim != 3 or not logits.is_floating_point()):
        raise _ProtocolError("invalid worker result")
    # Reserve 32 bytes per floating-point value before building Python objects.
    if logits.numel() > (max_response_bytes - 512) // 32:
        raise _ResultTooLarge
    if not bool(torch.isfinite(logits).all()):
        raise _ProtocolError("non-finite worker result")
    return {"shape": list(logits.shape), "dtype": str(logits.dtype),
            "logits": logits.reshape(-1).tolist()}


class IPCServer:
    """One request per authenticated local connection, backed by a trusted broker.

    The server never constructs the model or closes the caller-owned broker.
    Factory/encoder overrides are trusted local configuration, not wire inputs.
    """

    def __init__(self, broker, *, request_factory=None, result_encoder=None,
                 max_connections=8, max_request_bytes=_MAX_REQUEST,
                 max_response_bytes=_MAX_RESPONSE, max_batch=8, max_tokens=256,
                 io_timeout=3.0, job_timeout=60.0, team_ids=None):
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
        self.broker = broker
        self._broker_id = getattr(broker, "broker_id", None)
        if type(self._broker_id) is not str or not _BROKER_ID.fullmatch(self._broker_id):
            raise ValueError("broker must expose its 32-character hexadecimal lifetime ID")
        self.request_factory = request_factory or _default_request_factory
        self.result_encoder = result_encoder
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.max_batch, self.max_tokens = max_batch, max_tokens
        self.io_timeout, self.job_timeout = io_timeout, job_timeout
        if team_ids is not None:
            if (type(team_ids) not in (list, tuple) or not 1 <= len(team_ids) <= 128 or
                    not all(_identifier(team) for team in team_ids) or len(set(team_ids)) != len(team_ids)):
                raise ValueError("team_ids must contain 1 to 128 distinct valid team identifiers")
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
        self._slots = threading.BoundedSemaphore(max_connections)
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._admission = threading.Condition()
        self._admissions_active = 0
        self._clients = set()
        self._handlers = set()
        self._stop = threading.Event()
        self._accepting_stopped = threading.Event()
        self._listener = None
        self._accept_thread = None
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

    def start(self):
        with self._lifecycle:
            if self._listener is not None or self._accepting_stopped.is_set():
                raise RuntimeError("server already started or closed")
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
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
                self._accept_thread.start()
            except BaseException:
                listener.close()
                raise
        return self

    def _accept(self):
        while not self._accepting_stopped.is_set():
            try:
                if not select.select([self._listener], [], [], 0.1)[0]:
                    continue
                client, peer = self._listener.accept()
                client.setblocking(False)
            except (OSError, ValueError):
                if self._accepting_stopped.is_set():
                    return
                continue
            if peer[0] != "127.0.0.1" or not self._slots.acquire(blocking=False):
                try:
                    _send(client, {"version": 1, "status": "rejected", "error": "busy"},
                          self.max_response_bytes, time.monotonic() + min(self.io_timeout, 0.2))
                except (OSError, EOFError, TimeoutError, ValueError, _ProtocolError, _ResultTooLarge):
                    pass
                finally:
                    client.close()
                continue
            thread = threading.Thread(target=self._serve, args=(client,), daemon=True,
                                      name="looking-glass-ipc-client")
            with self._lock:
                if self._accepting_stopped.is_set():
                    client.close()
                    self._slots.release()
                    return
                self._clients.add(client)
                self._handlers.add(thread)
                thread.start()

    def _send(self, client, message):
        _send(client, message, self.max_response_bytes, time.monotonic() + self.io_timeout)

    def _serve(self, client):
        handle = None
        terminal = False
        request_id = None
        try:
            self._send(client, {"version": 1, "status": "ready"})
            message = _receive(client, self.max_request_bytes, time.monotonic() + self.io_timeout)
            _validate_request(message, self.token, self.team_tokens, self._invalid_token,
                              self.max_batch, self.max_tokens)
            request_id = message["request_id"]
            request = self.request_factory(request_id, message["inputs"], message["seed"])
            if self._accepting_stopped.is_set():
                return
            with self._admission:
                if self._accepting_stopped.is_set():
                    raise IPCRejected("unavailable")
                self._admissions_active += 1
            try:
                try:
                    handle = self.broker.submit(message["team_id"], request)
                except Exception as exc:
                    code = {"QueueFull": "queue_full", "BrokerUnavailable": "unavailable"}.get(
                        type(exc).__name__, "admission_failed")
                    raise IPCRejected(code) from exc
                if self._accepting_stopped.is_set():
                    handle.cancel()
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
                try:
                    outcome = handle.wait(timeout=0)
                except TimeoutError:
                    outcome = None
                if outcome is not None:
                    self._send_outcome(client, outcome, handle, request_id, message["team_id"])
                    terminal = True
                    return
                if _remaining(deadline) == 0:
                    handle.cancel()
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
                    if handle.cancel():
                        cancellation_status = "cancel_requested"
                        self._send(client, {"version": 1, "status": "cancel_requested",
                                            "request_id": request_id, "team_id": message["team_id"],
                                            "broker_id": self.broker_id, "job_id": handle.job_id})
                        continue
                    try:
                        outcome = handle.wait(timeout=0)
                    except TimeoutError:
                        cancellation_status = "cancel_not_accepted"
                        self._send(client, {"version": 1, "status": "cancel_not_accepted",
                                            "request_id": request_id, "team_id": message["team_id"],
                                            "broker_id": self.broker_id, "job_id": handle.job_id})
                        continue
                    self._send_outcome(client, outcome, handle, request_id, message["team_id"])
                    terminal = True
                    return
        except IPCRejected as exc:
            self._try_reject(client, exc.code, request_id)
        except (_ProtocolError, _ResultTooLarge, ValueError, TypeError):
            if handle is None:
                self._try_reject(client, "invalid_request", request_id)
        except (OSError, EOFError, TimeoutError):
            pass
        except Exception:
            if handle is None:
                self._try_reject(client, "internal_error", request_id)
        finally:
            if handle is not None and not terminal:
                try:
                    handle.cancel()
                except Exception:
                    pass
            # Drop potentially large result/input references before a slow peer
            # drains the socket. The broker retains any running work it owns.
            handle = request = message = outcome = None
            try:
                if terminal:
                    self._graceful_finish(client)
                client.close()
            finally:
                with self._lock:
                    self._clients.discard(client)
                    self._handlers.discard(threading.current_thread())
                self._slots.release()

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
            self._send(client, {"version": 1, "status": "failed", "error": "invalid_result",
                                "request_id": request_id, "team_id": team_id,
                                "broker_id": self.broker_id, "job_id": handle.job_id})
            return
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
        except (MemoryError, _ResultTooLarge, _ProtocolError, ValueError, TypeError) as exc:
            response.pop("result", None)
            response["status"] = "failed"
            if isinstance(exc, MemoryError):
                response["error"] = "serialization_out_of_memory"
            elif isinstance(exc, _ResultTooLarge):
                response["error"] = "result_too_large"
            else:
                response["error"] = "invalid_result"
            self._send(client, response)

    def stop_accepting(self, timeout=5.0):
        """Stop new submissions without interrupting already accepted jobs."""
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        with self._admission:
            self._accepting_stopped.set()
        if not self._lifecycle.acquire(timeout=_remaining(deadline)):
            return False
        try:
            if self._listener is not None:
                self._listener.close()
        finally:
            self._lifecycle.release()
        with self._admission:
            if not self._admission.wait_for(lambda: self._admissions_active == 0,
                                            timeout=_remaining(deadline)):
                return False
        if self._accept_thread is not None and self._accept_thread is not threading.current_thread():
            self._accept_thread.join(timeout=_remaining(deadline))
        return self._accept_thread is None or not self._accept_thread.is_alive()

    def drain(self, timeout=5.0):
        """Wait for client handlers without cancelling accepted work."""
        if not self._accepting_stopped.is_set():
            raise IPCError("stop accepting before draining")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        if not self.stop_accepting(timeout=_remaining(deadline)):
            return False
        if self._accept_thread is not None and self._accept_thread is not threading.current_thread():
            self._accept_thread.join(timeout=_remaining(deadline))
        with self._lock:
            handlers = list(self._handlers)
        for thread in handlers:
            if thread is not threading.current_thread():
                thread.join(timeout=_remaining(deadline))
        with self._lock:
            return (not self._handlers and
                    (self._accept_thread is None or not self._accept_thread.is_alive()))

    def close(self, timeout=5.0):
        """Stop admission and cancel clients; return False while handlers remain.

        A trusted factory or broker submit may be blocked. In that case this
        method does not claim shutdown succeeded, and can be called again.
        """
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        self._stop.set()
        self._accepting_stopped.set()
        if not self._lifecycle.acquire(timeout=_remaining(deadline)):
            return False
        try:
            if self._listener is not None:
                self._listener.close()
        finally:
            self._lifecycle.release()
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
        if self._accept_thread is not None and self._accept_thread is not threading.current_thread():
            self._accept_thread.join(timeout=_remaining(deadline))
        with self._lock:
            handlers = list(self._handlers)
        for thread in handlers:
            if thread is not threading.current_thread():
                thread.join(timeout=_remaining(deadline))
        with self._lock:
            return (not self._handlers and
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


class IPCClient:
    """Connect to a known loopback server using an out-of-band secret."""

    def __init__(self, address, token, *, io_timeout=3.0, max_response_bytes=_MAX_RESPONSE):
        if (type(address) is not tuple or len(address) != 2 or address[0] != "127.0.0.1" or
                type(address[1]) is not int or not 0 < address[1] < 65536):
            raise ValueError("only an IPv4 loopback endpoint is allowed")
        if type(token) is not str or not token.isascii() or not token:
            raise ValueError("invalid token")
        if type(io_timeout) not in (int, float) or not math.isfinite(io_timeout) or io_timeout <= 0:
            raise ValueError("invalid io_timeout")
        if type(max_response_bytes) is not int or not 512 <= max_response_bytes <= _MAX_RESPONSE:
            raise ValueError("invalid max_response_bytes")
        self.address, self.token = address, token
        self.io_timeout, self.max_response_bytes = io_timeout, max_response_bytes

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
                             self.max_response_bytes)
        except BaseException:
            sock.close()
            raise


class IPCHandle:
    def __init__(self, sock, request_id, team_id, broker_id, job_id, io_timeout, max_response_bytes):
        self._sock, self.request_id, self.team_id = sock, request_id, team_id
        self.broker_id, self.job_id = broker_id, job_id
        self._io_timeout, self._max_response_bytes = io_timeout, max_response_bytes
        self._send_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._closed = False
        self._cancel_sent = False
        self.cancel_acknowledged = False

    def cancel(self):
        """Request cooperative cancellation; call wait() for its acknowledgement."""
        with self._send_lock:
            if self._closed:
                raise IPCError("handle closed")
            if self._cancel_sent:
                return
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
                        self.cancel_acknowledged = response["status"] == "cancel_requested"
                        continue
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
