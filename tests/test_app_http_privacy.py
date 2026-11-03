"""Unexpected errors and lost clients must not print private request data."""

import contextlib
import http.client
import io
import json
import socket
import struct
import sys
import threading

from looking_glass.app import AppState, create_server


class ObservedLock:
    """Expose when a second request has reached the serialized state lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.queued = threading.Event()

    def __enter__(self):
        if self.lock.locked():
            self.queued.set()
        self.lock.acquire()
        return self

    def __exit__(self, *_):
        self.lock.release()


def test_unexpected_get_error_returns_generic_500_without_logging_private_text(tmp_path):
    marker = "PRIVATE_GET_DIAGNOSTIC_MARKER"
    state = AppState(runtime=object(), export_root=tmp_path)

    def fail_status():
        raise RuntimeError(marker)

    state.status = fail_status
    server = create_server(state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    errors = io.StringIO()
    connection = None
    try:
        with contextlib.redirect_stderr(errors):
            worker.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            connection.request("GET", "/api/status", headers={"X-MLG-Token": state.token})
            try:
                response = connection.getresponse()
                status, body = response.status, response.read()
            except http.client.RemoteDisconnected:
                status, body = None, b""
            connection.close()
            server.shutdown()
            worker.join(5)
        assert not worker.is_alive()
        assert status == 500
        assert marker not in body.decode("utf-8")
        assert marker not in errors.getvalue()
        assert "Traceback" not in errors.getvalue()
    finally:
        if connection is not None:
            connection.close()
        if worker.is_alive():
            server.shutdown()
            worker.join(5)
        server.server_close()


def test_disconnected_queued_request_does_not_log_private_error(tmp_path):
    marker = "PRIVATE_POST_DIAGNOSTIC_MARKER"
    state = AppState(runtime=object(), export_root=tmp_path)
    state.lock = ObservedLock()
    first_started, release_first, second_called, second_finished = (threading.Event() for _ in range(4))

    def post(_action, data):
        if data["id"] == 1:
            first_started.set()
            assert release_first.wait(5)
            return {"ok": True}
        second_called.set()
        raise ValueError(marker)

    state.post = post
    server = create_server(state)
    original_process = server.process_request_thread
    ports = {}

    def process(request, address):
        try:
            return original_process(request, address)
        finally:
            event = ports.get(address[1])
            if event is not None:
                event.set()

    server.process_request_thread = process
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    errors = io.StringIO()
    first = other = None

    def send(client, payload):
        body = json.dumps(payload).encode("ascii")
        header = (f"POST /api/sensitivity HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                  f"X-MLG-Token: {state.token}\r\nContent-Type: application/json\r\n"
                  f"Content-Length: {len(body)}\r\n\r\n").encode("ascii")
        client.sendall(header + body)

    try:
        with contextlib.redirect_stderr(errors):
            worker.start()
            first = socket.create_connection(("127.0.0.1", server.server_port), timeout=5)
            first.settimeout(5)
            send(first, {"id": 1})
            assert first_started.wait(5)
            other = socket.create_connection(("127.0.0.1", server.server_port), timeout=5)
            ports[other.getsockname()[1]] = second_finished
            send(other, {"id": 2})
            assert state.lock.queued.wait(5)
            # A reset makes the response fail even if the client OS buffers writes.
            linger = struct.pack("HH" if sys.platform == "win32" else "ii", 1, 0)
            other.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
            other.close()
            other = None
            release_first.set()
            assert second_called.wait(5)
            assert second_finished.wait(5)
            while first.recv(4096):
                pass
            first.close()
            first = None
            server.shutdown()
            worker.join(5)
        assert not worker.is_alive()
        assert marker not in errors.getvalue()
        assert "Traceback" not in errors.getvalue()
    finally:
        release_first.set()
        if first is not None:
            first.close()
        if other is not None:
            other.close()
        if worker.is_alive():
            server.shutdown()
            worker.join(5)
        server.server_close()


def test_unpaired_unicode_in_post_error_does_not_log_request_text(tmp_path):
    marker = "PRIVATE_UNICODE_DIAGNOSTIC_MARKER"
    state = AppState(runtime=object(), export_root=tmp_path)

    def post(_action, _data):
        raise ValueError(marker + chr(0xD800))

    state.post = post
    server = create_server(state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    errors = io.StringIO()
    connection = None
    try:
        with contextlib.redirect_stderr(errors):
            worker.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            connection.request("POST", "/api/sensitivity", body="{}", headers={
                "X-MLG-Token": state.token, "Content-Type": "application/json"})
            try:
                response = connection.getresponse()
                status, body = response.status, response.read()
            except http.client.RemoteDisconnected:
                status, body = None, b""
            connection.close()
            server.shutdown()
            worker.join(5)
        assert not worker.is_alive()
        assert status == 400
        assert json.loads(body)["error"] == marker + chr(0xD800)
        assert marker not in errors.getvalue()
        assert "Traceback" not in errors.getvalue()
    finally:
        if connection is not None:
            connection.close()
        if worker.is_alive():
            server.shutdown()
            worker.join(5)
        server.server_close()
