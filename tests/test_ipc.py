"""Real local-socket tests with a deterministic CPU broker stand-in."""
import importlib.util
import json
import socket
import struct
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch


class FakeHandle:
    def __init__(self, job_id, team_id, request):
        self.request = request
        self.job_id = job_id
        self.team_id = team_id
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.outcome = None

    def finish(self):
        self.outcome = types.SimpleNamespace(
            broker_id="a" * 32, job_id=self.job_id, team_id=self.team_id,
            request_id=self.request["request_id"], state="succeeded",
            result={"sum": sum(sum(row) for row in self.request["inputs"]["input_ids"])},
            error_type=None, error_message=None,
            submitted_ns=1, started_ns=2, finished_ns=3,
        )
        self.done.set()

    def wait(self, timeout=None):
        if not self.done.wait(timeout):
            raise TimeoutError
        return self.outcome

    def cancel(self):
        if self.done.is_set():
            return False
        self.cancelled.set()
        self.outcome = types.SimpleNamespace(
            broker_id="a" * 32, job_id=self.job_id, team_id=self.team_id,
            request_id=self.request["request_id"], state="cancelled", result=None,
            error_type=None, error_message=None,
            submitted_ns=1, started_ns=None, finished_ns=2,
        )
        self.done.set()
        return True


class FakeBroker:
    broker_id = "a" * 32
    def __init__(self, *, block=False):
        self.block = block
        self.handles = []
        self.lock = threading.Lock()

    def submit(self, team_id, request):
        with self.lock:
            handle = FakeHandle(len(self.handles) + 1, team_id, request)
            self.handles.append(handle)
        if not self.block:
            handle.finish()
        return handle


def make_request(request_id, inputs, seed):
    return {"request_id": request_id, "inputs": inputs, "seed": seed}


class IPCTest(unittest.TestCase):
    def make_server(self, broker=None, **options):
        from looking_glass.ipc import IPCServer
        server = IPCServer(broker or FakeBroker(), request_factory=make_request,
                           result_encoder=lambda value: value, **options)
        server.start()
        self.addCleanup(server.close)
        return server

    def read_frame(self, sock):
        def receive(size):
            data = b""
            while len(data) < size:
                part = sock.recv(size - len(data))
                if not part:
                    return None
                data += part
            return data
        header = receive(4)
        if header is None:
            return None
        data = receive(struct.unpack("!I", header)[0])
        return None if data is None else json.loads(data)

    def send_raw(self, server, payload):
        with socket.create_connection(server.address, timeout=2) as sock:
            sock.settimeout(2)
            self.assertEqual(self.read_frame(sock)["status"], "ready")
            sock.sendall(struct.pack("!I", len(payload)) + payload)
            return self.read_frame(sock)

    def test_independent_processes_submit_and_receive_owned_results(self):
        self.assertIsNotNone(importlib.util.find_spec("looking_glass.ipc"),
                             "local IPC transport is not implemented")
        from looking_glass.ipc import IPCServer

        broker = FakeBroker()
        with IPCServer(broker, request_factory=make_request,
                       result_encoder=lambda value: value) as server:
            child_code = """
import json, sys
from looking_glass.ipc import IPCClient
client = IPCClient((sys.argv[1], int(sys.argv[2])), sys.argv[3])
handle = client.submit(team_id=sys.argv[4], request_id=sys.argv[5],
                       inputs={'input_ids': [[int(sys.argv[6]), 2]]})
print(json.dumps(handle.wait(timeout=5)))
"""
            children = [subprocess.Popen(
                [sys.executable, "-c", child_code, server.address[0],
                 str(server.address[1]), server.token, f"team-{i}", f"req-{i}", str(i)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ) for i in (1, 4)]
            results = []
            for child in children:
                output, error = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, error)
                results.append(json.loads(output))
            self.assertEqual([(x["request_id"], x["result"]["sum"]) for x in results],
                             [("req-1", 3), ("req-4", 6)])
            self.assertEqual([x["broker_id"] for x in results], ["a" * 32, "a" * 32])
            self.assertEqual({h.team_id for h in broker.handles}, {"team-1", "team-4"})

    def test_unauthorized_and_unknown_jobs_do_not_reach_broker(self):
        broker = FakeBroker()
        server = self.make_server(broker)
        request = {"version": 1, "op": "submit", "token": "wrong",
                   "team_id": "t", "request_id": "r", "job_type": "forward",
                   "inputs": {"input_ids": [[1, 2]]}, "seed": 0}
        self.assertEqual(self.send_raw(server, json.dumps(request).encode())["status"], "rejected")
        request["token"] = server.token
        request["job_type"] = "arbitrary_code"
        self.assertEqual(self.send_raw(server, json.dumps(request).encode())["status"], "rejected")
        self.assertEqual(broker.handles, [])

    def test_per_team_tokens_cannot_impersonate_another_team(self):
        from looking_glass.ipc import IPCClient, IPCRejected
        broker = FakeBroker()
        server = self.make_server(broker, team_ids=["team-a", "team-b"])
        self.assertIsNone(server.token)
        self.assertNotEqual(server.team_tokens["team-a"], server.team_tokens["team-b"])
        with self.assertRaises(TypeError):
            server.team_tokens["team-a"] = "replacement"
        with self.assertRaises(AttributeError):
            server.team_tokens = {}
        with self.assertRaises(AttributeError):
            server.token = "replacement"
        team_a = IPCClient(server.address, server.team_tokens["team-a"])
        with self.assertRaises(IPCRejected) as rejected:
            team_a.submit(team_id="team-b", request_id="bad", inputs={"input_ids": [[1]]})
        self.assertEqual(rejected.exception.code, "unauthorized")
        result = team_a.submit(team_id="team-a", request_id="ok", inputs={"input_ids": [[2]]}).wait(3)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual([handle.team_id for handle in broker.handles], ["team-a"])

    def test_team_configuration_rejects_invalid_or_ambiguous_identities(self):
        from looking_glass.ipc import IPCServer
        for team_ids in ([], ["same", "same"], ["bad team"], [True]):
            with self.subTest(team_ids=team_ids):
                with self.assertRaises(ValueError):
                    IPCServer(FakeBroker(), team_ids=team_ids)

    def test_invalid_token_shapes_and_values_never_reach_the_broker(self):
        from looking_glass.ipc import IPCClient, IPCRejected
        broker = FakeBroker()
        server = self.make_server(broker, max_batch=2, max_tokens=3)
        client = IPCClient(server.address, server.token)
        bad_inputs = [
            {"input_ids": [[1, 2], [3]]},
            {"input_ids": [[1, 2, 3, 4]]},
            {"input_ids": [[True]]},
            {"input_ids": [[-1]]},
            {"input_ids": [[1]], "attention_mask": [[2]]},
            {"input_ids": [[1, 2]], "attention_mask": [[1]]},
            {"input_ids": [[1]], "past_key_values": [[1]]},
        ]
        for index, inputs in enumerate(bad_inputs):
            with self.subTest(inputs=inputs):
                with self.assertRaises(IPCRejected):
                    client.submit(team_id="t", request_id=f"bad-{index}", inputs=inputs)
        self.assertEqual(broker.handles, [])

    def test_queue_full_is_explicit_backpressure(self):
        from looking_glass.ipc import IPCClient, IPCRejected
        class QueueFull(Exception):
            pass

        class FullBroker:
            broker_id = "a" * 32
            def submit(self, team_id, request):
                raise QueueFull()

        server = self.make_server(FullBroker())
        with self.assertRaises(IPCRejected) as rejected:
            IPCClient(server.address, server.token).submit(
                team_id="a", request_id="a", inputs={"input_ids": [[1]]})
        self.assertEqual(rejected.exception.code, "queue_full")

    def test_client_rejects_noninteger_admission_metadata(self):
        from looking_glass.ipc import IPCClient, IPCError
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        self.addCleanup(listener.close)

        def write_frame(sock, value):
            payload = json.dumps(value).encode("utf-8")
            sock.sendall(struct.pack("!I", len(payload)) + payload)

        def malformed_server():
            sock, _ = listener.accept()
            with sock:
                sock.settimeout(3)
                write_frame(sock, {"version": 1, "status": "ready"})
                self.read_frame(sock)
                write_frame(sock, {"version": 1, "status": "accepted", "request_id": "r",
                                   "team_id": "t", "job_id": True, "broker_id": "a" * 32})

        thread = threading.Thread(target=malformed_server)
        thread.start()
        with self.assertRaises(IPCError):
            IPCClient(listener.getsockname(), "secret").submit(
                team_id="t", request_id="r", inputs={"input_ids": [[1]]})
        thread.join(3)
        self.assertFalse(thread.is_alive())

    def test_rejects_duplicate_fields_and_oversized_frames_before_admission(self):
        broker = FakeBroker()
        server = self.make_server(broker, max_request_bytes=512)
        self.assertEqual(self.send_raw(server, b'{"op":"submit","op":"cancel"}')["status"], "rejected")
        with socket.create_connection(server.address, timeout=2) as sock:
            sock.settimeout(2)
            self.assertEqual(self.read_frame(sock)["status"], "ready")
            sock.sendall(struct.pack("!I", 513))
            self.assertEqual(self.read_frame(sock)["status"], "rejected")
        self.assertEqual(broker.handles, [])

    def test_fragmented_request_is_reassembled_and_processed(self):
        server = self.make_server()
        request = {"version": 1, "op": "submit", "token": server.token,
                   "team_id": "t", "request_id": "r", "job_type": "forward",
                   "inputs": {"input_ids": [[4, 7]]}, "seed": 0}
        payload = json.dumps(request).encode("utf-8")
        frame = struct.pack("!I", len(payload)) + payload
        with socket.create_connection(server.address, timeout=2) as sock:
            sock.settimeout(3)
            self.assertEqual(self.read_frame(sock)["status"], "ready")
            for byte in frame:
                sock.sendall(bytes((byte,)))
            self.assertEqual(self.read_frame(sock)["status"], "accepted")
            result = self.read_frame(sock)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["result"]["sum"], 11)

    def test_json_exponent_overflow_is_rejected_as_nonfinite(self):
        from looking_glass.ipc import _ProtocolError, _receive
        reader, writer = socket.socketpair()
        self.addCleanup(reader.close)
        self.addCleanup(writer.close)
        payload = b'{"value":1e999}'
        writer.sendall(struct.pack("!I", len(payload)) + payload)
        reader.setblocking(False)
        with self.assertRaises(_ProtocolError):
            _receive(reader, 512, time.monotonic() + 2)

    def test_cancellation_and_disconnect_cancel_only_own_request(self):
        from looking_glass.ipc import IPCClient
        broker = FakeBroker(block=True)
        server = self.make_server(broker)
        client = IPCClient(server.address, server.token)
        a = client.submit(team_id="team-a", request_id="a", inputs={"input_ids": [[1]]})
        b = client.submit(team_id="team-b", request_id="b", inputs={"input_ids": [[2]]})
        a.cancel()
        self.assertEqual(a.wait(timeout=3)["status"], "cancelled")
        self.assertTrue(broker.handles[0].cancelled.wait(3))
        self.assertFalse(broker.handles[1].cancelled.is_set())
        b.close()
        self.assertTrue(broker.handles[1].cancelled.wait(3))

    def test_worker_failure_after_cancel_request_reaches_the_client(self):
        from looking_glass.ipc import IPCClient

        class SlowCancelHandle(FakeHandle):
            def cancel(self):
                self.cancelled.set()
                return True

        class SlowCancelBroker(FakeBroker):
            def submit(self, team_id, request):
                handle = SlowCancelHandle(1, team_id, request)
                self.handles.append(handle)
                return handle

        broker = SlowCancelBroker()
        server = self.make_server(broker)
        handle = IPCClient(server.address, server.token).submit(
            team_id="t", request_id="r", inputs={"input_ids": [[1]]})
        handle.cancel()
        handle.cancel()  # A repeated request must not lose a later failure.
        self.assertTrue(broker.handles[0].cancelled.wait(3))
        finished = threading.Event()
        received = []

        def wait():
            try:
                received.append(handle.wait(timeout=3))
            except Exception as exc:
                received.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=wait)
        thread.start()
        try:
            self.assertFalse(finished.wait(0.2), "cancel closed the connection before the worker stopped")
        finally:
            broker.handles[0].outcome = types.SimpleNamespace(
                broker_id="a" * 32, job_id=1, team_id="t", request_id="r", state="failed", result=None,
                error_type="WorkerOutOfMemory", error_message="simulated", submitted_ns=1,
                ready_ns=2, started_ns=3, finished_ns=4,
            )
            broker.handles[0].done.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(received), 1)
        outcome = received[0]
        self.assertEqual(outcome["status"], "failed")
        self.assertEqual(outcome["error"], "WorkerOutOfMemory")

    def test_cancel_racing_with_completion_does_not_discard_the_result(self):
        from looking_glass.ipc import IPCClient, IPCServer
        entered, release = threading.Event(), threading.Event()

        def slow_encoder(value):
            entered.set()
            release.wait(3)
            return value

        server = IPCServer(FakeBroker(), request_factory=make_request,
                           result_encoder=slow_encoder).start()
        self.addCleanup(server.close)
        self.addCleanup(release.set)
        handle = IPCClient(server.address, server.token).submit(
            team_id="t", request_id="r", inputs={"input_ids": [[1, 2]]})
        self.assertTrue(entered.wait(3))
        handle.cancel()  # Sent while the already-finished result is being encoded.
        release.set()
        result = handle.wait(timeout=3)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["result"]["sum"], 3)

    def test_connections_are_bounded_and_busy_requests_fail_closed(self):
        from looking_glass.ipc import IPCClient, IPCRejected
        broker = FakeBroker(block=True)
        server = self.make_server(broker, max_connections=1)
        client = IPCClient(server.address, server.token)
        first = client.submit(team_id="a", request_id="a", inputs={"input_ids": [[1]]})
        with self.assertRaises(IPCRejected) as rejected:
            client.submit(team_id="b", request_id="b", inputs={"input_ids": [[2]]})
        self.assertEqual(rejected.exception.code, "busy")
        first.close()
        self.assertTrue(broker.handles[0].cancelled.wait(3))
        self.assertEqual(len(broker.handles), 1)

    def test_job_deadline_requests_cancellation_and_reports_timeout(self):
        from looking_glass.ipc import IPCClient
        broker = FakeBroker(block=True)
        server = self.make_server(broker, job_timeout=0.15)
        handle = IPCClient(server.address, server.token).submit(
            team_id="t", request_id="r", inputs={"input_ids": [[1]]})
        self.assertEqual(handle.wait(timeout=3)["status"], "timeout")
        self.assertTrue(broker.handles[0].cancelled.wait(3))

    def test_client_wait_timeout_closes_connection_and_requests_cancellation(self):
        from looking_glass.ipc import IPCClient, IPCError
        broker = FakeBroker(block=True)
        server = self.make_server(broker, job_timeout=5)
        handle = IPCClient(server.address, server.token).submit(
            team_id="t", request_id="r", inputs={"input_ids": [[1]]})
        with self.assertRaises(TimeoutError):
            handle.wait(timeout=0.05)
        self.assertTrue(broker.handles[0].cancelled.wait(3))
        with self.assertRaises(IPCError):
            handle.wait(timeout=1)

    def test_shutdown_does_not_admit_after_blocked_request_factory_returns(self):
        from looking_glass.ipc import IPCClient, IPCServer
        entered, release = threading.Event(), threading.Event()
        broker = FakeBroker(block=True)

        def blocked_factory(request_id, inputs, seed):
            entered.set()
            release.wait(3)
            return make_request(request_id, inputs, seed)

        server = IPCServer(broker, request_factory=blocked_factory,
                           result_encoder=lambda value: value).start()
        self.addCleanup(lambda: release.set())
        self.addCleanup(lambda: server.close(timeout=3))
        failures = []

        def submit():
            try:
                IPCClient(server.address, server.token).submit(
                    team_id="t", request_id="r", inputs={"input_ids": [[1]]})
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=submit)
        thread.start()
        self.assertTrue(entered.wait(3))
        self.assertFalse(server.close(timeout=0.05))
        release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(server.close(timeout=3))
        self.assertEqual(broker.handles, [])
        self.assertTrue(failures)

    def test_stopping_admission_allows_terminal_failure_delivery(self):
        from looking_glass.ipc import IPCClient
        broker = FakeBroker(block=True)
        server = self.make_server(broker)
        handle = IPCClient(server.address, server.token).submit(
            team_id="t", request_id="r", inputs={"input_ids": [[1]]})
        self.assertTrue(server.stop_accepting(timeout=3))
        with self.assertRaises(OSError):
            socket.create_connection(server.address, timeout=1)
        self.assertFalse(server.drain(timeout=0.05))
        self.assertFalse(broker.handles[0].cancelled.is_set())
        broker.handles[0].outcome = types.SimpleNamespace(
            broker_id=broker.broker_id, job_id=1, team_id="t", request_id="r",
            state="failed", result=None, error_type="WorkerOutOfMemory", error_message="simulated",
            submitted_ns=1, ready_ns=2, started_ns=3, finished_ns=4,
        )
        broker.handles[0].done.set()
        outcome = handle.wait(timeout=3)
        self.assertEqual((outcome["status"], outcome["error"]), ("failed", "WorkerOutOfMemory"))
        self.assertTrue(server.drain(timeout=3))

    def test_slow_submission_does_not_block_another_team(self):
        from looking_glass.ipc import IPCClient
        entered, release, fast_done = threading.Event(), threading.Event(), threading.Event()

        class SlowSubmitBroker(FakeBroker):
            def submit(self, team_id, request):
                if team_id == "slow":
                    entered.set()
                    release.wait(3)
                return super().submit(team_id, request)

        broker = SlowSubmitBroker()
        server = self.make_server(broker)
        client = IPCClient(server.address, server.token)
        results = []

        def submit(team, done):
            try:
                outcome = client.submit(team_id=team, request_id=team,
                                        inputs={"input_ids": [[1]]}).wait(timeout=3)
                results.append((team, outcome["status"]))
            finally:
                done.set()

        slow_done = threading.Event()
        slow = threading.Thread(target=submit, args=("slow", slow_done))
        fast = threading.Thread(target=submit, args=("fast", fast_done))
        slow.start()
        self.assertTrue(entered.wait(3))
        fast.start()
        try:
            self.assertTrue(fast_done.wait(0.5), "slow submission blocked another team")
        finally:
            release.set()
            slow.join(3)
            fast.join(3)
        self.assertIn(("fast", "succeeded"), results)
        self.assertIn(("slow", "succeeded"), results)

    def test_stop_accepting_waits_for_inflight_submit_to_be_cancelled(self):
        from looking_glass.ipc import IPCClient, IPCRejected
        entered, release = threading.Event(), threading.Event()

        class SlowSubmitBroker(FakeBroker):
            def submit(self, team_id, request):
                entered.set()
                release.wait(3)
                return super().submit(team_id, request)

        broker = SlowSubmitBroker(block=True)
        server = self.make_server(broker)
        errors = []

        def submit():
            try:
                IPCClient(server.address, server.token).submit(
                    team_id="t", request_id="r", inputs={"input_ids": [[1]]})
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=submit)
        thread.start()
        self.assertTrue(entered.wait(3))
        try:
            self.assertFalse(server.stop_accepting(timeout=0.05))
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(server.stop_accepting(timeout=3))
        self.assertTrue(broker.handles[0].cancelled.wait(3))
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], IPCRejected)

    def test_drain_requires_admission_to_be_stopped(self):
        from looking_glass.ipc import IPCError
        server = self.make_server()
        with self.assertRaises(IPCError):
            server.drain(timeout=0.1)

    def test_close_during_start_does_not_leave_an_open_listener(self):
        from looking_glass.ipc import IPCServer
        entered, release = threading.Event(), threading.Event()
        original_socket = socket.socket
        created = []

        class DelayedSocket:
            def __init__(self, *args, **kwargs):
                self.inner = original_socket(*args, **kwargs)
                created.append(self.inner)

            def bind(self, address):
                entered.set()
                release.wait(3)
                return self.inner.bind(address)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        server = IPCServer(FakeBroker(), request_factory=make_request,
                           result_encoder=lambda value: value)
        errors = []

        def start():
            try:
                server.start()
            except Exception as exc:
                errors.append(exc)

        with patch("looking_glass.ipc.socket.socket", DelayedSocket):
            thread = threading.Thread(target=start)
            thread.start()
            self.assertTrue(entered.wait(3))
            try:
                self.assertFalse(server.close(timeout=0.05))
            finally:
                release.set()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertTrue(server.close(timeout=3))
        self.assertEqual(created[0].fileno(), -1)
        self.assertTrue(errors)

    def test_drain_cannot_claim_success_while_startup_is_blocked(self):
        from looking_glass.ipc import IPCServer
        entered, release = threading.Event(), threading.Event()
        original_socket = socket.socket

        class DelayedSocket:
            def __init__(self, *args, **kwargs):
                self.inner = original_socket(*args, **kwargs)

            def bind(self, address):
                entered.set()
                release.wait(3)
                return self.inner.bind(address)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        server = IPCServer(FakeBroker(), request_factory=make_request)
        errors = []

        def start():
            try:
                server.start()
            except Exception as exc:
                errors.append(exc)

        with patch("looking_glass.ipc.socket.socket", DelayedSocket):
            thread = threading.Thread(target=start)
            thread.start()
            self.assertTrue(entered.wait(3))
            try:
                self.assertFalse(server.stop_accepting(timeout=0.05))
                self.assertFalse(server.drain(timeout=0.05))
            finally:
                release.set()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertTrue(server.stop_accepting(timeout=3))
            self.assertTrue(server.drain(timeout=3))
            self.assertTrue(server.close(timeout=3))
        self.assertTrue(errors)

    def test_oversized_result_is_rejected_without_truncating(self):
        from looking_glass.ipc import IPCClient
        broker = FakeBroker(block=True)
        server = self.make_server(broker, max_response_bytes=512)
        handle = IPCClient(server.address, server.token).submit(
            team_id="t", request_id="r", inputs={"input_ids": [[1]]})
        broker.handles[0].outcome = types.SimpleNamespace(
            broker_id="a" * 32, job_id=1, team_id="t", request_id="r", state="succeeded",
            result={"logits": [123.456] * 200}, error_type=None, error_message=None,
            submitted_ns=1, ready_ns=2, started_ns=3, finished_ns=4,
        )
        broker.handles[0].done.set()
        result = handle.wait(timeout=3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "result_too_large")
        self.assertNotIn("result", result)

    def test_memory_error_during_serialization_is_a_failed_outcome(self):
        from looking_glass.ipc import IPCClient, IPCServer

        def out_of_memory(value):
            raise MemoryError("simulated CPU allocation failure")

        server = IPCServer(FakeBroker(), request_factory=make_request,
                           result_encoder=out_of_memory).start()
        self.addCleanup(server.close)
        handle = IPCClient(server.address, server.token).submit(
            team_id="t", request_id="r", inputs={"input_ids": [[1]]})
        result = handle.wait(timeout=3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "serialization_out_of_memory")

    def test_mismatched_broker_result_is_not_disclosed_to_the_client(self):
        from looking_glass.ipc import IPCClient
        broker = FakeBroker(block=True)
        server = self.make_server(broker)
        handle = IPCClient(server.address, server.token).submit(
            team_id="alice", request_id="my-request", inputs={"input_ids": [[1]]})
        broker.handles[0].outcome = types.SimpleNamespace(
            broker_id="a" * 32, job_id=999, team_id="bob", request_id="other-request", state="succeeded",
            result={"private": "other team's output"}, error_type=None, error_message=None,
            submitted_ns=1, ready_ns=2, started_ns=3, finished_ns=4,
        )
        broker.handles[0].done.set()
        result = handle.wait(timeout=3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "invalid_result")
        self.assertNotIn("result", result)

    def test_result_from_different_broker_lifetime_is_not_disclosed(self):
        from looking_glass.ipc import IPCClient
        broker = FakeBroker(block=True)
        server = self.make_server(broker)
        handle = IPCClient(server.address, server.token).submit(
            team_id="alice", request_id="my-request", inputs={"input_ids": [[1]]})
        broker.handles[0].outcome = types.SimpleNamespace(
            broker_id="b" * 32, job_id=1, team_id="alice", request_id="my-request", state="succeeded",
            result={"private": "old broker output"}, error_type=None, error_message=None,
            submitted_ns=1, ready_ns=2, started_ns=3, finished_ns=4,
        )
        broker.handles[0].done.set()
        result = handle.wait(timeout=3)
        self.assertEqual(result["broker_id"], "a" * 32)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "invalid_result")
        self.assertNotIn("result", result)

    def test_mismatched_worker_result_id_is_not_disclosed(self):
        import torch
        from looking_glass.ipc import IPCClient, IPCServer
        broker = FakeBroker(block=True)
        server = IPCServer(broker, request_factory=make_request).start()
        self.addCleanup(server.close)
        handle = IPCClient(server.address, server.token).submit(
            team_id="alice", request_id="my-request", inputs={"input_ids": [[1]]})
        broker.handles[0].outcome = types.SimpleNamespace(
            broker_id="a" * 32, job_id=1, team_id="alice", request_id="my-request", state="succeeded",
            result=types.SimpleNamespace(request_id="other-request", logits=torch.tensor([[[42.0]]])),
            error_type=None, error_message=None, submitted_ns=1, ready_ns=2, started_ns=3, finished_ns=4,
        )
        broker.handles[0].done.set()
        result = handle.wait(timeout=3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "invalid_result")
        self.assertNotIn("result", result)

    def test_real_cpu_worker_and_broker_share_one_model_across_clients(self):
        if (importlib.util.find_spec("looking_glass.broker") is None or
                importlib.util.find_spec("looking_glass.resident_worker") is None):
            self.skipTest("resident worker and broker not integrated in this checkout")
        import torch
        from torch import nn
        from looking_glass.broker import Broker
        from looking_glass.ipc import IPCClient, IPCServer
        from looking_glass.resident_worker import ResidentWorker

        class SmallModel(nn.Module):
            def forward(self, input_ids, attention_mask=None):
                values = input_ids.float()
                if attention_mask is not None:
                    values = values * attention_mask
                return {"logits": torch.stack((values, values + 1), dim=-1)}

        worker = ResidentWorker("cpu-test-model", SmallModel().eval(), device="cpu",
                                max_batch_size=2, max_sequence_length=8)
        broker = Broker(worker, max_pending=4, max_pending_per_team=2)
        try:
            with IPCServer(broker, team_ids=["team-a", "team-b"]) as server:
                child_code = """
import json, sys
from looking_glass.ipc import IPCClient
spec = json.load(sys.stdin)
client = IPCClient((spec['host'], spec['port']), spec['token'])
handle = client.submit(team_id=spec['team'], request_id='same-id', inputs=spec['inputs'])
print(json.dumps(handle.wait(timeout=8)))
"""
                specs = [
                    {"host": server.address[0], "port": server.address[1], "token": server.team_tokens["team-a"],
                     "team": "team-a", "inputs": {"input_ids": [[5, 8]], "attention_mask": [[1, 1]]}},
                    {"host": server.address[0], "port": server.address[1], "token": server.team_tokens["team-b"],
                     "team": "team-b", "inputs": {"input_ids": [[3, 2]]}},
                ]
                children = [subprocess.Popen([sys.executable, "-c", child_code],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True) for _ in specs]
                for child, spec in zip(children, specs):
                    child.stdin.write(json.dumps(spec))
                    child.stdin.close()
                outputs = []
                for child in children:
                    child.wait(timeout=10)
                    output, error = child.stdout.read(), child.stderr.read()
                    self.assertEqual(child.returncode, 0, error)
                    outputs.append(json.loads(output))
                    child.stdout.close()
                    child.stderr.close()
                a, b = outputs
                self.assertEqual((a["status"], b["status"]), ("succeeded", "succeeded"))
                self.assertEqual(a["broker_id"], broker.broker_id)
                self.assertEqual(b["broker_id"], broker.broker_id)
                self.assertNotEqual(a["job_id"], b["job_id"])
                self.assertEqual(a["result"]["shape"], [1, 2, 2])
                self.assertEqual(a["result"]["logits"], [5.0, 6.0, 8.0, 9.0])
                self.assertEqual(b["result"]["logits"], [3.0, 4.0, 2.0, 3.0])
        finally:
            self.assertTrue(broker.close(timeout=5))


if __name__ == "__main__":
    unittest.main()
