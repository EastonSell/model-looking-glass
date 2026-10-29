"""CPU-only local-socket tests for guardian notification of IPC references."""
import inspect
import json
import socket
import struct
import threading
import time
import types
import unittest
from unittest.mock import patch

from looking_glass.ipc import IPCClient, IPCError, IPCRejected, IPCServer
from looking_glass.ipc_custody import IPCCustodyCapacity, IPCCustodyKey


class FakeGuardian:
    def __init__(self):
        self.records = {}
        self.lock = threading.Lock()
        self.reserved = threading.Event()
        self.notified = threading.Event()
        self.failure = None
        self.callback_state = None

    def reserve(self, key):
        with self.lock:
            if key in self.records:
                raise AssertionError("duplicate")
            self.records[key] = {"authenticated": None, "handle": None,
                                 "notice": False, "charged": True}
        self.reserved.set()
        if self.failure == "reserve_after_commit":
            raise MemoryError("uncertain reserve")

    def authenticate(self, key, team_id, request_id):
        with self.lock:
            self.records[key]["authenticated"] = (team_id, request_id)
        if self.failure == "authenticate_after_commit":
            raise MemoryError("uncertain binding")

    def handler_closed(self, key):
        frame = inspect.currentframe().f_back
        assert frame.f_code.co_name == "_serve"
        self.callback_state = {name: frame.f_locals.get(name)
                               for name in ("handle", "outcome", "request", "message")}
        with self.lock:
            self.records[key]["notice"] = True
        self.notified.set()
        if self.failure == "notice_after_commit":
            raise MemoryError("uncertain notice")


class FakeHandle:
    def __init__(self, broker_id, job_id, team_id, request, *, done):
        self.job_id = job_id
        self._done = done
        self.cancel_requested = threading.Event()
        self.outcome = types.SimpleNamespace(
            broker_id=broker_id, job_id=job_id, team_id=team_id,
            request_id=request["request_id"], state="succeeded", result={"value": 7},
            error_type=None, error_message=None, submitted_ns=1, ready_ns=2,
            started_ns=3, finished_ns=4,
        )

    def wait(self, timeout=0):
        if not self._done.wait(timeout):
            raise TimeoutError
        return self.outcome

    def cancel(self):
        self.cancel_requested.set()
        return True


class RegisteredBroker:
    broker_id = "a" * 32

    def __init__(self, guardian, *, completed=True, fail_after_registration=False):
        self.guardian = guardian
        self.done = threading.Event()
        if completed:
            self.done.set()
        self.fail_after_registration = fail_after_registration
        self.calls = []

    @property
    def custody_guardian(self):
        return self.guardian

    def submit_registered(self, key, team_id, request):
        with self.guardian.lock:
            assert self.guardian.records[key]["authenticated"] == (team_id, request["request_id"])
            handle = FakeHandle(self.broker_id, len(self.calls) + 1, team_id, request, done=self.done)
            self.guardian.records[key]["handle"] = handle
        self.calls.append((key, team_id, request["request_id"]))
        if self.fail_after_registration:
            raise RuntimeError("submit return was lost")
        return handle


def factory(request_id, inputs, seed):
    return {"request_id": request_id, "inputs": inputs, "seed": seed}


class IPCCustodyNotificationTests(unittest.TestCase):
    def make_server(self, guardian, broker, **options):
        server = IPCServer(broker, team_ids=["team"], request_factory=factory,
                           result_encoder=lambda result: result,
                           custody_guardian=guardian, custody_epoch="b" * 32,
                           **options)
        server.start()
        self.addCleanup(server.close)
        return server

    def test_exact_key_validation_and_configuration(self):
        for epoch, generation in (("short", 1), ("b" * 32, True),
                                  ("b" * 32, 0), ("B" * 32, 1)):
            with self.assertRaises(ValueError):
                IPCCustodyKey(epoch, generation)
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian)
        with self.assertRaises(ValueError):
            IPCServer(broker, custody_guardian=guardian, custody_epoch="b" * 32)
        with self.assertRaises(ValueError):
            IPCServer(broker, team_ids=["team"], custody_guardian=guardian,
                      custody_epoch="short")
        with self.assertRaises(ValueError):
            IPCServer(broker, team_ids=["team"], custody_guardian=FakeGuardian(),
                      custody_epoch="b" * 32)

    def test_notice_follows_clear_and_does_not_release_owned_result(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        client = IPCClient(server.address, server.team_tokens["team"])
        result = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]}).wait(3)
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(guardian.notified.wait(3))
        key = broker.calls[0][0]
        self.assertIs(type(key), IPCCustodyKey)
        self.assertEqual(key.server_epoch, "b" * 32)
        self.assertEqual(key.connection_generation, 1)
        self.assertEqual(guardian.callback_state, {name: None for name in
                                                   ("handle", "outcome", "request", "message")})
        record = guardian.records[key]
        self.assertTrue(record["notice"])
        self.assertTrue(record["charged"])
        self.assertEqual(record["handle"].outcome.result, {"value": 7})
        self.assertFalse(server.custody_quarantined)

    def test_reservation_precedes_receive_and_unknown_auth_has_no_admission(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        with socket.create_connection(server.address, timeout=2) as sock:
            sock.settimeout(2)
            self.assertTrue(guardian.reserved.wait(2))
            header = sock.recv(4)
            size = struct.unpack("!I", header)[0]
            sock.recv(size)
            request = {"version": 1, "op": "submit", "token": "wrong", "team_id": "team",
                       "request_id": "case", "job_type": "forward", "inputs": {"input_ids": [[1]]}, "seed": 0}
            raw = json.dumps(request).encode()
            sock.sendall(struct.pack("!I", len(raw)) + raw)
        self.assertTrue(guardian.notified.wait(3))
        record = next(iter(guardian.records.values()))
        self.assertIsNone(record["authenticated"])
        self.assertTrue(record["charged"])
        self.assertEqual(broker.calls, [])
        self.assertFalse(server.custody_quarantined)

    def test_timeout_requests_cancel_but_does_not_release_running_job(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian, completed=False)
        server = self.make_server(guardian, broker, job_timeout=.1)
        client = IPCClient(server.address, server.team_tokens["team"])
        result = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]}).wait(3)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(guardian.notified.wait(3))
        record = guardian.records[broker.calls[0][0]]
        self.assertTrue(record["handle"].cancel_requested.is_set())
        self.assertFalse(broker.done.is_set())
        self.assertTrue(record["charged"])
        broker.done.set()

    def test_disconnected_client_retains_running_job(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian, completed=False)
        server = self.make_server(guardian, broker)
        client = IPCClient(server.address, server.team_tokens["team"])
        handle = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]})
        handle.close()
        self.assertTrue(guardian.notified.wait(3))
        record = guardian.records[broker.calls[0][0]]
        self.assertTrue(record["handle"].cancel_requested.is_set())
        self.assertFalse(broker.done.is_set())
        self.assertTrue(record["charged"])
        broker.done.set()

    def test_uncertain_guardian_registration_does_not_notify_or_release(self):
        guardian = FakeGuardian()
        guardian.failure = "reserve_after_commit"
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        with socket.create_connection(server.address, timeout=2) as sock:
            sock.settimeout(2)
            sock.recv(1)
        self.assertTrue(guardian.reserved.wait(2))
        self.assertTrue(server.close(3))
        record = next(iter(guardian.records.values()))
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertEqual(broker.calls, [])
        self.assertTrue(server.custody_quarantined)

    def test_proven_capacity_rejection_is_busy_without_quarantine(self):
        class LimitedGuardian(FakeGuardian):
            full = True

            def reserve(self, key):
                if self.full:
                    # Trusted guarantee: no record, claim or pending work.
                    raise IPCCustodyCapacity()
                return super().reserve(key)

        guardian = LimitedGuardian()
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        client = IPCClient(server.address, server.team_tokens["team"])
        with self.assertRaises(IPCRejected) as caught:
            client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]})
        self.assertEqual(caught.exception.code, "busy")
        self.assertFalse(server.custody_quarantined)
        self.assertEqual(guardian.records, {})
        self.assertFalse(guardian.notified.is_set())
        guardian.full = False
        result = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]}).wait(3)
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(guardian.notified.wait(3))
        self.assertEqual(broker.calls[0][0].connection_generation, 2)
        self.assertFalse(server.custody_quarantined)

    def test_saturation_survives_disconnect_until_owner_disposes_job(self):
        class SingleClaimGuardian(FakeGuardian):
            def reserve(self, key):
                with self.lock:
                    if any(record["charged"] for record in self.records.values()):
                        raise IPCCustodyCapacity()
                    self.records[key] = {"authenticated": None, "handle": None,
                                         "notice": False, "charged": True}
                self.reserved.set()

        guardian = SingleClaimGuardian()
        broker = RegisteredBroker(guardian, completed=False)
        server = self.make_server(guardian, broker)
        client = IPCClient(server.address, server.team_tokens["team"])
        running = client.submit(team_id="team", request_id="running", inputs={"input_ids": [[1]]})
        key = broker.calls[0][0]

        def assert_busy(request_id):
            with self.assertRaises(IPCRejected) as caught:
                client.submit(team_id="team", request_id=request_id, inputs={"input_ids": [[1]]})
            self.assertEqual(caught.exception.code, "busy")
            self.assertFalse(server.custody_quarantined)

        assert_busy("while-running")
        running.close()
        self.assertTrue(guardian.notified.wait(3))
        self.assertTrue(guardian.records[key]["handle"].cancel_requested.is_set())
        self.assertTrue(guardian.records[key]["charged"])
        self.assertFalse(broker.done.is_set())
        assert_busy("after-disconnect")

        # The trusted fake owner, not a disconnect or notice, establishes
        # completion and discards its retained handle before releasing quota.
        broker.done.set()
        with guardian.lock:
            guardian.records[key]["handle"] = None
            guardian.records[key]["charged"] = False
        result = client.submit(team_id="team", request_id="next", inputs={"input_ids": [[2]]}).wait(3)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(broker.calls[-1][0].connection_generation, 4)
        self.assertFalse(server.custody_quarantined)

    def test_capacity_subclass_after_partial_reserve_is_quarantined(self):
        class UncertainCapacity(IPCCustodyCapacity):
            pass

        class BadGuardian(FakeGuardian):
            def reserve(self, key):
                super().reserve(key)
                raise UncertainCapacity()

        guardian = BadGuardian()
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        with self.assertRaises((IPCError, OSError, EOFError)):
            IPCClient(server.address, server.team_tokens["team"]).submit(
                team_id="team", request_id="case", inputs={"input_ids": [[1]]})
        self.assertTrue(server.close(3))
        record = next(iter(guardian.records.values()))
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertTrue(server.custody_quarantined)

    def test_capacity_exception_outside_reserve_is_not_clean_overload(self):
        for where in ("authenticate", "submit", "notify"):
            with self.subTest(where=where):
                class FaultGuardian(FakeGuardian):
                    def authenticate(self, key, team_id, request_id):
                        super().authenticate(key, team_id, request_id)
                        if where == "authenticate":
                            raise IPCCustodyCapacity()

                    def handler_closed(self, key):
                        if where == "notify":
                            raise IPCCustodyCapacity()
                        return super().handler_closed(key)

                class FaultBroker(RegisteredBroker):
                    def submit_registered(self, key, team_id, request):
                        handle = super().submit_registered(key, team_id, request)
                        if where == "submit":
                            raise IPCCustodyCapacity()
                        return handle

                guardian = FaultGuardian()
                broker = FaultBroker(guardian)
                server = self.make_server(guardian, broker)
                client = IPCClient(server.address, server.team_tokens["team"])
                if where == "notify":
                    result = client.submit(team_id="team", request_id="case",
                                           inputs={"input_ids": [[1]]}).wait(3)
                    self.assertEqual(result["status"], "succeeded")
                else:
                    with self.assertRaises((IPCError, OSError, EOFError)):
                        client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]})
                self.assertTrue(server.close(3))
                record = next(iter(guardian.records.values()))
                self.assertFalse(record["notice"])
                self.assertTrue(record["charged"])
                self.assertTrue(server.custody_quarantined)

    def test_uncertain_binding_does_not_notify_or_release(self):
        guardian = FakeGuardian()
        guardian.failure = "authenticate_after_commit"
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        client = IPCClient(server.address, server.team_tokens["team"])
        with self.assertRaises((IPCError, OSError)):
            client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]})
        self.assertTrue(server.close(3))
        record = next(iter(guardian.records.values()))
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertTrue(server.custody_quarantined)
        self.assertEqual(broker.calls, [])

    def test_failed_request_factory_keeps_record_quarantined(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        retained = []

        def failing_factory(request_id, inputs, seed):
            retained.append(inputs)
            raise RuntimeError("unknown input alias ownership")

        server.request_factory = failing_factory
        client = IPCClient(server.address, server.team_tokens["team"])
        with self.assertRaises((IPCError, OSError, EOFError)):
            client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]})
        self.assertTrue(server.close(3))
        record = next(iter(guardian.records.values()))
        self.assertEqual(retained, [{"input_ids": [[1]]}])
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertTrue(server.custody_quarantined)
        self.assertEqual(broker.calls, [])

    def test_failed_handle_observation_keeps_record_quarantined(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian, completed=False)
        server = self.make_server(guardian, broker)

        def fail_wait(handle, timeout=0):
            raise RuntimeError("handle observation failed")

        with patch.object(FakeHandle, "wait", fail_wait):
            with self.assertRaises((IPCError, OSError, EOFError)):
                IPCClient(server.address, server.team_tokens["team"]).submit(
                    team_id="team", request_id="case", inputs={"input_ids": [[1]]}).wait(3)
            self.assertTrue(server.close(3))
        record = guardian.records[broker.calls[0][0]]
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertTrue(server.custody_quarantined)

    def test_failed_cancellation_remains_uncertain_after_retry(self):
        for mode in ("deadline", "explicit"):
            with self.subTest(mode=mode):
                guardian = FakeGuardian()
                broker = RegisteredBroker(guardian, completed=False)
                server = self.make_server(guardian, broker, job_timeout=.1 if mode == "deadline" else 3)
                calls = []

                def flaky_cancel(handle):
                    handle.cancel_requested.set()
                    calls.append(1)
                    if len(calls) == 1:
                        raise RuntimeError("cancellation outcome unknown")
                    return True

                with patch.object(FakeHandle, "cancel", flaky_cancel):
                    client = IPCClient(server.address, server.team_tokens["team"])
                    handle = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]})
                    if mode == "explicit":
                        handle.cancel()
                    with self.assertRaises((IPCError, OSError, EOFError)):
                        handle.wait(3)
                    self.assertTrue(server.close(3))
                self.assertEqual(len(calls), 2)
                record = guardian.records[broker.calls[0][0]]
                self.assertTrue(record["handle"].cancel_requested.is_set())
                self.assertFalse(record["notice"])
                self.assertTrue(record["charged"])
                self.assertTrue(server.custody_quarantined)

    def test_failed_cancellation_during_stopped_admission_is_uncertain(self):
        entered, resume = threading.Event(), threading.Event()

        class GatedBroker(RegisteredBroker):
            def submit_registered(self, key, team_id, request):
                handle = super().submit_registered(key, team_id, request)
                entered.set()
                resume.wait(3)
                return handle

        guardian = FakeGuardian()
        broker = GatedBroker(guardian, completed=False)
        server = self.make_server(guardian, broker)
        calls, observed = [], []

        def flaky_cancel(handle):
            handle.cancel_requested.set()
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("cancellation outcome unknown")
            return True

        def submit():
            try:
                IPCClient(server.address, server.team_tokens["team"]).submit(
                    team_id="team", request_id="case", inputs={"input_ids": [[1]]})
            except (IPCError, OSError, EOFError) as exc:
                observed.append(exc)

        with patch.object(FakeHandle, "cancel", flaky_cancel):
            client_thread = threading.Thread(target=submit)
            client_thread.start()
            self.assertTrue(entered.wait(2))
            self.assertFalse(server.stop_accepting(.01))
            resume.set()
            client_thread.join(3)
            self.assertFalse(client_thread.is_alive())
            self.assertTrue(server.close(3))
        self.assertTrue(observed)
        self.assertEqual(len(calls), 2)
        record = guardian.records[broker.calls[0][0]]
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertTrue(server.custody_quarantined)

    def test_baseexception_during_final_cancel_propagates_and_quarantines(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian, completed=False)
        server = self.make_server(guardian, broker)
        exceptions = []
        reported = threading.Event()

        def fail_cancel(handle):
            handle.cancel_requested.set()
            raise SystemExit("interrupted cancellation")

        def record_exception(args):
            exceptions.append(args.exc_type)
            reported.set()

        with patch.object(FakeHandle, "cancel", fail_cancel), patch.object(threading, "excepthook", record_exception):
            handle = IPCClient(server.address, server.team_tokens["team"]).submit(
                team_id="team", request_id="case", inputs={"input_ids": [[1]]})
            handle.close()
            self.assertTrue(reported.wait(3))
            self.assertTrue(server.close(3))
        self.assertEqual(exceptions, [SystemExit])
        record = guardian.records[broker.calls[0][0]]
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertTrue(server.custody_quarantined)

    def test_uncertain_encoder_does_not_notify_or_release(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        server.result_encoder = lambda result: (_ for _ in ()).throw(MemoryError("uncertain"))
        client = IPCClient(server.address, server.team_tokens["team"])
        result = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]}).wait(3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "serialization_out_of_memory")
        self.assertTrue(server.close(3))
        record = guardian.records[broker.calls[0][0]]
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertTrue(server.custody_quarantined)

    def test_lost_submit_return_stays_registered_and_quarantined(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian, fail_after_registration=True)
        server = self.make_server(guardian, broker)
        client = IPCClient(server.address, server.team_tokens["team"])
        with self.assertRaises((IPCError, OSError, EOFError)):
            client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]})
        self.assertTrue(server.close(3))
        key = broker.calls[0][0]
        record = guardian.records[key]
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertIsNotNone(record["handle"])
        self.assertTrue(server.custody_quarantined)

    def test_unhandled_serializer_baseexception_never_sends_notice(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        server.result_encoder = lambda result: (_ for _ in ()).throw(SystemExit("uncertain"))
        client = IPCClient(server.address, server.team_tokens["team"])
        reported = threading.Event()
        exceptions = []

        def record_exception(args):
            exceptions.append(args.exc_type)
            reported.set()

        with patch.object(threading, "excepthook", record_exception):
            with self.assertRaises((IPCError, OSError, EOFError)):
                client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]}).wait(3)
            self.assertTrue(reported.wait(3))
            self.assertTrue(server.close(3))
        self.assertEqual(exceptions, [SystemExit])
        record = guardian.records[broker.calls[0][0]]
        self.assertFalse(record["notice"])
        self.assertTrue(record["charged"])
        self.assertTrue(server.custody_quarantined)

    def test_interrupted_notification_latches_quarantine(self):
        guardian = FakeGuardian()
        guardian.failure = "notice_after_commit"
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        client = IPCClient(server.address, server.team_tokens["team"])
        result = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]}).wait(3)
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(server.close(3))
        record = guardian.records[broker.calls[0][0]]
        self.assertTrue(record["notice"])
        self.assertTrue(record["charged"])
        self.assertTrue(server.custody_quarantined)

    def test_blocked_notification_remains_in_shutdown_accounting(self):
        entered, resume = threading.Event(), threading.Event()

        class BlockingGuardian(FakeGuardian):
            def handler_closed(self, key):
                entered.set()
                resume.wait(3)
                with self.lock:
                    self.records[key]["notice"] = True
                self.notified.set()

        guardian = BlockingGuardian()
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        client = IPCClient(server.address, server.team_tokens["team"])
        result = client.submit(team_id="team", request_id="case", inputs={"input_ids": [[1]]}).wait(3)
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(entered.wait(2))
        self.assertFalse(server.close(.01))
        self.assertTrue(guardian.records[broker.calls[0][0]]["charged"])
        resume.set()
        self.assertTrue(server.close(3))
        self.assertTrue(guardian.notified.is_set())

    def test_post_notice_bookkeeping_failure_latches_before_thread_exit(self):
        for location in ("clients", "handlers", "slots"):
            for after_mutation in (False, True):
                with self.subTest(location=location, after_mutation=after_mutation):
                    class ObservingGuardian(FakeGuardian):
                        def handler_closed(self, key):
                            self.handler_thread = threading.current_thread()
                            with self.lock:
                                self.records[key]["notice"] = True
                            self.notified.set()

                    class FaultSet(set):
                        def discard(self, item):
                            if after_mutation:
                                super().discard(item)
                            raise RuntimeError("bookkeeping outcome unknown")

                    guardian = ObservingGuardian()
                    broker = RegisteredBroker(guardian)
                    server = self.make_server(guardian, broker)
                    seen = threading.Event()
                    failures = []

                    def observe_exception(args):
                        failures.append(args.exc_type)
                        seen.set()

                    released = []
                    if location == "slots":
                        original = server._slots.release

                        def fail_release():
                            if after_mutation:
                                original()
                                released.append(True)
                            raise RuntimeError("slot outcome unknown")

                        fault = patch.object(server._slots, "release", fail_release)
                    else:
                        name = "_" + location
                        fault = patch.object(server, name, FaultSet(getattr(server, name)))

                    with fault, patch.object(threading, "excepthook", observe_exception):
                        result = IPCClient(server.address, server.team_tokens["team"]).submit(
                            team_id="team", request_id="case", inputs={"input_ids": [[1]]}).wait(3)
                        self.assertEqual(result["status"], "succeeded")
                        self.assertTrue(guardian.notified.wait(3))
                        self.assertTrue(seen.wait(3))
                        guardian.handler_thread.join(3)
                        self.assertFalse(guardian.handler_thread.is_alive())
                        self.assertEqual(failures, [RuntimeError])
                        self.assertTrue(server.custody_quarantined)
                        record = guardian.records[broker.calls[0][0]]
                        self.assertTrue(record["notice"])
                        self.assertTrue(record["charged"])
                        if not (location == "handlers" and after_mutation):
                            self.assertFalse(server.close(.01))
                    # Test-only reconciliation after confirmed thread exit;
                    # production must independently reconcile uncertain state.
                    with server._lock:
                        server._handlers = set()
                        server._clients = set()
                    if location == "slots" and not released:
                        server._slots.release()
                    self.assertTrue(server.close(3))

    def test_generations_are_distinct_within_server_lifetime(self):
        guardian = FakeGuardian()
        broker = RegisteredBroker(guardian)
        server = self.make_server(guardian, broker)
        client = IPCClient(server.address, server.team_tokens["team"])
        for index in range(2):
            result = client.submit(team_id="team", request_id=f"case-{index}",
                                   inputs={"input_ids": [[1]]}).wait(3)
            self.assertEqual(result["status"], "succeeded")
        self.assertEqual([call[0].connection_generation for call in broker.calls], [1, 2])
        self.assertEqual(len(guardian.records), 2)
        self.assertTrue(all(item["charged"] for item in guardian.records.values()))


if __name__ == "__main__":
    unittest.main()
