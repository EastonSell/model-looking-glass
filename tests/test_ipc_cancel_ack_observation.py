"""Client acknowledgement observation over real local sockets; no model work."""
from dataclasses import FrozenInstanceError
import inspect
import json
import socket
import struct
import sys
import threading
import time
import unittest
from unittest import mock

from test_ipc import FakeBroker, make_request


class CancelAckObservationTests(unittest.TestCase):
    def observed_client(self, server):
        from looking_glass.ipc import IPCClient
        self.assertIn('observe_cancel_ack', inspect.signature(IPCClient).parameters,
                      'the opt-in acknowledgement observation is not implemented')
        return IPCClient(server.address, server.token, observe_cancel_ack=True)

    def make_server(self):
        from looking_glass.ipc import IPCServer
        server = IPCServer(FakeBroker(block=True), request_factory=make_request,
                           result_encoder=lambda value: value)
        server.start()
        self.addCleanup(server.close)
        return server

    def test_accepted_ack_is_observed_after_cancel_and_before_terminal_return(self):
        server = self.make_server()
        handle = self.observed_client(server).submit(team_id='team', request_id='job',
                                                      inputs={'input_ids': [[1, 2]]})
        before = time.perf_counter_ns()
        handle.cancel()
        result = handle.wait(3)
        after = time.perf_counter_ns()
        receipt = handle.cancel_ack_observation
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue(handle.cancel_acknowledged)
        self.assertIs(receipt.accepted, True)
        self.assertIs(type(receipt.received_ns), int)
        self.assertLessEqual(before, receipt.received_ns)
        self.assertLessEqual(receipt.received_ns, after)
        self.assertFalse(handle.cancel_ack_observation_failed)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            receipt.accepted = False

    def test_default_client_does_not_read_a_timestamp_or_capture_a_receipt(self):
        from looking_glass.ipc import IPCClient
        server = self.make_server()
        handle = IPCClient(server.address, server.token).submit(team_id='team', request_id='job',
                                                                 inputs={'input_ids': [[1, 2]]})
        self.assertTrue(hasattr(handle, 'cancel_ack_observation'),
                        'the opt-in acknowledgement observation is not implemented')
        handle.cancel()
        with mock.patch('looking_glass.ipc.time.perf_counter_ns', side_effect=AssertionError('clock called')):
            result = handle.wait(3)
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue(handle.cancel_acknowledged)
        self.assertIsNone(handle.cancel_ack_observation)
        self.assertFalse(handle.cancel_ack_observation_failed)

    def test_failed_receipt_clock_does_not_replace_the_real_outcome(self):
        server = self.make_server()
        handle = self.observed_client(server).submit(team_id='team', request_id='job',
                                                      inputs={'input_ids': [[1, 2]]})
        handle.cancel()
        with mock.patch('looking_glass.ipc.time.perf_counter_ns', side_effect=OSError('unavailable')):
            result = handle.wait(3)
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue(handle.cancel_acknowledged)
        self.assertIsNone(handle.cancel_ack_observation)
        self.assertTrue(handle.cancel_ack_observation_failed)

    def test_invalid_clock_value_marks_the_observation_failed(self):
        server = self.make_server()
        handle = self.observed_client(server).submit(team_id='team', request_id='job',
                                                      inputs={'input_ids': [[1, 2]]})
        handle.cancel()
        with mock.patch('looking_glass.ipc.time.perf_counter_ns', return_value=True):
            result = handle.wait(3)
        self.assertEqual(result['status'], 'cancelled')
        self.assertIsNone(handle.cancel_ack_observation)
        self.assertTrue(handle.cancel_ack_observation_failed)

    def test_control_flow_interruption_marks_receipt_unknown_and_propagates(self):
        server = self.make_server()
        handle = self.observed_client(server).submit(team_id='team', request_id='job',
                                                      inputs={'input_ids': [[1, 2]]})
        handle.cancel()
        with mock.patch('looking_glass.ipc.time.perf_counter_ns', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                handle.wait(3)
        self.assertIsNone(handle.cancel_ack_observation)
        self.assertTrue(handle.cancel_ack_observation_failed)

    def test_not_accepted_ack_is_distinct_from_successful_terminal(self):
        from looking_glass.ipc import IPCHandle
        self.assertIn('observe_cancel_ack', inspect.signature(IPCHandle).parameters,
                      'the opt-in acknowledgement observation is not implemented')
        client, peer = socket.socketpair()
        self.addCleanup(peer.close)
        client.setblocking(False)
        handle = IPCHandle(client, 'job', 'team', 'a' * 32, 1, 1, 4096, observe_cancel_ack=True)
        self.addCleanup(handle.close)
        before = time.perf_counter_ns()
        handle.cancel()
        for status in ('cancel_not_accepted', 'succeeded'):
            data = json.dumps({'version': 1, 'status': status, 'request_id': 'job',
                               'team_id': 'team', 'broker_id': 'a' * 32, 'job_id': 1}).encode()
            peer.sendall(struct.pack('!I', len(data)) + data)
        result = handle.wait(3)
        after = time.perf_counter_ns()
        self.assertEqual(result['status'], 'succeeded')
        self.assertIs(handle.cancel_ack_observation.accepted, False)
        self.assertLessEqual(before, handle.cancel_ack_observation.received_ns)
        self.assertLessEqual(handle.cancel_ack_observation.received_ns, after)
        self.assertFalse(handle.cancel_ack_observation_failed)

    def test_repeated_ack_is_flagged_incomplete_without_replacing_terminal(self):
        from looking_glass.ipc import IPCHandle
        self.assertIn('observe_cancel_ack', inspect.signature(IPCHandle).parameters,
                      'the opt-in acknowledgement observation is not implemented')
        client, peer = socket.socketpair()
        self.addCleanup(peer.close)
        client.setblocking(False)
        handle = IPCHandle(client, 'job', 'team', 'a' * 32, 1, 1, 4096, observe_cancel_ack=True)
        self.addCleanup(handle.close)
        handle.cancel()
        for status in ('cancel_requested', 'cancel_requested', 'cancelled'):
            data = json.dumps({'version': 1, 'status': status, 'request_id': 'job',
                               'team_id': 'team', 'broker_id': 'a' * 32, 'job_id': 1}).encode()
            peer.sendall(struct.pack('!I', len(data)) + data)
        self.assertEqual(handle.wait(3)['status'], 'cancelled')
        self.assertTrue(handle.cancel_ack_observation_failed)

    def test_unsolicited_ack_is_not_valid_client_cancellation_evidence(self):
        from looking_glass.ipc import IPCHandle
        client, peer = socket.socketpair()
        self.addCleanup(peer.close)
        client.setblocking(False)
        handle = IPCHandle(client, 'job', 'team', 'a' * 32, 1, 1, 4096, observe_cancel_ack=True)
        self.addCleanup(handle.close)
        for status in ('cancel_requested', 'succeeded'):
            data = json.dumps({'version': 1, 'status': status, 'request_id': 'job',
                               'team_id': 'team', 'broker_id': 'a' * 32, 'job_id': 1}).encode()
            peer.sendall(struct.pack('!I', len(data)) + data)
        self.assertEqual(handle.wait(3)['status'], 'succeeded')
        self.assertIsNone(handle.cancel_ack_observation)
        self.assertTrue(handle.cancel_ack_observation_failed)

    def test_accepted_ack_then_success_is_retained_but_evidence_is_incomplete(self):
        from looking_glass.ipc import IPCHandle
        client, peer = socket.socketpair()
        self.addCleanup(peer.close)
        client.setblocking(False)
        handle = IPCHandle(client, 'job', 'team', 'a' * 32, 1, 1, 4096, observe_cancel_ack=True)
        self.addCleanup(handle.close)
        handle.cancel()
        for status in ('cancel_requested', 'succeeded'):
            data = json.dumps({'version': 1, 'status': status, 'request_id': 'job',
                               'team_id': 'team', 'broker_id': 'a' * 32, 'job_id': 1}).encode()
            peer.sendall(struct.pack('!I', len(data)) + data)
        self.assertEqual(handle.wait(3)['status'], 'succeeded')
        self.assertTrue(handle.cancel_ack_observation_failed)

    def test_interrupted_capture_entry_does_not_look_complete(self):
        from looking_glass.ipc import IPCHandle
        server = self.make_server()
        handle = self.observed_client(server).submit(team_id='team', request_id='job',
                                                      inputs={'input_ids': [[1, 2]]})
        handle.cancel()

        def interrupt(frame, event, arg):
            if frame.f_code is IPCHandle._capture_cancel_ack.__code__ and event == 'call':
                raise KeyboardInterrupt('capture entry')
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                handle.wait(3)
        finally:
            sys.settrace(None)
        self.assertIsNone(handle.cancel_ack_observation)
        self.assertTrue(handle.cancel_ack_observation_failed)

    def test_ack_during_cancel_send_is_not_mistaken_for_an_unsolicited_frame(self):
        import looking_glass.ipc as ipc
        server = self.make_server()
        handle = self.observed_client(server).submit(team_id='team', request_id='job',
                                                      inputs={'input_ids': [[1, 2]]})
        sent, release = threading.Event(), threading.Event()
        original_send = ipc._send
        errors = []

        def delayed_send(sock, *args):
            original_send(sock, *args)
            if sock is handle._sock:
                sent.set()
                if not release.wait(3):
                    raise RuntimeError('cancel send did not resume')

        def cancel():
            try:
                handle.cancel()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=cancel)
        try:
            with mock.patch.object(ipc, '_send', side_effect=delayed_send):
                thread.start()
                self.assertTrue(sent.wait(3))
                result = handle.wait(3)
                release.set()
                thread.join(3)
            self.assertEqual(result['status'], 'cancelled')
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertIs(handle.cancel_ack_observation.accepted, True)
            self.assertFalse(handle.cancel_ack_observation_failed)
        finally:
            release.set()
            thread.join(3)

    def test_client_timeout_without_ack_does_not_imply_a_broker_decision(self):
        from looking_glass.ipc import IPCHandle
        self.assertIn('observe_cancel_ack', inspect.signature(IPCHandle).parameters,
                      'the opt-in acknowledgement observation is not implemented')
        client, peer = socket.socketpair()
        self.addCleanup(peer.close)
        client.setblocking(False)
        handle = IPCHandle(client, 'job', 'team', 'a' * 32, 1, 1, 4096, observe_cancel_ack=True)
        self.addCleanup(handle.close)
        handle.cancel()
        with self.assertRaises(TimeoutError):
            handle.wait(0.05)
        self.assertIsNone(handle.cancel_ack_observation)
        self.assertTrue(handle._closed)


if __name__ == '__main__':
    unittest.main()
