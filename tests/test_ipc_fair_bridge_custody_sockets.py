"""Loopback regressions for fair admission and recipient custody."""

import threading
from unittest.mock import patch

import torch

from looking_glass.ipc import IPCClient
from test_ipc_fair_bridge import eventually, make_stack


def test_waiting_team_can_cancel_while_other_team_decodes():
    worker, broker, adapter, bridge = make_stack()
    entered = threading.Event()
    resume = threading.Event()
    original_tensor = torch.tensor

    def gate_decoder(*args, **kwargs):
        if threading.current_thread().name == 'fair-ipc-dispatch' and not entered.is_set():
            entered.set()
            if not resume.wait(12):
                raise TimeoutError('test did not release the decoder')
        return original_tensor(*args, **kwargs)

    bridge.start()
    server = bridge.make_server(io_timeout=3, job_timeout=20).start()
    try:
        with patch('torch.tensor', side_effect=gate_decoder):
            team_a = IPCClient(server.address, server.team_tokens['A'], io_timeout=3).submit(
                team_id='A', request_id='claimed-a', inputs={'input_ids': [[5]]})
            assert entered.wait(3)
            with bridge._condition:
                record_a = next(record for record in bridge._records.values() if record.team == 'A')
                ticket_a = record_a.ticket
            with adapter._lock:
                assert ticket_a.sequence not in adapter._owners  # Already claimed.
            assert adapter.recover_receipt(ticket_a) is None  # Decoder has not published a handle.
            assert adapter._target.committed_bytes >= 2048

            team_b = IPCClient(server.address, server.team_tokens['B'], io_timeout=3).submit(
                team_id='B', request_id='waiting-b', inputs={'input_ids': [[6]]})
            with bridge._condition:
                record_b = next(record for record in bridge._records.values() if record.team == 'B')
                ticket_b = record_b.ticket
            with adapter._lock:
                assert ticket_b.sequence in adapter._owners
            assert worker.seen == []
            team_b.cancel()
            result_b = team_b.wait(6)
            assert result_b['status'] == 'cancelled'
            assert team_b.cancel_acknowledged
            with adapter._lock:
                assert ticket_b.sequence not in adapter._owners
            assert not resume.is_set()
            # The blocked dispatcher has not yet retired either connection.
            assert bridge.committed_bytes == 4_000_000

            resume.set()
            result_a = team_a.wait(6)
            assert result_a['status'] == 'succeeded'
            assert result_a['result']['logits'] == [5.0]
            assert worker.seen == [5]  # Team B was canceled before execution.
        eventually(lambda: bridge.committed_bytes == 0)
        assert not bridge.quarantined
    finally:
        resume.set()
        worker.release.set()
        assert server.close(timeout=3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_handler_exit_keeps_custody_until_exact_target_retirement_and_ack():
    _, broker, adapter, bridge = make_stack()
    target = adapter._target
    entered = threading.Event()
    resume = threading.Event()
    acknowledgements = []
    original_retire = broker._retire_job_result
    original_acknowledge = adapter.acknowledge_receipt

    def gate_retirement(job):
        entered.set()
        if not resume.wait(12):
            raise TimeoutError('test did not release target retirement')
        return original_retire(job)

    def observe_acknowledgement(ticket, handle):
        assert handle.wait_for_retirement(timeout=0)
        with bridge._condition:
            record = next(record for record in bridge._records.values() if record.ticket == ticket)
            assert record.handler_closed and not record.handler_thread.is_alive()
        acknowledgements.append((ticket.sequence, id(handle)))
        return original_acknowledge(ticket, handle)

    def handler_exited():
        with bridge._condition:
            if len(bridge._records) != 1:
                return False
            record = next(iter(bridge._records.values()))
            return record.handler_closed and not record.handler_thread.is_alive()

    bridge.start()
    server = bridge.make_server(io_timeout=3, job_timeout=10).start()
    try:
        with patch.object(broker, '_retire_job_result', side_effect=gate_retirement), \
                patch.object(adapter, 'acknowledge_receipt', side_effect=observe_acknowledgement):
            client_handle = IPCClient(server.address, server.team_tokens['A'], io_timeout=3).submit(
                team_id='A', request_id='retirement', inputs={'input_ids': [[7]]})
            result = client_handle.wait(4)
            assert result['status'] == 'succeeded'
            del result, client_handle
            assert entered.wait(3)
            eventually(handler_exited)
            with bridge._condition:
                ticket = next(iter(bridge._records.values())).ticket
            target_handle = adapter.recover_receipt(ticket)
            assert target_handle is not None
            expected = ticket.sequence, id(target_handle)
            del target_handle
            assert acknowledgements == []
            assert bridge.committed_bytes == 2_000_000
            assert target.receipt_committed_bytes >= 128

            resume.set()
            eventually(lambda: bridge.committed_bytes == 0 and target.receipt_committed_bytes == 0)
            assert acknowledgements == [expected]
            assert not bridge.quarantined
    finally:
        resume.set()
        assert server.close(timeout=3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)
