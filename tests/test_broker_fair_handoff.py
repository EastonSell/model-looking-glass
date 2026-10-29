import threading
import unittest
import gc
import linecache
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch

from looking_glass.broker import Broker, BrokerShutdownError, BrokerUnavailable, RequestState
from looking_glass.fair_adapter import (
    ExperimentalFairAdapter, FatalHandoffFailure, HandoffFailure, WaitingTicket,
)
from looking_glass.resident_worker import ForwardRequest, ForwardResult


class Cancelled(Exception):
    pass


class Worker:
    cancellation_exception = Cancelled
    max_input_bytes = 8
    max_output_bytes = 8

    def __init__(self):
        self.poisoned = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.closed = False

    def snapshot_request(self, request):
        if type(request) is not ForwardRequest:
            raise ValueError('invalid request')
        return ForwardRequest(request.request_id,
                              {name: value.clone() for name, value in request.inputs.items()},
                              request.seed)

    def execute(self, request, cancel_event):
        self.started.set()
        self.release.wait()
        if cancel_event.is_set():
            raise Cancelled()
        return ForwardResult(request.request_id, request.inputs['input_ids'].float().unsqueeze(-1))

    def close(self):
        self.closed = True


def verify(credential):
    table = {b'alice': 'alice', b'bob': 'bob'}
    if credential not in table:
        raise PermissionError('unauthorized')
    return table[credential]


def decode(payload):
    return ForwardRequest(payload.decode('ascii'), {'input_ids': torch.tensor([[payload[0]]])})


def transfer_handle(adapter):
    receipt = adapter.transfer_next()
    if receipt is None:
        raise AssertionError('expected an admitted receipt')
    handle = adapter.recover_receipt(receipt.ticket)
    if handle is None:
        raise AssertionError('expected a recoverable handle')
    return handle


class FairBrokerHandoffTests(unittest.TestCase):
    def make_adapter(self, broker, *, decoder=decode, max_bytes=8, preparation_bytes=64,
                     total_preparation_bytes=128, max_unacknowledged=8):
        target = broker.enable_fair_handoff(
            ('A', 'B'), decoder=decoder, max_item_bytes=max_bytes,
            max_decoded_bytes=8,
            preparation_bytes=preparation_bytes,
            max_preparation_bytes=total_preparation_bytes,
            max_result_bytes=16, max_unacknowledged=max_unacknowledged,
        )
        adapter = ExperimentalFairAdapter(
            {'alice': 'A', 'bob': 'B'}, verifier=verify, target=target,
            max_waiters=4, max_waiters_per_team=3,
            max_waiting_bytes=32, max_item_bytes=max_bytes,
        )
        return adapter, target

    def test_real_broker_claims_and_executes_one_waiter(self):
        worker = Worker()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            adapter, _ = self.make_adapter(broker)
            ticket = adapter.submit(b'alice', b'hello')
            receipt = adapter.transfer_next()
            self.assertEqual(receipt.ticket, ticket)
            handle = adapter.recover_receipt(receipt.ticket)
            self.assertEqual(handle.team_id, 'A')
            outcome = handle.wait(2)
            self.assertEqual(outcome.state, RequestState.SUCCEEDED)
            self.assertTrue(torch.equal(outcome.result.logits, torch.tensor([[[104.]]])))

    def test_cpu_resident_worker_runs_a_bounded_decoded_forward(self):
        import torch
        from torch import nn
        from looking_glass.resident_worker import ForwardRequest, ResidentWorker

        class Model(nn.Module):
            def forward(self, input_ids):
                return {'logits': input_ids.float().unsqueeze(-1)}

        def decoder(payload):
            return ForwardRequest('input', {'input_ids': torch.tensor([[payload[0]]])})

        worker = ResidentWorker('test-model', Model().eval(), max_batch_size=1,
                                max_sequence_length=1, max_input_bytes=8,
                                max_output_bytes=8)
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker, decoder=decoder)
            adapter.submit(b'alice', b'\x03')
            outcome = transfer_handle(adapter).wait(2)
            self.assertEqual(outcome.state, RequestState.SUCCEEDED)
            self.assertTrue(torch.equal(outcome.result.logits, torch.tensor([[[3.]]])))
            self.assertTrue(target.wait_for_capacity(timeout=2))

    def test_cpu_budgeted_worker_runs_through_the_fair_handoff(self):
        from torch import nn
        from looking_glass.budgeted_worker import BudgetedWorker, DeviceAdmission, WorkloadProfile
        from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
        from looking_glass.resident_worker import ResidentWorker

        class Model(nn.Module):
            def forward(self, input_ids):
                return {'logits': input_ids.float().unsqueeze(-1)}

        model_id = 'a' * 64
        worker = ResidentWorker(model_id, Model().eval(), max_batch_size=1,
                                max_sequence_length=1, max_input_bytes=8,
                                max_output_bytes=8)
        budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        admission = DeviceAdmission(budget, device='cpu', max_estimated_incremental_bytes=1024)
        profile = WorkloadProfile('cpu-fair', model_id, 'cpu', (1, 1),
                                  (('input_ids', 'torch.int64'),), 8, 16)
        wrapper = BudgetedWorker(worker, admission, (profile,))
        with Broker(wrapper, max_pending=2, max_pending_per_team=2) as broker:
            adapter, _ = self.make_adapter(broker)
            adapter.submit(b'alice', b'a')
            handle = transfer_handle(adapter)
            outcome = handle.wait(2)
            self.assertEqual(outcome.state, RequestState.SUCCEEDED)
            self.assertTrue(torch.equal(outcome.result.logits, torch.tensor([[[97.]]])))

    def test_rejected_or_identity_changed_worker_snapshot_is_quarantined(self):
        class TracedSnapshot(ForwardRequest):
            pass

        for mode in ('subclass', 'identity_changed'):
            with self.subTest(mode=mode):
                class BadWorker(Worker):
                    def snapshot_request(self, request):
                        if mode == 'subclass':
                            snapshot = TracedSnapshot(request.request_id, request.inputs, request.seed)
                        else:
                            snapshot = ForwardRequest('changed', request.inputs, request.seed)
                        self.snapshot_ref = weakref.ref(snapshot)
                        return snapshot

                worker = BadWorker()
                broker = Broker(worker, max_pending=2, max_pending_per_team=2)
                adapter, target = self.make_adapter(broker)
                ticket = adapter.submit(b'alice', b'one')
                with self.assertRaises(HandoffFailure) as caught:
                    adapter.transfer_next()
                self.assertEqual(caught.exception.ticket, ticket)
                self.assertTrue(broker.quarantined)
                self.assertEqual(target.committed_bytes, 64)
                self.assertEqual(target.receipt_committed_bytes, 16)
                self.assertIsNotNone(worker.snapshot_ref())
                with self.assertRaises(BrokerShutdownError):
                    broker.close(timeout=3)

    def test_rejected_decoder_output_is_retained_and_quarantined(self):
        class UnknownRequest(ForwardRequest):
            pass

        refs = []

        def decoder(payload):
            result = UnknownRequest('unsafe', {'input_ids': torch.tensor([[payload[0]]])})
            refs.append(weakref.ref(result))
            return result

        broker = Broker(Worker(), max_pending=2, max_pending_per_team=2)
        adapter, target = self.make_adapter(broker, decoder=decoder)
        ticket = adapter.submit(b'alice', b'one')
        with self.assertRaises(HandoffFailure) as caught:
            adapter.transfer_next()
        self.assertEqual(caught.exception.ticket, ticket)
        self.assertTrue(broker.quarantined)
        self.assertIsNotNone(refs[0]())
        self.assertEqual(target.committed_bytes, 64)
        self.assertEqual(target.receipt_committed_bytes, 16)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=3)

    def test_poisoned_preparation_retains_job_and_claims_through_close(self):
        from looking_glass.resident_worker import WorkerOutOfMemory

        refs = []

        class PoisonWorker(Worker):
            def snapshot_request(self, request):
                refs.append(weakref.ref(request.inputs['input_ids']))
                self.poisoned = True
                raise WorkerOutOfMemory('snapshot OOM')

        worker = PoisonWorker()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        adapter, target = self.make_adapter(broker)
        ticket = adapter.submit(b'alice', b'one')
        with self.assertRaises(HandoffFailure) as caught:
            adapter.transfer_next()
        self.assertEqual(caught.exception.ticket, ticket)
        self.assertTrue(broker.quarantined)
        self.assertEqual((target.committed_bytes, target.receipt_committed_bytes), (64, 16))
        self.assertIsNotNone(refs[0]())
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=3)
        self.assertFalse(worker.closed)

    def test_overlapping_poisoned_preparations_each_retain_their_own_error(self):
        from looking_glass.resident_worker import WorkerOutOfMemory

        entered = {name: threading.Event() for name in ('one', 'two')}
        proceed = {name: threading.Event() for name in ('one', 'two')}
        refs, aliases, statuses = [], [], []

        class Owner:
            pass

        class PoisonWorker(Worker):
            def snapshot_request(self, request):
                name = request.request_id
                entered[name].set()
                if not proceed[name].wait(5):
                    raise TimeoutError('test did not release snapshot')
                if name == 'one':
                    return ForwardRequest('changed', request.inputs, request.seed)
                original = torch.ones(1)
                original.owner = Owner()
                refs.append(weakref.ref(original.owner))
                aliases.append(original.detach())
                self.poisoned = True
                raise WorkerOutOfMemory('second snapshot OOM')

        worker = PoisonWorker()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        adapter, target = self.make_adapter(broker)
        adapter.submit(b'alice', b'one')
        adapter.submit(b'alice', b'two')

        def dispatch():
            try:
                adapter.transfer_next()
                statuses.append('unexpected success')
            except HandoffFailure:
                statuses.append('failed')
            except BaseException as exc:
                statuses.append(type(exc).__name__)

        first = threading.Thread(target=dispatch)
        second = threading.Thread(target=dispatch)
        try:
            first.start()
            self.assertTrue(entered['one'].wait(3))
            second.start()
            self.assertTrue(entered['two'].wait(3))
            proceed['one'].set()
            first.join(3)
            self.assertFalse(first.is_alive())
            self.assertTrue(broker.quarantined)
            proceed['two'].set()
            second.join(3)
            self.assertFalse(second.is_alive())
            self.assertEqual(statuses, ['failed', 'failed'])
            self.assertEqual((target.committed_bytes, target.receipt_committed_bytes), (128, 32))
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=3)
            broker_ref = weakref.ref(broker)
            del broker, target, adapter, worker
            gc.collect()
            self.assertIsNotNone(broker_ref())
            self.assertIsNotNone(refs[0]())
            self.assertEqual(len(aliases), 1)
        finally:
            for event in proceed.values():
                event.set()
            if first.ident is not None:
                first.join(3)
            if second.ident is not None:
                second.join(3)

    def test_multiple_keyboard_interrupts_preserve_their_interruption(self):
        self.assert_concurrent_status_interruptions(KeyboardInterrupt)

    def test_multiple_system_exits_preserve_their_interruption(self):
        self.assert_concurrent_status_interruptions(SystemExit)

    def assert_concurrent_status_interruptions(self, interruption):
        barrier = threading.Barrier(3)
        local = threading.local()
        statuses = []

        class UnknownRequest(ForwardRequest):
            pass

        class StatusWorker(Worker):
            @property
            def poisoned(self):
                if getattr(local, 'fail_status', False):
                    raise interruption('status interrupted')
                return self._poisoned

            @poisoned.setter
            def poisoned(self, value):
                self._poisoned = value

        def bad_decoder(payload):
            barrier.wait(timeout=5)
            local.fail_status = True
            return UnknownRequest(payload.decode('ascii'), {'input_ids': torch.tensor([[1]])})

        broker = Broker(StatusWorker(), max_pending=3, max_pending_per_team=3)
        adapter, target = self.make_adapter(
            broker, decoder=bad_decoder, total_preparation_bytes=192)
        for payload in (b'one', b'two', b'three'):
            adapter.submit(b'alice', payload)

        def dispatch():
            try:
                adapter.transfer_next()
                statuses.append('unexpected success')
            except BaseException as exc:
                statuses.append((type(exc), type(exc.__cause__)))

        threads = [threading.Thread(target=dispatch) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(6)
            self.assertFalse(thread.is_alive())
        self.assertEqual(statuses, [(FatalHandoffFailure, interruption)] * 3)
        self.assertTrue(broker.quarantined)
        self.assertEqual((target.committed_bytes, target.receipt_committed_bytes), (192, 48))
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=3)

    def test_poisoned_running_job_and_input_remain_process_rooted(self):
        from looking_glass.resident_worker import WorkerOutOfMemory

        refs = []

        class PoisonWorker(Worker):
            def snapshot_request(self, request):
                snapshot = super().snapshot_request(request)
                refs.append(weakref.ref(snapshot.inputs['input_ids']))
                return snapshot

            def execute(self, request, cancel_event):
                self.poisoned = True
                raise WorkerOutOfMemory('execution OOM')

        worker = PoisonWorker()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        adapter, target = self.make_adapter(broker)
        adapter.submit(b'alice', b'one')
        handle = transfer_handle(adapter)
        self.assertEqual(handle.wait(2).state, RequestState.FAILED)
        self.assertTrue(broker.quarantined)
        self.assertIsNotNone(broker._fair_running_job)
        self.assertEqual((target.committed_bytes, target.receipt_committed_bytes), (64, 16))
        broker_ref, target_ref = weakref.ref(broker), weakref.ref(target)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=3)
        del broker, target, adapter, worker, handle
        gc.collect()
        self.assertIsNotNone(broker_ref())
        self.assertIsNotNone(target_ref())
        self.assertIsNotNone(refs[0]())

    def test_quarantined_inflight_snapshot_is_rooted_until_process_exit(self):
        refs = []

        class ObservedWorker(Worker):
            def snapshot_request(self, request):
                snapshot = super().snapshot_request(request)
                refs.append(weakref.ref(snapshot.inputs['input_ids']))
                return snapshot

        worker = ObservedWorker()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        adapter, target = self.make_adapter(broker)
        ticket = adapter.submit(b'alice', b'one')

        def trace(frame, event, arg):
            if (event == 'line' and frame.f_code.co_name == '_accept_next' and
                    linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip() ==
                    'publication_failure = None'):
                raise KeyboardInterrupt()
            return trace

        try:
            sys.settrace(trace)
            try:
                adapter.transfer_next()
            except FatalHandoffFailure as exc:
                self.assertEqual(exc.ticket, ticket)
        finally:
            sys.settrace(None)
        self.assertTrue(broker.quarantined)
        broker_ref, target_ref = weakref.ref(broker), weakref.ref(target)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=3)
        del adapter, broker, target, worker, trace
        gc.collect()
        self.assertIsNotNone(broker_ref())
        self.assertIsNotNone(target_ref())
        self.assertIsNotNone(refs[0]())

    def test_fair_mode_disables_direct_submission(self):
        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            self.make_adapter(broker)
            with self.assertRaises(BrokerUnavailable):
                broker.submit('A', decode(b'x'))

    def test_rejects_underdeclared_input_commitments_without_changing_mode(self):
        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            with self.assertRaises(ValueError):
                broker.enable_fair_handoff(
                    ('A', 'B'), decoder=decode, max_item_bytes=8,
                    max_decoded_bytes=8, preparation_bytes=55,
                    max_preparation_bytes=128, max_result_bytes=16, max_unacknowledged=8,
                )
            handle = broker.submit('A', decode(b'x'))
            self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)

    def test_rejects_result_commitment_below_worker_output_limit(self):
        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            with self.assertRaises(ValueError):
                broker.enable_fair_handoff(
                    ('A', 'B'), decoder=decode, max_item_bytes=8, max_decoded_bytes=8,
                    preparation_bytes=64, max_preparation_bytes=128,
                    max_result_bytes=7, max_unacknowledged=8,
                )
            self.assertEqual(broker.submit('A', decode(b'x')).wait(2).state,
                             RequestState.SUCCEEDED)

    def test_capacity_check_is_false_for_poisoned_worker(self):
        worker = Worker()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            _, target = self.make_adapter(broker)
            worker.poisoned = True
            self.assertFalse(target.wait_for_capacity(timeout=0))

    def test_target_cannot_be_shared_by_two_adapters(self):
        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            _, target = self.make_adapter(broker)
            with self.assertRaises(BrokerUnavailable):
                ExperimentalFairAdapter(
                    {'alice': 'A', 'bob': 'B'}, verifier=verify, target=target,
                    max_waiters=4, max_waiters_per_team=3,
                    max_waiting_bytes=32, max_item_bytes=8,
                )

    def test_unacknowledged_completed_receipt_is_recoverable_and_bounded(self):
        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker, max_unacknowledged=1)
            first_ticket = adapter.submit(b'alice', b'one')
            receipt = adapter.transfer_next()
            handle = adapter.recover_receipt(receipt.ticket)
            self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)
            self.assertEqual(target.receipt_committed_bytes, 16)
            second = adapter.submit(b'bob', b'two')
            self.assertIsNone(adapter.transfer_next())
            self.assertIs(adapter.recover_receipt(first_ticket), handle)
            foreign = WaitingTicket('another-adapter', first_ticket.sequence)
            self.assertIsNone(adapter.recover_receipt(foreign))
            self.assertFalse(adapter.acknowledge_receipt(foreign, handle))
            with self.assertRaises(ValueError):
                adapter.acknowledge_receipt(first_ticket, object())
            self.assertTrue(adapter.acknowledge_receipt(first_ticket, handle))
            self.assertFalse(adapter.acknowledge_receipt(first_ticket, handle))
            self.assertTrue(target.wait_for_capacity(timeout=2))
            self.assertEqual(target.receipt_committed_bytes, 0)
            self.assertEqual(adapter.transfer_next().ticket, second)

    def test_interrupt_after_publication_retains_recoverable_running_handle(self):
        worker = Worker()
        worker.release.clear()
        errors = []
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker)
            ticket = adapter.submit(b'alice', b'one')

            def trace(frame, event, arg):
                if (event == 'line' and frame.f_code.co_name == '_accept_next' and
                        linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip() ==
                        'if publication_failure is not None:'):
                    raise KeyboardInterrupt()
                return trace

            def transfer():
                sys.settrace(trace)
                try:
                    adapter.transfer_next()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    sys.settrace(None)

            thread = threading.Thread(target=transfer, daemon=True)
            thread.start()
            try:
                thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], FatalHandoffFailure)
                self.assertEqual(errors[0].ticket, ticket)
                self.assertTrue(worker.started.wait(2))
                handle = adapter.recover_receipt(ticket)
                self.assertIsNotNone(handle)
                self.assertTrue(handle.cancel())
                self.assertEqual(target.receipt_committed_bytes, 16)
            finally:
                worker.release.set()
            self.assertEqual(handle.wait(2).state, RequestState.CANCELLED)
            self.assertTrue(adapter.acknowledge_receipt(ticket, handle))
            with broker._condition:
                self.assertTrue(broker._condition.wait_for(lambda: target.receipt_committed_bytes == 0, 2))
            self.assertEqual(target.receipt_committed_bytes, 0)

    def test_retained_handoff_exceptions_do_not_pin_acknowledged_results(self):
        for mode in ('post_publication_interrupt', 'receipt_allocation_failure'):
            with self.subTest(mode=mode):
                refs = []

                class ObservedWorker(Worker):
                    def execute(self, request, cancel_event):
                        result = super().execute(request, cancel_event)
                        refs.append(weakref.ref(result.logits))
                        return result

                with Broker(ObservedWorker(), max_pending=2, max_pending_per_team=2) as broker:
                    adapter, target = self.make_adapter(broker, max_unacknowledged=1)
                    ticket = adapter.submit(b'alice', b'one')
                    error = None
                    if mode == 'receipt_allocation_failure':
                        with patch('looking_glass.fair_adapter.TransferReceipt', side_effect=MemoryError):
                            try:
                                adapter.transfer_next()
                            except HandoffFailure as exc:
                                error = exc
                    else:
                        def trace(frame, event, arg):
                            if (event == 'line' and frame.f_code.co_name == '_accept_next' and
                                    linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip() ==
                                    'if publication_failure is not None:'):
                                raise KeyboardInterrupt()
                            return trace

                        try:
                            sys.settrace(trace)
                            try:
                                adapter.transfer_next()
                            except FatalHandoffFailure as exc:
                                error = exc
                        finally:
                            sys.settrace(None)
                    self.assertIsNotNone(error)
                    self.assertIsNotNone(error.__traceback__)  # Keep the real traceback alive.
                    handle = adapter.recover_receipt(ticket)
                    handle_ref = weakref.ref(handle)
                    self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)
                    self.assertTrue(adapter.acknowledge_receipt(ticket, handle))
                    with broker._condition:
                        self.assertTrue(broker._condition.wait_for(lambda: target.receipt_committed_bytes == 0, 2))
                    del handle
                    gc.collect()
                    self.assertIsNone(handle_ref())
                    self.assertIsNone(refs[0]())
                    self.assertEqual(error.ticket, ticket)

    def test_interrupt_during_decode_reports_ticket_without_published_handle(self):
        def interrupt(_payload):
            raise KeyboardInterrupt()

        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker, decoder=interrupt)
            ticket = adapter.submit(b'alice', b'one')
            with self.assertRaises(FatalHandoffFailure) as caught:
                adapter.transfer_next()
            self.assertEqual(caught.exception.ticket, ticket)
            self.assertIsNone(adapter.recover_receipt(ticket))
            self.assertEqual(target.committed_bytes, 0)
            self.assertEqual(target.receipt_committed_bytes, 0)

    def test_interrupt_at_unprotected_stage_boundaries_fails_closed(self):
        for source_line in ('if request is None:', 'publication_failure = None'):
            with self.subTest(source_line=source_line):
                broker = Broker(Worker(), max_pending=2, max_pending_per_team=2)
                adapter, target = self.make_adapter(broker)
                ticket = adapter.submit(b'alice', b'one')

                def trace(frame, event, arg):
                    if (event == 'line' and frame.f_code.co_name == '_accept_next' and
                            linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip() == source_line and
                            not broker._condition._is_owned()):
                        raise KeyboardInterrupt()
                    return trace

                try:
                    sys.settrace(trace)
                    with self.assertRaises(FatalHandoffFailure) as caught:
                        adapter.transfer_next()
                finally:
                    sys.settrace(None)
                self.assertEqual(caught.exception.ticket, ticket)
                self.assertTrue(broker.failed)
                self.assertEqual(broker._pending_slots, 1)
                self.assertEqual(broker._fair_preparations, set())
                self.assertEqual(target.committed_bytes, 64)
                self.assertEqual(target.receipt_committed_bytes, 16)
                self.assertIsNone(adapter.recover_receipt(ticket))
                self.assertEqual(len(target._inflight), 1)
                rooted = next(iter(target._inflight.values()))
                if source_line == 'if request is None:':
                    self.assertIsNotNone(rooted.staged_raw)
                else:
                    self.assertIsNotNone(rooted.request)
                with self.assertRaises(BrokerShutdownError):
                    broker.close(timeout=3)

    def test_interrupted_queue_claim_requires_owner_reconciliation(self):
        broker = Broker(Worker(), max_pending=2, max_pending_per_team=2)
        adapter, target = self.make_adapter(broker)
        ticket = adapter.submit(b'alice', b'one')
        original = adapter._queue.claim_next

        def remove_then_interrupt(eligible):
            original(eligible)
            raise KeyboardInterrupt()

        with patch.object(adapter._queue, 'claim_next', side_effect=remove_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                adapter.transfer_next()
        self.assertTrue(broker.failed)
        self.assertEqual(adapter.close(), (ticket,))
        self.assertEqual(target.committed_bytes, 64)
        self.assertEqual(target.receipt_committed_bytes, 16)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=3)

    def test_completed_receipt_survives_broker_close_until_owner_acknowledges(self):
        broker = Broker(Worker(), max_pending=2, max_pending_per_team=2)
        adapter, target = self.make_adapter(broker)
        ticket = adapter.submit(b'alice', b'one')
        handle = transfer_handle(adapter)
        self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)
        self.assertTrue(broker.close(timeout=3))
        self.assertIs(adapter.recover_receipt(ticket), handle)
        self.assertEqual(target.receipt_committed_bytes, 16)
        self.assertTrue(adapter.acknowledge_receipt(ticket, handle))
        self.assertEqual(target.receipt_committed_bytes, 0)

    def test_failed_team_binding_does_not_claim_target(self):
        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            target = broker.enable_fair_handoff(
                ('A', 'B'), decoder=decode, max_item_bytes=8, max_decoded_bytes=8,
                preparation_bytes=64, max_preparation_bytes=128,
                max_result_bytes=16, max_unacknowledged=8,
            )
            self.assertFalse(target.wait_for_capacity(timeout=0))
            options = dict(verifier=verify, target=target, max_waiters=4,
                           max_waiters_per_team=3, max_waiting_bytes=32, max_item_bytes=8)
            with self.assertRaises(ValueError):
                ExperimentalFairAdapter({'alice': 'A', 'bob': 'C'}, **options)
            adapter = ExperimentalFairAdapter({'alice': 'A', 'bob': 'B'}, **options)
            self.assertTrue(target.wait_for_capacity(timeout=0))
            adapter.submit(b'alice', b'good')
            self.assertEqual(transfer_handle(adapter).wait(2).state,
                             RequestState.SUCCEEDED)

    def test_mode_switch_and_direct_submit_race_cannot_bypass_admission(self):
        for _ in range(20):
            worker = Worker()
            worker.release.clear()
            barrier = threading.Barrier(2)
            with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
                def submit():
                    barrier.wait()
                    try:
                        return broker.submit('A', decode(b'x'))
                    except BrokerUnavailable:
                        return None

                def enable():
                    barrier.wait()
                    try:
                        return broker.enable_fair_handoff(
                            ('A', 'B'), decoder=decode, max_item_bytes=8, max_decoded_bytes=8,
                            preparation_bytes=64, max_preparation_bytes=128,
                            max_result_bytes=16, max_unacknowledged=8,
                        )
                    except BrokerUnavailable:
                        return None

                try:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        direct = pool.submit(submit)
                        fair = pool.submit(enable)
                        direct, fair = direct.result(3), fair.result(3)
                    self.assertNotEqual(direct is None, fair is None)
                finally:
                    worker.release.set()
                if direct is not None:
                    self.assertEqual(direct.wait(2).state, RequestState.SUCCEEDED)

    def test_adapter_cannot_exceed_the_targets_byte_or_team_limits(self):
        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            _, target = self.make_adapter(broker)
            with self.assertRaises(ValueError):
                ExperimentalFairAdapter(
                    {'alice': 'A', 'bob': 'B'}, verifier=verify, target=target,
                    max_waiters=3, max_waiters_per_team=2,
                    max_waiting_bytes=32, max_item_bytes=9,
                )
            with self.assertRaises(ValueError):
                ExperimentalFairAdapter(
                    {'alice': 'A', 'bob': 'unregistered'}, verifier=verify, target=target,
                    max_waiters=3, max_waiters_per_team=2,
                    max_waiting_bytes=32, max_item_bytes=8,
                )

    def test_decoder_failure_reports_ticket_and_releases_reservation(self):
        def decoder(payload):
            if payload == b'bad':
                raise ValueError('bad input')
            return decode(payload)

        with Broker(Worker(), max_pending=1, max_pending_per_team=1) as broker:
            adapter, _ = self.make_adapter(broker, decoder=decoder)
            failed = adapter.submit(b'alice', b'bad')
            with self.assertRaises(HandoffFailure) as caught:
                adapter.transfer_next()
            self.assertEqual(caught.exception.ticket, failed)
            adapter.submit(b'bob', b'good')
            self.assertEqual(transfer_handle(adapter).wait(2).state,
                             RequestState.SUCCEEDED)

    def test_running_request_keeps_custody_reservation_until_completion(self):
        worker = Worker()
        worker.release.clear()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker, total_preparation_bytes=64)
            try:
                adapter.submit(b'alice', b'one')
                first = transfer_handle(adapter)
                self.assertTrue(worker.started.wait(2))
                second = adapter.submit(b'bob', b'two')
                self.assertIsNone(adapter.transfer_next())
                self.assertEqual(adapter.waiting_bytes, 3)
                self.assertEqual(target.committed_bytes, 64)
            finally:
                worker.release.set()
            self.assertEqual(first.wait(2).state, RequestState.SUCCEEDED)
            # Terminal status may precede the owner dropping its input references.
            self.assertTrue(target.wait_for_capacity(timeout=2))
            receipt = adapter.transfer_next()
            self.assertEqual(receipt.ticket, second)
            self.assertEqual(adapter.recover_receipt(receipt.ticket).wait(2).state, RequestState.SUCCEEDED)

    def test_handoff_frame_drops_snapshot_before_worker_can_retire_it(self):
        entered, resume = threading.Event(), threading.Event()
        references, receipts, errors = [], [], []

        class ObservedWorker(Worker):
            def snapshot_request(self, request):
                snapshot = super().snapshot_request(request)
                references.append(weakref.ref(snapshot.inputs['input_ids']))
                return snapshot

        worker = ObservedWorker()
        worker.release.clear()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker, total_preparation_bytes=64)
            adapter.submit(b'alice', b'one')

            def trace(frame, event, arg):
                if (event == 'line' and frame.f_code.co_name == '_accept_next' and
                        linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip() ==
                        'if publication_failure is not None:'):
                    entered.set()
                    if not resume.wait(3):
                        raise RuntimeError('test timed out')
                return trace

            def transfer():
                sys.settrace(trace)
                try:
                    receipts.append(adapter.transfer_next())
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    sys.settrace(None)

            thread = threading.Thread(target=transfer, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                worker.release.set()
                self.assertTrue(target.wait_for_capacity(timeout=2))
                self.assertIsNone(references[0]())
            finally:
                worker.release.set()
                resume.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(adapter.recover_receipt(receipts[0].ticket).wait(2).state, RequestState.SUCCEEDED)

    def test_ack_keeps_result_commitment_until_broker_drops_its_aliases(self):
        entered, resume = threading.Event(), threading.Event()
        result_refs = []

        class ObservedWorker(Worker):
            def execute(self, request, cancel_event):
                result = super().execute(request, cancel_event)
                result_refs.append(weakref.ref(result.logits))
                return result

        with Broker(ObservedWorker(), max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker, max_unacknowledged=1)
            original_retire = broker._retire_job_input

            def delayed_retirement(job):
                entered.set()
                if not resume.wait(3):
                    raise RuntimeError('test timed out')
                return original_retire(job)

            with patch.object(broker, '_retire_job_input', side_effect=delayed_retirement):
                ticket = adapter.submit(b'alice', b'one')
                receipt = adapter.transfer_next()
                handle = adapter.recover_receipt(receipt.ticket)
                outcome = handle.wait(2)
                try:
                    self.assertTrue(entered.wait(2))
                    self.assertTrue(adapter.acknowledge_receipt(ticket, handle))
                    del outcome, receipt, handle
                    self.assertIsNotNone(result_refs[0]())
                    self.assertEqual(target.receipt_committed_bytes, 16)
                finally:
                    resume.set()
                self.assertTrue(target.wait_for_capacity(timeout=2))
                self.assertIsNone(result_refs[0]())
                self.assertEqual(target.receipt_committed_bytes, 0)

    def test_running_cancellation_keeps_custody_until_worker_returns(self):
        worker = Worker()
        worker.release.clear()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker, total_preparation_bytes=64)
            try:
                adapter.submit(b'alice', b'one')
                handle = transfer_handle(adapter)
                self.assertTrue(worker.started.wait(2))
                self.assertTrue(handle.cancel())
                self.assertEqual(target.committed_bytes, 64)
                adapter.submit(b'bob', b'two')
                self.assertIsNone(adapter.transfer_next())
            finally:
                worker.release.set()
            self.assertEqual(handle.wait(2).state, RequestState.CANCELLED)
            self.assertTrue(target.wait_for_capacity(timeout=2))
            self.assertEqual(transfer_handle(adapter).wait(2).state,
                             RequestState.SUCCEEDED)

    def test_cancelling_pending_job_releases_its_lifetime_commitment(self):
        worker = Worker()
        worker.release.clear()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker)
            try:
                adapter.submit(b'alice', b'one')
                first = transfer_handle(adapter)
                self.assertTrue(worker.started.wait(2))
                adapter.submit(b'bob', b'two')
                queued = transfer_handle(adapter)
                adapter.submit(b'alice', b'three')
                self.assertIsNone(adapter.transfer_next())
                self.assertEqual(target.committed_bytes, 128)
                self.assertTrue(queued.cancel())
                self.assertEqual(queued.wait(2).state, RequestState.CANCELLED)
                self.assertEqual(target.committed_bytes, 64)
                next_job = transfer_handle(adapter)
            finally:
                worker.release.set()
            self.assertEqual(first.wait(2).state, RequestState.SUCCEEDED)
            self.assertEqual(next_job.wait(2).state, RequestState.SUCCEEDED)

    def test_enabling_requires_an_idle_broker_and_happens_only_once(self):
        worker = Worker()
        worker.release.clear()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            try:
                handle = broker.submit('A', decode(b'x'))
                self.assertTrue(worker.started.wait(2))
                with self.assertRaises(BrokerUnavailable):
                    self.make_adapter(broker)
            finally:
                worker.release.set()
            self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)
            self.make_adapter(broker)
            with self.assertRaises(BrokerUnavailable):
                self.make_adapter(broker)

    def test_busy_decoder_holds_slot_and_budget_without_blocking_cancellation(self):
        entered, resume = threading.Event(), threading.Event()
        results = []
        errors = []

        def slow_decoder(payload):
            entered.set()
            if not resume.wait(3):
                raise RuntimeError('test timed out')
            return decode(payload)

        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker, decoder=slow_decoder,
                                                total_preparation_bytes=64)
            first = adapter.submit(b'alice', b'first')
            second = adapter.submit(b'bob', b'other')

            def transfer():
                try:
                    results.append(adapter.transfer_next())
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=transfer, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(target.committed_bytes, 64)
                self.assertIsNone(adapter.transfer_next())
                self.assertTrue(adapter.cancel(b'bob', second))
            finally:
                resume.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results[0].ticket, first)
            self.assertEqual(adapter.recover_receipt(results[0].ticket).wait(2).state, RequestState.SUCCEEDED)

    def test_empty_claim_reservation_ids_are_not_reused_during_release(self):
        entered, resume = threading.Event(), threading.Event()
        results, errors = [], []
        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            adapter, target = self.make_adapter(broker)
            original_release = target._release
            calls = 0
            calls_lock = threading.Lock()

            def slow_release(custody):
                nonlocal calls
                with calls_lock:
                    calls += 1
                    first = calls == 1
                if first:
                    entered.set()
                    if not resume.wait(3):
                        raise RuntimeError('test timed out')
                return original_release(custody)

            def transfer():
                try:
                    results.append(adapter.transfer_next())
                except BaseException as exc:
                    errors.append(exc)

            with patch.object(target, '_release', side_effect=slow_release):
                thread = threading.Thread(target=transfer, daemon=True)
                thread.start()
                try:
                    self.assertTrue(entered.wait(2))
                    self.assertIsNone(adapter.transfer_next())
                finally:
                    resume.set()
                    thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results, [None])

    def test_waiting_cancel_cannot_cancel_claimed_decode(self):
        entered, resume = threading.Event(), threading.Event()
        results = []

        def slow_decoder(payload):
            entered.set()
            if not resume.wait(3):
                raise RuntimeError('test timed out')
            return decode(payload)

        with Broker(Worker(), max_pending=2, max_pending_per_team=2) as broker:
            adapter, _ = self.make_adapter(broker, decoder=slow_decoder)
            ticket = adapter.submit(b'alice', b'first')
            thread = threading.Thread(target=lambda: results.append(adapter.transfer_next()), daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertFalse(adapter.cancel(b'alice', ticket))
            finally:
                resume.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(results[0].ticket, ticket)
            self.assertEqual(adapter.recover_receipt(results[0].ticket).wait(2).state, RequestState.SUCCEEDED)

    def test_uncertain_queue_publication_retains_the_host_claim(self):
        broker = Broker(Worker(), max_pending=2, max_pending_per_team=2)
        adapter, target = self.make_adapter(broker)
        ticket = adapter.submit(b'alice', b'first')
        original_put = broker._queue.put

        def publish_then_fail(*args, **kwargs):
            original_put(*args, **kwargs)
            raise MemoryError('publication outcome unknown')

        with patch.object(broker._queue, 'put', side_effect=publish_then_fail):
            with self.assertRaises(HandoffFailure) as caught:
                adapter.transfer_next()
        self.assertEqual(caught.exception.ticket, ticket)
        self.assertTrue(broker.failed)
        self.assertEqual(target.committed_bytes, 64)
        self.assertEqual(target.receipt_committed_bytes, 16)
        with self.assertRaises(BrokerUnavailable):
            adapter.recover_receipt(ticket)
        self.assertEqual(len(broker._queue), 1)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=3)
        self.assertEqual(target.committed_bytes, 64)

    def test_fatal_publication_interruptions_preserve_their_kind(self):
        for kind in (KeyboardInterrupt, SystemExit):
            with self.subTest(kind=kind):
                broker = Broker(Worker(), max_pending=2, max_pending_per_team=2)
                adapter, target = self.make_adapter(broker)
                ticket = adapter.submit(b'alice', b'one')
                with patch.object(broker._queue, 'put', side_effect=kind):
                    with self.assertRaises(FatalHandoffFailure) as caught:
                        adapter.transfer_next()
                self.assertEqual(caught.exception.ticket, ticket)
                self.assertTrue(broker.quarantined)
                self.assertEqual(target.committed_bytes, 64)
                self.assertEqual(target.receipt_committed_bytes, 16)
                with self.assertRaises(BrokerShutdownError):
                    broker.close(timeout=3)

    def test_fatal_custody_release_interruptions_are_not_converted(self):
        for kind in (KeyboardInterrupt, SystemExit):
            with self.subTest(kind=kind):
                worker = Worker()
                worker.release.clear()
                broker = Broker(worker, max_pending=2, max_pending_per_team=2)
                adapter, _ = self.make_adapter(broker)
                try:
                    adapter.submit(b'alice', b'first')
                    transfer_handle(adapter)
                    self.assertTrue(worker.started.wait(2))
                    adapter.submit(b'bob', b'second')
                    pending = transfer_handle(adapter)
                    custody = pending._job.input_custody
                    with patch.object(custody._claim, 'release', side_effect=kind):
                        with self.assertRaises(kind):
                            pending.cancel()
                    self.assertTrue(broker.quarantined)
                finally:
                    worker.release.set()
                    with self.assertRaises(BrokerShutdownError):
                        broker.close(timeout=3)
    def test_custody_release_failure_stops_broker_and_retains_other_claims(self):
        worker = Worker()
        worker.release.clear()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        adapter, target = self.make_adapter(broker)
        try:
            adapter.submit(b'alice', b'first')
            first = transfer_handle(adapter)
            self.assertTrue(worker.started.wait(2))
            adapter.submit(b'bob', b'second')
            pending = transfer_handle(adapter)
            custody = pending._job.input_custody
            with patch.object(custody._claim, 'release', side_effect=MemoryError):
                with self.assertRaises(BrokerUnavailable):
                    pending.cancel()
            self.assertTrue(broker.failed)
            self.assertEqual(target.committed_bytes, 128)
        finally:
            worker.release.set()
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=3)
        self.assertEqual(target.committed_bytes, 128)
        self.assertIn(first.state, (RequestState.SUCCEEDED, RequestState.CANCELLED))

    def test_shutdown_during_preparation_fails_the_claimed_ticket(self):
        entered, resume = threading.Event(), threading.Event()
        errors = []

        def slow_decoder(payload):
            entered.set()
            if not resume.wait(3):
                raise RuntimeError('test timed out')
            return decode(payload)

        broker = Broker(Worker(), max_pending=1, max_pending_per_team=1)
        adapter, target = self.make_adapter(broker, decoder=slow_decoder)
        ticket = adapter.submit(b'alice', b'one')

        def transfer():
            try:
                adapter.transfer_next()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=transfer, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertFalse(broker.close(wait=False))
        finally:
            resume.set()
            thread.join(3)
            broker.close(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], HandoffFailure)
        self.assertEqual(errors[0].ticket, ticket)
        self.assertEqual(target.committed_bytes, 0)


if __name__ == '__main__':
    unittest.main()
