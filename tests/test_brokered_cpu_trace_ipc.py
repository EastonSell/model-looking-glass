"""CPU-only composition of the resident Broker and versioned trace IPC."""
import json
import inspect
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

import torch
from looking_glass.cpu_broker_trace_service import BrokeredCpuTraceService, BrokerTraceIncomplete
from looking_glass.cpu_shared_trace_service import SharedTraceError
from looking_glass.host_budget import HostBudget
from looking_glass.trace_ipc_v2 import TraceIPCClient, TraceIPCIncomplete, TraceIPCRejected, TraceIPCServer
from looking_glass.trace_producer import ClaimLimit, ProducerProfile
from looking_glass.trace_store import HostTraceStore
from looking_glass.traced_broker_worker import SampleTrace, SiteTrace, TraceLimits
from looking_glass.traced_worker import TracedGPT2Worker
if __package__:
    from .test_trace_ipc_v2 import _CLIENT_CODE, _reader
else:
    from test_trace_ipc_v2 import _CLIENT_CODE, _reader

_RETAINED = []


class BrokeredTraceIPCTests(unittest.TestCase):
    def make_service(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(71)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=8, n_ctx=8,
                n_embd=8, n_layer=1, n_head=2, use_cache=False,
                resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
        worker = TracedGPT2Worker('broker-trace-tiny', model, device='cpu',
            max_batch_size=1, max_sequence_length=3, max_input_bytes=24, max_output_bytes=384)
        budget = HostBudget(max_bytes=3_000_000, max_items=128, max_requests=32, max_operations=32)
        profile = ProducerProfile(ClaimLimit(48, 8), ClaimLimit(768, 2),
                                  ClaimLimit(64, 2), ClaimLimit(100_000, 1))
        service = BrokeredCpuTraceService(worker=worker, budget=budget,
            store=HostTraceStore(max_tensor_bytes=1024, max_items=4), profile=profile,
            limits=TraceLimits(1, 3, 1, 2, 2, 64, 3), team_ids=('alpha','beta'),
            max_slots=2, max_event_bytes=32)
        server = TraceIPCServer(service, max_event_bytes=32, max_trace_bytes=64, timeout=20).start()
        def cleanup():
            if not server.close(timeout=5) or not service.close(timeout=5):
                _RETAINED.append((server, service, worker, budget))
        self.addCleanup(cleanup)
        return server, service, worker, budget

    def client(self, server, team):
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=8, max_operations=8)
        return (TraceIPCClient(server.address, token=server.team_tokens[team], team_id=team,
            broker_id=server.broker_id, process_generation=server.process_generation,
            budget=budget, max_event_bytes=32, max_trace_bytes=64, timeout=20), budget)

    def test_real_two_process_clients_share_the_broker_without_trace_leakage(self):
        server, service, worker, budget = self.make_service()
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        children = []
        owner_threads = []
        execute_traced = worker.execute_traced
        def observe_owner(*args, **kwargs):
            owner_threads.append(threading.current_thread())
            return execute_traced(*args, **kwargs)
        observer = patch.object(worker, 'execute_traced', new=observe_owner)
        def launch(team, row, position, block):
            child = subprocess.Popen([sys.executable, '-B', '-c', _CLIENT_CODE], cwd=root,
                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1)
            children.append(child)
            messages = queue.Queue()
            threading.Thread(target=_reader, args=(child.stdout, messages), daemon=True).start()
            config = {'address':server.address, 'token':server.team_tokens[team], 'team':team,
                'broker_id':server.broker_id, 'process_generation':server.process_generation,
                'input_ids':row, 'position':position, 'block':block}
            child.stdin.write(json.dumps(config) + '\n')
            child.stdin.flush()
            return child, messages
        try:
            observer.start()
            alpha, aq = launch('alpha', [1,2,3], 0, True)
            self.assertEqual(json.loads(aq.get(timeout=20)), {'kind':'event', 'team':'alpha'})
            beta, bq = launch('beta', [3,2,1], 2, False)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                with service._broker._condition:
                    if service._broker._pending_slots == 1:
                        self.assertEqual(service._broker._running.team_id, 'alpha')
                        break
                time.sleep(.01)
            else:
                self.fail('the second trace did not queue in the resident broker')
            self.assertTrue(bq.empty())
            claims = budget.snapshot().claims
            self.assertEqual(sum(c.nbytes for c in claims if c.stage == 'broker_trace_snapshot'), 48)
            self.assertTrue(any(c.stage == 'callback_exposure' and c.nbytes == 32 for c in claims))
            self.assertIsNotNone(service._attempts['alpha'].snapshot_ref())
            alpha.stdin.write('continue\n')
            alpha.stdin.flush()
            a = json.loads(aq.get(timeout=20))
            self.assertEqual(json.loads(bq.get(timeout=20)), {'kind':'event', 'team':'beta'})
            b = json.loads(bq.get(timeout=20))
            self.assertEqual(service._broker.broker_id, server.broker_id)
            for value, team in ((a, 'alpha'), (b, 'beta')):
                self.assertEqual(value['kind'], 'done')
                done = value['completion']
                self.assertEqual((done['team_id'], done['request_id'], done['event_count'], done['trace_bytes']),
                                 (team, 'same-request', 2, 64))
                self.assertEqual([event[:3] for event in value['events']],
                                 [['attn', 0, team+'-sample'], ['mlp', 1, team+'-sample']])
                self.assertEqual(value['budget_bytes'], 0)
            self.assertNotEqual(a['completion']['job_id'], b['completion']['job_id'])
            self.assertNotEqual(a['events'][0][3], b['events'][0][3])
            self.assertEqual(owner_threads, [service._broker._thread, service._broker._thread])
            with self.assertRaises(SharedTraceError):
                service.client('beta').acknowledge(service._attempts['alpha'].receipt)
            for child in children:
                self.assertEqual(child.wait(timeout=20), 0, child.stderr.read())
            self.assertTrue(server.close(timeout=5))
            self.assertTrue(service.close(timeout=5), (service._failure, service._broker._shutdown_error))
            self.assertEqual(budget.snapshot().used_bytes, 0)
            self.assertIsNone(worker._model)
            self.assertIsNone(service._owner._model)
            self.assertIsNone(service._bound_model)
        finally:
            observer.stop()
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                for stream in (child.stdin, child.stdout, child.stderr):
                    if stream is not None:
                        stream.close()

    def test_oom_keeps_broker_snapshot_and_owner_incomplete(self):
        server, service, worker, budget = self.make_service()
        client, receiver_budget = self.client(server, 'alpha')
        with patch.object(worker, 'execute_traced', side_effect=torch.OutOfMemoryError('injected CPU OOM')):
            with self.assertRaises(TraceIPCIncomplete):
                client.trace(request_id='oom', input_ids=[1,2,3], sample_id='row',
                    positions=[0], sites=[[0,'attn']], on_event=lambda *_: None)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not server.incomplete:
            time.sleep(.01)
        self.assertTrue(client.incomplete and server.incomplete and service.quarantined)
        self.assertIsNotNone(worker._model)
        self.assertTrue(any(c.stage == 'broker_trace_snapshot' and c.nbytes == 24
                            for c in budget.snapshot().claims))
        self.assertFalse(service.close(timeout=1))
        _RETAINED.append((client, receiver_budget))

    def test_trace_byte_limit_is_rejected_before_broker_admission(self):
        server, service, _worker, budget = self.make_service()
        client, receiver_budget = self.client(server, 'alpha')
        with self.assertRaisesRegex(TraceIPCRejected, 'trace_limit'):
            client.trace(request_id='too-large', input_ids=[1,2,3], sample_id='row',
                positions=[0,1], sites=[[0,'attn']], on_event=lambda *_: None)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and server.active_count:
            time.sleep(.01)
        self.assertEqual(service._broker._next_job_id, 0)
        self.assertEqual(service._attempts, {})
        self.assertFalse(service.quarantined or server.incomplete)
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertEqual(receiver_budget.snapshot().used_bytes, 0)

    def test_broker_cancel_during_delivery_leaves_both_clients_incomplete(self):
        server, service, _worker, budget = self.make_service()
        alpha, alpha_budget = self.client(server, 'alpha')
        beta, beta_budget = self.client(server, 'beta')
        entered, release = threading.Event(), threading.Event()
        outcomes = {}
        def callback(*_):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('test did not release the event')
        def run(client, team, sink):
            try:
                outcomes[team] = client.trace(request_id=team, input_ids=[1,2,3],
                    sample_id=team, positions=[0], sites=[[0,'attn']], on_event=sink)
            except BaseException as error:
                outcomes[team] = error
        a = threading.Thread(target=run, args=(alpha,'alpha',callback))
        b = threading.Thread(target=run, args=(beta,'beta',lambda *_: None))
        try:
            a.start()
            self.assertTrue(entered.wait(15))
            b.start()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with service._broker._condition:
                    if service._broker._pending_slots == 1:
                        break
                time.sleep(.01)
            else:
                self.fail('the second client did not reach the broker queue')
            self.assertFalse(service._broker.close(wait=False))
            release.set()
            a.join(15)
            b.join(15)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertIsInstance(outcomes.get('alpha'), TraceIPCIncomplete)
            self.assertIsInstance(outcomes.get('beta'), TraceIPCIncomplete)
            self.assertTrue(alpha.incomplete and beta.incomplete and service.quarantined)
            self.assertTrue(any(c.stage == 'broker_trace_snapshot' for c in budget.snapshot().claims))
            self.assertFalse(service.close(timeout=1))
            _RETAINED.append((alpha, alpha_budget, beta, beta_budget))
        finally:
            release.set()
            if a.ident is not None:
                a.join(15)
            if b.ident is not None:
                b.join(15)

    def test_non_none_callback_result_keeps_trace_custody(self):
        _server, service, _worker, budget = self.make_service()
        request = budget.open_request('caller-input')
        claim = request.reserve('caller', 'input', 24, items=1)
        inputs = {'input_ids':torch.tensor([[1,2,3]], dtype=torch.long)}
        with self.assertRaises(Exception):
            service.client('alpha').submit(request_id='bad-callback', inputs=inputs,
                input_owner=request, input_claim=claim, samples=(SampleTrace('row', (0,)),),
                sites=(SiteTrace(0, 'attn'),), on_event=lambda *_: 'not a synchronous release')
        self.assertTrue(service.quarantined)
        self.assertTrue(any(c.stage == 'callback_exposure' for c in budget.snapshot().claims))
        self.assertTrue(any(c.stage == 'broker_trace_snapshot' for c in budget.snapshot().claims))
        _RETAINED.append((request, claim, inputs))

    def test_owner_callback_reentry_is_rejected_before_broker_admission(self):
        server, service, _worker, budget = self.make_service()
        request = budget.open_request('reentry-input')
        claim = request.reserve('caller', 'input', 24, items=1)
        inputs = {'input_ids':torch.tensor([[1,2,3]], dtype=torch.long)}
        outcomes = {}
        def callback(*_):
            try:
                service.client('beta').submit(request_id='nested', inputs=inputs,
                    input_owner=request, input_claim=claim, samples=(SampleTrace('nested', (0,)),),
                    sites=(SiteTrace(0, 'attn'),), on_event=lambda *_: None)
            except BaseException as error:
                outcomes['nested'] = error
        def run():
            try:
                outcomes['receipt'] = service.client('alpha').submit(
                    request_id='outer', inputs=inputs, input_owner=request, input_claim=claim,
                    samples=(SampleTrace('outer', (0,)),), sites=(SiteTrace(0, 'attn'),),
                    on_event=callback)
            except BaseException as error:
                outcomes['outer_error'] = error
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(8)
        if thread.is_alive():
            service._broker.close(wait=False)
            thread.join(5)
        self.assertFalse(thread.is_alive(), 'nested callback blocked the broker owner')
        self.assertIsInstance(outcomes.get('nested'), SharedTraceError)
        self.assertNotIn('outer_error', outcomes)
        self.assertEqual(set(service._attempts), {'alpha'})
        self.assertEqual(service._broker._next_job_id, 1)
        self.assertFalse(service.quarantined)
        service.client('alpha').acknowledge(outcomes['receipt'])
        outcomes.pop('nested').__traceback__ = None
        inputs = None
        claim.release()
        request.close()
        self.assertTrue(server.close(timeout=5))
        self.assertTrue(service.close(timeout=5))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_failed_delivery_stops_queued_team_and_retains_both_claims(self):
        server, service, _worker, budget = self.make_service()
        alpha, alpha_budget = self.client(server, 'alpha')
        beta, beta_budget = self.client(server, 'beta')
        entered, release = threading.Event(), threading.Event()
        outcomes = {}
        def fail(*_):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('test did not release alpha')
            raise RuntimeError('client did not release the event')
        def run(client, team, sink):
            try:
                outcomes[team] = client.trace(request_id=team, input_ids=[1,2,3],
                    sample_id=team, positions=[0], sites=[[0,'attn']], on_event=sink)
            except BaseException as error:
                outcomes[team] = error
        a = threading.Thread(target=run, args=(alpha,'alpha',fail))
        b = threading.Thread(target=run, args=(beta,'beta',lambda *_: None))
        try:
            a.start()
            self.assertTrue(entered.wait(15))
            b.start()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with service._broker._condition:
                    if service._broker._pending_slots == 1:
                        break
                time.sleep(.01)
            else:
                self.fail('the second client did not reach the broker queue')
            release.set()
            a.join(15)
            b.join(15)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertIsInstance(outcomes.get('alpha'), TraceIPCIncomplete)
            self.assertIsInstance(outcomes.get('beta'), TraceIPCIncomplete)
            self.assertTrue(alpha.incomplete and beta.incomplete and service.quarantined)
            self.assertEqual(service._inner._generation, 1)
            self.assertTrue(any(c.stage == 'trace_ipc_receive' and c.nbytes == 32
                                for c in alpha_budget.snapshot().claims))
            self.assertTrue(any(c.stage == 'callback_exposure' and c.nbytes == 32
                                for c in budget.snapshot().claims))
            self.assertTrue(any(c.stage == 'broker_trace_snapshot' for c in budget.snapshot().claims))
            _RETAINED.append((alpha, alpha_budget, beta, beta_budget))
        finally:
            release.set()
            if a.ident is not None:
                a.join(15)
            if b.ident is not None:
                b.join(15)

    def test_interrupted_registration_keeps_the_attempt_and_faults_owner(self):
        _server, service, _worker, budget = self.make_service()
        request = budget.open_request('registration-input')
        claim = request.reserve('caller', 'input', 24, items=1)
        inputs = {'input_ids':torch.tensor([[1,2,3]], dtype=torch.long)}
        lines, first = inspect.getsourcelines(BrokeredCpuTraceService._submit)
        target = first + next(i for i, line in enumerate(lines) if 'self._tickets[token] = attempt' in line)
        def interrupt(frame, event, _arg):
            if frame.f_code is BrokeredCpuTraceService._submit.__code__ and event == 'line' and frame.f_lineno == target:
                raise KeyboardInterrupt('registration interruption')
            return interrupt
        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                service.client('alpha').submit(request_id='interrupt', inputs=inputs,
                    input_owner=request, input_claim=claim, samples=(SampleTrace('row', (0,)),),
                    sites=(SiteTrace(0, 'attn'),), on_event=lambda *_: None)
        finally:
            sys.settrace(None)
        self.assertTrue(service.quarantined)
        self.assertIsNotNone(service._registering)
        self.assertIs(service._attempts['alpha'], service._registering)
        self.assertIsInstance(service._registering.failure, KeyboardInterrupt)
        self.assertEqual(service._inner._generation, 0)
        self.assertFalse(service.close(timeout=1))
        _RETAINED.append((request, claim, inputs))

    def test_interrupted_receipt_acknowledgment_keeps_snapshot_claim(self):
        _server, service, _worker, budget = self.make_service()
        request = budget.open_request('ack-input')
        claim = request.reserve('caller', 'input', 24, items=1)
        inputs = {'input_ids':torch.tensor([[1,2,3]], dtype=torch.long)}
        client = service.client('alpha')
        receipt = client.submit(request_id='ack', inputs=inputs, input_owner=request,
            input_claim=claim, samples=(SampleTrace('row', (0,)),),
            sites=(SiteTrace(0, 'attn'),), on_event=lambda *_: None)
        method = type(client).acknowledge
        lines, first = inspect.getsourcelines(method)
        target = first + next(i for i, line in enumerate(lines) if 'attempt.acknowledged = True' in line)
        def interrupt(frame, event, _arg):
            if frame.f_code is method.__code__ and event == 'line' and frame.f_lineno == target:
                raise KeyboardInterrupt('acknowledgment interruption')
            return interrupt
        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                client.acknowledge(receipt)
        finally:
            sys.settrace(None)
        self.assertTrue(service.quarantined)
        self.assertIsInstance(service._attempts['alpha'].failure, KeyboardInterrupt)
        self.assertTrue(any(c.stage == 'broker_trace_snapshot' for c in budget.snapshot().claims))
        self.assertFalse(service.close(timeout=1))
        _RETAINED.append((request, claim, inputs, receipt))

    def test_lost_direct_return_cannot_recover_or_acknowledge_after_quarantine(self):
        _server, service, _worker, budget = self.make_service()
        request = budget.open_request('lost-return-input')
        claim = request.reserve('caller', 'input', 24, items=1)
        inputs = {'input_ids':torch.tensor([[1,2,3]], dtype=torch.long)}
        client = service.client('alpha')
        lines, first = inspect.getsourcelines(BrokeredCpuTraceService._submit)
        target = first + next(i for i, line in enumerate(lines) if line.strip() == 'return receipt')
        def interrupt(frame, event, _arg):
            if frame.f_code is BrokeredCpuTraceService._submit.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                raise KeyboardInterrupt('lost direct return')
            return interrupt
        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                client.submit(request_id='lost', inputs=inputs, input_owner=request, input_claim=claim,
                    samples=(SampleTrace('row', (0,)),), sites=(SiteTrace(0, 'attn'),),
                    on_event=lambda *_: None)
        finally:
            sys.settrace(None)
        attempt = service._attempts['alpha']
        self.assertTrue(attempt.completed and service.quarantined)
        self.assertIsNotNone(attempt.receipt)
        with self.assertRaises(SharedTraceError):
            client.recover('lost')
        with self.assertRaises(SharedTraceError):
            client.acknowledge(attempt.receipt)
        self.assertFalse(attempt.acknowledged)
        self.assertTrue(any(c.stage == 'broker_trace_snapshot' for c in budget.snapshot().claims))
        self.assertFalse(service.close(timeout=1))
        _RETAINED.append((request, claim, inputs))

    def test_peer_fault_before_completion_decision_prevents_direct_receipt(self):
        _server, service, _worker, budget = self.make_service()
        request = budget.open_request('peer-fault-input')
        claim = request.reserve('caller', 'input', 24, items=1)
        inputs = {'input_ids':torch.tensor([[1,2,3]], dtype=torch.long)}
        original_submit = service._broker.submit
        class FaultAfterBrokerOutcome:
            def __init__(self, handle):
                self.handle = handle
            def wait(self):
                outcome = self.handle.wait()
                service._fail(service._attempts['alpha'], RuntimeError('peer fault'))
                return outcome
        def submit_with_fault(*args, **kwargs):
            return FaultAfterBrokerOutcome(original_submit(*args, **kwargs))
        with patch.object(service._broker, 'submit', new=submit_with_fault):
            with self.assertRaises(BrokerTraceIncomplete):
                service.client('alpha').submit(request_id='peer-fault', inputs=inputs,
                    input_owner=request, input_claim=claim, samples=(SampleTrace('row', (0,)),),
                    sites=(SiteTrace(0, 'attn'),), on_event=lambda *_: None)
        attempt = service._attempts['alpha']
        self.assertTrue(service.quarantined)
        self.assertIsNotNone(attempt.receipt)
        self.assertFalse(attempt.completed or attempt.acknowledged)
        self.assertTrue(any(c.stage == 'broker_trace_snapshot' for c in budget.snapshot().claims))
        self.assertFalse(service.close(timeout=1))
        _RETAINED.append((request, claim, inputs))

    def test_a_second_brokered_owner_cannot_bind_the_same_model(self):
        _server, service, worker, budget = self.make_service()
        with self.assertRaisesRegex(ValueError, 'already has a brokered trace owner'):
            BrokeredCpuTraceService(worker=worker, budget=budget,
                store=HostTraceStore(max_tensor_bytes=1024, max_items=4),
                profile=service._inner._profile, limits=service._inner._limits,
                team_ids=('alpha','beta'), max_slots=2, max_event_bytes=32)

    def test_poisoned_worker_or_changed_model_blocks_service_retirement(self):
        for mode in ('poisoned', 'model-changed', 'both-models-changed'):
            with self.subTest(mode=mode):
                _server, service, worker, _budget = self.make_service()
                model = worker._model
                if mode == 'poisoned':
                    worker._poison_pending.set()
                else:
                    worker._model = None
                    if mode == 'both-models-changed':
                        service._owner._model = None
                self.assertTrue(service.quarantined)
                self.assertIs(service._bound_model, model)
                self.assertFalse(service.close(timeout=.1))

    def test_model_change_after_close_starts_blocks_worker_retirement(self):
        server, service, worker, _budget = self.make_service()
        model = worker._model
        self.assertTrue(server.close(timeout=5))
        original_close = service._broker.close
        def change_then_close(*args, **kwargs):
            self.assertTrue(service._closing)
            worker._model = None
            return original_close(*args, **kwargs)
        with patch.object(service._broker, 'close', side_effect=change_then_close):
            self.assertFalse(service.close(timeout=5))
        self.assertTrue(service.quarantined)
        self.assertIs(service._bound_model, model)
        self.assertIs(service._owner._model, model)

    def test_clean_empty_service_releases_owner_model(self):
        server, service, worker, budget = self.make_service()
        self.assertTrue(server.close(timeout=5))
        self.assertTrue(service.close(timeout=5))
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertIsNone(worker._model)
        self.assertIsNone(service._owner._model)
        self.assertIsNone(service._bound_model)

    def test_lost_final_ack_blocks_broker_retirement_even_after_local_completion(self):
        server, service, _worker, budget = self.make_service()
        client, receiver_budget = self.client(server, 'alpha')
        from looking_glass import trace_ipc_v2 as module
        original = module._receive
        def lose_final(sock, limit, deadline):
            message = original(sock, limit, deadline)
            if (threading.current_thread().name == 'trace-ipc-client'
                    and message.get('op') == 'closed_received'):
                raise EOFError('simulated lost final acknowledgement')
            return message
        with patch.object(module, '_receive', side_effect=lose_final):
            result = client.trace(request_id='final', input_ids=[1,2,3], sample_id='row',
                positions=[0], sites=[[0,'attn']], on_event=lambda *_: None)
            self.assertEqual((result.event_count, result.trace_bytes), (1,32))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not server.incomplete:
                time.sleep(.01)
            self.assertTrue(server.incomplete)
        self.assertFalse(client.incomplete)
        self.assertEqual(receiver_budget.snapshot().used_bytes, 0)
        self.assertTrue(service.quarantined)
        self.assertFalse(service.close(timeout=1))
        self.assertTrue(any(c.stage == 'broker_trace_snapshot' for c in budget.snapshot().claims))

    def test_close_timeout_respects_an_occupied_service_or_server_lock(self):
        server, service, _worker, _budget = self.make_service()
        for lock in (service._lock, server._lock):
            locked, release = threading.Event(), threading.Event()
            def hold():
                with lock:
                    locked.set()
                    release.wait(10)
            thread = threading.Thread(target=hold)
            try:
                thread.start()
                self.assertTrue(locked.wait(5))
                start = time.monotonic()
                self.assertFalse(service.close(timeout=.05))
                self.assertLess(time.monotonic() - start, .5)
            finally:
                release.set()
                thread.join(5)


if __name__ == '__main__':
    unittest.main()
