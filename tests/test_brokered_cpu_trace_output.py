"""Output delivery on the same brokered, traced tiny CPU forward."""
import hashlib
import json
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

from looking_glass.cpu_broker_trace_service import BrokeredCpuTraceService
from looking_glass.host_budget import HostBudget, HostBudgetExceeded, HostRequest
from looking_glass.resident_worker import ForwardResult
from looking_glass.trace_ipc_v2 import TraceIPCClient, TraceIPCIncomplete, TraceIPCRejected, TraceIPCServer
from looking_glass.trace_producer import ClaimLimit, ProducerProfile
from looking_glass.trace_store import HostTraceStore
from looking_glass.traced_broker_worker import TraceLimits
from looking_glass.traced_worker import TracedForwardResult, TracedGPT2Worker


_RETAINED = []
_CLIENT = r'''
import hashlib, json, sys
from dataclasses import asdict
from looking_glass.host_budget import HostBudget
from looking_glass.trace_ipc_v2 import TraceIPCClient
c = json.loads(sys.stdin.readline())
budget = HostBudget(max_bytes=1000000, max_items=64, max_requests=8, max_operations=8)
client = TraceIPCClient(tuple(c['address']), token=c['token'], team_id=c['team'],
    broker_id=c['broker'], process_generation=c['generation'], budget=budget,
    max_event_bytes=32, max_trace_bytes=64, timeout=20)
events = []
def consume(meta, payload):
    print(json.dumps({'kind':'output', 'team':meta.team_id, 'digest':hashlib.sha256(payload).hexdigest(),
                      'job':meta.job_id, 'transfer':meta.transfer_id}), flush=True)
    if c['block'] and sys.stdin.readline().strip() != 'continue':
        raise RuntimeError('missing test release')
result = client.trace(request_id='same-name', input_ids=c['row'], sample_id=c['team'],
    positions=[0], sites=[[0,'attn']], on_event=lambda event, payload: events.append(event.team_id),
    on_output=consume, max_output_bytes=384, vocab_size=32)
print(json.dumps({'kind':'done', 'completion':asdict(result), 'events':events,
                  'budget_bytes':budget.snapshot().used_bytes}), flush=True)
'''


def make_system(capacity=384):
    from transformers import GPT2Config, GPT2LMHeadModel
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=8, n_ctx=8,
            n_embd=8, n_layer=1, n_head=2, use_cache=False,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
    worker = TracedGPT2Worker('trace-output-tiny', model, device='cpu',
        max_batch_size=1, max_sequence_length=3, max_input_bytes=24, max_output_bytes=384)
    sender = HostBudget(max_bytes=3_000_000, max_items=128, max_requests=32, max_operations=32)
    profile = ProducerProfile(ClaimLimit(48, 8), ClaimLimit(768, 2),
                              ClaimLimit(64, 2), ClaimLimit(100_000, 1))
    service = BrokeredCpuTraceService(worker=worker, budget=sender,
        store=HostTraceStore(max_tensor_bytes=1024, max_items=4), profile=profile,
        limits=TraceLimits(1, 3, 1, 2, 2, 64, 3), team_ids=('alpha', 'beta'),
        max_slots=2, max_event_bytes=32)
    server = TraceIPCServer(service, max_event_bytes=32, max_trace_bytes=64,
                            max_output_bytes=capacity, timeout=20).start()
    return server, service, worker, sender


def make_client(server, team):
    budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=8, max_operations=8)
    client = TraceIPCClient(server.address, token=server.team_tokens[team], team_id=team,
        broker_id=server.broker_id, process_generation=server.process_generation,
        budget=budget, max_event_bytes=32, max_trace_bytes=64, timeout=20)
    return client, budget


def wait_incomplete(server):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not server.incomplete:
        time.sleep(.01)
    return server.incomplete


class BrokeredTraceOutputTests(unittest.TestCase):
    def test_opt_in_output_is_from_the_same_forward_and_keeps_claims_live(self):
        server, service, worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        observed, captures, events = [], [], []
        original = worker.execute_traced

        def tracked(*args, **kwargs):
            value = original(*args, **kwargs)
            # This tiny test-owned copy compares the outgoing bytes with the
            # exact worker result, not a second forward.
            observed.append((threading.current_thread(), value.forward_result.logits.numpy().tobytes()))
            return value

        def output(metadata, payload):
            sender_stages = {claim.stage for claim in sender.snapshot().claims}
            receiver_stages = {claim.stage for claim in receiver.snapshot().claims}
            captures.append((metadata, bytes(payload), sender_stages, receiver_stages))

        try:
            with patch.object(worker, 'execute_traced', new=tracked):
                result = client.trace(request_id='output-one', input_ids=[1, 2, 3], sample_id='row',
                    positions=[0], sites=[[0, 'attn']], on_event=lambda event, _: events.append(event),
                    on_output=output, max_output_bytes=384, vocab_size=32)
            self.assertEqual(len(observed), 1)
            self.assertIs(observed[0][0], service._broker._thread)
            self.assertEqual(len(captures), 1)
            metadata, payload, sender_stages, receiver_stages = captures[0]
            self.assertEqual(payload, observed[0][1])
            self.assertEqual(metadata.shape, (1, 3, 32))
            self.assertEqual(metadata.payload_bytes, 384)
            self.assertEqual(metadata.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual((metadata.team_id, metadata.request_id), ('alpha', 'output-one'))
            self.assertEqual((metadata.job_id, metadata.generation, metadata.transfer_id),
                             (result.job_id, result.generation, result.transfer_id))
            self.assertEqual(events[0].job_id, metadata.job_id)
            self.assertEqual((result.output_bytes, result.output_sha256),
                             (384, metadata.sha256))
            self.assertIn('output', sender_stages)
            self.assertIn('trace_output_send', sender_stages)
            self.assertIn('trace_output_receive', receiver_stages)
            self.assertEqual(receiver.snapshot().used_bytes, 0)
            self.assertTrue(server.close(timeout=5))
            self.assertTrue(service.close(timeout=5))
            self.assertEqual(sender.snapshot().used_bytes, 0)
        finally:
            server.close(timeout=5)
            service.close(timeout=5)

    def test_output_limit_is_rejected_before_any_forward(self):
        server, service, worker, sender = make_system(capacity=383)
        client, receiver = make_client(server, 'alpha')
        seen = []
        try:
            with patch.object(worker, 'execute_traced', wraps=worker.execute_traced) as forward:
                with self.assertRaisesRegex(TraceIPCRejected, 'output_limit'):
                    client.trace(request_id='bounded', input_ids=[1, 2, 3], sample_id='row',
                        positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                        on_output=lambda *args: seen.append(args), max_output_bytes=384, vocab_size=32)
                forward.assert_not_called()
            self.assertEqual(seen, [])
            self.assertEqual(service._broker._next_job_id, 0)
            self.assertFalse(client.incomplete or server.incomplete or service.quarantined)
            self.assertEqual(receiver.snapshot().used_bytes, 0)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and server.active_count:
                time.sleep(.01)
            self.assertEqual(sender.snapshot().used_bytes, 0)
            self.assertTrue(server.close(timeout=5))
            self.assertTrue(service.close(timeout=5))
        finally:
            server.close(timeout=5)
            service.close(timeout=5)

    def test_wrong_vocabulary_is_rejected_before_any_forward(self):
        server, service, worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        try:
            with patch.object(worker, 'execute_traced', wraps=worker.execute_traced) as forward:
                with self.assertRaisesRegex(TraceIPCRejected, 'invalid_request'):
                    client.trace(request_id='wrong-vocab', input_ids=[1, 2, 3], sample_id='row',
                        positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                        on_output=lambda *_: None, max_output_bytes=384, vocab_size=31)
                forward.assert_not_called()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and server.active_count:
                time.sleep(.01)
            self.assertFalse(client.incomplete or server.incomplete or service.quarantined)
            self.assertEqual(service._broker._next_job_id, 0)
            self.assertEqual(receiver.snapshot().used_bytes, 0)
            self.assertEqual(sender.snapshot().used_bytes, 0)
            self.assertTrue(server.close(timeout=5))
            self.assertTrue(service.close(timeout=5))
        finally:
            server.close(timeout=5)
            service.close(timeout=5)

    def test_output_and_trace_only_clients_can_share_the_same_broker(self):
        server, service, _worker, sender = make_system()
        alpha, alpha_budget = make_client(server, 'alpha')
        beta, beta_budget = make_client(server, 'beta')
        seen = []
        try:
            output = alpha.trace(request_id='output', input_ids=[1, 2, 3], sample_id='a',
                positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                on_output=lambda meta, _: seen.append(meta), max_output_bytes=384, vocab_size=32)
            trace = beta.trace(request_id='plain', input_ids=[3, 2, 1], sample_id='b',
                positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None)
            self.assertEqual(len(seen), 1)
            self.assertEqual(output.output_bytes, 384)
            self.assertFalse(hasattr(trace, 'output_bytes'))
            self.assertNotEqual(output.job_id, trace.job_id)
            self.assertEqual(alpha_budget.snapshot().used_bytes, 0)
            self.assertEqual(beta_budget.snapshot().used_bytes, 0)
            self.assertTrue(server.close(timeout=5))
            self.assertTrue(service.close(timeout=5))
            self.assertEqual(sender.snapshot().used_bytes, 0)
        finally:
            server.close(timeout=5)
            service.close(timeout=5)

    def test_two_clients_receive_only_their_same_forward_outputs(self):
        server, service, worker, sender = make_system()
        alpha, alpha_budget = make_client(server, 'alpha')
        beta, beta_budget = make_client(server, 'beta')
        entered, release = threading.Event(), threading.Event()
        observations, outputs, completions, failures = {}, {}, {}, {}
        original = worker.execute_traced

        def tracked(request, *args, **kwargs):
            row = tuple(request.inputs['input_ids'][0].tolist())
            result = original(request, *args, **kwargs)
            observations[row] = result.forward_result.logits.numpy().tobytes()
            return result

        def run(client, team, row):
            def consume(meta, payload):
                outputs[team] = (meta, bytes(payload))
                if team == 'alpha':
                    entered.set()
                    if not release.wait(10):
                        raise TimeoutError('test output was not released')
            try:
                completions[team] = client.trace(request_id='same-name', input_ids=row,
                    sample_id=team, positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                    on_output=consume, max_output_bytes=384, vocab_size=32)
            except BaseException as exc:
                failures[team] = exc

        a = threading.Thread(target=run, args=(alpha, 'alpha', [1, 2, 3]), daemon=True)
        b = threading.Thread(target=run, args=(beta, 'beta', [3, 2, 1]), daemon=True)
        try:
            with patch.object(worker, 'execute_traced', new=tracked):
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
                    self.fail('second team did not queue')
                self.assertNotIn('beta', outputs)
                release.set()
                a.join(15)
                b.join(15)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertEqual(failures, {})
            self.assertEqual(set(outputs), {'alpha', 'beta'})
            for team, row in (('alpha', (1, 2, 3)), ('beta', (3, 2, 1))):
                meta, payload = outputs[team]
                self.assertEqual(payload, observations[row])
                self.assertEqual((meta.team_id, meta.request_id), (team, 'same-name'))
                self.assertEqual((meta.job_id, meta.transfer_id),
                                 (completions[team].job_id, completions[team].transfer_id))
            self.assertNotEqual(outputs['alpha'][1], outputs['beta'][1])
            self.assertNotEqual(outputs['alpha'][0].job_id, outputs['beta'][0].job_id)
            self.assertEqual(alpha_budget.snapshot().used_bytes, 0)
            self.assertEqual(beta_budget.snapshot().used_bytes, 0)
            self.assertTrue(server.close(timeout=5))
            self.assertTrue(service.close(timeout=5))
            self.assertEqual(sender.snapshot().used_bytes, 0)
        finally:
            release.set()
            if a.ident is not None:
                a.join(15)
            if b.ident is not None:
                b.join(15)
            server.close(timeout=5)
            service.close(timeout=5)

    def test_two_processes_receive_only_their_own_output_and_trace(self):
        server, service, worker, sender = make_system()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        children, observations = [], {}
        original = worker.execute_traced
        def tracked(request, *args, **kwargs):
            row = tuple(request.inputs['input_ids'][0].tolist())
            result = original(request, *args, **kwargs)
            observations[row] = hashlib.sha256(result.forward_result.logits.numpy().tobytes()).hexdigest()
            return result
        def launch(team, row, block):
            proc = subprocess.Popen([sys.executable, '-B', '-c', _CLIENT], cwd=Path(__file__).resolve().parents[1],
                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1)
            children.append(proc)
            messages = queue.Queue()
            def read():
                for line in proc.stdout:
                    messages.put(line)
            threading.Thread(target=read, daemon=True).start()
            proc.stdin.write(json.dumps({'address': server.address, 'token': server.team_tokens[team],
                'team': team, 'broker': server.broker_id, 'generation': server.process_generation,
                'row': row, 'block': block}) + '\n')
            proc.stdin.flush()
            return proc, messages
        try:
            with patch.object(worker, 'execute_traced', new=tracked):
                alpha, aq = launch('alpha', [1, 2, 3], True)
                a_output = json.loads(aq.get(timeout=20))
                beta, bq = launch('beta', [3, 2, 1], False)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    with service._broker._condition:
                        if service._broker._pending_slots == 1:
                            break
                    time.sleep(.01)
                else:
                    self.fail('second process did not queue')
                self.assertTrue(bq.empty())
                alpha.stdin.write('continue\n')
                alpha.stdin.flush()
                a_done = json.loads(aq.get(timeout=20))
                b_output = json.loads(bq.get(timeout=20))
                b_done = json.loads(bq.get(timeout=20))
            for team, row, output, done in (('alpha', (1, 2, 3), a_output, a_done),
                                             ('beta', (3, 2, 1), b_output, b_done)):
                self.assertEqual((output['kind'], output['team']), ('output', team))
                self.assertEqual(output['digest'], observations[row])
                self.assertEqual(done['kind'], 'done')
                self.assertEqual(done['events'], [team])
                self.assertEqual(done['budget_bytes'], 0)
                self.assertEqual((output['job'], output['transfer']),
                                 (done['completion']['job_id'], done['completion']['transfer_id']))
            self.assertNotEqual(a_output['digest'], b_output['digest'])
            self.assertNotEqual(a_output['job'], b_output['job'])
            for child in children:
                self.assertEqual(child.wait(timeout=20), 0, child.stderr.read())
            self.assertTrue(server.close(timeout=5))
            self.assertTrue(service.close(timeout=5))
            self.assertEqual(sender.snapshot().used_bytes, 0)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                for stream in (child.stdin, child.stdout, child.stderr):
                    if stream is not None:
                        stream.close()
            server.close(timeout=5)
            service.close(timeout=5)

    def test_failed_output_callback_retains_both_claims_and_has_no_receipt(self):
        server, service, _worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        held, events = [], []
        def fail(_metadata, payload):
            held.append(payload)
            raise RuntimeError('receiver did not release output')
        with self.assertRaises(TraceIPCIncomplete):
            client.trace(request_id='output-failure', input_ids=[1, 2, 3], sample_id='row',
                positions=[0], sites=[[0, 'attn']], on_event=lambda *_: events.append(1),
                on_output=fail, max_output_bytes=384, vocab_size=32)
        self.assertTrue(wait_incomplete(server))
        self.assertTrue(client.incomplete and service.quarantined)
        self.assertEqual(events, [])
        self.assertEqual(len(held[0]), 384)
        self.assertTrue(any(c.stage == 'trace_output_receive' and c.nbytes == 384
                            for c in receiver.snapshot().claims))
        self.assertTrue(any(c.stage == 'trace_output_send' and c.nbytes == 384
                            for c in sender.snapshot().claims))
        self.assertTrue(any(c.stage == 'output' for c in sender.snapshot().claims))
        with self.assertRaises(TraceIPCIncomplete):
            client.recover()
        self.assertFalse(service.close(timeout=1))
        _RETAINED.append((server, service, sender, receiver, client, held))

    def test_output_callback_must_return_none_before_release(self):
        server, service, _worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        with self.assertRaises(TraceIPCIncomplete):
            client.trace(request_id='deferred', input_ids=[1, 2, 3], sample_id='row',
                positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                on_output=lambda *_: 'not released', max_output_bytes=384, vocab_size=32)
        self.assertTrue(wait_incomplete(server))
        self.assertTrue(client.incomplete and service.quarantined)
        self.assertTrue(any(c.stage == 'trace_output_send' for c in sender.snapshot().claims))
        self.assertTrue(any(c.stage == 'trace_output_receive' for c in receiver.snapshot().claims))
        _RETAINED.append((server, service, sender, receiver, client))

    def test_output_callback_interruption_preserves_receiver_and_sender_claims(self):
        server, service, _worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        original = KeyboardInterrupt('interrupted output callback')
        def interrupted(_meta, _payload):
            raise original
        with self.assertRaises(KeyboardInterrupt) as raised:
            client.trace(request_id='interrupted', input_ids=[1, 2, 3], sample_id='row',
                positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                on_output=interrupted, max_output_bytes=384, vocab_size=32)
        self.assertIs(raised.exception, original)
        self.assertTrue(wait_incomplete(server))
        self.assertTrue(client.incomplete and service.quarantined)
        self.assertIs(client._active.failure, original)
        self.assertTrue(any(c.stage == 'trace_output_receive' for c in receiver.snapshot().claims))
        self.assertTrue(any(c.stage == 'trace_output_send' for c in sender.snapshot().claims))
        _RETAINED.append((server, service, sender, receiver, client, original))

    def test_receiver_capacity_failure_after_forward_is_incomplete(self):
        server, service, worker, sender = make_system()
        receiver = HostBudget(max_bytes=8 * 65536, max_items=64, max_requests=8, max_operations=8)
        client = TraceIPCClient(server.address, token=server.team_tokens['alpha'], team_id='alpha',
            broker_id=server.broker_id, process_generation=server.process_generation,
            budget=receiver, max_event_bytes=32, max_trace_bytes=64, timeout=20)
        observed = []
        with patch.object(worker, 'execute_traced', wraps=worker.execute_traced) as forward:
            with self.assertRaises(TraceIPCIncomplete):
                client.trace(request_id='receiver-capacity', input_ids=[1, 2, 3], sample_id='row',
                    positions=[0], sites=[[0, 'attn']], on_event=lambda *_: observed.append('event'),
                    on_output=lambda *_: observed.append('output'), max_output_bytes=384, vocab_size=32)
            self.assertEqual(forward.call_count, 1)
        self.assertTrue(wait_incomplete(server))
        self.assertEqual(observed, [])
        self.assertTrue(client.incomplete and service.quarantined)
        self.assertTrue(any(c.stage == 'trace_output_send' for c in sender.snapshot().claims))
        self.assertTrue(any(c.stage == 'trace_ipc_control' for c in receiver.snapshot().claims))
        with self.assertRaises(TraceIPCIncomplete):
            client.recover()
        _RETAINED.append((server, service, worker, sender, receiver, client))

    def test_output_admission_failure_does_not_run_the_model(self):
        server, service, worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        original = HostRequest.claim_guard
        def deny_output(request, owner, stage, nbytes, **kwargs):
            if stage == 'trace_output_send':
                raise HostBudgetExceeded('injected output admission failure')
            return original(request, owner, stage, nbytes, **kwargs)
        with patch.object(HostRequest, 'claim_guard', new=deny_output), \
             patch.object(worker, 'execute_traced', wraps=worker.execute_traced) as forward:
            with self.assertRaises(TraceIPCIncomplete):
                client.trace(request_id='capacity-failure', input_ids=[1, 2, 3], sample_id='row',
                    positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                    on_output=lambda *_: None, max_output_bytes=384, vocab_size=32)
            forward.assert_not_called()
        self.assertTrue(wait_incomplete(server))
        self.assertTrue(client.incomplete and service.quarantined)
        self.assertTrue(any(c.stage == 'output' for c in sender.snapshot().claims))
        with self.assertRaises(TraceIPCIncomplete):
            client.recover()
        _RETAINED.append((server, service, worker, sender, receiver, client))

    def test_wrong_output_dtype_is_never_delivered(self):
        server, service, worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        original = worker.execute_traced
        observed = []
        def wrong_dtype(*args, **kwargs):
            value = original(*args, **kwargs)
            changed = ForwardResult(value.forward_result.request_id, value.forward_result.logits.double())
            return TracedForwardResult(changed, value.events)
        with patch.object(worker, 'execute_traced', new=wrong_dtype):
            with self.assertRaises(TraceIPCIncomplete):
                client.trace(request_id='wrong-dtype', input_ids=[1, 2, 3], sample_id='row',
                    positions=[0], sites=[[0, 'attn']], on_event=lambda *_: observed.append('event'),
                    on_output=lambda *_: observed.append('output'), max_output_bytes=384, vocab_size=32)
        self.assertTrue(wait_incomplete(server))
        self.assertEqual(observed, [])
        self.assertTrue(client.incomplete and service.quarantined)
        self.assertTrue(any(c.stage == 'trace_output_send' for c in sender.snapshot().claims))
        self.assertTrue(any(c.stage == 'output' for c in sender.snapshot().claims))
        _RETAINED.append((server, service, worker, sender, receiver, client))

    def test_cross_team_output_header_is_rejected_without_callback(self):
        server, service, _worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        published = server._publish
        seen = []
        def swap(session, message, deadline):
            if message.get('op') == 'output':
                message = dict(message, team_id='beta')
            return published(session, message, deadline)
        with patch.object(server, '_publish', new=swap):
            with self.assertRaises(TraceIPCIncomplete):
                client.trace(request_id='swap', input_ids=[1, 2, 3], sample_id='row',
                    positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                    on_output=lambda *args: seen.append(args), max_output_bytes=384, vocab_size=32)
            self.assertTrue(wait_incomplete(server))
        self.assertEqual(seen, [])
        self.assertTrue(client.incomplete and service.quarantined)
        self.assertTrue(any(c.stage == 'trace_output_send' for c in sender.snapshot().claims))
        _RETAINED.append((server, service, sender, receiver, client))

    def test_output_digest_mismatch_is_not_exposed_to_the_client_callback(self):
        from looking_glass import trace_ipc_v2 as wire
        server, service, _worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        send_bytes = wire._send_bytes
        seen = []
        def corrupt_payload(sock, payload, deadline):
            if type(payload) is bytearray and len(payload) == 384:
                payload = bytearray(payload)
                payload[0] ^= 1
            return send_bytes(sock, payload, deadline)
        with patch.object(wire, '_send_bytes', new=corrupt_payload):
            with self.assertRaises(TraceIPCIncomplete):
                client.trace(request_id='corrupt', input_ids=[1, 2, 3], sample_id='row',
                    positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                    on_output=lambda *args: seen.append(args), max_output_bytes=384, vocab_size=32)
            self.assertTrue(wait_incomplete(server))
        self.assertEqual(seen, [])
        self.assertTrue(client.incomplete and service.quarantined)
        self.assertTrue(any(c.stage == 'trace_output_send' for c in sender.snapshot().claims))
        self.assertTrue(any(c.stage == 'trace_output_receive' for c in receiver.snapshot().claims))
        _RETAINED.append((server, service, sender, receiver, client))

    def test_cancellation_during_output_does_not_produce_a_complete_receipt(self):
        server, service, _worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        entered, release = threading.Event(), threading.Event()
        observed, events, outcomes = [], [], []
        def consume(meta, payload):
            observed.append((meta, bytes(payload)))
            entered.set()
            if not release.wait(10):
                raise TimeoutError('test output was not released')
        def run():
            try:
                outcomes.append(client.trace(request_id='cancel', input_ids=[1, 2, 3], sample_id='row',
                    positions=[0], sites=[[0, 'attn']], on_event=lambda *_: events.append(1),
                    on_output=consume, max_output_bytes=384, vocab_size=32))
            except BaseException as exc:
                outcomes.append(exc)
        thread = threading.Thread(target=run, daemon=True)
        try:
            thread.start()
            self.assertTrue(entered.wait(15))
            self.assertFalse(service._broker.close(wait=False))
            release.set()
            thread.join(15)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(observed), 1)
            self.assertEqual(events, [])
            self.assertEqual(len(outcomes), 1)
            self.assertIsInstance(outcomes[0], TraceIPCIncomplete)
            self.assertTrue(wait_incomplete(server))
            self.assertTrue(client.incomplete and service.quarantined)
            self.assertTrue(any(c.stage == 'trace_output_send' for c in sender.snapshot().claims))
            with self.assertRaises(TraceIPCIncomplete):
                client.recover()
        finally:
            release.set()
            thread.join(15)
            _RETAINED.append((server, service, sender, receiver, client, observed))

    def test_lost_final_ack_leaves_server_incomplete_after_local_completion(self):
        from looking_glass import trace_ipc_v2 as wire
        server, service, _worker, sender = make_system()
        client, receiver = make_client(server, 'alpha')
        original = wire._receive
        def lose_final(sock, limit, deadline):
            message = original(sock, limit, deadline)
            if threading.current_thread().name == 'trace-ipc-client' and message.get('op') == 'closed_received':
                raise EOFError('simulated lost final acknowledgement')
            return message
        seen = []
        with patch.object(wire, '_receive', new=lose_final):
            result = client.trace(request_id='final-output', input_ids=[1, 2, 3], sample_id='row',
                positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None,
                on_output=lambda meta, _: seen.append(meta.sha256), max_output_bytes=384, vocab_size=32)
            self.assertEqual(result.output_sha256, seen[0])
            self.assertTrue(wait_incomplete(server))
        self.assertFalse(client.incomplete)
        self.assertEqual(receiver.snapshot().used_bytes, 0)
        self.assertTrue(any(c.stage == 'trace_ipc_control' for c in sender.snapshot().claims))
        self.assertFalse(server.close(timeout=5))
        _RETAINED.append((server, service, sender, receiver, client))


if __name__ == '__main__':
    unittest.main()
