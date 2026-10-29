"""CPU-only integration tests for the separate traced loopback protocol."""
import json
import gc
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch
import weakref

import torch
from looking_glass.cpu_trace_owner import CpuTraceOwner
from looking_glass.cpu_shared_trace_service import CpuSharedTraceService
from looking_glass.host_budget import HostBudget
from looking_glass.ipc import _read_exact, _receive, _send
from looking_glass.trace_ipc_v2 import TraceIPCClient, TraceIPCIncomplete, TraceIPCRejected, TraceIPCServer
from looking_glass.trace_ipc_v2 import _ack, _send_bytes
from looking_glass.trace_producer import ClaimLimit, ProducerProfile
from looking_glass.trace_store import HostTraceStore
from looking_glass.traced_broker_worker import TraceLimits
from looking_glass.traced_worker import TracedGPT2Worker

_FAULT_ROOTS = []

_CLIENT_CODE = r'''
import hashlib, json, sys
from dataclasses import asdict
from looking_glass.host_budget import HostBudget
from looking_glass.trace_ipc_v2 import TraceIPCClient
c = json.loads(sys.stdin.readline())
budget = HostBudget(max_bytes=1000000, max_items=64, max_requests=8, max_operations=8)
client = TraceIPCClient(tuple(c['address']), token=c['token'], team_id=c['team'],
    broker_id=c['broker_id'], process_generation=c['process_generation'], budget=budget,
    max_event_bytes=32, max_trace_bytes=64, timeout=20)
events = []
def consume(meta, payload):
    events.append([meta.event.module_kind, meta.event.sequence,
                   meta.event.identity.sample_id, hashlib.sha256(payload).hexdigest()])
    if len(events) == 1:
        print(json.dumps({'kind':'event', 'team':c['team']}), flush=True)
        if c['block']:
            assert sys.stdin.readline().strip() == 'continue'
result = client.trace(request_id='same-request', input_ids=c['input_ids'],
    sample_id=c['team'] + '-sample', positions=[c['position']],
    sites=[[0, 'attn'], [0, 'mlp']], on_event=consume)
print(json.dumps({'kind':'done', 'completion':asdict(result), 'events':events,
                  'budget_bytes':budget.snapshot().used_bytes}), flush=True)
'''

def _reader(stream, output):
    try:
        for line in stream:
            output.put(line)
    finally:
        output.put(None)

def _budget():
    return HostBudget(max_bytes=1_000_000, max_items=64, max_requests=8, max_operations=8)

def _wire_event(broker, generation, transfer, event_id, request_id):
    run = 'run-' + broker + '-1'
    event = {'schema_version':'1.0', 'identity':{'run_id':run, 'job_id':'lg-'+broker+'-1',
                 'request_id':request_id, 'sample_id':'row'},
        'generation':{'stage':'unknown', 'generation_step':None, 'cache_position':None,
                      'cache_length_before':None, 'cache_length_after':None, 'use_cache':None},
        'sequence':0, 'forward_index':0, 'module_path':'transformer.h.0.attn',
        'invocation':0, 'dtype':'torch.float32', 'device':'cpu', 'capture_mode':'direct',
        'module_kind':'attn', 'layer_index':0, 'batch_axis':0, 'token_axis':1,
        'saved_dtype':'torch.float32', 'saved_device':'cpu', 'shape':[1,3,8],
        'output_path':[0], 'batch_positions':[0], 'token_positions':[0], 'saved_shape':[1,8]}
    return {'version':2, 'op':'event', 'broker_id':broker, 'process_generation':generation,
        'job_id':1, 'generation':1, 'transfer_id':transfer, 'event_id':event_id,
        'team_id':'alpha', 'request_id':request_id, 'event':event, 'payload_bytes':32, 'byte_order':'little'}

class TraceIPCV2Tests(unittest.TestCase):
    def make_service(self, *, dtype=torch.float32):
        from transformers import GPT2Config, GPT2LMHeadModel
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(71)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=8, n_ctx=8,
                n_embd=8, n_layer=1, n_head=2, use_cache=False,
                resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval().to(dtype=dtype)
        worker = TracedGPT2Worker('ipc-tiny-gpt2', model, device='cpu', max_batch_size=1,
            max_sequence_length=3, max_input_bytes=24, max_output_bytes=384)
        budget = HostBudget(max_bytes=3_000_000, max_items=128, max_requests=32, max_operations=32)
        owner = CpuTraceOwner(worker=worker, budget=budget,
            store=HostTraceStore(max_tensor_bytes=1024, max_items=4),
            broker_id='a' * 32, process_generation='b' * 32, max_slots=2, max_event_bytes=32)
        profile = ProducerProfile(ClaimLimit(48, 8), ClaimLimit(768, 2),
                                  ClaimLimit(64, 2), ClaimLimit(100_000, 1))
        service = CpuSharedTraceService(owner=owner, profile=profile,
            limits=TraceLimits(1, 3, 1, 2, 2, 64, 3), team_ids=('alpha', 'beta'))
        try:
            server = TraceIPCServer(service, max_control_bytes=8192, max_event_bytes=32,
                                    max_trace_bytes=64, timeout=20).start()
        except BaseException:
            service.close()
            worker.close()
            raise
        def cleanup():
            if server.close(timeout=5) and not server.incomplete and not service.quarantined:
                service.close()
                worker.close()
            else:
                _FAULT_ROOTS.append((server, service, worker, budget))
        self.addCleanup(cleanup)
        return server, service, budget

    def client(self, server, team, *, token=None):
        budget = _budget()
        client = TraceIPCClient(server.address, token=token or server.team_tokens[team],
            team_id=team, broker_id=server.broker_id, process_generation=server.process_generation,
            budget=budget, max_event_bytes=32, max_trace_bytes=64, timeout=20)
        return client, budget

    def test_two_real_client_processes_receive_only_their_own_complete_traces(self):
        server, service, budget = self.make_service()
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        children = []
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
            alpha, aq = launch('alpha', [1, 2, 3], 0, True)
            self.assertEqual(json.loads(aq.get(timeout=20)), {'kind':'event', 'team':'alpha'})
            beta, bq = launch('beta', [3, 2, 1], 2, False)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                with service._condition:
                    if 'beta' in service._pending:
                        break
                time.sleep(.01)
            else:
                self.fail('the second client did not wait behind the first trace')
            self.assertTrue(bq.empty())
            alpha.stdin.write('continue\n')
            alpha.stdin.flush()
            a = json.loads(aq.get(timeout=20))
            self.assertEqual(json.loads(bq.get(timeout=20)), {'kind':'event', 'team':'beta'})
            b = json.loads(bq.get(timeout=20))
            for value, team in ((a, 'alpha'), (b, 'beta')):
                self.assertEqual(value['kind'], 'done')
                done = value['completion']
                self.assertEqual((done['team_id'], done['request_id'], done['event_count'], done['trace_bytes']),
                                 (team, 'same-request', 2, 64))
                self.assertEqual([event[:3] for event in value['events']],
                                 [['attn', 0, team + '-sample'], ['mlp', 1, team + '-sample']])
                self.assertEqual(value['budget_bytes'], 0)
            self.assertNotEqual(a['completion']['job_id'], b['completion']['job_id'])
            self.assertNotEqual(a['events'][0][3], b['events'][0][3])
            for child in children:
                code = child.wait(timeout=20)
                self.assertEqual(code, 0, child.stderr.read())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and server.active_count:
                time.sleep(.01)
            self.assertEqual(server.active_count, 0)
            self.assertEqual(budget.snapshot().used_bytes, 0)
            self.assertFalse(server.incomplete)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)

    def test_failed_receiver_retains_its_payload_and_both_sides_are_incomplete(self):
        server, service, server_budget = self.make_service()
        client, client_budget = self.client(server, 'alpha')
        held = []
        def failing(_metadata, payload):
            held.append(payload)
            raise RuntimeError('receiver stopped before release')
        with self.assertRaises(TraceIPCIncomplete):
            client.trace(request_id='partial', input_ids=[1, 2, 3], sample_id='row',
                         positions=[0], sites=[[0, 'attn']], on_event=failing)
        self.assertTrue(client.incomplete)
        self.assertEqual(len(held), 1)
        self.assertTrue(any(c.stage == 'trace_ipc_receive' and c.nbytes >= len(held[0])
                            for c in client_budget.snapshot().claims))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not server.incomplete:
            time.sleep(.01)
        self.assertTrue(server.incomplete)
        self.assertTrue(service.quarantined)
        self.assertTrue(any(c.stage == 'callback_exposure' and c.nbytes >= len(held[0])
                            for c in server_budget.snapshot().claims))
        self.assertFalse(server.close(timeout=5))
        _FAULT_ROOTS.append((client, client_budget, held))

    def test_partial_send_keeps_claims_and_stops_queued_other_team(self):
        server, service, sender_budget = self.make_service()
        alpha, alpha_budget = self.client(server, 'alpha')
        beta, beta_budget = self.client(server, 'beta')
        partial_written, release_sender = threading.Event(), threading.Event()
        outcomes = {}
        from looking_glass import trace_ipc_v2 as module
        original = module._send_bytes
        def partial_send(sock, payload, deadline):
            if type(payload) is bytes and len(payload) == 32:
                view = memoryview(payload)[:7]
                try:
                    original(sock, view, deadline)
                finally:
                    view.release()
                partial_written.set()
                if not release_sender.wait(10):
                    raise TimeoutError('partial sender was not released')
                raise ConnectionResetError('synthetic partial-event disconnect')
            return original(sock, payload, deadline)
        def run(client, team):
            try:
                outcomes[team] = client.trace(request_id=team + '-partial',
                    input_ids=[1,2,3], sample_id=team + '-row', positions=[0],
                    sites=[[0,'attn']], on_event=lambda *_: None)
            except BaseException as error:
                outcomes[team] = error
        a = threading.Thread(target=run, args=(alpha, 'alpha'))
        b = threading.Thread(target=run, args=(beta, 'beta'))
        try:
            with patch.object(module, '_send_bytes', partial_send):
                a.start()
                self.assertTrue(partial_written.wait(10))
                b.start()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    with service._condition:
                        queued = 'beta' in service._pending
                    if queued and alpha._active is not None and alpha._active.payload is not None:
                        break
                    time.sleep(.01)
                else:
                    self.fail('partial receive and queued second team were not observed')
                payload = alpha._active.payload
                self.assertIs(type(payload), bytearray)
                self.assertEqual(len(payload), 32)
                self.assertTrue(any(c.stage == 'trace_ipc_receive' and c.nbytes == 32
                                    for c in alpha_budget.snapshot().claims))
                self.assertTrue(any(c.stage == 'callback_exposure' and c.nbytes == 32
                                    for c in sender_budget.snapshot().claims))
                self.assertIsNotNone(service._active.exposed_payload)
                release_sender.set()
                a.join(15)
                b.join(15)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertIsInstance(outcomes.get('alpha'), TraceIPCIncomplete)
            self.assertIsInstance(outcomes.get('beta'), TraceIPCIncomplete)
            self.assertTrue(alpha.incomplete and beta.incomplete and server.incomplete and service.quarantined)
            self.assertEqual(service._generation, 1)
            self.assertIs(alpha._active.payload, payload)
            self.assertTrue(any(c.stage == 'trace_ipc_receive' and c.nbytes == 32
                                for c in alpha_budget.snapshot().claims))
            self.assertTrue(any(c.stage == 'callback_exposure' and c.nbytes == 32
                                for c in sender_budget.snapshot().claims))
            self.assertGreater(beta_budget.snapshot().used_bytes, 0)
            _FAULT_ROOTS.append((alpha, alpha_budget, beta, beta_budget))
        finally:
            release_sender.set()
            if a.ident is not None:
                a.join(15)
            if b.ident is not None:
                b.join(15)

    def test_queued_client_timeout_does_not_cancel_its_accepted_job(self):
        server, service, _sender_budget = self.make_service()
        alpha, _alpha_budget = self.client(server, 'alpha')
        beta_budget = _budget()
        beta = TraceIPCClient(server.address, token=server.team_tokens['beta'], team_id='beta',
            broker_id=server.broker_id, process_generation=server.process_generation,
            budget=beta_budget, max_event_bytes=32, max_trace_bytes=64, timeout=1)
        entered, release, beta_done = threading.Event(), threading.Event(), threading.Event()
        outcomes = {}
        def alpha_callback(*_):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('test did not release alpha')
        def run(client, team, callback):
            try:
                outcomes[team] = client.trace(request_id=team+'-queued', input_ids=[1,2,3],
                    sample_id=team+'-row', positions=[0], sites=[[0,'attn']], on_event=callback)
            except BaseException as error:
                outcomes[team] = error
            finally:
                if team == 'beta':
                    beta_done.set()
        a = threading.Thread(target=run, args=(alpha, 'alpha', alpha_callback))
        b = threading.Thread(target=run, args=(beta, 'beta', lambda *_: None))
        try:
            a.start()
            self.assertTrue(entered.wait(15))
            b.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with service._condition:
                    if 'beta' in service._pending:
                        break
                time.sleep(.01)
            else:
                self.fail('beta never reached the service queue')
            self.assertTrue(beta_done.wait(5))
            self.assertIsInstance(outcomes['beta'], TraceIPCIncomplete)
            self.assertTrue(beta.incomplete)
            self.assertEqual(service._generation, 1)
            release.set()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with service._condition:
                    delivered = (service._generation == 2 and service._active is not None
                                 and service._active.stage == 'delivery')
                if delivered:
                    break
                time.sleep(.01)
            else:
                self.fail('the disconnected queued job did not reach delivery')
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not service.quarantined:
                time.sleep(.01)
            self.assertTrue(service.quarantined)
            _FAULT_ROOTS.append((beta, beta_budget))
        finally:
            release.set()
            if a.ident is not None:
                a.join(15)
            if b.ident is not None:
                b.join(15)

    def test_live_send_and_receive_have_separate_claims_and_input_dies_after_drain(self):
        server, _service, server_budget = self.make_service()
        client, client_budget = self.client(server, 'alpha')
        observed, input_refs = [], []
        def consume(_metadata, _payload):
            sender = {claim.stage: claim.nbytes for claim in server_budget.snapshot().claims}
            receiver = {claim.stage: claim.nbytes for claim in client_budget.snapshot().claims}
            with server._lock:
                session = next(iter(server._sessions.values()))
                input_refs.append(weakref.ref(session.inputs['input_ids']))
            observed.append((sender, receiver))
        result = client.trace(request_id='live-claims', input_ids=[1,2,3], sample_id='row',
            positions=[0], sites=[[0,'attn']], on_event=consume)
        self.assertEqual(result.trace_bytes, 32)
        self.assertIs(client.recover(), result)
        with self.assertRaises(TraceIPCIncomplete):
            client.trace(request_id='again', input_ids=[1,2,3], sample_id='row',
                positions=[0], sites=[[0,'attn']], on_event=consume)
        self.assertFalse(client.incomplete)
        sender, receiver = observed[0]
        self.assertGreaterEqual(sender['trace_ipc_control'], 8 * 65536)
        self.assertEqual(sender['trace_ipc_input'], 24)
        self.assertEqual(sender['store-copy'], 32)
        self.assertEqual(sender['serialization'], 32)
        self.assertEqual(sender['callback_exposure'], 32)
        self.assertGreaterEqual(receiver['trace_ipc_control'], 8 * 65536)
        self.assertEqual(receiver['trace_ipc_receive'], 32)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and server.active_count:
            time.sleep(.01)
        gc.collect()
        self.assertIsNone(input_refs[0]())
        self.assertEqual(server_budget.snapshot().used_bytes, 0)
        self.assertEqual(client_budget.snapshot().used_bytes, 0)

    def test_wrong_team_token_is_rejected_before_model_work(self):
        server, service, budget = self.make_service()
        client, client_budget = self.client(server, 'alpha', token=server.team_tokens['beta'])
        with self.assertRaises(TraceIPCRejected):
            client.trace(request_id='wrong-team', input_ids=[1, 2, 3], sample_id='row',
                         positions=[0], sites=[[0, 'attn']], on_event=lambda *_: None)
        self.assertFalse(server.incomplete)
        self.assertFalse(service.quarantined)
        self.assertEqual(client_budget.snapshot().used_bytes, 0)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and server.active_count:
            time.sleep(.01)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_event_byte_limit_rejects_before_forward(self):
        server, service, budget = self.make_service()
        client, client_budget = self.client(server, 'alpha')
        seen = []
        with self.assertRaisesRegex(TraceIPCRejected, 'trace_limit'):
            client.trace(request_id='too-large', input_ids=[1,2,3], sample_id='row',
                         positions=[0,1], sites=[[0,'attn']], on_event=lambda *_: seen.append(1))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and server.active_count:
            time.sleep(.01)
        self.assertEqual(seen, [])
        self.assertEqual(service._generation, 0)
        self.assertFalse(service.quarantined)
        self.assertFalse(server.incomplete)
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertEqual(client_budget.snapshot().used_bytes, 0)

    def test_callback_rejection_and_interrupt_are_incomplete_not_pre_forward_rejection(self):
        for original in (TraceIPCRejected('callback failure'), KeyboardInterrupt('callback interruption')):
            with self.subTest(kind=type(original).__name__):
                server, service, sender_budget = self.make_service()
                client, receiver_budget = self.client(server, 'alpha')
                def fail(_metadata, _payload):
                    raise original
                expected = KeyboardInterrupt if isinstance(original, KeyboardInterrupt) else TraceIPCIncomplete
                with self.assertRaises(expected):
                    client.trace(request_id='callback-fault', input_ids=[1,2,3], sample_id='row',
                        positions=[0], sites=[[0,'attn']], on_event=fail)
                self.assertTrue(client.incomplete)
                self.assertIs(client._active.failure, original)
                self.assertEqual(len(client._active.payload), 32)
                self.assertTrue(any(c.stage == 'trace_ipc_receive' and c.nbytes == 32
                                    for c in receiver_budget.snapshot().claims))
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not server.incomplete:
                    time.sleep(.01)
                self.assertTrue(server.incomplete)
                self.assertTrue(service.quarantined)
                self.assertTrue(any(c.stage == 'callback_exposure' and c.nbytes == 32
                                    for c in sender_budget.snapshot().claims))
                _FAULT_ROOTS.append((client, receiver_budget))

    def test_forged_ready_received_and_release_are_not_accepted(self):
        for stage, field, wrong in (('ready', 'generation', True),
                                    ('received', 'sequence', False),
                                    ('released', 'team_id', 'beta')):
            with self.subTest(stage=stage):
                server, service, budget = self.make_service()
                sock = socket.create_connection(server.address, timeout=10)
                sock.setblocking(False)
                deadline = time.monotonic() + 20
                try:
                    _send(sock, {'version':2, 'op':'trace', 'token':server.team_tokens['alpha'],
                        'team_id':'alpha', 'request_id':'forged', 'input_ids':[1,2,3],
                        'sample_id':'row', 'positions':[0], 'sites':[[0,'attn']]}, 8192, deadline)
                    self.assertEqual(_receive(sock, 8192, deadline)['op'], 'accepted')
                    event = _receive(sock, 8192, deadline)
                    self.assertEqual(event['op'], 'event')
                    if stage != 'ready':
                        _send(sock, _ack(event, 'ready'), 8192, deadline)
                        nbytes = struct.unpack('!I', _read_exact(sock, 4, deadline))[0]
                        self.assertEqual(nbytes, 32)
                        _read_exact(sock, nbytes, deadline)
                    if stage == 'released':
                        _send(sock, _ack(event, 'received'), 8192, deadline)
                    forged = _ack(event, stage)
                    forged[field] = wrong
                    _send(sock, forged, 8192, deadline)
                    end = time.monotonic() + 10
                    while time.monotonic() < end and not server.incomplete:
                        time.sleep(.01)
                    self.assertTrue(server.incomplete)
                    self.assertTrue(service.quarantined)
                    self.assertTrue(any(c.stage == 'callback_exposure' and c.nbytes == 32
                                        for c in budget.snapshot().claims))
                finally:
                    sock.close()

    def test_short_event_read_keeps_allocated_receiver_buffer_and_claim(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        address = listener.getsockname()
        broker, generation, transfer, event_id = 'a'*32, 'b'*32, 'c'*32, 'd'*32
        header = _wire_event(broker, generation, transfer, event_id, 'short')
        failures = []
        def fake_server():
            try:
                sock, _ = listener.accept()
                sock.setblocking(False)
                with sock:
                    end = time.monotonic() + 20
                    _receive(sock, 8192, end)
                    _send(sock, {'version':2, 'op':'accepted', 'broker_id':broker,
                        'process_generation':generation, 'transfer_id':transfer,
                        'team_id':'alpha', 'request_id':'short'}, 8192, end)
                    _send(sock, header, 8192, end)
                    self.assertEqual(_receive(sock, 8192, end), _ack(header, 'ready'))
                    _send_bytes(sock, struct.pack('!I', 32), end)
                    _send_bytes(sock, b'12345', end)
            except BaseException as error:
                failures.append(error)
            finally:
                listener.close()
        thread = threading.Thread(target=fake_server, daemon=True)
        thread.start()
        budget = _budget()
        client = TraceIPCClient(address, token='secret', team_id='alpha', broker_id=broker,
            process_generation=generation, budget=budget, max_event_bytes=32, max_trace_bytes=64, timeout=20)
        seen = []
        with self.assertRaises(TraceIPCIncomplete):
            client.trace(request_id='short', input_ids=[1,2,3], sample_id='row',
                         positions=[0], sites=[[0,'attn']], on_event=lambda *_: seen.append(1))
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(seen, [])
        self.assertTrue(client.incomplete)
        self.assertEqual(len(client._active.payload), 32)
        self.assertEqual(client._active.payload[:5], b'12345')
        self.assertTrue(any(c.stage == 'trace_ipc_receive' and c.nbytes == 32
                            for c in budget.snapshot().claims))
        _FAULT_ROOTS.append((client, budget))

    def test_forged_completion_types_do_not_publish_a_receipt(self):
        for field, wrong in (('event_count', True), ('version', 2.0)):
            with self.subTest(field=field):
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.bind(('127.0.0.1', 0))
                listener.listen(1)
                address = listener.getsockname()
                broker, generation, transfer, event_id = 'a'*32, 'b'*32, 'c'*32, 'd'*32
                header = _wire_event(broker, generation, transfer, event_id, 'fake')
                failures = []
                def fake_server():
                    try:
                        sock, _ = listener.accept()
                        sock.setblocking(False)
                        with sock:
                            end = time.monotonic() + 20
                            _receive(sock, 8192, end)
                            _send(sock, {'version':2, 'op':'accepted', 'broker_id':broker,
                                'process_generation':generation, 'transfer_id':transfer,
                                'team_id':'alpha', 'request_id':'fake'}, 8192, end)
                            _send(sock, header, 8192, end)
                            self.assertEqual(_receive(sock, 8192, end), _ack(header, 'ready'))
                            _send_bytes(sock, struct.pack('!I', 32), end)
                            _send_bytes(sock, bytes(32), end)
                            self.assertEqual(_receive(sock, 8192, end), _ack(header, 'received'))
                            self.assertEqual(_receive(sock, 8192, end), _ack(header, 'released'))
                            complete = {'version':2, 'op':'complete', 'broker_id':broker,
                                'process_generation':generation, 'job_id':1, 'generation':1,
                                'transfer_id':transfer, 'team_id':'alpha', 'request_id':'fake',
                                'run_id':'run-'+broker+'-1', 'event_count':1, 'trace_bytes':32}
                            complete[field] = wrong
                            _send(sock, complete, 8192, end)
                    except BaseException as error:
                        failures.append(error)
                    finally:
                        listener.close()
                thread = threading.Thread(target=fake_server, daemon=True)
                thread.start()
                budget = _budget()
                client = TraceIPCClient(address, token='secret', team_id='alpha', broker_id=broker,
                    process_generation=generation, budget=budget, max_event_bytes=32,
                    max_trace_bytes=64, timeout=20)
                seen = []
                with self.assertRaises(TraceIPCIncomplete):
                    client.trace(request_id='fake', input_ids=[1,2,3], sample_id='row',
                        positions=[0], sites=[[0,'attn']], on_event=lambda *_: seen.append(1))
                thread.join(10)
                self.assertFalse(thread.is_alive())
                self.assertEqual(failures, [])
                self.assertEqual(seen, [1])
                self.assertTrue(client.incomplete)
                with self.assertRaises(TraceIPCIncomplete):
                    client.recover()
                self.assertTrue(any(c.stage == 'trace_ipc_control' for c in budget.snapshot().claims))
                _FAULT_ROOTS.append((client, budget))

    def test_lost_final_ack_leaves_sender_incomplete_despite_local_completion(self):
        server, service, server_budget = self.make_service()
        client, client_budget = self.client(server, 'alpha')
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
            self.assertEqual((result.event_count, result.trace_bytes), (1, 32))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not server.incomplete:
                time.sleep(.01)
            self.assertTrue(server.incomplete)
        self.assertFalse(client.incomplete)
        self.assertEqual(client_budget.snapshot().used_bytes, 0)
        self.assertFalse(service.quarantined)
        self.assertTrue(any(c.stage == 'trace_ipc_input' for c in server_budget.snapshot().claims))
        self.assertTrue(any(c.stage == 'trace_ipc_control' for c in server_budget.snapshot().claims))
        self.assertFalse(server.close(timeout=5))

    def test_close_timeout_does_not_wait_for_a_blocked_send_or_busy_lock(self):
        server, _service, _sender_budget = self.make_service()
        client, _client_budget = self.client(server, 'alpha')
        entered, release = threading.Event(), threading.Event()
        outcome = []
        from looking_glass import trace_ipc_v2 as module
        original = module._send
        def blocked_send(sock, message, limit, deadline):
            if message.get('op') == 'accepted':
                entered.set()
                if not release.wait(10):
                    raise TimeoutError('test did not release send')
            return original(sock, message, limit, deadline)
        def run():
            try:
                outcome.append(client.trace(request_id='close-busy', input_ids=[1,2,3],
                    sample_id='row', positions=[0], sites=[[0,'attn']], on_event=lambda *_: None))
            except BaseException as error:
                outcome.append(error)
        thread = threading.Thread(target=run)
        try:
            with patch.object(module, '_send', blocked_send):
                thread.start()
                self.assertTrue(entered.wait(10))
                start = time.monotonic()
                self.assertFalse(server.close(timeout=.05))
                self.assertLess(time.monotonic() - start, .5)
                release.set()
                thread.join(15)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(outcome), 1)
            self.assertEqual(outcome[0].event_count, 1)
            self.assertTrue(server.close(timeout=5))
        finally:
            release.set()
            if thread.ident is not None:
                thread.join(15)

        # A concurrent lock holder must also be bounded by the same deadline.
        other, _service, _budget = self.make_service()
        locked, unlock = threading.Event(), threading.Event()
        def hold_lock():
            with other._lock:
                locked.set()
                unlock.wait(10)
        holder = threading.Thread(target=hold_lock)
        try:
            holder.start()
            self.assertTrue(locked.wait(5))
            start = time.monotonic()
            self.assertFalse(other.close(timeout=.05))
            self.assertLess(time.monotonic() - start, .5)
        finally:
            unlock.set()
            holder.join(5)
        self.assertTrue(other.close(timeout=5))

    def test_fault_latches_during_a_previously_authorized_stalled_send(self):
        server, service, sender_budget = self.make_service()
        client, receiver_budget = self.client(server, 'alpha')
        entered, release = threading.Event(), threading.Event()
        outcome, accepted_sent = [], []
        from looking_glass import trace_ipc_v2 as module
        original = module._send
        def blocked_send(sock, message, limit, deadline):
            if message.get('op') == 'accepted':
                entered.set()
                if not release.wait(10):
                    raise TimeoutError('test did not release send')
                original(sock, message, limit, deadline)
                accepted_sent.append(True)
                return
            return original(sock, message, limit, deadline)
        def run():
            try:
                client.trace(request_id='ordered-fault', input_ids=[1,2,3], sample_id='row',
                    positions=[0], sites=[[0,'attn']], on_event=lambda *_: None)
            except BaseException as error:
                outcome.append(error)
        thread = threading.Thread(target=run)
        try:
            with patch.object(module, '_send', blocked_send):
                thread.start()
                self.assertTrue(entered.wait(10))
                with socket.create_connection(server.address, timeout=5) as peer:
                    peer.sendall(struct.pack('!I', 8193))
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and not server.incomplete:
                        time.sleep(.01)
                    self.assertTrue(server.incomplete)
                    with self.assertRaises(TraceIPCIncomplete):
                        server._publish(None, {'op':'must-not-send'}, deadline)
                release.set()
                thread.join(15)
            self.assertFalse(thread.is_alive())
            self.assertEqual(accepted_sent, [True])
            self.assertEqual(len(outcome), 1)
            self.assertIsInstance(outcome[0], TraceIPCIncomplete)
            self.assertEqual(service._generation, 0)
            self.assertTrue(client.incomplete)
            self.assertTrue(any(c.stage == 'trace_ipc_input' for c in sender_budget.snapshot().claims))
            self.assertTrue(any(c.stage == 'trace_ipc_control' for c in receiver_budget.snapshot().claims))
            _FAULT_ROOTS.append((client, receiver_budget))
        finally:
            release.set()
            if thread.ident is not None:
                thread.join(15)

    def test_unbounded_request_values_reject_before_encoding(self):
        server, _service, _server_budget = self.make_service()
        client, budget = self.client(server, 'alpha')
        with self.assertRaises(ValueError):
            client.trace(request_id='bad', input_ids=[1,2,3], sample_id='row',
                         positions=[0], sites=[{'x':'a' * 1_000_000}], on_event=lambda *_: None)
        with self.assertRaises(ValueError):
            client.trace(request_id='bad', input_ids=[10**1000], sample_id='row',
                         positions=[0], sites=[[0,'attn']], on_event=lambda *_: None)
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertFalse(client.incomplete)

    def test_control_minimum_fits_maximum_bounded_request_and_event_header(self):
        server, service, _server_budget = self.make_service()
        with self.assertRaises(ValueError):
            TraceIPCServer(service, max_control_bytes=1024, max_event_bytes=32, max_trace_bytes=64)
        with self.assertRaises(ValueError):
            TraceIPCClient(server.address, token=server.team_tokens['alpha'], team_id='alpha',
                broker_id=server.broker_id, process_generation=server.process_generation,
                budget=_budget(), max_control_bytes=1024, max_event_bytes=32, max_trace_bytes=64)
        maximum_request = {'version':2, 'op':'trace', 'token':'\x00'*128,
            'team_id':'t'*64, 'request_id':'r'*64, 'input_ids':[2**31-1]*128,
            'sample_id':'s'*64, 'positions':list(range(128)),
            'sites':[[2**31-1-i//3, ('block','attn','mlp')[i%3]] for i in range(256)]}
        encoded = lambda message: json.dumps(message, allow_nan=False, ensure_ascii=True,
                                             separators=(',', ':')).encode('utf-8')
        self.assertLessEqual(len(encoded(maximum_request)), 8192)
        header = _wire_event('a'*32, 'b'*32, 'c'*32, 'd'*32, 'r'*64)
        job_id, layer = 2**63-1, 2**31-1
        header.update(job_id=job_id, generation=job_id, team_id='t'*64, payload_bytes=2**24)
        event = header['event']
        event['identity'].update(run_id='run-'+'a'*32+'-'+str(job_id),
            job_id='lg-'+'a'*32+'-'+str(job_id), sample_id='s'*64)
        event.update(sequence=255, layer_index=layer,
            module_path=f'transformer.h.{layer}.attn', shape=[1,128,2**22],
            token_positions=list(range(128)), saved_shape=[128,2**22])
        self.assertLessEqual(len(encoded(header)), 8192)
        self.assertLessEqual(len(encoded(_ack(header, 'released'))), 8192)
        complete = {'version':2, 'op':'complete_received', 'broker_id':'a'*32,
            'process_generation':'b'*32, 'job_id':job_id, 'generation':job_id,
            'transfer_id':'c'*32, 'team_id':'t'*64, 'request_id':'r'*64,
            'run_id':'run-'+'a'*32+'-'+str(job_id), 'event_count':256, 'trace_bytes':2**26}
        self.assertLessEqual(len(encoded(complete)), 8192)

    def test_non_float32_model_rejected_before_server_starts(self):
        with self.assertRaisesRegex(ValueError, 'float32'):
            self.make_service(dtype=torch.float64)

    def test_caller_mutation_cannot_change_the_validated_request_snapshot(self):
        server, _service, sender_budget = self.make_service()
        client, receiver_budget = self.client(server, 'alpha')
        ids, positions, sites = [1,2,3], [0], [[0,'attn']]
        seen = []
        from looking_glass import trace_ipc_v2 as module
        original = module.socket.create_connection
        def mutate_then_connect(*args, **kwargs):
            # Encoding the mutated row would exceed the entire control claim.
            ids.extend([1] * 300_000)
            positions[0] = 2
            sites[0][:] = [999999, 'mlp']
            return original(*args, **kwargs)
        with patch.object(module.socket, 'create_connection', mutate_then_connect):
            result = client.trace(request_id='stable-snapshot', input_ids=ids, sample_id='row',
                positions=positions, sites=sites,
                on_event=lambda event, _payload: seen.append((event.event.token_positions,
                                                               event.event.module_kind)))
        self.assertEqual((result.event_count, result.trace_bytes), (1,32))
        self.assertEqual(seen, [((0,), 'attn')])
        self.assertEqual(len(ids), 300_003)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and server.active_count:
            time.sleep(.01)
        self.assertEqual(sender_budget.snapshot().used_bytes, 0)
        self.assertEqual(receiver_budget.snapshot().used_bytes, 0)

    def test_existing_oversize_lists_reject_before_claim_or_connection(self):
        server, _service, _sender_budget = self.make_service()
        client, budget = self.client(server, 'alpha')
        from looking_glass import trace_ipc_v2 as module
        with patch.object(module.socket, 'create_connection') as connect:
            with self.assertRaises(ValueError):
                client.trace(request_id='huge', input_ids=[1] * 1_000_000,
                    sample_id='row', positions=[0], sites=[[0,'attn']], on_event=lambda *_: None)
        connect.assert_not_called()
        self.assertFalse(client.incomplete)
        self.assertEqual(budget.snapshot().used_bytes, 0)

if __name__ == '__main__':
    unittest.main()
