import gc
import inspect
import subprocess
import sys
import textwrap
import threading
import time
import weakref
from unittest.mock import patch

import pytest
import torch

from looking_glass.broker import Broker, BrokerShutdownError, BrokerUnavailable, RequestState
from looking_glass.fair_adapter import ExperimentalFairAdapter
from looking_glass.ipc import IPCClient, IPCRejected
from looking_glass.ipc_custody import IPCCustodyKey
from looking_glass.ipc_fair_bridge import FairIPCBridge, ForwardCodec
from looking_glass.resident_worker import ForwardRequest, ForwardResult


EPOCH = 'a' * 32


class Cancelled(Exception):
    pass


class CPUWorker:
    cancellation_exception = Cancelled
    max_input_bytes = 64
    max_output_bytes = 64

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.seen = []
        self.poisoned = False

    def snapshot_request(self, request):
        return ForwardRequest(request.request_id,
                              {k: v.clone() for k, v in request.inputs.items()}, request.seed)

    def execute(self, request, cancel_event):
        self.seen.append(int(request.inputs['input_ids'][0, 0]))
        self.started.set()
        self.release.wait()
        if cancel_event.is_set():
            raise Cancelled()
        return ForwardResult(request.request_id, request.inputs['input_ids'].float().unsqueeze(-1))

    def close(self):
        pass


def make_stack(codec_max_encoded_bytes=512, request_validator=None, max_connections=4):
    worker = CPUWorker()
    broker = Broker(worker, max_pending=2, max_pending_per_team=2)
    codec = ForwardCodec(max_batch=1, max_tokens=4, max_encoded_bytes=codec_max_encoded_bytes,
                         max_decoded_bytes=512)
    target = broker.enable_fair_handoff(
        ('A', 'B'), decoder=codec.decode, max_item_bytes=512,
        max_decoded_bytes=512, preparation_bytes=2048,
        max_preparation_bytes=4096, max_result_bytes=128, max_unacknowledged=4,
    )
    credentials = {'A': b'alice-key', 'B': b'bob-key'}
    verifier = lambda token: {b'alice-key': 'alice', b'bob-key': 'bob'}[token]
    adapter = ExperimentalFairAdapter(
        {'alice': 'A', 'bob': 'B'}, verifier=verifier, target=target,
        max_waiters=4, max_waiters_per_team=3,
        max_waiting_bytes=2048, max_item_bytes=512,
    )
    bridge = FairIPCBridge(adapter, codec=codec, team_credentials=credentials,
                           custody_epoch=EPOCH, max_connections=max_connections,
                           max_request_bytes=2048, max_response_bytes=4096,
                           connection_commitment_bytes=2_000_000,
                           request_validator=request_validator)
    return worker, broker, adapter, bridge


def request(name, value):
    return ForwardRequest(name, {'input_ids': torch.tensor([[value]], dtype=torch.long)}, 0)


def eventually(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


def submit(bridge, sequence, team, name, value):
    key = IPCCustodyKey(EPOCH, sequence)
    bridge.reserve(key)
    bridge.authenticate(key, team, name)
    handle = bridge.submit_registered(key, team, request(name, value))
    return key, handle


def notify_closed(bridge, key):
    thread = threading.Thread(target=bridge.handler_closed, args=(key,))
    thread.start()
    thread.join(2)
    assert not thread.is_alive()


def test_loopback_two_team_results_retain_identity_and_release_custody():
    worker, broker, _, bridge = make_stack()
    bridge.start()
    server = bridge.make_server(io_timeout=2, job_timeout=3).start()
    try:
        for team, value in [('A', 11), ('B', 22)]:
            client = IPCClient(server.address, server.team_tokens[team], io_timeout=2)
            handle = client.submit(team_id=team, request_id='req-' + team,
                                   inputs={'input_ids': [[value]]})
            response = handle.wait(3)
            assert response['status'] == 'succeeded'
            assert response['team_id'] == team
            assert response['request_id'] == 'req-' + team
            assert response['broker_id'] == bridge.broker_id
            assert response['result']['logits'] == [float(value)]
        eventually(lambda: bridge.committed_bytes == 0)
        assert worker.seen == [11, 22]
    finally:
        assert server.close(timeout=3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_unstarted_waiter_cancellation_is_owner_scoped_and_peer_runs():
    _, broker, _, bridge = make_stack()
    ka, first = submit(bridge, 1, 'A', 'a', 3)
    kb, second = submit(bridge, 2, 'B', 'b', 4)
    try:
        assert first.cancel()
        assert first.wait(timeout=0).state is RequestState.CANCELLED
        notify_closed(bridge, ka)
        bridge.start()
        outcome = second.wait(timeout=3)
        assert outcome.state is RequestState.SUCCEEDED
        assert outcome.team_id == 'B'
        assert outcome.result.logits.item() == 4
        assert outcome.submitted_ns == second._record.submitted_ns
        assert outcome.submitted_ns <= outcome.ready_ns <= outcome.finished_ns
        del outcome
        notify_closed(bridge, kb)
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_closed_handler_does_not_release_claim_while_worker_runs():
    worker, broker, _, bridge = make_stack()
    worker.release.clear()
    bridge.start()
    key, handle = submit(bridge, 1, 'A', 'a', 5)
    try:
        assert worker.started.wait(2)
        notify_closed(bridge, key)
        assert bridge.committed_bytes == 2_000_000
        worker.release.set()
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        worker.release.set()
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_admitted_waiters_rotate_by_team_without_reordering_each_team():
    worker, broker, _, bridge = make_stack()
    pairs = [submit(bridge, i, team, str(i), value)
             for i, team, value in [(1, 'A', 1), (2, 'A', 2), (3, 'B', 3)]]
    bridge.start()
    try:
        for key, handle in pairs:
            outcome = handle.wait(timeout=3)
            assert outcome.state is RequestState.SUCCEEDED
            del outcome
            notify_closed(bridge, key)
        eventually(lambda: bridge.committed_bytes == 0)
        assert worker.seen == [1, 3, 2]
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_authenticated_team_cannot_be_substituted_at_submit():
    _, broker, _, bridge = make_stack()
    key = IPCCustodyKey(EPOCH, 1)
    bridge.reserve(key)
    bridge.authenticate(key, 'A', 'a')
    with pytest.raises(PermissionError):
        bridge.submit_registered(key, 'B', request('a', 1))
    notify_closed(bridge, key)
    bridge.start()
    eventually(lambda: bridge.committed_bytes == 0)
    assert bridge.close(timeout=3)
    assert broker.close(timeout=3)


def test_cancel_after_target_completion_preserves_success():
    _, broker, _, bridge = make_stack()
    bridge.start()
    key, handle = submit(bridge, 1, 'A', 'a', 5)
    try:
        outcome = handle.wait(timeout=3)
        assert outcome.state is RequestState.SUCCEEDED
        del outcome
        assert not handle.cancel()
        outcome = handle.wait(timeout=0)
        assert outcome.state is RequestState.SUCCEEDED
        del outcome
        notify_closed(bridge, key)
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_deferred_cancel_does_not_rewrite_a_success_that_finishes_first():
    worker, broker, adapter, bridge = make_stack()
    entered, resume = threading.Event(), threading.Event()
    original_snapshot = worker.snapshot_request

    def delayed_snapshot(request_value):
        entered.set()
        assert resume.wait(3)
        return original_snapshot(request_value)

    worker.snapshot_request = delayed_snapshot
    bridge.start()
    key, handle = submit(bridge, 1, 'A', 'a', 5)
    try:
        assert entered.wait(2)
        assert handle.cancel()  # Intent recorded while the receipt is unpublished.
        resume.set()
        record = handle._record
        eventually(lambda: adapter.recover_receipt(record.ticket) is not None)
        underlying = adapter.recover_receipt(record.ticket)
        actual = underlying.wait(timeout=3)
        assert actual.state is RequestState.SUCCEEDED
        del actual, underlying
        outcome = handle.wait(timeout=3)
        assert outcome.state is RequestState.SUCCEEDED
        del outcome
        notify_closed(bridge, key)
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        resume.set()
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_recorded_closed_notice_is_not_disposal_until_handler_returns():
    worker, broker, _, bridge = make_stack()
    bridge.start()
    server = bridge.make_server(io_timeout=2, job_timeout=3).start()
    entered, release = threading.Event(), threading.Event()
    original = bridge.handler_closed

    def interrupted_notice(key):
        original(key)
        entered.set()
        assert release.wait(3)
        raise RuntimeError('notice return was interrupted')

    bridge.handler_closed = interrupted_notice
    try:
        client = IPCClient(server.address, server.team_tokens['A'], io_timeout=2)
        handle = client.submit(team_id='A', request_id='notice', inputs={'input_ids': [[9]]})
        assert handle.wait(3)['status'] == 'succeeded'
        assert entered.wait(2)
        time.sleep(.1)
        assert bridge.committed_bytes == 2_000_000
        release.set()
        eventually(lambda: server.custody_quarantined)
        eventually(lambda: bridge.quarantined)
        assert bridge.committed_bytes == 2_000_000
    finally:
        release.set()
        assert server.close(timeout=3)
        assert not bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_codec_bounds_tensor_expansion_before_python_list_allocation():
    codec = ForwardCodec(max_batch=1, max_tokens=1000, max_encoded_bytes=32, max_decoded_bytes=8000)
    request_value = ForwardRequest('r', {'input_ids': torch.ones((1, 1000), dtype=torch.long)}, 0)
    with patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('unbounded expansion')):
        with pytest.raises(ValueError):
            codec.encode(request_value)


def test_codec_rejects_excess_decoded_storage_before_tensor_creation():
    codec = ForwardCodec(max_batch=1, max_tokens=2, max_encoded_bytes=256, max_decoded_bytes=8)
    payload = b'{"request_id":"r","seed":0,"inputs":{"input_ids":[[1,2]]}}'
    with patch('torch.tensor', side_effect=AssertionError('oversized tensor allocation')):
        with pytest.raises(ValueError):
            codec.decode(payload)


def test_bridge_rejects_target_with_smaller_decoded_envelope():
    worker = CPUWorker()
    broker = Broker(worker, max_pending=1, max_pending_per_team=1)
    codec = ForwardCodec(max_batch=1, max_tokens=2, max_encoded_bytes=256, max_decoded_bytes=16)
    target = broker.enable_fair_handoff(('A',), decoder=codec.decode, max_item_bytes=256,
                                        max_decoded_bytes=8, preparation_bytes=1024,
                                        max_preparation_bytes=1024, max_result_bytes=128,
                                        max_unacknowledged=1)
    adapter = ExperimentalFairAdapter({'alice': 'A'}, verifier=lambda x: 'alice', target=target,
                                      max_waiters=1, max_waiters_per_team=1,
                                      max_waiting_bytes=256, max_item_bytes=256)
    try:
        with pytest.raises(ValueError):
            FairIPCBridge(adapter, codec=codec, team_credentials={'A': b'a'}, custody_epoch=EPOCH,
                          max_connections=1, max_request_bytes=256, max_response_bytes=512,
                          connection_commitment_bytes=1_000_000)
    finally:
        adapter.close()
        assert broker.close(timeout=3)


def test_dispatch_thread_start_failure_latches_quarantine():
    _, broker, _, bridge = make_stack()
    try:
        with patch.object(threading.Thread, 'start', side_effect=RuntimeError('thread start failed')):
            with pytest.raises(BrokerUnavailable):
                bridge.start()
        assert bridge.quarantined
        assert not bridge.close(timeout=0)
    finally:
        assert broker.close(timeout=3)


def test_ambiguous_submission_roots_private_traceback_and_sanitizes_public_error():
    _, broker, _, bridge = make_stack()
    key = IPCCustodyKey(EPOCH, 1)
    bridge.reserve(key)
    bridge.authenticate(key, 'A', 'a')
    refs = []

    def fail_encode(_request):
        temporary = torch.ones(10)
        refs.append(weakref.ref(temporary))
        raise MemoryError('private failure')

    try:
        with patch.object(bridge._codec, 'encode', side_effect=fail_encode):
            with pytest.raises(BrokerUnavailable) as caught:
                bridge.submit_registered(key, 'A', request('a', 1))
        assert bridge.quarantined
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
        caught.value.__traceback__ = None
        gc.collect()
        assert refs[0]() is not None
        assert bridge.committed_bytes == 2_000_000
    finally:
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)


def test_dispatch_exception_retains_private_temporary():
    _, broker, adapter, bridge = make_stack()
    refs = []

    def fail_transfer():
        temporary = torch.ones(10)
        refs.append(weakref.ref(temporary))
        raise MemoryError('private dispatch failure')

    try:
        with patch.object(adapter, 'transfer_next', side_effect=fail_transfer):
            bridge.start()
            eventually(lambda: bridge.quarantined)
        gc.collect()
        assert refs[0]() is not None
    finally:
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)


def test_waiter_limit_yields_clean_failure_without_poisoning_other_jobs():
    _, broker, _, bridge = make_stack()
    pairs = [submit(bridge, i, 'A', str(i), i) for i in range(1, 5)]
    try:
        refused = pairs[-1][1].wait(timeout=0)
        assert refused.state is RequestState.FAILED
        assert refused.error_type == 'QueueFull'
        assert not bridge.quarantined
        del refused
        notify_closed(bridge, pairs[-1][0])
        bridge.start()
        for key, handle in pairs[:-1]:
            outcome = handle.wait(timeout=3)
            assert outcome.state is RequestState.SUCCEEDED
            del outcome
            notify_closed(bridge, key)
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_codec_limit_failure_does_not_quarantine_existing_service():
    _, broker, _, bridge = make_stack(codec_max_encoded_bytes=32)
    key, handle = submit(bridge, 1, 'A', 'a', 1)
    try:
        outcome = handle.wait(timeout=0)
        assert outcome.state is RequestState.FAILED
        assert outcome.error_type == 'InvalidRequest'
        assert not bridge.quarantined
        del outcome
        notify_closed(bridge, key)
        bridge.start()
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_bridge_rejects_adapter_item_limit_below_codec_limit():
    worker = CPUWorker()
    broker = Broker(worker, max_pending=1, max_pending_per_team=1)
    codec = ForwardCodec(max_batch=1, max_tokens=2, max_encoded_bytes=256, max_decoded_bytes=16)
    target = broker.enable_fair_handoff(('A',), decoder=codec.decode, max_item_bytes=256,
                                        max_decoded_bytes=16, preparation_bytes=1024,
                                        max_preparation_bytes=1024, max_result_bytes=128,
                                        max_unacknowledged=1)
    adapter = ExperimentalFairAdapter({'alice': 'A'}, verifier=lambda x: 'alice', target=target,
                                      max_waiters=1, max_waiters_per_team=1,
                                      max_waiting_bytes=64, max_item_bytes=64)
    try:
        with pytest.raises(ValueError):
            FairIPCBridge(adapter, codec=codec, team_credentials={'A': b'a'}, custody_epoch=EPOCH,
                          max_connections=1, max_request_bytes=256, max_response_bytes=512,
                          connection_commitment_bytes=1_000_000)
    finally:
        adapter.close()
        assert broker.close(timeout=3)


def test_expected_snapshot_rejection_fails_its_request_without_poisoning_peer():
    worker, broker, _, bridge = make_stack()
    original = worker.snapshot_request

    def validated_snapshot(value):
        if value.inputs['input_ids'][0, 0].item() == 99:
            raise ValueError('model-specific token rejected')
        return original(value)

    worker.snapshot_request = validated_snapshot
    first = submit(bridge, 1, 'A', 'invalid', 99)
    second = submit(bridge, 2, 'B', 'valid', 7)
    bridge.start()
    try:
        outcome = first[1].wait(timeout=3)
        assert outcome.state is RequestState.FAILED
        assert outcome.error_type == 'HandoffFailure'
        del outcome
        outcome = second[1].wait(timeout=3)
        assert outcome.state is RequestState.SUCCEEDED
        assert outcome.result.logits.item() == 7
        assert not bridge.quarantined
        del outcome
        notify_closed(bridge, first[0])
        notify_closed(bridge, second[0])
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_model_domain_validator_rejects_bad_wire_input_and_keeps_peer_healthy():
    def validate(owned_request):
        if int(owned_request.inputs['input_ids'].max()) >= 10 or owned_request.seed != 0:
            raise ValueError('outside frozen model domain')

    worker, broker, _, bridge = make_stack(request_validator=validate)
    bridge.start()
    server = bridge.make_server(io_timeout=2, job_timeout=3).start()
    try:
        client_a = IPCClient(server.address, server.team_tokens['A'], io_timeout=2)
        bad = client_a.submit(team_id='A', request_id='bad', inputs={'input_ids': [[99]]})
        assert bad.wait(3)['status'] == 'failed'
        client_b = IPCClient(server.address, server.team_tokens['B'], io_timeout=2)
        good = client_b.submit(team_id='B', request_id='good', inputs={'input_ids': [[7]]})
        assert good.wait(3)['status'] == 'succeeded'
        assert worker.seen == [7]
        assert not bridge.quarantined and not server.custody_quarantined
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert server.close(timeout=3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_decoder_traceback_retains_partial_cpu_tensors_on_allocation_failure():
    codec = ForwardCodec(max_batch=1, max_tokens=1, max_encoded_bytes=256, max_decoded_bytes=16)
    payload = b'{"request_id":"r","seed":0,"inputs":{"input_ids":[[1]],"attention_mask":[[1]]}}'
    real_tensor = torch.tensor
    refs = []

    def allocate(*args, **kwargs):
        if refs:
            raise MemoryError('second tensor allocation failed')
        tensor = real_tensor(*args, **kwargs)
        refs.append(weakref.ref(tensor))
        return tensor

    with patch('torch.tensor', side_effect=allocate):
        try:
            codec.decode(payload)
        except MemoryError as error:
            retained = error
    gc.collect()
    assert refs[0]() is not None
    assert retained.__traceback__ is not None


def test_validator_uses_private_snapshot_even_if_source_changes_later():
    entered, resume = threading.Event(), threading.Event()

    def validate(snapshot):
        entered.set()
        assert resume.wait(3)
        assert snapshot.inputs['input_ids'].item() == 3

    _, broker, _, bridge = make_stack(request_validator=validate)
    key = IPCCustodyKey(EPOCH, 1)
    bridge.reserve(key)
    bridge.authenticate(key, 'A', 'a')
    source = request('a', 3)
    handles, errors = [], []

    def produce():
        try:
            handles.append(bridge.submit_registered(key, 'A', source))
        except BaseException as error:
            errors.append(error)

    producer = threading.Thread(target=produce)
    producer.start()
    try:
        assert entered.wait(2)
        source.inputs['input_ids'].fill_(9)
        resume.set()
        producer.join(3)
        assert not producer.is_alive() and not errors
        bridge.start()
        outcome = handles[0].wait(timeout=3)
        assert outcome.result.logits.item() == 3
        del outcome
        notify_closed(bridge, key)
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        resume.set()
        producer.join(3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_detached_worker_saturates_custody_with_clean_busy_rejection():
    worker, broker, _, bridge = make_stack(max_connections=1)
    worker.release.clear()
    bridge.start()
    server = bridge.make_server(io_timeout=2, job_timeout=3).start()
    try:
        client = IPCClient(server.address, server.team_tokens['A'], io_timeout=2)
        first = client.submit(team_id='A', request_id='first', inputs={'input_ids': [[1]]})
        assert worker.started.wait(2)
        first.close()
        eventually(lambda: not server._handlers)
        assert bridge.committed_bytes == 2_000_000
        with pytest.raises(IPCRejected) as caught:
            client.submit(team_id='A', request_id='busy', inputs={'input_ids': [[2]]})
        assert caught.value.code == 'busy'
        assert not server.custody_quarantined and not bridge.quarantined
        worker.release.set()
        eventually(lambda: bridge.committed_bytes == 0)
        third = client.submit(team_id='A', request_id='third', inputs={'input_ids': [[3]]})
        assert third.wait(3)['status'] == 'succeeded'
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        worker.release.set()
        assert server.close(timeout=3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_duplicate_key_is_not_misreported_as_no_commit_capacity_denial():
    _, broker, _, bridge = make_stack(max_connections=1)
    key = IPCCustodyKey(EPOCH, 1)
    bridge.reserve(key)
    try:
        with pytest.raises(ValueError):
            bridge.reserve(key)
        assert bridge.committed_bytes == 2_000_000
        notify_closed(bridge, key)
        bridge.start()
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_server_quarantine_is_rechecked_after_notice_thread_exit():
    _, broker, _, bridge = make_stack()
    key = IPCCustodyKey(EPOCH, 1)
    bridge.reserve(key)
    server = bridge.make_server(io_timeout=2, job_timeout=3)
    bridge.handler_closed(key)

    class ExitingWithFailure:
        def is_alive(self):
            # Models a post-notice IPC cleanup fault which occurs after the
            # collector's first health check but before it sees thread exit.
            server._custody_quarantine.set()
            return False

    bridge._records[key].notice_thread = ExitingWithFailure()
    try:
        with pytest.raises(BrokerUnavailable):
            bridge._collect_closed()
        assert bridge.quarantined
        assert bridge.committed_bytes == 2_000_000
        assert key in bridge._records
    finally:
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)


def test_codec_rejects_a_small_view_with_oversized_backing_storage():
    codec = ForwardCodec(max_batch=1, max_tokens=1, max_encoded_bytes=256, max_decoded_bytes=8)
    backing = torch.zeros(1000, dtype=torch.long)
    view = backing[:1].view(1, 1)
    assert view.is_contiguous() and view.numel() == 1
    with pytest.raises(ValueError):
        codec.snapshot(ForwardRequest('r', {'input_ids': view}, 0))


@pytest.mark.parametrize(('method', 'last_mutation'), [
    ('_submit', 'record.proxy = proxy'),
    ('_cancel_inner', 'record.cancel_requested = True'),
    ('_authenticate', 'record.team, record.request_id = team_id, request_id'),
    ('_handler_closed', 'record.handler_closed = True'),
])
def test_interrupted_transition_latches_before_partial_record_mutation(method, last_mutation):
    # A subprocess bounds the CPython lock-interruption failure mode.
    script = textwrap.dedent(f'''
        import inspect, sys
        sys.path.insert(0, 'tests')
        from test_ipc_fair_bridge import make_stack, request, EPOCH
        from looking_glass.ipc_custody import IPCCustodyKey
        worker, broker, adapter, bridge = make_stack()
        key = IPCCustodyKey(EPOCH, 1)
        bridge.reserve(key)
        if {method!r} != '_authenticate':
            bridge.authenticate(key, 'A', 'a')
        handle = None
        if {method!r} == '_cancel_inner':
            handle = bridge.submit_registered(key, 'A', request('a', 1))
        function = getattr(type(bridge), {method!r})
        source, first = inspect.getsourcelines(function)
        index = next(i for i, line in enumerate(source) if {last_mutation!r} in line)
        target = first + index + 1
        def trace(frame, event, arg):
            if frame.f_code is function.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                raise KeyboardInterrupt()
            return trace
        sys.settrace(trace)
        try:
            if {method!r} == '_authenticate':
                bridge.authenticate(key, 'A', 'a')
            elif {method!r} == '_handler_closed':
                bridge.handler_closed(key)
            elif handle is None:
                bridge.submit_registered(key, 'A', request('a', 1))
            else:
                handle.cancel()
        except KeyboardInterrupt:
            pass
        finally:
            sys.settrace(None)
        assert bridge.quarantined
    ''')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_lost_submission_return_is_recoverable_by_exact_connection_identity():
    _, broker, _, bridge = make_stack()
    key = IPCCustodyKey(EPOCH, 1)
    bridge.reserve(key)
    bridge.authenticate(key, 'A', 'a')
    code = type(bridge).submit_registered.__code__

    def trace(frame, event, arg):
        if frame.f_code is code and event == 'return':
            sys.settrace(None)
            raise KeyboardInterrupt('lost return')
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt):
            bridge.submit_registered(key, 'A', request('a', 4))
    finally:
        sys.settrace(None)
    try:
        with pytest.raises(PermissionError):
            bridge.recover_proxy(key, 'B', 'a')
        proxy = bridge.recover_proxy(key, 'A', 'a')
        assert proxy is bridge.recover_proxy(key, 'A', 'a')
        bridge.start()
        outcome = proxy.wait(timeout=3)
        assert outcome.state is RequestState.SUCCEEDED
        assert outcome.result.logits.item() == 4
        del outcome
        notify_closed(bridge, key)
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_notify_failure_during_submission_quarantines_rooted_request():
    _, broker, _, bridge = make_stack()
    key = IPCCustodyKey(EPOCH, 1)
    bridge.reserve(key)
    bridge.authenticate(key, 'A', 'a')
    try:
        with patch.object(bridge._condition, 'notify_all', side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                bridge.submit_registered(key, 'A', request('a', 4))
        assert bridge.quarantined
        assert bridge.committed_bytes == 2_000_000
    finally:
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)


def test_lost_cancel_return_keeps_queryable_cancellation_intent():
    _, broker, _, bridge = make_stack()
    key, proxy = submit(bridge, 1, 'A', 'a', 4)
    code = type(proxy).cancel.__code__

    def trace(frame, event, arg):
        if frame.f_code is code and event == 'return':
            sys.settrace(None)
            raise KeyboardInterrupt('lost cancel return')
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt):
            proxy.cancel()
    finally:
        sys.settrace(None)
    try:
        assert proxy.cancel_requested
        assert proxy.wait(timeout=0).state is RequestState.CANCELLED
        notify_closed(bridge, key)
        bridge.start()
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


@pytest.mark.parametrize('foreign_team', ['A', 'B'])
def test_foreign_broker_receipt_quarantines_without_admitting_new_jobs(foreign_team):
    worker, broker, adapter, bridge = make_stack()
    other = Broker(CPUWorker(), max_pending=2, max_pending_per_team=2)
    try:
        foreign = other.submit(foreign_team, request('same-id', 2))
        assert foreign.wait(timeout=3).result.logits.item() == 2
        _, proxy = submit(bridge, 1, 'A', 'same-id', 1)
        with patch.object(adapter, 'recover_receipt', return_value=foreign):
            with pytest.raises(BrokerUnavailable):
                proxy.wait(timeout=0)
        assert bridge.quarantined
        assert bridge.committed_bytes == 2_000_000
        assert proxy._record.target_handle is foreign
        assert not worker.seen
        with pytest.raises(BrokerUnavailable):
            bridge.reserve(IPCCustodyKey(EPOCH, 2))
    finally:
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)
        assert other.close(timeout=3)


def test_same_broker_duplicate_request_ids_cannot_swap_receipts():
    _, broker, adapter, bridge = make_stack()
    try:
        _, first = submit(bridge, 1, 'A', 'same-id', 1)
        assert adapter.transfer_next() is not None
        first_handle = adapter.recover_receipt(first._record.ticket)
        assert first_handle.wait(timeout=3).result.logits.item() == 1
        _, second = submit(bridge, 2, 'A', 'same-id', 2)
        assert adapter.transfer_next() is not None
        second_handle = adapter.recover_receipt(second._record.ticket)
        assert second_handle.wait(timeout=3).result.logits.item() == 2
        with patch.object(adapter, 'recover_receipt', return_value=second_handle):
            with pytest.raises(BrokerUnavailable):
                first.wait(timeout=0)
        assert bridge.quarantined
        assert bridge.committed_bytes == 4_000_000
        assert first._record.target_handle is second_handle
        with pytest.raises(BrokerUnavailable):
            bridge.reserve(IPCCustodyKey(EPOCH, 3))
    finally:
        assert not bridge.close(timeout=0)
        broker.close(timeout=3)


@pytest.mark.parametrize('path', ['wait', 'cancel', 'handoff_failure'])
def test_receipt_mismatch_quarantines_every_handoff_path(path):
    _, broker, adapter, bridge = make_stack()
    other = Broker(CPUWorker(), max_pending=2, max_pending_per_team=2)
    try:
        foreign = other.submit('A', request('same-id', 2))
        foreign.wait(timeout=3)
        _, proxy = submit(bridge, 1, 'A', 'same-id', 1)
        with patch.object(adapter, 'recover_receipt', return_value=foreign):
            with pytest.raises(BrokerUnavailable):
                if path == 'wait':
                    proxy.wait(timeout=0)
                elif path == 'cancel':
                    proxy.cancel()
                else:
                    bridge._handle_handoff_failure(proxy._record.ticket)
        assert bridge.quarantined
        assert proxy._record.target_handle is foreign
        assert bridge.committed_bytes == 2_000_000
        with pytest.raises(BrokerUnavailable):
            bridge.reserve(IPCCustodyKey(EPOCH, 2))
    finally:
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)
        assert other.close(timeout=3)


def test_cached_handle_is_revalidated_against_live_receipt_before_delivery():
    _, broker, adapter, bridge = make_stack()
    try:
        _, first = submit(bridge, 1, 'A', 'same-id', 1)
        assert adapter.transfer_next() is not None
        assert first.wait(timeout=3).result.logits.item() == 1
        _, second = submit(bridge, 2, 'A', 'same-id', 2)
        assert adapter.transfer_next() is not None
        second_handle = adapter.recover_receipt(second._record.ticket)
        assert second_handle.wait(timeout=3).result.logits.item() == 2
        with patch.object(adapter._target, 'recover_receipt', return_value=second_handle):
            with pytest.raises(BrokerUnavailable):
                first.wait(timeout=0)
        assert bridge.quarantined
        assert bridge.committed_bytes == 4_000_000
    finally:
        assert not bridge.close(timeout=0)
        broker.close(timeout=3)


def test_target_quarantine_prevents_receipt_delivery_and_new_admission():
    _, broker, adapter, bridge = make_stack()
    try:
        _, proxy = submit(bridge, 1, 'A', 'same-id', 1)
        assert adapter.transfer_next() is not None
        assert adapter.recover_receipt(proxy._record.ticket).wait(timeout=3)
        broker._quarantine_latched = True
        assert bridge.quarantined
        with pytest.raises(BrokerUnavailable):
            proxy.wait(timeout=0)
        with pytest.raises(BrokerUnavailable):
            bridge.reserve(IPCCustodyKey(EPOCH, 2))
        assert bridge.committed_bytes == 2_000_000
    finally:
        assert not bridge.close(timeout=0)
        with pytest.raises(BrokerShutdownError):
            broker.close(timeout=3)


@pytest.mark.parametrize('path', ['wait', 'cancel', 'handoff_failure'])
def test_missing_adapter_receipt_for_published_ticket_quarantines(path):
    _, broker, adapter, bridge = make_stack()
    try:
        _, proxy = submit(bridge, 1, 'A', 'same-id', 1)
        assert adapter.transfer_next() is not None
        actual = adapter.recover_receipt(proxy._record.ticket)
        assert actual.wait(timeout=3).result.logits.item() == 1
        with patch.object(adapter, 'recover_receipt', return_value=None):
            with pytest.raises(BrokerUnavailable):
                if path == 'wait':
                    proxy.wait(timeout=0)
                elif path == 'cancel':
                    proxy.cancel()
                else:
                    bridge._handle_handoff_failure(proxy._record.ticket)
        assert bridge.quarantined
        assert proxy._record.synthetic is None
        assert bridge.committed_bytes == 2_000_000
        with pytest.raises(BrokerUnavailable):
            bridge.reserve(IPCCustodyKey(EPOCH, 2))
    finally:
        assert not bridge.close(timeout=0)
        broker.close(timeout=3)


def test_receipt_comparison_serializes_a_concurrent_publication():
    _, broker, adapter, bridge = make_stack()
    key, proxy = submit(bridge, 1, 'A', 'same-id', 1)
    entered = threading.Event()
    release = threading.Event()
    dispatched = threading.Event()
    results = []
    original = adapter.recover_receipt

    def pause_after_lookup(ticket):
        handle = original(ticket)
        if not entered.is_set():
            entered.set()
            assert release.wait(3)
        return handle

    def wait_for_result():
        results.append(proxy.wait(timeout=3))

    def dispatch():
        adapter.transfer_next()
        dispatched.set()

    waiter = threading.Thread(target=wait_for_result)
    dispatcher = threading.Thread(target=dispatch)
    try:
        with patch.object(adapter, 'recover_receipt', side_effect=pause_after_lookup):
            waiter.start()
            assert entered.wait(3)
            dispatcher.start()
            assert not dispatched.wait(.1)
            release.set()
            dispatcher.join(3)
            waiter.join(3)
        assert not waiter.is_alive() and not dispatcher.is_alive()
        assert results[0].state is RequestState.SUCCEEDED
        assert results[0].result.logits.item() == 1
        assert not bridge.quarantined
        results.clear()
        notify_closed(bridge, key)
        bridge.start()
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        release.set()
        waiter.join(3)
        if dispatcher.ident is not None:
            dispatcher.join(3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


@pytest.mark.parametrize('first_action', ['wait', 'cancel'])
def test_one_proxy_operation_at_a_time_preserves_original_failure(first_action):
    worker, broker, adapter, bridge = make_stack()
    worker.release.clear()
    _, proxy = submit(bridge, 1, 'A', 'same-id', 1)
    assert adapter.transfer_next() is not None
    assert worker.started.wait(3)
    handle = adapter.recover_receipt(proxy._record.ticket)
    if first_action == 'wait':
        # A poll of an unfinished request no longer calls handle.wait();
        # observe a terminal outcome to exercise its guarded return path.
        worker.release.set()
        handle.wait(timeout=3)
    entered = threading.Event()
    resume = threading.Event()
    refs, public_errors = [], []
    real_latch, real_retain = bridge._latch, bridge._retain_wait_failure

    def fail(*args, **kwargs):
        temporary = torch.tensor([123.0])
        refs.append(weakref.ref(temporary))
        raise MemoryError('private owner failure')

    def delayed_latch(marker, error, record=None):
        if type(error) is MemoryError:
            entered.set()
            assert resume.wait(3)
        return real_latch(marker, error, record)

    def delayed_retain(error, record=None):
        if type(error) is MemoryError and threading.current_thread() is thread:
            entered.set()
            assert resume.wait(3)
        return real_retain(error, record)

    def first():
        try:
            if first_action == 'wait':
                proxy.wait(timeout=0)
            else:
                proxy.cancel()
        except BaseException as error:
            public_errors.append(error)

    action = patch.object(handle, first_action, side_effect=fail)
    delay = (patch.object(bridge, '_retain_wait_failure', side_effect=delayed_retain)
             if first_action == 'wait' else
             patch.object(bridge, '_latch', side_effect=delayed_latch))
    thread = threading.Thread(target=first)
    try:
        with action, delay:
            thread.start()
            assert entered.wait(3)
            with pytest.raises(BrokerUnavailable, match='proxy operation already in progress'):
                proxy.wait(timeout=0)
            resume.set()
            thread.join(3)
        assert not thread.is_alive()
        assert bridge.quarantined
        assert bridge.committed_bytes == 2_000_000
        public_errors.clear()
        gc.collect()
        assert refs[0]() is not None
    finally:
        resume.set()
        thread.join(3)
        worker.release.set()
        assert not bridge.close(timeout=0)
        broker.close(timeout=3)


def test_transition_owner_failure_survives_a_competing_observer_error():
    _, broker, _, bridge = make_stack()
    key = IPCCustodyKey(EPOCH, 1)
    bridge.reserve(key)
    bridge.authenticate(key, 'A', 'same-id')
    entered = threading.Event()
    resume = threading.Event()
    refs, public_errors = [], []
    real_latch = bridge._latch

    def fail(_request):
        temporary = torch.tensor([456.0])
        refs.append(weakref.ref(temporary))
        raise MemoryError('private transition failure')

    def delayed_latch(marker, error, record=None):
        if type(error) is MemoryError:
            entered.set()
            assert resume.wait(3)
        return real_latch(marker, error, record)

    def submit_one():
        try:
            bridge.submit_registered(key, 'A', request('same-id', 1))
        except BaseException as error:
            public_errors.append(error)

    thread = threading.Thread(target=submit_one)
    try:
        with patch.object(bridge._codec, 'encode', side_effect=fail), \
                patch.object(bridge, '_latch', side_effect=delayed_latch):
            thread.start()
            assert entered.wait(3)
            with pytest.raises(BrokerUnavailable):
                bridge.handler_closed(key)
            resume.set()
            thread.join(3)
        assert not thread.is_alive()
        assert bridge.quarantined
        assert type(bridge._records[key].transition_failure) is MemoryError
        assert type(bridge._records[key].failure) is BrokerUnavailable
        public_errors.clear()
        gc.collect()
        assert refs[0]() is not None
        assert bridge.committed_bytes == 2_000_000
    finally:
        resume.set()
        thread.join(3)
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)


def test_busy_proxy_does_not_block_other_team_or_poison_service():
    _, broker, _, bridge = make_stack()
    key_a, first = submit(bridge, 1, 'A', 'a', 1)
    key_b, second = submit(bridge, 2, 'B', 'b', 2)
    entered = threading.Event()
    resume = threading.Event()
    original_wait = bridge._wait
    errors = []

    def delayed_wait(record, timeout):
        if record is first._record:
            entered.set()
            assert resume.wait(3)
        return original_wait(record, timeout)

    def first_wait():
        try:
            first.wait(timeout=0)
        except TimeoutError:
            pass
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=first_wait)
    try:
        with patch.object(bridge, '_wait', side_effect=delayed_wait):
            thread.start()
            assert entered.wait(3)
            with pytest.raises(BrokerUnavailable, match='proxy operation already in progress'):
                first.cancel()
            assert second.cancel()
            assert second.wait(timeout=0).state is RequestState.CANCELLED
            assert not bridge.quarantined
            resume.set()
            thread.join(3)
        assert not thread.is_alive() and not errors
        assert first.cancel()
        notify_closed(bridge, key_a)
        notify_closed(bridge, key_b)
        bridge.start()
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        resume.set()
        thread.join(3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


def test_proxy_release_interruption_quarantines_and_blocks_new_admission():
    _, broker, _, bridge = make_stack()
    _, proxy = submit(bridge, 1, 'A', 'a', 1)
    lines, start = inspect.getsourcelines(type(proxy)._invoke)
    candidates = [i for i, line in enumerate(lines) if 'if acquired:' in line]
    if not candidates:
        candidates = [i for i, line in enumerate(lines) if 'self._operation_lock.release()' in line]
    target = start + candidates[0]
    code = type(proxy)._invoke.__code__

    def trace(frame, event, arg):
        if frame.f_code is code and event == 'line' and frame.f_lineno == target:
            sys.settrace(None)
            raise KeyboardInterrupt('release interrupted')
        return trace

    try:
        sys.settrace(trace)
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                _ = proxy.cancel_requested
        finally:
            sys.settrace(None)
        assert caught.value.__context__ is None
        assert bridge.quarantined
        assert bridge.committed_bytes == 2_000_000
        with pytest.raises(BrokerUnavailable):
            bridge.reserve(IPCCustodyKey(EPOCH, 2))
    finally:
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)


@pytest.mark.parametrize('stage', ['acquire', 'release'])
def test_proxy_gate_failure_privately_roots_and_sanitizes_its_exception(stage):
    _, broker, _, bridge = make_stack()
    _, proxy = submit(bridge, 1, 'A', 'a', 1)
    refs = []

    class FaultyLock:
        def acquire(self, *, blocking):
            if stage == 'acquire':
                self.fail()
            return True

        def release(self):
            if stage == 'release':
                self.fail()

        def fail(self):
            temporary = torch.tensor([789.0])
            refs.append(weakref.ref(temporary))
            raise MemoryError('private gate failure')

    proxy._operation_lock = FaultyLock()
    try:
        with pytest.raises(BrokerUnavailable) as caught:
            _ = proxy.cancel_requested
        assert caught.value.__context__ is None and caught.value.__cause__ is None
        caught.value.__traceback__ = None
        gc.collect()
        assert refs[0]() is not None
        assert bridge.quarantined and bridge.committed_bytes == 2_000_000
    finally:
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)


@pytest.mark.parametrize('alias', ['request', 'target_handle'])
def test_handler_alias_finalizer_runs_without_holding_bridge_condition(alias):
    _, broker, adapter, bridge = make_stack()
    key = IPCCustodyKey(EPOCH, 1)
    entered = threading.Event()
    release = threading.Event()
    acquired = threading.Event()

    class Owner:
        def __del__(self):
            entered.set()
            release.wait(3)

    def probe_condition():
        with bridge._condition:
            acquired.set()

    bridge.reserve(key)
    bridge.authenticate(key, 'A', 'a')
    value = request('a', 1)
    if alias == 'request':
        object.__setattr__(value, 'owner', Owner())
    proxy = bridge.submit_registered(key, 'A', value)
    del value
    if alias == 'request':
        assert proxy.cancel()
    else:
        assert adapter.transfer_next() is not None
        outcome = proxy.wait(timeout=3)
        assert outcome.state is RequestState.SUCCEEDED
        del outcome
        handle = adapter.recover_receipt(proxy._record.ticket)
        assert handle.wait_for_retirement(timeout=3)
        handle.owner = Owner()
        del handle
    notify_closed(bridge, key)
    bridge.start()
    probe = threading.Thread(target=probe_condition, daemon=True)
    try:
        assert entered.wait(3)
        assert bridge.committed_bytes == 2_000_000
        probe.start()
        assert acquired.wait(.5), 'finalizer held the bridge condition'
        release.set()
        eventually(lambda: bridge.committed_bytes == 0)
    finally:
        release.set()
        if probe.ident is not None:
            probe.join(3)
        assert bridge.close(timeout=3)
        assert broker.close(timeout=3)


@pytest.mark.parametrize('after_drop', [False, True])
def test_interrupted_alias_disposal_quarantines_before_claim_release(after_drop):
    _, broker, _, bridge = make_stack()
    key, proxy = submit(bridge, 1, 'A', 'a', 1)
    assert proxy.cancel()
    notify_closed(bridge, key)
    lines, start = inspect.getsourcelines(type(bridge)._collect_closed_inner)
    drop = next(i for i, line in enumerate(lines) if 'candidate.disposal = None' in line)
    target = start + drop + int(after_drop)
    code = type(bridge)._collect_closed_inner.__code__

    def trace(frame, event, arg):
        if frame.f_code is code and event == 'line' and frame.f_lineno == target:
            sys.settrace(None)
            raise KeyboardInterrupt('disposal interrupted')
        return trace

    try:
        sys.settrace(trace)
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                bridge._collect_closed()
        finally:
            sys.settrace(None)
        assert caught.value.__context__ is None
        assert bridge.quarantined
        assert bridge.committed_bytes == 2_000_000
        assert bridge._records[key].retiring
        assert (bridge._records[key].disposal is None) is after_drop
        with pytest.raises(BrokerUnavailable):
            bridge.reserve(IPCCustodyKey(EPOCH, 2))
    finally:
        assert not bridge.close(timeout=0)
        assert broker.close(timeout=3)
