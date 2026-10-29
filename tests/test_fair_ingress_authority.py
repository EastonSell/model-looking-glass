"""CPU composition checks for the experimental authority-bound fair ingress."""

import gc
import inspect
import sys
import threading
import traceback
import weakref
from unittest.mock import patch

import pytest
import torch

from looking_glass.broker import BrokerUnavailable
from looking_glass.fair_adapter import AdapterClosed, ExperimentalFairAdapter
from looking_glass.ipc import IPCClient
from looking_glass.ipc_custody import IPCCustodyKey
from looking_glass.ipc_fair_bridge import FairIPCBridge, ForwardCodec
from looking_glass.owner_completion import OwnerCompletionAuthority
from looking_glass.resident_worker import ForwardRequest
from test_ipc_completion_authority import stop_owner
from test_owner_completion import make_owner


EPOCH = 'b' * 32


def bound_stack(*, request_validator=None):
    model, _, budget, load, workspace, authority, _, broker, _ = make_owner()
    codec = ForwardCodec(max_batch=1, max_tokens=2, max_encoded_bytes=512, max_decoded_bytes=512)
    target = broker.enable_fair_handoff(
        ('team',), decoder=codec.decode, max_item_bytes=512, max_decoded_bytes=512,
        preparation_bytes=2048, max_preparation_bytes=4096,
        max_result_bytes=128, max_unacknowledged=4)
    adapter = ExperimentalFairAdapter(
        {'alice': 'team'}, verifier=lambda token: 'alice' if token == b'secret' else None,
        target=target, max_waiters=4, max_waiters_per_team=3,
        max_waiting_bytes=2048, max_item_bytes=512, completion_authority=authority)
    guard = adapter.bridge_construction_guard()
    bridge = guard.construct(codec=codec, team_credentials={'team': b'secret'},
        custody_epoch=EPOCH, max_connections=2, max_request_bytes=2048,
        max_response_bytes=4096, connection_commitment_bytes=2_000_000,
        request_validator=request_validator)
    assert guard.adopt() is bridge
    return model, budget, load, workspace, authority, broker, target, adapter, guard, bridge


def test_guarded_factory_binds_one_real_loopback_service():
    model, _, _, _, authority, broker, target, adapter, guard, bridge = bound_stack()
    server = None
    try:
        assert adapter.recover_bridge_construction_guard() is guard
        with pytest.raises(AdapterClosed):
            adapter.bridge_construction_guard()
        model.release.set()
        bridge.start()
        server = bridge.make_server(io_timeout=2, job_timeout=5)
        assert bridge._server is server and bridge._server_construction.server is server
        assert server._completion_binding == (authority, broker, target)
        assert bridge._target_broker is broker and bridge._target is target
        server.start()
        client = IPCClient(server.address, server.team_tokens['team'], io_timeout=2)
        result = client.submit(team_id='team', request_id='fixture', inputs={'input_ids': [[1, 2]]}).wait(5)
        assert result['status'] == 'succeeded' and result['request_id'] == 'fixture'
        assert result['team_id'] == 'team' and result['result']['logits'] == [1.0, 2.0, 2.0, 3.0]
    finally:
        if server is not None:
            assert server.close(timeout=5)
        assert bridge.close(timeout=5)
        assert broker.close(timeout=5)


def test_interrupted_adapter_queue_mutation_marks_the_shared_authority():
    model, budget, load, workspace, authority, broker, _, adapter, _, _ = bound_stack()
    try:
        registration = adapter.registration_guard(b'secret', b'payload', team_id='team')
        with patch.object(adapter._queue, 'submit', side_effect=MemoryError('uncertain admission')):
            with pytest.raises(MemoryError):
                registration.submit()
        assert authority.unsafe and adapter.indeterminate
        assert set(budget.active_leases()) == {load, workspace}
        with pytest.raises(AdapterClosed):
            adapter.registration_guard(b'secret', b'later', team_id='team')
    finally:
        stop_owner(model, broker)


def test_bridge_snapshot_failure_marks_owner_and_retains_connection_claim():
    def fail(_):
        raise MemoryError('uncertain validation')
    model, budget, load, workspace, authority, broker, _, _, _, bridge = bound_stack(request_validator=fail)
    key = IPCCustodyKey(EPOCH, 1)
    try:
        bridge.reserve(key)
        bridge.authenticate(key, 'team', 'fixture')
        request = ForwardRequest('fixture', {'input_ids': torch.tensor([[1, 2]])})
        with pytest.raises(BrokerUnavailable):
            bridge.submit_registered(key, 'team', request)
        assert authority.unsafe and bridge.quarantined
        assert bridge.committed_bytes == 2_000_000
        assert set(budget.active_leases()) == {load, workspace}
    finally:
        stop_owner(model, broker)


def test_unsafe_owner_prevents_waiter_cancellation_and_retains_payload():
    model, _, _, _, authority, broker, _, adapter, _, _ = bound_stack()
    try:
        ticket = adapter.submit(b'secret', b'pending')
        authority.mark_unsafe()
        with pytest.raises(AdapterClosed):
            adapter.cancel(b'secret', ticket)
        assert len(adapter) == 1 and adapter.waiting_bytes == len(b'pending')
    finally:
        stop_owner(model, broker)


def test_unsafe_owner_prevents_connection_cleanup_and_retains_claim():
    model, _, _, _, authority, broker, _, _, _, bridge = bound_stack()
    key = IPCCustodyKey(EPOCH, 1)
    try:
        bridge.reserve(key)
        authority.mark_unsafe()
        with pytest.raises(BrokerUnavailable):
            bridge.handler_closed(key)
        assert bridge.committed_bytes == 2_000_000
        assert not bridge.close(timeout=0.1)
    finally:
        stop_owner(model, broker)


def test_unsafe_owner_prevents_receipt_ack_and_retains_result_commitment():
    model, _, _, _, authority, broker, target, adapter, _, bridge = bound_stack()
    try:
        request = ForwardRequest('fixture', {'input_ids': torch.tensor([[1, 2]])})
        payload = bridge._codec.encode(request)
        ticket = adapter.submit(b'secret', payload)
        receipt = adapter.transfer_next()
        assert receipt.ticket == ticket
        handle = adapter.recover_receipt(ticket)
        assert handle is not None and model.entered.wait(5)
        committed = target.receipt_committed_bytes
        assert committed == 128
        authority.mark_unsafe()
        with pytest.raises(BrokerUnavailable):
            adapter.acknowledge_receipt(ticket, handle)
        assert target.receipt_committed_bytes == committed
    finally:
        stop_owner(model, broker)


@pytest.mark.parametrize('source', ['adapter', 'bridge', 'target'])
def test_interrupted_owner_signal_still_latches_exact_authority(source):
    model, _, _, _, authority, broker, target, adapter, _, bridge = bound_stack()
    signal = {'adapter': adapter._mark_indeterminate, 'bridge': bridge._mark_quarantined,
              'target': target._signal_owner_unsafe}[source]
    try:
        with patch.object(authority, 'mark_unsafe', side_effect=KeyboardInterrupt('injected')):
            with pytest.raises(KeyboardInterrupt):
                signal()
        assert authority.unsafe
        if source == 'adapter':
            assert adapter.indeterminate
        if source == 'bridge':
            assert bridge.quarantined
    finally:
        stop_owner(model, broker)


@pytest.mark.parametrize('initially_unsafe', [False, True])
@pytest.mark.parametrize('change', [
    'authority_removed', 'authority_replaced', 'authority_deleted', 'target_broker', 'broker_worker',
    'authority_broker', 'authority_worker', 'worker_authority', 'broker_target',
])
def test_changed_target_graph_quarantines_original_owner_before_receipt_ack(change, initially_unsafe):
    model, _, _, _, authority, broker, target, adapter, _, bridge = bound_stack()
    worker = broker._worker
    saved = None
    try:
        payload = bridge._codec.encode(ForwardRequest('fixture', {'input_ids': torch.tensor([[1, 2]])}))
        ticket = adapter.submit(b'secret', payload)
        assert adapter.transfer_next().ticket == ticket
        handle = adapter.recover_receipt(ticket)
        assert model.entered.wait(5)
        if initially_unsafe:
            authority.mark_unsafe()
        fields = {
            'authority_removed': (broker, '_completion_authority', None),
            'authority_replaced': (broker, '_completion_authority', OwnerCompletionAuthority()),
            'authority_deleted': (broker, '_completion_authority', None),
            'target_broker': (target, '_broker', object()),
            'broker_worker': (broker, '_worker', object()),
            'authority_broker': (authority, '_broker', object()),
            'authority_worker': (authority, '_worker', object()),
            'worker_authority': (worker, '_authority', object()),
            'broker_target': (broker, '_fair_handoff', object()),
        }
        owner, field, replacement = fields[change]
        saved = owner, field, getattr(owner, field)
        if change == 'authority_deleted':
            delattr(owner, field)
        else:
            try:
                setattr(owner, field, replacement)
            except RuntimeError:
                # A broker with the separately reviewed immutable-authority
                # setter rejects this earlier and latches the same owner.
                assert change in ('authority_removed', 'authority_replaced')
        with pytest.raises(BrokerUnavailable):
            target.acknowledge_receipt(ticket.sequence, handle)
        assert authority.unsafe and broker.quarantined
        assert target.committed_bytes == 2048 and target.receipt_committed_bytes == 128
        assert target._receipts[ticket.sequence].handle is handle
    finally:
        if saved is not None:
            setattr(*saved)
        stop_owner(model, broker)


@pytest.mark.parametrize('component', ['adapter', 'bridge'])
def test_changed_ingress_authority_retains_waiter_and_connection(component):
    model, _, _, _, authority, broker, _, adapter, _, bridge = bound_stack()
    key = IPCCustodyKey(EPOCH, 1)
    original = None
    try:
        ticket = adapter.submit(b'secret', b'pending')
        bridge.reserve(key)
        owner = adapter if component == 'adapter' else bridge
        original = owner._completion_authority
        owner._completion_authority = None
        if component == 'adapter':
            with pytest.raises(AdapterClosed):
                adapter.cancel(b'secret', ticket)
        else:
            with pytest.raises(BrokerUnavailable):
                bridge.handler_closed(key)
        assert authority.unsafe and broker.quarantined
        assert len(adapter) == 1 and adapter.waiting_bytes == len(b'pending')
        assert bridge.committed_bytes == 2_000_000
    finally:
        if original is not None:
            owner._completion_authority = original
        stop_owner(model, broker)


@pytest.mark.parametrize('component,field', [
    ('bridge', '_adapter'), ('bridge', '_target'), ('bridge', '_target_broker'),
    ('adapter', '_target'), ('bridge', '_completion_authority'),
])
def test_changed_bridge_graph_preserves_connection_custody(component, field):
    model, _, _, _, authority, broker, _, adapter, _, bridge = bound_stack()
    key = IPCCustodyKey(EPOCH, 1)
    owner = bridge if component == 'bridge' else adapter
    original = getattr(owner, field)
    try:
        bridge.reserve(key)
        setattr(owner, field, object())
        with pytest.raises(BrokerUnavailable):
            bridge.handler_closed(key)
        assert authority.unsafe and broker.quarantined and bridge.quarantined
        assert bridge.committed_bytes == 2_000_000
    finally:
        setattr(owner, field, original)
        stop_owner(model, broker)


@pytest.mark.parametrize('component', ['adapter', 'bridge'])
def test_status_observation_latches_changed_graph_before_queued_transfer(component):
    model, _, _, _, authority, broker, target, adapter, _, bridge = bound_stack()
    owner = adapter if component == 'adapter' else bridge
    original = owner._completion_authority
    try:
        ticket = adapter.submit(b'secret', b'pending')
        owner._completion_authority = None
        assert adapter.indeterminate if component == 'adapter' else bridge.quarantined
        assert authority.unsafe and broker.quarantined
        with pytest.raises((AdapterClosed, BrokerUnavailable)):
            adapter.transfer_next()
        assert adapter.waiting_bytes == len(b'pending') and len(adapter) == 1
        assert ticket.sequence in adapter._owners
    finally:
        owner._completion_authority = original
        stop_owner(model, broker)


@pytest.mark.parametrize('interrupt_at', ['none', 'store', 'handler'])
def test_simultaneous_failed_status_reads_root_each_private_exception(interrupt_at):
    model, _, _, _, authority, broker, target, adapter, _, bridge = bound_stack()
    barrier = threading.Barrier(2)
    refs, aliases, public_errors = [], [], []

    class Owner:
        pass

    def fail():
        tensor = torch.tensor([1.0])
        tensor.owner = Owner()
        refs.append(weakref.ref(tensor.owner))
        aliases.append(tensor.detach())
        barrier.wait(timeout=3)
        raise RuntimeError('private graph-read failure')

    from looking_glass.broker_fair_handoff import BrokerFairHandoff
    lines, start = inspect.getsourcelines(BrokerFairHandoff._observe_owner_binding_inner)
    boundary = 'slot.error = error' if interrupt_at == 'store' else 'notification = None'
    store_line = start + next(i for i, source in enumerate(lines) if source.strip() == boundary)
    injected = []

    def interrupt(frame, event, _arg):
        if (frame.f_code is BrokerFairHandoff._observe_owner_binding_inner.__code__ and
                event == 'line' and frame.f_lineno == store_line and not injected):
            injected.append(True)
            raise KeyboardInterrupt('interrupted private error retention')
        return interrupt

    def observe(component):
        try:
            if interrupt_at != 'none' and component == 'adapter':
                sys.settrace(interrupt)
            if component == 'adapter':
                _ = adapter.indeterminate
            else:
                _ = bridge.quarantined
        except BaseException as error:
            public_errors.append(error)
        finally:
            sys.settrace(None)

    threads = [threading.Thread(target=observe, args=(name,)) for name in ('adapter', 'bridge')]
    try:
        with patch.object(target, '_owner_binding_valid', side_effect=fail):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        assert all(not thread.is_alive() for thread in threads)
        assert len(public_errors) == len(refs) == len(aliases) == 2
        assert authority.unsafe and broker.quarantined
        assert sum(slot.error is not None or slot.guard_error is not None
                   for slot in target._binding_slots) == 2
        if interrupt_at != 'none':
            assert injected
            assert any(isinstance(error, KeyboardInterrupt) for error in public_errors)
        if interrupt_at == 'store':
            assert sum(isinstance(slot.retention_error, KeyboardInterrupt)
                       for slot in target._binding_slots) == 1
        if interrupt_at == 'handler':
            assert any(isinstance(slot.guard_error, KeyboardInterrupt) and
                       isinstance(slot.guard_error.__context__, RuntimeError)
                       for slot in target._binding_slots)
        for error in public_errors:
            assert error.__context__ is None and error.__cause__ is None
            traceback.clear_frames(error.__traceback__)
        public_errors.clear()
        gc.collect()
        assert all(ref() is not None for ref in refs)
    finally:
        stop_owner(model, broker)


def test_graph_observation_capacity_exhaustion_latches_without_inspection():
    model, _, _, _, authority, broker, target, _, _, bridge = bound_stack()
    try:
        slots = [target._free_binding_slots.get_nowait() for _ in target._binding_slots]
        with patch.object(bridge, '_owner_binding_valid', side_effect=AssertionError('must not inspect')):
            assert bridge.quarantined
        assert authority.unsafe and broker.quarantined
        assert len(slots) == 256
    finally:
        stop_owner(model, broker)


def test_substituted_graph_slot_producer_is_never_called():
    model, _, _, _, authority, broker, target, _, _, bridge = bound_stack()
    original = target._free_binding_slots
    calls = []

    class SubstitutedPool:
        def get_nowait(self):
            calls.append(True)
            raise AssertionError('substituted producer must not run')

    try:
        target._free_binding_slots = SubstitutedPool()
        assert bridge.quarantined
        assert not calls and authority.unsafe and broker.quarantined
    finally:
        target._free_binding_slots = original
        stop_owner(model, broker)
