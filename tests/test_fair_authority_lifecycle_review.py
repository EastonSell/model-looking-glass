"""Exercise lifecycle composition around the real cooperative CPU authority."""

from threading import Event, Thread

from looking_glass.fair_service_lifecycle import _FairOwnerLifecycle
from looking_glass.ipc import IPCClient
from looking_glass.service import OwnerStatus, serve_owner
from test_fair_ingress_authority import bound_stack
from test_ipc_completion_authority import stop_owner


def test_authority_bound_loopback_lifecycle_serves_and_drains():
    model, _, _, _, authority, broker, target, _, _, bridge = bound_stack()
    worker = broker._worker
    server = bridge.make_server(io_timeout=2, job_timeout=5)
    owner = _FairOwnerLifecycle(worker, broker, bridge, server,
                                owner_quarantined=lambda: False)
    stop, ready, results = Event(), Event(), []
    def run():
        results.append(serve_owner(owner.server_view, owner.broker_view, worker, stop,
            announce=lambda _: ready.set(), verify_drain=lambda: True,
            poll_interval=.01, shutdown_timeout=5))
    thread = Thread(target=run)
    model.release.set()
    thread.start()
    try:
        assert ready.wait(5)
        assert server._completion_binding == (authority, broker, target)
        client = IPCClient(server.address, server.team_tokens['team'], io_timeout=2)
        result = client.submit(team_id='team', request_id='fixture',
                               inputs={'input_ids': [[1, 2]]}).wait(5)
        assert result['status'] == 'succeeded'
        assert result['result']['logits'] == [1.0, 2.0, 2.0, 3.0]
    finally:
        stop.set()
        thread.join(8)
    assert not thread.is_alive()
    assert len(results) == 1 and results[0].status is OwnerStatus.CLEAN
    assert results[0].teardown_verified and not authority.unsafe


def test_authority_fault_during_forward_quarantines_lifecycle_and_retains_budget():
    model, budget, load, workspace, authority, broker, _, _, _, bridge = bound_stack()
    worker = broker._worker
    server = bridge.make_server(io_timeout=2, job_timeout=5)
    owner = _FairOwnerLifecycle(worker, broker, bridge, server,
                                owner_quarantined=lambda: False)
    stop, ready, results = Event(), Event(), []
    thread = Thread(target=lambda: results.append(serve_owner(
        owner.server_view, owner.broker_view, worker, stop,
        announce=lambda _: ready.set(), verify_drain=lambda: True,
        poll_interval=.01, shutdown_timeout=2)))
    thread.start()
    try:
        assert ready.wait(5)
        client = IPCClient(server.address, server.team_tokens['team'], io_timeout=2)
        client.submit(team_id='team', request_id='blocked', inputs={'input_ids': [[1, 2]]})
        assert model.entered.wait(5)
        authority.mark_unsafe()
        thread.join(5)
        assert not thread.is_alive()
        assert len(results) == 1 and results[0].status is OwnerStatus.QUARANTINED
        assert not results[0].teardown_verified
        assert owner.quarantined and set(budget.active_leases()) == {load, workspace}
    finally:
        stop.set()
        stop_owner(model, broker)
