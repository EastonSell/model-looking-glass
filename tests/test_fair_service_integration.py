"""Real loopback bridge composition with synthetic CPU work only."""

from threading import Event, Thread
from time import monotonic, sleep
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from looking_glass.broker import Broker
from looking_glass.fair_adapter import ExperimentalFairAdapter
from looking_glass.fair_service_lifecycle import _FairOwnerLifecycle
from looking_glass.ipc import IPCClient
from looking_glass.ipc_fair_bridge import FairIPCBridge, ForwardCodec
from looking_glass.resident_worker import ForwardRequest, ForwardResult
from looking_glass.service import OwnerStatus, make_gpt2_request_factory, serve_owner
from looking_glass.trusted_service_profile import validate_service_request


def install_cuda_guard():
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU-only fair service tests must never call CUDA')
    patches = []
    for name in dir(torch.cuda):
        value = getattr(torch.cuda, name)
        if callable(value) and not (isinstance(value, type) and issubclass(value, BaseException)):
            item = patch.object(torch.cuda, name, forbidden)
            item.start()
            patches.append(item)
    for name in dir(torch._C):
        value = getattr(torch._C, name)
        if name.startswith('_cuda_') and callable(value) and not isinstance(value, type):
            item = patch.object(torch._C, name, forbidden)
            item.start()
            patches.append(item)
    original_to = torch.Tensor.to
    def cpu_only_to(tensor, *args, **kwargs):
        destination = kwargs.get('device')
        if destination is None and args:
            destination = args[0]
        if isinstance(destination, (str, torch.device)) and str(destination).split(':', 1)[0] == 'cuda':
            return forbidden()
        return original_to(tensor, *args, **kwargs)
    item = patch.object(torch.Tensor, 'to', cpu_only_to)
    item.start()
    patches.append(item)
    return patches


@pytest.fixture(autouse=True)
def cuda_guard():
    patches = install_cuda_guard()
    try:
        yield
    finally:
        for item in reversed(patches):
            item.stop()


class CPUWorker:
    max_input_bytes = 64 * 1024
    max_output_bytes = 512 * 1024
    poisoned = False
    failed = False
    quarantined = False
    cancellation_exception = InterruptedError

    def __init__(self):
        self.seen = []
        self.started, self.release = Event(), Event()
        self.release.set()

    def snapshot_request(self, request):
        return ForwardRequest(request.request_id,
                              {key: value.clone() for key, value in request.inputs.items()}, request.seed)

    def execute(self, request, cancel_event):
        self.seen.append((request.inputs['input_ids'].tolist(), request.seed))
        self.started.set()
        self.release.wait(3)
        if cancel_event.is_set():
            raise InterruptedError('cancelled')
        return ForwardResult(request.request_id, request.inputs['input_ids'].to(dtype=torch.float32).unsqueeze(-1))

    def close(self):
        self.poisoned = True


def stack():
    worker = CPUWorker()
    broker = Broker(worker, max_pending=2, max_pending_per_team=1)
    codec = ForwardCodec(max_batch=1, max_tokens=2, max_encoded_bytes=4096, max_decoded_bytes=64)
    target = broker.enable_fair_handoff(('A', 'B'), decoder=codec.decode,
        max_item_bytes=4096, max_decoded_bytes=64, preparation_bytes=1_249_344,
        max_preparation_bytes=2_498_688, max_result_bytes=512 * 1024, max_unacknowledged=2)
    credentials = {'A': b'a-private-credential', 'B': b'b-private-credential'}
    def verifier(value):
        return {credentials['A']: 'A', credentials['B']: 'B'}[value]
    adapter = ExperimentalFairAdapter({'A': 'A', 'B': 'B'}, verifier=verifier, target=target,
        max_waiters=2, max_waiters_per_team=1, max_waiting_bytes=8192, max_item_bytes=4096)
    def validate(snapshot):
        validate_service_request(snapshot.inputs, snapshot.seed, cold=False)
        return None
    bridge = FairIPCBridge(adapter, codec=codec, team_credentials=credentials,
        custody_epoch='e' * 32, max_connections=2, max_request_bytes=4096,
        max_response_bytes=4 * 1024 * 1024, connection_commitment_bytes=1_075_838_976,
        request_validator=validate)
    factory = make_gpt2_request_factory(vocab_size=50257, max_positions=1024, max_batch=1, max_tokens=2)
    server = bridge.make_server(request_factory=factory, io_timeout=2, job_timeout=3)
    owner = _FairOwnerLifecycle(worker, broker, bridge, server, owner_quarantined=lambda: False)
    return owner, worker


def serve(owner, stop, ready, output):
    output.append(serve_owner(owner.server_view, owner.broker_view, owner.worker, stop,
        announce=lambda server: ready.set(), verify_drain=lambda: True,
        poll_interval=.01, shutdown_timeout=3))


def test_two_team_real_loopback_validates_fixed_inputs_and_drains():
    owner, worker = stack()
    stop, ready, output = Event(), Event(), []
    thread = Thread(target=serve, args=(owner, stop, ready, output))
    thread.start()
    try:
        assert ready.wait(3)
        first = IPCClient(owner.server.address, owner.server.team_tokens['A'], io_timeout=2)
        bad = first.submit(team_id='A', request_id='invalid',
                           inputs={'input_ids': [[15496, 996]]}, seed=29)
        assert bad.wait(3)['status'] == 'failed'
        deadline = monotonic() + 3
        while owner.bridge.committed_bytes and monotonic() < deadline:
            sleep(.01)
        assert owner.bridge.committed_bytes == 0
        go, outcomes, errors = Event(), {}, []
        def submit_team(team, seed):
            try:
                client = IPCClient(owner.server.address, owner.server.team_tokens[team], io_timeout=2)
                go.wait(2)
                handle = client.submit(team_id=team, request_id='valid-' + team,
                                       inputs={'input_ids': [[15496, 995]]}, seed=seed)
                outcomes[team] = handle.wait(3)
            except BaseException as error:
                errors.append(error)
        teams = [Thread(target=submit_team, args=('A', 29)),
                 Thread(target=submit_team, args=('B', 43))]
        for team in teams:
            team.start()
        go.set()
        for team in teams:
            team.join(4)
            assert not team.is_alive()
        assert not errors
        assert set(outcomes) == {'A', 'B'}
        assert all(outcomes[team]['status'] == 'succeeded' and outcomes[team]['team_id'] == team
                   for team in outcomes)
        assert sorted(worker.seen, key=lambda pair: pair[1]) == [([[15496, 995]], 29), ([[15496, 995]], 43)]
        deadline = monotonic() + 3
        while owner.bridge.committed_bytes and monotonic() < deadline:
            sleep(.01)
        assert owner.bridge.committed_bytes == 0
    finally:
        stop.set()
        thread.join(5)
    assert not thread.is_alive()
    assert output[0].status is OwnerStatus.CLEAN and output[0].teardown_verified
    assert not torch.cuda._initialized


def test_disconnected_server_custody_quarantine_stops_real_owner():
    # This case deliberately leaves a blocked quarantined worker. Isolate it
    # in a disposable CPU-only process, as the real supervisor must do.
    code = r'''
import os, runpy, sys
from threading import Event, Thread
import torch
ns = runpy.run_path(sys.argv[1])
guards = ns['install_cuda_guard']()
owner, worker = ns['stack']()
worker.release.clear()
stop, ready, output = Event(), Event(), []
thread = Thread(target=ns['serve'], args=(owner, stop, ready, output))
thread.start()
assert ready.wait(3)
client = ns['IPCClient'](owner.server.address, owner.server.team_tokens['A'], io_timeout=2)
handle = client.submit(team_id='A', request_id='disconnect',
                       inputs={'input_ids': [[15496, 995]]}, seed=29)
assert worker.started.wait(3)
handle.close()
owner.server._custody_quarantine.set()
thread.join(3)
assert not thread.is_alive()
assert output[0].status is ns['OwnerStatus'].QUARANTINED and not output[0].teardown_verified
assert owner.server and owner.bridge and owner.broker and worker
assert not torch.cuda._initialized
print('quarantined', flush=True)
os._exit(0)
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).resolve())],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'quarantined'
