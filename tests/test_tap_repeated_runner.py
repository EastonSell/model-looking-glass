import json
import os
import inspect
from pathlib import Path
import subprocess
import sys
import gc
import weakref

import pytest
import torch

from looking_glass import tap_repeated_protocol as protocol
from looking_glass import tap_repeated_runner as runner


SCHEDULE = Path(__file__).resolve().parents[1] / 'evidence/gpu-tap-latency-repeated-v1/schedule.json'
TOTAL, CAP = 8 * protocol.GIB, 2 * protocol.GIB


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class Journal:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(row)


def test_preflight_holds_both_captures_and_all_outputs_across_completion(one_thread):
    device = runner._SimulatedDevice(0)
    guard = runner._GuardedDevice(device, runner._SimulatedOwner())
    journal = Journal()
    syncs = []

    def sync():
        syncs.append((len(guard.roots.outputs), len(guard.roots.captures), len(guard.roots.sessions)))

    device.synchronize = sync
    runner._preflight(guard, journal, 0, TOTAL, CAP)
    assert syncs == [(0, 0, 0), (4, 2, 3), (0, 0, 0)]
    assert [row['stage'] for row in journal.rows] == ['preflight', 'postcheck']


def test_id_only_preflight_requires_an_observed_event_without_a_value(one_thread):
    device = runner._SimulatedDevice(0)
    roots = runner._Roots()
    device.execute('id', roots)
    metadata = runner._validate_output(roots, 'id', 0)
    assert metadata['events'] == 1 and metadata['recorded_tensor_bytes'] == 0
    assert metadata['trace_mode'] == 'id' and metadata['capture_finite'] is None
    roots.session['events'].clear()
    with pytest.raises(ValueError, match='one event'):
        runner._validate_output(roots, 'id', 0)


@pytest.mark.parametrize('mode', protocol.MODES)
def test_sample_syncs_before_and_after_disposal_and_records_real_cpu_work(one_thread, mode):
    device = runner._SimulatedDevice(0)
    guard = runner._GuardedDevice(device, runner._SimulatedOwner())
    journal, syncs = Journal(), []
    device.synchronize = lambda: syncs.append((guard.roots.output is not None, guard.roots.capture is not None))
    item = {'stage': 'measured', 'round': 0, 'position': 0, 'mode': mode,
            'observation_id': 'measured:0:0'}
    runner._sample(guard, journal, item, 0, TOTAL, CAP, minimum_wall_ns=1)
    assert syncs == [(False, False), (True, mode != 'none'), (False, False)]
    sample, post = journal.rows
    assert sample['wall_ns'] == sample['ended_ns'] - sample['started_ns'] > 0
    assert sample['events'] == (mode != 'none')
    assert sample['output_finite'] is True and sample['capture_finite'] is (None if mode == 'none' else True)
    assert sample['recorded_tensor_bytes'] == (3072 if mode != 'none' else 0)
    assert post['status'] == 'passed'


def test_failed_completion_retains_reachable_output_and_makes_no_later_device_call(one_thread):
    device = runner._SimulatedDevice(0)
    guard = runner._GuardedDevice(device, runner._SimulatedOwner())
    journal, calls = Journal(), []

    def sync():
        calls.append('sync')
        if len(calls) == 2:
            raise KeyboardInterrupt('simulated uncertain completion')

    device.synchronize = sync
    item = {'stage': 'measured', 'round': 0, 'position': 0, 'mode': 'direct',
            'observation_id': 'measured:0:0'}
    with pytest.raises(KeyboardInterrupt):
        runner._sample(guard, journal, item, 0, TOTAL, CAP, minimum_wall_ns=1)
    assert guard.unsafe is True and isinstance(guard.roots.failure, KeyboardInterrupt)
    assert guard.roots.output is not None and guard.roots.capture is not None and journal.rows == []
    with pytest.raises(runner.DeviceOwnerLost):
        guard.call(device.synchronize)
    assert calls == ['sync', 'sync']


def test_revocation_during_preflight_completion_keeps_outputs_and_captures(one_thread):
    class Owner:
        revoked = False

        def check(self):
            if self.revoked:
                raise RuntimeError('owner revoked during synchronization')

    owner, device, journal = Owner(), runner._SimulatedDevice(0), Journal()
    guard = runner._GuardedDevice(device, owner)
    references, calls = [], []

    def sync():
        calls.append('sync')
        if len(calls) == 2:
            references.extend(weakref.ref(value) for value in guard.roots.outputs + guard.roots.captures)
            owner.revoked = True

    device.synchronize = sync
    with pytest.raises(RuntimeError, match='revoked'):
        runner._preflight(guard, journal, 0, TOTAL, CAP)
    gc.collect()
    assert len(references) == 6 and all(reference() is not None for reference in references)
    assert len(guard.roots.outputs) == 4 and len(guard.roots.captures) == 2
    assert guard.unsafe and calls == ['sync', 'sync'] and journal.rows == []
    with pytest.raises(runner.DeviceOwnerLost):
        guard.call(device.synchronize)
    assert calls == ['sync', 'sync']


def test_owner_loss_prevents_operation_and_latches_failure():
    class Owner:
        def check(self):
            raise RuntimeError('owner lost')

    called = []
    guard = runner._GuardedDevice(runner._SimulatedDevice(0), Owner())
    with pytest.raises(RuntimeError, match='owner lost'):
        guard.call(lambda: called.append(True))
    assert called == [] and guard.unsafe and str(guard.roots.failure) == 'owner lost'
    with pytest.raises(runner.DeviceOwnerLost):
        guard.call(lambda: called.append(True))
    assert called == []


def test_interrupted_failure_handler_keeps_inflight_guard_latched():
    guard = runner._GuardedDevice(runner._SimulatedDevice(0), runner._SimulatedOwner())
    lines, first_line = inspect.getsourcelines(runner._GuardedDevice.call)
    unsafe_line = first_line + next(i for i, line in enumerate(lines) if 'self._unsafe = True' in line)
    interrupted = []

    def trace(frame, event, arg):
        if frame.f_code is runner._GuardedDevice.call.__code__ and event == 'line' and frame.f_lineno == unsafe_line:
            interrupted.append(True)
            raise KeyboardInterrupt('failure handler interrupted')
        return trace

    def fail():
        raise MemoryError('fake operation failed')

    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt, match='failure handler interrupted'):
            guard.call(fail)
    finally:
        sys.settrace(None)
    assert interrupted == [True] and guard.unsafe
    with pytest.raises(runner.DeviceOwnerLost):
        guard.call(lambda: pytest.fail('unsafe guard must prevent another operation'))


def test_postcheck_loss_keeps_an_assigned_return_rooted():
    class Owner:
        checks = 0

        def check(self):
            self.checks += 1
            if self.checks == 2:
                raise RuntimeError('owner lost after return')

    guard = runner._GuardedDevice(runner._SimulatedDevice(0), Owner())
    refs = []

    def allocate():
        value = torch.tensor([1.0])
        refs.append(weakref.ref(value))
        return value

    with pytest.raises(RuntimeError, match='owner lost after return'):
        guard.call(allocate)
    guard.roots.failure = None
    gc.collect()
    assert guard.unsafe and refs[0]() is guard.roots.pending_return


def test_outer_failure_handling_stops_after_return_and_handler_interruption():
    class InterruptedHandlerGuard(runner._GuardedDevice):
        interrupt_handler = True

        def __setattr__(self, name, value):
            if name == '_unsafe' and value is True and self.interrupt_handler:
                self.interrupt_handler = False
                raise KeyboardInterrupt('second fault')
            return super().__setattr__(name, value)

    guard = InterruptedHandlerGuard(runner._SimulatedDevice(0), runner._SimulatedOwner())
    lines, first_line = inspect.getsourcelines(runner._GuardedDevice.call)
    return_line = first_line + next(i for i, line in enumerate(lines) if line.strip() == 'return result')
    references, injected = [], []

    def allocate():
        value = torch.tensor([1.0])
        references.append(weakref.ref(value))
        return value

    def trace(frame, event, arg):
        if frame.f_code is runner._GuardedDevice.call.__code__ and event == 'line' and frame.f_lineno == return_line:
            injected.append(True)
            raise MemoryError('return interrupted')
        return trace

    try:
        sys.settrace(trace)
        try:
            guard.call(allocate)
        except KeyboardInterrupt as exc:
            # The outer CPU driver must stop on any escaping interruption,
            # regardless of whether the inner guard reports unsafe.
            guard.fail(exc)
    finally:
        sys.settrace(None)
    gc.collect()
    assert injected == [True] and guard.unsafe
    assert references[0]() is guard.roots.pending_return
    with pytest.raises(runner.DeviceOwnerLost):
        guard.call(lambda: pytest.fail('outer handling must prevent another operation'))


def test_failed_post_sample_memory_read_leaves_pending_observation(one_thread):
    device = runner._SimulatedDevice(0)
    guard = runner._GuardedDevice(device, runner._SimulatedOwner())
    journal, calls = Journal(), []
    original = device.snapshot

    def snapshot():
        calls.append('snapshot')
        if len(calls) == 2:
            raise RuntimeError('simulated postcheck loss')
        return original()

    device.snapshot = snapshot
    item = {'stage': 'measured', 'round': 0, 'position': 0, 'mode': 'none',
            'observation_id': 'measured:0:0'}
    with pytest.raises(RuntimeError, match='postcheck loss'):
        runner._sample(guard, journal, item, 0, TOTAL, CAP, minimum_wall_ns=1)
    assert len(journal.rows) == 1 and journal.rows[0]['postcheck_status'] == 'pending'
    assert guard.unsafe and isinstance(guard.roots.failure, RuntimeError)
    with pytest.raises(runner.DeviceOwnerLost):
        guard.call(device.snapshot)
    assert calls == ['snapshot', 'snapshot']


def test_journal_interruption_keeps_process_owner_root_and_blocks_more_calls(tmp_path, monkeypatch, one_thread):
    monkeypatch.setattr(runner, '_OWNER_USED', False)
    monkeypatch.setattr(runner, '_OWNER_ROOT', [None])
    original = runner._Journal.append
    count = []

    def append(self, row):
        count.append(row['stage'])
        if len(count) == 2:
            raise KeyboardInterrupt('simulated lost postcheck')
        return original(self, row)

    monkeypatch.setattr(runner._Journal, 'append', append)
    output = tmp_path / 'interrupted'
    with pytest.raises(KeyboardInterrupt, match='lost postcheck'):
        runner.run_cpu_simulation(SCHEDULE, session=0, output=output)
    guard = runner._OWNER_ROOT[0]
    assert guard is not None and guard.unsafe and isinstance(guard.roots.failure, KeyboardInterrupt)
    assert len((output / 'observations.jsonl').read_bytes().splitlines()) == 1
    failure = json.loads((output / 'failure.json').read_bytes())
    assert failure['stage'] == 'preflight' and failure['observation_id'] == 'preflight'
    assert failure['unsafe'] and not failure['completion_verified']
    with pytest.raises(runner.DeviceOwnerLost):
        guard.call(guard.device.synchronize)


def test_gpu_entry_is_disabled_before_device_access(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('must not touch CUDA'))
    with pytest.raises(RuntimeError, match='admission is not available'):
        runner.run_gpu()


def test_existing_output_directory_is_not_modified(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_OWNER_USED', False)
    monkeypatch.setattr(runner, '_OWNER_ROOT', [None])
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'sentinel').write_text('untouched')
    with pytest.raises(FileExistsError):
        runner.run_cpu_simulation(SCHEDULE, session=0, output=output)
    assert sorted(path.name for path in output.iterdir()) == ['sentinel']


@pytest.mark.parametrize('target', ['simulation.json', 'completed.json'])
def test_post_write_failure_cannot_publish_complete_simulation(tmp_path, monkeypatch, one_thread, target):
    monkeypatch.setattr(runner, '_OWNER_USED', False)
    monkeypatch.setattr(runner, '_OWNER_ROOT', [None])
    original = runner._write_json_new

    def write(path, value):
        original(path, value)
        if Path(path).name == target:
            raise OSError('simulated post-write failure')

    monkeypatch.setattr(runner, '_write_json_new', write)
    output = tmp_path / target
    with pytest.raises(OSError, match='post-write failure'):
        runner.run_cpu_simulation(SCHEDULE, session=0, output=output)
    observed = runner.inspect_cpu_simulation(output, schedule_path=SCHEDULE, child_returncode=1, child_reaped=True)
    assert observed['state'] in ('failed', 'incomplete') and observed['gpu_eligible'] is False
    assert (output / 'failure.json').exists()


def test_complete_cpu_simulation_uses_frozen_rounds_and_durable_records(tmp_path):
    output = tmp_path / 'session'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    process = subprocess.run([sys.executable, '-m', 'looking_glass.tap_repeated_runner',
                              '--schedule', str(SCHEDULE), '--session', '2', '--output', str(output)],
                             env=env, capture_output=True, timeout=45, check=False)
    assert process.returncode == 0, process.stderr.decode(errors='replace')
    result = json.loads((output / 'simulation.json').read_bytes())
    assert result['synthetic_cpu_simulation'] is True and result['gpu_eligible'] is False
    assert result['device_event_and_memory_are_simulated'] is True
    assert result['summary']['warmup_samples'] == 6 and result['summary']['measured_samples'] == 36
    journal = (output / 'observations.jsonl').read_bytes()
    assert journal.endswith(b'\n') and len(journal.splitlines()) == 86
    rows = [json.loads(line) for line in journal.splitlines()]
    check = protocol.summarize_session(protocol.load_schedule(SCHEDULE), 2, rows, clock=result['clock'],
                                       total_bytes=TOTAL, cap_bytes=CAP, device='cpu')
    assert check == result['summary']
    inspected = runner.inspect_cpu_simulation(output, schedule_path=SCHEDULE,
                                              child_returncode=process.returncode, child_reaped=True)
    assert inspected['state'] == 'completed_simulation' and inspected['summary'] == check
    assert runner.inspect_cpu_simulation(output, schedule_path=SCHEDULE,
        child_returncode=1, child_reaped=True)['state'] == 'incomplete'
