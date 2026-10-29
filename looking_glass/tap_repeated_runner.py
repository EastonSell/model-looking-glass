"""CPU-only exercise of the repeated tap timing and journal boundaries.

This implementation does not load a checkpoint, attach real model hooks, query
GPU activity, or admit CUDA. Its fake device counters, stream events and tap
metadata are synthetic test inputs, never GPU benchmark observations.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import time

import torch

from . import tap_repeated_protocol as protocol
from .probes import gradient_readout

_OWNER_LOCK = threading.Lock()
_OWNER_USED = False
_OWNER_ROOT = [None]


class DeviceOwnerLost(RuntimeError):
    """The simulated owner or an operation failed; no further device calls."""


class _Roots:
    def __init__(self):
        self.output = None
        self.capture = None
        self.session = None
        self.outputs = []
        self.captures = []
        self.sessions = []
        self.failure = None
        self.cleanup_failure = None
        self.pending_return = None


class _GuardedDevice:
    """Fail closed on a fake-device/owner exception, retaining known roots."""
    def __init__(self, device, owner):
        self.device, self.owner = device, owner
        self.roots = _Roots()
        self._unsafe = False
        self._inflight = False

    @property
    def unsafe(self):
        # An interrupted operation/exception handler leaves its marker set.
        # During a normal call this is a conservative in-flight indication.
        return self._unsafe or self._inflight

    def call(self, operation, *args):
        if self.unsafe:
            raise DeviceOwnerLost('device ownership or completion is unverified')
        self._inflight = True
        try:
            self.owner.check()
            result = operation(*args)
            # Keep an assigned return value alive while verifying that the
            # owner was not revoked during the call. A native return that is
            # interrupted before assignment cannot be guaranteed rooted.
            self.roots.pending_return = result
            self.owner.check()
            self._inflight = False
            return result
        except BaseException as exc:
            self._unsafe = True
            self.roots.failure = exc
            raise

    def fail(self, exc):
        self._inflight = True
        self._unsafe = True
        if self.roots.failure is None:
            self.roots.failure = exc


class _SimulatedOwner:
    def check(self):
        return None


class _SimulatedDevice:
    """Deterministic CPU tensors and deliberately fake allocator observations."""
    def __init__(self, session):
        self.session = session
        self.vocab_size, self.hidden_size = 50257, 768
        self._start = self._end = None
        self.memory = {'free_bytes': 6 * protocol.GIB, 'total_bytes': 8 * protocol.GIB,
                       'allocated_bytes': 384 * 1024 ** 2, 'reserved_bytes': 512 * 1024 ** 2}

    def synchronize(self):
        pass

    def default_stream(self):
        return True

    def reset_peaks(self):
        pass

    def snapshot(self):
        return dict(self.memory)

    def peaks(self):
        return self.memory['allocated_bytes'], self.memory['reserved_bytes']

    def start_event(self):
        self._start = time.perf_counter_ns()

    def end_event(self):
        self._end = time.perf_counter_ns()

    def event_ms(self):
        return (self._end - self._start) / 1_000_000

    def execute(self, mode, roots):
        # Fake model and tap use real CPU tensor operations. The reported path
        # and device counters below are simulated, not real GPT-2/CUDA data.
        hidden = torch.arange(1536, dtype=torch.float32).reshape(1, 2, 768)
        roots.output = torch.full((1, 2, 50257), 0.125, dtype=torch.float32)
        if mode not in ('none', 'id'):
            selected = hidden[:, [1], :]
            roots.capture = selected.detach() if mode == 'direct' else gradient_readout(selected)
        if mode != 'none':
            roots.session = {'events': [{'identity': protocol.IDENTITY, 'sequence': 0,
                'run_id': f'{protocol.PROTOCOL}-session-{self.session}', 'mode': mode, 'value': roots.capture,
                'source_shape': [1, 2, self.hidden_size], 'source_dtype': 'torch.float32',
                'source_device': 'cpu'}]}


class _CPUTraceDevice(_SimulatedDevice):
    """Use real CPU GPT-2 hooks with fake device events and counters."""
    def __init__(self, session, model, input_ids):
        super().__init__(session)
        if (type(session) is not int or session not in range(5) or model.training
                or getattr(model.config, 'model_type', None) != 'gpt2'
                or getattr(model.config, '_attn_implementation', None) != 'eager'
                or not isinstance(input_ids, torch.Tensor) or input_ids.device.type != 'cpu'
                or input_ids.dtype != torch.int64 or tuple(input_ids.shape) != (1, 2)
                or input_ids.tolist() != [[15496, 995]]
                or any(value.device.type != 'cpu' for value in list(model.parameters()) + list(model.buffers()))):
            raise ValueError('CPU trace test requires the fixed input and an eager CPU GPT-2 model')
        self.model, self.input_ids = model, input_ids
        self.vocab_size, self.hidden_size = model.config.vocab_size, model.config.n_embd

    def execute(self, mode, roots):
        if mode == 'none':
            roots.output = self.model(input_ids=self.input_ids, use_cache=False, return_dict=True).logits
            return
        from .transformer_trace import GPT2TraceSession
        roots.session = GPT2TraceSession(self.model, run_id=f'{protocol.PROTOCOL}-session-{self.session}',
            mode=mode, targets=('block',), layers=(0,), samples=(0,), tokens=(1,),
            capture_device='source', max_events=2, max_trace_bytes=64 * 1024)
        with roots.session:
            roots.output = self.model(input_ids=self.input_ids, use_cache=False, return_dict=True).logits
        if mode != 'id' and len(roots.session.events) == 1:
            roots.capture = roots.session.events[0].value


class _Journal:
    def __init__(self, path):
        self.handle = Path(path).open('xb')

    def append(self, row):
        raw = (json.dumps(row, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
        if len(raw) > 128 * 1024:
            raise ValueError('journal entry exceeds its byte bound')
        count = self.handle.write(raw)
        if count != len(raw):
            raise OSError('short journal write')
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        self.handle.close()


def _clock():
    info = time.get_clock_info('perf_counter')
    result = {'name': 'perf_counter_ns', 'implementation': info.implementation,
              'monotonic': info.monotonic, 'adjustable': info.adjustable,
              'resolution_seconds': info.resolution}
    protocol.clock_minimum(result)
    return result


def _write_json_new(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + '\n').encode('utf-8')
    with Path(path).open('xb') as handle:
        if handle.write(raw) != len(raw):
            raise OSError('short evidence write')
        handle.flush()
        os.fsync(handle.fileno())


def _validate_output(roots, mode, session, vocab_size=50257, hidden_size=768):
    output = roots.output
    if (not isinstance(output, torch.Tensor) or list(output.shape) != [1, 2, vocab_size]
            or output.dtype != torch.float32 or str(output.device) != 'cpu'
            or not bool(torch.isfinite(output).all())):
        raise ValueError('synthetic output is invalid')
    capture, trace = roots.capture, roots.session
    if mode == 'none':
        if capture is not None or trace is not None:
            raise ValueError('no-capture sample has a trace')
        return {'events': 0, 'recorded_tensor_bytes': 0, 'output_finite': True, 'capture_finite': None,
                'trace_identity': None,
                'trace_sequence': None, 'trace_run_id': None, 'trace_mode': None,
                'value_shape': None, 'value_dtype': None, 'value_device': None}
    if mode not in ('id', 'direct', 'gradient'):
        raise ValueError('invalid capture mode')
    if type(trace) is dict:
        if type(trace.get('events')) is not list or len(trace['events']) != 1:
            raise ValueError('synthetic session must contain one event')
        event = trace['events'][0]
    else:
        from .transformer_trace import GPT2TraceSession
        if type(trace) is not GPT2TraceSession or trace.successful is not True or len(trace.events) != 1:
            raise ValueError('CPU trace session must contain one event')
        item = trace.events[0]
        event = {'identity': [item.module_path, item.module_kind, item.layer_index, item.forward_index,
                              item.invocation, list(item.output_path), list(item.batch_positions),
                              list(item.token_positions)], 'sequence': item.sequence, 'run_id': item.run_id,
                 'mode': item.capture_mode, 'value': item.value, 'source_shape': list(item.shape),
                 'source_dtype': item.dtype, 'source_device': item.device}
        expected_bytes = 0 if mode == 'id' else hidden_size * 4
        if type(trace.tensor_bytes) is not int or trace.tensor_bytes != expected_bytes:
            raise ValueError('CPU trace bytes differ')
    if (type(event) is not dict or event.get('identity') != protocol.IDENTITY
            or type(event.get('sequence')) is not int or event['sequence'] != 0
            or event.get('run_id') != f'{protocol.PROTOCOL}-session-{session}'
            or event.get('mode') != mode or event.get('value') is not capture
            or event.get('source_shape') != [1, 2, hidden_size]
            or event.get('source_dtype') != 'torch.float32' or event.get('source_device') != 'cpu'):
        raise ValueError('synthetic event identity or value differs')
    if mode == 'id':
        if capture is not None:
            raise ValueError('ID-only session retained a value')
        return {'events': 1, 'recorded_tensor_bytes': 0, 'output_finite': True, 'capture_finite': None,
                'trace_identity': event['identity'], 'trace_sequence': event['sequence'],
                'trace_run_id': event['run_id'], 'trace_mode': event['mode'],
                'value_shape': None, 'value_dtype': None, 'value_device': None}
    if (not isinstance(capture, torch.Tensor) or list(capture.shape) != [1, 1, hidden_size]
            or capture.dtype != torch.float32 or str(capture.device) != 'cpu'
            or capture.numel() * capture.element_size() != hidden_size * 4 or not bool(torch.isfinite(capture).all())):
        raise ValueError('synthetic trace is invalid')
    return {'events': 1, 'recorded_tensor_bytes': hidden_size * 4, 'output_finite': True, 'capture_finite': True,
            'trace_identity': event['identity'], 'trace_sequence': event['sequence'],
            'trace_run_id': event['run_id'], 'trace_mode': event['mode'],
            'value_shape': list(capture.shape), 'value_dtype': str(capture.dtype),
            'value_device': 'cpu'}


def _preflight(guard, journal, session, total, cap):
    roots, device = guard.roots, guard.device
    started = time.perf_counter_ns()
    guard.call(device.synchronize)
    before = guard.call(device.snapshot)
    protocol.check_memory(before, expected_total=total, cap_bytes=cap)
    guard.call(device.reset_peaks)
    metadata = {}
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        for mode in ('none', 'id', 'direct', 'gradient'):
            guard.call(device.execute, mode, roots)
            metadata[mode] = guard.call(_validate_output, roots, mode, session,
                                        device.vocab_size, device.hidden_size)
            roots.outputs.append(roots.output)
            if roots.capture is not None:
                roots.captures.append(roots.capture)
            if roots.session is not None:
                roots.sessions.append(roots.session)
            roots.output = roots.capture = roots.session = None
    if (len(roots.outputs) != 4 or len(roots.captures) != 2 or len(roots.sessions) != 3
            or [metadata[mode]['events'] for mode in ('none', 'id', 'direct', 'gradient')] != [0, 1, 1, 1]
            or not all(torch.equal(roots.outputs[0], value) for value in roots.outputs[1:])
            or not torch.equal(roots.captures[0], roots.captures[1])):
        raise ValueError('synthetic preflight equality failed')
    output_shape = list(roots.outputs[0].shape)
    output_dtype, output_device = str(roots.outputs[0].dtype), str(roots.outputs[0].device)
    # Hold every output and capture through completion, then dispose and sync.
    guard.call(device.synchronize)
    roots.outputs.clear()
    roots.captures.clear()
    roots.sessions.clear()
    guard.call(device.synchronize)
    ended = time.perf_counter_ns()
    peak_a, peak_r = guard.call(device.peaks)
    checks = {'outputs_finite': True, 'outputs_exact': True, 'values_exact': True,
              'preflight_modes': list(metadata), 'id_value_absent': metadata['id']['recorded_tensor_bytes'] == 0,
              'run_id': metadata['direct']['trace_run_id'], 'trace_sequences': [metadata['direct']['trace_sequence']],
              'capture_modes': [metadata[mode]['trace_mode'] for mode in ('id', 'direct', 'gradient')],
              'output_shape': output_shape, 'output_dtype': output_dtype, 'output_device': output_device,
              'events_per_mode': metadata['direct']['events'],
              'value_tensor_bytes': metadata['direct']['recorded_tensor_bytes'],
              'event_identities': [metadata['direct']['trace_identity']],
              'value_layouts': [{'shape': metadata['direct']['value_shape'],
                                 'dtype': metadata['direct']['value_dtype'], 'device': metadata['direct']['value_device']}],
              'comparison': {'operator': 'torch.equal', 'tolerance': 'none',
                             'guarantees_bitwise_identity': False, 'subnormal_handling': 'backend-dependent'}}
    journal.append({'stage': 'preflight', 'observation_id': 'preflight', 'postcheck_status': 'pending',
                    'started_ns': started, 'ended_ns': ended,
                    'checks': checks, 'memory_before': before, 'peak_allocated_bytes': peak_a,
                    'peak_reserved_bytes': peak_r})
    after = guard.call(device.snapshot)
    protocol.check_memory(after, expected_total=total, cap_bytes=cap)
    journal.append({'stage': 'postcheck', 'observation_id': 'preflight', 'status': 'passed',
                    'memory_after': after})


def _sample(guard, journal, item, session, total, cap, minimum_wall_ns):
    roots, device = guard.roots, guard.device
    guard.call(device.synchronize)
    if guard.call(device.default_stream) is not True:
        raise ValueError('default stream is required')
    guard.call(device.reset_peaks)
    before = guard.call(device.snapshot)
    protocol.check_memory(before, expected_total=total, cap_bytes=cap)
    started = time.perf_counter_ns()
    guard.call(device.start_event)
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        guard.call(device.execute, item['mode'], roots)
        metadata = guard.call(_validate_output, roots, item['mode'], session,
                              device.vocab_size, device.hidden_size)
        output_metadata = {'output_shape': list(roots.output.shape), 'output_dtype': str(roots.output.dtype),
                           'output_device': str(roots.output.device)}
    if guard.call(device.default_stream) is not True:
        raise ValueError('stream changed during sample')
    guard.call(device.end_event)
    guard.call(device.synchronize)
    event_ms = guard.call(device.event_ms)
    roots.output = roots.capture = roots.session = None
    guard.call(device.synchronize)
    ended = time.perf_counter_ns()
    peak_a, peak_r = guard.call(device.peaks)
    row = {**item, 'session': session, 'postcheck_status': 'pending', 'started_ns': started,
           'ended_ns': ended, 'wall_ns': ended - started, 'device_event_ms': event_ms,
           'memory_before': before, 'peak_allocated_bytes': peak_a, 'peak_reserved_bytes': peak_r,
           'default_stream': True, **output_metadata, **metadata}
    journal.append(row)
    if row['wall_ns'] < minimum_wall_ns:
        raise ValueError('sample is too short for the clock resolution')
    after = guard.call(device.snapshot)
    protocol.check_memory(after, expected_total=total, cap_bytes=cap)
    journal.append({'stage': 'postcheck', 'observation_id': item['observation_id'],
                    'status': 'passed', 'memory_after': after})


def run_cpu_simulation(schedule_path, *, session, output):
    """Exercise one session with fake CPU timing/telemetry; never a GPU result."""
    global _OWNER_USED
    schedule = protocol.load_schedule(schedule_path)
    order = protocol.session_order(schedule, session)
    clock = _clock()
    guard = _GuardedDevice(_SimulatedDevice(session), _SimulatedOwner())
    # One owner per process, retained on every uncertain exit. A future CUDA
    # owner needs independently reviewed roots, claims and supervisor wiring.
    with _OWNER_LOCK:
        if _OWNER_USED:
            raise RuntimeError('this process has already started an owner')
        _OWNER_USED = True
        _OWNER_ROOT[0] = guard
    out = Path(output)
    journal = None
    created = False
    progress = {'stage': 'setup', 'observation_id': None}
    total, cap = 8 * protocol.GIB, 2 * protocol.GIB
    try:
        out.mkdir(exist_ok=False)
        created = True
        _write_json_new(out / 'started.json', {'protocol': protocol.PROTOCOL, 'session': session,
                        'schedule_sha256': protocol.SCHEDULE_SHA256, 'synthetic_cpu_simulation': True,
                        'started_ns': time.perf_counter_ns()})
        journal = _Journal(out / 'observations.jsonl')
        progress = {'stage': 'preflight', 'observation_id': 'preflight'}
        _preflight(guard, journal, session, total, cap)
        for item in order:
            progress = {'stage': item['stage'], 'observation_id': item['observation_id']}
            _sample(guard, journal, item, session, total, cap, protocol.clock_minimum(clock))
        progress = {'stage': 'summary', 'observation_id': None}
        journal.close()
        rows = [json.loads(line) for line in (out / 'observations.jsonl').read_bytes().splitlines()]
        summary = protocol.summarize_session(schedule, session, rows, clock=clock,
                                             total_bytes=total, cap_bytes=cap, device='cpu')
        result = {'protocol': protocol.PROTOCOL, 'session': session, 'clock': clock,
                  'synthetic_cpu_simulation': True, 'gpu_eligible': False,
                  'device_event_and_memory_are_simulated': True, 'summary': summary}
        _write_json_new(out / 'simulation.json', result)
        _write_json_new(out / 'completed.json', {'protocol': protocol.PROTOCOL, 'session': session,
                        'synthetic_cpu_simulation': True,
                        'simulation_sha256': sha256((out / 'simulation.json').read_bytes()).hexdigest(),
                        'journal_sha256': sha256((out / 'observations.jsonl').read_bytes()).hexdigest()})
        _OWNER_ROOT[0] = None
        return result
    except BaseException as exc:
        guard.fail(exc)
        # No device calls in this failure path. The caller/process owns the
        # guard and its roots until termination; CPU simulation only today.
        try:
            if journal is not None:
                journal.close()
        except BaseException as cleanup:
            guard.roots.cleanup_failure = cleanup
        try:
            if created:
                _write_json_new(out / 'failure.json', {'protocol': protocol.PROTOCOL, 'session': session,
                                'synthetic_cpu_simulation': True, 'stage': progress['stage'],
                                'observation_id': progress['observation_id'], 'error_type': type(exc).__name__,
                                'unsafe': True, 'completion_verified': False})
        except BaseException as cleanup:
            guard.roots.cleanup_failure = cleanup
        raise


def run_gpu(*_args, **_kwargs):
    """Fail before touching CUDA: no reviewed/admitted GPU owner exists yet."""
    raise RuntimeError('repeated GPU tap benchmark admission is not available')


def inspect_cpu_simulation(directory, *, schedule_path, child_returncode, child_reaped):
    """Reject partial/conflicting fake terminals; never produce GPU evidence.

    Child exit/reap is a trusted external observation, not authenticated here.
    """
    root = Path(directory)
    failed, completed = root / 'failure.json', root / 'completed.json'
    if failed.exists() and completed.exists():
        return {'state': 'incomplete', 'reason': 'conflicting_terminal_records', 'gpu_eligible': False}
    if failed.exists():
        return {'state': 'failed', 'reason': 'owner_failure_record', 'gpu_eligible': False}
    if (not completed.is_file() or type(child_returncode) is not int or child_returncode != 0
            or child_reaped is not True):
        return {'state': 'incomplete', 'reason': 'terminal_or_reap_unverified', 'gpu_eligible': False}
    try:
        def read_limited(path, limit):
            with path.open('rb') as stream:
                value = stream.read(limit + 1)
            if len(value) > limit:
                raise ValueError('evidence exceeds its bound')
            return value
        terminal = json.loads(read_limited(completed, 16 * 1024))
        simulation_bytes = read_limited(root / 'simulation.json', 1024 * 1024)
        journal_bytes = read_limited(root / 'observations.jsonl', 16 * 1024 * 1024)
        simulation = json.loads(simulation_bytes)
        if (type(terminal) is not dict or terminal.get('protocol') != protocol.PROTOCOL
                or terminal.get('synthetic_cpu_simulation') is not True
                or terminal.get('simulation_sha256') != sha256(simulation_bytes).hexdigest()
                or terminal.get('journal_sha256') != sha256(journal_bytes).hexdigest()
                or type(simulation) is not dict or simulation.get('synthetic_cpu_simulation') is not True
                or simulation.get('gpu_eligible') is not False
                or terminal.get('session') != simulation.get('session') or not journal_bytes.endswith(b'\n')):
            raise ValueError('terminal identity or hashes differ')
        rows = [json.loads(line) for line in journal_bytes.splitlines()]
        summary = protocol.summarize_session(protocol.load_schedule(schedule_path), simulation['session'], rows,
            clock=simulation['clock'], total_bytes=8 * protocol.GIB, cap_bytes=2 * protocol.GIB, device='cpu')
        if simulation.get('summary') != summary:
            raise ValueError('summary differs from the journal')
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return {'state': 'incomplete', 'reason': 'invalid_terminal_or_evidence', 'gpu_eligible': False}
    return {'state': 'completed_simulation', 'session': simulation['session'], 'gpu_eligible': False,
            'summary': summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--session', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    run_cpu_simulation(args.schedule, session=args.session, output=args.output)


if __name__ == '__main__':
    main()
