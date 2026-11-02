"""Disabled four-path preflight for the prospective repeated tap owner.

This seam requires the separately disabled timing/owner guards. It is not a
supervisor, admission, physical-device binding, or a runnable GPU entrypoint.
Its gate is a trusted, non-device callback: the eventual owner must bind it
to the live parent grant and stop the process after any escaping failure.
"""

import weakref

from . import tap_repeated_owner as owner
from . import tap_repeated_timing as timing
from .tap_repeated_protocol import EXPERIMENT_PROTOCOL, FINAL_POSTCHECK_MAX_NS, HEADROOM, IDENTITY


_PREFLIGHT_NAMES = (tuple(f'preflight_result_{i}' for i in range(4))
                    + tuple(f'preflight_output_{i}' for i in range(4))
                    + tuple(f'preflight_capture_{i}' for i in range(2))
                    + tuple(f'preflight_trace_{i}' for i in range(3))
                    + tuple(f'preflight_events_{i}' for i in range(3))
                    + ('current_stream_before', 'current_stream_after'))
_START_GATE_MAX_NS = 250_000_000


class _FixedPreflight:
    """Compare all fixed paths while retaining their known device owners."""

    def __init__(self, sample):
        if (type(sample) is not timing._TimedSample or sample.state != 'ready'
                or sample._preflight_verified is not False):
            raise ValueError('the exact ready timing owner is required')
        self.sample, self.operations = sample, sample.operations
        self.state, self.failure, self.partial = 'new', None, None

    def _validate(self, index, mode):
        import torch
        from .transformer_trace import GPT2TraceSession
        roots, runtime = self.operations.roots, self.sample.runtime
        output = getattr(roots, f'preflight_output_{index}')
        result = getattr(roots, f'preflight_result_{index}')
        if (roots.model.training or not hasattr(result, 'past_key_values')
                or result.past_key_values is not None
                or getattr(result, 'hidden_states', None) is not None
                or getattr(result, 'attentions', None) is not None
                or getattr(result, 'cross_attentions', None) is not None
                or getattr(result, 'loss', None) is not None
                or not isinstance(output, torch.Tensor)
                or tuple(output.shape) != (1, 2, runtime.vocab_size)
                or output.dtype != torch.float32 or str(output.device) != runtime.device
                or not bool(torch.isfinite(output).all())):
            raise ValueError('fixed preflight output is invalid')
        base = {'output_shape': list(output.shape), 'output_dtype': str(output.dtype),
                'output_device': str(output.device), 'events': 0, 'recorded_tensor_bytes': 0}
        if mode == 'none':
            return base
        trace_index = index - 1
        trace = getattr(roots, f'preflight_trace_{trace_index}')
        events = getattr(roots, f'preflight_events_{trace_index}')
        if (type(trace) is not GPT2TraceSession or trace.successful is not True
                or trace.events is not events or len(events) != 1):
            raise ValueError('preflight trace did not complete once')
        event = events[0]
        identity = [event.module_path, event.module_kind, event.layer_index, event.forward_index,
                    event.invocation, list(event.output_path), list(event.batch_positions), list(event.token_positions)]
        if (identity != IDENTITY or type(event.sequence) is not int or event.sequence != 0
                or event.run_id != f'{EXPERIMENT_PROTOCOL}-session-{self.sample.session}'
                or event.capture_mode != mode or tuple(event.shape) != (1, 2, runtime.hidden_size)
                or event.dtype != 'torch.float32' or event.device != runtime.device):
            raise ValueError('preflight trace identity is invalid')
        base.update(events=1, trace_identity=identity, trace_sequence=event.sequence,
                    trace_run_id=event.run_id, trace_mode=mode)
        if mode == 'id':
            if event.value is not None or trace.tensor_bytes != 0:
                raise ValueError('preflight ID-only trace retained a value')
            return base
        capture = getattr(roots, f'preflight_capture_{index - 2}')
        if (event.value is not capture or not isinstance(capture, torch.Tensor)
                or tuple(capture.shape) != (1, 1, runtime.hidden_size)
                or capture.dtype != torch.float32 or str(capture.device) != runtime.device
                or trace.tensor_bytes != runtime.hidden_size * 4
                or not bool(torch.isfinite(capture).all())):
            raise ValueError('preflight capture is invalid')
        return {**base, 'recorded_tensor_bytes': trace.tensor_bytes,
                'value_shape': list(capture.shape), 'value_dtype': str(capture.dtype),
                'value_device': str(capture.device)}

    def _prepare(self):
        operations, sample, roots, runtime = (self.operations, self.sample,
                                               self.operations.roots, self.sample.runtime)

        def require_empty():
            names = _PREFLIGHT_NAMES + ('model_bundle', 'model_result', 'output', 'capture', 'trace', 'trace_events')
            if any(getattr(roots, name) is not None for name in names):
                raise RuntimeError('preflight or timed resources are already occupied')
            if operations._benchmark_session != sample.session:
                raise RuntimeError('preflight schedule slot changed')

        def require_default(slot):
            if runtime.is_default_stream(slot) is not True:
                raise RuntimeError('the fixed default stream is required')

        operations.call(require_empty)
        sample._none_call(runtime.synchronize)
        sample._none_call(runtime.reset_peaks)
        self.partial['preparation_memory'] = sample._snapshot()
        operations.call(runtime.current_stream, root='current_stream_before')
        operations.call(require_default, 'current_stream_before')

    def _run(self, gate):
        operations, sample, roots, runtime = (self.operations, self.sample,
                                               self.operations.roots, self.sample.runtime)

        def require_default(slot):
            if runtime.is_default_stream(slot) is not True:
                raise RuntimeError('the fixed default stream is required')

        def first_forward():
            import torch
            if roots.model.training:
                raise ValueError('the fixed model must be in evaluation mode')
            with torch.no_grad(), torch.autocast(device_type=runtime.device.split(':')[0], enabled=False):
                memory_started = sample.clock()
                if type(memory_started) is not int or not 0 <= memory_started < 2 ** 63:
                    raise ValueError('preflight memory clock is invalid')
                # The enclosing owner operation guards this sequential CUDA
                # snapshot. Validate each value before issuing the next query.
                self.partial['memory_before'] = sample._snapshot_in_guard()
                memory_ended = sample.clock()
                if type(memory_ended) is not int or not memory_started <= memory_ended < 2 ** 63:
                    raise ValueError('preflight memory clock is invalid')
                self.partial.update(memory_snapshot_started_ns=memory_started,
                                    memory_snapshot_ended_ns=memory_ended)
                lower = sample.clock()
                if (type(lower) is not int or not memory_ended <= lower
                        or not lower < 2 ** 63 - _START_GATE_MAX_NS):
                    raise ValueError('preflight gate clock is invalid')
                upper = lower + _START_GATE_MAX_NS
                self.partial.update(stage='gate_pending', gate_lower_ns=lower, gate_upper_ns=upper)
                # Validate early ordering at lower and expiry through upper.
                # A delayed gate cannot backdate a forward into a valid window.
                if gate(lower, upper) is not None:
                    raise RuntimeError('start gate did not affirm authorization')
                # This is a direct check, not a nested operations.call.
                operations._check()
                started = sample.clock()
                if (type(started) is not int or not lower <= started <= upper
                        or started - memory_started > 2_000_000_000):
                    raise ValueError('preflight start gate took too long')
                self.partial.update(stage='first_forward_attempt', started_ns=started)
                return roots.model(input_ids=roots.input_ids, use_cache=False, return_dict=True)

        def refs():
            return tuple(weakref.ref(getattr(roots, name)) for name in _PREFLIGHT_NAMES
                         if getattr(roots, name) is not None)

        def disposed():
            if any(ref() is not None for ref in references):
                raise RuntimeError('a synchronized preflight resource remains externally reachable')

        operations.call(first_forward, root='preflight_result_0')
        before = self.partial['memory_before']
        operations.call(lambda: roots.preflight_result_0.logits, root='preflight_output_0')
        metadata = {'none': operations.call(self._validate, 0, 'none')}
        for index, mode in enumerate(('id', 'direct', 'gradient'), start=1):
            trace_slot = f'preflight_trace_{index - 1}'
            event_slot = f'preflight_events_{index - 1}'
            operations.call(lambda: sample._new_trace(mode), root=trace_slot)
            operations.bind_trace_events(trace_slot, session=sample.session)
            operations.call(lambda: getattr(roots, trace_slot).__enter__())
            operations.call(sample._forward, root=f'preflight_result_{index}')
            operations.call(lambda: getattr(roots, trace_slot).__exit__(None, None, None))
            operations.call(lambda: getattr(roots, f'preflight_result_{index}').logits,
                            root=f'preflight_output_{index}')
            if mode != 'id':
                operations.call(lambda: getattr(roots, event_slot)[0].value,
                                root=f'preflight_capture_{index - 2}')
            metadata[mode] = operations.call(self._validate, index, mode)
        import torch
        for index in range(1, 4):
            if operations.call(lambda: torch.equal(roots.preflight_output_0,
                                                    getattr(roots, f'preflight_output_{index}'))) is not True:
                raise ValueError('preflight outputs differ')
        if operations.call(lambda: torch.equal(roots.preflight_capture_0, roots.preflight_capture_1)) is not True:
            raise ValueError('preflight captures differ')
        operations.call(runtime.current_stream, root='current_stream_after')
        operations.call(require_default, 'current_stream_after')
        sample._none_call(runtime.synchronize)
        references = operations.call(refs)
        operations.clear_after_synchronization(*_PREFLIGHT_NAMES)
        operations.call(disposed)
        sample._none_call(runtime.synchronize)
        ended = operations.call(sample.clock)
        started = self.partial['started_ns']
        if type(ended) is not int or not started < ended < 2 ** 63:
            raise ValueError('preflight end clock is invalid')
        self.partial.update(stage='postcheck_pending', ended_ns=ended)
        peak_a = operations.call(runtime.peak_allocated_bytes)
        if (type(peak_a) is not int
                or not before['allocated_bytes'] <= peak_a <= operations.admitted.cap_bytes - HEADROOM):
            raise ValueError('invalid preflight peak allocation')
        self.partial['peak_allocated_bytes'] = peak_a
        peak_r = operations.call(runtime.peak_reserved_bytes)
        if (type(peak_r) is not int
                or not max(before['reserved_bytes'], peak_a) <= peak_r <= operations.admitted.cap_bytes - HEADROOM):
            raise ValueError('invalid preflight peak reservation')
        self.partial['peak_reserved_bytes'] = peak_r
        after = sample._snapshot()
        self.partial['memory_after'] = after
        if peak_a < after['allocated_bytes'] or peak_r < after['reserved_bytes']:
            raise ValueError('preflight peak counters precede the final snapshot')
        completed = operations.call(sample.clock)
        if type(completed) is not int or not ended <= completed <= ended + FINAL_POSTCHECK_MAX_NS:
            raise ValueError('preflight postcheck timing is invalid')
        direct = metadata['direct']
        checks = {'outputs_finite': True, 'outputs_exact': True, 'values_exact': True,
                  'preflight_modes': list(metadata), 'id_value_absent': metadata['id']['recorded_tensor_bytes'] == 0,
                  'run_id': direct['trace_run_id'], 'trace_sequences': [direct['trace_sequence']],
                  'capture_modes': [metadata[mode]['trace_mode'] for mode in ('id', 'direct', 'gradient')],
                  'output_shape': direct['output_shape'], 'output_dtype': direct['output_dtype'],
                  'output_device': direct['output_device'], 'events_per_mode': direct['events'],
                  'value_tensor_bytes': direct['recorded_tensor_bytes'],
                  'event_identities': [direct['trace_identity']],
                  'value_layouts': [{'shape': direct['value_shape'], 'dtype': direct['value_dtype'],
                                     'device': direct['value_device']}],
                  'comparison': {'operator': 'torch.equal', 'tolerance': 'none',
                                 'guarantees_bitwise_identity': False,
                                 'subnormal_handling': 'backend-dependent'}}
        self.partial['stage'] = 'complete'
        return ({'stage': 'preflight', 'observation_id': 'preflight', 'postcheck_status': 'pending',
                 'started_ns': started, 'ended_ns': ended, 'checks': checks, 'memory_before': before,
                 'gate_lower_ns': self.partial['gate_lower_ns'], 'gate_upper_ns': self.partial['gate_upper_ns'],
                 'memory_snapshot_started_ns': self.partial['memory_snapshot_started_ns'],
                 'memory_snapshot_ended_ns': self.partial['memory_snapshot_ended_ns'],
                 'peak_allocated_bytes': peak_a, 'peak_reserved_bytes': peak_r},
                {'stage': 'postcheck', 'observation_id': 'preflight', 'status': 'passed',
                 'completed_ns': completed, 'memory_after': after})

    def prepare(self):
        if self.state != 'new':
            raise RuntimeError('preflight is single-use')
        kind = None
        try:
            self.state = 'preparing'
            self.partial = {'stage': 'preflight_setup'}
            if self.sample.state != 'ready' or self.sample._preflight_verified is not False:
                raise RuntimeError('this timing owner has already begun preflight')
            self.sample.state = 'preflight_running'
            self._prepare()
            self.partial['stage'] = 'prepared'
            self.state = 'prepared'
            return None
        except BaseException as exc:
            self.failure = exc
            self.operations.unsafe = True
            if self.operations.roots.failure is None:
                self.operations.roots.failure = exc
            self.sample.state = 'failed'
            self.state = 'failed'
            kind = owner._failure_kind(exc)
        owner._raise_detached(kind)

    def run(self, gate):
        if self.state != 'prepared':
            raise RuntimeError('preflight preparation is required')
        kind = None
        try:
            self.state = 'running'
            if not callable(gate):
                raise ValueError('a live start-grant gate is required')
            if self.sample.state != 'preflight_running' or self.sample._preflight_verified is not False:
                raise RuntimeError('this timing owner has already begun preflight')
            result = self._run(gate)
            self.sample._preflight_verified = True
            self.sample.state = 'ready'
            self.state = 'complete'
            return result
        except BaseException as exc:
            self.failure = exc
            self.operations.unsafe = True
            if self.operations.roots.failure is None:
                self.operations.roots.failure = exc
            self.sample.state = 'failed'
            self.state = 'failed'
            kind = owner._failure_kind(exc)
        owner._raise_detached(kind)
