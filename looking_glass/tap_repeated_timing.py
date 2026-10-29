"""Disabled fixed-path timing core for the prospective repeated tap owner.

The backend is a trusted test seam, not a CUDA admission or supervision
interface. No real backend or runnable GPU entry is supplied by this module.
The future owner must bind its exact device/runtime, memory admission, grants,
source-stamped journal, and hard-exit lifecycle before enabling anything.
"""

import gc
import math
import time
import weakref

from . import tap_repeated_owner as owner
from .tap_repeated_protocol import (EVENT_ENCLOSURE_TOLERANCE_NS, EXPERIMENT_PROTOCOL,
                                    FINAL_POSTCHECK_MAX_NS, GIB, HEADROOM, IDENTITY, OUTSIDE_CAP,
                                    check_memory, clock_minimum)


_TIMING_ENABLED = False


class _TimedSample:
    """Run one sequential sample with the same top-level guards per mode."""

    def __init__(self, operations, runtime, *, session, clock_info, clock=time.perf_counter_ns):
        if _TIMING_ENABLED is not True:
            raise RuntimeError('repeated tap timing remains disabled')
        if (type(operations) is not owner._OwnerOperations or type(session) is not int
                or session not in range(5) or not callable(clock)
                or type(runtime.device) is not str or runtime.device not in ('cpu', 'cuda:0')
                or type(runtime.vocab_size) is not int or runtime.vocab_size <= 0
                or type(runtime.hidden_size) is not int or runtime.hidden_size <= 0):
            raise ValueError('invalid fixed timing configuration')
        if operations._gc_disabled_required is not True or gc.isenabled():
            raise RuntimeError('the owner must latch disabled cyclic collection first')
        if operations._benchmark_session != session:
            raise RuntimeError('the fixed schedule slot is not bound')
        self.operations, self.runtime, self.session, self.clock = operations, runtime, session, clock
        self.minimum_wall_ns = clock_minimum(clock_info)
        self.state = 'ready'
        self._preflight_verified = False
        self.failure = None
        self.pending_return = None
        self.partial = None

    def _none_call(self, callback):
        def run():
            value = None
            try:
                value = callback()
                self.pending_return = value
                if value is not None:
                    raise RuntimeError('device callback returned an unexpected object')
            except BaseException:
                if value is not None:
                    self.pending_return = value
                raise
        self.operations.call(run)

    def _snapshot(self):
        return self._snapshot_with(self.operations.call)

    def _snapshot_in_guard(self):
        """Read within an already-guarded preflight operation, not independently."""
        return self._snapshot_with(lambda callback: callback())

    def _snapshot_with(self, invoke):
        operations, runtime = self.operations, self.runtime
        free_total = invoke(runtime.free_total_bytes)
        total, cap = operations.admitted.total_bytes, operations.admitted.cap_bytes
        if (type(free_total) is not tuple or len(free_total) != 2
                or any(type(value) is not int for value in free_total)
                or free_total[1] != total or not 0 <= free_total[0] <= total
                or free_total[0] < max(GIB, (15 * total + 99) // 100) + HEADROOM + OUTSIDE_CAP):
            raise ValueError('invalid device free/total observation')
        allocated = invoke(runtime.allocated_bytes)
        if type(allocated) is not int or not 0 <= allocated <= cap - HEADROOM:
            raise ValueError('invalid allocated-byte observation')
        reserved = invoke(runtime.reserved_bytes)
        row = {'free_bytes': free_total[0], 'total_bytes': free_total[1],
               'allocated_bytes': allocated, 'reserved_bytes': reserved}
        check_memory(row, expected_total=total, cap_bytes=cap)
        return row

    def _new_trace(self, mode):
        if mode == 'none':
            return None
        from .transformer_trace import GPT2TraceSession
        return GPT2TraceSession(self.operations.roots.model,
            run_id=f'{EXPERIMENT_PROTOCOL}-session-{self.session}', mode=mode,
            targets=('block',), layers=(0,), samples=(0,), tokens=(1,),
            capture_device='source', max_events=2, max_trace_bytes=64 * 1024)

    def _forward(self):
        import torch
        roots = self.operations.roots
        if roots.model.training:
            raise ValueError('the fixed model must be in evaluation mode')
        with torch.no_grad(), torch.autocast(device_type=self.runtime.device.split(':')[0], enabled=False):
            return roots.model(input_ids=roots.input_ids, use_cache=False, return_dict=True)

    def _validate(self, mode):
        import torch
        from .transformer_trace import GPT2TraceSession
        roots, runtime = self.operations.roots, self.runtime
        output, trace, capture = roots.output, roots.trace, roots.capture
        result = roots.model_result
        if (roots.model.training or not hasattr(result, 'past_key_values')
                or result.past_key_values is not None
                or getattr(result, 'hidden_states', None) is not None
                or getattr(result, 'attentions', None) is not None
                or getattr(result, 'cross_attentions', None) is not None
                or getattr(result, 'loss', None) is not None
                or not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 2, runtime.vocab_size)
                or output.dtype != torch.float32 or str(output.device) != runtime.device
                or not bool(torch.isfinite(output).all())):
            raise ValueError('fixed model output is invalid')
        base = {'output_shape': list(output.shape), 'output_dtype': str(output.dtype),
                'output_device': str(output.device), 'output_finite': True}
        if mode == 'none':
            if trace is not None or capture is not None or roots.trace_events is not None:
                raise ValueError('uncaptured mode retained a trace')
            return {**base, 'events': 0, 'recorded_tensor_bytes': 0, 'capture_finite': None,
                    'trace_identity': None, 'trace_sequence': None, 'trace_run_id': None,
                    'trace_mode': None, 'value_shape': None, 'value_dtype': None, 'value_device': None}
        if (type(trace) is not GPT2TraceSession or trace.successful is not True
                or trace.events is not roots.trace_events or len(trace.events) != 1):
            raise ValueError('fixed capture did not complete once')
        event = trace.events[0]
        identity = [event.module_path, event.module_kind, event.layer_index, event.forward_index,
                    event.invocation, list(event.output_path), list(event.batch_positions), list(event.token_positions)]
        if (identity != IDENTITY or type(event.sequence) is not int or event.sequence != 0
                or event.run_id != f'{EXPERIMENT_PROTOCOL}-session-{self.session}'
                or event.capture_mode != mode or event.value is not capture
                or tuple(event.shape) != (1, 2, runtime.hidden_size)
                or event.dtype != 'torch.float32' or event.device != runtime.device
                or not isinstance(capture, torch.Tensor)
                or tuple(capture.shape) != (1, 1, runtime.hidden_size)
                or capture.dtype != torch.float32 or str(capture.device) != runtime.device
                or trace.tensor_bytes != runtime.hidden_size * 4
                or not bool(torch.isfinite(capture).all())):
            raise ValueError('fixed trace output or identity is invalid')
        return {**base, 'events': 1, 'recorded_tensor_bytes': runtime.hidden_size * 4,
                'capture_finite': True, 'trace_identity': identity, 'trace_sequence': event.sequence,
                'trace_run_id': event.run_id, 'trace_mode': mode, 'value_shape': list(capture.shape),
                'value_dtype': str(capture.dtype), 'value_device': str(capture.device)}

    def _sample(self, mode):
        operations, roots, runtime = self.operations, self.operations.roots, self.runtime

        def require_default(slot):
            if runtime.is_default_stream(slot) is not True:
                raise RuntimeError('the fixed default stream is required')

        def require_session():
            if operations._benchmark_session != self.session:
                raise RuntimeError('the fixed schedule slot changed')

        def require_preflight_empty():
            if any(getattr(roots, f'preflight_{kind}_{index}') is not None
                   for kind, count in (('trace', 3), ('events', 3), ('result', 4), ('output', 4), ('capture', 2))
                   for index in range(count)):
                raise RuntimeError('preflight ownership remains')
            if any(getattr(roots, name) is not None
                   for name in ('model_bundle', 'model_result', 'output', 'capture', 'trace', 'trace_events',
                                'current_stream_before', 'current_stream_after')):
                raise RuntimeError('a prior sample remains rooted')

        names = ('model_result', 'output', 'capture', 'trace', 'trace_events',
                 'current_stream_before', 'current_stream_after')

        def transient_refs():
            return tuple(weakref.ref(value) for value in (getattr(roots, name) for name in names)
                         if value is not None)

        def require_disposed():
            if any(ref() is not None for ref in references):
                raise RuntimeError('a synchronized transient remains externally reachable')

        operations.call(require_session)
        operations.call(require_preflight_empty)
        self._none_call(runtime.synchronize)
        self._none_call(runtime.reset_peaks)
        before = self._snapshot()
        self.partial['memory_before'] = before
        operations.call(runtime.current_stream, root='current_stream_before')
        operations.call(require_default, 'current_stream_before')
        started = operations.call(self.clock)
        if type(started) is not int or started < 0:
            raise ValueError('invalid sample start clock')
        self.partial.update(stage='running', started_ns=started)
        self._none_call(runtime.record_start)
        operations.call(lambda: self._new_trace(mode), root='trace')
        if mode == 'none':
            operations.call(lambda: None, root='trace_events')
            operations.call(lambda: None)
        else:
            operations.bind_trace_events('trace', session=self.session)
        operations.call(lambda: None if roots.trace is None else roots.trace.__enter__())
        operations.call(self._forward, root='model_result')
        operations.call(lambda: None if roots.trace is None else roots.trace.__exit__(None, None, None))
        operations.call(lambda: roots.model_result.logits, root='output')
        operations.call(lambda: None if roots.trace is None else roots.trace.events[0].value, root='capture')
        metadata = operations.call(lambda: self._validate(mode))
        self.partial['metadata'] = metadata
        operations.call(runtime.current_stream, root='current_stream_after')
        operations.call(require_default, 'current_stream_after')
        self._none_call(runtime.record_end)
        self._none_call(runtime.synchronize)
        event_ms = operations.call(runtime.elapsed_ms)
        if type(event_ms) not in (int, float) or not math.isfinite(event_ms) or event_ms < 0:
            raise ValueError('invalid device event duration')
        self.partial['device_event_ms'] = event_ms
        references = operations.call(transient_refs)
        operations.clear_after_synchronization(*names)
        operations.call(require_disposed)
        self._none_call(runtime.synchronize)
        ended = operations.call(self.clock)
        if type(ended) is not int or ended <= started:
            raise ValueError('invalid sample end clock')
        self.partial.update(stage='postcheck_pending', ended_ns=ended, wall_ns=ended - started)
        if (ended - started < self.minimum_wall_ns
                or event_ms * 1_000_000 > ended - started + EVENT_ENCLOSURE_TOLERANCE_NS):
            raise ValueError('sample clock or event interval is inconsistent')
        peak_allocated = operations.call(runtime.peak_allocated_bytes)
        if (type(peak_allocated) is not int
                or not before['allocated_bytes'] <= peak_allocated <= operations.admitted.cap_bytes - HEADROOM):
            raise ValueError('invalid peak allocated-byte observation')
        self.partial['peak_allocated_bytes'] = peak_allocated
        peak_reserved = operations.call(runtime.peak_reserved_bytes)
        if (type(peak_reserved) is not int
                or not max(before['reserved_bytes'], peak_allocated) <= peak_reserved <= operations.admitted.cap_bytes - HEADROOM):
            raise ValueError('invalid peak reserved-byte observation')
        self.partial['peak_reserved_bytes'] = peak_reserved
        after = self._snapshot()
        self.partial['memory_after'] = after
        if (type(peak_allocated) is not int or type(peak_reserved) is not int
                or not 0 <= peak_allocated <= peak_reserved <= operations.admitted.cap_bytes - HEADROOM
                or peak_allocated < max(before['allocated_bytes'], after['allocated_bytes'])
                or peak_reserved < max(before['reserved_bytes'], after['reserved_bytes'])):
            raise ValueError('invalid peak allocator observation')
        postcheck = operations.call(self.clock)
        if type(postcheck) is int:
            self.partial['postcheck_completed_ns'] = postcheck
        if type(postcheck) is not int or not ended <= postcheck <= ended + FINAL_POSTCHECK_MAX_NS:
            raise ValueError('final postcheck timing is invalid')
        self.partial['stage'] = 'postcheck_complete'
        return {**metadata, 'started_ns': started, 'ended_ns': ended,
                'wall_ns': ended - started, 'device_event_ms': event_ms, 'default_stream': True,
                'memory_before': before, 'memory_after': after, 'peak_allocated_bytes': peak_allocated,
                'peak_reserved_bytes': peak_reserved, 'postcheck_completed_ns': postcheck}

    def sample(self, mode):
        if self.state != 'ready':
            raise RuntimeError('timing owner is not ready')
        kind = None
        try:
            self.state = 'running'
            if mode not in ('none', 'direct', 'gradient'):
                raise ValueError('invalid fixed capture mode')
            if self.runtime.device == 'cuda:0' and self._preflight_verified is not True:
                raise RuntimeError('the guarded preflight has not completed')
            self.partial = {'mode': mode, 'stage': 'pre_sample'}
            result = self._sample(mode)
            self.state = 'ready'
            return result
        except BaseException as exc:
            # Keep both an existing private operation failure and this outer
            # failure, and emit a detached public category. An interruption
            # in this handler still requires immediate process replacement.
            self.failure = exc
            self.operations.unsafe = True
            if self.operations.roots.failure is None:
                self.operations.roots.failure = exc
            self.state = 'failed'
            kind = owner._failure_kind(exc)
        owner._raise_detached(kind)
