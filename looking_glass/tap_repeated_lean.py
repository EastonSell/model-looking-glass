"""Disabled lean, completion-synchronized repeated-tap timing seam.

The whole measured workload is one owner-guarded operation. It intentionally
does not recheck the OS lease between internal CUDA calls. A loss during that
operation can therefore be observed only at its postcheck; this requires the
documented cooperative supervisor/lease and immediate fail-stop on error.
"""

import math
import weakref

from . import tap_repeated_owner as owner
from . import tap_repeated_timing as timing
from .tap_repeated_protocol import EVENT_ENCLOSURE_TOLERANCE_NS, FINAL_POSTCHECK_MAX_NS, HEADROOM


_LEAN_ENABLED = False
_TRANSIENTS = ('model_result', 'output', 'capture', 'trace', 'trace_events',
               'current_stream_before', 'current_stream_after')


class _LeanTimedSample:
    """Run a fixed mode with one guarded, completion-synchronized workload."""

    def __init__(self, sample):
        if _LEAN_ENABLED is not True:
            raise RuntimeError('lean repeated tap timing remains disabled')
        if (type(sample) is not timing._TimedSample or sample.state != 'ready'
                or sample._preflight_verified is not True):
            raise ValueError('one completed preflight on the exact timing owner is required')
        self.sample, self.operations = sample, sample.operations
        self.state, self.failure, self.partial = 'ready', None, None

    def _capture(self, slot, callback):
        """Root an assigned Python return before issuing the next operation."""
        roots = self.operations.roots
        value = None
        try:
            if getattr(roots, slot) is not None:
                raise RuntimeError('a lean sample root is already occupied')
            value = callback()
            setattr(roots, slot, value)
        except BaseException:
            if value is not None:
                roots.pending_return = value
            raise

    def _bind_trace(self, mode):
        from .transformer_trace import GPT2TraceEvent, GPT2TraceSession
        roots, sample = self.operations.roots, self.sample
        trace = roots.trace
        if (type(trace) is not GPT2TraceSession or trace.model is not roots.model
                or trace._used or trace._entering or trace._failed
                or type(trace.events) is not list or trace.events
                or trace.mode != mode or trace.capture_device != 'source'
                or trace.max_events != 2 or trace.max_trace_bytes != 64 * 1024
                or trace.run_id != f'{timing.EXPERIMENT_PROTOCOL}-session-{sample.session}'
                or trace.samples != (0,) or trace.tokens != (1,)
                or len(trace._selected_modules) != 1
                or trace._selected_modules[0][1:] != ('transformer.h.0', 'block', 0)):
            raise RuntimeError('the fixed fresh trace is required')
        self._capture('trace_events', lambda: owner._RetainedEvents(GPT2TraceEvent))
        trace.events = roots.trace_events

    def _expect(self, callback, expected):
        value = None
        try:
            value = callback()
            if value is not expected:
                if value is not None:
                    self.operations.roots.pending_return = value
                raise RuntimeError('a workload callback returned an unexpected value')
        except BaseException:
            if value is not None:
                self.operations.roots.pending_return = value
            raise

    def _measured(self, mode):
        roots, runtime, sample = self.operations.roots, self.sample.runtime, self.sample
        self._capture('current_stream_before', runtime.current_stream)
        if runtime.is_default_stream('current_stream_before') is not True:
            raise RuntimeError('the fixed default stream is required')
        started = sample.clock()
        if type(started) is not int or not 0 <= started < 2 ** 63:
            raise ValueError('lean sample start clock is invalid')
        self.partial.update(stage='running', started_ns=started)
        self._expect(runtime.record_start, None)
        self._capture('trace', lambda: sample._new_trace(mode))
        if mode != 'none':
            self._bind_trace(mode)
            self._expect(roots.trace.__enter__, roots.trace)
        self._capture('model_result', sample._forward)
        if mode != 'none':
            self._expect(lambda: roots.trace.__exit__(None, None, None), False)
        self._capture('output', lambda: roots.model_result.logits)
        if mode != 'none':
            self._capture('capture', lambda: roots.trace.events[0].value)
        metadata = sample._validate(mode)
        self.partial['metadata'] = metadata
        self._capture('current_stream_after', runtime.current_stream)
        if runtime.is_default_stream('current_stream_after') is not True:
            raise RuntimeError('the fixed default stream changed')
        self._expect(runtime.record_end, None)
        self._expect(runtime.synchronize, None)
        completion = sample.clock()
        if type(completion) is not int or not started < completion < 2 ** 63:
            raise ValueError('lean completion clock is invalid')
        self.partial.update(stage='completion_synchronized', completion_ns=completion)
        event_ms = runtime.elapsed_ms()
        if type(event_ms) not in (int, float) or not math.isfinite(event_ms) or event_ms < 0:
            raise ValueError('invalid lean device event duration')
        self.partial['device_event_ms'] = event_ms
        return metadata, started, completion, event_ms

    def _run(self, mode):
        operations, roots, sample, runtime = (self.operations, self.operations.roots,
                                               self.sample, self.sample.runtime)

        def require_empty():
            preflight = (tuple(f'preflight_result_{i}' for i in range(4))
                         + tuple(f'preflight_output_{i}' for i in range(4))
                         + tuple(f'preflight_capture_{i}' for i in range(2))
                         + tuple(f'preflight_trace_{i}' for i in range(3))
                         + tuple(f'preflight_events_{i}' for i in range(3)))
            if (operations._benchmark_session != sample.session
                    or any(getattr(roots, name) is not None for name in preflight + _TRANSIENTS + ('model_bundle',))):
                raise RuntimeError('a prior sample or preflight resource remains')

        def refs():
            return tuple(weakref.ref(getattr(roots, name)) for name in _TRANSIENTS
                         if getattr(roots, name) is not None)

        def disposed():
            if any(ref() is not None for ref in references):
                raise RuntimeError('a synchronized lean resource remains externally reachable')

        guarded_started = sample.clock()
        if type(guarded_started) is not int or not 0 <= guarded_started < 2 ** 63:
            raise ValueError('guarded sample start clock is invalid')
        self.partial['guarded_started_ns'] = guarded_started
        operations.call(require_empty)
        sample._none_call(runtime.synchronize)
        sample._none_call(runtime.reset_peaks)
        before = sample._snapshot()
        self.partial['memory_before'] = before
        # The outer call checks the owner before and after the entire workload.
        # All known model/trace/output values remain rooted until it returns.
        metadata, started, completion, event_ms = operations.call(self._measured, mode)
        wall = completion - started
        if wall < sample.minimum_wall_ns or event_ms * 1_000_000 > wall + EVENT_ENCLOSURE_TOLERANCE_NS:
            raise ValueError('lean clock or device event interval is inconsistent')
        references = operations.call(refs)
        operations.clear_after_synchronization(*_TRANSIENTS)
        operations.call(disposed)
        sample._none_call(runtime.synchronize)
        ended = operations.call(sample.clock)
        if type(ended) is not int or not completion <= ended < 2 ** 63:
            raise ValueError('lean disposal completion clock is invalid')
        self.partial.update(stage='postcheck_pending', ended_ns=ended)
        peak_a = operations.call(runtime.peak_allocated_bytes)
        if (type(peak_a) is not int
                or not before['allocated_bytes'] <= peak_a <= operations.admitted.cap_bytes - HEADROOM):
            raise ValueError('invalid lean peak allocation')
        self.partial['peak_allocated_bytes'] = peak_a
        peak_r = operations.call(runtime.peak_reserved_bytes)
        if (type(peak_r) is not int
                or not max(before['reserved_bytes'], peak_a) <= peak_r <= operations.admitted.cap_bytes - HEADROOM):
            raise ValueError('invalid lean peak reservation')
        self.partial['peak_reserved_bytes'] = peak_r
        after = sample._snapshot()
        self.partial['memory_after'] = after
        if peak_a < after['allocated_bytes'] or peak_r < after['reserved_bytes']:
            raise ValueError('lean peak counters precede the final snapshot')
        postcheck = operations.call(sample.clock)
        if type(postcheck) is not int or not ended <= postcheck <= ended + FINAL_POSTCHECK_MAX_NS:
            raise ValueError('lean postcheck timing is invalid')
        guarded_ended = sample.clock()
        if (type(guarded_ended) is not int or not postcheck <= guarded_ended < 2 ** 63
                or guarded_started > started):
            raise ValueError('guarded sample interval is invalid')
        self.partial.update(stage='complete', postcheck_completed_ns=postcheck,
                            guarded_ended_ns=guarded_ended)
        return {**metadata, 'started_ns': started, 'completion_ns': completion, 'ended_ns': ended,
                'wall_ns': wall, 'device_event_ms': event_ms, 'default_stream': True,
                'guarded_started_ns': guarded_started, 'guarded_ended_ns': guarded_ended,
                'guarded_wall_ns': guarded_ended - guarded_started,
                'memory_before': before, 'memory_after': after, 'peak_allocated_bytes': peak_a,
                'peak_reserved_bytes': peak_r, 'postcheck_completed_ns': postcheck}

    def sample_mode(self, mode):
        if self.state != 'ready' or self.sample.state != 'ready':
            raise RuntimeError('lean timing owner is not ready')
        kind = None
        try:
            self.state = 'running'
            self.sample.state = 'lean_running'
            if mode not in ('none', 'direct', 'gradient'):
                raise ValueError('invalid fixed capture mode')
            self.partial = {'mode': mode, 'stage': 'pre_sample'}
            result = self._run(mode)
            self.sample.state = 'ready'
            self.state = 'ready'
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
