"""Disabled CUDA-call adapter for the prospective repeated tap owner.

This module does not import Torch or initialize CUDA on import. It is a
private trusted seam, not a supervised owner or permission to launch one.
The caller must wrap every method in the exact owner operations guard and
must terminate after any escaping error or ambiguous return.
"""

import threading
import weakref

from . import tap_repeated_owner as owner


_CUDA_BACKEND_ENABLED = False


class _CudaRuntime:
    """One fixed CUDA device and pair of timing events, initially disabled."""

    device, vocab_size, hidden_size = 'cuda:0', 50257, 768

    def __init__(self, operations):
        if _CUDA_BACKEND_ENABLED is not True:
            raise RuntimeError('repeated tap CUDA backend remains disabled')
        if (type(operations) is not owner._OwnerOperations or operations.unsafe
                or type(operations._benchmark_session) is not int or operations._benchmark_session not in range(5)
                or operations._gc_disabled_required is not True):
            raise ValueError('the exact bound owner is required')
        self.operations = operations
        self._roots, self._session, self._device_lease = operations.roots, operations.session, operations.device_lease
        self._admitted, self._slot = operations.admitted, operations._benchmark_session
        self._thread = threading.get_ident()
        self.state = 'new'
        self.failure = None
        self.pending_return = None
        self.default_stream_id = None
        self._stream_ref = self._start_ref = self._end_ref = None

    def _none_call(self, callback):
        def invoke():
            result = None
            try:
                result = callback()
                self.pending_return = result
                if result is not None:
                    raise RuntimeError('CUDA initialization returned an unexpected object')
            except BaseException:
                if result is not None:
                    self.pending_return = result
                raise
        self.operations.call(invoke)

    def _binding(self):
        ops = self.operations
        if (ops.roots is not self._roots or ops.session is not self._session
                or ops.device_lease is not self._device_lease or ops.admitted is not self._admitted
                or ops._benchmark_session != self._slot or ops._gc_disabled_required is not True
                or ops.unsafe):
            raise RuntimeError('CUDA runtime owner binding changed')

    def _ready(self, *, events=False):
        if self.state != 'ready' or threading.get_ident() != self._thread:
            raise RuntimeError('CUDA runtime is not ready on its owner thread')
        self._binding()
        if events:
            self._event_binding()

    def _event_binding(self):
        self._binding()
        if (self._stream_ref is None or self._stream_ref() is None
                or self._roots.default_stream is not self._stream_ref()
                or self._start_ref is None or self._end_ref is None
                or self._start_ref() is None or self._end_ref() is None
                or self._roots.start_event is not self._start_ref()
                or self._roots.end_event is not self._end_ref()):
            raise RuntimeError('CUDA timing event custody changed')

    def initialize(self):
        """Set/read the fixed cap and root weak-referenceable CUDA events."""
        if self.state != 'new' or threading.get_ident() != self._thread:
            raise RuntimeError('CUDA initialization is single-use on the owner thread')
        kind = None
        try:
            self.state = 'initializing'
            self._binding()

            def set_cap():
                import torch
                return torch.cuda.set_per_process_memory_fraction(0.25, device=0)

            def check_cap():
                import torch
                value = torch.cuda.get_per_process_memory_fraction(device=0)
                if type(value) is not float or value != 0.25:
                    raise RuntimeError('CUDA allocator cap readback differs')

            def default_stream():
                import torch
                return torch.cuda.default_stream(device=0)

            def bind_default():
                stream = self._roots.default_stream
                value, device_index = stream.cuda_stream, stream.device.index
                if type(value) is not int or value < 0 or type(device_index) is not int or device_index != 0:
                    raise RuntimeError('CUDA default stream identity is invalid')
                self.default_stream_id = value
                self._stream_ref = weakref.ref(stream)

            def event():
                import torch
                return torch.cuda.Event(enable_timing=True)

            def check_events():
                roots = self.operations.roots
                if roots.start_event is None or roots.end_event is None or roots.start_event is roots.end_event:
                    raise RuntimeError('CUDA timing events are invalid')
                self._start_ref = weakref.ref(roots.start_event)
                self._end_ref = weakref.ref(roots.end_event)

            def prime_start():
                self._event_binding()
                return self._roots.start_event.record(self._roots.default_stream)

            def prime_end():
                self._event_binding()
                return self._roots.end_event.record(self._roots.default_stream)

            def synchronize():
                self._event_binding()
                import torch
                return torch.cuda.synchronize(device=0)

            self._none_call(set_cap)
            self.operations.call(check_cap)
            self.operations.call(default_stream, root='default_stream')
            self.operations.call(bind_default)
            self.operations.call(event, root='start_event')
            self.operations.call(event, root='end_event')
            self.operations.call(check_events)
            # CUDA events are lazily created on first record. Prime both on
            # the checked default stream and synchronize before preflight or
            # the first timed warmup, as in the fixed reference path.
            self._none_call(prime_start)
            self._none_call(prime_end)
            self._none_call(synchronize)
            self.state = 'ready'
            return None
        except BaseException as exc:
            self.failure = exc
            self.operations.unsafe = True
            if self.operations.roots.failure is None:
                self.operations.roots.failure = exc
            self.state = 'failed'
            kind = owner._failure_kind(exc)
        owner._raise_detached(kind)

    def current_stream(self):
        self._ready(events=True)
        import torch
        # The timing owner calls this through operations.call with a dedicated
        # root. Do not nest another owner call inside the already guarded call.
        return torch.cuda.current_stream(device=0)

    def is_default_stream(self, slot):
        self._ready(events=True)
        if slot not in ('current_stream_before', 'current_stream_after'):
            raise ValueError('invalid CUDA current stream slot')
        stream = getattr(self._roots, slot)
        if stream is None:
            raise RuntimeError('CUDA current stream is not rooted')
        return (type(stream.cuda_stream) is int and stream.cuda_stream == self.default_stream_id
                and type(stream.device.index) is int and stream.device.index == 0)

    def record_start(self):
        self._ready(events=True)
        return self.operations.roots.start_event.record(self._roots.default_stream)

    def record_end(self):
        self._ready(events=True)
        return self.operations.roots.end_event.record(self._roots.default_stream)

    def synchronize(self):
        self._ready()
        import torch
        return torch.cuda.synchronize(device=0)

    def elapsed_ms(self):
        self._ready(events=True)
        return self.operations.roots.start_event.elapsed_time(self.operations.roots.end_event)

    def reset_peaks(self):
        self._ready()
        import torch
        return torch.cuda.reset_peak_memory_stats(device=0)

    def free_total_bytes(self):
        self._ready()
        import torch
        return torch.cuda.mem_get_info(device=0)

    def allocated_bytes(self):
        self._ready()
        import torch
        return torch.cuda.memory_allocated(device=0)

    def reserved_bytes(self):
        self._ready()
        import torch
        return torch.cuda.memory_reserved(device=0)

    def peak_allocated_bytes(self):
        self._ready()
        import torch
        return torch.cuda.max_memory_allocated(device=0)

    def peak_reserved_bytes(self):
        self._ready()
        import torch
        return torch.cuda.max_memory_reserved(device=0)
