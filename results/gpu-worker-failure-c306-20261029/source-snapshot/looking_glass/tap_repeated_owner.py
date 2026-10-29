"""Disabled, single-owner custody seam for the repeated tap study.

There is no runnable CUDA owner here. The operation seam is exercised only
with fake owner/device objects in CPU tests; it stays disabled until its full
runner, live telemetry, supervisor and failure paths are independently reviewed.
"""

import gc

from .device_identity import DeviceIdentity
from .device_lease import DeviceLease
from .supervisor import OwnerSession
from .tap_repeated_admission import _Admission
from .tap_repeated_protocol import EXPERIMENT_PROTOCOL


_OPERATIONS_ENABLED = False
_SLOTS = (('model_bundle', 'model', 'input_ids', 'default_stream', 'current_stream_before', 'current_stream_after',
           'start_event', 'end_event', 'model_result',
           'output', 'trace', 'trace_events', 'capture', 'converted')
          + tuple(f'preflight_result_{index}' for index in range(4))
          + tuple(f'preflight_output_{index}' for index in range(4))
          + tuple(f'preflight_trace_{index}' for index in range(3))
          + tuple(f'preflight_events_{index}' for index in range(3))
          + tuple(f'preflight_capture_{index}' for index in range(2)))


def _failure_kind(error):
    if isinstance(error, KeyboardInterrupt):
        return 'interrupt'
    if isinstance(error, SystemExit):
        return 'exit'
    if isinstance(error, GeneratorExit):
        return 'generator_exit'
    if isinstance(error, MemoryError):
        return 'memory'
    return 'failure'


def _raise_detached(kind):
    # Called outside the exception handler: the public exception has no
    # implicit context pointing at the privately retained failure graph.
    if kind == 'interrupt':
        raise KeyboardInterrupt()
    if kind == 'exit':
        raise SystemExit(1)
    if kind == 'generator_exit':
        raise GeneratorExit()
    if kind == 'memory':
        raise MemoryError('owner operation failed')
    raise RuntimeError('owner operation failed')


class _RetainedEvents(list):
    """Owner-private event storage surviving a failed trace's clear().

    The trusted trace implementation only calls append and clear. This is not
    a public trace result: a failed session remains unsuccessful. Dispose of
    the complete container only after a verified synchronization.
    """

    def __init__(self, event_type):
        super().__init__()
        self._event_type = event_type

    def append(self, event):
        if type(event) is not self._event_type or len(self) >= 2:
            raise RuntimeError('retained trace event inventory is invalid')
        super().append(event)

    def clear(self):
        # GPT2TraceSession intentionally clears failed sessions. Hold the
        # privately rooted event values until owner exit or safe disposal.
        return None


class _Roots:
    """Preallocate names for independently retained device owners."""

    def __init__(self):
        for name in _SLOTS:
            setattr(self, name, None)
        self.pending_return = None
        self.failure = None
        self.cleanup_failure = None


class _OwnerOperations:
    """Checks the exact supervised binding around one synchronous operation.

    The trusted single-threaded owner must not catch an operation failure and
    continue. It must terminate under its supervisor; a Python interruption
    before a returned object or exception is assigned cannot always be rooted.
    """

    def __init__(self, session, device_lease, admitted):
        if _OPERATIONS_ENABLED is not True:
            raise RuntimeError('repeated tap CUDA owner operations are disabled')
        if (type(session) is not OwnerSession or type(device_lease) is not DeviceLease
                or type(admitted) is not _Admission):
            raise ValueError('exact supervised owner, device lease and admission are required')
        self.session, self.device_lease, self.admitted = session, device_lease, admitted
        self.roots = _Roots()
        self.unsafe = False
        self._gc_disabled_required = False
        self._benchmark_session = None
        OwnerSession.check(session, device_lease)
        identity = session.identity
        if (type(identity) is not DeviceIdentity or session._runtime_verified is not True
                or session._bound_lease is not device_lease):
            raise RuntimeError('physical runtime identity is not bound')
        self.expected_uuid = identity.gpu_uuid
        self.expected_workspace = identity.workspace_root
        self._check()

    def _check_identity(self):
        identity = self.session.identity
        if (type(identity) is not DeviceIdentity or identity.gpu_uuid != self.expected_uuid
                or identity.workspace_root != self.expected_workspace
                or self.session._runtime_verified is not True
                or self.session._bound_lease is not self.device_lease):
            raise RuntimeError('supervised physical binding changed')

    def _check(self):
        if self._gc_disabled_required and gc.isenabled():
            raise RuntimeError('automatic cyclic collection was reenabled')
        OwnerSession.check(self.session, self.device_lease)
        self._check_identity()
        self.admitted.verify()
        OwnerSession.check(self.session, self.device_lease)
        self._check_identity()

    def require_disabled_gc(self):
        """Latch the single-owner no-automatic-collection policy."""
        def enable():
            if gc.isenabled():
                raise RuntimeError('automatic cyclic collection must be disabled')
            self._gc_disabled_required = True
        self.call(enable)

    def bind_session(self, session):
        """Bind the owner's immutable schedule slot before any trace."""
        def bind():
            if type(session) is not int or session not in range(5):
                raise ValueError('invalid repeated-tap session')
            if self._benchmark_session is not None and self._benchmark_session != session:
                raise RuntimeError('the schedule slot is already bound')
            self._benchmark_session = session
        self.call(bind)

    def call(self, operation, *args, root=None):
        """Call and optionally root an assigned result before the postcheck."""
        if self.unsafe:
            raise RuntimeError('owner completion is unverified')
        result = None
        kind = None
        try:
            self._check()
            if not callable(operation) or (root is not None and (type(root) is not str or root not in _SLOTS)):
                raise ValueError('invalid operation or root slot')
            if root is not None and getattr(self.roots, root) is not None:
                raise RuntimeError('the requested owner root is already occupied')
            result = operation(*args)
            if root is not None:
                setattr(self.roots, root, result)
            self._check()
            return result
        except BaseException as exc:
            # An interruption can strike after Python assigned a returned
            # object but before its designated slot. Retain that known value
            # independently of the exception traceback before marking failure.
            # A further interruption before this assignment remains uncertain.
            if result is not None or self.roots.pending_return is None:
                self.roots.pending_return = result
            self.unsafe = True
            if self.roots.failure is None:
                self.roots.failure = exc
            kind = _failure_kind(exc)
        _raise_detached(kind)

    def bind_trace_events(self, trace_slot, *, session):
        """Pre-root bounded event storage before entering a fresh trace."""
        if type(session) is not int or session not in range(5):
            raise ValueError('invalid repeated-tap session')
        if trace_slot == 'trace':
            event_slot = 'trace_events'
            expected_modes = ('direct', 'gradient')
        elif type(trace_slot) is str and trace_slot in tuple(f'preflight_trace_{i}' for i in range(3)):
            event_slot = trace_slot.replace('preflight_trace_', 'preflight_events_')
            expected_modes = (('id', 'direct', 'gradient')[int(trace_slot[-1])],)
        else:
            raise ValueError('unsupported trace root')

        def fresh_trace(trace_type):
            trace = getattr(self.roots, trace_slot)
            if (self._benchmark_session != session or type(trace) is not trace_type
                    or trace.model is not self.roots.model
                    or trace._used or trace._entering or trace._failed
                    or type(trace.events) is not list or trace.events
                    or trace.max_events != 2 or trace.max_trace_bytes != 64 * 1024
                    or trace.capture_device != 'source' or trace.mode not in expected_modes
                    or trace.run_id != f'{EXPERIMENT_PROTOCOL}-session-{session}'
                    or trace.samples != (0,) or trace.tokens != (1,)
                    or len(trace._selected_modules) != 1
                    or trace._selected_modules[0][1:] != ('transformer.h.0', 'block', 0)
                    or (trace_slot == 'trace' and any(
                        getattr(self.roots, f'preflight_{kind}_{index}') is not None
                        for kind in ('trace', 'events') for index in range(3)))
                    or (trace_slot == 'trace' and any(
                        getattr(self.roots, f'preflight_{kind}_{index}') is not None
                        for kind, count in (('result', 4), ('output', 4), ('capture', 2)) for index in range(count)))
                    or (trace_slot == 'trace' and any(getattr(self.roots, name) is not None
                                                       for name in ('model_bundle', 'model_result', 'output', 'capture')))
                    or (trace_slot != 'trace' and (self.roots.trace is not None
                                                   or self.roots.trace_events is not None
                                                   or self.roots.capture is not None))):
                raise RuntimeError('a fresh bounded owner trace is required')
            return trace

        def allocate():
            # Importing this module imports Torch. It happens only inside an
            # owner-checked operation, not when the controller imports us.
            from .transformer_trace import GPT2TraceEvent, GPT2TraceSession
            fresh_trace(GPT2TraceSession)
            return _RetainedEvents(GPT2TraceEvent)

        retained = self.call(allocate, root=event_slot)

        def install():
            from .transformer_trace import GPT2TraceSession
            trace = fresh_trace(GPT2TraceSession)
            if getattr(self.roots, event_slot) is not retained:
                raise RuntimeError('retained trace root changed')
            trace.events = retained

        self.call(install)

    def clear_after_synchronization(self, *slots):
        """Clear named roots only after the caller verified completion.

        This does not itself synchronize or prove the caller's disposal
        precondition; the yet-unimplemented owner must do so and then perform a
        second synchronization before releasing the admission.
        """
        if self.unsafe:
            raise RuntimeError('owner completion is unverified')
        kind = None
        try:
            self._check()
            if any(type(name) is not str or name not in _SLOTS for name in slots):
                raise ValueError('invalid owner root slot')
            for name in slots:
                self._check()
                setattr(self.roots, name, None)
                self._check()
        except BaseException as exc:
            self.unsafe = True
            self.roots.failure = exc
            kind = _failure_kind(exc)
        if kind is not None:
            _raise_detached(kind)
