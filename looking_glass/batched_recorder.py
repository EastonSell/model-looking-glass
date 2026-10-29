"""Bounded, loss-detecting storage for already observed batch events.

This recorder does not instrument a broker, read a clock or verify CUDA work.
Producers supply source timestamps and must stop before sealing the buffer.
"""
from copy import deepcopy
import threading

from .batched_events import _EVENT_FIELDS


_MAX_EVENTS = 4096
_MAX_WIRE_BYTES = 1024 * 1024
_EMPTY_ENVELOPE_BYTES = len('{"events":[]}')


class _InvalidEvent(ValueError):
    pass


class _CapacityExceeded(ValueError):
    pass


def _charge(budget, amount):
    budget[0] -= amount
    if budget[0] < 0:
        raise _CapacityExceeded('event exceeds the remaining buffer budget')


def _copy_value(value, budget, depth=0):
    """Copy exact JSON primitives, charging a conservative JSON wire bound."""
    if depth > 3:
        raise _InvalidEvent('event nesting exceeds the supported schema')
    if value is None:
        _charge(budget, 4)
        return None
    if type(value) is bool:
        _charge(budget, 5)
        return value
    if type(value) is int:
        if not 0 <= value < 2 ** 63:
            raise _InvalidEvent('event integers must be bounded and nonnegative')
        _charge(budget, 19)
        return value
    if type(value) is str:
        if len(value) > 512 or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise _InvalidEvent('event string is too long or not valid Unicode')
        # A code point takes at most 12 JSON bytes even when escaped as a
        # surrogate pair; the adapter uses the smaller UTF-8 representation.
        _charge(budget, 2 + 12 * len(value))
        return value
    if type(value) is list:
        if len(value) > 256:
            raise _InvalidEvent('event list is too long')
        _charge(budget, 2 + max(0, len(value) - 1))
        return [_copy_value(item, budget, depth + 1) for item in value]
    if type(value) is dict:
        if len(value) > 16:
            raise _InvalidEvent('event object has too many fields')
        _charge(budget, 2 + max(0, len(value) - 1) + len(value))
        result = {}
        for key, item in value.items():
            if type(key) is not str or len(key) > 64:
                raise _InvalidEvent('event object key is invalid')
            copied_key = _copy_value(key, budget, depth + 1)
            result[copied_key] = _copy_value(item, budget, depth + 1)
        return result
    raise _InvalidEvent('event must contain only ordinary JSON primitives')


def _copy_event(event, available_bytes):
    if type(event) is not dict:
        raise _InvalidEvent('event must be an ordinary dictionary')
    # Inspect key types before string lookup: a user-defined key with a hash
    # collision could otherwise execute arbitrary __eq__ inside our lock.
    if any(type(key) is not str for key in event):
        raise _InvalidEvent('event keys must be ordinary strings')
    kind = event.get('kind')
    if type(kind) is not str or kind not in _EVENT_FIELDS:
        raise _InvalidEvent('event kind is unknown')
    fields = _EVENT_FIELDS[kind]
    if len(event) != len(fields) or any(name not in event for name in fields):
        raise _InvalidEvent('event fields differ from the adapter schema')
    timestamp = event['time_ns']
    if type(timestamp) is not int or not 0 <= timestamp < 2 ** 63:
        raise _InvalidEvent('event timestamp must be a bounded integer')
    budget = [available_bytes]
    result = _copy_value(event, budget)
    return result, available_bytes - budget[0]


class BatchedEventRecorder:
    """Capture bounded copies of trusted observations without device access.

    ``record`` accepts the exact event dictionaries documented in
    BATCHED_EVENTS_V1. Timestamps must be read at the actual source before this
    call; they are retained verbatim, not sampled inside this class. Ordinary
    recording failures return False and set a sticky incomplete flag. The
    caller must keep input dictionaries exclusively owned during each call.
    """

    def __init__(self, *, max_events=_MAX_EVENTS, max_wire_bytes=_MAX_WIRE_BYTES):
        if type(max_events) is not int or not 1 <= max_events <= _MAX_EVENTS:
            raise ValueError('max_events must be between 1 and 4096')
        if (type(max_wire_bytes) is not int
                or not _EMPTY_ENVELOPE_BYTES <= max_wire_bytes <= _MAX_WIRE_BYTES):
            raise ValueError('max_wire_bytes must cover the envelope and be at most one MiB')
        self._max_events = max_events
        self._max_wire_bytes = max_wire_bytes
        self._wire_bound = _EMPTY_ENVELOPE_BYTES
        self._events = []
        self._complete = True
        self._sealed = False
        self._lost_capacity = self._lost_invalid = self._lost_internal = 0
        self._late_records = self._explicit_faults = 0
        self._lock = threading.Lock()

    def record(self, event):
        """Store one source observation, returning False if it could not be stored.

        No callback or file I/O occurs while holding the recorder lock. It can
        block on another producer and its overhead belongs in the measured
        window. Control-flow interruptions are re-raised after marking the
        record incomplete when the fault latch itself remains available.
        """
        try:
            return self._record_once(event)
        except BaseException as error:
            # Include failures during lock acquisition and the bookkeeping
            # preceding the protected copy. The driver's independent failure
            # latch/watchdog is still needed if this latch cannot run.
            with self._lock:
                self._complete = False
                self._lost_internal += 1
            if isinstance(error, Exception):
                return False
            raise

    def _record_once(self, event):
        with self._lock:
            if self._sealed:
                self._complete = False
                self._late_records += 1
                return False
            was_complete = self._complete
            # An interruption at any point in the copy must never leave a
            # partially observed event marked complete.
            self._complete = False
            try:
                if len(self._events) >= self._max_events:
                    raise _CapacityExceeded('event count limit reached')
                # One extra byte conservatively covers an array separator.
                available = self._max_wire_bytes - self._wire_bound - 1
                snapshot, bound = _copy_event(event, available)
                self._events.append(snapshot)
                self._wire_bound += bound + 1
                self._complete = was_complete
                return True
            except _CapacityExceeded:
                self._complete = False
                self._lost_capacity += 1
                return False
            except _InvalidEvent:
                self._complete = False
                self._lost_invalid += 1
                return False

    def mark_incomplete(self):
        """Record a hook-side fault that prevented an event from reaching us."""
        with self._lock:
            self._complete = False
            self._explicit_faults += 1

    def seal(self, *, producers_stopped):
        """Seal only after the caller has stopped and joined all producers.

        The exact True argument is a caller attestation, not proof of thread or
        device quiescence. A subsequent event is dropped and marks the current
        recorder incomplete; a previously copied report cannot be revoked.
        """
        if producers_stopped is not True:
            raise ValueError('all event producers must be stopped before sealing')
        with self._lock:
            self._sealed = True

    def report(self):
        """Return a detached report after producers have stopped and sealing."""
        with self._lock:
            if not self._sealed:
                raise RuntimeError('seal the recorder before reporting')
            return {'events_complete': self._complete, 'recorded': len(self._events),
                    'lost_capacity': self._lost_capacity, 'lost_invalid': self._lost_invalid,
                    'lost_internal': self._lost_internal, 'late_records': self._late_records,
                    'explicit_faults': self._explicit_faults,
                    'wire_bytes_upper_bound': self._wire_bound,
                    'max_wire_bytes': self._max_wire_bytes, 'max_events': self._max_events,
                    'events': deepcopy(self._events)}
