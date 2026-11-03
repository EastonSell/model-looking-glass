"""Bounded cooperative-stop signals for one local experiment at a time.

The registry stores no prompts, tensors, model references, or results. A stop
request only sets an event; the caller decides when a model operation has
reached a safe, fully restored boundary. The registry lock is never the model
owner lock and must not be held while waiting for that lock.
"""
from __future__ import annotations

from dataclasses import dataclass
import hmac
import math
import re
import secrets
import threading
import time


EXPERIMENT_ACTIONS = frozenset(("sensitivity", "sensitivity-repeat", "patch-cross-prompt"))
_JOB_ID = re.compile(r"[0-9a-f]{48}\Z")


@dataclass(frozen=True, eq=False, slots=True)
class ExperimentLease:
    job_id: str
    action: str
    capture_id: str
    cancel_event: threading.Event


@dataclass(slots=True)
class _Reservation:
    lease: ExperimentLease
    deadline: float
    claimed: bool = False
    started: bool = False


class ExperimentJobs:
    """Linearizable one-job reservation with one-use claim and targeted stop."""

    def __init__(self, *, reservation_ttl=30, clock=time.monotonic):
        if (type(reservation_ttl) not in (int, float) or not math.isfinite(reservation_ttl)
                or not 0 < reservation_ttl <= 300 or not callable(clock)):
            raise ValueError("Invalid experiment reservation lifetime")
        self._ttl = reservation_ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._active = None

    @staticmethod
    def _validate_request(action, capture_id):
        if type(action) is not str or action not in EXPERIMENT_ACTIONS:
            raise ValueError("Unknown experiment action")
        if type(capture_id) is not str or not capture_id or len(capture_id) > 64:
            raise ValueError("A bounded current capture ID is required")

    def _expire_unclaimed(self):
        if self._active is not None and not self._active.claimed and self._clock() >= self._active.deadline:
            self._active = None

    def _matching(self, job_id):
        if type(job_id) is not str or _JOB_ID.fullmatch(job_id) is None:
            raise ValueError("Invalid experiment job ID")
        self._expire_unclaimed()
        if self._active is None or not hmac.compare_digest(self._active.lease.job_id, job_id):
            raise ValueError("Experiment job is not active or has expired")
        return self._active

    def reserve(self, action, capture_id):
        self._validate_request(action, capture_id)
        with self._lock:
            self._expire_unclaimed()
            if self._active is not None:
                raise ValueError("An experiment is already reserved or running")
            job_id = secrets.token_hex(24)
            lease = ExperimentLease(job_id, action, capture_id, threading.Event())
            self._active = _Reservation(lease, self._clock() + self._ttl)
            return {"job_id": job_id, "action": action, "status": "reserved",
                    "expires_in_seconds": self._ttl}

    def claim(self, job_id, action, capture_id):
        self._validate_request(action, capture_id)
        with self._lock:
            reservation = self._matching(job_id)
            lease = reservation.lease
            if reservation.claimed or lease.action != action or lease.capture_id != capture_id:
                raise ValueError("Experiment reservation was already claimed or does not match")
            reservation.claimed = True
            return lease

    def mark_started(self, lease):
        """Mark a claimed job running only after its caller owns the model lock."""
        with self._lock:
            if (self._active is None or not self._active.claimed or self._active.started
                    or self._active.lease is not lease):
                return False
            self._active.started = True
            return True

    def cancel(self, job_id):
        """Request a stop; acceptance does not mean computation has stopped."""
        with self._lock:
            reservation = self._matching(job_id)
            reservation.lease.cancel_event.set()
            return {"job_id": reservation.lease.job_id, "status": "stopping"}

    def status(self, job_id):
        with self._lock:
            reservation = self._matching(job_id)
            lease = reservation.lease
            status = ("stopping" if lease.cancel_event.is_set() else "running" if reservation.started
                      else "queued" if reservation.claimed else "reserved")
            return {"job_id": lease.job_id, "action": lease.action, "status": status}

    def release(self, lease):
        """Finish a claimed job without ever retiring a newer reservation."""
        with self._lock:
            if self._active is None or not self._active.claimed or self._active.lease is not lease:
                return False
            self._active = None
            return True
