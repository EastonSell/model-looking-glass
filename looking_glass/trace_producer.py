"""Cooperative, CPU-only custody around one trusted synchronous forward.

The profile is an operator-supplied upper-bound commitment, not an allocator
interceptor or an operating-system memory limit.  This prototype is not wired
into the broker, the trace store, or IPC.
"""

from concurrent.futures import Future
from dataclasses import dataclass
import inspect
from threading import RLock
from uuid import uuid4

from .host_budget import HostBudget, HostBusy


class ProducerCancelled(RuntimeError):
    """The producer was cancelled before its result could be returned."""


class ProducerQuarantined(RuntimeError):
    """Work may still be using claimed memory; a coordinator must recover it."""


@dataclass(frozen=True)
class ClaimLimit:
    """Declared maximum bytes and item count for one producer stage."""

    nbytes: int
    items: int

    def __post_init__(self):
        if type(self.nbytes) is not int or self.nbytes < 0:
            raise ValueError("nbytes must be a nonnegative integer")
        if type(self.items) is not int or self.items < 1:
            raise ValueError("items must be a positive integer")


@dataclass(frozen=True)
class ProducerProfile:
    """Trusted, workload-specific host commitments for one CPU forward."""

    input_copies: ClaimLimit
    output: ClaimLimit
    trace_bundle: ClaimLimit
    workspace: ClaimLimit
    device: str = "cpu"

    def __post_init__(self):
        for name in _STAGES:
            if type(getattr(self, name)) is not ClaimLimit:
                raise ValueError(f"{name} must be a ClaimLimit")
        if self.device != "cpu" or type(self.device) is not str:
            raise ValueError("only the CPU producer profile is supported")


_STAGES = ("input_copies", "output", "trace_bundle", "workspace")
_FACTORY_KEY = object()


def _deferred(value):
    return (isinstance(value, Future) or inspect.isawaitable(value)
            or inspect.isgenerator(value) or inspect.isasyncgen(value))


def _invoke_sync(forward):
    """Distinguish a Python-body exception from an ambiguous call boundary."""
    try:
        return forward(), None
    except BaseException as error:
        # An interruption can arrive after the callable returned but before
        # its return value was assigned, concealing possible deferred work.
        code = getattr(forward, "__code__", None)
        traceback = error.__traceback__
        while traceback is not None and traceback.tb_frame.f_code is not code:
            traceback = traceback.tb_next
        if code is None or traceback is None:
            raise
        traceback = None
        # Claims remain held even if clearing the original callback traceback
        # releases references from the callback's local frame.
        error.__traceback__ = None
        error.__cause__ = None
        error.__context__ = None
        return None, error


class ProducerEnvelope:
    """Pin four pre-admitted claims throughout one trusted synchronous call.

    Claims remain charged after success, failure, and cancellation.  A trusted
    owner must attest disposal stage by stage; this class cannot prove disposal.
    """

    def __init__(self, key, budget, request, profile, claims):
        if key is not _FACTORY_KEY:
            raise TypeError("use ProducerEnvelope.acquire")
        self._budget = budget
        self._request = request
        self._profile = profile
        self._claims = claims
        self._lock = RLock()
        self._state = "reserved"
        self._cancelled = False
        self._operation = None
        self._deferred = None

    @property
    def profile(self):
        """The immutable commitments used for admission."""
        return self._profile

    @classmethod
    def acquire(cls, budget, request_id, profile):
        """Reserve all commitments before any execution or captured allocation."""
        if cls is not ProducerEnvelope:
            raise TypeError("ProducerEnvelope cannot be subclassed for admission")
        if type(budget) is not HostBudget or type(profile) is not ProducerProfile:
            raise ValueError("a HostBudget and ProducerProfile are required")
        if type(request_id) is not str or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a nonempty string of at most 128 characters")
        guard = budget.request_guard(request_id)
        owner = f"producer-{uuid4().hex}"
        claims = {}
        try:
            guard.open()
            request = guard.adopt()
            for name in _STAGES:
                limit = getattr(profile, name)
                claims[name] = request.reserve(owner, name, limit.nbytes,
                                               items=limit.items)
            return cls(_FACTORY_KEY, budget, request, profile, claims)
        except BaseException:
            # No forward has run, so claims created by this acquisition have
            # no associated payloads.  Scan the ledger as well as local
            # handles, because interruption can occur after reserve returns
            # but before its result is assigned.  Match the exact generation,
            # acquisition-unique owner and untransferred stage so other
            # owners' claims remain charged.
            try:
                request = guard.adopt()
                generation = guard.generation
                request.cancel()
                for claim in budget.active_claims():
                    info = claim.info()
                    if (info.request_id == request_id
                            and info.request_generation == generation
                            and info.owners == (owner,) and info.stage in _STAGES):
                        claim.release()
                guard.cancel_and_close_empty()
            except BaseException:
                # Conservatively leave anything not proved safe to a
                # trusted ledger recovery coordinator. Registered guards and
                # outstanding handles remain discoverable in the ledger.
                pass
            raise

    def cancel(self):
        """Tombstone the request; claims and in-flight work remain charged."""
        with self._lock:
            already_cancelled = self._cancelled
            self._request.cancel()
            self._cancelled = True
            return not already_cancelled

    def run(self, forward):
        """Run a trusted synchronous callable once, returning only a live result.

        The caller must separately synchronize publication with late
        cancellation. Deferred results are quarantined with the operation
        still pinned because their completion cannot be established here.
        """
        if not callable(forward):
            raise ValueError("forward must be callable")
        if (inspect.iscoroutinefunction(forward) or inspect.isgeneratorfunction(forward)
                or inspect.isasyncgenfunction(forward)):
            raise ValueError("forward must be synchronous")
        owned = False
        operation = None
        result = None
        failure = None
        cancelled = False
        try:
            with self._lock:
                if self._cancelled:
                    raise ProducerCancelled("producer request is cancelled")
                if self._state != "reserved":
                    raise RuntimeError("producer envelope has already been used")
                if tuple(self._claims) != _STAGES:
                    raise RuntimeError("producer commitments were already disposed")
                owned = True
                self._state = "starting"
                operation = self._request.begin_operation(
                    "producer-forward", tuple(self._claims.values()))
                self._operation = operation
                self._state = "running"

            result, failure = _invoke_sync(forward)
            forward = None

            with self._lock:
                if failure is None and _deferred(result):
                    self._deferred = result
                    result = None
                    self._state = "quarantined"
                    self._cancelled = True
                    self._request.cancel()
                    raise ProducerQuarantined("forward returned deferred work")
                self._state = "finishing"
                live = operation.finish()
                self._operation = None
                cancelled = self._cancelled or not live
                if cancelled:
                    result = None
                    failure = None
                    self._state = "cancelled"
                elif failure is not None:
                    self._state = "failed"
                else:
                    self._state = "completed"
            if cancelled:
                raise ProducerCancelled("producer request was cancelled")
            if failure is not None:
                raise failure from None
            return result
        except BaseException:
            if owned:
                with self._lock:
                    if self._state in ("reserved", "starting", "running", "finishing",
                                       "quarantined"):
                        # An interrupted start/finish might have registered an
                        # operation even when its handle was not assigned.
                        # Never infer that such an operation has finished.
                        result = None
                        forward = None
                        self._state = "quarantined"
                        self._cancelled = True
                        try:
                            self._request.cancel()
                        except BaseException:
                            pass
            raise

    def confirm_disposed(self, stage):
        """Attest that all aliases covered by one claim have been disposed.

        Transfers to another owner require a separate overlapping claim first;
        this is not a transfer to another HostBudget request.  An operation pin
        causes HostBusy instead of releasing capacity prematurely.
        """
        if type(stage) is not str or stage not in _STAGES:
            raise ValueError("unknown producer stage")
        with self._lock:
            if self._state == "quarantined":
                raise HostBusy("quarantined producer requires external recovery")
            claim = self._claims.get(stage)
            if claim is None:
                return False
            claim.release()
            del self._claims[stage]
            return True

    def close(self):
        """Close only after all claims and operations have been resolved."""
        with self._lock:
            if self._state == "quarantined":
                raise HostBusy("quarantined producer requires external recovery")
            return self._request.close()
