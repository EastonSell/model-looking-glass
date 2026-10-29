"""Validate one trusted broker job's CPU trace selections.

This module constructs plans only. It is not a broker worker, payload transport,
host-memory reservation, or authorization boundary. No trace execution or
result delivery is enabled until those ownership protocols are integrated.
"""
from dataclasses import dataclass

from .capture_dispatcher import CaptureRequest, SampleSelection

_MAX_INT = 2**63 - 1
_MAX_TEXT = 128
_PLAN_FACTORY = object()


@dataclass(frozen=True, slots=True)
class BrokerJobIdentity:
    broker_id: str
    job_id: int
    team_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class SampleTrace:
    sample_id: str
    token_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SiteTrace:
    layer_index: int
    module_kind: str


@dataclass(frozen=True, slots=True)
class TraceSpec:
    run_id: str
    samples: tuple[SampleTrace, ...]
    sites: tuple[SiteTrace, ...]
    forward_index: int
    sequence_start: int
    generation_step: int | None
    max_events: int
    max_trace_bytes: int


@dataclass(frozen=True, slots=True)
class TraceLimits:
    max_batch_size: int
    max_sequence_length: int
    max_layers: int
    max_sites: int
    max_events: int
    max_trace_bytes: int
    max_token_positions: int


def _integer(name, value, *, minimum=0):
    if type(value) is not int or not minimum <= value <= _MAX_INT:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_MAX_INT}]")


def _text(name, value):
    if type(value) is not str or not value.strip() or len(value) > _MAX_TEXT:
        raise ValueError(f"{name} must be a nonempty plain string of at most {_MAX_TEXT} characters")


@dataclass(frozen=True, slots=True, init=False)
class BoundTracePlan:
    """Private copies of one owner's validated trace selection metadata.

    There is intentionally no ``execute`` or broker worker interface here.
    Constructed by :func:`bind_trace_plan`, with caller-visible values
    reconstructed on access. Arbitrary same-process code is not sandboxed.
    """
    _identity: tuple
    _run_id: str
    _samples: tuple
    _sites: tuple
    _forward_index: int
    _sequence_start: int
    _generation_step: int | None
    _max_events: int
    _max_trace_bytes: int

    def __init__(self, identity, run_id, samples, sites, forward_index,
                 sequence_start, generation_step, max_events, max_trace_bytes,
                 *, _factory=None):
        if _factory is not _PLAN_FACTORY:
            raise TypeError("use bind_trace_plan to construct a plan")
        for name, value in (("_identity", identity), ("_run_id", run_id),
                            ("_samples", samples), ("_sites", sites),
                            ("_forward_index", forward_index),
                            ("_sequence_start", sequence_start),
                            ("_generation_step", generation_step),
                            ("_max_events", max_events),
                            ("_max_trace_bytes", max_trace_bytes)):
            object.__setattr__(self, name, value)

    @property
    def identity(self):
        return BrokerJobIdentity(*self._identity)

    @property
    def worker_request_id(self):
        broker_id, job_id, _, _ = self._identity
        return f"lg-{broker_id}-{job_id}"

    @property
    def sites(self):
        return tuple(SiteTrace(layer, kind) for layer, kind in self._sites)

    def capture_request(self):
        _, _, _, request_id = self._identity
        samples = tuple(SampleSelection(row, sample_id, positions)
                        for row, (sample_id, positions) in enumerate(self._samples))
        return CaptureRequest(self._run_id, self.worker_request_id, request_id, samples,
                              self._forward_index, self._generation_step,
                              self._max_events, self._max_trace_bytes)

    def sequence_starts(self):
        return {self._run_id: self._sequence_start}


def bind_trace_plan(identity, spec, limits, *, batch_size, sequence_length,
                    layer_count, device="cpu", transport="in_process"):
    """Bind one job's selection to all batch rows using trusted owner facts.

    Identity, actual input shape, model layer count, and limits must come from
    the trusted broker/worker owner. This function does not authenticate them.
    Caller objects must not be changed while they are being copied.
    """
    if type(device) is not str or device != "cpu":
        raise ValueError("only a CPU device declaration is accepted")
    if type(transport) is not str or transport != "in_process":
        raise ValueError("only an in-process transport declaration is accepted")
    if type(identity) is not BrokerJobIdentity or type(spec) is not TraceSpec \
            or type(limits) is not TraceLimits:
        raise ValueError("identity, spec and limits must be exact plan types")

    # Snapshot before validation; no caller-owned dataclass remains in the plan.
    identity = (identity.broker_id, identity.job_id, identity.team_id, identity.request_id)
    policy = (limits.max_batch_size, limits.max_sequence_length, limits.max_layers,
              limits.max_sites, limits.max_events, limits.max_trace_bytes,
              limits.max_token_positions)
    run_id, samples, sites = spec.run_id, spec.samples, spec.sites
    forward_index, sequence_start = spec.forward_index, spec.sequence_start
    generation_step, max_events, max_trace_bytes = spec.generation_step, spec.max_events, spec.max_trace_bytes
    if type(samples) is not tuple or type(sites) is not tuple:
        raise ValueError("sample and site selections must be tuples")
    broker_id, job_id, team_id, request_id = identity
    if type(broker_id) is not str or len(broker_id) != 32 \
            or any(char not in "0123456789abcdef" for char in broker_id):
        raise ValueError("broker_id must be a lowercase 32-character hex identifier")
    _integer("job_id", job_id, minimum=1)
    _text("team_id", team_id)
    _text("request_id", request_id)
    _text("run_id", run_id)
    for name, value in zip(("max_batch_size", "max_sequence_length", "max_layers",
                            "max_sites", "max_events", "max_trace_bytes",
                            "max_token_positions"), policy):
        _integer(name, value, minimum=1)
    max_batch, max_length, max_layers, max_sites, limit_events, limit_bytes, max_tokens = policy
    if max_tokens > 128:
        raise ValueError("token-position limit cannot exceed the transport schema bound")
    for name, value in (("batch_size", batch_size), ("sequence_length", sequence_length),
                        ("layer_count", layer_count)):
        _integer(name, value, minimum=1)
    if batch_size > max_batch or sequence_length > max_length or layer_count > max_layers:
        raise ValueError("input or model dimensions exceed configured bounds")
    if len(samples) != batch_size or not sites or len(sites) > max_sites:
        raise ValueError("every row requires a sample and sites must be bounded")
    _integer("forward_index", forward_index)
    _integer("sequence_start", sequence_start)
    if generation_step is not None:
        _integer("generation_step", generation_step)
    _integer("max_events", max_events, minimum=1)
    _integer("max_trace_bytes", max_trace_bytes, minimum=1)
    if (max_events > limit_events or max_trace_bytes > limit_bytes
            or len(samples) * len(sites) > max_events
            or sequence_start + max_events - 1 > _MAX_INT):
        raise ValueError("trace selection exceeds its declared bounds")
    if any(type(sample) is not SampleTrace for sample in samples) \
            or any(type(site) is not SiteTrace for site in sites):
        raise ValueError("sample and site selections must use exact plan types")
    samples = tuple((sample.sample_id, sample.token_positions) for sample in samples)
    sites = tuple((site.layer_index, site.module_kind) for site in sites)
    seen_samples = set()
    for sample_id, positions in samples:
        _text("sample_id", sample_id)
        if sample_id in seen_samples:
            raise ValueError("sample IDs must be distinct within the job")
        seen_samples.add(sample_id)
        if type(positions) is not tuple or not positions or len(positions) > max_tokens:
            raise ValueError("token positions must be a bounded nonempty tuple")
        seen_positions = set()
        for position in positions:
            _integer("token position", position)
            if position >= sequence_length or position in seen_positions:
                raise ValueError("token positions must be distinct and in range")
            seen_positions.add(position)
    seen_sites = set()
    for layer, kind in sites:
        _integer("layer_index", layer)
        if layer >= layer_count or type(kind) is not str or kind not in ("block", "attn", "mlp"):
            raise ValueError("trace site is outside the supported model")
        if (layer, kind) in seen_sites:
            raise ValueError("trace sites must be distinct")
        seen_sites.add((layer, kind))
    return BoundTracePlan(identity, run_id, samples, sites, forward_index,
                          sequence_start, generation_step, max_events, max_trace_bytes,
                          _factory=_PLAN_FACTORY)
