"""Private, cooperative CPU custody for one complete traced GPT-2 job.

This owner is not a broker, allocator, transport, or isolation boundary. Its
caller must own the model and inputs exclusively and attest actual disposal.
"""

import threading

import torch

from .host_budget import HostBudget, HostClaim, HostRequest
from .producer_slot import ProducerSlotRegistry
from .resident_worker import ForwardRequest
from .trace_batch_owner import BatchStateError, TraceBatchRegistry
from .trace_job_validation import _strip_failure_frames, make_trace_job_layout
from .trace_producer import ClaimLimit, ProducerProfile
from .trace_store import HostTraceStore
from .traced_broker_worker import (BoundTracePlan, SampleTrace, TraceLimits,
                                   TraceSpec, bind_trace_plan)
from .traced_worker import TraceSite, TracedGPT2Worker


class CpuTraceStateError(RuntimeError):
    """The private job is unavailable, stale, cancelled, or requires recovery."""


def _hex(value):
    return (type(value) is str and len(value) == 32
            and all(character in "0123456789abcdef" for character in value))


def _profile_snapshot(profile, worker, plan):
    if type(profile) is not ProducerProfile:
        raise ValueError("profile must be an exact ProducerProfile")
    limits = []
    for name in ("input_copies", "output", "trace_bundle", "workspace"):
        item = getattr(profile, name)
        if type(item) is not ClaimLimit:
            raise ValueError("profile stages must be exact ClaimLimit objects")
        limits.append(ClaimLimit(item.nbytes, item.items))
    copied = ProducerProfile(*limits, device=profile.device)
    if (copied.input_copies.nbytes < 2 * worker.max_input_bytes
            or copied.input_copies.items < 8
            or copied.output.nbytes < 2 * worker.max_output_bytes
            or copied.output.items < 2
            or copied.trace_bundle.nbytes < plan._max_trace_bytes
            or copied.trace_bundle.items < plan._max_events
            or copied.workspace.nbytes < 1):
        raise ValueError("producer profile does not cover required overlapping stages")
    return copied


def _model_facts(worker, model, batch_size, sequence_length):
    if worker._model is not model or worker.device.type != "cpu" or worker.poisoned:
        raise ValueError("the bound CPU worker/model is unavailable or changed")
    if type(batch_size) is not int or type(sequence_length) is not int:
        raise ValueError("batch and sequence dimensions must be plain integers")
    if not (1 <= batch_size <= worker.max_batch_size
            and 1 <= sequence_length <= worker.max_sequence_length):
        raise ValueError("batch or sequence exceeds the worker limits")
    config = model.config
    if (model.training or config.use_cache is not False
            or bool(getattr(config, "output_hidden_states", False))
            or bool(getattr(config, "output_attentions", False))):
        raise ValueError("model must be in evaluation mode without cache or auxiliary outputs")
    if any(parameter.device.type != "cpu" or parameter.dtype is not torch.float32
           for parameter in model.parameters()):
        raise ValueError("model parameters must be CPU float32")
    width = config.n_embd
    layers = len(model.transformer.h)
    head = model.lm_head
    if (type(width) is not int or width < 1 or layers < 1
            or type(head) is not torch.nn.Linear or head.in_features != width
            or type(head.out_features) is not int or head.out_features < 1
            or tuple(head.weight.shape) != (head.out_features, width)):
        raise ValueError("unsupported GPT-2 model dimensions")
    if batch_size * sequence_length * head.out_features * 4 > worker.max_output_bytes:
        raise ValueError("the raw model logits would exceed the worker output limit")
    return width, layers


def _copy_plan(plan, worker, *, batch_size, sequence_length, layer_count, max_slots):
    if type(plan) is not BoundTracePlan:
        raise ValueError("plan must be an exact BoundTracePlan")
    # Do not reconstruct attacker-supplied oversized selections merely to
    # reject them. The input object itself is owned and metered by its caller.
    if (type(plan._samples) is not tuple or len(plan._samples) != batch_size
            or type(plan._sites) is not tuple or not plan._sites
            or len(plan._sites) > max_slots
            or len(plan._samples) * len(plan._sites) > max_slots
            or type(plan._max_events) is not int or not 1 <= plan._max_events <= max_slots):
        raise ValueError("trace selection exceeds this owner's metadata limits")
    if (type(plan._identity) is not tuple or len(plan._identity) != 4
            or not _hex(plan._identity[0]) or type(plan._identity[1]) is not int
            or any(type(value) is not str or not 1 <= len(value) <= 128
                   for value in plan._identity[2:])
            or type(plan._run_id) is not str or not 1 <= len(plan._run_id) <= 128):
        raise ValueError("invalid bounded trace identity")
    for sample in plan._samples:
        if (type(sample) is not tuple or len(sample) != 2
                or type(sample[0]) is not str or not 1 <= len(sample[0]) <= 128
                or type(sample[1]) is not tuple or not 1 <= len(sample[1]) <= 128):
            raise ValueError("invalid bounded sample selection")
    for site in plan._sites:
        if (type(site) is not tuple or len(site) != 2 or type(site[0]) is not int
                or type(site[1]) is not str or site[1] not in ("block", "attn", "mlp")):
            raise ValueError("invalid bounded site selection")
    # Snapshot and revalidate caller-owned frozen objects; callers must not
    # mutate them concurrently while this snapshot is being constructed.
    request = plan.capture_request()
    identity = plan.identity
    samples = tuple(SampleTrace(sample.sample_id, sample.token_positions)
                    for sample in request.samples)
    sites = plan.sites
    starts = plan.sequence_starts()
    if type(starts) is not dict or set(starts) != {request.run_id}:
        raise ValueError("invalid run sequence allocation")
    spec = TraceSpec(request.run_id, samples, sites, request.forward_index,
                     starts[request.run_id], request.generation_step,
                     request.max_events, request.max_trace_bytes)
    limits = TraceLimits(worker.max_batch_size, worker.max_sequence_length,
                         layer_count, max_slots, max_slots,
                         spec.max_trace_bytes, 128)
    return bind_trace_plan(identity, spec, limits, batch_size=batch_size,
                           sequence_length=sequence_length, layer_count=layer_count)


class CpuTraceOwner:
    """Root one trusted in-process job around one exact CPU worker and model."""

    def __init__(self, *, worker, budget, store, broker_id, process_generation,
                 max_slots, max_event_bytes):
        if (type(worker) is not TracedGPT2Worker or type(budget) is not HostBudget
                or type(store) is not HostTraceStore or not _hex(broker_id)
                or not _hex(process_generation) or type(max_slots) is not int
                or not 1 <= max_slots <= 65536 or type(max_event_bytes) is not int
                or max_event_bytes < 1):
            raise ValueError("invalid trusted CPU trace owner configuration")
        if worker.device.type != "cpu" or worker._model is None:
            raise ValueError("an available CPU traced worker is required")
        self._worker = worker
        self._model = worker._model
        self._budget = budget
        self._broker_id = broker_id
        self._process_generation = process_generation
        self._max_slots = max_slots
        self._max_event_bytes = max_event_bytes
        self._producers = ProducerSlotRegistry(max_slots=1)
        self._batches = TraceBatchRegistry(budget=budget, store=store, broker_id=broker_id,
                                           process_generation=process_generation,
                                           max_jobs=1, max_slots_per_job=max_slots)
        self._lock = threading.Lock()
        self._job = None
        self._generation = 0

    def register(self, plan, *, profile, registration_id, generation,
                 batch_size, sequence_length):
        if not _hex(registration_id) or type(generation) is not int or not 1 <= generation < 2**63:
            raise ValueError("invalid registration key")
        pending = object()
        try:
            with self._lock:
                if self._job is not None or generation <= self._generation:
                    raise CpuTraceStateError("owner is busy or generation is stale")
                self._generation = generation
                self._job = pending
            width, layers = _model_facts(self._worker, self._model, batch_size, sequence_length)
            copied_plan = _copy_plan(plan, self._worker, batch_size=batch_size,
                                     sequence_length=sequence_length, layer_count=layers,
                                     max_slots=self._max_slots)
            if copied_plan.identity.broker_id != self._broker_id:
                raise ValueError("plan belongs to another broker")
            copied_profile = _profile_snapshot(profile, self._worker, copied_plan)
            make_trace_job_layout(copied_plan, batch_size=batch_size,
                                  sequence_length=sequence_length, hidden_width=width,
                                  layer_count=layers, max_slots=self._max_slots,
                                  max_event_bytes=self._max_event_bytes)
            job = _CpuTraceJob(self, copied_plan, copied_profile, registration_id,
                               generation, batch_size, sequence_length, width, layers)
            with self._lock:
                if self._job is not pending:
                    raise CpuTraceStateError("registration ownership changed")
                self._job = job
        except BaseException:
            with self._lock:
                if self._job is pending:
                    self._job = None
            raise

    def lookup(self, job_id, *, registration_id, generation):
        if (type(job_id) is not int or not 1 <= job_id < 2**63
                or not _hex(registration_id) or type(generation) is not int
                or not 1 <= generation < 2**63):
            raise CpuTraceStateError("unknown or stale job")
        with self._lock:
            job = self._job
            if (type(job) is not _CpuTraceJob or job._job_id != job_id
                    or registration_id != job._registration_id or generation != job._generation):
                raise CpuTraceStateError("unknown or stale job")
        return job

    def retire(self, job):
        if type(job) is not _CpuTraceJob or job._owner is not self:
            raise CpuTraceStateError("foreign job")
        with job._lock:
            if job._phase != "disposed" or job._active is not None:
                raise CpuTraceStateError("job has not drained")
            with self._lock:
                if self._job is not job:
                    raise CpuTraceStateError("stale job")
                self._job = None
                job._phase = "retired"


class _CpuTraceJob:
    def __init__(self, owner, plan, profile, registration_id, generation,
                 batch_size, sequence_length, width, layers):
        self._owner, self._plan, self._profile = owner, plan, profile
        self._job_id = plan.identity.job_id
        self._registration_id, self._generation = registration_id, generation
        self._batch_size, self._sequence_length = batch_size, sequence_length
        self._width, self._layers = width, layers
        self._producer_key = (f"p-{registration_id}", generation)
        self._input_label = f"input-{self._job_id}-{registration_id}-{generation}"
        self._producer = self._producer_slot = self._batch = None
        self._result = self._input_operation = self._input_request = None
        self._abort_sources_confirmed = False
        self._input_state = "unattempted"
        self._run_state = "unattempted"
        self._failed = False
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._phase = "registered"
        self._active = None

    def __copy__(self):
        raise TypeError("job capabilities cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("job capabilities cannot be copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("job capabilities cannot be serialized")

    @property
    def state(self):
        with self._lock:
            state = self._phase
        return state

    def _begin(self, expected, phase, token, *, allow_cancel=False):
        with self._lock:
            if (self._active is not None or self._phase not in expected
                    or (self._cancel.is_set() and not allow_cancel)):
                raise CpuTraceStateError("job is not in the required state")
            self._active = token
            self._phase = phase

    def _quarantine(self, token):
        with self._lock:
            if self._active is token:
                self._cancel.set()
                self._phase = "quarantined"
                self._active = None

    def prepare(self):
        """Pre-admit producer and every delivery source before any forward."""
        token = object()
        try:
            self._begin(("registered",), "preparing", token)
            slots = self._owner._producers
            slots.register(*self._producer_key)
            self._producer_slot = slots.lookup(*self._producer_key)
            self._producer_slot.construct(self._owner._budget,
                                          f"producer-{self._owner._broker_id}-{self._registration_id}",
                                          self._profile)
            self._producer = self._producer_slot.adopt()
            registry = self._owner._batches
            registry.register(self._plan, registration_id=self._registration_id,
                              generation=self._generation, batch_size=self._batch_size,
                              sequence_length=self._sequence_length, hidden_width=self._width,
                              layer_count=self._layers, max_event_bytes=self._owner._max_event_bytes)
            self._batch = registry.lookup(self._job_id, registration_id=self._registration_id,
                                          generation=self._generation)
            self._batch.prepare()
            with self._lock:
                if self._cancel.is_set():
                    raise CpuTraceStateError("job was cancelled")
                self._phase = "prepared"
                self._active = None
        except BaseException:
            # No forward starts in prepare. Exact child slots remain rooted and
            # can be inspected by abort_before_forward; ambiguous children are
            # kept quarantined rather than guessed away.
            with self._lock:
                if self._active is token:
                    self._cancel.set()
                    self._phase = "admission_failed"
                    self._active = None
            raise

    def execute(self, request, *, input_owner, input_claim):
        """Capture and privately stage one complete job; return no payloads."""
        token = object()
        pin_attempted = False
        try:
            self._begin(("prepared",), "validating", token)
            owner = self._owner
            if _model_facts(owner._worker, owner._model, self._batch_size,
                            self._sequence_length) != (self._width, self._layers):
                raise ValueError("bound model dimensions changed")
            _profile_snapshot(self._profile, owner._worker, self._plan)
            if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
                raise ValueError("ambient autocast is unsupported")
            if (type(request) is not ForwardRequest
                    or request.request_id != self._plan.worker_request_id):
                raise ValueError("request does not match the job")
            owner._worker.validate_request(request)
            if tuple(request.inputs["input_ids"].shape) != (self._batch_size, self._sequence_length):
                raise ValueError("input shape does not match the job")
            if any(not value.is_contiguous() or value.storage_offset() != 0
                   or value.untyped_storage().nbytes() != value.numel() * value.element_size()
                   for value in request.inputs.values()):
                raise ValueError("inputs must occupy full contiguous reported storage")
            if (type(input_owner) is not HostRequest or type(input_claim) is not HostClaim
                    or input_owner._budget is not owner._budget or input_claim._budget is not owner._budget):
                raise ValueError("input request and claim must belong to this host budget")
            info = input_claim.info()
            if (info.request_id != input_owner._request_id
                    or info.nbytes < sum(value.numel() * value.element_size() for value in request.inputs.values())
                    or info.items < len(request.inputs)):
                raise ValueError("input claim is too small or belongs to a different request")
            pin_attempted = True
            self._input_state = "begin_pending"
            self._input_operation = input_owner.begin_operation(self._input_label, (input_claim,))
            self._input_state = "active"
            # Root the source only after the caller's claim is pinned. A lost
            # begin-operation return is quarantined and never installs a new
            # owner-held reference to an unpinned input.
            self._input_request = request
            with self._lock:
                if self._cancel.is_set():
                    raise CpuTraceStateError("job was cancelled")
                self._phase = "running"
            def forward():
                self._result = owner._worker.execute_traced(
                    self._input_request, captures=(self._plan.capture_request(),),
                    sites=tuple(TraceSite(site.layer_index, site.module_kind) for site in self._plan.sites),
                    sequence_starts=self._plan.sequence_starts(),
                    max_events=self._plan._max_events, max_trace_bytes=self._plan._max_trace_bytes,
                    cancel_event=self._cancel)
                return None
            self._run_state = "running"
            self._producer.run(forward)
            self._run_state = "returned"
            # These are the owner's last known references to caller input. Drop
            # them while its claim is still pinned; external caller aliases are
            # subject to their own separately charged lifetime.
            request = None
            self._input_request = None
            self._input_state = "finish_pending"
            input_live = self._input_operation.finish()
            self._input_operation = None
            self._input_state = "finished"
            if not input_live:
                raise CpuTraceStateError("input request was cancelled")
            self._batch.stage(self._result.events)
            with self._lock:
                if self._cancel.is_set():
                    raise CpuTraceStateError("job was cancelled")
                self._phase = "staged"
                self._active = None
        except BaseException as error:
            if pin_attempted:
                # A completed synchronous producer and an intact input pin
                # admit cooperative cleanup after explicit alias attestation.
                # Lost begin/finish or interruption outcomes remain uncertain.
                recoverable = False
                if isinstance(error, Exception) and self._input_state in ("active", "finished"):
                    with self._producer._lock:
                        recoverable = (self._producer._operation is None
                                       and (self._producer._state in ("completed", "failed", "cancelled")
                                            or (self._run_state == "unattempted"
                                                and self._producer._state == "reserved")))
                if recoverable:
                    try:
                        self._cancel.set()
                        self._batch.cancel()
                        with self._lock:
                            if self._active is token:
                                self._failed = True
                                self._phase = "failed"
                                self._active = None
                    except BaseException:
                        self._quarantine(token)
                        raise
                else:
                    self._quarantine(token)
            else:
                with self._lock:
                    if self._active is token:
                        self._phase = "prepared"
                        self._active = None
                # Retaining a rejected request's ordinary exception must not
                # itself keep caller input through this frame or validation
                # frames. This remains best effort for custom exceptions and
                # oversized exception groups, which require owner attestation.
                try:
                    _strip_failure_frames(error)
                except BaseException:
                    pass
            request = input_owner = input_claim = None
            raise

    def discard_result(self):
        """Drop the owner's complete worker result; caller must dispose aliases."""
        token = object()
        result = None
        try:
            self._begin(("staged", "failed"), "discarding", token, allow_cancel=True)
            with self._lock:
                result = self._result
                self._result = None
                request = self._input_request
                self._input_request = None
            # Tensor finalizers may call back into this owner. Never destroy the
            # bundle while holding its non-reentrant state lock.
            result = None
            request = None
            with self._lock:
                self._phase = "result_discarded"
                self._active = None
        except BaseException:
            self._quarantine(token)
            raise

    def abort_before_forward(self, *, timeout):
        """Cancel and drain a known unstarted job, keeping uncertain claims."""
        if type(timeout) not in (int, float) or not 0 <= timeout <= 86400:
            raise ValueError("invalid timeout")
        token = object()
        try:
            self._begin(("registered", "prepared", "admission_failed", "abort_pending"),
                        "aborting", token,
                        allow_cancel=True)
            self._cancel.set()
            batches = self._owner._batches
            if self._job_id in batches.active_jobs():
                batch = batches.lookup(self._job_id, registration_id=self._registration_id,
                                       generation=self._generation)
                batch.cancel()
                if not self._abort_sources_confirmed:
                    batch.confirm_sources_disposed()
                    self._abort_sources_confirmed = True
                if not batch.dispose(timeout=timeout):
                    with self._lock:
                        self._phase = "abort_pending"
                        self._active = None
                    return False
                batches.retire(batch)
                self._batch = None
            slots = self._owner._producers
            active = slots.active_slots()
            if active:
                slot = slots.lookup(*self._producer_key)
                state = slot.state
                if state in ("ready", "cancelled"):
                    envelope = slot.owned_envelope()
                    for stage in ("input_copies", "output", "trace_bundle", "workspace"):
                        envelope.confirm_disposed(stage)
                    envelope.close()
                elif state not in ("reserved", "failed_clean"):
                    raise CpuTraceStateError("producer admission needs supervised recovery")
                slot.retire()
                self._producer = self._producer_slot = None
            with self._lock:
                self._phase = "disposed"
                self._active = None
            return True
        except BaseException:
            self._quarantine(token)
            raise

    def confirm_producer_disposed(self):
        """Attest all source, output, workspace and worker aliases are gone."""
        token = object()
        try:
            self._begin(("result_discarded",), "confirming", token, allow_cancel=True)
            if self._input_state == "active":
                self._input_state = "finish_pending"
                self._input_operation.finish()
                self._input_operation = None
                self._input_state = "finished"
            self._batch.confirm_sources_disposed()
            for stage in ("input_copies", "output", "trace_bundle", "workspace"):
                self._producer.confirm_disposed(stage)
            self._producer.close()
            self._producer_slot.retire()
            self._producer = self._producer_slot = None
            with self._lock:
                self._phase = "confirmed_failed" if self._failed else "confirmed"
                self._active = None
        except BaseException:
            self._quarantine(token)
            raise

    def seal(self):
        token = object()
        try:
            self._begin(("confirmed",), "sealing", token)
            self._batch.seal()
            with self._lock:
                if self._cancel.is_set():
                    raise CpuTraceStateError("job was cancelled")
                self._phase = "sealed"
                self._active = None
        except BaseException:
            self._quarantine(token)
            raise

    def manifest(self):
        with self._lock:
            if self._phase != "sealed" or self._active is not None or self._cancel.is_set():
                raise CpuTraceStateError("complete manifest is unavailable")
            batch = self._batch
        return batch.manifest()

    def cancel(self):
        self._cancel.set()
        with self._lock:
            batch = self._batch
        if batch is not None:
            batch.cancel()

    def dispose(self, *, timeout):
        token = object()
        try:
            self._begin(("confirmed", "confirmed_failed", "sealed"), "disposing", token,
                        allow_cancel=True)
            self._cancel.set()
            if not self._batch.dispose(timeout=timeout):
                with self._lock:
                    self._phase = "confirmed_failed" if self._failed else "confirmed"
                    self._active = None
                return False
            self._owner._batches.retire(self._batch)
            self._batch = None
            with self._lock:
                self._phase = "disposed"
                self._active = None
            return True
        except BaseException:
            self._quarantine(token)
            raise
