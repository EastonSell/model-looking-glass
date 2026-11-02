"""Small, synthetic worker-failure exercise; this module has no launch entry.

The direct trace probe and the zero-trace fixed worker are distinct phases.
This is not a trace-enabled service or a production workload profile.
"""
import threading
import weakref


MODEL_ID = 'a' * 64  # Synthetic test label, not a checkpoint attestation.
TRACE_BYTES = 64
TRACE_WORKSPACE_BYTES = 16 * 1024**2
WORKER_WORKSPACE_BYTES = 32 * 1024**2


class SyntheticTraceCancellation(RuntimeError):
    """A test-only cancellation used to exercise failed trace cleanup."""


class TraceCustody:
    """Caller-rooted state; an uncertain completion keeps every known owner."""
    def __init__(self):
        self.lease = None
        self.trace = None
        self.events = None
        self.input = None
        self.output = None
        self.error = None
        self.capture_ref = None
        self.input_ref = self.output_ref = self.logits_ref = None
        self.charge_expected = 0


class WorkerCustody:
    """Retain the actual worker, authority, broker, leases and failure outcome."""
    def __init__(self):
        self.model = self.resident = self.workspace = self.authority = None
        self.worker = self.broker = self.handle = self.outcome = None
        self.budget = self.resident_lease = None
        self.error = None


def _await_broker_retirement(root, guard, timeout=20):
    """Observe the trusted single-producer broker's idle retirement point.

    This does not dispose aliases held by clients or prove native cleanup.
    """
    broker, authority = root.broker, root.authority
    try:
        with broker._condition:
            def unhealthy():
                return (authority.unsafe or broker.quarantined or broker._fatal_cleanup_failure
                        or broker._closed or broker._unavailable is not None or not broker._thread.is_alive())
            def retired():
                return (broker._running is None and broker._preparing == 0 and broker._pending_slots == 0
                        and not broker._fair_preparations and broker._fair_handoff is None
                        and not len(broker._queue) and all(job is None for job in broker._job_roots))
            if not broker._wait_condition_locked(lambda: unhealthy() or retired(), timeout):
                raise TimeoutError('broker retirement did not finish')
            if unhealthy() or not retired():
                raise RuntimeError('broker retirement is unverified')
        custody = root.worker._custody
        if (any(getattr(custody, name) is not None for name in
                ('request', 'logits', 'result', 'error', 'producer_error', 'forward_error', 'state_error'))
                or root.worker._retiring_cancel is not None or root.worker._retiring_transfer is not None
                or root.resident._drain_pending):
            raise RuntimeError('worker request custody did not retire')
        guard()
    except BaseException:
        try:
            authority.mark_unsafe()
        finally:
            authority._unsafe = True
        raise


def run_trace_probe(model, *, budget, device, guard, synchronize, custody,
                    inject_cancellation=False):
    """Run one separate bounded trace and retire it only after completion.

    The intentional cancellation exercises trace-session failure cleanup. It
    is distinct from the worker's cooperative request cancellation below.
    Any unexpected error retains the lease and private objects for process exit.
    """
    import torch
    from .tap_repeated_owner import _RetainedEvents
    from .transformer_trace import GPT2TraceEvent, GPT2TraceSession

    if type(custody) is not TraceCustody or type(inject_cancellation) is not bool:
        raise TypeError('exact custody and cancellation flag required')
    if device not in ('cpu', 'cuda:0') or any(getattr(custody, name) is not None
                                                for name in ('lease', 'trace', 'events', 'input', 'output')):
        raise ValueError('fresh custody and a fixed device are required')
    location = 'gpu' if device == 'cuda:0' else 'host'
    try:
        guard()
        custody.lease = budget.acquire(TRACE_WORKSPACE_BYTES,
            max_gpu_trace_bytes=TRACE_BYTES if location == 'gpu' else 0,
            max_host_trace_bytes=TRACE_BYTES if location == 'host' else 0)
        custody.lease.reserve_trace(location, TRACE_BYTES)
        custody.charge_expected = TRACE_BYTES
        guard()
        custody.input = torch.tensor([[1, 2]], dtype=torch.int64).to(device)
        guard()
        custody.trace = GPT2TraceSession(model, run_id='synthetic-failure-trace', targets=('block',),
            layers=(0,), samples=(0,), tokens=(1,), mode='direct', max_events=1,
            max_trace_bytes=TRACE_BYTES, capture_device='source')
        custody.events = _RetainedEvents(GPT2TraceEvent)
        custody.trace.events = custody.events
        try:
            with custody.trace:
                guard()
                with torch.inference_mode():
                    custody.output = model(input_ids=custody.input, use_cache=False)
                if inject_cancellation:
                    raise SyntheticTraceCancellation('deliberate trace cancellation')
        except SyntheticTraceCancellation:
            if not inject_cancellation:
                raise
        if (len(custody.events) != 1 or custody.events[0].tensor_bytes != TRACE_BYTES
                or custody.events[0].value is None):
            raise RuntimeError('the fixed trace capture was not retained')
        event = custody.events[0]
        if (event.run_id != 'synthetic-failure-trace' or event.sequence != 0
                or event.module_path != 'transformer.h.0' or event.capture_mode != 'direct'
                or tuple(event.value.shape) != (1, 1, 16)
                or event.value.dtype != torch.float32 or str(event.value.device) != device):
            raise RuntimeError('trace capture identity or shape differs')
        del event
        if custody.trace.successful is inject_cancellation:
            raise RuntimeError('trace terminal status is inconsistent')
        custody.capture_ref = weakref.ref(custody.events[0].value)
        custody.input_ref = weakref.ref(custody.input)
        custody.output_ref = weakref.ref(custody.output)
        if type(custody.output.logits) is not torch.Tensor:
            raise RuntimeError('trace output logits have an unsupported type')
        custody.logits_ref = weakref.ref(custody.output.logits)
        guard()
        synchronize()
        guard()
        refs = (custody.capture_ref, custody.input_ref, custody.output_ref, custody.logits_ref)
        retained = all(reference() is not None for reference in refs)
        if not retained:
            raise RuntimeError('a trace resource vanished before completion synchronization')
        # Check the owner around each last-reference drop. No cyclic GC or
        # unguarded cleanup is used to establish capture disposal.
        for name in ('output', 'input', 'trace', 'events'):
            guard()
            setattr(custody, name, None)
            guard()
        if any(reference() is not None for reference in refs):
            raise RuntimeError('a trace resource remains externally reachable')
        guard()
        synchronize()
        guard()
        custody.lease.release_trace(location, TRACE_BYTES)
        custody.charge_expected = 0
        guard()
        custody.lease.release()
        return {'outcome': 'synthetic_trace_cancellation' if inject_cancellation else 'success',
                'event_bytes': TRACE_BYTES, 'capture_retained_until_sync': retained,
                'capture_released': custody.capture_ref() is None,
                'trace_resources_retained_until_sync': retained,
                'trace_resources_released': all(reference() is None for reference in refs),
                'trace_claim_released': custody.lease.released}
    except BaseException as error:
        custody.error = error
        raise


def run_worker_probe(model, *, budget, resident_lease, device, guard, checkpoint, record, custody):
    """Run a no-trace worker baseline, cooperative cancel, recovery and OOM.

    The final OOM is deliberately raised by the trusted synthetic model after
    real tensor work. Once the authority becomes unsafe, this function performs
    no checkpoint or deliberate device operation and returns retained custody.
    """
    import torch
    from torch import nn
    from .broker import Broker, RequestState
    from .budgeted_worker import WorkloadProfile
    from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
    from .resident_worker import ForwardRequest, ResidentWorker

    if (device not in ('cpu', 'cuda:0') or not callable(guard) or not callable(checkpoint)
            or not callable(record)
            or type(custody) is not WorkerCustody
            or any(getattr(custody, name) is not None for name in
                   ('model', 'resident', 'workspace', 'authority', 'worker', 'broker', 'handle',
                    'budget', 'resident_lease', 'outcome', 'error'))):
        raise ValueError('fixed device, trusted callbacks and fresh custody required')

    class ControlledModel(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.config = inner.config
            self.mode = 'normal'
            self.entered = threading.Event()
            self.resume = threading.Event()
            self.invocations = 0
            self.oom_injected = False

        def forward(self, input_ids):
            self.invocations += 1
            output = self.inner(input_ids=input_ids, use_cache=False)
            if self.mode == 'cancel':
                self.entered.set()
                if not self.resume.wait(10):
                    raise RuntimeError('cancel coordination timed out')
            elif self.mode == 'oom':
                self.oom_injected = True
                raise torch.OutOfMemoryError('deliberately simulated worker OOM')
            return output

    root = custody
    root.budget, root.resident_lease = budget, resident_lease

    def submit(name):
        request = ForwardRequest(name, {'input_ids': torch.tensor([[1, 2]], dtype=torch.int64)})
        return root.broker.submit('synthetic', request)

    def success(name):
        handle = submit(name)
        outcome = handle.wait(20)
        if outcome.state is not RequestState.SUCCEEDED or tuple(outcome.result.logits.shape) != (1, 2, 64):
            raise RuntimeError('the expected synthetic worker success was not observed')
        if not bool(torch.isfinite(outcome.result.logits).all()):
            raise RuntimeError('the synthetic output was nonfinite')
        _await_broker_retirement(root, guard)
        return 'succeeded'

    try:
        root.model = ControlledModel(model).eval()
        root.resident = ResidentWorker(MODEL_ID, root.model, device=device, max_batch_size=1,
                                       max_sequence_length=2, max_input_bytes=64, max_output_bytes=512)
        root.workspace = budget.acquire(WORKER_WORKSPACE_BYTES, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        root.authority = OwnerCompletionAuthority(device_guard=guard)
        profile = WorkloadProfile('synthetic-only', MODEL_ID, device, (1, 2),
                                  (('input_ids', 'torch.int64'),), 512, WORKER_WORKSPACE_BYTES)
        root.worker = LifetimeWorkspaceWorker(root.resident, budget, resident_lease, root.workspace,
                                               profiles=(profile,), authority=root.authority)
        root.broker = Broker(root.worker, max_pending=2, max_pending_per_team=2,
                             completion_authority=root.authority)
        def snapshot(stage):
            value = checkpoint(stage)
            record(stage, value)
            return value
        checkpoints = [snapshot('before_worker')]
        baseline = success('baseline')
        record('baseline', {'outcome': baseline})
        checkpoints.append(snapshot('after_baseline'))
        root.model.mode = 'cancel'
        handle = submit('cooperative-cancel')
        if not root.model.entered.wait(20):
            raise RuntimeError('the cancellation forward did not enter')
        try:
            accepted = handle.cancel()
        finally:
            root.model.resume.set()
        cancelled = handle.wait(20)
        if accepted is not True or cancelled.state is not RequestState.CANCELLED or root.authority.unsafe:
            raise RuntimeError('the synchronized cancellation did not finish safely')
        _await_broker_retirement(root, guard)
        record('cancellation', {'accepted': True, 'outcome': 'cancelled'})
        checkpoints.append(snapshot('after_cancellation'))
        root.model.mode = 'normal'
        recovery = success('after-cancellation')
        record('post_cancel', {'outcome': recovery})
        checkpoints.append(snapshot('after_post_cancel'))
        checkpoints.append(snapshot('before_simulated_oom'))
        root.model.mode = 'oom'
        root.handle = submit('simulated-oom')
        try:
            root.outcome = root.handle.wait(20)
            observed = root.outcome.state.value
        except Exception as error:
            root.error = error
            observed = 'unresolved'
        if observed not in ('failed', 'unresolved'):
            raise RuntimeError('simulated OOM was published as an unsupported outcome')
        if root.outcome is not None and root.outcome.result is not None:
            raise RuntimeError('failed OOM outcome unexpectedly contains a result')
        root.broker._thread.join(20)
        if root.broker._thread.is_alive():
            raise RuntimeError('broker did not stop after unsafe worker failure')
        if not root.model.oom_injected or not root.authority.unsafe:
            raise RuntimeError('the simulated OOM did not reach the unsafe boundary')
        before_retry = root.model.invocations
        try:
            submit('must-refuse-after-oom')
        except Exception:
            refused = True
        else:
            refused = False
        if not refused or root.model.invocations != before_retry:
            raise RuntimeError('unsafe worker accepted another request')
        private = root.worker._custody.forward_error
        active = budget.active_leases()  # Host accounting only; no telemetry call.
        if (not isinstance(private, torch.OutOfMemoryError)
                or set(active) != {resident_lease, root.workspace}):
            raise RuntimeError('simulated OOM custody or commitments were lost')
        oom_report = {'injection_reached': True, 'outcome': observed,
                    'public_error_type': (None if root.outcome is None else root.outcome.error_type),
                    'private_forward_error_type': type(private).__name__,
                    'worker_unsafe': root.authority.unsafe, 'subsequent_work_refused': refused,
                    'active_claim_count': len(active), 'no_explicit_checkpoint_after_unsafe': True}
        record('simulated_oom', oom_report)
        return {'baseline': baseline, 'cancellation': {'accepted': accepted, 'outcome': 'cancelled'},
                'post_cancel': recovery, 'checkpoints': checkpoints, 'simulated_oom': oom_report}
    except BaseException as error:
        root.error = error
        if root.model is not None:
            root.model.resume.set()
        raise
