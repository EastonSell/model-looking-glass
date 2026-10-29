"""Shared-service load benchmarking; no isolated-latency claims.

The benchmark deliberately keeps raw attempt and broker timestamps so that
reported rates and distributions can be independently recomputed.
"""
from dataclasses import asdict, dataclass, fields, is_dataclass
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import re
from math import ceil, isfinite
from statistics import median
import threading
import time
import uuid


_STATES = ('succeeded', 'failed', 'cancelled', 'rejected', 'unresolved', 'invalid_output', 'submit_error')


def _distribution(values):
    if not values:
        return {'p50': None, 'p95': None, 'max': None, 'n': 0}
    ordered = sorted(values)
    return {'p50': median(ordered), 'p95': ordered[ceil(.95 * len(ordered)) - 1],
            'max': ordered[-1], 'n': len(ordered)}


def _validate_sample(row):
    if row['state'] not in _STATES:
        raise ValueError('unknown terminal state')
    if not all(isinstance(row[k], str) and row[k] for k in ('job_id', 'team_id', 'workload_class')):
        raise ValueError('job, team and class must be nonempty strings')
    if type(row['units']) is not int or row['units'] <= 0:
        raise ValueError('units must be positive integers')
    attempt, submitted, started, finished, returned = (row[k] for k in
        ('attempt_ns', 'submitted_ns', 'started_ns', 'finished_ns', 'returned_ns'))
    ready = row.get('ready_ns')
    timestamps = (attempt, submitted, ready, started, finished, returned)
    if any(t is not None and (type(t) is not int or t < 0) for t in timestamps):
        raise ValueError('timestamps must be nonnegative integer nanoseconds')
    if attempt is None or returned is None or returned < attempt:
        raise ValueError('attempt/return timestamps are invalid')
    if row['state'] in ('failed', 'rejected', 'unresolved', 'invalid_output', 'submit_error') and not row['error']:
        raise ValueError('unsuccessful work requires error evidence')
    if row['state'] in ('rejected', 'submit_error'):
        if any(t is not None for t in (submitted, ready, started, finished)):
            raise ValueError('rejected work cannot have broker timestamps')
        return
    if submitted is None or not attempt <= submitted <= returned:
        raise ValueError('invalid broker submission timestamp')
    if ready is not None and not submitted <= ready <= (started if started is not None else (finished if finished is not None else returned)):
        raise ValueError('invalid ready-to-dispatch timestamp')
    if started is not None and not submitted <= started <= returned:
        raise ValueError('invalid broker start timestamp')
    if finished is not None and (finished < (started if started is not None else submitted) or finished > returned):
        raise ValueError('invalid broker finish timestamp')
    if row['state'] == 'unresolved' and finished is not None:
        raise ValueError('unresolved work cannot have a terminal timestamp')
    if row['state'] == 'succeeded' and row['error'] is not None:
        raise ValueError('successful work cannot contain an error')
    if row['state'] != 'unresolved' and finished is None:
        raise ValueError('terminal accepted work needs a finish timestamp')
    if row['state'] in ('succeeded', 'invalid_output') and started is None:
        raise ValueError('successful work must have started')
    if row['state'] == 'succeeded' and (not isinstance(row['output_digest'], str) or not row['output_digest']):
        raise ValueError('successful work requires an output digest')


def _summarize_group(samples, seconds):
    states = {state: sum(row['state'] == state for row in samples) for state in _STATES}
    counts = {'attempted': len(samples), 'accepted': len(samples) - states['rejected'] - states['submit_error'], **states}
    successes = [row for row in samples if row['state'] == 'succeeded']
    units = sum(row['units'] for row in successes)
    return {
        'counts': counts,
        'successful_units': units,
        'units_definition': 'caller_declared',
        'successes_per_second': len(successes) / seconds,
        'successful_units_per_second': units / seconds,
        'nonsuccess_rate': 1 - len(successes) / len(samples),
        'queue_wait_ns_started': _distribution([row['started_ns'] - row['ready_ns']
                                                 for row in samples if row['started_ns'] is not None
                                                 and row.get('ready_ns') is not None]),
        'submission_to_service_ns_started': _distribution([row['started_ns'] - row['submitted_ns']
                                                          for row in samples if row['started_ns'] is not None]),
        'service_ns_started': _distribution([row['finished_ns'] - row['started_ns']
                                              for row in samples if row['started_ns'] is not None
                                              and row['finished_ns'] is not None]),
        'broker_response_ns_terminal': _distribution([row['finished_ns'] - row['submitted_ns']
                                                       for row in samples if row['finished_ns'] is not None]),
        'response_ns_all_attempts': _distribution([row['returned_ns'] - row['attempt_ns'] for row in samples]),
        'response_ns_succeeded': _distribution([row['returned_ns'] - row['attempt_ns'] for row in successes]),
        'timing_by_state': {state: {
            'response_ns': _distribution([r['returned_ns'] - r['attempt_ns'] for r in samples if r['state'] == state]),
            'queue_wait_ns': _distribution([r['started_ns'] - r['ready_ns'] for r in samples
                                             if r['state'] == state and r['started_ns'] is not None
                                             and r.get('ready_ns') is not None]),
            'submission_to_service_ns': _distribution([r['started_ns'] - r['submitted_ns'] for r in samples
                                                        if r['state'] == state and r['started_ns'] is not None]),
            'service_ns': _distribution([r['finished_ns'] - r['started_ns'] for r in samples
                                         if r['state'] == state and r['started_ns'] is not None
                                         and r['finished_ns'] is not None]),
        } for state in _STATES},
    }


def summarize_window(samples, *, start_ns, end_ns):
    """Summarize one fully drained load window with a shared elapsed denominator.

    p50 is the usual midpoint median; p95 is nearest rank (ceil(.95*n)).
    Distributions expose n and max; null values mean no eligible observations.
    """
    if (not samples or type(start_ns) is not int or type(end_ns) is not int
            or start_ns < 0 or end_ns <= start_ns):
        raise ValueError('a nonempty window with valid boundaries is required')
    elapsed_ns = end_ns - start_ns
    for row in samples:
        _validate_sample(row)
        if not start_ns <= row['attempt_ns'] <= row['returned_ns'] <= end_ns:
            raise ValueError('sample lies outside the measurement window')
    if len({row['job_id'] for row in samples}) != len(samples):
        raise ValueError('duplicate job IDs')
    seconds = elapsed_ns / 1_000_000_000
    summary = _summarize_group(samples, seconds)
    summary['elapsed_ns'] = elapsed_ns
    summary['teams'] = {
        team: _summarize_group([row for row in samples if row['team_id'] == team], seconds)
        for team in sorted({row['team_id'] for row in samples})
    }
    summary['workload_classes'] = {
        cls: _summarize_group([row for row in samples if row['workload_class'] == cls], seconds)
        for cls in sorted({row['workload_class'] for row in samples})
    }
    fractions = [v['counts']['succeeded'] / v['counts']['attempted'] for v in summary['teams'].values()]
    denominator = len(fractions) * sum(x * x for x in fractions)
    summary['completion_balance_jain'] = sum(fractions) ** 2 / denominator if denominator else None
    return summary

@dataclass(frozen=True)
class WorkItem:
    """One immutable job specification; the request payload remains caller-owned.

    The input digest must be recomputed before each window. Callers must also
    prevent concurrent payload mutation while the broker snapshots requests.
    """
    job_id: str
    team_id: str
    workload_class: str
    units: int
    seed: int
    input_digest: str
    capture_policy: str
    request: object


class SynchronizedWorker:
    """Proxy the resident worker and synchronize in its execution thread.

    The supplied synchronization callback must cover the same CUDA device as
    the wrapped worker. For CPU-only validation pass a no-op callback.
    """
    def __init__(self, worker, synchronize):
        self._worker = worker
        self._synchronize = synchronize
        self._synchronization_failed = False

    @property
    def cancellation_exception(self):
        return self._worker.cancellation_exception

    @property
    def poisoned(self):
        return self._synchronization_failed or self._worker.poisoned

    def snapshot_request(self, request):
        return self._worker.snapshot_request(request)

    def _sync(self):
        try:
            self._synchronize()
        except BaseException:
            self._synchronization_failed = True
            raise

    def execute(self, request, cancel_event):
        self._sync()
        try:
            result = self._worker.execute(request, cancel_event)
        except BaseException as original:
            try:
                self._sync()
            except BaseException as sync_error:
                raise original from sync_error
            raise
        self._sync()
        return result

    @property
    def quarantined(self):
        return self._synchronization_failed or bool(getattr(self._worker, 'quarantined', False))

    def close(self):
        if self._synchronization_failed:
            raise RuntimeError('device synchronization failed; retain worker and quarantine this process')
        return self._worker.close()


def _canonical(value):
    """Represent every supported request field without retaining tensor values."""
    if value is None or type(value) in (int, float, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {'type': f'{type(value).__module__}.{type(value).__qualname__}',
                'fields': {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, dict):
        if not all(type(k) is str for k in value):
            raise ValueError('request dictionary keys must be strings')
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return {'sequence_type': type(value).__name__, 'items': [_canonical(v) for v in value]}
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and type(value) is torch.Tensor:
        if value.device.type != 'cpu' or value.layout != torch.strided or value.is_quantized:
            raise ValueError('request tensors must be dense, nonquantized CPU tensors')
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
        if type(raw) is not torch.Tensor or raw.device.type != 'cpu' or not raw.is_contiguous() or raw.dtype != torch.uint8:
            raise ValueError('unsafe tensor byte view')
        digest = hashlib.sha256()
        if raw.numel():
            # The owning tensor remains live throughout the native buffer read.
            # This avoids Python iteration over every byte and works for bfloat16.
            buffer = (ctypes.c_ubyte * raw.numel()).from_address(raw.data_ptr())
            digest.update(memoryview(buffer).cast('B'))
        return {'tensor_shape': list(value.shape), 'tensor_dtype': str(value.dtype),
                'sha256': digest.hexdigest()}
    raise TypeError(f'unsupported request field type {type(value).__name__}')


def workload_manifest(jobs, input_digest, *, model_spec, unit_name):
    """Hash complete supported requests and their declared model/work metadata."""
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError('jobs must be nonempty with unique IDs')
    if not isinstance(model_spec, dict) or not model_spec or type(unit_name) is not str or not unit_name:
        raise ValueError('a model specification and work unit are required')
    items = []
    for job in jobs:
        if not all(type(getattr(job, field)) is str and getattr(job, field)
                   for field in ('job_id', 'team_id', 'workload_class', 'capture_policy', 'input_digest')):
            raise ValueError('job metadata must be nonempty strings')
        if type(job.units) is not int or job.units <= 0 or type(job.seed) is not int:
            raise ValueError('units must be positive integers and seed an integer')
        actual = input_digest(job.request)
        if not isinstance(actual, str) or actual != job.input_digest:
            raise ValueError(f'input digest changed for {job.job_id}')
        request_seed = getattr(job.request, 'seed', job.seed)
        request_id = getattr(job.request, 'request_id', job.job_id)
        if request_seed != job.seed or request_id != job.job_id:
            raise ValueError(f'request metadata differs for {job.job_id}')
        item = {key: getattr(job, key) for key in
                ('job_id', 'team_id', 'workload_class', 'units', 'seed', 'input_digest', 'capture_policy')}
        canonical_request = json.dumps(_canonical(job.request), sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode('utf-8')
        item['request_sha256'] = hashlib.sha256(canonical_request).hexdigest()
        items.append(item)
    manifest = {'model_spec': _canonical(model_spec), 'unit_name': unit_name, 'jobs': items}
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return manifest, hashlib.sha256(encoded).hexdigest()


def _describe_error(exc):
    try:
        return f'{type(exc).__name__}: {exc}'
    except Exception:
        return f'{type(exc).__name__}: <message unavailable>'


def _one_attempt(broker, job, *, expected_request_sha256, output_digest, rejection_errors, wait_timeout, cancel_timeout, on_attempt, on_accept, should_stop):
    row = {'job_id': job.job_id, 'team_id': job.team_id, 'workload_class': job.workload_class,
           'units': job.units, 'attempt_ns': time.perf_counter_ns(), 'submitted_ns': None,
           'ready_ns': None, 'started_ns': None, 'finished_ns': None, 'returned_ns': None,
           'state': None, 'output_digest': None, 'error': None, 'error_type': None,
           'broker_id': getattr(broker, 'broker_id', None),
           'timed_out': False, 'cancel_requested': False}
    on_attempt(job.job_id, row)
    try:
        if canonical_digest(job.request) != expected_request_sha256:
            raise ValueError('request manifest changed before submission')
    except Exception as exc:
        row.update(state='submit_error', error=_describe_error(exc), error_type=type(exc).__name__,
                   returned_ns=time.perf_counter_ns())
        return row
    if should_stop():
        row.update(state='submit_error', error='Window stopped before submission', error_type='WindowStopped',
                   returned_ns=time.perf_counter_ns())
        return row
    try:
        handle = broker.submit(job.team_id, job.request)
    except rejection_errors as exc:
        row.update(state='rejected', error=_describe_error(exc), error_type=type(exc).__name__,
                   returned_ns=time.perf_counter_ns())
        return row
    except Exception as exc:
        row.update(state='submit_error', error=_describe_error(exc), error_type=type(exc).__name__,
                   returned_ns=time.perf_counter_ns())
        return row
    row['broker_job_id'] = getattr(handle, 'job_id', None)
    on_accept(job.job_id, handle, row)
    try:
        try:
            outcome = handle.wait(wait_timeout)
        except TimeoutError:
            row['timed_out'] = True
            row['cancel_requested'] = bool(handle.cancel())
            outcome = handle.wait(cancel_timeout)
    except Exception as exc:
        cancellation_error = None
        if not row['cancel_requested']:
            try:
                row['cancel_requested'] = bool(handle.cancel())
            except Exception as cancel_exc:
                cancellation_error = _describe_error(cancel_exc)
        message = _describe_error(exc)
        if cancellation_error:
            message += f'; cancellation also failed: {cancellation_error}'
        row.update(state='unresolved', submitted_ns=handle.submitted_ns,
                   ready_ns=getattr(handle, 'ready_ns', None),
                   returned_ns=time.perf_counter_ns(), error_type=type(exc).__name__, error=message)
        return row
    state = getattr(outcome.state, 'value', outcome.state)
    identity_matches = (getattr(outcome, 'broker_id', row['broker_id']) == row['broker_id']
                        and getattr(outcome, 'job_id', row['broker_job_id']) == row['broker_job_id']
                        and getattr(outcome, 'team_id', job.team_id) == job.team_id
                        and getattr(outcome, 'request_id', job.job_id) == job.job_id)
    if not identity_matches:
        row.update(state='unresolved', submitted_ns=handle.submitted_ns,
                   ready_ns=getattr(handle, 'ready_ns', None),
                   returned_ns=time.perf_counter_ns(), error_type='BrokerIdentityMismatch',
                   error='BrokerIdentityMismatch: outcome does not match the submitted job')
        return row
    row.update(state=state, submitted_ns=outcome.submitted_ns,
               ready_ns=getattr(outcome, 'ready_ns', None),
               started_ns=outcome.started_ns, finished_ns=outcome.finished_ns,
               error_type=getattr(outcome, 'error_type', None))
    if state == 'succeeded':
        try:
            result_digest = output_digest(outcome.result)
            if not isinstance(result_digest, str) or not result_digest:
                raise ValueError('output digest must be a nonempty string')
            row['output_digest'] = result_digest
        except Exception as exc:
            row.update(state='invalid_output', error=_describe_error(exc), error_type=type(exc).__name__)
    elif state == 'failed':
        row['error'] = f"{outcome.error_type}: {outcome.error_message}"
    row['returned_ns'] = time.perf_counter_ns()
    return row


def run_window(broker, jobs, *, mode, output_digest, input_digest, rejection_errors,
               model_spec, unit_name, wait_timeout=120.0, cancel_timeout=5.0):
    """Run fixed work under one client or one concurrent closed-loop client/team.

    The broker must own a SynchronizedWorker for CUDA measurements. An
    unresolved wait makes the window incomplete and may leave device work in
    flight: drain/close the broker before starting any subsequent window.
    """
    if mode not in ('serial', 'shared'):
        raise ValueError('mode must be serial or shared')
    if not isinstance(rejection_errors, tuple) or not rejection_errors:
        raise ValueError('provide expected synchronous rejection exception types')
    if wait_timeout <= 0 or cancel_timeout <= 0:
        raise ValueError('wait and cancellation timeouts must be positive')
    manifest, manifest_hash = workload_manifest(jobs, input_digest, model_spec=model_spec, unit_name=unit_name)
    request_hashes = {item['job_id']: item['request_sha256'] for item in manifest['jobs']}
    groups = [list(jobs)] if mode == 'serial' else [
        [job for job in jobs if job.team_id == team] for team in sorted({j.team_id for j in jobs})]
    window_start = []
    barrier = threading.Barrier(len(groups) + 1, action=lambda: window_start.append(time.perf_counter_ns()))
    lock = threading.Lock()
    rows, fatal, active_handles, pending_attempts = [], [], {}, {}
    stop_event = threading.Event()

    def attempted(job_id, row):
        with lock:
            pending_attempts[job_id] = dict(row)

    def accepted(job_id, handle, row):
        with lock:
            pending_attempts.pop(job_id, None)
            active_handles[job_id] = (handle, dict(row))
        if stop_event.is_set():
            try:
                handle.cancel()
            except Exception:
                pass  # Quiescence remains unknown; the caller must recover.

    def halt_clients():
        stop_event.set()
        with lock:
            handles = [item[0] for item in active_handles.values()]
        for handle in handles:
            try:
                handle.cancel()
            except BaseException:
                pass

    def client(group):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return
        for job in group:
            if stop_event.is_set():
                return
            try:
                row = _one_attempt(broker, job, expected_request_sha256=request_hashes[job.job_id], output_digest=output_digest,
                                   rejection_errors=rejection_errors, wait_timeout=wait_timeout,
                                   cancel_timeout=cancel_timeout, on_attempt=attempted, on_accept=accepted,
                                   should_stop=stop_event.is_set)
                with lock:
                    pending_attempts.pop(job.job_id, None)
                    if row['state'] != 'unresolved':
                        active_handles.pop(job.job_id, None)
                    rows.append(row)
                if row['state'] in ('unresolved', 'submit_error'):
                    halt_clients()
                    return
            except Exception as exc:
                with lock:
                    fatal.append({'job_id': job.job_id, 'error': _describe_error(exc)})
                halt_clients()
                return

    threads = [threading.Thread(target=client, args=(group,), daemon=True) for group in groups]
    try:
        for thread in threads:
            thread.start()
        barrier.wait()
        start = window_start[0]
        for thread in threads:
            thread.join()
        end = time.perf_counter_ns()
    except BaseException as exc:
        stop_event.set()
        barrier.abort()
        with lock:
            completed = [dict(row) for row in rows]
            pending = list(active_handles.items())
            unknown_submissions = [dict(row, state='submission_status_unknown')
                                   for row in pending_attempts.values()]
            errors = list(fatal)
        active = []
        for job_id, (handle, row) in pending:
            active.append({'job_id': job_id, 'team_id': row['team_id'],
                           'attempt_ns': row['attempt_ns'], 'submitted_ns': getattr(handle, 'submitted_ns', None),
                           'broker_job_id': getattr(handle, 'job_id', None), 'broker_id': getattr(broker, 'broker_id', None),
                           'state': 'unknown_at_interruption'})
            try:
                handle.cancel()
            except BaseException:
                pass  # A cancellation request is best-effort, never proof of drain.
        try:
            exc.throughput_partial = {'mode': mode, 'start_ns': window_start[0] if window_start else None,
                                      'interrupted_ns': time.perf_counter_ns(), 'samples': completed,
                                      'active_jobs': active, 'submissions_unknown': unknown_submissions, 'fatal_errors': errors}
        except Exception:
            pass
        raise
    rows.sort(key=lambda row: (row['attempt_ns'], row['job_id']))
    attempted = {row['job_id'] for row in rows}
    missing = [job.job_id for job in jobs if job.job_id not in attempted]
    incomplete = bool(fatal or missing or any(row['state'] in ('unresolved', 'submit_error') for row in rows))
    return {'mode': mode, 'status': 'incomplete' if incomplete else 'complete',
            'manifest': manifest, 'manifest_sha256': manifest_hash, 'start_ns': start,
            'end_ns': end, 'samples': rows, 'unattempted_job_ids': missing, 'fatal_errors': fatal,
            'summary': summarize_window(rows, start_ns=start, end_ns=end) if rows else None}


def _valid_window_inventory(window):
    """Reject malformed windows or jobs omitted from the frozen manifest."""
    try:
        manifest = window['manifest']
        jobs = manifest['jobs']
        if type(jobs) is not list or not jobs:
            return False
        encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        if hashlib.sha256(encoded).hexdigest() != window['manifest_sha256']:
            return False
        by_id = {job['job_id']: job for job in jobs}
        rows = window['samples']
        if (len(by_id) != len(jobs) or len(rows) != len(jobs)
                or {row['job_id'] for row in rows} != set(by_id)
                or window.get('unattempted_job_ids') != [] or window.get('fatal_errors') != []):
            return False
        for row in rows:
            job = by_id[row['job_id']]
            if any(row.get(key) != job[key] for key in ('team_id', 'workload_class', 'units')):
                return False
        summarize_window(rows, start_ns=window['start_ns'], end_ns=window['end_ns'])
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def compare_windows(first, second):
    """Compare immutable manifests and exact output hashes; no speedup claim."""
    same_manifest = first['manifest_sha256'] == second['manifest_sha256']
    left = {row['job_id']: row['output_digest'] for row in first['samples'] if row['state'] == 'succeeded'}
    right = {row['job_id']: row['output_digest'] for row in second['samples'] if row['state'] == 'succeeded'}
    common = left.keys() & right.keys()
    matching = all(left[key] == right[key] for key in common) if common else None
    valid_inventory = _valid_window_inventory(first) and _valid_window_inventory(second)
    all_success = (len(left) == len(first['samples']) and len(right) == len(second['samples'])
                   and left.keys() == right.keys() and bool(left) and valid_inventory)
    valid_pair = ({first['mode'], second['mode']} == {'serial', 'shared'}
                  and (first['end_ns'] <= second['start_ns'] or second['end_ns'] <= first['start_ns']))
    valid = (valid_pair and first['status'] == second['status'] == 'complete' and same_manifest
             and all_success and matching is True)
    return {'valid_mode_pair': valid_pair, 'same_manifest': same_manifest, 'compared_successes': len(common),
            'matching_common_outputs': matching, 'all_jobs_succeeded_in_both': all_success,
            'valid_paired_comparison': valid}


def canonical_digest(value):
    """SHA-256 over normalized metadata and bit-exact CPU tensor bytes.

    Values are never written to evidence files. Unsupported objects and GPU
    tensors fail closed instead of being silently omitted from the fingerprint.
    """
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


class CudaMemoryTelemetry:
    """Boundary snapshots and PyTorch process allocator high-water marks.

    Create only while the caller holds a measurement reservation and the
    resident broker is drained. These snapshots are not a device-wide peak.
    """
    def __init__(self, device):
        import torch
        self._torch = torch
        self._device = torch.device(device)
        if self._device.type != 'cuda':
            raise ValueError('CUDA telemetry requires a CUDA device')

    def _snapshot(self):
        cuda, device = self._torch.cuda, self._device
        free, total = cuda.mem_get_info(device)
        return {'device_free_bytes': int(free), 'device_total_bytes': int(total),
                'process_allocated_bytes': int(cuda.memory_allocated(device)),
                'process_reserved_bytes': int(cuda.memory_reserved(device))}

    def before(self):
        cuda = self._torch.cuda
        cuda.synchronize(self._device)
        snapshot = self._snapshot()
        cuda.reset_peak_memory_stats(self._device)
        return snapshot

    def after(self):
        cuda = self._torch.cuda
        cuda.synchronize(self._device)
        snapshot = self._snapshot()
        snapshot.update(process_peak_allocated_bytes=int(cuda.max_memory_allocated(self._device)),
                        process_peak_reserved_bytes=int(cuda.max_memory_reserved(self._device)))
        return snapshot


def _atomic_write_json(path, value):
    # Hard-link publication is atomic and fails if the destination exists;
    # the temporary file resides on the same filesystem as its destination.
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    published = False
    original_error = None
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        published = True
    except BaseException as exc:
        original_error = exc
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if original_error is not None:
                raise original_error from cleanup_error
            if not published:
                raise
            # The terminal link is authoritative; a stale temporary link does
            # not turn a committed result into a conflicting failure record.


def _evidence_hashes(directory):
    hashes = {}
    for name in ('manifest.json', 'warmup_samples.jsonl', 'raw_samples.jsonl', 'partial_windows.jsonl', 'interrupted_windows.jsonl'):
        path = directory / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            hashes[name] = digest.hexdigest()
    return hashes


def _append_jsonl(path, value):
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()


def run_protocol(output_dir, broker, jobs, *, input_digest, output_digest, rejection_errors,
                 model_spec, unit_name, metadata, telemetry, repetitions=3, warmup_rounds=1,
                 protocol_seed=0, wait_timeout=120.0, cancel_timeout=5.0):
    """Run version-1 paired load windows and write immutable local evidence.

    The supplied broker is shared by all windows and must already own a single
    correctly synchronized resident model. Measurements abort on unresolved
    work; the caller must drain/close the broker before reusing the device.
    """
    if type(repetitions) is not int or repetitions < 1 or type(warmup_rounds) is not int or warmup_rounds < 1:
        raise ValueError('positive integer repetitions and per-class warmups are required')
    if type(protocol_seed) is not int or not isinstance(metadata, dict):
        raise ValueError('an integer protocol seed and metadata mapping are required')
    if not metadata.get('external_gpu_activity') or not metadata.get('source_commit'):
        raise ValueError('record source_commit and external_gpu_activity')
    if telemetry is None:
        raise ValueError('explicit device memory telemetry is required')
    manifest, manifest_hash = workload_manifest(jobs, input_digest, model_spec=model_spec, unit_name=unit_name)
    # Every exact request is warmed, including shapes and capture options that
    # callers might otherwise group under the same free-form workload class.
    warmup_jobs = list(jobs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    clock = time.get_clock_info('perf_counter')
    admission_status = metadata.get('admission_status')
    evidence_label = {'admission_status': admission_status} if admission_status is not None else {}
    result = {'protocol': 'shared-model-throughput-v1', 'status': 'running',
              'admission_status': admission_status,
              'benchmark_session_id': uuid.uuid4().hex, 'broker_id': getattr(broker, 'broker_id', None),
              'metadata': metadata, 'model_spec': model_spec, 'unit_name': unit_name,
              'manifest': manifest, 'manifest_sha256': manifest_hash,
              'config': {'protocol_seed': protocol_seed, 'repetitions': repetitions,
                         'warmup_rounds_per_job_and_mode': warmup_rounds,
                         'shared_clients': len({job.team_id for job in jobs}),
                         'max_outstanding_per_client': 1, 'wait_timeout_s': wait_timeout,
                         'cancel_timeout_s': cancel_timeout},
              'clock': {'name': 'perf_counter_ns', 'implementation': clock.implementation,
                        'resolution_seconds': clock.resolution, 'monotonic': clock.monotonic,
                        'adjustable': clock.adjustable},
              'environment': {'python': platform.python_version(), 'platform': platform.platform()},
              'benchmark_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'windows': [], 'comparisons': []}
    kwargs = {'input_digest': input_digest, 'output_digest': output_digest,
              'rejection_errors': rejection_errors, 'model_spec': model_spec, 'unit_name': unit_name,
              'wait_timeout': wait_timeout, 'cancel_timeout': cancel_timeout}
    def verify_frozen():
        try:
            _, actual_hash = workload_manifest(jobs, input_digest, model_spec=model_spec, unit_name=unit_name)
        except Exception as exc:
            raise RuntimeError('workload manifest changed') from exc
        if actual_hash != manifest_hash:
            raise RuntimeError('workload manifest changed')

    active_context = {'phase': 'setup', 'mode': None, 'repetition': None, 'warmup_round': None}
    try:
        _atomic_write_json(output_dir / 'manifest.json', result)
        rng = random.Random(protocol_seed)
        for repetition in range(repetitions):
            modes = ['serial', 'shared']
            rng.shuffle(modes)
            pair = {}
            for mode in modes:
                for warmup_round in range(warmup_rounds):
                    active_context.update(phase='warmup', mode=mode, repetition=repetition, warmup_round=warmup_round)
                    verify_frozen()
                    warmup = run_window(broker, warmup_jobs, mode=mode, **kwargs)
                    for sample in warmup['samples']:
                        _append_jsonl(output_dir / 'warmup_samples.jsonl',
                                      {'benchmark_session_id': result['benchmark_session_id'], 'repetition': repetition, 'mode': mode,
                                       'warmup_round': warmup_round, **evidence_label, **sample})
                    if warmup['manifest_sha256'] != manifest_hash:
                        raise RuntimeError('workload manifest changed during warmup')
                    verify_frozen()
                    if warmup['status'] != 'complete' or warmup['summary']['counts']['succeeded'] != len(warmup_jobs):
                        raise RuntimeError('warmup failed or did not drain; stop and recover broker')
                active_context.update(phase='measurement', mode=mode, repetition=repetition, warmup_round=None)
                verify_frozen()
                before = telemetry.before()
                window = run_window(broker, jobs, mode=mode, **kwargs)
                window['repetition'] = repetition
                window.update(evidence_label)
                for sample in window['samples']:
                    _append_jsonl(output_dir / 'raw_samples.jsonl',
                                  {'benchmark_session_id': result['benchmark_session_id'], 'repetition': repetition,
                                   'mode': mode, **evidence_label, **sample})
                if window['status'] != 'complete':
                    window['memory'] = {'before': before, 'after': None}
                    _append_jsonl(output_dir / 'partial_windows.jsonl', window)
                    raise RuntimeError('measurement did not drain; stop and recover broker')
                window['memory'] = {'before': before, 'after': telemetry.after()}
                _append_jsonl(output_dir / 'partial_windows.jsonl', window)
                result['windows'].append(window)
                if window['manifest_sha256'] != manifest_hash:
                    raise RuntimeError('workload manifest changed during measurement')
                verify_frozen()
                pair[mode] = window
            comparison = compare_windows(pair['serial'], pair['shared'])
            result['comparisons'].append({'repetition': repetition, **comparison})
        result['evidence_sha256'] = _evidence_hashes(output_dir)
        result['status'] = ('complete' if all(c['valid_paired_comparison'] for c in result['comparisons'])
                            else 'invalid_comparison')
        _atomic_write_json(output_dir / 'results.json', result)
        return result
    except BaseException as exc:
        # Publication may have completed just before an interruption. A
        # terminal results record already present in this exclusive directory
        # wins; never create a contradictory failure terminal beside it.
        if (output_dir / 'results.json').exists():
            raise
        try:
            partial = getattr(exc, 'throughput_partial', None)
            if partial is not None:
                _append_jsonl(output_dir / 'interrupted_windows.jsonl',
                              {'benchmark_session_id': result['benchmark_session_id'], **active_context,
                               **evidence_label, **partial})
                path = output_dir / ('warmup_samples.jsonl' if active_context['phase'] == 'warmup' else 'raw_samples.jsonl')
                for sample in partial['samples']:
                    _append_jsonl(path, {'benchmark_session_id': result['benchmark_session_id'],
                                        **active_context, **evidence_label, 'partial': True, **sample})
            try:
                evidence_hashes, hash_error = _evidence_hashes(output_dir), None
            except Exception as digest_error:
                evidence_hashes, hash_error = None, _describe_error(digest_error)
            _atomic_write_json(output_dir / 'failure.json', {'status': 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                'admission_status': admission_status,
                'type': type(exc).__name__, 'message': str(exc), 'completed_windows': len(result['windows']),
                'work_may_still_be_running': True, 'evidence_sha256': evidence_hashes,
                'evidence_hash_error': hash_error})
        except Exception as recording_error:
            raise exc from recording_error
        raise


# Fixed, public short prompts: each team receives the same workload mix.
DISTILGPT2_PROMPTS = (
    'The small bird landed on',
    'A bright light appeared in',
    'The train arrived at the',
    'A quiet garden grew beside',
)


def make_distilgpt2_workload(bundle, *, prompts=DISTILGPT2_PROMPTS):
    """Build two equal closed-loop team streams and count actual input positions."""
    from .resident_worker import ForwardRequest
    if not prompts or any(type(prompt) is not str or not prompt for prompt in prompts):
        raise ValueError('nonempty public prompts are required')
    jobs = []
    for team in ('team-a', 'team-b'):
        for index, prompt in enumerate(prompts):
            request = ForwardRequest(f'{team}-{index}', bundle.encode(prompt), seed=1000 + index)
            count = request.inputs['input_ids'].numel()
            if not count:
                raise ValueError('tokenized input is empty')
            jobs.append(WorkItem(request.request_id, team, f'{count}-positions', count,
                                 request.seed, canonical_digest(request), 'none', request))
    return jobs


def digest_forward_result(result, *, vocab_size):
    """Validate the concrete worker result before counting a completed job."""
    import torch
    from .resident_worker import ForwardResult
    if type(result) is not ForwardResult or type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError('expected a resident-worker result and positive vocabulary size')
    logits = result.logits
    if (type(logits) is not torch.Tensor or logits.device.type != 'cpu'
            or logits.dtype != torch.float32 or logits.ndim != 3
            or logits.shape[0] < 1 or logits.shape[1] < 1 or logits.shape[2] != vocab_size):
        raise ValueError('unexpected result logits shape, dtype or device')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite result logits')
    return canonical_digest(result)


class CpuMemoryTelemetry:
    """Explicit marker for CPU smoke runs; no GPU-memory figure is invented."""
    def before(self):
        return {'device': 'cpu', 'gpu_memory_measured': False}

    def after(self):
        return {'device': 'cpu', 'gpu_memory_measured': False}


_SOURCE_FILES = ('looking_glass/__init__.py', 'looking_glass/throughput_benchmark.py', 'looking_glass/pretrained.py',
                 'looking_glass/resident_worker.py', 'looking_glass/broker.py',
                 'looking_glass/fair_queue.py', 'looking_glass/memory_budget.py',
                 'looking_glass/budgeted_worker.py', 'looking_glass/device_lease.py',
                 'looking_glass/device_identity.py', 'looking_glass/supervisor.py',
                 'docs/THROUGHPUT_PROTOCOL_V1.md', 'scripts/throughput_bootstrap.py')


def _source_metadata():
    import subprocess
    root = Path(__file__).resolve().parent.parent
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in _SOURCE_FILES if (root / name).is_file()}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
                                capture_output=True, text=True, timeout=5).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain=v1', '--untracked-files=all'],
                                cwd=root, check=True, capture_output=True, text=True, timeout=5).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = 'unavailable', None
    return {'source_commit': commit, 'source_dirty': dirty, 'source_sha256': files}


def _nvidia_inventory():
    import subprocess
    try:
        output = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,memory.free',
             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=5)
        return {'query': 'index,uuid,name,driver_version,memory.total,memory.free',
                'units': 'MiB', 'rows': output.stdout.strip().splitlines()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'error': _describe_error(exc)}


def configure_cuda_determinism(torch_module, environment):
    """Set cuBLAS determinism before any CUDA context is initialized."""
    if torch_module.cuda.is_initialized():
        raise RuntimeError('CUDA is already initialized; launch a fresh benchmark process')
    setting = environment.get('CUBLAS_WORKSPACE_CONFIG')
    if setting is None:
        setting = ':4096:8'
        environment['CUBLAS_WORKSPACE_CONFIG'] = setting
    if setting not in (':4096:8', ':16:8'):
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8')
    return setting


_FORBIDDEN_CUDA_ENV = (
    'CUDA_LAUNCH_BLOCKING', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
    'PYTORCH_NO_CUDA_MEMORY_CACHING', 'NVIDIA_TF32_OVERRIDE',
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'TORCH_CUDNN_V8_API_DISABLED',
    'TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT', 'TORCH_CUDNN_V8_API_DEBUG',
    'CUDNN_CONV_WSCAP_DBG', 'CUDNN_ERRATA_JSON_FILE',
)
_TRACKED_CUDA_ENV = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_MODULE_LOADING',
                     'CUDA_CACHE_DISABLE', 'CUDA_CACHE_MAXSIZE', 'CUDA_DEVICE_MAX_CONNECTIONS',
                     'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS')
_CUDA_ENV_PREFIXES = ('CUDA_', 'CUBLAS_', 'CUDNN_', 'NVIDIA_', 'PYTORCH_', 'TORCH_', 'NCCL_',
                      'OMP_', 'MKL_', 'KMP_', 'OPENBLAS_', 'BLIS_', 'NUMEXPR_', 'VECLIB_',
                      'ONEDNN_', 'DNNL_')
_CHILD_OS_ENV = ('PATH', 'PATHEXT', 'SYSTEMROOT', 'WINDIR', 'SYSTEMDRIVE', 'COMSPEC',
                 'TEMP', 'TMP', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'HOME',
                 'LOCALAPPDATA', 'APPDATA')


def _capture_cuda_environment(environment):
    return {name: environment[name] for name in _TRACKED_CUDA_ENV if name in environment}


def _configure_cuda_environment(environment):
    """Freeze a simple native-allocator, nonblocking, one-thread baseline."""
    present = [name for name in _FORBIDDEN_CUDA_ENV if name in environment]
    if present:
        raise ValueError('unsupported CUDA or PyTorch environment overrides: ' + ', '.join(present))
    unknown = [name for name in environment if isinstance(name, str)
               and name.upper().startswith(_CUDA_ENV_PREFIXES)
               and (name != name.upper() or name not in _TRACKED_CUDA_ENV)]
    if unknown:
        raise ValueError('unrecognized runtime environment keys: ' + ', '.join(sorted(unknown)))
    for name, value in (('CUDA_MODULE_LOADING', 'LAZY'), ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),
                        ('OPENBLAS_NUM_THREADS', '1'), ('NUMEXPR_NUM_THREADS', '1'),
                        ('BLIS_NUM_THREADS', '1'), ('VECLIB_MAXIMUM_THREADS', '1')):
        if name in environment and environment[name] != value:
            raise ValueError(f'{name} must be {value} for this protocol')
        environment[name] = value
    if 'CUDA_VISIBLE_DEVICES' in environment:
        from .device_identity import normalize_gpu_uuid
        normalize_gpu_uuid(environment['CUDA_VISIBLE_DEVICES'])
    if 'CUDA_DEVICE_ORDER' in environment and environment['CUDA_DEVICE_ORDER'] not in ('PCI_BUS_ID', 'FASTEST_FIRST'):
        raise ValueError('CUDA_DEVICE_ORDER is unsupported')
    for name in ('CUDA_CACHE_DISABLE', 'CUDA_CACHE_MAXSIZE', 'CUDA_DEVICE_MAX_CONNECTIONS'):
        if name in environment:
            value = environment[name]
            if type(value) is not str or re.fullmatch(r'[0-9]{1,12}', value) is None:
                raise ValueError(f'{name} must be a bounded nonnegative integer')
            number = int(value)
            if ((name == 'CUDA_CACHE_DISABLE' and number not in (0, 1))
                    or (name == 'CUDA_DEVICE_MAX_CONNECTIONS' and not 1 <= number <= 32)):
                raise ValueError(f'{name} is outside the supported range')
    workspace = environment.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in (':4096:8', ':16:8'):
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8')
    environment['CUBLAS_WORKSPACE_CONFIG'] = workspace
    return _capture_cuda_environment(environment)


def _supervised_child_environment(runtime_environment, environment):
    """Pass only selected OS context and the validated measurement settings."""
    if (type(runtime_environment) is not dict
            or _configure_cuda_environment(dict(runtime_environment)) != runtime_environment):
        raise ValueError('the child runtime environment is not a validated policy')
    child = {key: environment[key] for key in _CHILD_OS_ENV if key in environment}
    child.update(runtime_environment)
    return child


def _claim_supervised_owner():
    """Check supervisor identity while holding its exact cross-process lease."""
    from .device_lease import DeviceLease
    from .supervisor import OwnerSession
    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
    if type(session) is not OwnerSession:
        raise RuntimeError('a real supervised owner session is required')
    OwnerSession.check(session, lease)
    if session.identity is None:
        raise RuntimeError('supervised physical GPU identity is required')
    return session, lease


def _check_owner_before_cuda(session, lease):
    """Recheck the actual lease and marker immediately before CUDA queries."""
    from .supervisor import OwnerSession
    if type(session) is not OwnerSession:
        raise RuntimeError('a real supervised owner session is required')
    OwnerSession.check(session, lease)
    if session.identity is None:
        raise RuntimeError('supervised physical GPU identity is required')


def _initialize_verified_cuda(session, lease, torch_module, environment):
    """Bind CUDA ordinal zero to the supervised physical GPU before model work."""
    from .device_identity import normalize_gpu_uuid
    _check_owner_before_cuda(session, lease)
    configure_cuda_determinism(torch_module, environment)
    if not torch_module.cuda.is_available():
        raise RuntimeError('the supervised CUDA device is unavailable')
    properties = torch_module.cuda.get_device_properties(0)
    runtime_uuid = properties.uuid
    cu_uuid_type = getattr(getattr(torch_module, '_C', None), '_CUuuid', None)
    if cu_uuid_type is not None and type(runtime_uuid) is cu_uuid_type:
        rendered = str(runtime_uuid)
        try:
            parsed = uuid.UUID(rendered)
        except (ValueError, AttributeError) as exc:
            raise ValueError('unsupported CUDA runtime UUID format') from exc
        if rendered != str(parsed):
            raise ValueError('unsupported CUDA runtime UUID format')
        runtime_uuid = 'GPU-' + rendered
    elif type(runtime_uuid) is not str:
        raise ValueError('unsupported CUDA runtime UUID type')
    runtime_uuid = normalize_gpu_uuid(runtime_uuid)
    session.verify_runtime_binding(lease, runtime_uuid=runtime_uuid,
        device_count=torch_module.cuda.device_count(), cuda_device_index=0)
    return runtime_uuid


def _attest_owner(session, device_lease, *, runtime_verified, budget, synchronize):
    """Attest only after verified identity, drained work and released claims."""
    if not runtime_verified:
        raise RuntimeError('physical CUDA identity was not verified')
    if budget is not None and budget.active_leases():
        raise RuntimeError('memory leases remain active after cleanup')
    synchronize()
    session.attest_clean(device_lease)


def _clear_cuda_allocator(torch_module, device):
    """Require zero live or cached PyTorch tensor bytes in the owner process."""
    import gc
    gc.collect()
    torch_module.cuda.synchronize(device)
    torch_module.cuda.empty_cache()
    allocated = torch_module.cuda.memory_allocated(device)
    reserved = torch_module.cuda.memory_reserved(device)
    if type(allocated) is not int or type(reserved) is not int or allocated != 0 or reserved != 0:
        raise RuntimeError(f'live PyTorch CUDA allocations remain after teardown: allocated={allocated}, reserved={reserved}')
    return {'allocated_bytes': allocated, 'reserved_bytes': reserved}


def _configure_cuda_allocator(torch_module, device):
    """Apply and verify the fixed native PyTorch caching-allocator cap."""
    backend = torch_module.cuda.get_allocator_backend()
    if backend != 'native':
        raise RuntimeError('the CUDA allocator backend must be native')
    torch_module.cuda.set_per_process_memory_fraction(0.75, device)
    fraction = float(torch_module.cuda.get_per_process_memory_fraction(device))
    if not isfinite(fraction) or abs(fraction - 0.75) > 1e-6:
        raise RuntimeError('the CUDA allocator fraction did not match the requested cap')
    return backend, fraction


def _fixed_workload_profiles(jobs, *, model_fingerprint, device,
                             output_limit_bytes, estimated_incremental_bytes):
    """Pin service-owned exact signatures to one conservative estimate."""
    from .budgeted_worker import WorkloadProfile
    profiles = []
    signatures = set()
    for job in jobs:
        inputs = job.request.inputs
        shape = tuple(inputs['input_ids'].shape)
        dtypes = tuple(sorted((name, str(tensor.dtype)) for name, tensor in inputs.items()))
        signature = (shape, dtypes)
        if signature in signatures:
            continue
        signatures.add(signature)
        profiles.append(WorkloadProfile(
            profile_id='fixed-v1-' + canonical_digest({'shape': shape, 'dtypes': dtypes})[:16],
            model_id=model_fingerprint, device=device, shape=shape, input_dtypes=dtypes,
            output_limit_bytes=output_limit_bytes,
            estimated_incremental_bytes=estimated_incremental_bytes))
    if not profiles:
        raise ValueError('at least one fixed workload profile is required')
    return tuple(profiles)


def _model_fingerprint(model_spec, environment):
    """Bind the verified assets and execution settings used by admission."""
    keys = ('torch', 'cuda_runtime', 'device', 'deterministic_algorithms',
            'cublas_workspace_config', 'matmul_allow_tf32', 'cudnn_benchmark',
            'cudnn_deterministic', 'cudnn_allow_tf32', 'runtime_gpu_uuid',
            'device_name', 'compute_capability', 'bytecode_isolation', 'python_no_site',
            'python_ignores_environment', 'python_isolated', 'python_hash_randomization',
            'cuda_environment', 'allocator_backend', 'allocator_fraction',
            'torch_threads', 'torch_interop_threads')
    return canonical_digest({'model': model_spec,
                             'runtime': {key: environment.get(key) for key in keys}})


def _require_fresh_bytecode_prefix(sys_module, *, expected=None):
    """Require an empty startup bytecode prefix and disabled bytecode writes."""
    import stat
    prefix = getattr(sys_module, 'pycache_prefix', None)
    option = getattr(sys_module, '_xoptions', {}).get('pycache_prefix')
    if (type(prefix) is not str or not prefix or type(option) is not str or option != prefix
            or getattr(sys_module.flags, 'dont_write_bytecode', 0) != 1
            or getattr(sys_module.flags, 'no_site', 0) != 1
            or getattr(sys_module.flags, 'ignore_environment', 0) != 1
            or getattr(sys_module.flags, 'isolated', 0) != 1
            or sys_module.dont_write_bytecode is not True
            or os.environ.get('LOOKING_GLASS_THROUGHPUT_BOOTSTRAP_PID') != str(os.getpid())):
        raise RuntimeError('launch the source bootstrap with -I -S -B and an absolute empty -X pycache_prefix')
    path = Path(prefix)
    if not path.is_absolute():
        raise RuntimeError('the bytecode prefix must be absolute')
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or path.is_symlink()
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise RuntimeError('the bytecode prefix must be a plain directory')
    if expected is not None and path.resolve(strict=True) != Path(expected).resolve(strict=True):
        raise RuntimeError('the bytecode prefix differs from the supervised prefix')
    if next(path.iterdir(), None) is not None:
        raise RuntimeError('the bytecode prefix must be empty')
    try:
        site_paths = json.loads(os.environ['LOOKING_GLASS_THROUGHPUT_SITE_PATHS'])
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError('bootstrap dependency directories are missing') from exc
    if (type(site_paths) is not list or not site_paths
            or any(type(item) is not str or not Path(item).is_absolute() or not Path(item).is_dir()
                   for item in site_paths)):
        raise RuntimeError('bootstrap dependency directories are invalid')
    return path.resolve(strict=True)


def _checked_bytecode_isolation(*, required=False):
    """Record isolation only after validating the source-bootstrap startup."""
    if required or 'LOOKING_GLASS_THROUGHPUT_BOOTSTRAP_PID' in os.environ:
        import sys
        _require_fresh_bytecode_prefix(sys)
        return 'empty_prefix_no_writes'
    return None


def _normalized_model_metadata(metadata):
    """Convert string-subclass library versions into stable evidence values."""
    result = dict(metadata)
    for key in ('torch_version', 'transformers_version'):
        if isinstance(result.get(key), str):
            result[key] = str(result[key])
    return result


def run_distilgpt2_benchmark(output_dir, *, cache_dir, device='cpu', gpu_activity='unknown',
                              repetitions=3, protocol_seed=0, quick=False,
                              development_unprofiled=False):
    """Run the pinned offline DistilGPT2 workload through one resident broker.

    A CUDA caller must be a supervised child with the physical-device lease.
    Memory admission estimates are not hard allocation limits. The quick run
    is a smoke check.
    """
    device_name = str(device)
    if device_name == 'cuda':
        device_name = 'cuda:0'
    if device_name not in ('cpu', 'cuda:0'):
        raise ValueError('device must be cpu or supervised cuda:0')
    if type(development_unprofiled) is not bool:
        raise ValueError('development_unprofiled must be a boolean')
    if device_name == 'cuda:0' and not development_unprofiled:
        raise RuntimeError('unprofiled GPU workload is disabled; a measured admission profile is required')
    if device_name == 'cuda:0' and development_unprofiled and quick is not True:
        raise ValueError('unprofiled GPU development is limited to the quick workload')
    if device_name == 'cpu' and development_unprofiled:
        raise ValueError('development_unprofiled applies only to GPU development runs')
    if type(quick) is not bool or type(gpu_activity) is not str or not gpu_activity.strip():
        raise ValueError('record observed external GPU activity and use a boolean quick flag')
    if type(repetitions) is not int or not 1 <= repetitions <= 10 or type(protocol_seed) is not int:
        raise ValueError('repetitions must be 1 through 10 and the protocol seed must be an integer')
    owner_session = device_lease = None
    runtime_environment = None
    bytecode_isolation = _checked_bytecode_isolation(required=device_name == 'cuda:0')
    if device_name == 'cuda:0':
        runtime_environment = _configure_cuda_environment(os.environ)
        owner_session, device_lease = _claim_supervised_owner()
    import torch
    from .broker import Broker, BrokerUnavailable
    from .budgeted_worker import BudgetedWorker, DeviceAdmission
    from .fair_queue import QueueFull
    from .memory_budget import GIB, MemoryBudget, cuda_memory_snapshot
    from .pretrained import ASSETS, MODEL_ID, REVISION, load_distilgpt2
    from .resident_worker import InvalidRequest, ResidentWorker

    resolved_device = torch.device(device_name)
    prompts = DISTILGPT2_PROMPTS[:2] if quick else DISTILGPT2_PROMPTS
    repetitions = 1 if quick else repetitions
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    broker = lease = bundle = worker = proxy = budgeted = budget = admission = None
    primary_error = None
    measurement_result = None
    measurement_started = False
    runtime_verified = False
    cuda_cleanup = None
    try:
        runtime_uuid = None
        allocator_backend = allocator_fraction = None
        if resolved_device.type == 'cuda':
            runtime_uuid = _initialize_verified_cuda(owner_session, device_lease, torch, os.environ)
            runtime_verified = True
            allocator_backend, allocator_fraction = _configure_cuda_allocator(torch, resolved_device)
        torch.set_num_threads(1)
        if resolved_device.type == 'cuda':
            torch.set_num_interop_threads(1)
        import transformers
        import numpy
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        source = _source_metadata()
        import sys
        environment = {'python': platform.python_version(), 'platform': platform.platform(),
                       'torch': str(torch.__version__), 'transformers': transformers.__version__,
                       'numpy': numpy.__version__, 'cuda_runtime': torch.version.cuda,
                       'device': str(resolved_device), 'torch_threads': torch.get_num_threads(),
                       'torch_interop_threads': torch.get_num_interop_threads(),
                       'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                       'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                       'cudnn_benchmark': torch.backends.cudnn.benchmark,
                       'cudnn_deterministic': torch.backends.cudnn.deterministic,
                       'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                       'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                       'nvidia_inventory': _nvidia_inventory() if resolved_device.type == 'cuda' else None,
                       'runtime_gpu_uuid': runtime_uuid,
                       'cuda_environment': runtime_environment,
                       'allocator_backend': allocator_backend, 'allocator_fraction': allocator_fraction,
                       'bytecode_isolation': bytecode_isolation,
                       'python_no_site': bool(sys.flags.no_site),
                       'python_ignores_environment': bool(sys.flags.ignore_environment),
                       'python_isolated': bool(sys.flags.isolated),
                       'python_hash_randomization': bool(sys.flags.hash_randomization)}
        if resolved_device.type == 'cuda':
            environment['device_name'] = torch.cuda.get_device_name(resolved_device)
            environment['compute_capability'] = list(torch.cuda.get_device_capability(resolved_device))
        setup = {'protocol': 'shared-model-throughput-v1', 'status': 'initializing',
                 'smoke_only': quick, 'source': source, 'environment': environment,
                 'admission_status': 'development_unprofiled' if resolved_device.type == 'cuda' else None,
                 'owner_run_id': owner_session.run_id if owner_session is not None else None,
                 'owner_pid': os.getpid() if owner_session is not None else None,
                 'external_gpu_activity': gpu_activity, 'model_id': MODEL_ID, 'revision': REVISION,
                 'expected_checkpoint_sha256': ASSETS['model.safetensors'][1],
                 'workload_replay': {'public_prompts': list(prompts), 'teams': ['team-a', 'team-b'],
                                     'seed_rule': '1000 + prompt index', 'batch': 1,
                                     'tokenizer_add_special_tokens': False, 'padding': False,
                                     'max_tokens': 128, 'capture_policy': 'none',
                                     'unit': 'input tensor positions, including any padding'},
                 'limits': {'max_batch_size': 1, 'max_sequence_length': 128,
                            'max_input_bytes': 1024 * 1024, 'max_output_bytes': 8 * 1024 * 1024,
                            'max_gpu_trace_bytes': 0, 'max_host_trace_bytes': 0,
                            'gpu_model_load_estimate_bytes': 2 * GIB if resolved_device.type == 'cuda' else None,
                            'gpu_per_forward_estimate_bytes': GIB if resolved_device.type == 'cuda' else None,
                            'gpu_estimates_status': 'provisional estimates; not measured hard limits'
                                if resolved_device.type == 'cuda' else None,
                            'broker_max_pending': 4, 'broker_max_pending_per_team': 1},
                 'protocol_seed': protocol_seed, 'repetitions': repetitions}
        _atomic_write_json(output_dir / 'setup.json', setup)
        bundle = load_distilgpt2(cache_dir=cache_dir, allow_download=False)
        if resolved_device.type == 'cuda' and _configure_cuda_environment(dict(os.environ)) != runtime_environment:
            raise RuntimeError('CUDA environment changed after startup')
        jobs = make_distilgpt2_workload(bundle, prompts=prompts)
        model_spec = _normalized_model_metadata(bundle.metadata)
        model_spec.update(device=str(resolved_device), dtype='float32', eval_mode=True,
                          capture_policy='none')
        model_fingerprint = _model_fingerprint(model_spec, environment)
        model_spec['service_model_fingerprint'] = model_fingerprint
        if resolved_device.type == 'cuda':
            budget = MemoryBudget(lambda: cuda_memory_snapshot(resolved_device),
                                  max_gpu_trace_bytes=0, max_host_trace_bytes=0)
            lease = budget.acquire(2 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
            bundle.model.to(resolved_device)
            torch.cuda.synchronize(resolved_device)
            synchronize = lambda: torch.cuda.synchronize(resolved_device)
            telemetry = CudaMemoryTelemetry(resolved_device)
        else:
            synchronize = lambda: None
            telemetry = CpuMemoryTelemetry()
        vocab_size = bundle.model.config.vocab_size
        worker = ResidentWorker(model_fingerprint if resolved_device.type == 'cuda' else MODEL_ID,
                                bundle.model, device=resolved_device,
                                max_batch_size=1, max_sequence_length=128,
                                max_input_bytes=1024 * 1024, max_output_bytes=8 * 1024 * 1024)
        bundle.model = None  # The worker is the only owner of the live model.
        if resolved_device.type == 'cuda':
            profiles = _fixed_workload_profiles(jobs, model_fingerprint=model_fingerprint,
                device=str(resolved_device), output_limit_bytes=8 * 1024 * 1024,
                estimated_incremental_bytes=GIB)
            admission = DeviceAdmission(budget, device=str(resolved_device),
                                        max_estimated_incremental_bytes=GIB)
            budgeted = BudgetedWorker(worker, admission, profiles)
            setup['limits']['profiles'] = [asdict(profile) for profile in profiles]
            _atomic_write_json(output_dir / 'admission_profiles.json', {
                'status': 'provisional_unmeasured_estimates', 'admission_status': setup['admission_status'],
                'profiles': setup['limits']['profiles']})
        proxy = SynchronizedWorker(budgeted if budgeted is not None else worker, synchronize)
        broker = Broker(proxy, max_pending=4, max_pending_per_team=1)
        metadata = {**source, 'external_gpu_activity': gpu_activity, 'environment': environment,
                    'workload_replay': setup['workload_replay'], 'smoke_only': quick,
                    'memory_policy': setup['limits'], 'admission_status': setup['admission_status']}
        measurement_started = True
        measurement_result = run_protocol(output_dir / 'measurements', broker, jobs,
                            input_digest=canonical_digest,
                            output_digest=lambda result: digest_forward_result(
                                result, vocab_size=vocab_size),
                            rejection_errors=(QueueFull, BrokerUnavailable, InvalidRequest),
                            model_spec=model_spec, unit_name='input_positions_including_padding',
                            metadata=metadata, telemetry=telemetry,
                            repetitions=repetitions, protocol_seed=protocol_seed)
        return measurement_result
    except BaseException as exc:
        primary_error = exc
        try:
            _atomic_write_json(output_dir / 'failure.json', {
                'status': 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                'admission_status': 'development_unprofiled' if device_name == 'cuda:0' else None,
                'type': type(exc).__name__, 'message': str(exc),
                'measurement_started': measurement_started,
                'nested_failure_recorded': (output_dir / 'measurements' / 'failure.json').exists(),
                'work_may_still_be_running': broker is not None})
        except Exception as recording_error:
            raise exc from recording_error
        raise
    finally:
        cleanup_error = None
        try:
            if broker is not None and not broker.close(wait=True, timeout=30):
                raise RuntimeError('broker did not stop; keep its process quarantined')
            if broker is None:
                if proxy is not None:
                    proxy.close()
                elif budgeted is not None:
                    budgeted.close()
                elif worker is not None:
                    worker.close()
            if proxy is not None and proxy.quarantined:
                raise RuntimeError('worker process is quarantined; do not reuse this device context')
            if admission is not None and admission.quarantined:
                raise RuntimeError('device admission is quarantined; do not reuse this process')
            if resolved_device.type == 'cuda':
                if not runtime_verified:
                    raise RuntimeError('physical CUDA identity was not verified; process remains quarantined')
                torch.cuda.synchronize(resolved_device)
                if bundle is not None and bundle.model is not None:
                    bundle.model.to('cpu')
                    bundle.model = None
                torch.cuda.synchronize(resolved_device)
                cuda_cleanup = _clear_cuda_allocator(torch, resolved_device)
                if lease is not None:
                    lease.release()
                _attest_owner(owner_session, device_lease, runtime_verified=runtime_verified,
                              budget=budget, synchronize=lambda: torch.cuda.synchronize(resolved_device))
        except BaseException as exc:
            cleanup_error = exc
        cleanup = ({'status': 'complete', 'broker_state': 'not_started' if broker is None else 'stopped',
                    'memory_lease_released': lease is None or lease.released,
                    'owner_attested_clean': owner_session is not None,
                    'cuda_allocator_after_cleanup': cuda_cleanup}
                   if cleanup_error is None else
                   {'status': 'failed', 'type': type(cleanup_error).__name__, 'message': str(cleanup_error),
                    'work_may_still_be_running': True})
        cleanup['admission_status'] = 'development_unprofiled' if device_name == 'cuda:0' else None
        publication_errors = []
        try:
            _atomic_write_json(output_dir / 'cleanup.json', cleanup)
        except Exception as exc:
            publication_errors.append(exc)
        measurement_status = measurement_result['status'] if measurement_result is not None else None
        if cleanup_error is not None:
            final_status = 'cleanup_failed'
        elif publication_errors:
            final_status = 'evidence_write_failed'
        elif primary_error is not None:
            final_status = 'cancelled' if isinstance(primary_error, KeyboardInterrupt) else 'failed'
        else:
            final_status = measurement_status
        try:
            _atomic_write_json(output_dir / 'run_status.json', {'status': final_status,
                'admission_status': 'development_unprofiled' if device_name == 'cuda:0' else None,
                'measurement_status': measurement_status, 'cleanup_status': cleanup['status'],
                'work_may_still_be_running': cleanup_error is not None})
        except Exception as exc:
            publication_errors.append(exc)
        if primary_error is not None:
            if hasattr(primary_error, 'add_note'):
                if cleanup_error is not None:
                    primary_error.add_note('Cleanup failed: ' + _describe_error(cleanup_error))
                for exc in publication_errors:
                    primary_error.add_note('Status publication failed: ' + _describe_error(exc))
        elif cleanup_error is not None:
            raise cleanup_error
        elif publication_errors:
            raise publication_errors[0]


def _pytorch_already_imported():
    import sys
    return 'torch' in sys.modules


def _preflight_distilgpt2(cache_dir, *, quick):
    """Verify cached assets and freeze the CPU workload before starting an owner."""
    if _pytorch_already_imported():
        raise RuntimeError('PyTorch is already imported; start a fresh CPU preflight process')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    was_set = 'CUDA_VISIBLE_DEVICES' in os.environ
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        import torch
        if torch.cuda.is_initialized() is not False:
            raise RuntimeError('PyTorch CUDA was initialized during CPU preflight')
        from .pretrained import load_distilgpt2
        bundle = load_distilgpt2(cache_dir=cache_dir, allow_download=False)
        prompts = DISTILGPT2_PROMPTS[:2] if quick else DISTILGPT2_PROMPTS
        jobs = make_distilgpt2_workload(bundle, prompts=prompts)
        metadata = _normalized_model_metadata(bundle.metadata)
        manifest, _ = workload_manifest(jobs, canonical_digest, model_spec=metadata,
                                        unit_name='input_positions_including_padding')
        if torch.cuda.is_initialized() is not False:
            raise RuntimeError('PyTorch CUDA was initialized during CPU preflight')
        return {'model_metadata': metadata, 'manifest_jobs': manifest['jobs'],
                'source': _source_metadata(), 'public_prompts': list(prompts),
                'pytorch_cuda_initialized': False}
    finally:
        if was_set:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)


def _verify_source_commit(source, *, root=None):
    """Compare each captured source digest with its actual recorded Git blob."""
    import subprocess
    root = Path(root or Path(__file__).resolve().parent.parent)
    for name in _SOURCE_FILES:
        try:
            proc = subprocess.run(['git', 'show', f"{source['source_commit']}:{name}"], cwd=root,
                                  check=True, capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f'cannot read committed source blob for {name}') from exc
        if hashlib.sha256(proc.stdout).hexdigest() != source['source_sha256'][name]:
            raise ValueError(f'working source differs from recorded commit: {name}')


def _validate_preflight(preflight, *, quick):
    """Require reconstructible source, pinned assets and the fixed job set."""
    from .pretrained import ASSETS, MODEL_ID, REVISION
    if preflight.get('pytorch_cuda_initialized') is not False:
        raise ValueError('CPU preflight did not verify uninitialized PyTorch CUDA state')
    cuda_environment = preflight.get('cuda_environment')
    if (type(cuda_environment) is not dict
            or any(type(key) is not str or type(value) is not str for key, value in cuda_environment.items())
            or _configure_cuda_environment(dict(cuda_environment)) != cuda_environment):
        raise ValueError('preflight CUDA environment policy is incomplete or unsupported')
    source = preflight.get('source', {})
    commit = source.get('source_commit')
    file_hashes = source.get('source_sha256')
    if (type(commit) is not str or re.fullmatch(r'[0-9a-f]{40}', commit) is None
            or source.get('source_dirty') is not False or type(file_hashes) is not dict
            or set(file_hashes) != set(_SOURCE_FILES)
            or any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None
                   for value in file_hashes.values())):
        raise ValueError('preflight requires a clean source commit and all source hashes')
    metadata = preflight.get('model_metadata', {})
    assets = {name: digest for name, (_, digest) in ASSETS.items()}
    if (metadata.get('model_id') != MODEL_ID or metadata.get('model_revision') != REVISION
            or metadata.get('tokenizer_id') != MODEL_ID or metadata.get('tokenizer_revision') != REVISION
            or metadata.get('model_class') != 'GPT2LMHeadModel'
            or metadata.get('tokenizer_class') != 'GPT2TokenizerFast'
            or type(metadata.get('parameter_count')) is not int or metadata['parameter_count'] != 81912576
            or type(metadata.get('torch_version')) is not str or not metadata['torch_version']
            or type(metadata.get('transformers_version')) is not str or not metadata['transformers_version']
            or metadata.get('asset_sha256') != assets
            or metadata.get('weights_sha256') != ASSETS['model.safetensors'][1]
            or metadata.get('device') != 'cpu' or metadata.get('dtype') != 'float32'
            or metadata.get('attention_implementation') != 'eager' or metadata.get('use_cache') is not False):
        raise ValueError('preflight did not verify all pinned model assets and settings')
    prompts = DISTILGPT2_PROMPTS[:2] if quick else DISTILGPT2_PROMPTS
    if preflight.get('public_prompts') != list(prompts):
        raise ValueError('preflight public prompts do not match the fixed protocol')
    jobs = preflight.get('manifest_jobs')
    expected = [(f'{team}-{index}', team, index) for team in ('team-a', 'team-b')
                for index in range(len(prompts))]
    if type(jobs) is not list or len(jobs) != len(expected):
        raise ValueError('preflight has the wrong fixed workload size')
    for job, (job_id, team, index) in zip(jobs, expected):
        if (type(job) is not dict or job.get('job_id') != job_id or job.get('team_id') != team
                or job.get('seed') != 1000 + index or job.get('capture_policy') != 'none'
                or type(job.get('units')) is not int or job['units'] <= 0
                or job.get('workload_class') != f"{job['units']}-positions"
                or type(job.get('input_digest')) is not str
                or re.fullmatch(r'[0-9a-f]{64}', job['input_digest']) is None
                or job.get('request_sha256') != job['input_digest']):
            raise ValueError('preflight has an invalid fixed job')
    _verify_source_commit(source)


def _physical_gpu_inventory():
    """Read the bounded physical UUID inventory without initializing CUDA."""
    import subprocess
    from .device_identity import parse_nvidia_smi_inventory
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=5)
    return parse_nvidia_smi_inventory(result.stdout)


def _direct_python_module(module, arguments, *, pycache_prefix, project=None, site_paths=None):
    """Run the real Python child, avoiding Windows virtualenv redirector PIDs."""
    import sys
    if type(module) is not str or not module or not isinstance(arguments, (list, tuple)):
        raise ValueError('a module and an argument sequence are required')
    if any(type(argument) is not str for argument in arguments):
        raise ValueError('module arguments must be strings')
    prefix = Path(pycache_prefix)
    if not prefix.is_absolute() or not prefix.is_dir() or next(prefix.iterdir(), None) is not None:
        raise ValueError('a fresh absolute empty owner bytecode prefix is required')
    root = Path(project or Path(__file__).resolve().parent.parent).resolve(strict=True)
    if site_paths is None:
        try:
            site_paths = json.loads(os.environ['LOOKING_GLASS_THROUGHPUT_SITE_PATHS'])
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError('bootstrap dependency directories are missing') from exc
    if (type(site_paths) is not list or not site_paths
            or any(type(path) is not str or not Path(path).is_absolute() or not Path(path).is_dir()
                   for path in site_paths)):
        raise RuntimeError('bootstrap dependency directories are invalid')
    base = Path(sys._base_executable).resolve(strict=True)
    spec = json.dumps({'module': module, 'project': str(root), 'site_paths': site_paths},
                      separators=(',', ':'))
    bootstrap = Path(__file__).resolve().parent.parent / 'scripts' / 'throughput_bootstrap.py'
    if not bootstrap.is_file():
        raise RuntimeError('source bootstrap script is missing')
    return [str(base), '-I', '-S', '-B', '-X', f'pycache_prefix={prefix.resolve()}',
            str(bootstrap), '--bootstrap-internal', spec, *arguments]


def _validate_recorded_windows(directory, result, *, jobs, repetitions, protocol_seed):
    """Reconcile every raw attempt with the saved windows and recompute them."""
    expected_jobs = {job['job_id']: job for job in jobs}
    admission_status = result.get('admission_status')
    if admission_status != result.get('metadata', {}).get('admission_status'):
        raise ValueError('measurement admission status differs from metadata')
    session_id, broker_id = result.get('benchmark_session_id'), result.get('broker_id')
    if any(type(value) is not str or re.fullmatch(r'[0-9a-f]{32}', value) is None
           for value in (session_id, broker_id)):
        raise ValueError('benchmark session or broker identity is missing')
    config = result.get('config', {})
    if (type(config.get('repetitions')) is not int or config['repetitions'] != repetitions
            or type(config.get('warmup_rounds_per_job_and_mode')) is not int
            or config['warmup_rounds_per_job_and_mode'] != 1
            or type(config.get('shared_clients')) is not int or config['shared_clients'] != 2
            or type(config.get('max_outstanding_per_client')) is not int
            or config['max_outstanding_per_client'] != 1):
        raise ValueError('saved protocol configuration differs from the fixed design')
    def read_jsonl(name):
        path = directory / name
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('evidence stream exceeds the bounded protocol size')
        with path.open('r', encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]
    raw = read_jsonl('raw_samples.jsonl')
    warmups = read_jsonl('warmup_samples.jsonl')
    partial = read_jsonl('partial_windows.jsonl')
    expected_count = repetitions * 2 * len(jobs)
    if len(raw) != expected_count or len(warmups) != expected_count:
        raise ValueError('raw or warmup attempt inventory is incomplete')
    windows = result.get('windows')
    if type(windows) is not list or len(windows) != repetitions * 2 or partial != windows:
        raise ValueError('saved partial and final window inventories differ')
    seen_broker_jobs = set()
    def index_attempts(rows, *, warmup):
        indexed = {}
        for original in rows:
            row = dict(original)
            if row.pop('benchmark_session_id', None) != session_id:
                raise ValueError('attempt session identity differs')
            if row.pop('admission_status', None) != admission_status:
                raise ValueError('attempt admission status differs from measurement')
            repetition, mode = row.pop('repetition', None), row.pop('mode', None)
            if type(repetition) is not int or repetition not in range(repetitions) or mode not in ('serial', 'shared'):
                raise ValueError('attempt repetition or mode is invalid')
            if warmup and row.pop('warmup_round', None) != 0:
                raise ValueError('warmup round differs from protocol')
            job_id = row.get('job_id')
            job = expected_jobs.get(job_id)
            if job is None or any(row.get(key) != job[key] for key in ('team_id', 'workload_class', 'units')):
                raise ValueError('attempt does not match its frozen job')
            if (row.get('state') != 'succeeded' or row.get('broker_id') != broker_id
                    or type(row.get('broker_job_id')) is not int or row['broker_job_id'] <= 0
                    or type(row.get('output_digest')) is not str
                    or re.fullmatch(r'[0-9a-f]{64}', row['output_digest']) is None
                    or row.get('timed_out') is not False or row.get('cancel_requested') is not False):
                raise ValueError('attempt is unsuccessful or lacks broker/output identity')
            _validate_sample(row)
            if row['broker_job_id'] in seen_broker_jobs:
                raise ValueError('broker job ID was reused')
            seen_broker_jobs.add(row['broker_job_id'])
            key = (repetition, mode, job_id)
            if key in indexed:
                raise ValueError('attempt is duplicated')
            indexed[key] = row
        return indexed
    raw_index = index_attempts(raw, warmup=False)
    warm_index = index_attempts(warmups, warmup=True)
    expected_keys = {(rep, mode, job_id) for rep in range(repetitions)
                     for mode in ('serial', 'shared') for job_id in expected_jobs}
    if set(raw_index) != expected_keys or set(warm_index) != expected_keys:
        raise ValueError('attempt inventory differs from the fixed workload')
    teams = sorted({job['team_id'] for job in jobs})
    for index in (raw_index, warm_index):
        for rep in range(repetitions):
            for mode in ('serial', 'shared'):
                streams = [jobs] if mode == 'serial' else [
                    [job for job in jobs if job['team_id'] == team] for team in teams]
                for stream in streams:
                    attempts = [index[(rep, mode, job['job_id'])] for job in stream]
                    if any(first['returned_ns'] > second['attempt_ns']
                           for first, second in zip(attempts, attempts[1:])):
                        raise ValueError('closed-loop client attempts overlap or violate fixed job order')
    service_intervals = sorted((row['started_ns'], row['finished_ns'])
                               for row in list(raw_index.values()) + list(warm_index.values()))
    if any(first[1] > second[0] for first, second in zip(service_intervals, service_intervals[1:])):
        raise ValueError('service intervals overlap for the single owner')
    schedule = []
    rng = random.Random(protocol_seed)
    for repetition in range(repetitions):
        modes = ['serial', 'shared']
        rng.shuffle(modes)
        schedule.extend((repetition, mode) for mode in modes)
    if [(window.get('repetition'), window.get('mode')) for window in windows] != schedule:
        raise ValueError('window order differs from the recorded protocol seed')
    indexed_windows = {}
    previous_end = 0
    for window in windows:
        key = (window.get('repetition'), window.get('mode'))
        if (key in indexed_windows or type(key[0]) is not int or key[0] not in range(repetitions)
                or key[1] not in ('serial', 'shared')
                or window.get('status') != 'complete' or window.get('unattempted_job_ids') != []
                or window.get('fatal_errors') != [] or window.get('manifest') != result['manifest']
                or window.get('manifest_sha256') != result['manifest_sha256']
                or window.get('admission_status') != admission_status):
            raise ValueError('window identity or manifest is invalid')
        samples = window.get('samples')
        if type(samples) is not list or len(samples) != len(jobs):
            raise ValueError('window has the wrong sample count')
        for row in samples:
            if raw_index.get((key[0], key[1], row.get('job_id'))) != row:
                raise ValueError('raw attempt differs from its window record')
        if (window['start_ns'] < previous_end or any(
                warm_index[(key[0], key[1], job_id)]['attempt_ns'] < previous_end
                or warm_index[(key[0], key[1], job_id)]['returned_ns'] > window['start_ns']
                for job_id in expected_jobs)):
            raise ValueError('warmups or measurement windows violate the recorded chronology')
        previous_end = window['end_ns']
        if summarize_window(samples, start_ns=window['start_ns'], end_ns=window['end_ns']) != window.get('summary'):
            raise ValueError('window summary does not match raw attempts')
        memory = window.get('memory', {})
        for phase in ('before', 'after'):
            snapshot = memory.get(phase, {})
            names = ['device_free_bytes', 'device_total_bytes', 'process_allocated_bytes', 'process_reserved_bytes']
            if phase == 'after':
                names += ['process_peak_allocated_bytes', 'process_peak_reserved_bytes']
            if (any(type(snapshot.get(name)) is not int or snapshot[name] < 0 for name in names)
                    or snapshot['device_total_bytes'] <= 0
                    or snapshot['device_free_bytes'] > snapshot['device_total_bytes']
                    or snapshot['process_allocated_bytes'] > snapshot['process_reserved_bytes']
                    or snapshot['process_reserved_bytes'] > snapshot['device_total_bytes']
                    or (phase == 'after' and
                        (snapshot['process_peak_allocated_bytes'] < snapshot['process_allocated_bytes']
                         or snapshot['process_peak_reserved_bytes'] < snapshot['process_reserved_bytes']
                         or snapshot['process_peak_allocated_bytes'] > snapshot['process_peak_reserved_bytes']))):
                raise ValueError('window lacks valid GPU memory telemetry')
        if memory['before']['device_total_bytes'] != memory['after']['device_total_bytes']:
            raise ValueError('GPU memory total changed within a window')
        if (memory['after']['process_peak_allocated_bytes'] < memory['before']['process_allocated_bytes']
                or memory['after']['process_peak_reserved_bytes'] < memory['before']['process_reserved_bytes']
                or memory['after']['process_peak_reserved_bytes'] > memory['after']['device_total_bytes']):
            raise ValueError('GPU allocator peak is inconsistent with the window baseline or total')
        indexed_windows[key] = window
    comparisons = result.get('comparisons')
    if type(comparisons) is not list or len(comparisons) != repetitions:
        raise ValueError('comparison inventory differs from the protocol')
    for rep in range(repetitions):
        serial, shared = indexed_windows[(rep, 'serial')], indexed_windows[(rep, 'shared')]
        comparison = {'repetition': rep, **compare_windows(serial, shared)}
        if (type(comparisons[rep].get('repetition')) is not int
                or comparisons[rep] != comparison or comparison['valid_paired_comparison'] is not True):
            raise ValueError('paired comparison does not match the saved windows')
        for mode in ('serial', 'shared'):
            for job_id in expected_jobs:
                if warm_index[(rep, mode, job_id)]['output_digest'] != raw_index[(rep, mode, job_id)]['output_digest']:
                    raise ValueError('warmup and measured output digests differ')
                if raw_index[(rep, mode, job_id)]['output_digest'] != raw_index[(0, 'serial', job_id)]['output_digest']:
                    raise ValueError('output digest changed between repetitions')


def _validate_owner_evidence(owner_dir, *, supervision, preflight, gpu_uuid, repetitions,
                             quick, protocol_seed):
    """Require matching identities and hashed, fully successful owner evidence."""
    _validate_preflight(preflight, quick=quick)
    owner_dir = Path(owner_dir)
    def read(name):
        path = owner_dir / name
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('owner JSON evidence exceeds the bounded protocol size')
        with path.open('r', encoding='utf-8') as stream:
            return json.load(stream)
    setup = read('setup.json')
    cleanup = read('cleanup.json')
    final = read('run_status.json')
    result = read('measurements/results.json')
    initial = read('measurements/manifest.json')
    if (setup.get('owner_run_id') != supervision.run_id or setup.get('owner_pid') != supervision.pid
            or setup.get('environment', {}).get('runtime_gpu_uuid') != gpu_uuid):
        raise ValueError('owner run or verified GPU identity differs from the supervisor')
    if any(record.get('admission_status') != 'development_unprofiled'
           for record in (setup, cleanup, final, result, initial)):
        raise ValueError('owner evidence must identify unprofiled development admission')
    allocator = cleanup.get('cuda_allocator_after_cleanup')
    if (cleanup.get('status') != 'complete' or cleanup.get('owner_attested_clean') is not True
            or cleanup.get('broker_state') != 'stopped' or cleanup.get('memory_lease_released') is not True
            or type(allocator) is not dict or set(allocator) != {'allocated_bytes', 'reserved_bytes'}
            or any(type(value) is not int or value != 0 for value in allocator.values())
            or final.get('status') != 'complete' or final.get('measurement_status') != 'complete'
            or final.get('cleanup_status') != 'complete' or final.get('work_may_still_be_running') is not False
            or result.get('status') != 'complete'):
        raise ValueError('owner measurement or cleanup did not complete')
    if setup.get('source', {}).get('source_commit') != preflight['source']['source_commit']:
        raise ValueError('source commit changed since CPU preflight')
    if setup.get('source', {}).get('source_sha256') != preflight['source']['source_sha256']:
        raise ValueError('source files changed since CPU preflight')
    if setup.get('source', {}).get('source_dirty') is not False:
        raise ValueError('owner source tree is dirty')
    if (setup.get('protocol') != 'shared-model-throughput-v1' or result.get('protocol') != 'shared-model-throughput-v1'
            or setup.get('smoke_only') is not quick or type(setup.get('repetitions')) is not int
            or setup['repetitions'] != repetitions or type(setup.get('protocol_seed')) is not int
            or setup['protocol_seed'] != protocol_seed
            or type(result.get('config', {}).get('protocol_seed')) is not int
            or result['config']['protocol_seed'] != protocol_seed
            or result.get('unit_name') != 'input_positions_including_padding'):
        raise ValueError('owner protocol or configuration differs from preflight')
    environment = setup.get('environment', {})
    if (environment.get('device') != 'cuda:0' or environment.get('deterministic_algorithms') is not True
            or environment.get('bytecode_isolation') != 'empty_prefix_no_writes'
            or environment.get('python_no_site') is not True
            or environment.get('python_ignores_environment') is not True
            or environment.get('python_isolated') is not True
            or environment.get('python_hash_randomization') is not True
            or environment.get('cuda_environment') != preflight['cuda_environment']
            or environment.get('allocator_backend') != 'native'
            or type(environment.get('allocator_fraction')) not in (int, float)
            or not isfinite(environment['allocator_fraction'])
            or abs(environment['allocator_fraction'] - 0.75) > 1e-6
            or type(environment.get('torch_threads')) is not int or environment['torch_threads'] != 1
            or type(environment.get('torch_interop_threads')) is not int or environment['torch_interop_threads'] != 1
            or environment.get('cudnn_benchmark') is not False or environment.get('cudnn_deterministic') is not True
            or environment.get('cudnn_allow_tf32') is not False or environment.get('matmul_allow_tf32') is not False
            or environment.get('cublas_workspace_config') not in (':4096:8', ':16:8')
            or environment.get('torch') != preflight['model_metadata'].get('torch_version')
            or environment.get('transformers') != preflight['model_metadata'].get('transformers_version')
            or type(environment.get('cuda_runtime')) is not str or not environment['cuda_runtime']
            or type(environment.get('device_name')) is not str or not environment['device_name']
            or type(environment.get('compute_capability')) is not list
            or len(environment['compute_capability']) != 2
            or any(type(value) is not int or value < 0 for value in environment['compute_capability'])):
        raise ValueError('owner runtime settings differ from the fixed deterministic protocol')
    if setup.get('workload_replay', {}).get('public_prompts') != preflight['public_prompts']:
        raise ValueError('owner replay prompts differ from CPU preflight')
    if result.get('manifest', {}).get('jobs') != preflight['manifest_jobs']:
        raise ValueError('owner workload differs from CPU preflight')
    model_spec = dict(result.get('model_spec', {}))
    fingerprint = model_spec.pop('service_model_fingerprint', None)
    expected_model = dict(preflight['model_metadata'])
    expected_model.update(device='cuda:0', dtype='float32', eval_mode=True, capture_policy='none')
    if (model_spec != expected_model
            or fingerprint != _model_fingerprint(model_spec, setup['environment'])):
        raise ValueError('owner model assets or runtime fingerprint differ from preflight')
    if (result.get('manifest', {}).get('model_spec') != _canonical(result['model_spec'])
            or result.get('manifest', {}).get('unit_name') != result['unit_name']):
        raise ValueError('manifest model specification or work units differ')
    limits = setup.get('limits', {})
    expected_limits = {'max_batch_size': 1, 'max_sequence_length': 128,
                       'max_input_bytes': 1024 * 1024, 'max_output_bytes': 8 * 1024 * 1024,
                       'max_gpu_trace_bytes': 0, 'max_host_trace_bytes': 0,
                       'gpu_model_load_estimate_bytes': 2 * 1024 ** 3,
                       'gpu_per_forward_estimate_bytes': 1024 ** 3,
                       'broker_max_pending': 4, 'broker_max_pending_per_team': 1}
    if any(type(limits.get(key)) is not int or limits[key] != value for key, value in expected_limits.items()):
        raise ValueError('owner memory or queue policy differs from the fixed protocol')
    profile_record = read('admission_profiles.json')
    if (profile_record.get('status') != 'provisional_unmeasured_estimates'
            or profile_record.get('admission_status') != 'development_unprofiled'):
        raise ValueError('admission estimate status is missing')
    signatures = {job['units'] for job in preflight['manifest_jobs']}
    dtypes = (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64'))
    expected_profiles = []
    for length in sorted(signatures):
        signature = {'shape': (1, length), 'dtypes': dtypes}
        expected_profiles.append({'profile_id': 'fixed-v1-' + canonical_digest(signature)[:16],
            'model_id': fingerprint, 'device': 'cuda:0', 'shape': [1, length],
            'input_dtypes': [list(value) for value in dtypes], 'output_limit_bytes': 8 * 1024 * 1024,
            'estimated_incremental_bytes': 1024 ** 3})
    profiles = profile_record.get('profiles')
    if type(profiles) is not list or sorted(profiles, key=lambda item: item.get('profile_id', '')) != sorted(expected_profiles, key=lambda item: item['profile_id']):
        raise ValueError('admission profiles differ from the fixed workload')
    expected_metadata = {**setup['source'], 'external_gpu_activity': setup['external_gpu_activity'],
        'environment': setup['environment'], 'workload_replay': setup['workload_replay'],
        'smoke_only': quick, 'memory_policy': {**setup['limits'], 'profiles': profiles},
        'admission_status': 'development_unprofiled'}
    if (result.get('metadata') != expected_metadata or initial.get('metadata') != expected_metadata
            or initial.get('model_spec') != result.get('model_spec')
            or initial.get('unit_name') != result.get('unit_name')
            or initial.get('config') != result.get('config')
            or initial.get('clock') != result.get('clock')
            or initial.get('environment') != result.get('environment')
            or initial.get('benchmark_session_id') != result.get('benchmark_session_id')
            or initial.get('broker_id') != result.get('broker_id')
            or initial.get('benchmark_source_sha256') != result.get('benchmark_source_sha256')):
        raise ValueError('initial, final and setup reproducibility metadata differ')
    if (result.get('manifest') != initial.get('manifest')
            or result.get('manifest_sha256') != initial.get('manifest_sha256')):
        raise ValueError('initial and completed manifests differ')
    encoded = json.dumps(result['manifest'], sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != result['manifest_sha256']:
        raise ValueError('manifest hash does not match')
    hashes = _evidence_hashes(owner_dir / 'measurements')
    required = {'manifest.json', 'warmup_samples.jsonl', 'raw_samples.jsonl', 'partial_windows.jsonl'}
    if not required.issubset(hashes) or hashes != result.get('evidence_sha256'):
        raise ValueError('owner evidence is missing or its digest differs')
    _validate_recorded_windows(owner_dir / 'measurements', result,
                               jobs=preflight['manifest_jobs'], repetitions=repetitions,
                               protocol_seed=protocol_seed)
    owner_files = ('setup.json', 'admission_profiles.json', 'cleanup.json',
                   'run_status.json', 'measurements/results.json')
    owner_hashes = {name: hashlib.sha256((owner_dir / name).read_bytes()).hexdigest() for name in owner_files}
    return {'status': 'complete', 'manifest_sha256': result['manifest_sha256'],
            'evidence_sha256': hashes, 'owner_file_sha256': owner_hashes}


def _controller_evidence_hashes(output_dir):
    """Hash the published parent records and preserve individual read errors."""
    hashes, errors = {}, {}
    for name in ('controller_manifest.json', 'preflight.json'):
        path = output_dir / name
        if not path.exists():
            continue
        try:
            if path.stat().st_size > 1024 * 1024:
                raise ValueError('controller evidence exceeds the bounded size')
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception as exc:
            errors[name] = _describe_error(exc)
    return hashes, errors


def run_supervised_distilgpt2_benchmark(output_dir, *, cache_dir, device_root, gpu_uuid,
                                        gpu_activity='unknown', repetitions=3, protocol_seed=0,
                                        quick=False, timeout=600, development_unprofiled=False):
    """Preflight on CPU, then supervise one UUID-bound GPU owner process."""
    import sys
    from .device_identity import resolve_device_identity
    from .supervisor import run_supervised
    if (type(quick) is not bool or type(development_unprofiled) is not bool
            or type(gpu_activity) is not str or not gpu_activity.strip()):
        raise ValueError('record GPU activity and use a boolean quick flag')
    if type(repetitions) is not int or not 1 <= repetitions <= 10 or type(protocol_seed) is not int:
        raise ValueError('repetitions must be 1 through 10 and the protocol seed must be an integer')
    if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
        raise ValueError('supervision timeout must be finite and positive')
    if development_unprofiled and not quick:
        raise ValueError('unprofiled GPU development is limited to the quick workload')
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('the benchmark output directory already exists')
    output_dir.mkdir(parents=True, exist_ok=False)
    supervision_started = False
    try:
        _atomic_write_json(output_dir / 'controller_manifest.json', {
            'protocol': 'supervised-shared-model-throughput-v1', 'status': 'initializing',
            'smoke_only': quick, 'repetitions': 1 if quick else repetitions,
            'protocol_seed': protocol_seed, 'external_gpu_activity': gpu_activity,
            'admission_status': 'development_unprofiled' if development_unprofiled else 'disabled_unprofiled',
            'bytecode_policy': 'empty_prefix_no_writes'})
        if not development_unprofiled:
            raise RuntimeError('unprofiled GPU workload is disabled; a measured admission profile is required')
        _require_fresh_bytecode_prefix(sys)
        runtime_environment = _configure_cuda_environment(os.environ)
        preflight = _preflight_distilgpt2(cache_dir, quick=quick)
        if _configure_cuda_environment(dict(os.environ)) != runtime_environment:
            raise RuntimeError('CUDA environment changed during CPU preflight')
        preflight['cuda_environment'] = runtime_environment
        _validate_preflight(preflight, quick=quick)
        _atomic_write_json(output_dir / 'preflight.json', preflight)
        inventory = _physical_gpu_inventory()
        identity = resolve_device_identity(device_root, gpu_uuid, physical_uuids=inventory)
        arguments = ['--owner-child', '--development-unprofiled', '--output', str(output_dir / 'owner'),
                   '--cache-dir', str(Path(cache_dir).resolve()), '--device', 'cuda:0',
                   '--gpu-activity', gpu_activity, '--repetitions', str(repetitions),
                   '--protocol-seed', str(protocol_seed)]
        if quick:
            arguments.append('--quick')
        owner_prefix = output_dir / 'owner-pycache'
        owner_prefix.mkdir(exist_ok=False)
        command = _direct_python_module('looking_glass.throughput_benchmark', arguments,
                                        pycache_prefix=owner_prefix)
        supervision_started = True
        supervision = run_supervised(command, device_lease_path=identity.lock_path,
            marker_path=identity.marker_path, identity=identity, timeout=timeout,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=_supervised_child_environment(runtime_environment, os.environ),
            inherit_environment=False)
        health = {key: getattr(supervision, key) for key in
                  ('run_id', 'pid', 'exit_code', 'clean', 'timed_out', 'reason')}
        record = {'status': 'failed', 'admission_status': 'development_unprofiled',
                  'supervisor': health, 'runtime_gpu_uuid': identity.gpu_uuid,
                  'evidence': None, 'error': None, 'owner_reported_status': None}
        if (supervision.clean is True and supervision.timed_out is False
                and type(supervision.exit_code) is int and supervision.exit_code == 0):
            try:
                record['evidence'] = _validate_owner_evidence(output_dir / 'owner',
                    supervision=supervision, preflight=preflight, gpu_uuid=identity.gpu_uuid,
                    repetitions=1 if quick else repetitions, quick=quick, protocol_seed=protocol_seed)
                if record['evidence'].get('status') != 'complete':
                    raise ValueError('owner evidence validation did not complete')
                record['status'] = 'development_only'
            except Exception as exc:
                record['error'] = _describe_error(exc)
        else:
            if supervision.timed_out:
                record['error'] = 'supervised owner timed out; device handoff is unsafe'
            elif not supervision.clean:
                record['error'] = 'supervised owner did not attest a clean device handoff'
            else:
                record['error'] = 'owner exited nonzero after a clean device handoff'
            try:
                with (output_dir / 'owner' / 'run_status.json').open('r', encoding='utf-8') as stream:
                    reported = json.load(stream).get('status')
                if reported in ('complete', 'invalid_comparison', 'failed', 'cancelled', 'cleanup_failed', 'evidence_write_failed'):
                    record['owner_reported_status'] = reported
            except (OSError, ValueError, AttributeError):
                pass
        hashes, hash_errors = _controller_evidence_hashes(output_dir)
        record['controller_evidence_sha256'] = hashes
        record['controller_evidence_hash_errors'] = hash_errors
        if hash_errors or set(hashes) != {'controller_manifest.json', 'preflight.json'}:
            record['status'] = 'failed'
            record['error'] = 'controller evidence is missing or could not be hashed'
        _atomic_write_json(output_dir / 'controller_status.json', record)
        return record
    except BaseException as exc:
        try:
            if not (output_dir / 'controller_status.json').exists():
                try:
                    hashes, hash_errors = _controller_evidence_hashes(output_dir)
                except Exception as hash_error:
                    hashes, hash_errors = {}, {'controller': _describe_error(hash_error)}
                _atomic_write_json(output_dir / 'controller_failure.json', {
                    'status': 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                    'admission_status': 'development_unprofiled' if development_unprofiled else 'disabled_unprofiled',
                    'error': _describe_error(exc),
                    'controller_evidence_sha256': hashes, 'controller_evidence_hash_errors': hash_errors,
                    'device_state_may_be_unsafe': supervision_started})
        except Exception as recording_error:
            raise exc from recording_error
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run the pinned shared-model load protocol.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path, help='verified offline Hugging Face cache')
    parser.add_argument('--device', default='cpu', help='cpu or a coordinator-reserved CUDA device')
    parser.add_argument('--gpu-activity', default='unknown; no device-exclusivity claim')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--protocol-seed', type=int, default=0)
    parser.add_argument('--quick', action='store_true', help='small smoke check, not the full protocol')
    parser.add_argument('--gpu-uuid', help='full physical GPU UUID for the supervised controller')
    parser.add_argument('--device-root', type=Path, help='shared local device-coordination root')
    parser.add_argument('--supervision-timeout', type=float, default=600)
    parser.add_argument('--development-unprofiled', action='store_true',
                        help='allow a supervised development run using unmeasured estimates')
    parser.add_argument('--owner-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device in ('cuda', 'cuda:0') and not args.owner_child:
        if not args.gpu_uuid or not args.device_root:
            parser.error('CUDA requires --gpu-uuid and the shared --device-root')
        result = run_supervised_distilgpt2_benchmark(args.output, cache_dir=args.cache_dir,
            device_root=args.device_root, gpu_uuid=args.gpu_uuid, gpu_activity=args.gpu_activity,
            repetitions=args.repetitions, protocol_seed=args.protocol_seed, quick=args.quick,
            timeout=args.supervision_timeout, development_unprofiled=args.development_unprofiled)
        print(json.dumps({'output': str(args.output), 'status': result['status'],
                          'admission_status': result['admission_status'], 'smoke_only': args.quick}))
        return 0 if result['status'] == 'complete' else 1
    if args.owner_child and args.device not in ('cuda', 'cuda:0'):
        parser.error('the owner child must use cuda:0')
    result = run_distilgpt2_benchmark(args.output, cache_dir=args.cache_dir, device=args.device,
                                     gpu_activity=args.gpu_activity, repetitions=args.repetitions,
                                     protocol_seed=args.protocol_seed, quick=args.quick,
                                     development_unprofiled=args.development_unprofiled)
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'admission_status': result.get('admission_status'),
                      'smoke_only': args.quick, 'comparisons': result['comparisons']}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
