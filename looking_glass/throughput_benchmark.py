"""Shared-service load benchmarking; no isolated-latency claims.

The benchmark deliberately keeps raw attempt and broker timestamps so that
reported rates and distributions can be independently recomputed.
"""
from dataclasses import dataclass, fields, is_dataclass
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import random
from math import ceil
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


def compare_windows(first, second):
    """Compare immutable manifests and exact output hashes; no speedup claim."""
    same_manifest = first['manifest_sha256'] == second['manifest_sha256']
    left = {row['job_id']: row['output_digest'] for row in first['samples'] if row['state'] == 'succeeded'}
    right = {row['job_id']: row['output_digest'] for row in second['samples'] if row['state'] == 'succeeded'}
    common = left.keys() & right.keys()
    matching = all(left[key] == right[key] for key in common) if common else None
    all_success = (len(left) == len(first['samples']) and len(right) == len(second['samples'])
                   and left.keys() == right.keys() and bool(left))
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
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        published = True
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
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
    result = {'protocol': 'shared-model-throughput-v1', 'status': 'running',
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
                                       'warmup_round': warmup_round, **sample})
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
                for sample in window['samples']:
                    _append_jsonl(output_dir / 'raw_samples.jsonl',
                                  {'benchmark_session_id': result['benchmark_session_id'], 'repetition': repetition, 'mode': mode, **sample})
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
                              {'benchmark_session_id': result['benchmark_session_id'], **active_context, **partial})
                path = output_dir / ('warmup_samples.jsonl' if active_context['phase'] == 'warmup' else 'raw_samples.jsonl')
                for sample in partial['samples']:
                    _append_jsonl(path, {'benchmark_session_id': result['benchmark_session_id'],
                                        **active_context, 'partial': True, **sample})
            try:
                evidence_hashes, hash_error = _evidence_hashes(output_dir), None
            except Exception as digest_error:
                evidence_hashes, hash_error = None, _describe_error(digest_error)
            _atomic_write_json(output_dir / 'failure.json', {'status': 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
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


def _source_metadata():
    import subprocess
    root = Path(__file__).resolve().parent.parent
    names = ('looking_glass/throughput_benchmark.py', 'looking_glass/pretrained.py',
             'looking_glass/resident_worker.py', 'looking_glass/broker.py',
             'looking_glass/fair_queue.py', 'looking_glass/memory_budget.py',
             'docs/THROUGHPUT_PROTOCOL_V1.md')
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in names if (root / name).is_file()}
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


def run_distilgpt2_benchmark(output_dir, *, cache_dir, device='cpu', gpu_activity='unknown',
                              repetitions=3, protocol_seed=0, quick=False):
    """Run the pinned offline DistilGPT2 workload through one resident broker.

    The caller must obtain a coordinator-controlled GPU window before choosing
    CUDA. A process-local memory lease is an estimate, not a global GPU lock or
    guarantee against external allocation/OOM. The quick run is a smoke check.
    """
    import torch
    import transformers
    import numpy
    from .broker import Broker, BrokerUnavailable
    from .fair_queue import QueueFull
    from .memory_budget import GIB, MemoryBudget, cuda_memory_snapshot
    from .pretrained import ASSETS, MODEL_ID, REVISION, load_distilgpt2
    from .resident_worker import InvalidRequest, ResidentWorker

    resolved_device = torch.device(device)
    if resolved_device.type not in ('cpu', 'cuda'):
        raise ValueError('device must be CPU or CUDA')
    if resolved_device.type == 'cuda' and resolved_device.index is None:
        resolved_device = torch.device('cuda', 0)
    if type(quick) is not bool or type(gpu_activity) is not str or not gpu_activity.strip():
        raise ValueError('record observed external GPU activity and use a boolean quick flag')
    if resolved_device.type == 'cuda':
        raise RuntimeError('CUDA throughput is disabled until supervised physical-device ownership is integrated')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    prompts = DISTILGPT2_PROMPTS[:2] if quick else DISTILGPT2_PROMPTS
    repetitions = 1 if quick else repetitions
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    broker = lease = bundle = worker = proxy = None
    primary_error = None
    measurement_result = None
    measurement_started = False
    try:
        source = _source_metadata()
        environment = {'python': platform.python_version(), 'platform': platform.platform(),
                       'torch': str(torch.__version__), 'transformers': transformers.__version__,
                       'numpy': numpy.__version__, 'cuda_runtime': torch.version.cuda,
                       'device': str(resolved_device), 'torch_threads': torch.get_num_threads(),
                       'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                       'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                       'cudnn_benchmark': torch.backends.cudnn.benchmark,
                       'cudnn_deterministic': torch.backends.cudnn.deterministic,
                       'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                       'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                       'nvidia_inventory': _nvidia_inventory() if resolved_device.type == 'cuda' else None}
        if resolved_device.type == 'cuda':
            environment['device_name'] = torch.cuda.get_device_name(resolved_device)
            environment['compute_capability'] = list(torch.cuda.get_device_capability(resolved_device))
        setup = {'protocol': 'shared-model-throughput-v1', 'status': 'initializing',
                 'smoke_only': quick, 'source': source, 'environment': environment,
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
                            'gpu_incremental_estimate_bytes': 2 * GIB if resolved_device.type == 'cuda' else None,
                            'broker_max_pending': 4, 'broker_max_pending_per_team': 1},
                 'protocol_seed': protocol_seed, 'repetitions': repetitions}
        _atomic_write_json(output_dir / 'setup.json', setup)
        bundle = load_distilgpt2(cache_dir=cache_dir, allow_download=False)
        jobs = make_distilgpt2_workload(bundle, prompts=prompts)
        model_spec = dict(bundle.metadata)
        model_spec.update(device=str(resolved_device), dtype='float32', eval_mode=True,
                          capture_policy='none')
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
        worker = ResidentWorker(MODEL_ID, bundle.model, device=resolved_device,
                                max_batch_size=1, max_sequence_length=128,
                                max_input_bytes=1024 * 1024, max_output_bytes=8 * 1024 * 1024)
        bundle.model = None  # The worker is the only owner of the live model.
        proxy = SynchronizedWorker(worker, synchronize)
        broker = Broker(proxy, max_pending=4, max_pending_per_team=1)
        metadata = {**source, 'external_gpu_activity': gpu_activity, 'environment': environment,
                    'workload_replay': setup['workload_replay'], 'smoke_only': quick,
                    'memory_policy': setup['limits']}
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
            if broker is None and worker is not None:
                worker.close()
            if proxy is not None and proxy.quarantined:
                raise RuntimeError('worker process is quarantined; do not reuse this device context')
            if resolved_device.type == 'cuda' and lease is not None:
                torch.cuda.synchronize(resolved_device)
                if bundle is not None and bundle.model is not None:
                    bundle.model.to('cpu')
                torch.cuda.synchronize(resolved_device)
                torch.cuda.empty_cache()
                lease.release()
        except BaseException as exc:
            cleanup_error = exc
        cleanup = ({'status': 'complete', 'broker_state': 'not_started' if broker is None else 'stopped',
                    'memory_lease_released': lease is None or lease.released}
                   if cleanup_error is None else
                   {'status': 'failed', 'type': type(cleanup_error).__name__, 'message': str(cleanup_error),
                    'work_may_still_be_running': True})
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
    args = parser.parse_args()
    result = run_distilgpt2_benchmark(args.output, cache_dir=args.cache_dir, device=args.device,
                                     gpu_activity=args.gpu_activity, repetitions=args.repetitions,
                                     protocol_seed=args.protocol_seed, quick=args.quick)
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'smoke_only': args.quick, 'comparisons': result['comparisons']}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
