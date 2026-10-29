"""A small, CPU-only characterization of batched event recording cost.

This is not a model or GPU benchmark. The frozen experiment and its limits are
documented in docs/RECORDER_OVERHEAD_V1.md.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from .batched_recorder import BatchedEventRecorder
from .throughput_benchmark import _atomic_write_json


VERSION = 'batched-recorder-cpu-overhead-v1'
SEED = 17
REPETITIONS = 7
WARMUP_EVENTS = 200
EVENTS_PER_WINDOW = 400
_TEMPLATES = (
    {'kind': 'selected', 'job_id': 'job-a'},
    {'kind': 'forward_entry', 'call_id': 'call-1', 'group_id': 'group-1',
     'profile_sha256': 'a' * 64, 'input_shape': [2, 5],
     'members': [{'job_id': 'job-a', 'team_id': 'team-a', 'broker_id': 'broker',
                  'broker_job_id': 'broker-j-1', 'row_offset': 0},
                 {'job_id': 'job-b', 'team_id': 'team-b', 'broker_id': 'broker',
                  'broker_job_id': 'broker-j-2', 'row_offset': 1}]},
)


class _RecorderLoss(RuntimeError):
    def __init__(self, *, attempted, report):
        super().__init__('event sink lost an observation')
        self.detail = {'attempted': attempted, 'recorder_report': report}


def _discard(event):
    return True


def _measure(mode, event_count):
    if mode not in ('discard', 'record') or type(event_count) is not int or event_count <= 0 or event_count % 2:
        raise ValueError('window requires a supported mode and an even positive event count')
    recorder = BatchedEventRecorder(max_events=event_count) if mode == 'record' else None
    sink = recorder.record if recorder is not None else _discard
    gc_before = {'counts': list(gc.get_count()),
                 'collections': [entry['collections'] for entry in gc.get_stats()]}
    start = time.perf_counter_ns()
    for index in range(event_count):
        # Both modes construct the same event and capture the same source clock.
        event = {**_TEMPLATES[index % 2], 'time_ns': time.perf_counter_ns()}
        if sink(event) is not True:
            if recorder is None:
                raise RuntimeError('discard sink lost an observation')
            recorder.seal(producers_stopped=True)
            raise _RecorderLoss(attempted=index + 1, report=recorder.report())
    end = time.perf_counter_ns()
    gc_after = {'counts': list(gc.get_count()),
                'collections': [entry['collections'] for entry in gc.get_stats()]}
    report = None
    if recorder is not None:
        # Snapshotting and serialization are deliberately outside the window.
        recorder.seal(producers_stopped=True)
        report = recorder.report()
        if report['events_complete'] is not True or report['recorded'] != event_count:
            raise RuntimeError('recorder did not retain the full workload')
    if end <= start:
        raise RuntimeError('clock did not advance during the window')
    return {'mode': mode, 'start_ns': start, 'end_ns': end, 'elapsed_ns': end - start,
            'event_count': event_count,
            'event_counts': {'selected': event_count // 2, 'forward_entry': event_count // 2},
            'recorded': report['recorded'] if report is not None else None,
            'events_complete': report['events_complete'] if report is not None else None,
            'gc_before': gc_before, 'gc_after': gc_after}


def _summarize(samples):
    pairs = {}
    for sample in samples:
        rep, mode, elapsed, count = (sample['repetition'], sample['mode'],
                                     sample['elapsed_ns'], sample['event_count'])
        if (type(rep) is not int or rep < 0 or mode not in ('record', 'discard')
                or type(elapsed) is not int or elapsed <= 0 or type(count) is not int or count <= 0):
            raise ValueError('invalid sample')
        if mode in pairs.setdefault(rep, {}):
            raise ValueError('duplicate mode within a pair')
        pairs[rep][mode] = (elapsed, count)
    if not pairs or any(set(pair) != {'record', 'discard'}
                        or pair['record'][1] != pair['discard'][1] for pair in pairs.values()):
        raise ValueError('each pair requires the same workload in both modes')
    ordered = [pairs[rep] for rep in sorted(pairs)]
    increments = [(pair['record'][0] - pair['discard'][0]) / pair['record'][1] for pair in ordered]
    return {'discard_median_ns': statistics.median(pair['discard'][0] for pair in ordered),
            'record_median_ns': statistics.median(pair['record'][0] for pair in ordered),
            'paired_incremental_ns_per_event': increments,
            'median_paired_incremental_ns_per_event': statistics.median(increments)}


def _clock_screen(samples, resolution_ns, *, monotonic=True, adjustable=False):
    if type(resolution_ns) is not int or resolution_ns <= 0:
        raise ValueError('clock resolution must be a positive integer')
    if type(monotonic) is not bool or type(adjustable) is not bool:
        raise ValueError('clock properties must be exact booleans')
    minimum = 100 * resolution_ns
    return {'multiplier': 100, 'minimum_window_ns': minimum,
            'monotonic': monotonic, 'nonadjustable': not adjustable,
            'passes': (monotonic and not adjustable and bool(samples)
                       and all(sample['elapsed_ns'] >= minimum for sample in samples))}


def _mode_orders():
    # Preallocate a balanced 4/3 order assignment, then randomize its blocks.
    orders = [('discard', 'record')] * 4 + [('record', 'discard')] * 3
    random.Random(SEED).shuffle(orders)
    return orders


def _source_metadata():
    root = Path(__file__).resolve().parent.parent
    git = lambda *args: subprocess.run(['git', *args], cwd=root, check=True,
                                       capture_output=True, timeout=10).stdout
    commit = git('rev-parse', 'HEAD').decode('ascii').strip()
    if git('status', '--porcelain=v1', '--untracked-files=all').strip():
        raise RuntimeError('source checkout must be clean before and after the run')
    files = {str(Path(__file__).resolve().relative_to(root)).replace('\\', '/')}
    for name, module in tuple(sys.modules.items()):
        if name == 'looking_glass' or name.startswith('looking_glass.'):
            source = getattr(module, '__file__', None)
            if source is not None:
                path = Path(source).resolve()
                try:
                    relative = path.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError('project module loaded outside the source checkout') from exc
                if path.suffix != '.py':
                    raise RuntimeError('project modules must be loaded from source files')
                files.add(str(relative).replace('\\', '/'))
    hashes = {}
    for relative in sorted(files):
        data = (root / relative).read_bytes()
        if data != git('show', f'{commit}:{relative}'):
            raise RuntimeError('source file differs from its recorded commit')
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return {'commit': commit, 'tracked_and_untracked_checkout_clean': True, 'sha256': hashes}


def _startup_metadata():
    prefix = sys.pycache_prefix
    if (sys.dont_write_bytecode is not True or not isinstance(prefix, str)
            or not Path(prefix).is_absolute() or not Path(prefix).is_dir()
            or any(Path(prefix).iterdir())):
        raise RuntimeError('start with -B and a fresh empty absolute -X pycache_prefix')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('set CUDA_VISIBLE_DEVICES to the empty string')
    torch = sys.modules.get('torch')
    if torch is None or torch.cuda.is_initialized():
        raise RuntimeError('PyTorch CUDA must not be initialized in this process')
    return {'bytecode_write_disabled': True, 'private_prefix_empty_when_checked': True,
            'cuda_visibility_empty': True, 'torch_cuda_initialized': False}


def run(*, partial=None):
    partial = {} if partial is None else partial
    partial.update({'version': VERSION, 'status': 'started', 'samples': [], 'phase': 'preflight'})
    startup = _startup_metadata()
    partial['startup'] = startup
    source = _source_metadata()
    partial['source'] = source
    clock = time.get_clock_info('perf_counter')
    resolution_ns = max(1, math.ceil(clock.resolution * 1_000_000_000))
    workload = {'seed': SEED, 'repetitions': REPETITIONS, 'warmup_events_per_mode': WARMUP_EVENTS,
                'events_per_window': EVENTS_PER_WINDOW, 'templates_without_timestamps': _TEMPLATES,
                'order': 'seeded randomized balanced 4/3 paired blocks', 'execution': 'single Python thread'}
    workload_bytes = json.dumps(workload, sort_keys=True, separators=(',', ':'),
                                allow_nan=False).encode('utf-8')
    partial['workload'] = workload
    partial['workload_sha256'] = hashlib.sha256(workload_bytes).hexdigest()
    partial['runtime'] = {'python': platform.python_version(),
                          'python_implementation': platform.python_implementation(),
                          'platform': platform.platform(), 'cpu': platform.processor() or None,
                          'architecture': platform.machine() or None,
                          'logical_cpu_count': os.cpu_count(),
                          'torch_version': str(sys.modules['torch'].__version__),
                          'clock': 'perf_counter_ns', 'clock_implementation': clock.implementation,
                          'clock_monotonic': clock.monotonic, 'clock_adjustable': clock.adjustable,
                          'clock_resolution_ns': resolution_ns,
                          'gc_enabled': gc.isenabled(), 'gc_thresholds': list(gc.get_threshold()),
                          'external_cpu_activity': 'not monitored'}
    partial['phase'] = 'warmup'
    for mode in ('discard', 'record'):
        partial['active'] = {'mode': mode}
        _measure(mode, WARMUP_EVENTS)
    samples = partial['samples']
    partial['phase'] = 'measurement'
    for repetition, modes in enumerate(_mode_orders()):
        for mode in modes:
            partial['active'] = {'repetition': repetition, 'mode': mode}
            sample = _measure(mode, EVENTS_PER_WINDOW)
            sample['repetition'] = repetition
            samples.append(sample)
    partial['active'] = None
    partial['phase'] = 'postflight'
    if _source_metadata() != source or _startup_metadata() != startup:
        raise RuntimeError('source or startup state changed during measurement')
    screen = _clock_screen(samples, resolution_ns, monotonic=clock.monotonic, adjustable=clock.adjustable)
    partial.update({'status': 'complete' if screen['passes'] else 'clock_screen_failed',
                    'phase': 'terminal', 'clock_screen': screen, 'summary': _summarize(samples)})
    return partial


def _save(path, value):
    _atomic_write_json(Path(path), value)


def _output_path(value):
    path = Path(value)
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError('output must be an absolute path with an existing parent')
    root = Path(__file__).resolve().parent.parent
    if path.resolve().is_relative_to(root):
        raise ValueError('output must be outside the source checkout')
    if path.exists() or path.is_symlink():
        raise FileExistsError('output already exists')
    return path


def _execute_to_file(output):
    partial = {}
    try:
        result = run(partial=partial)
    except BaseException as error:
        partial['status'] = 'failed' if isinstance(error, Exception) else 'cancelled'
        partial['error_type'] = type(error).__name__[:128]
        if type(error) is _RecorderLoss:
            partial['failure_detail'] = error.detail
        try:
            _save(output, partial)
        except BaseException as publication_error:
            raise error from publication_error
        raise
    _save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='new output file outside the source checkout')
    args = parser.parse_args()
    output = _output_path(args.output)
    result = _execute_to_file(output)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
