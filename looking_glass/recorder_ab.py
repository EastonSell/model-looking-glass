"""Controlled single-process CPU comparison of two fixed recorder sources.

See docs/RECORDER_ASCII_AB_V1.md. No model or GPU work is performed.
"""
import argparse
from copy import deepcopy
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

from . import batched_events
from .recorder_overhead import (_TEMPLATES, _clock_screen, _output_path, _save,
                                _source_metadata, _startup_metadata)


VERSION = 'recorder-ascii-ab-cpu-v1'
BASELINE_COMMIT = 'f3c6cbb2211f04b46ff5b25dcde9bf251558dc41'
FAST_COMMIT = '23a19abb75248474151c0e84af29016d71bd334f'
BASELINE_SHA256 = 'd557159a7650a5cb4ad5f4a0354570edba04be6c3666f01a6f76421e12ef62f9'
FAST_SHA256 = '8e241c8b6670539ee9d130dea587220fe3132c3452330605e612cc26c1eb53cd'
SCHEMA_SHA256 = '79cc10978caca7a026824be35ddf01013be601b94bce5afcd3303a361d83bffc'
SCHEMA_MAPPING_SHA256 = '4e68e27b8a9a07bb20ac7dda36a33ecb15d107759eca3ebe924aeb3018ab779f'
TEMPLATES_SHA256 = 'a592f928a20404ec83a88c6e37bea514da0f0245e74ca21a38494885d9994083'
SEED = 41
WARMUP_EVENTS = 200
EVENTS_PER_WINDOW = 400
_SOURCE_PATH = 'looking_glass/batched_recorder.py'
_SCHEMA_PATH = 'looking_glass/batched_events.py'
_OLD_CHECK = b'        if len(value) > 512 or any(0xD800 <= ord(char) <= 0xDFFF for char in value):\n'
_NEW_CHECK = (b"        # ASCII cannot contain surrogate code points; use Python's cached\n"
              b'        # ASCII check for the common identifiers and hashes, while retaining\n'
              b'        # explicit validation of all non-ASCII strings.\n'
              b'        if (len(value) > 512\n'
              b'                or (not value.isascii() and any(0xD800 <= ord(char) <= 0xDFFF for char in value))):\n')


class _WindowFailure(RuntimeError):
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail


def _git_blob(commit, path):
    root = Path(__file__).resolve().parent.parent
    return subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=root, check=True,
                          capture_output=True, timeout=10).stdout


def _load_variants():
    sources = {'baseline': _git_blob(BASELINE_COMMIT, _SOURCE_PATH),
               'fast': _git_blob(FAST_COMMIT, _SOURCE_PATH)}
    expected = {'baseline': BASELINE_SHA256, 'fast': FAST_SHA256}
    for name, source in sources.items():
        if hashlib.sha256(source).hexdigest() != expected[name]:
            raise ValueError('recorder source hash does not match the frozen protocol')
    if sources['baseline'].count(_OLD_CHECK) != 1 or sources['baseline'].replace(_OLD_CHECK, _NEW_CHECK) != sources['fast']:
        raise ValueError('recorder sources differ by more than the frozen ASCII check')
    schema = _git_blob(BASELINE_COMMIT, _SCHEMA_PATH)
    if (hashlib.sha256(schema).hexdigest() != SCHEMA_SHA256
            or _git_blob(FAST_COMMIT, _SCHEMA_PATH) != schema
            or Path(batched_events.__file__).read_bytes() != schema):
        raise ValueError('the shared event schema differs from the frozen source')
    classes, metadata = {}, {}
    for name, commit in (('baseline', BASELINE_COMMIT), ('fast', FAST_COMMIT)):
        namespace_name = f'looking_glass._recorder_ab_{name}'
        if namespace_name in sys.modules:
            raise RuntimeError('benchmark module namespace already exists')
        filename = f'<git:{commit}:{_SOURCE_PATH}>'
        namespace = {'__name__': namespace_name, '__package__': 'looking_glass', '__file__': filename}
        # Execute exactly the trusted Git bytes that were hashed above, with
        # separate module globals. Never replace the production recorder.
        exec(compile(sources[name], filename, 'exec'), namespace)
        if namespace['_EVENT_FIELDS'] is not batched_events._EVENT_FIELDS:
            raise RuntimeError('recorder variants do not share the loaded event schema')
        classes[name] = namespace['BatchedEventRecorder']
        metadata[name] = {'commit': commit, 'sha256': expected[name],
                          'namespace': namespace_name, 'compiled_from_verified_git_bytes': True}
    if classes['baseline'] is classes['fast']:
        raise RuntimeError('recorder variants must be distinct')
    return classes, metadata


def _ab_source_metadata():
    # Under ``python -m``, this file is __main__, so the imported-module
    # inventory in recorder_overhead does not necessarily include it.
    metadata = _source_metadata()
    path = 'looking_glass/recorder_ab.py'
    actual = Path(__file__).read_bytes()
    if actual != _git_blob(metadata['commit'], path):
        raise RuntimeError('A/B runner source differs from its recorded commit')
    metadata['sha256'][path] = hashlib.sha256(actual).hexdigest()
    return metadata


def _shared_input_hashes():
    def digest(value):
        payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    templates, schema = digest(_TEMPLATES), digest(batched_events._EVENT_FIELDS)
    if templates != TEMPLATES_SHA256 or schema != SCHEMA_MAPPING_SHA256:
        raise RuntimeError('the frozen workload templates or schema mapping changed')
    return {'templates_sha256': templates, 'schema_mapping_sha256': schema}


def _measure(recorder_type, event_count):
    if type(event_count) is not int or event_count <= 0 or event_count % 2:
        raise ValueError('a window requires an even positive event count')
    # Keep an independent expected payload in case a recorder mutates its
    # caller-owned nested input. This copy is outside either timed window.
    expected_templates = deepcopy(_TEMPLATES)
    recorder = recorder_type(max_events=event_count)
    gc_before = {'counts': list(gc.get_count()),
                 'collections': [entry['collections'] for entry in gc.get_stats()]}
    source_times = []
    start = time.perf_counter_ns()
    for index in range(event_count):
        stamp = time.perf_counter_ns()
        source_times.append(stamp)
        event = {**_TEMPLATES[index % 2], 'time_ns': stamp}
        if recorder.record(event) is not True:
            recorder.seal(producers_stopped=True)
            raise _WindowFailure('recorder lost an event',
                                 {'attempted': index + 1, 'source_times_ns': source_times,
                                  'recorder_report': recorder.report()})
    end = time.perf_counter_ns()
    gc_after = {'counts': list(gc.get_count()),
                'collections': [entry['collections'] for entry in gc.get_stats()]}
    recorder.seal(producers_stopped=True)
    report = recorder.report()
    if (report['events_complete'] is not True or report['recorded'] != event_count
            or type(report['events']) is not list or len(report['events']) != event_count):
        raise _WindowFailure('recorder did not retain all events', {'recorder_report': report})
    if _TEMPLATES != expected_templates:
        raise _WindowFailure('recorder mutated an input payload', {'recorder_report': report})
    if any(first > second for first, second in zip(source_times, source_times[1:])):
        raise _WindowFailure('source event clock moved backwards',
                             {'source_times_ns': source_times, 'recorder_report': report})
    recorded_times = []
    for index, event in enumerate(report['events']):
        stamp = event.get('time_ns')
        recorded_times.append(stamp)
        expected = {**expected_templates[index % 2], 'time_ns': source_times[index]}
        if (type(stamp) is not int or stamp != source_times[index]
                or not start <= stamp <= end or event != expected):
            raise _WindowFailure('recorded payload or source timestamp differs',
                                 {'event_index': index, 'source_times_ns': source_times,
                                  'recorder_report': report})
    if end <= start:
        raise RuntimeError('clock did not advance')
    return {'start_ns': start, 'end_ns': end, 'elapsed_ns': end - start,
            'event_count': event_count,
            'event_counts': {'selected': event_count // 2, 'forward_entry': event_count // 2},
            'recorded': report['recorded'], 'events_complete': report['events_complete'],
            'payloads_match': True, 'source_times_ns': source_times, 'recorded_times_ns': recorded_times,
            'gc_before': gc_before, 'gc_after': gc_after}


def _mode_orders():
    modes = [('baseline', 'fast')] * 6 + [('fast', 'baseline')] * 5
    random.Random(SEED).shuffle(modes)
    return modes


def _summarize(samples):
    pairs = {}
    for sample in samples:
        rep, name, elapsed, count = (sample['repetition'], sample['variant'],
                                     sample['elapsed_ns'], sample['event_count'])
        if (type(rep) is not int or rep < 0 or name not in ('baseline', 'fast')
                or type(elapsed) is not int or elapsed <= 0 or type(count) is not int or count <= 0):
            raise ValueError('invalid sample')
        if name in pairs.setdefault(rep, {}):
            raise ValueError('duplicate variant in a block')
        pairs[rep][name] = (elapsed, count)
    if not pairs or any(set(pair) != {'baseline', 'fast'}
                        or pair['baseline'][1] != pair['fast'][1] for pair in pairs.values()):
        raise ValueError('each paired block needs the same workload for both variants')
    ordered = [pairs[index] for index in sorted(pairs)]
    differences = [(pair['baseline'][0] - pair['fast'][0]) / pair['baseline'][1] for pair in ordered]
    return {'baseline_median_ns': statistics.median(pair['baseline'][0] for pair in ordered),
            'fast_median_ns': statistics.median(pair['fast'][0] for pair in ordered),
            'paired_baseline_minus_fast_ns_per_event': differences,
            'median_paired_baseline_minus_fast_ns_per_event': statistics.median(differences)}


def run(*, partial=None):
    partial = {} if partial is None else partial
    partial.update({'version': VERSION, 'status': 'started', 'phase': 'preflight', 'samples': []})
    startup = _startup_metadata()
    source = _ab_source_metadata()
    partial.update(startup=startup, source=source)
    shared_inputs = _shared_input_hashes()
    classes, variants = _load_variants()
    clock = time.get_clock_info('perf_counter')
    resolution = max(1, math.ceil(clock.resolution * 1_000_000_000))
    workload = {'seed': SEED, 'repetitions': 11, 'warmup_events_per_variant': WARMUP_EVENTS,
                'events_per_window': EVENTS_PER_WINDOW, 'templates_without_timestamps': _TEMPLATES,
                'order': 'seeded randomized balanced 6/5 paired blocks', 'execution': 'single Python thread',
                'variants': variants, 'shared_schema_source_sha256': SCHEMA_SHA256,
                'shared_schema_mapping_sha256': shared_inputs['schema_mapping_sha256'],
                'templates_sha256': shared_inputs['templates_sha256']}
    payload = json.dumps(workload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    partial['workload'] = workload
    partial['workload_sha256'] = hashlib.sha256(payload).hexdigest()
    partial['runtime'] = {'python': platform.python_version(), 'python_implementation': platform.python_implementation(),
                          'platform': platform.platform(), 'cpu': platform.processor() or None,
                          'architecture': platform.machine() or None, 'logical_cpu_count': os.cpu_count(),
                          'torch_version': str(sys.modules['torch'].__version__), 'clock': 'perf_counter_ns',
                          'clock_implementation': clock.implementation, 'clock_resolution_ns': resolution,
                          'clock_monotonic': clock.monotonic, 'clock_adjustable': clock.adjustable,
                          'gc_enabled': gc.isenabled(), 'gc_thresholds': list(gc.get_threshold()),
                          'external_cpu_activity': 'not monitored'}
    partial['phase'] = 'warmup'
    for name in ('baseline', 'fast'):
        partial['active'] = {'variant': name}
        _measure(classes[name], WARMUP_EVENTS)
    partial['phase'] = 'measurement'
    for repetition, order in enumerate(_mode_orders()):
        for name in order:
            partial['active'] = {'repetition': repetition, 'variant': name}
            sample = _measure(classes[name], EVENTS_PER_WINDOW)
            sample.update(repetition=repetition, variant=name)
            partial['samples'].append(sample)
    partial.update(active=None, phase='postflight')
    if (_ab_source_metadata() != source or _startup_metadata() != startup
            or _shared_input_hashes() != shared_inputs):
        raise RuntimeError('source or startup state changed during measurement')
    screen = _clock_screen(partial['samples'], resolution, monotonic=clock.monotonic, adjustable=clock.adjustable)
    partial.update(status='complete' if screen['passes'] else 'clock_screen_failed', phase='terminal',
                   clock_screen=screen, summary=_summarize(partial['samples']))
    return partial


def _execute_to_file(output):
    output = _output_path(output)
    partial = {}
    try:
        result = run(partial=partial)
    except BaseException as error:
        partial.update(status='failed' if isinstance(error, Exception) else 'cancelled',
                       error_type=type(error).__name__[:128])
        if type(error) is _WindowFailure:
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
    result = _execute_to_file(parser.parse_args().output)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
