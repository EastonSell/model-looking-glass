import copy
import hashlib
import json

import pytest

from looking_glass import recorder_ab
from looking_glass.batched_recorder import BatchedEventRecorder


def test_balanced_seeded_schedule_is_fixed():
    order = recorder_ab._mode_orders()
    assert order == recorder_ab._mode_orders()
    assert len(order) == 11
    assert order.count(('baseline', 'fast')) == 6
    assert order.count(('fast', 'baseline')) == 5


def test_window_verifies_real_copied_events_and_original_timestamps():
    sample = recorder_ab._measure(BatchedEventRecorder, 20)
    assert sample['event_count'] == sample['recorded'] == 20
    assert sample['events_complete'] is True
    assert sample['payloads_match'] is True
    assert sample['elapsed_ns'] == sample['end_ns'] - sample['start_ns'] > 0
    assert sample['source_times_ns'] == sample['recorded_times_ns']
    assert len(sample['source_times_ns']) == 20
    assert len(sample['gc_before']['collections']) == len(sample['gc_after']['collections']) == 3


def test_corrupt_recorder_snapshot_invalidates_the_window():
    class CorruptRecorder(BatchedEventRecorder):
        def report(self):
            result = super().report()
            result['events'][0]['job_id'] = 'wrong-job'
            return result

    with pytest.raises(RuntimeError, match='payload'):
        recorder_ab._measure(CorruptRecorder, 20)


def test_changed_but_in_range_timestamp_is_not_accepted():
    class TimestampCorruptor(BatchedEventRecorder):
        def report(self):
            result = super().report()
            result['events'][0]['time_ns'] = result['events'][-1]['time_ns']
            return result

    with pytest.raises(RuntimeError, match='timestamp'):
        recorder_ab._measure(TimestampCorruptor, 20)


def test_missing_snapshot_event_is_not_accepted():
    class MissingEvent(BatchedEventRecorder):
        def report(self):
            result = super().report()
            result['events'].pop()
            return result

    with pytest.raises(RuntimeError, match='retain'):
        recorder_ab._measure(MissingEvent, 20)


def test_backwards_source_event_clock_is_not_accepted(monkeypatch):
    stamps = iter([1000, 1004, 1003, 1010])
    monkeypatch.setattr(recorder_ab.time, 'perf_counter_ns', lambda: next(stamps))
    with pytest.raises(RuntimeError, match='clock moved backwards'):
        recorder_ab._measure(BatchedEventRecorder, 2)


def test_mutating_nested_caller_input_is_not_accepted(monkeypatch):
    class MutatingRecorder(BatchedEventRecorder):
        def record(self, event):
            if event['kind'] == 'forward_entry':
                event['members'][0]['job_id'] = 'changed'
            return super().record(event)

    monkeypatch.setattr(recorder_ab, '_TEMPLATES', copy.deepcopy(recorder_ab._TEMPLATES))
    with pytest.raises(RuntimeError, match='mutated'):
        recorder_ab._measure(MutatingRecorder, 20)


def test_lost_event_retains_bounded_partial_recorder_evidence():
    class LosingRecorder(BatchedEventRecorder):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def record(self, event):
            self.calls += 1
            if self.calls == 3:
                self.mark_incomplete()
                return False
            return super().record(event)

    with pytest.raises(recorder_ab._WindowFailure) as error:
        recorder_ab._measure(LosingRecorder, 20)
    assert error.value.detail['attempted'] == 3
    assert error.value.detail['recorder_report']['recorded'] == 2
    assert error.value.detail['recorder_report']['events_complete'] is False


def test_summary_recomputes_each_paired_difference():
    samples = [
        {'repetition': 0, 'variant': 'baseline', 'elapsed_ns': 1300, 'event_count': 10},
        {'repetition': 0, 'variant': 'fast', 'elapsed_ns': 1000, 'event_count': 10},
        {'repetition': 1, 'variant': 'fast', 'elapsed_ns': 1100, 'event_count': 10},
        {'repetition': 1, 'variant': 'baseline', 'elapsed_ns': 1300, 'event_count': 10},
        {'repetition': 2, 'variant': 'baseline', 'elapsed_ns': 1400, 'event_count': 10},
        {'repetition': 2, 'variant': 'fast', 'elapsed_ns': 1500, 'event_count': 10},
    ]
    result = recorder_ab._summarize(samples)
    assert result['paired_baseline_minus_fast_ns_per_event'] == [30, 20, -10]
    assert result['median_paired_baseline_minus_fast_ns_per_event'] == 20
    assert result['baseline_median_ns'] == 1300
    assert result['fast_median_ns'] == 1100


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unequal_count', 'bad_variant'])
def test_incomplete_or_mismatched_pairs_are_rejected(change):
    samples = [
        {'repetition': 0, 'variant': 'baseline', 'elapsed_ns': 1300, 'event_count': 10},
        {'repetition': 0, 'variant': 'fast', 'elapsed_ns': 1000, 'event_count': 10},
    ]
    if change == 'missing':
        samples.pop()
    elif change == 'duplicate':
        samples.append(copy.copy(samples[0]))
    elif change == 'unequal_count':
        samples[1]['event_count'] = 11
    else:
        samples[1]['variant'] = 'unknown'
    with pytest.raises(ValueError):
        recorder_ab._summarize(samples)


def test_exact_git_blobs_load_as_distinct_classes_with_one_shared_schema():
    classes, metadata = recorder_ab._load_variants()
    assert classes['baseline'] is not classes['fast']
    assert classes['baseline'].__module__ != classes['fast'].__module__
    assert metadata['baseline']['sha256'] == recorder_ab.BASELINE_SHA256
    assert metadata['fast']['sha256'] == recorder_ab.FAST_SHA256
    for cls in classes.values():
        sample = recorder_ab._measure(cls, 4)
        assert sample['recorded'] == 4 and sample['payloads_match']


def test_frozen_templates_and_schema_mapping_must_not_drift(monkeypatch):
    assert recorder_ab._shared_input_hashes() == {
        'templates_sha256': recorder_ab.TEMPLATES_SHA256,
        'schema_mapping_sha256': recorder_ab.SCHEMA_MAPPING_SHA256}
    changed = copy.deepcopy(recorder_ab._TEMPLATES)
    changed[0]['job_id'] = 'different'
    monkeypatch.setattr(recorder_ab, '_TEMPLATES', changed)
    with pytest.raises(RuntimeError, match='changed'):
        recorder_ab._shared_input_hashes()


def test_frozen_schema_mapping_must_not_drift(monkeypatch):
    mapping = dict(recorder_ab.batched_events._EVENT_FIELDS)
    mapping['selected'] = ('kind', 'time_ns')
    monkeypatch.setattr(recorder_ab.batched_events, '_EVENT_FIELDS', mapping)
    with pytest.raises(RuntimeError, match='changed'):
        recorder_ab._shared_input_hashes()


def test_wrong_source_hash_is_rejected_before_code_execution(monkeypatch):
    payload = b'raise AssertionError("untrusted code ran")\n'
    monkeypatch.setattr(recorder_ab, '_git_blob', lambda *args: payload)
    assert hashlib.sha256(payload).hexdigest() != recorder_ab.BASELINE_SHA256
    with pytest.raises(ValueError, match='source hash'):
        recorder_ab._load_variants()


def test_runner_source_is_inventoried_even_without_a_named_module(tmp_path, monkeypatch):
    source = tmp_path / 'recorder_ab.py'
    source.write_bytes(b'example source')
    metadata = {'commit': 'commit', 'sha256': {}}
    monkeypatch.setattr(recorder_ab, '__file__', str(source))
    monkeypatch.setattr(recorder_ab, '_source_metadata', lambda: metadata)
    monkeypatch.setattr(recorder_ab, '_git_blob', lambda *args: b'example source')
    result = recorder_ab._ab_source_metadata()
    assert result['sha256']['looking_glass/recorder_ab.py'] == hashlib.sha256(b'example source').hexdigest()
    monkeypatch.setattr(recorder_ab, '_git_blob', lambda *args: b'different source')
    with pytest.raises(RuntimeError, match='differs'):
        recorder_ab._ab_source_metadata()


def test_failure_publication_retains_prior_samples_without_false_summary(tmp_path, monkeypatch):
    def broken(*, partial):
        partial.update(phase='measurement', active={'repetition': 1, 'variant': 'fast'},
                       samples=[{'repetition': 0, 'variant': 'baseline', 'elapsed_ns': 20}])
        raise RuntimeError('injected loss')

    monkeypatch.setattr(recorder_ab, 'run', broken)
    output = tmp_path / 'result.json'
    with pytest.raises(RuntimeError, match='injected loss'):
        recorder_ab._execute_to_file(output)
    data = json.loads(output.read_text(encoding='utf-8'))
    assert data['status'] == 'failed' and data['error_type'] == 'RuntimeError'
    assert len(data['samples']) == 1 and 'summary' not in data
    with pytest.raises(FileExistsError):
        recorder_ab._execute_to_file(output)
