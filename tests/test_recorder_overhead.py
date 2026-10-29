import copy
import json

import pytest

from looking_glass import recorder_overhead


def test_both_modes_run_same_fixed_mixed_payloads_without_event_loss():
    reference = recorder_overhead._measure('discard', 20)
    recorded = recorder_overhead._measure('record', 20)
    assert reference['event_count'] == recorded['event_count'] == 20
    assert reference['event_counts'] == recorded['event_counts'] == {
        'selected': 10, 'forward_entry': 10}
    assert reference['recorded'] is None
    assert recorded['recorded'] == 20 and recorded['events_complete'] is True
    for sample in (reference, recorded):
        assert sample['elapsed_ns'] == sample['end_ns'] - sample['start_ns'] > 0
        for boundary in ('gc_before', 'gc_after'):
            assert len(sample[boundary]['counts']) == len(sample[boundary]['collections']) == 3


def test_summary_recomputes_paired_differences_without_subtracting_gpu_results():
    samples = [
        {'repetition': 0, 'mode': 'discard', 'elapsed_ns': 1000, 'event_count': 10},
        {'repetition': 0, 'mode': 'record', 'elapsed_ns': 1300, 'event_count': 10},
        {'repetition': 1, 'mode': 'record', 'elapsed_ns': 1200, 'event_count': 10},
        {'repetition': 1, 'mode': 'discard', 'elapsed_ns': 1000, 'event_count': 10},
        {'repetition': 2, 'mode': 'discard', 'elapsed_ns': 1000, 'event_count': 10},
        {'repetition': 2, 'mode': 'record', 'elapsed_ns': 1600, 'event_count': 10},
    ]
    result = recorder_overhead._summarize(samples)
    assert result['discard_median_ns'] == 1000
    assert result['record_median_ns'] == 1300
    assert result['paired_incremental_ns_per_event'] == [30, 20, 60]
    assert result['median_paired_incremental_ns_per_event'] == 30


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unequal_count', 'bad_mode'])
def test_summary_rejects_incomplete_or_incomparable_pairs(change):
    samples = [
        {'repetition': 0, 'mode': 'discard', 'elapsed_ns': 1000, 'event_count': 10},
        {'repetition': 0, 'mode': 'record', 'elapsed_ns': 1300, 'event_count': 10},
    ]
    if change == 'missing':
        samples.pop()
    elif change == 'duplicate':
        samples.append(copy.copy(samples[0]))
    elif change == 'unequal_count':
        samples[1]['event_count'] = 11
    else:
        samples[1]['mode'] = 'invented'
    with pytest.raises(ValueError):
        recorder_overhead._summarize(samples)


def test_clock_screen_marks_too_short_windows_without_claiming_accuracy():
    samples = [{'elapsed_ns': 9999}, {'elapsed_ns': 10000}]
    assert recorder_overhead._clock_screen(samples, resolution_ns=100) == {
        'multiplier': 100, 'minimum_window_ns': 10000,
        'monotonic': True, 'nonadjustable': True, 'passes': False}
    samples[0]['elapsed_ns'] = 10000
    assert recorder_overhead._clock_screen(samples, resolution_ns=100)['passes'] is True
    assert not recorder_overhead._clock_screen(samples, resolution_ns=100, monotonic=False)['passes']
    assert not recorder_overhead._clock_screen(samples, resolution_ns=100, adjustable=True)['passes']


def test_seeded_mode_order_is_balanced_and_reproducible():
    orders = recorder_overhead._mode_orders()
    assert orders == recorder_overhead._mode_orders()
    assert len(orders) == 7
    assert orders.count(('discard', 'record')) == 4
    assert orders.count(('record', 'discard')) == 3


def test_atomic_result_publication_never_overwrites(tmp_path):
    path = tmp_path / 'result.json'
    recorder_overhead._save(path, {'status': 'complete'})
    with pytest.raises(FileExistsError):
        recorder_overhead._save(path, {'status': 'replacement'})
    assert 'complete' in path.read_text(encoding='utf-8')


def test_output_must_be_new_absolute_file_outside_checkout(tmp_path):
    new_path = tmp_path / 'sample.json'
    assert recorder_overhead._output_path(new_path) == new_path
    new_path.write_text('existing', encoding='utf-8')
    with pytest.raises(FileExistsError):
        recorder_overhead._output_path(new_path)
    with pytest.raises(ValueError):
        recorder_overhead._output_path('relative.json')
    inside_checkout = recorder_overhead.Path(recorder_overhead.__file__).resolve().parent / 'sample.json'
    with pytest.raises(ValueError):
        recorder_overhead._output_path(inside_checkout)


def test_failed_measurement_publishes_partial_samples_and_preserves_exception(tmp_path, monkeypatch):
    def interrupted(*, partial):
        partial.update(phase='measurement', active={'repetition': 1, 'mode': 'record'},
                       samples=[{'repetition': 0, 'mode': 'discard', 'elapsed_ns': 10}])
        raise ValueError('measurement lost an event')

    monkeypatch.setattr(recorder_overhead, 'run', interrupted)
    output = tmp_path / 'failure.json'
    with pytest.raises(ValueError, match='lost an event'):
        recorder_overhead._execute_to_file(output)
    evidence = json.loads(output.read_text(encoding='utf-8'))
    assert evidence['status'] == 'failed' and evidence['error_type'] == 'ValueError'
    assert len(evidence['samples']) == 1 and evidence['active']['repetition'] == 1
    assert 'summary' not in evidence


def test_recording_loss_preserves_the_recorder_failure_report(monkeypatch):
    class LosingRecorder:
        def __init__(self, **kwargs):
            self.calls = 0

        def record(self, event):
            self.calls += 1
            return self.calls < 3

        def seal(self, *, producers_stopped):
            assert producers_stopped is True

        def report(self):
            return {'events_complete': False, 'recorded': 2, 'lost_capacity': 1, 'events': []}

    monkeypatch.setattr(recorder_overhead, 'BatchedEventRecorder', LosingRecorder)
    with pytest.raises(recorder_overhead._RecorderLoss) as error:
        recorder_overhead._measure('record', 10)
    assert error.value.detail['attempted'] == 3
    assert error.value.detail['recorder_report']['lost_capacity'] == 1
