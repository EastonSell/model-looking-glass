import copy
import inspect
import json
import sys
import threading

import pytest

from looking_glass.batched_events import assemble_window
from looking_glass.batched_recorder import BatchedEventRecorder
from tests.test_batched_events import _fixture


def _seal(recorder):
    recorder.seal(producers_stopped=True)
    return recorder.report()


def test_recorded_observations_are_detached_and_assemble_without_inferred_times():
    protocol, digest, contexts, _, events, _ = _fixture()
    saved = copy.deepcopy(events)
    recorder = BatchedEventRecorder(max_events=100, max_wire_bytes=500_000)
    for event in events:
        assert recorder.record(event) is True
    next(event for event in events if event['kind'] == 'forward_entry')['members'][0]['job_id'] = 'changed'
    result = _seal(recorder)
    assert result['events_complete'] is True
    assert result['events'] == saved
    assert result['recorded'] == len(saved)
    window = assemble_window(protocol, expected_protocol_sha256=digest,
                             context={**contexts[1], 'events_complete': result['events_complete']},
                             events=result['events'])
    assert window['calls'][0]['launch_ns'] == 220
    assert window['calls'][0]['finish_ns'] == 240
    result['events'].clear()
    assert recorder.report()['events'] == saved


def test_recorded_primitive_shape_does_not_replace_offline_semantic_validation():
    protocol, digest, contexts, _, events, _ = _fixture()
    next(event for event in events if event['kind'] == 'selected')['job_id'] = 42
    recorder = BatchedEventRecorder(max_events=100)
    assert all(recorder.record(event) for event in events)
    result = _seal(recorder)
    assert result['events_complete'] is True
    with pytest.raises(ValueError):
        assemble_window(protocol, expected_protocol_sha256=digest,
                        context={**contexts[1], 'events_complete': result['events_complete']},
                        events=result['events'])


def test_overflow_is_sticky_and_preserves_accepted_partial_records():
    recorder = BatchedEventRecorder(max_events=2, max_wire_bytes=4096)
    for number in range(2):
        assert recorder.record({'kind': 'selected', 'time_ns': number, 'job_id': str(number)})
    assert not recorder.record({'kind': 'selected', 'time_ns': 2, 'job_id': 'lost'})
    result = _seal(recorder)
    assert result['events_complete'] is False
    assert result['lost_capacity'] == 1
    assert result['recorded'] == 2
    assert [event['job_id'] for event in result['events']] == ['0', '1']


def test_wire_budget_is_conservative_for_escaped_unicode_and_nested_members():
    recorder = BatchedEventRecorder(max_events=10, max_wire_bytes=4000)
    event = {'kind': 'forward_entry', 'time_ns': 1, 'call_id': 'a\n\U0001f4a1', 'group_id': 'g',
             'profile_sha256': 'a' * 64, 'input_shape': [1, 2],
             'members': [{'job_id': 'j', 'team_id': 'é', 'broker_id': 'b',
                          'broker_job_id': 'b-j-1', 'row_offset': 0}]}
    assert recorder.record(event)
    result = _seal(recorder)
    encoded = json.dumps({'events': result['events']}, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    assert len(encoded) <= result['wire_bytes_upper_bound'] <= result['max_wire_bytes']
    small = BatchedEventRecorder(max_events=10, max_wire_bytes=128)
    assert not small.record(event)
    assert _seal(small)['lost_capacity'] == 1


@pytest.mark.parametrize('bad', [
    {'kind': 'selected', 'time_ns': True, 'job_id': 'j'},
    {'kind': 'selected', 'time_ns': -1, 'job_id': 'j'},
    {'kind': 'selected', 'time_ns': 2 ** 63, 'job_id': 'j'},
    {'kind': 'selected', 'time_ns': 1, 'job_id': 'x' * 513},
    {'kind': 'selected', 'time_ns': 1, 'job_id': '\ud800'},
    {'kind': 'selected', 'time_ns': 1, 'job_id': 'j', 'extra': None},
    {'kind': 'selected', 'time_ns': 1},
    {'kind': 'invented', 'time_ns': 1},
])
def test_invalid_events_are_dropped_without_propagating(bad):
    recorder = BatchedEventRecorder()
    assert recorder.record(bad) is False
    result = _seal(recorder)
    assert result['events_complete'] is False
    assert result['lost_invalid'] == 1
    assert result['events'] == []


def test_dynamic_container_subclasses_are_not_invoked():
    class Hostile(dict):
        def items(self):
            raise AssertionError('user code must not run')

    recorder = BatchedEventRecorder()
    assert not recorder.record(Hostile(kind='selected', time_ns=1, job_id='j'))
    assert _seal(recorder)['lost_invalid'] == 1


def test_non_string_keys_are_rejected_before_any_hash_collision_callback():
    class HostileKey:
        def __hash__(self):
            return hash('kind')

        def __eq__(self, other):
            raise AssertionError('the recorder must not compare this key')

    recorder = BatchedEventRecorder()
    assert not recorder.record({HostileKey(): 'selected'})
    assert _seal(recorder)['lost_invalid'] == 1


@pytest.mark.parametrize('identifier', ['ascii-identifier', 'é', '\U0001f4a1', 'a\n\t'])
def test_ascii_and_valid_unicode_are_preserved(identifier):
    recorder = BatchedEventRecorder()
    assert recorder.record({'kind': 'selected', 'time_ns': 1, 'job_id': identifier})
    assert _seal(recorder)['events'][0]['job_id'] == identifier


@pytest.mark.parametrize('identifier', ['ascii\ud800', 'é\ud800', '\U0001f4a1\udfff'])
def test_surrogates_are_rejected_even_in_mixed_unicode(identifier):
    recorder = BatchedEventRecorder()
    assert not recorder.record({'kind': 'selected', 'time_ns': 1, 'job_id': identifier})
    assert _seal(recorder)['lost_invalid'] == 1


def test_observer_errors_are_sticky_but_regular_failures_do_not_escape(monkeypatch):
    recorder = BatchedEventRecorder()
    event = {'kind': 'selected', 'time_ns': 1, 'job_id': 'j'}

    def failed(*args):
        raise MemoryError('injected')

    monkeypatch.setattr('looking_glass.batched_recorder._copy_event', failed)
    assert recorder.record(event) is False
    recorder.mark_incomplete()
    result = _seal(recorder)
    assert result['events_complete'] is False
    assert result['lost_internal'] == 1
    assert result['explicit_faults'] == 1


def test_interruption_marks_incomplete_and_remains_an_interruption(monkeypatch):
    recorder = BatchedEventRecorder()

    def interrupted(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr('looking_glass.batched_recorder._copy_event', interrupted)
    with pytest.raises(KeyboardInterrupt):
        recorder.record({'kind': 'selected', 'time_ns': 1, 'job_id': 'j'})
    result = _seal(recorder)
    assert result['events_complete'] is False
    assert result['lost_internal'] == 1


@pytest.mark.parametrize('error_type', [MemoryError, KeyboardInterrupt])
def test_failure_at_lock_entry_is_latched_before_report(error_type):
    recorder = BatchedEventRecorder()
    lines, first = inspect.getsourcelines(BatchedEventRecorder._record_once)
    target = first + next(index for index, line in enumerate(lines)
                          if 'was_complete = self._complete' in line)

    def trace(frame, event, arg):
        if (event == 'line' and frame.f_code is BatchedEventRecorder._record_once.__code__
                and frame.f_lineno == target):
            sys.settrace(None)
            raise error_type('injected at locked bookkeeping')
        return trace

    sys.settrace(trace)
    try:
        if error_type is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt):
                recorder.record({'kind': 'selected', 'time_ns': 1, 'job_id': 'j'})
        else:
            assert not recorder.record({'kind': 'selected', 'time_ns': 1, 'job_id': 'j'})
    finally:
        sys.settrace(None)
    result = _seal(recorder)
    assert result['events_complete'] is False and result['lost_internal'] == 1


def test_explicit_quiescence_is_required_and_late_event_invalidates_subsequent_report():
    recorder = BatchedEventRecorder()
    with pytest.raises(ValueError):
        recorder.seal(producers_stopped=False)
    with pytest.raises(RuntimeError):
        recorder.report()
    recorder.seal(producers_stopped=True)
    assert recorder.report()['events_complete'] is True
    assert not recorder.record({'kind': 'selected', 'time_ns': 1, 'job_id': 'late'})
    report = recorder.report()
    assert report['events_complete'] is False and report['late_records'] == 1


def test_concurrent_producers_keep_all_observed_records():
    recorder = BatchedEventRecorder(max_events=1200)
    errors = []

    def producer(team):
        for number in range(100):
            if not recorder.record({'kind': 'selected', 'time_ns': number,
                                    'job_id': f'{team}-{number}'}):
                errors.append((team, number))

    threads = [threading.Thread(target=producer, args=(number,)) for number in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    report = _seal(recorder)
    assert not errors and report['events_complete'] is True
    assert report['recorded'] == 600
    assert len({event['job_id'] for event in report['events']}) == 600


@pytest.mark.parametrize('kwargs', [
    {'max_events': True}, {'max_events': 0}, {'max_events': 4097},
    {'max_wire_bytes': False}, {'max_wire_bytes': 0}, {'max_wire_bytes': 1024 * 1024 + 1},
])
def test_limits_are_exact_bounded_integers(kwargs):
    with pytest.raises(ValueError):
        BatchedEventRecorder(**kwargs)
