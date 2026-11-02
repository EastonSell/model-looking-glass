import io
import json
import struct
from pathlib import Path

import pytest
from scripts import copy_behavior_screen_v1 as owner


def sample_case():
    return {"case_id": "fake-1", "token_ids": list(range(11)), "group": 1, "variant": 0, "target_id": 8}


def test_single_clean_row_is_bound_and_retained(tmp_path):
    row = bytes(owner.ROW_BYTES)
    counts = {"submitted": 0, "model_returns": 0, "returned": 0, "retained": 0}
    completed, activity = [], {}
    def forward(model, ids, action, observation):
        observation.update({"model_call_return_observed": True, "payload": {"logits": row}})
        return observation["payload"]
    with (tmp_path / 'journal').open('wb') as log:
        owner.perform_action(sample_case(), 0, model=None, runtime_before={}, path=tmp_path / 'screen.f32',
            log=log, counts=counts, activity=activity, completed=completed, run_forward=forward,
            runtime=lambda m: {})
    events = [json.loads(x) for x in (tmp_path / 'journal').read_bytes().splitlines()]
    assert [e['event'] for e in events] == ['start', 'raw_recorded', 'complete']
    assert (tmp_path / 'screen.f32').read_bytes() == row
    assert completed[0]['raw']['logits']['offset'] == 0
    assert counts == {"submitted": 1, "model_returns": 1, "returned": 1, "retained": len(row)}


def test_returned_payload_on_failure_is_preserved(tmp_path):
    row = b'\0' * owner.ROW_BYTES
    counts = {"submitted": 0, "model_returns": 0, "returned": 0, "retained": 0}
    completed, activity = [], {}
    def forward(model, ids, action, observation):
        observation.update({"model_call_return_observed": True, "payload": {"logits": row}})
        raise RuntimeError('injected after model return')
    with (tmp_path / 'journal').open('wb') as log:
        with pytest.raises(RuntimeError):
            owner.perform_action(sample_case(), 0, model=None, runtime_before={}, path=tmp_path / 'screen.f32',
                log=log, counts=counts, activity=activity, completed=completed, run_forward=forward,
                runtime=lambda m: {})
    assert (tmp_path / 'screen.f32').read_bytes() == row
    assert counts == {"submitted": 1, "model_returns": 1, "returned": 0, "retained": len(row)}
    assert activity['marker']['partial']
    assert activity['observation']['helper_return_unbound_uncertain']
    assert not completed

@pytest.mark.parametrize('needle', [
    'observation["helper_return_observed"] = True',
    'marker["recorded"] = True',
])
def test_interrupted_after_return_retains_row(tmp_path, needle):
    import inspect
    import sys
    row = bytes(owner.ROW_BYTES)
    counts = {"submitted": 0, "model_returns": 0, "returned": 0, "retained": 0}
    activity, completed = {}, []
    def forward(model, ids, action, observation):
        observation.update({"model_call_return_observed": True, "payload": {"logits": row}})
        return observation['payload']
    lines, first = inspect.getsourcelines(owner.perform_action)
    offset = next(i for i, line in enumerate(lines) if needle in line)
    target_line = first + offset
    class Injected(BaseException):
        pass
    def tracer(frame, event, arg):
        if frame.f_code is owner.perform_action.__code__ and event == 'line' and frame.f_lineno == target_line:
            raise Injected()
        return tracer
    with (tmp_path / 'journal').open('wb') as log:
        try:
            sys.settrace(tracer)
            with pytest.raises(Injected):
                owner.perform_action(sample_case(), 0, model=None, runtime_before={}, path=tmp_path / 'screen.f32',
                    log=log, counts=counts, activity=activity, completed=completed, run_forward=forward,
                    runtime=lambda m: {})
        finally:
            sys.settrace(None)
    assert (tmp_path / 'screen.f32').read_bytes() == row
    assert counts == {"submitted": 1, "model_returns": 1, "returned": 1, "retained": len(row)}
    assert activity['marker']['partial']
    assert not completed


def test_partial_write_recovers_returned_row(tmp_path, monkeypatch):
    row = bytes(owner.ROW_BYTES)
    counts = {"submitted": 0, "model_returns": 0, "returned": 0, "retained": 0}
    activity, completed = {}, []
    def forward(model, ids, action, observation):
        observation.update({"model_call_return_observed": True, "payload": {"logits": row}})
        return observation['payload']
    def partial(stream, payload):
        stream.write(payload['logits'][:128])
        stream.flush()
        raise OSError('injected partial write')
    monkeypatch.setattr(owner.base, '_write_payload', partial)
    with (tmp_path / 'journal').open('wb') as log:
        with pytest.raises(OSError):
            owner.perform_action(sample_case(), 0, model=None, runtime_before={}, path=tmp_path / 'screen.f32',
                log=log, counts=counts, activity=activity, completed=completed, run_forward=forward,
                runtime=lambda m: {})
    assert (tmp_path / 'screen.f32').read_bytes() == row
    assert counts == {"submitted": 1, "model_returns": 1, "returned": 1, "retained": len(row)}
    assert activity['marker']['partial']
    assert not completed


def test_journal_capacity_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'MAX_JOURNAL_BYTES', 3)
    with (tmp_path / 'bounded').open('wb') as stream:
        bounded = owner.BoundedJournal(stream)
        assert bounded.write(b'abc') == 3
        with pytest.raises(OSError):
            bounded.write(b'd')
    assert (tmp_path / 'bounded').read_bytes() == b'abc'
