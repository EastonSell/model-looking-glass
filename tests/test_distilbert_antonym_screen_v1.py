"""Model-free screen selection against fabricated full logit rows."""
import importlib
import importlib.util
import struct

import pytest
from looking_glass import distilbert_color_edit_records_v1 as records


def _module():
    name = 'looking_glass.distilbert_antonym_screen_v1'
    assert importlib.util.find_spec(name) is not None, 'antonym screen reader is not implemented'
    return importlib.import_module(name)


def _fixture():
    order = ['box', 'rope', 'drink', 'room']
    cases, events = [], []
    for obj in order:
        for role in ('discovery', 'calibration-01', 'calibration-02'):
            row = {'id': f'{obj}-{role}', 'object': obj, 'role': role,
                   'factual': 'factual', 'counterfactual': 'cue', 'wrong_target': 'red',
                   'accepted': ['factual'], 'word_ids': {'factual': 0, 'cue': 1, 'red': 2},
                   'input_ids': [101, 103, 102], 'attention_mask': [1, 1, 1], 'mask_index': 1}
            cases.append(row)
            action = len(cases) - 1
            binding = {'action': action, 'phase': 'screen', 'object': obj, 'case_id': row['id'],
                       'condition': 'original', 'input_ids': row['input_ids'][:],
                       'attention_mask': row['attention_mask'][:], 'mask_index': 1}
            raw = struct.pack('<5f', 2., 0., -2., -3., -4.)
            events.append({'seq': 2 * action, 'kind': 'start', **binding})
            events.append({'seq': 2 * action + 1, 'kind': 'complete', **binding,
                           'parameter': None, 'parameter_sha256': None, 'realized_norm': 0.,
                           'logits': records.encode_raw(raw)})
    return {'candidate_order': order, 'cases': cases}, events


def _set_logits(events, index, *values):
    events[2 * index + 1]['logits'] = records.encode_raw(struct.pack('<5f', *values))


def test_calibration_third_token_top_does_not_fail_pairwise_gate():
    manifest, events = _fixture()
    _set_logits(events, 1, 1., 0., 2., -3., -4.)
    result = _module().rebuild_screen(manifest, events, vocabulary=5)
    assert result['selected_object'] == 'box'
    assert result['rows'][1]['unique_top'] == 2
    assert result['rows'][1]['eligible'] is True


def test_discovery_top_tie_excludes_first_candidate():
    manifest, events = _fixture()
    _set_logits(events, 0, 2., 0., 2., -3., -4.)
    result = _module().rebuild_screen(manifest, events, vocabulary=5)
    assert result['candidate_eligible']['box'] is False
    assert result['selected_object'] == 'rope'


def test_strict_decimal_threshold_uses_exact_saved_f32_values():
    manifest, events = _fixture()
    below = struct.unpack('<f', bytes.fromhex('cccccc3d'))[0]
    above = struct.unpack('<f', bytes.fromhex('cdcccc3d'))[0]
    _set_logits(events, 2, below, 0., -1., -2., -3.)
    result = _module().rebuild_screen(manifest, events, vocabulary=5)
    assert result['rows'][2]['eligible'] is False
    _set_logits(events, 2, above, 0., -1., -2., -3.)
    result = _module().rebuild_screen(manifest, events, vocabulary=5)
    assert result['rows'][2]['eligible'] is True


def test_contradictory_completion_input_is_rejected():
    manifest, events = _fixture()
    events[1]['input_ids'] = [101, 104, 102]
    with pytest.raises(ValueError, match='binding'):
        _module().rebuild_screen(manifest, events, vocabulary=5)


def test_screen_runs_all_twelve_fixed_rows_in_order():
    manifest, _ = _fixture()
    calls = []

    class Engine:
        def run(self, journal, row, *, phase, condition):
            calls.append((row['id'], phase, condition))

    run_screen = getattr(_module(), 'run_screen', None)
    assert callable(run_screen), 'antonym screen schedule is not implemented'
    run_screen(Engine(), None, manifest)
    assert calls == [(row['id'], 'screen', 'original') for row in manifest['cases']]
