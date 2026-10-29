"""Full-journal and eight-family checks using only a fabricated observer."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from looking_glass.bert_matched_evidence import aggregate_families, audit_event_journal, audit_family_events
from looking_glass.bert_matched_runner import _run_family


def _fixture():
    # Path loading works under both pytest's normal and importlib modes.
    path = Path(__file__).with_name('test_bert_matched_runner.py')
    spec = importlib.util.spec_from_file_location('matched_fake_runner_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _family(family):
    fixture = _fixture()
    manifest = fixture.sample_family()
    prefix = f'family-{family}-'
    for row in manifest['rows']:
        row['family'] = family
        row['row_id'] = prefix + row['row_id']
    for target in manifest['targets']:
        target['family'] = family
        target['case_id'] = prefix + target['case_id']
        target['target_row_id'] = prefix + target['target_row_id']
        target['donor_row_ids'] = {key: prefix + value for key, value in target['donor_row_ids'].items()}
    model = SimpleNamespace(distilbert=SimpleNamespace(transformer=SimpleNamespace(layer=[None] * 6)))
    records = []
    summary = _run_family(model, manifest, split='development', family=family, is_id=20, are_id=21,
                          mask_id=7, emit=records.append, observer=fixture.fake_observation)
    return manifest, records, summary


@pytest.fixture(scope='module')
def complete_family():
    return _family(1)


def _bits(value):
    return int.from_bytes(struct.pack('<f', value), 'little')


def _with_known_nonzero_contrasts(manifest, original):
    records = copy.deepcopy(original)
    targets = {row['case_id']: row for row in manifest['targets']}
    rows = {row['row_id']: row for row in manifest['rows']}
    for event in records:
        if event.get('status') != 'completed' or event.get('kind') not in ('B_same', 'B_opposite'):
            continue
        target = targets[event['case_id']]
        number = rows[target['target_row_id']]['subject_number']
        sign = 1 if number == 'singular' else -1
        effect = .25 if event['site'] == 'block0' else .125
        value = .5 if event['kind'] == 'B_same' else .5 - sign * effect
        event['is_bits'], event['is_logit'] = _bits(value), value
        event['are_bits'], event['are_logit'] = _bits(0.0), 0.0
    return records


def test_nonzero_saved_contrasts_use_b_same_and_number_sign(complete_family):
    manifest, original, _ = complete_family
    records = _with_known_nonzero_contrasts(manifest, original)
    result = audit_family_events(manifest, records, split='development', family=1, mask_id=7)
    for case in result['cases']:
        assert case['sites']['block0']['primary_effect'] == .25
        assert case['sites']['block2']['primary_effect'] == .125
        assert case['paired_difference'] == .125
        assert 'A_same' in case['sites']['block0']['margins']
    assert result['family_means']['paired'] == .125


@pytest.mark.parametrize('corruption', (
    'missing', 'extra', 'started_bool', 'reordered', 'donor', 'metadata', 'input',
    'logit', 'nonfinite', 'clean', 'control', 'plan_radius', 'plan_metric',
    'pre', 'post', 'source', 'final', 'failure'))
def test_incomplete_or_mutated_full_journal_is_rejected(complete_family, corruption):
    manifest, original, _ = complete_family
    events = copy.deepcopy(original)
    completed = lambda kind: next(row for row in events if row['status'] == 'completed' and row['kind'] == kind)
    if corruption == 'missing':
        events.pop()
    elif corruption == 'extra':
        events.append({'status': 'extra'})
    elif corruption == 'started_bool':
        events[0]['forward_index'] = True
    elif corruption == 'reordered':
        events[0], events[1] = events[1], events[0]
    elif corruption == 'donor':
        next(row for row in events if row['status'] == 'started' and row['kind'] == 'B_same')['donor_row_id'] = 'wrong'
    elif corruption == 'metadata':
        completed('B_same')['events'][1]['token_index'] = 3
    elif corruption == 'input':
        completed('capture')['model_input_ids'][2] = True
    elif corruption == 'logit':
        completed('B_same')['is_logit'] += .5
    elif corruption == 'nonfinite':
        completed('B_same')['is_bits'] = 0x7f800000
    elif corruption == 'clean':
        next(row for row in events if row['status'] == 'completed' and row['kind'] == 'capture'
             and row['site'] == 'block2')['mask_logits_sha256'] = 'f' * 64
    elif corruption == 'control':
        next(row for row in events if row['status'] == 'control')['passed'] = False
    elif corruption == 'plan_radius':
        next(row for row in events if row['status'] == 'dose_plan')['radius'] += 1
    elif corruption == 'plan_metric':
        next(row for row in events if row['status'] == 'dose_plan')['sites']['block0']['B_same']['fraction'] = .123
    elif corruption == 'pre':
        completed('B_same')['before_float32_le_base64'] = completed('B_opposite')['after_float32_le_base64']
    elif corruption == 'post':
        completed('B_same')['after_float32_le_base64'] = completed('B_same')['before_float32_le_base64']
    elif corruption == 'source':
        completed('B_same')['submitted_float32_le_base64'] = completed('B_same')['before_float32_le_base64']
    elif corruption == 'final':
        completed('final_clean')['before_float32_le_base64'] = completed('B_same')['after_float32_le_base64']
    else:
        next(row for row in events if row['status'] == 'dose_plan')['status'] = 'dose_plan_failed'
    with pytest.raises(ValueError):
        audit_family_events(manifest, events, split='development', family=1, mask_id=7)


@pytest.fixture(scope='module')
def eight_families():
    families, all_records, combined = [], [], {'rows': [], 'targets': []}
    for family in range(1, 9):
        manifest, records, live = _family(family)
        saved = audit_family_events(manifest, records, split='development', family=family, mask_id=7)
        assert saved == live
        modified = _with_known_nonzero_contrasts(manifest, records)
        families.append(audit_family_events(manifest, modified, split='development', family=family, mask_id=7))
        all_records.extend(modified)
        for key in combined:
            combined[key].extend(manifest[key])
    return combined, all_records, families


def test_eight_audited_families_and_mutated_summaries(eight_families):
    _, _, families = eight_families
    result = aggregate_families(families)
    assert result['forward_count'] == 480
    assert len(result['cases']) == 32
    assert len(result['families']) == 8
    assert result['block0_mean'] == .25
    assert result['block2_mean'] == .125
    assert result['paired_mean'] == .125
    assert result['positive_families'] == 8
    assert result['criterion_met'] is True
    for corruption in ('order', 'family_mean', 'site_mean', 'dose', 'radius', 'case_id', 'split'):
        damaged = copy.deepcopy(families)
        if corruption == 'order':
            damaged[0], damaged[1] = damaged[1], damaged[0]
        elif corruption == 'family_mean':
            damaged[0]['family_means']['paired'] += 1.0
        elif corruption == 'site_mean':
            damaged[0]['cases'][0]['sites']['block0']['primary_effect'] += 1.0
        elif corruption == 'dose':
            damaged[0]['cases'][0]['sites']['block0']['doses']['B_same']['relative_error'] += 1.0
        elif corruption == 'radius':
            damaged[0]['cases'][0]['radius'] *= 2.0
        elif corruption == 'case_id':
            damaged[0]['cases'][1]['case_id'] = damaged[0]['cases'][0]['case_id']
        else:
            damaged[1]['split'] = 'other'
        with pytest.raises(ValueError):
            aggregate_families(damaged)


def _jsonl(records):
    return b''.join((json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
                    for record in records)


def _audit_bytes(tmp_path, manifest, raw, *, families, expected=None):
    path = tmp_path / 'fabricated-events.jsonl'
    path.write_bytes(raw)
    return audit_event_journal(manifest, path, split='development', mask_id=7,
        expected_sha256=expected if expected is not None else hashlib.sha256(raw).hexdigest(), family_ids=families)


def test_closed_journal_reconstructs_all_eight_and_single_rehearsal(tmp_path, eight_families):
    manifest, records, families = eight_families
    raw = _jsonl(records)
    result = _audit_bytes(tmp_path, manifest, raw, families=tuple(range(1, 9)))
    assert result['sha256'] == hashlib.sha256(raw).hexdigest()
    assert result['size_bytes'] == len(raw)
    assert result['record_count'] == 1152
    assert result['forward_count'] == 480
    assert result['families'] == families
    assert result['aggregate']['paired_mean'] == .125
    assert result['aggregate']['criterion_met'] is True
    one = _audit_bytes(tmp_path, manifest, _jsonl(records[:144]), families=(1,))
    assert one['forward_count'] == 60
    assert one['record_count'] == 144
    assert one['families'] == families[:1]
    assert one['aggregate'] is None


@pytest.mark.parametrize('corruption', ('hash', 'missing', 'extra', 'order', 'duplicate_key', 'constant',
                                        'invalid_utf8', 'no_final_newline', 'line_limit', 'file_limit', 'families'))
def test_closed_journal_rejects_invalid_framing_and_integrity(tmp_path, complete_family, corruption):
    manifest, records, _ = complete_family
    if corruption == 'hash':
        with pytest.raises(ValueError):
            _audit_bytes(tmp_path, manifest, _jsonl(records), families=(1,), expected='0' * 64)
        return
    raw, families = _jsonl(records), (1,)
    if corruption == 'missing':
        raw = _jsonl(records[:-1])
    elif corruption == 'extra':
        raw += _jsonl([records[-1]])
    elif corruption == 'order':
        raw = _jsonl([records[1], records[0], *records[2:]])
    elif corruption == 'duplicate_key':
        raw = raw.replace(b'"status":', b'"status":"started","status":', 1)
    elif corruption == 'constant':
        raw = raw.replace(b'"family":1', b'"family":NaN', 1)
    elif corruption == 'invalid_utf8':
        raw = b'\xff' + raw
    elif corruption == 'no_final_newline':
        raw = raw[:-1]
    elif corruption == 'line_limit':
        raw = b' ' * 65537 + b'\n' + raw
    elif corruption == 'file_limit':
        raw = b' ' * (16 * 1024 * 1024 + 1)
    else:
        families = (1, 2)
    with pytest.raises(ValueError):
        _audit_bytes(tmp_path, manifest, raw, families=families)
