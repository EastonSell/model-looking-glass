"""Fabricated, model-free checks of the independent subject-study reader."""
from collections import Counter
import copy
import hashlib
import json
import math
import os
import py_compile
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from looking_glass import agreement_subject_science_v1 as reader
from looking_glass import agreement_science_v1 as science


def cases():
    conditions = ('none', 'mass', 'attractor_singular', 'attractor_plural')
    pairs = (('R1', 'I1'), ('R1', 'I2'), ('R2', 'I1'), ('R2', 'I2'))
    rows = []
    for frame in range(4):
        for condition, label in enumerate(conditions):
            for pair, name in enumerate(pairs[frame]):
                for number in ('singular', 'plural'):
                    rows.append({'case_id': f'{frame}-{condition}-{name}-{number}',
                        'frame_index': frame, 'condition_index': condition, 'condition': label,
                        'pair_index': pair, 'subject_key': name, 'subject_number': number,
                        'role': 'target', 'morphology': ('regular', 'irregular')[pair],
                        'token_ids': [10 + frame, 20 + pair, 30 + (number == 'plural'), 40 + condition],
                        'subject_index': 2, 'final_index': 3})
            rows.append({'case_id': f'{frame}-{condition}-context', 'frame_index': frame,
                'condition_index': condition, 'condition': label, 'pair_index': None,
                'subject_key': None, 'subject_number': None, 'role': 'context', 'morphology': None,
                'token_ids': [10 + frame, 50 + condition], 'subject_index': None, 'final_index': 1})
    return rows


def measurements(rows, *, differing=False):
    result = {}
    for row in rows:
        number = row['subject_number']
        clean = 1.0 if number == 'singular' else -1.0 if number == 'plural' else 0.0
        magnitude = 2.0 if differing and number == 'singular' else 1.0
        margins = {'clean': {'present': clean}}
        for direction, odd in (('u', magnitude), ('v1', 0.25), ('v2', 0.125)):
            for sign, multiplier in (('plus', 1.0), ('minus', -1.0)):
                margins[f'{direction}_{sign}'] = {'present': clean + multiplier * odd}
        result[row['case_id']] = {'margins': margins,
            'capture': (1.0, 0.0, 0.0) if number == 'singular' else (0.0, 0.0, 0.0)}
    return result


def test_schedule_preserves_720_actions_balanced_slots_and_pair_order():
    rows = cases()
    schedule = reader.expected_schedule(rows)
    assert len(schedule) == 720
    assert [entry['action'] for entry in schedule[:9]] == [
        'clean', 'noop_l3_final', 'u_plus', 'u_minus', 'v1_plus', 'v1_minus',
        'v2_plus', 'v2_minus', 'clean_after']
    counts = Counter((entry['action'], entry['slot']) for entry in schedule if entry['slot'] is not None)
    assert len(counts) == 36 and set(counts.values()) == {13, 14}
    assert len({entry['case_id'] for entry in schedule}) == 80
    changed = copy.deepcopy(rows)
    changed[0], changed[1] = changed[1], changed[0]
    with pytest.raises(ValueError):
        reader.expected_schedule(changed)


def test_case_alignment_requires_only_the_subject_or_last_noun_to_change():
    rows = cases()
    assert reader.validate_cases(rows, vocabulary=600) is True
    changed = copy.deepcopy(rows)
    changed[1]['token_ids'][1] += 10
    with pytest.raises(ValueError):
        reader.validate_cases(changed, vocabulary=600)
    changed = copy.deepcopy(rows)
    changed[5]['token_ids'][-2] += 10
    with pytest.raises(ValueError):
        reader.validate_cases(changed, vocabulary=600)


def test_constant_additive_response_can_pass_general_check_but_fail_specific_reference():
    rows = cases()
    directions = {'u': (1.0, 0.0, 0.0), 'v1': (0.0, 1.0, 0.0), 'v2': (0.0, 0.0, 1.0)}
    result = reader.score_cases(rows, measurements(rows), directions, reference_odd=0.5)
    assert result['baseline']['adequate'] is True
    assert result['constant_additive']['passes'] is True
    assert result['constant_additive']['mean_abs_k'] == 0.0
    assert result['specific_reference']['passes'] is False
    assert result['specific_reference']['mean_abs_error'] == 0.5
    assert result['sensitivity']['passes'] is False
    assert len(result['pairs']) == 32 and len(result['fragments']) == 16


def test_subject_response_and_geometry_criteria_use_all_fixed_cells():
    rows = cases()
    directions = {'u': (1.0, 0.0, 0.0), 'v1': (0.0, 1.0, 0.0), 'v2': (0.0, 0.0, 1.0)}
    result = reader.score_cases(rows, measurements(rows, differing=True), directions, reference_odd=1.0)
    assert result['sensitivity']['passes'] is True
    assert result['sensitivity']['mean_k_u'] == 1.0
    assert result['sensitivity']['positive_zero_negative'] == [32, 0, 0]
    assert result['geometry']['passes'] is True
    assert result['geometry']['mean_g_u'] == 1.0
    assert result['constant_additive']['passes'] is False


def test_baseline_reports_individual_subject_and_pair_coverage_with_ties_incorrect():
    rows = cases()
    observed = measurements(rows)
    observed[rows[0]['case_id']]['margins']['clean']['present'] = 0.0
    directions = {'u': (1.0, 0.0, 0.0), 'v1': (0.0, 1.0, 0.0), 'v2': (0.0, 0.0, 1.0)}
    baseline = reader.score_cases(rows, observed, directions, reference_odd=0.5)['baseline']
    assert baseline['correct_pairs'] == 31
    assert baseline['individual_by_number']['singular'] == {'correct': 31, 'total': 32}
    assert baseline['individual_by_number']['plural'] == {'correct': 32, 'total': 32}
    assert baseline['by_group']['condition']['none'] == 7


def test_nonlinear_input_gain_can_produce_a_positive_k_without_proving_grammar():
    rows = cases()
    observed = measurements(rows)
    for row in rows:
        if row['role'] != 'target':
            continue
        x = 1.0 if row['subject_number'] == 'singular' else 0.0
        margins = observed[row['case_id']]['margins']
        offset = margins['clean']['present'] - 0.25 * x * x
        margins['u_plus']['present'] = 0.25 * (x + 2.0) ** 2 + offset
        margins['u_minus']['present'] = 0.25 * (x - 2.0) ** 2 + offset
    directions = {'u': (1.0, 0.0, 0.0), 'v1': (0.0, 1.0, 0.0), 'v2': (0.0, 0.0, 1.0)}
    result = reader.score_cases(rows, observed, directions, reference_odd=0.5)
    assert result['sensitivity']['passes'] is True
    assert all(pair['k']['u'] == 1.0 for pair in result['pairs'])


def test_baseline_limited_label_and_fixed_group_aggregates_keep_raw_results():
    rows = cases()
    observed = measurements(rows, differing=True)
    for row in rows:
        if row['role'] == 'target':
            observed[row['case_id']]['margins']['clean']['present'] = 0.0
    directions = {'u': (1.0, 0.0, 0.0), 'v1': (0.0, 1.0, 0.0), 'v2': (0.0, 0.0, 1.0)}
    result = reader.score_cases(rows, observed, directions, reference_odd=0.5)
    assert result['sensitivity']['passes'] is True
    assert result['sensitivity']['interpretation'] == 'baseline_limited'
    assert result['condition_aggregates']['none']['pair_count'] == 8
    assert result['condition_aggregates']['none']['case_counts'] == {'target': 16, 'context': 4}
    assert result['condition_aggregates']['none']['mean_k']['u'] == 1.0
    assert len(result['frame_subject_aggregates']) == 8
    assert all(group['pair_count'] == 4 for group in result['frame_subject_aggregates'])


def test_centered_cosine_is_shift_invariant_and_constant_vectors_are_undefined():
    baseline = reader.centered_cosine((1.0, 3.0, -1.0), (2.0, 0.0, 5.0))
    shifted = reader.centered_cosine((8.0, 10.0, 6.0), (-2.0, -4.0, 1.0))
    assert baseline['cosine'] == pytest.approx(shifted['cosine'], abs=1e-15)
    assert reader.centered_cosine((1.0, 1.0, 1.0), (0.0, 1.0, 2.0))['cosine'] is None


def synthetic_journal(directory):
    """Make an on-disk journal from simple f32 vectors, without a model."""
    rows = cases()
    schedule = reader.expected_schedule(rows)
    directions = {'u': (1.0, 0.0, 0.0), 'v1': (0.0, 1.0, 0.0), 'v2': (0.0, 0.0, 1.0)}
    binding = {'prelaunch_sha256': '1' * 64, 'reservation_sha256': '2' * 64,
               'latest_exposure_sha256': '3' * 64}
    directory.mkdir()
    by_id = {row['case_id']: row for row in rows}
    placement = {name: [sum(entry['action'] == name and entry['slot'] == slot for entry in schedule)
                        for slot in range(6)] for name in
                 ('u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus')}
    schedule_sha = hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
    contract = {'kind': 'run_contract', 'expected_forwards': 720, 'seed': 20261030,
        'environment': {}, 'execution_configuration': [], 'model_state_sha256': '4' * 64,
        'external_binding': binding, 'schedule_sha256': schedule_sha,
        'arm_placement_counts': placement, 'evidence_budget': {}}
    records = [{'index': 0, 'event': contract, 'tensors': {}}]
    def save(raw, shape):
        digest = hashlib.sha256(raw).hexdigest()
        path = directory / (digest + '.f32le')
        if not path.exists():
            path.write_bytes(raw)
        return {'sha256': digest, 'shape': list(shape), 'dtype': 'float32-le', 'bytes': len(raw)}
    for action in schedule:
        row = by_id[action['case_id']]
        number = row['subject_number']
        clean = 1.0 if number == 'singular' else -1.0 if number == 'plural' else 0.0
        arm = action['action']
        change = 0.0
        if arm.endswith(('_plus', '_minus')):
            direction, sign = arm.rsplit('_', 1)
            change = {'u': 1.0, 'v1': 0.25, 'v2': 0.125}[direction] * (1 if sign == 'plus' else -1)
        logits = [0.0] * 600
        logits[318] = clean + change
        raw_logits = struct.pack('<600f', *logits)
        refs = {'last_logits': save(raw_logits, (600,))}
        event = {'kind': 'forward', **action, 'input_ids': list(row['token_ids']),
                 'attention_mask': [1] * len(row['token_ids']), 'plan_metadata': {'steps': []},
                 'all_sequence_logits_finite': True, 'dose': None}
        if arm not in ('clean', 'clean_after'):
            before = struct.pack('<3f', 1.0 if number == 'singular' else 0.0, 0.0, 0.0)
            if arm == 'noop_l3_final':
                after, dose = before, None
            else:
                direction, sign = arm.rsplit('_', 1)
                amount = 2.0 if sign == 'plus' else -2.0
                after, dose = science.direction_delivery(before, directions[direction], amount)
                values, original = struct.unpack('<3f', after), struct.unpack('<3f', before)
                projection = math.fsum((a - b) * d for a, b, d in zip(values, original, directions[direction]))
                dose['signed_projected_displacement'] = projection
                dose['projected_relative_error'] = abs(abs(projection) - 2.0) / 2.0
            prefix = bytes((len(row['token_ids']) - 1) * 3 * 4)
            refs.update({'before': save(before, (3,)), 'after': save(after, (3,)),
                'whole_before': save(prefix + before, (1, len(row['token_ids']), 3)),
                'whole_after': save(prefix + after, (1, len(row['token_ids']), 3))})
            site, plan = science.expected_hook_plan(arm, sequence_length=len(row['token_ids']), width=3)
            event.update(site=site, donor=None, plan_metadata=plan, dose=dose)
        records.append({'index': len(records), 'event': event, 'tensors': refs})
    reference = [0.0] * 600
    reference[318] = 0.25
    return rows, records, directions, tuple(reference), binding


def test_saved_journal_replay_checks_all_actions_and_controls(tmp_path):
    rows, records, directions, reference, binding = synthetic_journal(tmp_path / 'tensors')
    report, seen = reader.audit_records(records, rows, tmp_path / 'tensors', directions, reference,
                                        expected_binding=binding, width=3, vocabulary=600,
                                        allow_synthetic=True)
    assert len(report['cases']) == 80 and len(report['pairs']) == 32
    assert report['constant_additive']['passes'] is True
    assert report['specific_reference']['passes'] is False
    assert report['baseline']['adequate'] is True
    assert report['source_tensor_files'] == len(seen)


@pytest.mark.parametrize('mutation', ['missing', 'order', 'noop', 'dose', 'delivery'])
def test_incomplete_or_rehashed_wrong_saved_witness_is_not_a_result(tmp_path, mutation):
    rows, records, directions, reference, binding = synthetic_journal(tmp_path / 'tensors')
    if mutation == 'missing':
        records.pop()
    elif mutation == 'order':
        records[3]['event']['action'] = 'v2_minus'
    elif mutation == 'noop':
        records[2]['tensors']['whole_after'] = records[3]['tensors']['whole_after']
    elif mutation == 'dose':
        records[3]['event']['dose']['signed_projected_displacement'] *= -1
    else:
        records[3]['tensors']['after'] = records[4]['tensors']['after']
        records[3]['tensors']['whole_after'] = records[4]['tensors']['whole_after']
    with pytest.raises(ValueError):
        reader.audit_records(records, rows, tmp_path / 'tensors', directions, reference,
                             expected_binding=binding, width=3, vocabulary=600, allow_synthetic=True)


def write_attempt(root):
    root.mkdir()
    rows, records, directions, reference, binding = synthetic_journal(root / 'tensors')
    dump = lambda value: (json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')
    events = b''.join(dump(record) for record in records)
    tensor_bytes = sum(path.stat().st_size for path in (root / 'tensors').iterdir())
    (root / 'events.jsonl').write_bytes(events)
    (root / 'started.json').write_bytes(dump({'scope': 'public prospective agreement subject-dependence',
        'external_binding': binding, 'source_revision': 'e' * 40}))
    (root / 'terminal.json').write_bytes(dump({'status': 'complete_candidate', 'events': 721,
        'event_log_bytes': len(events), 'retained_tensor_bytes': tensor_bytes, 'storage_accounting_complete': True,
        'detail': {'status': 'complete_candidate', 'forwards': 720,
            'model_state_sha256_before_after': '4' * 64, 'retained_tensor_bytes': tensor_bytes,
            'schedule_sha256': records[0]['event']['schedule_sha256']}}))
    return rows, directions, reference, binding


def test_closed_attempt_replay_checks_terminal_and_exact_tensor_inventory(tmp_path):
    root = tmp_path / 'attempt'
    rows, directions, reference, binding = write_attempt(root)
    report = reader.audit_attempt(root, rows, directions, reference, expected_binding=binding,
                                  expected_source_revision='e' * 40, width=3, vocabulary=600,
                                  allow_synthetic=True)
    assert report['journal']['status'] == 'consistent_candidate'
    assert report['journal']['events'] == 721
    (root / 'tensors' / ('f' * 64 + '.f32le')).write_bytes(b'unknown')
    with pytest.raises(ValueError):
        reader.audit_attempt(root, rows, directions, reference, expected_binding=binding,
                             expected_source_revision='e' * 40, width=3, vocabulary=600,
                             allow_synthetic=True)


def test_failed_terminal_cannot_be_promoted_as_a_scientific_result(tmp_path):
    root = tmp_path / 'attempt'
    rows, directions, reference, binding = write_attempt(root)
    terminal = json.loads((root / 'terminal.json').read_bytes())
    terminal['status'] = 'failed'
    (root / 'terminal.json').write_text(json.dumps(terminal))
    with pytest.raises(ValueError):
        reader.audit_attempt(root, rows, directions, reference, expected_binding=binding,
                             expected_source_revision='e' * 40, width=3, vocabulary=600,
                             allow_synthetic=True)


def test_public_input_loader_rejects_changed_case_bytes_before_audit(tmp_path, monkeypatch):
    direction_dir = tmp_path / 'directions'
    direction_dir.mkdir()
    shas = {}
    for index, name in enumerate(('u', 'v1', 'v2')):
        vector = [0.0] * 768
        vector[index] = 1.0
        raw = struct.pack('<768d', *vector)
        (direction_dir / (name + '.f64le')).write_bytes(raw)
        shas[name] = hashlib.sha256(raw).hexdigest()
    reference = struct.pack('<50257d', *([0.0] * 50257))
    reference_path = tmp_path / 'reference.f64le'
    reference_path.write_bytes(reference)
    protocol = '4' * 64
    payload = {'schema': 'distilgpt2-agreement-subject-dependence-cases-v1',
               'protocol_sha256': protocol, 'reference_sha256': hashlib.sha256(reference).hexdigest(),
               'cases': cases()}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('ascii')
    case_path = tmp_path / 'cases.json'
    case_path.write_bytes(raw)
    monkeypatch.setattr(reader, '_CASE_SHA256', hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(reader, '_REFERENCE_SHA256', hashlib.sha256(reference).hexdigest())
    monkeypatch.setattr(reader, '_PROTOCOL_SHA256', protocol)
    monkeypatch.setattr(reader, '_DIRECTION_SHAS', shas)
    rows, directions, observed_reference = reader.load_frozen_inputs(case_path, direction_dir, reference_path)
    assert len(rows) == 80 and len(observed_reference) == 50257
    assert directions['u'][0] == 1.0
    case_path.write_bytes(raw + b' ')
    with pytest.raises(ValueError):
        reader.load_frozen_inputs(case_path, direction_dir, reference_path)


def test_pinned_helper_ignores_a_same_size_same_mtime_stale_bytecode(tmp_path):
    package = tmp_path / 'synthetic_package'
    package.mkdir()
    (package / '__init__.py').write_bytes(b'')
    (package / 'reader.py').write_bytes(Path(reader.__file__).read_bytes())
    helper = package / 'agreement_science_v1.py'
    genuine = Path(science.__file__).read_bytes()
    fake_prefix = b'MARKER = "stale_bytecode"\n'
    helper.write_bytes(fake_prefix + b' ' * (len(genuine) - len(fake_prefix)))
    stamp = helper.stat().st_mtime_ns
    py_compile.compile(str(helper), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    helper.write_bytes(genuine)
    os.utime(helper, ns=(stamp, stamp))
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    process = subprocess.run([sys.executable, '-B', '-c',
        "from synthetic_package import reader; print(getattr(reader.science, 'MARKER', 'verified_source'))"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == 'verified_source'


def test_source_first_entry_executes_only_the_exact_expected_reader_bytes(tmp_path):
    source = tmp_path / 'reader.py'
    marker = tmp_path / 'executed'
    raw = ("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('ran')\n").encode()
    source.write_bytes(raw)
    bootstrap = Path(reader.__file__).parents[1] / 'tools/audit_agreement_subject_v1.py'
    command = [sys.executable, '-I', '-S', '-B', str(bootstrap), str(source), hashlib.sha256(raw).hexdigest()]
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert marker.is_file()
    assert marker.read_text() == 'ran'
    marker.unlink()
    source.write_bytes(raw + b' ')
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert not marker.exists()
