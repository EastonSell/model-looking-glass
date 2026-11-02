"""Replay and challenge a completely fabricated 612-forward saved journal."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from looking_glass import agreement_invariant_science_v1 as reader
from looking_glass.agreement_journal import AgreementJournal
from tests.test_agreement_invariant_science_v1 import fabricated_case_rows, fabricated_margins


NUMERIC = Path(__file__).resolve().parents[1] / 'looking_glass/agreement_science_v1.py'


def test_numeric_helper_requires_exact_source_bytes(tmp_path):
    numeric = reader.load_numeric(NUMERIC)
    assert numeric.decode_float32(struct.pack('<f', 1.0), expected_count=1) == (1.0,)
    wrong = tmp_path / 'wrong.py'
    wrong.write_text('raise RuntimeError("must not execute")', encoding='utf-8')
    with pytest.raises(ValueError):
        reader.load_numeric(wrong)


@pytest.fixture(scope='module')
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp('invariant-saved')
    numeric = reader.load_numeric(NUMERIC)
    cases = fabricated_case_rows()
    margins = fabricated_margins()
    directions = {'u': (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                  'v1': (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                  'v2': (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
    binding = {'prelaunch_sha256': '1' * 64, 'reservation_sha256': '2' * 64,
               'latest_exposure_sha256': '3' * 64}
    scope, revision, width, vocabulary = 'fabricated invariant', 'f' * 40, 8, 800
    schedule = reader.expected_records(cases, vocabulary=vocabulary)
    placements = {axis + '_' + sign: [sum(row['action'] == axis + '_' + sign and row['slot'] == slot
                                           for row in schedule) for slot in range(6)]
                  for axis in reader.AXES for sign in ('plus', 'minus')}
    success = 612 * vocabulary * 4 + 476 * (2 * 12 * width * 4 + 2 * width * 4)
    budget = {'successful_logical_bytes_upper_bound': success,
              'failed_return_bytes_reserve': 4 * (12 * vocabulary + 2 * 12 * width + 2 * width),
              'invalid_candidate_bytes_reserve': 2 * width * 4,
              'retained_tensor_bytes_limit': 256 * 1024**2,
              'per_forward_bytes_limit': 1024**2, 'event_log_bytes_limit': 32 * 1024**2}
    attempt = root / 'attempt'
    journal = AgreementJournal(attempt, {'scope': scope, 'external_binding': binding,
        'source_revision': revision}, max_forward_bytes=budget['per_forward_bytes_limit'],
        max_total_bytes=budget['retained_tensor_bytes_limit'], max_event_log_bytes=budget['event_log_bytes_limit'],
        max_events=614)
    action_sha, model_sha = 'fbf526fd867ed48127c24b3b35004727b1757782e90798d60dcfdec37ac9b2f6', 'a' * 64
    journal.record({'kind': 'run_contract', 'expected_forwards': 612, 'seed': 20261030,
        'environment': {}, 'execution_configuration': [], 'model_state_sha256': model_sha,
        'external_binding': binding, 'schedule_sha256': action_sha, 'arm_placement_counts': placements,
        'evidence_budget': budget}, {})
    by_id = {case['case_id']: case for case in cases}
    base = struct.pack('<8f', *([0.0] * width))
    metric_ids = {'present': (318, 389), 'past': (373, 547), 'the_vs_a': (262, 257), 'in_vs_on': (287, 319)}
    for item in schedule:
        case, action = by_id[item['case_id']], item['action']
        length = len(case['token_ids'])
        logits = [0.0] * vocabulary
        for metric, (left, right) in metric_ids.items():
            logits[left], logits[right] = margins[case['case_id']][action][metric], 0.0
        event = {'kind': 'forward', **item, 'input_ids': case['token_ids'], 'attention_mask': [1] * length,
                 'all_sequence_logits_finite': True, 'plan_metadata': {'steps': []}, 'dose': None}
        tensors = {'last_logits': ((vocabulary,), struct.pack('<800f', *logits))}
        if action not in ('clean', 'clean_after'):
            site, plan = numeric.expected_hook_plan('noop_l3_final' if action == 'noop' else action,
                                                    sequence_length=length, width=width)
            after = base
            if action != 'noop':
                axis, sign = action.rsplit('_', 1)
                amount = 2.0 if sign == 'plus' else -2.0
                after, dose = numeric.direction_delivery(base, directions[axis], amount)
                before_values = numeric.decode_float32(base, expected_count=width)
                after_values = numeric.decode_float32(after, expected_count=width)
                projection = math.fsum((a-b)*d for a, b, d in zip(after_values, before_values, directions[axis]))
                relative = abs(abs(projection) - 2.0) / 2.0
                dose['signed_projected_displacement'] = projection
                dose['projected_relative_error'] = relative
                dose['dose_valid'] = dose['dose_valid'] and projection * amount > 0.0 and relative <= 1e-5
                event['dose'] = dose
            event.update({'site': site, 'plan_metadata': plan, 'donor': None})
            tensors.update({'before': ((width,), base), 'after': ((width,), after),
                'whole_before': ((1, length, width), base * length),
                'whole_after': ((1, length, width), base * (length-1) + after)})
        journal.record(event, tensors)
    journal.finish('complete', {'status': 'complete_candidate', 'forwards': 612,
        'model_state_sha256_before_after': model_sha, 'retained_tensor_bytes': journal.retained_bytes,
        'schedule_sha256': action_sha})
    raw = (attempt / 'events.jsonl').read_bytes()
    return {'attempt': attempt, 'numeric': numeric, 'cases': cases, 'directions': directions,
            'binding': binding, 'scope': scope, 'revision': revision,
            'events_sha256': hashlib.sha256(raw).hexdigest(),
            'records': [json.loads(line) for line in raw.splitlines()]}


def test_complete_fabricated_journal_replays_all_cases_and_witnesses(saved):
    report = reader.audit_attempt(saved['attempt'], saved['cases'], saved['directions'], saved['numeric'],
        expected_binding=saved['binding'], expected_source_revision=saved['revision'],
        expected_events_sha256=saved['events_sha256'], expected_scope=saved['scope'],
        width=8, vocabulary=800, allow_synthetic=True)
    assert report['journal']['events'] == 613
    assert len(report['cases']) == 68 and len(report['fragments']) == 12
    assert len(report['demonstratives']) == 8
    assert all(len(row['vocabulary']) == 9 and len(row['deliveries']) == 6 for row in report['cases'])
    for row in report['cases']:
        assert row['vocabulary']['noop']['kl_clean_to_arm'] == 0.0
        assert row['vocabulary']['noop']['kl_underflowed_to_zero'] is False
        assert row['vocabulary']['noop']['top_one_changed'] is False
        assert all(item['kl_clean_to_arm'] >= 0.0 for item in row['vocabulary'].values())
    assert report['partitions']['evaluation']['baseline']['overall_correct'] == 24
    assert report['partitions']['evaluation']['criteria']['sensitivity']['decision'] == 'pass'


def test_changed_order_input_finiteness_dose_and_landing_are_rejected(saved):
    samples = []
    for mutate in (
        lambda record: record['event'].__setitem__('index', 100),
        lambda record: record['event']['input_ids'].__setitem__(0, 200),
        lambda record: record['event'].__setitem__('all_sequence_logits_finite', False),
    ):
        altered = copy.deepcopy(saved['records']); mutate(altered[1]); samples.append(altered)
    altered = copy.deepcopy(saved['records'])
    altered[3]['event']['dose']['signed_projected_displacement'] += 1.0; samples.append(altered)
    altered = copy.deepcopy(saved['records'])
    altered[3]['tensors']['after'] = altered[3]['tensors']['before']; samples.append(altered)
    samples.append(saved['records'][:-1])
    for records in samples:
        with pytest.raises(ValueError):
            reader.audit_records(records, saved['cases'], saved['attempt'] / 'tensors', saved['directions'],
                saved['numeric'], expected_binding=saved['binding'], width=8, vocabulary=800, allow_synthetic=True)


def test_incomplete_terminal_and_wrong_event_digest_are_not_results(saved, monkeypatch):
    kwargs = {'expected_binding': saved['binding'], 'expected_source_revision': saved['revision'],
              'expected_events_sha256': saved['events_sha256'], 'expected_scope': saved['scope'],
              'width': 8, 'vocabulary': 800, 'allow_synthetic': True}
    bad = dict(kwargs, expected_events_sha256='0' * 64)
    with pytest.raises(ValueError):
        reader.audit_attempt(saved['attempt'], saved['cases'], saved['directions'], saved['numeric'], **bad)
    original = reader.bounded_file
    def altered(path, limit):
        raw = original(path, limit)
        if Path(path).name == 'terminal.json':
            doc = json.loads(raw); doc['status'] = 'failed'
            return json.dumps(doc).encode('ascii')
        return raw
    monkeypatch.setattr(reader, 'bounded_file', altered)
    with pytest.raises(ValueError):
        reader.audit_attempt(saved['attempt'], saved['cases'], saved['directions'], saved['numeric'], **kwargs)


def test_public_case_and_direction_inputs_are_pinned_before_use(tmp_path, monkeypatch):
    cases = fabricated_case_rows()
    rows = json.dumps([row['token_ids'] for row in cases], separators=(',', ':'), ensure_ascii=True).encode('ascii')
    document = {'schema': 'agreement-invariant-noun-cases-v1', 'protocol_sha256': reader.PROTOCOL_SHA,
                'protocol_revision': 'f404c5bf8446457275bcab046c4ecedbb5a83687',
                'expected_forwards': 612,
                'model_id': 'distilbert/distilgpt2',
                'model_revision': '2290a62682d06624634c1f46a6ad5be0f47f38aa',
                'tokenizer_sha256': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
                'actions_sha256': reader.ACTION_SHA, 'ordered_rows_sha256': hashlib.sha256(rows).hexdigest(),
                'cases': cases}
    case_file = tmp_path / 'cases.json'
    case_file.write_text(json.dumps(document), encoding='utf-8')
    monkeypatch.setattr(reader, 'CASE_SHA', hashlib.sha256(case_file.read_bytes()).hexdigest())
    direction_shas = {}
    for index, name in enumerate(reader.AXES):
        vector = [0.0] * 768; vector[index] = 1.0
        raw = struct.pack('<768d', *vector)
        (tmp_path / (name + '.f64le')).write_bytes(raw)
        direction_shas[name] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(reader, 'DIRECTION_SHA', direction_shas)
    loaded_cases, directions = reader.load_frozen_inputs(case_file, tmp_path)
    assert loaded_cases == cases and directions['u'][0] == directions['v1'][1] == directions['v2'][2] == 1.0
    case_file.write_bytes(case_file.read_bytes() + b' ')
    with pytest.raises(ValueError):
        reader.load_frozen_inputs(case_file, tmp_path)


def test_relative_attempt_alias_cannot_place_output_inside_attempt(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempt'; attempt.mkdir()
    alias = tmp_path / 'alias'; alias.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reader, 'load_numeric', lambda _: pytest.fail('must reject before loading'))
    common = ['--cases', 'unused', '--directions', 'unused', '--numeric-source', 'unused',
              '--source-revision', 'a' * 40, '--prelaunch-sha256', '1' * 64,
              '--reservation-sha256', '2' * 64, '--latest-exposure-sha256', '3' * 64,
              '--events-sha256', '4' * 64, '--start-scope', 'public prospective agreement invariant-noun']
    for name in ('attempt', str(alias / '..' / 'attempt')):
        with pytest.raises(ValueError, match='outside'):
            reader.main(['--attempt', name, '--output', str(attempt / 'report.json'), *common])
        assert not (attempt / 'report.json').exists()


def test_flushed_subnormals_reject_before_reading_evidence(saved, monkeypatch):
    monkeypatch.setattr(reader.math, 'ldexp', lambda *_: 0.0)
    monkeypatch.setattr(reader, 'bounded_file', lambda *_: pytest.fail('must reject before reading'))
    with pytest.raises(ValueError, match='subnormal'):
        reader.audit_attempt(saved['attempt'], saved['cases'], saved['directions'], saved['numeric'],
            expected_binding=saved['binding'], expected_source_revision=saved['revision'],
            expected_events_sha256=saved['events_sha256'], expected_scope=saved['scope'],
            width=8, vocabulary=800, allow_synthetic=True)
