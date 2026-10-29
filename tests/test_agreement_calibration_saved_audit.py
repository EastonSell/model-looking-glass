"""CPU-only tests for independent saved calibration evidence checks.

The sampler fixture uses a fake owner and synthetic telemetry; it does not
load a checkpoint, execute a model, or initialize CUDA.
"""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

_FIXTURE_SHA256 = '09321b657e7ed7dfbefc03828aed746d3ad1883b7a552ad4e932afa20330b908'


def frozen_fixture():
    raw = (Path(__file__).parent / 'fixtures' / 'agreement-sampler-synthetic-v1.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _FIXTURE_SHA256
    fixture = json.loads(raw)
    assert fixture['schema'] == 'synthetic-agreement-sampler-audit-fixture-v1'
    return fixture


@pytest.fixture
def sampled_records():
    fixture = frozen_fixture()
    return fixture['records'], fixture['warm_key']


def auditor():
    # A missing implementation is a failed feature assertion, rather than an
    # import error concealing why the first test is red.
    name = 'looking_glass.agreement_calibration_saved_audit'
    assert importlib.util.find_spec(name) is not None, 'saved evidence auditor is missing'
    return importlib.import_module(name)


def test_reconstructs_fourteen_samples_and_independent_peak_increment(sampled_records):
    records, key = sampled_records
    result = auditor().audit_stage_records(records, key)
    assert result['sample_count'] == 14
    assert result['stage_count'] == 84
    # Synthetic fixture baseline allocated is 400 MiB and its allocated peak
    # is 450 MiB; reserved rises by 40 MiB and device-free is unchanged.
    assert result['cold_observed_increment_bytes'] == 50 * 1024 * 1024
    assert result['warm_observed_increment_bytes'] == 50 * 1024 * 1024


@pytest.mark.parametrize('mutation', [
    'missing_stage', 'swapped_stage', 'wrong_seed', 'wrong_case', 'wrong_site',
    'wrong_token', 'wrong_invocation_type', 'wrong_selector', 'wrong_capture_bytes',
    'failed_landing', 'baseline_mismatch', 'after_mismatch', 'peak_below_endpoint',
    'low_free', 'bad_budget_claim', 'forged_admissible', 'over_allocator_cap',
])
def test_rejects_incomplete_or_inconsistent_stage_evidence(sampled_records, mutation):
    records, key = deepcopy(sampled_records)
    if mutation == 'missing_stage': records.pop()
    elif mutation == 'swapped_stage': records[7], records[8] = records[8], records[7]
    elif mutation == 'wrong_seed': records[6]['seed'] = 1
    elif mutation == 'wrong_case': records[6]['case_sha256'] = 'f' * 64
    elif mutation == 'wrong_site': records[17]['callback']['plan_metadata']['steps'][2]['module_path'] = 'transformer.h.2.mlp'
    elif mutation == 'wrong_token': records[17]['callback']['plan_metadata']['steps'][2]['token_index'] = 1
    elif mutation == 'wrong_invocation_type': records[17]['callback']['plan_metadata']['steps'][2]['invocation'] = False
    elif mutation == 'wrong_selector': records[17]['callback']['plan_metadata']['steps'][2]['output_selector'] = []
    elif mutation == 'wrong_capture_bytes': records[17]['callback']['plan_metadata']['capture_bytes'] -= 1
    elif mutation == 'failed_landing': records[17]['callback']['capture_or_landing_checked'] = False
    elif mutation == 'baseline_mismatch': records[17]['sample']['before']['process_allocated_bytes'] += 1
    elif mutation == 'after_mismatch': records[17]['sample']['after']['device_free_bytes'] -= 1
    elif mutation == 'peak_below_endpoint': records[17]['sample']['peak_allocated_bytes'] = 0
    elif mutation == 'low_free': records[13]['snapshot']['device_free_bytes'] = 1
    elif mutation == 'bad_budget_claim': records[14]['budget']['committed_incremental_bytes'] -= 1
    elif mutation == 'forged_admissible': records[14]['budget']['admissible_estimate_bytes'] += 1
    elif mutation == 'over_allocator_cap': records[17]['sample']['peak_reserved_bytes'] = key['device_total_bytes'] // 4 + 1
    with pytest.raises(ValueError):
        auditor().audit_stage_records(records, key)


@pytest.fixture
def sampled_profiles():
    fixture = frozen_fixture()
    return (fixture['records'], fixture['warm_key'],
            [item.encode() for item in fixture['profile_documents']],
            [item.encode() for item in fixture['evidence_documents']],
            fixture['source'].encode(), fixture['profile_ids'])


def test_recomputes_cold_and_warm_profile_estimates_from_stage_evidence(sampled_profiles):
    records, key, profiles, evidence, source, profile_ids = sampled_profiles
    module = auditor()
    assert hasattr(module, 'audit_profiles'), 'independent profile audit is missing'
    stages = module.audit_stage_records(records, key)
    # Declared 3,102,372 bytes + 50 MiB observed, plus 512 MiB and
    # ceil(20% of the 55,531,172-byte basis) = 603,508,319 bytes.
    result = module.audit_profiles(stages, key, profiles, evidence, source,
                                  profile_ids, [603508319, 603508319])
    assert result['estimates_bytes'] == [603508319, 603508319]
    assert len(set(result['profile_ids'])) == 2


@pytest.mark.parametrize('mutation', [
    'profile_id', 'candidate_estimate', 'source', 'missing_evidence', 'wrong_evidence',
    'noncanonical_profile', 'duplicate_json_key', 'margin', 'warm_key', 'sample_order',
])
def test_rejects_inconsistent_profile_provenance_or_arithmetic(sampled_profiles, mutation):
    records, key, profiles, evidence, source, profile_ids = deepcopy(sampled_profiles)
    module = auditor()
    assert hasattr(module, 'audit_profiles'), 'independent profile audit is missing'
    stages = module.audit_stage_records(records, key)
    estimates = [603508319, 603508319]
    if mutation == 'profile_id': profile_ids[0] = 'a' * 64
    elif mutation == 'candidate_estimate': estimates[1] -= 1
    elif mutation == 'source': source = b'different source manifest'
    elif mutation == 'missing_evidence': evidence.pop()
    elif mutation == 'wrong_evidence': evidence[1] += b' '
    elif mutation == 'noncanonical_profile': profiles[0] += b' '
    elif mutation == 'duplicate_json_key': profiles[0] = b'{"schema":"resource-profile-v1","schema":"resource-profile-v1"}'
    elif mutation in ('margin', 'warm_key', 'sample_order'):
        doc = json.loads(profiles[1])
        if mutation == 'margin': doc['margin_bytes'] -= 1
        elif mutation == 'warm_key': doc['key']['sequence_length'] = 4
        else: doc['samples'][0], doc['samples'][1] = doc['samples'][1], doc['samples'][0]
        profiles[1] = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        # Re-hash the altered candidate so rejection cannot rely merely on a
        # stale caller-supplied ID.
        profile_ids[1] = hashlib.sha256(profiles[1]).hexdigest()
    with pytest.raises(ValueError):
        module.audit_profiles(stages, key, profiles, evidence, source, profile_ids, estimates)


@pytest.fixture
def synthetic_journal(tmp_path, sampled_records):
    from looking_glass.agreement_journal import AgreementJournal
    records, key = sampled_records
    stages = auditor().audit_stage_records(records, key)
    run_id, source_sha, profile_ids = '1' * 32, '2' * 64, ['3' * 64, '4' * 64]
    directory = tmp_path.resolve() / 'journal'
    journal = AgreementJournal(directory, {'schema': 'agreement-calibration-journal-v1',
        'run_id': run_id, 'source_sha256': source_sha}, max_total_bytes=1024**2,
        max_forward_bytes=1024**2, max_events=32)
    for index, callback in enumerate(stages['callbacks']):
        journal.record({'kind': 'gpu_warmup' if index == 0 else 'gpu_calibration_action', **callback}, {})
    journal.finish('complete', {'samples': 14, 'profile_ids': profile_ids})
    return directory, stages, run_id, source_sha, profile_ids


def test_reconciles_closed_journal_with_all_fourteen_exact_callbacks(synthetic_journal):
    module = auditor()
    assert hasattr(module, 'audit_journal'), 'independent journal audit is missing'
    path, stages, run_id, source, ids = synthetic_journal
    result = module.audit_journal(path, stages, run_id, source, ids)
    assert result['event_count'] == 14
    assert result['retained_tensor_bytes'] == 0


@pytest.mark.parametrize('mutation', [
    'wrong_run', 'wrong_source', 'wrong_warmup_state', 'wrong_event', 'event_reordered',
    'missing_event', 'wrong_terminal', 'missing_terminal', 'pending_terminal',
    'event_bytes', 'extra_tensor', 'unexpected_metadata',
])
def test_rejects_unclosed_or_unmatched_calibration_journal(synthetic_journal, mutation):
    module = auditor()
    assert hasattr(module, 'audit_journal'), 'independent journal audit is missing'
    path, stages, run_id, source, ids = synthetic_journal
    def encode(value):
        return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    events_path = path / 'events.jsonl'
    if mutation == 'wrong_run': run_id = 'f' * 32
    elif mutation == 'wrong_source': source = 'f' * 64
    elif mutation == 'wrong_warmup_state': stages['callbacks'][0]['model_state_sha256'] = 'f' * 64
    elif mutation in ('wrong_event', 'event_reordered', 'missing_event'):
        values = [json.loads(line) for line in events_path.read_bytes().splitlines()]
        if mutation == 'wrong_event': values[2]['event']['plan_metadata']['steps'][2]['token_index'] = 1
        elif mutation == 'event_reordered': values[2], values[3] = values[3], values[2]
        else: values.pop()
        events_path.write_bytes(b''.join(encode(value) for value in values))
    elif mutation == 'wrong_terminal':
        terminal_path = path / 'terminal.json'; terminal = json.loads(terminal_path.read_bytes())
        terminal['status'] = 'failed'; terminal_path.write_bytes(encode(terminal))
    elif mutation == 'missing_terminal': (path / 'terminal.json').unlink()
    elif mutation == 'pending_terminal': (path / '.terminal-test.pending').write_bytes(b'pending')
    elif mutation == 'event_bytes':
        terminal_path = path / 'terminal.json'; terminal = json.loads(terminal_path.read_bytes())
        terminal['event_log_bytes'] -= 1; terminal_path.write_bytes(encode(terminal))
    elif mutation == 'extra_tensor': (path / 'tensors' / 'unexpected').write_bytes(b'x')
    elif mutation == 'unexpected_metadata': (path / 'unexpected.json').write_bytes(b'{}')
    with pytest.raises(ValueError):
        module.audit_journal(path, stages, run_id, source, ids)
