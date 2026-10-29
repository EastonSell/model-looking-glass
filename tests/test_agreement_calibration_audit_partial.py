"""Validate recorded prefixes without inferring missing GPU samples."""
from copy import deepcopy

import pytest

from looking_glass import agreement_calibration_saved_audit as audit
from test_agreement_calibration_saved_audit import sampled_records


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4, 5, 6, 7, 41, 42, 43, 83])
def test_reports_only_available_ordered_stage_prefix(sampled_records, count):
    records, key = sampled_records
    assert hasattr(audit, 'audit_partial_stage_records'), 'partial stage audit is missing'
    result = audit.audit_partial_stage_records(records[:count], key)
    expected_kind = ('warmup', *audit.KINDS)[count // 6]
    assert result['status'] == 'incomplete_prefix_consistent_not_approved'
    assert result['stage_count'] == count and result['completed_samples'] == count // 6
    assert result['next_kind'] == expected_kind and result['next_phase'] == audit.PHASES[count % 6]
    assert 'estimates_bytes' not in result and 'profile_ids' not in result


@pytest.mark.parametrize('mutation', ['reordered', 'changed_seed', 'wrong_case', 'bad_baseline',
                                     'bad_budget', 'bad_after', 'bad_postcheck', 'bad_completed',
                                     'wrong_plan', 'full_sequence'])
def test_rejects_inconsistent_or_not_partial_prefix(sampled_records, mutation):
    records, key = deepcopy(sampled_records)
    records = records[:42] if mutation not in ('bad_after', 'bad_postcheck', 'full_sequence') else records
    if mutation == 'reordered': records[7], records[8] = records[8], records[7]
    elif mutation == 'changed_seed': records[6]['seed'] = 1
    elif mutation == 'wrong_case': records[6]['case_sha256'] = 'f' * 64
    elif mutation == 'bad_baseline': records[7]['snapshot']['process_reserved_bytes'] = 0
    elif mutation == 'bad_budget': records[8]['budget']['committed_incremental_bytes'] -= 1
    elif mutation == 'bad_after':
        records = records[:4]; records[3]['snapshot']['device_total_bytes'] -= 1
    elif mutation == 'bad_postcheck':
        records = records[:5]; records[4]['budget']['committed_incremental_bytes'] -= 1
    elif mutation == 'bad_completed': records[17]['sample']['peak_reserved_bytes'] = 0
    elif mutation == 'wrong_plan': records[17]['callback']['plan_metadata']['steps'][2]['token_index'] = 1
    with pytest.raises(ValueError): audit.audit_partial_stage_records(records, key)


@pytest.mark.parametrize('count', [0, 1, 6, 41, 42, 83, 84])
def test_preserves_a_trailing_failed_stage_without_accepting_the_attempt(sampled_records, count):
    records, key = sampled_records
    prior = records[:count]
    kind = ('preflight' if count == 0 else ('warmup', *audit.KINDS)[min((count - 1) // 6, 13)])
    result = audit.audit_partial_stage_records(prior + [{'phase': 'failed', 'kind': kind,
                                                         'error_type': 'MemoryError'}], key)
    assert result['status'] == 'incomplete_prefix_consistent_not_approved'
    assert result['stage_count'] == count + 1 and result['completed_samples'] == count // 6
    assert result['failure'] == {'kind': kind, 'error_type': 'MemoryError'}
    assert 'profile_ids' not in result and 'estimates_bytes' not in result


def test_rejects_wrong_failure_kind_or_records_after_failure(sampled_records):
    records, key = sampled_records
    with pytest.raises(ValueError):
        audit.audit_partial_stage_records(records[:7] + [{'phase': 'failed', 'kind': 'wrong',
                                                          'error_type': 'RuntimeError'}], key)
    with pytest.raises(ValueError):
        audit.audit_partial_stage_records(records[:7] + [{'phase': 'failed', 'kind': 'clean_capture',
                                                          'error_type': 'RuntimeError'}, records[7]], key)
