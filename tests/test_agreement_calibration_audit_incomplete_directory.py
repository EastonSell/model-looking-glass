"""Partial raw-directory checks use only synthetic CPU artifacts."""
import json
from pathlib import Path

import pytest

from looking_glass import agreement_calibration_saved_audit as audit
from test_agreement_calibration_audit_directory import saved_attempt
from test_agreement_calibration_audit_identity import frozen_identity
from test_agreement_calibration_audit_sources import committed_tree


def incomplete_args(saved_attempt):
    return {name: value for name, value in saved_attempt.items() if name != 'controller_result'}


def remove_after(run, count):
    for index in range(count, 84):
        (run / f'stage-{index:03}.json').unlink()


def test_reads_only_the_actual_incomplete_contiguous_stage_files(saved_attempt):
    run = saved_attempt['run_dir']; remove_after(run, 43)
    assert hasattr(audit, 'audit_incomplete_directory'), 'raw incomplete-attempt audit is missing'
    result = audit.audit_incomplete_directory(**incomplete_args(saved_attempt))
    assert result['status'] == 'incomplete_prefix_consistent_not_approved'
    assert result['stage_count'] == 43 and result['completed_samples'] == 7
    assert result['next_kind'] == audit.KINDS[6] and result['next_phase'] == 'baseline'
    assert result['supervision_checked'] is False and result['journal_validated'] is False
    assert 'profile_ids' not in result and 'estimates_bytes' not in result


def test_all_ordinary_stages_do_not_attest_owner_completion(saved_attempt):
    result = audit.audit_incomplete_directory(**incomplete_args(saved_attempt))
    assert result['status'] == 'stage_sequence_consistent_owner_unverified_not_approved'
    assert result['stage_count'] == 84 and result['completed_samples'] == 14
    assert 'profile_ids' not in result and 'estimates_bytes' not in result


@pytest.mark.parametrize('mutation', ['gap', 'extra_nonstage', 'source_changed', 'wrong_anchor'])
def test_rejects_missing_stage_or_invalid_identity(saved_attempt, mutation):
    run = saved_attempt['run_dir']; remove_after(run, 43)
    if mutation == 'gap': (run / 'stage-020.json').unlink()
    elif mutation == 'extra_nonstage': (run / 'stage-999.json').write_bytes(b'{}\n')
    elif mutation == 'source_changed':
        (saved_attempt['source_checkout'] / 'looking_glass/__init__.py').write_bytes(b'changed')
    else: saved_attempt['expected_freeze_sha256'] = 'f' * 64
    with pytest.raises(ValueError): audit.audit_incomplete_directory(**incomplete_args(saved_attempt))


def test_records_failure_sidecar_hash_without_exposing_its_contents(saved_attempt):
    run = saved_attempt['run_dir']; remove_after(run, 43)
    (run / 'failure.json').write_bytes(b'{"private_traceback":"private path"}\n')
    report = audit.audit_incomplete_directory(**incomplete_args(saved_attempt))
    assert len(report['failure_sidecar_sha256']) == 64
    assert 'private path' not in json.dumps(report)
