"""End-to-end saved-file audit on fake CPU telemetry, never a GPU or model."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from looking_glass import agreement_calibration_saved_audit as audit
from looking_glass.agreement_journal import AgreementJournal
from looking_glass.memory_budget import CudaMemorySnapshot
from looking_glass.resource_profile import (
    MemorySample, MeasuredResourceProfile, ProfileProvenance, WorkloadKey, encode_evidence,
)
from test_agreement_calibration_audit_identity import frozen_identity, canonical, digest
from test_agreement_calibration_audit_sources import committed_tree
from test_agreement_calibration_audit_reap import line, make_reaped
from test_agreement_calibration_saved_audit import frozen_fixture


@pytest.fixture
def saved_attempt(frozen_identity, committed_tree):
    records, _, uuid, _ = deepcopy(frozen_identity)
    root, git, commit, manifest, source_partial = committed_tree
    base = json.loads(records['source_base'])
    base['repository_files'] = source_partial['base']['repository_files']
    records['source_base'] = canonical(base)
    source = json.loads(records['source'])
    source['base'] = base
    source['protocols'] = source_partial['protocols']
    records['source'] = canonical(source)
    records['controller_source'] = line(manifest)
    freeze = json.loads(records['run_manifest'])
    freeze['source'] = {'commit': commit, 'sha256': digest(canonical(manifest)), 'checkout': str(root),
                        'source_base_sha256': digest(records['source_base'])}
    records['run_manifest'] = line(freeze)
    freeze_sha = digest(records['run_manifest'])
    identity = audit.audit_identity(records, expected_commit=commit, expected_device_uuid=uuid,
                                    expected_freeze_sha256=freeze_sha)
    run = Path(freeze['directory']['path'])
    for key, name in {'run_manifest': 'run-manifest.json', 'controller_source': 'controller-source.json',
                      'source_base': 'controller-source-base.json', 'source': 'source-manifest.json',
                      'runtime': 'runtime-manifest.json', 'assets': 'asset-manifest.json'}.items():
        (run / name).write_bytes(records[key])

    fields = deepcopy(identity['warm_key'])
    fields['compute_capability'] = tuple(fields['compute_capability'])
    fields['input_dtypes'] = tuple(tuple(value) for value in fields['input_dtypes'])
    key = WorkloadKey(**fields)
    cold_key = replace(key, warmup_state='cold')
    stages = frozen_fixture()['records']
    samples = []
    for index, value in enumerate(item for item in stages if item['phase'] == 'completed'):
        item = value['sample']
        item['workload_sha256'] = (cold_key if index == 0 else key).fingerprint()
        fields = deepcopy(item)
        fields['before'] = CudaMemorySnapshot(**fields['before'])
        fields['after'] = CudaMemorySnapshot(**fields['after'])
        samples.append(MemorySample(**fields))
    for index, value in enumerate(stages):
        (run / f'stage-{index:03}.json').write_bytes(line(value))
    profiles, evidence = [], []
    for sample_key, selected in ((cold_key, tuple(samples[:1])), (key, tuple(samples[1:]))):
        raw = encode_evidence(sample_key, selected)
        provenance = ProfileProvenance(
            'resource-profile-v1', digest(records['source']), digest(raw),
            '2026-10-29T00:00:00Z', 'cooperative-exclusive', 'synchronize-reset-synchronize',
        )
        profiles.append(MeasuredResourceProfile.build(
            sample_key, selected, provenance, margin_bytes=512 * 1024**2, margin_bps=2000,
        ))
        evidence.append(raw)
    for name, profile, raw in zip(('cold', 'warm'), profiles, evidence):
        (run / (name + '-profile.json')).write_bytes(profile.to_json().encode())
        (run / (name + '-evidence.json')).write_bytes(raw)
    checked_profiles = {'profile_ids': [profile.profile_id for profile in profiles],
                        'estimates_bytes': [profile.estimated_incremental_bytes for profile in profiles]}
    journal = AgreementJournal(run / 'journal', {'schema': 'agreement-calibration-journal-v1',
        'run_id': identity['run_id'], 'source_sha256': identity['source_sha256']},
        max_total_bytes=1024**2, max_forward_bytes=1024**2, max_events=32)
    for index, value in enumerate(item['callback'] for item in stages if item['phase'] == 'completed'):
        journal.record({'kind': 'gpu_warmup' if index == 0 else 'gpu_calibration_action', **value}, {})
    journal.finish('complete', {'samples': 14, 'profile_ids': checked_profiles['profile_ids']})
    _, _, _, outcome, owner = make_reaped(identity, freeze_sha, checked_profiles)
    names = {'started': 'started.json', 'owner': 'owner.json', 'candidate': 'candidate.json',
             'final': 'final.json', 'supervision': 'supervision.json', 'model_checks': 'model-checks.json',
             'preload_admission': 'preload-admission.json', 'cleanup': 'cleanup.json'}
    for key, value in owner.items(): (run / names[key]).write_bytes(value)
    controller = run.parent / 'controller-result.json'
    controller.write_bytes(line(outcome))
    site = Path(json.loads(records['source'])['dependency_origins'][0]['path']).parent
    return {'run_dir': run, 'controller_result': controller, 'source_checkout': root, 'git_executable': git,
            'site': site, 'expected_commit': commit, 'expected_device_uuid': uuid,
            'expected_freeze_sha256': freeze_sha}


def test_audits_the_complete_synthetic_saved_attempt(saved_attempt):
    assert hasattr(audit, 'audit_directory'), 'full saved-attempt audit is missing'
    report = audit.audit_directory(**saved_attempt)
    assert report['status'] == 'saved_records_consistent_not_approved'
    assert (report['sample_count'], report['stage_count'], report['journal_event_count']) == (14, 84, 14)
    assert len(report['profile_ids']) == 2 and len(report['estimates_bytes']) == 2
    assert report['freeze_observation'] == 'controller-reported'


def test_cli_audits_without_writing_or_promoting(saved_attempt):
    args = [sys.executable, '-B', '-m', 'looking_glass.agreement_calibration_saved_audit']
    for name, value in saved_attempt.items():
        args.extend(['--' + name.replace('_', '-'), str(value)])
    completed = subprocess.run(args, capture_output=True, check=True, timeout=60)
    report = json.loads(completed.stdout)
    assert report['status'] == 'saved_records_consistent_not_approved'
    assert report['stage_count'] == 84 and report['journal_event_count'] == 14
    assert 'private_run_dir' not in report
    assert report['freeze_observation'] == 'controller-reported'


def test_isolated_standalone_cli_does_not_import_checkout_root_shadow(saved_attempt):
    parent = saved_attempt['run_dir'].parent
    prefix = parent / 'empty-bytecode-prefix'; prefix.mkdir()
    marker = parent / 'shadow-imported'
    (parent / 'copy.py').write_text(f'open({str(marker)!r}, "w").write("unexpected")\n')
    source = Path(audit.__file__).resolve(strict=True)
    args = [sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix), str(source)]
    for name, value in saved_attempt.items(): args.extend(['--' + name.replace('_', '-'), str(value)])
    completed = subprocess.run(args, cwd=parent, capture_output=True, check=True, timeout=60)
    assert json.loads(completed.stdout)['status'] == 'saved_records_consistent_not_approved'
    assert not marker.exists() and not any(prefix.iterdir())


def test_freeze_chronology_requires_a_distinct_operator_declaration(saved_attempt):
    report = audit.audit_directory(**saved_attempt, freeze_observation='independent-prelaunch')
    assert report['freeze_observation'] == 'independent-prelaunch'
    with pytest.raises(ValueError):
        audit.audit_directory(**saved_attempt, freeze_observation='unreviewed')


@pytest.mark.parametrize('mutation', ['missing_stage', 'extra_stage', 'wrong_site', 'source_bytes',
                                     'dependency_bytes', 'wrong_profile', 'wrong_journal', 'unclean',
                                     'failure_sidecar', 'different_directory'])
def test_rejects_inconsistent_raw_attempt_even_when_other_files_are_untouched(saved_attempt, mutation):
    assert hasattr(audit, 'audit_directory'), 'full saved-attempt audit is missing'
    paths = saved_attempt
    run = paths['run_dir']
    if mutation == 'missing_stage': (run / 'stage-083.json').unlink()
    elif mutation == 'extra_stage': (run / 'stage-084.json').write_bytes(b'{}\n')
    elif mutation == 'wrong_site':
        p = run / 'stage-017.json'; value = json.loads(p.read_bytes())
        value['callback']['plan_metadata']['steps'][2]['token_index'] = 1; p.write_bytes(line(value))
    elif mutation == 'source_bytes': (paths['source_checkout'] / 'looking_glass/__init__.py').write_bytes(b'changed')
    elif mutation == 'dependency_bytes': (paths['site'] / 'torch.py').write_bytes(b'changed')
    elif mutation == 'wrong_profile': (run / 'warm-profile.json').write_bytes(b'{}')
    elif mutation == 'wrong_journal':
        p = run / 'journal' / 'events.jsonl'; p.write_bytes(p.read_bytes()[:-1])
    elif mutation == 'unclean':
        p = run / 'cleanup.json'; value = json.loads(p.read_bytes())
        value['after']['allocated_bytes'] = 1; p.write_bytes(line(value))
    elif mutation == 'failure_sidecar': (run / 'failure.json').write_bytes(b'{}\n')
    elif mutation == 'different_directory': paths['run_dir'] = run.parent
    with pytest.raises(ValueError): audit.audit_directory(**paths)
