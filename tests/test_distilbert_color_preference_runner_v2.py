"""Fabricated direct-owner and currentness tests; no checkpoint is loaded."""
import json
import os
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from looking_glass import distilbert_color_edit_records_v1 as records
from looking_glass import distilbert_color_edit_stages_v1 as stages
from looking_glass import distilbert_color_preference_runner_v2 as runner
from looking_glass import distilbert_color_preference_screen_v2 as preference
from scripts import check_distilbert_color_preference_exposures_v2 as comparison
from scripts import observe_distilbert_color_preference_v2 as observer

REAL_REPO = Path(__file__).resolve().parents[1]


def test_observer_targets_the_v2_owner():
    with patch.object(observer.observer, 'BOOTSTRAP', 'original'), \
         patch.object(observer.observer, 'main') as main:
        observer.main()
        assert 'distilbert_color_preference_runner_v2' in observer.observer.BOOTSTRAP
        main.assert_called_once_with()


def test_runtime_witness_is_complete_cpu_bound_and_linked_to_claim():
    claim = {'site_path_sha256': 'a' * 64, 'scratch_root_sha256': 'b' * 64}
    runtime = fake_runtime(claim)
    runner._validate_runtime(runtime, claim)
    for change in ({'device': 'cuda'}, {'intraop_threads': True},
                   {'cuda_initialized': True}, {'model_state_sha256': None},
                   {'site_path_sha256': 'c' * 64}):
        with pytest.raises(ValueError, match='runtime'):
            runner._validate_runtime({**runtime, **change}, claim)
    with pytest.raises(ValueError, match='runtime'):
        runner._validate_runtime(None, claim)


def directories(tmp_path):
    repo, cache, site, scratch = (tmp_path / x for x in ('repo', 'cache', 'site', 'scratch'))
    for path in (repo, cache, site, scratch):
        path.mkdir()
    (repo / 'experiments').mkdir()
    (repo / runner.public.CASE_PATH).write_bytes((REAL_REPO / runner.public.CASE_PATH).read_bytes())
    prelaunch = tmp_path / 'prelaunch.json'
    prelaunch.write_bytes(b'{}\n')
    return repo, cache, site, scratch, prelaunch


def emit_fake(journal, row):
    binding = {'action': journal.forwards, 'phase': 'screen', 'condition': 'original',
               'object': row['object'], 'case_id': row['id'], 'input_ids': row['input_ids'],
               'attention_mask': row['attention_mask'], 'mask_index': row['mask_index']}
    journal.start(binding)
    values = [0.0] * 30522
    values[row['word_ids'][row['factual']]] = 0.05 if row['object'] == 'banana' else 0.125
    values[row['word_ids'][row['wrong_target']]] = 8.0
    journal.emit({'kind': 'complete', **binding, 'parameter': None, 'parameter_sha256': None,
                  'realized_norm': 0.0, 'logits': records.encode_raw(struct.pack('<30522f', *values))})


def admission():
    return {'current_scan': {'commit': 'b' * 40, 'fabricated': True}}


def fake_runtime(claim):
    return {'runtime': runner.RUNTIME, 'python': '3.11.15', 'model': 'distilbert/distilbert-base-uncased',
            'revision': '12040accade4e8a0f71eabdb258fecc2e7e948be', 'model_state_sha256': 'e' * 64,
            'device': 'cpu', 'dtype': 'float32', 'eager': True, 'intraop_threads': 1, 'interop_threads': 1,
            'deterministic_algorithms': True, 'cuda_initialized': False,
            'decoder_weight_tied_to_input': True, 'decoder_bias_sha256': runner.DECODER_BIAS_SHA256,
            'site_path_sha256': claim['site_path_sha256'], 'scratch_root_sha256': claim['scratch_root_sha256']}


def test_fabricated_supervisor_saves_nine_rows_and_cannot_run_twice(tmp_path):
    repo, cache, site, scratch, prelaunch = directories(tmp_path)
    root, source = tmp_path / 'run', 'a' * 40
    manifest = json.loads((repo / runner.public.CASE_PATH).read_bytes())

    class FakeChild:
        pid = 4242
        def __init__(self, command, **_kwargs):
            directory = Path(command[command.index('--run-root') + 1]) / 'screen'
            claim = runner._read_json(directory / 'claim.json')
            assert claim['remaining_starts'] == 9
            runner._write(directory / 'worker-start.json',
                          runner._json_bytes({'phase': 'screen', 'pid': self.pid, 'source_commit': source}))
            with records.Journal(directory / 'events.jsonl', byte_cap=claim['journal_byte_cap'],
                                 forward_cap=claim['remaining_starts']) as journal:
                for row in stages.screen_rows(manifest):
                    emit_fake(journal, row)
            raw = (directory / 'events.jsonl').read_bytes()
            summary = preference.rebuild_screen(manifest, records.parse_jsonl(raw))
            result = {'schema': 'distilbert-color-stage-result-v1', 'phase': 'screen',
                      'source_commit': source, 'status': 'complete', 'source_sha256': {},
                      'runtime': fake_runtime(claim),
                      'case_sha256': runner.CASE_SHA256, 'events_sha256': runner._sha(raw),
                      'event_bytes': len(raw), 'starts': 9, 'summary': summary}
            runner._write(directory / 'result.json', runner._json_bytes(result))
        def wait(self, timeout=None):
            return 0

    with patch.object(runner, '_check_launch'), patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * 1024**3)), \
         patch.object(runner, 'verify_git_sources', return_value={}), \
         patch.object(runner, '_verify_prelaunch', return_value=admission()), \
         patch.object(runner.subprocess, 'Popen', FakeChild):
        kwargs = dict(repo=repo, cache=cache, site=site, scratch=scratch, root=root, phase='screen',
                      source=source, prelaunch_path=prelaunch, prelaunch_sha=runner._file_sha(prelaunch)[1])
        result = runner.supervise(**kwargs)
        assert result['started_forwards'] == 9
        assert (root / 'screen/validated.json').is_file()
        saved = runner._read_json(root / 'screen/result.json')
        assert saved['summary']['selected_object'] == 'strawberry'
        assert all(row['unique_top'] == row['top_representative'] for row in saved['summary']['rows'])
        with pytest.raises(FileExistsError):
            runner.supervise(**kwargs)
        with pytest.raises(ValueError, match='phase'):
            runner.supervise(**{**kwargs, 'root': tmp_path / 'other', 'phase': 'development'})


def test_timeout_and_lost_spawn_remain_incomplete(tmp_path):
    repo, cache, site, scratch, prelaunch = directories(tmp_path)

    class TimeoutChild:
        pid = 5151
        def __init__(self, *_args, **_kwargs):
            self.calls = 0
        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired('fabricated', timeout)
            return -9
        def kill(self):
            pass

    with patch.object(runner, '_check_launch'), patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * 1024**3)), \
         patch.object(runner, 'verify_git_sources', return_value={}), \
         patch.object(runner, '_verify_prelaunch', return_value=admission()):
        kwargs = dict(repo=repo, cache=cache, site=site, scratch=scratch, phase='screen',
                      source='a' * 40, prelaunch_path=prelaunch, prelaunch_sha=runner._file_sha(prelaunch)[1])
        with patch.object(runner.subprocess, 'Popen', TimeoutChild):
            with pytest.raises(RuntimeError, match='unsuccessfully'):
                runner.supervise(root=tmp_path / 'timeout', **kwargs)
        exit_ = runner._read_json(tmp_path / 'timeout/screen/exit.json')
        assert exit_['timed_out'] and exit_['cleanup'] == 'direct-child-reaped'
        assert not (tmp_path / 'timeout/screen/validated.json').exists()
        with patch.object(runner.subprocess, 'Popen', side_effect=OSError('fake spawn')):
            with pytest.raises(RuntimeError, match='unknown'):
                runner.supervise(root=tmp_path / 'unknown', **kwargs)
        exit_ = runner._read_json(tmp_path / 'unknown/screen/exit.json')
        assert exit_['pid'] is None and exit_['events_sha256'] is None
        assert not (tmp_path / 'unknown/screen/validated.json').exists()


def test_worker_reconstructs_fake_full_width_logits_and_launch_scan(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    repo, cache, site, scratch, _ = directories(tmp_path)
    child_scratch = scratch / 'color-owner-fixture'
    child_scratch.mkdir()
    root, source = tmp_path / 'run', 'a' * 40
    root.mkdir()
    stage = root / 'screen'
    stage.mkdir()
    run_raw = runner._json_bytes({'source_commit': source, 'case_sha256': runner.CASE_SHA256})
    (root / 'run.json').write_bytes(run_raw)
    prelaunch = b'{}\n'
    (stage / 'prelaunch.json').write_bytes(prelaunch)
    launch_scan = admission()['current_scan']
    claim = {'phase': 'screen', 'source_commit': source, 'prelaunch_sha256': runner._sha(prelaunch),
             'source_sha256': {}, 'publication_ancestry': [source],
             'scratch_root_sha256': runner._sha(os.fsencode(str(scratch.resolve()))),
             'site_path_sha256': runner._sha(os.fsencode(str(site.resolve()))),
             'external_prelaunch_bytes': len(prelaunch), 'prior_retained_bytes': len(run_raw),
             'prior_public_index_bytes': 0, 'prior_starts': 0, 'prior_publications': {},
             'journal_byte_cap': 8 * 1024 * 1024, 'remaining_starts': 9,
             'launch_main': launch_scan['commit'], 'launch_public_scan_sha256': runner._sha(runner._json_bytes(launch_scan))}
    (stage / 'claim.json').write_bytes(runner._json_bytes(claim))

    class FakeEngine:
        model = object()
        state = object()
        def run(self, journal, row, *, phase, condition):
            assert phase == 'screen' and condition == 'original'
            emit_fake(journal, row)

    model_runtime = fake_runtime(claim)
    model_runtime.pop('site_path_sha256')
    model_runtime.pop('scratch_root_sha256')
    with patch.object(runner, '_check_launch'), patch.object(runner, 'verify_git_sources', return_value={}), \
         patch.object(runner, '_verify_prelaunch', return_value=admission()), \
         patch.object(runner.public, 'scan', return_value=launch_scan), \
         patch.object(runner, '_load_model', return_value=(FakeEngine(), model_runtime)), \
         patch('looking_glass.bert_agreement_study.verify_model_state', return_value=True):
        runner._worker(repo, cache, site, child_scratch, root, 'screen')
    result = runner._read_json(stage / 'result.json')
    assert result['starts'] == 9 and result['summary']['selected_object'] == 'strawberry'
    assert len(records.parse_jsonl((stage / 'events.jsonl').read_bytes())) == 18
    assert not (stage / 'failure.json').exists()


def test_worker_rejects_wrong_launch_scan_before_loading_a_model(tmp_path):
    repo, cache, site, scratch, _ = directories(tmp_path)
    child_scratch = scratch / 'color-owner-fixture'
    child_scratch.mkdir()
    root, source = tmp_path / 'run', 'a' * 40
    root.mkdir()
    stage = root / 'screen'
    stage.mkdir()
    run_raw = runner._json_bytes({'source_commit': source, 'case_sha256': runner.CASE_SHA256})
    (root / 'run.json').write_bytes(run_raw)
    prelaunch = b'{}\n'
    (stage / 'prelaunch.json').write_bytes(prelaunch)
    claim = {'phase': 'screen', 'source_commit': source, 'prelaunch_sha256': runner._sha(prelaunch),
             'source_sha256': {}, 'publication_ancestry': [source],
             'scratch_root_sha256': runner._sha(os.fsencode(str(scratch.resolve()))),
             'site_path_sha256': runner._sha(os.fsencode(str(site.resolve()))),
             'external_prelaunch_bytes': len(prelaunch), 'prior_retained_bytes': len(run_raw),
             'prior_public_index_bytes': 0, 'prior_starts': 0, 'prior_publications': {},
             'journal_byte_cap': 8 * 1024 * 1024, 'remaining_starts': 9,
             'launch_main': 'b' * 40, 'launch_public_scan_sha256': '0' * 64}
    (stage / 'claim.json').write_bytes(runner._json_bytes(claim))
    with patch.object(runner, '_check_launch'), patch.object(runner, 'verify_git_sources', return_value={}), \
         patch.object(runner, '_verify_prelaunch', return_value=admission()), \
         patch.object(runner.public, 'scan', return_value=admission()['current_scan']), \
         patch.object(runner, '_load_model') as load:
        with pytest.raises(ValueError, match='launch-boundary public scan'):
            runner._worker(repo, cache, site, child_scratch, root, 'screen')
        load.assert_not_called()
    assert (stage / 'failure.json').is_file()
    assert not (stage / 'events.jsonl').exists()


def test_main_advance_rechecks_current_inputs_instead_of_rejecting_any_change():
    repo = REAL_REPO
    original, latest, source = 'b' * 40, 'c' * 40, 'a' * 40
    before = {'commit': original, 'coverage': 'first'}
    after = {'commit': latest, 'coverage': 'updated-disjoint'}
    observation = {'schema': 'distilbert-color-preference-prelaunch-v2', 'phase': 'screen',
                   'source_commit': source, 'case_sha256': runner.CASE_SHA256,
                   'public_check_sha256': runner.CHECK_SHA256, 'public_scan': before,
                   'additional_public': [], 'scope': 'fabricated named inputs'}
    fixed = json.loads((repo / 'experiments/bert_color_preference_public_comparison_v2.json').read_bytes())
    scans = []
    def fake_scan(_repo, commit, _planned):
        scans.append(commit)
        return before if commit == original else after
    with patch.object(comparison, 'check', return_value=fixed), \
         patch.object(runner.public, 'scan', side_effect=fake_scan), \
         patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(
             stdout=(latest + '\trefs/heads/main\n').encode())):
        result = runner._verify_prelaunch(repo, source, 'screen', runner._json_bytes(observation))
    assert scans == [original, latest]
    assert result['current_scan'] == after


def test_changed_public_scan_with_collision_stops_before_claim(tmp_path):
    repo, cache, site, scratch, prelaunch = directories(tmp_path)
    with patch.object(runner, '_check_launch'), \
         patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * 1024**3)), \
         patch.object(runner, 'verify_git_sources', return_value={}), \
         patch.object(runner, '_verify_prelaunch', side_effect=ValueError('public input prefix relation')):
        with pytest.raises(ValueError, match='prefix relation'):
            runner.supervise(repo=repo, cache=cache, site=site, scratch=scratch, root=tmp_path / 'run',
                             phase='screen', source='a' * 40, prelaunch_path=prelaunch,
                             prelaunch_sha=runner._file_sha(prelaunch)[1])
    assert not (tmp_path / 'run').exists()


def test_run_root_and_dependency_site_must_be_disjoint(tmp_path):
    repo, cache, site, scratch, prelaunch = directories(tmp_path)
    with patch.object(runner, '_check_launch'):
        for root in (site, site / 'nested-run', tmp_path):
            with pytest.raises(ValueError, match='dependency site'):
                runner.supervise(repo=repo, cache=cache, site=site, scratch=scratch, root=root,
                                 phase='screen', source='a' * 40, prelaunch_path=prelaunch,
                                 prelaunch_sha=runner._file_sha(prelaunch)[1])


def test_low_headroom_is_checked_before_claim(tmp_path):
    repo, cache, site, scratch, prelaunch = directories(tmp_path)
    with patch.object(runner, '_check_launch'), \
         patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024)):
        with pytest.raises(RuntimeError, match='insufficient free space'):
            runner.supervise(repo=repo, cache=cache, site=site, scratch=scratch, root=tmp_path / 'run',
                             phase='screen', source='a' * 40, prelaunch_path=prelaunch,
                             prelaunch_sha=runner._file_sha(prelaunch)[1])
    assert not (tmp_path / 'run').exists()
