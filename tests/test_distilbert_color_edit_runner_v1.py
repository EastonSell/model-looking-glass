"""Fabricated supervisor, one-shot and aggregate-budget regressions."""
import json
import os
import subprocess
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from looking_glass import distilbert_color_edit_records_v1 as records
from looking_glass import distilbert_color_edit_runner_v1 as runner
from looking_glass import distilbert_color_edit_stages_v1 as stages


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.repo, self.cache = self.base / 'repo', self.base / 'cache'
        self.site = self.base / 'trusted-site'
        self.scratch = self.base / 'scratch'
        self.child_scratch = self.scratch / 'color-owner-fixture'
        self.repo.mkdir()
        self.cache.mkdir()
        self.site.mkdir()
        self.scratch.mkdir()
        self.child_scratch.mkdir()
        (self.repo / 'experiments').mkdir()
        source = Path(__file__).resolve().parents[1] / 'experiments/bert_color_edit_public_cases_v1.json'
        (self.repo / 'experiments/bert_color_edit_public_cases_v1.json').write_bytes(source.read_bytes())
        check = Path(__file__).resolve().parents[1] / 'experiments/bert_color_edit_public_exposure_check_v1.json'
        (self.repo / 'experiments/bert_color_edit_public_exposure_check_v1.json').write_bytes(check.read_bytes())
        self.prelaunch = self.base / 'prelaunch.json'
        self.prelaunch.write_bytes(b'{}\n')
        self.source = 'a' * 40
        launch = patch.object(runner, '_check_launch', return_value=None)
        launch.start()
        self.addCleanup(launch.stop)
        space = patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * 1024**3))
        space.start()
        self.addCleanup(space.stop)

    def tearDown(self):
        self.temp.cleanup()

    def test_supervisor_records_all_nine_and_disallows_a_second_claim(self):
        root = self.base / 'run'
        manifest = json.loads((self.repo / 'experiments/bert_color_edit_public_cases_v1.json').read_bytes())
        source = self.source

        class FakeChild:
            pid = 4242
            def __init__(self, command, **_kwargs):
                phase = command[command.index('--phase') + 1]
                run_root = Path(command[command.index('--run-root') + 1])
                directory = run_root / phase
                claim = runner._read_json(directory / 'claim.json')
                runner._write(directory / 'worker-start.json', runner._json_bytes(
                    {'phase': phase, 'pid': self.pid, 'source_commit': source}))
                with records.Journal(directory / 'events.jsonl', byte_cap=claim['journal_byte_cap'],
                                     forward_cap=claim['remaining_starts']) as journal:
                    for row in stages.screen_rows(manifest):
                        action = journal.forwards
                        binding = {'action': action, 'phase': 'screen', 'object': row['object'],
                                   'case_id': row['id'], 'condition': 'original', 'input_ids': row['input_ids'],
                                   'attention_mask': row['attention_mask'], 'mask_index': row['mask_index']}
                        journal.start(binding)
                        values = [0.0] * 30522
                        values[row['word_ids'][row['factual']]] = 1.0
                        journal.emit({'kind': 'complete', **binding, 'parameter': None, 'parameter_sha256': None,
                                      'realized_norm': 0.0, 'logits': records.encode_raw(struct.pack('<30522f', *values))})
                raw = (directory / 'events.jsonl').read_bytes()
                result = {'schema': 'distilbert-color-stage-result-v1',
                          'phase': phase, 'source_commit': source, 'status': 'complete',
                          'source_sha256': {},
                          'case_sha256': runner.CASE_SHA256, 'events_sha256': runner._sha(raw),
                          'event_bytes': len(raw), 'starts': 9,
                          'summary': records.rebuild_screen(manifest, records.parse_jsonl(raw))}
                runner._write(directory / 'result.json', runner._json_bytes(result))
            def wait(self, timeout=None):
                return 0

        with patch.object(runner, 'verify_git_sources', return_value={}), \
             patch.object(runner, '_verify_prelaunch', return_value={}), \
             patch.object(runner.subprocess, 'Popen', FakeChild):
            result = runner.supervise(repo=self.repo, cache=self.cache, site=self.site, scratch=self.scratch, root=root, phase='screen', source=source,
                                      prelaunch_path=self.prelaunch, prelaunch_sha=runner._file_sha(self.prelaunch)[1])
            self.assertEqual(result['started_forwards'], 9)
            self.assertLess(result['retained_bytes'], records.MAX_BYTES)
            with self.assertRaises(FileExistsError):
                runner.supervise(repo=self.repo, cache=self.cache, site=self.site, scratch=self.scratch, root=root, phase='screen', source=source,
                                 prelaunch_path=self.prelaunch, prelaunch_sha=runner._file_sha(self.prelaunch)[1])
        receipt = runner._read_json(root / 'screen/exit.json')
        self.assertEqual(receipt['pid'], 4242)
        self.assertEqual(receipt['returncode'], 0)
        stage = root / 'screen'
        self.assertTrue((stage / 'validated.json').is_file())
        marker = runner._read_json(stage / 'validated.json')
        observation = {'schema': 'distilbert-color-supervisor-observation-v1', 'phase': 'screen',
                       'source_commit': self.source, 'supervisor_pid': marker['supervisor_pid'],
                       'returncode': 0, 'validation_sha256': runner._file_sha(stage / 'validated.json')[1],
                       'run_sha256': runner._file_sha(root / 'run.json')[1], 'command_sha256': 'a' * 64,
                       'observer_external_bytes': 5,
                       'scope': 'fabricated cooperative observer'}
        (stage / 'supervisor-observation.json').write_bytes(runner._json_bytes(observation))
        runner._valid_prior(stage, 'screen', self.source, {})
        (stage / 'supervisor-observation-failure.json').write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            runner._valid_prior(stage, 'screen', self.source, {})
        (stage / 'supervisor-observation-failure.json').unlink()
        with self.assertRaisesRegex(ValueError, 'source|receipt'):
            runner._valid_prior(stage, 'screen', self.source, {'changed.py': 'b' * 64})
        observation['supervisor_pid'] += 1
        (stage / 'supervisor-observation.json').write_bytes(runner._json_bytes(observation))
        with self.assertRaisesRegex(ValueError, 'supervisor exit observation'):
            runner._valid_prior(stage, 'screen', self.source, {})

    def test_budget_counts_all_stages_and_unknown_files_fail(self):
        root = self.base / 'run'
        root.mkdir()
        (root / 'run.json').write_bytes(b'{}\n')
        for phase in ('screen', 'development'):
            stage = root / phase
            stage.mkdir()
            (stage / 'claim.json').write_bytes(runner._json_bytes({'external_prelaunch_bytes': 9}))
            (stage / 'events.jsonl').write_bytes(
                runner._json_bytes({'seq': 0, 'kind': 'start', 'phase': phase, 'action': 0}))
        total, starts = runner._inspect_run(root)
        self.assertEqual(starts, 2)
        self.assertEqual(total, sum(path.stat().st_size for path in root.rglob('*') if path.is_file()) + 18)
        (root / 'screen/unknown.bin').write_bytes(b'x')
        with self.assertRaises(ValueError):
            runner._inspect_run(root)

    def test_incomplete_prior_cannot_be_promoted(self):
        stage = self.base / 'screen'
        stage.mkdir()
        for name in ('claim.json', 'worker-start.json', 'events.jsonl', 'result.json', 'exit.json'):
            (stage / name).write_bytes(b'{}\n')
        (stage / 'failure.json').write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            runner._valid_prior(stage, 'screen', self.source, {})

    def test_timeout_is_reaped_and_the_claim_cannot_be_reused(self):
        root = self.base / 'run'
        class TimedOutChild:
            pid = 5151
            def __init__(self, command, **_kwargs):
                self.calls = 0
            def wait(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired('fabricated', timeout)
                return -9
            def kill(self):
                pass
        with patch.object(runner, 'verify_git_sources', return_value={}), \
             patch.object(runner, '_verify_prelaunch', return_value={}), \
             patch.object(runner.subprocess, 'Popen', TimedOutChild):
            with self.assertRaisesRegex(RuntimeError, 'unsuccessfully'):
                runner.supervise(repo=self.repo, cache=self.cache, site=self.site, scratch=self.scratch, root=root, phase='screen', source=self.source,
                                 prelaunch_path=self.prelaunch, prelaunch_sha=runner._file_sha(self.prelaunch)[1])
            receipt = runner._read_json(root / 'screen/exit.json')
            self.assertTrue(receipt['timed_out'])
            self.assertEqual(receipt['returncode'], -9)
            with self.assertRaises(FileExistsError):
                runner.supervise(repo=self.repo, cache=self.cache, site=self.site, scratch=self.scratch, root=root, phase='screen', source=self.source,
                                 prelaunch_path=self.prelaunch, prelaunch_sha=runner._file_sha(self.prelaunch)[1])

    def test_new_public_prefix_collision_blocks_prelaunch(self):
        from scripts import check_distilbert_color_edit_exposures_v1 as comparison
        doc = json.loads((self.repo / 'experiments/bert_color_edit_public_cases_v1.json').read_bytes())
        extra = runner._json_bytes({'cases': [doc['cases'][0]]})
        observation = {'schema': 'distilbert-color-prelaunch-v1', 'phase': 'screen', 'source_commit': self.source,
                       'case_sha256': runner.CASE_SHA256, 'public_check_sha256': runner.CHECK_SHA256,
                       'official_main': 'b' * 40, 'scope': 'named public inputs only',
                       'additional_public': [{'commit': 'c' * 40, 'path': 'experiments/extra.json',
                                              'sha256': runner._sha(extra), 'format': 'case-rows', 'rows': 1}]}
        public_check = json.loads((self.repo / 'experiments/bert_color_edit_public_exposure_check_v1.json').read_bytes())
        with patch.object(comparison, 'check', return_value=public_check), \
             patch.object(runner.publication, 'git_blob', return_value=extra):
            with self.assertRaisesRegex(ValueError, 'prefix check'):
                runner._verify_prelaunch(self.repo, self.source, 'screen', runner._json_bytes(observation), check_main=False)

    def test_cross_model_bert_row_collision_blocks_prelaunch(self):
        from scripts import check_distilbert_color_edit_exposures_v1 as comparison
        doc = json.loads((self.repo / 'experiments/bert_color_edit_public_cases_v1.json').read_bytes())
        extra = runner._json_bytes({'cases': [{'bert': doc['cases'][0]}]})
        observation = {'schema': 'distilbert-color-prelaunch-v1', 'phase': 'screen', 'source_commit': self.source,
                       'case_sha256': runner.CASE_SHA256, 'public_check_sha256': runner.CHECK_SHA256,
                       'official_main': 'b' * 40, 'scope': 'named public inputs only',
                       'additional_public': [{'commit': 'c' * 40, 'path': 'experiments/cross.json',
                                              'sha256': runner._sha(extra), 'format': 'bert-cases', 'rows': 1}]}
        public_check = json.loads((self.repo / 'experiments/bert_color_edit_public_exposure_check_v1.json').read_bytes())
        with patch.object(comparison, 'check', return_value=public_check), \
             patch.object(runner.publication, 'git_blob', return_value=extra):
            with self.assertRaisesRegex(ValueError, 'prefix check'):
                runner._verify_prelaunch(self.repo, self.source, 'screen', runner._json_bytes(observation), check_main=False)

    def test_worker_enforces_budget_and_rebuilds_fabricated_screen(self):
        root = self.base / 'run'
        root.mkdir()
        run_raw = runner._json_bytes({'source_commit': self.source, 'case_sha256': runner.CASE_SHA256})
        (root / 'run.json').write_bytes(run_raw)
        stage = root / 'screen'
        stage.mkdir()
        prelaunch_raw = b'{}\n'
        (stage / 'prelaunch.json').write_bytes(prelaunch_raw)
        claim = {'phase': 'screen', 'source_commit': self.source, 'prelaunch_sha256': runner._sha(prelaunch_raw),
                 'source_sha256': {}, 'publication_ancestry': [self.source],
                 'scratch_root_sha256': runner._sha(os.fsencode(str(self.scratch.resolve()))),
                 'external_prelaunch_bytes': len(prelaunch_raw), 'prior_retained_bytes': len(run_raw),
                 'prior_public_index_bytes': 0, 'prior_starts': 0, 'prior_publications': {},
                 'journal_byte_cap': records.MAX_BYTES - len(run_raw) - 2 * len(prelaunch_raw)
                                     - runner.TERMINAL_RESERVE - 16384,
                 'remaining_starts': records.MAX_FORWARDS}
        (stage / 'claim.json').write_bytes(runner._json_bytes(claim))

        class FakeEngine:
            vocabulary = 30522
            model = object()
            state = object()
            def run(self, journal, row, *, phase, condition):
                binding = {'action': journal.forwards, 'phase': phase, 'condition': condition,
                           'object': row['object'], 'case_id': row['id'], 'input_ids': row['input_ids'],
                           'attention_mask': row['attention_mask'], 'mask_index': row['mask_index']}
                journal.start(binding)
                values = [0.0] * self.vocabulary
                values[row['word_ids'][row['factual']]] = 1.0
                journal.emit({'kind': 'complete', **binding, 'parameter': None, 'parameter_sha256': None,
                              'realized_norm': 0.0, 'logits': records.encode_raw(struct.pack('<30522f', *values))})

        with patch.object(runner, 'verify_git_sources', return_value={}), \
             patch.object(runner, '_verify_prelaunch', return_value={}), \
             patch.object(runner, '_load_model', return_value=(FakeEngine(), {'fabricated': True})), \
             patch('looking_glass.bert_agreement_study.verify_model_state', return_value=True):
            runner._worker(self.repo, self.cache, self.site, self.child_scratch, root, 'screen')
        result = runner._read_json(stage / 'result.json')
        self.assertEqual(result['starts'], 9)
        self.assertEqual(result['summary']['selected_object'], 'banana')
        self.assertTrue((stage / 'worker-start.json').is_file())
        self.assertFalse((stage / 'failure.json').exists())

    def test_mismatched_loaded_state_stops_before_development_forward(self):
        root = self.base / 'run'
        root.mkdir()
        run_raw = runner._json_bytes({'source_commit': self.source, 'case_sha256': runner.CASE_SHA256})
        (root / 'run.json').write_bytes(run_raw)
        stage = root / 'development'
        stage.mkdir()
        prelaunch_raw = b'{}\n'
        (stage / 'prelaunch.json').write_bytes(prelaunch_raw)
        predecessor = {'commit': 'b' * 40, 'prefix': 'results/fake'}
        claim = {'phase': 'development', 'source_commit': self.source, 'source_sha256': {},
                 'scratch_root_sha256': runner._sha(os.fsencode(str(self.scratch.resolve()))),
                 'prelaunch_sha256': runner._sha(prelaunch_raw), 'external_prelaunch_bytes': len(prelaunch_raw),
                 'prior_retained_bytes': len(run_raw), 'prior_public_index_bytes': 0, 'prior_starts': 0,
                 'prior_publications': {'screen': predecessor}, 'publication_ancestry': [self.source, 'b' * 40],
                 'journal_byte_cap': records.MAX_BYTES - len(run_raw) - 2 * len(prelaunch_raw)
                                     - runner.TERMINAL_RESERVE - 16384,
                 'remaining_starts': records.MAX_FORWARDS}
        (stage / 'claim.json').write_bytes(runner._json_bytes(claim))
        prior_result = {'runtime': {'model_state_sha256': 'a' * 64}}
        with patch.object(runner, 'verify_git_sources', return_value={}), \
             patch.object(runner, '_verify_prelaunch', return_value={}), \
             patch.object(runner, '_publication_ancestry', return_value=[self.source, 'b' * 40]), \
             patch.object(runner, '_valid_prior', return_value=([], prior_result)), \
             patch.object(runner.publication, 'verify_publication', return_value={'index_bytes': 0, 'run_sidecar_bytes': 0}), \
             patch.object(runner, '_load_model', return_value=(object(), {'model_state_sha256': 'b' * 64})), \
             patch.object(stages, 'run_development') as run_development:
            with self.assertRaisesRegex(ValueError, 'model state or runtime differs'):
                runner._worker(self.repo, self.cache, self.site, self.child_scratch, root, 'development')
            run_development.assert_not_called()
        self.assertFalse((stage / 'events.jsonl').exists())
        self.assertTrue((stage / 'failure.json').exists())

    def test_cross_stage_runtime_requires_both_exact_predecessors(self):
        current = {'model_state_sha256': 'a' * 64, 'threads': 1}
        runner._check_prior_runtime(current, {'runtime': current}, {'runtime': dict(current)})
        for malformed in ({'model_state_sha256': 'A' * 64, 'threads': 1},
                          {'model_state_sha256': 'a' * 64, 'threads': 2},
                          {'model_state_sha256': 'b' * 64, 'threads': 1}):
            with self.assertRaises(ValueError):
                runner._check_prior_runtime(current, {'runtime': current}, {'runtime': malformed})

    def test_publication_ancestry_must_be_ordered_and_distinct(self):
        pubs = {'screen': {'commit': 'b' * 40, 'prefix': 'results/screen'},
                'development': {'commit': 'c' * 40, 'prefix': 'results/development'}}
        with patch.object(runner.subprocess, 'run') as git:
            self.assertEqual(runner._publication_ancestry(self.repo, self.source, 'evaluation', pubs),
                             [self.source, 'b' * 40, 'c' * 40])
            self.assertEqual(git.call_count, 2)
            pubs['development']['commit'] = 'b' * 40
            with self.assertRaisesRegex(ValueError, 'distinct'):
                runner._publication_ancestry(self.repo, self.source, 'evaluation', pubs)

    def test_scratch_must_not_overlap_source_or_run_in_either_direction(self):
        root = self.base / 'run'
        with patch.object(runner, 'verify_git_sources', return_value={}), \
             patch.object(runner, '_verify_prelaunch', return_value={}):
            for overlapping in (self.base, self.repo, self.cache, self.site):
                with self.subTest(scratch=overlapping):
                    with self.assertRaisesRegex(ValueError, 'scratch must be disjoint'):
                        runner.supervise(repo=self.repo, cache=self.cache, site=self.site, scratch=overlapping,
                                         root=root, phase='screen', source=self.source, prelaunch_path=self.prelaunch,
                                         prelaunch_sha=runner._file_sha(self.prelaunch)[1])
            self.assertFalse(root.exists())

    def test_low_headroom_blocks_before_claim(self):
        root = self.base / 'run'
        with patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024)):
            with self.assertRaisesRegex(RuntimeError, 'insufficient free space'):
                runner.supervise(repo=self.repo, cache=self.cache, site=self.site, scratch=self.scratch,
                                 root=root, phase='screen', source=self.source, prelaunch_path=self.prelaunch,
                                 prelaunch_sha=runner._file_sha(self.prelaunch)[1])
        self.assertFalse(root.exists())

    def test_interrupted_wait_reaps_known_child_and_does_not_promote(self):
        root = self.base / 'run'
        children = []
        class InterruptedChild:
            pid = 6060
            def __init__(self, command, **_kwargs):
                self.calls = 0
                self.killed = False
                children.append(self)
            def wait(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt()
                return -9
            def kill(self):
                self.killed = True
        with patch.object(runner, 'verify_git_sources', return_value={}), \
             patch.object(runner, '_verify_prelaunch', return_value={}), \
             patch.object(runner.subprocess, 'Popen', InterruptedChild):
            with self.assertRaises(KeyboardInterrupt):
                runner.supervise(repo=self.repo, cache=self.cache, site=self.site, scratch=self.scratch, root=root, phase='screen', source=self.source,
                                 prelaunch_path=self.prelaunch, prelaunch_sha=runner._file_sha(self.prelaunch)[1])
        self.assertTrue(children[0].killed)
        receipt = runner._read_json(root / 'screen/exit.json')
        self.assertEqual(receipt['cleanup'], 'direct-child-reaped')
        self.assertFalse((root / 'screen/validated.json').exists())
        with self.assertRaises(ValueError):
            runner._valid_prior(root / 'screen', 'screen', self.source, {})

    def test_spawn_failure_reports_unknown_child_state(self):
        root = self.base / 'run'
        with patch.object(runner, 'verify_git_sources', return_value={}), \
             patch.object(runner, '_verify_prelaunch', return_value={}), \
             patch.object(runner.subprocess, 'Popen', side_effect=OSError('fabricated')):
            with self.assertRaisesRegex(RuntimeError, 'unknown'):
                runner.supervise(repo=self.repo, cache=self.cache, site=self.site, scratch=self.scratch, root=root, phase='screen', source=self.source,
                                 prelaunch_path=self.prelaunch, prelaunch_sha=runner._file_sha(self.prelaunch)[1])
        receipt = runner._read_json(root / 'screen/exit.json')
        self.assertIn('unknown', receipt['scope'])
        self.assertIsNone(receipt['events_sha256'])
        self.assertFalse((root / 'screen/validated.json').exists())

    def test_launch_guard_accepts_only_isolated_empty_bytecode_prefix(self):
        prefix = self.base / 'fresh-bytecode'
        prefix.mkdir()
        command = [sys._base_executable, '-E', '-s', '-B', '-X', 'pycache_prefix=' + str(prefix), '-c',
                   'import sys;sys.path.insert(1,sys.argv[1]);'
                   'from looking_glass.distilbert_color_edit_runner_v1 import _check_launch; _check_launch(sys.argv[1])',
                   str(self.site)]
        real_repo = Path(__file__).resolve().parents[1]
        self.assertEqual(subprocess.run(command, cwd=real_repo, capture_output=True).returncode, 0)
        pid_command = command.copy()
        pid_command[-2] += ';import os;print(os.getpid())'
        direct = subprocess.Popen(pid_command, cwd=real_repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = direct.communicate(timeout=30)
        self.assertEqual(direct.returncode, 0, err.decode(errors='replace'))
        self.assertEqual(int(out.strip()), direct.pid)
        (prefix / 'stale.pyc').write_bytes(b'not bytecode')
        self.assertNotEqual(subprocess.run(command, cwd=real_repo, capture_output=True).returncode, 0)

    def test_missing_post_validation_marker_cannot_be_promoted(self):
        stage = self.base / 'screen'
        stage.mkdir()
        for name in ('claim.json', 'prelaunch.json', 'worker-start.json', 'events.jsonl', 'result.json', 'exit.json'):
            (stage / name).write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            runner._valid_prior(stage, 'screen', self.source, {})


if __name__ == '__main__':
    unittest.main()
