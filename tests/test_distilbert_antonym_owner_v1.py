"""No-model wiring checks for the reused bounded CPU supervisor."""
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import pytest


def _module():
    name = 'looking_glass.distilbert_antonym_runner_v1'
    assert importlib.util.find_spec(name) is not None, 'antonym CPU owner is not implemented'
    return importlib.import_module(name)


def test_tokenizer_adapter_preserves_ids_and_tokens():
    module = _module()

    class HFTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [101, 103, 102] if add_special_tokens else [103]

        def convert_ids_to_tokens(self, ids):
            return [{101: '[CLS]', 102: '[SEP]', 103: '[MASK]'}[item] for item in ids]

    adapter = module.TokenizerAdapter(HFTokenizer())
    encoded = adapter.encode('[MASK]')
    assert encoded.ids == [101, 103, 102]
    assert encoded.tokens == ['[CLS]', '[MASK]', '[SEP]']
    assert adapter.id_to_token(103) == '[MASK]'
    assert adapter.encode('[MASK]', add_special_tokens=False).ids == [103]


def test_configure_selects_the_twelve_row_screen_without_loading_model():
    module = _module()
    configure = getattr(module, 'configure', None)
    assert callable(configure), 'antonym supervisor configuration is not implemented'
    from looking_glass import distilbert_color_preference_runner_v2 as base
    names = ('CASE_SHA256', 'CHECK_SHA256', 'SOURCE_PATHS', 'MAX_SCREEN_STARTS', 'BOOTSTRAP',
             'preference', 'stages', 'public', '_load_model', '_verify_prelaunch')
    saved = {name: getattr(base, name) for name in names}
    try:
        configured = configure()
        assert configured is base and base.MAX_SCREEN_STARTS == 12
        assert base.CASE_SHA256 == '284fc9adc0cc3caa52ea5d122c1f08db50c8dda2d881297e2dcd1f2229cb6386'
        assert base.preference.__name__.endswith('distilbert_antonym_screen_v1')
        assert base.stages is base.preference
        assert 'looking_glass/distilbert_antonym_runner_v1.py' in base.SOURCE_PATHS
        assert 'distilbert_antonym_runner_v1' in base.BOOTSTRAP
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def test_launch_scan_has_same_commit_binding_for_worker_and_supervisor(monkeypatch):
    module = _module()
    fake_commit = '1' * 40
    monkeypatch.setattr(module, 'ensure_commit', lambda repo, commit: None)
    monkeypatch.setattr(module.public, 'scan', lambda repo, commit, cases, **kwargs:
                        {'public_commit': commit, 'related_pairs': []})
    scan = module.CurrentPublic.scan('.', fake_commit, [])
    assert scan['commit'] == scan['public_commit'] == fake_commit


def test_ensure_commit_uses_default_git_and_rejects_a_blob():
    module = _module()
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
    blob = subprocess.check_output(['git', '-C', str(repo), 'rev-parse',
                                    'HEAD:looking_glass/distilbert_antonym_runner_v1.py']).decode().strip()
    module.ensure_commit(repo, commit)
    with pytest.raises(ValueError, match='not a commit'):
        module.ensure_commit(repo, blob)


def test_ensure_commit_fetches_missing_object_then_checks_type(monkeypatch):
    module = _module()
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            return SimpleNamespace(returncode=1, stdout=b'', stderr=b'missing')
        if len(commands) == 2:
            assert 'fetch' in command
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
        return SimpleNamespace(returncode=0, stdout=b'commit\n', stderr=b'')
    monkeypatch.setattr(module.subprocess, 'run', run)
    module.ensure_commit('repo', 'a' * 40)
    assert len(commands) == 3
    assert commands[0][-3:] == ['cat-file', '-t', 'a' * 40]
    assert commands[2][-3:] == ['cat-file', '-t', 'a' * 40]


def test_observer_targets_the_antonym_supervisor():
    name = 'scripts.observe_distilbert_antonym_v1'
    assert importlib.util.find_spec(name) is not None, 'antonym observer is not implemented'
    module = importlib.import_module(name)
    assert 'looking_glass.distilbert_antonym_runner_v1' in module.BOOTSTRAP


def test_prelaunch_document_binds_the_fixed_cases_and_scope():
    name = 'scripts.prepare_distilbert_antonym_prelaunch_v1'
    assert importlib.util.find_spec(name) is not None, 'antonym prelaunch builder is not implemented'
    module = importlib.import_module(name)
    scan = {'public_commit': '1' * 40, 'commit': '1' * 40, 'related_pairs': []}
    document = module.make_document('2' * 40, scan, [], 'named public sources')
    assert document['source_commit'] == '2' * 40
    assert document['case_sha256'] == _module().CASE_SHA256
    assert document['public_scan'] == scan
    assert document['scope'] == 'named public sources'


def test_prelaunch_checks_the_independent_tiny_tail_regression():
    module = _module()
    check = getattr(module, 'check_tiny_tail', None)
    assert callable(check), 'pre-forward tiny-tail check is not implemented'
    result = check()
    assert result['precision_digits'] >= 100
    assert result['positive_correction'] is True


def test_fabricated_supervisor_retains_twelve_rows_and_cannot_repeat(tmp_path, monkeypatch):
    module = _module()
    from looking_glass import distilbert_color_preference_runner_v2 as original
    for name in ('CASE_SHA256', 'CHECK_SHA256', 'SOURCE_PATHS', 'MAX_SCREEN_STARTS', 'BOOTSTRAP',
                 'preference', 'stages', 'public', '_load_model', '_verify_prelaunch'):
        monkeypatch.setattr(original, name, getattr(original, name))
    base = module.configure()
    from looking_glass import distilbert_color_edit_records_v1 as records
    from looking_glass import distilbert_antonym_screen_v1 as screen
    repo, cache, site, scratch = (tmp_path / item for item in ('repo', 'cache', 'site', 'scratch'))
    for path in (repo, cache, site, scratch):
        path.mkdir()
    (repo / 'experiments').mkdir()
    source_case = Path(__file__).resolve().parents[1] / module.public.CASE_PATH
    (repo / module.public.CASE_PATH).write_bytes(source_case.read_bytes())
    manifest = json.loads(source_case.read_bytes())
    prelaunch = tmp_path / 'prelaunch.json'
    prelaunch.write_bytes(b'{}\n')
    root, source = tmp_path / 'run', 'a' * 40
    scan = {'commit': 'b' * 40, 'public_commit': 'b' * 40, 'related_pairs': []}

    class FakeChild:
        pid = 4242
        def __init__(self, command, **_kwargs):
            stage = Path(command[command.index('--run-root') + 1]) / 'screen'
            claim = base._read_json(stage / 'claim.json')
            assert claim['remaining_starts'] == 12
            base._write(stage / 'worker-start.json', base._json_bytes(
                {'phase': 'screen', 'pid': self.pid, 'source_commit': source}))
            with records.Journal(stage / 'events.jsonl', byte_cap=claim['journal_byte_cap'],
                                 forward_cap=claim['remaining_starts']) as journal:
                for row in screen.screen_rows(manifest):
                    binding = {'action': journal.forwards, 'phase': 'screen', 'condition': 'original',
                               'object': row['object'], 'case_id': row['id'], 'input_ids': row['input_ids'],
                               'attention_mask': row['attention_mask'], 'mask_index': row['mask_index']}
                    journal.start(binding)
                    logits = [0.] * 30522
                    logits[row['word_ids'][row['factual']]] = 2.
                    journal.emit({'kind': 'complete', **binding, 'parameter': None, 'parameter_sha256': None,
                                  'realized_norm': 0., 'logits': records.encode_raw(struct.pack('<30522f', *logits))})
            raw = (stage / 'events.jsonl').read_bytes()
            runtime = {'runtime': base.RUNTIME, 'python': '3.11.15', 'model': 'distilbert/distilbert-base-uncased',
                       'revision': '12040accade4e8a0f71eabdb258fecc2e7e948be', 'model_state_sha256': 'e' * 64,
                       'device': 'cpu', 'dtype': 'float32', 'eager': True, 'intraop_threads': 1,
                       'interop_threads': 1, 'deterministic_algorithms': True, 'cuda_initialized': False,
                       'decoder_weight_tied_to_input': True, 'decoder_bias_sha256': base.DECODER_BIAS_SHA256,
                       'site_path_sha256': claim['site_path_sha256'], 'scratch_root_sha256': claim['scratch_root_sha256']}
            result = {'schema': 'distilbert-color-stage-result-v1', 'phase': 'screen', 'source_commit': source,
                      'status': 'complete', 'source_sha256': {}, 'runtime': runtime, 'case_sha256': module.CASE_SHA256,
                      'events_sha256': base._sha(raw), 'event_bytes': len(raw), 'starts': 12,
                      'summary': screen.rebuild_screen(manifest, records.parse_jsonl(raw))}
            base._write(stage / 'result.json', base._json_bytes(result))

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(base, '_check_launch', lambda site: None)
    monkeypatch.setattr(base.shutil, 'disk_usage', lambda path: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(base, 'verify_git_sources', lambda *args: {})
    monkeypatch.setattr(base, '_verify_prelaunch', lambda *args, **kwargs: {'current_scan': scan})
    monkeypatch.setattr(base.subprocess, 'Popen', FakeChild)
    result = base.supervise(repo=repo, cache=cache, site=site, scratch=scratch, root=root, phase='screen',
                            source=source, prelaunch_path=prelaunch, prelaunch_sha=base._file_sha(prelaunch)[1])
    assert result['started_forwards'] == 12
    assert base._read_json(root / 'screen/result.json')['summary']['selected_object'] == 'box'
    assert (root / 'screen/validated.json').is_file()
    with pytest.raises(FileExistsError):
        base.supervise(repo=repo, cache=cache, site=site, scratch=scratch, root=root, phase='screen',
                       source=source, prelaunch_path=prelaunch, prelaunch_sha=base._file_sha(prelaunch)[1])
