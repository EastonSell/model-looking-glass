import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import natural_agreement_heldout_child as child


def _config(tmp_path):
    remote = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'a' * 40 + '/x'
    return {'schema': 'mlg-natural-heldout-child-config-v1',
            'frozen_root': str(tmp_path / 'frozen'), 'cache_dir': str(tmp_path / 'cache'),
            'phase_dir': str(tmp_path / 'phase'), 'preparation_supervisor': str(tmp_path / 'prep-supervisor'),
            'public_files': {name: str(tmp_path / (name + '.json')) for name in child._FILE_KEYS},
            'public_urls': {name: remote for name in child._URL_KEYS}}


def test_config_accepts_paths_and_urls_only_and_rejects_private_content(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    _, loaded = child._load_config(path)
    assert loaded == config
    config['nonce'] = 'never-allow-secret-material'
    path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(child.BootstrapError):
        child._load_config(path)
    path.write_text('{"schema":0,"schema":1}', encoding='utf-8')
    with pytest.raises(child.BootstrapError):
        child._load_config(path)


def test_config_rejects_relative_paths_and_unpinned_urls(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / 'config.json'
    config['phase_dir'] = 'relative'
    path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(child.BootstrapError):
        child._load_config(path)
    config = _config(tmp_path)
    config['public_urls']['binding_url'] = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/main/x'
    path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(child.BootstrapError):
        child._load_config(path)


@pytest.mark.parametrize('name', ['MKL_CBWR', 'ATEN_CPU_CAPABILITY'])
def test_unrecorded_cpu_override_is_rejected_even_if_empty(monkeypatch, name):
    monkeypatch.delenv('MKL_CBWR', raising=False)
    monkeypatch.delenv('ATEN_CPU_CAPABILITY', raising=False)
    monkeypatch.setenv(name, '')
    with pytest.raises(child.BootstrapError):
        child._check_overrides()


def test_actual_unset_overrides_are_recorded_once(tmp_path, monkeypatch):
    monkeypatch.delenv('MKL_CBWR', raising=False)
    monkeypatch.delenv('ATEN_CPU_CAPABILITY', raising=False)
    (tmp_path / 'owner').mkdir()
    child._record_overrides(tmp_path)
    observed = json.loads((tmp_path / 'owner/runtime-overrides.json').read_bytes())
    assert observed == {'schema':'mlg-natural-heldout-runtime-overrides-v1',
                        'MKL_CBWR':None, 'ATEN_CPU_CAPABILITY':None}
    with pytest.raises(FileExistsError):
        child._record_overrides(tmp_path)


def test_staging_checks_free_space_and_preserves_unresolved_directories(tmp_path, monkeypatch):
    cache, phase = tmp_path / 'cache', tmp_path / 'phase'
    cache.mkdir()
    (phase / 'owner').mkdir(parents=True)
    monkeypatch.setattr(child.shutil, 'disk_usage', lambda path: SimpleNamespace(free=3 * 1024 ** 3))
    with pytest.raises(child.BootstrapError, match='staging entries remain'):
        with child._observe_staging(cache, phase):
            (cache / 'looking-glass-stage-uncertain').mkdir()
    after = json.loads((phase / 'owner/checkpoint-staging-after.json').read_bytes())
    assert after['new_unresolved_names'] == ['looking-glass-stage-uncertain']
    assert (cache / 'looking-glass-stage-uncertain').is_dir()


def test_staging_failure_records_after_and_preserves_primary_error(tmp_path, monkeypatch):
    cache, phase = tmp_path / 'cache', tmp_path / 'phase'
    cache.mkdir()
    (phase / 'owner').mkdir(parents=True)
    monkeypatch.setattr(child.shutil, 'disk_usage', lambda path: SimpleNamespace(free=3 * 1024 ** 3))
    with pytest.raises(ValueError, match='primary'):
        with child._observe_staging(cache, phase):
            raise ValueError('primary')
    assert (phase / 'owner/checkpoint-staging-after.json').is_file()


def test_staging_rejects_insufficient_free_space_before_loading(tmp_path, monkeypatch):
    cache, phase = tmp_path / 'cache', tmp_path / 'phase'
    cache.mkdir()
    (phase / 'owner').mkdir(parents=True)
    monkeypatch.setattr(child.shutil, 'disk_usage', lambda path: SimpleNamespace(free=3 * 1024 ** 3 - 1))
    with pytest.raises(child.BootstrapError, match='3 GiB'):
        with child._observe_staging(cache, phase):
            pytest.fail('must reject before loading')


def test_source_finder_uses_only_frozen_source(tmp_path):
    package = tmp_path / 'looking_glass'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'agreement_heldout.py').write_text('answer=42\n', encoding='utf-8')
    inventory = {'looking_glass/__init__.py': hashlib.sha256(b'').hexdigest(),
                 'looking_glass/agreement_heldout.py': hashlib.sha256(b'answer=42\n').hexdigest()}
    finder = child._FrozenSourceFinder(tmp_path, inventory)
    assert Path(finder.find_spec('looking_glass.agreement_heldout').origin) == package / 'agreement_heldout.py'
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec('scripts.unapproved')
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec('looking_glass.missing')


def test_modified_controller_is_rejected_before_its_code_runs(tmp_path):
    package = tmp_path / 'looking_glass'
    package.mkdir()
    (package / '__init__.py').write_bytes(b'')
    marker = tmp_path / 'executed'
    (package / 'natural_agreement_controller.py').write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n", encoding='utf-8')
    inventory = {'looking_glass/natural_agreement_controller.py': hashlib.sha256(b'original').hexdigest()}
    finder = child._FrozenSourceFinder(tmp_path, inventory)
    spec = finder.find_spec('looking_glass.natural_agreement_controller')
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(child.BootstrapError):
        spec.loader.exec_module(module)
    assert not marker.exists()


def test_modified_frozen_launcher_is_rejected_before_execution(tmp_path):
    tools = tmp_path / 'tools'
    tools.mkdir()
    marker = tmp_path / 'executed'
    (tools / 'natural_agreement_cpu.py').write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n", encoding='utf-8')
    with pytest.raises(child.BootstrapError):
        child._frozen_launcher(tmp_path, {'tools/natural_agreement_cpu.py': hashlib.sha256(b'original').hexdigest()})
    assert not marker.exists()


@pytest.mark.parametrize('token,expected_marker', [(b'', False), (b'x', True)])
def test_real_child_entry_gates_project_import_before_supervisor_signal(tmp_path, token, expected_marker):
    root = tmp_path / 'frozen'
    package = root / 'looking_glass'
    package.mkdir(parents=True)
    (package / '__init__.py').write_bytes(b'')
    marker = tmp_path / 'project-executed'
    source = ("def await_start(timeout=10):\n"
              "    import sys\n"
              "    if sys.stdin.buffer.read(1) != b'x':\n"
              "        raise RuntimeError('no start')\n"
              "from pathlib import Path\n"
              f"Path({str(marker)!r}).write_text('executed')\n").encode()
    (package / 'natural_agreement_controller.py').write_bytes(source)
    binding = {'source_commit': child.SOURCE_COMMIT,
               'source_sha256': {'looking_glass/__init__.py': hashlib.sha256(b'').hexdigest(),
                                 'looking_glass/natural_agreement_controller.py': hashlib.sha256(source).hexdigest()}}
    binding_raw = json.dumps(binding).encode()
    config = _config(tmp_path)
    config['frozen_root'] = str(root)
    binding_path = Path(config['public_files']['binding'])
    binding_path.write_bytes(binding_raw)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    script_path = Path(child.__file__).resolve()
    shim = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('heldout_child',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            f"m.BINDING_SHA256='{hashlib.sha256(binding_raw).hexdigest()}'; "
            "sys.argv=[sys.argv[1],sys.argv[2]]; m.main()")
    env = dict(os.environ)
    env.pop('MKL_CBWR', None)
    env.pop('ATEN_CPU_CAPABILITY', None)
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', shim, str(script_path),
                             str(config_path)], input=token, capture_output=True, env=env, timeout=10)
    assert result.returncode == 1
    assert marker.exists() is expected_marker


def test_operational_binding_checks_both_executed_source_hashes():
    bootstrap, custody = b'child', b'custody'
    op = {'sources': {'child': {'sha256': hashlib.sha256(bootstrap).hexdigest()},
                      'custody': {'sha256': hashlib.sha256(custody).hexdigest()}}}
    raw = json.dumps(op).encode()
    child._check_operational_sources(raw, bootstrap, custody)
    with pytest.raises(child.BootstrapError):
        child._check_operational_sources(raw, b'changed', custody)
    with pytest.raises(child.BootstrapError):
        child._check_operational_sources(raw, bootstrap, b'changed')


def test_public_inputs_are_constructed_from_only_bound_files_and_urls(tmp_path):
    config = _config(tmp_path)
    for name, path in config['public_files'].items():
        Path(path).write_bytes(name.encode())
    class Custody:
        PublicInputs = staticmethod(lambda **kwargs: kwargs)
    files, public = child._public_inputs(config, Custody())
    assert public['inventory_sha256'] == hashlib.sha256(b'inventory').hexdigest()
    assert public['operational_binding'] == b'operational_binding'
    assert public['wrapper_url'] == config['public_urls']['wrapper_url']
    assert 'commitment_url' not in public
    assert set(files) == child._FILE_KEYS


def test_validation_mode_dispatches_without_entering_run(monkeypatch, capsys, tmp_path):
    calls = []
    def execute(path, state, *, validation_only):
        calls.append((path, validation_only))
        state['phase'] = 'validated_no_model_forward'
    monkeypatch.setattr(child, '_execute', execute)
    monkeypatch.setattr(sys, 'argv', ['child', '--validate-only', str(tmp_path / 'config')])
    child.main()
    assert calls == [(str(tmp_path / 'config'), True)]
    assert json.loads(capsys.readouterr().out) == {'status':'validated_no_model_forward'}


def test_child_reports_keyboard_cancellation_without_case_content(monkeypatch, capsys, tmp_path):
    def execute(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(child, '_execute', execute)
    monkeypatch.setattr(sys, 'argv', ['child', str(tmp_path / 'config')])
    with pytest.raises(SystemExit) as error:
        child.main()
    assert error.value.code == 130
    assert json.loads(capsys.readouterr().err) == {'status':'cancelled', 'phase':'configuration',
                                                   'error_type':'KeyboardInterrupt'}


def test_public_validation_does_not_read_a_private_commitment(tmp_path, monkeypatch):
    config = _config(tmp_path)
    calls = []
    monkeypatch.setattr(child, '_check_published', lambda *args: calls.append(args))
    child._check_public_operation(config, b'public-operation', require_commitment=False)
    assert len(calls) == 1
    with pytest.raises(FileNotFoundError):
        child._check_public_operation(config, b'public-operation', require_commitment=True)


def test_validation_rejects_a_claimed_phase_before_project_import(tmp_path, monkeypatch):
    config = _config(tmp_path)
    phase = Path(config['phase_dir'])
    phase.mkdir()
    (phase / 'claim.json').write_bytes(b'not-read')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr(child, '_check_environment', lambda: None)
    with pytest.raises(child.BootstrapError, match='separate unclaimed'):
        child._execute(path, {}, validation_only=True)


def test_validation_rejects_an_unresolved_marker_path(tmp_path, monkeypatch):
    config = _config(tmp_path)
    phase = Path(config['phase_dir'])
    phase.mkdir()
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr(child, '_check_environment', lambda: None)
    actual_lexists = os.path.lexists
    monkeypatch.setattr(child.os.path, 'lexists',
                        lambda value: True if Path(value) == phase / 'nonce.bin' else actual_lexists(value))
    with pytest.raises(child.BootstrapError, match='separate unclaimed'):
        child._execute(path, {}, validation_only=True)


def _fake_pipeline(tmp_path, *, fail_verification=False, fail_owner=False, wrong_state=False):
    steps = []
    freeze, snapshot = SimpleNamespace(sha256='freeze'), SimpleNamespace(sha256='inventory')
    commitment = {'heldout_commitment': 'c' * 64}
    prepared = (freeze, snapshot, b'private-nonce', b'private-record', commitment)
    class Custody:
        observe_current_inventory = object()
        def verify_prepared(self, *args, **kwargs):
            steps.append('verify')
            if fail_verification:
                raise RuntimeError('fabricated verification failure')
            return prepared
        def start_owner_once(self, *args):
            steps.append('start')
            if fail_owner:
                raise RuntimeError('fabricated exclusive start failure')
        def tokenize_assignment(self, allocation, tokenizer):
            steps.append('heldout-alignment')
            return 'fabricated-cases'
    class Preflight:
        def _tokenizer(self, cache):
            steps.append('tokenizer')
            return 'tokenizer'
        def tokenize_inputs(self, tokenizer):
            steps.append('public-alignment')
            return ['rows'], ['groups']
    class Heldout:
        def run_committed_heldout(self, runner, **kwargs):
            steps.append('run')
            assert kwargs['freeze'] is freeze and kwargs['record_bytes'] == b'private-record'
            assert kwargs['heldout_commitment'] == commitment['heldout_commitment']
    class Commitments:
        def sample_families(self, *args, **kwargs):
            steps.append('resample')
            return 'fabricated-allocation'
        def seal_heldout(self, *args, **kwargs):
            steps.append('reseal')
            return SimpleNamespace(record_bytes=b'private-record',
                                   commitment_sha256=commitment['heldout_commitment'])
    class Control:
        def bind_public_cases(self, manifest, rows, groups):
            return ['cases']
    binding = {'checkpoint_revision': 'revision', 'assets_sha256': {'a':'b'}, 'cases': ['cases']}
    bundle = SimpleNamespace(model=object(), tokenizer='loaded-tokenizer',
                             metadata={'model_revision':'revision', 'asset_sha256': {'a':'b'},
                                       'device':'cpu', 'dtype':'float32', 'attention_implementation':'eager',
                                       'use_cache':False})
    def load_model(**kwargs):
        steps.append('model')
        return bundle
    def state_hash(model):
        steps.append('hash')
        return 'wrong' if wrong_state else child.MODEL_STATE_SHA256
    def journal(path, metadata, **kwargs):
        steps.append('journal')
        assert path == tmp_path / 'phase' / 'owner' / 'attempt'
        assert kwargs == {'max_total_bytes':160 * 1024 * 1024, 'max_event_log_bytes':48 * 1024 * 1024}
        return object()
    def runner(model, journal, **kwargs):
        steps.append('runner')
        assert kwargs == {'device':'cpu', 'expected_quartets':4}
        return object()
    deps = {'preflight': Preflight(), 'commitments': Commitments(), 'heldout': Heldout(), 'control': Control(),
            'load_model': load_model, 'state_hash': state_hash, 'journal': journal, 'runner': runner,
            'manifest': b'manifest'}
    public = SimpleNamespace(binding=b'binding', prediction=b'prediction')
    config = _config(tmp_path)
    frozen = {'_start_record': lambda raw, b: {'binding_sha256':'bound'}}
    state = {}
    return steps, lambda: child._run_pipeline(frozen, config, binding, public, Custody(), b'wrapper', deps, state)


def test_committed_verification_and_one_shot_start_precede_model_and_forward(tmp_path):
    steps, run = _fake_pipeline(tmp_path)
    run()
    assert steps == ['tokenizer', 'verify', 'start', 'model', 'hash', 'public-alignment',
                     'resample', 'heldout-alignment', 'reseal', 'journal', 'runner', 'run', 'hash']


@pytest.mark.parametrize('failure', ['verification', 'owner', 'state'])
def test_rejected_gates_never_reach_a_forward(tmp_path, failure):
    steps, run = _fake_pipeline(tmp_path, fail_verification=failure == 'verification',
                                fail_owner=failure == 'owner', wrong_state=failure == 'state')
    with pytest.raises((RuntimeError, child.BootstrapError)):
        run()
    assert 'run' not in steps
    if failure != 'state':
        assert 'model' not in steps
