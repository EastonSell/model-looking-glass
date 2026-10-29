"""Binding and publication gates use only fixed public inputs and fake assets."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from looking_glass import agreement_preflight_v1 as preflight
from looking_glass import pretrained
from looking_glass.agreement_journal import AgreementJournal
from looking_glass import natural_agreement_controller as control
from tools import natural_agreement_cpu as launch
import huggingface_hub


@pytest.fixture
def setup_binding(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[1]
    names = ['docs/PRETRAINED_NUMBER_PATCH_V1.md', launch.CLARIFICATION_PATH, launch.MANIFEST_PATH,
             'results/distilgpt2-agreement-preflight-v1-20261028/run/result.json',
             'results/distilgpt2-agreement-preflight-v1-20261028/run/last_logits.f32']
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, target)
    manifest = json.loads((tmp_path / launch.MANIFEST_PATH).read_bytes())
    required = ['tools/natural_agreement_cpu.py', launch.AUDITOR_PATH, 'scripts/v2_source_guards.py',
                'looking_glass/natural_agreement_controller.py',
                'looking_glass/agreement_patch_runner.py', 'looking_glass/agreement_journal.py',
                'looking_glass/agreement_heldout.py', 'looking_glass/pretrained_number_commitments.py',
                'looking_glass/natural_prediction_freeze.py', 'looking_glass/agreement_metrics.py',
                'looking_glass/agreement_numeric.py', 'looking_glass/distilgpt2_intervention_dev.py',
                'looking_glass/intervention_plan.py', 'looking_glass/interventions.py',
                'looking_glass/model_ownership.py', 'looking_glass/pretrained_exposures.py',
                'looking_glass/pretrained_exposures_v2.py', 'looking_glass/pretrained_exposures_v3.py',
                'looking_glass/pretrained_exposures_v4.py', 'looking_glass/pretrained_exposures_v5.py',
                'looking_glass/pretrained_exposures_v6.py',
                'looking_glass/pretrained_exposures_v7.py',
                'looking_glass/agreement_preflight_v1.py', 'looking_glass/pretrained.py']
    sources = {name: 'a' * 64 for name in required}
    sources[launch.AUDITOR_PATH] = launch.AUDITOR
    monkeypatch.setattr(launch, 'ROOT', tmp_path)
    monkeypatch.setattr(control, 'source_inventory', lambda root: ('b' * 40, sources))
    monkeypatch.setattr(launch, '_runtime', lambda: {'device': 'cpu', 'cpu_threads': 1})
    monkeypatch.setattr(preflight, '_tokenizer', lambda cache: object())
    monkeypatch.setattr(preflight, 'tokenize_inputs', lambda tokenizer: (manifest['rows'], manifest['groups']))
    monkeypatch.setattr(pretrained, 'load_distilgpt2', lambda **kw: pytest.fail('model loading must not occur'))
    assets_dir = tmp_path / 'assets'
    assets_dir.mkdir()
    fake_assets = {name: (1, digest) for name, digest in manifest['asset_sha256'].items()}
    monkeypatch.setattr(pretrained, 'ASSETS', fake_assets)
    for name in fake_assets:
        (assets_dir / name).write_bytes(b'x')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *, filename, **kw: str(assets_dir / filename))
    actual_sha = launch._sha
    corrupt = set()
    def hash_fixture(path):
        path = Path(path)
        if path.parent == assets_dir:
            return '0' * 64 if path.name in corrupt else fake_assets[path.name][1]
        return actual_sha(path)
    monkeypatch.setattr(launch, '_sha', hash_fixture)
    return tmp_path, manifest, corrupt


def test_prepare_binds_public_rows_without_loading_model(setup_binding):
    root, _, _ = setup_binding
    binding = launch.build_binding(root / 'assets')
    assert binding['source_commit'] == 'b' * 40
    assert len(binding['cases']) == 16
    assert binding['cases'][0]['input_ids'] == [464, 3797, 1474, 262, 9168]
    assert binding['preflight_status'] == 'passed; public exposed development'
    assert binding['natural_prediction_clarification_sha256'] == launch.CLARIFICATION
    assert binding['independent_natural_audit_source_sha256'] == launch.AUDITOR
    assert binding['development_exposure_inventory_sha256'] == launch.EXPOSURE
    budget = binding['study_budget']
    assert budget['prior_unique_tensor_bytes'] == 3216448
    assert budget['phase_reservations_bytes'] == {'natural_discovery': 128 * 1024**2,
                                                 'natural_heldout': 160 * 1024**2,
                                                 'injection_discovery': 48 * 1024**2,
                                                 'injection_heldout': 72 * 1024**2}
    assert budget['unallocated_bytes'] == 105835456
    assert budget['host_disk_allowance_bytes'] == 1024**3
    assert budget['host_disk_phase_reservations_bytes'] == {'natural_discovery': 224 * 1024**2,
                                                           'natural_heldout': 224 * 1024**2,
                                                           'injection_discovery': 128 * 1024**2,
                                                           'injection_heldout': 192 * 1024**2}
    assert budget['host_disk_unallocated_bytes'] == 256 * 1024**2
    assert budget['coordination'].startswith('external single study owner')


def test_budget_claim_is_durable_and_not_reusable(setup_binding, tmp_path):
    root, _, _ = setup_binding
    binding = launch.build_binding(root / 'assets')
    raw = launch._bytes(binding)
    binding_path = tmp_path / 'frozen-binding.json'
    output = tmp_path / 'run'
    output.mkdir()
    launch._claim_budget(binding_path, raw, binding, output)
    record = json.loads((tmp_path / 'frozen-binding.json.natural-claim.json').read_bytes())
    assert record['binding_sha256'] == hashlib.sha256(raw).hexdigest()
    assert record['output'] == str(output)
    assert record['free_disk_bytes'] >= binding['study_budget']['required_free_disk_bytes']
    with pytest.raises(FileExistsError):
        launch._claim_budget(binding_path, raw, binding, output)


def test_prepare_rejects_missing_bound_runner_before_model(setup_binding, monkeypatch):
    root, _, _ = setup_binding
    monkeypatch.setattr(control, 'source_inventory', lambda root: ('b' * 40, {}))
    with pytest.raises(RuntimeError, match='required committed study source'):
        launch.build_binding(root / 'assets')


@pytest.mark.parametrize('missing', ['looking_glass/agreement_heldout.py',
                                    'looking_glass/pretrained_number_commitments.py',
                                    'looking_glass/natural_prediction_freeze.py', launch.AUDITOR_PATH])
def test_prepare_rejects_missing_prospective_analysis_before_model(setup_binding, monkeypatch, missing):
    root, _, _ = setup_binding
    binding = launch.build_binding(root / 'assets')
    sources = dict(binding['source_sha256'])
    del sources[missing]
    monkeypatch.setattr(control, 'source_inventory', lambda root: ('b' * 40, sources))
    with pytest.raises(RuntimeError, match='required committed study source'):
        launch.build_binding(root / 'assets')


def test_prepare_rejects_changed_independent_audit_source(setup_binding, monkeypatch):
    root, _, _ = setup_binding
    binding = launch.build_binding(root / 'assets')
    sources = dict(binding['source_sha256'])
    sources[launch.AUDITOR_PATH] = '0' * 64
    monkeypatch.setattr(control, 'source_inventory', lambda root: ('b' * 40, sources))
    with pytest.raises(RuntimeError, match='audit source differs'):
        launch.build_binding(root / 'assets')


@pytest.mark.parametrize('artifact', ['protocol', 'clarification', 'manifest', 'result', 'archive', 'asset', 'tokenizer'])
def test_prepare_rejects_mutated_public_inputs_before_model(setup_binding, monkeypatch, artifact):
    root, manifest, corrupt = setup_binding
    if artifact == 'protocol':
        path = root / 'docs/PRETRAINED_NUMBER_PATCH_V1.md'
        path.write_bytes(path.read_bytes() + b'!')
    elif artifact == 'clarification':
        path = root / launch.CLARIFICATION_PATH
        path.write_bytes(path.read_bytes() + b'!')
    elif artifact == 'manifest':
        path = root / launch.MANIFEST_PATH
        path.write_bytes(path.read_bytes() + b' ')
    elif artifact == 'result':
        path = root / 'results/distilgpt2-agreement-preflight-v1-20261028/run/result.json'
        path.write_bytes(path.read_bytes() + b' ')
    elif artifact == 'archive':
        path = root / 'results/distilgpt2-agreement-preflight-v1-20261028/run/last_logits.f32'
        path.write_bytes(path.read_bytes() + b'!')
    elif artifact == 'asset':
        corrupt.add('model.safetensors')
    else:
        rows = json.loads(json.dumps(manifest['rows']))
        rows[0]['subject_index'] = 2
        monkeypatch.setattr(preflight, 'tokenize_inputs', lambda tokenizer: (rows, manifest['groups']))
    with pytest.raises((RuntimeError, control.InventoryError)):
        launch.build_binding(root / 'assets')


def test_publication_requires_exact_remote_binding_and_source(monkeypatch):
    binding = {'source_commit': 'a' * 40,
               'natural_prediction_clarification_sha256': hashlib.sha256(launch.CLARIFICATION_PATH.encode()).hexdigest(),
               'independent_natural_audit_source_sha256': hashlib.sha256(launch.AUDITOR_PATH.encode()).hexdigest(),
               'source_sha256': {name: hashlib.sha256(name.encode()).hexdigest() for name in
                                 ('tools/natural_agreement_cpu.py', launch.AUDITOR_PATH,
                                  'looking_glass/natural_agreement_controller.py',
                                  'looking_glass/agreement_patch_runner.py', 'looking_glass/agreement_journal.py',
                                  'looking_glass/agreement_heldout.py', 'looking_glass/pretrained_number_commitments.py',
                                  'looking_glass/natural_prediction_freeze.py', 'scripts/v2_source_guards.py')}}
    raw = launch._bytes(binding)
    url = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'b' * 40 + '/results/binding.json'
    def open_url(request, timeout):
        if request == url:
            return io.BytesIO(raw)
        return io.BytesIO(request.split('/' + 'a' * 40 + '/', 1)[1].encode())
    monkeypatch.setattr(launch.urllib.request, 'urlopen', open_url)
    launch._published(raw, url)
    with pytest.raises(RuntimeError):
        launch._published(raw + b' ', url)
    binding['source_commit'] = 'c' * 40
    wrong_source = launch._bytes(binding)
    monkeypatch.setattr(launch.urllib.request, 'urlopen',
                        lambda request, timeout: io.BytesIO(wrong_source if request == url else b'wrong source'))
    with pytest.raises(RuntimeError, match='source is not publicly retrievable'):
        launch._published(wrong_source, url)


def test_child_rejects_changed_runtime_binding_before_loading_model(setup_binding, monkeypatch):
    root, _, _ = setup_binding
    binding = launch.build_binding(root / 'assets')
    binding['runtime']['cpu_threads'] = 2
    path = root / 'binding.json'
    path.write_bytes(launch._bytes(binding))
    monkeypatch.setattr(launch, '_published', lambda raw, url: None)
    with pytest.raises(RuntimeError, match='child runtime/source/assets differ'):
        launch._child_guarded(root / 'assets', path, root, 'url')
    assert not (root / 'attempt').exists()


def test_validation_child_returns_before_model_loading(setup_binding, monkeypatch, capsys):
    root, _, _ = setup_binding
    binding = launch.build_binding(root / 'assets')
    path = root / 'binding.json'
    path.write_bytes(launch._bytes(binding))
    monkeypatch.setattr(launch, '_published', lambda raw, url: None)
    launch._child_guarded(root / 'assets', path, root, 'url', validation_only=True)
    message = json.loads(capsys.readouterr().out)
    assert message['status'] == 'validated_no_model_forward'
    assert message['binding_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not (root / 'attempt').exists()


def test_child_enforces_frozen_retained_tensor_cap_without_model_forward(setup_binding, monkeypatch):
    from looking_glass import agreement_patch_runner
    from looking_glass.agreement_journal import JournalError
    root, _, _ = setup_binding
    binding = launch.build_binding(root / 'assets')
    binding['study_budget']['phase_reservations_bytes']['natural_discovery'] = 12
    path = root / 'binding.json'
    path.write_bytes(launch._bytes(binding))
    monkeypatch.setattr(launch, 'build_binding', lambda cache, **kwargs: binding)
    monkeypatch.setattr(launch, '_published', lambda raw, url: None)
    fake = SimpleNamespace(model=object(), tokenizer=object(), metadata={
        'model_revision': launch.CHECKPOINT, 'asset_sha256': binding['assets_sha256'],
        'device': 'cpu', 'dtype': 'float32', 'attention_implementation': 'eager', 'use_cache': False})
    monkeypatch.setattr(pretrained, 'load_distilgpt2', lambda **kw: fake)
    class FakeRunner:
        def __init__(self, model, journal, **kw):
            self.journal = journal
        def run(self, cases):
            self.journal.record({'kind': 'fabricated'}, {'x': ([2], b'\x00' * 8)})
            self.journal.record({'kind': 'fabricated'}, {'y': ([2], b'\x01' * 8)})
    monkeypatch.setattr(agreement_patch_runner, 'NaturalAgreementRunner', FakeRunner)
    with pytest.raises(JournalError, match='retained tensor payload exceeds'):
        launch._child_guarded(root / 'assets', path, root, 'url')
    assert sum(p.stat().st_size for p in (root / 'attempt/tensors').iterdir()) == 8


def test_child_holds_its_own_source_lease_during_work(monkeypatch, tmp_path):
    import scripts.v2_source_guards as guards
    state = {'held': False}
    class Lease:
        def __init__(self, root, files):
            pass
        def __enter__(self):
            state['held'] = True
        def __exit__(self, *args):
            state['held'] = False
    monkeypatch.setattr(control, 'await_start', lambda: None)
    monkeypatch.setattr(launch, '_startup', lambda: None)
    monkeypatch.setattr(launch, '_configure_runtime', lambda: None)
    monkeypatch.setattr(launch, '_install_owner_audit_hook', lambda: state.update({'hook_installed': True}))
    source = ('a' * 40, {'looking_glass/x.py': 'b' * 64})
    binding_path = tmp_path / 'binding'
    binding_path.write_bytes(launch._bytes({'source_commit': source[0], 'source_sha256': source[1]}))
    monkeypatch.setattr(control, 'verify_source_inventory', lambda root, revision, files: source)
    monkeypatch.setattr(guards, 'SourceFileLease', Lease)
    def guarded(*args, **kwargs):
        assert state['held'] is True
    monkeypatch.setattr(launch, '_child_guarded', guarded)
    launch._child(tmp_path, binding_path, tmp_path, 'url')
    assert state['held'] is False and state['hook_installed'] is True


def test_started_journal_records_exact_runtime_binding(tmp_path):
    binding = {'runtime': {'device': 'cpu', 'cpu_threads': 1}, 'source_commit': 'a' * 40}
    raw = launch._bytes(binding)
    expected = launch._start_record(raw, binding)
    journal = AgreementJournal(tmp_path / 'attempt', expected)
    journal.finish('complete', {'forwards': 0})
    inventory = control.audit_storage(tmp_path / 'attempt')
    control.check_start(inventory['_started'], expected)
    assert inventory['_started']['binding']['runtime']['cpu_threads'] == 1


def test_run_records_preflight_failure_without_starting_child(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, '_startup', lambda: None)
    monkeypatch.setattr(launch, '_configure_runtime', lambda: None)
    def bad_source(root):
        raise control.InventoryError('source checkout changed')
    monkeypatch.setattr(control, 'source_inventory', bad_source)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['natural_agreement_cpu.py', 'run', '--cache', str(tmp_path),
                                      '--binding', str(tmp_path / 'binding'), '--output', str(output),
                                      '--published-url', 'https://invalid.example'])
    with pytest.raises(control.InventoryError):
        launch.main()
    terminal = json.loads((output / 'launcher-terminal.json').read_text())
    assert terminal['status'] == 'preflight_failed'
    assert terminal['phase'] == 'source_acquisition'
    assert terminal['child_started'] is False
    assert (output / 'launcher-error.txt').is_file()
    assert not (output / 'owner').exists()


def test_preflight_terminal_survives_error_log_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, '_startup', lambda: (_ for _ in ()).throw(ValueError('primary')))
    original_write = launch._write
    def failing_log(path, data):
        if Path(path).name == 'launcher-error.txt':
            raise OSError('diagnostic-only failure')
        original_write(path, data)
    monkeypatch.setattr(launch, '_write', failing_log)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['natural_agreement_cpu.py', 'run', '--cache', str(tmp_path),
                                      '--binding', str(tmp_path / 'binding'), '--output', str(output),
                                      '--published-url', 'https://invalid.example'])
    with pytest.raises(ValueError, match='primary'):
        launch.main()
    terminal = json.loads((output / 'launcher-terminal.json').read_text())
    assert terminal['status'] == 'preflight_failed' and terminal['error_type'] == 'ValueError'
    assert terminal['error_log_status'] == 'unavailable' and terminal['error_log_error_type'] == 'OSError'
    assert terminal['child_started'] is False


def test_fresh_cancellation_during_diagnostic_preserves_primary_context(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, '_startup', lambda: (_ for _ in ()).throw(ValueError('primary')))
    original_write = launch._write
    def interrupt_log(path, data):
        if Path(path).name == 'launcher-error.txt':
            raise KeyboardInterrupt()
        original_write(path, data)
    monkeypatch.setattr(launch, '_write', interrupt_log)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['natural_agreement_cpu.py', 'run', '--cache', str(tmp_path),
                                      '--binding', str(tmp_path / 'binding'), '--output', str(output),
                                      '--published-url', 'https://invalid.example'])
    with pytest.raises(KeyboardInterrupt) as exc:
        launch.main()
    assert isinstance(exc.value.__cause__, ValueError)
    terminal = json.loads((output / 'launcher-terminal.json').read_text())
    assert terminal['error_type'] == 'ValueError' and terminal['error_log_error_type'] == 'KeyboardInterrupt'


def test_fresh_cancellation_survives_a_terminal_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, '_startup', lambda: (_ for _ in ()).throw(ValueError('primary')))
    original_write = launch._write
    def failing_writes(path, data):
        if Path(path).name == 'launcher-error.txt':
            raise KeyboardInterrupt()
        if Path(path).name == 'launcher-terminal.json':
            raise OSError('terminal failure')
        original_write(path, data)
    monkeypatch.setattr(launch, '_write', failing_writes)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['natural_agreement_cpu.py', 'run', '--cache', str(tmp_path),
                                      '--binding', str(tmp_path / 'binding'), '--output', str(output),
                                      '--published-url', 'https://invalid.example'])
    with pytest.raises(KeyboardInterrupt) as exc:
        launch.main()
    assert isinstance(exc.value.__cause__, OSError)
    assert isinstance(exc.value.__cause__.__cause__, ValueError)
    assert not (output / 'launcher-terminal.json').exists()


def test_dependency_configuration_rejects_missing_home(tmp_path, monkeypatch):
    venv = tmp_path / 'fake-venv'
    site = venv / 'Lib/site-packages'
    site.mkdir(parents=True)
    (venv / 'pyvenv.cfg').write_text('implementation = CPython\nversion_info = ' +
                                    '.'.join(map(str, sys.version_info[:3])) +
                                    '\ninclude-system-site-packages = false\n')
    monkeypatch.setenv('MLG_NATURAL_SITE', str(site.resolve()))
    monkeypatch.syspath_prepend(str(site.resolve()))
    monkeypatch.setattr(sys, 'base_prefix', str(tmp_path))
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'python.exe'))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match='matching direct base interpreter'):
        launch._dependency_configuration()


def test_terminal_binds_exact_supervisor_record(tmp_path, monkeypatch):
    import scripts.v2_source_guards as guards
    monkeypatch.setattr(launch, '_startup', lambda: None)
    monkeypatch.setattr(launch, '_configure_runtime', lambda: None)
    monkeypatch.setattr(control, 'source_inventory', lambda root: ('a' * 40, {'looking_glass/x.py': 'b' * 64}))
    class Lease:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(guards, 'SourceFileLease', Lease)
    output = tmp_path / 'run'
    supervisor = b'{"status":"evidence_complete"}\n'
    def guarded(args, binding_path, state):
        (output / 'owner').mkdir()
        (output / 'owner/supervisor.json').write_bytes(supervisor)
        return {'status': 'evidence_complete', 'child_reaped': True,
                'cleanup_verified': True, 'retry_safe': True}
    monkeypatch.setattr(launch, '_guarded_main', guarded)
    monkeypatch.setattr(sys, 'argv', ['natural_agreement_cpu.py', 'run', '--cache', str(tmp_path),
                                      '--binding', str(tmp_path / 'binding'), '--output', str(output),
                                      '--published-url', 'https://invalid.example'])
    launch.main()
    terminal = json.loads((output / 'launcher-terminal.json').read_bytes())
    assert terminal['owner_supervisor_sha256'] == hashlib.sha256(supervisor).hexdigest()
    assert terminal['owner_supervisor_sha256'] != hashlib.sha256(b'other').hexdigest()


def test_runtime_observation_does_not_reset_denormal_state(monkeypatch):
    import torch
    monkeypatch.setattr(launch, '_dependency_configuration', lambda: {})
    assert torch.set_flush_denormal(True)
    try:
        with pytest.raises(RuntimeError, match='subnormal probe failed'):
            launch._runtime()
        tiny = torch.tensor([1], dtype=torch.int32).view(torch.float32)
        assert int((tiny * 1.0).view(torch.int32).item()) == 0
    finally:
        torch.set_flush_denormal(False)


def test_runtime_observation_does_not_spawn_shell(monkeypatch):
    monkeypatch.setattr(launch, '_dependency_configuration', lambda: {})
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('runtime spawned a subprocess'))
    observed = launch._runtime()
    assert observed['platform'] == 'Windows'
    assert type(observed['windows_version']['build']) is int


def test_owner_audit_hook_rejects_python_subprocess_creation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    marker = tmp_path / 'subprocess-started'
    code = ("import sys,subprocess; sys.path.insert(0,sys.argv[1]); "
            "from tools.natural_agreement_cpu import _install_owner_audit_hook; _install_owner_audit_hook(); "
            "\ntry: sys.audit('_winapi.CreateProcess', 'fabricated')"
            "\nexcept RuntimeError as error: print('winapi '+str(error))"
            "\ntry: subprocess.run([sys.executable,'-c',\"open(\"+repr(sys.argv[2])+\",'w').write('x')\"])"
            "\nexcept RuntimeError as error: print(str(error))")
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', code, str(root), str(marker)],
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert b'subprocess creation is prohibited' in result.stdout
    assert b'winapi CPU owner subprocess creation is prohibited' in result.stdout
    assert not marker.exists()
    for event in ('_winapi.CreateProcess', 'os.system', 'os.spawn', 'os.posix_spawn', 'os.exec', 'os.fork',
                  'os.forkpty', 'os.startfile', 'os.startfile/2'):
        with pytest.raises(RuntimeError, match='subprocess creation is prohibited'):
            launch._owner_audit_hook(event, ())


def test_windows_platform_probe_and_torch_import_run_without_subprocess(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    site = Path(sys.prefix) / 'Lib/site-packages'
    env = os.environ.copy()
    env.update({'MLG_NATURAL_SITE': str(site.resolve()), 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1'})
    code = ("import sys,os; sys.path.append(os.environ['MLG_NATURAL_SITE']); sys.path.insert(0,sys.argv[1]); "
            "from tools.natural_agreement_cpu import _install_owner_audit_hook,_configure_runtime; "
            "_install_owner_audit_hook(); _configure_runtime(); import platform; "
            "print(platform.platform()); print(platform.machine())")
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', code, str(root)],
                            env=env, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert b'Windows-' in result.stdout


def test_owner_detects_suppressed_audit_hook_installation():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; sys.path.insert(0,sys.argv[1]); "
            "from tools.natural_agreement_cpu import _install_owner_audit_hook\n"
            "def suppress(event,args):\n"
            " if event == 'sys.addaudithook': raise RuntimeError('suppress installation')\n"
            "sys.addaudithook(suppress)\n"
            "try: _install_owner_audit_hook()\n"
            "except RuntimeError as error: print(str(error))")
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', code, str(root)],
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert b'audit hook could not be verified' in result.stdout


def test_toy_runner_contract_is_accepted_by_runtime_controller(tmp_path):
    import torch
    from looking_glass.agreement_patch_runner import NaturalAgreementRunner
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('agreement_toy_fixture', root / 'tests/test_agreement_patch_runner.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_flush_denormal(False)
        model = fixture.ToyModel().eval()
        journal = AgreementJournal(tmp_path / 'toy-attempt', {'protocol': 'fabricated-only'})
        result = NaturalAgreementRunner(model, journal, device='cpu', expected_quartets=1).run(fixture.cases())
        assert result['forwards'] == 152
        inventory = control.audit_storage(tmp_path / 'toy-attempt')
        contract = inventory['_records'][0]['event']
        assert inventory['events'] == 170 and inventory['retained_tensor_bytes'] > 0
        runtime = {'cpu_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                   'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                   'float32_matmul_precision': torch.get_float32_matmul_precision(), 'denormal_probe': [1, 4194304]}
        control.check_execution_configuration(contract, runtime)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)


def test_isolated_launcher_does_not_import_sourceless_checkout_root_module(tmp_path):
    original = Path(__file__).resolve().parents[1]
    for relative in ('tools/natural_agreement_cpu.py', 'looking_glass/__init__.py',
                     'looking_glass/natural_agreement_controller.py'):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, target)
    marker = tmp_path / 'shadow-executed'
    bad = tmp_path / 'fractions.py'
    bad.write_text(f"open({str(marker)!r}, 'w').write('executed')\n", encoding='utf-8')
    py_compile.compile(str(bad), cfile=str(tmp_path / 'fractions.pyc'), doraise=True)
    bad.unlink()
    script = ("import runpy,sys; target=sys.argv[1]; sys.argv=[target, '--help']; "
              "\ntry: runpy.run_path(target, run_name='__main__')"
              "\nexcept SystemExit: pass"
              "\nsys.modules.pop('fractions',None); import fractions")
    command = [sys.executable, '-I', '-B', '-X', f'pycache_prefix={tmp_path / "unused-cache"}',
               '-c', script, str(tmp_path / 'tools/natural_agreement_cpu.py')]
    result = subprocess.run(command, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert not marker.exists()


def test_documented_direct_base_bootstrap_checks_runtime_without_model(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    site = Path(sys.prefix) / 'Lib/site-packages'
    env = os.environ.copy()
    env.update({'MLG_NATURAL_SITE': str(site.resolve()), 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1'})
    command = [sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={tmp_path / "unused-cache"}',
               '-c', launch.BOOTSTRAP, str(root / 'tools/natural_agreement_cpu.py'), 'check-runtime',
               '--cache', str(tmp_path), '--binding', str(tmp_path / 'unused-binding')]
    result = subprocess.run(command, env=env, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    info = json.loads(result.stdout)
    assert info['status'] == 'ok'
    assert Path(info['interpreter']).resolve() == Path(sys._base_executable).resolve()
    assert info['dependencies']['site_packages_sha256'] == hashlib.sha256(str(site.resolve()).encode('utf-8')).hexdigest()


def test_supervised_direct_base_bootstrap_reaps_without_model(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    site = Path(sys.prefix) / 'Lib/site-packages'
    env = os.environ.copy()
    env.update({'MLG_NATURAL_SITE': str(site.resolve()), 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1'})
    command = [sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={tmp_path / "unused-cache"}',
               '-c', 'import sys; sys.stdin.buffer.readline(); ' + launch.BOOTSTRAP,
               str(root / 'tools/natural_agreement_cpu.py'), 'check-runtime',
               '--cache', str(tmp_path), '--binding', str(tmp_path / 'unused-binding')]
    out = tmp_path / 'supervised'
    result = control.supervise(command, out, [], env)
    assert result['child_reaped'] is True and result['cleanup_verified'] is True
    assert result['exit_code'] == 0 and result['status'] == 'incomplete'
    assert json.loads((out / 'stdout.txt').read_bytes())['status'] == 'ok'


def test_bootstrap_rejects_preloaded_source_guard(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = ("import runpy,sys,types; sys.modules['scripts.v2_source_guards']=types.ModuleType('fake'); "
              "sys.argv=[sys.argv[1], '--help']; runpy.run_path(sys.argv[0],run_name='__main__')")
    command = [sys._base_executable, '-I', '-S', '-B', '-c', script, str(root / 'tools/natural_agreement_cpu.py')]
    result = subprocess.run(command, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert b'project modules were imported before the source bootstrap' in result.stderr
