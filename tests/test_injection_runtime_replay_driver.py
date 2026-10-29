"""Fabricated source-bound checks for the public replay driver."""
import hashlib
import io
import os
from pathlib import Path
import py_compile
import sqlite3
import subprocess
import sys

import pytest

from tools import injection_runtime_replay_driver as driver


def _inputs(tmp_path):
    locations, identities, remote = {}, {}, {}
    for index, (name, suffix) in enumerate(driver.SOURCE_SUFFIXES.items()):
        path = tmp_path.joinpath(*suffix.split('/'))
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ('# fabricated source %s\n' % name).encode('ascii')
        path.write_bytes(raw)
        url = driver.RAW_PREFIX + ('%040x' % (index + 1)) + '/' + suffix
        locations[name] = {'path': str(path), 'url': url}
        identities[name] = hashlib.sha256(raw).hexdigest()
        remote[url] = raw
    return locations, identities, remote


def test_every_exact_source_is_read_back_before_capture(tmp_path):
    locations, identities, remote = _inputs(tmp_path)
    observed = []
    def fetch(url):
        observed.append(url)
        return remote[url]
    snapshot = driver.capture_sources(locations, identities, fetch=fetch)
    assert set(snapshot) == set(driver.SOURCE_SUFFIXES)
    assert set(observed) == set(remote)
    for name, source in snapshot.items():
        assert source == Path(locations[name]['path']).read_bytes()
    driver.recheck_sources(locations, snapshot)
    Path(locations['supervisor']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        driver.recheck_sources(locations, snapshot)


def test_mismatched_source_url_or_local_bytes_fail_before_execution(tmp_path):
    locations, identities, remote = _inputs(tmp_path)
    marker = tmp_path / 'executed'
    name = 'controller'
    path = Path(locations[name]['path'])
    path.write_text('open(%r, "w").write("bad")' % str(marker), encoding='utf-8')
    with pytest.raises(ValueError):
        driver.capture_sources(locations, identities, fetch=remote.__getitem__)
    assert not marker.exists()
    locations, identities, remote = _inputs(tmp_path)
    locations[name]['url'] = locations[name]['url'].replace('/looking_glass/', '/other/')
    with pytest.raises(ValueError):
        driver.capture_sources(locations, identities, fetch=remote.__getitem__)
    assert not marker.exists()


def test_captured_bytes_are_executed_only_after_digest_verification(tmp_path):
    marker = tmp_path / 'marker'
    source = ('open(%r, "w").write("ok")\n' % str(marker)).encode('utf-8')
    module_name = '_mlg_fabricated_driver_test'
    with pytest.raises(ValueError):
        driver.load_captured(module_name, tmp_path / 'source.py', source, '0' * 64)
    assert not marker.exists()
    module = driver.load_captured(module_name, tmp_path / 'source.py', source,
                                  hashlib.sha256(source).hexdigest())
    assert marker.read_text() == 'ok'
    assert sys.modules[module_name] is module
    sys.modules.pop(module_name, None)


def test_public_files_require_exact_immutable_bytes_and_unique_paths(tmp_path):
    artifacts, specs, remote = {}, {}, {}
    for index, name in enumerate(('freeze', 'prediction')):
        raw = ('public-' + name).encode('ascii')
        path = tmp_path / (name + '.json')
        path.write_bytes(raw)
        url = driver.RAW_PREFIX + ('%040x' % (index + 1)) + '/results/' + name + '.json'
        artifacts[name] = str(path)
        specs[name] = (url, hashlib.sha256(raw).hexdigest(), 1024)
        remote[url] = raw
    assert driver.capture_public_files(artifacts, specs, fetch=remote.__getitem__) == {
        name: Path(path).read_bytes() for name, path in artifacts.items()}
    changed = dict(remote)
    changed[specs['freeze'][0]] = b'changed'
    with pytest.raises(ValueError):
        driver.capture_public_files(artifacts, specs, fetch=changed.__getitem__)
    aliased = dict(artifacts, prediction=artifacts['freeze'])
    with pytest.raises(ValueError):
        driver.capture_public_files(aliased, specs, fetch=remote.__getitem__)


def test_source_only_importer_ignores_poisoned_pyc_and_rejects_unknown_module(tmp_path):
    package = tmp_path / 'mlg_fixture_package'
    package.mkdir()
    (package / '__init__.py').write_bytes(b'# package\n')
    (package / 'alpha.py').write_bytes(b'from .beta import value\n')
    beta = package / 'beta.py'
    beta.write_bytes(b"value = 'EVIL'\n")
    stamp = beta.stat().st_mtime_ns
    py_compile.compile(str(beta), doraise=True)
    beta.write_bytes(b"value = 'GOOD'\n")
    os.utime(beta, ns=(stamp, stamp))
    (package / 'unknown.py').write_bytes(b'value = 42\n')
    script = """
import hashlib, importlib, pathlib, sys, types
driver_path, root = map(pathlib.Path, sys.argv[1:])
driver = types.ModuleType('_isolated_test_driver')
driver.__file__ = str(driver_path)
exec(compile(driver_path.read_bytes(), str(driver_path), 'exec'), driver.__dict__)
package = root / 'mlg_fixture_package'
entries = {}
for name, filename, package_flag in (('mlg_fixture_package', '__init__.py', True),
                                     ('mlg_fixture_package.alpha', 'alpha.py', False),
                                     ('mlg_fixture_package.beta', 'beta.py', False)):
    path = package / filename
    raw = path.read_bytes()
    entries[name] = (str(path), raw, hashlib.sha256(raw).hexdigest(), package_flag)
def exercise(finder):
    try: finder.entries['mlg_fixture_package.beta'] = ('bad', b'bad', '0' * 64, False)
    except TypeError: pass
    else: raise AssertionError('bound source map was mutable')
    assert importlib.import_module('mlg_fixture_package.alpha').value == 'GOOD'
    assert finder.assert_intact() is True
    sys.path.append(str(root))
    try: finder.assert_intact()
    except ValueError: pass
    else: raise AssertionError('changed import path was accepted')
    try: importlib.import_module('mlg_fixture_package.unknown')
    except ModuleNotFoundError: pass
    else: raise AssertionError('unknown project module imported')
    sys.path.pop()
    assert finder.assert_intact() is True
driver.run_bound_imports(entries, exercise, roots=('mlg_fixture_package',))
assert not any(name == 'mlg_fixture_package' or name.startswith('mlg_fixture_package.') for name in sys.modules)
def interrupt_entry(frame, event, arg):
    if frame.f_code is driver._BoundImports.__enter__.__code__ and event == 'return':
        sys.settrace(None)
        raise KeyboardInterrupt()
    return interrupt_entry
sys.settrace(interrupt_entry)
try: driver.run_bound_imports(entries, lambda finder: None, roots=('mlg_fixture_package',))
except KeyboardInterrupt: pass
else: raise AssertionError('entry interruption was not propagated')
finally: sys.settrace(None)
assert sys.meta_path == [importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                         importlib.machinery.PathFinder]
assert not any(name == 'mlg_fixture_package' or name.startswith('mlg_fixture_package.') for name in sys.modules)
"""
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', script,
                             str(Path(driver.__file__)), str(tmp_path)], capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')


def test_source_only_importer_rejects_preloaded_project_modules():
    with pytest.raises(ValueError):
        driver.run_bound_imports({}, lambda finder: None, roots=('tools',))


def test_frozen_and_operational_graph_is_bound_to_recorded_bytes(tmp_path):
    frozen = tmp_path / 'frozen'
    package = frozen / 'looking_glass'
    package.mkdir(parents=True)
    (package / '__init__.py').write_bytes(b'# package\n')
    (package / 'example.py').write_bytes(b'value = 1\n')
    record = {'git_commit': driver.DISCOVERY_SOURCE, 'git_tree': 'a' * 40,
              'files': {path.relative_to(frozen).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (package / '__init__.py', package / 'example.py')}}
    raw = driver.canonical(record)
    locations, identities, remote = _inputs(tmp_path / 'operational')
    captured = driver.capture_sources(locations, identities, fetch=remote.__getitem__)
    graph = driver.module_graph(str(frozen), raw, hashlib.sha256(raw).hexdigest(), locations, captured, identities)
    assert graph['looking_glass.example'][1] == b'value = 1\n'
    assert graph['looking_glass.natural_agreement_controller'][1] == captured['controller']
    assert graph['scripts.v2_source_guards'][1] == captured['source_guard']
    (package / 'example.py').write_bytes(b'value = 2\n')
    with pytest.raises(ValueError):
        driver.module_graph(str(frozen), raw, hashlib.sha256(raw).hexdigest(), locations, captured, identities)


def test_public_specs_pin_original_freeze_prediction_and_all_reference_roles():
    from tools import injection_runtime_replay_child as child
    plan = {'reference_sha256': {role: '%064x' % (index + 1) for index, role in enumerate(child._ROLES)}}
    specifications = driver.fixed_public_specs(plan, child)
    assert specifications['freeze'][1] == child._FREEZE
    assert specifications['prediction'][1] == child._PREDICTION
    for role in child._ROLES:
        url, digest, _ = specifications['reference_' + role]
        assert ('/' + child._REFERENCE_COMMIT + '/') in url
        assert url.endswith('/observation-00-' + role + '.json')
        assert digest == plan['reference_sha256'][role]
    freeze = {'schema': 'mlg-number-prediction-freezes-v1', 'protocol_sha256': child._PROTOCOL,
              'checkpoint_revision': child._CHECKPOINT,
              'tracks': {'injection': {'prediction_sha256': child._PREDICTION,
                                       'runtime_sha256': child._DISCOVERY_RUNTIME,
                                       'source_commit': child._SOURCE}}}
    driver.verify_fixed_freeze(driver.canonical(freeze), child)
    freeze['tracks']['injection']['source_commit'] = '0' * 40
    with pytest.raises(ValueError):
        driver.verify_fixed_freeze(driver.canonical(freeze), child)


def test_driver_gate_and_plan_are_checked_before_project_import(tmp_path):
    driver.await_start(io.BytesIO(driver.START_TOKEN))
    for bad in (b'', b'wrong\n', driver.START_TOKEN[:-1]):
        with pytest.raises(ValueError):
            driver.await_start(io.BytesIO(bad))
    plan = {'sources': {name: '1' * 64 for name in driver.SOURCE_SUFFIXES}, 'fabricated': True}
    raw = driver.canonical(plan)
    path = tmp_path / 'plan.json'
    path.write_bytes(raw)
    url = driver.RAW_PREFIX + '1' * 40 + driver.PLAN_PATH
    value, captured = driver.capture_plan(str(path), url, fetch=lambda observed: raw)
    assert value == plan and captured == raw
    with pytest.raises(ValueError):
        driver.capture_plan(str(path), url, fetch=lambda observed: b'changed')
    with pytest.raises(ValueError):
        driver.capture_plan(str(path), url.replace(driver.PLAN_PATH, '/wrong.json'), fetch=lambda observed: raw)
    plan['sources'].pop('controller')
    path.write_bytes(driver.canonical(plan))
    with pytest.raises(ValueError):
        driver.capture_plan(str(path), url, fetch=lambda observed: path.read_bytes())


def test_claim_row_is_bound_to_guarded_database_and_rechecked(tmp_path):
    from types import SimpleNamespace
    ledger = tmp_path / 'ledger.sqlite3'
    phase = tmp_path / 'phase'
    row = ('injection_heldout', str(phase), 72 * 1024**2, 192 * 1024**2, 12 * 1024**3)
    with sqlite3.connect(ledger) as database:
        database.execute('CREATE TABLE claims(phase TEXT, output TEXT, logical INT, disk INT, free INT)')
        database.execute('INSERT INTO claims VALUES(?, ?, ?, ?, ?)', row)
    identity = (ledger.stat().st_dev, ledger.stat().st_ino)
    seen = []
    guard = SimpleNamespace(ClaimAnchor=lambda **kwargs: SimpleNamespace(**kwargs),
                            verify_existing_claim=lambda *args, **kwargs: seen.append(kwargs['anchor']) or True)
    kwargs = {'db_identity': identity, 'state_sha256': '1' * 64, 'history_sha256': '2' * 64,
              'scope_sha256': '3' * 64, 'original_claims_sha256': '4' * 64}
    assert driver.guarded_claim_row(str(ledger), str(phase), guard, **kwargs) == list(row)
    assert seen[0].heldout_claim_sha256 == driver.claim_row_sha256(list(row))
    guard.verify_existing_claim = lambda *args, **kwargs: False
    with pytest.raises(ValueError):
        driver.guarded_claim_row(str(ledger), str(phase), guard, **kwargs)
    def mutate(*args, **kwargs):
        with sqlite3.connect(ledger) as database:
            database.execute('UPDATE claims SET free = free + 1')
        return True
    guard.verify_existing_claim = mutate
    with pytest.raises(ValueError):
        driver.guarded_claim_row(str(ledger), str(phase), guard, **kwargs)


def test_claim_read_preserves_interrupt_when_database_close_fails(monkeypatch, tmp_path):
    class FailingDatabase:
        def execute(self, *args):
            raise KeyboardInterrupt()
        def close(self):
            raise OSError('fabricated close failure')
    monkeypatch.setattr(driver.sqlite3, 'connect', lambda *args, **kwargs: FailingDatabase())
    with pytest.raises(KeyboardInterrupt) as failure:
        driver._read_claim_row(tmp_path / 'db', str(tmp_path / 'phase'))
    assert isinstance(failure.value.__cause__, OSError)


def test_observed_exit_chain_requires_exact_process_and_terminal_bindings():
    terminal = b'private terminal bytes'
    outer = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
             'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
             'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
             'parent_pid': 111, 'parent_birth_100ns': 222, 'config_sha256': '1' * 64,
             'operation_sha256': '2' * 64, 'command_sha256': '3' * 64,
             'supervisor_source_sha256': '4' * 64,
             'supervisor_terminal_sha256': hashlib.sha256(terminal).hexdigest()}
    outer_raw = driver.canonical(outer)
    observation = {'schema': 'mlg-injection-replay-observer-exit-v1', 'purpose': 'claim',
                   'status': 'completed', 'exit_code': 0, 'observer_reaped': True,
                   'observer_handle_closed': True, 'job_drained': True, 'job_closed': True,
                   'active_job_members': 0, 'observer_pid': 333, 'observer_birth_100ns': 444,
                   'plan_sha256': '2' * 64, 'observer_command_sha256': '5' * 64,
                   'observer_source_sha256': '6' * 64,
                   'outer_observation_sha256': hashlib.sha256(outer_raw).hexdigest()}
    kwargs = {'purpose': 'claim', 'observer_identity': (333, 444), 'parent_identity': (111, 222),
              'plan_sha256': '2' * 64, 'config_sha256': '1' * 64, 'parent_command_sha256': '3' * 64,
              'parent_source_sha256': '4' * 64, 'observer_command_sha256': '5' * 64,
              'observer_source_sha256': '6' * 64, 'terminal_check': lambda raw, pid, birth: raw == terminal}
    assert driver.verify_exit_chain(driver.canonical(observation), outer_raw, terminal, **kwargs) == {
        'status': 'saved_chain_consistent', 'purpose': 'claim'}
    for key, bad in (('exit_code', False), ('observer_reaped', 1), ('active_job_members', True),
                     ('observer_pid', 334), ('outer_observation_sha256', '0' * 64)):
        changed = dict(observation, **{key: bad})
        with pytest.raises(ValueError):
            driver.verify_exit_chain(driver.canonical(changed), outer_raw, terminal, **kwargs)
    changed_outer = dict(outer, parent_birth_100ns=223)
    changed_raw = driver.canonical(changed_outer)
    changed_observation = dict(observation, outer_observation_sha256=hashlib.sha256(changed_raw).hexdigest())
    with pytest.raises(ValueError):
        driver.verify_exit_chain(driver.canonical(changed_observation), changed_raw, terminal, **kwargs)
    with pytest.raises(ValueError):
        driver.verify_exit_chain(driver.canonical(observation), outer_raw, terminal,
                                 **dict(kwargs, terminal_check=lambda *args: False))


def test_asset_inventory_checks_exact_cached_bytes_without_loading_a_model(tmp_path):
    revision = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
    snapshot = tmp_path / 'models--distilbert--distilgpt2' / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    assets = []
    for name in sorted(driver.ASSET_NAMES):
        raw = ('fake-' + name).encode('ascii')
        (snapshot / name).write_bytes(raw)
        assets.append({'name': name, 'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    manifest = {'schema': 'resource-bootstrap-assets-v1', 'model_id': 'distilbert/distilgpt2',
                'revision': revision, 'assets': assets}
    checked = driver.check_assets(str(tmp_path), manifest)
    assert {path.name for path in checked} == driver.ASSET_NAMES
    (snapshot / 'config.json').write_bytes(b'changed')
    with pytest.raises(ValueError):
        driver.check_assets(str(tmp_path), manifest)


def test_exit_receipt_is_built_only_after_clean_outer_process_observation():
    terminal = b'parent terminal'
    inner = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
             'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
             'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
             'parent_pid': 11, 'parent_birth_100ns': 22, 'config_sha256': '1' * 64,
             'operation_sha256': '2' * 64, 'command_sha256': '3' * 64,
             'supervisor_source_sha256': '4' * 64,
             'supervisor_terminal_sha256': hashlib.sha256(terminal).hexdigest()}
    inner_raw = driver.canonical(inner)
    outer = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
             'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
             'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
             'parent_pid': 33, 'parent_birth_100ns': 44, 'config_sha256': '5' * 64,
             'operation_sha256': '2' * 64, 'command_sha256': '6' * 64,
             'supervisor_source_sha256': '7' * 64,
             'supervisor_terminal_sha256': hashlib.sha256(inner_raw).hexdigest()}
    kwargs = {'purpose': 'replay', 'plan_sha256': '2' * 64, 'parent_config_sha256': '1' * 64,
              'parent_command_sha256': '3' * 64, 'parent_source_sha256': '4' * 64,
              'observer_config_sha256': '5' * 64, 'observer_command_sha256': '6' * 64,
              'observer_source_sha256': '7' * 64, 'terminal_check': lambda raw, pid, birth: raw == terminal}
    receipt = driver.make_observed_exit(outer, driver.canonical(outer), inner_raw, terminal, **kwargs)
    assert receipt['observer_pid'] == 33 and receipt['observer_birth_100ns'] == 44
    assert receipt['outer_observation_sha256'] == hashlib.sha256(inner_raw).hexdigest()
    changed = dict(outer, parent_reaped=False)
    with pytest.raises(ValueError):
        driver.make_observed_exit(changed, driver.canonical(changed), inner_raw, terminal, **kwargs)
    with pytest.raises(ValueError):
        driver.make_observed_exit(outer, driver.canonical(outer), inner_raw, terminal,
                                  **dict(kwargs, terminal_check=lambda *args: False))


def test_claimant_emits_terminal_only_after_claim_and_final_authority(tmp_path):
    from types import SimpleNamespace
    from tools import injection_replay_claim as claim
    from tools import injection_runtime_replay_child as child
    workspace = tmp_path / 'workspace'
    parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
    diagnostic = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    ledger = workspace / 'work' / 'agreement-study-ledger-v5' / 'ledger.sqlite3'
    parent.mkdir(parents=True)
    diagnostic.mkdir()
    (diagnostic / 'no-model.log').write_bytes(b'no model')
    ledger.parent.mkdir()
    with sqlite3.connect(ledger) as database:
        database.execute('CREATE TABLE claims(phase TEXT, output TEXT, logical INT, disk INT, free INT)')
    parent_id = (parent.stat().st_dev, parent.stat().st_ino)
    ledger_id = (ledger.stat().st_dev, ledger.stat().st_ino)
    calls = []
    def make_claim(path, phase, *, output_path):
        calls.append(phase)
        with sqlite3.connect(path) as database:
            database.execute('INSERT INTO claims VALUES(?, ?, ?, ?, ?)',
                (phase, str(output_path), 72 * 1024**2, 192 * 1024**2, 12 * 1024**3))
        return SimpleNamespace(phase=phase, output_path=str(output_path),
                               logical_limit_bytes=72 * 1024**2, disk_limit_bytes=192 * 1024**2)
    ledger_module = SimpleNamespace(claim_phase=make_claim)
    guard = SimpleNamespace(ClaimAnchor=lambda **kwargs: SimpleNamespace(**kwargs),
                            verify_existing_claim=lambda *args, **kwargs: True)
    owner = SimpleNamespace(_current_process_identity=lambda: (123, 456))
    plan = b'{"fabricated":true}'
    arguments = {'workspace': str(workspace), 'plan_bytes': plan,
                 'plan_sha256': hashlib.sha256(plan).hexdigest(), 'config_sha256': '1' * 64,
                 'driver_sha256': '2' * 64, 'parent_identity': parent_id, 'ledger_identity': ledger_id,
                 'state_sha256': '3' * 64, 'history_sha256': '4' * 64,
                 'scope_sha256': '5' * 64, 'original_claims_sha256': '6' * 64,
                 'claim': claim, 'child': child, 'guard': guard, 'ledger_module': ledger_module, 'owner': owner}
    with pytest.raises(ValueError):
        driver.claim_once(authority=lambda: False, **arguments)
    assert not calls and not (parent / 'replay-claimant-terminal.json').exists()
    terminal = driver.claim_once(authority=lambda: True, **arguments)
    assert calls == ['injection_heldout']
    assert terminal['parent_pid'] == 123 and terminal['parent_birth_100ns'] == 456
    assert (parent / 'replay-claimant-terminal.json').read_bytes() == driver.canonical(terminal)
    terminal_bindings = {'parent_identity': (123, 456), 'plan_sha256': terminal['plan_sha256'],
                         'config_sha256': '1' * 64, 'driver_sha256': '2' * 64,
                         'intent_sha256': terminal['intent_sha256'], 'receipt_sha256': terminal['receipt_sha256'],
                         'claim_row_sha256': terminal['claim_row_sha256']}
    assert driver.verify_claimant_terminal(driver.canonical(terminal), **terminal_bindings) is True
    with pytest.raises(ValueError):
        driver.verify_claimant_terminal(driver.canonical(terminal), **dict(terminal_bindings, parent_identity=(123, 457)))


def test_claimant_failure_after_terminal_cannot_be_treated_as_success(monkeypatch, tmp_path):
    from types import SimpleNamespace
    workspace = tmp_path / 'workspace'
    parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
    parent.mkdir(parents=True)
    phase = parent / ('injection-heldout-' + 'a' * 64)
    row = ['injection_heldout', str(phase), 72 * 1024**2, 192 * 1024**2, 12 * 1024**3]
    intent_raw = b'fabricated intent'
    receipt = {'intent_sha256': hashlib.sha256(intent_raw).hexdigest(),
               'injection_claim_row_sha256': driver.claim_row_sha256(row)}
    def claim_replay_once(**kwargs):
        (parent / 'replay-claim-intent.json').write_bytes(intent_raw)
        (parent / 'replay-claim-receipt.json').write_bytes(driver.canonical(receipt))
        return receipt
    monkeypatch.setattr(driver, 'guarded_claim_row', lambda *args, **kwargs: row)
    calls = []
    def authority():
        calls.append(True)
        return len(calls) < 3
    plan = b'{"fake":true}'
    with pytest.raises(ValueError):
        driver.claim_once(workspace=str(workspace), plan_bytes=plan, plan_sha256=hashlib.sha256(plan).hexdigest(),
            config_sha256='1' * 64, driver_sha256='2' * 64, parent_identity=(1, 2), ledger_identity=(3, 4),
            state_sha256='3' * 64, history_sha256='4' * 64, scope_sha256='5' * 64,
            original_claims_sha256='6' * 64, claim=SimpleNamespace(claim_replay_once=claim_replay_once),
            child=SimpleNamespace(_FREEZE='a' * 64, _check_claim_receipt=lambda *args: receipt),
            guard=object(), ledger_module=SimpleNamespace(claim_phase=lambda *args, **kwargs: None),
            owner=SimpleNamespace(_current_process_identity=lambda: (123, 456)), authority=authority)
    assert (parent / 'replay-claimant-terminal.json').exists()
    assert len(calls) == 3


def test_unverified_frozen_launcher_never_executes(tmp_path):
    root = tmp_path / 'frozen'
    launcher = root / 'scripts' / 'pretrained_number_cpu_launcher.py'
    launcher.parent.mkdir(parents=True)
    marker = tmp_path / 'unexpected'
    launcher.write_text('open(%r, "w").write("bad")' % str(marker), encoding='utf-8')
    runtime = {'source_commit': driver.DISCOVERY_SOURCE, 'repository_source_sha256': '1' * 64,
               'source_tree': '2' * 40}
    with pytest.raises(ValueError):
        driver.capture_frozen_repository(str(root), runtime, fetch=lambda url: launcher.read_bytes())
    assert not marker.exists()


def test_preflight_rejects_a_different_executing_driver_before_import(monkeypatch, tmp_path):
    plan = {'sources': {name: '1' * 64 for name in driver.SOURCE_SUFFIXES}}
    locations = {name: {'path': str(tmp_path / suffix), 'url': 'fake'}
                 for name, suffix in driver.SOURCE_SUFFIXES.items()}
    monkeypatch.setattr(driver, 'capture_plan', lambda *args, **kwargs: (plan, b'plan'))
    monkeypatch.setattr(driver, 'capture_sources', lambda *args, **kwargs: {
        name: b'fake' for name in driver.SOURCE_SUFFIXES})
    monkeypatch.setattr(driver, 'load_captured', lambda *args, **kwargs: pytest.fail('project source executed'))
    with pytest.raises(ValueError):
        driver.capture_preflight(plan_path=str(tmp_path / 'plan'), plan_url='fake',
            source_locations=locations, public_paths={}, frozen_root=str(tmp_path / 'frozen'),
            cache_dir=str(tmp_path / 'cache'), fetch=lambda url: b'fake')


def test_preflight_recheck_rejects_changed_paths_and_public_bytes(monkeypatch, tmp_path):
    from types import SimpleNamespace
    locations, identities, remote = _inputs(tmp_path / 'sources')
    sources = driver.capture_sources(locations, identities, fetch=remote.__getitem__)
    plan_path = tmp_path / 'plan.json'
    plan = {'sources': identities}
    plan_raw = driver.canonical(plan)
    plan_path.write_bytes(plan_raw)
    public_path = tmp_path / 'public.json'
    public_path.write_bytes(b'public')
    record = {'git_commit': driver.DISCOVERY_SOURCE, 'git_tree': 'a' * 40, 'files': {}}
    context = {'plan_path': str(plan_path), 'plan_url': 'url', 'plan_url_captured': 'url',
               'plan': plan, 'plan_bytes': plan_raw, 'child': None, 'source_locations': locations,
               'source_locations_bytes': driver.canonical(locations), 'sources': sources,
               'public_paths': {'test': str(public_path)}, 'public_paths_bytes': driver.canonical({'test': str(public_path)}),
               'public': {'test': b'public'}, 'frozen_root': str(tmp_path),
               'frozen_record': driver.canonical(record), 'launcher': SimpleNamespace(
                   _verify_unloaded_source=lambda *args: record), 'cache_dir': str(tmp_path),
               'asset_manifest': {}, 'asset_paths': (tmp_path / 'asset',)}
    monkeypatch.setattr(driver, 'check_assets', lambda *args: context['asset_paths'])
    monkeypatch.setattr(driver, 'fixed_public_specs', lambda *args: {
        'test': ('url', hashlib.sha256(b'public').hexdigest(), 100)})
    finder = SimpleNamespace(assert_intact=lambda: True)
    assert driver.recheck_preflight(context, finder) is True
    public_path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        driver.recheck_preflight(context, finder)
    public_path.write_bytes(b'public')
    public_path.write_bytes(b'changed')
    context['public']['test'] = b'changed'
    with pytest.raises(ValueError):
        driver.recheck_preflight(context, finder)
    context['public']['test'] = b'public'
    public_path.write_bytes(b'public')
    locations['driver']['url'] = 'changed'
    with pytest.raises(ValueError):
        driver.recheck_preflight(context, finder)


def test_lease_inventory_covers_frozen_public_operational_and_assets(tmp_path):
    workspace = tmp_path / 'workspace'
    frozen = workspace / 'frozen'
    frozen.mkdir(parents=True)
    (frozen / 'module.py').write_bytes(b'module')
    locations, identities, remote = _inputs(workspace / 'operational')
    public = workspace / 'public.json'
    plan = workspace / 'plan.json'
    config = workspace / 'config.json'
    asset = workspace / 'asset.bin'
    for path in (public, plan, config, asset):
        path.write_bytes(b'input')
    context = {'frozen_root': str(frozen), 'frozen_record': driver.canonical(
        {'git_commit': driver.DISCOVERY_SOURCE, 'git_tree': 'a' * 40,
         'files': {'module.py': hashlib.sha256(b'module').hexdigest()}}),
        'source_locations': locations, 'public_paths': {'public': str(public)},
        'plan_path': str(plan), 'asset_paths': (asset,)}
    frozen_files, operational_files, asset_files = driver.lease_inputs(context, str(workspace),
                                                                         extra_files=(str(config),))
    assert frozen_files == (frozen / 'module.py',)
    assert {public, plan, config} <= set(operational_files)
    assert set(asset_files) == {asset}
    outside = tmp_path / 'outside'
    outside.write_bytes(b'outside')
    with pytest.raises(ValueError):
        driver.lease_inputs(context, str(workspace), extra_files=(str(outside),))


def test_runtime_environment_is_fixed_to_the_published_system_fingerprint(tmp_path):
    workspace = tmp_path / 'workspace'
    home = workspace / 'work' / 'hf-cache'
    cache = home / 'hub'
    cache.mkdir(parents=True)
    system = {'PATH': 'example', 'TEMP': str(workspace)}
    fixed = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
             'OPENBLAS_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE', 'HF_HUB_OFFLINE': '1',
             'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
    environment = {**fixed, **system, 'MLG_STUDY_WORKSPACE': str(workspace), 'HF_HOME': str(home),
                   'HF_HUB_CACHE': str(cache), 'TRANSFORMERS_CACHE': str(cache)}
    manifest = {'environment': fixed,
                'system_environment_sha256': hashlib.sha256(driver.canonical(system)).hexdigest()}
    assert driver.check_runtime_environment(environment, str(workspace), str(cache), manifest) is True
    for altered in (dict(environment, PATH='different'), dict(environment, UNREVIEWED='1'),
                    dict(environment, OMP_NUM_THREADS='2'), dict(environment, HF_HOME=str(workspace))):
        with pytest.raises(ValueError):
            driver.check_runtime_environment(altered, str(workspace), str(cache), manifest)


def test_replay_configuration_is_derived_from_the_captured_public_inputs(tmp_path):
    from tools import injection_runtime_replay_child as child
    locations, _, _ = _inputs(tmp_path / 'sources')
    public = {name: str(tmp_path / (name + '.json')) for name in
              ('discovery_runtime', 'heldout_runtime', 'discovery_plan', 'runtime_deviation')}
    context = {'frozen_root': str(tmp_path / 'frozen'), 'cache_dir': str(tmp_path / 'cache'),
               'plan_path': str(tmp_path / 'plan.json'), 'plan_url': child.TEST_PLAN_URL,
               'source_locations': locations, 'public_paths': public, 'reference_dir': str(tmp_path / 'refs')}
    values = {'schema': child.CONFIG_SCHEMA, 'plan_url': child.TEST_PLAN_URL,
              **{key: str(tmp_path / key) for key in child.CONFIG_PATHS if key != 'pycache_prefix'}}
    values.update({'frozen_root': context['frozen_root'], 'cache_dir': context['cache_dir'],
                   'replay_plan_file': context['plan_path'], 'reference_dir': context['reference_dir'],
                   'producer_source_file': locations['producer']['path'],
                   **{name + '_file': path for name, path in public.items()}})
    prefix = str(tmp_path / 'protected-prefix')
    result = driver.bind_replay_config(context, child, values, prefix)
    assert result['pycache_prefix'] == prefix
    assert child.parse_config(driver.canonical(result)) == result
    changed = dict(values, producer_source_file=str(tmp_path / 'other.py'))
    with pytest.raises(ValueError):
        driver.bind_replay_config(context, child, changed, prefix)
    changed = dict(values, plan_url=child.TEST_PLAN_URL.replace('/' + '0' * 40 + '/', '/' + '1' * 40 + '/'))
    with pytest.raises(ValueError):
        driver.bind_replay_config(context, child, changed, prefix)


def test_private_replay_inputs_are_leased_but_validation_reads_no_private_files(tmp_path):
    config = {key: str(tmp_path / key) for key in
              ('site_nonce_file', 'site_record_file', 'claim_receipt_file')}
    assert driver.private_replay_inputs(config, True, ()) == ()
    assert set(driver.private_replay_inputs(config, False, ())) == set(config.values())
    with pytest.raises(ValueError):
        driver.private_replay_inputs(dict(config, site_nonce_file='relative'), False, ())


def test_replay_outputs_must_be_fresh_disjoint_workspace_paths(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    paths = [str(workspace / name) for name in ('config', 'supervisor', 'terminal', 'postcheck', 'attempt')]
    assert driver.check_replay_output_paths(str(workspace), *paths) is True
    with pytest.raises(ValueError):
        driver.check_replay_output_paths(str(workspace), *paths[:3], paths[0], paths[4])
    with pytest.raises(ValueError):
        driver.check_replay_output_paths(str(workspace), *paths[:4], str(tmp_path / 'outside'))
    Path(paths[2]).write_bytes(b'existing')
    with pytest.raises(ValueError):
        driver.check_replay_output_paths(str(workspace), *paths)


@pytest.mark.skipif(os.name != 'nt', reason='requires the reviewed Windows Job and file leases')
def test_protected_replay_runs_a_gated_no_model_child_with_generated_config(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from looking_glass import natural_heldout_supervisor as supervisor
    from scripts import number_cpu_outer, v2_source_guards as guards
    from tools import injection_heldout_owner as adapter, injection_runtime_replay_child as child
    from tools import injection_runtime_replay_owner as owner
    workspace = tmp_path / 'workspace'
    frozen = workspace / 'frozen'
    frozen.mkdir(parents=True)
    frozen_file = frozen / 'module.py'
    frozen_file.write_bytes(b'module')
    locations, identities, _ = _inputs(workspace / 'sources')
    source = Path(locations['child']['path'])
    source.write_text(
        "import hashlib,json,pathlib,sys\n"
        "assert sys.stdin.buffer.readline() == b'MLG-natural-discovery-start-v1\\n'\n"
        "c=json.loads(pathlib.Path(sys.argv[-1]).read_bytes())\n"
        "p=hashlib.sha256(pathlib.Path(c['replay_plan_file']).read_bytes()).hexdigest()\n"
        "s=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()\n"
        "v={'schema':'mlg-injection-runtime-replay-child-postcheck-v1','status':'validated_no_model_forward',"
        "'plan_sha256':p,'heldout_runtime_sha256':%r,'child_source_sha256':s,'forward_count':0}\n"
        "pathlib.Path(c['postcheck_file']).write_bytes(json.dumps(v,sort_keys=True,separators=(',',':')).encode())\n"
        % child._HELDOUT_RUNTIME, encoding='utf-8')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    public = {name: str(workspace / (name + '.json')) for name in
              ('discovery_runtime', 'heldout_runtime', 'discovery_plan', 'runtime_deviation')}
    for path in public.values():
        Path(path).write_bytes(b'public')
    cache = workspace / 'hf' / 'hub'
    cache.mkdir(parents=True)
    asset = cache / 'asset'
    asset.write_bytes(b'asset')
    plan = {'schema': 'mlg-injection-runtime-replay-plan-v1', 'discovery_source_commit': child._SOURCE,
            'protocol_sha256': child._PROTOCOL, 'checkpoint_revision': child._CHECKPOINT,
            'prediction_freeze_sha256': child._FREEZE, 'prediction_sha256': child._PREDICTION,
            'source_supplement_sha256': child._SOURCE_SUPPLEMENT, 'discovery_runtime_sha256': child._DISCOVERY_RUNTIME,
            'heldout_runtime_sha256': child._HELDOUT_RUNTIME, 'discovery_plan_sha256': child._DISCOVERY_PLAN,
            'runtime_deviation_sha256': child._DEVIATION, 'site_commitment': child._SITE_COMMITMENT,
            'model_state_sha256': child._MODEL_STATE, 'reference_commit': child._REFERENCE_COMMIT,
            'roles': list(child._ROLES), 'budgets': dict(child._BUDGETS),
            'reference_sha256': {role: '1' * 64 for role in child._ROLES},
            'sources': {name: child._PRODUCER if name == 'producer' else source_sha if name == 'child' else '2' * 64
                        for name in child._SOURCE_NAMES}}
    plan_path = workspace / 'plan.json'
    plan_raw = child._canonical(plan)
    plan_path.write_bytes(plan_raw)
    context = {'frozen_root': str(frozen), 'frozen_record': driver.canonical(
        {'git_commit': driver.DISCOVERY_SOURCE, 'git_tree': 'a' * 40,
         'files': {'module.py': hashlib.sha256(b'module').hexdigest()}}),
        'cache_dir': str(cache), 'asset_paths': (asset,), 'source_locations': locations,
        'public_paths': public, 'public': {'heldout_runtime': b'{}'}, 'plan_path': str(plan_path), 'plan_bytes': plan_raw,
        'plan_url': child.TEST_PLAN_URL, 'reference_dir': str(workspace / 'refs'), 'plan': plan}
    config = {'schema': child.CONFIG_SCHEMA, 'plan_url': child.TEST_PLAN_URL,
              **{key: str(workspace / key) for key in child.CONFIG_PATHS if key != 'pycache_prefix'}}
    config.update({'frozen_root': str(frozen), 'cache_dir': str(cache), 'replay_plan_file': str(plan_path),
                   'reference_dir': context['reference_dir'], 'producer_source_file': locations['producer']['path'],
                   'postcheck_file': str(workspace / 'postcheck.json'), 'attempt_dir': str(workspace / 'attempt'),
                   **{name + '_file': path for name, path in public.items()}})
    modules = {'owner': owner, 'guard': guards, 'child': child, 'adapter': adapter,
               'supervisor': supervisor, 'cpu_job': number_cpu_outer}
    monkeypatch.setattr(driver, '_execution_modules', lambda finder: modules)
    checks = []
    monkeypatch.setattr(driver, 'recheck_preflight', lambda *args: checks.append('source') or True)
    monkeypatch.setattr(driver, 'check_runtime_environment', lambda *args: True)
    result = driver.run_protected_replay(context, SimpleNamespace(), workspace=str(workspace),
        cache_parent=str(workspace), config_values=config, config_path=str(workspace / 'config.json'),
        output=str(workspace / 'supervisor'), terminal_path=str(workspace / 'parent-terminal.json'),
        environment=os.environ.copy(), authority=lambda stage, value: checks.append(stage) or True,
        validation_only=True, timeout=30)
    assert result['status'] == 'validated_no_model_forward'
    assert checks.count('source') >= 3 and checks.count('before_launch') == 1 and checks.count('after_child') >= 1
    assert Path(workspace / 'parent-terminal.json').is_file()
    assert not owner._QUARANTINED


@pytest.mark.skipif(os.name != 'nt', reason='requires the reviewed Windows Job and file leases')
def test_observer_retains_and_reaps_a_gated_fabricated_parent(monkeypatch, tmp_path):
    import types
    from types import SimpleNamespace
    from scripts import v2_source_guards as guards
    from tools import injection_runtime_replay_owner as owner
    source = subprocess.run(['git', 'show', '3f3ddcce46b5c64d4b08b8e6e177b643206eca07:'
                             'looking_glass/injection_outer_observer.py'], capture_output=True, check=True).stdout
    assert hashlib.sha256(source).hexdigest() == 'fa878a9f8dbd555e23aab8090005ecd632b3e1061288b87d7a6399e2932ee95e'
    audit = types.ModuleType('looking_glass.injection_preparation_audit')
    audit._read = lambda path, limit: Path(path).read_bytes() if Path(path).stat().st_size <= limit else b''
    custody = types.ModuleType('looking_glass.injection_split_custody')
    def write_new(path, raw):
        with open(path, 'xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    custody._write_new = write_new
    monkeypatch.setitem(sys.modules, audit.__name__, audit)
    monkeypatch.setitem(sys.modules, custody.__name__, custody)
    observer = types.ModuleType('looking_glass._fabricated_outer_observer')
    observer.__package__ = 'looking_glass'
    exec(compile(source, '<published-observer>', 'exec'), observer.__dict__)
    workspace = tmp_path / 'workspace'
    frozen = workspace / 'frozen'
    frozen.mkdir(parents=True)
    (frozen / 'module.py').write_bytes(b'module')
    locations, _, _ = _inputs(workspace / 'sources')
    parent_source = Path(locations['driver']['path'])
    parent_source.write_text(
        "import json,os,pathlib,sys\n"
        "assert sys.stdin.buffer.readline() == b'MLG-natural-discovery-start-v1\\n'\n"
        "c=json.loads(pathlib.Path(sys.argv[-1]).read_bytes())\n"
        "pathlib.Path(c['terminal']).write_text(json.dumps({'pid':os.getpid()},sort_keys=True,separators=(',',':')))\n",
        encoding='utf-8')
    parent_sha = hashlib.sha256(parent_source.read_bytes()).hexdigest()
    public = workspace / 'public.json'
    public.write_bytes(b'public')
    asset = workspace / 'asset'
    asset.write_bytes(b'asset')
    plan = workspace / 'plan.json'
    plan.write_bytes(b'plan')
    terminal = workspace / 'parent-terminal.json'
    parent_config = workspace / 'parent-config.json'
    parent_config.write_bytes(driver.canonical({'terminal': str(terminal)}))
    context = {'frozen_root': str(frozen), 'frozen_record': driver.canonical(
        {'git_commit': driver.DISCOVERY_SOURCE, 'git_tree': 'a' * 40,
         'files': {'module.py': hashlib.sha256(b'module').hexdigest()}}),
        'source_locations': locations, 'public_paths': {'public': str(public)}, 'plan_path': str(plan),
        'plan_bytes': b'plan', 'plan': {'sources': {'driver': parent_sha}},
        'asset_paths': (asset,), 'cache_dir': str(workspace), 'public': {'heldout_runtime': b'{}'}}
    monkeypatch.setattr(driver, '_execution_modules', lambda finder: {'owner': owner, 'guard': guards,
                                                                       'outer_observer': observer})
    monkeypatch.setattr(driver, 'recheck_preflight', lambda *args: True)
    monkeypatch.setattr(driver, 'check_runtime_environment', lambda *args: True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    result = driver.observe_replay_parent(context, SimpleNamespace(), workspace=str(workspace),
        cache_parent=str(workspace), purpose='validation', config_path=str(parent_config),
        terminal_path=str(terminal), output=str(workspace / 'outer'),
        command_path=str(workspace / 'parent-command.json'), environment=environment,
        authority=lambda: True,
        terminal_check=lambda raw, pid, birth: driver._canonical_object(raw, 8192) == {'pid': pid} and birth > 0,
        timeout=30)
    assert result['status'] == 'completed' and result['parent_reaped'] is True
    assert result['parent_handle_closed'] is True and result['active_job_members'] == 0
    command = driver.verify_parent_command((workspace / 'parent-command.json').read_bytes(), context,
                                           owner, str(workspace), 'validation', str(parent_config))
    assert hashlib.sha256(driver.canonical(command)).hexdigest() == result['command_sha256']
    assert not observer._QUARANTINED and not owner._QUARANTINED
    parent_source.write_text("import sys\nsys.stdin.buffer.readline()\nraise SystemExit(1)\n", encoding='utf-8')
    context['plan']['sources']['driver'] = hashlib.sha256(parent_source.read_bytes()).hexdigest()
    failed_terminal = workspace / 'failed-terminal.json'
    failed_config = workspace / 'failed-config.json'
    failed_config.write_bytes(driver.canonical({'terminal': str(failed_terminal)}))
    # Avoid the production hold in this controlled, already-reaped fixture so
    # the test can inspect the retained handles and then explicitly close them.
    monkeypatch.setattr(owner, 'run_guarded', owner.under_protection)
    try:
        with pytest.raises(observer.ObserverError):
            driver.observe_replay_parent(context, SimpleNamespace(), workspace=str(workspace),
                cache_parent=str(workspace), purpose='validation', config_path=str(failed_config),
                terminal_path=str(failed_terminal), output=str(workspace / 'failed-outer'),
                command_path=str(workspace / 'failed-command.json'), environment=environment,
                authority=lambda: True, terminal_check=lambda raw, pid, birth: True, timeout=30)
        assert not failed_terminal.exists() and not observer._QUARANTINED and len(owner._QUARANTINED) == 1
        with pytest.raises(OSError):
            parent_source.write_bytes(b'changed while custody is retained')
    finally:
        while owner._QUARANTINED:
            record = owner._QUARANTINED.pop()
            for lease in reversed(record['leases']):
                lease.close()


def test_validation_parent_config_binds_the_existing_diagnostic_run(tmp_path):
    workspace = tmp_path / 'workspace'
    diagnostic = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    run = diagnostic / 'validation-01'
    run.mkdir(parents=True)
    config_path = run / 'parent-config.json'
    config = {'schema': 'mlg-injection-replay-validation-parent-config-v1', 'workspace': str(workspace),
              'plan_path': str(workspace / 'plan.json'), 'plan_url': driver.RAW_PREFIX + '1' * 40 + driver.PLAN_PATH,
              'source_locations': {}, 'public_paths': {}, 'frozen_root': str(workspace / 'frozen'),
              'cache_dir': str(workspace / 'cache'), 'run_dir': str(run),
              'parent_identity': [1, 2], 'ledger_identity': [3, 4]}
    raw = driver.canonical(config)
    config_path.write_bytes(raw)
    assert driver.parse_validation_parent_config(raw, str(config_path)) == config
    bad = dict(config, run_dir=str(workspace))
    with pytest.raises(ValueError):
        driver.parse_validation_parent_config(driver.canonical(bad), str(config_path))
    bad = dict(config, ledger_identity=[True, 4])
    with pytest.raises(ValueError):
        driver.parse_validation_parent_config(driver.canonical(bad), str(config_path))


def test_validation_parent_gate_precedes_any_parent_work(monkeypatch):
    calls = []
    monkeypatch.setattr(driver, 'run_validation_parent', lambda path: calls.append(path))
    monkeypatch.setattr(driver, 'run_validation_observer', lambda path: calls.append('observe:' + path))
    monkeypatch.setattr(driver, 'run_validation_operator', lambda path: calls.append('operator:' + path))
    monkeypatch.setattr(driver, 'run_validation_root', lambda path: calls.append('root:' + path))
    assert driver.main(['--validation-parent', 'private-config'], io.BytesIO(b'')) == 1
    assert calls == []
    assert driver.main(['--validation-parent', 'private-config'], io.BytesIO(driver.START_TOKEN)) == 0
    assert calls == ['private-config']
    assert driver.main(['--claim-parent', 'private-config'], io.BytesIO(driver.START_TOKEN)) == 1
    assert calls == ['private-config']
    assert driver.main(['--validation-observer', 'private-config'], io.BytesIO(driver.START_TOKEN)) == 0
    assert calls == ['private-config', 'observe:private-config']
    assert driver.main(['--validation-operator', 'private-config'], io.BytesIO(driver.START_TOKEN)) == 0
    assert calls == ['private-config', 'observe:private-config', 'operator:private-config']
    assert driver.main(['--validation-root', 'private-config'], io.BytesIO(driver.START_TOKEN)) == 0
    assert calls == ['private-config', 'observe:private-config', 'operator:private-config', 'root:private-config']


def test_validation_parent_eof_never_reads_config_or_starts_external_work(tmp_path):
    source = Path(driver.__file__)
    prefix = tmp_path / 'prefix'
    prefix.mkdir()
    config = tmp_path / 'absent-config.json'
    script = """
import atexit, hashlib, sys
source, digest, config = sys.argv[1:]
events = []
def audit(event, args):
    if event in ('subprocess.Popen', 'socket.connect', 'socket.getaddrinfo'):
        events.append(event)
    elif event == 'open' and args and str(args[0]) == config:
        events.append(event)
    elif event == 'import' and args and args[0].split('.')[0] in ('torch', 'transformers', 'looking_glass'):
        events.append(event)
sys.addaudithook(audit)
atexit.register(lambda: print('unexpected-before-gate') if events else None)
raw = open(source, 'rb').read()
assert hashlib.sha256(raw).hexdigest() == digest
sys.argv = [source, '--validation-parent', config]
exec(compile(raw, source, 'exec'), {'__name__': '__main__', '__file__': source})
"""
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                             '-c', script, str(source), hashlib.sha256(source.read_bytes()).hexdigest(), str(config)],
                            input=b'', capture_output=True, timeout=30)
    assert result.returncode == 1 and result.stdout == b'' and result.stderr == b''
    assert not config.exists() and not any(prefix.iterdir())


def test_no_model_validation_requires_an_unclaimed_ledger(tmp_path):
    workspace = tmp_path / 'workspace'
    ledger = workspace / 'work' / 'agreement-study-ledger-v5' / 'ledger.sqlite3'
    ledger.parent.mkdir(parents=True)
    parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
    parent.mkdir()
    database = sqlite3.connect(ledger)
    database.execute('CREATE TABLE claims (phase TEXT)')
    database.commit()
    identity = (ledger.stat().st_dev, ledger.stat().st_ino)
    assert driver.check_unclaimed_validation(str(workspace), identity) is True
    database.execute("INSERT INTO claims VALUES ('injection_heldout')")
    database.commit()
    with pytest.raises(ValueError):
        driver.check_unclaimed_validation(str(workspace), identity)
    database.close()


def test_validation_observer_rebuilds_the_exact_child_config_and_parent_terminal(tmp_path):
    from types import SimpleNamespace
    from tools import injection_runtime_replay_child as child
    workspace = tmp_path / 'workspace'
    run = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1' / 'validation-01'
    run.mkdir(parents=True)
    locations, _, _ = _inputs(workspace / 'sources')
    public = {name: str(workspace / (name + '.json')) for name in
              ('discovery_runtime', 'heldout_runtime', 'discovery_plan', 'runtime_deviation')}
    context = {'source_locations': locations, 'public_paths': public, 'reference_dir': str(workspace / 'refs'),
               'plan_path': str(workspace / 'plan.json'), 'plan_url': child.TEST_PLAN_URL,
               'frozen_root': str(workspace / 'frozen'), 'cache_dir': str(workspace / 'cache'),
               'plan': {'sources': {'child': '1' * 64, 'driver': '2' * 64}}, 'plan_bytes': b'plan'}
    config = {'workspace': str(workspace), 'run_dir': str(run), 'plan_url': child.TEST_PLAN_URL,
              'plan_path': context['plan_path'], 'frozen_root': context['frozen_root'],
              'cache_dir': context['cache_dir'], 'public_paths': public, 'source_locations': locations}
    values = driver.validation_child_values(context, config, child)
    actual = driver.bind_replay_config(context, child, values, str(run / 'v2-cache-fake'))
    (run / 'child-config.json').write_bytes(child._canonical(actual))
    supervisor = run / 'child-supervisor'
    supervisor.mkdir()
    (supervisor / 'supervisor.json').write_bytes(driver.canonical({'child_pid': 17}))
    expected = {'schema': 'fake-validation-terminal', 'parent_pid': 42, 'parent_birth_100ns': 123}
    owner = SimpleNamespace(_BOOTSTRAP='bootstrap', child_command=lambda *args, **kwargs: ['fixed-command'],
        check_supervisor_record=lambda *args, **kwargs: expected)
    raw = driver.canonical(expected)
    assert driver.check_validation_parent_terminal(context, config, child, owner, raw, 42, 123) is True
    (supervisor / 'supervisor.json').write_bytes(driver.canonical({'child_pid': 42}))
    with pytest.raises(ValueError):
        driver.check_validation_parent_terminal(context, config, child, owner, raw, 42, 123)
    (supervisor / 'supervisor.json').write_bytes(driver.canonical({'child_pid': 17}))
    altered = dict(actual, pycache_prefix=str(run / 'unprotected-prefix'))
    (run / 'child-config.json').write_bytes(child._canonical(altered))
    with pytest.raises(ValueError):
        driver.check_validation_parent_terminal(context, config, child, owner, raw, 42, 123)
    (run / 'child-config.json').write_bytes(child._canonical(actual))
    parent_config = run / 'parent-config.json'
    parent_config.write_bytes(b'config')
    (run / 'parent-terminal.json').write_bytes(raw)
    command = [str(sys._base_executable), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(run / 'v2-cache-fake'),
               '-c', owner._BOOTSTRAP, locations['driver']['path'], '2' * 64, '--validation-parent', str(parent_config)]
    (run / 'parent-command.json').write_bytes(driver.canonical(command))
    observation = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
        'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
        'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
        'parent_pid': 42, 'parent_birth_100ns': 123, 'config_sha256': hashlib.sha256(b'config').hexdigest(),
        'operation_sha256': hashlib.sha256(b'plan').hexdigest(),
        'command_sha256': hashlib.sha256(driver.canonical(command)).hexdigest(),
        'supervisor_source_sha256': '2' * 64, 'supervisor_terminal_sha256': hashlib.sha256(raw).hexdigest()}
    observed_raw = driver.canonical(observation)
    terminal = driver.make_validation_observer_terminal(context, config, child, owner, observed_raw, 77, 456)
    assert terminal['observer_pid'] == 77 and terminal['parent_observation_sha256'] == hashlib.sha256(observed_raw).hexdigest()
    for alias in (42, 17):
        with pytest.raises(ValueError):
            driver.make_validation_observer_terminal(context, config, child, owner, observed_raw, alias, 456)
    observer_dir = run / 'parent-observer'
    observer_dir.mkdir()
    (observer_dir / 'observation.json').write_bytes(observed_raw)
    terminal_raw = driver.canonical(terminal)
    (run / 'observer-terminal.json').write_bytes(terminal_raw)
    outer_command = list(command)
    outer_command[10] = '--validation-observer'
    (run / 'observer-command.json').write_bytes(driver.canonical(outer_command))
    outer = dict(observation, parent_pid=77, parent_birth_100ns=456,
                 command_sha256=hashlib.sha256(driver.canonical(outer_command)).hexdigest(),
                 supervisor_terminal_sha256=hashlib.sha256(terminal_raw).hexdigest())
    outer_raw = driver.canonical(outer)
    exit_receipt = driver.make_validation_exit_receipt(context, config, child, owner, outer, outer_raw, 99, 789)
    assert exit_receipt['status'] == 'awaiting_operator_exit' and exit_receipt['observer_pid'] == 77
    outer_dir = run / 'observer-observer'
    outer_dir.mkdir()
    (outer_dir / 'observation.json').write_bytes(outer_raw)
    exit_raw = driver.canonical(exit_receipt)
    assert driver.check_validation_exit_receipt(context, config, child, owner, exit_raw, 99, 789) is True
    with pytest.raises(ValueError):
        driver.check_validation_exit_receipt(context, config, child, owner, exit_raw, 100, 789)
    for alias in (77, 42, 17):
        with pytest.raises(ValueError):
            driver.make_validation_exit_receipt(context, config, child, owner, outer, outer_raw, alias, 789)
    corrupt_outer = dict(outer, parent_reaped=1)
    with pytest.raises(ValueError):
        driver.make_validation_exit_receipt(context, config, child, owner, corrupt_outer,
                                            driver.canonical(corrupt_outer), 99, 789)
    corrupt = dict(observation, parent_reaped=1)
    with pytest.raises(ValueError):
        driver.make_validation_observer_terminal(context, config, child, owner, driver.canonical(corrupt), 77, 456)
    actual['producer_source_file'] = str(workspace / 'wrong-producer.py')
    (run / 'child-config.json').write_bytes(child._canonical(actual))
    with pytest.raises(ValueError):
        driver.check_validation_parent_terminal(context, config, child, owner, raw, 42, 123)
