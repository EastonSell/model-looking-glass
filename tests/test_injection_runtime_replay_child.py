"""No-model tests for the source-first public-replay child bootstrap."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import injection_runtime_replay_child as child


def test_child_start_gate_requires_exact_token():
    child.await_start(io.BytesIO(b'MLG-natural-discovery-start-v1\n'))
    for bad in (b'', b'MLG-natural-discovery-start-v1', b'wrong\n'):
        with pytest.raises(ValueError):
            child.await_start(io.BytesIO(bad))


def test_changed_source_does_not_execute_before_digest_check(tmp_path):
    marker = tmp_path / 'marker'
    source = tmp_path / 'source.py'
    source.write_text('open(%r,"w").write("executed")' % str(marker), encoding='utf-8')
    with pytest.raises(ValueError):
        child.load_source(source, '0' * 64, '_fabricated_replay_child_source', '')
    assert not marker.exists()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    child.load_source(source, digest, '_fabricated_replay_child_source', '')
    assert marker.exists()
    sys.modules.pop('_fabricated_replay_child_source', None)


def test_private_config_is_exact_path_only_and_rejects_secret_values(tmp_path):
    config = {'schema': child.CONFIG_SCHEMA, **{key: str(tmp_path / key) for key in child.CONFIG_PATHS},
              'plan_url': child.TEST_PLAN_URL}
    raw = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    assert child.parse_config(raw) == config
    config['nonce'] = 'fabricated-secret'
    with pytest.raises(ValueError):
        child.parse_config(json.dumps(config).encode())
    config.pop('nonce')
    config['site_nonce_file'] = 'relative'
    with pytest.raises(ValueError):
        child.parse_config(json.dumps(config).encode())


def _plan():
    return {'schema': 'mlg-injection-runtime-replay-plan-v1',
            'discovery_source_commit': child._SOURCE, 'protocol_sha256': child._PROTOCOL,
            'checkpoint_revision': child._CHECKPOINT, 'discovery_runtime_sha256': child._DISCOVERY_RUNTIME,
            'prediction_freeze_sha256': child._FREEZE, 'prediction_sha256': child._PREDICTION,
            'source_supplement_sha256': child._SOURCE_SUPPLEMENT,
            'heldout_runtime_sha256': child._HELDOUT_RUNTIME, 'discovery_plan_sha256': child._DISCOVERY_PLAN,
            'runtime_deviation_sha256': child._DEVIATION, 'site_commitment': child._SITE_COMMITMENT,
            'model_state_sha256': child._MODEL_STATE, 'reference_commit': child._REFERENCE_COMMIT,
            'roles': list(child._ROLES), 'budgets': dict(child._BUDGETS),
            'reference_sha256': {role: '1' * 64 for role in child._ROLES},
            'sources': {name: child._PRODUCER if name == 'producer' else '2' * 64
                        for name in child._SOURCE_NAMES}}


def test_published_plan_binds_fixed_design_and_exact_child():
    plan = _plan()
    assert child.validate_plan(child._canonical(plan), child_source_sha256='2' * 64) == plan
    for mutate in (lambda p: p['roles'].reverse(),
                   lambda p: p['budgets'].__setitem__('pilot_logical_bytes', True),
                   lambda p: p['sources'].__setitem__('producer', '3' * 64),
                   lambda p: p['sources'].__setitem__('child', '3' * 64),
                   lambda p: p['sources'].pop('cpu_job_adapter'),
                   lambda p: p['reference_sha256'].pop('clean')):
        candidate = _plan()
        mutate(candidate)
        with pytest.raises(ValueError):
            child.validate_plan(child._canonical(candidate), child_source_sha256='2' * 64)


def test_runtime_difference_is_exactly_the_disclosed_fingerprint():
    before = {'schema': 'fake', 'system_environment_sha256': '1' * 64, 'numerics': {'x': 1}}
    after = {'schema': 'fake', 'system_environment_sha256': '2' * 64, 'numerics': {'x': 1}}
    assert child._check_runtime(child._canonical(before), child._canonical(after)) == after
    after['numerics']['x'] = 2
    with pytest.raises(ValueError):
        child._check_runtime(child._canonical(before), child._canonical(after))


def test_claim_binding_requires_exact_plan_and_pilot_paths(tmp_path):
    pilot = tmp_path / 'runtime-compatibility-replay-v1'
    config = {'attempt_dir': str(pilot / 'attempt'), 'postcheck_file': str(pilot / 'child-postcheck.json')}
    receipt = {'schema': 'mlg-injection-replay-claim-receipt-v1', 'status': 'preclaimed',
               'plan_sha256': '1' * 64, 'intent_sha256': '2' * 64, 'ledger_path': str(tmp_path / 'db'),
               'phase_path': str(tmp_path / ('injection-heldout-' + child._FREEZE)), 'pilot_path': str(pilot),
               'diagnostic_path': str(tmp_path / 'diagnostic'), 'parent_identity': [1, 2],
               'ledger_identity': [3, 4], **child._BUDGETS, 'diagnostic_actual_bytes': 12,
               'diagnostic_inventory_sha256': '3' * 64, 'injection_claim_row_sha256': '4' * 64}
    child._check_claim_receipt(child._canonical(receipt), '1' * 64, config)
    for key, bad in (('plan_sha256', '5' * 64), ('phase_logical_bytes', True),
                     ('injection_claim_row_sha256', 'bogus'), ('pilot_path', str(tmp_path / 'other'))):
        changed = dict(receipt)
        changed[key] = bad
        with pytest.raises(ValueError):
            child._check_claim_receipt(child._canonical(changed), '1' * 64, config)


def test_postcheck_observes_runtime_without_reinitializing(monkeypatch, tmp_path):
    from types import SimpleNamespace
    recorded = {'numerics': {'tested': True}, 'system_environment_sha256': 'a' * 64,
                'environment': {'ONLY_FOR_TEST': 'value'}}
    monkeypatch.setenv('ONLY_FOR_TEST', 'value')
    loader = SimpleNamespace(_CONTEXT={'repo': 'repo', 'prefix': tmp_path},
        _numeric_state=lambda torch: {'tested': True}, _environment_policy=lambda: 'a' * 64,
        _cache_policy=lambda cache: ('home', 'cache'), _verify_effective_cache=lambda home, cache: None,
        _verify_unloaded_source=lambda repo, revision: 'repository')
    state = {'torch': object(), 'home': 'home', 'cache': 'cache', 'events': [], 'repository': 'repository'}
    child._post_runtime_check(loader, state, recorded)
    loader._numeric_state = lambda torch: {'tested': False}
    with pytest.raises(ValueError):
        child._post_runtime_check(loader, state, recorded)


def test_validation_path_has_no_claim_private_read_or_model_load(monkeypatch, tmp_path):
    from types import SimpleNamespace
    paths = {key: str(tmp_path / key) for key in child.CONFIG_PATHS}
    references = tmp_path / 'references'
    references.mkdir()
    paths['reference_dir'] = str(references)
    paths['postcheck_file'] = str(tmp_path / 'postcheck.json')
    old = child._canonical({'system_environment_sha256': '1' * 64, 'numerics': {}, 'environment': {}})
    new = child._canonical({'system_environment_sha256': '2' * 64, 'numerics': {}, 'environment': {}})
    for key, raw in (('discovery_runtime_file', old), ('heldout_runtime_file', new),
                     ('discovery_plan_file', b'public-plan'), ('runtime_deviation_file', b'public-deviation')):
        Path(paths[key]).write_bytes(raw)
    for constant, raw in (('_DISCOVERY_RUNTIME', old), ('_HELDOUT_RUNTIME', new),
                          ('_DISCOVERY_PLAN', b'public-plan'), ('_DEVIATION', b'public-deviation')):
        monkeypatch.setattr(child, constant, hashlib.sha256(raw).hexdigest())
    plan = _plan()
    plan['sources']['child'] = hashlib.sha256(Path(child.__file__).read_bytes()).hexdigest()
    for role in child._ROLES:
        raw = ('public-' + role).encode()
        (references / ('observation-00-' + role + '.json')).write_bytes(raw)
        plan['reference_sha256'][role] = hashlib.sha256(raw).hexdigest()
    plan_bytes = child._canonical(plan)
    Path(paths['replay_plan_file']).write_bytes(plan_bytes)
    config = {'schema': child.CONFIG_SCHEMA, **paths, 'plan_url': child.TEST_PLAN_URL}
    config_path = tmp_path / 'config.json'
    config_path.write_bytes(child._canonical(config))
    loader = SimpleNamespace(_boot=lambda site, source: {'repo': 'fake'},
                             _initialize_runtime=lambda **kw: {'runtime_bytes': new})
    loader._load_trusted_model = lambda state: pytest.fail('validation loaded the model')
    producer = SimpleNamespace(_ROLES=child._ROLES,
        _REFERENCE_SHA256=tuple(plan['reference_sha256'][role] for role in child._ROLES))
    monkeypatch.setattr(child, '_assert_isolated', lambda prefix: None)
    monkeypatch.setattr(child, '_fetch_immutable', lambda url: plan_bytes)
    monkeypatch.setattr(child, 'load_source', lambda *args: producer if args[2].endswith('_replay') else loader)
    monkeypatch.setattr(child, '_post_runtime_check', lambda *args: None)
    assert child.execute(str(config_path), validation_only=True)['status'] == 'validated_no_model_forward'
    assert not Path(paths['claim_receipt_file']).exists()
    assert not Path(paths['site_nonce_file']).exists()
    assert not Path(paths['site_record_file']).exists()
    assert not Path(paths['attempt_dir']).exists()


def test_main_requires_gate_and_never_reports_systemexit_zero_as_success(monkeypatch):
    calls = []
    def execute(*args, **kwargs):
        calls.append('ran')
        raise SystemExit(0)
    monkeypatch.setattr(child, 'execute', execute)
    assert child.main(['--validate-only', 'fake'], io.BytesIO(b'')) == 1
    assert not calls
    assert child.main(['--validate-only', 'fake'], io.BytesIO(child._START)) == 1
    assert calls == ['ran']


def test_direct_isolated_interpreter_accepts_protected_empty_prefix(tmp_path):
    prefix = tmp_path / 'prefix'
    prefix.mkdir()
    script = ("p = %r\nns = {'__file__': p, '__name__': '_test_replay_bootstrap'}\n"
              "exec(compile(open(p, 'rb').read(), p, 'exec'), ns)\nns['_assert_isolated'](%r)\n"
              % (str(Path(child.__file__).resolve()), str(prefix)))
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-X',
                             'pycache_prefix=' + str(prefix), '-c', script],
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
    assert not list(prefix.iterdir())


def test_postcheck_close_failure_does_not_mask_cancellation(monkeypatch, tmp_path):
    close = child.os.close
    def interrupted_write(fd, raw):
        raise KeyboardInterrupt()
    def failed_close(fd):
        close(fd)
        raise OSError('fabricated close failure')
    monkeypatch.setattr(child.os, 'write', interrupted_write)
    monkeypatch.setattr(child.os, 'close', failed_close)
    with pytest.raises(KeyboardInterrupt):
        child._write_once(tmp_path / 'postcheck', {'status': 'incomplete'})


def test_postcheck_preserves_a_fatal_close_after_write_failure(monkeypatch, tmp_path):
    close = child.os.close
    def failed_write(fd, raw):
        raise OSError('fabricated write failure')
    def interrupted_close(fd):
        close(fd)
        raise KeyboardInterrupt()
    monkeypatch.setattr(child.os, 'write', failed_write)
    monkeypatch.setattr(child.os, 'close', interrupted_close)
    with pytest.raises(KeyboardInterrupt):
        child._write_once(tmp_path / 'postcheck', {'status': 'incomplete'})


def test_private_terminal_comparison_distinguishes_bool_from_int():
    produced = {'status': 'replay-produced-private-awaiting-independent-audit',
                'forward_count': 16, 'comparisons': [{'within_tolerance': True, 'bit_equal': False}]}
    child._check_producer_terminal(child._canonical(produced), produced)
    modified = dict(produced, comparisons=[{'within_tolerance': 1, 'bit_equal': 0}])
    with pytest.raises(ValueError):
        child._check_producer_terminal(child._canonical(modified), produced)
