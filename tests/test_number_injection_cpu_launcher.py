"""No-model tests for the public CPU launcher gates."""
import importlib
import io
import hashlib
import json
import os
import py_compile
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

try:
    launcher = importlib.import_module('scripts.pretrained_number_cpu_launcher')
except ModuleNotFoundError as error:
    if error.name != 'scripts.pretrained_number_cpu_launcher':
        raise
    launcher = None


class Response(io.BytesIO):
    def __init__(self, payload, url):
        super().__init__(payload)
        self._url = url

    def geturl(self):
        return self._url


def runtime_for(source):
    from looking_glass.pretrained_number_commitments import PROTOCOL_SHA256, CHECKPOINT_REVISION
    return json.dumps({'schema': 'mlg-number-injection-cpu-runtime-v1', 'source_commit': source,
                       'protocol_sha256': PROTOCOL_SHA256, 'checkpoint_revision': CHECKPOINT_REVISION},
                      sort_keys=True, separators=(',', ':')).encode('ascii')


def plan_attempt(tmp_path, source, plan, runtime):
    directory = tmp_path / 'plan-attempt'
    directory.mkdir()
    (directory / 'plan.json').write_bytes(plan)
    (directory / 'started.json').write_bytes(launcher._canonical({
        'schema': 'mlg-number-plan-start-v1', 'source_commit': source,
        'runtime_sha256': hashlib.sha256(runtime).hexdigest()}))
    (tmp_path / 'number-injection-plan-attempt.json').write_bytes(launcher._canonical({
        'schema': 'mlg-number-plan-root-v1', 'source_commit': source,
        'runtime_sha256': hashlib.sha256(runtime).hexdigest(), 'plan_directory': directory.name}))
    return directory


@pytest.fixture
def safe_environment():
    with mock.patch.object(launcher, '_environment_policy', return_value='0' * 64):
        yield


def test_publication_gate_checks_exact_bytes_and_pinned_github_url():
    """A changed or redirected artifact must not authorize a forward."""
    assert launcher is not None, 'CPU launcher is not implemented'
    url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
           + 'a' * 40 + '/results/public-plan.json')
    with mock.patch.object(launcher.urllib.request, 'urlopen',
                           side_effect=lambda request, **kw: Response(b'published', request.full_url)):
        launcher.verify_published_bytes(url, b'published')
        with pytest.raises(ValueError):
            launcher.verify_published_bytes(url, b'different')
        with pytest.raises(ValueError):
            launcher.verify_published_bytes(url.replace('a' * 40, 'main'), b'published')
    with mock.patch.object(launcher.urllib.request, 'urlopen',
                           side_effect=lambda request, **kw: Response(b'published', 'https://example.com/')):
        with pytest.raises(ValueError):
            launcher.verify_published_bytes(url, b'published')


def test_boot_refuses_unverified_source_before_executing_bootstrap(tmp_path, safe_environment):
    """A mismatched package or bootstrap cannot run its Python initializer."""
    with mock.patch.object(launcher, 'verify_published_bytes'), \
            mock.patch.object(launcher, '_verify_unloaded_source', side_effect=ValueError('source mismatch')), \
            mock.patch.object(launcher.importlib.util, 'spec_from_file_location') as load_spec:
        with pytest.raises(ValueError, match='source mismatch'):
            launcher._boot(tmp_path, 'a' * 40)
        load_spec.assert_not_called()


def test_bootstrap_executes_verified_source_not_timestamp_valid_stale_bytecode(tmp_path):
    """-B alone does not stop reads from an existing pycache."""
    source = tmp_path / 'bootstrap.py'
    source.write_bytes(b"value = 'stale'\n")
    with mock.patch.object(sys, 'pycache_prefix', str(tmp_path / 'pycache')):
        cached = importlib.util.cache_from_source(str(source))
        py_compile.compile(str(source), cfile=cached, doraise=True)
        before = source.stat()
        source.write_bytes(b"value = 'fresh'\n")
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        with mock.patch.object(importlib.util, 'spec_from_file_location',
                               side_effect=AssertionError('pyc-capable loader used')):
            result = launcher._execute_bootstrap_source(source, hashlib.sha256(source.read_bytes()).hexdigest())
    assert result.value == 'fresh'


def test_preimport_source_check_rejects_modified_or_untracked_package(tmp_path):
    """The real Git blobs are checked before any project Python is imported."""
    repo = tmp_path / 'checkout'
    repo.mkdir()
    paths = ('looking_glass/__init__.py', 'looking_glass/component.py',
             'scripts/pretrained_number_cpu_launcher.py',
             'scripts/distilgpt2_intervention_dev_bootstrap.py',
             'docs/PRETRAINED_NUMBER_PATCH_V1.md', 'docs/PRETRAINED_NUMBER_INJECTION_REQUESTS.md')
    for name in paths:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('fixture ' + name + '\n').encode('ascii'))
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    git('init', '--quiet')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '--quiet', '-m', 'fixture')
    revision = git('rev-parse', 'HEAD').decode('ascii').strip()
    git('checkout', '--quiet', '--detach')
    hook, marker = tmp_path / 'hook.sh', tmp_path / 'marker'
    hook.write_text('#!/bin/sh\nprintf x > "' + marker.as_posix() + '"\ncat\n', encoding='ascii')
    git('config', '--local', 'core.fsmonitor', hook.as_posix())
    git('config', '--local', 'filter.probe.clean', hook.as_posix())
    (repo / '.git' / 'info' / 'attributes').write_text('*.py filter=probe\n', encoding='ascii')
    launcher._verify_unloaded_source(repo, revision)
    assert not marker.exists(), 'source verification executed a Git hook or clean filter'
    (repo / 'looking_glass' / '__init__.py').write_bytes(b'changed\n')
    with pytest.raises(ValueError):
        launcher._verify_unloaded_source(repo, revision)
    git('checkout', '--', 'looking_glass/__init__.py')
    (repo / 'looking_glass' / 'untracked.py').write_bytes(b'untracked\n')
    with pytest.raises(ValueError):
        launcher._verify_unloaded_source(repo, revision)


def test_workspace_cache_policy_requires_exact_local_hub_and_transformers_paths(tmp_path):
    home = tmp_path / 'huggingface'
    cache = home / 'hub'
    cache.mkdir(parents=True)
    expected = {'MLG_STUDY_WORKSPACE': str(tmp_path), 'HF_HOME': str(home),
                'HF_HUB_CACHE': str(cache), 'TRANSFORMERS_CACHE': str(cache)}
    with mock.patch.dict(launcher.os.environ, expected, clear=True):
        assert launcher._cache_policy(cache) == (home.resolve(), cache.resolve())
        launcher.os.environ['TRANSFORMERS_CACHE'] = str(tmp_path / 'elsewhere')
        with pytest.raises(ValueError):
            launcher._cache_policy(cache)
        launcher.os.environ['TRANSFORMERS_CACHE'] = str(cache)
        launcher.os.environ['HF_HOME'] = str(tmp_path.parent)
        with pytest.raises(ValueError):
            launcher._cache_policy(cache)


def test_effective_hub_cache_readback_must_match_configured_paths(tmp_path):
    home, cache = tmp_path / 'hf', tmp_path / 'hf' / 'hub'
    cache.mkdir(parents=True)
    constants = SimpleNamespace(HF_HOME=str(home), HF_HUB_CACHE=str(cache),
                                HF_HUB_OFFLINE=True, HF_HUB_DISABLE_TELEMETRY=True)
    transformer_hub = SimpleNamespace(TRANSFORMERS_CACHE=str(cache), is_offline_mode=lambda: True)
    with mock.patch.object(launcher.importlib, 'import_module',
            side_effect=lambda name: constants if name == 'huggingface_hub.constants' else transformer_hub):
        launcher._verify_effective_cache(home, cache)
        constants.HF_HUB_CACHE = str(tmp_path / 'different')
        with pytest.raises(ValueError):
            launcher._verify_effective_cache(home, cache)


def test_numeric_environment_pins_mkl_and_rejects_unrecorded_backend_controls():
    fixed = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
             'OPENBLAS_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE'}
    with mock.patch.dict(launcher.os.environ, fixed, clear=True):
        assert launcher._numeric_environment() == fixed
        launcher.os.environ['MKL_CBWR'] = 'AUTO'
        with pytest.raises(ValueError):
            launcher._numeric_environment()
        launcher.os.environ['MKL_CBWR'] = 'COMPATIBLE'
        launcher.os.environ['OMP_SCHEDULE'] = 'dynamic'
        with pytest.raises(ValueError):
            launcher._numeric_environment()
        del launcher.os.environ['OMP_SCHEDULE']
        launcher.os.environ['ATEN_CPU_CAPABILITY'] = 'default'
        with pytest.raises(ValueError):
            launcher._numeric_environment()


def test_runtime_environment_rejects_all_unlisted_variables_and_binds_os_values():
    fixed = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'MLG_STUDY_WORKSPACE': 'workspace', 'HF_HOME': 'home', 'HF_HUB_CACHE': 'cache',
        'TRANSFORMERS_CACHE': 'cache', 'PATH': 'trusted-bin'}
    with mock.patch.dict(launcher.os.environ, fixed, clear=True):
        first = launcher._environment_policy()
        launcher.os.environ['PATH'] = 'different-bin'
        assert launcher._environment_policy() != first
        launcher.os.environ['NPY_DISABLE_CPU_FEATURES'] = 'AVX2'
        with pytest.raises(ValueError):
            launcher._environment_policy()
        del launcher.os.environ['NPY_DISABLE_CPU_FEATURES']
        launcher.os.environ['UNREVIEWED_CONTROL'] = 'anything'
        with pytest.raises(ValueError):
            launcher._environment_policy()


def test_real_cpu_numeric_readback_rejects_default_dtype_and_mkldnn_drift():
    code = '''
import torch
from scripts import pretrained_number_cpu_launcher as m
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_float32_matmul_precision('highest')
torch.set_default_dtype(torch.float32)
torch.set_flush_denormal(False)
torch.backends.mkldnn.enabled = True
torch.backends.mkldnn.deterministic = True
record = m._numeric_state(torch)
assert record['default_dtype'] == 'float32' and record['mkldnn_enabled'] is True
assert record['mkldnn_deterministic'] is True and record['deterministic_warn_only'] is False
assert record['cpu_capability'] and len(record['torch_config_sha256']) == 64
for update in (lambda: torch.set_default_dtype(torch.float64),
               lambda: setattr(torch.backends.mkldnn, 'enabled', False),
               lambda: setattr(torch.backends.mkldnn, 'deterministic', False),
               lambda: torch.use_deterministic_algorithms(True, warn_only=True)):
    torch.set_default_dtype(torch.float32)
    torch.backends.mkldnn.enabled = True
    torch.backends.mkldnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    update()
    try: m._numeric_state(torch)
    except ValueError: pass
    else: raise AssertionError('numeric drift accepted')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode == 0, result.stderr


def test_tokenizer_checks_all_public_rows_before_forward():
    from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
    rows = [{'text': text, 'input_ids': list(ids)} for text, ids in PUBLIC_DISCOVERY]
    tokenization = mock.Mock(return_value=(rows, []))
    module = SimpleNamespace(tokenize_inputs=tokenization)
    tokenizer = object()
    with mock.patch.dict(sys.modules, {'looking_glass.agreement_preflight_v1': module}):
        launcher._verify_public_tokenizer(tokenizer)
        tokenization.assert_called_once_with(tokenizer)
        rows[-1]['input_ids'][-1] += 1
        with pytest.raises(ValueError):
            launcher._verify_public_tokenizer(tokenizer)


@pytest.mark.parametrize('write_error,close_error,expected', [
    (KeyboardInterrupt('cancel'), OSError('close failed'), KeyboardInterrupt),
    (OSError('write failed'), KeyboardInterrupt('cancel'), KeyboardInterrupt),
    (SystemExit(7), OSError('close failed'), SystemExit),
    (OSError('write failed'), SystemExit(9), SystemExit),
])
def test_evidence_write_preserves_cancellation_across_close_error(tmp_path, write_error, close_error, expected):
    with mock.patch.object(launcher.os, 'open', return_value=42), \
            mock.patch.object(launcher.os, 'write', side_effect=write_error), \
            mock.patch.object(launcher.os, 'close', side_effect=close_error):
        with pytest.raises(expected) as captured:
            launcher._write_new(tmp_path / 'evidence', b'partial')
        if expected is SystemExit:
            assert captured.value.code == (7 if isinstance(write_error, SystemExit) else 9)


def test_site_nonce_is_not_drawn_until_the_public_plan_is_verified(tmp_path, safe_environment):
    """A failed publication check must leave no generated site or custody directory."""
    assert launcher is not None
    assert callable(getattr(launcher, 'create_site_custody', None)), 'site custody gate is not implemented'
    destination = tmp_path / 'private-custody'
    with mock.patch.object(launcher, 'verify_published_bytes', side_effect=ValueError('publication mismatch')):
        with mock.patch.object(launcher.secrets, 'token_bytes', side_effect=AssertionError('nonce drawn too early')):
            with pytest.raises(ValueError, match='publication mismatch'):
                launcher.create_site_custody(source_commit='a' * 40, runtime_bytes=b'public runtime',
                    runtime_url='https://example.invalid/runtime', plan_bytes=b'public plan',
                    plan_url='https://example.invalid/plan', plan_attempt_dir=tmp_path / 'plan',
                    custody_dir=destination)
    assert not destination.exists()


def test_site_custody_saves_one_fabricated_nonce_and_rejects_reuse(tmp_path, safe_environment):
    """A successful commitment has private inputs and its attempt cannot be rerolled."""
    from looking_glass.pretrained_number_commitments import verify_site_reveal
    from looking_glass.pretrained_number_requests import create_family_request_plan
    source = 'a' * 40
    plan = create_family_request_plan(b'p' * 32, source)
    plan_dir = plan_attempt(tmp_path, source, plan, runtime_for(source))
    dest = tmp_path / 'private-custody'
    with mock.patch.object(launcher, 'verify_published_bytes'), \
            mock.patch.object(launcher.secrets, 'token_bytes', return_value=b's' * 32):
        digest = launcher.create_site_custody(source_commit=source, runtime_bytes=runtime_for(source),
                    runtime_url='runtime', plan_bytes=plan, plan_url='plan', plan_attempt_dir=plan_dir,
                    custody_dir=dest)
        assert (dest / 'nonce.bin').read_bytes() == b's' * 32
        assert (dest / 'commitment.txt').read_bytes() == (digest + '\n').encode('ascii')
        assert 1 <= verify_site_reveal((dest / 'nonce.bin').read_bytes(),
                    (dest / 'site-record.json').read_bytes(), digest, expected_source_commit=source) <= 4
        with pytest.raises(ValueError):
            launcher.create_site_custody(source_commit=source, runtime_bytes=runtime_for(source),
                    runtime_url='runtime', plan_bytes=plan, plan_url='plan', plan_attempt_dir=plan_dir,
                    custody_dir=tmp_path / 'different-private-attempt')


def test_equal_public_and_private_entropy_aborts_without_reroll(tmp_path, safe_environment):
    """A collision retains its nonce and a failed attempt cannot be reused."""
    from looking_glass.pretrained_number_requests import create_family_request_plan
    source = 'b' * 40
    plan = create_family_request_plan(b'p' * 32, source)
    plan_dir = plan_attempt(tmp_path, source, plan, runtime_for(source))
    dest = tmp_path / 'collision'
    with mock.patch.object(launcher, 'verify_published_bytes'), \
            mock.patch.object(launcher.secrets, 'token_bytes', return_value=b'p' * 32):
        with pytest.raises(ValueError, match='equals public'):
            launcher.create_site_custody(source_commit=source, runtime_bytes=runtime_for(source),
                    runtime_url='runtime', plan_bytes=plan, plan_url='plan', plan_attempt_dir=plan_dir,
                    custody_dir=dest)
        assert (dest / 'nonce.bin').read_bytes() == b'p' * 32
        assert not (dest / 'commitment.txt').exists()
        with pytest.raises(ValueError):
            launcher.create_site_custody(source_commit=source, runtime_bytes=runtime_for(source),
                    runtime_url='runtime', plan_bytes=plan, plan_url='plan', plan_attempt_dir=plan_dir,
                    custody_dir=tmp_path / 'different-private-attempt')


def test_plan_draw_happens_only_after_runtime_publication(tmp_path, safe_environment):
    """An unpublished runtime must not consume request entropy or make an attempt."""
    assert callable(getattr(launcher, 'create_public_plan', None)), 'public plan gate is not implemented'
    dest = tmp_path / 'plan-attempt'
    with mock.patch.object(launcher, 'verify_published_bytes', side_effect=ValueError('not published')), \
            mock.patch.object(launcher.secrets, 'token_bytes', side_effect=AssertionError('entropy drawn too early')):
        with pytest.raises(ValueError, match='not published'):
            launcher.create_public_plan(source_commit='a' * 40, runtime_bytes=b'runtime',
                    runtime_url='runtime', plan_dir=dest)
    assert not dest.exists()


def test_plan_rejects_a_published_runtime_bound_to_different_source_before_entropy(tmp_path, safe_environment):
    """A source mismatch cannot consume entropy or create a plan attempt."""
    destination = tmp_path / 'plan-attempt'
    with mock.patch.object(launcher, 'verify_published_bytes'), \
            mock.patch.object(launcher.secrets, 'token_bytes', side_effect=AssertionError('entropy drawn too early')):
        with pytest.raises(ValueError):
            launcher.create_public_plan(source_commit='a' * 40, runtime_bytes=runtime_for('b' * 40),
                                        runtime_url='runtime', plan_dir=destination)
    assert not destination.exists()


def test_public_plan_attempt_draws_once_and_cannot_be_reused(tmp_path, safe_environment):
    from looking_glass.pretrained_number_requests import load_family_request_plan
    source = 'a' * 40
    destination = tmp_path / 'plan-attempt'
    with mock.patch.object(launcher, 'verify_published_bytes'), \
            mock.patch.object(launcher.secrets, 'token_bytes', return_value=b'p' * 32) as draw:
        digest = launcher.create_public_plan(source_commit=source, runtime_bytes=runtime_for(source),
                                             runtime_url='runtime', plan_dir=destination)
        load_family_request_plan((destination / 'plan.json').read_bytes(),
                                 expected_source_commit=source, expected_sha256=digest)
        with pytest.raises(ValueError):
            launcher.create_public_plan(source_commit=source, runtime_bytes=runtime_for(source),
                                        runtime_url='runtime', plan_dir=tmp_path / 'different-plan-attempt')
        draw.assert_called_once_with(32)


def test_run_rejects_unpublished_plan_before_reading_private_custody(tmp_path, safe_environment):
    """A publication failure cannot expose the private record or start a model."""
    assert callable(getattr(launcher, 'run_published_discovery', None)), 'discovery launch gate is not implemented'
    source = 'a' * 40
    with mock.patch.object(launcher, 'verify_published_bytes', side_effect=ValueError('not published')):
        with pytest.raises(ValueError, match='not published'):
            launcher.run_published_discovery(source_commit=source, runtime_bytes=runtime_for(source),
                runtime_url='runtime', plan_bytes=b'not yet public', plan_url='plan',
                commitment_bytes=b'0' * 64 + b'\n', commitment_url='commitment',
                plan_attempt_dir=tmp_path / 'plan', custody_dir=tmp_path / 'unreadable-private',
                attempt_dir=tmp_path / 'new-attempt',
                cache_dir=tmp_path, site_packages=tmp_path)
    assert not (tmp_path / 'new-attempt').exists()


def test_run_requires_custody_start_and_commitment_to_match_published_inputs(tmp_path, safe_environment):
    """A plan or runtime switch cannot reuse a private custody attempt."""
    from looking_glass.pretrained_number_requests import create_family_request_plan
    source = 'a' * 40
    runtime = runtime_for(source)
    plan = create_family_request_plan(b'p' * 32, source)
    plan_dir = plan_attempt(tmp_path, source, plan, runtime)
    custody = tmp_path / 'custody'
    with mock.patch.object(launcher, 'verify_published_bytes'), \
            mock.patch.object(launcher.secrets, 'token_bytes', return_value=b's' * 32):
        digest = launcher.create_site_custody(source_commit=source, runtime_bytes=runtime,
            runtime_url='runtime', plan_bytes=plan, plan_url='plan', plan_attempt_dir=plan_dir,
            custody_dir=custody)
    args = dict(source_commit=source, runtime_bytes=runtime, runtime_url='runtime',
        plan_bytes=plan, plan_url='plan', commitment_bytes=(digest + '\n').encode('ascii'),
        commitment_url='commitment', plan_attempt_dir=plan_dir, custody_dir=custody,
        attempt_dir=tmp_path / 'attempt',
        cache_dir=tmp_path, site_packages=tmp_path)
    started = custody / 'started.json'
    original = started.read_bytes()
    record = json.loads(original)
    record['plan_sha256'] = '0' * 64
    started.write_bytes(launcher._canonical(record))
    with mock.patch.object(launcher, 'verify_published_bytes'), \
            mock.patch.object(launcher, '_execute_verified') as execute:
        with pytest.raises(ValueError, match='custody'):
            launcher.run_published_discovery(**args)
        execute.assert_not_called()
        started.write_bytes(original)
        (custody / 'commitment.txt').write_bytes(b'0' * 64 + b'\n')
        with pytest.raises(ValueError, match='custody'):
            launcher.run_published_discovery(**args)
        execute.assert_not_called()
        (custody / 'commitment.txt').write_bytes(args['commitment_bytes'])
        launcher.run_published_discovery(**args)
        execute.assert_called_once()


def test_changed_actual_runtime_stops_before_model_load_or_attempt_creation(tmp_path):
    """A runtime differing from its published bytes cannot launch a model."""
    source = 'a' * 40
    assert callable(getattr(launcher, '_initialize_runtime', None)), 'runtime bootstrap is not implemented'
    with mock.patch.object(launcher, '_CONTEXT', object()), \
            mock.patch.object(launcher, '_initialize_runtime', return_value={'runtime_bytes': b'changed'}), \
            mock.patch.object(launcher, '_load_trusted_model', side_effect=AssertionError('model loaded')):
        with pytest.raises(ValueError, match='runtime'):
            launcher._execute_verified(source_commit=source, runtime_bytes=runtime_for(source),
                plan_bytes=b'plan', plan_sha256='b' * 64, site_nonce=b'c' * 32,
                site_record_bytes=b'private', site_commitment='d' * 64,
                attempt_dir=tmp_path / 'attempt', cache_dir=tmp_path, site_packages=tmp_path)
    assert not (tmp_path / 'attempt').exists()


def test_phase_grant_is_consumed_before_model_load_and_denial_stops_launch(tmp_path):
    source = 'a' * 40
    runtime = runtime_for(source)
    claim = mock.Mock(side_effect=ValueError('phase already claimed'))
    module = SimpleNamespace(claim_phase=claim)
    with mock.patch.dict(sys.modules, {'looking_glass.agreement_phase_ledger': module}), \
            mock.patch.object(launcher, '_initialize_runtime', return_value={'runtime_bytes': runtime}), \
            mock.patch.object(launcher, '_load_trusted_model') as load:
        with pytest.raises(ValueError, match='already claimed'):
            launcher._execute_verified(source_commit=source, runtime_bytes=runtime, plan_bytes=b'plan',
                plan_sha256='b' * 64, site_nonce=b'c' * 32, site_record_bytes=b'private',
                site_commitment='d' * 64, attempt_dir=tmp_path / 'attempt', cache_dir=tmp_path,
                site_packages=tmp_path, phase_ledger=tmp_path / 'study.db')
        claim.assert_called_once_with(tmp_path / 'study.db', 'injection_discovery', output_path=tmp_path / 'attempt')
        load.assert_not_called()
    assert not (tmp_path / 'attempt.launcher-started.json').exists()
    assert not (tmp_path / 'attempt.launcher-terminal.json').exists()


def test_phase_share_includes_launcher_markers_on_a_failed_supervisor_attempt(tmp_path):
    source = 'a' * 40
    runtime = runtime_for(source)
    events = []
    grant = SimpleNamespace(logical_limit_bytes=48 * 1024 ** 2, disk_limit_bytes=128 * 1024 ** 2)
    claim = mock.Mock(side_effect=lambda *args, **kwargs: events.append('claim') or grant)
    forward = mock.Mock(side_effect=ValueError('synthetic execution failure'))
    modules = {'looking_glass.agreement_phase_ledger': SimpleNamespace(claim_phase=claim),
               'looking_glass.pretrained_number_supervisor': SimpleNamespace(run_discovery=forward)}
    with mock.patch.dict(sys.modules, modules), \
            mock.patch.object(launcher, '_initialize_runtime', return_value={'runtime_bytes': runtime}), \
            mock.patch.object(launcher, '_load_trusted_model',
                              side_effect=lambda state: events.append('load') or SimpleNamespace(model=object())):
        with pytest.raises(ValueError, match='synthetic execution failure'):
            launcher._execute_verified(source_commit=source, runtime_bytes=runtime, plan_bytes=b'plan',
                plan_sha256='b' * 64, site_nonce=b'c' * 32, site_record_bytes=b'private',
                site_commitment='d' * 64, attempt_dir=tmp_path / 'attempt', cache_dir=tmp_path,
                site_packages=tmp_path, phase_ledger=tmp_path / 'study.db')
    marker = tmp_path / 'attempt.launcher-started.json'
    terminal = tmp_path / 'attempt.launcher-terminal.json'
    assert events == ['claim', 'load']
    assert forward.call_args.kwargs['logical_limit_bytes'] == grant.logical_limit_bytes
    assert forward.call_args.kwargs['disk_limit_bytes'] == grant.disk_limit_bytes - marker.stat().st_size - 4096
    assert terminal.stat().st_size <= 4096
    assert json.loads(terminal.read_bytes())['status'] == 'incomplete'


@pytest.mark.parametrize('failure,terminal_error,expected', [
    (KeyboardInterrupt('cancel'), OSError('terminal failed'), KeyboardInterrupt),
    (OSError('load failed'), KeyboardInterrupt('cancel'), KeyboardInterrupt),
    (SystemExit(7), OSError('terminal failed'), SystemExit),
    (OSError('load failed'), SystemExit(9), SystemExit),
])
def test_failure_marker_preserves_cancellation(tmp_path, failure, terminal_error, expected):
    runtime = runtime_for('a' * 40)
    grant = SimpleNamespace(logical_limit_bytes=48 * 1024 ** 2, disk_limit_bytes=128 * 1024 ** 2)
    real_write = launcher._write_new
    def write_marker(path, data):
        if str(path).endswith('.launcher-terminal.json'):
            raise terminal_error
        return real_write(path, data)
    with mock.patch.dict(sys.modules, {'looking_glass.agreement_phase_ledger':
                                     SimpleNamespace(claim_phase=lambda *a, **k: grant)}), \
            mock.patch.object(launcher, '_initialize_runtime', return_value={'runtime_bytes': runtime}), \
            mock.patch.object(launcher, '_load_trusted_model', side_effect=failure), \
            mock.patch.object(launcher, '_write_new', side_effect=write_marker):
        with pytest.raises(expected) as captured:
            launcher._execute_verified(source_commit='a' * 40, runtime_bytes=runtime, plan_bytes=b'plan',
                plan_sha256='b' * 64, site_nonce=b'c' * 32, site_record_bytes=b'private',
                site_commitment='d' * 64, attempt_dir=tmp_path / 'attempt', cache_dir=tmp_path,
                site_packages=tmp_path, phase_ledger=tmp_path / 'study.db')
        if expected is SystemExit:
            assert captured.value.code == (7 if isinstance(failure, SystemExit) else 9)


def test_model_is_reverified_before_success_is_promoted(tmp_path):
    source = 'a' * 40
    manifest = json.loads(runtime_for(source))
    manifest['numerics'] = {'fixture': True}
    runtime = launcher._canonical(manifest)
    preflight, repo_record, model = object(), {'source': 'fixture'}, object()
    state = {'runtime_bytes': runtime, 'preflight': preflight, 'repository': repo_record,
             'events': [], 'torch': object(), 'cache': tmp_path}
    empty_prefix = tmp_path / 'empty-prefix'
    empty_prefix.mkdir()
    context = {'repo': tmp_path, 'bootstrap': SimpleNamespace(_sources=lambda *args: {}),
               'extra_files': (), 'extra_sources': {}, 'prefix': empty_prefix}
    grant = SimpleNamespace(logical_limit_bytes=48 * 1024 ** 2, disk_limit_bytes=128 * 1024 ** 2)
    complete = SimpleNamespace(status='complete-local', record_count=128,
        prediction_sha256='1' * 64, archive_sha256='2' * 64)
    verify = mock.Mock(return_value=False)
    modules = {'looking_glass.agreement_phase_ledger': SimpleNamespace(claim_phase=lambda *a, **k: grant),
        'looking_glass.pretrained_number_supervisor': SimpleNamespace(run_discovery=lambda **kw: complete),
        'looking_glass.live_profile': SimpleNamespace(collect_repository_source=lambda *a: repo_record),
        'looking_glass.trusted_service_profile': SimpleNamespace(prepare_trusted_source=lambda *a: preflight,
                                                                 verify_trusted_cpu_model=verify)}
    with mock.patch.dict(sys.modules, modules), mock.patch.object(launcher, '_CONTEXT', context), \
            mock.patch.object(launcher, '_initialize_runtime', return_value=state), \
            mock.patch.object(launcher, '_load_trusted_model', return_value=SimpleNamespace(model=model)), \
            mock.patch.object(launcher, '_numeric_state', return_value={'fixture': True}):
        with pytest.raises(ValueError, match='model'):
            launcher._execute_verified(source_commit=source, runtime_bytes=runtime, plan_bytes=b'plan',
                plan_sha256='b' * 64, site_nonce=b'c' * 32, site_record_bytes=b'private',
                site_commitment='d' * 64, attempt_dir=tmp_path / 'attempt', cache_dir=tmp_path,
                site_packages=tmp_path, phase_ledger=tmp_path / 'study.db')
    verify.assert_called_once_with(preflight, model)
    terminal = json.loads((tmp_path / 'attempt.launcher-terminal.json').read_bytes())
    assert terminal['status'] == 'incomplete'


def test_cli_prepare_creates_public_runtime_and_reports_only_digest(tmp_path, capsys, safe_environment):
    source = 'a' * 40
    dest = tmp_path / 'runtime.json'
    runtime = runtime_for(source)
    context = {'repo': tmp_path / 'checkout'}
    with mock.patch.object(launcher, '_boot', return_value=context), \
            mock.patch.object(launcher, '_initialize_runtime', return_value={'runtime_bytes': runtime}):
        result = launcher.main(['--source-commit', source, '--site-packages', str(tmp_path),
                                'prepare', '--cache-dir', str(tmp_path), '--runtime-output', str(dest)])
    assert result == 0 and dest.read_bytes() == runtime
    assert json.loads(capsys.readouterr().out) == {
        'status': 'prepared', 'runtime_sha256': hashlib.sha256(runtime).hexdigest()}


def test_cli_checks_environment_before_any_source_or_git_access(tmp_path, capsys):
    with mock.patch.object(launcher, '_environment_policy', side_effect=ValueError('unsafe')), \
            mock.patch.object(launcher, '_boot') as boot:
        result = launcher.main(['--source-commit', 'a' * 40, '--site-packages', str(tmp_path),
                                'prepare', '--cache-dir', str(tmp_path),
                                '--runtime-output', str(tmp_path / 'runtime.json')])
    assert result == 1
    boot.assert_not_called()
    assert 'incomplete' in capsys.readouterr().err


def test_cli_run_failure_does_not_print_private_exception(tmp_path, capsys, safe_environment):
    source = 'a' * 40
    runtime = tmp_path / 'runtime.json'
    plan = tmp_path / 'plan.json'
    commitment = tmp_path / 'commitment.txt'
    for path, value in ((runtime, runtime_for(source)), (plan, b'plan'),
                        (commitment, b'0' * 64 + b'\n')):
        path.write_bytes(value)
    with mock.patch.object(launcher, '_boot', return_value={'repo': tmp_path / 'checkout'}), \
            mock.patch.object(launcher, 'run_published_discovery',
                              side_effect=ValueError('PRIVATE-SITE-DETAIL')):
        result = launcher.main(['--source-commit', source, '--site-packages', str(tmp_path),
            'run', '--cache-dir', str(tmp_path), '--runtime', str(runtime), '--runtime-url', 'runtime',
            '--plan', str(plan), '--plan-url', 'plan', '--commitment', str(commitment),
            '--commitment-url', 'commitment', '--custody-dir', str(tmp_path / 'private'),
            '--attempt-dir', str(tmp_path / 'attempt'), '--phase-ledger', str(tmp_path / 'ledger.db')])
    captured = capsys.readouterr()
    assert result == 1 and 'PRIVATE-SITE-DETAIL' not in captured.out + captured.err
    assert 'incomplete' in captured.err


def test_cli_subprocess_sanitizes_systemexit_with_private_payload(tmp_path):
    """A BaseException cannot print custodian-private details to the console."""
    source = 'a' * 40
    code = '''
import sys
from unittest import mock
from scripts import pretrained_number_cpu_launcher as launcher
with mock.patch.object(launcher, '_environment_policy', return_value='0' * 64), \\
     mock.patch.object(launcher, '_boot', return_value={'repo': sys.argv[1]}), \\
     mock.patch.object(launcher, '_initialize_runtime', side_effect=SystemExit('FABRICATED_PRIVATE_DETAIL')):
    code = launcher.main(['--source-commit', sys.argv[2], '--site-packages', sys.argv[1],
                          'prepare', '--cache-dir', sys.argv[1], '--runtime-output', sys.argv[3]])
sys.exit(code)
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path / 'checkout'),
                             source, str(tmp_path / 'runtime.json')], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'FABRICATED_PRIVATE_DETAIL' not in result.stdout + result.stderr
    assert 'incomplete' in result.stderr


def test_cli_parser_does_not_echo_unrecognized_private_arguments():
    code = 'from scripts import pretrained_number_cpu_launcher as m; m.main()'
    result = subprocess.run([sys.executable, '-B', '-c', code, '--source-commit', 'a' * 40,
        '--site-packages', 'site', 'plan', '--runtime', 'runtime', '--runtime-url', 'url',
        '--plan-dir', 'plan', '--unknown=FABRICATED_PRIVATE_PATH'], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'FABRICATED_PRIVATE_PATH' not in result.stdout + result.stderr
    help_result = subprocess.run([sys.executable, '-B', '-c', code, '--help'],
                                 capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0 and 'prepare' in help_result.stdout
