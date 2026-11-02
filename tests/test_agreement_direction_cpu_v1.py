"""No-model admission and source-bound entry tests."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest


PATH = Path(__file__).parents[1] / 'tools/agreement_direction_cpu_v1.py'


@pytest.fixture
def entry():
    spec = importlib.util.spec_from_file_location('_direction_entry_test', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prelaunch(entry):
    return {
        'schema': 'distilgpt2-agreement-direction-prelaunch-v1',
        'owner': {'source_revision': '1' * 40, 'source_tree': '2' * 40,
                  'executor_sha256': '3' * 64, 'entry_script_sha256': '4' * 64},
        'protocol': {'revision': entry.PROTOCOL_COMMIT, 'path': entry.PROTOCOL_PATH,
                     'sha256': entry.PROTOCOL_SHA},
        'cases': {'revision': entry.CASE_COMMIT, 'path': entry.CASE_PATH,
                  'sha256': entry.CASE_SHA, 'ordered_rows_sha256': entry.ROWS_SHA},
        'directions': {'revision': entry.DIRECTION_COMMIT, 'manifest_sha256': entry.MANIFEST_SHA,
                       'u_sha256': entry.DIRECTION_SHA['u'], 'v1_sha256': entry.DIRECTION_SHA['v1'],
                       'v2_sha256': entry.DIRECTION_SHA['v2']},
        'exposure': {'reservation_revision': entry.RESERVATION_COMMIT,
                     'reservation_version': entry.RESERVATION_VERSION,
                     'reservation_sha256': entry.RESERVATION_SHA,
                     'latest_revision': entry.LATEST_COMMIT,
                     'latest_version': entry.LATEST_VERSION,
                     'latest_sha256': entry.LATEST_SHA},
        'custody': {'natural': {'case_sha256': entry.CASE_SHA, 'comparison_receipt_sha256': entry.NATURAL_RECEIPT_SHA,
                                'comparison_parent_observation_sha256': entry.NATURAL_PARENT_SHA,
                                'status': 'clear', 'scope': 'bidirectional full-row-prefix against existing sealed split'},
                    'injection': {'status': 'no_protected_rows', 'direct_target_conflict': 'none reported',
                                  'scope': 'fixed L3 final/subject and L1 final attention outputs'}},
        'checkpoint': {'model_id': entry.MODEL_ID, 'revision': entry.MODEL_COMMIT,
                       'weights_sha256': entry.WEIGHTS_SHA, 'tokenizer_sha256': entry.TOKENIZER_SHA},
        'runtime': {'python': '3.11.15', 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
                    'tokenizers': '0.22.2', 'numpy': '2.4.6', 'huggingface_hub': '0.36.2',
                    'cpu_threads': 1, 'cpu_interop_threads': 1, 'claim_cuda_visibility': '',
                    'environment': entry.REQUIRED_ENV},
        'limits': {'expected_forwards': 784, 'logical_raw_bytes': 512 * 1024**2,
                   'physical_attempt_bytes': 1024**3, 'minimum_free_disk_bytes': 2 * 1024**3,
                   'automatic_retry': False},
        'independent_reader_status': 'under development at launch', 'recorded_utc': '2026-10-29T00:00:00Z',
        'scope': 'operator observation, not authenticated execution or exposure history',
    }


def test_prelaunch_rejects_inconsistent_fixed_inputs_custody_and_limits(entry):
    record = prelaunch(entry)
    entry.validate_prelaunch(record)
    record['custody']['natural']['status'] = 'unavailable'
    with pytest.raises(ValueError):
        entry.validate_prelaunch(record)
    record = prelaunch(entry)
    record['runtime']['environment'] = {**entry.REQUIRED_ENV, 'CUDA_VISIBLE_DEVICES': ''}
    with pytest.raises(ValueError):
        entry.validate_prelaunch(record)
    record = prelaunch(entry)
    record['limits']['expected_forwards'] = 783
    with pytest.raises(ValueError):
        entry.validate_prelaunch(record)
    record = prelaunch(entry)
    record['cases']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        entry.validate_prelaunch(record)
    record = prelaunch(entry)
    record['exposure'].update(latest_revision='9' * 40, latest_version='made-up-v99', latest_sha256='a' * 64)
    with pytest.raises(ValueError):
        entry.validate_prelaunch(record)


def test_latest_exposure_requires_all_planned_rows_and_rejects_new_prefixes(entry):
    cases = [{'case_id': f'case-{i}', 'role': 'target' if i < 32 else 'context',
              'text': f'example {i}', 'token_ids': [1000 + i, 2000]} for i in range(40)]
    payload = {'records': [{'record_id': 'planned-' + row['case_id'], 'category': 'planned',
                            'case_role': row['role'], 'public_text': row['text'],
                            'token_ids': row['token_ids'], 'candidate_token_ids': [318, 389, 373, 547]}
                           for row in cases]}
    entry.validate_exposure(payload, cases)
    payload['records'].pop()
    with pytest.raises(ValueError):
        entry.validate_exposure(payload, cases)
    payload['records'].append({'record_id': 'planned-' + cases[-1]['case_id'], 'category': 'planned',
                               'case_role': cases[-1]['role'], 'public_text': cases[-1]['text'],
                               'token_ids': cases[-1]['token_ids'], 'candidate_token_ids': [318, 389, 373, 547]})
    payload['records'].append({'record_id': 'other', 'token_ids': [1000]})
    with pytest.raises(ValueError):
        entry.validate_exposure(payload, cases)


def test_inventory_status_change_may_preserve_but_not_drop_known_rows(entry):
    previous = [{'record_id': 'planned-old', 'token_ids': [7, 8], 'candidate_token_ids': [318, 389]}]
    latest = [{'record_id': 'observed-old', 'token_ids': [7, 8], 'candidate_token_ids': [318, 389]}]
    assert entry._known_sequences(previous) <= entry._known_sequences(latest)
    latest[0]['candidate_token_ids'].pop()
    assert not entry._known_sequences(previous) <= entry._known_sequences(latest)


def test_bounded_strict_json_rejects_duplicate_keys_nonfinite_and_oversize(entry, tmp_path):
    path = tmp_path / 'bad.json'
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b' ' * 33):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            entry.read_json(path, 32)


def test_natural_comparison_binds_ordered_cases_receipt_and_parent(entry):
    root = Path(__file__).parents[1]
    cases = entry.read_json(root / entry.CASE_PATH, 2 * 1024 * 1024)['cases']
    receipt = entry.read_json(root / entry.NATURAL_RECEIPT_PATH, 256 * 1024)
    bundle = entry.read_json(root / entry.NATURAL_INPUT_PATH, 256 * 1024)
    parent = {'schema': 'mlg-natural-comparison-parent-observation-v1', 'exit_code': 0,
              'reaped': True, 'receipt_verified': True, 'receipt_sha256': entry.NATURAL_RECEIPT_SHA,
              'adapted_public_file_sha256': entry.NATURAL_INPUT_SHA,
              'original_public_file_sha256': entry.CASE_SHA, 'ordered_public_rows_sha256': entry.ROWS_SHA}
    entry.validate_natural_records(receipt, bundle, parent, cases)
    cases[0]['token_ids'][0] += 1
    with pytest.raises(ValueError):
        entry.validate_natural_records(receipt, bundle, parent, cases)


def test_project_loader_executes_captured_bytes_and_rejects_unlisted_source(entry, tmp_path):
    source = tmp_path / 'looking_glass'
    source.mkdir()
    (source / 'sample.py').write_text("raise RuntimeError('unverified shadow')\n", encoding='utf-8')
    finder = entry._SourceFinder(tmp_path, {'looking_glass/__init__.py': b'',
                                           'looking_glass/sample.py': b'VALUE = 17\n'})
    spec = finder.find_spec('looking_glass.sample')
    module = types.ModuleType('looking_glass.sample')
    spec.loader.exec_module(module)
    assert module.VALUE == 17
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec('looking_glass.unlisted')


def test_git_source_inventory_ignores_inherited_git_dir_and_checks_blob_bytes(entry, tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)
    git('init', '-q')
    git('config', 'user.email', 'fixture@example.invalid')
    git('config', 'user.name', 'Fixture')
    (root / 'looking_glass').mkdir()
    (root / 'tools').mkdir()
    (root / 'looking_glass/__init__.py').write_bytes(b'')
    (root / 'looking_glass/sample.py').write_bytes(b'VALUE = 1\n')
    (root / 'tools/agreement_direction_cpu_v1.py').write_bytes(b'# fixture\n')
    git('add', '.')
    git('commit', '-q', '-m', 'fixture')
    revision = git('rev-parse', 'HEAD').decode().strip()
    tree = git('show', '-s', '--format=%T', 'HEAD').decode().strip()
    for name in ('PROTOCOL_COMMIT', 'CASE_COMMIT', 'DIRECTION_COMMIT', 'RESERVATION_COMMIT'):
        monkeypatch.setattr(entry, name, revision)
    monkeypatch.setenv('GIT_DIR', str(tmp_path / 'unrelated'))
    sources = entry.source_inventory(root, revision, tree)
    assert sources['looking_glass/sample.py'] == b'VALUE = 1\n'
    monkeypatch.delenv('GIT_DIR')
    git('update-index', '--assume-unchanged', 'looking_glass/sample.py')
    (root / 'looking_glass/sample.py').write_bytes(b'VALUE = 2\n')
    assert not git('status', '--porcelain')
    with pytest.raises(ValueError, match='committed blob'):
        entry.source_inventory(root, revision, tree)


def test_admission_write_failure_records_incomplete_before_imports(entry, tmp_path, monkeypatch):
    record = prelaunch(entry)
    raw = json.dumps(record).encode()
    prelaunch_path = tmp_path / 'prelaunch.json'
    prelaunch_path.write_bytes(raw)
    cache = tmp_path / 'cache'
    cache.mkdir()
    output = tmp_path / 'attempt'
    class Session:
        run_id = 'a' * 32
        def check(self, lease):
            pass
    monkeypatch.setattr(entry, '_bootstrap', lambda *args: (Session(), object(), {}))
    monkeypatch.setattr(entry, '_runtime_precheck', lambda value: {})
    monkeypatch.setattr(entry, '_checked_source', lambda *args: {})
    monkeypatch.setattr(entry, '_fixed_inputs', lambda *args: ([], {}, b'{}'))
    monkeypatch.setattr(entry, 'validate_natural_comparison', lambda *args: (b'{}', b'{}', {}))
    monkeypatch.setattr(entry, 'bind_latest_committed_inventory', lambda *args: None)
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    original_write = entry._write
    def fault(path, value):
        if Path(path).name == 'admission.json':
            raise OSError('simulated failed admission write')
        return original_write(path, value)
    monkeypatch.setattr(entry, '_write', fault)
    args = ['--prelaunch', str(prelaunch_path), '--prelaunch-sha256', hashlib.sha256(raw).hexdigest(),
            '--latest-exposure', str(tmp_path / 'latest.json'), '--natural-parent-observation', str(tmp_path / 'parent.json'),
            '--case-file', str(tmp_path / 'cases.json'), '--direction-dir', str(tmp_path), '--cache-dir', str(cache),
            '--study-root', str(tmp_path), '--venv', str(tmp_path), '--child-custody', str(tmp_path / 'child-custody.json'),
            '--output-dir', str(output)]
    with pytest.raises(OSError, match='admission write'):
        entry.main(args)
    failure = json.loads((output / 'entry-failure.json').read_bytes())
    assert failure['status'] == 'failed' and failure['error_type'] == 'OSError'
    assert not (output / 'attempt').exists()


def test_supervised_claim_precedes_visibility_transition_and_dependencies(entry, tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    monkeypatch.setattr(entry, '_early_precheck', lambda record: None)
    monkeypatch.setattr(entry, '_capture_local_sources', lambda root, record: {})
    events = []
    def claim(root):
        events.append(('claim', entry.os.environ.get('CUDA_VISIBLE_DEVICES')))
        return object(), object()
    def dependencies(venv):
        events.append(('dependencies', entry.os.environ.get('CUDA_VISIBLE_DEVICES')))
    monkeypatch.setattr(entry, '_claim_cpu', claim)
    monkeypatch.setattr(entry, '_install_venv', dependencies)
    entry._bootstrap(tmp_path, prelaunch(entry), tmp_path, tmp_path)
    assert events == [('claim', ''), ('dependencies', '-1')]


def test_failed_supervisor_claim_does_not_add_dependencies(entry, tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    monkeypatch.setattr(entry, '_early_precheck', lambda record: None)
    monkeypatch.setattr(entry, '_capture_local_sources', lambda root, record: {})
    monkeypatch.setattr(entry, '_claim_cpu', lambda root: (_ for _ in ()).throw(RuntimeError('no grant')))
    monkeypatch.setattr(entry, '_install_venv', lambda path: pytest.fail('dependencies added before grant'))
    with pytest.raises(RuntimeError, match='no grant'):
        entry._bootstrap(tmp_path, prelaunch(entry), tmp_path, tmp_path)
    assert entry.os.environ['CUDA_VISIBLE_DEVICES'] == ''
