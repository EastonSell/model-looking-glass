"""Fabricated-only checks for the disabled prospective supervised entry."""

import importlib.util
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest


def _entry():
    path = Path(__file__).parents[1] / 'tools' / 'agreement_subject_cpu_v1.py'
    if not path.is_file():
        pytest.fail('subject CPU entry is not implemented')
    spec = importlib.util.spec_from_file_location('_subject_cpu_entry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __init__(self, answers, rows):
        self.encodings = {text: [token] for text, token in answers}
        for row in rows:
            self.encodings[row['text']] = list(row['token_ids'])
            for text, token in answers:
                self.encodings[row['text'] + text] = list(row['token_ids']) + [token]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return self.encodings[text]


def test_all_eight_continuations_and_full_input_are_checked():
    entry = _entry()
    rows = [{'text': 'fabricated one', 'token_ids': [1, 2]},
            {'text': 'fabricated two', 'token_ids': [3, 4]}]
    tokenizer = FakeTokenizer(entry.ANSWERS, rows)
    entry.verify_tokenizer(tokenizer, rows)
    tokenizer.encodings['fabricated two on'] = [3, 4, 999]
    with pytest.raises(ValueError, match='continuation'):
        entry.verify_tokenizer(tokenizer, rows)
    tokenizer = FakeTokenizer(entry.ANSWERS, rows)
    tokenizer.encodings['fabricated one'] = [1, 9]
    with pytest.raises(ValueError, match='tokenization'):
        entry.verify_tokenizer(tokenizer, rows)


def test_unpublished_gate_cannot_enter_a_study_or_load_model(monkeypatch):
    entry = _entry()
    monkeypatch.setattr(entry, 'RESERVATION_REVISION', None)
    with pytest.raises(RuntimeError, match='published case'):
        entry.main([])
    source = ("import runpy,sys; runpy.run_path(sys.argv[1],run_name='subject_entry_probe'); "
              "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    entry_path = Path(__file__).parents[1] / 'tools' / 'agreement_subject_cpu_v1.py'
    subprocess.run([sys.executable, '-I', '-S', '-B', '-c', source, str(entry_path)],
                   check=True, timeout=30)


def test_entry_validates_prelaunch_before_claiming_or_loading(tmp_path, monkeypatch):
    entry = _entry()
    monkeypatch.setattr(entry, 'RESERVATION_REVISION', '1' * 40)
    prelaunch = tmp_path / 'prelaunch.json'
    raw = b'{}'
    prelaunch.write_bytes(raw)
    def reject(_record):
        raise ValueError('fabricated prelaunch rejected')
    def forbidden(*_args, **_kwargs):
        raise AssertionError('owner was entered')
    monkeypatch.setattr(entry, 'validate_prelaunch', reject)
    monkeypatch.setattr(entry, 'run_owner', forbidden)
    names = ('latest-exposure', 'operator-exposure', 'operator-observation', 'natural-observation',
             'injection-observation', 'case-file',
             'direction-dir', 'reference-file', 'reference-manifest', 'cache-dir', 'output-dir',
             'study-root', 'venv', 'child-custody')
    argv = ['--prelaunch', str(prelaunch), '--prelaunch-sha256', hashlib.sha256(raw).hexdigest()]
    for name in names:
        argv.extend(['--' + name, str(tmp_path / name)])
    with pytest.raises(ValueError, match='prelaunch rejected'):
        entry.main(argv)


@pytest.mark.parametrize('purpose,validate_only', [('rehearsal', False), ('study', True)])
def test_rehearsal_and_study_prelaunch_cannot_be_interchanged(tmp_path, monkeypatch, purpose, validate_only):
    entry = _entry()
    prelaunch = tmp_path / 'prelaunch.json'
    raw = json.dumps({'purpose': purpose}).encode('ascii')
    prelaunch.write_bytes(raw)
    monkeypatch.setattr(entry, 'validate_prelaunch', lambda record: record)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('owner was entered')
    monkeypatch.setattr(entry, 'run_owner', forbidden)
    names = ('latest-exposure', 'operator-exposure', 'operator-observation', 'natural-observation',
             'injection-observation', 'case-file', 'direction-dir', 'reference-file', 'reference-manifest',
             'cache-dir', 'output-dir', 'study-root', 'venv', 'child-custody')
    argv = ['--prelaunch', str(prelaunch), '--prelaunch-sha256', hashlib.sha256(raw).hexdigest()]
    for name in names:
        argv.extend(['--' + name, str(tmp_path / name)])
    if validate_only:
        argv.append('--validate-only')
    with pytest.raises(ValueError, match='purpose'):
        entry.main(argv)


def test_binary_assets_are_length_and_hash_bound_before_use(tmp_path, monkeypatch):
    entry = _entry()
    vectors = {'u': struct.pack('<2d', 1.0, 0.0), 'v1': struct.pack('<2d', 0.0, 1.0),
               'v2': struct.pack('<2d', 0.5, 0.5)}
    manifest = b'{"synthetic":true}\n'
    reference = struct.pack('<3d', 0.25, -0.5, 0.0)
    reference_manifest = b'{"synthetic_reference":true}\n'
    digest = lambda value: hashlib.sha256(value).hexdigest()
    monkeypatch.setattr(entry, 'WIDTH', 2)
    monkeypatch.setattr(entry, 'VOCABULARY', 3)
    monkeypatch.setattr(entry, 'DIRECTION_SHA256', {name: digest(raw) for name, raw in vectors.items()})
    monkeypatch.setattr(entry, 'DIRECTION_MANIFEST_SHA256', digest(manifest))
    monkeypatch.setattr(entry, 'REFERENCE_SHA256', digest(reference))
    monkeypatch.setattr(entry, 'REFERENCE_MANIFEST_SHA256', digest(reference_manifest))
    direction_dir = tmp_path / 'directions'
    direction_dir.mkdir()
    for name, raw in vectors.items():
        (direction_dir / (name + '.f64le')).write_bytes(raw)
    (direction_dir / 'manifest.json').write_bytes(manifest)
    reference_path = tmp_path / 'reference.f64le'
    manifest_path = tmp_path / 'reference-manifest.json'
    reference_path.write_bytes(reference)
    manifest_path.write_bytes(reference_manifest)
    loaded = entry.verify_binary_assets(direction_dir, reference_path, manifest_path)
    assert loaded['u'] == (1.0, 0.0)
    reference_path.write_bytes(reference[:-1])
    with pytest.raises(ValueError, match='reference'):
        entry.verify_binary_assets(direction_dir, reference_path, manifest_path)
    reference_path.write_bytes(reference)
    (direction_dir / 'v1.f64le').write_bytes(vectors['v2'])
    with pytest.raises(ValueError, match='direction'):
        entry.verify_binary_assets(direction_dir, reference_path, manifest_path)


def test_exposure_requires_all_planned_rows_and_checks_other_prefixes():
    entry = _entry()
    cases = [{'case_id': f'fake-{i}', 'role': 'target' if i % 5 < 4 else 'context',
              'text': f'fabricated {i}', 'token_ids': [1000 + i, 1]}
             for i in range(80)]
    candidates = [token for _, token in entry.ANSWERS]
    records = [{'record_id': 'planned-' + row['case_id'], 'category': 'planned',
                'case_role': row['role'], 'public_text': row['text'],
                'token_ids': row['token_ids'], 'candidate_token_ids': candidates}
               for row in cases]
    entry.validate_exposure({'records': records}, cases)
    with pytest.raises(ValueError, match='planned'):
        entry.validate_exposure({'records': records[:-1]}, cases)
    changed = [dict(row) for row in records]
    changed[0]['candidate_token_ids'] = candidates[:-1]
    with pytest.raises(ValueError, match='planned'):
        entry.validate_exposure({'records': changed}, cases)
    with pytest.raises(ValueError, match='collision'):
        entry.validate_exposure({'records': records + [{'record_id': 'unrelated',
                                                        'token_ids': [1000]}]}, cases)
    with pytest.raises(ValueError, match='group'):
        entry.validate_exposure({'records': records + [{'record_id': 'unrelated', 'token_ids': [999],
                                                        'candidate_token_groups': {'year': [True]}}]}, cases)


def test_preservation_inventory_includes_grouped_continuations():
    entry = _entry()
    records = [{'token_ids': [9, 8], 'candidate_token_ids': [3], 'target_token_id': 4,
                'candidate_token_groups': {'year': [5, 6]}}]
    assert entry.known_sequences(records) == {(9, 8), (9, 8, 3), (9, 8, 4), (9, 8, 5), (9, 8, 6)}
    with pytest.raises(ValueError, match='group'):
        entry.known_sequences([{'token_ids': [9], 'candidate_token_groups': {'year': [False]}}])


def test_case_document_binds_exact_bytes_order_and_fixed_protocol(monkeypatch):
    entry = _entry()
    rows = [{'token_ids': [i + 1, 1]} for i in range(80)]
    ordered = json.dumps([row['token_ids'] for row in rows], separators=(',', ':')).encode('ascii')
    rows_sha = hashlib.sha256(ordered).hexdigest()
    document = {'schema': 'distilgpt2-agreement-subject-dependence-cases-v1',
                'protocol_revision': entry.PROTOCOL_REVISION, 'protocol_sha256': entry.PROTOCOL_SHA256,
                'reference_sha256': entry.REFERENCE_SHA256, 'model_id': entry.MODEL_ID,
                'model_revision': entry.MODEL_REVISION, 'tokenizer_sha256': entry.TOKENIZER_SHA256,
                'answer_ids': [item[1] for item in entry.ANSWERS], 'expected_forwards': 720,
                'ordered_rows_sha256': rows_sha, 'cases': rows}
    raw = json.dumps(document, sort_keys=True).encode('utf-8')
    monkeypatch.setattr(entry, 'CASE_SHA256', hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(entry, 'ROWS_SHA256', rows_sha)
    assert entry.verify_case_document(raw) == rows
    document['cases'] = list(reversed(rows))
    changed = json.dumps(document, sort_keys=True).encode('utf-8')
    monkeypatch.setattr(entry, 'CASE_SHA256', hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match='rows'):
        entry.verify_case_document(changed)
    document['cases'] = rows
    document['protocol_sha256'] = '0' * 64
    changed = json.dumps(document, sort_keys=True).encode('utf-8')
    monkeypatch.setattr(entry, 'CASE_SHA256', hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match='fixed'):
        entry.verify_case_document(changed)


def test_prelaunch_rejects_unpublished_or_mismatched_bindings(monkeypatch):
    entry = _entry()
    monkeypatch.setattr(entry, 'RESERVATION_REVISION', '1' * 40)
    monkeypatch.setattr(entry, 'RESERVATION_SHA256', '2' * 64)
    monkeypatch.setattr(entry, 'RESERVATION_VERSION', 'fabricated-v1')
    record = {
        'schema': 'distilgpt2-agreement-subject-prelaunch-v1', 'purpose': 'study',
        'owner': {'source_revision': 'a' * 40, 'source_tree': 'b' * 40,
                  'executor_sha256': 'c' * 64, 'entry_script_sha256': 'd' * 64},
        'protocol': {'revision': entry.PROTOCOL_REVISION, 'path': entry.PROTOCOL_PATH,
                     'sha256': entry.PROTOCOL_SHA256},
        'cases': {'revision': entry.CASE_REVISION, 'path': entry.CASE_PATH,
                  'sha256': entry.CASE_SHA256, 'ordered_rows_sha256': entry.ROWS_SHA256},
        'directions': {'revision': entry.DIRECTION_REVISION, 'manifest_sha256': entry.DIRECTION_MANIFEST_SHA256,
                       **{name + '_sha256': digest for name, digest in entry.DIRECTION_SHA256.items()}},
        'reference': {'sha256': entry.REFERENCE_SHA256, 'manifest_sha256': entry.REFERENCE_MANIFEST_SHA256},
        'checkpoint': {'model_id': entry.MODEL_ID, 'revision': entry.MODEL_REVISION,
                       'weights_sha256': entry.WEIGHTS_SHA256, 'tokenizer_sha256': entry.TOKENIZER_SHA256},
        'limits': dict(entry.LIMITS),
        'runtime': {**entry.VERSIONS, 'cpu_threads': 1, 'cpu_interop_threads': 1,
                    'claim_cuda_visibility': '', 'environment': dict(entry.REQUIRED_ENV)},
        'exposure': {'reservation_revision': '1' * 40, 'reservation_sha256': '2' * 64,
                     'reservation_version': 'fabricated-v1', 'latest_revision': '3' * 40,
                     'latest_sha256': '4' * 64, 'latest_version': 'fabricated-v2',
                     'latest_module': 'looking_glass/pretrained_exposures_v99.py',
                     'operator_revision': '7' * 40, 'operator_sha256': '8' * 64,
                     'operator_version': 'fabricated-v3', 'operator_observation_sha256': '9' * 64,
                     'operator_refreshed_utc': '2026-10-29T10:00:00Z'},
        'custody': {'natural': {'status': 'clear', 'case_sha256': entry.CASE_SHA256,
                                'support_receipt_sha256': entry.NATURAL_RECEIPT_SHA256,
                                'scope': entry.NATURAL_SCOPE,
                                'observation_sha256': '5' * 64, 'refreshed_utc': '2026-10-29T10:00:00Z'},
                    'injection': {'status': 'no_protected_rows', 'direct_target_conflict': 'none reported',
                                  'observation_sha256': '6' * 64, 'refreshed_utc': '2026-10-29T10:00:00Z'}},
        'independent_reader_status': 'reviewed', 'recorded_utc': '2026-10-29T10:00:00Z',
        'scope': 'fabricated test',
    }
    assert entry.validate_prelaunch(record) is record
    record['independent_reader_status'] = 'under_review'
    assert entry.validate_prelaunch(record) is record
    record['independent_reader_status'] = 'reviewed'
    record['cases']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='fixed'):
        entry.validate_prelaunch(record)
    record['cases']['sha256'] = entry.CASE_SHA256
    monkeypatch.setattr(entry, 'RESERVATION_REVISION', None)
    with pytest.raises(RuntimeError, match='reservation'):
        entry.validate_prelaunch(record)


def test_reviewed_helper_is_verified_before_its_bytes_execute(tmp_path, monkeypatch):
    entry = _entry()
    tools_dir = tmp_path / 'tools'
    tools_dir.mkdir()
    helper = tools_dir / 'agreement_direction_cpu_v1.py'
    raw = b'VALUE = 17\n'
    helper.write_bytes(raw)
    monkeypatch.setattr(entry, 'DIRECTION_ENTRY_SHA256', hashlib.sha256(raw).hexdigest())
    assert entry.load_reviewed_helper(tmp_path).VALUE == 17
    helper.write_bytes(b"raise AssertionError('tampered helper executed')\n")
    with pytest.raises(ValueError, match='helper'):
        entry.load_reviewed_helper(tmp_path)


def test_source_binding_compares_captured_worker_and_committed_entry(tmp_path, monkeypatch):
    entry = _entry()
    monkeypatch.setattr(entry, 'RESERVATION_REVISION', '4' * 40)
    source = b'fabricated worker source\n'
    entry_raw = b'fabricated entry source\n'
    entry_path = tmp_path / 'entry.py'
    entry_path.write_bytes(entry_raw)
    captured = {'looking_glass/agreement_subject_executor_v1.py': source}
    owner = {'source_revision': '1' * 40, 'source_tree': '2' * 40,
             'executor_sha256': hashlib.sha256(source).hexdigest(),
             'entry_script_sha256': hashlib.sha256(entry_raw).hexdigest()}
    record = {'owner': owner, 'exposure': {'latest_revision': '3' * 40}}
    class FakeHelper:
        committed = entry_raw
        def source_inventory(self, _root, revision, tree):
            assert revision == owner['source_revision'] and tree == owner['source_tree']
            return dict(captured)
        def _git(self, _root, *args):
            return self.committed if args[0] == 'show' else b''
    helper = FakeHelper()
    entry.verify_source(tmp_path, record, captured, helper, entry_path)
    helper.committed = b'different committed source\n'
    with pytest.raises(ValueError, match='entry'):
        entry.verify_source(tmp_path, record, captured, helper, entry_path)


def test_capture_checks_worker_before_cpu_claim(tmp_path, monkeypatch):
    entry = _entry()
    worker = b'fabricated worker\n'
    entry_raw = b'fabricated entry\n'
    entry_path = tmp_path / 'entry.py'
    entry_path.write_bytes(entry_raw)
    captured = {'looking_glass/agreement_subject_executor_v1.py': worker}
    owner = {'executor_sha256': hashlib.sha256(worker).hexdigest(),
             'entry_script_sha256': hashlib.sha256(entry_raw).hexdigest()}
    record = {'owner': owner, 'runtime': {'claim_cuda_visibility': ''}}
    class FakeHelper:
        claimed = False
        def _early_precheck(self, _record): pass
        def _capture_local_sources(self, _root, _record): return dict(captured)
        def _SourceFinder(self, _root, _sources): return object()
        def _claim_cpu(self, _root):
            self.claimed = True
            return 'session', 'lease'
        def _install_venv(self, _venv): pass
    helper = FakeHelper()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    record['owner']['executor_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='worker'):
        entry.capture_and_claim(tmp_path, record, tmp_path, tmp_path, helper, entry_path)
    assert helper.claimed is False
    record['owner']['executor_sha256'] = hashlib.sha256(worker).hexdigest()
    session, lease, sources = entry.capture_and_claim(tmp_path, record, tmp_path, tmp_path, helper, entry_path)
    assert (session, lease, sources) == ('session', 'lease', captured)
    assert helper.claimed is True


def test_latest_inventory_binds_committed_module_and_preserves_reservation(tmp_path, monkeypatch):
    entry = _entry()
    cases = [{'case_id': f'fake-{i}', 'role': 'target' if i % 5 < 4 else 'context',
              'text': f'fabricated {i}', 'token_ids': [1000 + i, 1]} for i in range(80)]
    candidates = [token for _, token in entry.ANSWERS]
    planned = [{'record_id': 'planned-' + row['case_id'], 'category': 'planned',
                'case_role': row['role'], 'public_text': row['text'], 'token_ids': row['token_ids'],
                'candidate_token_ids': candidates} for row in cases]
    previous = {'version': 'fake-v27', 'records': planned}
    latest = {'version': 'fake-v28', 'records': planned + [{'record_id': 'other', 'token_ids': [999]}]}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=True, allow_nan=False).encode('ascii')
    prev_sha = hashlib.sha256(canonical(previous)).hexdigest()
    latest_raw = canonical(latest)
    latest_sha = hashlib.sha256(latest_raw).hexdigest()
    path = tmp_path / 'latest.json'
    path.write_bytes(latest_raw)
    monkeypatch.setattr(entry, 'RESERVATION_REVISION', '1' * 40)
    monkeypatch.setattr(entry, 'RESERVATION_SHA256', prev_sha)
    monkeypatch.setattr(entry, 'RESERVATION_VERSION', 'fake-v27')
    monkeypatch.setattr(entry, 'RESERVATION_MODULE', 'looking_glass/pretrained_exposures_v27.py')
    modules = {
        'looking_glass.pretrained_exposures_v27': type('Previous', (), {
            'INVENTORY_VERSION': 'fake-v27', 'INVENTORY_SHA256': prev_sha,
            'inventory_payload': staticmethod(lambda: previous)}),
        'looking_glass.pretrained_exposures_v28': type('Latest', (), {
            'INVENTORY_VERSION': 'fake-v28', 'INVENTORY_SHA256': latest_sha,
            'inventory_payload': staticmethod(lambda: latest)}),
    }
    monkeypatch.setattr(entry.importlib, 'import_module', lambda name: modules[name])
    sources = {'looking_glass/pretrained_exposures_v27.py': b'previous',
               'looking_glass/pretrained_exposures_v28.py': b'latest'}
    class FakeHelper:
        def _git(self, _root, *args):
            if args[0] == 'show':
                return sources[args[1].split(':', 1)[1]]
            return b''
        def _known_sequences(self, records):
            return {tuple(row['token_ids']) for row in records}
    record = {'exposure': {'latest_revision': '2' * 40, 'latest_sha256': latest_sha,
                           'latest_version': 'fake-v28',
                           'latest_module': 'looking_glass/pretrained_exposures_v28.py'}}
    entry.verify_latest_inventory(tmp_path, record, path, cases, FakeHelper(), sources)
    path.write_bytes(latest_raw + b' ')
    with pytest.raises(ValueError, match='inventory'):
        entry.verify_latest_inventory(tmp_path, record, path, cases, FakeHelper(), sources)


def test_operator_inventory_is_separate_and_preserves_reconstructed_rows(tmp_path):
    entry = _entry()
    cases = [{'case_id': f'fake-{i}', 'role': 'target' if i % 5 < 4 else 'context',
              'text': f'fabricated {i}', 'token_ids': [1000 + i, 1]} for i in range(80)]
    candidates = [token for _, token in entry.ANSWERS]
    planned = [{'record_id': 'planned-' + row['case_id'], 'category': 'planned',
                'case_role': row['role'], 'public_text': row['text'], 'token_ids': row['token_ids'],
                'candidate_token_ids': candidates} for row in cases]
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=True, allow_nan=False).encode('ascii')
    reconstructed = canonical({'version': 'fake-v27', 'records': planned})
    operator = canonical({'version': 'fake-v28', 'records': planned + [{'record_id': 'other', 'token_ids': [999]}]})
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    stamp = '2026-10-29T10:00:00Z'
    observation = canonical({'schema': 'agreement-subject-exposure-refresh-v1', 'status': 'clear',
                             'case_sha256': entry.CASE_SHA256, 'reconstructed_sha256': sha(reconstructed),
                             'operator_revision': '5' * 40, 'operator_version': 'fake-v28',
                             'operator_sha256': sha(operator), 'refreshed_utc': stamp})
    operator_path, observation_path = tmp_path / 'operator.json', tmp_path / 'observation.json'
    operator_path.write_bytes(operator)
    observation_path.write_bytes(observation)
    record = {'exposure': {'latest_sha256': sha(reconstructed), 'operator_revision': '5' * 40,
                           'operator_version': 'fake-v28', 'operator_sha256': sha(operator),
                           'operator_observation_sha256': sha(observation), 'operator_refreshed_utc': stamp}}
    assert entry.verify_operator_inventory(record, operator_path, observation_path, reconstructed, cases) == (operator, observation)
    operator_path.write_bytes(operator + b' ')
    with pytest.raises(ValueError, match='operator'):
        entry.verify_operator_inventory(record, operator_path, observation_path, reconstructed, cases)


def test_refreshed_custody_observations_are_bound_to_the_fixed_case(tmp_path, monkeypatch):
    entry = _entry()
    receipt = json.dumps({'schema': 'mlg-agreement-subject-public-support-comparison-v1',
                          'status': 'clear', 'proposal_file_sha256': entry.CASE_SHA256,
                          'proposal_ordered_rows_sha256': entry.ROWS_SHA256}).encode('ascii')
    receipt_sha = hashlib.sha256(receipt).hexdigest()
    monkeypatch.setattr(entry, 'NATURAL_RECEIPT_SHA256', receipt_sha)
    utc = '2026-10-29T10:00:00Z'
    natural = {'schema': 'agreement-subject-natural-refresh-v1', 'status': 'clear',
               'case_sha256': entry.CASE_SHA256, 'support_receipt_sha256': receipt_sha,
               'scope': entry.NATURAL_SCOPE, 'successor_status': 'undrawn',
               'successor_support_sha256': entry.SUCCESSOR_SUPPORT_SHA256, 'refreshed_utc': utc}
    injection = {'schema': 'agreement-subject-injection-refresh-v1', 'status': 'no_protected_rows',
                 'direct_target_conflict': 'none reported', 'site': 'transformer.h.3.attn',
                 'refreshed_utc': utc}
    paths = []
    raw_files = [receipt, json.dumps(natural).encode('ascii'), json.dumps(injection).encode('ascii')]
    for index, raw in enumerate(raw_files):
        path = tmp_path / f'file-{index}.json'
        path.write_bytes(raw)
        paths.append(path)
    record = {'custody': {'natural': {'observation_sha256': hashlib.sha256(raw_files[1]).hexdigest(),
                                      'refreshed_utc': utc},
                          'injection': {'observation_sha256': hashlib.sha256(raw_files[2]).hexdigest(),
                                        'refreshed_utc': utc}}}
    assert entry.verify_custody(record, *paths) == tuple(raw_files)
    natural['successor_status'] = 'drawn_covered_by_public_successor_inputs'
    updated = json.dumps(natural).encode('ascii')
    paths[1].write_bytes(updated)
    record['custody']['natural']['observation_sha256'] = hashlib.sha256(updated).hexdigest()
    assert entry.verify_custody(record, *paths)[1] == updated
    natural['successor_status'] = 'unknown'
    updated = json.dumps(natural).encode('ascii')
    paths[1].write_bytes(updated)
    record['custody']['natural']['observation_sha256'] = hashlib.sha256(updated).hexdigest()
    with pytest.raises(ValueError, match='successor'):
        entry.verify_custody(record, *paths)
    natural['successor_status'] = 'undrawn'
    updated = json.dumps(natural).encode('ascii')
    paths[1].write_bytes(updated)
    record['custody']['natural']['observation_sha256'] = hashlib.sha256(updated).hexdigest()
    injection['site'] = 'transformer.h.1.attn'
    changed = json.dumps(injection).encode('ascii')
    paths[2].write_bytes(changed)
    record['custody']['injection']['observation_sha256'] = hashlib.sha256(changed).hexdigest()
    with pytest.raises(ValueError, match='fixed'):
        entry.verify_custody(record, *paths)


def test_published_artifact_must_match_checkout_and_recorded_commit(tmp_path):
    entry = _entry()
    path = tmp_path / 'case.json'
    raw = b'fabricated public case bytes'
    path.write_bytes(raw)
    class FakeHelper:
        committed = raw
        def _ordinary(self, _path, _root): pass
        def _git(self, _root, *args): return self.committed
    helper = FakeHelper()
    assert entry.verify_published_file(tmp_path, helper, '1' * 40, 'case.json',
                                       hashlib.sha256(raw).hexdigest(), 1024) == raw
    helper.committed = b'different committed bytes'
    with pytest.raises(ValueError, match='published'):
        entry.verify_published_file(tmp_path, helper, '1' * 40, 'case.json',
                                    hashlib.sha256(raw).hexdigest(), 1024)


def test_one_shot_claim_is_case_bound_across_new_output_paths(tmp_path):
    entry = _entry()
    class FakeHelper:
        def _write(self, path, value):
            with path.open('x', encoding='utf-8') as stream:
                json.dump(value, stream)
    helper = FakeHelper()
    entry.claim_attempt(tmp_path, tmp_path / 'first', '1' * 64, helper)
    with pytest.raises(FileExistsError):
        entry.claim_attempt(tmp_path, tmp_path / 'second', '2' * 64, helper)
