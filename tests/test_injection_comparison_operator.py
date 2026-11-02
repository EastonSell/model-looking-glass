"""Fabricated public bindings for the source-first comparison operator."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root = Path(__file__).resolve().parents[1]
operator = load('_comparison_operator_test', root / 'tools/injection_comparison_operator.py')
comparison = load('_comparison_operator_comparison', root / 'tools/injection_split_comparison.py')
from looking_glass import injection_split_custody as custody


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def public_case():
    freeze = canonical({'schema': 'mlg-number-prediction-freezes-v1',
        'checkpoint_revision': operator._CHECKPOINT,
        'protocol_sha256': operator._PROTOCOL,
        'tracks': {'injection': {'source_commit': operator._SOURCE,
            'runtime_sha256': operator._RUNTIME, 'prediction_sha256': operator._PREDICTION}}})
    assert sha(freeze) == comparison._INJECTION_FREEZE
    operation = {key: 'f' * 64 for key in custody._OP_V3_KEYS}
    operation.update({'schema': 'mlg-injection-heldout-operational-binding-v3',
        'protocol_sha256': operator._PROTOCOL, 'checkpoint_revision': operator._CHECKPOINT,
        'discovery_source_commit': operator._SOURCE, 'inventory_source_commit': 'a' * 40,
        'prediction_freeze_sha256': comparison._INJECTION_FREEZE,
        'runtime_sha256': operator._RUNTIME, 'prediction_sha256': operator._PREDICTION,
        'plan_sha256': operator._PLAN, 'site_commitment': comparison._SITE_COMMITMENT,
        'custody_source_sha256': comparison._SOURCE_SPECS[-1][1],
        'comparator_source_sha256': comparison._SOURCE_SPECS[-1][1]})
    operation_raw = canonical(operation)
    commitment = canonical({'schema': 'mlg-injection-heldout-public-commitment-v1',
        'prediction_freeze_sha256': comparison._INJECTION_FREEZE,
        'exposure_inventory_sha256': operation['inventory_sha256'],
        'heldout_commitment': 'e' * 64,
        'custody_source_sha256': operation['custody_source_sha256'],
        'operational_binding_sha256': sha(operation_raw)})
    manifest = canonical({'schema': 'mlg-injection-comparison-source-manifest-v1',
        'sources': {name: {'sha256': digest, 'url': url}
                    for name, digest, url in operator._PINNED_SOURCES}})
    url = comparison._RAW + 'a' * 40 + '/results/comparison-sources.json'
    operation_url = comparison._RAW + 'd' * 40 + '/results/operation.json'
    published = {comparison._PUBLIC_URLS['injection_freeze']: freeze,
                 comparison._RAW + 'b' * 40 + '/results/commitment.json': commitment,
                 url: manifest, operation_url: operation_raw}
    args = {'custody': custody, 'comparison': comparison, 'operation': operation_raw,
        'operation_sha256': sha(operation_raw), 'operation_url': operation_url,
        'freeze': freeze, 'commitment': commitment,
        'commitment_sha256': sha(commitment), 'commitment_url': list(published)[1],
        'manifest': manifest, 'manifest_sha256': sha(manifest), 'manifest_url': url,
        'reader': lambda address: published[address]}
    return SimpleNamespace(args=args, operation=operation, published=published)


def test_fixed_public_operation_and_commitment_are_checked_before_private_work(public_case):
    result = operator.validate_public_binding(**public_case.args)
    assert result['operation_sha256'] == public_case.args['operation_sha256']
    assert result['manifest_sha256'] == public_case.args['manifest_sha256']
    assert result['injection_commitment_sha256'] == public_case.args['commitment_sha256']


@pytest.mark.parametrize('field', ['prediction_freeze_sha256', 'discovery_source_commit',
    'runtime_sha256', 'prediction_sha256', 'plan_sha256', 'site_commitment',
    'custody_source_sha256', 'comparator_source_sha256'])
def test_self_consistent_wrong_operation_cannot_pass_fixed_study_binding(public_case, field):
    operation = dict(public_case.operation)
    operation[field] = ('1' * (40 if field == 'discovery_source_commit' else 64))
    raw = canonical(operation)
    commitment = json.loads(public_case.args['commitment'])
    commitment['operational_binding_sha256'] = sha(raw)
    if field == 'custody_source_sha256':
        commitment['custody_source_sha256'] = operation[field]
    commitment_raw = canonical(commitment)
    args = dict(public_case.args, operation=raw, operation_sha256=sha(raw),
                commitment=commitment_raw, commitment_sha256=sha(commitment_raw))
    public_case.published[args['commitment_url']] = commitment_raw
    public_case.published[args['operation_url']] = raw
    with pytest.raises(operator.OperatorError):
        operator.validate_public_binding(**args)


def test_manifest_must_match_independent_digest_and_published_bytes(public_case):
    args = dict(public_case.args, manifest_sha256='0' * 64)
    with pytest.raises(operator.OperatorError):
        operator.validate_public_binding(**args)
    public_case.published[args['manifest_url']] = b'different'
    with pytest.raises(operator.OperatorError):
        operator.validate_public_binding(**public_case.args)


def test_operation_publication_is_read_back_even_when_commitment_binds_it(public_case):
    public_case.published[public_case.args['operation_url']] = b'different'
    with pytest.raises(operator.OperatorError):
        operator.validate_public_binding(**public_case.args)


def test_self_consistent_manifest_with_changed_required_source_is_rejected(public_case):
    manifest = json.loads(public_case.args['manifest'])
    key = operator._PINNED_SOURCES[0][0]
    manifest['sources'][key]['sha256'] = '0' * 64
    raw = canonical(manifest)
    args = dict(public_case.args, manifest=raw, manifest_sha256=sha(raw))
    public_case.published[args['manifest_url']] = raw
    with pytest.raises(operator.OperatorError):
        operator.validate_public_binding(**args)


def test_published_manifest_binds_the_executing_operator_source(public_case, tmp_path):
    source = tmp_path / 'operator.py'
    source.write_bytes(b'# fabricated operator source\n')
    raw_source = source.read_bytes()
    source_url = comparison._RAW + 'c' * 40 + '/tools/operator.py'
    manifest = json.loads(public_case.args['manifest'])
    manifest['sources']['operator'] = {'sha256': sha(raw_source), 'url': source_url}
    raw_manifest = canonical(manifest)
    public_case.published[source_url] = raw_source
    public_case.published[public_case.args['manifest_url']] = raw_manifest
    assert operator.verify_operator_manifest(raw_manifest, sha(raw_manifest),
        public_case.args['manifest_url'], source, public_case.published.__getitem__) is True
    source.write_bytes(b'# altered\n')
    with pytest.raises(operator.OperatorError):
        operator.verify_operator_manifest(raw_manifest, sha(raw_manifest),
            public_case.args['manifest_url'], source, public_case.published.__getitem__)


def test_missing_operator_manifest_entry_is_rejected(public_case, tmp_path):
    source = tmp_path / 'operator.py'
    source.write_bytes(b'# fabricated source\n')
    with pytest.raises(operator.OperatorError):
        operator.verify_operator_manifest(public_case.args['manifest'], public_case.args['manifest_sha256'],
            public_case.args['manifest_url'], source, public_case.published.__getitem__)


def test_all_published_source_bytes_are_checked_before_any_project_execution(monkeypatch, tmp_path):
    first = tmp_path / 'first.py'
    second = tmp_path / 'second.py'
    first_raw = b'import builtins\nbuiltins._comparison_probe = True\n'
    second_raw = b'value = 1\n'
    first.write_bytes(first_raw)
    second.write_bytes(b'changed')
    urls = [comparison._RAW + c * 40 + '/tools/' + name + '.py'
            for c, name in [('a', 'first'), ('b', 'second')]]
    monkeypatch.setattr(operator, '_PINNED_SOURCES', (
        ('first', sha(first_raw), urls[0]), ('second', sha(second_raw), urls[1])))
    monkeypatch.setattr(operator, '_MODULE_SPECS', (
        ('first', '_comparison_probe_first', ''), ('second', '_comparison_probe_second', '')))
    published = {urls[0]: first_raw, urls[1]: second_raw}
    import builtins
    monkeypatch.delattr(builtins, '_comparison_probe', raising=False)
    with pytest.raises(operator.OperatorError):
        operator.capture_sources({'first': str(first), 'second': str(second)}, published.__getitem__)
    assert not hasattr(builtins, '_comparison_probe')
    second.write_bytes(second_raw)
    captured = operator.capture_sources({'first': str(first), 'second': str(second)}, published.__getitem__)
    assert not hasattr(builtins, '_comparison_probe')
    changed = list(captured)
    changed[1] = (*changed[1][:-1], b'value = 999\n')
    with pytest.raises(operator.OperatorError):
        operator.execute_sources(tuple(changed))
    assert not hasattr(builtins, '_comparison_probe')
    modules = operator.execute_sources(captured)
    try:
        assert builtins._comparison_probe is True
        assert modules['second'].value == 1
    finally:
        import sys
        sys.modules.pop('_comparison_probe_first', None)
        sys.modules.pop('_comparison_probe_second', None)
        monkeypatch.delattr(builtins, '_comparison_probe', raising=False)


def test_public_source_mismatch_is_rejected_without_execution(monkeypatch, tmp_path):
    path = tmp_path / 'source.py'
    path.write_bytes(b'value = 1\n')
    url = comparison._RAW + 'a' * 40 + '/tools/source.py'
    monkeypatch.setattr(operator, '_PINNED_SOURCES', (('source', sha(path.read_bytes()), url),))
    monkeypatch.setattr(operator, '_MODULE_SPECS', (('source', '_comparison_probe', ''),))
    with pytest.raises(operator.OperatorError, match='published'):
        operator.capture_sources({'source': str(path)}, lambda _: b'different')


def site_snapshot(directory):
    return {'user': 'S-1-5-21-100-200-300-1001', 'owner': 'S-1-5-21-100-200-300-1001',
        'protected': directory, 'rules': [
            {'sid': sid, 'type': 0, 'inheritance': 3 if directory else 0,
             'propagation': 0, 'inherited': not directory, 'rights': 0x1f01ff}
            for sid in ('S-1-3-4', 'S-1-5-18', 'S-1-5-32-544')]}


@pytest.mark.parametrize('directory', [True, False])
def test_existing_site_custody_acl_is_checked_without_reading_site_contents(tmp_path, directory):
    path = tmp_path / 'protected'
    if directory:
        path.mkdir()
    else:
        path.write_bytes(b'do not read this fabricated private content')
    info = path.stat()
    provider = SimpleNamespace(_windows_acl_snapshot=lambda _: site_snapshot(directory))
    assert operator.verify_site_custody_metadata(provider, path, (info.st_dev, info.st_ino),
                                                 directory=directory) is True


@pytest.mark.parametrize('mutation', ['extra_principal', 'owner', 'inheritance', 'identity'])
def test_site_custody_metadata_rejects_policy_or_identity_change(tmp_path, mutation):
    path = tmp_path / 'protected'
    path.mkdir()
    info = path.stat()
    snapshot = site_snapshot(True)
    identity = (info.st_dev, info.st_ino)
    if mutation == 'extra_principal':
        snapshot['rules'].append(dict(snapshot['rules'][0], sid='S-1-1-0'))
    elif mutation == 'owner':
        snapshot['owner'] = 'S-1-5-18'
    elif mutation == 'inheritance':
        snapshot['rules'][0]['inherited'] = True
    else:
        identity = (info.st_dev, info.st_ino + 1)
    with pytest.raises(operator.OperatorError):
        operator.verify_site_custody_metadata(SimpleNamespace(_windows_acl_snapshot=lambda _: snapshot),
                                              path, identity, directory=True)


def test_private_path_check_uses_exact_phase_files_and_track_acl_helpers(tmp_path, monkeypatch):
    natural_parent = tmp_path / 'natural'
    natural_phase = natural_parent / ('natural-heldout-' + comparison._NATURAL_FREEZE)
    injection_parent = tmp_path / 'injection'
    injection_phase = injection_parent / ('injection-heldout-' + comparison._INJECTION_FREEZE)
    site_parent = tmp_path / 'site'
    for path in (natural_phase, injection_phase, site_parent):
        path.mkdir(parents=True)
    config = {'natural': {}, 'injection': {}, 'site_nonce': str(site_parent / 'nonce.bin'),
              'site_record': str(site_parent / 'site-record.json')}
    for track, phase in (('natural', natural_phase), ('injection', injection_phase)):
        for field, name in (('nonce', 'nonce.bin'), ('record', 'heldout-record.json'),
                            ('commitment', 'commitment.json')):
            config[track][field] = str(phase / name)
    config['injection']['inventory'] = str(injection_phase / 'inventory.json')
    files = [Path(config[track][field]) for track in ('natural', 'injection')
             for field in ('nonce', 'record', 'commitment')]
    files.extend((Path(config['injection']['inventory']), Path(config['site_nonce']), Path(config['site_record'])))
    for path in files:
        path.write_bytes(b'fabricated private input; do not read')
    paths = {'natural_parent': natural_parent, 'natural_phase': natural_phase,
        'injection_parent': injection_parent, 'injection_phase': injection_phase,
        'site_parent': site_parent, 'natural_nonce': files[0], 'natural_record': files[1],
        'natural_commitment': files[2], 'injection_nonce': files[3],
        'injection_record': files[4], 'injection_commitment': files[5],
        'injection_inventory': files[6], 'site_nonce': files[7], 'site_record': files[8]}
    identities = {name: (path.stat().st_dev, path.stat().st_ino) for name, path in paths.items()}
    seen = []
    acl = SimpleNamespace(verify_private_parent=lambda path, identity: seen.append(('parent', path)) or True)
    query = SimpleNamespace(_verify_inherited_private=lambda acl, path, identity, directory=True:
                            seen.append(('directory' if directory else 'file', path)))
    monkeypatch.setattr(operator, 'verify_site_custody_metadata',
                        lambda acl, path, identity, directory: seen.append(('site', path)) or True)
    assert operator.verify_private_inputs(comparison, config, acl=acl, natural_query=query,
        natural_parent=natural_parent, injection_parent=injection_parent,
        site_parent=site_parent, identities=identities) is True
    assert len(seen) == 14
    config['natural']['nonce'] = str(injection_phase / 'nonce.bin')
    with pytest.raises(operator.OperatorError):
        operator.verify_private_inputs(comparison, config, acl=acl, natural_query=query,
            natural_parent=natural_parent, injection_parent=injection_parent,
            site_parent=site_parent, identities=identities)


def test_protected_operator_rechecks_public_and_private_inputs_before_child(public_case, tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_bytes(b'value = 1\n')
    source_url = comparison._RAW + 'e' * 40 + '/tools/source.py'
    monkeypatch.setattr(operator, '_PINNED_SOURCES', (('comparison', sha(source.read_bytes()), source_url),))
    monkeypatch.setattr(operator, '_MODULE_SPECS', (('comparison', '_comparison_probe', ''),))
    operator_source = Path(operator.__file__)
    operator_url = comparison._RAW + 'f' * 40 + '/tools/operator.py'
    manifest = canonical({'schema': 'mlg-injection-comparison-source-manifest-v1', 'sources': {
        'comparison': {'sha256': sha(source.read_bytes()), 'url': source_url},
        'operator': {'sha256': sha(operator_source.read_bytes()), 'url': operator_url}}})
    public_case.published[source_url] = source.read_bytes()
    public_case.published[operator_url] = operator_source.read_bytes()
    public_case.published[public_case.args['manifest_url']] = manifest
    captured = operator.capture_sources({'comparison': str(source)}, public_case.published.__getitem__)
    natural_parent, injection_parent, site_parent = (tmp_path / name for name in ('natural', 'injection', 'site'))
    natural_phase = natural_parent / ('natural-heldout-' + comparison._NATURAL_FREEZE)
    injection_phase = injection_parent / ('injection-heldout-' + comparison._INJECTION_FREEZE)
    for path in (natural_phase, injection_phase, site_parent, tmp_path / 'public', injection_parent / 'evidence'):
        path.mkdir(parents=True)
    receipt = injection_parent / 'evidence' / 'receipt.json'
    config = {'schema': 'mlg-injection-split-comparison-config-v1',
        'source_root': str(tmp_path), 'custody_source': str(source), 'pycache_prefix': str(tmp_path / 'cache'),
        'output': str(receipt), 'site_nonce': str(site_parent / 'nonce.bin'),
        'site_record': str(site_parent / 'site-record.json'),
        'natural': {'freeze': str(tmp_path / 'public' / 'natural-freeze'),
            'inventory': str(tmp_path / 'public' / 'natural-inventory'),
            'nonce': str(natural_phase / 'nonce.bin'), 'record': str(natural_phase / 'heldout-record.json'),
            'commitment': str(natural_phase / 'commitment.json')},
        'injection': {'freeze': str(tmp_path / 'public' / 'injection-freeze'),
            'inventory': str(injection_phase / 'inventory.json'),
            'nonce': str(injection_phase / 'nonce.bin'), 'record': str(injection_phase / 'heldout-record.json'),
            'commitment': str(injection_phase / 'commitment.json')},
        'injection_commitment_sha256': public_case.args['commitment_sha256'],
        'injection_commitment_url': public_case.args['commitment_url']}
    Path(config['injection']['freeze']).write_bytes(public_case.args['freeze'])
    Path(config['injection']['commitment']).write_bytes(public_case.args['commitment'])
    Path(config['natural']['freeze']).write_bytes(b'public natural freeze')
    Path(config['natural']['commitment']).write_bytes(b'public natural commitment')
    Path(config['natural']['inventory']).write_bytes(b'public natural inventory')
    Path(config['pycache_prefix']).mkdir()
    config_path, operation_path, manifest_path = (tmp_path / 'public' / name
                                                  for name in ('config.json', 'operation.json', 'manifest.json'))
    config_path.write_bytes(canonical(config))
    operation_path.write_bytes(public_case.args['operation'])
    manifest_path.write_bytes(manifest)
    operator_config_path = injection_parent / 'operator-config.json'
    operator_config_path.write_bytes(b'protected operator configuration')
    events = []
    class Lease:
        def __init__(self, root, files): self.root, self.files = root, files
        def __enter__(self): events.append('lease'); return self
        def close(self): events.append('close')
        def handles(self): return (len(events) + id(self),)
    class Guard:
        SourceFileLease = Lease
        @staticmethod
        def duplicate_to_process(handles, process):
            events.append('duplicate')
            return handles
    def run_guarded(leases, precheck, launch, postcheck, **kwargs):
        try:
            for lease in leases: lease.__enter__()
            assert precheck(leases) is True
            return postcheck(launch(leases))
        finally:
            for lease in reversed(leases): lease.close()
    def run_exit(**kwargs):
        assert kwargs['source_check']() is True
        assert kwargs['adapter_options']['protect_process'](object()) is True
        events.append('child')
        return {'status': 'collision', 'receipt_sha256': 'a' * 64}
    modules = {'comparison': comparison, 'custody': custody, 'source_guards': Guard,
        'replay_owner': SimpleNamespace(run_guarded=run_guarded),
        'natural_query': SimpleNamespace(), 'adapter': SimpleNamespace(_BOOTSTRAP='bootstrap'),
        'exit': SimpleNamespace(run_operator=run_exit, verify_public_receipt=lambda **kwargs: True),
        'observer': SimpleNamespace(observe_supervisor=lambda **kwargs: None)}
    monkeypatch.setattr(operator, 'verify_private_inputs',
                        lambda *args, **kwargs: events.append('private_metadata') or True)
    args = dict(modules=modules, captured=captured, operator_path=operator_source,
        operator_config_path=operator_config_path, operator_config_sha256=sha(operator_config_path.read_bytes()),
        config_path=config_path, config_sha256=sha(config_path.read_bytes()),
        operation_path=operation_path, operation_sha256=sha(operation_path.read_bytes()),
        operation_url=public_case.args['operation_url'], manifest_path=manifest_path,
        manifest_sha256=sha(manifest), manifest_url=public_case.args['manifest_url'],
        natural_parent=natural_parent, injection_parent=injection_parent, site_parent=site_parent,
        identities={}, terminal_path=injection_parent/'evidence'/'terminal.json',
        inner_output=injection_parent/'evidence'/'inner', base_python=Path(__import__('sys')._base_executable),
        environment={'CUDA_VISIBLE_DEVICES': '-1'}, timeout=60, reader=public_case.published.__getitem__)
    result = operator.run_protected_comparison(**args)
    assert result['status'] == 'collision'
    assert events.index('private_metadata') < events.index('child')
    assert 'duplicate' in events and events[-1] == 'close'
    events.clear()
    with pytest.raises(operator.OperatorError):
        operator.run_protected_comparison(**dict(args, terminal_path=tmp_path/'outside.json'))
    assert 'child' not in events
    events.clear()
    public_case.published[args['operation_url']] = b'different'
    with pytest.raises(operator.OperatorError):
        operator.run_protected_comparison(**args)
    assert 'child' not in events


def test_operator_configuration_requires_independently_pinned_manifest(tmp_path):
    paths = {name: str(tmp_path / (name + '.py')) for name, _, _ in operator._PINNED_SOURCES}
    keys = ('comparison_config_path', 'operation_path', 'manifest_path', 'natural_parent',
            'injection_parent', 'site_parent', 'terminal_path', 'inner_output', 'base_python', 'pycache_prefix')
    value = {key: str(tmp_path / key) for key in keys}
    value.update({'schema': 'mlg-injection-comparison-operator-config-v1', 'source_paths': paths,
        'comparison_config_sha256': '1' * 64, 'operation_sha256': '2' * 64,
        'manifest_sha256': '3' * 64, 'operation_url': comparison._RAW + 'a' * 40 + '/op.json',
        'manifest_url': comparison._RAW + 'b' * 40 + '/manifest.json', 'timeout': 60,
        'identities': {name: [1, 2] for name in operator._PRIVATE_IDENTITIES}})
    parsed = operator.parse_operator_config(canonical(value), '3' * 64, value['manifest_url'])
    assert parsed['identities']['natural_parent'] == (1, 2)
    with pytest.raises(operator.OperatorError):
        operator.parse_operator_config(canonical(value), '4' * 64, value['manifest_url'])
    value['timeout'] = True
    with pytest.raises(operator.OperatorError):
        operator.parse_operator_config(canonical(value), '3' * 64, value['manifest_url'])
