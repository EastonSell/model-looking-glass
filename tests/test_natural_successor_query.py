"""A one-shot caller exercised only with fabricated rows and a fake rehearsal."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

try:
    from tools import natural_successor_query as query
except ImportError:
    query = None


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def sha(value):
    return hashlib.sha256(value).hexdigest()


class SyntheticLease:
    """Recorded test resource; deliberately NOT a Windows source lock."""
    def __init__(self, root, paths, events):
        self.root, self.paths, self.events = Path(root), list(paths), events
        self.entered = self.closed = False
    def __enter__(self):
        assert not self.entered and not self.closed
        self.entered = True
        self.events.append(('enter', self))
        return self
    def handles(self):
        assert self.entered and not self.closed
        return [id(self)]
    def close(self):
        assert self.entered and not self.closed
        self.closed = True
        self.events.append(('close', self))


def install_synthetic_query_custody(monkeypatch, events):
    """Keep real query/parser logic; simulate only Windows custody boundaries.

    The comparator runs against fabricated rows in a portable isolated child.
    Saved Job/handle fields are synthetic schema fixtures, not OS evidence.
    """
    original = query._project_modules
    def project(captured, paths):
        prior, guards, supervisor = original(captured, paths)
        guards.SourceFileLease = lambda root, files: SyntheticLease(root, files, events)
        class Handles(SyntheticLease):
            def __init__(self, handles):
                super().__init__('.', [], events)
                self.values = list(handles)
            def __enter__(self):
                super().__enter__()
                return self.values
        guards.InheritableHandles = Handles
        def supervise(command, output, environment, *, audit_result, timeout, inherited_handles):
            assert inherited_handles and all(lease.entered and not lease.closed
                                            for action, lease in events if action == 'enter')
            output.mkdir()
            portable = ('import hashlib,sys; p=sys.argv.pop(1); h=sys.argv.pop(1); '
                        'raw=open(p,"rb").read(1000001); '
                        'assert len(raw)<=1000000 and hashlib.sha256(raw).hexdigest()==h; '
                        'sys.argv[0]=p; exec(compile(raw,p,"exec"),'
                        '{"__name__":"__main__","__file__":p})')
            assert command[5] == query._BOOTSTRAP
            actual = [sys._base_executable] + command[1:5] + [portable] + command[6:]
            child = subprocess.Popen(actual, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                stdout, stderr = child.communicate(timeout=timeout)
            except BaseException:
                child.kill()
                child.communicate()
                raise
            (output / 'stdout.txt').write_bytes(stdout)
            (output / 'stderr.txt').write_bytes(stderr)
            assert child.returncode == 0, (stdout, stderr)
            verdict = {'status': 'incomplete', 'command': command, 'exit_code': child.returncode,
                'child_pid': child.pid, 'cleanup_verified': True, 'child_reaped': True,
                'job_drained': True, 'job_closed': True, 'logs_complete': True,
                'postexit_active_processes': 0, 'supervisor_error': None, 'log_overflow': False,
                'audit_error_type': None, 'cancellation_requested': False, 'quarantined': False,
                'retry_safe': False, 'stdout_sha256': sha(stdout), 'stderr_sha256': sha(stderr),
                'retained_tensor_files': {'accounting_complete': True, 'total_bytes': 0, 'files': []},
                'test_only_custody': 'synthetic Windows fields; portable comparator subprocess only'}
            verdict['audit'] = audit_result(output, child.returncode, True)
            (output / 'supervisor.json').write_bytes(canonical(verdict))
            return verdict
        supervisor.supervise_heldout = supervise
        return prior, guards, supervisor
    monkeypatch.setattr(query, '_project_modules', project)


def project_module_snapshot():
    return {name: module for name, module in sys.modules.items()
            if name in ('looking_glass', 'scripts') or name.startswith(('looking_glass.', 'scripts.'))}


@pytest.fixture
def setup(tmp_path, request, monkeypatch):
    if query is None:
        return None
    # Identity-only fixture: never execute this file or weaken the reader's size bound.
    interpreter = tmp_path / 'synthetic-interpreter'
    interpreter.write_bytes(b'fabricated interpreter identity; not executable')
    monkeypatch.setattr(query, 'sys', SimpleNamespace(**dict(vars(sys), _base_executable=str(interpreter))))
    events = []
    install_synthetic_query_custody(monkeypatch, events)
    repo = Path(__file__).resolve().parents[1]
    root, cache, parent = (tmp_path / name for name in ('root', 'cache', 'old-parent'))
    old, supervisor = parent / 'old', parent / 'supervisor'
    for directory in (root, cache, parent, old, supervisor):
        directory.mkdir()
    freeze_raw = canonical({'fabricated': 'prediction freeze'})
    freeze_path = tmp_path / 'freeze.json'
    freeze_path.write_bytes(freeze_raw)
    freeze, inventory_old, old_source = sha(freeze_raw), '2' * 64, '3' * 40
    nonce = b'x' * 32
    collide = getattr(request, 'param', None) == 'collision'
    record = canonical({'schema': 'mlg-number-heldout-v1', 'prediction_freeze_sha256': freeze,
                        'exposure_inventory_sha256': inventory_old, 'source_commits': {'natural': old_source},
                        'families': [{'cases': [{'input_ids': ([500, 1000] if collide and i == j == 0
                                                              else [600, 100 + 4 * i + j])}
                                                 for j in range(4)]}
                                     for i in range(4)]})
    old_commitment = sha(b'MLG-number-v1-heldout-commit\0' + nonce + b'\0' + record)
    commitment = canonical({'schema': 'mlg-natural-heldout-public-commitment-v1',
                            'heldout_commitment': old_commitment, 'prediction_freeze_sha256': freeze,
                            'exposure_inventory_sha256': inventory_old})
    for name, raw in (('nonce.bin', nonce), ('heldout-record.json', record), ('commitment.json', commitment)):
        (old / name).write_bytes(raw)
    audit = canonical({'audit': {'preparation': {'nonce.bin': sha(nonce), 'heldout-record.json': sha(record),
                                                 'commitment.json': sha(commitment)}}})
    observation = canonical({'status': 'completed', 'exit_code': 0, 'supervisor_record_sha256': sha(audit)})
    (supervisor / 'supervisor.json').write_bytes(audit)
    (supervisor / 'supervisor-observation.json').write_bytes(observation)
    proposal = canonical({'schema': 'mlg-natural-successor-candidate-inputs-v1',
                          'pool_sha256': '4' * 64, 'tokenizer_sha256': '5' * 64,
                          'cases': [{'input_ids': [500, 1000 + i]} for i in range(256)]})
    proposal_path = tmp_path / 'inputs.json'
    proposal_path.write_bytes(proposal)
    inventory = canonical({'version': 'distilgpt2-development-exposures-v21',
                           'records': [{'record_id': 'fabricated', 'category': 'observed', 'token_ids': [700, 1]}]})
    inventory_path = tmp_path / 'inventory.json'
    inventory_path.write_bytes(inventory)
    sources = {'caller': repo / 'tools/natural_successor_query.py',
               'comparator': repo / 'tools/natural_known_row_comparator.py',
               'claim': repo / 'looking_glass/natural_successor_claim.py',
               'accounting': repo / 'looking_glass/natural_successor_accounting.py',
               'sampler': repo / 'looking_glass/natural_agreement_successor.py',
               'rehearsal_verifier': tmp_path / 'rehearsal_verifier.py',
               'acl_verifier': tmp_path / 'acl_verifier.py',
               'controller': repo / 'looking_glass/natural_agreement_controller.py',
               'guards': repo / 'scripts/v2_source_guards.py',
               'supervisor': repo / 'looking_glass/natural_heldout_supervisor.py',
               'owner_wrapper': repo / 'tools/natural_agreement_heldout_owner.py',
               'protocol': repo / 'docs/NATURAL_AGREEMENT_SUCCESSOR_V1.md',
               'clarification': repo / 'docs/NATURAL_AGREEMENT_SUCCESSOR_QUERY_CLARIFICATION_V1.md',
               'inputs': proposal_path, 'inventory_source': tmp_path / 'inventory_source.py'}
    sources['rehearsal_verifier'].write_text('def verify(config, operation):\n    return True\n', encoding='ascii')
    sources['acl_verifier'].write_text(
        "def verify_private_parent(path, identity):\n    return True\n"
        "def _windows_acl_snapshot(path):\n"
        "    fixed = {'type': 0, 'inheritance': 3 if path.is_dir() else 0, 'propagation': 0, 'inherited': True, 'rights': 0x1f01ff}\n"
        "    return {'user':'S-1-5-21-1', 'owner':'S-1-5-21-1', 'protected':False,\n"
        "            'rules':[dict(fixed, sid='S-1-5-21-1'), dict(fixed, sid='S-1-5-18')]}\n",
        encoding='ascii')
    sources['inventory_source'].write_text('# fabricated public inventory source\n', encoding='ascii')
    pins = {'freeze': freeze, 'inventory': inventory_old, 'source_commit': old_source,
            'proposal': sha(proposal), 'commitment': sha(commitment), 'supervisor': sha(audit),
            'observation': sha(observation), 'private_count': 16, 'public_count': 256,
            'comparator_source': sha(sources['comparator'].read_bytes())}
    pins_path = tmp_path / 'pins.json'
    pins_path.write_bytes(canonical(pins))
    entries = {name: {'sha256': sha(path.read_bytes()), 'url': f'https://raw.githubusercontent.com/'
                      f'EastonSell/model-looking-glass/{"a" * 40}/{name}.py'} for name, path in sources.items()}
    info, old_info, parent_info, supervisor_info = root.stat(), old.stat(), parent.stat(), supervisor.stat()
    normalized = os.path.normcase(os.path.realpath(root))
    operation = {'schema': 'mlg-natural-successor-query-operation-v1', 'sources': entries,
                 'root_path_sha256': sha(normalized.encode()), 'root_device': info.st_dev, 'root_inode': info.st_ino,
                 'old_parent_device': parent_info.st_dev, 'old_parent_inode': parent_info.st_ino,
                 'old_phase_device': old_info.st_dev, 'old_phase_inode': old_info.st_ino,
                 'old_supervisor_device': supervisor_info.st_dev, 'old_supervisor_inode': supervisor_info.st_ino,
                 'pool_input_sha256': sha(proposal), 'pool_sha256': '4' * 64, 'tokenizer_sha256': '5' * 64,
                 'freeze_sha256': freeze, 'old_commitment': old_commitment, 'pins_sha256': sha(canonical(pins)),
                 'inventory_sha256': sha(inventory), 'inventory_version': 21,
                 'inventory_source_sha256': entries['inventory_source']['sha256'],
                 'protocol_sha256': entries['protocol']['sha256'],
                 'query_clarification_sha256': entries['clarification']['sha256'],
                 'old_inventory_sha256': inventory_old, 'old_source_commit': old_source,
                 'old_commitment_file_sha256': sha(commitment), 'old_supervisor_sha256': sha(audit),
                 'old_observation_sha256': sha(observation),
                 'freeze_url': f'https://raw.githubusercontent.com/EastonSell/model-looking-glass/{"a" * 40}/freeze.json',
                 'old_commitment_url': f'https://raw.githubusercontent.com/EastonSell/model-looking-glass/{"a" * 40}/old.json'}
    operation_raw = canonical(operation)
    operation_path = tmp_path / 'operation.json'
    operation_path.write_bytes(operation_raw)
    config = {'schema': 'mlg-natural-successor-query-config-v1', 'root': str(root), 'cache': str(cache),
              'old_private_parent': str(parent), 'old_phase': str(old), 'old_supervisor': str(supervisor),
              'inputs': str(proposal_path), 'public_freeze': str(freeze_path), 'rehearsal': {},
              'pins': str(pins_path), 'inventory': str(inventory_path), 'operation': str(operation_path),
              'operation_url': f'https://raw.githubusercontent.com/EastonSell/model-looking-glass/{"b" * 40}/op.json',
              'sources': {name: str(path) for name, path in sources.items()}}
    config_raw = canonical(config)
    config_path = tmp_path / 'config.json'
    config_path.write_bytes(config_raw)
    remote = {config['operation_url']: operation_raw}
    remote.update({entries[name]['url']: path.read_bytes() for name, path in sources.items()})
    remote.update({operation['freeze_url']: freeze_raw, operation['old_commitment_url']: commitment})
    observe = lambda op: {'version': op['inventory_version'], 'source_sha256': op['inventory_source_sha256'],
                          'inventory_sha256': op['inventory_sha256'], 'head': 'c' * 40}
    (parent / 'old-ledger').write_bytes(b'old ledger')
    (parent / 'old-gate').mkdir()
    (parent / 'old-gate' / 'old-marker').write_bytes(b'old gate')
    return {'synthetic_interpreter': interpreter, 'synthetic_custody_events': events, 'root': root, 'old_parent': parent, 'config_path': config_path, 'config_sha256': sha(config_raw),
            'operation_sha256': sha(operation_raw), 'remote': remote, 'observe': observe}


def run(value):
    before = project_module_snapshot()
    active = list(query._ACTIVE_OWNER)
    events = value['synthetic_custody_events']
    start = len(events)
    try:
        return query.run_query(value['config_path'], config_sha256=value['config_sha256'],
                               operation_sha256=value['operation_sha256'], reader=value['remote'].__getitem__,
                               observe_current=value['observe'])
    finally:
        # Assert restoration; never clear real quarantine or repair modules silently.
        after = project_module_snapshot()
        assert after.keys() == before.keys()
        assert all(after[name] is module for name, module in before.items())
        assert query._ACTIVE_OWNER == active
        entered = [lease for action, lease in events[start:] if action == 'enter']
        closed = [lease for action, lease in events[start:] if action == 'close']
        assert closed == list(reversed(entered))
        assert all(lease.closed for lease in entered)


def old_snapshot(parent):
    """Only fabricated files are enumerated in this regression fixture."""
    return {item.relative_to(parent).as_posix(): (item.stat().st_dev, item.stat().st_ino,
            item.read_bytes() if item.is_file() else None) for item in parent.rglob('*')}


def assert_no_new_phase(root):
    assert not (root / 'phase-claim.json').exists()
    assert not (root / 'nonce.bin').exists()


def rebind_config(value, change):
    config = json.loads(value['config_path'].read_bytes())
    operation_path = Path(config['operation'])
    operation = json.loads(operation_path.read_bytes())
    change(config, operation)
    operation_raw = canonical(operation)
    operation_path.write_bytes(operation_raw)
    config_raw = canonical(config)
    value['config_path'].write_bytes(config_raw)
    value['config_sha256'] = sha(config_raw)
    value['operation_sha256'] = sha(operation_raw)
    value['remote'][config['operation_url']] = operation_raw


def test_fabricated_fixed_query_is_recorded_and_cannot_repeat(setup):
    assert query is not None
    before = old_snapshot(setup['old_parent'])
    result = run(setup)
    assert result['status'] == 'clear' and result['reaped'] is True
    assert result['job_drained'] is True and result['job_closed'] is True
    assert (setup['root'] / 'pool-query-claim.json').is_file()
    assert (setup['root'] / 'pool-query-receipt.json').is_file()
    assert (setup['root'] / 'pool-query-observation.json').is_file()
    assert_no_new_phase(setup['root'])
    with pytest.raises((ValueError, RuntimeError, OSError)):
        run(setup)
    assert old_snapshot(setup['old_parent']) == before


@pytest.mark.parametrize('name', ['nonce.bin', 'heldout-record.json', 'phase-claim.json', 'pool-query-supervisor'])
@pytest.mark.parametrize('during_precheck', [False, True])
def test_existing_or_late_phase_artifact_stops_before_query(setup, name, during_precheck):
    root = setup['root']
    path = root / name
    def insert():
        if name == 'pool-query-supervisor':
            path.mkdir()
        else:
            path.write_bytes(b'fabricated orphan')
    if during_precheck:
        prior = setup['observe']
        def observe(operation):
            result = prior(operation)
            insert()
            return result
        setup['observe'] = observe
    else:
        insert()
    before = old_snapshot(setup['old_parent'])
    with pytest.raises((ValueError, RuntimeError, OSError)):
        run(setup)
    assert not (root / 'pool-query-claim.json').exists()
    assert not (root / 'pool-query-receipt.json').exists()
    assert not (root / 'pool-query-observation.json').exists()
    assert old_snapshot(setup['old_parent']) == before


def test_stale_official_inventory_stops_before_query_claim(setup):
    assert query is not None
    setup['observe'] = lambda _: {'version': 22, 'source_sha256': '9' * 64,
                                 'inventory_sha256': '0' * 64, 'head': 'c' * 40}
    with pytest.raises((ValueError, RuntimeError)):
        run(setup)
    assert not list(setup['root'].iterdir())


def test_unavailable_rehearsal_stops_before_query_claim(setup):
    assert query is not None
    setup['remote'].pop(next(url for url in setup['remote'] if url.endswith('/rehearsal_verifier.py')))
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        run(setup)
    assert not list(setup['root'].iterdir())


@pytest.mark.parametrize('setup', ['collision'], indirect=True)
def test_collision_is_recorded_but_never_authorizes_phase_or_redraw(setup):
    assert query is not None
    before = old_snapshot(setup['old_parent'])
    with pytest.raises(query.QueryCollision):
        run(setup)
    observation = json.loads((setup['root'] / 'pool-query-observation.json').read_bytes())
    assert observation['status'] == 'collision' and observation['reaped'] is True
    assert_no_new_phase(setup['root'])
    with pytest.raises((ValueError, RuntimeError, OSError)):
        run(setup)
    assert old_snapshot(setup['old_parent']) == before


def test_lost_observation_write_retains_the_query_and_stops_retry(setup, monkeypatch):
    assert query is not None
    before = old_snapshot(setup['old_parent'])
    def interrupted(path, raw):
        with Path(path).open('xb') as stream:
            stream.write(raw[:5])
        raise OSError('fabricated short observation')
    monkeypatch.setattr(query, '_write_new', interrupted)
    with pytest.raises(OSError):
        run(setup)
    assert (setup['root'] / 'pool-query-claim.json').is_file()
    assert (setup['root'] / 'pool-query-receipt.json').is_file()
    assert_no_new_phase(setup['root'])
    with pytest.raises((ValueError, RuntimeError, OSError)):
        run(setup)
    assert old_snapshot(setup['old_parent']) == before


@pytest.mark.parametrize('which', ['root', 'cache'])
def test_new_paths_cannot_be_siblings_inside_old_parent(setup, which):
    before = old_snapshot(setup['old_parent'])
    sibling = setup['old_parent'] / ('fabricated-' + which)
    sibling.mkdir()
    def change(config, operation):
        config[which] = str(sibling)
        if which == 'root':
            info = sibling.stat()
            operation['root_device'], operation['root_inode'] = info.st_dev, info.st_ino
            operation['root_path_sha256'] = sha(os.path.normcase(os.path.realpath(sibling)).encode())
    rebind_config(setup, change)
    with pytest.raises(query.QueryError, match='roots overlap'):
        run(setup)
    assert not list(setup['root'].iterdir())
    assert old_snapshot(setup['old_parent']) == before | {'fabricated-' + which: (sibling.stat().st_dev, sibling.stat().st_ino, None)}


def test_old_child_acl_mismatch_stops_before_claim(setup):
    before = old_snapshot(setup['old_parent'])
    def change(config, operation):
        path = Path(config['sources']['acl_verifier'])
        path.write_text("def verify_private_parent(path, identity):\n    return True\n"
                        "def _windows_acl_snapshot(path):\n    return {'protected': True}\n", encoding='ascii')
        entry = operation['sources']['acl_verifier']
        entry['sha256'] = sha(path.read_bytes())
        setup['remote'][entry['url']] = path.read_bytes()
    rebind_config(setup, change)
    with pytest.raises(query.QueryError, match='child ACL'):
        run(setup)
    assert not list(setup['root'].iterdir())
    assert old_snapshot(setup['old_parent']) == before


def test_input_file_in_output_root_is_rejected_before_claim(setup):
    before = old_snapshot(setup['old_parent'])
    def change(config, operation):
        destination = setup['root'] / 'pins.json'
        destination.write_bytes(Path(config['pins']).read_bytes())
        config['pins'] = str(destination)
    rebind_config(setup, change)
    with pytest.raises(query.QueryError, match='separate directory'):
        run(setup)
    assert not (setup['root'] / 'pool-query-claim.json').exists()
    assert_no_new_phase(setup['root'])
    assert old_snapshot(setup['old_parent']) == before


def test_old_private_file_acl_mismatch_stops_before_read_or_claim(setup, monkeypatch):
    before = old_snapshot(setup['old_parent'])
    def change(config, operation):
        path = Path(config['sources']['acl_verifier'])
        source = path.read_text(encoding='ascii')
        source = source.replace("    fixed = {'type': 0,", "    if path.name == 'heldout-record.json': return {'protected': True}\n    fixed = {'type': 0,")
        path.write_text(source, encoding='ascii')
        entry = operation['sources']['acl_verifier']
        entry['sha256'] = sha(path.read_bytes())
        setup['remote'][entry['url']] = path.read_bytes()
    rebind_config(setup, change)
    original_read, protected_reads = query._read, []
    def tracked(path, *args, **kwargs):
        if Path(path).parent in (setup['old_parent'] / 'old', setup['old_parent'] / 'supervisor'):
            protected_reads.append(Path(path).name)
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(query, '_read', tracked)
    with pytest.raises(query.QueryError, match='child ACL'):
        run(setup)
    assert protected_reads == []
    assert not list(setup['root'].iterdir())
    assert old_snapshot(setup['old_parent']) == before


def test_changed_inventory_after_query_keeps_claim_and_stops(setup):
    before = old_snapshot(setup['old_parent'])
    correct = setup['observe']
    calls = []
    def changed(operation):
        calls.append(1)
        state = correct(operation)
        if len(calls) > 1:
            state['version'] += 1
        return state
    setup['observe'] = changed
    with pytest.raises(query.QueryError, match='current exposure'):
        run(setup)
    assert (setup['root'] / 'pool-query-claim.json').is_file()
    assert (setup['root'] / 'pool-query-receipt.json').is_file()
    assert not (setup['root'] / 'pool-query-observation.json').exists()
    assert_no_new_phase(setup['root'])
    with pytest.raises((ValueError, RuntimeError, OSError)):
        run(setup)
    assert old_snapshot(setup['old_parent']) == before


@pytest.mark.skipif(os.name != 'nt', reason='Windows gated child regression')
def test_child_waits_for_gate_before_opening_comparator(tmp_path):
    missing = tmp_path / 'must-not-open.py'
    process = subprocess.Popen([sys._base_executable, '-I', '-S', '-B', '-c', query._BOOTSTRAP,
                                str(missing), '0' * 64], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        time.sleep(0.2)
        assert process.poll() is None
        stdout, stderr = process.communicate(input=b'wrong\n', timeout=15)
        assert process.returncode != 0 and stdout == b'' and stderr == b''
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    opened = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', query._BOOTSTRAP,
                             str(missing), '0' * 64], input=b'MLG-natural-discovery-start-v1\n',
                            capture_output=True, timeout=15)
    assert opened.returncode != 0 and b'FileNotFoundError' in opened.stderr
