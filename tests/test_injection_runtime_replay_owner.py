"""Fabricated process-custody checks for the public replay parent."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import injection_runtime_replay_child as child
from tools import injection_runtime_replay_owner as owner


class Lease:
    def __init__(self, events, name, *, close_error=None):
        self.events, self.name, self.close_error = events, name, close_error

    def __enter__(self):
        self.events.append(('enter', self.name))
        return self

    def close(self):
        self.events.append(('close', self.name))
        if self.close_error:
            raise self.close_error


@pytest.fixture(autouse=True)
def clear_fake_custody():
    assert not owner._QUARANTINED
    yield
    owner._QUARANTINED.clear()  # All resources here are inert fabricated leases.


def test_protection_covers_launch_and_postcheck():
    events = []
    leases = [Lease(events, 'source'), Lease(events, 'cache')]
    def precheck(acquired):
        assert acquired == leases
        events.append(('precheck', None))
        return True
    def launch(acquired):
        assert acquired == leases
        events.append(('launch', None))
        return {'cleanup_verified': True}
    def postcheck(verdict):
        events.append(('postcheck', None))
        return 'checked'
    assert owner.under_protection(leases, precheck, launch, postcheck) == 'checked'
    assert events == [('enter', 'source'), ('enter', 'cache'), ('precheck', None),
                      ('launch', None), ('postcheck', None), ('close', 'cache'), ('close', 'source')]
    assert not owner._QUARANTINED


def test_unknown_launch_keeps_resources_rooted_and_blocks_restart():
    events = []
    lease = Lease(events, 'source')
    def uncertain(_):
        raise MemoryError('fabricated launch uncertainty')
    with pytest.raises(MemoryError):
        owner.under_protection([lease], lambda _: True, uncertain, lambda _: None)
    assert owner._QUARANTINED[0]['leases'] == [lease]
    assert events == [('enter', 'source')]
    with pytest.raises(owner.ReplayOwnerError):
        owner.under_protection([], lambda _: None, lambda _: {}, lambda _: None)


def test_prelaunch_failure_closes_and_does_not_claim_cleanup():
    events = []
    lease = Lease(events, 'source')
    def rejected(_):
        raise ValueError('fabricated mismatch')
    with pytest.raises(ValueError):
        owner.under_protection([lease], rejected, lambda _: pytest.fail('launched'), lambda _: None)
    assert events == [('enter', 'source'), ('close', 'source')]
    assert not owner._QUARANTINED


def test_false_precheck_never_authorizes_launch():
    with pytest.raises(owner.ReplayOwnerError):
        owner.under_protection([], lambda _: False, lambda _: pytest.fail('launched'), lambda _: None)
    assert not owner._QUARANTINED


def test_uncertain_close_retains_lease_and_primary_interrupt():
    events = []
    lease = Lease(events, 'source', close_error=OSError('fabricated close failure'))
    def interrupted(_):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        owner.under_protection([lease], interrupted, lambda _: None, lambda _: None)
    assert owner._QUARANTINED[0]['leases'] == [lease]


def test_signal_restoration_cannot_mask_original_interrupt(monkeypatch):
    restored = []
    monkeypatch.setattr(owner.signal, 'getsignal', lambda number: 'old')
    def set_signal(number, handler):
        if handler == 'old':
            restored.append(number)
            if len(restored) == 1:
                raise OSError('fabricated signal restore failure')
    monkeypatch.setattr(owner.signal, 'signal', set_signal)
    def interrupted(_):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as raised:
        owner.run_guarded([], interrupted, lambda _: None, lambda _: None)
    assert isinstance(raised.value.__cause__, OSError)
    assert len(restored) == (2 if getattr(owner.signal, 'SIGBREAK', None) is not None else 1)


def test_fatal_signal_restoration_supersedes_ordinary_precheck_error(monkeypatch):
    restored = []
    monkeypatch.setattr(owner.signal, 'getsignal', lambda number: 'old')
    def set_signal(number, handler):
        if handler == 'old':
            restored.append(number)
            if len(restored) == 1:
                raise SystemExit(0)
    monkeypatch.setattr(owner.signal, 'signal', set_signal)
    def failed(_):
        raise ValueError('fabricated precheck failure')
    with pytest.raises(SystemExit) as raised:
        owner.run_guarded([], failed, lambda _: None, lambda _: None)
    assert isinstance(raised.value.__cause__, ValueError)
    assert len(restored) == (2 if getattr(owner.signal, 'SIGBREAK', None) is not None else 1)


def test_child_postcheck_is_bounded_and_explicitly_not_scientific_completion(tmp_path):
    config = {'postcheck_file': str(tmp_path / 'child-postcheck.json'), 'attempt_dir': str(tmp_path / 'attempt')}
    post = {'schema': 'mlg-injection-runtime-replay-child-postcheck-v1',
            'status': 'validated_no_model_forward', 'plan_sha256': '1' * 64,
            'heldout_runtime_sha256': child._HELDOUT_RUNTIME, 'child_source_sha256': '2' * 64,
            'forward_count': 0}
    Path(config['postcheck_file']).write_bytes(child._canonical(post))
    audit = owner.inspect_child_postcheck(child, config, '1' * 64, '2' * 64, validation_only=True)
    assert audit['status'] == 'incomplete' and audit['validation_only'] is True
    assert audit['child_postcheck_sha256'] == hashlib.sha256(child._canonical(post)).hexdigest()
    post['forward_count'] = False
    Path(config['postcheck_file']).write_bytes(child._canonical(post))
    with pytest.raises(owner.ReplayOwnerError):
        owner.inspect_child_postcheck(child, config, '1' * 64, '2' * 64, validation_only=True)


def test_private_replay_postcheck_binds_exact_terminal_and_receipt_bytes(tmp_path):
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    terminal = child._canonical({'status': 'replay-produced-private-awaiting-independent-audit',
                                 'forward_count': 16, 'comparisons': []})
    receipts = b'fabricated retained receipt stream\n'
    (attempt / 'owner-terminal.json').write_bytes(terminal)
    (attempt / 'receipts.jsonl').write_bytes(receipts)
    config = {'postcheck_file': str(tmp_path / 'child-postcheck.json'), 'attempt_dir': str(attempt)}
    post = {'schema': 'mlg-injection-runtime-replay-child-postcheck-v1',
            'status': 'private_replay_pending_independent_audit', 'plan_sha256': '1' * 64,
            'heldout_runtime_sha256': child._HELDOUT_RUNTIME, 'child_source_sha256': '2' * 64,
            'discovery_runtime_sha256': child._DISCOVERY_RUNTIME, 'runtime_deviation_sha256': child._DEVIATION,
            'source_supplement_sha256': child._SOURCE_SUPPLEMENT, 'forward_count': 16,
            'owner_terminal_sha256': hashlib.sha256(terminal).hexdigest(),
            'receipts_sha256': hashlib.sha256(receipts).hexdigest()}
    Path(config['postcheck_file']).write_bytes(child._canonical(post))
    audit = owner.inspect_child_postcheck(child, config, '1' * 64, '2' * 64, validation_only=False)
    assert audit['status'] == 'incomplete' and audit['forward_count'] == 16
    (attempt / 'receipts.jsonl').write_bytes(receipts + b'tampered')
    with pytest.raises(owner.ReplayOwnerError):
        owner.inspect_child_postcheck(child, config, '1' * 64, '2' * 64, validation_only=False)


def test_supervisor_receipt_binds_exit_audit_command_and_logs(tmp_path):
    config = {'postcheck_file': str(tmp_path / 'child-postcheck.json'), 'attempt_dir': str(tmp_path / 'attempt')}
    post = {'schema': 'mlg-injection-runtime-replay-child-postcheck-v1',
            'status': 'validated_no_model_forward', 'plan_sha256': '1' * 64,
            'heldout_runtime_sha256': child._HELDOUT_RUNTIME, 'child_source_sha256': '2' * 64,
            'forward_count': 0}
    Path(config['postcheck_file']).write_bytes(child._canonical(post))
    output = tmp_path / 'supervisor'
    output.mkdir()
    for name in ('stdout.txt', 'stderr.txt'):
        (output / name).write_bytes(b'')
    audit = owner.inspect_child_postcheck(child, config, '1' * 64, '2' * 64, validation_only=True)
    digest = hashlib.sha256(b'').hexdigest()
    verdict = {'status': 'incomplete', 'exit_code': 0, 'child_reaped': True, 'child_pid': 123,
               'job_drained': True, 'job_closed': True, 'logs_complete': True,
               'postexit_active_processes': 0, 'cleanup_verified': True, 'retry_safe': False,
               'cancellation_requested': False, 'quarantined': False, 'log_overflow': False,
               'supervisor_error': None, 'command': ['base', 'config'],
               'stdout_sha256': digest, 'stderr_sha256': digest, 'audit': audit}
    raw = child._canonical(verdict)
    (output / 'supervisor.json').write_bytes(raw)
    result = owner.check_supervisor_record(child, config, '1' * 64, '2' * 64, output,
                                           ['base', 'config'], verdict, validation_only=True)
    assert result['status'] == 'validated_no_model_forward'
    assert result['supervisor_sha256'] == hashlib.sha256(raw).hexdigest()
    for key, bad in (('exit_code', False), ('postexit_active_processes', False),
                     ('child_pid', True), ('job_closed', False), ('command', ['other'])):
        changed = dict(verdict, **{key: bad})
        (output / 'supervisor.json').write_bytes(child._canonical(changed))
        with pytest.raises(owner.ReplayOwnerError):
            owner.check_supervisor_record(child, config, '1' * 64, '2' * 64, output,
                                          ['base', 'config'], changed, validation_only=True)
    oversized = b'x' * (256 * 1024 + 1)
    (output / 'stdout.txt').write_bytes(oversized)
    changed = dict(verdict, stdout_sha256=hashlib.sha256(oversized).hexdigest())
    (output / 'supervisor.json').write_bytes(child._canonical(changed))
    with pytest.raises((owner.ReplayOwnerError, ValueError)):
        owner.check_supervisor_record(child, config, '1' * 64, '2' * 64, output,
                                      ['base', 'config'], changed, validation_only=True)


def test_supervisor_inventory_rejects_extra_entries_before_loading(tmp_path):
    for name in ('stdout.txt', 'stderr.txt', 'supervisor.json', 'extra'):
        (tmp_path / name).write_bytes(b'')
    with pytest.raises(owner.ReplayOwnerError):
        owner.check_supervisor_record(child, {}, '1' * 64, '2' * 64, tmp_path,
                                      ['unused'], {}, validation_only=True)


def test_real_isolated_bootstrap_hashes_source_and_honors_gate(tmp_path):
    prefix = tmp_path / 'prefix'
    prefix.mkdir()
    marker, source, config = tmp_path / 'marker', tmp_path / 'child.py', tmp_path / 'config.json'
    source.write_text("import pathlib,sys\n"
                      "if sys.stdin.buffer.readline() == b'MLG-natural-discovery-start-v1\\n':\n"
                      "    pathlib.Path(%r).write_text('started')\n" % str(marker), encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    command = owner.child_command(sys._base_executable, prefix, source, digest, config, validation_only=True)
    stopped = subprocess.run(command, input=b'', capture_output=True, timeout=20)
    assert stopped.returncode == 0 and not marker.exists()
    granted = subprocess.run(command, input=child._START, capture_output=True, timeout=20)
    assert granted.returncode == 0 and marker.read_text() == 'started'
    marker.unlink()
    wrong = owner.child_command(sys._base_executable, prefix, source, '0' * 64, config, validation_only=True)
    rejected = subprocess.run(wrong, input=child._START, capture_output=True, timeout=20)
    assert rejected.returncode != 0 and not marker.exists()
    assert not list(prefix.iterdir())


def test_private_supervisor_log_limit_does_not_mutate_original():
    from looking_glass import natural_heldout_supervisor as supervisor
    private = owner.bounded_supervisor(supervisor)
    assert supervisor._LOG_LIMIT == 1024 * 1024
    assert private._LOG_LIMIT == 256 * 1024
    assert private.supervise_heldout.__globals__ is private.__dict__
    assert supervisor.supervise_heldout.__globals__ is supervisor.__dict__
    assert private._QUARANTINED is supervisor._QUARANTINED


@pytest.mark.skipif(os.name != 'nt', reason='requires the reviewed Windows Job')
def test_gated_windows_child_inherits_source_lease_and_is_reaped(tmp_path):
    from looking_glass import natural_heldout_supervisor as supervisor
    from scripts import number_cpu_outer, v2_source_guards as guards
    from tools import injection_heldout_owner as adapter

    source, config_path, plan_path = (tmp_path / n for n in ('fake-child.py', 'config.json', 'plan.json'))
    source.write_text(
        "import hashlib,json,pathlib,sys\n"
        "if sys.stdin.buffer.readline()!=b'MLG-natural-discovery-start-v1\\n': raise SystemExit(1)\n"
        "c=json.loads(pathlib.Path(sys.argv[-1]).read_bytes())\n"
        "p=hashlib.sha256(pathlib.Path(c['replay_plan_file']).read_bytes()).hexdigest()\n"
        "s=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()\n"
        "v={'schema':'mlg-injection-runtime-replay-child-postcheck-v1','status':'validated_no_model_forward',"
        "'plan_sha256':p,'heldout_runtime_sha256':%r,'child_source_sha256':s,'forward_count':0}\n"
        "pathlib.Path(c['postcheck_file']).write_bytes(json.dumps(v,sort_keys=True,separators=(',',':')).encode())\n"
        % child._HELDOUT_RUNTIME, encoding='utf-8')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
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
    plan_raw = child._canonical(plan)
    plan_path.write_bytes(plan_raw)
    config = {'schema': child.CONFIG_SCHEMA, **{key: str(tmp_path / key) for key in child.CONFIG_PATHS},
              'plan_url': child.TEST_PLAN_URL, 'replay_plan_file': str(plan_path),
              'postcheck_file': str(tmp_path / 'postcheck.json'), 'attempt_dir': str(tmp_path / 'attempt')}
    lease = guards.SourceFileLease(tmp_path, [source, plan_path])
    cache = guards.PrivateEmptyCache(tmp_path)
    state = {}
    output = tmp_path / 'supervisor'
    def precheck(acquired):
        config['pycache_prefix'] = str(cache.path)
        state['config_raw'] = child._canonical(config)
        config_path.write_bytes(state['config_raw'])
        config_lease = guards.SourceFileLease(tmp_path, [config_path])
        acquired.append(config_lease)
        config_lease.__enter__()
        return child._read(config_path, 65536) == state['config_raw']
    def launch(acquired):
        return owner.launch_child(child=child, child_path=source, child_sha256=source_sha,
            config_path=config_path, expected_config=state['config_raw'], plan_bytes=plan_raw,
            base_python=sys._base_executable, output=output, environment=os.environ.copy(),
            guard=guards, adapter=adapter, supervisor=supervisor, cpu_job_type=number_cpu_outer.WindowsJob,
            acquired=acquired, validation_only=True, timeout=30)
    def postcheck(result):
        return owner.check_supervisor_record(child, result['config'], result['plan_sha256'], source_sha,
            output, result['command'], result['verdict'], validation_only=True)
    result = owner.under_protection([lease, cache], precheck, launch, postcheck, retained=supervisor)
    assert result['status'] == 'validated_no_model_forward'
    assert not owner._QUARANTINED and not cache.path.exists()


@pytest.mark.skipif(os.name != 'nt', reason='requires the reviewed Windows Job')
def test_private_log_cap_stops_output_overflow_without_audit(tmp_path):
    from looking_glass import natural_heldout_supervisor as supervisor
    from scripts import number_cpu_outer, v2_source_guards as guards
    from tools import injection_heldout_owner as adapter
    private = owner.bounded_supervisor(supervisor)
    command = [sys._base_executable, '-I', '-S', '-B', '-c',
               "import sys; sys.stdin.buffer.readline(); sys.stdout.buffer.write(b'x'*400000); sys.stdout.flush()"]
    called = []
    result = adapter.supervise_with_cpu_job(command, tmp_path / 'out', os.environ.copy(),
        audit_result=lambda *args: called.append(True), supervisor=private, guard_module=guards,
        cpu_job_type=number_cpu_outer.WindowsJob, timeout=30)
    assert result['status'] == 'incomplete' and result['log_overflow'] is True and not called
    assert (tmp_path / 'out' / 'stdout.txt').stat().st_size <= 256 * 1024
    assert supervisor._LOG_LIMIT == 1024 * 1024


def test_private_acl_policy_rejects_extra_or_inherited_principals():
    user = 'S-1-5-21-1-2-3-1000'
    rule = lambda sid: {'sid': sid, 'type': 0, 'inheritance': 3, 'propagation': 0,
                        'inherited': False, 'rights': 0x1f01ff}
    snapshot = {'user': user, 'owner': user, 'protected': True,
                'rules': [rule(user), rule('S-1-5-18')]}
    owner._validate_private_acl(snapshot)
    altered = dict(snapshot, rules=snapshot['rules'] + [rule('S-1-1-0')])
    with pytest.raises(owner.ReplayOwnerError):
        owner._validate_private_acl(altered)
    altered = dict(snapshot, rules=[dict(snapshot['rules'][0], inherited=True), snapshot['rules'][1]])
    with pytest.raises(owner.ReplayOwnerError):
        owner._validate_private_acl(altered)


@pytest.mark.skipif(os.name != 'nt', reason='requires Windows ACLs')
def test_real_private_parent_acl_detects_an_added_everyone_grant(tmp_path):
    private = tmp_path / 'private'
    private.mkdir()
    user = owner._windows_acl_snapshot(private)['user']
    executable = str(Path(os.environ['WINDIR']) / 'System32' / 'icacls.exe')
    def icacls(*arguments):
        result = subprocess.run([executable, str(private), *arguments], capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr.decode(errors='replace')
    icacls('/inheritance:r', '/grant:r', '*' + user + ':(OI)(CI)(F)', '*S-1-5-18:(OI)(CI)(F)')
    icacls('/remove:g', '*S-1-5-32-544', '*S-1-3-4')
    info = private.stat()
    owner.verify_private_parent(private, (info.st_dev, info.st_ino))
    icacls('/grant:r', '*S-1-1-0:(OI)(CI)(RX)')
    with pytest.raises(owner.ReplayOwnerError):
        owner.verify_private_parent(private, (info.st_dev, info.st_ino))


@pytest.mark.skipif(os.name != 'nt', reason='requires Windows system-directory lookup')
def test_acl_reader_ignores_a_forged_windir(monkeypatch, tmp_path):
    monkeypatch.setenv('WINDIR', str(tmp_path))
    snapshot = owner._windows_acl_snapshot(tmp_path)
    assert snapshot['user'].startswith('S-1-')


def test_preclaim_handoff_binds_intent_diagnostic_and_guarded_row(monkeypatch, tmp_path):
    from tools import injection_replay_claim as claim
    workspace = tmp_path / 'workspace'
    parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
    diagnostic = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    ledger = workspace / 'work' / 'agreement-study-ledger-v5' / 'ledger.sqlite3'
    parent.mkdir(parents=True)
    diagnostic.mkdir()
    (diagnostic / 'log').write_bytes(b'no model')
    ledger.parent.mkdir()
    ledger.write_bytes(b'fake database')
    identity = lambda path: (path.stat().st_dev, path.stat().st_ino)
    parent_id, ledger_id = identity(parent), identity(ledger)
    plan_sha = '1' * 64
    phase = parent / ('injection-heldout-' + child._FREEZE)
    pilot = parent / 'runtime-compatibility-replay-v1'
    count, inventory_sha = claim._diagnostic_bytes(diagnostic)
    intent = {'schema': 'mlg-injection-replay-claim-intent-v1', 'plan_sha256': plan_sha,
              'ledger_path': str(ledger), 'phase_path': str(phase), 'pilot_path': str(pilot),
              'diagnostic_path': str(diagnostic), 'parent_identity': list(parent_id),
              'ledger_identity': list(ledger_id), 'diagnostic_actual_bytes': count,
              'diagnostic_inventory_sha256': inventory_sha}
    intent_raw = child._canonical(intent)
    (parent / 'replay-claim-intent.json').write_bytes(intent_raw)
    receipt = {'schema': 'mlg-injection-replay-claim-receipt-v1', 'status': 'preclaimed',
               'plan_sha256': plan_sha, 'intent_sha256': hashlib.sha256(intent_raw).hexdigest(),
               'ledger_path': str(ledger), 'phase_path': str(phase), 'pilot_path': str(pilot),
               'diagnostic_path': str(diagnostic), 'parent_identity': list(parent_id),
               'ledger_identity': list(ledger_id), **child._BUDGETS, 'diagnostic_actual_bytes': count,
               'diagnostic_inventory_sha256': inventory_sha, 'injection_claim_row_sha256': '2' * 64}
    receipt_raw = child._canonical(receipt)
    (parent / 'replay-claim-receipt.json').write_bytes(receipt_raw)
    config = {'attempt_dir': str(pilot / 'attempt'), 'postcheck_file': str(pilot / 'child-postcheck.json'),
              'claim_receipt_file': str(parent / 'replay-claim-receipt.json')}
    seen = []
    guard = SimpleNamespace(ClaimAnchor=lambda **kwargs: SimpleNamespace(**kwargs),
                            verify_existing_claim=lambda *args, **kwargs: seen.append(kwargs['anchor']) or True)
    monkeypatch.setattr(owner, 'verify_private_parent', lambda *args: True)
    kwargs = dict(child=child, claim=claim, guard=guard, workspace=workspace, config=config,
                  plan_sha256=plan_sha, receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
                  parent_identity=parent_id, ledger_identity=ledger_id, state_sha256='3' * 64,
                  history_sha256='4' * 64, scope_sha256='5' * 64, original_claims_sha256='6' * 64)
    result = owner.verify_preclaim_handoff(**kwargs)
    assert result['status'] == 'row_and_receipt_consistent_awaiting_observed_exit'
    assert seen[0].heldout_claim_sha256 == '2' * 64
    wrong_config = dict(config, claim_receipt_file=str(parent / 'other-receipt.json'))
    with pytest.raises(owner.ReplayOwnerError):
        owner.verify_preclaim_handoff(**dict(kwargs, config=wrong_config))
    guarded = guard.verify_existing_claim
    guard.verify_existing_claim = lambda *args, **kwargs: False
    with pytest.raises(owner.ReplayOwnerError):
        owner.verify_preclaim_handoff(**kwargs)
    guard.verify_existing_claim = guarded
    def corrupt_during_guard(*args, **kwargs):
        (parent / 'replay-claim-receipt.json').write_bytes(b'changed')
        return True
    guard.verify_existing_claim = corrupt_during_guard
    with pytest.raises(owner.ReplayOwnerError):
        owner.verify_preclaim_handoff(**kwargs)
    (parent / 'replay-claim-receipt.json').write_bytes(receipt_raw)
    guard.verify_existing_claim = guarded
    (diagnostic / 'log').write_bytes(b'changed!')
    with pytest.raises(owner.ReplayOwnerError):
        owner.verify_preclaim_handoff(**kwargs)
    (diagnostic / 'log').write_bytes(b'no model')
    with pytest.raises(owner.ReplayOwnerError):
        owner.verify_preclaim_handoff(**dict(kwargs, receipt_sha256='0' * 64))
    (parent / 'replay-claim-intent.json').write_bytes(b'corrupted')
    with pytest.raises(owner.ReplayOwnerError):
        owner.verify_preclaim_handoff(**kwargs)
