"""Verify retained query and outer-observer metadata before a successor phase.

The caller must source-bind this module, the operation and both bootstraps,
authenticate their official publication and current inventory, and establish
the observer process's own clean exit. Before phase claim it must hold leases
on all handoff files, source and configuration through re-verification and the
claim. These checks do not inspect private rows or prove execution against a
same-account adversary.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat


class HandoffError(ValueError):
    pass


_SHA = re.compile(r'[0-9a-f]{64}\Z')
_START = b'MLG-natural-discovery-start-v1\n'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise HandoffError('duplicate JSON key')
        result[key] = value
    return result


def _encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def _json(raw, *, canonical=True):
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
        encoded = _encode(value)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise HandoffError('invalid handoff JSON') from error
    if canonical and encoded != raw:
        raise HandoffError('handoff JSON is not canonical')
    return value


def _read(path, limit, *, empty=False):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise HandoffError('an absolute handoff path is required')
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise HandoffError('handoff path crosses a reparse point')
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or not (0 if empty else 1) <= before.st_size <= limit):
        raise HandoffError('handoff file is unsafe or oversized')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        data = stream.read(limit + 1)
        finished = os.fstat(stream.fileno())
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink)
    if (len(data) != before.st_size or identity(before) != identity(opened)
            or identity(before) != identity(finished) or identity(before) != identity(after)):
        raise HandoffError('handoff file changed while reading')
    return data


def _digest(value):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise HandoffError('a bound digest is required')
    return value


def _identity(root, operation):
    root = Path(root)
    if (not root.is_absolute() or '..' in root.parts
            or os.path.normcase(os.path.abspath(root)) != os.path.normcase(os.path.realpath(root))):
        raise HandoffError('successor root is unsafe')
    for part in (root, *root.parents):
        info = part.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise HandoffError('successor root crosses a link')
    info = root.lstat()
    path_hash = _sha(os.path.normcase(os.path.realpath(root)).encode('utf-8'))
    if (path_hash != operation.get('root_path_sha256')
            or type(operation.get('root_device')) is not int or type(operation.get('root_inode')) is not int
            or (info.st_dev, info.st_ino) != (operation['root_device'], operation['root_inode'])):
        raise HandoffError('successor root identity differs')
    return root


def verify_inner(config_path, config_sha256, operation_sha256, *, interpreter, interpreter_sha256,
                 bootstrap, query_source_sha256):
    """Recheck a clear query's claim, gated child, Job, logs and receipt.

    This reports consistency of retained records only. An independently
    observed successful outer caller exit is still required before phase claim.
    """
    config_sha256, operation_sha256, query_source_sha256, interpreter_sha256 = map(
        _digest, (config_sha256, operation_sha256, query_source_sha256, interpreter_sha256))
    if (type(interpreter) is not str or not Path(interpreter).is_absolute()
            or type(bootstrap) is not str or not bootstrap):
        raise HandoffError('the fixed interpreter and bootstrap are required')
    resolved_interpreter = Path(interpreter).resolve(strict=True)
    if (_sha(_read(resolved_interpreter, 16 * 1024 * 1024)) != interpreter_sha256
            or Path(interpreter).resolve(strict=True) != resolved_interpreter):
        raise HandoffError('the fixed interpreter differs')
    config_raw = _read(config_path, 65536)
    if _sha(config_raw) != config_sha256:
        raise HandoffError('query configuration differs')
    config = _json(config_raw)
    try:
        operation_raw = _read(config['operation'], 1000000)
        if _sha(operation_raw) != operation_sha256:
            raise HandoffError('query operation differs')
        operation = _json(operation_raw)
        root = _identity(config['root'], operation)
        query_path = Path(config['sources']['caller'])
        comparator_sha = _digest(operation['sources']['comparator']['sha256'])
        query_raw = _read(query_path, 1000000)
        if (operation['sources']['caller']['sha256'] != query_source_sha256
                or _sha(query_raw) != query_source_sha256):
            raise HandoffError('query source differs')
        tree = ast.parse(query_raw, filename=str(query_path))
        values = [node.value for node in tree.body if isinstance(node, ast.Assign)
                  and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == '_BOOTSTRAP']
        if len(values) != 1 or ast.literal_eval(values[0]) != bootstrap:
            raise HandoffError('gated bootstrap differs from the bound query source')
        if any(os.path.lexists(root / name) for name in ('phase-claim.json', 'nonce.bin', 'heldout-record.json')):
            raise HandoffError('query handoff must precede phase preparation')
        claim_raw = _read(root / 'pool-query-claim.json', 8192)
        claim = _json(claim_raw)
        expected_claim = {'schema': 'mlg-natural-successor-claim-v1', 'kind': 'pool-query',
                          'operation_sha256': operation_sha256, 'pool_input_sha256': operation['pool_input_sha256'],
                          'old_commitment': operation['old_commitment'], 'comparator_sha256': comparator_sha}
        if claim != expected_claim:
            raise HandoffError('query claim differs')
        receipt_raw = _read(root / 'pool-query-receipt.json', 8192)
        receipt = _json(receipt_raw)
        keys = {'schema', 'status', 'comparison', 'natural_heldout_commitment',
                'natural_prediction_freeze_sha256', 'public_input_file_sha256', 'natural_case_count',
                'public_case_count', 'comparator_source_sha256', 'observed_utc'}
        if (type(receipt) is not dict or set(receipt) != keys
                or receipt['schema'] != 'mlg-natural-external-input-comparison-v1'
                or receipt['status'] != 'clear' or receipt['comparison'] != 'bidirectional-complete-row-prefix'
                or receipt['natural_heldout_commitment'] != operation['old_commitment']
                or receipt['natural_prediction_freeze_sha256'] != operation['freeze_sha256']
                or receipt['public_input_file_sha256'] != operation['pool_input_sha256']
                or type(receipt['natural_case_count']) is not int or receipt['natural_case_count'] != 16
                or type(receipt['public_case_count']) is not int or receipt['public_case_count'] != 256
                or receipt['comparator_source_sha256'] != comparator_sha
                or type(receipt['observed_utc']) is not str or not 1 <= len(receipt['observed_utc']) <= 64):
            raise HandoffError('a matching clear pool receipt is required')
        output = root / 'pool-query-supervisor'
        stdout = _read(output / 'stdout.txt', 1024 * 1024, empty=True)
        stderr = _read(output / 'stderr.txt', 1024 * 1024, empty=True)
        if stdout not in (b'clear\n', b'clear\r\n') or stderr:
            raise HandoffError('query logs differ')
        supervisor_raw = _read(output / 'supervisor.json', 1000000)
        supervisor = _json(supervisor_raw, canonical=False)
        command = [interpreter, '-I', '-S', '-B', '-c', bootstrap,
                   config['sources']['comparator'], comparator_sha,
                   '--phase', config['old_phase'], '--supervisor', config['old_supervisor'],
                   '--proposal', config['inputs'], '--pins', config['pins'],
                   '--pins-sha256', operation['pins_sha256'], '--receipt', str(root / 'pool-query-receipt.json')]
        if (type(supervisor) is not dict or supervisor.get('status') != 'incomplete'
                or supervisor.get('command') != command
                or type(supervisor.get('exit_code')) is not int or supervisor['exit_code'] != 0
                or any(supervisor.get(key) is not True for key in
                       ('cleanup_verified', 'child_reaped', 'job_drained', 'job_closed', 'logs_complete'))
                or type(supervisor.get('postexit_active_processes')) is not int
                or supervisor['postexit_active_processes'] != 0
                or supervisor.get('supervisor_error') is not None or supervisor.get('log_overflow') is not False
                or supervisor.get('audit_error_type') is not None
                or supervisor.get('cancellation_requested') is not False or supervisor.get('quarantined') is not False
                or supervisor.get('retry_safe') is not False
                or supervisor.get('stdout_sha256') != _sha(stdout) or supervisor.get('stderr_sha256') != _sha(stderr)
                or _encode(supervisor.get('audit')) != _encode({'status': 'incomplete', 'query_result': 'clear',
                                                                'receipt_sha256': _sha(receipt_raw)})
                or _encode(supervisor.get('retained_tensor_files')) != _encode(
                    {'accounting_complete': True, 'total_bytes': 0, 'files': []})
                or type(supervisor.get('child_pid')) is not int or not 0 < supervisor['child_pid'] <= 0xffffffff):
            raise HandoffError('query supervisor did not complete cleanly')
        observation_raw = _read(root / 'pool-query-observation.json', 65536)
        observation = _json(observation_raw)
        observation_keys = {'schema', 'status', 'query_claim_sha256', 'receipt_sha256', 'config_sha256',
                            'operation_sha256', 'comparator_sha256', 'supervisor_sha256', 'pre_head', 'post_head',
                            'command', 'pid', 'exit_code', 'reaped', 'job_drained', 'job_closed',
                            'stdout_sha256', 'stderr_sha256'}
        if (type(observation) is not dict or set(observation) != observation_keys
                or observation['schema'] != 'mlg-natural-successor-query-observation-v1'
                or observation['status'] != 'clear' or observation['query_claim_sha256'] != _sha(claim_raw)
                or observation['receipt_sha256'] != _sha(receipt_raw) or observation['config_sha256'] != config_sha256
                or observation['operation_sha256'] != operation_sha256 or observation['comparator_sha256'] != comparator_sha
                or observation['supervisor_sha256'] != _sha(supervisor_raw) or observation['command'] != command
                or type(observation['pid']) is not int or observation['pid'] != supervisor['child_pid']
                or type(observation['exit_code']) is not int or observation['exit_code'] != 0
                or any(observation[key] is not True for key in ('reaped', 'job_drained', 'job_closed'))
                or observation['stdout_sha256'] != _sha(stdout) or observation['stderr_sha256'] != _sha(stderr)
                or any(type(observation[key]) is not str or re.fullmatch(r'[0-9a-f]{40}', observation[key]) is None
                       for key in ('pre_head', 'post_head'))):
            raise HandoffError('query observation differs')
        _identity(root, operation)
        for path, before, limit, empty in (
            (config_path, config_raw, 65536, False), (config['operation'], operation_raw, 1000000, False),
            (query_path, query_raw, 1000000, False), (root / 'pool-query-claim.json', claim_raw, 8192, False),
            (root / 'pool-query-receipt.json', receipt_raw, 8192, False),
            (output / 'stdout.txt', stdout, 1024 * 1024, True),
            (output / 'stderr.txt', stderr, 1024 * 1024, True),
            (output / 'supervisor.json', supervisor_raw, 1000000, False),
            (root / 'pool-query-observation.json', observation_raw, 65536, False),
        ):
            if _read(path, limit, empty=empty) != before:
                raise HandoffError('query handoff changed during verification')
        if Path(interpreter).resolve(strict=True) != resolved_interpreter:
            raise HandoffError('the interpreter path changed during verification')
        return {'claim_sha256': _sha(claim_raw), 'receipt_sha256': _sha(receipt_raw),
                'observation_sha256': _sha(observation_raw), 'child_pid': observation['pid']}
    except (KeyError, TypeError, OSError, RecursionError) as error:
        raise HandoffError('query handoff is incomplete') from error


def verify_outer(config_path, config_sha256, operation_sha256, *, interpreter, interpreter_sha256,
                 bootstrap, query_source_sha256):
    """Check the invocation marker, outer observation and exact gated command.

    The process that produced the outer observation must itself be independently
    observed exiting successfully and reaped by its trusted caller. The marker
    and every handoff file must remain leased through the later phase claim.
    """
    inner = verify_inner(config_path, config_sha256, operation_sha256, interpreter=interpreter,
                         interpreter_sha256=interpreter_sha256, bootstrap=bootstrap,
                         query_source_sha256=query_source_sha256)
    config_raw = _read(config_path, 65536)
    if _sha(config_raw) != _digest(config_sha256):
        raise HandoffError('query configuration differs')
    config = _json(config_raw)
    operation_raw = _read(config['operation'], 1000000)
    if _sha(operation_raw) != _digest(operation_sha256):
        raise HandoffError('query operation differs')
    operation = _json(operation_raw)
    root = _identity(config['root'], operation)
    command = [interpreter, '-I', '-S', '-B', '-c', bootstrap, config['sources']['caller'],
               _digest(query_source_sha256), '--config', str(config_path), '--config-sha256', config_sha256,
               '--operation-sha256', operation_sha256]
    command_sha = _sha(_encode(command))
    invocation_raw = _read(root / 'pool-query-invocation.json', 8192)
    expected_invocation = {'schema': 'mlg-natural-successor-query-invocation-v1',
                           'operation_sha256': operation_sha256, 'config_sha256': config_sha256,
                           'query_source_sha256': query_source_sha256, 'command_sha256': command_sha,
                           'pool_input_sha256': operation['pool_input_sha256'],
                           'old_commitment': operation['old_commitment']}
    if _json(invocation_raw) != expected_invocation:
        raise HandoffError('outer query invocation marker differs')
    folder = root / 'pool-query-outer'
    started_raw = _read(folder / 'started.json', 8192)
    started = _json(started_raw)
    expected_start = {'schema': 'mlg-injection-outer-process-start-v1', 'purpose': 'preparation',
                      'config_sha256': config_sha256, 'operation_sha256': operation_sha256,
                      'command_sha256': command_sha, 'supervisor_source_sha256': query_source_sha256,
                      'start_token_sha256': _sha(_START)}
    if started != expected_start:
        raise HandoffError('outer query start record differs')
    raw = _read(folder / 'observation.json', 8192)
    observation = _json(raw)
    keys = {'schema', 'purpose', 'status', 'exit_code', 'parent_reaped', 'parent_handle_closed',
            'outer_job_drained', 'outer_job_closed', 'active_job_members', 'parent_pid', 'parent_birth_100ns',
            'config_sha256', 'operation_sha256', 'command_sha256', 'supervisor_source_sha256',
            'supervisor_terminal_sha256'}
    if (type(inner) is not dict or type(inner.get('child_pid')) is not int
            or type(observation) is not dict or set(observation) != keys
            or observation['schema'] != 'mlg-injection-outer-process-observation-v1'
            or observation['purpose'] != 'preparation' or observation['status'] != 'completed'
            or type(observation['exit_code']) is not int or observation['exit_code'] != 0
            or any(observation[key] is not True for key in ('parent_reaped', 'parent_handle_closed',
                                                             'outer_job_drained', 'outer_job_closed'))
            or type(observation['active_job_members']) is not int or observation['active_job_members'] != 0
            or type(observation['parent_pid']) is not int or not 0 < observation['parent_pid'] <= 0xffffffff
            or observation['parent_pid'] == inner['child_pid']
            or type(observation['parent_birth_100ns']) is not int
            or not 0 < observation['parent_birth_100ns'] < 2 ** 64
            or observation['config_sha256'] != config_sha256 or observation['operation_sha256'] != operation_sha256
            or observation['command_sha256'] != command_sha or observation['supervisor_source_sha256'] != query_source_sha256
            or observation['supervisor_terminal_sha256'] != inner.get('observation_sha256')):
        raise HandoffError('outer query process observation differs')
    _identity(root, operation)
    checked_again = verify_inner(config_path, config_sha256, operation_sha256, interpreter=interpreter,
                                 interpreter_sha256=interpreter_sha256, bootstrap=bootstrap,
                                 query_source_sha256=query_source_sha256)
    if (checked_again != inner or _read(folder / 'started.json', 8192) != started_raw
            or _read(folder / 'observation.json', 8192) != raw
            or _read(root / 'pool-query-invocation.json', 8192) != invocation_raw):
        raise HandoffError('outer query handoff changed during verification')
    return {'outer_observation_sha256': _sha(raw), 'parent_pid': observation['parent_pid'],
            'parent_birth_100ns': observation['parent_birth_100ns'],
            'invocation_sha256': _sha(invocation_raw), **inner}
