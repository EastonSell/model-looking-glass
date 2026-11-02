"""Compose a gated outer observer with one fixed successor pool query.

This is a library component, not a source-first launcher. Its dedicated caller
must authenticate these dependencies and the published operation before
import, supply complete source/config leases, hold on uncertain cleanup, and
independently observe the caller's own successful exit and reaping. A failure
before the invocation marker is also a stop without retry under the protocol.
"""
import hashlib
import os
from pathlib import Path
import signal
import sys
import threading


class OuterQueryError(ValueError):
    pass


_HOLD = threading.Event()


def _hold_uncertain(owner, observer):
    """Keep all source and process custody rooted after an uncertain return."""
    while owner._QUARANTINED or observer._QUARANTINED:
        try:
            _HOLD.wait()
        except BaseException:
            continue


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def observe_query(*, query, handoff, observer, guards, owner, accounting, acl,
                  config_path, config_sha256, operation_sha256, interpreter,
                  interpreter_sha256, query_source_sha256, source_check, leases,
                  environment, timeout=300):
    """Observe the source-bound query parent and verify its complete handoff.

    The caller owns publication/currentness checks, complete source leases,
    private-root ACL policy, exclusion of concurrent output writers, and the
    trusted parent observation. A successful return is a process-record check,
    not a phase claim or scientific result.
    """
    if (not callable(source_check) or source_check() is not True or type(leases) is not list
            or not leases or type(environment) is not dict or environment.get('CUDA_VISIBLE_DEVICES') != '-1'
            or type(timeout) is not int or not 1 <= timeout <= 7200):
        raise OuterQueryError('bound sources, leases and CPU environment are required')
    config_raw = query._read(config_path, 65536)
    if _sha(config_raw) != query._hex(config_sha256):
        raise OuterQueryError('query configuration differs')
    config = query._json(config_raw)
    operation_raw = query._read(config['operation'], 1000000)
    if _sha(operation_raw) != query._hex(operation_sha256):
        raise OuterQueryError('query operation differs')
    operation = query._json(operation_raw)
    root = accounting._directory(config['root'])
    cache = accounting._directory(config['cache'])
    info = root.lstat()
    normalized = os.path.normcase(os.path.realpath(root))
    if (query._digest(normalized.encode('utf-8')) != operation.get('root_path_sha256')
            or type(operation.get('root_device')) is not int or type(operation.get('root_inode')) is not int
            or (info.st_dev, info.st_ino) != (operation['root_device'], operation['root_inode'])
            or acl.verify_private_parent(root, (info.st_dev, info.st_ino)) is not True
            or query._digest(query._read(config['sources']['caller'], 1000000)) != query_source_sha256
            or operation['sources']['caller']['sha256'] != query_source_sha256):
        raise OuterQueryError('query root, ACL or source differs')
    if (not Path(interpreter).is_absolute() or type(interpreter_sha256) is not str
            or query._digest(query._read(Path(interpreter).resolve(strict=True), 16 * 1024 * 1024)) != interpreter_sha256):
        raise OuterQueryError('query interpreter differs')
    marker = root / 'pool-query-invocation.json'
    output = root / 'pool-query-outer'
    terminal = root / 'pool-query-observation.json'
    def require_fresh_outputs():
        names = ('pool-query-outer', 'pool-query-observation.json', 'pool-query-claim.json',
                 'pool-query-receipt.json', 'pool-query-supervisor', 'phase-claim.json',
                 'nonce.bin', 'heldout-record.json')
        if any(os.path.lexists(root / name) for name in names):
            raise OuterQueryError('query or phase already started or is uncertain')
    if os.path.lexists(marker):
        raise OuterQueryError('query was already started or is uncertain')
    require_fresh_outputs()
    command = [interpreter, '-I', '-S', '-B', '-c', query._BOOTSTRAP,
               config['sources']['caller'], query_source_sha256, '--config', str(config_path),
               '--config-sha256', config_sha256, '--operation-sha256', operation_sha256]
    command_sha = _sha(query._encode(command))
    token = b'MLG-natural-discovery-start-v1\n'
    state = {}

    def unchanged():
        if (query._read(config_path, 65536) != config_raw
                or query._read(config['operation'], 1000000) != operation_raw
                or query._digest(query._read(config['sources']['caller'], 1000000)) != query_source_sha256
                or source_check() is not True):
            raise OuterQueryError('query source or configuration changed')
        current = root.lstat()
        if ((current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
                or acl.verify_private_parent(root, (info.st_dev, info.st_ino)) is not True):
            raise OuterQueryError('query root identity or ACL changed')
        return True

    def precheck():
        unchanged()
        accounting.account_phase(root, reserve_side_bytes=4 * 1024 * 1024)
        accounting.inspect_resources(root, cache)
        require_fresh_outputs()
        raw = query._encode({'schema': 'mlg-natural-successor-query-invocation-v1',
                             'operation_sha256': operation_sha256, 'config_sha256': config_sha256,
                             'query_source_sha256': query_source_sha256, 'command_sha256': command_sha,
                             'pool_input_sha256': operation['pool_input_sha256'],
                             'old_commitment': operation['old_commitment']})
        query._write_new(marker, raw)
        if query._read(marker, 8192) != raw:
            raise OuterQueryError('query invocation marker differs')
        require_fresh_outputs()
        state['marker_sha256'] = _sha(raw)

    def terminal_check(raw, purpose, parent_pid, _birth):
        inner = handoff.verify_inner(config_path, config_sha256, operation_sha256,
            interpreter=interpreter, interpreter_sha256=interpreter_sha256,
            bootstrap=query._BOOTSTRAP, query_source_sha256=query_source_sha256)
        return (purpose == 'preparation' and type(parent_pid) is int and parent_pid != inner['child_pid']
                and _sha(raw) == inner['observation_sha256'])

    def launch(acquired):
        handles = [handle for lease in acquired for handle in lease.handles()]
        def protect(process):
            return bool(guards.duplicate_to_process(handles, process))
        result = observer.observe_supervisor(purpose='preparation', command=command, output=output,
            environment=environment, config_path=config_path, config_sha256=config_sha256,
            operation_path=config['operation'], operation_sha256=operation_sha256,
            command_sha256=command_sha, supervisor_source_sha256=query_source_sha256,
            terminal_path=terminal, start_token=token, timeout=timeout,
            source_check=unchanged, protect_process=protect, terminal_check=terminal_check)
        return {'cleanup_verified': True, 'outer': result}

    def postcheck(verdict):
        unchanged()
        result = handoff.verify_outer(config_path, config_sha256, operation_sha256,
            interpreter=interpreter, interpreter_sha256=interpreter_sha256,
            bootstrap=query._BOOTSTRAP, query_source_sha256=query_source_sha256)
        saved_outer = query._read(output / 'observation.json', 8192)
        if (type(verdict.get('outer')) is not dict or query._encode(verdict['outer']) != saved_outer
                or _sha(saved_outer) != result['outer_observation_sha256']
                or result['invocation_sha256'] != state['marker_sha256']
                or _sha(query._read(marker, 8192)) != state['marker_sha256']):
            raise OuterQueryError('query outer handoff differs')
        accounting.account_phase(root, reserve_side_bytes=65536)
        accounting.inspect_resources(root, cache)
        return {'status': 'clear', 'invocation_sha256': state['marker_sha256'], **result}

    previous = owner._ignore_console_interrupts()
    try:
        try:
            return owner.under_source_leases(leases, precheck, launch, postcheck, retained=state)
        except BaseException:
            _hold_uncertain(owner, observer)
            raise
    finally:
        if not owner._QUARANTINED and not observer._QUARANTINED:
            for number, old in reversed(previous):
                signal.signal(number, old)
