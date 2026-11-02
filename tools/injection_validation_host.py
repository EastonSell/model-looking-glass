"""Last retained-handle observer for the no-model injection rehearsal.

This is a one-shot diagnostic, not a claim, replay, or evaluation launcher.
The invoking process must verify this source before execution and separately
observe this host's own exit. Any uncertain child cleanup retains custody.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType


# This host is prospectively published for this single reviewed plan. These
# constants are checked before any caller-supplied driver code is executed.
_DRIVER_SHA256 = '188148128d643dab11e85fe4ea01e77859049f2718b1d47faf5b2b1d7bb02209'
_PLAN_SHA256 = '051f8263c0c2630fb6438b28dfebc9d8c4da644f727d4938bfa0d3904117512c'


def check_root_terminal(driver, context, config, child, owner, raw, root_pid, root_birth):
    """Rebuild the operator observation and check the complete nested chain."""
    driver._process_identity((root_pid, root_birth))
    run = Path(config['run_dir'])
    config_raw = driver._ordinary_source(str(run / 'parent-config.json'), 65536)
    command_raw = driver._ordinary_source(str(run / 'operator-command.json'), 8192)
    driver.verify_parent_command(command_raw, context, owner, str(run), 'validation',
                                 str(run / 'parent-config.json'), role='operator')
    terminal_raw = driver._ordinary_source(str(run / 'validation-exit.json'), 8192)
    operator = driver._canonical_object(raw, 8192)
    operator_pid, operator_birth = driver._process_identity(
        (operator.get('parent_pid'), operator.get('parent_birth_100ns')))
    inner = driver._canonical_object(driver._ordinary_source(
        str(run / 'parent-observer' / 'observation.json'), 8192), 8192)
    outer = driver._canonical_object(driver._ordinary_source(
        str(run / 'observer-observer' / 'observation.json'), 8192), 8192)
    _, supervisor = driver._validation_supervisor(config)
    if root_pid in (operator_pid, inner.get('parent_pid'), outer.get('parent_pid'), supervisor['child_pid']):
        raise ValueError('the validation root and nested process identities overlap')
    expected = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
        'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
        'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
        'parent_pid': operator_pid, 'parent_birth_100ns': operator_birth,
        'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'operation_sha256': hashlib.sha256(context['plan_bytes']).hexdigest(),
        'command_sha256': hashlib.sha256(command_raw).hexdigest(),
        'supervisor_source_sha256': context['plan']['sources']['driver'],
        'supervisor_terminal_sha256': hashlib.sha256(terminal_raw).hexdigest()}
    if (raw != driver.canonical(expected)
            or driver.check_validation_exit_receipt(context, config, child, owner, terminal_raw,
                                                     operator_pid, operator_birth) is not True):
        raise ValueError('the retained operator evidence differs')
    return True


def check_host_terminal(driver, context, config, child, owner, raw, host_pid, host_birth, host_sha256):
    """Recheck saved bytes against an independently retained host identity.

    This verifies consistency only. The caller must itself observe a zero exit
    and reap on the same retained host handle before using the result.
    """
    driver._process_identity((host_pid, host_birth))
    if type(host_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', host_sha256) is None:
        raise ValueError('the host source identity differs')
    run = Path(config['run_dir'])
    config_path = str(run / 'parent-config.json')
    config_raw = driver._ordinary_source(config_path, 65536)
    command_raw = driver._ordinary_source(str(run / 'root-command.json'), 8192)
    try:
        command = json.loads(command_raw, object_pairs_hook=driver._unique)
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('the saved root command differs') from error
    if (type(command) is not list or len(command) != 12 or driver.canonical(command) != command_raw
            or any(type(part) is not str or not part for part in command)
            or not command[5].startswith('pycache_prefix=')):
        raise ValueError('the saved root command differs')
    prefix = Path(command[5][len('pycache_prefix='):])
    if (not prefix.is_absolute() or prefix.parent != run or not prefix.name.startswith('v2-cache-')
            or '..' in prefix.parts):
        raise ValueError('the saved root prefix differs')
    expected_command = [str(sys._base_executable), '-I', '-S', '-B', '-X',
        'pycache_prefix=' + str(prefix), '-c', owner._BOOTSTRAP,
        context['source_locations']['driver']['path'], context['plan']['sources']['driver'],
        '--validation-root', config_path]
    if command != expected_command:
        raise ValueError('the saved root command is not source-bound')
    operator_raw = driver._ordinary_source(str(run / 'operator-observer' / 'observation.json'), 8192)
    obs_raw = driver._ordinary_source(str(run / 'root-observer' / 'observation.json'), 8192)
    observation = driver._canonical_object(obs_raw, 8192)
    root_pid, root_birth = driver._process_identity(
        (observation.get('parent_pid'), observation.get('parent_birth_100ns')))
    expected_observation = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
        'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
        'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
        'parent_pid': root_pid, 'parent_birth_100ns': root_birth,
        'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'operation_sha256': hashlib.sha256(context['plan_bytes']).hexdigest(),
        'command_sha256': hashlib.sha256(command_raw).hexdigest(),
        'supervisor_source_sha256': context['plan']['sources']['driver'],
        'supervisor_terminal_sha256': hashlib.sha256(operator_raw).hexdigest()}
    operator = driver._canonical_object(operator_raw, 8192)
    inner = driver._canonical_object(driver._ordinary_source(
        str(run / 'parent-observer' / 'observation.json'), 8192), 8192)
    outer = driver._canonical_object(driver._ordinary_source(
        str(run / 'observer-observer' / 'observation.json'), 8192), 8192)
    _, supervisor = driver._validation_supervisor(config)
    if (host_pid in (root_pid, operator.get('parent_pid'), inner.get('parent_pid'),
                     outer.get('parent_pid'), supervisor['child_pid'])
            or obs_raw != driver.canonical(expected_observation)
            or check_root_terminal(driver, context, config, child, owner,
                                   operator_raw, root_pid, root_birth) is not True):
        raise ValueError('the saved root observation differs')
    expected_terminal = {'schema': 'mlg-injection-validation-host-terminal-v1',
        'status': 'awaiting_host_exit', 'host_pid': host_pid, 'host_birth_100ns': host_birth,
        'root_pid': root_pid, 'root_birth_100ns': root_birth, 'host_source_sha256': host_sha256,
        'driver_source_sha256': context['plan']['sources']['driver'],
        'plan_sha256': hashlib.sha256(context['plan_bytes']).hexdigest(),
        'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'root_observation_sha256': hashlib.sha256(obs_raw).hexdigest(), 'forward_count': 0}
    if type(raw) is not bytes or raw != driver.canonical(expected_terminal):
        raise ValueError('the observed host terminal differs')
    return True


def _load_driver(path, expected):
    if (type(path) is not str or not Path(path).is_absolute() or type(expected) is not str
            or re.fullmatch(r'[0-9a-f]{64}', expected) is None):
        raise ValueError('the validation driver identity differs')
    with open(path, 'rb') as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('the validation driver bytes differ')
    module = ModuleType('_mlg_validation_driver')
    module.__file__ = path
    sys.modules[module.__name__] = module
    try:
        exec(compile(raw, path, 'exec', dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        del sys.modules[module.__name__]
        raise
    return module


def run(config_path, driver_path, driver_sha256, host_url, host_sha256, plan_sha256):
    """Run one gated root while retaining its source, assets and process."""
    flags = sys.flags
    if (not (flags.isolated and flags.ignore_environment and flags.no_site and flags.no_user_site
             and flags.safe_path and sys.dont_write_bytecode and flags.optimize == 0)
            or Path(sys.executable).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)):
        raise ValueError('the validation host is not isolated')
    if (type(driver_sha256) is not str or driver_sha256 != _DRIVER_SHA256
            or type(plan_sha256) is not str or plan_sha256 != _PLAN_SHA256):
        raise ValueError('the predeclared driver or plan identity differs')
    driver = _load_driver(driver_path, driver_sha256)
    own_path = str(Path(__file__).resolve(strict=True))
    if (type(host_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', host_sha256) is None
            or type(plan_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', plan_sha256) is None
            or type(host_url) is not str or re.fullmatch(re.escape(driver.RAW_PREFIX)
                + r'[0-9a-f]{40}/tools/injection_validation_host\.py', host_url) is None):
        raise ValueError('the published host identity differs')
    own_raw = driver._ordinary_source(own_path)
    if hashlib.sha256(own_raw).hexdigest() != host_sha256 or driver._fetch_source(host_url) != own_raw:
        raise ValueError('the published host bytes differ')
    config_raw = driver._ordinary_source(config_path, 65536)
    config = driver.parse_validation_parent_config(config_raw, config_path)
    workspace, run_dir = Path(config['workspace']), Path(config['run_dir'])
    context = driver.capture_preflight(plan_path=config['plan_path'], plan_url=config['plan_url'],
        source_locations=config['source_locations'], public_paths=config['public_paths'],
        frozen_root=config['frozen_root'], cache_dir=config['cache_dir'])
    if (context['plan']['sources']['driver'] != driver_sha256
            or hashlib.sha256(context['plan_bytes']).hexdigest() != plan_sha256):
        raise ValueError('the reviewed driver or plan differs')
    diagnostic = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    environment = dict(os.environ)
    environment_raw = driver.canonical(environment)

    def execute(finder):
        modules = driver._execution_modules(finder)
        owner, child, guard, observer = (modules[k] for k in ('owner', 'child', 'guard', 'outer_observer'))
        import importlib
        claim = importlib.import_module('tools.injection_replay_claim')
        if (getattr(getattr(claim, '__spec__', None), 'loader', None) is not finder
                or finder.assert_intact() is not True):
            raise ValueError('the diagnostic accounting source differs')
        parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
        def authority():
            if (driver._ordinary_source(config_path, 65536) != config_raw
                    or driver._ordinary_source(own_path) != own_raw
                    or driver.canonical(environment) != environment_raw
                    or owner.verify_private_parent(parent, tuple(config['parent_identity'])) is not True
                    or driver.check_unclaimed_validation(str(workspace), tuple(config['ledger_identity'])) is not True):
                return False
            return claim._diagnostic_bytes(diagnostic)[0] <= 4 * 1024 * 1024
        if claim._diagnostic_bytes(diagnostic)[0] > 1024 * 1024 or authority() is not True:
            raise ValueError('the validation diagnostic reservation differs')
        frozen, operational, assets = driver.lease_inputs(context, str(workspace),
                                                           extra_files=(config_path, own_path))
        leases = [guard.SourceFileLease(workspace, files) for files in (frozen, operational, assets)]
        cache = guard.PrivateEmptyCache(run_dir)
        leases.append(cache)
        state = {}
        command_path = str(run_dir / 'root-command.json')
        terminal_path = str(run_dir / 'operator-observer' / 'observation.json')
        output = str(run_dir / 'root-observer')
        def source_check():
            return (driver.recheck_preflight(context, finder) is True and authority() is True
                    and driver.check_runtime_environment(environment, str(workspace), context['cache_dir'],
                        driver._canonical_object(context['public']['heldout_runtime'], 1024 * 1024)) is True
                    and (state.get('command_raw') is None
                         or driver._ordinary_source(command_path, 8192) == state['command_raw']))
        def precheck(acquired):
            if source_check() is not True:
                raise ValueError('the validation host authority differs')
            command = [str(sys._base_executable), '-I', '-S', '-B', '-X',
                'pycache_prefix=' + str(cache.path), '-c', owner._BOOTSTRAP, driver_path,
                driver_sha256, '--validation-root', config_path]
            command_raw = driver.canonical(command)
            if len(command_raw) > 8192 or sum(len(part) for part in command) > 16000:
                raise ValueError('the validation root command exceeds its bound')
            driver._write_once(command_path, command_raw)
            lease = guard.SourceFileLease(workspace, [Path(command_path)])
            acquired.append(lease)
            lease.__enter__()
            state.update({'command': command, 'command_raw': command_raw})
            return source_check() is True
        def launch(acquired):
            def protect(process):
                handles = tuple(handle for lease in acquired for handle in lease.handles())
                duplicated = guard.duplicate_to_process(handles, process)
                return (type(duplicated) is tuple and len(duplicated) == len(handles)
                        and all(type(handle) is int and handle > 0 for handle in duplicated))
            observed = observer.observe_supervisor(purpose='execution', command=state['command'], output=output,
                environment=environment, config_path=config_path,
                config_sha256=hashlib.sha256(config_raw).hexdigest(), operation_path=context['plan_path'],
                operation_sha256=plan_sha256, command_sha256=hashlib.sha256(state['command_raw']).hexdigest(),
                supervisor_source_sha256=driver_sha256, terminal_path=terminal_path,
                start_token=driver.START_TOKEN, timeout=7200, source_check=source_check,
                protect_process=protect, terminal_check=lambda raw, purpose, pid, birth:
                    purpose == 'execution' and check_root_terminal(driver, context, config, child, owner, raw, pid, birth))
            return {'cleanup_verified': True, 'observation': observed}
        def postcheck(result):
            raw = driver._ordinary_source(str(Path(output) / 'observation.json'), 8192)
            if raw != driver.canonical(result['observation']) or source_check() is not True:
                raise ValueError('the retained root observation differs')
            host_pid, host_birth = driver._process_identity(owner._current_process_identity())
            if host_pid == result['observation']['parent_pid']:
                raise ValueError('the host and root identities overlap')
            record = {'schema': 'mlg-injection-validation-host-terminal-v1',
                'status': 'awaiting_host_exit', 'host_pid': host_pid, 'host_birth_100ns': host_birth,
                'root_pid': result['observation']['parent_pid'],
                'root_birth_100ns': result['observation']['parent_birth_100ns'],
                'host_source_sha256': host_sha256, 'driver_source_sha256': driver_sha256,
                'plan_sha256': plan_sha256, 'config_sha256': hashlib.sha256(config_raw).hexdigest(),
                'root_observation_sha256': hashlib.sha256(raw).hexdigest(), 'forward_count': 0}
            terminal = str(run_dir / 'host-terminal.json')
            driver._write_once(terminal, driver.canonical(record))
            if source_check() is not True or driver._ordinary_source(terminal, 8192) != driver.canonical(record):
                raise ValueError('the host terminal could not be reconciled')
            return record
        return owner.run_guarded(leases, precheck, launch, postcheck, retained=modules)
    return driver.run_bound_imports(context['graph'], execute)


def main(argv=None, stream=None):
    try:
        stream = sys.stdin.buffer if stream is None else stream
        if stream.readline(33) != b'MLG-natural-discovery-start-v1\n':
            raise ValueError('the validation start gate is required')
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) != 6:
            raise ValueError('the validation host arguments differ')
        run(*args)
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
