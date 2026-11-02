"""One-shot retained-handle custodian for a no-model validation host.

The caller must hash-verify this source before execution and separately
retain and reap this custodian. It does not claim a phase or run a model.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType


DRIVER_SHA = '188148128d643dab11e85fe4ea01e77859049f2718b1d47faf5b2b1d7bb02209'
HOST_SHA = '2e60b419761aad5ee42e1ab45b787db12088e751855319ca63da6a08e4ed61c7'
PLAN_SHA = '051f8263c0c2630fb6438b28dfebc9d8c4da644f727d4938bfa0d3904117512c'
HOST_URL = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
            'cb316aa900ee51c5e8ec682de7a08db46dad33e3/tools/injection_validation_host.py')
START = b'MLG-natural-discovery-start-v1\n'


def load(path, digest, name):
    if (type(path) is not str or not Path(path).is_absolute() or type(digest) is not str
            or re.fullmatch(r'[0-9a-f]{64}', digest) is None or name in sys.modules):
        raise ValueError('the source identity differs')
    with open(path, 'rb') as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('the reviewed source differs')
    module = ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    try:
        exec(compile(raw, path, 'exec', dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def check_custodian_terminal(driver, host, context, config, child, owner, raw,
                             custodian_pid, custodian_birth, custodian_sha256, host_path):
    """Check saved evidence against the caller's retained custodian identity.

    The caller must separately observe zero exit and reap on that same handle.
    """
    driver._process_identity((custodian_pid, custodian_birth))
    if (type(custodian_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', custodian_sha256) is None
            or type(host_path) is not str or not Path(host_path).is_absolute()):
        raise ValueError('the custodian source identity differs')
    run = Path(config['run_dir'])
    config_path = str(run / 'parent-config.json')
    config_raw = driver._ordinary_source(config_path, 65536)
    command_raw = driver._ordinary_source(str(run / 'custodian-host-command.json'), 8192)
    try:
        command = json.loads(command_raw, object_pairs_hook=driver._unique)
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('the saved host command differs') from error
    if (type(command) is not list or len(command) != 16 or driver.canonical(command) != command_raw
            or any(type(part) is not str or not part for part in command)
            or not command[5].startswith('pycache_prefix=')):
        raise ValueError('the saved host command differs')
    prefix = Path(command[5][len('pycache_prefix='):])
    if (not prefix.is_absolute() or prefix.parent != run or not prefix.name.startswith('v2-cache-')
            or '..' in prefix.parts):
        raise ValueError('the saved host prefix differs')
    expected_command = [str(sys._base_executable), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
        '-c', owner._BOOTSTRAP, host_path, HOST_SHA, config_path,
        context['source_locations']['driver']['path'], DRIVER_SHA, HOST_URL, HOST_SHA, PLAN_SHA]
    if command != expected_command:
        raise ValueError('the host command is not source-bound')
    value = driver._canonical_object(raw, 8192)
    host_pid, host_birth = driver._process_identity((value.get('host_pid'), value.get('host_birth_100ns')))
    host_raw = driver._ordinary_source(str(run / 'host-terminal.json'), 8192)
    descendants = [host_pid]
    for name in ('root-observer', 'operator-observer', 'observer-observer', 'parent-observer'):
        observation = driver._canonical_object(driver._ordinary_source(
            str(run / name / 'observation.json'), 8192), 8192)
        descendants.append(observation.get('parent_pid'))
    _, supervisor = driver._validation_supervisor(config)
    descendants.append(supervisor['child_pid'])
    if (custodian_pid in descendants
            or host.check_host_terminal(driver, context, config, child, owner, host_raw,
                                        host_pid, host_birth, HOST_SHA) is not True):
        raise ValueError('the host process evidence differs')
    expected = {'schema': 'mlg-injection-validation-custodian-terminal-v1',
        'status': 'awaiting_custodian_exit', 'custodian_pid': custodian_pid,
        'custodian_birth_100ns': custodian_birth, 'host_pid': host_pid, 'host_birth_100ns': host_birth,
        'host_terminal_sha256': hashlib.sha256(host_raw).hexdigest(),
        'host_command_sha256': hashlib.sha256(command_raw).hexdigest(), 'host_source_sha256': HOST_SHA,
        'driver_source_sha256': DRIVER_SHA, 'plan_sha256': PLAN_SHA,
        'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'custodian_source_sha256': custodian_sha256, 'forward_count': 0}
    if raw != driver.canonical(expected):
        raise ValueError('the observed custodian terminal differs')
    return True


def run(config_path, host_path, driver_path):
    flags = sys.flags
    if (not (flags.isolated and flags.ignore_environment and flags.no_site and flags.no_user_site
             and flags.safe_path and sys.dont_write_bytecode and flags.optimize == 0)
            or Path(sys.executable).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)):
        raise ValueError('the validation custodian is not isolated')
    driver = load(driver_path, DRIVER_SHA, '_mlg_custodian_driver')
    host = load(host_path, HOST_SHA, '_mlg_custodian_host')
    own_path = str(Path(__file__).resolve(strict=True))
    own_raw = driver._ordinary_source(own_path)
    host_raw = driver._ordinary_source(host_path)
    if driver._fetch_source(HOST_URL) != host_raw:
        raise ValueError('the published host differs')
    config_raw = driver._ordinary_source(config_path, 65536)
    config = driver.parse_validation_parent_config(config_raw, config_path)
    workspace, run_dir = Path(config['workspace']), Path(config['run_dir'])
    context = driver.capture_preflight(plan_path=config['plan_path'], plan_url=config['plan_url'],
        source_locations=config['source_locations'], public_paths=config['public_paths'],
        frozen_root=config['frozen_root'], cache_dir=config['cache_dir'])
    if (context['plan']['sources']['driver'] != DRIVER_SHA
            or hashlib.sha256(context['plan_bytes']).hexdigest() != PLAN_SHA):
        raise ValueError('the predeclared plan differs')
    environment = dict(os.environ)
    environment_raw = driver.canonical(environment)
    diagnostic = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    def execute(finder):
        modules = driver._execution_modules(finder)
        owner, child, guard, observer = (modules[k] for k in ('owner', 'child', 'guard', 'outer_observer'))
        import importlib
        claim = importlib.import_module('tools.injection_replay_claim')
        if (getattr(getattr(claim, '__spec__', None), 'loader', None) is not finder
                or finder.assert_intact() is not True):
            raise ValueError('the diagnostic accounting source differs')
        parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
        def source_check():
            return (driver._ordinary_source(config_path, 65536) == config_raw
                    and driver._ordinary_source(host_path) == host_raw
                    and driver._ordinary_source(own_path) == own_raw
                    and driver.canonical(environment) == environment_raw
                    and driver.recheck_preflight(context, finder) is True
                    and driver.check_runtime_environment(environment, str(workspace), context['cache_dir'],
                        driver._canonical_object(context['public']['heldout_runtime'], 1024 * 1024)) is True
                    and owner.verify_private_parent(parent, tuple(config['parent_identity'])) is True
                    and driver.check_unclaimed_validation(str(workspace), tuple(config['ledger_identity'])) is True
                    and claim._diagnostic_bytes(diagnostic)[0] <= 4 * 1024 * 1024)
        if claim._diagnostic_bytes(diagnostic)[0] > 1024 * 1024 or source_check() is not True:
            raise ValueError('the no-model diagnostic reservation differs')
        frozen, operational, assets = driver.lease_inputs(context, str(workspace),
            extra_files=(config_path, host_path, own_path))
        leases = [guard.SourceFileLease(workspace, paths) for paths in (frozen, operational, assets)]
        cache = guard.PrivateEmptyCache(run_dir)
        leases.append(cache)
        command_path = str(run_dir / 'custodian-host-command.json')
        state = {}
        def precheck(acquired):
            if source_check() is not True:
                raise ValueError('the custodian inputs differ')
            command = [str(sys._base_executable), '-I', '-S', '-B', '-X',
                'pycache_prefix=' + str(cache.path), '-c', owner._BOOTSTRAP, host_path, HOST_SHA,
                config_path, driver_path, DRIVER_SHA, HOST_URL, HOST_SHA, PLAN_SHA]
            raw = driver.canonical(command)
            if len(raw) > 8192 or sum(map(len, command)) > 16000:
                raise ValueError('the host command exceeds its bound')
            driver._write_once(command_path, raw)
            command_lease = guard.SourceFileLease(workspace, [Path(command_path)])
            acquired.append(command_lease)
            command_lease.__enter__()
            state.update({'command': command, 'command_raw': raw})
            return (driver._ordinary_source(command_path, 8192) == raw and source_check() is True)
        def launch(acquired):
            custody = next((item for item in owner._QUARANTINED if item['leases'] is acquired), None)
            if custody is None:
                raise ValueError('the host process custody is not rooted')
            process = subprocess.Popen.__new__(subprocess.Popen)
            custody['objects'].append(process)
            subprocess.Popen.__init__(process, state['command'], env=environment, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            birth = observer._birth_time(process)
            handles = tuple(handle for lease in acquired for handle in lease.handles())
            duplicated = guard.duplicate_to_process(handles, process)
            if (type(duplicated) is not tuple or len(duplicated) != len(handles)
                    or any(type(handle) is not int or handle <= 0 for handle in duplicated)
                    or source_check() is not True
                    or driver._ordinary_source(command_path, 8192) != state['command_raw']):
                raise ValueError('the host source protection differs')
            if process.stdin.write(START) != len(START):
                raise ValueError('the host start gate is incomplete')
            process.stdin.close()
            code = process.wait(timeout=7200)
            terminal = driver._ordinary_source(str(run_dir / 'host-terminal.json'), 8192)
            if (type(code) is not int or code != 0 or source_check() is not True
                    or host.check_host_terminal(driver, context, config, child, owner, terminal,
                                                process.pid, birth, HOST_SHA) is not True):
                raise ValueError('the retained host exit is incomplete')
            observer._close_process_handle(process)
            return {'cleanup_verified': True, 'host_pid': process.pid, 'host_birth_100ns': birth,
                    'host_terminal_sha256': hashlib.sha256(terminal).hexdigest()}
        def postcheck(result):
            pid, birth = driver._process_identity(owner._current_process_identity())
            if pid == result['host_pid'] or source_check() is not True:
                raise ValueError('the custodian process identity differs')
            record = {'schema': 'mlg-injection-validation-custodian-terminal-v1',
                'status': 'awaiting_custodian_exit', 'custodian_pid': pid, 'custodian_birth_100ns': birth,
                'host_pid': result['host_pid'], 'host_birth_100ns': result['host_birth_100ns'],
                'host_terminal_sha256': result['host_terminal_sha256'],
                'host_command_sha256': hashlib.sha256(state['command_raw']).hexdigest(),
                'host_source_sha256': HOST_SHA, 'driver_source_sha256': DRIVER_SHA,
                'plan_sha256': PLAN_SHA, 'config_sha256': hashlib.sha256(config_raw).hexdigest(),
                'custodian_source_sha256': hashlib.sha256(own_raw).hexdigest(), 'forward_count': 0}
            terminal = str(run_dir / 'custodian-terminal.json')
            driver._write_once(terminal, driver.canonical(record))
            if source_check() is not True or driver._ordinary_source(terminal, 8192) != driver.canonical(record):
                raise ValueError('the custodian terminal changed')
            return record
        return owner.run_guarded(leases, precheck, launch, postcheck, retained=modules)
    return driver.run_bound_imports(context['graph'], execute)


def main(argv=None, stream=None):
    try:
        stream = sys.stdin.buffer if stream is None else stream
        if stream.readline(len(START) + 1) != START:
            raise ValueError('the supervised start token is required')
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) != 3:
            raise ValueError('the validation custodian arguments differ')
        run(*args)
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
