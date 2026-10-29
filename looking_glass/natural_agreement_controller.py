"""Independent storage and process checks for the public CPU discovery.

The journal terminal is only a candidate; the controller never changes or
deletes a failed attempt. These checks are integrity checks, not an independent
recomputation of the experiment's scientific conclusions.
"""

import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
import time


class InventoryError(RuntimeError):
    """The child's retained evidence is incomplete or inconsistent."""


_PUBLIC_MANIFEST_SHA256 = '1b1adf4d72163bfa3e0399ed8ee92eba6b5c0eb82a47edf28cecde5ffb1deafe'


def await_start(*, timeout=10):
    """Wait for the start token; the cooperating supervisor assigns the job."""
    if os.name != 'nt' or type(timeout) is not int or not 1 <= timeout <= 30:
        raise ValueError('invalid startup timeout')
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.PeekNamedPipe.argtypes = (wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                     wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
    kernel.PeekNamedPipe.restype = wintypes.BOOL
    fd = sys.stdin.fileno()
    handle = msvcrt.get_osfhandle(fd)
    deadline = time.monotonic() + timeout
    data = bytearray()
    while time.monotonic() < deadline:
        available = wintypes.DWORD()
        if not kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
            break
        if available.value:
            data.extend(os.read(fd, min(available.value, 128 - len(data))))
            if b'\n' in data or len(data) >= 128:
                break
        else:
            time.sleep(0.01)
    if bytes(data) != b'MLG-natural-discovery-start-v1\n':
        raise RuntimeError('owner did not receive the supervisor startup signal')


def source_inventory(root):
    """Bind current project Python files to a clean committed checkout."""
    root = Path(root).resolve(strict=True)
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    environment.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                        'GIT_GRAFT_FILE': os.devnull, 'GIT_NO_REPLACE_OBJECTS': '1',
                        'GIT_TERMINAL_PROMPT': '0', 'GIT_OPTIONAL_LOCKS': '0'})
    def git(*args):
        try:
            return subprocess.check_output(['git', '--no-replace-objects', '-C', str(root), *args], timeout=30,
                                           stderr=subprocess.DEVNULL, env=environment)
        except (OSError, subprocess.SubprocessError) as error:
            raise InventoryError('cannot inspect committed source') from error
    revision = git('rev-parse', '--verify', 'HEAD').strip().decode('ascii')
    if re.fullmatch('[0-9a-f]{40}', revision) is None:
        raise InventoryError('invalid source revision')
    if git('status', '--porcelain=v1', '--untracked-files=all').strip():
        raise InventoryError('source checkout must be clean')
    entries = git('ls-tree', '-rz', '--full-tree', revision).split(b'\0')
    inventory = {}
    for entry in entries:
        if not entry:
            continue
        header, name_bytes = entry.split(b'\t', 1)
        mode, kind, blob = header.decode('ascii').split()
        name = name_bytes.decode('utf-8')
        selected = ((name.startswith('looking_glass/') and name.endswith('.py'))
                    or name in ('tools/natural_agreement_cpu.py', 'tools/audit_natural_agreement_discovery.py',
                                'scripts/v2_source_guards.py'))
        if not selected:
            continue
        if mode not in ('100644', '100755') or kind != 'blob' or '..' in Path(name).parts:
            raise InventoryError('unsupported source entry')
        path = root / name
        _regular(path)
        if not path.resolve(strict=True).is_relative_to(root):
            raise InventoryError('source escapes the checkout')
        data = path.read_bytes()
        if len(data) > 8 * 1024 * 1024:
            raise InventoryError('source file exceeds its bound')
        git_sha = hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()
        if git_sha != blob:
            raise InventoryError('source file differs from committed bytes')
        inventory[name] = hashlib.sha256(data).hexdigest()
    actual = {path.relative_to(root).as_posix() for path in (root / 'looking_glass').rglob('*.py')}
    if not inventory or actual != {name for name in inventory if name.startswith('looking_glass/')}:
        raise InventoryError('project Python inventory is incomplete or has extra files')
    return revision, dict(sorted(inventory.items()))


def verify_source_inventory(root, revision, inventory):
    """Recheck frozen bytes without starting subprocesses inside the owner job."""
    root = Path(root).resolve(strict=True)
    if (type(revision) is not str or re.fullmatch(r'[0-9a-f]{40}', revision) is None
            or type(inventory) is not dict or not inventory):
        raise InventoryError('invalid frozen source identity')
    for name, digest in inventory.items():
        if (type(name) is not str or type(digest) is not str
                or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                or not ((name.startswith('looking_glass/') and name.endswith('.py'))
                        or name in ('tools/natural_agreement_cpu.py', 'tools/audit_natural_agreement_discovery.py',
                                    'scripts/v2_source_guards.py'))
                or '\\' in name or '..' in Path(name).parts):
            raise InventoryError('invalid frozen source entry')
        path = root / name
        _regular(path)
        if not path.resolve(strict=True).is_relative_to(root):
            raise InventoryError('source escapes the checkout')
        data = path.read_bytes()
        if len(data) > 8 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest:
            raise InventoryError('source differs from frozen bytes')
    actual = {path.relative_to(root).as_posix() for path in (root / 'looking_glass').rglob('*.py')}
    if actual != {name for name in inventory if name.startswith('looking_glass/')}:
        raise InventoryError('frozen project Python inventory differs from checkout')
    return revision, dict(sorted(inventory.items()))


def bind_public_cases(raw, tokenized_rows, tokenized_groups):
    """Bind only the published discovery rows to fresh tokenizer alignment."""
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != _PUBLIC_MANIFEST_SHA256:
        raise InventoryError('public preflight manifest digest differs from the frozen bytes')
    manifest = _json(raw)
    if (manifest.get('model_id') != 'distilbert/distilgpt2'
            or manifest.get('model_revision') != '2290a62682d06624634c1f46a6ad5be0f47f38aa'
            or manifest.get('targets') != {' is': 318, ' are': 389, ' was': 373, ' were': 547}
            or manifest.get('rows') != tokenized_rows or manifest.get('groups') != tokenized_groups
            or type(tokenized_rows) is not list or len(tokenized_rows) != 16):
        raise InventoryError('manifest and tokenizer alignment do not agree')
    cases = []
    for index, row in enumerate(tokenized_rows):
        lemma = tokenized_rows[index - index % 2]['subject']
        cases.append({'case_id': f'c{index:02}', 'quartet': f'q{row["group"]}',
                      'lemma': lemma, 'number': row['subject_number'],
                      'input_ids': list(row['input_ids']), 'attention_mask': list(row['attention_mask']),
                      'subject_index': row['subject_index']})
    return cases


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError('duplicate JSON key')
        result[key] = value
    return result


def _json(raw):
    def constant(_):
        raise InventoryError('nonfinite JSON number')
    try:
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=constant)
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as error:
        raise InventoryError('invalid JSON') from error
    def check(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise InventoryError('nonfinite JSON number')
        if isinstance(item, dict):
            for entry in item.values():
                check(entry)
        elif isinstance(item, list):
            for entry in item:
                check(entry)
    try:
        check(value)
    except RecursionError as error:
        raise InventoryError('JSON nesting exceeds the supported limit') from error
    return value


def _regular(path):
    try:
        info = path.lstat()
    except OSError as error:
        raise InventoryError('missing evidence file') from error
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400 or info.st_nlink != 1):
        raise InventoryError('evidence is not an unaliased regular file')
    return info


def _directory(path):
    try:
        info = path.lstat()
    except OSError as error:
        raise InventoryError('missing evidence directory') from error
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400):
        raise InventoryError('evidence directory is a link or reparse point')


def _read(path, limit):
    info = _regular(path)
    if info.st_size > limit:
        raise InventoryError('evidence file exceeds its limit')
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise InventoryError('evidence file changed while reading')
    return raw


def audit_storage(directory):
    """Audit every journal event and referenced tensor, without mutating it."""
    directory = Path(directory)
    _directory(directory)
    if {p.name for p in directory.iterdir()} != {'started.json', 'events.jsonl', 'terminal.json', 'tensors'}:
        raise InventoryError('unexpected or missing attempt artifacts')
    tensor_dir = directory / 'tensors'
    _directory(tensor_dir)
    started_raw = _read(directory / 'started.json', 256 * 1024)
    started = _json(started_raw)
    if type(started) is not dict:
        raise InventoryError('invalid start metadata')
    terminal_raw = _read(directory / 'terminal.json', 256 * 1024)
    terminal = _json(terminal_raw)
    if (type(terminal) is not dict or terminal.get('status') != 'complete_candidate'
            or terminal.get('storage_accounting_complete') is not True
            or any(type(terminal.get(name)) is not int or terminal[name] < 0
                   for name in ('events', 'retained_tensor_bytes', 'event_log_bytes'))):
        raise InventoryError('terminal is absent, incomplete, or not a candidate')
    raw = _read(directory / 'events.jsonl', 64 * 1024 * 1024)
    if raw and not raw.endswith(b'\n'):
        raise InventoryError('event log is truncated')
    lines = raw.splitlines()
    if len(lines) > 10000 or terminal.get('events') != len(lines) or terminal.get('event_log_bytes') != len(raw):
        raise InventoryError('event log accounting mismatch')
    refs = {}
    records = []
    for index, line in enumerate(lines):
        if len(line) > 256 * 1024:
            raise InventoryError('event is too large')
        row = _json(line)
        if (type(row) is not dict or set(row) != {'index', 'event', 'tensors'}
                or type(row['index']) is not int or row['index'] != index
                or type(row['event']) is not dict or type(row['tensors']) is not dict
                or len(row['tensors']) > 128):
            raise InventoryError('invalid event structure or order')
        total = 0
        for name, ref in row['tensors'].items():
            if (type(name) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', name)
                    or type(ref) is not dict or set(ref) != {'sha256', 'shape', 'dtype', 'bytes'}):
                raise InventoryError('invalid tensor reference')
            digest, shape, size = ref['sha256'], ref['shape'], ref['bytes']
            if (type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                    or type(shape) is not list or not 1 <= len(shape) <= 8
                    or any(type(d) is not int or d <= 0 for d in shape)
                    or type(size) is not int or math.prod(shape) * 4 != size
                    or ref['dtype'] != 'float32-le'):
                raise InventoryError('tensor metadata is inconsistent')
            total += size
            if digest in refs and refs[digest] != size:
                raise InventoryError('conflicting tensor sizes')
            refs[digest] = size
        if total > 1024 * 1024:
            raise InventoryError('per-event tensor limit exceeded')
        records.append(row)
    if sum(refs.values()) > 512 * 1024 * 1024 or terminal.get('retained_tensor_bytes') != sum(refs.values()):
        raise InventoryError('retained tensor accounting mismatch')
    if {p.name for p in tensor_dir.iterdir()} != {digest + '.f32le' for digest in refs}:
        raise InventoryError('unreferenced or missing tensor files')
    for digest, size in refs.items():
        data = _read(tensor_dir / (digest + '.f32le'), size)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise InventoryError('tensor bytes differ from reference')
        for i in range(0, size, 4):
            bits = int.from_bytes(data[i:i + 4], 'little')
            if bits & 0x7f800000 == 0x7f800000:
                raise InventoryError('nonfinite stored float32 value')
    return {'events': len(records), 'retained_tensor_bytes': sum(refs.values()),
            'tensor_sha256': sorted(refs), 'started_sha256': hashlib.sha256(started_raw).hexdigest(),
            'events_sha256': hashlib.sha256(raw).hexdigest(),
            'terminal_sha256': hashlib.sha256(terminal_raw).hexdigest(),
            '_started': started, '_terminal': terminal, '_records': records}


def check_start(actual, expected):
    """Require type-exact start metadata and re-hash the embedded binding."""
    if (type(actual) is not dict or type(expected) is not dict
            or set(actual) != {'schema', 'binding_sha256', 'binding'}
            or actual.get('schema') != 'mlg-natural-cpu-attempt-v1'):
        raise InventoryError('attempt start binding is malformed')
    try:
        canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=True,
                                             allow_nan=False, separators=(',', ':'))
        embedded = (json.dumps(actual['binding'], sort_keys=True, ensure_ascii=True,
                               allow_nan=False, indent=2) + '\n').encode('utf-8')
        if (canonical(actual) != canonical(expected)
                or hashlib.sha256(embedded).hexdigest() != actual['binding_sha256']):
            raise InventoryError('attempt runtime and source binding do not match the published configuration')
    except (TypeError, ValueError, RecursionError) as error:
        raise InventoryError('attempt binding is invalid JSON') from error


def audit_result(directory, *, exit_code, reaped, expected_cases=None, expected_start=None):
    """Require a reaped zero exit plus a complete, audited discovery journal."""
    verdict = {'status': 'incomplete', 'exit_code': exit_code, 'child_reaped': reaped}
    if type(exit_code) is not int or exit_code != 0 or reaped is not True:
        verdict['reason'] = 'child did not exit successfully and get reaped'
        return verdict
    try:
        inventory = audit_storage(directory)
        check_start(inventory['_started'], expected_start)
        check_discovery(inventory['_records'], inventory['_terminal'], expected_cases)
        if expected_start is not None:
            check_execution_configuration(inventory['_records'][0]['event'], expected_start['binding']['runtime'])
            cap = expected_start['binding']['study_budget']['phase_reservations_bytes']['natural_discovery']
            if type(cap) is not int or cap <= 0 or inventory['retained_tensor_bytes'] > cap:
                raise InventoryError('retained tensors exceed the frozen discovery reservation')
    except (InventoryError, OSError, ValueError, TypeError, KeyError, IndexError) as error:
        verdict['reason'] = str(error)
        return verdict
    inventory.pop('_records')
    inventory.pop('_terminal')
    inventory.pop('_started')
    verdict.update({'status': 'evidence_complete', 'inventory': inventory,
                    'scientific_validation': 'pending independent raw numerical and landing audit'})
    return verdict


def check_execution_configuration(contract, runtime):
    """Bind recorded configuration bytes and their global values to the runtime."""
    configuration = contract.get('execution_configuration')
    if (type(configuration) is not list or len(configuration) != 2
            or type(configuration[1]) is not list or not configuration[1]):
        raise InventoryError('execution configuration is missing or malformed')
    try:
        raw = json.dumps(configuration, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
        expected_global = [['deterministic_algorithms', runtime['deterministic_algorithms']],
                           ['deterministic_warn_only', runtime['deterministic_warn_only']],
                           ['cpu_threads', runtime['cpu_threads']],
                           ['cpu_interop_threads', runtime['interop_threads']],
                           ['float32_matmul_precision', runtime['float32_matmul_precision']],
                           ['calling_thread_denormal_product_bits', runtime['denormal_probe'][1]]]
        if (hashlib.sha256(raw).hexdigest() != contract.get('execution_configuration_sha256')
                or not _same_json(configuration[0], expected_global)):
            raise InventoryError('execution configuration differs from the frozen runtime')
    except (TypeError, ValueError, KeyError, IndexError, RecursionError) as error:
        raise InventoryError('execution configuration is invalid') from error


def _retained_tensor_files(attempt, cleanup_verified):
    """Count even orphan or partial tensor files, without trusting the terminal."""
    directory = Path(attempt) / 'tensors'
    result = {'accounting_complete': bool(cleanup_verified), 'total_bytes': 0, 'files': []}
    if not cleanup_verified:
        result['reason'] = 'pending verified process cleanup; no live files were read'
        return result
    try:
        if not directory.exists():
            return result
        if not directory.is_dir() or directory.is_symlink():
            raise OSError('invalid tensor directory')
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise OSError('invalid tensor entry')
            digest = hashlib.sha256()
            size = 0
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    size += len(block)
                    digest.update(block)
            if size != path.stat().st_size:
                raise OSError('tensor changed while counting')
            result['total_bytes'] += size
            result['files'].append({'name': path.name, 'bytes': size, 'sha256': digest.hexdigest()})
    except OSError as error:
        result['accounting_complete'] = False
        result['error_type'] = type(error).__name__
    return result


def supervise(command, output, expected_cases, environment, *, expected_start=None, timeout=7200):
    """Run a gated owner under a kill-on-close job, retaining bounded logs."""
    if os.name != 'nt' or type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError('the Windows CPU owner requires a timeout of at most two hours')
    from scripts.v2_source_guards import WindowsJob
    output = Path(output)
    if not output.is_absolute() or not output.parent.is_dir():
        raise ValueError('output needs an absolute path and an existing parent')
    output.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    code, reaped, failure, child, job = None, False, None, None, None
    job_drained, job_closed = False, False
    overflow = threading.Event()
    pump_errors = []
    pumps = []
    with (output / 'stdout.txt').open('xb') as stdout, (output / 'stderr.txt').open('xb') as stderr:
        try:
            job = WindowsJob()
            child = subprocess.Popen(command, env=environment, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=subprocess.DETACHED_PROCESS)
            job.assign(child)
            def pump(source, destination):
                remaining = 1024 * 1024
                try:
                    while True:
                        data = source.read1(65536)
                        if not data:
                            break
                        keep = data[:remaining]
                        if destination.write(keep) != len(keep):
                            raise OSError('short log write')
                        remaining -= len(keep)
                        if len(keep) != len(data):
                            overflow.set()
                            try:
                                child.terminate()
                            except OSError:
                                pass
                except BaseException as error:
                    pump_errors.append(type(error).__name__)
                    try:
                        child.terminate()
                    except OSError:
                        pass
                finally:
                    source.close()
            for source, destination in ((child.stdout, stdout), (child.stderr, stderr)):
                thread = threading.Thread(target=pump, args=(source, destination), daemon=True)
                thread.start()
                pumps.append(thread)
            child.stdin.write(b'MLG-natural-discovery-start-v1\n')
            child.stdin.close()
            code = child.wait(timeout=timeout)
            reaped = True
        except BaseException as error:
            failure = type(error).__name__
        finally:
            if job is not None:
                try:
                    job.terminate_and_wait()
                    job_drained = True
                except BaseException as cleanup_error:
                    failure = (failure or 'job cleanup') + '; drain: ' + type(cleanup_error).__name__
                try:
                    job.close()
                    job_closed = True
                except BaseException as cleanup_error:
                    failure = (failure or 'job cleanup') + '; close: ' + type(cleanup_error).__name__
            if child is not None and not reaped:
                try:
                    if child.stdin is not None and not child.stdin.closed:
                        child.stdin.close()
                    if child.poll() is None:
                        child.kill()
                    code = child.wait(timeout=10)
                    reaped = True
                except BaseException as cleanup_error:
                    failure = (failure or 'child cleanup') + '; reap: ' + type(cleanup_error).__name__
            for thread in pumps:
                thread.join(timeout=10)
                if thread.is_alive():
                    failure = (failure or 'log drain') + '; log drain unverified'
        for stream in (stdout, stderr):
            stream.flush()
            os.fsync(stream.fileno())
    cleanup_verified = bool(reaped and job_drained and job_closed)
    if cleanup_verified:
        verdict = audit_result(output / 'attempt', exit_code=code, reaped=reaped,
                               expected_cases=expected_cases, expected_start=expected_start)
    else:
        verdict = {'status': 'incomplete', 'exit_code': code, 'child_reaped': reaped,
                   'reason': 'process cleanup is unverified; evidence was not audited'}
    if failure is not None or overflow.is_set() or pump_errors:
        verdict.update({'status': 'incomplete', 'supervisor_error': failure})
    verdict['log_overflow'] = overflow.is_set()
    verdict['job_drained'] = job_drained
    verdict['job_closed'] = job_closed
    verdict['cleanup_verified'] = cleanup_verified
    verdict['retry_safe'] = verdict['cleanup_verified']
    verdict['retained_tensor_files'] = _retained_tensor_files(output / 'attempt', verdict['cleanup_verified'])
    if not verdict['retained_tensor_files']['accounting_complete']:
        verdict['status'] = 'incomplete'
        verdict['retry_safe'] = False
    elif (verdict['status'] == 'evidence_complete'
          and verdict['retained_tensor_files']['total_bytes'] != verdict['inventory']['retained_tensor_bytes']):
        verdict['status'] = 'incomplete'
        verdict['reason'] = 'retained tensor accounting changed after inventory'
    if not verdict['cleanup_verified']:
        verdict['status'] = 'incomplete'
        verdict['protection_state'] = 'unverified; stop and verify process cleanup before any retry'
    elif failure == 'KeyboardInterrupt' and not overflow.is_set() and not pump_errors:
        verdict['status'] = 'cancelled'
    if pump_errors:
        verdict['log_errors'] = pump_errors
    verdict.update({'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
                    'command': command,
                    'stdout_sha256': hashlib.sha256((output / 'stdout.txt').read_bytes()).hexdigest(),
                    'stderr_sha256': hashlib.sha256((output / 'stderr.txt').read_bytes()).hexdigest()})
    raw = (json.dumps(verdict, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n').encode('utf-8')
    with (output / 'supervisor.json').open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short supervisor record write')
        stream.flush()
        os.fsync(stream.fileno())
    return verdict


def _site_metadata(site_id):
    layer, component, role = site_id.split(':')
    path = f'transformer.h.{layer}' + ('' if component == 'block' else f'.{component}')
    return {'id': site_id, 'layer': int(layer), 'component': component, 'token_role': role,
            'module_path': path, 'output_selector': [0] if component in ('attn', 'block') else []}


def _same_json(actual, expected):
    return json.dumps(actual, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        expected, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _plan_step(name, operation, site, order, case, *, whole=False, preceding=(), source_capture=None):
    metadata = _site_metadata(site)
    return {'name': name, 'operation': operation, 'order': order,
            'module_path': metadata['module_path'], 'module_aliases': [metadata['module_path']],
            'invocation': 0, 'forward_index': 0, 'call_site_known': False,
            'batch_index': None if whole else 0,
            'token_index': None if whole else (case['subject_index'] if metadata['token_role'] == 'subject'
                                               else len(case['input_ids']) - 1),
            'output_selector': metadata['output_selector'], 'output_shape': [1, 5, 768],
            'selected_shape': [1, 5, 768] if whole else [768], 'dtype': 'torch.float32',
            'device': 'cpu', 'preceding_same_site_actions': list(preceding), 'source_capture': source_capture}


def _check_plan(event, case, all_sites):
    kind = event['kind']
    plan = event.get('plan_metadata')
    if type(plan) is not dict or type(plan.get('steps')) is not list:
        raise InventoryError('forward plan metadata is missing')
    if kind in ('clean_capture', 'clean_repeat'):
        steps = [_plan_step(f'capture_{i}', 'capture', site, i, case,
                            preceding=('capture_8',) if i == 9 else ())
                 for i, site in enumerate(all_sites)]
        capture_bytes, source_bytes = 10 * 768 * 4, 0
    else:
        site = event.get('site', {}).get('id')
        if site not in all_sites or not _same_json(event['site'], _site_metadata(site)):
            raise InventoryError('declared site metadata differs from the fixed candidate')
        if kind == 'null' and site != all_sites[0]:
            raise InventoryError('null control used the wrong site')
        names = ('whole_before', 'before', 'action', 'after', 'whole_after')
        operation = 'null' if kind == 'null' else 'patch'
        self_patch = kind == 'patch' and event.get('target') == event.get('donor')
        steps = [_plan_step(name, operation if name == 'action' else 'capture', site, i, case,
                            whole=name.startswith('whole_'), preceding=names[:i],
                            source_capture='before' if name == 'action' and self_patch else None)
                 for i, name in enumerate(names)]
        capture_bytes = (2 * 5 + 2) * 768 * 4
        source_bytes = 0 if operation == 'null' or self_patch else 768 * 4
    actual = plan['steps']
    if (len(actual) != len(steps) or type(plan.get('capture_bytes')) is not int
            or plan['capture_bytes'] != capture_bytes or type(plan.get('source_bytes')) is not int
            or plan['source_bytes'] != source_bytes):
        raise InventoryError('forward plan inventory is inconsistent')
    for row, expected in zip(actual, steps):
        if (type(row) is not dict or any(name not in row for name in expected)
                or not _same_json({name: row[name] for name in expected}, expected)):
            raise InventoryError('forward plan coordinates or order differ from the declared design')


def check_discovery(records, terminal, expected_cases):
    """Check fixed-case coverage and control outcomes, not effect correctness."""
    if (type(expected_cases) is not list or len(expected_cases) != 16
            or type(records) is not list or len(records) != 626
            or not _same_json(terminal.get('detail'), {'forwards': 608})):
        raise InventoryError('discovery requires sixteen bound cases and 608 forwards')
    sites = [f'{layer}:{component}:last' for layer in range(1, 5)
             for component in ('attn', 'mlp')]
    all_sites = sites + ['5:block:last', '5:block:subject']
    cases = {case['case_id']: case for case in expected_cases}
    def input_sha(case_id):
        case = cases[case_id]
        raw = json.dumps({'ids': case['input_ids'], 'subject_index': case['subject_index']},
                         sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()
    groups = {}
    for case in expected_cases:
        groups.setdefault(case['quartet'], []).append(case['case_id'])
    if len(cases) != 16 or len(groups) != 4 or any(len(group) != 4 for group in groups.values()):
        raise InventoryError('invalid case contract')
    events = [row['event'] for row in records]
    kinds = Counter(event.get('kind') for event in events)
    expected_kinds = {'run_contract': 1, 'clean_capture': 16, 'clean_repeat': 16,
                      'null': 16, 'patch': 512, 'delivery': 16, 'subject_position': 16,
                      'final_clean': 16, 'present_site_metrics': 8,
                      'secondary_past_site_metrics': 8, 'summary': 1}
    if kinds != expected_kinds:
        raise InventoryError('discovery event kind counts do not match the fixed design')
    expected_order = [('run_contract', None, None, None, None)]
    for quartet, members in groups.items():
        for kind in ('clean_capture', 'clean_repeat', 'null'):
            expected_order.extend((kind, quartet, case, None, None) for case in members)
        expected_order.extend(('patch', quartet, target, site, donor)
                              for site in sites for target in members for donor in members)
        expected_order.extend((kind, quartet, target, None, None)
                              for target in members for kind in ('delivery', 'subject_position'))
        expected_order.extend(('final_clean', quartet, case, None, None) for case in members)
    expected_order.extend((kind, None, None, site, None)
                          for kind in ('present_site_metrics', 'secondary_past_site_metrics') for site in sites)
    expected_order.append(('summary', None, None, None, None))
    observed_order = []
    for event in events:
        kind = event.get('kind')
        if kind in ('clean_capture', 'clean_repeat', 'final_clean'):
            observed_order.append((kind, event.get('quartet'), event.get('case'), None, None))
        elif kind in ('null', 'delivery', 'subject_position'):
            observed_order.append((kind, event.get('quartet'), event.get('target'), None, None))
        elif kind == 'patch':
            site = event.get('site')
            observed_order.append((kind, event.get('quartet'), event.get('target'),
                                   site.get('id') if type(site) is dict else None, event.get('donor')))
        elif kind in ('present_site_metrics', 'secondary_past_site_metrics'):
            observed_order.append((kind, None, None, event.get('site'), None))
        else:
            observed_order.append((kind, None, None, None, None))
    if observed_order != expected_order:
        raise InventoryError('event order differs from the declared family and control sequence')
    contract = events[0]
    if (contract.get('kind') != 'run_contract' or not _same_json(contract.get('cases'), expected_cases)
            or type(contract.get('cpu_rng_seed')) is not int or contract['cpu_rng_seed'] != 20261028
            or type(contract.get('cpu_threads')) is not int or contract['cpu_threads'] != 1
            or contract.get('deterministic_algorithms') is not True
            or not _same_json(contract.get('sites'), [_site_metadata(site) for site in all_sites])):
        raise InventoryError('run contract differs from the published case or CPU configuration')
    summary = events[-1]
    result = summary.get('result', {})
    expected_controls = {'null': 16, 'self': 128, 'delivery': 16,
                         'subject_position': 16, 'repeat': 16, 'final_clean': 16}
    if (summary.get('kind') != 'summary' or type(result) is not dict
            or result.get('status') != 'complete' or type(result.get('forwards')) is not int
            or result['forwards'] != 608 or not _same_json(result.get('controls'), expected_controls)
            or result.get('model_state_unchanged') is not True
            or type(result.get('model_state_sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}', result['model_state_sha256']) is None
            or type(contract.get('seeded_cpu_rng_sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}', contract['seeded_cpu_rng_sha256']) is None
            or result.get('seeded_cpu_rng_sha256') != contract['seeded_cpu_rng_sha256']
            or type(result.get('cpu_rng_seed')) is not int or result['cpu_rng_seed'] != 20261028):
        raise InventoryError('incomplete result summary')
    selection = result.get('selection')
    if type(selection) is not dict:
        raise InventoryError('discovery selection is missing')
    predictions, magnitudes = selection.get('predictions'), selection.get('magnitude_means')
    ranking = selection.get('ranking')
    random_index = hashlib.sha256(b'MLG-number-v1-random\x00' + (20261028).to_bytes(8, 'big')).digest()[0] % 8
    if (type(predictions) is not dict or set(predictions) != set(sites)
            or type(magnitudes) is not dict or set(magnitudes) != set(sites)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in list(predictions.values()) + list(magnitudes.values()))
            or ranking != sorted(sites, key=lambda site: -predictions[site])
            or selection.get('selected_site') != (ranking[0] if predictions[ranking[0]] > 0 else None)
            or selection.get('earliest_site') != sites[0] or selection.get('random_index') != random_index
            or selection.get('random_site') != sites[random_index]
            or selection.get('magnitude_site') != sorted(sites, key=lambda site: -magnitudes[site])[0]):
        raise InventoryError('selection or fixed baseline is incomplete or inconsistent')
    expected_magnitude = {(site, case['quartet'], case['lemma'], case['case_id'],
                           next(other['case_id'] for other in expected_cases
                                if other['quartet'] == case['quartet'] and other['lemma'] == case['lemma']
                                and other['number'] == 'plural'))
                          for site in sites for case in expected_cases if case['number'] == 'singular'}
    pairs = selection.get('magnitude_pairs')
    if (type(pairs) is not list or len(pairs) != 64
            or {(item.get('site'), item.get('quartet'), item.get('lemma'), item.get('singular'), item.get('plural'))
                for item in pairs} != expected_magnitude
            or any(type(item.get('l2')) not in (int, float) or not math.isfinite(item['l2']) or item['l2'] < 0
                   for item in pairs)):
        raise InventoryError('magnitude baseline pairs are incomplete')

    captures = {row['event']['case']: row['tensors'] for row in records
                if row['event'].get('kind') == 'clean_capture'}
    for row in records:
        event, tensors = row['event'], row['tensors']
        kind = event['kind']
        if kind in ('clean_capture', 'clean_repeat'):
            shapes = {f'capture_{i}': [768] for i in range(10)}
            shapes['last_logits'] = [50257]
        elif kind in ('null', 'patch', 'delivery', 'subject_position'):
            shapes = {'before': [768], 'after': [768], 'whole_before': [1, 5, 768],
                      'whole_after': [1, 5, 768]}
            logits_name = ('contrast_logits' if kind == 'patch' and event.get('target') != event.get('donor')
                           else 'last_logits')
            shapes[logits_name] = [4] if logits_name == 'contrast_logits' else [50257]
        elif kind == 'final_clean':
            shapes = {'last_logits': [50257]}
        else:
            shapes = {}
        if set(tensors) != set(shapes) or any(tensors[name].get('shape') != shape for name, shape in shapes.items()):
            raise InventoryError('per-forward witness tensor inventory or shape is incomplete')
        if kind in ('clean_capture', 'clean_repeat', 'null', 'patch', 'delivery', 'subject_position'):
            case_id = event.get('case') if kind in ('clean_capture', 'clean_repeat') else event.get('target')
            if case_id not in cases:
                raise InventoryError('plan references an unknown case')
            _check_plan(event, cases[case_id], all_sites)
        if kind == 'clean_capture' and event.get('input_sha256') != input_sha(event['case']):
            raise InventoryError('clean capture input identity differs from its bound case')
        if kind == 'null' and tensors['before']['sha256'] != tensors['after']['sha256']:
            raise InventoryError('null witness differs across the action')
        if kind in ('patch', 'delivery', 'subject_position'):
            target, donor = event.get('target'), event.get('donor')
            site = event.get('site', {}).get('id')
            if kind == 'patch' and target == donor:
                source_sha = tensors['before']['sha256']
            else:
                if donor not in captures or site not in all_sites:
                    raise InventoryError('unknown donor or witness site')
                source_sha = captures[donor][f'capture_{all_sites.index(site)}']['sha256']
            if tensors['after']['sha256'] != source_sha:
                raise InventoryError('post-action witness does not match its named source')
            if kind == 'patch' and event.get('source_tensor_sha256') != source_sha:
                raise InventoryError('patch source metadata differs from its witness')
            if kind == 'patch' and (event.get('source_input_sha256') != input_sha(donor)
                                    or event.get('target_input_sha256') != input_sha(target)):
                raise InventoryError('patch input identity differs from the bound cases')
            if kind in ('delivery', 'subject_position'):
                if (donor not in cases or cases[donor]['quartet'] != cases[target]['quartet']
                        or cases[donor]['lemma'] != cases[target]['lemma']
                        or cases[donor]['number'] == cases[target]['number']
                        or event.get('reference') != (donor if kind == 'delivery' else target)
                        or site != ('5:block:last' if kind == 'delivery' else '5:block:subject')):
                    raise InventoryError('position control has the wrong donor or site')
    expected_ids = set(cases)
    for kind in ('clean_capture', 'clean_repeat', 'null', 'delivery', 'subject_position', 'final_clean'):
        case_key = 'case' if kind in ('clean_capture', 'clean_repeat', 'final_clean') else 'target'
        matching = [event for event in events if event.get('kind') == kind]
        if {event.get(case_key) for event in matching} != expected_ids:
            raise InventoryError('missing or duplicated per-case control')
        for event in matching:
            if event.get('quartet') != cases[event[case_key]]['quartet']:
                raise InventoryError('control references the wrong quartet')
            if kind in ('clean_repeat', 'null'):
                checks = event.get('checks')
                required = set(['last_logits'] + all_sites) if kind == 'clean_repeat' else {'activation', 'logits'}
                if (type(checks) is not dict or set(checks) != required
                        or any(type(check) is not dict or check.get('within_tolerance') is not True
                               for check in checks.values())):
                    raise InventoryError('clean repeat or null control failed')
            if kind in ('null', 'delivery', 'subject_position') and event.get('untouched_output_bits_equal') is not True:
                raise InventoryError('control changed the untargeted output')
            if kind in ('delivery', 'subject_position'):
                if event.get('post_action_matches_donor_bits') is not True:
                    raise InventoryError('control did not land')
            if kind in ('delivery', 'subject_position', 'final_clean'):
                if (type(event.get('logits_check')) is not dict
                        or event['logits_check'].get('within_tolerance') is not True):
                    raise InventoryError('control logits failed')
    expected_patches = {(quartet, site, target, donor) for quartet, members in groups.items()
                        for site in sites for target in members for donor in members}
    patch_rows = [event for event in events if event.get('kind') == 'patch']
    observed = set()
    for event in patch_rows:
        site = event.get('site')
        key = (event.get('quartet'), site.get('id') if type(site) is dict else None,
               event.get('target'), event.get('donor'))
        if key in observed or key not in expected_patches:
            raise InventoryError('duplicated or unexpected patch')
        observed.add(key)
        self_patch = key[2] == key[3]
        if (event.get('self_patch') is not self_patch
                or event.get('post_action_matches_donor_bits') is not True
                or event.get('untouched_output_bits_equal') is not True):
            raise InventoryError('patch or landing check failed')
        checks = event.get('checks')
        if type(checks) is not dict or (self_patch and set(checks) != {'activation', 'logits'}):
            raise InventoryError('self-patch control is incomplete')
        if any(type(check) is not dict or check.get('within_tolerance') is not True
               for check in checks.values()):
            raise InventoryError('self-patch control failed')
    if observed != expected_patches:
        raise InventoryError('patch coverage is incomplete')
    for kind in ('present_site_metrics', 'secondary_past_site_metrics'):
        rows = [event for event in events if event.get('kind') == kind]
        matching = [event.get('site') for event in rows]
        if len(matching) != 8 or set(matching) != set(sites):
            raise InventoryError('site metric inventory is incomplete')
        for event in rows:
            site, metrics = event['site'], event.get('metrics')
            contrast = ({'first_token': ' is', 'first_id': 318, 'second_token': ' are', 'second_id': 389}
                        if kind == 'present_site_metrics' else
                        {'first_token': ' was', 'first_id': 373, 'second_token': ' were', 'second_id': 547})
            if event.get('contrast') != contrast:
                raise InventoryError('metric contrast token labels are incorrect')
            if (type(metrics) is not dict or set(metrics) != {'baseline', 'site_summary', 'quartets', 'pairs'}
                    or type(metrics['baseline']) is not dict or type(metrics['baseline'].get('n')) is not int
                    or metrics['baseline']['n'] != 16
                    or type(metrics['site_summary']) is not dict or metrics['site_summary'].get('site') != site
                    or type(metrics['site_summary'].get('n_quartets')) is not int
                    or metrics['site_summary']['n_quartets'] != 4
                    or type(metrics['quartets']) is not list or len(metrics['quartets']) != 4
                    or {q.get('quartet') for q in metrics['quartets']} != set(groups)
                    or type(metrics['pairs']) is not list or len(metrics['pairs']) != 64):
                raise InventoryError('site metrics omit a baseline, family, or pair')
            cells = {'same_lemma_same_number', 'same_lemma_opposite_number',
                     'different_lemma_same_number', 'different_lemma_opposite_number'}
            if any(q.get('site') != site or type(q.get('cell_means')) is not dict
                   or set(q['cell_means']) != cells or not _same_json(q.get('cell_counts'), {cell: 4 for cell in cells})
                   or 'number_main_contrast' not in q or 'lemma_number_interaction' not in q
                   for q in metrics['quartets']):
                raise InventoryError('family metrics omit a factorial cell')
            expected_site_pairs = {(q, t, d) for q, members in groups.items() for t in members for d in members}
            logit_fields = ({'clean_is', 'clean_are', 'patched_is', 'patched_are'}
                            if kind == 'present_site_metrics' else
                            {'clean_was', 'clean_were', 'patched_was', 'patched_were'})
            if ({(p.get('quartet'), p.get('target'), p.get('donor')) for p in metrics['pairs']} != expected_site_pairs
                    or any(p.get('site') != site or not {'cell', 'weight', 'clean_margin', 'patched_margin',
                                                          'signed_target_shift', 'clean_margin_sign', 'patched_margin_sign',
                                                          'strict_sign_reversal', 'zero_transition'} | logit_fields <= set(p)
                           for p in metrics['pairs'])):
                raise InventoryError('per-target metric pairs are incomplete')
            if kind == 'present_site_metrics' and predictions[site] != metrics['site_summary'].get('mean_number_main_contrast'):
                raise InventoryError('prediction differs from its site metric')
