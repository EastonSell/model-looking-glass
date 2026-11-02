"""Process-custody primitives for the prospectively bound public replay.

This module is not a production authorization or command-line launcher. A
trusted caller must source-verify the supplied child, supervisor and Windows
Job implementation, independently verify the published plan and the one-use
claim, protect the source/configuration/cache handles, and observe this
parent's own successful exit. Ambiguous cleanup must keep this process alive.
"""
import hashlib
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
from types import FunctionType, ModuleType


class ReplayOwnerError(RuntimeError):
    """The private replay cannot be promoted or its custody is uncertain."""


_QUARANTINED = []
_QUARANTINE_LOCK = threading.Lock()
_QUARANTINE_EVENT = threading.Event()
_BOOTSTRAP = ("import hashlib,sys\n"
              "p=sys.argv[1]\n"
              "with open(p,'rb') as f: raw=f.read(16*1024*1024+1)\n"
              "if len(raw)>16*1024*1024 or hashlib.sha256(raw).hexdigest()!=sys.argv[2]: raise SystemExit(1)\n"
              "sys.argv=[p]+sys.argv[3:]\n"
              "exec(compile(raw,p,'exec',dont_inherit=True,optimize=0),{'__name__':'__main__','__file__':p})\n")
_ACL_PROBE = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$p = [Console]::In.ReadToEnd()
$a = Microsoft.PowerShell.Security\Get-Acl -LiteralPath $p
$rules = @($a.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]))
if ($rules.Count -gt 8) { throw 'too many rules' }
$items = @($rules | ForEach-Object { [ordered]@{
  sid = $_.IdentityReference.Value; type = [int]$_.AccessControlType;
  inheritance = [int]$_.InheritanceFlags; propagation = [int]$_.PropagationFlags;
  inherited = [bool]$_.IsInherited; rights = [int64]$_.FileSystemRights
} })
$v = [ordered]@{
  user = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value;
  owner = [System.Security.Principal.NTAccount]::new($a.Owner).Translate([System.Security.Principal.SecurityIdentifier]).Value;
  protected = [bool]$a.AreAccessRulesProtected; rules = $items
}
[Console]::Out.Write(($v | ConvertTo-Json -Compress -Depth 4))
"""


def under_protection(leases, precheck, launch, postcheck, *, retained=None):
    """Keep pre-rooted leases and process objects through verified cleanup.

    A dedicated caller must hold on quarantine and must not treat a callback
    or a saved record as proof of a published plan or successful parent exit.
    """
    if (type(leases) not in (list, tuple) or not all(callable(f) for f in (precheck, launch, postcheck))):
        raise ReplayOwnerError('invalid protected replay composition')
    custody = {'leases': [], 'retained': retained, 'objects': []}
    with _QUARANTINE_LOCK:
        if _QUARANTINED:
            raise ReplayOwnerError('earlier replay custody is unverified')
        _QUARANTINED.append(custody)  # Root before acquiring or launching anything.
    acquired, launched, verified = custody['leases'], False, False
    try:
        for lease in leases:
            acquired.append(lease)
            lease.__enter__()
        if precheck(acquired) is not True:
            raise ReplayOwnerError('protected replay precheck was not verified')
        launched = True
        verdict = launch(acquired)
        verified = type(verdict) is dict and verdict.get('cleanup_verified') is True
        if not verified:
            raise ReplayOwnerError('replay process cleanup is unverified')
        return postcheck(verdict)
    finally:
        primary = sys.exc_info()[1]
        if not launched or verified:
            while acquired:
                lease = acquired[-1]
                try:
                    lease.close()
                except BaseException as cleanup_error:
                    if (primary is not None and
                            (isinstance(primary, (KeyboardInterrupt, SystemExit))
                             or not isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)))):
                        raise primary from cleanup_error
                    raise
                acquired.pop()
            with _QUARANTINE_LOCK:
                _QUARANTINED.remove(custody)


def hold_quarantine():
    """Retain process and lease handles after an unknown cleanup outcome.

    The dedicated caller must ignore normal console interrupts before launch
    and re-enter this hold if it is interrupted. Forced process termination and
    arbitrary native return-boundary failures remain outside this guarantee.
    """
    while _QUARANTINED:
        try:
            _QUARANTINE_EVENT.wait()
        except BaseException:
            continue


def run_guarded(leases, precheck, launch, postcheck, *, retained=None):
    """Use a dedicated process and retain quarantine under console interrupts."""
    previous = []
    try:
        for number in (signal.SIGINT, getattr(signal, 'SIGBREAK', None)):
            if number is not None:
                old = signal.getsignal(number)
                signal.signal(number, signal.SIG_IGN)
                previous.append((number, old))
        try:
            return under_protection(leases, precheck, launch, postcheck, retained=retained)
        except BaseException:
            while _QUARANTINED:
                try:
                    hold_quarantine()
                except BaseException:
                    continue
            raise
    finally:
        if not _QUARANTINED:
            primary = sys.exc_info()[1]
            errors = []
            for number, old in reversed(previous):
                try:
                    signal.signal(number, old)
                except BaseException as error:
                    errors.append(error)
            if errors:
                fatal = next((error for error in errors if isinstance(error, (KeyboardInterrupt, SystemExit))), None)
                if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                    raise primary from errors[0]
                if fatal is not None:
                    raise fatal from primary
                if primary is not None:
                    raise primary from errors[0]
                raise errors[0]


def _windows_acl_snapshot(path):
    """Read a bounded SID-based directory ACL through the system PowerShell."""
    if os.name != 'nt' or not Path(path).is_absolute() or any(c in str(path) for c in ('\r', '\n', '\0')):
        raise ReplayOwnerError('a Windows private directory is required')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetSystemDirectoryW.argtypes = (wintypes.LPWSTR, wintypes.UINT)
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel.GetSystemDirectoryW(buffer, len(buffer))
    if not length or length >= len(buffer):
        raise ReplayOwnerError('the Windows system directory is unavailable')
    system = Path(buffer.value)
    executable = system / 'WindowsPowerShell' / 'v1.0' / 'powershell.exe'
    if not system.is_absolute() or not executable.is_file():
        raise ReplayOwnerError('the system ACL reader is unavailable')
    environment = os.environ.copy()
    environment['PSModulePath'] = str(executable.parent / 'Modules')
    environment['WINDIR'] = str(system.parent)
    environment['SYSTEMROOT'] = str(system.parent)
    try:
        process = subprocess.run([str(executable), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', _ACL_PROBE],
                                 input=str(path).encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=20, env=environment)
        if process.returncode != 0 or not 0 < len(process.stdout) <= 16384 or len(process.stderr) > 16384:
            raise ReplayOwnerError('private directory ACL could not be read')
        return json.loads(process.stdout.decode('utf-8'))
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
        raise ReplayOwnerError('private directory ACL could not be read') from error


def _validate_private_acl(snapshot):
    if (type(snapshot) is not dict or set(snapshot) != {'user', 'owner', 'protected', 'rules'}
            or type(snapshot['user']) is not str
            or re.fullmatch(r'S-1-(?:[0-9]+-){2,14}[0-9]+', snapshot['user']) is None
            or snapshot['user'] == 'S-1-5-18' or snapshot['owner'] != snapshot['user']
            or snapshot['protected'] is not True or type(snapshot['rules']) is not list
            or len(snapshot['rules']) != 2):
        raise ReplayOwnerError('private directory ACL policy differs')
    principals = set()
    for rule in snapshot['rules']:
        fixed = {'type': 0, 'inheritance': 3, 'propagation': 0, 'inherited': False, 'rights': 0x1f01ff}
        if (type(rule) is not dict or set(rule) != set(fixed) | {'sid'} or type(rule['sid']) is not str
                or any(type(rule.get(k)) is not type(value) or rule[k] != value for k, value in fixed.items())):
            raise ReplayOwnerError('private directory ACL policy differs')
        principals.add(rule['sid'])
    if principals != {snapshot['user'], 'S-1-5-18'}:
        raise ReplayOwnerError('private directory ACL principals differ')


def verify_private_parent(path, identity):
    """Verify the protected parent identity and its exact private DACL policy."""
    path = Path(path)
    if (not path.is_absolute() or type(identity) is not tuple or len(identity) != 2
            or any(type(value) is not int or value <= 0 for value in identity)):
        raise ReplayOwnerError('invalid private parent identity')
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ReplayOwnerError('private directory path is linked')
    before = path.lstat()
    if not stat.S_ISDIR(before.st_mode) or (before.st_dev, before.st_ino) != identity:
        raise ReplayOwnerError('private directory identity differs')
    _validate_private_acl(_windows_acl_snapshot(path))
    after = path.lstat()
    if (after.st_dev, after.st_ino) != identity:
        raise ReplayOwnerError('private directory identity changed')
    return True


def verify_preclaim_handoff(*, child, claim, guard, workspace, config, plan_sha256, receipt_sha256,
                            parent_identity, ledger_identity, state_sha256, history_sha256,
                            scope_sha256, original_claims_sha256):
    """Recheck private intent, receipt and the independently anchored v6 row.

    `receipt_sha256` and the remaining anchors must come from independently
    verified protected configuration and the observed claimant handoff. This
    read-only consistency check does not itself establish a successful process
    exit, a public plan, or permission to replay or draw a split.
    """
    digests = (plan_sha256, receipt_sha256, state_sha256, history_sha256, scope_sha256, original_claims_sha256)
    if any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None for value in digests):
        raise ReplayOwnerError('invalid preclaim anchor')
    workspace = Path(workspace)
    claim._ordinary(workspace, directory=True)
    work = workspace / 'work'
    parent = work / 'number-injection-heldout-cpu-v1'
    ledger = work / 'agreement-study-ledger-v5' / 'ledger.sqlite3'
    diagnostic = work / 'injection-heldout-runtime-rehearsal-v1'
    phase = parent / ('injection-heldout-' + child._FREEZE)
    pilot = parent / 'runtime-compatibility-replay-v1'
    intent_path, receipt_path = parent / 'replay-claim-intent.json', parent / 'replay-claim-receipt.json'
    if (type(config) is not dict or type(config.get('claim_receipt_file')) is not str
            or child._absolute(config['claim_receipt_file']) != receipt_path):
        raise ReplayOwnerError('child claim receipt path differs from the verified handoff')
    claim._identity(parent, parent_identity, directory=True)
    claim._identity(ledger, ledger_identity, directory=False)
    if verify_private_parent(parent, parent_identity) is not True:
        raise ReplayOwnerError('private parent ACL is unverified')
    raw = child._read(receipt_path, 65536)
    if hashlib.sha256(raw).hexdigest() != receipt_sha256:
        raise ReplayOwnerError('observed claimant receipt differs')
    receipt = child._check_claim_receipt(raw, plan_sha256, config)
    expected_fields = {'ledger_path': str(ledger), 'phase_path': str(phase), 'pilot_path': str(pilot),
                       'diagnostic_path': str(diagnostic), 'parent_identity': list(parent_identity),
                       'ledger_identity': list(ledger_identity)}
    if any(child._canonical(receipt.get(key)) != child._canonical(value) for key, value in expected_fields.items()):
        raise ReplayOwnerError('preclaim protected paths or identities differ')
    diagnostic_state = claim._diagnostic_bytes(diagnostic)
    if diagnostic_state != (receipt['diagnostic_actual_bytes'], receipt['diagnostic_inventory_sha256']):
        raise ReplayOwnerError('retained no-model diagnostic changed')
    intent = {'schema': 'mlg-injection-replay-claim-intent-v1', 'plan_sha256': plan_sha256,
              **expected_fields, 'diagnostic_actual_bytes': diagnostic_state[0],
              'diagnostic_inventory_sha256': diagnostic_state[1]}
    intent_raw = child._read(intent_path, 65536)
    if (intent_raw != child._canonical(intent)
            or hashlib.sha256(intent_raw).hexdigest() != receipt['intent_sha256']):
        raise ReplayOwnerError('claim intent differs from the retained receipt')
    anchor = guard.ClaimAnchor(ledger_path=str(ledger), db_dev=ledger_identity[0], db_ino=ledger_identity[1],
        phase_path=str(phase), state_sha256=state_sha256, history_sha256=history_sha256,
        scope_sha256=scope_sha256, original_claims_sha256=original_claims_sha256,
        heldout_claim_sha256=receipt['injection_claim_row_sha256'])
    if guard.verify_existing_claim(ledger, 'injection_heldout', output_path=phase, anchor=anchor) is not True:
        raise ReplayOwnerError('the protected preclaim row is unverified')
    claim._identity(parent, parent_identity, directory=True)
    claim._identity(ledger, ledger_identity, directory=False)
    if (verify_private_parent(parent, parent_identity) is not True
            or child._read(intent_path, 65536) != intent_raw or child._read(receipt_path, 65536) != raw
            or claim._diagnostic_bytes(diagnostic) != diagnostic_state):
        raise ReplayOwnerError('preclaim evidence changed during verification')
    return {'status': 'row_and_receipt_consistent_awaiting_observed_exit', 'receipt_sha256': receipt_sha256,
            'intent_sha256': receipt['intent_sha256'], 'claim_row_sha256': receipt['injection_claim_row_sha256']}


def child_command(base_python, prefix, child_path, child_sha256, config_path, *, validation_only):
    """Construct the path-only isolated command for the exact gated child."""
    values = (base_python, prefix, child_path, config_path)
    if (type(validation_only) is not bool
            or any(not Path(value).is_absolute() for value in values)
            or Path(base_python).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)
            or type(child_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', child_sha256) is None):
        raise ReplayOwnerError('invalid isolated replay command')
    command = [str(base_python), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix), '-c',
               _BOOTSTRAP, str(child_path), child_sha256]
    if validation_only:
        command.append('--validate-only')
    command.append(str(config_path))
    if sum(len(item) for item in command) > 16000:
        raise ReplayOwnerError('isolated command exceeds the metadata reserve')
    return command


def bounded_supervisor(supervisor):
    """Use a private globals dictionary with a smaller per-stream cap."""
    function = getattr(supervisor, 'supervise_heldout', None)
    if type(function) is not FunctionType or function.__globals__ is not supervisor.__dict__:
        raise ReplayOwnerError('the verified supervisor source is required')
    private = ModuleType('_mlg_replay_bounded_supervisor')
    private.__dict__.update(supervisor.__dict__)
    private.__dict__['_LOG_LIMIT'] = 256 * 1024
    clone = FunctionType(function.__code__, private.__dict__, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    private.supervise_heldout = clone
    return private


def launch_child(*, child, child_path, child_sha256, config_path, expected_config, plan_bytes,
                 base_python, output, environment, guard, adapter, supervisor, cpu_job_type,
                 acquired, validation_only, timeout=7200):
    """Launch only inside rooted protection; caller authenticates all modules.

    The caller's precheck must bind the plan/publication, exact lease objects,
    protected cache, ledger row and observed claimant exit. This helper records
    no success by itself and returns the raw supervised custody result.
    """
    custody = next((item for item in _QUARANTINED if item['leases'] is acquired), None)
    if (custody is None or type(expected_config) is not bytes or type(plan_bytes) is not bytes
            or type(environment) is not dict or type(validation_only) is not bool):
        raise ReplayOwnerError('rooted source and cache protection is required')
    config_raw = child._read(config_path, 65536)
    if config_raw != expected_config:
        raise ReplayOwnerError('child configuration changed')
    config = child.parse_config(config_raw)
    source = child._read(child_path, 16 * 1024 * 1024)
    if hashlib.sha256(source).hexdigest() != child_sha256:
        raise ReplayOwnerError('child source changed')
    if child._read(config['replay_plan_file'], 2 * 1024 * 1024) != plan_bytes:
        raise ReplayOwnerError('replay plan changed')
    child.validate_plan(plan_bytes, child_source_sha256=child_sha256)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    command = child_command(base_python, config['pycache_prefix'], child_path, child_sha256,
                            config_path, validation_only=validation_only)
    handles = tuple(handle for lease in acquired for handle in lease.handles())
    duplicates = guard.InheritableHandles(handles)
    acquired.append(duplicates)  # Root the object before duplication can fail.
    inherited = duplicates.__enter__()
    private_supervisor = bounded_supervisor(supervisor)
    custody['objects'].append(private_supervisor)
    def audit(_attempt, exit_code, reaped):
        if (type(exit_code) is not int or exit_code != 0 or reaped is not True
                or child._read(config_path, 65536) != config_raw
                or child._read(config['replay_plan_file'], 2 * 1024 * 1024) != plan_bytes
                or child._read(child_path, 16 * 1024 * 1024) != source):
            raise ReplayOwnerError('protected replay inputs changed during the child')
        return inspect_child_postcheck(child, config, plan_sha256, child_sha256, validation_only=validation_only)
    verdict = adapter.supervise_with_cpu_job(command, output, environment, audit_result=audit,
        supervisor=private_supervisor, guard_module=guard, cpu_job_type=cpu_job_type, timeout=timeout,
        inherited_handles=inherited)
    return {'cleanup_verified': type(verdict) is dict and verdict.get('cleanup_verified') is True,
            'verdict': verdict, 'command': command, 'config': config, 'plan_sha256': plan_sha256}


def inspect_child_postcheck(child, config, plan_sha256, child_sha256, *, validation_only):
    """Inspect saved child bytes, without making a scientific completion claim."""
    if (type(validation_only) is not bool
            or any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None
                   for value in (plan_sha256, child_sha256))):
        raise ReplayOwnerError('invalid replay inspection identity')
    raw = child._read(config['postcheck_file'], 16384)
    value = child._json(raw)
    expected = {'schema': 'mlg-injection-runtime-replay-child-postcheck-v1',
                'status': 'validated_no_model_forward' if validation_only else 'private_replay_pending_independent_audit',
                'plan_sha256': plan_sha256, 'heldout_runtime_sha256': child._HELDOUT_RUNTIME,
                'child_source_sha256': child_sha256, 'forward_count': 0 if validation_only else 16}
    if validation_only:
        if os.path.lexists(config['attempt_dir']) or raw != child._canonical(expected):
            raise ReplayOwnerError('no-model child postcheck differs')
    else:
        keys = {'owner_terminal_sha256', 'receipts_sha256'}
        expected.update({'discovery_runtime_sha256': child._DISCOVERY_RUNTIME,
                         'runtime_deviation_sha256': child._DEVIATION,
                         'source_supplement_sha256': child._SOURCE_SUPPLEMENT})
        if (type(value) is not dict or set(value) != set(expected) | keys
                or any(child._canonical(value.get(k)) != child._canonical(v) for k, v in expected.items())
                or any(type(value[k]) is not str or re.fullmatch(r'[0-9a-f]{64}', value[k]) is None for k in keys)):
            raise ReplayOwnerError('private child postcheck differs')
        attempt = Path(config['attempt_dir'])
        terminal = child._read(attempt / 'owner-terminal.json', 65536)
        receipts = child._read(attempt / 'receipts.jsonl', 1024 * 1024)
        record = child._json(terminal)
        if (type(record) is not dict or record.get('status') != 'replay-produced-private-awaiting-independent-audit'
                or type(record.get('forward_count')) is not int or record['forward_count'] != 16
                or hashlib.sha256(terminal).hexdigest() != value['owner_terminal_sha256']
                or not receipts or hashlib.sha256(receipts).hexdigest() != value['receipts_sha256']):
            raise ReplayOwnerError('private replay evidence digest differs')
    return {'status': 'incomplete', 'validation_only': validation_only,
            'child_postcheck_sha256': hashlib.sha256(raw).hexdigest(),
            'forward_count': expected['forward_count'],
            'scientific_validation': 'not applicable' if validation_only else 'pending independent saved-evidence audit'}


def _current_process_identity():
    if os.name != 'nt':
        raise ReplayOwnerError('the replay owner requires a Windows process identity')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.argtypes = ()
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME))
    kernel.GetProcessTimes.restype = wintypes.BOOL
    created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), ctypes.byref(created), ctypes.byref(exited),
                                  ctypes.byref(kernel_time), ctypes.byref(user_time)):
        raise ReplayOwnerError('the replay owner birth identity is unavailable')
    return os.getpid(), (created.dwHighDateTime << 32) | created.dwLowDateTime


def check_supervisor_record(child, config, plan_sha256, child_sha256, output, command, verdict, *,
                            validation_only, parent_identity=None):
    """Bind clean child/Job cleanup and exact logs to the private postcheck.

    This observes only the child supervisor's own return. A separate observer
    must still retain and reap the process that called this function.
    """
    output = Path(output)
    info = output.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400):
        raise ReplayOwnerError('supervisor output inventory differs')
    names = set()
    with os.scandir(output) as entries:
        for entry in entries:
            if len(names) >= 3:
                raise ReplayOwnerError('supervisor output inventory exceeds its bound')
            names.add(entry.name)
    if names != {'stdout.txt', 'stderr.txt', 'supervisor.json'}:
        raise ReplayOwnerError('supervisor output inventory differs')
    raw = child._read(output / 'supervisor.json', 2 * 1024 * 1024)
    try:
        saved = json.loads(raw, object_pairs_hook=child._unique)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ReplayOwnerError('supervisor record is malformed') from error
    required_true = ('child_reaped', 'job_drained', 'job_closed', 'logs_complete', 'cleanup_verified')
    required_false = ('retry_safe', 'cancellation_requested', 'quarantined', 'log_overflow')
    if (type(verdict) is not dict or type(saved) is not dict or child._canonical(saved) != child._canonical(verdict)
            or verdict.get('status') != 'incomplete' or type(verdict.get('exit_code')) is not int
            or verdict['exit_code'] != 0 or type(verdict.get('postexit_active_processes')) is not int
            or verdict['postexit_active_processes'] != 0 or type(verdict.get('child_pid')) is not int
            or not 0 < verdict['child_pid'] <= 0xffffffff or verdict.get('supervisor_error') is not None
            or any(verdict.get(key) is not True for key in required_true)
            or any(verdict.get(key) is not False for key in required_false)
            or child._canonical(verdict.get('command')) != child._canonical(command)):
        raise ReplayOwnerError('supervised child did not finish and clean up')
    audit = inspect_child_postcheck(child, config, plan_sha256, child_sha256, validation_only=validation_only)
    if child._canonical(verdict.get('audit')) != child._canonical(audit):
        raise ReplayOwnerError('supervisor audit differs from the saved postcheck')
    for name in ('stdout', 'stderr'):
        data = child._read(output / (name + '.txt'), 256 * 1024)
        if verdict.get(name + '_sha256') != hashlib.sha256(data).hexdigest():
            raise ReplayOwnerError('supervisor stream digest differs')
    identity = _current_process_identity() if parent_identity is None else parent_identity
    if (type(identity) is not tuple or len(identity) != 2 or type(identity[0]) is not int
            or not 0 < identity[0] <= 0xffffffff or type(identity[1]) is not int
            or not 0 < identity[1] < 2 ** 64):
        raise ReplayOwnerError('replay parent process identity differs')
    return {'schema': 'mlg-injection-runtime-replay-parent-terminal-v1',
            'status': 'validated_no_model_forward' if validation_only else 'private_replay_pending_independent_audit',
            'plan_sha256': plan_sha256, 'child_source_sha256': child_sha256,
            'config_sha256': hashlib.sha256(child._canonical(config)).hexdigest(),
            'command_sha256': hashlib.sha256(child._canonical(command)).hexdigest(),
            'supervisor_sha256': hashlib.sha256(raw).hexdigest(),
            'child_postcheck_sha256': audit['child_postcheck_sha256'],
            'parent_pid': identity[0], 'parent_birth_100ns': identity[1],
            'cleanup_verified': True,
            'scientific_validation': 'not applicable' if validation_only else 'pending independent saved-evidence audit'}
