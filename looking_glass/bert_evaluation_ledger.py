"""Cooperative append-only attempt reservations for one frozen BERT study.

The trusted local filesystem and every caller must honor this protocol. These
records are not an execution attestation or a complete exposure history.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading

from .device_lease import DeviceLease, LeaseTimeoutError
from .supervisor import _supervisor_birth

MAX_ATTEMPTS = 32
MAX_RECORD_BYTES = 1024 * 1024
RESERVATION_SCHEMA = 'bert-evaluation-reservation-v1'
EXIT_SCHEMA = 'bert-evaluation-controller-exit-v1'
LOCK_NAME = 'bert-evaluation.lock'
_ID = re.compile(r'[0-9a-f]{32}\Z')
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def root_identity(root):
    root = Path(root)
    if not root.is_absolute():
        raise ValueError('ledger root must be absolute')
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError('ledger root must be an existing directory')
    return root, hashlib.sha256(os.path.normcase(str(root)).encode('utf-8')).hexdigest()


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate ledger key')
        out[key] = value
    return out


def read_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('ledger record is missing or not a regular file')
    with path.open('rb') as stream:
        raw = stream.read(MAX_RECORD_BYTES + 1)
    if not raw or len(raw) > MAX_RECORD_BYTES or not raw.endswith(b'\n'):
        raise ValueError('ledger record is incomplete or oversized')
    try:
        value = json.loads(raw.decode('ascii'), object_pairs_hook=_unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid ledger record') from error
    if type(value) is not dict:
        raise ValueError('ledger record must be an object')
    return value, hashlib.sha256(raw).hexdigest()


def write_record(path, value):
    """Publish one fsynced record without replacing an existing name."""
    raw = _canonical(value) + b'\n'
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError('ledger record is oversized')
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    primary = None
    try:
        with os.fdopen(fd, 'wb', buffering=0) as stream:
            if stream.write(raw) != len(raw):
                raise OSError('incomplete ledger write')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            os.unlink(temporary)
        except BaseException as error:
            if primary is not None:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise error from primary
                raise primary.with_traceback(primary.__traceback__) from error
            raise
    return hashlib.sha256(raw).hexdigest()


def _checked_binding(binding):
    """Take an owned JSON snapshot and validate the declared 64-row shape.

    The caller must derive these values from the verified frozen sources and
    canonical manifest; this function alone does not authenticate them.
    """
    try:
        owned = json.loads(_canonical(binding))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('ledger binding is not finite JSON') from error
    if (type(owned) is not dict or set(owned) != {'source_commit', 'source_sha256', 'protocol_sha256',
            'manifest_sha256', 'exposure_sha256', 'model', 'cpu_study', 'controller_cpu_study',
            'ledger_root_sha256', 'rows'}
            or type(owned['source_commit']) is not str or re.fullmatch('[0-9a-f]{40}', owned['source_commit']) is None
            or any(type(owned[k]) is not str or _SHA.fullmatch(owned[k]) is None for k in
                   ('protocol_sha256', 'manifest_sha256', 'exposure_sha256', 'ledger_root_sha256'))
            or type(owned['source_sha256']) is not dict or not owned['source_sha256']
            or any(type(k) is not str or type(v) is not str or _SHA.fullmatch(v) is None
                   for k, v in owned['source_sha256'].items())
            or type(owned['model']) is not dict or type(owned['cpu_study']) is not dict
            or type(owned['controller_cpu_study']) is not dict
            or type(owned['rows']) is not list or len(owned['rows']) != 64):
        raise ValueError('invalid frozen evaluation binding')
    ids = set()
    for row in owned['rows']:
        if (type(row) is not dict or set(row) != {'row_id', 'input_ids', 'attention_mask'}
                or type(row['row_id']) is not str or not row['row_id'].startswith('evaluation-')
                or row['row_id'] in ids or type(row['input_ids']) is not list or len(row['input_ids']) != 10
                or type(row['attention_mask']) is not list or len(row['attention_mask']) != 10
                or any(type(item) is not int or item < 0 for item in row['input_ids'])
                or any(type(item) is not int or item != 1 for item in row['attention_mask'])):
            raise ValueError('invalid evaluation input row')
        ids.add(row['row_id'])
    return owned


def verify_reservation(directory, run_id, binding):
    if type(run_id) is not str or _ID.fullmatch(run_id) is None:
        raise ValueError('invalid evaluation attempt identity')
    value, digest = read_record(Path(directory) / 'reservation.json')
    if (set(value) != {'schema', 'run_id', 'controller_pid', 'controller_birth', 'controller_supervision_run_id',
                       'binding', 'prior_attempts',
                       'prior_may_have_exposed', 'prior_complete', 'prior_audits'}
            or value['schema'] != RESERVATION_SCHEMA or value['run_id'] != run_id
            or type(value['controller_pid']) is not int or value['controller_pid'] <= 0
            or type(value['controller_birth']) is not str or not value['controller_birth']
            or type(value['controller_supervision_run_id']) is not str
            or _ID.fullmatch(value['controller_supervision_run_id']) is None
            or _canonical(value['binding']) != _canonical(_checked_binding(binding))
            or type(value['prior_attempts']) is not list
            or type(value['prior_may_have_exposed']) is not bool
            or value['prior_may_have_exposed'] != bool(value['prior_attempts'])
            or type(value['prior_complete']) is not bool or type(value['prior_audits']) is not list):
        raise ValueError('evaluation reservation does not match the frozen workload')
    ids = set()
    for item in value['prior_attempts']:
        if (type(item) is not dict or set(item) != {'run_id', 'reservation_sha256', 'terminal_sha256',
                                                   'exit_sha256', 'failure_sha256', 'cancel_sha256', 'status'}
                or type(item['run_id']) is not str or _ID.fullmatch(item['run_id']) is None
                or item['run_id'] in ids or item['run_id'] == run_id
                or any(type(item[key]) is not str or _SHA.fullmatch(item[key]) is None for key in
                       ('reservation_sha256', 'terminal_sha256', 'exit_sha256'))
                or any(item[key] is not None and (type(item[key]) is not str or _SHA.fullmatch(item[key]) is None)
                       for key in ('failure_sha256', 'cancel_sha256'))
                or item['status'] not in ('complete', 'incomplete')):
            raise ValueError('invalid prior-attempt commitment')
        ids.add(item['run_id'])
    if value['prior_complete'] != any(item['status'] == 'complete' for item in value['prior_attempts']):
        raise ValueError('reservation prior-completion flag does not match its history')
    expected_audits = [item['run_id'] for item in value['prior_attempts'] if item['status'] == 'complete']
    if ([item.get('run_id') if type(item) is dict else None for item in value['prior_audits']] != expected_audits
            or any(set(item) != {'run_id', 'python', 'torch', 'cuda_initialized_before_after',
                                 'denormals_preserved_calling_thread', 'model_forwarded'}
                   or type(item['python']) is not str or not item['python']
                   or item['torch'] != '2.8.0+cu128' or item['cuda_initialized_before_after'] is not False
                   or item['denormals_preserved_calling_thread'] is not True or item['model_forwarded'] is not False
                   for item in value['prior_audits'])):
        raise ValueError('reservation prior-audit record is inconsistent')
    return value, digest


def reservation_binding(reservation, digest):
    return {'ledger_root_sha256': reservation['binding']['ledger_root_sha256'],
            'reservation_sha256': digest, 'controller_supervision_run_id': reservation['controller_supervision_run_id'],
            'prior_attempts': reservation['prior_attempts'],
            'prior_may_have_exposed': reservation['prior_may_have_exposed'],
            'prior_complete': reservation['prior_complete'], 'prior_audits': reservation['prior_audits']}


def verify_owner_claim(root, expected_root_sha256, directory, run_id, binding, *, supervisor_pid, supervisor_birth):
    """Check the direct parent identity and lock contention, not lock ownership.

    The caller must also validate its supervisor session and rely on the
    trusted controller to retain its ledger session for the entire attempt.
    """
    root, identity = root_identity(root)
    directory = Path(directory).resolve(strict=True)
    if identity != expected_root_sha256 or directory.parent != root or directory.name != run_id:
        raise ValueError('owner reservation is outside the frozen ledger')
    reservation, digest = verify_reservation(directory, run_id, binding)
    if (type(supervisor_pid) is not int or supervisor_pid != os.getppid()
            or reservation['controller_pid'] != supervisor_pid
            or type(supervisor_birth) is not str or reservation['controller_birth'] != supervisor_birth):
        raise ValueError('owner is not the reserving controller child')
    try:
        unexpected = DeviceLease.acquire(root / LOCK_NAME, timeout=0)
    except LeaseTimeoutError:
        pass
    else:
        unexpected.release()
        raise ValueError('evaluation ledger lock is not contended')
    return reservation, digest


def _optional_sidecar(directory, name, run_id):
    path = directory / name
    if not path.exists() and not path.is_symlink():
        return None
    value, digest = read_record(path)
    if value.get('run_id') != run_id or value.get('status') != 'incomplete':
        raise ValueError('controller failure sidecar is malformed')
    return digest


def _artifact_digest(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ValueError('completed artifact is missing or not a regular file')
    size, digest = 0, hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError('completed artifact is oversized')
            digest.update(chunk)
    if not size:
        raise ValueError('completed artifact is empty')
    return {'bytes': size, 'sha256': digest.hexdigest()}


def _validate_controller_observation(observed, reservation):
    """Validate the trusted launcher's reaped direct-controller observation."""
    if (type(observed) is not dict or set(observed) != {
            'scope', 'run_id', 'pid', 'exit_code', 'clean', 'timed_out', 'reason'}
            or observed['scope'] != 'cooperative-cpu-controller-custody-only'
            or type(observed['run_id']) is not str or observed['run_id'] != reservation['controller_supervision_run_id']
            or type(observed['pid']) is not int or observed['pid'] != reservation['controller_pid']
            or type(observed['exit_code']) is not int or observed['clean'] is not True
            or observed['timed_out'] is not False
            or type(observed['reason']) is not str or not observed['reason']):
        raise ValueError('controller observation does not identify a clean, reaped controller')


def _validate_complete(directory, terminal, binding, reservation, reservation_sha):
    supervision = terminal.get('cpu_supervision')
    model = terminal.get('model')
    expected = {'started.json', 'preflight.json', 'owner-started.json',
                'owner-candidate.json', 'events.jsonl', 'reservation.json'}
    artifacts = terminal.get('artifacts')
    if (terminal.get('schema') != 'bert-agreement-controller-result-v1'
            or terminal.get('split') != 'evaluation' or terminal.get('family') is not None
            or _canonical(terminal.get('source_sha256')) != _canonical(binding['source_sha256'])
            or terminal.get('manifest_sha256') != binding['manifest_sha256']
            or _canonical(terminal.get('cpu_study')) != _canonical(binding['cpu_study'])
            or terminal.get('owner_state') != 'reaped' or type(terminal.get('owner_pid')) is not int
            or terminal['owner_pid'] <= 0 or type(terminal.get('owner_exit_code')) is not int
            or terminal['owner_exit_code'] != 0 or type(supervision) is not dict
            or supervision.get('scope') != 'cooperative-cpu-study-custody-only'
            or supervision.get('clean') is not True or supervision.get('timed_out') is not False
            or type(supervision.get('pid')) is not int or supervision['pid'] != terminal['owner_pid']
            or type(supervision.get('exit_code')) is not int or supervision['exit_code'] != 0
            or type(supervision.get('run_id')) is not str or _ID.fullmatch(supervision['run_id']) is None
            or type(model) is not dict or any(_canonical(model.get(key)) != _canonical(value)
                                                for key, value in binding['model'].items())
            or type(artifacts) is not dict or set(artifacts) != expected):
        raise ValueError('completed terminal lacks the frozen source, model or custody binding')
    for name in expected:
        limit = 16 * 1024 * 1024 if name == 'events.jsonl' else MAX_RECORD_BYTES
        if _canonical(artifacts[name]) != _canonical(_artifact_digest(directory / name, limit)):
            raise ValueError('completed artifact inventory does not match')
    if terminal.get('journal_sha256') != artifacts['events.jsonl']['sha256']:
        raise ValueError('completed terminal journal digest does not match')
    ledger_binding = reservation_binding(reservation, reservation_sha)
    interpretation = {'designation': 'replay' if reservation['prior_complete'] else 'first-complete-primary',
                      'prior_attempt_may_have_exposed_rows': reservation['prior_may_have_exposed'],
                      'unknown_history_and_pretraining': True}
    started, _ = read_record(directory / 'started.json')
    preflight, _ = read_record(directory / 'preflight.json')
    owner_started, _ = read_record(directory / 'owner-started.json')
    candidate, _ = read_record(directory / 'owner-candidate.json')
    run_id, pid = reservation['run_id'], terminal['owner_pid']
    expected_started = {'schema': 'bert-agreement-controller-start-v1', 'run_id': run_id,
                        'source_commit': binding['source_commit'], 'split': 'evaluation', 'family': None}
    if (set(started) != set(expected_started) | {'started_utc'}
            or any(_canonical(started.get(k)) != _canonical(v) for k, v in expected_started.items())
            or type(started['started_utc']) is not str or not started['started_utc']):
        raise ValueError('controller start does not match this reservation')
    expected_preflight = {'schema': 'bert-agreement-preflight-v1', 'run_id': run_id,
                          'source_commit': binding['source_commit'], 'source_sha256': binding['source_sha256'],
                          'manifest_sha256': binding['manifest_sha256'], 'split': 'evaluation', 'family': None,
                          'cpu_study': binding['cpu_study'], 'evaluation_ledger': ledger_binding}
    expected_owner = {'schema': 'bert-agreement-owner-start-v1', 'run_id': run_id, 'pid': pid,
                      'source_commit': binding['source_commit'], 'manifest_sha256': binding['manifest_sha256'],
                      'split': 'evaluation', 'family': None, 'cpu_study': binding['cpu_study'],
                      'evaluation_ledger': ledger_binding, 'cpu_owner_session_run_id': supervision['run_id']}
    if (_canonical(preflight) != _canonical(expected_preflight)
            or _canonical(owner_started) != _canonical(expected_owner)
            or _canonical(terminal.get('evaluation_ledger')) != _canonical(ledger_binding)
            or _canonical(terminal.get('evaluation_interpretation')) != _canonical(interpretation)
            or _canonical(candidate.get('evaluation_ledger')) != _canonical(ledger_binding)
            or candidate.get('cpu_owner_session_run_id') != supervision['run_id']
            or _canonical(candidate.get('source_sha256')) != _canonical(binding['source_sha256'])
            or candidate.get('manifest_sha256') != binding['manifest_sha256']
            or _canonical(candidate.get('cpu_study')) != _canonical(binding['cpu_study'])
            or candidate.get('split') != 'evaluation' or candidate.get('family') is not None
            or any(_canonical(candidate.get(key)) != _canonical(terminal.get(key))
                   for key in ('runtime', 'model', 'model_state_sha256_before_after'))):
        raise ValueError('completed owner evidence is not linked to this reservation')
    from .bert_agreement_cases import build_manifest, manifest_digest
    from .bert_agreement_study import (audit_cpu_evaluation_journal, validate_cpu_runtime_and_model,
                                       validate_owner_result)
    validate_cpu_runtime_and_model(terminal.get('runtime'), terminal.get('model'),
                                   terminal.get('model_state_sha256_before_after'))
    controller_audit = terminal.get('controller_audit_runtime')
    if (type(controller_audit) is not dict or set(controller_audit) != {'python', 'torch',
            'cuda_initialized_before_after', 'denormals_preserved_calling_thread', 'model_forwarded'}
            or controller_audit['python'] != terminal['runtime']['python']
            or controller_audit['torch'] != '2.8.0+cu128' or controller_audit['cuda_initialized_before_after'] is not False
            or controller_audit['denormals_preserved_calling_thread'] is not True
            or controller_audit['model_forwarded'] is not False):
        raise ValueError('completed controller audit runtime is inconsistent')
    manifest = build_manifest()
    expected_rows = [{'row_id': row['row_id'], 'input_ids': row['input_ids'],
                      'attention_mask': row['attention_mask']} for row in manifest['rows'] if row['split'] == 'evaluation']
    if manifest_digest() != binding['manifest_sha256'] or _canonical(expected_rows) != _canonical(binding['rows']):
        raise ValueError('ledger rows differ from the canonical evaluation manifest')
    validate_owner_result(candidate, run_id=run_id, pid=pid, exit_code=0,
                          source_commit=binding['source_commit'], journal_sha256=terminal['journal_sha256'])
    audited, observed_runtime = audit_cpu_evaluation_journal(manifest, directory / 'events.jsonl',
                                                             terminal['journal_sha256'])
    if (_canonical(candidate.get('audited')) != _canonical(audited)
            or _canonical(terminal.get('families')) != _canonical(audited['families'])
            or _canonical(terminal.get('summary')) != _canonical(audited['summary'])):
        raise ValueError('completed journal audit or CPU state does not match')
    return {'run_id': run_id, **observed_runtime}


class EvaluationLedger:
    """Hold one ledger lock while reserving and running a controller attempt."""

    def __init__(self, root, root_sha256, binding, lease):
        self.root = root
        self.root_sha256 = root_sha256
        self._binding = _canonical(binding)
        self.lease = lease
        self._thread = threading.current_thread()
        self._pid = os.getpid()
        self._closed = False
        self._reserved = False
        self._prior_audits = []

    @property
    def binding(self):
        return json.loads(self._binding)

    def _check(self):
        if self._closed or os.getpid() != self._pid or threading.current_thread() is not self._thread:
            raise RuntimeError('ledger session is closed or belongs to another thread/process')
        self.lease.check()

    @classmethod
    def acquire(cls, root, expected_root_sha256, binding, *, timeout=5):
        root, digest = root_identity(root)
        owned = _checked_binding(binding)
        if (type(expected_root_sha256) is not str or digest != expected_root_sha256
                or owned['ledger_root_sha256'] != digest):
            raise ValueError('evaluation ledger root does not match the frozen identity')
        lease = DeviceLease.acquire(root / LOCK_NAME, timeout=timeout)
        try:
            return cls(root, digest, owned, lease)
        except BaseException:
            lease.release()
            raise

    def prior_attempts(self):
        """Require complete local exit observations and one linear history."""
        self._check()
        entries = list(self.root.iterdir())
        attempts = []
        audit_metadata = {}
        for entry in entries:
            if entry.name == LOCK_NAME:
                continue
            if (entry.is_symlink() or not entry.is_dir() or _ID.fullmatch(entry.name) is None
                    or len(attempts) >= MAX_ATTEMPTS):
                raise ValueError('ledger has an unknown or excessive entry')
            reservation, digest = verify_reservation(entry, entry.name, self.binding)
            terminal, terminal_sha = read_record(entry / 'result.json')
            observed, exit_sha = read_record(entry / 'controller-exit.json')
            failure_sha = _optional_sidecar(entry, 'controller-failure.json', entry.name)
            cancel_sha = _optional_sidecar(entry, 'controller-cancelled.json', entry.name)
            if (type(terminal.get('status')) is not str or terminal['status'] not in ('complete', 'incomplete')
                    or terminal.get('run_id') != entry.name or terminal.get('source_commit') != self.binding['source_commit']
                    or terminal.get('owner_state') not in ('not_started', 'reaped')
                    or set(observed) != {'schema', 'run_id', 'exit_code', 'reservation_sha256', 'supervision'}
                    or observed['schema'] != EXIT_SCHEMA or observed['run_id'] != entry.name
                    or type(observed['exit_code']) is not int or observed['reservation_sha256'] != digest
                    or (observed['exit_code'] == 0 and
                        (terminal['status'] != 'complete' or failure_sha is not None or cancel_sha is not None))):
                raise ValueError('a prior attempt has unresolved status or exit evidence')
            _validate_controller_observation(observed['supervision'], reservation)
            if observed['supervision']['exit_code'] != observed['exit_code']:
                raise ValueError('controller exit does not match its supervision observation')
            status = 'complete' if observed['exit_code'] == 0 else 'incomplete'
            if status == 'complete':
                audit_metadata[entry.name] = _validate_complete(entry, terminal, self.binding, reservation, digest)
            commitment = {'run_id': entry.name, 'reservation_sha256': digest,
                          'terminal_sha256': terminal_sha, 'exit_sha256': exit_sha,
                          'failure_sha256': failure_sha, 'cancel_sha256': cancel_sha, 'status': status}
            attempts.append((reservation, commitment))
        attempts.sort(key=lambda item: len(item[0]['prior_attempts']))
        order = []
        for reservation, commitment in attempts:
            if reservation['prior_attempts'] != order:
                raise ValueError('evaluation ledger history is not a single append-only chain')
            order.append(commitment)
        self._prior_audits = [audit_metadata[item['run_id']] for item in order if item['status'] == 'complete']
        return order, any(item['status'] == 'complete' for item in order)

    def reserve(self, run_id, *, controller_supervision_run_id):
        self._check()
        if self._reserved:
            raise RuntimeError('a ledger session can reserve only one attempt')
        if (type(run_id) is not str or _ID.fullmatch(run_id) is None
                or type(controller_supervision_run_id) is not str
                or _ID.fullmatch(controller_supervision_run_id) is None):
            raise ValueError('invalid evaluation attempt identity')
        prior, completed = self.prior_attempts()
        if len(prior) >= MAX_ATTEMPTS or any(item['run_id'] == run_id for item in prior):
            raise ValueError('evaluation ledger is full or the attempt already exists')
        self._reserved = True
        directory = self.root / run_id
        directory.mkdir()
        value = {'schema': RESERVATION_SCHEMA, 'run_id': run_id, 'controller_pid': os.getpid(),
                 'controller_birth': _supervisor_birth(), 'controller_supervision_run_id': controller_supervision_run_id,
                 'binding': self.binding,
                 'prior_attempts': prior, 'prior_may_have_exposed': bool(prior), 'prior_complete': completed,
                 'prior_audits': self._prior_audits}
        digest = write_record(directory / 'reservation.json', value)
        return directory, digest, completed

    def release(self):
        self._check()
        # An asynchronous exception after the OS unlock must not leave a
        # stale session able to reserve while another process owns the lock.
        self._closed = True
        self.lease.release()


def record_controller_exit(root, expected_root_sha256, run_id, binding, supervision):
    """Record the trusted launcher's observation of its reaped controller."""
    if type(run_id) is not str or _ID.fullmatch(run_id) is None:
        raise ValueError('controller exit code or attempt identity is invalid')
    ledger = EvaluationLedger.acquire(root, expected_root_sha256, binding)
    try:
        directory = ledger.root / run_id
        if directory.is_symlink() or not directory.is_dir() or directory.parent != ledger.root:
            raise ValueError('invalid evaluation attempt directory')
        reservation, digest = verify_reservation(directory, run_id, ledger.binding)
        _validate_controller_observation(supervision, reservation)
        return write_record(directory / 'controller-exit.json',
                            {'schema': EXIT_SCHEMA, 'run_id': run_id, 'exit_code': supervision['exit_code'],
                             'reservation_sha256': digest, 'supervision': supervision})
    finally:
        ledger.release()
