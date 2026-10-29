"""Source-bound composition of the fixed owner and two concurrent clients.

This entry defaults closed until a trusted operator supplies an approved,
matching real-owner host observation. Its retained logical host claim is not
an OS memory reservation; the owner and supervisor remain responsible for
device work, failures and teardown.
"""
from dataclasses import dataclass
import ctypes
import hashlib
import json
import math
import os
import re
from threading import Lock
from types import MethodType

from .direct_concurrent_clients import ConcurrentClientReport, ConcurrentClients, _Slot
from .gpu_service import ProfileFiles, _load_profile_registry
from .trusted_service_profile import TrustedServiceSource, prepare_trusted_source
from .two_team_service_smoke import _clean_supervisor
from .two_token_service import _PreparedService


class ConcurrentServiceError(RuntimeError):
    """The supervised two-client composition is unavailable or incomplete."""


_HOST_CLAIM_LOCK = Lock()
_HOST_CLAIM_SLOT = [None]


class _HostClaim:
    """One retained process-local logical charge, not an OS reservation."""

    def __init__(self, owner=None):
        self._record = None
        self._owner = owner

    @property
    def commitment_bytes(self):
        return None if self._record is None else self._record[0]

    @property
    def required_headroom_bytes(self):
        return None if self._record is None else self._record[1]

    @property
    def admission_mode(self):
        return None if self._record is None else self._record[3]


@dataclass(frozen=True)
class SupervisedConcurrentResult:
    """Source pins and client observations after checked supervisor teardown."""

    source_base_sha256: str
    asset_manifest_sha256: str
    approved_profile_ids: tuple[str, str, str]
    clients: ConcurrentClientReport
    teardown_verified: bool
    admission_mode: str = 'approved-host-evidence'


@dataclass(frozen=True)
class ApprovedHostEvidence:
    """Private evidence selected by a trusted operator using a separate SHA-256 pin.

    Matching the digest selects bytes; it does not authenticate their origin
    or prove the observation actually took place.
    """

    document: bytes
    approved_sha256: str


def _validate_host_evidence(raw, approval_sha256, binding):
    """Validate the bytes selected by a trusted operator; hashes are not attestations."""
    def reject():
        raise ConcurrentServiceError('approved concurrent host evidence is invalid or mismatched')

    def integer(value, minimum=0, maximum=2 ** 63 - 1):
        return type(value) is int and minimum <= value <= maximum

    if (type(raw) is not bytes or not 0 < len(raw) <= 2 * 1024 ** 2
            or type(approval_sha256) is not str
            or re.fullmatch(r'[0-9a-f]{64}', approval_sha256) is None
            or hashlib.sha256(raw).hexdigest() != approval_sha256):
        reject()
    try:
        document = json.loads(raw)
        canonical = json.dumps(document, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=True, allow_nan=False).encode('ascii')
    except (TypeError, ValueError, UnicodeError):
        reject()
    if canonical != raw or type(document) is not dict:
        reject()
    fields = {'schema', 'source_base_sha256', 'asset_manifest_sha256', 'runtime_settings_sha256',
              'profile_ids', 'workload', 'measurement', 'commitment_bytes', 'margin_bytes', 'reserve_bytes'}
    if (set(document) != fields or document['schema'] != 'serial-two-client-host-evidence-v1'
            or type(binding) is not dict or set(binding) != {'source_base_sha256', 'asset_manifest_sha256',
                'runtime_settings_sha256', 'python_executable_sha256', 'profile_ids', 'plan'}):
        reject()
    for key in ('source_base_sha256', 'asset_manifest_sha256', 'runtime_settings_sha256'):
        value = binding[key]
        if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None or document[key] != value:
            reject()
    executable = binding['python_executable_sha256']
    if type(executable) is not str or re.fullmatch(r'[0-9a-f]{64}', executable) is None:
        reject()
    pins, plan = binding['profile_ids'], binding['plan']
    if (type(pins) is not tuple or len(pins) != 3
            or any(type(pin) is not str or re.fullmatch(r'[0-9a-f]{64}', pin) is None for pin in pins)
            or len(set(pins)) != 3
            or document['profile_ids'] != list(pins) or type(plan) is not tuple or len(plan) != 2
            or any(type(item) is not tuple or len(item) != 3 or type(item[0]) is not str
                   or not item[0] or type(item[1]) is not int or item[1] not in (29, 43, 71)
                   or item[2] is not False for item in plan)
            or plan[0][0] == plan[1][0]):
        reject()
    workload = {'input_ids': [15496, 995], 'shape': [1, 2], 'max_connections': 2,
                'max_pending': 2, 'max_pending_per_team': 1, 'max_request_bytes': 4096,
                'max_response_bytes': 4 * 1024 ** 2, 'trace_bytes': 0,
                'plan': [{'team': team, 'seed': seed, 'cancel': cancel} for team, seed, cancel in plan]}
    if (type(document['workload']) is not dict
            or json.dumps(document['workload'], sort_keys=True, separators=(',', ':'), allow_nan=False)
            != json.dumps(workload, sort_keys=True, separators=(',', ':'), allow_nan=False)):
        reject()
    measurement = document['measurement']
    measurement_fields = {'kind', 'controller_exit', 'owner_exit', 'supervisor_clean', 'supervisor_reaped',
                          'timed_out', 'marker_absent', 'clients_reconciled', 'client_statuses',
                          'model_forward_calls', 'physical_total_bytes', 'sample_caller', 'processes', 'process_scope',
                          'lifetime_peaks', 'samples'}
    if (type(measurement) is not dict or set(measurement) != measurement_fields
            or measurement['kind'] != 'supervised-gpu-owner'
            or type(measurement['controller_exit']) is not int or measurement['controller_exit'] != 0
            or type(measurement['owner_exit']) is not int or measurement['owner_exit'] != 0
            or measurement['supervisor_clean'] is not True or measurement['supervisor_reaped'] is not True
            or measurement['timed_out'] is not False or measurement['marker_absent'] is not True
            or measurement['clients_reconciled'] is not True
            or measurement['client_statuses'] != ['succeeded', 'succeeded']
            or type(measurement['model_forward_calls']) is not int or measurement['model_forward_calls'] != 3
            or not integer(measurement['physical_total_bytes'], 1024 ** 3, 4 * 1024 ** 4)
            or measurement['sample_caller'] != 'controller'
            or measurement['process_scope'] != 'controller-and-owner-only; observer-and-helpers-excluded'):
        reject()
    processes, peaks = measurement['processes'], measurement['lifetime_peaks']
    if (type(processes) is not dict or set(processes) != {'controller', 'owner'}
            or any(type(value) is not dict or set(value) != {'pid', 'created_100ns', 'executable_sha256',
                                                            'handle_verified'}
                   or not integer(value['pid'], 1, 2 ** 32 - 1)
                   or not integer(value['created_100ns'], 1)
                   or value['executable_sha256'] != executable or value['handle_verified'] is not True
                   for value in processes.values())
            or processes['controller']['pid'] == processes['owner']['pid']):
        reject()
    peak_fields = {'controller_working_set_bytes', 'owner_working_set_bytes',
                   'controller_private_commit_bytes', 'owner_private_commit_bytes'}
    if (type(peaks) is not dict or set(peaks) != peak_fields
            or any(not integer(value, 1) for value in peaks.values())):
        reject()
    samples = measurement['samples']
    phases = ('before_start', 'owner_ready', 'clients_active', 'clients_done', 'owner_exiting', 'owner_reaped')
    names = {'timestamp_ns', 'phase', 'controller_working_set_bytes', 'controller_private_commit_bytes',
             'owner_working_set_bytes', 'owner_private_commit_bytes', 'available_physical_bytes',
             'available_system_commit_bytes', 'available_caller_commit_bytes'}
    if type(samples) is not list or not 6 <= len(samples) <= 8192:
        reject()
    phase_index = -1
    previous = None
    max_working = max_private = 0
    for sample in samples:
        if (type(sample) is not dict or set(sample) != names or sample['phase'] not in phases
                or any(not integer(sample[key]) for key in names - {'phase'})
                or sample['controller_working_set_bytes'] == 0
                or sample['controller_private_commit_bytes'] == 0
                or sample['available_physical_bytes'] > measurement['physical_total_bytes']):
            reject()
        current = phases.index(sample['phase'])
        if current < phase_index or current > phase_index + 1:
            reject()
        phase_index = current
        stamp = sample['timestamp_ns']
        if previous is not None and not 0 < stamp - previous <= 2_000_000_000:
            reject()
        previous = stamp
        owner_live = sample['phase'] not in ('before_start', 'owner_reaped')
        if ((sample['owner_working_set_bytes'] > 0) != owner_live
                or (sample['owner_private_commit_bytes'] > 0) != owner_live):
            reject()
        if any(sample[name] > peaks[name] for name in peak_fields):
            reject()
        max_working = max(max_working, sample['controller_working_set_bytes'] + sample['owner_working_set_bytes'])
        max_private = max(max_private, sample['controller_private_commit_bytes'] + sample['owner_private_commit_bytes'])
    peak_working = peaks['controller_working_set_bytes'] + peaks['owner_working_set_bytes']
    peak_private = peaks['controller_private_commit_bytes'] + peaks['owner_private_commit_bytes']
    commitment, margin, reserve = (document[key] for key in ('commitment_bytes', 'margin_bytes', 'reserve_bytes'))
    if (phase_index != len(phases) - 1 or not integer(margin, 1024 ** 3, 16 * 1024 ** 3)
            or not integer(reserve, 2 * 1024 ** 3, 64 * 1024 ** 3)
            or not integer(commitment, max(max_working, max_private, peak_working, peak_private) + margin,
                           16 * 1024 ** 3)
            or any(sample[name] < reserve for sample in samples for name in
                   ('available_physical_bytes', 'available_system_commit_bytes', 'available_caller_commit_bytes'))):
        reject()
    return {'commitment_bytes': commitment, 'margin_bytes': margin, 'reserve_bytes': reserve,
            'physical_total_bytes': measurement['physical_total_bytes'],
            'observed_peak_working_set_bytes': max_working,
            'observed_peak_private_commit_bytes': max_private,
            'lifetime_peak_working_set_sum_bytes': peak_working,
            'lifetime_peak_private_commit_sum_bytes': peak_private}


def _windows_host_snapshot():
    """Read physical, system-commit and caller-commit headroom on Windows."""
    if os.name != 'nt':
        raise ConcurrentServiceError('Windows host telemetry is unavailable')

    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                    ('total_physical', ctypes.c_uint64), ('available_physical', ctypes.c_uint64),
                    ('total_pagefile', ctypes.c_uint64), ('available_pagefile', ctypes.c_uint64),
                    ('total_virtual', ctypes.c_uint64), ('available_virtual', ctypes.c_uint64),
                    ('available_extended_virtual', ctypes.c_uint64)]

    class PerformanceInfo(ctypes.Structure):
        _fields_ = [('size', ctypes.c_uint32),
                    *[(name, ctypes.c_size_t) for name in ('commit_total', 'commit_limit', 'commit_peak',
                        'physical_total', 'physical_available', 'system_cache', 'kernel_total',
                        'kernel_paged', 'kernel_nonpaged', 'page_size')],
                    ('handle_count', ctypes.c_uint32), ('process_count', ctypes.c_uint32),
                    ('thread_count', ctypes.c_uint32)]

    try:
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        performance = PerformanceInfo()
        performance.size = ctypes.sizeof(performance)
        global_status = ctypes.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx
        global_status.argtypes = [ctypes.POINTER(MemoryStatus)]
        global_status.restype = ctypes.c_int
        performance_info = ctypes.WinDLL('psapi', use_last_error=True).GetPerformanceInfo
        performance_info.argtypes = [ctypes.POINTER(PerformanceInfo), ctypes.c_uint32]
        performance_info.restype = ctypes.c_int
        if not global_status(ctypes.byref(memory)) or not performance_info(ctypes.byref(performance), performance.size):
            raise OSError('Windows memory query failed')
        if (not 0 < performance.page_size <= 2 * 1024 ** 2
                or performance.commit_total > performance.commit_limit):
            raise OSError('Windows memory counters are inconsistent')
        snapshot = {'physical_total_bytes': int(memory.total_physical),
                    'available_physical_bytes': int(memory.available_physical),
                    'available_system_commit_bytes': int((performance.commit_limit - performance.commit_total)
                                                         * performance.page_size),
                    # This is this controller process's available commit, not a child-process limit.
                    'available_caller_commit_bytes': int(memory.available_pagefile)}
        if (not 0 < snapshot['available_physical_bytes'] <= snapshot['physical_total_bytes']
                or any(value < 0 or value >= 2 ** 63 for value in snapshot.values())):
            raise OSError('Windows memory counters are invalid')
        return snapshot
    except (OSError, AttributeError, ValueError, OverflowError):
        raise ConcurrentServiceError('Windows host telemetry is unavailable or invalid') from None


def _check_host_headroom(snapshot, envelope):
    """Check a fresh point observation; this does not reserve OS memory."""
    fields = {'physical_total_bytes', 'available_physical_bytes',
              'available_system_commit_bytes', 'available_caller_commit_bytes'}
    if (type(snapshot) is not dict or set(snapshot) != fields or type(envelope) is not dict
            or any(type(snapshot[key]) is not int or not 0 <= snapshot[key] < 2 ** 63 for key in fields)
            or type(envelope.get('physical_total_bytes')) is not int
            or type(envelope.get('commitment_bytes')) is not int
            or type(envelope.get('reserve_bytes')) is not int
            or not 0 < envelope['commitment_bytes'] <= 16 * 1024 ** 3
            or not 2 * 1024 ** 3 <= envelope['reserve_bytes'] <= 64 * 1024 ** 3
            or snapshot['physical_total_bytes'] != envelope['physical_total_bytes']
            or snapshot['available_physical_bytes'] > snapshot['physical_total_bytes']):
        raise ConcurrentServiceError('host telemetry or admitted envelope is invalid')
    required = max(8 * 1024 ** 3, envelope['commitment_bytes'] + envelope['reserve_bytes'])
    if any(snapshot[name] < required for name in fields - {'physical_total_bytes'}):
        raise ConcurrentServiceError('insufficient fresh host physical or commit headroom')
    return required


def _reserve_host_claim(claim, envelope, *, development_bootstrap=False):
    """Atomically retain one logical host charge until this process exits."""
    if type(claim) is not _HostClaim or claim._record is not None:
        raise ConcurrentServiceError('host claim is invalid or already used')
    if development_bootstrap and envelope != {'commitment_bytes': 8 * 1024 ** 3,
                                               'reserve_bytes': 2 * 1024 ** 3}:
        raise ConcurrentServiceError('invalid development host allowance')
    with _HOST_CLAIM_LOCK:
        if _HOST_CLAIM_SLOT[0] is not None:
            raise ConcurrentServiceError('a previous host claim remains active')
        snapshot = _windows_host_snapshot()
        if development_bootstrap:
            envelope = dict(envelope, physical_total_bytes=snapshot.get('physical_total_bytes'))
        required = _check_host_headroom(snapshot, envelope)
        claim._record = (envelope['commitment_bytes'], required, dict(snapshot),
                         'development-bootstrap' if development_bootstrap else 'approved-host-evidence')
        # The global slot keeps the charge even if returning to the caller is
        # interrupted. It is deliberately never released within this process.
        _HOST_CLAIM_SLOT[0] = claim


def _prepared_host_binding(service, source_base_sha256, profile_ids, *, clients=None, claim=None):
    """Validate the shared source, profile, client and retained-claim binding."""
    if (type(service) is not _PreparedService
            or type(clients) is not ConcurrentClients or clients._service is not service or clients._started
            or type(clients._slots) is not tuple or len(clients._slots) != 2
            or any(type(slot) is not _Slot for slot in clients._slots)
            or tuple(slot.team for slot in clients._slots) != service._teams
            or type(service._source) is not TrustedServiceSource
            or type(service._source.source_base) is not bytes
            or not 0 < len(service._source.source_base) <= 32 * 1024 ** 2
            or type(service._source.asset_manifest) is not bytes
            or hashlib.sha256(service._source.source_base).hexdigest() != source_base_sha256
            or type(service._profiles) is not tuple or len(service._profiles) != 3
            or any(type(item) is not ProfileFiles for item in service._profiles)
            or tuple(item.approved_profile_id for item in service._profiles) != profile_ids
            or type(claim) is not _HostClaim or type(claim._owner) is not SupervisedConcurrentRun
            or claim._owner._service is not service or claim._owner._clients is not clients
            or claim._owner._host_claim is not claim):
        raise ConcurrentServiceError('source-bound concurrent host evidence is unavailable')
    _, keys = _load_profile_registry(service._profiles, service._identity)
    if (type(keys) is not tuple or len(keys) != 3
            or any(type(getattr(key, 'runtime_settings_sha256', None)) is not str for key in keys)
            or len({key.runtime_settings_sha256 for key in keys}) != 1):
        raise ConcurrentServiceError('approved serial profiles do not share one runtime')
    try:
        source = json.loads(service._source.source_base)
        executable = source['python_executable']
        if (type(source) is not dict or source.get('schema') != 'trusted-service-source-base-v1'
                or type(executable) is not dict or set(executable) != {'size', 'sha256'}
                or type(executable['size']) is not int or executable['size'] <= 0):
            raise ValueError('invalid executable binding')
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise ConcurrentServiceError('prepared source executable identity is unavailable') from None
    binding = {'source_base_sha256': source_base_sha256,
               'asset_manifest_sha256': hashlib.sha256(service._source.asset_manifest).hexdigest(),
               'runtime_settings_sha256': keys[0].runtime_settings_sha256,
               'python_executable_sha256': executable['sha256'], 'profile_ids': profile_ids,
               'plan': tuple((slot.team, slot.seed, slot.cancel) for slot in clients._slots)}
    if (type(binding['python_executable_sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', binding['python_executable_sha256']) is None
            or type(binding['runtime_settings_sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', binding['runtime_settings_sha256']) is None
            or any(type(team) is not str or not team or type(seed) is not int
                   or seed not in (29, 43, 71) or cancel is not False
                   for team, seed, cancel in binding['plan'])
            or binding['plan'][0][0] == binding['plan'][1][0]):
        raise ConcurrentServiceError('fixed no-cancellation client plan is unavailable')
    return binding


def _require_measured_host_envelope(service, source_base_sha256, profile_ids, *, evidence=None,
                                    clients=None, claim=None):
    """Require an operator-approved exact-path observation and fresh host screen."""
    if type(evidence) is not ApprovedHostEvidence:
        raise ConcurrentServiceError('source-bound concurrent host evidence is unavailable')
    binding = _prepared_host_binding(service, source_base_sha256, profile_ids, clients=clients, claim=claim)
    envelope = _validate_host_evidence(evidence.document, evidence.approved_sha256, binding)
    _reserve_host_claim(claim, envelope)


def _require_development_bootstrap(service, source_base_sha256, profile_ids, *, clients, claim):
    """Screen one unmeasured fixed-path run under a conservative policy allowance."""
    _prepared_host_binding(service, source_base_sha256, profile_ids, clients=clients, claim=claim)
    _reserve_host_claim(claim, {'commitment_bytes': 8 * 1024 ** 3, 'reserve_bytes': 2 * 1024 ** 3},
                        development_bootstrap=True)


def _canonical_methods(service, clients):
    """Reject accidental instance overrides of the source-bound call path."""
    def matches(owner, name, function):
        method = getattr(owner, name, None)
        return (type(method) is MethodType and method.__self__ is owner
                and method.__func__ is function)
    return (matches(service, 'start', _PreparedService.start)
            and matches(service, 'client', _PreparedService.client)
            and matches(clients, 'start', ConcurrentClients.start)
            and matches(clients, 'wait', ConcurrentClients.wait))


class SupervisedConcurrentRun:
    """A caller-retained, one-shot composition; retain it even on interruption.

    The caller separately prepares and retains the exact fixed service and
    its two-client plan before constructing this object. On an uncertain
    return, use those same objects to reconcile client threads, the supervisor
    and the unsafe marker; do not retry a request or launch another owner.
    """
    def __init__(self, service, clients, *, host_evidence=None):
        if type(service) is not _PreparedService or type(clients) is not ConcurrentClients:
            raise TypeError('the exact prepared service and concurrent clients are required')
        if clients._service is not service or clients._started:
            raise ConcurrentServiceError('client plan does not match the unstarted service')
        if (type(clients._slots) is not tuple or len(clients._slots) != 2
                or any(type(slot) is not _Slot for slot in clients._slots)):
            raise ConcurrentServiceError('invalid concurrent client plan')
        if host_evidence is not None and type(host_evidence) is not ApprovedHostEvidence:
            raise TypeError('host evidence must be an approved evidence object')
        source, profiles = service._source, service._profiles
        if (type(source) is not TrustedServiceSource or source.repo_root != service._repo
                or type(source.source_base) is not bytes or not source.source_base
                or type(source.asset_manifest) is not bytes or not source.asset_manifest
                or type(profiles) is not tuple or len(profiles) != 3
                or any(type(item) is not ProfileFiles for item in profiles)):
            raise ConcurrentServiceError('invalid prepared source or profile binding')
        pins = tuple(item.approved_profile_id for item in profiles)
        if (any(type(pin) is not str or re.fullmatch(r'[0-9a-f]{64}', pin) is None for pin in pins)
                or len(set(pins)) != 3):
            raise ConcurrentServiceError('three distinct approved profile pins are required')
        self._service, self._clients = service, clients
        self._source, self._profiles, self._pins = source, profiles, pins
        self._repo, self._identity, self._controller, self._launch, self._teams = (
            service._repo, service._identity, service._controller, service._launch, service._teams)
        # Root a separate facade over the original controller for cleanup if
        # a cooperative caller accidentally changes the prepared binding.
        self._recovery = _PreparedService(controller=self._controller, launch=self._launch,
            identity=self._identity, source=self._source, profiles=self._profiles,
            teams=self._teams, repo=self._repo)
        self._source_hash = hashlib.sha256(source.source_base).hexdigest()
        self._asset_hash = hashlib.sha256(source.asset_manifest).hexdigest()
        self._host_evidence = host_evidence
        self._approved_host_evidence = host_evidence
        self._host_document = None if host_evidence is None else host_evidence.document
        self._host_pin = None if host_evidence is None else host_evidence.approved_sha256
        self._slots = clients._slots
        self._slot_plan = tuple((slot.team, slot.seed, slot.cancel) for slot in clients._slots)
        self._host_claim = _HostClaim(self)
        self._original_host_claim = self._host_claim
        self._claim = Lock()
        self._claimed = False
        # The retained caller-owned object roots errors from each boundary.
        self._run_failure = self._shutdown_failure = self._wait_failure = self._verification_failure = None
        self._client_report = self._supervision = None

    def _check_binding(self):
        service = self._service
        if (self._clients._service is not service or service._source is not self._source
                or service._profiles is not self._profiles or service._repo != self._repo
                or service._identity is not self._identity or service._controller is not self._controller
                or service._launch is not self._launch or service._teams is not self._teams
                or tuple(item.approved_profile_id for item in service._profiles) != self._pins
                or self._clients._slots is not self._slots
                or any(type(slot) is not _Slot or type(slot.team) is not str
                       or type(slot.seed) is not int or type(slot.cancel) is not bool
                       or (slot.team, slot.seed, slot.cancel) != original
                       for slot, original in zip(self._slots, self._slot_plan))
                or self._host_claim is not self._original_host_claim
                or self._host_claim._owner is not self
                or self._host_evidence is not self._approved_host_evidence
                or (self._host_evidence is not None and
                    (self._host_evidence.document is not self._host_document
                     or self._host_evidence.approved_sha256 != self._host_pin))
                or not _canonical_methods(service, self._clients)):
            raise ConcurrentServiceError('prepared service binding changed')

    def run(self, *, client_timeout=60, shutdown_timeout=30):
        """Run once, returning only after supervisor and source checks pass.

        A timeout or exception is not a clean result. The caller retains this
        object and may use the original clients and service to finish recovery.
        """
        return self._run(client_timeout, shutdown_timeout, development_bootstrap=False)

    def run_development_bootstrap(self, *, client_timeout=60, shutdown_timeout=30):
        """Run the fixed path once under the unmeasured development allowance.

        This is not host-profile approval and still requires all three exact
        independently approved worker profiles and supervised device admission.
        """
        return self._run(client_timeout, shutdown_timeout, development_bootstrap=True)

    def _run(self, client_timeout, shutdown_timeout, *, development_bootstrap):
        for value in (client_timeout, shutdown_timeout):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 300:
                raise ValueError('timeouts must be finite and positive')
        with self._claim:
            if self._claimed:
                raise ConcurrentServiceError('the supervised client run is one-shot')
            self._claimed = True
        self._check_binding()
        if self._clients._started:
            raise ConcurrentServiceError('client plan was already started')
        if development_bootstrap:
            if self._host_evidence is not None:
                raise ConcurrentServiceError('development bootstrap cannot consume approved host evidence')
            _require_development_bootstrap(self._service, self._source_hash, self._pins,
                                           clients=self._clients, claim=self._host_claim)
        else:
            _require_measured_host_envelope(self._service, self._source_hash, self._pins,
                                            evidence=self._host_evidence, clients=self._clients,
                                            claim=self._host_claim)
        self._check_binding()
        attempted = False
        try:
            # The fixed controller rechecks the exact source, assets, three
            # approved profiles, runtime and physical identity on its own path.
            attempted = True
            self._service.start(startup_timeout=60, owner_timeout=600,
                                lease_timeout=30, terminate_grace=5)
            self._check_binding()
            self._clients.start()
            self._client_report = self._clients.wait(timeout=client_timeout)
            if self._client_report is None:
                raise ConcurrentServiceError('client completion is unresolved')
        except BaseException as exc:
            self._run_failure = exc
        finally:
            if attempted:
                try:
                    self._supervision = self._recovery.shutdown(timeout=shutdown_timeout)
                except BaseException as exc:
                    self._shutdown_failure = exc
                    try:
                        self._supervision = self._recovery.wait(timeout=shutdown_timeout)
                    except BaseException as wait_exc:
                        self._wait_failure = wait_exc
        # A later interruption is not an ordinary cleanup failure: preserve
        # every original error in its own slot and propagate the newest
        # non-Exception before choosing an earlier ordinary failure.
        for failure in (self._wait_failure, self._shutdown_failure, self._run_failure):
            if failure is not None and not isinstance(failure, Exception):
                raise failure
        if self._run_failure is not None:
            raise self._run_failure
        if self._shutdown_failure is not None:
            raise self._shutdown_failure
        if not _clean_supervisor(self._supervision, self._recovery.unsafe_marker_path):
            raise ConcurrentServiceError('supervised owner teardown was not verified')
        try:
            self._check_binding()
            if (type(self._client_report) is not ConcurrentClientReport
                    or prepare_trusted_source(self._source.repo_root, self._source.cache_root) != self._source):
                raise ConcurrentServiceError('source or client report changed before publication')
        except BaseException as exc:
            self._verification_failure = exc
            raise
        return SupervisedConcurrentResult(self._source_hash, self._asset_hash, self._pins,
                                          self._client_report, True,
                                          'development-bootstrap' if development_bootstrap
                                          else 'approved-host-evidence')

    def wait_supervisor(self, *, timeout=30):
        """Recheck the originally pinned controller after an uncertain return."""
        return self._recovery.wait(timeout=timeout)
