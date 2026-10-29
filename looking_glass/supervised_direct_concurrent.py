"""Source-bound composition of the fixed owner and two concurrent clients.

The concurrent host envelope is not measured yet, so this entry cannot launch
an owner. Its caller retains the run, prepared service and client objects for
reconciliation; the existing owner and supervisor remain responsible for all
device work and teardown.
"""
from dataclasses import dataclass
import hashlib
import math
import re
from threading import Lock
from types import MethodType

from .direct_concurrent_clients import ConcurrentClientReport, ConcurrentClients
from .gpu_service import ProfileFiles
from .trusted_service_profile import TrustedServiceSource, prepare_trusted_source
from .two_team_service_smoke import _clean_supervisor
from .two_token_service import _PreparedService


class ConcurrentServiceError(RuntimeError):
    """The supervised two-client composition is unavailable or incomplete."""


@dataclass(frozen=True)
class SupervisedConcurrentResult:
    """Source pins and client observations after checked supervisor teardown."""

    source_base_sha256: str
    asset_manifest_sha256: str
    approved_profile_ids: tuple[str, str, str]
    clients: ConcurrentClientReport
    teardown_verified: bool


def _require_measured_host_envelope(service, source_base_sha256, profile_ids):
    """Fail closed until the exact concurrent IPC host envelope is reviewed.

    The worker's GPU profiles and IPC frame limits are not host-RSS profiles.
    A future implementation must validate separately approved, source-bound
    measurements and admission for this composition. It must not accept a
    grouped-worker profile as evidence for this serial owner.
    """
    raise ConcurrentServiceError('source-bound concurrent host envelope is not measured and approved')


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
    def __init__(self, service, clients):
        if type(service) is not _PreparedService or type(clients) is not ConcurrentClients:
            raise TypeError('the exact prepared service and concurrent clients are required')
        if clients._service is not service or clients._started:
            raise ConcurrentServiceError('client plan does not match the unstarted service')
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
                or not _canonical_methods(service, self._clients)):
            raise ConcurrentServiceError('prepared service binding changed')

    def run(self, *, client_timeout=60, shutdown_timeout=30):
        """Run once, returning only after supervisor and source checks pass.

        A timeout or exception is not a clean result. The caller retains this
        object and may use the original clients and service to finish recovery.
        """
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
        _require_measured_host_envelope(self._service, self._source_hash, self._pins)
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
                                          self._client_report, True)

    def wait_supervisor(self, *, timeout=30):
        """Recheck the originally pinned controller after an uncertain return."""
        return self._recovery.wait(timeout=timeout)
