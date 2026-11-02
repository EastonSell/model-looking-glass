"""Bounded operator smoke for the reviewed fixed two-team service.

This is not a launcher, profiler, fairness benchmark or general IPC client.
Each entry accepts only a prepared fixed service and remains gated until its
actual direct or fair owner declares the matching reviewed contract.
"""
from dataclasses import dataclass
import hashlib
import math
import os
import re
import struct
from threading import Lock
from weakref import WeakSet

from . import gpu_service as _owner_module
from .supervisor import SupervisionResult
from .two_token_service import _PreparedService, _REQUIRED_COMPLETION_CONTRACT


_REQUIRED_FAIR_CONTRACT = 'two-team-fair-service-v1'
_REQUIRED_DIRECT_CONTRACT = 'two-team-direct-service-v1'
_REQUEST_ID = 'two-team-service-smoke-v1'
_SHAPE = (1, 2, 50257)
_NUM_VALUES = 100514
_BROKER_ID = re.compile(r'[0-9a-f]{32}\Z', re.ASCII)
_CLAIM_LOCK = Lock()
_CLAIMED = WeakSet()


class ServiceSmokeError(RuntimeError):
    """The fixed smoke or its independently checked teardown did not pass."""


@dataclass(frozen=True)
class TwoTeamSmokeResult:
    """Shareable summary; no endpoint, credential or physical UUID is included."""

    requests_succeeded: int
    logits_shape: tuple
    float32_sha256: str
    teardown_verified: bool


def _result_identity(response, team):
    if (type(response) is not dict or type(response.get('version')) is not int
            or response['version'] != 1 or response.get('status') != 'succeeded'
            or response.get('team_id') != team or response.get('request_id') != _REQUEST_ID
            or type(response.get('job_id')) is not int or response['job_id'] <= 0
            or type(response.get('broker_id')) is not str
            or _BROKER_ID.fullmatch(response['broker_id']) is None):
        raise ServiceSmokeError('service returned an invalid request outcome')
    payload = response.get('result')
    if (type(payload) is not dict or set(payload) != {'shape', 'dtype', 'logits'}
            or type(payload['shape']) is not list or tuple(payload['shape']) != _SHAPE
            or any(type(dimension) is not int for dimension in payload['shape'])
            or payload['dtype'] != 'torch.float32' or type(payload['logits']) is not list
            or len(payload['logits']) != _NUM_VALUES):
        raise ServiceSmokeError('service returned an invalid bounded logit payload')
    data = bytearray(_NUM_VALUES * 4)
    try:
        for index, value in enumerate(payload['logits']):
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError('nonfinite or invalid logit')
            struct.pack_into('<f', data, index * 4, value)
            rounded = struct.unpack_from('<f', data, index * 4)[0]
            if rounded != value or (rounded == 0.0 and
                    math.copysign(1.0, rounded) != math.copysign(1.0, value)):
                raise ValueError('logit does not represent an exact float32 value')
    except (OverflowError, TypeError, ValueError, struct.error):
        raise ServiceSmokeError('service returned an invalid float32 logit') from None
    return response['broker_id'], response['job_id'], hashlib.sha256(data).hexdigest()


def _clean_supervisor(result, marker):
    return (type(result) is SupervisionResult and result.clean is True
            and result.timed_out is False and type(result.exit_code) is int
            and result.exit_code == 0
            and type(result.pid) is int and result.pid > 0 and not os.path.lexists(marker))


def _exercise(service, teams):
    """Trusted internal seam; caller must already have checked its owner gate."""
    handles = [None, None]
    outcome = None
    client = None
    failure = False
    interruption = None
    cleanup_failed = False
    teardown = None
    stage = 'startup'
    attempted = False
    try:
        # The caller retains the prepared controller even if startup raises.
        attempted = True
        service.start(startup_timeout=60, owner_timeout=600, lease_timeout=30, terminate_grace=5)
        stage = 'submission'
        for index, team in enumerate(teams):
            client = service.client(team)
            handles[index] = client.submit(team_id=team, request_id=_REQUEST_ID,
                                           inputs={'input_ids': [[15496, 995]]}, seed=29)
            client = None
        stage = 'result validation'
        first = _result_identity(handles[0].wait(timeout=60), teams[0])
        second = _result_identity(handles[1].wait(timeout=60), teams[1])
        if first[0] != second[0] or first[1] == second[1] or first[2] != second[2]:
            raise ServiceSmokeError('two team outcomes did not agree')
        outcome = TwoTeamSmokeResult(2, _SHAPE, first[2], True)
    except Exception:
        # Do not retain a client exception or traceback containing credentials.
        failure = True
    except BaseException as exc:
        interruption = exc
    finally:
        client = None
        for handle in handles:
            if handle is not None:
                try:
                    handle.close()
                except BaseException as exc:
                    cleanup_failed = True
                    if interruption is None and not isinstance(exc, Exception):
                        interruption = exc
        if attempted:
            try:
                teardown = service.shutdown(timeout=30)
            except BaseException as exc:
                cleanup_failed = True
                if interruption is None and not isinstance(exc, Exception):
                    interruption = exc
                # The controller is retained by the caller for reconciliation.
                try:
                    teardown = service.wait(timeout=30)
                except BaseException as wait_exc:
                    if interruption is None and not isinstance(wait_exc, Exception):
                        interruption = wait_exc
    try:
        clean = not cleanup_failed and _clean_supervisor(teardown, service.unsafe_marker_path)
    except BaseException as exc:
        clean = False
        if interruption is None and not isinstance(exc, Exception):
            interruption = exc
    if interruption is not None:
        if not clean:
            note = 'service supervisor teardown was not verified; retain the controller and marker'
            if hasattr(interruption, 'add_note'):
                interruption.add_note(note)
            else:
                interruption.__context__ = ServiceSmokeError(note)
        raise interruption
    if failure:
        detail = '; supervisor teardown was not verified' if not clean else '; clean teardown was verified'
        raise ServiceSmokeError('service smoke failed during ' + stage + detail) from None
    if not clean:
        raise ServiceSmokeError('service supervisor teardown was not verified')
    if outcome is None:
        raise ServiceSmokeError('service smoke did not complete')
    return outcome


def _claim_and_exercise(service):
    teams = service._teams
    if type(teams) is not tuple or len(teams) != 2 or teams[0] == teams[1]:
        raise ValueError('the smoke requires exactly two distinct configured teams')
    # One controller cannot safely be started or shut down by two smoke calls.
    # Retain the one-shot claim while the caller retains its prepared handle.
    with _CLAIM_LOCK:
        if service in _CLAIMED:
            raise ServiceSmokeError('prepared service was already claimed by a smoke')
        _CLAIMED.add(service)
    return _exercise(service, teams)


def run_two_team_smoke(service):
    """Exercise two fixed requests through the reviewed fair owner.

    The caller retains the prepared service for recovery and must separately
    establish approved exact-path profiles and the coordinated device window.
    The two submissions are not a fairness or throughput measurement.
    """
    if type(service) is not _PreparedService:
        raise TypeError('a prepared fixed service is required')
    marker = getattr(_owner_module, 'FAIR_SERVICE_OWNER_CONTRACT', None)
    if type(marker) is not str or marker != _REQUIRED_FAIR_CONTRACT:
        raise ServiceSmokeError('reviewed fair service owner is unavailable')
    return _claim_and_exercise(service)


def run_direct_two_team_smoke(service):
    """Exercise two fixed requests through the separately reviewed direct owner.

    The prepared controller and supervised child still verify the source,
    collector, supervisor, exact profiles and completion contracts before
    model work. This smoke does not establish fairness, throughput, or
    simultaneous device execution.
    """
    if type(service) is not _PreparedService:
        raise TypeError('a prepared fixed service is required')
    marker = getattr(_owner_module, 'DIRECT_SERVICE_OWNER_CONTRACT', None)
    if type(marker) is not str or marker != _REQUIRED_DIRECT_CONTRACT:
        raise ServiceSmokeError('reviewed direct service owner is unavailable')
    completion = getattr(_owner_module, 'OWNER_COMPLETION_CUSTODY_CONTRACT', None)
    if type(completion) is not str or completion != _REQUIRED_COMPLETION_CONTRACT:
        raise ServiceSmokeError('reviewed completion custody is unavailable')
    return _claim_and_exercise(service)
