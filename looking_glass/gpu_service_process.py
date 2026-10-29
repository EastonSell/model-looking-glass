"""Fixed supervised two-token service child; not a standalone GPU launcher.

The controller must provide an identity-aware supervisor, measured profiles,
trusted source snapshot and private startup channel. No model work is allowed
before those inputs have been independently checked.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from threading import Event

from .gpu_service_control import _encode_ready, _monitor_stop, _teams, _unique_object
from .trusted_service_launch import read_trusted_config


_HEX = re.compile(r'[0-9a-f]{64}\Z', re.ASCII)
_LEASE_HOLD = None


@dataclass(frozen=True)
class _Config:
    repo_root: Path
    cache_root: Path
    workspace_root: Path
    stop_path: Path
    teams: tuple
    profiles: tuple
    source_base_sha256: str
    asset_manifest_sha256: str


def _absolute(value, *, directory):
    if type(value) is not str or not value or '\0' in value:
        raise ValueError('invalid configured path')
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('configured paths must be absolute')
    if directory:
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise ValueError('configured directory is unavailable')
    return path


def _parse_config(data):
    try:
        value = json.loads(data, object_pairs_hook=_unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        fields = {'schema', 'repo_root', 'cache_root', 'workspace_root', 'stop_path',
                  'teams', 'profiles', 'source_base_sha256', 'asset_manifest_sha256'}
        if type(value) is not dict or set(value) != fields or value['schema'] != 'gpu-service-owner-v1':
            raise ValueError('invalid owner configuration')
        repo = _absolute(value['repo_root'], directory=True)
        cache = _absolute(value['cache_root'], directory=True)
        workspace = _absolute(value['workspace_root'], directory=True)
        stop = _absolute(value['stop_path'], directory=False)
        if type(value['teams']) is not list:
            raise ValueError('invalid owner teams')
        teams = _teams(tuple(value['teams']))
        for key in ('source_base_sha256', 'asset_manifest_sha256'):
            if type(value[key]) is not str or _HEX.fullmatch(value[key]) is None:
                raise ValueError('invalid owner digest')
        if type(value['profiles']) is not list or len(value['profiles']) != 3:
            raise ValueError('three measured profiles are required')
        profiles = []
        for item in value['profiles']:
            if (type(item) is not dict or set(item) != {'document', 'evidence', 'source_manifest', 'approved_profile_id'}
                    or type(item['approved_profile_id']) is not str
                    or _HEX.fullmatch(item['approved_profile_id']) is None):
                raise ValueError('invalid measured profile configuration')
            paths = tuple(_absolute(item[name], directory=False)
                          for name in ('document', 'evidence', 'source_manifest'))
            profiles.append((*paths, item['approved_profile_id']))
        if len({item[3] for item in profiles}) != 3:
            raise ValueError('profile pins must be distinct')
        return _Config(repo, cache, workspace, stop, teams, tuple(profiles),
                       value['source_base_sha256'], value['asset_manifest_sha256'])
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError, OSError):
        raise ValueError('invalid private service configuration') from None


def _result_code(result, *, monitor_failed, monitor_alive):
    """Classify workload outcome; teardown is established by the supervisor."""
    status = getattr(result, 'status', None)
    if status == 'quarantined':
        return 32
    if status == 'shutdown_incomplete':
        return 33
    if getattr(result, 'teardown_verified', None) is not True:
        return 32
    if monitor_alive:
        return 35
    if monitor_failed:
        return 34
    if status == 'clean':
        return 0
    if status == 'worker_failed':
        if getattr(result, 'error_type', None) in ('WorkerOutOfMemory', 'OutOfMemoryError', 'MemoryError'):
            return 36
        return 31
    return 32


def _owner(config):
    """Run only as the direct child of the identity-aware supervisor."""
    global _LEASE_HOLD
    # The launcher establishes the cache environment before any of these
    # dependencies are imported. Hugging Face Hub caches paths at import time;
    # an environment check alone cannot detect an earlier import under a
    # different HF_HOME. This is not an authentication of the interpreter.
    if any(name.partition('.')[0] in ('torch', 'transformers', 'huggingface_hub')
           for name in sys.modules):
        raise RuntimeError('model dependencies were imported before owner acquisition')
    from .two_token_service import _check_owner_contract
    _check_owner_contract(config.repo_root)
    from .device_identity import DeviceIdentity
    from .device_lease import DeviceLease
    from .supervisor import OwnerSession
    from .gpu_service import ProfileFiles, _load_profile_registry, _run_bound_owner, _TorchServiceBackend
    from .trusted_service_profile import (prepare_trusted_source, build_trusted_service_identity,
        resolve_trusted_service_profiles, verify_trusted_cpu_model, verify_trusted_device_tie,
        JOB_COMMITMENT)

    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
    _LEASE_HOLD = lease
    session.check(lease)
    identity = session.identity
    if type(identity) is not DeviceIdentity or identity.workspace_root != config.workspace_root:
        raise RuntimeError('supervised GPU identity does not match the configured workspace')
    source = prepare_trusted_source(config.repo_root, config.cache_root)
    if (hashlib.sha256(source.source_base).hexdigest() != config.source_base_sha256
            or hashlib.sha256(source.asset_manifest).hexdigest() != config.asset_manifest_sha256):
        raise RuntimeError('trusted source or checkpoint preflight differs from configuration')
    files = tuple(ProfileFiles(*record) for record in config.profiles)
    pins = tuple(record[3] for record in config.profiles)
    registry, _ = _load_profile_registry(files, identity)
    latest = [None]

    def live_keys(actual_identity):
        live = build_trusted_service_identity(source, actual_identity, session, lease)
        latest[0] = live
        return live.keys

    class CheckedBackend(_TorchServiceBackend):
        def resolve_profiles(self, candidate, keys):
            live = latest[0]
            if live is None or live.keys != keys:
                raise RuntimeError('live service identity was not reconstructed')
            checked = resolve_trusted_service_profiles(candidate, live, pins)
            # The independently checked measured estimates establish
            # eligibility. The worker reserves the larger fixed commitment.
            checked = (checked[0],
                       replace(checked[1], estimated_incremental_bytes=JOB_COMMITMENT),
                       replace(checked[2], estimated_incremental_bytes=JOB_COMMITMENT))
            ordinary = super().resolve_profiles(candidate, keys)
            if checked != ordinary:
                raise RuntimeError('service profile conversions differ')
            return ordinary

    backend = CheckedBackend(session=session, device_lease=lease, cache_dir=config.cache_root,
        team_ids=config.teams, live_key_builder=live_keys,
        verify_loaded_model=lambda model: verify_trusted_cpu_model(source, model),
        verify_device_model=verify_trusted_device_tie, shutdown_timeout=10)
    stop = Event()
    monitor, failed, _errors = _monitor_stop(config.stop_path, stop)

    def announce(server):
        session.send_startup(lease, _encode_ready(server, config.teams))

    try:
        result = _run_bound_owner(session, lease, backend=backend, registry=registry,
                                  stop_event=stop, announce=announce)
    finally:
        stop.set()
        monitor.join(timeout=2)
    return _result_code(result, monitor_failed=failed[0], monitor_alive=monitor.is_alive())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-id', required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        parts = args.config_id.split(':')
        if len(parts) != 2 or any(not part.isascii() or not part.isdigit() for part in parts):
            raise ValueError('invalid configuration file identity')
        data = read_trusted_config(args.config, expected_identity=tuple(map(int, parts)),
                                   expected_sha256=args.config_sha256)
        config = _parse_config(data)
    except Exception:
        return 20  # Invalid private configuration; no GPU operation was attempted.
    try:
        return _owner(config)
    except Exception:
        # No result means teardown was not verified. Never block on a logging
        # pipe or report a memory subtype ahead of that uncertainty.
        return 32


def _exit_owner():
    """Exit without Python finalizers, retaining owned resources to OS exit."""
    try:
        code = main()
        if type(code) is not int or not 0 <= code <= 255:
            code = 70
    except BaseException:
        code = 70
    os._exit(code)


if __name__ == '__main__':
    _exit_owner()
