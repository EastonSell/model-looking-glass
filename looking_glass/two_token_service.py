"""Prepare a caller-held controller for the fixed profiled GPU service.

This is a cooperative local-owner interface. The caller supplies a current,
trusted physical inventory and separately approved profile IDs. It is not a
general request router, a source sandbox, or a memory-bound guarantee.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from . import gpu_service as _owner_module
from . import supervisor as _supervisor_module
from .device_identity import DeviceIdentity, normalize_gpu_uuid, resolve_device_identity
from .gpu_service import ProfileFiles, _load_profile_registry
from .gpu_service_control import _prepare_owner_controller, _start_owner_controller, _teams
from .trusted_service_launch import prepare_trusted_launch
from .trusted_service_profile import prepare_trusted_source


_REQUIRED_OWNER_CONTRACT = "fixed-two-token-commitments-v1"
_REQUIRED_SPAWN_CONTRACT = "gated-rooted-child-v1"
_REQUIRED_COLLECTOR_CONTRACT = "two-token-worker-profile-v1"


def _collector_contract_matches(path):
    """Read a bounded literal from source without importing the collector."""
    with path.open("rb") as stream:
        source = stream.read(2 * 1024 * 1024 + 1)
    if len(source) > 2 * 1024 * 1024:
        return False
    module = ast.parse(source, filename=str(path))
    assignments = [node for node in module.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "TRUSTED_SERVICE_COLLECTOR_CONTRACT"
                           for target in node.targets)]
    return (len(assignments) == 1 and len(assignments[0].targets) == 1
            and isinstance(assignments[0].value, ast.Constant)
            and type(assignments[0].value.value) is str
            and assignments[0].value.value == _REQUIRED_COLLECTOR_CONTRACT)


def _check_owner_contract(repo):
    """Require the reviewed owner and collector from this source snapshot."""
    root = Path(repo)
    owner = root / "looking_glass" / "gpu_service.py"
    collector = root / "looking_glass" / "trusted_profile_collector.py"
    supervisor = root / "looking_glass" / "supervisor.py"
    try:
        valid = (getattr(_owner_module, "FIXED_SERVICE_OWNER_CONTRACT", None) == _REQUIRED_OWNER_CONTRACT
                 and Path(_owner_module.__file__).resolve(strict=True) == owner.resolve(strict=True)
                 and getattr(_supervisor_module, "SUPERVISOR_LAUNCH_CUSTODY_CONTRACT", None) == _REQUIRED_SPAWN_CONTRACT
                 and Path(_supervisor_module.__file__).resolve(strict=True) == supervisor.resolve(strict=True)
                 and _collector_contract_matches(collector))
    except (OSError, SyntaxError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError("fixed service owner, launch-custody supervisor or worker profile collector is unavailable")


class _PreparedService:
    """Keep this handle before starting; it also controls interrupted startup."""

    def __init__(self, *, controller, launch, identity, source, profiles, teams, repo):
        self._controller = controller
        self._launch = launch
        self._identity = identity
        self._source = source
        self._profiles = profiles
        self._teams = teams
        self._repo = repo

    def __repr__(self):
        return "<prepared fixed supervised service>"

    def start(self, *, startup_timeout=60, owner_timeout=600,
              lease_timeout=30, terminate_grace=5):
        """Start the fixed owner through the identity-aware supervisor.

        The caller must retain this object before calling. An exception from
        startup does not prove that no child was launched; use this handle and
        inspect the supervisor/unsafe marker to reconcile the outcome.
        """
        _check_owner_contract(self._repo)
        self._identity.validate_paths()
        if prepare_trusted_source(self._source.repo_root, self._source.cache_root) != self._source:
            raise RuntimeError("service source or checkpoint changed before launch")
        _load_profile_registry(self._profiles, self._identity)
        self._launch.verify_ready()
        _start_owner_controller(
            self._controller, self._launch.command,
            device_lease_path=self._identity.lock_path,
            marker_path=self._identity.marker_path, identity=self._identity,
            env=self._launch.environment, cwd=self._repo, teams=self._teams,
            startup_timeout=startup_timeout, owner_timeout=owner_timeout,
            lease_timeout=lease_timeout, terminate_grace=terminate_grace)

    def client(self, team):
        """Issue a team-scoped client while the supervised owner is alive."""
        return self._controller.client(team)

    @property
    def unsafe_marker_path(self):
        """Return the local coordination marker; its path contains the GPU UUID."""
        return self._identity.marker_path

    def shutdown(self, *, timeout=30):
        """Cancel pre-start or signal the owner and return its teardown result.

        ``None`` means the supervisor was not yet claimed and cannot start.
        Otherwise the supervisor result, or an exception, reports its outcome.
        """
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("invalid shutdown timeout")
        # Claiming the underlying controller uses this same lock. Cancellation
        # during source/profile preflight must close the gate before a later
        # start can claim it, rather than merely returning while start proceeds.
        with self._controller._lock:
            if not self._controller._claimed:
                self._controller._closed = True
                self._controller._error_type = "StartupCancelled"
                self._controller._done.set()
                return None
        return self._controller.shutdown(timeout=timeout)

    def wait(self, *, timeout=30):
        """Recheck the same supervisor after a timeout or failed startup."""
        return self._controller.wait(timeout=timeout)


def prepare_two_token_service(*, repo_root, cache_root, workspace_root,
                              private_parent, site_packages, environment_root,
                              hf_home, gpu_uuid, physical_uuids, profiles, teams,
                              trusted_path):
    """Prepare the fixed owner without initializing CUDA or starting a process.

    ``physical_uuids`` must be a current trusted physical inventory; exactly
    the selected full UUID will be visible to the child. ``profiles`` is an
    ordered tuple of separately approved model-load, cold and warm
    :class:`ProfileFiles`. The actual runtime identity and case/seed evidence
    are reconstructed and checked again under the child's device lease.
    """
    teams = _teams(teams)
    if (type(profiles) is not tuple or len(profiles) != 3
            or any(type(item) is not ProfileFiles for item in profiles)):
        raise ValueError("three ordered approved profile files are required")
    if type(physical_uuids) is not tuple:
        raise ValueError("a trusted physical GPU inventory tuple is required")
    uuid = normalize_gpu_uuid(gpu_uuid)
    environment = {"CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    identity = resolve_device_identity(workspace_root, uuid, physical_uuids=physical_uuids,
                                       environ=environment)
    if type(identity) is not DeviceIdentity:
        raise RuntimeError("canonical device identity is required")
    source = prepare_trusted_source(repo_root, cache_root)
    home = Path(hf_home)
    if not home.is_absolute():
        raise ValueError("HF_HOME must be an absolute directory")
    home = home.resolve(strict=True)
    cache = Path(source.cache_root)
    if not home.is_dir() or cache == home or not cache.is_relative_to(home):
        raise ValueError("checkpoint cache must be inside HF_HOME")
    _load_profile_registry(profiles, identity)

    parent = Path(private_parent)
    if not parent.is_absolute():
        raise ValueError("private launch parent must be absolute")
    parent = parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("private launch parent must be a directory")
    launch_dir = Path(tempfile.mkdtemp(prefix="looking-glass-service-", dir=parent))
    stop = launch_dir / "stop"
    config = {
        "schema": "gpu-service-owner-v1", "repo_root": source.repo_root,
        "cache_root": source.cache_root, "workspace_root": str(identity.workspace_root),
        "stop_path": str(stop), "teams": list(teams),
        "profiles": [{"document": str(item.document), "evidence": str(item.evidence),
                      "source_manifest": str(item.source_manifest),
                      "approved_profile_id": item.approved_profile_id} for item in profiles],
        "source_base_sha256": hashlib.sha256(source.source_base).hexdigest(),
        "asset_manifest_sha256": hashlib.sha256(source.asset_manifest).hexdigest(),
    }
    config_path = launch_dir / "config.json"
    payload = json.dumps(config, separators=(",", ":"), allow_nan=False).encode("utf-8")
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("private service configuration was not fully written")
        stream.flush()
        os.fsync(stream.fileno())
    launch = prepare_trusted_launch(
        repo_root=source.repo_root, site_packages=site_packages, environment_root=environment_root,
        private_parent=launch_dir, hf_home=home, gpu_uuid=uuid, module="looking_glass.gpu_service_process",
        config_path=config_path, trusted_path=trusted_path)
    controller = _prepare_owner_controller(stop)
    return _PreparedService(controller=controller, launch=launch, identity=identity,
                            source=source, profiles=profiles, teams=teams, repo=source.repo_root)
