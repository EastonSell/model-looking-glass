"""Canonical cooperative GPU paths without initializing CUDA.

The caller supplies a trusted snapshot of physical NVIDIA GPU UUIDs and must
recheck the actual CUDA binding after acquiring the returned device lease.
This module does not query hardware or create a CUDA context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat

from .device_lease import DeviceLease, LeaseError


class DeviceIdentityError(ValueError):
    """The physical device, visibility mapping, or paths are ambiguous."""


_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE | re.ASCII,
)
_MAX_GPUS = 128
_MAX_INVENTORY_TEXT = 16_384
_STATE_DIR = ".looking-glass-devices"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def normalize_gpu_uuid(value: str) -> str:
    """Normalize a complete physical GPU UUID; reject MIG and abbreviations."""
    if not isinstance(value, str) or _UUID.fullmatch(value) is None:
        raise DeviceIdentityError("a complete physical GPU UUID is required")
    return "GPU-" + value[4:].lower()


def _inventory(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DeviceIdentityError("physical GPU inventory must be a sequence of UUIDs")
    if not 1 <= len(values) <= _MAX_GPUS:
        raise DeviceIdentityError("physical GPU inventory is empty or too large")
    normalized = tuple(normalize_gpu_uuid(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise DeviceIdentityError("physical GPU inventory contains duplicate UUIDs")
    return normalized


def parse_nvidia_smi_inventory(text: str) -> tuple[str, ...]:
    """Parse bounded `nvidia-smi --query-gpu=uuid --format=csv,noheader` text.

    Parsing does not establish that the source is trustworthy or current.
    """
    if not isinstance(text, str) or len(text) > _MAX_INVENTORY_TEXT:
        raise DeviceIdentityError("physical GPU inventory output is invalid or too large")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _inventory(lines)


def _environment(environ: Mapping[str, str] | None) -> tuple[str | None, str | None]:
    env = os.environ if environ is None else environ
    if not isinstance(env, Mapping):
        raise DeviceIdentityError("environment must be a mapping")
    for key in env:
        if (isinstance(key, str) and key.upper() in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
                and key != key.upper()):
            raise DeviceIdentityError("CUDA environment keys must use canonical uppercase spelling")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    order = env.get("CUDA_DEVICE_ORDER")
    if visible is not None and not isinstance(visible, str):
        raise DeviceIdentityError("CUDA_VISIBLE_DEVICES must be a string")
    if order is not None and order not in ("PCI_BUS_ID", "FASTEST_FIRST"):
        raise DeviceIdentityError("CUDA_DEVICE_ORDER is unsupported")
    return visible, order


def _check_visibility(
    target: str, inventory: tuple[str, ...], index: int, visible: str | None
) -> None:
    if type(index) is not int or index != 0:
        raise DeviceIdentityError("only a single CUDA-visible device at index 0 is supported")
    if target not in inventory:
        raise DeviceIdentityError("the target UUID is absent from the physical inventory")
    if visible is None:
        if len(inventory) != 1 or inventory[0] != target:
            raise DeviceIdentityError("unset visibility requires exactly one physical GPU")
    elif normalize_gpu_uuid(visible) != target:
        raise DeviceIdentityError("CUDA visibility does not select the requested physical GPU")


def _is_reparse(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


def _file_id(info: os.stat_result) -> tuple[int, int]:
    if not info.st_ino:
        raise DeviceIdentityError("filesystem does not provide an unambiguous file identity")
    return info.st_dev, info.st_ino


def _plain_directory(path: Path) -> tuple[int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise DeviceIdentityError("workspace state directory is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
        raise DeviceIdentityError("workspace state directory must not be a link or reparse point")
    return _file_id(info)


def _plain_file_or_missing(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DeviceIdentityError("device coordination path cannot be inspected") from exc
    if not stat.S_ISREG(info.st_mode) or _is_reparse(info) or info.st_nlink != 1:
        raise DeviceIdentityError("device coordination paths must not be links or reparse points")
    _file_id(info)


def _canonical_workspace(path: str | os.PathLike[str]) -> tuple[Path, tuple[int, int]]:
    try:
        requested = Path(path)
    except (TypeError, ValueError) as exc:
        raise DeviceIdentityError("workspace root must be an absolute directory") from exc
    if not requested.is_absolute():
        raise DeviceIdentityError("workspace root must be an absolute directory")
    try:
        root = requested.resolve(strict=True)
        info = root.stat()
    except (OSError, RuntimeError) as exc:
        raise DeviceIdentityError("workspace root cannot be resolved") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise DeviceIdentityError("workspace root must be a directory")
    if os.name == "nt":
        # A trusted local fixed drive avoids silently relying on UNC or mapped
        # network lock semantics.  Junctions in the root have already resolved.
        import ctypes

        get_drive_type = ctypes.windll.kernel32.GetDriveTypeW
        get_drive_type.argtypes = [ctypes.c_wchar_p]
        get_drive_type.restype = ctypes.c_uint
        if get_drive_type(root.anchor) != 3:  # DRIVE_FIXED
            raise DeviceIdentityError("workspace root must be on a local fixed drive")
    return root, _file_id(info)


def _derived_paths(root: Path, uuid: str) -> tuple[Path, Path, Path, Path]:
    state = root / _STATE_DIR
    lock = state / (uuid + ".lock")
    marker = state / (uuid + ".unsafe.json")
    return lock, marker, Path(str(marker) + ".controller.lock"), Path(str(marker) + ".attestation")


@dataclass(frozen=True)
class DeviceIdentity:
    """Immutable path bundle derived from one trusted inventory snapshot."""

    gpu_uuid: str
    workspace_root: Path
    lock_path: Path
    marker_path: Path
    controller_path: Path
    attestation_path: Path
    cuda_device_index: int
    _physical_uuids: tuple[str, ...]
    _environment_snapshot: tuple[str | None, str | None]
    _root_file_id: tuple[int, int]
    _state_file_id: tuple[int, int]

    def validate_paths(self) -> None:
        """Recheck paths, particularly while holding the device lease."""
        if normalize_gpu_uuid(self.gpu_uuid) != self.gpu_uuid:
            raise DeviceIdentityError("device UUID is not canonical")
        inventory = _inventory(self._physical_uuids)
        if (not isinstance(self._environment_snapshot, tuple)
                or len(self._environment_snapshot) != 2):
            raise DeviceIdentityError("CUDA visibility snapshot is invalid")
        visible, order = self._environment_snapshot
        if _environment({key: value for key, value in (
                ("CUDA_VISIBLE_DEVICES", visible), ("CUDA_DEVICE_ORDER", order)
        ) if value is not None}) != self._environment_snapshot:
            raise DeviceIdentityError("CUDA visibility snapshot is invalid")
        _check_visibility(self.gpu_uuid, inventory, self.cuda_device_index, visible)
        root, root_id = _canonical_workspace(self.workspace_root)
        if root != self.workspace_root or root_id != self._root_file_id:
            raise DeviceIdentityError("configured workspace root changed")
        expected = _derived_paths(root, self.gpu_uuid)
        if (self.lock_path, self.marker_path, self.controller_path, self.attestation_path) != expected:
            raise DeviceIdentityError("device coordination paths do not match the UUID and workspace")
        if _plain_directory(root / _STATE_DIR) != self._state_file_id:
            raise DeviceIdentityError("workspace state directory changed")
        for path in (self.lock_path, self.marker_path, self.controller_path, self.attestation_path):
            _plain_file_or_missing(path)

    def verify_runtime_binding(
        self,
        *,
        lease: DeviceLease,
        runtime_uuid: str,
        device_count: int,
        cuda_device_index: int = 0,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Validate a caller-supplied, authoritative post-lease CUDA UUID.

        The caller must obtain ``runtime_uuid`` from the CUDA runtime after
        acquiring the matching device lease; a device name or a copied
        preflight UUID is not proof.
        """
        self.validate_paths()
        if type(lease) is not DeviceLease:
            raise DeviceIdentityError("a live matching device lease is required")
        if os.path.normcase(str(lease.path)) != os.path.normcase(str(self.lock_path)):
            raise DeviceIdentityError("device lease path does not match this physical UUID")
        try:
            DeviceLease.check(lease)
        except LeaseError as exc:
            raise DeviceIdentityError("device lease is not safely held") from exc
        snapshot = _environment(environ)
        if snapshot != self._environment_snapshot:
            raise DeviceIdentityError("CUDA visibility environment changed after preflight")
        _check_visibility(self.gpu_uuid, self._physical_uuids, cuda_device_index, snapshot[0])
        if type(device_count) is not int or device_count != 1:
            raise DeviceIdentityError("the CUDA runtime must expose exactly one device")
        if normalize_gpu_uuid(runtime_uuid) != self.gpu_uuid:
            raise DeviceIdentityError("CUDA runtime UUID does not match the leased physical GPU")


def resolve_device_identity(
    workspace_root: str | os.PathLike[str],
    physical_uuid: str,
    *,
    physical_uuids: Sequence[str],
    cuda_device_index: int = 0,
    environ: Mapping[str, str] | None = None,
) -> DeviceIdentity:
    """Resolve the shared lease/marker paths without querying CUDA.

    All participants must supply the same trusted local workspace root and a
    current, trusted physical inventory. The process may use only one visible
    full UUID or an unset visibility mask on a single-physical-GPU host.
    """
    target = normalize_gpu_uuid(physical_uuid)
    inventory = _inventory(physical_uuids)
    snapshot = _environment(environ)
    _check_visibility(target, inventory, cuda_device_index, snapshot[0])
    root, root_id = _canonical_workspace(workspace_root)
    state = root / _STATE_DIR
    try:
        state.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise DeviceIdentityError("cannot create the shared device state directory") from exc
    state_id = _plain_directory(state)
    lock, marker, controller, attestation = _derived_paths(root, target)
    result = DeviceIdentity(
        gpu_uuid=target,
        workspace_root=root,
        lock_path=lock,
        marker_path=marker,
        controller_path=controller,
        attestation_path=attestation,
        cuda_device_index=cuda_device_index,
        _physical_uuids=inventory,
        _environment_snapshot=snapshot,
        _root_file_id=root_id,
        _state_file_id=state_id,
    )
    result.validate_paths()
    return result
