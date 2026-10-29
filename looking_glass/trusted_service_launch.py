"""Fixed direct-child launch settings for the supervised two-token pilot.

This prepares a command, not a GPU service or permission to use a device.
The controller must supply a verified physical identity, measured profiles,
the common supervisor/lease and a trusted immutable checkout/runtime.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import MappingProxyType

from .device_identity import normalize_gpu_uuid


_MODULES = frozenset({"looking_glass.gpu_service_process",
                      "looking_glass.trusted_profile_collector"})
_BOOTSTRAP = ("import sys,runpy;"
              "repo=sys.argv.pop(1);site=sys.argv.pop(1);module=sys.argv.pop(1);"
              "sys.path.extend((repo,site));"
              "runpy.run_module(module,run_name='__main__',alter_sys=True)")
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _directory(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("launch directories must be absolute")
    path = path.resolve(strict=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & _REPARSE:
        raise ValueError("launch directory is not ordinary")
    return path


def _capture_config(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("private configuration path must be absolute")
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & _REPARSE
            or info.st_nlink != 1 or not info.st_ino or not 0 < info.st_size <= 65536):
        raise ValueError("private configuration must be a bounded ordinary file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if ((opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
                or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1):
            raise ValueError("private configuration changed during opening")
        data = stream.read(65537)
        after = os.fstat(stream.fileno())
    if (len(data) != info.st_size or len(data) > 65536
            or (after.st_dev, after.st_ino, after.st_size) != (info.st_dev, info.st_ino, info.st_size)):
        raise ValueError("private configuration changed during reading")
    return path.resolve(strict=True), (info.st_dev, info.st_ino), data


def read_trusted_config(path, *, expected_identity, expected_sha256):
    """Read exact child configuration bytes and bind them to the parent pin."""
    if (type(expected_identity) is not tuple or len(expected_identity) != 2
            or any(type(value) is not int or value < 0 for value in expected_identity)
            or type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)):
        raise ValueError("invalid private configuration identity")
    _, identity, data = _capture_config(path)
    if identity != expected_identity or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("private configuration differs from the launch pin")
    return data


@dataclass(frozen=True)
class TrustedLaunch:
    command: tuple
    environment: object
    pycache_prefix: Path
    _prefix_identity: tuple
    config_path: Path
    _config_identity: tuple
    _config_sha256: str

    def verify_ready(self):
        """Reject a replaced config or a populated/replaced bytecode prefix."""
        info = self.pycache_prefix.lstat()
        if (not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & _REPARSE
                or (info.st_dev, info.st_ino) != self._prefix_identity
                or any(self.pycache_prefix.iterdir())):
            raise ValueError("private bytecode prefix changed or is not empty")
        path, identity, data = _capture_config(self.config_path)
        if (path != self.config_path or identity != self._config_identity
                or hashlib.sha256(data).hexdigest() != self._config_sha256):
            raise ValueError("private configuration changed before launch")


def prepare_trusted_launch(*, repo_root, site_packages, environment_root, private_parent,
                           hf_home, gpu_uuid, module, config_path, trusted_path, inherited_environment=None):
    """Prepare one fixed source-first launch without inheriting runtime knobs.

    The configuration contains paths and fixed study settings, never bearer
    tokens. The returned environment must replace the inherited environment
    in ``run_supervised``. Revalidate the frozen source/profile separately.
    """
    if type(module) is not str or module not in _MODULES:
        raise ValueError("unsupported trusted owner module")
    if type(gpu_uuid) is not str:
        raise ValueError("physical GPU UUID must be a string")
    gpu_uuid = normalize_gpu_uuid(gpu_uuid)
    repo, site, env_root, private, home = map(_directory,
        (repo_root, site_packages, environment_root, private_parent, hf_home))
    if not site.is_relative_to(env_root):
        raise ValueError("site-packages is outside the selected environment")
    if type(trusted_path) is not str or not trusted_path:
        raise ValueError("controlled executable PATH is required")
    entries = trusted_path.split(os.pathsep)
    if any(not item for item in entries):
        raise ValueError("controlled executable PATH contains an empty entry")
    checked_path = os.pathsep.join(str(_directory(item)) for item in entries)
    config, config_id, config_bytes = _capture_config(config_path)
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    if not config.is_relative_to(private):
        raise ValueError("configuration must be inside the private launch directory")
    base = Path(sys._base_executable).resolve(strict=True)
    if not base.is_file():
        raise ValueError("base Python executable is unavailable")
    source = os.environ if inherited_environment is None else inherited_environment
    if type(source) is not dict and source is not os.environ:
        raise TypeError("inherited environment must be a dictionary")
    keep = {"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP"}
    environment = {key: value for key, value in source.items()
                   if type(key) is str and key.upper() in keep and type(value) is str}
    program_files = [(key, value) for key, value in source.items()
                     if type(key) is str and key.upper() == 'PROGRAMFILES']
    if len(program_files) > 1:
        raise ValueError('ambiguous PROGRAMFILES environment')
    if program_files:
        value = program_files[0][1]
        try:
            if type(value) is not str or str(_directory(value)) != value:
                raise ValueError('PROGRAMFILES must be a canonical directory')
        except (OSError, RuntimeError) as exc:
            raise ValueError('PROGRAMFILES must be a canonical directory') from exc
        environment['PROGRAMFILES'] = value
    prefix = Path(tempfile.mkdtemp(prefix="looking-glass-service-pyc-", dir=private))
    info = prefix.lstat()
    environment.update({
        "PATH": checked_path, "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HOME": str(home),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "17",
        "PYTHONPYCACHEPREFIX": str(prefix),
    })
    command = (str(base), "-B", "-S", "-P", "-X", "pycache_prefix=" + str(prefix),
               "-c", _BOOTSTRAP, str(repo), str(site), module, "--config", str(config),
               "--config-id", f"{config_id[0]}:{config_id[1]}", "--config-sha256", config_digest)
    return TrustedLaunch(command, MappingProxyType(environment), prefix,
                         (info.st_dev, info.st_ino), config, config_id, config_digest)
