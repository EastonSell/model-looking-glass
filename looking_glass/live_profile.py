"""Reconstruct bootstrap-v1 workload identity from a trusted supervised owner.

This module does not authenticate arbitrary Python callers or prove an empirical
memory estimate is an upper bound. The first CUDA access requires a live owner.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import importlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys

from .device_identity import DeviceIdentity, DeviceIdentityError
from .device_lease import DeviceLease
from .resource_profile import (ProfileRegistry, WorkloadKey, canonical_fingerprint,
                               model_runtime_fingerprint)
from .supervisor import OwnerSession


_GIB = 1024 ** 3
_PINNED_ASSETS = {
    'config.json': (762, '4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a'),
    'generation_config.json': (124, 'b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805'),
    'tokenizer.json': (1355256, '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'),
    'tokenizer_config.json': (26, '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f'),
    'vocab.json': (1042301, '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'),
    'merges.txt': (456318, '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),
    'model.safetensors': (352824413, 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'),
}
_UUID_SUFFIX = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z', re.ASCII)
_REQUIRED_ENV = {
    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
    'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONHASHSEED': '17',
    'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
}
_ABSENT_ENV = ('PYTORCH_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING', 'CUDA_LAUNCH_BLOCKING', 'PYTHONPATH', 'PYTHONHOME')
_CONTROL_PREFIXES = ('CUDA_', 'PYTORCH_', 'TORCH_', 'CUBLAS_', 'HF_',
                     'TRANSFORMERS_', 'OMP_', 'MKL_', 'OPENBLAS_', 'PYTHON', 'LD_', 'DYLD_')


def decode_cuda_uuid(value, expected_type):
    """Decode PyTorch 2.8's actual CUDA UUID without accepting abbreviated IDs."""
    if not isinstance(expected_type, type) or type(value) is not expected_type:
        raise ValueError('unsupported CUDA runtime UUID type')
    text = str(value)
    if not _UUID_SUFFIX.fullmatch(text):
        raise ValueError('unsupported CUDA runtime UUID format')
    return 'GPU-' + text


def check_allocator_readback(total_before, total_after, effective_fraction):
    """Validate PyTorch 2.8 native allocator's integer-byte-truncated limit."""
    if (type(total_before) is not int or type(total_after) is not int
            or not 7 * _GIB <= total_before <= 9 * _GIB or total_before != total_after):
        raise ValueError('CUDA total changed or is outside the approved range')
    cap = int(0.20 * total_before)
    if type(effective_fraction) is not float or effective_fraction != cap / total_before:
        raise ValueError('effective allocator limit differs from the requested fraction')
    return cap


def checked_environment(environ, device_uuid):
    """Validate and return the complete approved nonsecret control settings."""
    if type(device_uuid) is not str or not re.fullmatch(r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', device_uuid):
        raise ValueError('physical UUID must be complete and canonical')
    required = {'CUDA_VISIBLE_DEVICES': device_uuid, **_REQUIRED_ENV}
    allowed = set(required) | set(_ABSENT_ENV) | {'PYTHONPYCACHEPREFIX'}
    for name, value in environ.items():
        if not isinstance(name, str):
            raise ValueError('environment keys must be strings')
        upper = name.upper()
        if upper.startswith(_CONTROL_PREFIXES):
            if name != upper or upper not in allowed:
                raise ValueError('unreviewed control environment variable')
            if type(value) is not str:
                raise ValueError('environment values must be strings')
    if any(environ.get(name) != value for name, value in required.items()):
        raise ValueError('required environment control is absent or changed')
    if any(name in environ for name in _ABSENT_ENV):
        raise ValueError('an incompatible environment control is present')
    cache_prefix = environ.get('PYTHONPYCACHEPREFIX')
    if type(cache_prefix) is not str or not Path(cache_prefix).is_absolute():
        raise ValueError('private Python cache prefix is missing')
    path = environ.get('PATH')
    if type(path) is not str or not path:
        raise ValueError('controlled executable and native-library PATH is missing')
    try:
        entries = path.split(os.pathsep)
        resolved_entries = []
        for entry in entries:
            candidate = Path(entry)
            if not entry or not candidate.is_absolute():
                raise ValueError('PATH entries must be absolute directories')
            resolved = candidate.resolve(strict=True)
            if not resolved.is_dir() or candidate != resolved:
                raise ValueError('PATH entries must be existing canonical directories')
            resolved_entries.append(resolved)
        if len(set(resolved_entries)) != len(resolved_entries):
            raise ValueError('PATH entries must be unique')
    except (OSError, RuntimeError) as exc:
        raise ValueError('controlled PATH cannot be validated') from exc
    # The unique cache path is private and intentionally excluded from the
    # reusable workload fingerprint; its actual state is checked separately.
    return {**required, 'PATH': path, **{name: None for name in _ABSENT_ENV}}


def _manifest_file_record(record, *, path_required):
    fields = {'size', 'sha256'} | ({'path'} if path_required else set())
    if type(record) is not dict or set(record) != fields:
        raise ValueError('invalid source manifest file record')
    if type(record['size']) is not int or record['size'] < 0:
        raise ValueError('source manifest file size must be a nonnegative integer')
    if type(record['sha256']) is not str or not re.fullmatch(r'[0-9a-f]{64}', record['sha256']):
        raise ValueError('source manifest file hash is invalid')
    if path_required:
        name = record['path']
        if (type(name) is not str or not name or name == '.' or '\x00' in name or '\\' in name or ':' in name
                or PurePosixPath(name).is_absolute() or str(PurePosixPath(name)) != name
                or any(part in ('.', '..') for part in PurePosixPath(name).parts)):
            raise ValueError('source manifest path is invalid')


def validate_source_manifest(manifest):
    """Validate bootstrap-v1 structure; this does not verify file contents."""
    if (type(manifest) is not dict or set(manifest) !=
            {'schema', 'git_commit', 'git_tree', 'repository_files', 'python_executable', 'distributions'}
            or manifest['schema'] != 'resource-bootstrap-source-v1'):
        raise ValueError('invalid bootstrap source manifest')
    for name in ('git_commit', 'git_tree'):
        if type(manifest[name]) is not str or not re.fullmatch(r'[0-9a-f]{40}', manifest[name]):
            raise ValueError('invalid source revision')
    _manifest_file_record(manifest['python_executable'], path_required=False)
    records = manifest['repository_files']
    if type(records) is not list or not records:
        raise ValueError('source manifest has no repository files')
    for record in records:
        _manifest_file_record(record, path_required=True)
        if not (record['path'].startswith('looking_glass/') or
                record['path'] == 'docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md'):
            raise ValueError('unexpected repository source file')
    names = [record['path'] for record in records]
    required = {'looking_glass/__init__.py', 'looking_glass/gpu_bootstrap.py', 'looking_glass/gpu_service.py',
                'looking_glass/live_profile.py', 'docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md'}
    if names != sorted(set(names)) or not required.issubset(names):
        raise ValueError('repository source is incomplete or unordered')
    packages = manifest['distributions']
    required = {'huggingface-hub', 'numpy', 'safetensors', 'tokenizers', 'torch', 'transformers'}
    if type(packages) is not list or any(type(item) is not dict for item in packages):
        raise ValueError('invalid distribution inventory')
    package_names = [item.get('name') for item in packages]
    if (any(type(name) is not str or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name) for name in package_names)
            or package_names != sorted(set(package_names)) or not required.issubset(package_names)):
        raise ValueError('distribution inventory is incomplete or unordered')
    all_paths = set()
    for item in packages:
        if set(item) != {'name', 'version', 'files'} or type(item['version']) is not str or not item['version']:
            raise ValueError('invalid distribution identity')
        if type(item['files']) is not list or not item['files']:
            raise ValueError('distribution file inventory is empty')
        for record in item['files']:
            _manifest_file_record(record, path_required=True)
        paths = [record['path'] for record in item['files']]
        if paths != sorted(set(paths)) or all_paths.intersection(paths):
            raise ValueError('distribution files are duplicated or unordered')
        all_paths.update(paths)
    return True


def check_import_cache_policy(expected_prefix, runtime_prefix, dont_write_bytecode):
    """Check the child's import cache; the parent must establish it before exec."""
    if type(dont_write_bytecode) is not bool or not dont_write_bytecode:
        raise ValueError('bytecode writes must be disabled before interpreter startup')
    try:
        expected = Path(expected_prefix)
        runtime = Path(runtime_prefix)
        if not expected.is_absolute() or not runtime.is_absolute():
            raise ValueError('import cache prefix must be absolute')
        info = expected.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
            raise ValueError('import cache must be a private ordinary directory')
        if expected.resolve(strict=True) != runtime.resolve(strict=True):
            raise ValueError('active Python cache prefix differs from the parent policy')
        if any(expected.iterdir()):
            raise ValueError('Python cache prefix must remain empty')
    except (OSError, TypeError) as exc:
        raise ValueError('Python import cache policy is unavailable') from exc
    return True


def check_profile_source_identity(key_source_sha256, provenance_source_sha256, source_manifest):
    validate_source_manifest(source_manifest)
    digest = canonical_fingerprint(source_manifest)
    if key_source_sha256 != digest or provenance_source_sha256 != digest:
        raise ValueError('bootstrap source identity and provenance differ')


def resolve_bootstrap_profile(registry, key, source_manifest, expected_profile_id):
    """Resolve against a separately approved pin and bootstrap source manifest."""
    if type(registry) is not ProfileRegistry or type(key) is not WorkloadKey:
        raise TypeError('a trusted profile registry and exact live key are required')
    if type(expected_profile_id) is not str or not re.fullmatch(r'[0-9a-f]{64}', expected_profile_id):
        raise ValueError('a separate approved full profile ID is required')
    profile = registry.resolve(key)
    if profile.profile_id != expected_profile_id:
        raise ValueError('profile does not match the independently approved ID')
    check_profile_source_identity(key.source_sha256, profile.provenance.source_manifest_sha256,
                                  source_manifest)
    return profile


def _git(root, *args):
    try:
        result = subprocess.run(['git', '-C', str(root), *args], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('cannot verify repository state') from exc
    return result.stdout


def _filtered_git_hashes(root, names):
    if any('\n' in name or '\r' in name for name in names):
        raise ValueError('source filenames cannot contain line breaks')
    try:
        result = subprocess.run(['git', '-C', str(root), 'hash-object', '--stdin-paths'],
                                input=('\n'.join(names) + '\n').encode('utf-8'),
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        hashes = result.stdout.decode('ascii').splitlines()
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise ValueError('cannot compute filtered repository hashes') from exc
    if len(hashes) != len(names) or any(not re.fullmatch(r'[0-9a-f]{40}', value) for value in hashes):
        raise ValueError('invalid filtered repository hashes')
    return dict(zip(names, hashes))


def _capture_source_bytes(path):
    """Capture one bounded immutable snapshot for both source and Git hashes."""
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
                or info.st_size > 64 * 1024 * 1024):
            raise ValueError('source must be a bounded regular file')
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            data = stream.read(64 * 1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
        if (len(data) != info.st_size or len(data) > 64 * 1024 * 1024
                or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) !=
                   (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                or (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise ValueError('source file changed while reading')
        return data
    except OSError as exc:
        raise ValueError('cannot read source file') from exc


def _git_hash_captured_source(root, name, data):
    try:
        result = subprocess.run(['git', '-C', str(root), 'hash-object', '--path=' + name, '--stdin'],
                                input=data, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        digest = result.stdout.decode('ascii').strip()
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise ValueError('cannot hash captured source as a Git blob') from exc
    if not re.fullmatch(r'[0-9a-f]{40}', digest):
        raise ValueError('invalid captured source Git hash')
    return digest


def _head_blobs(root, commit):
    raw = _git(root, 'ls-tree', '-r', '-z', '--full-tree', commit, '--',
               'looking_glass/', 'docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md')
    result = {}
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        try:
            metadata, path = entry.split(b'\t', 1)
            mode, kind, digest = metadata.decode('ascii').split(' ')
            name = path.decode('utf-8')
        except (UnicodeError, ValueError) as exc:
            raise ValueError('invalid committed tree entry') from exc
        if mode not in ('100644', '100755') or kind != 'blob' or not re.fullmatch(r'[0-9a-f]{40}', digest) or name in result:
            raise ValueError('source tree contains a nonregular or duplicate entry')
        result[name] = digest
    return result


def _plain_file_hash(path):
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
            raise ValueError('manifest inputs must be regular files')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(stream.fileno())
        if ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) !=
                (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                or (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise ValueError('manifest file changed while hashing')
        return {'size': opened.st_size, 'sha256': digest.hexdigest()}
    except OSError as exc:
        raise ValueError('cannot hash manifest file') from exc


def collect_repository_source(repo_root):
    """Hash a clean repository's tracked implementation and protocol bytes."""
    root = Path(repo_root).resolve(strict=True)
    inside = _git(root, 'rev-parse', '--is-inside-work-tree').strip()
    prefix = _git(root, 'rev-parse', '--show-prefix').strip()
    branch = _git(root, 'rev-parse', '--abbrev-ref', 'HEAD').strip()
    if inside != b'true' or prefix or branch != b'HEAD' or _git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=all'):
        raise ValueError('source checkout must be clean and match its Git root')
    flags = _git(root, 'ls-files', '-v', '-z').split(b'\0')
    if any(entry and not entry.startswith(b'H ') for entry in flags):
        raise ValueError('assume-unchanged, sparse, or unmerged files are not allowed')
    commit = _git(root, 'rev-parse', 'HEAD').decode('ascii').strip()
    tree = _git(root, 'show', '-s', '--format=%T', 'HEAD').decode('ascii').strip()
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or not re.fullmatch(r'[0-9a-f]{40}', tree):
        raise ValueError('repository commit and tree must be full SHA-1 identities')
    raw = _git(root, 'ls-files', '-z', '--', 'looking_glass/', 'docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md')
    try:
        names = [item.decode('utf-8') for item in raw.split(b'\0') if item]
    except UnicodeError as exc:
        raise ValueError('source paths must be UTF-8') from exc
    if len(names) != len(set(names)):
        raise ValueError('duplicate tracked source paths')
    required = {'looking_glass/__init__.py', 'looking_glass/gpu_bootstrap.py', 'looking_glass/gpu_service.py',
                'looking_glass/live_profile.py', 'docs/RESOURCE_PROFILE_BOOTSTRAP_V1.md'}
    if not required.issubset(names):
        raise ValueError('tracked source or protocol is missing')
    tracked = set(names)
    committed_blobs = _head_blobs(root, commit)
    if set(committed_blobs) != tracked or _filtered_git_hashes(root, sorted(names)) != committed_blobs:
        raise ValueError('working source bytes do not match the committed tree')
    package = root / 'looking_glass'
    for base, dirs, files in os.walk(package, followlinks=False):
        dirs[:] = [name for name in dirs if name != '__pycache__']
        for name in dirs:
            directory = Path(base) / name
            info = directory.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
                raise ValueError('package source directories must not be links')
        for name in files:
            relative = (Path(base) / name).relative_to(root).as_posix()
            if relative not in tracked:
                raise ValueError('untracked or ignored package source is present')
    result = []
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid tracked source path')
        path = root / relative
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('source path escapes the repository')
        data = _capture_source_bytes(path)
        if _git_hash_captured_source(root, name, data) != committed_blobs[name]:
            raise ValueError('captured source bytes differ from the committed Git blob')
        result.append({'path': relative.as_posix(), 'size': len(data),
                       'sha256': hashlib.sha256(data).hexdigest()})
    # Verify revision and index flags again as well as worktree dirtiness: a
    # concurrent commit can leave status clean while changing the bytes read.
    if (_git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=all')
            or _git(root, 'rev-parse', '--abbrev-ref', 'HEAD').strip() != b'HEAD'
            or _git(root, 'rev-parse', 'HEAD').decode('ascii').strip() != commit
            or _git(root, 'show', '-s', '--format=%T', 'HEAD').decode('ascii').strip() != tree
            or _git(root, 'ls-files', '-v', '-z').split(b'\0') != flags
            or _filtered_git_hashes(root, sorted(names)) != committed_blobs):
        raise ValueError('source checkout changed while hashing')
    return {'git_commit': commit, 'git_tree': tree, 'repository_files': result}


def _validated_site(site_packages, root):
    """Check the supplied site path before normalizing away any component."""
    site = Path(site_packages)
    if not site.is_absolute() or '..' in site.parts:
        raise ValueError('site-packages path must be absolute and canonical')
    try:
        relative = site.relative_to(root)
    except ValueError as exc:
        raise ValueError('site-packages is outside the selected environment') from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        info = cursor.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
            raise ValueError('site-packages must not use intermediate links')
    if site.resolve(strict=True) != site:
        raise ValueError('site-packages path must resolve without aliases')
    return site


def collect_distribution_files(distribution, site_packages, environment_root):
    """Hash every recorded installed file under the explicitly selected environment."""
    try:
        root = Path(environment_root).resolve(strict=True)
        site = _validated_site(site_packages, root)
        location = Path(distribution.locate_file(''))
    except (OSError, TypeError) as exc:
        raise ValueError('package environment is unavailable') from exc
    if location != site or not site.is_relative_to(root):
        raise ValueError('distribution is outside the selected environment')
    # Walk each original RECORD component rather than resolving away an
    # intermediate alias (including alias/../ paths).
    if distribution.files is None or not distribution.files:
        raise ValueError('distribution file inventory is unavailable')
    result = []
    seen = set()
    for entry in distribution.files:
        try:
            entry_text = os.fspath(entry)
            if type(entry_text) is not str or any(char in entry_text for char in ('\\', ':', '\x00')):
                raise ValueError('distribution record path must use portable POSIX syntax')
            relative_entry = Path(entry_text)
            if relative_entry.is_absolute() or relative_entry.drive:
                raise ValueError('distribution record path must be relative')
            target = site
            for part in relative_entry.parts:
                if part == '..':
                    if target == root:
                        raise ValueError('distribution file escapes the selected environment')
                    target = target.parent
                    continue
                if part == '.':
                    continue
                target = target / part
                info = target.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
                    raise ValueError('distribution file path must not use links')
            resolved = target.resolve(strict=True)
            located = Path(distribution.locate_file(entry)).resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise ValueError('recorded distribution file is unavailable') from exc
        if resolved != located or not resolved.is_relative_to(root):
            raise ValueError('distribution file escapes the selected environment')
        relative = resolved.relative_to(root).as_posix()
        if relative in seen:
            raise ValueError('duplicate distribution file')
        seen.add(relative)
        result.append({'path': relative, **_plain_file_hash(target)})
    return sorted(result, key=lambda item: item['path'])


def verify_asset_file(path, cache_root, expected_size, expected_sha256):
    """Verify the actual offline checkpoint bytes against the pinned asset."""
    if (type(expected_size) is not int or expected_size < 0
            or type(expected_sha256) is not str
            or not re.fullmatch(r'[0-9a-f]{64}', expected_sha256)):
        raise ValueError('offline asset identity is invalid')
    try:
        root = Path(cache_root).resolve(strict=True)
        resolved = Path(path).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ValueError('offline asset is unavailable') from exc
    if not resolved.is_relative_to(root):
        raise ValueError('offline asset escapes the selected cache')
    actual = _plain_file_hash(resolved)
    if actual != {'size': expected_size, 'sha256': expected_sha256}:
        raise ValueError('offline asset does not match its pinned bytes')
    return True


def _normalized_distribution_name(value):
    if type(value) is not str:
        raise ValueError('distribution name is unavailable')
    name = re.sub(r'[-_.]+', '-', value).lower()
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name):
        raise ValueError('invalid distribution name')
    return name


def collect_installed_distributions(site_packages, environment_root):
    """Hash all recorded files from the selected installed environment."""
    root = Path(environment_root).resolve(strict=True)
    site = _validated_site(site_packages, root)
    packages = []
    names = set()
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = _normalized_distribution_name(distribution.metadata.get('Name'))
        version = distribution.version
        if name in names or type(version) is not str or not version:
            raise ValueError('duplicate or unversioned distribution')
        names.add(name)
        packages.append({'name': name, 'version': version,
                         'files': collect_distribution_files(distribution, site, root)})
    required = {'huggingface-hub', 'numpy', 'safetensors', 'tokenizers', 'torch', 'transformers'}
    if not required.issubset(names):
        raise ValueError('required runtime distribution is missing')
    return sorted(packages, key=lambda item: item['name'])


def collect_offline_assets(repo_root, cache_root):
    """Hash the exact seven pinned checkpoint assets from the offline cache."""
    pretrained = importlib.import_module('looking_glass.pretrained')
    expected_module = (Path(repo_root).resolve(strict=True) / 'looking_glass' / 'pretrained.py').resolve(strict=True)
    if Path(pretrained.__file__).resolve(strict=True) != expected_module:
        raise ValueError('pretrained loader did not come from the verified checkout')
    model_id = 'distilbert/distilgpt2'
    revision = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
    if pretrained.MODEL_ID != model_id or pretrained.REVISION != revision:
        raise ValueError('offline loader does not use the bootstrap checkpoint')
    if pretrained.ASSETS != _PINNED_ASSETS:
        raise ValueError('offline loader assets differ from the bootstrap pin')
    expected_names = set(_PINNED_ASSETS)
    cache = Path(cache_root).resolve(strict=True)
    snapshot = cache / 'models--distilbert--distilgpt2' / 'snapshots' / revision
    records, paths = [], []
    for name in sorted(expected_names):
        size, digest = pretrained.ASSETS[name]
        target = snapshot / name
        verify_asset_file(target, cache, size, digest)
        records.append({'name': name, 'size': size, 'sha256': digest})
        paths.append((name, str(target.resolve(strict=True))))
    return {'schema': 'resource-bootstrap-assets-v1', 'model_id': model_id,
            'revision': revision, 'assets': records}, tuple(paths)


def _canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class SourcePreflight:
    repo_root: str
    cache_root: str
    site_packages: str
    environment_root: str
    source_manifest: bytes
    asset_manifest: bytes
    asset_paths: tuple

    @property
    def source_sha256(self):
        return hashlib.sha256(self.source_manifest).hexdigest()


def preflight_sources(repo_root, cache_root, site_packages, environment_root):
    """Collect CPU-only source/runtime-file and pinned-asset evidence.

    This uses actual file bytes, not a profile's declared identity. A trusted
    launcher must prevent source changes and revalidate before execution.
    """
    try:
        repo = Path(repo_root).resolve(strict=True)
        cache = Path(cache_root).resolve(strict=True)
        env = Path(environment_root).resolve(strict=True)
        site = _validated_site(site_packages, env)
    except (OSError, TypeError) as exc:
        raise ValueError('source, cache or runtime path is unavailable') from exc
    if not site.is_relative_to(env):
        raise ValueError('site-packages must be inside the selected environment')
    repo_info = collect_repository_source(repo)
    packages = collect_installed_distributions(site, env)
    executable = _plain_file_hash(Path(sys._base_executable).resolve(strict=True))
    source = {'schema': 'resource-bootstrap-source-v1', **repo_info,
              'python_executable': executable, 'distributions': packages}
    validate_source_manifest(source)
    assets, paths = collect_offline_assets(repo, cache)
    return SourcePreflight(str(repo), str(cache), str(site), str(env),
                           _canonical_json_bytes(source), _canonical_json_bytes(assets), paths)


def _verify_cpu_tensors(torch, model, weights):
    """Compare CPU model values, nonpersistent masks and tied storage to the pin."""
    if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise ValueError('model conversion flags differ from the fixed tie policy')
    state = model.state_dict()
    if not isinstance(state, dict):
        raise ValueError('model state dictionary is unavailable')
    mask_names = {f'transformer.h.{index}.attn.bias' for index in range(6)}
    masked_names = {f'transformer.h.{index}.attn.masked_bias' for index in range(6)}
    expected_names = set(weights.keys())
    if ('lm_head.weight' not in state or 'transformer.wte.weight' not in state
            or expected_names != (set(state) - {'lm_head.weight'}) | mask_names
            or set(dict(model.named_buffers())) != mask_names | masked_names):
        raise ValueError('model tensors or buffers do not match the pinned checkpoint')
    embedding, head = state['transformer.wte.weight'], state['lm_head.weight']
    if model.get_parameter('transformer.wte.weight') is not model.get_parameter('lm_head.weight'):
        raise ValueError('language-model head and token embeddings must be the same Parameter')
    for tensor in (embedding, head):
        if type(tensor) is not torch.Tensor or tensor.device.type != 'cpu':
            raise ValueError('model tensors must be ordinary CPU tensors')
    if (embedding.untyped_storage().data_ptr() != head.untyped_storage().data_ptr()
            or embedding.storage_offset() != head.storage_offset()
            or embedding.shape != head.shape or embedding.stride() != head.stride()):
        raise ValueError('language-model head must remain tied to token embeddings')
    storage_ranges = []
    for name in sorted(expected_names):
        expected = weights.get_tensor(name)
        actual = model.get_buffer(name) if name in mask_names else state[name]
        if (type(actual) is not torch.Tensor or type(expected) is not torch.Tensor
                or actual.device.type != 'cpu' or expected.device.type != 'cpu'
                or actual.shape != expected.shape):
            raise ValueError('checkpoint tensor shape or device differs from the model')
        storage = actual.untyped_storage()
        nbytes = actual.numel() * actual.element_size()
        if (actual.stride() != expected.stride() or actual.storage_offset() != 0
                or storage.nbytes() != nbytes or not nbytes):
            raise ValueError('model tensor storage layout differs from the checkpoint')
        start = storage.data_ptr()
        storage_ranges.append((start, start + nbytes, name))
        if name in mask_names:
            if actual.dtype is not torch.bool or not torch.equal(actual, expected.to(dtype=torch.bool)):
                raise ValueError('attention mask buffer differs from the pinned checkpoint')
        elif actual.dtype != expected.dtype or not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
            raise ValueError('model tensor differs from the pinned checkpoint')
    for name in sorted(masked_names):
        value = model.get_buffer(name)
        if (type(value) is not torch.Tensor or value.device.type != 'cpu'
                or value.dtype is not torch.float32 or value.shape != torch.Size([])
                or value.storage_offset() != 0 or value.untyped_storage().nbytes() != 4
                or value.item() != -10000.0):
            raise ValueError('nonpersistent attention masked-bias buffer changed')
        start = value.untyped_storage().data_ptr()
        storage_ranges.append((start, start + 4, name))
    storage_ranges.sort()
    if any(left[1] > right[0] for left, right in zip(storage_ranges, storage_ranges[1:])):
        raise ValueError('unrelated model tensors unexpectedly share storage')
    return True


def _check_global_module_hooks(torch):
    module_impl = torch.nn.modules.module
    if any(value for name, value in vars(module_impl).items()
           if name.startswith('_global_') and 'hooks' in name):
        raise ValueError('process-global module hooks must be absent')


def _verify_architecture(torch, model, reference):
    """Compare a loaded model with a freshly constructed meta-device model.

    This catches accidental local mutations, not malicious changes to trusted
    Python classes or functions. The reference comes from pinned configuration
    and the same verified Transformers implementation.
    """
    _check_global_module_hooks(torch)
    instance_overrides = ('forward', '_call_impl', '_wrapped_call_impl', '_slow_forward',
                          'to', '_apply', 'children', 'named_children', 'named_modules', 'modules',
                          'get_parameter', 'get_buffer', 'get_submodule', 'state_dict',
                          '_save_to_state_dict', 'named_buffers', 'named_parameters',
                          'parameters', 'buffers', '_named_members')
    if any(method in vars(model) for method in instance_overrides):
        raise ValueError('instance-level module execution overrides are prohibited')
    actual = dict(model.named_modules(remove_duplicate=False))
    expected = dict(reference.named_modules(remove_duplicate=False))
    if list(actual) != list(expected):
        raise ValueError('model module tree differs from the pinned architecture')
    simple = (str, int, float, bool, tuple, list, set, type(None))
    ignored = {'name_or_path', '_is_hf_initialized'}
    for name, module in actual.items():
        trusted = expected[name]
        if type(module) is not type(trusted) or module.training is not False:
            raise ValueError('model module type or mode differs from the pinned architecture')
        if any(method in vars(module) for method in instance_overrides):
            raise ValueError('instance-level module execution overrides are prohibited')
        if getattr(module, '_compiled_call_impl', None) is not None:
            raise ValueError('compiled model modules are prohibited')
        if any(value for key, value in vars(module).items()
               if key.startswith('_') and 'hook' in key and isinstance(value, (dict, set))):
            raise ValueError('model module hooks must be absent')
        if list(module._modules) != list(trusted._modules):
            raise ValueError('model child modules differ from the pinned architecture')
        for key, value in vars(trusted).items():
            if key in ignored or type(value) not in simple:
                continue
            actual_value = vars(module).get(key, object())
            if type(actual_value) is not type(value) or actual_value != value:
                raise ValueError(f'model execution setting changed: {name}.{key}')
        for key, value in vars(module).items():
            if key not in ignored and type(value) in simple and key not in vars(trusted):
                raise ValueError(f'unexpected model execution setting: {name}.{key}')
        if vars(trusted).get('config') is reference.config and vars(module).get('config') is not model.config:
            raise ValueError('model modules must use the verified shared configuration')
    return True


def verify_loaded_model(preflight, model):
    """Verify a loaded CPU DistilGPT2 against the exact offline checkpoint.

    Returns the singleton True only after every check. The caller must keep the
    model protected from mutation and separately revalidate the runtime binding
    before its supervised copy to CUDA.
    """
    if type(preflight) is not SourcePreflight:
        raise TypeError('a real source preflight is required')
    torch = _load_torch()
    transformers = importlib.import_module('transformers')
    safetensors = importlib.import_module('safetensors')
    if type(model) is not transformers.GPT2LMHeadModel:
        raise ValueError('model is not the pinned GPT2 language-model class')
    paths = dict(preflight.asset_paths)
    if set(paths) != set(_PINNED_ASSETS):
        raise ValueError('preflight asset paths are incomplete')
    for name in ('config.json', 'model.safetensors'):
        size, digest = _PINNED_ASSETS[name]
        verify_asset_file(paths[name], preflight.cache_root, size, digest)
    try:
        raw_config = json.loads(Path(paths['config.json']).read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError('pinned model configuration cannot be read') from exc
    if not isinstance(raw_config, dict):
        raise ValueError('pinned model configuration must be an object')
    config = model.config
    if type(config) is not transformers.GPT2Config:
        raise ValueError('model configuration has an unexpected type')
    reference_config = transformers.GPT2Config.from_dict(raw_config)
    reference_config.use_cache = False
    reference_config._attn_implementation = 'eager'
    expected_config = reference_config.to_dict()
    actual_config = config.to_dict()
    # The loader stages assets in a temporary directory and requests float32.
    # Neither staging path nor this loader-generated dtype changes the pin.
    expected_config.pop('_name_or_path', None)
    actual_config.pop('_name_or_path', None)
    expected_config['dtype'] = 'float32'
    if actual_config != expected_config:
        raise ValueError('loaded model configuration differs from the pinned configuration')
    for name in ('vocab_size', 'n_positions', 'n_ctx', 'n_embd', 'n_head', 'n_layer',
                 'activation_function', 'attn_pdrop', 'embd_pdrop', 'resid_pdrop',
                 'layer_norm_epsilon', 'initializer_range'):
        if name not in raw_config or getattr(config, name, None) != raw_config[name]:
            raise ValueError(f'loaded model configuration differs for {name}')
    expected_modes = {'use_cache': False, 'output_attentions': False,
                      'output_hidden_states': False, 'use_return_dict': True,
                      '_attn_implementation': 'eager', 'add_cross_attention': False,
                      'tie_word_embeddings': True}
    if any(getattr(config, name, None) != expected for name, expected in expected_modes.items()):
        raise ValueError('loaded model execution modes differ from the bootstrap')
    # The meta-device reference has the exact module tree and scalar execution
    # settings without allocating a second model's weights on CPU or CUDA.
    _check_global_module_hooks(torch)
    with torch.random.fork_rng(devices=[]):
        with torch.device('meta'):
            reference = transformers.GPT2LMHeadModel(reference_config).eval()
    if _verify_architecture(torch, model, reference) is not True:
        raise ValueError('model architecture verification did not complete')
    with safetensors.safe_open(paths['model.safetensors'], framework='pt', device='cpu') as weights:
        if _verify_cpu_tensors(torch, model, weights) is not True:
            raise ValueError('model tensor verification did not complete')
    return True


def _guard_owner(identity, owner_session, device_lease):
    if type(identity) is not DeviceIdentity or type(owner_session) is not OwnerSession or type(device_lease) is not DeviceLease:
        raise ValueError('a supervised owner, physical identity and actual device lease are required')
    DeviceIdentity.validate_paths(identity)
    DeviceLease.check(device_lease)
    OwnerSession.check(owner_session, device_lease)
    if owner_session.identity != identity:
        raise DeviceIdentityError('owner identity does not match the selected physical device')


def _load_torch():
    return importlib.import_module('torch')


def _driver_version(device_uuid):
    """Read the driver version for this physical UUID from the local inventory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('driver inventory is unavailable') from exc
    if len(result.stdout) > 16384:
        raise ValueError('driver inventory exceeds its allowed size')
    versions = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 2 or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', parts[0]):
            raise ValueError('invalid driver inventory')
        if 'GPU-' + parts[0][4:].lower() == device_uuid:
            versions.append(parts[1])
    if len(versions) != 1 or not re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}', versions[0]):
        raise ValueError('driver version does not identify the supervised device')
    return versions[0]


def _measure_runtime(torch, identity, owner_session, device_lease):
    """Apply and read back fixed bootstrap settings after supervised ownership.

    Allocator, device and backend values are checked here. Model/forward-mode
    entries are the planned fixed contract, not observations of a model or an
    execution thread; the owner must separately verify them at execution time.
    This does not guarantee future memory consumption. The calling owner must
    retain its lease until device teardown.
    """
    _guard_owner(identity, owner_session, device_lease)
    count = torch.cuda.device_count()
    if type(count) is not int or count != 1:
        raise ValueError('exactly one visible CUDA device is required')
    _guard_owner(identity, owner_session, device_lease)
    props = torch.cuda.get_device_properties(0)
    actual_uuid = decode_cuda_uuid(props.uuid, torch._C._CUuuid)
    OwnerSession.verify_runtime_binding(owner_session, device_lease,
                                        runtime_uuid=actual_uuid, device_count=count,
                                        cuda_device_index=0)
    if actual_uuid != identity.gpu_uuid:
        raise ValueError('runtime GPU UUID differs from the supervised identity')
    _guard_owner(identity, owner_session, device_lease)
    if torch.cuda.get_allocator_backend() != 'native':
        raise ValueError('the bootstrap requires the native CUDA allocator')
    _guard_owner(identity, owner_session, device_lease)
    before_free, before_total = torch.cuda.mem_get_info(0)
    if (type(before_total) is not int or not 7 * _GIB <= before_total <= 9 * _GIB
            or type(before_free) is not int or not 0 <= before_free <= before_total):
        raise ValueError('invalid initial CUDA memory telemetry')
    _guard_owner(identity, owner_session, device_lease)
    torch.cuda.set_per_process_memory_fraction(0.20, device=0)
    _guard_owner(identity, owner_session, device_lease)
    after_free, after_total = torch.cuda.mem_get_info(0)
    if (type(after_total) is not int or after_total != before_total or type(after_free) is not int
            or not 0 <= after_free <= after_total):
        raise ValueError('invalid CUDA memory telemetry after allocator setup')
    _guard_owner(identity, owner_session, device_lease)
    fraction = torch.cuda.get_per_process_memory_fraction(0)
    cap = check_allocator_readback(before_total, after_total, fraction)

    _guard_owner(identity, owner_session, device_lease)
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _guard_owner(identity, owner_session, device_lease)
    cudnn_version = torch.backends.cudnn.version()
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or torch.are_deterministic_algorithms_enabled() is not True
            or torch.get_float32_matmul_precision() != 'highest'
            or torch.backends.cuda.matmul.allow_tf32 is not False
            or torch.backends.cudnn.allow_tf32 is not False
            or torch.backends.cudnn.enabled is not True
            or torch.backends.cudnn.benchmark is not False
            or torch.backends.cudnn.deterministic is not True
            or type(cudnn_version) is not int or cudnn_version <= 0):
        raise ValueError('runtime settings could not be verified')
    if (type(props.name) is not str or not props.name or type(props.major) is not int
            or type(props.minor) is not int or props.major < 0 or props.minor < 0
            or type(torch.version.cuda) is not str or not torch.version.cuda):
        raise ValueError('CUDA runtime properties are incomplete')
    _guard_owner(identity, owner_session, device_lease)
    driver_version = _driver_version(actual_uuid)
    _guard_owner(identity, owner_session, device_lease)
    return {
        'device': 'cuda:0', 'device_uuid': actual_uuid, 'device_name': props.name,
        'device_total_bytes': before_total, 'compute_capability': [props.major, props.minor],
        'torch_version': str(torch.__version__), 'cuda_runtime_version': torch.version.cuda,
        'driver_version': driver_version, 'cudnn_version': cudnn_version,
        'allocator_backend': 'native', 'allocator_fraction': '0.20',
        'allocator_effective_limit_bytes': cap,
        'model_dtype': 'torch.float32', 'compute_dtype': 'torch.float32',
        'attention_backend': 'eager', 'model_mode': 'eval', 'grad_mode': 'inference',
        'autocast': 'disabled', 'use_cache': False, 'compiled': False,
        'float32_matmul_precision': 'highest', 'cuda_matmul_allow_tf32': False,
        'cudnn_allow_tf32': False, 'cudnn_enabled': True, 'cudnn_benchmark': False,
        'cudnn_deterministic': True, 'deterministic_algorithms': True,
        'execution_path': 'looking_glass.resident_worker.ResidentWorker.execute',
        'torch_num_threads': 1, 'torch_num_interop_threads': 1,
        'input_limit_bytes': 64 * 1024, 'output_tensor_limit_bytes': 512 * 1024,
        'response_wire_limit_bytes': 4 * 1024 * 1024,
    }


def _make_workload_keys(source_sha256, runtime):
    """Construct three distinct keys from the independently observed runtime."""
    fields = {
        'checkpoint_sha256': _PINNED_ASSETS['model.safetensors'][1],
        'checkpoint_revision': '2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'model_config_sha256': _PINNED_ASSETS['config.json'][1],
        'source_sha256': source_sha256,
        'runtime_settings_sha256': canonical_fingerprint(runtime),
    }
    for name in ('torch_version', 'transformers_version', 'cuda_runtime_version',
                 'driver_version', 'python_version', 'device', 'device_uuid',
                 'device_name', 'device_total_bytes', 'model_dtype', 'compute_dtype',
                 'attention_backend', 'model_mode', 'grad_mode', 'autocast',
                 'use_cache', 'compiled'):
        fields[name] = runtime[name]
    fields['compute_capability'] = tuple(runtime['compute_capability'])
    model_id = model_runtime_fingerprint(fields)
    warmup = canonical_fingerprint({
        'schema': 'resource-bootstrap-warmup-v1',
        'procedure': 'one_completed_cold_forward_same_case',
        'cold_seed': 17, 'warm_seeds': [29, 43, 71],
    })
    capture = canonical_fingerprint({
        'schema': 'resource-bootstrap-capture-v1', 'mode': 'none',
        'max_gpu_trace_bytes': 0, 'max_host_trace_bytes': 0,
    })
    common = {'model_id': model_id, **fields, 'warmup_protocol_sha256': warmup,
              'capture_mode': 'none', 'capture_policy_sha256': capture,
              'max_gpu_trace_bytes': 0, 'max_host_trace_bytes': 0}
    load = WorkloadKey(**common, operation='model_load', warmup_state='cold',
                       batch_size=0, sequence_length=0, input_dtypes=(), output_limit_bytes=0)
    forward = {'operation': 'forward', 'batch_size': 1, 'sequence_length': 2,
               'input_dtypes': (('input_ids', 'torch.int64'),),
               'output_limit_bytes': 512 * 1024}
    return (load, WorkloadKey(**common, **forward, warmup_state='cold'),
            WorkloadKey(**common, **forward, warmup_state='warm'))


def build_live_keys(preflight, identity, owner_session, device_lease):
    """Build live bootstrap keys only after supervised ownership is verified."""
    _guard_owner(identity, owner_session, device_lease)
    raise NotImplementedError('live bootstrap identity is under development')
