"""Stage or verify audited natural-discovery publication; never draw held-out cases.

Run this checked-in wrapper as an isolated, no-bytecode Python process. Its
frozen analysis checkout and private input files are separately read-protected.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types


WRAPPER_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_FILES = ('tools/publish_natural_prediction_freeze.py',
                 'looking_glass/natural_prediction_publication.py', 'scripts/v2_source_guards.py')


class GpuVisibilityError(RuntimeError):
    """Staging must hide CUDA before importing the frozen analysis."""


def _existing_path(path):
    lexical = Path(os.path.abspath(path))
    for part in (lexical, *lexical.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise RuntimeError('reparse points and symlinks are unsupported')
    return lexical.resolve(strict=True)


def _safe_output(path, protected_roots):
    lexical = Path(os.path.abspath(path))
    parent = _existing_path(lexical.parent)
    if not parent.is_dir() or not lexical.name:
        raise RuntimeError('output parent is not an existing directory')
    output = parent / lexical.name
    if os.path.lexists(output):
        raise RuntimeError('output already exists')
    if any(output.is_relative_to(_existing_path(root)) for root in protected_roots):
        raise RuntimeError('output lies inside a protected directory')
    return output


def _read(path, limit):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400 or info.st_size > limit):
        raise RuntimeError('input is not a bounded regular file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    after = path.lstat()
    if (len(raw) != info.st_size or len(raw) > limit or info.st_size != after.st_size
            or info.st_mtime_ns != after.st_mtime_ns):
        raise RuntimeError('input changed while being read')
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _environment():
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                'GIT_GRAFT_FILE': os.devnull, 'GIT_NO_REPLACE_OBJECTS': '1',
                'GIT_TERMINAL_PROMPT': '0', 'GIT_OPTIONAL_LOCKS': '0'})
    return env


def _git(root, *args):
    return subprocess.check_output(['git', '--no-replace-objects', '-C', str(root), *args],
                                   env=_environment(), stderr=subprocess.DEVNULL, timeout=30)


def _clean_revision(root, expected):
    if type(expected) is not str or re.fullmatch(r'[0-9a-f]{40}', expected) is None:
        raise RuntimeError('expected revision is invalid')
    if _git(root, 'rev-parse', '--verify', 'HEAD').strip().decode('ascii') != expected:
        raise RuntimeError('checkout revision differs from the declared revision')
    if _git(root, 'status', '--porcelain=v1', '--untracked-files=all').strip():
        raise RuntimeError('checkout is not clean')


def _wrapper_sources(expected):
    _clean_revision(WRAPPER_ROOT, expected)
    found = {}
    for relative in WRAPPER_FILES:
        raw = _read(WRAPPER_ROOT / relative, 8 * 1024 * 1024)
        committed = _git(WRAPPER_ROOT, 'show', expected + ':' + relative)
        if raw != committed:
            raise RuntimeError('wrapper source differs from its committed bytes')
        found[relative] = _sha(raw)
    return found


def _source_files(root, inventory):
    if type(inventory) is not dict or not 1 <= len(inventory) <= 4096:
        raise RuntimeError('frozen source inventory is invalid')
    files = []
    for name, digest in inventory.items():
        if (type(name) is not str or type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                or not ((name.startswith('looking_glass/') and name.endswith('.py'))
                        or name in ('scripts/v2_source_guards.py', 'tools/natural_agreement_cpu.py',
                                    'tools/audit_natural_agreement_discovery.py'))
                or '\\' in name or any(part in ('', '.', '..') for part in name.split('/'))):
            raise RuntimeError('frozen source entry is invalid')
        path = root / name
        if not path.resolve(strict=True).is_relative_to(root):
            raise RuntimeError('frozen source leaves its checkout')
        if _sha(_read(path, 8 * 1024 * 1024)) != digest:
            raise RuntimeError('frozen source bytes changed')
        files.append(path)
    actual = {path.relative_to(root).as_posix() for path in (root / 'looking_glass').rglob('*.py')}
    if actual != {name for name in inventory if name.startswith('looking_glass/')}:
        raise RuntimeError('frozen Python source inventory is incomplete')
    return files


class _VerifiedLoader(importlib.abc.Loader):
    def __init__(self, path, digest):
        self.path, self.digest = path, digest

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = _read(self.path, 8 * 1024 * 1024)
        if _sha(source) != self.digest:
            raise RuntimeError('source changed before import')
        module.__file__ = str(self.path)
        exec(compile(source, str(self.path), 'exec', dont_inherit=True), module.__dict__)


class _VerifiedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, frozen_root, inventory, wrapper_digest):
        self.root, self.inventory, self.wrapper_digest = frozen_root, inventory, wrapper_digest

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'looking_glass' and not fullname.startswith('looking_glass.'):
            return None
        if any(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) is None for part in fullname.split('.')):
            raise ModuleNotFoundError('invalid project module')
        if fullname == 'looking_glass.natural_prediction_publication':
            relative, root, digest, package = ('looking_glass/natural_prediction_publication.py',
                                               WRAPPER_ROOT, self.wrapper_digest, False)
        else:
            stem = fullname.replace('.', '/')
            package_name, module_name = stem + '/__init__.py', stem + '.py'
            package = package_name in self.inventory
            relative = package_name if package else module_name
            root, digest = self.root, self.inventory.get(relative)
        if digest is None:
            raise ModuleNotFoundError('project source was not present in the frozen binding')
        source = root / relative
        loader = _VerifiedLoader(source, digest)
        return importlib.util.spec_from_loader(fullname, loader, origin=str(source), is_package=package)


def _guard_source(wrapper_hashes):
    path = WRAPPER_ROOT / 'scripts/v2_source_guards.py'
    raw = _read(path, 8 * 1024 * 1024)
    if _sha(raw) != wrapper_hashes['scripts/v2_source_guards.py']:
        raise RuntimeError('wrapper guard changed before import')
    module = types.ModuleType('_publication_checked_guard')
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module.SourceFileLease


def _group_files(files):
    groups = {}
    for path in files:
        path = Path(path).absolute()
        groups.setdefault(path.parent, set()).add(path)
    return groups


def _stage(args, wrapper_hashes):
    root = _existing_path(args.source_root)
    binding_path = _existing_path(args.binding)
    original_binding = _read(binding_path, 2 * 1024 * 1024)
    binding = json.loads(original_binding)
    if (type(binding) is not dict or binding.get('source_commit') != args.source_revision
            or binding.get('independent_natural_audit_source_sha256') != args.auditor_sha256
            or binding.get('natural_prediction_clarification_sha256') != args.clarification_sha256):
        raise RuntimeError('binding does not match independently supplied identities')
    _clean_revision(root, args.source_revision)
    inventory = binding['source_sha256']
    frozen_files = _source_files(root, inventory)
    if (inventory.get('tools/audit_natural_agreement_discovery.py') != args.auditor_sha256
            or 'looking_glass/natural_prediction_freeze.py' not in inventory):
        raise RuntimeError('frozen analysis or auditor source is missing')
    owner = _existing_path(args.owner)
    attempt = owner / 'attempt'
    tensors = _existing_path(attempt / 'tensors')
    tensor_paths = [_existing_path(path) for path in tensors.iterdir()]
    if not 0 < len(tensor_paths) <= 4096:
        raise RuntimeError('tensor custody inventory is unavailable or oversized')
    paths = {'events': (attempt / 'events.jsonl', 64 * 1024 * 1024),
             'started': (attempt / 'started.json', 256 * 1024),
             'terminal': (attempt / 'terminal.json', 256 * 1024),
             'supervisor': (owner / 'supervisor.json', 2 * 1024 * 1024),
             'launcher_terminal': (Path(args.launcher_terminal).absolute(), 256 * 1024),
             'binding': (binding_path, 2 * 1024 * 1024),
             'audit': (Path(args.audit).absolute(), 2 * 1024 * 1024),
             'preflight_manifest': (Path(args.preflight_manifest).absolute(), 256 * 1024)}
    paths = {name: (_existing_path(path), limit) for name, (path, limit) in paths.items()}
    groups = _group_files([path for path, _ in paths.values()] + tensor_paths)
    output = _safe_output(args.output, [root, WRAPPER_ROOT, *groups])
    Guard = _guard_source(wrapper_hashes)
    with ExitStack() as stack:
        stack.enter_context(Guard(WRAPPER_ROOT, [WRAPPER_ROOT / name for name in WRAPPER_FILES]))
        stack.enter_context(Guard(root, frozen_files))
        for parent, files in sorted(groups.items(), key=lambda item: str(item[0])):
            stack.enter_context(Guard(parent, sorted(files)))
        if _wrapper_sources(args.wrapper_revision) != wrapper_hashes:
            raise RuntimeError('wrapper changed before protection')
        _clean_revision(root, args.source_revision)
        _source_files(root, inventory)
        data = {name: _read(path, limit) for name, (path, limit) in paths.items()}
        if data['binding'] != original_binding:
            raise RuntimeError('binding changed before protection')
        if any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules):
            raise RuntimeError('project modules were already imported')
        finder = _VerifiedFinder(root, inventory, wrapper_hashes['looking_glass/natural_prediction_publication.py'])
        sys.meta_path.insert(0, finder)
        from looking_glass import natural_prediction_publication as publication
        data.update(expected_source_commit=args.source_revision, expected_audit_source_sha256=args.auditor_sha256,
                    expected_builder_sha256=inventory['looking_glass/natural_prediction_freeze.py'],
                    expected_clarification_sha256=args.clarification_sha256)
        publication.prepare_stage(inputs=data, tensor_dir=tensors, output_dir=output,
                                  binding_url=args.binding_url, wrapper_commit=args.wrapper_revision)
        publication._tensor_inventory(tensors, data['supervisor'])
        if any(_read(path, limit) != data[name] for name, (path, limit) in paths.items()):
            raise RuntimeError('evidence changed during staging')
        if _wrapper_sources(args.wrapper_revision) != wrapper_hashes:
            raise RuntimeError('wrapper changed during staging')
        _source_files(root, inventory)
        _clean_revision(root, args.source_revision)
    return _sha(_read(output / 'stage.json', 2 * 1024 * 1024))


def _verify(args, wrapper_hashes):
    Guard = _guard_source(wrapper_hashes)
    stage = _existing_path(args.stage)
    stage_files = [stage / name for name in ('stage.json', 'runtime-binding.json',
                                              'natural-prediction-v2.json', 'natural-prediction-freeze.json')]
    track = _existing_path(args.track_receipt) if args.mode == 'aggregate' else None
    roots = [stage, WRAPPER_ROOT] + ([track.parent] if track is not None else [])
    receipt = _safe_output(args.receipt, roots)
    with ExitStack() as stack:
        stack.enter_context(Guard(WRAPPER_ROOT, [WRAPPER_ROOT / name for name in WRAPPER_FILES]))
        stack.enter_context(Guard(stage, stage_files))
        if args.mode == 'aggregate':
            stack.enter_context(Guard(track.parent, [track]))
        if _wrapper_sources(args.wrapper_revision) != wrapper_hashes:
            raise RuntimeError('wrapper changed before protection')
        path = WRAPPER_ROOT / 'looking_glass/natural_prediction_publication.py'
        raw = _read(path, 8 * 1024 * 1024)
        if _sha(raw) != wrapper_hashes['looking_glass/natural_prediction_publication.py']:
            raise RuntimeError('publication code changed before import')
        module = types.ModuleType('_publication_checked_io')
        module.__package__ = 'looking_glass'
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
        if args.mode == 'track':
            record = module.verify_track_publication(stage, args.stage_sha256, args.prediction_commit, receipt)
        else:
            record = module.verify_aggregate_publication(stage, args.stage_sha256, args.aggregate_commit,
                                                         track, receipt)
    return _sha(_read(receipt, 256 * 1024)), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('stage', 'track', 'aggregate'))
    parser.add_argument('--wrapper-revision', required=True)
    for name in ('source-root', 'source-revision', 'binding', 'binding-url', 'owner', 'launcher-terminal',
                 'audit', 'preflight-manifest', 'auditor-sha256', 'clarification-sha256', 'output',
                 'stage', 'stage-sha256', 'prediction-commit', 'aggregate-commit', 'track-receipt', 'receipt'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError('run the wrapper using Python -I -B')
    if args.mode == 'stage' and os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise GpuVisibilityError('set CUDA_VISIBLE_DEVICES=-1 before staging')
    required = {'stage': ('source_root', 'source_revision', 'binding', 'binding_url', 'owner',
                          'launcher_terminal', 'audit', 'preflight_manifest', 'auditor_sha256',
                          'clarification_sha256', 'output'),
                'track': ('stage', 'stage_sha256', 'prediction_commit', 'receipt'),
                'aggregate': ('stage', 'stage_sha256', 'aggregate_commit', 'track_receipt', 'receipt')}
    if any(not getattr(args, name) for name in required[args.mode]):
        raise RuntimeError('required publication arguments are missing')
    hashes = _wrapper_sources(args.wrapper_revision)
    if args.mode == 'stage':
        print(json.dumps({'status': 'stage_complete', 'stage_sha256': _stage(args, hashes)}, sort_keys=True))
    else:
        receipt_sha, _ = _verify(args, hashes)
        print(json.dumps({'status': args.mode + '_observed', 'receipt_sha256': receipt_sha}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
