"""Source-first entry for injection held-out preparation.

This component does not itself authorize a split, claim, or model forward.
Its caller must bind the published operation and observe the child process.
"""

import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType
from urllib.request import Request, urlopen


_START = b'MLG-natural-discovery-start-v1\n'
_DISCOVERY_SOURCE = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
_LAUNCHER_SHA = '930fee8d77ece12b6b5598391a2fac63d688faa9237d1e6bd04ba96a60e3c110'
_HELDOUT_RUNTIME_SHA = '7a967712a896d19ccfe4164c11a57e3ab1c65ece1a0ae26fedb220b9fc80e327'
_PREPARATION_SOURCE = 'b3cf11f5d38e76db70d116c414d0944f725c1780'
_EXTRA_SOURCES = (
    ('injection_split_custody', '964d90b3674671e00fe9f8d72c67eb124d0afde2bbb499629c267e79fd5b27ff'),
    ('injection_preclaim_guard', '425a8b70c4512ef99bdc840a6095f536c9d7e0df18edee0ff2db65d729d0a320'),
    ('injection_phase_accounting', '133605ee534e1647c46215c3dae78970ec188f906ef1e238c0a264b5a7e65e0f'),
    ('injection_preparation_audit', '75af0477142e8aa002c73beba987668904284612c2dc88ca0eb6d06abb524a35'),
)


def parse_probe_config(raw):
    """Accept only the public paths needed for a no-model startup probe."""
    paths = {'frozen_root', 'site_packages', 'cache_dir', 'source_root',
             'heldout_runtime_file', 'pycache_prefix'}
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise ValueError('a bounded configuration is required')
    try:
        value = json.loads(raw)
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError('canonical configuration is required') from None
    if (canonical != raw or type(value) is not dict or set(value) != paths | {'schema'}
            or value['schema'] != 'mlg-injection-preparation-probe-v1'):
        raise ValueError('no-model configuration differs')
    for key in paths:
        item = value[key]
        if (type(item) is not str or not item or not Path(item).is_absolute()
                or '..' in Path(item).parts):
            raise ValueError('absolute no-model configuration paths are required')
    return value


def _read_bounded(path, limit):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ValueError('bounded bytes are required')
    return raw


def read_probe_config(path):
    return parse_probe_config(_read_bounded(path, 65536))


def _read_source(path, expected):
    path = Path(path)
    if (not path.is_absolute() or type(expected) is not str
            or re.fullmatch(r'[0-9a-f]{64}', expected) is None):
        raise ValueError('an absolute source path and SHA-256 are required')
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_nlink != 1
            or getattr(before, 'st_file_attributes', 0) & 0x400
            or not 0 < before.st_size <= 1024 * 1024):
        raise ValueError('a bounded regular source file is required')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(1024 * 1024 + 1)
        finished = os.fstat(stream.fileno())
    after = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_nlink)
    if (len(raw) != before.st_size or identity(before) != identity(opened)
            or identity(before) != identity(finished) or identity(before) != identity(after)
            or hashlib.sha256(raw).hexdigest() != expected):
        raise ValueError('bound source bytes differ or changed during reading')
    return raw


def load_verified_modules(sources):
    """Check all bytes before executing any additional project source.

    Each entry is ``(module_name, package_name, absolute_path, sha256)``.
    Published identities and the source-first process are caller obligations.
    """
    if type(sources) not in (list, tuple) or not sources:
        raise ValueError('bound source entries are required')
    entries, names = [], set()
    for name, package, path, expected in sources:
        if (type(name) is not str or not name or type(package) is not str
                or name in names or name in sys.modules):
            raise ValueError('source module identity is invalid')
        names.add(name)
        entries.append((name, package, Path(path), _read_source(path, expected)))
    loaded = {}
    try:
        for name, package, path, raw in entries:
            module = ModuleType(name)
            module.__file__ = str(path)
            module.__package__ = package
            sys.modules[name] = module
            loaded[name] = module
            exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        for name in loaded:
            sys.modules.pop(name, None)
        raise
    return loaded


def bootstrap_runtime(launcher_source, sources, *, site_packages, cache_dir,
                      source_commit, expected_runtime, publication_check):
    """Boot the frozen checkout and exact runtime before executing new modules.

    The publication callback must compare each extra source to the intended
    immutable public bytes. The trusted parent binds this entry's own source,
    the callback, private paths and the process lifecycle.
    """
    if (type(expected_runtime) is not bytes or not expected_runtime
            or not callable(publication_check)):
        raise ValueError('the published runtime and source observer are required')
    for _, _, path, expected in sources:
        raw = _read_source(path, expected)
        if publication_check(Path(path), raw) is not True:
            raise ValueError('published preparation source differs')
    name = launcher_source[0]
    launcher = load_verified_modules([launcher_source])[name]
    launcher._CONTEXT = launcher._boot(site_packages, source_commit)
    state = launcher._initialize_runtime(source_commit=source_commit, cache_dir=cache_dir,
                                         site_packages=site_packages)
    if state.get('runtime_bytes') != expected_runtime:
        raise ValueError('actual runtime differs from the published held-out runtime')
    return launcher, state, load_verified_modules(sources)


def _assert_isolated(prefix):
    flags = sys.flags
    if (not (flags.isolated and flags.ignore_environment and flags.no_site and flags.no_user_site
             and flags.safe_path and sys.dont_write_bytecode and flags.optimize == 0)
            or not sys.pycache_prefix or Path(sys.pycache_prefix) != Path(prefix)
            or Path(sys.executable).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)
            or any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules)
            or sys.meta_path != [importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                                 importlib.machinery.PathFinder]):
        raise ValueError('an isolated source-only interpreter is required')
    path = Path(prefix)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink() or any(path.iterdir()):
        raise ValueError('a fresh empty bytecode prefix is required')


def _read_public_source(path, raw):
    url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
           + _PREPARATION_SOURCE + '/looking_glass/' + path.name)
    request = Request(url, headers={'User-Agent': 'looking-glass-cpu-study'})
    with urlopen(request, timeout=20) as response:
        return response.geturl() == url and response.read(1024 * 1024 + 1) == raw


def validate_no_model(config):
    """Probe source and runtime startup; do not claim, draw, or load a model."""
    _assert_isolated(config['pycache_prefix'])
    root = Path(config['source_root']) / 'looking_glass'
    sources = [('looking_glass.' + name, 'looking_glass', root / (name + '.py'), digest)
               for name, digest in _EXTRA_SOURCES]
    runtime = _read_source(config['heldout_runtime_file'], _HELDOUT_RUNTIME_SHA)
    launcher_path = Path(config['frozen_root']) / 'scripts' / 'pretrained_number_cpu_launcher.py'
    launcher, state, modules = bootstrap_runtime(
        ('_mlg_preparation_launcher', '', launcher_path, _LAUNCHER_SHA), sources,
        site_packages=config['site_packages'], cache_dir=config['cache_dir'],
        source_commit=_DISCOVERY_SOURCE, expected_runtime=runtime, publication_check=_read_public_source,
    )
    site = str(Path(config['site_packages']).resolve(strict=True))
    if (sys.path.count(site) != 1 or state['events'] or state['torch'].cuda.is_initialized()
            or any(launcher._CONTEXT['prefix'].iterdir())
            or launcher._verify_unloaded_source(launcher._CONTEXT['repo'], _DISCOVERY_SOURCE)
               != state['repository']):
        raise ValueError('source, site or no-model runtime changed')
    for _, _, path, digest in sources:
        _read_source(path, digest)
    launcher._verify_effective_cache(state['home'], state['cache'])
    return {'schema': 'mlg-injection-preparation-probe-result-v1',
            'status': 'validated_no_model_forward', 'forward_count': 0,
            'heldout_runtime_sha256': _HELDOUT_RUNTIME_SHA,
            'preparation_source_commit': _PREPARATION_SOURCE,
            'preparation_source_sha256': hashlib.sha256(_read_bounded(__file__, 1024 * 1024)).hexdigest(),
            'verified_module_count': len(modules)}


def main(argv=None, stream=None):
    try:
        start = sys.stdin.buffer if stream is None else stream
        if start.readline(len(_START) + 1) != _START:
            raise ValueError('the supervisor start token is required')
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) != 2 or args[0] != '--validate-only':
            raise ValueError('only a no-model validation is available')
        path = Path(args[1])
        if not path.is_absolute():
            raise ValueError('an absolute configuration path is required')
        result = validate_no_model(read_probe_config(path))
        print(json.dumps(result, sort_keys=True, separators=(',', ':')), flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
