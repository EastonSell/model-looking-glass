"""Source-bound public inputs for a private split comparison.

The independently pinned manifest is established by the held-out owner before
the split is drawn. This component is not itself a comparison or a forward
authorization. It never reads private split or site records.
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


_RAW = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
_PROTOCOL = '95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee'
_CHECKPOINT = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
_SOURCE = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
_RUNTIME = 'b6860ec21fe6f0ab626164b86b494d783f3c18c92a697744b118be4cdb33754e'
_PREDICTION = '11376d8aae517db76d6a33c1ae5dc1f4b0c747a9d5216c9b02a17b1b0b300107'
_PLAN = 'e59d98638e474ed9f915d24fa527edd6357caa0e62c8bdabdda0433c20c683f8'
_PINNED_SOURCES = (
    ('exposures', '6ec5a00a6950a739ddfab1c0335065609aead275f0f393323ecb22699d93a9cd',
     _RAW + '02e1c5d5c79813e3d67bbd0468b599926b4b9f83/looking_glass/pretrained_exposures.py'),
    ('exposures_v2', 'fa9014fb5cce2c3d8be3497a561754e47115ec07b9be3a896bb3bf1b20f8c474',
     _RAW + '02e1c5d5c79813e3d67bbd0468b599926b4b9f83/looking_glass/pretrained_exposures_v2.py'),
    ('exposures_v3', '2546bb672a461bfea10f0aed5e83c3a2110b2c898b56ff17542cb4718fc32886',
     _RAW + '02e1c5d5c79813e3d67bbd0468b599926b4b9f83/looking_glass/pretrained_exposures_v3.py'),
    ('commitments', '635f25553c20dd437f5cc4a8a9ca3a7a1453e7f8a517875fe75eb01ede2d5f35',
     _RAW + '02e1c5d5c79813e3d67bbd0468b599926b4b9f83/looking_glass/pretrained_number_commitments.py'),
    ('custody', '964d90b3674671e00fe9f8d72c67eb124d0afde2bbb499629c267e79fd5b27ff',
     _RAW + 'b3cf11f5d38e76db70d116c414d0944f725c1780/looking_glass/injection_split_custody.py'),
    ('preparation_audit', '75af0477142e8aa002c73beba987668904284612c2dc88ca0eb6d06abb524a35',
     _RAW + 'b3cf11f5d38e76db70d116c414d0944f725c1780/looking_glass/injection_preparation_audit.py'),
    ('source_guards', '9a9a6c2db19fc5923a1845fb620aebf43135a1cc72150f16b095a019576c7c5c',
     _RAW + '479b41fa3ab1b3131644fa14d68d761ce9672f42/scripts/v2_source_guards.py'),
    ('observer', 'fa878a9f8dbd555e23aab8090005ecd632b3e1061288b87d7a6399e2932ee95e',
     _RAW + '3f3ddcce46b5c64d4b08b8e6e177b643206eca07/looking_glass/injection_outer_observer.py'),
    ('comparison', '3e61675a962a205aebf336b84699ee7a4fc6d500b01cb25ab7ebea0fda094ab4',
     _RAW + 'b83cc2cf7b2534318f66145f07873c890feb5f1e/tools/injection_split_comparison.py'),
    ('adapter', '4570ab75d1858dcdbd9b274651b2c12efad8973cc5188167daa6829419f4405c',
     _RAW + '12405019193d341d732c97d0956231ffa9025bd4/tools/injection_comparison_observation.py'),
    ('exit', 'a6c6c0fbf4299f8f0053703054a9e33639ecc031d4af734739be98d7a894f533',
     _RAW + '6d79c0f5a6e5e1f43c13dd64e7852cc1a8b6a0b3/tools/injection_comparison_exit.py'),
    ('replay_owner', '2370697df2f30a86abf3b774935a0978687a45fe81d6e0cab79214e326c5fbb2',
     _RAW + 'd03499a24ff5350d3d693d0376a65e3fa1e2dabd/tools/injection_runtime_replay_owner.py'),
    ('natural_query', '3f871096f9e2e111b1d456db03397bb149bfe11d9ba2af52206db3c3cf31ac26',
     _RAW + '9898ce1599bb9c9b3c88108a37219c107f3d75cb/tools/natural_successor_query.py'),
)
_MODULE_SPECS = (
    ('exposures', 'looking_glass.pretrained_exposures', 'looking_glass'),
    ('exposures_v2', 'looking_glass.pretrained_exposures_v2', 'looking_glass'),
    ('exposures_v3', 'looking_glass.pretrained_exposures_v3', 'looking_glass'),
    ('commitments', 'looking_glass.pretrained_number_commitments', 'looking_glass'),
    ('custody', 'looking_glass.injection_split_custody', 'looking_glass'),
    ('preparation_audit', 'looking_glass.injection_preparation_audit', 'looking_glass'),
    ('source_guards', 'scripts.v2_source_guards', 'scripts'),
    ('observer', 'looking_glass.injection_outer_observer', 'looking_glass'),
    ('comparison', '_mlg_split_comparison', ''),
    ('adapter', '_mlg_comparison_observation', ''),
    ('exit', '_mlg_comparison_exit', ''),
    ('replay_owner', '_mlg_comparison_replay_owner', ''),
    ('natural_query', '_mlg_comparison_natural_query', ''),
)
_PRIVATE_IDENTITIES = frozenset({'natural_parent', 'natural_phase', 'injection_parent', 'injection_phase',
    'site_parent', 'natural_nonce', 'natural_record', 'natural_commitment', 'injection_nonce',
    'injection_record', 'injection_commitment', 'injection_inventory', 'site_nonce', 'site_record'})
_START = b'MLG-natural-discovery-start-v1\n'
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_HEX40 = re.compile(r'[0-9a-f]{40}\Z')
_URL = re.compile(re.escape(_RAW) + r'[0-9a-f]{40}/[A-Za-z0-9._/-]+\Z')


class OperatorError(ValueError):
    """Public identities or an independently bound source differ."""


def _sha(raw):
    if type(raw) is not bytes:
        raise OperatorError('bounded public bytes are required')
    return hashlib.sha256(raw).hexdigest()


def _parse(raw, maximum):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise OperatorError('bounded public bytes are required')
    try:
        value = json.loads(raw)
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                             separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise OperatorError('canonical public JSON is required') from error
    if type(value) is not dict or raw != encoded:
        raise OperatorError('canonical public JSON is required')
    return value


def _digest(value):
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise OperatorError('a lowercase SHA-256 digest is required')
    return value


def _url(value):
    if (type(value) is not str or _URL.fullmatch(value) is None
            or any(part in ('', '.', '..') for part in value[len(_RAW):].split('/'))):
        raise OperatorError('an immutable repository URL is required')
    return value


def _fetch(url):
    _url(url)
    with urlopen(Request(url, headers={'User-Agent': 'looking-glass-cpu-study'}), timeout=20) as response:
        if response.geturl() != url:
            raise OperatorError('published source redirected')
        raw = response.read(1024 * 1024 + 1)
    if not 0 < len(raw) <= 1024 * 1024:
        raise OperatorError('published bytes exceed their bound')
    return raw


def parse_operator_config(raw, expected_manifest_sha256, expected_manifest_url):
    """Parse paths and identities against the owner's independent manifest pin."""
    value = _parse(raw, 65536)
    paths = {'comparison_config_path', 'operation_path', 'manifest_path', 'natural_parent',
             'injection_parent', 'site_parent', 'terminal_path', 'inner_output', 'base_python', 'pycache_prefix'}
    keys = paths | {'schema', 'source_paths', 'comparison_config_sha256', 'operation_sha256',
                    'operation_url', 'manifest_sha256', 'manifest_url', 'identities', 'timeout'}
    if (set(value) != keys or value.get('schema') != 'mlg-injection-comparison-operator-config-v1'
            or type(value['source_paths']) is not dict
            or set(value['source_paths']) != {name for name, _, _ in _PINNED_SOURCES}
            or type(value['identities']) is not dict or set(value['identities']) != _PRIVATE_IDENTITIES
            or type(value['timeout']) is not int or not 1 <= value['timeout'] <= 7200):
        raise OperatorError('the comparison operator configuration differs')
    for item in [*(value[key] for key in paths), *value['source_paths'].values()]:
        if type(item) is not str or not item or not Path(item).is_absolute() or '..' in Path(item).parts:
            raise OperatorError('absolute configuration paths are required')
    for key in ('comparison_config_sha256', 'operation_sha256', 'manifest_sha256'):
        _digest(value[key])
    _url(value['operation_url'])
    _url(value['manifest_url'])
    if (value['manifest_sha256'] != _digest(expected_manifest_sha256)
            or value['manifest_url'] != _url(expected_manifest_url)):
        raise OperatorError('the independent manifest pin differs')
    result = dict(value)
    result['identities'] = {}
    for name, identity in value['identities'].items():
        if (type(identity) is not list or len(identity) != 2
                or any(type(item) is not int or item <= 0 for item in identity)):
            raise OperatorError('a private path identity is invalid')
        result['identities'][name] = tuple(identity)
    return result


def _assert_isolated(prefix):
    flags = sys.flags
    if (not (flags.isolated and flags.ignore_environment and flags.no_site and flags.no_user_site
             and flags.safe_path and sys.dont_write_bytecode and flags.optimize == 0)
            or not sys.pycache_prefix or Path(sys.pycache_prefix) != Path(prefix)
            or Path(sys.executable).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)
            or any(name in ('looking_glass', 'scripts') or name.startswith(('looking_glass.', 'scripts.'))
                   for name in sys.modules)
            or sys.meta_path != [importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                                 importlib.machinery.PathFinder]):
        raise OperatorError('an isolated source-first interpreter is required')
    path = Path(prefix)
    if not path.is_absolute() or '..' in path.parts or not path.is_dir() or any(path.iterdir()):
        raise OperatorError('a fresh empty operator bytecode prefix is required')
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise OperatorError('the operator bytecode prefix crosses a link')


def _read_source(path, expected):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise OperatorError('absolute source paths are required')
    try:
        for item in (path, *path.parents):
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise OperatorError('source path crosses a link or reparse point')
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= 1024 * 1024:
            raise OperatorError('a bounded regular source is required')
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read(1024 * 1024 + 1)
            finished = os.fstat(stream.fileno())
        after = path.lstat()
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_nlink)
        if (len(raw) != before.st_size or identity(before) != identity(opened)
                or identity(before) != identity(finished) or identity(before) != identity(after)
                or _sha(raw) != expected):
            raise OperatorError('source differs or changed while reading')
        return raw
    except OSError as error:
        raise OperatorError('source is unavailable') from error


def capture_sources(paths, reader):
    """Read all fixed local and immutable public source bytes before execution."""
    if (type(paths) is not dict or set(paths) != {name for name, _, _ in _PINNED_SOURCES}
            or not callable(reader)):
        raise OperatorError('the fixed source paths are required')
    modules = {name: (module, package) for name, module, package in _MODULE_SPECS}
    if set(modules) != set(paths):
        raise OperatorError('source module mapping differs')
    captured = []
    for name, digest, url in _PINNED_SOURCES:
        raw = _read_source(paths[name], digest)
        if reader(url) != raw:
            raise OperatorError('published source bytes differ')
        module, package = modules[name]
        captured.append((name, module, package, Path(paths[name]), raw))
    return tuple(captured)


def execute_sources(captured):
    """Execute previously captured sources in an otherwise clean namespace."""
    if (type(captured) is not tuple or len(captured) != len(_PINNED_SOURCES)
            or [entry[0] for entry in captured] != [entry[0] for entry in _PINNED_SOURCES]):
        raise OperatorError('all verified source bytes are required')
    for entry, (name, expected, _), (spec_name, module, package) in zip(
            captured, _PINNED_SOURCES, _MODULE_SPECS):
        if (type(entry) is not tuple or len(entry) != 5 or spec_name != name
                or entry[:3] != (name, module, package) or _sha(entry[4]) != expected):
            raise OperatorError('captured source or module identity differs')
    packages = {package for _, _, package, _, _ in captured if package}
    names = [module for _, module, _, _, _ in captured]
    if (len(set(names)) != len(names)
            or any(name in sys.modules for name in packages | set(names))
            or any(any(loaded.startswith(package + '.') for loaded in sys.modules) for package in packages)):
        raise OperatorError('project modules were already loaded')
    loaded = []
    result = {}
    try:
        for name in sorted(packages):
            package = ModuleType(name)
            package.__package__ = name
            package.__path__ = []
            sys.modules[name] = package
            loaded.append(name)
        for key, name, package, path, raw in captured:
            module = ModuleType(name)
            module.__file__ = str(path)
            module.__package__ = package
            sys.modules[name] = module
            loaded.append(name)
            exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
            result[key] = module
        return result
    except BaseException:
        for name in reversed(loaded):
            sys.modules.pop(name, None)
        raise


def verify_site_custody_metadata(acl, path, identity, *, directory):
    """Check the original site's existing ACL and identity without reading it.

    Its protected directory grants OWNER RIGHTS, SYSTEM and Administrators;
    the contained files inherit those grants. This verifies that existing
    policy, not secrecy from administrators or an OS security boundary.
    """
    path = Path(path)
    if (not path.is_absolute() or '..' in path.parts or type(directory) is not bool
            or type(identity) is not tuple or len(identity) != 2
            or any(type(value) is not int or value < 0 for value in identity)
            or not callable(getattr(acl, '_windows_acl_snapshot', None))):
        raise OperatorError('bound site custody metadata is required')
    try:
        before = path.lstat()
        if ((not stat.S_ISDIR(before.st_mode) if directory else not stat.S_ISREG(before.st_mode))
                or stat.S_ISLNK(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 0x400
                or (not directory and before.st_nlink != 1)
                or (before.st_dev, before.st_ino) != identity):
            raise OperatorError('site custody path identity differs')
        snapshot = acl._windows_acl_snapshot(path)
        if (type(snapshot) is not dict or set(snapshot) != {'user', 'owner', 'protected', 'rules'}
                or type(snapshot['user']) is not str
                or re.fullmatch(r'S-1-(?:[0-9]+-){2,14}[0-9]+', snapshot['user']) is None
                or snapshot['user'] == 'S-1-5-18' or snapshot['owner'] != snapshot['user']
                or snapshot['protected'] is not directory or type(snapshot['rules']) is not list
                or len(snapshot['rules']) != 3):
            raise OperatorError('site custody ACL differs')
        principals = set()
        for rule in snapshot['rules']:
            fixed = {'type': 0, 'inheritance': 3 if directory else 0,
                     'propagation': 0, 'inherited': not directory, 'rights': 0x1f01ff}
            if (type(rule) is not dict or set(rule) != set(fixed) | {'sid'} or type(rule['sid']) is not str
                    or any(type(rule.get(key)) is not type(value) or rule[key] != value
                           for key, value in fixed.items())):
                raise OperatorError('site custody ACL differs')
            principals.add(rule['sid'])
        if principals != {'S-1-3-4', 'S-1-5-18', 'S-1-5-32-544'}:
            raise OperatorError('site custody ACL principals differ')
        after = path.lstat()
        if (after.st_dev, after.st_ino) != identity:
            raise OperatorError('site custody identity changed')
        return True
    except OSError as error:
        raise OperatorError('site custody metadata is unavailable') from error


def _private_paths(comparison, config, natural_parent, injection_parent, site_parent):
    if type(config) is not dict:
        raise OperatorError('bound private paths are required')
    try:
        natural_parent, injection_parent, site_parent = map(Path, (natural_parent, injection_parent, site_parent))
        if any(not path.is_absolute() or '..' in path.parts
               for path in (natural_parent, injection_parent, site_parent)):
            raise OperatorError('absolute private parent paths are required')
        if any(first == second or first in second.parents or second in first.parents
               for index, first in enumerate((natural_parent, injection_parent, site_parent))
               for second in (natural_parent, injection_parent, site_parent)[index + 1:]):
            raise OperatorError('the private parent roots must be distinct')
        natural_phase = natural_parent / ('natural-heldout-' + comparison._NATURAL_FREEZE)
        injection_phase = injection_parent / ('injection-heldout-' + comparison._INJECTION_FREEZE)
        paths = {'natural_parent': natural_parent, 'natural_phase': natural_phase,
            'injection_parent': injection_parent, 'injection_phase': injection_phase,
            'site_parent': site_parent}
        for track, phase in (('natural', natural_phase), ('injection', injection_phase)):
            for field, leaf in (('nonce', 'nonce.bin'), ('record', 'heldout-record.json'),
                                ('commitment', 'commitment.json')):
                path = phase / leaf
                if Path(config[track][field]) != path:
                    raise OperatorError('a private split path differs')
                paths[track + '_' + field] = path
        paths['injection_inventory'] = injection_phase / 'inventory.json'
        paths['site_nonce'] = site_parent / 'nonce.bin'
        paths['site_record'] = site_parent / 'site-record.json'
        if (Path(config['injection']['inventory']) != paths['injection_inventory']
                or Path(config['site_nonce']) != paths['site_nonce']
                or Path(config['site_record']) != paths['site_record']):
            raise OperatorError('private site or inventory paths differ')
        return paths
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, OperatorError):
            raise
        raise OperatorError('private paths are invalid') from error


def verify_private_inputs(comparison, config, *, acl, natural_query, natural_parent,
                          injection_parent, site_parent, identities):
    """Recheck exact private paths, identities and their distinct ACL policies.

    The caller independently anchors the three parents and file identities,
    holds the file leases, and establishes the injection phase after its one
    permitted preparation. This routine reads metadata, never private bytes.
    """
    if (type(identities) is not dict or not callable(getattr(acl, 'verify_private_parent', None))
            or not callable(getattr(natural_query, '_verify_inherited_private', None))):
        raise OperatorError('bound private path and ACL inputs are required')
    try:
        paths = _private_paths(comparison, config, natural_parent, injection_parent, site_parent)
        if set(identities) != set(paths):
            raise OperatorError('private path identities differ')
        for identity in identities.values():
            if (type(identity) is not tuple or len(identity) != 2
                    or any(type(value) is not int or value < 0 for value in identity)):
                raise OperatorError('private path identities are invalid')
        for name in ('natural_parent', 'injection_parent'):
            if acl.verify_private_parent(paths[name], identities[name]) is not True:
                raise OperatorError('a private parent ACL differs')
        for name in ('natural_phase', 'injection_phase'):
            natural_query._verify_inherited_private(acl, paths[name], identities[name], directory=True)
        for name in ('natural_nonce', 'natural_record', 'natural_commitment', 'injection_nonce',
                     'injection_record', 'injection_commitment', 'injection_inventory'):
            natural_query._verify_inherited_private(acl, paths[name], identities[name], directory=False)
        for name in ('site_parent', 'site_nonce', 'site_record'):
            verify_site_custody_metadata(acl, paths[name], identities[name], directory=name == 'site_parent')
        return True
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, OperatorError):
            raise
        raise OperatorError('private input metadata was not established') from error


def validate_source_manifest(raw, expected_sha256, url, reader):
    """Check the independently pinned manifest and reviewed source identities."""
    _digest(expected_sha256)
    _url(url)
    if not callable(reader) or _sha(raw) != expected_sha256 or reader(url) != raw:
        raise OperatorError('published manifest differs from the pinned identity')
    value = _parse(raw, 65536)
    if (set(value) != {'schema', 'sources'}
            or value['schema'] != 'mlg-injection-comparison-source-manifest-v1'
            or type(value['sources']) is not dict or not 1 <= len(value['sources']) <= 64):
        raise OperatorError('the comparison source manifest is invalid')
    for name, entry in value['sources'].items():
        if (type(name) is not str or re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name) is None
                or type(entry) is not dict or set(entry) != {'sha256', 'url'}):
            raise OperatorError('a source manifest entry is invalid')
        _digest(entry['sha256'])
        _url(entry['url'])
    for name, digest, source_url in _PINNED_SOURCES:
        if value['sources'].get(name) != {'sha256': digest, 'url': source_url}:
            raise OperatorError('a reviewed source identity differs')
    return value


def verify_operator_manifest(raw, expected_sha256, url, operator_path, reader):
    """Check this executing source against the independently pinned manifest."""
    manifest = validate_source_manifest(raw, expected_sha256, url, reader)
    entry = manifest['sources'].get('operator')
    if type(entry) is not dict:
        raise OperatorError('the executing operator is not bound by the manifest')
    source = _read_source(operator_path, entry['sha256'])
    if reader(entry['url']) != source:
        raise OperatorError('published operator source differs')
    return True


def validate_public_binding(*, custody, comparison, operation, operation_sha256, operation_url,
                            freeze, commitment, commitment_sha256, commitment_url,
                            manifest, manifest_sha256, manifest_url, reader):
    """Check the fixed study, published operation and independently pinned manifest.

    The owner supplies the manifest identity it pinned before allocation and
    supplies a reader that fetches exact immutable bytes. This is a public-only
    point-in-time check; protected paths and source leases are separate.
    """
    if not callable(reader):
        raise OperatorError('an independent public readback is required')
    _digest(operation_sha256)
    _digest(commitment_sha256)
    _digest(manifest_sha256)
    _url(commitment_url)
    _url(manifest_url)
    _url(operation_url)
    if (_sha(operation) != operation_sha256 or _sha(freeze) != comparison._INJECTION_FREEZE
            or _sha(commitment) != commitment_sha256 or _sha(manifest) != manifest_sha256):
        raise OperatorError('a bound public digest differs')
    data = _parse(operation, 2 * 1024 * 1024)
    if (set(data) != custody._OP_V3_KEYS
            or data.get('schema') != 'mlg-injection-heldout-operational-binding-v3'):
        raise OperatorError('the held-out operation schema differs')
    fixed = {'protocol_sha256': _PROTOCOL, 'checkpoint_revision': _CHECKPOINT,
        'discovery_source_commit': _SOURCE, 'prediction_freeze_sha256': comparison._INJECTION_FREEZE,
        'runtime_sha256': _RUNTIME, 'prediction_sha256': _PREDICTION, 'plan_sha256': _PLAN,
        'site_commitment': comparison._SITE_COMMITMENT,
        'custody_source_sha256': comparison._SOURCE_SPECS[-1][1],
        'comparator_source_sha256': comparison._SOURCE_SPECS[-1][1]}
    if any(data.get(key) != value for key, value in fixed.items()):
        raise OperatorError('the operation does not bind the fixed study')
    if (type(data.get('inventory_source_commit')) is not str
            or _HEX40.fullmatch(data['inventory_source_commit']) is None):
        raise OperatorError('the inventory revision is invalid')
    for key, value in data.items():
        if key.endswith('_sha256') or key == 'site_commitment':
            _digest(value)
    frozen = _parse(freeze, 65536)
    expected_freeze = {'schema': 'mlg-number-prediction-freezes-v1', 'protocol_sha256': _PROTOCOL,
        'checkpoint_revision': _CHECKPOINT, 'tracks': {'injection': {'source_commit': _SOURCE,
        'runtime_sha256': _RUNTIME, 'prediction_sha256': _PREDICTION}}}
    if frozen != expected_freeze:
        raise OperatorError('the frozen prediction differs')
    try:
        public = comparison._public_commitment('injection', commitment, comparison._INJECTION_FREEZE)
    except (TypeError, ValueError, KeyError) as error:
        raise OperatorError('the injection commitment is invalid') from error
    if (public['operational_binding_sha256'] != operation_sha256
            or public['exposure_inventory_sha256'] != data['inventory_sha256']
            or public['custody_source_sha256'] != data['custody_source_sha256']):
        raise OperatorError('the commitment does not bind this operation')
    validate_source_manifest(manifest, manifest_sha256, manifest_url, reader)
    for url, raw in ((comparison._PUBLIC_URLS['injection_freeze'], freeze),
                     (commitment_url, commitment), (operation_url, operation)):
        if reader(url) != raw:
            raise OperatorError('published public bytes differ')
    return {'operation_sha256': operation_sha256, 'manifest_sha256': manifest_sha256,
            'injection_commitment_sha256': commitment_sha256}


def run_protected_comparison(*, modules, captured, operator_path, operator_config_path,
                             operator_config_sha256, config_path, config_sha256,
                             operation_path, operation_sha256, operation_url, manifest_path,
                             manifest_sha256, manifest_url, natural_parent, injection_parent,
                             site_parent, identities, terminal_path, inner_output, base_python,
                             environment, timeout, reader):
    """Run the bound comparison under protected file handles and a child Job.

    A source-first parent must first capture and execute the published modules,
    bind the manifest and identities before allocation, and independently
    observe this operator process's exit. This routine does not announce a
    forward, prove a clock, or convert a collision into a clear result.
    """
    required = {'comparison', 'custody', 'source_guards', 'replay_owner',
                'natural_query', 'adapter', 'exit', 'observer'}
    if (type(modules) is not dict or not required <= set(modules) or not callable(reader)
            or type(environment) is not dict or environment.get('CUDA_VISIBLE_DEVICES') != '-1'
            or type(timeout) is not int or not 1 <= timeout <= 7200):
        raise OperatorError('bound comparison process inputs are required')
    comparison, custody = modules['comparison'], modules['custody']
    guard, owner, adapter, exit_check = (modules[key] for key in
                                         ('source_guards', 'replay_owner', 'adapter', 'exit'))
    if (type(captured) is not tuple or len(captured) != len(_PINNED_SOURCES)
            or [item[0] for item in captured] != [item[0] for item in _PINNED_SOURCES]):
        raise OperatorError('the verified source set is required')
    operator_path, operator_config_path, config_path, operation_path, manifest_path, terminal_path, inner_output, base_python = map(
        Path, (operator_path, operator_config_path, config_path, operation_path, manifest_path,
               terminal_path, inner_output, base_python))
    if (operator_path.resolve(strict=True) != Path(__file__).resolve(strict=True)
            or not base_python.is_absolute()
            or base_python.resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)):
        raise OperatorError('the actual operator and direct-base Python are required')
    config_raw = comparison._read_file(config_path, 65536)
    operator_config_raw = comparison._read_file(operator_config_path, 65536)
    operation_raw = comparison._read_file(operation_path, 2 * 1024 * 1024)
    manifest_raw = comparison._read_file(manifest_path, 65536)
    if (_sha(config_raw) != _digest(config_sha256)
            or _sha(operator_config_raw) != _digest(operator_config_sha256)):
        raise OperatorError('comparison configuration differs')
    config = comparison.parse_config(config_raw)
    private = _private_paths(comparison, config, natural_parent, injection_parent, site_parent)
    receipt = Path(config['output'])
    output_paths = (receipt, terminal_path, inner_output)
    if (not operator_config_path.is_relative_to(private['injection_parent'])
            or operator_config_path.is_relative_to(private['injection_phase'])
            or any(not path.is_absolute() or '..' in path.parts
                   or not path.is_relative_to(private['injection_parent'])
                   or path.is_relative_to(private['injection_phase']) for path in output_paths)
            or len(set(output_paths)) != 3
            or any(path.is_relative_to(inner_output) for path in (receipt, terminal_path))):
        raise OperatorError('comparison evidence must use the bounded injection phase parent')
    freeze = comparison._read_file(config['injection']['freeze'], 65536)
    commitment = comparison._read_file(config['injection']['commitment'], 8192)
    verify_operator_manifest(manifest_raw, manifest_sha256, manifest_url, operator_path, reader)
    validate_public_binding(custody=custody, comparison=comparison, operation=operation_raw,
        operation_sha256=operation_sha256, operation_url=operation_url, freeze=freeze,
        commitment=commitment, commitment_sha256=config['injection_commitment_sha256'],
        commitment_url=config['injection_commitment_url'], manifest=manifest_raw,
        manifest_sha256=manifest_sha256, manifest_url=manifest_url, reader=reader)
    prefix = Path(config['pycache_prefix'])
    if not prefix.is_dir() or any(prefix.iterdir()):
        raise OperatorError('a fresh empty bytecode prefix is required')
    public_paths = {config_path: (config_raw, 65536), operation_path: (operation_raw, 2 * 1024 * 1024),
                    manifest_path: (manifest_raw, 65536), operator_config_path: (operator_config_raw, 65536),
                    Path(config['injection']['freeze']): (freeze, 65536),
                    Path(config['injection']['commitment']): (commitment, 8192)}
    for key, limit in (('freeze', 65536), ('commitment', 8192), ('inventory', 16 * 1024 * 1024)):
        path = Path(config['natural'][key])
        public_paths[path] = (comparison._read_file(path, limit), limit)
    source_entry = _parse(manifest_raw, 65536)['sources']['operator']
    operator_raw = _read_source(operator_path, source_entry['sha256'])
    all_files = {operator_path, *public_paths, *(item[3] for item in captured),
                 *(path for name, path in private.items() if name not in
                   ('natural_parent', 'natural_phase', 'injection_parent', 'injection_phase', 'site_parent'))}
    groups = {}
    for path in all_files:
        groups.setdefault(path.parent, []).append(path)
    leases = [guard.SourceFileLease(parent, sorted(files, key=str))
              for parent, files in sorted(groups.items(), key=lambda item: str(item[0]))]

    def source_check():
        try:
            if (_read_source(operator_path, source_entry['sha256']) != operator_raw
                    or any(comparison._read_file(path, limit) != raw
                           for path, (raw, limit) in public_paths.items())
                    or any(_read_source(path, expected) != raw
                           for (_, _, _, path, raw), (_, expected, _) in zip(captured, _PINNED_SOURCES))):
                return False
            return prefix.is_dir() and not any(prefix.iterdir())
        except (OSError, ValueError, TypeError):
            return False

    def precheck(_leases):
        if source_check() is not True:
            raise OperatorError('bound sources or public inputs changed')
        verify_operator_manifest(manifest_raw, manifest_sha256, manifest_url, operator_path, reader)
        validate_public_binding(custody=custody, comparison=comparison, operation=operation_raw,
            operation_sha256=operation_sha256, operation_url=operation_url, freeze=freeze,
            commitment=commitment, commitment_sha256=config['injection_commitment_sha256'],
            commitment_url=config['injection_commitment_url'], manifest=manifest_raw,
            manifest_sha256=manifest_sha256, manifest_url=manifest_url, reader=reader)
        return verify_private_inputs(comparison, config, acl=owner, natural_query=modules['natural_query'],
            natural_parent=natural_parent, injection_parent=injection_parent,
            site_parent=site_parent, identities=identities)

    command = [str(base_python), '-I', '-S', '-B', '-X', 'pycache_prefix=' + config['pycache_prefix'],
               '-c', adapter._BOOTSTRAP, str(dict((item[0], item[3]) for item in captured)['comparison']),
               dict((name, digest) for name, digest, _ in _PINNED_SOURCES)['comparison'], str(config_path)]
    command_sha256 = _sha(json.dumps(command, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                     separators=(',', ':')).encode('utf-8'))

    def launch(acquired):
        handles = tuple(handle for lease in acquired for handle in lease.handles())
        def protect_process(process):
            return len(guard.duplicate_to_process(handles, process)) == len(handles)
        options = {'comparison': comparison, 'custody': custody,
            'caller_source': dict((item[0], item[3]) for item in captured)['comparison'],
            'config_path': config_path, 'config_sha256': config_sha256,
            'operation_path': operation_path, 'operation_sha256': operation_sha256,
            'output': inner_output, 'base_python': base_python, 'environment': environment,
            'timeout': timeout, 'source_check': source_check, 'protect_process': protect_process,
            'reader': reader, 'observer': modules['observer'].observe_supervisor}
        checked = exit_check.run_operator(adapter=adapter, adapter_options=options, comparison=comparison,
            terminal_path=terminal_path, receipt_path=receipt, source_check=source_check,
            receipt_check=lambda raw: exit_check.verify_public_receipt(comparison=comparison,
                custody=custody, config=config, receipt=raw, reader=reader),
            comparator_command_sha256=command_sha256, operator_source_sha256=source_entry['sha256'])
        return {'cleanup_verified': True, 'result': checked}

    def postcheck(verdict):
        if source_check() is not True or verify_private_inputs(comparison, config, acl=owner,
                natural_query=modules['natural_query'], natural_parent=natural_parent,
                injection_parent=injection_parent, site_parent=site_parent, identities=identities) is not True:
            raise OperatorError('private paths or source inputs changed during comparison')
        return verdict['result']

    return owner.run_guarded(leases, precheck, launch, postcheck)


def main(argv=None, stream=None):
    """Start one gated operator after source and manifest verification."""
    try:
        start = sys.stdin.buffer if stream is None else stream
        if start.readline(len(_START) + 1) != _START:
            raise OperatorError('the operator start token is required')
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) != 5 or args[0] not in ('--validate-sources', '--run'):
            raise OperatorError('one bounded operator command is required')
        config_path, config_sha256, manifest_sha256, manifest_url = args[1:]
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
            raise OperatorError('the comparison process must be CPU-only')
        config_raw = _read_source(config_path, _digest(config_sha256))
        config = parse_operator_config(config_raw, manifest_sha256, manifest_url)
        _assert_isolated(config['pycache_prefix'])
        manifest = _read_source(config['manifest_path'], manifest_sha256)
        verify_operator_manifest(manifest, manifest_sha256, manifest_url, __file__, _fetch)
        captured = capture_sources(config['source_paths'], _fetch)
        modules = execute_sources(captured)
        if args[0] == '--validate-sources':
            print('comparison_operator_sources_validated', flush=True)
            return 0
        result = run_protected_comparison(modules=modules, captured=captured, operator_path=__file__,
            operator_config_path=config_path, operator_config_sha256=config_sha256,
            config_path=config['comparison_config_path'], config_sha256=config['comparison_config_sha256'],
            operation_path=config['operation_path'], operation_sha256=config['operation_sha256'],
            operation_url=config['operation_url'], manifest_path=config['manifest_path'],
            manifest_sha256=manifest_sha256, manifest_url=manifest_url,
            natural_parent=config['natural_parent'], injection_parent=config['injection_parent'],
            site_parent=config['site_parent'], identities=config['identities'],
            terminal_path=config['terminal_path'], inner_output=config['inner_output'],
            base_python=config['base_python'], environment=dict(os.environ), timeout=config['timeout'],
            reader=_fetch)
        if type(result) is not dict or result.get('status') not in ('clear', 'collision'):
            raise OperatorError('the comparison operator result is incomplete')
        print('comparison_operator_completed', flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
