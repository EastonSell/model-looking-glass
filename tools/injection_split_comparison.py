"""Compare two committed held-out splits without publishing their contents.

The operator must bind this source, the protected paths and process exit, and
serialize the later forward decision with the shared gate. A saved receipt is
a trusted-custodian observation, not a proof of execution or chronology.
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


_HEX = re.compile(r'[0-9a-f]{64}\Z')
_SPLIT_KEYS = frozenset({'freeze', 'commitment', 'inventory', 'nonce', 'record'})
_EXPECTED_KEYS = frozenset({'natural_freeze_sha256', 'injection_freeze_sha256',
                            'natural_commitment_sha256', 'injection_commitment_sha256',
                            'comparator_source_sha256'})
_COMMITMENT_KEYS = {
    'natural': frozenset({'schema', 'prediction_freeze_sha256', 'exposure_inventory_sha256',
                          'heldout_commitment', 'wrapper_sha256', 'deviation_sha256',
                          'operational_binding_sha256'}),
    'injection': frozenset({'schema', 'prediction_freeze_sha256', 'exposure_inventory_sha256',
                            'heldout_commitment', 'custody_source_sha256',
                            'operational_binding_sha256'}),
}
_SOURCE_SPECS = (
    ('pretrained_exposures', '6ec5a00a6950a739ddfab1c0335065609aead275f0f393323ecb22699d93a9cd', '02e1c5d5c79813e3d67bbd0468b599926b4b9f83'),
    ('pretrained_exposures_v2', 'fa9014fb5cce2c3d8be3497a561754e47115ec07b9be3a896bb3bf1b20f8c474', '02e1c5d5c79813e3d67bbd0468b599926b4b9f83'),
    ('pretrained_exposures_v3', '2546bb672a461bfea10f0aed5e83c3a2110b2c898b56ff17542cb4718fc32886', '02e1c5d5c79813e3d67bbd0468b599926b4b9f83'),
    ('pretrained_number_commitments', '635f25553c20dd437f5cc4a8a9ca3a7a1453e7f8a517875fe75eb01ede2d5f35', '02e1c5d5c79813e3d67bbd0468b599926b4b9f83'),
    ('injection_split_custody', '964d90b3674671e00fe9f8d72c67eb124d0afde2bbb499629c267e79fd5b27ff', 'b3cf11f5d38e76db70d116c414d0944f725c1780'),
)
_NATURAL_FREEZE = '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb'
_NATURAL_COMMITMENT = '7f4f96fee2fd90bc5efbf8b85bb722021b3b698905256dc9f2e2e626d4a4f4ba'
_INJECTION_FREEZE = 'c327e191214cfb661b429c6d0a66900ca773c148782e1cc88be669c54deaee98'
_SITE_COMMITMENT = 'c3739cb173d3c1bc3e3adda70025ff7f26a73c4690e50ecf5baf06ff8b359b9d'
_RAW = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
_PUBLIC_URLS = {
    'natural_freeze': _RAW + 'c8d37acc064ecb80d522fcdd0c98aea7a1d78f87/results/distilgpt2-natural-agreement-v1-20261028/natural-prediction-freeze.json',
    'natural_commitment': _RAW + '4d4c29683d58ea796c9bd14656d2e637bdfe0913/results/distilgpt2-natural-agreement-v1-20261028/heldout-commitment-v1.json',
    'injection_freeze': _RAW + 'afbdedb6fec40fa56252f67c1eaf4fbd8ea4a1db/results/pretrained-number-injection-discovery-v1-20261028/injection-prediction-freeze.json',
}
_START = b'MLG-natural-discovery-start-v1\n'


class ComparisonError(ValueError):
    """The comparison has not established a usable result."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ComparisonError('a lowercase SHA-256 digest is required')
    return value


def _ordinary(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ComparisonError('an absolute non-traversing path is required')
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ComparisonError('a path crosses a link or reparse point')
    return path


def _read_file(path, limit):
    path = _ordinary(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or not 0 < before.st_size <= limit):
        raise ComparisonError('a bounded regular input is required')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(limit + 1)
        finished = os.fstat(stream.fileno())
    after = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_nlink)
    if (len(raw) != before.st_size or identity(before) != identity(opened)
            or identity(before) != identity(finished) or identity(before) != identity(after)):
        raise ComparisonError('input changed while being read')
    return raw


def _fetch(url, limit=1024 * 1024):
    request = Request(url, headers={'User-Agent': 'looking-glass-cpu-study'})
    with urlopen(request, timeout=20) as response:
        if response.geturl() != url:
            raise ComparisonError('published bytes redirected')
        raw = response.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ComparisonError('published bytes exceed the bound')
    return raw


def _remote_source(name, commit):
    return _fetch(_RAW + commit + '/looking_glass/' + name + '.py')


def load_sources(source_root, custody_source, *, reader=_remote_source):
    """Verify all source and public bytes before executing any project code."""
    root = _ordinary(source_root)
    captured = []
    for name, expected, commit in _SOURCE_SPECS:
        path = Path(custody_source) if name == 'injection_split_custody' else root / 'looking_glass' / (name + '.py')
        raw = _read_file(path, 1024 * 1024)
        remote = reader(name, commit)
        if type(remote) is not bytes or raw != remote or _sha(raw) != expected:
            raise ComparisonError('source differs from published bytes')
        captured.append((name, path, raw))
    if any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules):
        raise ComparisonError('project code was loaded before source verification')
    package = ModuleType('looking_glass')
    package.__package__ = 'looking_glass'
    package.__path__ = []
    names = ['looking_glass']
    sys.modules['looking_glass'] = package
    try:
        for name, path, raw in captured:
            qualified = 'looking_glass.' + name
            module = ModuleType(qualified)
            module.__file__ = str(path)
            module.__package__ = 'looking_glass'
            sys.modules[qualified] = module
            names.append(qualified)
            exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
        return sys.modules['looking_glass.injection_split_custody']
    except BaseException:
        for name in reversed(names):
            sys.modules.pop(name, None)
        raise


def _public_commitment(track, raw, freeze_sha):
    try:
        value = json.loads(raw)
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode('utf-8')
        if (type(value) is not dict or set(value) != _COMMITMENT_KEYS[track]
                or canonical != raw or value['schema'] != 'mlg-' + track + '-heldout-public-commitment-v1'
                or value['prediction_freeze_sha256'] != freeze_sha):
            raise ComparisonError('public commitment does not match its frozen track')
        for key in _COMMITMENT_KEYS[track] - {'schema'}:
            _digest(value[key])
        return value
    except (KeyError, TypeError, UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ComparisonError('public commitment is invalid') from error


def compare_files(custody, *, natural, injection, site_nonce, site_record, site_commitment,
                  expected, output, publication_check):
    """Verify public identities before reading the two private records.

    The generic entry accepts caller-supplied hashes for fabricated tests. A
    real study entry must independently pin the published study identities,
    source, private paths and publication observer before calling it.
    """
    if (type(natural) is not dict or set(natural) != _SPLIT_KEYS
            or type(injection) is not dict or set(injection) != _SPLIT_KEYS
            or type(expected) is not dict or set(expected) != _EXPECTED_KEYS
            or not callable(publication_check)):
        raise ComparisonError('bound split inputs are required')
    for value in expected.values():
        _digest(value)
    _digest(site_commitment)
    captured = {}
    public = {}
    for track, files in (('natural', natural), ('injection', injection)):
        for field in ('freeze', 'commitment'):
            key = track + '_' + field
            raw = _read_file(files[field], 1024 * 1024)
            if _sha(raw) != expected[key + '_sha256']:
                raise ComparisonError('public bytes do not match their bound digest')
            if field == 'commitment':
                public[track] = _public_commitment(track, raw, expected[track + '_freeze_sha256'])
            if publication_check(key, raw) is not True:
                raise ComparisonError('public publication readback failed')
            captured[key] = raw

    path = Path(output)
    if not path.is_absolute() or '..' in path.parts:
        raise ComparisonError('an absolute receipt path is required')
    _ordinary(path.parent)
    try:
        stream = path.open('xb')
    except FileExistsError as error:
        raise ComparisonError('receipt already exists') from error
    with stream:
        def build(track, files):
            inventory = _read_file(files['inventory'], 16 * 1024 * 1024)
            inventory_sha = public[track]['exposure_inventory_sha256']
            if _sha(inventory) != inventory_sha:
                raise ComparisonError('inventory differs from the public commitment')
            nonce = _read_file(files['nonce'], 32)
            if len(nonce) != 32:
                raise ComparisonError('a complete private nonce is required')
            return dict(track=track, freeze_bytes=captured[track + '_freeze'],
                        freeze_sha256=expected[track + '_freeze_sha256'], nonce=nonce,
                        record_bytes=_read_file(files['record'], 1024 * 1024),
                        commitment_bytes=captured[track + '_commitment'],
                        inventory_bytes=inventory, inventory_sha256=inventory_sha)

        n = custody.CommittedSplit(**build('natural', natural))
        i = build('injection', injection)
        i['site_nonce'] = _read_file(site_nonce, 32)
        if len(i['site_nonce']) != 32:
            raise ComparisonError('a complete private site nonce is required')
        i['site_record_bytes'] = _read_file(site_record, 1024 * 1024)
        i['site_commitment'] = site_commitment
        result = custody.compare_committed_splits(
            n, custody.CommittedSplit(**i),
            comparator_source_sha256=expected['comparator_source_sha256'],
            publication_check=lambda name, raw: captured.get(name) == raw)
        if type(result) is not bytes or not 0 < len(result) <= 8192:
            raise ComparisonError('bounded comparison receipt is required')
        status = json.loads(result).get('status')
        if status not in ('clear', 'collision'):
            raise ComparisonError('comparison did not produce a valid status')
        if stream.write(result) != len(result):
            raise ComparisonError('receipt write was incomplete')
        stream.flush()
        os.fsync(stream.fileno())
    if _read_file(path, 8192) != result:
        raise ComparisonError('receipt readback differs')
    return {'status': status, 'receipt_sha256': _sha(result)}


def parse_config(raw):
    """Accept only the protected paths and immutable publication identity."""
    keys = {'schema', 'source_root', 'custody_source', 'pycache_prefix', 'output',
            'site_nonce', 'site_record', 'natural', 'injection',
            'injection_commitment_sha256', 'injection_commitment_url'}
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise ComparisonError('a bounded configuration is required')
    try:
        value = json.loads(raw)
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode('utf-8')
        if (type(value) is not dict or set(value) != keys or canonical != raw
                or value['schema'] != 'mlg-injection-split-comparison-config-v1'
                or any(type(value[track]) is not dict or set(value[track]) != _SPLIT_KEYS
                       for track in ('natural', 'injection'))):
            raise ComparisonError('the comparison configuration differs')
        paths = [value[key] for key in ('source_root', 'custody_source', 'pycache_prefix',
                                        'output', 'site_nonce', 'site_record')]
        paths.extend(value[track][field] for track in ('natural', 'injection') for field in _SPLIT_KEYS)
        if any(type(path) is not str or not path or not Path(path).is_absolute()
               or '..' in Path(path).parts for path in paths):
            raise ComparisonError('absolute non-traversing paths are required')
        _digest(value['injection_commitment_sha256'])
        url = value['injection_commitment_url']
        pattern = re.escape(_RAW) + r'[0-9a-f]{40}/[A-Za-z0-9._/-]+'
        if (type(url) is not str or re.fullmatch(pattern, url) is None
                or any(part in ('', '.', '..') for part in url[len(_RAW):].split('/'))):
            raise ComparisonError('an immutable repository commitment URL is required')
        return value
    except (KeyError, TypeError, UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ComparisonError('the comparison configuration is invalid') from error


def compare_study(custody, config, *, reader=_fetch):
    """Compare the original natural split to the frozen injection track."""
    if type(config) is not dict:
        raise ComparisonError('a bound study configuration is required')
    try:
        raw = json.dumps(config, sort_keys=True, ensure_ascii=True, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, RecursionError) as error:
        raise ComparisonError('the comparison configuration is invalid') from error
    config = parse_config(raw)
    urls = dict(_PUBLIC_URLS, injection_commitment=config['injection_commitment_url'])
    expected = {'natural_freeze_sha256': _NATURAL_FREEZE,
                'natural_commitment_sha256': _NATURAL_COMMITMENT,
                'injection_freeze_sha256': _INJECTION_FREEZE,
                'injection_commitment_sha256': config['injection_commitment_sha256'],
                'comparator_source_sha256': _SOURCE_SPECS[-1][1]}
    return compare_files(custody, natural=config['natural'], injection=config['injection'],
                         site_nonce=config['site_nonce'], site_record=config['site_record'],
                         site_commitment=_SITE_COMMITMENT, expected=expected, output=config['output'],
                         publication_check=lambda name, raw: reader(urls[name]) == raw)


def _assert_isolated(prefix):
    flags = sys.flags
    if (not (flags.isolated and flags.ignore_environment and flags.no_site and flags.no_user_site
             and flags.safe_path and sys.dont_write_bytecode and flags.optimize == 0)
            or not sys.pycache_prefix or Path(sys.pycache_prefix) != Path(prefix)
            or Path(sys.executable).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)
            or any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules)
            or sys.meta_path != [importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                                 importlib.machinery.PathFinder]):
        raise ComparisonError('an isolated source-only interpreter is required')
    path = _ordinary(prefix)
    if not path.is_dir() or any(path.iterdir()):
        raise ComparisonError('a fresh empty bytecode prefix is required')


def main(argv=None, stream=None):
    try:
        start = sys.stdin.buffer if stream is None else stream
        if start.readline(len(_START) + 1) != _START:
            raise ComparisonError('the supervisor start token is required')
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) != 2 or args[0] not in ('--validate-sources', '--compare'):
            raise ComparisonError('a single comparison command is required')
        config = parse_config(_read_file(args[1], 65536))
        _assert_isolated(config['pycache_prefix'])
        custody = load_sources(config['source_root'], config['custody_source'])
        if args[0] == '--validate-sources':
            print('comparison_sources_validated', flush=True)
        else:
            result = compare_study(custody, config)
            print('comparison:' + result['status'] + ':' + result['receipt_sha256'], flush=True)
        return 0
    except KeyboardInterrupt:
        print('comparison_interrupted', file=sys.stderr, flush=True)
        return 130
    except Exception:
        print('comparison_incomplete', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
