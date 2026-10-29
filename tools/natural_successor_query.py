"""Source-bound, one-use private pool comparator caller.

The operation must be independently published/reviewed, its rehearsal verifier
must establish the actual production lineage, and this caller must itself be
started via a separately observed hash-before-execution bootstrap. It keeps a
query claim once that claim is created. Earlier failures may leave no marker:
the independently bound outer caller must enforce the protocol's one-attempt
stop for those failures. An ambiguous child or caller outcome requires external
process reconciliation and never authorizes a retry. This module does not
create a phase claim, draw a nonce or load a model.
"""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import sys
import threading
import types
import urllib.request


class QueryError(RuntimeError):
    pass


class QueryCollision(QueryError):
    """The fixed pool collides; the consumed query must not be repeated."""


_SHA = re.compile(r'[0-9a-f]{64}\Z')
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_ROLES = frozenset({'caller', 'claim', 'accounting', 'sampler', 'rehearsal_verifier', 'comparator',
                    'supervisor', 'controller', 'guards', 'owner_wrapper', 'acl_verifier',
                    'protocol', 'clarification', 'inputs', 'inventory_source'})
_ACTIVE_OWNER = []
_HOLD = threading.Event()
_BOOTSTRAP = '''import ctypes,hashlib,msvcrt,os,sys,time
from ctypes import wintypes
if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or sys.flags.optimize:
    raise SystemExit(1)
p=sys.argv.pop(1); h=sys.argv.pop(1)
k=ctypes.WinDLL("kernel32",use_last_error=True)
k.PeekNamedPipe.argtypes=(wintypes.HANDLE,wintypes.LPVOID,wintypes.DWORD,wintypes.LPVOID,ctypes.POINTER(wintypes.DWORD),wintypes.LPVOID)
k.PeekNamedPipe.restype=wintypes.BOOL
fd=sys.stdin.fileno(); handle=msvcrt.get_osfhandle(fd); token=b"MLG-natural-discovery-start-v1\\n"; data=bytearray()
deadline=time.monotonic()+10
while time.monotonic()<deadline and len(data)<len(token):
    available=wintypes.DWORD()
    if not k.PeekNamedPipe(handle,None,0,None,ctypes.byref(available),None): break
    if available.value: data.extend(os.read(fd,min(available.value,len(token)-len(data))))
    else: time.sleep(.01)
if bytes(data)!=token: raise SystemExit(1)
raw=open(p,'rb').read(1000001)
if len(raw)>1000000 or hashlib.sha256(raw).hexdigest()!=h:
    raise SystemExit(1)
sys.argv[0]=p
exec(compile(raw,p,'exec',dont_inherit=True),{'__name__':'__main__','__file__':p})
'''


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise QueryError('a bound SHA-256 digest is required')
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QueryError('duplicate JSON key')
        result[key] = value
    return result


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode('ascii')
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise QueryError('invalid canonical JSON') from error
    if canonical != raw:
        raise QueryError('JSON is not canonical')
    return value


def _encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def _read(path, maximum, *, allow_empty=False):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise QueryError('absolute paths are required')
    for item in (path, *path.parents):
        part = item.lstat()
        if stat.S_ISLNK(part.st_mode) or getattr(part, 'st_file_attributes', 0) & 1024:
            raise QueryError('a bound path crosses a reparse point')
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1
            or getattr(info, 'st_file_attributes', 0) & 1024
            or not (0 if allow_empty else 1) <= info.st_size <= maximum):
        raise QueryError('a bound input is unsafe or oversized')
    raw = path.read_bytes()
    after = path.lstat()
    identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_nlink)
    if len(raw) != info.st_size or identity(info) != identity(after):
        raise QueryError('a bound input changed while reading')
    return raw


def _url(value):
    if (type(value) is not str or re.fullmatch(
            r'https://raw\.githubusercontent\.com/EastonSell/model-looking-glass/'
            r'[0-9a-f]{40}/[A-Za-z0-9_./-]+', value) is None or '..' in value.split('/')):
        raise QueryError('an immutable official source URL is required')
    return value


def _read_public(url):
    _url(url)
    with urllib.request.urlopen(url, timeout=20) as response:
        if response.geturl() != url:
            raise QueryError('public source redirected')
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise QueryError('public source is oversized')
    return data


def _api(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'MLG-natural-successor-query',
                                                    'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.geturl() != url:
            raise QueryError('official-main observation redirected')
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise QueryError('official-main observation is oversized')
    return json.loads(data)


def _observe_current(operation):
    api = 'https://api.github.com/repos/EastonSell/model-looking-glass'
    head_value = _api(api + '/commits/main')
    head = head_value.get('sha') if type(head_value) is dict else None
    if type(head) is not str or _COMMIT.fullmatch(head) is None:
        raise QueryError('official main is unavailable')
    listing = _api(api + '/contents/looking_glass?ref=' + head)
    if type(listing) is not list or len(listing) >= 1000:
        raise QueryError('official exposure listing is unavailable')
    versions = []
    for entry in listing:
        name = entry.get('name') if type(entry) is dict else None
        match = re.fullmatch(r'pretrained_exposures_v([1-9][0-9]*)\.py', name or '')
        if match:
            versions.append(int(match.group(1)))
    version = operation['inventory_version']
    if not versions or max(versions) != version:
        raise QueryError('the official exposure version differs')
    url = (f'https://raw.githubusercontent.com/EastonSell/model-looking-glass/{head}/'
           f'looking_glass/pretrained_exposures_v{version}.py')
    return {'head': head, 'version': version, 'source_sha256': _digest(_read_public(url)),
            'inventory_sha256': operation['inventory_sha256']}


def _load(name, raw, path):
    module = types.ModuleType('_natural_successor_' + name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def _project_modules(captured, paths):
    """Install only already-captured sources for the duration of this process."""
    names = [name for name in sys.modules if name == 'looking_glass' or name.startswith('looking_glass.')
             or name == 'scripts' or name.startswith('scripts.')]
    prior = {name: sys.modules.pop(name) for name in names}
    for package in ('looking_glass', 'scripts'):
        module = types.ModuleType(package)
        module.__path__ = []
        module.__package__ = package
        sys.modules[package] = module
    try:
        for role, name in (('controller', 'looking_glass.natural_agreement_controller'),
                           ('guards', 'scripts.v2_source_guards'),
                           ('supervisor', 'looking_glass.natural_heldout_supervisor')):
            module = types.ModuleType(name)
            module.__file__ = str(paths[role])
            module.__package__ = name.rpartition('.')[0]
            sys.modules[name] = module
            exec(compile(captured[role], str(paths[role]), 'exec', dont_inherit=True), module.__dict__)
        return prior, sys.modules['scripts.v2_source_guards'], sys.modules['looking_glass.natural_heldout_supervisor']
    except BaseException:
        _restore_modules(prior)
        raise


def _restore_modules(prior):
    for name in list(sys.modules):
        if name == 'looking_glass' or name.startswith('looking_glass.') or name == 'scripts' or name.startswith('scripts.'):
            del sys.modules[name]
    sys.modules.update(prior)


def _write_new(path, raw):
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('incomplete observation write')
        stream.flush()
        os.fsync(stream.fileno())


def _current(operation, observe):
    state = observe(operation)
    if (type(state) is not dict or type(state.get('head')) is not str
            or _COMMIT.fullmatch(state['head']) is None
            or type(state.get('version')) is not int or state['version'] != operation['inventory_version']
            or state.get('source_sha256') != operation['inventory_source_sha256']
            or state.get('inventory_sha256') != operation['inventory_sha256']):
        raise QueryError('current exposure source is not the bound version')
    return state


def _verify_inherited_private(acl, path, identity, *, directory=True):
    """Check an old directory or file's inherited private ACL and identity."""
    before = path.lstat()
    if ((not stat.S_ISDIR(before.st_mode) if directory else not stat.S_ISREG(before.st_mode))
            or stat.S_ISLNK(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 1024
            or (not directory and before.st_nlink != 1)
            or (before.st_dev, before.st_ino) != identity):
        raise QueryError('old protected path identity differs')
    snapshot = acl._windows_acl_snapshot(path)
    if (type(snapshot) is not dict or set(snapshot) != {'user', 'owner', 'protected', 'rules'}
            or type(snapshot['user']) is not str
            or re.fullmatch(r'S-1-(?:[0-9]+-){2,14}[0-9]+', snapshot['user']) is None
            or snapshot['user'] == 'S-1-5-18' or snapshot['owner'] != snapshot['user']
            or snapshot['protected'] is not False or type(snapshot['rules']) is not list
            or len(snapshot['rules']) != 2):
        raise QueryError('old protected child ACL differs')
    principals = set()
    for rule in snapshot['rules']:
        fixed = {'type': 0, 'inheritance': 3 if directory else 0, 'propagation': 0,
                 'inherited': True, 'rights': 0x1f01ff}
        if (type(rule) is not dict or set(rule) != set(fixed) | {'sid'} or type(rule['sid']) is not str
                or any(type(rule.get(key)) is not type(value) or rule[key] != value
                       for key, value in fixed.items())):
            raise QueryError('old protected child ACL differs')
        principals.add(rule['sid'])
    if principals != {snapshot['user'], 'S-1-5-18'}:
        raise QueryError('old protected child principals differ')
    after = path.lstat()
    if (after.st_dev, after.st_ino) != identity:
        raise QueryError('old protected path identity changed')


def run_query(config_path, *, config_sha256, operation_sha256,
              reader=_read_public, observe_current=_observe_current):
    """Run the fixed query once after bound public and rehearsal checks.

    The verifier role must independently validate the actual owner rehearsal,
    source and process observations. A verifier merely returning a status is
    not sufficient unless that exact verifier has been reviewed and published.
    An outer observer must retain and reap this caller before treating its
    observation as evidence.
    """
    config_raw = _read(config_path, 65536)
    if not hmac.compare_digest(_digest(config_raw), _hex(config_sha256)):
        raise QueryError('query configuration differs')
    config = _json(config_raw)
    keys = {'schema', 'root', 'cache', 'old_private_parent', 'old_phase', 'old_supervisor',
            'inputs', 'pins', 'inventory', 'public_freeze', 'operation', 'operation_url', 'sources', 'rehearsal'}
    if (type(config) is not dict or set(config) != keys
            or config['schema'] != 'mlg-natural-successor-query-config-v1'
            or type(config['sources']) is not dict or set(config['sources']) != _ROLES):
        raise QueryError('query configuration schema differs')
    operation_raw = _read(config['operation'], 1000000)
    if (not hmac.compare_digest(_digest(operation_raw), _hex(operation_sha256))
            or reader(_url(config['operation_url'])) != operation_raw):
        raise QueryError('published query operation differs')
    operation = _json(operation_raw)
    op_keys = {'schema', 'sources', 'root_path_sha256', 'root_device', 'root_inode', 'pool_input_sha256',
               'pool_sha256', 'tokenizer_sha256', 'freeze_sha256', 'old_commitment', 'pins_sha256',
               'inventory_sha256', 'inventory_version', 'inventory_source_sha256',
               'protocol_sha256', 'query_clarification_sha256', 'freeze_url', 'old_commitment_url',
               'old_inventory_sha256', 'old_source_commit', 'old_commitment_file_sha256',
               'old_supervisor_sha256', 'old_observation_sha256', 'old_parent_device', 'old_parent_inode',
               'old_phase_device', 'old_phase_inode', 'old_supervisor_device', 'old_supervisor_inode'}
    if (type(operation) is not dict or set(operation) != op_keys
            or operation['schema'] != 'mlg-natural-successor-query-operation-v1'
            or type(operation['sources']) is not dict or set(operation['sources']) != _ROLES
            or type(operation['inventory_version']) is not int or operation['inventory_version'] < 7):
        raise QueryError('query operation schema differs')
    for key in ('root_path_sha256', 'pool_input_sha256', 'pool_sha256', 'tokenizer_sha256',
                'freeze_sha256', 'old_commitment', 'pins_sha256', 'inventory_sha256', 'inventory_source_sha256',
                'protocol_sha256', 'query_clarification_sha256', 'old_inventory_sha256',
                'old_commitment_file_sha256', 'old_supervisor_sha256', 'old_observation_sha256'):
        _hex(operation[key])
    if type(operation['old_source_commit']) is not str or _COMMIT.fullmatch(operation['old_source_commit']) is None:
        raise QueryError('old source commit identity differs')
    for stem in ('root', 'old_parent', 'old_phase', 'old_supervisor'):
        if (type(operation[stem + '_device']) is not int or operation[stem + '_device'] < 0
                or type(operation[stem + '_inode']) is not int or operation[stem + '_inode'] <= 0):
            raise QueryError('protected directory identity is invalid')
    captured = {}
    for role in sorted(_ROLES):
        entry = operation['sources'][role]
        if type(entry) is not dict or set(entry) != {'sha256', 'url'}:
            raise QueryError('query source binding differs')
        raw = _read(config['sources'][role], 1000000)
        if (_digest(raw) != _hex(entry['sha256']) or reader(_url(entry['url'])) != raw):
            raise QueryError('query source bytes differ')
        captured[role] = raw
    if _read(Path(__file__).resolve(), 1000000) != captured['caller']:
        raise QueryError('caller source differs')
    if (_digest(captured['protocol']) != operation['protocol_sha256']
            or _digest(captured['clarification']) != operation['query_clarification_sha256']
            or _digest(captured['inputs']) != operation['pool_input_sha256']
            or _digest(captured['inventory_source']) != operation['inventory_source_sha256']):
        raise QueryError('reviewed public bindings differ')
    claim, accounting, sampler, rehearsal, acl, owner = (
        _load(role, captured[role], config['sources'][role]) for role in
        ('claim', 'accounting', 'sampler', 'rehearsal_verifier', 'acl_verifier', 'owner_wrapper'))
    root = accounting._directory(config['root'])
    cache = accounting._directory(config['cache'])
    old_parent = accounting._directory(config['old_private_parent'])
    old_phase = accounting._directory(config['old_phase'])
    old_supervisor = accounting._directory(config['old_supervisor'])
    if old_phase.parent != old_parent or old_supervisor.parent != old_parent:
        raise QueryError('old protected directories have unexpected parents')
    def overlaps(first, second):
        return first == second or first in second.parents or second in first.parents
    if overlaps(root, cache) or overlaps(root, old_parent) or overlaps(cache, old_parent):
        raise QueryError('successor, cache and prior parent roots overlap')
    private_files = (old_phase / 'nonce.bin', old_phase / 'heldout-record.json', old_phase / 'commitment.json',
                     old_supervisor / 'supervisor.json', old_supervisor / 'supervisor-observation.json')
    private_file_identities = {}
    for path in private_files:
        info = path.lstat()
        private_file_identities[path] = (info.st_dev, info.st_ino)
    def private_identities():
        if (acl.verify_private_parent(root, (operation['root_device'], operation['root_inode'])) is not True
                or acl.verify_private_parent(old_parent, (operation['old_parent_device'], operation['old_parent_inode'])) is not True):
            raise QueryError('private parent ACL is unverified')
        for path, stem in ((old_phase, 'old_phase'), (old_supervisor, 'old_supervisor')):
            _verify_inherited_private(acl, path, (operation[stem + '_device'], operation[stem + '_inode']))
        for path, identity in private_file_identities.items():
            _verify_inherited_private(acl, path, identity, directory=False)
    private_identities()
    input_raw, inventory_raw, pins_raw = (_read(config[key], 1000000) for key in ('inputs', 'inventory', 'pins'))
    if (input_raw != captured['inputs'] or _digest(input_raw) != operation['pool_input_sha256']
            or _digest(inventory_raw) != operation['inventory_sha256']
            or _digest(pins_raw) != operation['pins_sha256']):
        raise QueryError('query input or exposure identity differs')
    freeze_raw = _read(config['public_freeze'], 1000000)
    old_commit_raw = _read(Path(config['old_phase']) / 'commitment.json', 8192)
    if (_digest(freeze_raw) != operation['freeze_sha256']
            or reader(_url(operation['freeze_url'])) != freeze_raw
            or _digest(old_commit_raw) != operation['old_commitment_file_sha256']
            or reader(_url(operation['old_commitment_url'])) != old_commit_raw):
        raise QueryError('published old freeze or commitment differs')
    old_public = _json(old_commit_raw)
    if (type(old_public) is not dict or old_public.get('heldout_commitment') != operation['old_commitment']
            or old_public.get('prediction_freeze_sha256') != operation['freeze_sha256']
            or old_public.get('exposure_inventory_sha256') != operation['old_inventory_sha256']):
        raise QueryError('old public commitment identity differs')
    public, inventory, pins = map(_json, (input_raw, inventory_raw, pins_raw))
    if (type(public) is not dict or set(public) != {'schema', 'pool_sha256', 'tokenizer_sha256', 'cases'}
            or public['schema'] != 'mlg-natural-successor-candidate-inputs-v1'
            or public['pool_sha256'] != operation['pool_sha256']
            or public['tokenizer_sha256'] != operation['tokenizer_sha256']
            or type(public['cases']) is not list or len(public['cases']) != 256
            or any(type(row) is not dict or set(row) != {'input_ids'} for row in public['cases'])
            or type(inventory) is not dict
            or inventory.get('version') != f"distilgpt2-development-exposures-v{operation['inventory_version']}"
            or type(inventory.get('records')) is not list):
        raise QueryError('query public inputs differ')
    rows = [row['input_ids'] for row in public['cases']]
    if (any(type(row) is not list or not row or len(row) > 32
            or any(type(token) is not int or not 0 <= token <= 50256 for token in row) for row in rows)
            or len({tuple(row) for row in rows}) != 256):
        raise QueryError('query candidate rows are invalid')
    pin_keys = {'freeze', 'inventory', 'source_commit', 'proposal', 'commitment', 'supervisor',
                'observation', 'private_count', 'public_count', 'comparator_source'}
    if (type(pins) is not dict or set(pins) != pin_keys or pins['freeze'] != operation['freeze_sha256']
            or pins['proposal'] != operation['pool_input_sha256'] or pins['inventory'] != operation['old_inventory_sha256']
            or pins['source_commit'] != operation['old_source_commit']
            or pins['commitment'] != operation['old_commitment_file_sha256']
            or pins['supervisor'] != operation['old_supervisor_sha256']
            or pins['observation'] != operation['old_observation_sha256']
            or pins['comparator_source'] != operation['sources']['comparator']['sha256']
            or type(pins['private_count']) is not int or pins['private_count'] != 16
            or type(pins['public_count']) is not int or pins['public_count'] != 256):
        raise QueryError('comparator pins differ')
    def require_fresh_query():
        if any(os.path.lexists(root / name) for name in
               ('pool-query-claim.json', 'pool-query-receipt.json', 'pool-query-observation.json',
                'pool-query-supervisor', 'phase-claim.json', 'nonce.bin', 'heldout-record.json')):
            raise QueryError('query or phase is already started or uncertain')
    require_fresh_query()
    sampler.reject_known_rows(rows, inventory['records'])
    def unchanged():
        if (_read(config_path, 65536) != config_raw or _read(config['operation'], 1000000) != operation_raw
                or _read(config['inputs'], 1000000) != input_raw or _read(config['inventory'], 1000000) != inventory_raw
                or _read(config['pins'], 1000000) != pins_raw
                or _read(config['public_freeze'], 1000000) != freeze_raw
                or _read(old_phase / 'commitment.json', 8192) != old_commit_raw
                or _digest(_read(old_supervisor / 'supervisor.json', 1000000)) != operation['old_supervisor_sha256']
                or _digest(_read(old_supervisor / 'supervisor-observation.json', 8192)) != operation['old_observation_sha256']
                or any(_read(config['sources'][role], 1000000) != captured[role] for role in _ROLES)):
            raise QueryError('query inputs changed during custody')
    receipt_path = root / 'pool-query-receipt.json'
    command = [sys._base_executable, '-I', '-S', '-B', '-c', _BOOTSTRAP,
               config['sources']['comparator'], operation['sources']['comparator']['sha256'],
               '--phase', str(old_phase), '--supervisor', str(old_supervisor), '--proposal', config['inputs'],
               '--pins', config['pins'], '--pins-sha256', operation['pins_sha256'], '--receipt', str(receipt_path)]
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = '-1'
    receipt_keys = {'schema', 'status', 'comparison', 'natural_heldout_commitment',
                    'natural_prediction_freeze_sha256', 'public_input_file_sha256',
                    'natural_case_count', 'public_case_count', 'comparator_source_sha256', 'observed_utc'}
    supervisor_output = root / 'pool-query-supervisor'
    state = {}

    def check_receipt():
        stdout = _read(supervisor_output / 'stdout.txt', 1024 * 1024, allow_empty=True)
        stderr = _read(supervisor_output / 'stderr.txt', 1024 * 1024, allow_empty=True)
        if stderr or stdout not in (b'clear\n', b'clear\r\n', b'collision\n', b'collision\r\n'):
            raise QueryError('query output differs')
        raw = _read(receipt_path, 8192)
        receipt = _json(raw)
        if (type(receipt) is not dict or set(receipt) != receipt_keys
                or receipt['schema'] != 'mlg-natural-external-input-comparison-v1'
                or receipt['status'] != stdout.strip().decode('ascii')
                or receipt['comparison'] != 'bidirectional-complete-row-prefix'
                or receipt['natural_heldout_commitment'] != operation['old_commitment']
                or receipt['natural_prediction_freeze_sha256'] != operation['freeze_sha256']
                or receipt['public_input_file_sha256'] != operation['pool_input_sha256']
                or type(receipt['natural_case_count']) is not int or receipt['natural_case_count'] != 16
                or type(receipt['public_case_count']) is not int or receipt['public_case_count'] != 256
                or receipt['comparator_source_sha256'] != operation['sources']['comparator']['sha256']
                or type(receipt['observed_utc']) is not str or not 1 <= len(receipt['observed_utc']) <= 64):
            raise QueryError('query receipt differs after reaping')
        return raw, receipt

    paths = [Path(config['sources'][role]) for role in sorted(_ROLES)]
    paths += [Path(config_path), Path(config['operation']), Path(config['inputs']), Path(config['pins']), Path(config['inventory']),
              Path(config['public_freeze']), old_phase / 'nonce.bin', old_phase / 'heldout-record.json',
              old_phase / 'commitment.json', old_supervisor / 'supervisor.json',
              old_supervisor / 'supervisor-observation.json']
    paths = list(dict.fromkeys(paths))
    if any(path == root or root in path.parents for path in paths):
        raise QueryError('query inputs require a separate directory from outputs')
    prior, guards, supervisor = _project_modules(captured, config['sources'])
    _ACTIVE_OWNER.append((owner, supervisor, prior))
    try:
        leases = [guards.SourceFileLease(path.parent, [path]) for path in paths]
        previous_signals = owner._ignore_console_interrupts()
    except BaseException:
        _ACTIVE_OWNER.pop()
        _restore_modules(prior)
        raise

    def precheck():
        unchanged()
        private_identities()
        if not callable(getattr(rehearsal, 'verify', None)) or rehearsal.verify(config, operation) is not True:
            raise QueryError('production rehearsal was not verified')
        accounting.account_phase(root, reserve_side_bytes=4 * 1024 * 1024)
        accounting.inspect_resources(root, cache)
        unchanged()
        private_identities()
        state['first'] = _current(operation, observe_current)
        require_fresh_query()
        state['claim_sha'] = claim.claim_query(root=root, root_path_sha256=operation['root_path_sha256'],
            root_device=operation['root_device'], root_inode=operation['root_inode'],
            operation_sha256=operation_sha256, pool_input_sha256=operation['pool_input_sha256'],
            old_commitment=operation['old_commitment'],
            comparator_sha256=operation['sources']['comparator']['sha256'])

    def audit(_attempt, code, reaped):
        if type(code) is not int or code != 0 or reaped is not True:
            raise QueryError('query child was not cleanly reaped')
        raw, receipt = check_receipt()
        state['receipt_raw'], state['receipt'] = raw, receipt
        return {'status': 'incomplete', 'query_result': receipt['status'], 'receipt_sha256': _digest(raw)}

    def launch(acquired):
        duplicates = guards.InheritableHandles([handle for lease in leases for handle in lease.handles()])
        acquired.append(duplicates)
        inherited = duplicates.__enter__()
        return supervisor.supervise_heldout(command, supervisor_output, environment, audit_result=audit,
                                             timeout=120, inherited_handles=inherited)

    def postcheck(verdict):
        if (verdict.get('cleanup_verified') is not True or verdict.get('child_reaped') is not True
                or verdict.get('job_drained') is not True or verdict.get('job_closed') is not True
                or verdict.get('logs_complete') is not True or verdict.get('exit_code') != 0
                or type(verdict.get('postexit_active_processes')) is not int
                or verdict['postexit_active_processes'] != 0 or verdict.get('supervisor_error') is not None
                or verdict.get('log_overflow') is not False or verdict.get('audit_error_type') is not None
                or verdict.get('audit', {}).get('receipt_sha256') != _digest(state.get('receipt_raw', b''))):
            raise QueryError('query supervisor did not complete cleanly')
        saved = _read(supervisor_output / 'supervisor.json', 1000000)
        if _encode(json.loads(saved, object_pairs_hook=_pairs)) != _encode(verdict):
            raise QueryError('query supervisor record differs')
        raw, receipt = check_receipt()
        if raw != state['receipt_raw'] or receipt != state['receipt']:
            raise QueryError('query receipt changed after audit')
        unchanged()
        private_identities()
        last = _current(operation, observe_current)
        accounting.account_phase(root, reserve_side_bytes=65536)
        accounting.inspect_resources(root, cache)
        observation = {'schema': 'mlg-natural-successor-query-observation-v1', 'status': receipt['status'],
                       'query_claim_sha256': state['claim_sha'], 'receipt_sha256': _digest(raw),
                       'config_sha256': config_sha256, 'operation_sha256': operation_sha256,
                       'comparator_sha256': operation['sources']['comparator']['sha256'],
                       'supervisor_sha256': _digest(saved), 'pre_head': state['first']['head'],
                       'post_head': last['head'], 'command': command, 'pid': verdict['child_pid'],
                       'exit_code': verdict['exit_code'], 'reaped': True, 'job_drained': True, 'job_closed': True,
                       'stdout_sha256': verdict['stdout_sha256'], 'stderr_sha256': verdict['stderr_sha256']}
        output = _encode(observation)
        _write_new(root / 'pool-query-observation.json', output)
        if _read(root / 'pool-query-observation.json', 65536) != output:
            raise QueryError('query observation was not retained intact')
        return observation

    try:
        observation = owner.under_source_leases(leases, precheck, launch, postcheck, retained=state)
    finally:
        if not owner._QUARANTINED and not supervisor._QUARANTINED:
            for number, old in reversed(previous_signals):
                import signal
                signal.signal(number, old)
            _ACTIVE_OWNER.pop()
            _restore_modules(prior)
    if observation['status'] == 'collision':
        raise QueryCollision('the fixed successor pool collided; stop')
    return observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--operation-sha256', required=True)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or sys.flags.optimize:
        raise QueryError('isolated non-optimized source bootstrap is required')
    try:
        result = run_query(Path(args.config), config_sha256=args.config_sha256,
                           operation_sha256=args.operation_sha256)
        print(result['status'])
    except QueryCollision:
        print('collision')
        raise SystemExit(2)
    except BaseException:
        try:
            print('query incomplete', file=sys.stderr)
        except BaseException:
            pass
        for owner, supervisor, _ in _ACTIVE_OWNER:
            if owner._QUARANTINED or supervisor._QUARANTINED:
                owner._hold_quarantine()
                while supervisor._QUARANTINED:
                    try:
                        _HOLD.wait()
                    except BaseException:
                        continue
        raise SystemExit(1)


if __name__ == '__main__':
    main()
