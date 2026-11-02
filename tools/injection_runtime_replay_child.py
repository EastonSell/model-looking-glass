"""Gated, path-only child for the prospective public runtime comparison.

The trusted parent must bind the published plan, claim, exact source files,
protected cache and source leases, and independently observe process cleanup.
This child never allocates a split or publishes a result.
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
from urllib.parse import urlsplit
from urllib.request import urlopen


CONFIG_SCHEMA = 'mlg-injection-runtime-replay-child-config-v1'
CONFIG_PATHS = frozenset(('frozen_root', 'site_packages', 'cache_dir', 'heldout_runtime_file',
    'discovery_runtime_file', 'discovery_plan_file', 'runtime_deviation_file', 'replay_plan_file',
    'producer_source_file', 'site_nonce_file', 'site_record_file', 'claim_receipt_file',
    'reference_dir', 'attempt_dir', 'postcheck_file', 'pycache_prefix'))
TEST_PLAN_URL = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + '0' * 40
                 + '/experiments/pretrained_number_injection_cpu_v1/runtime-replay-plan-v1.json')
_PLAN_PATH = '/experiments/pretrained_number_injection_cpu_v1/runtime-replay-plan-v1.json'
_START = b'MLG-natural-discovery-start-v1\n'
_SOURCE = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
_PROTOCOL = '95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee'
_CHECKPOINT = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
_REFERENCE_COMMIT = '6b0dd7e1c7e30288d4f563175d589540ee8480fd'
_FREEZE = 'c327e191214cfb661b429c6d0a66900ca773c148782e1cc88be669c54deaee98'
_PREDICTION = '11376d8aae517db76d6a33c1ae5dc1f4b0c747a9d5216c9b02a17b1b0b300107'
_SOURCE_SUPPLEMENT = '2ccd3c6515e18953e960b6d5fcdb87abafd177058bd3e06a6f855c7fd86a8834'
_LAUNCHER_SHA = '930fee8d77ece12b6b5598391a2fac63d688faa9237d1e6bd04ba96a60e3c110'
_DISCOVERY_RUNTIME = 'b6860ec21fe6f0ab626164b86b494d783f3c18c92a697744b118be4cdb33754e'
_HELDOUT_RUNTIME = '7a967712a896d19ccfe4164c11a57e3ab1c65ece1a0ae26fedb220b9fc80e327'
_DISCOVERY_PLAN = 'e59d98638e474ed9f915d24fa527edd6357caa0e62c8bdabdda0433c20c683f8'
_DEVIATION = '6b333232d66caa348211c932d4b4244070651e625b0a0c0d7686551b23fd895b'
_PRODUCER = '6bf925af465f4761cde6d32c3454c6d0810b089af3d17ca914dff676ea27e8cd'
_SITE_COMMITMENT = 'c3739cb173d3c1bc3e3adda70025ff7f26a73c4690e50ecf5baf06ff8b359b9d'
_MODEL_STATE = '94b22301fa1a03550f8b75a255d6c7bb2dda25ee6d9079b70321a81ac884c1f1'
_ROLES = ('clean', 'repeat', 'injected', *(f'forward_{i}' for i in range(1, 5)),
          *(f'reverse_{i}' for i in range(1, 5)), 'null', 'self', 'upstream', 'downstream', 'verify_clean')
_SOURCE_NAMES = frozenset(('claim', 'owner', 'child', 'producer', 'supervisor', 'outer_observer',
                           'cpu_job_adapter', 'row_guard', 'phase_ledger', 'source_guard', 'auditor',
                           'driver', 'controller', 'cpu_job', 'observer_audit', 'observer_custody',
                           'auditor_core'))
_BUDGETS = {'phase_logical_bytes': 72 * 1024**2, 'phase_physical_bytes': 192 * 1024**2,
            'pilot_logical_bytes': 8 * 1024**2, 'pilot_physical_bytes': 16 * 1024**2,
            'diagnostic_reserved_bytes': 4 * 1024**2}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _json(raw):
    try:
        result = json.loads(raw, object_pairs_hook=_unique)
        if _canonical(result) != raw:
            raise ValueError('canonical JSON is required')
        return result
    except (TypeError, UnicodeError, RecursionError) as error:
        raise ValueError('invalid canonical JSON') from error


def _absolute(path):
    if type(path) is not str or not path or not Path(path).is_absolute():
        raise ValueError('absolute path required')
    return Path(path)


def _read(path, limit):
    path = _absolute(str(path))
    for component in (path, *path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('linked source path is unsupported')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
        raise ValueError('bounded ordinary source file required')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    after = path.lstat()
    if (len(raw) != before.st_size or len(raw) > limit or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('file changed during bounded read')
    return raw


def await_start(stream=None):
    stream = sys.stdin.buffer if stream is None else stream
    if stream.readline(len(_START) + 1) != _START:
        raise ValueError('gated parent start token is required')


def load_source(path, expected_sha256, name, package):
    if (type(expected_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None
            or type(name) is not str or name in sys.modules):
        raise ValueError('invalid source identity')
    raw = _read(path, 16 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('bound source bytes differ')
    module = ModuleType(name)
    module.__file__, module.__package__ = str(path), package
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def parse_config(raw):
    value = _json(raw)
    if (type(value) is not dict or set(value) != CONFIG_PATHS | {'schema', 'plan_url'}
            or value['schema'] != CONFIG_SCHEMA):
        raise ValueError('private child configuration differs')
    for key in CONFIG_PATHS:
        _absolute(value[key])
    parsed = urlsplit(value['plan_url']) if type(value['plan_url']) is str else None
    if (parsed is None or parsed.scheme != 'https' or parsed.netloc != 'raw.githubusercontent.com'
            or parsed.query or parsed.fragment or
            re.fullmatch(r'/EastonSell/model-looking-glass/[0-9a-f]{40}' + re.escape(_PLAN_PATH), parsed.path) is None):
        raise ValueError('immutable public replay plan URL required')
    return value


def validate_plan(raw, *, child_source_sha256):
    plan = _json(raw)
    fixed = {'schema': 'mlg-injection-runtime-replay-plan-v1', 'discovery_source_commit': _SOURCE,
             'protocol_sha256': _PROTOCOL, 'checkpoint_revision': _CHECKPOINT,
             'prediction_freeze_sha256': _FREEZE, 'prediction_sha256': _PREDICTION,
             'source_supplement_sha256': _SOURCE_SUPPLEMENT,
             'discovery_runtime_sha256': _DISCOVERY_RUNTIME, 'heldout_runtime_sha256': _HELDOUT_RUNTIME,
             'discovery_plan_sha256': _DISCOVERY_PLAN, 'runtime_deviation_sha256': _DEVIATION,
             'site_commitment': _SITE_COMMITMENT, 'model_state_sha256': _MODEL_STATE,
             'reference_commit': _REFERENCE_COMMIT, 'roles': list(_ROLES), 'budgets': _BUDGETS}
    if (type(plan) is not dict or set(plan) != set(fixed) | {'reference_sha256', 'sources'}
            or any(_canonical(plan.get(key)) != _canonical(value) for key, value in fixed.items())):
        raise ValueError('published replay plan differs from the fixed design')
    references, sources = plan['reference_sha256'], plan['sources']
    if (type(references) is not dict or set(references) != set(_ROLES)
            or type(sources) is not dict or set(sources) != _SOURCE_NAMES
            or any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None
                   for value in (*references.values(), *sources.values()))
            or sources['producer'] != _PRODUCER or sources['child'] != child_source_sha256):
        raise ValueError('published replay source or reference identities differ')
    return plan


def _read_digest(path, expected, limit):
    raw = _read(path, limit)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('bound file digest differs')
    return raw


def _check_runtime(discovery, heldout):
    left, right = _json(discovery), _json(heldout)
    if type(left) is not dict or type(right) is not dict:
        raise ValueError('runtime manifests must be objects')
    before, after = left.pop('system_environment_sha256', None), right.pop('system_environment_sha256', None)
    if (type(before) is not str or type(after) is not str or before == after
            or re.fullmatch(r'[0-9a-f]{64}', before) is None or re.fullmatch(r'[0-9a-f]{64}', after) is None
            or _canonical(left) != _canonical(right)):
        raise ValueError('runtime differs beyond the documented system fingerprint')
    return _json(heldout)


def _write_once(path, value):
    raw = _canonical(value)
    if len(raw) > 16384:
        raise ValueError('private replay postcheck exceeds its bound')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if type(count) is not int or not 0 < count <= len(view):
                raise OSError('short private replay postcheck')
            view = view[count:]
        os.fsync(fd)
    except BaseException as error:
        try:
            os.close(fd)
        except BaseException as close_error:
            if (isinstance(close_error, (KeyboardInterrupt, SystemExit))
                    and not isinstance(error, (KeyboardInterrupt, SystemExit))):
                raise close_error from error
            raise error from close_error
        raise
    else:
        os.close(fd)


def _check_claim_receipt(raw, plan_sha256, config):
    """Check consistency only; the parent must verify the row and observed exit."""
    value = _json(raw)
    keys = {'schema', 'status', 'plan_sha256', 'intent_sha256', 'ledger_path', 'phase_path',
            'pilot_path', 'diagnostic_path', 'parent_identity', 'ledger_identity',
            'diagnostic_actual_bytes', 'diagnostic_inventory_sha256', 'injection_claim_row_sha256'} | set(_BUDGETS)
    if (type(value) is not dict or set(value) != keys
            or value['schema'] != 'mlg-injection-replay-claim-receipt-v1' or value['status'] != 'preclaimed'
            or value['plan_sha256'] != plan_sha256
            or any(_canonical(value[k]) != _canonical(v) for k, v in _BUDGETS.items())
            or any(type(value[k]) is not str or re.fullmatch(r'[0-9a-f]{64}', value[k]) is None
                   for k in ('plan_sha256', 'intent_sha256', 'diagnostic_inventory_sha256', 'injection_claim_row_sha256'))
            or any(type(value[k]) is not list or len(value[k]) != 2
                   or any(type(n) is not int or n <= 0 for n in value[k])
                   for k in ('parent_identity', 'ledger_identity'))
            or type(value['diagnostic_actual_bytes']) is not int
            or not 0 <= value['diagnostic_actual_bytes'] <= _BUDGETS['diagnostic_reserved_bytes']):
        raise ValueError('preclaim receipt differs')
    ledger, phase, pilot, diagnostic = (_absolute(value[k]) for k in
                                        ('ledger_path', 'phase_path', 'pilot_path', 'diagnostic_path'))
    if (pilot.name != 'runtime-compatibility-replay-v1' or phase.name != 'injection-heldout-' + _FREEZE
            or pilot.parent != phase.parent or _absolute(config['attempt_dir']) != pilot / 'attempt'
            or _absolute(config['postcheck_file']) != pilot / 'child-postcheck.json'
            or ledger == phase or diagnostic == pilot or diagnostic == phase):
        raise ValueError('preclaim paths differ')
    return value


def _post_runtime_check(launcher, state, manifest):
    """Observe the existing state; do not reset backend controls after a run."""
    if (state['events'] or launcher._numeric_state(state['torch']) != manifest['numerics']
            or launcher._environment_policy() != manifest['system_environment_sha256']
            or any(os.environ.get(key) != value for key, value in manifest['environment'].items())
            or launcher._cache_policy(state['cache']) != (state['home'], state['cache'])
            or launcher._verify_unloaded_source(launcher._CONTEXT['repo'], _SOURCE) != state['repository']
            or any(launcher._CONTEXT['prefix'].iterdir())):
        raise ValueError('post-replay runtime or source changed')
    launcher._verify_effective_cache(state['home'], state['cache'])


def _assert_isolated(prefix):
    flags = sys.flags
    if (not (flags.isolated and flags.ignore_environment and flags.no_site and flags.no_user_site
             and flags.safe_path and sys.dont_write_bytecode and flags.optimize == 0)
            or not sys.pycache_prefix or _absolute(sys.pycache_prefix) != _absolute(prefix)
            or Path(sys.executable).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)
            or any(name == 'looking_glass' or name.startswith('looking_glass.') or name == 'scripts'
                   or name.startswith('scripts.') for name in sys.modules)
            or sys.meta_path != [importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                                 importlib.machinery.PathFinder]):
        raise ValueError('isolated source-only interpreter is required')
    path = _absolute(prefix)
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('protected bytecode prefix is linked')
    if not path.is_dir() or any(path.iterdir()):
        raise ValueError('protected bytecode prefix is not empty')


def _fetch_immutable(url):
    with urlopen(url, timeout=20) as response:
        data = response.read(2 * 1024 * 1024 + 1)
        if response.geturl() != url or not 0 < len(data) <= 2 * 1024 * 1024:
            raise ValueError('published plan readback is unavailable')
    return data


def _check_producer_terminal(raw, result):
    _json(raw)
    if (type(result) is not dict or _canonical(result) != raw
            or result.get('status') != 'replay-produced-private-awaiting-independent-audit'
            or type(result.get('forward_count')) is not int or result['forward_count'] != 16):
        raise ValueError('private replay terminal differs')


def execute(config_path, *, validation_only=False):
    """Run the bound child after the parent gate; this is not an authorization."""
    if type(validation_only) is not bool:
        raise ValueError('validation mode is invalid')
    config = parse_config(_read(config_path, 65536))
    _assert_isolated(config['pycache_prefix'])
    if os.path.lexists(config['postcheck_file']) or os.path.lexists(config['attempt_dir']):
        raise ValueError('replay output already exists')
    plan_bytes = _read(config['replay_plan_file'], 2 * 1024 * 1024)
    if _fetch_immutable(config['plan_url']) != plan_bytes:
        raise ValueError('published replay plan differs')
    child_sha256 = hashlib.sha256(_read(__file__, 16 * 1024 * 1024)).hexdigest()
    plan = validate_plan(plan_bytes, child_source_sha256=child_sha256)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    original = _read_digest(config['discovery_runtime_file'], _DISCOVERY_RUNTIME, 1024 * 1024)
    actual = _read_digest(config['heldout_runtime_file'], _HELDOUT_RUNTIME, 1024 * 1024)
    manifest = _check_runtime(original, actual)
    discovery_plan = _read_digest(config['discovery_plan_file'], _DISCOVERY_PLAN, 2 * 1024 * 1024)
    _read_digest(config['runtime_deviation_file'], _DEVIATION, 1024 * 1024)
    references = tuple(_read_digest(Path(config['reference_dir']) / ('observation-00-' + role + '.json'),
                                    plan['reference_sha256'][role], 2 * 1024 * 1024) for role in _ROLES)
    if not validation_only:
        _check_claim_receipt(_read(config['claim_receipt_file'], 65536), plan_sha256, config)
    root = _absolute(config['frozen_root'])
    launcher = load_source(root / 'scripts' / 'pretrained_number_cpu_launcher.py', _LAUNCHER_SHA,
                           '_mlg_bound_replay_launcher', '')
    launcher._CONTEXT = launcher._boot(config['site_packages'], _SOURCE)
    state = launcher._initialize_runtime(source_commit=_SOURCE, cache_dir=config['cache_dir'],
                                         site_packages=config['site_packages'])
    if state['runtime_bytes'] != actual:
        raise ValueError('actual runtime differs from its prospective binding')
    producer = load_source(config['producer_source_file'], _PRODUCER, 'looking_glass._mlg_bound_replay',
                           'looking_glass')
    if (tuple(producer._ROLES) != _ROLES or tuple(producer._REFERENCE_SHA256)
            != tuple(plan['reference_sha256'][role] for role in _ROLES)):
        raise ValueError('fixed replay reference identities differ')
    if validation_only:
        _post_runtime_check(launcher, state, manifest)
        result = {'schema': 'mlg-injection-runtime-replay-child-postcheck-v1',
                  'status': 'validated_no_model_forward', 'plan_sha256': plan_sha256,
                  'heldout_runtime_sha256': _HELDOUT_RUNTIME, 'child_source_sha256': child_sha256,
                  'forward_count': 0}
    else:
        nonce = _read(config['site_nonce_file'], 64)
        site_record = _read(config['site_record_file'], 65536)
        producer.verify_site_reveal(nonce, site_record, _SITE_COMMITMENT, expected_source_commit=_SOURCE)
        bundle = launcher._load_trusted_model(state)
        result = producer.run_public_replay(model=bundle.model, attempt_dir=config['attempt_dir'],
            plan_bytes=discovery_plan, source_commit=_SOURCE, plan_sha256=_DISCOVERY_PLAN,
            discovery_runtime_sha256=_DISCOVERY_RUNTIME, heldout_runtime_sha256=_HELDOUT_RUNTIME,
            site_nonce=nonce, site_record_bytes=site_record, site_commitment=_SITE_COMMITMENT,
            references=references, expected_model_state_sha256=_MODEL_STATE)
        from looking_glass.pretrained_number_injection import _check_model_for_cpu
        from looking_glass.trusted_service_profile import prepare_trusted_source, verify_trusted_cpu_model
        _check_model_for_cpu(bundle.model)
        if (verify_trusted_cpu_model(state['preflight'], bundle.model) is not True
                or prepare_trusted_source(launcher._CONTEXT['repo'], state['cache']) != state['preflight']):
            raise ValueError('post-replay model or assets changed')
        _post_runtime_check(launcher, state, manifest)
        terminal = _read(Path(config['attempt_dir']) / 'owner-terminal.json', 65536)
        receipts = _read(Path(config['attempt_dir']) / 'receipts.jsonl', 1024 * 1024)
        _check_producer_terminal(terminal, result)
        result = {'schema': 'mlg-injection-runtime-replay-child-postcheck-v1',
                  'status': 'private_replay_pending_independent_audit', 'plan_sha256': plan_sha256,
                  'heldout_runtime_sha256': _HELDOUT_RUNTIME, 'child_source_sha256': child_sha256,
                  'discovery_runtime_sha256': _DISCOVERY_RUNTIME, 'runtime_deviation_sha256': _DEVIATION,
                  'source_supplement_sha256': _SOURCE_SUPPLEMENT, 'forward_count': 16,
                  'owner_terminal_sha256': hashlib.sha256(terminal).hexdigest(),
                  'receipts_sha256': hashlib.sha256(receipts).hexdigest()}
    _write_once(config['postcheck_file'], result)
    return result


def main(argv=None, stream=None):
    try:
        await_start(stream)
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) == 2 and args[0] == '--validate-only':
            execute(args[1], validation_only=True)
        elif len(args) == 1:
            execute(args[0], validation_only=False)
        else:
            raise ValueError('invalid invocation')
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
