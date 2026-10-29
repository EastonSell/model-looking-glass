"""Source-bound preparation for the prospective injection runtime replay.

The command-line modes run only a gated, no-model validation and its observer.
Claim and process helpers require separately verified authority and observed
exits; no helper alone authorizes a claim, model forward or scientific promotion.
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import sys
from types import MappingProxyType, ModuleType
from urllib.request import urlopen


RAW_PREFIX = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
DISCOVERY_SOURCE = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
PLAN_PATH = '/experiments/pretrained_number_injection_cpu_v1/runtime-replay-plan-v1.json'
START_TOKEN = b'MLG-natural-discovery-start-v1\n'
CHECKPOINT_REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
FROZEN_LAUNCHER_SHA256 = '930fee8d77ece12b6b5598391a2fac63d688faa9237d1e6bd04ba96a60e3c110'
ASSET_NAMES = frozenset(('config.json', 'generation_config.json', 'merges.txt', 'model.safetensors',
                         'tokenizer.json', 'tokenizer_config.json', 'vocab.json'))
SOURCE_SUFFIXES = {
    'claim': 'tools/injection_replay_claim.py',
    'owner': 'tools/injection_runtime_replay_owner.py',
    'child': 'tools/injection_runtime_replay_child.py',
    'producer': 'looking_glass/injection_runtime_replay.py',
    'supervisor': 'looking_glass/natural_heldout_supervisor.py',
    'outer_observer': 'looking_glass/injection_outer_observer.py',
    'cpu_job_adapter': 'tools/injection_heldout_owner.py',
    'row_guard': 'looking_glass/injection_preclaim_guard.py',
    'phase_ledger': 'looking_glass/agreement_phase_ledger.py',
    'source_guard': 'scripts/v2_source_guards.py',
    'auditor': 'tools/audit_injection_runtime_replay.py',
    'driver': 'tools/injection_runtime_replay_driver.py',
    'controller': 'looking_glass/natural_agreement_controller.py',
    'cpu_job': 'scripts/number_cpu_outer.py',
    'observer_audit': 'looking_glass/injection_preparation_audit.py',
    'observer_custody': 'looking_glass/injection_split_custody.py',
    'auditor_core': 'tools/audit_injection_heldout.py',
}
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_MAX_SOURCE = 16 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _ordinary_source(path, limit=_MAX_SOURCE):
    if type(path) is not str or not Path(path).is_absolute() or '..' in Path(path).parts:
        raise ValueError('an absolute source path is required')
    path = Path(path)
    for component in (path, *path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('linked source paths are unsupported')
    before = path.lstat()
    if (type(limit) is not int or not 0 < limit <= _MAX_SOURCE or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1 or not 0 < before.st_size <= limit):
        raise ValueError('a bounded ordinary source file is required')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    after = path.lstat()
    if (len(raw) != before.st_size or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('source changed during the bounded read')
    return raw


def _fetch_source(url):
    with urlopen(url, timeout=20) as response:
        raw = response.read(_MAX_SOURCE + 1)
        if response.geturl() != url or not 0 < len(raw) <= _MAX_SOURCE:
            raise ValueError('immutable source readback failed')
    return raw


def await_start(stream=None):
    """Wait for the trusted supervisor before loading any project module."""
    stream = sys.stdin.buffer if stream is None else stream
    if stream.readline(len(START_TOKEN) + 1) != START_TOKEN:
        raise ValueError('the supervised start token is required')


def capture_plan(path, url, *, fetch=_fetch_source):
    """Read back a canonical plan from its exact immutable public location."""
    if (type(url) is not str or re.fullmatch(re.escape(RAW_PREFIX) + r'[0-9a-f]{40}'
                                             + re.escape(PLAN_PATH), url) is None
            or not callable(fetch)):
        raise ValueError('the immutable replay plan URL differs')
    raw = _ordinary_source(path, 2 * 1024 * 1024)
    if fetch(url) != raw:
        raise ValueError('the published replay plan differs')
    try:
        plan = json.loads(raw, object_pairs_hook=_unique)
        if (type(plan) is not dict or canonical(plan) != raw or type(plan.get('sources')) is not dict
                or set(plan['sources']) != set(SOURCE_SUFFIXES)
                or any(type(value) is not str or _HEX64.fullmatch(value) is None
                       for value in plan['sources'].values())):
            raise ValueError('the replay plan source set differs')
    except (TypeError, UnicodeError, RecursionError) as error:
        raise ValueError('the replay plan is not canonical') from error
    return plan, raw


def capture_sources(locations, identities, *, fetch=_fetch_source):
    """Read back exact local/public source bytes before executing any of them."""
    if (type(locations) is not dict or type(identities) is not dict
            or set(locations) != set(SOURCE_SUFFIXES) or set(identities) != set(SOURCE_SUFFIXES)
            or not callable(fetch)):
        raise ValueError('the complete source set is required')
    captured, paths = {}, set()
    for name, suffix in SOURCE_SUFFIXES.items():
        item, expected = locations[name], identities[name]
        if (type(item) is not dict or set(item) != {'path', 'url'}
                or type(expected) is not str or _HEX64.fullmatch(expected) is None
                or type(item['path']) is not str or type(item['url']) is not str
                or re.fullmatch(re.escape(RAW_PREFIX) + r'[0-9a-f]{40}/' + re.escape(suffix), item['url']) is None):
            raise ValueError('source identity or immutable URL differs')
        path = Path(item['path'])
        expected_parts = tuple(suffix.split('/'))
        if tuple(path.parts[-len(expected_parts):]) != expected_parts:
            raise ValueError('source path has the wrong role')
        normalized = os.path.normcase(os.path.abspath(item['path']))
        if normalized in paths:
            raise ValueError('source paths must be distinct')
        paths.add(normalized)
        raw = _ordinary_source(item['path'])
        if hashlib.sha256(raw).hexdigest() != expected or fetch(item['url']) != raw:
            raise ValueError('local or published source differs')
        captured[name] = raw
    return captured


def recheck_sources(locations, captured):
    """Re-read captured sources while the caller retains its file leases."""
    if (type(locations) is not dict or type(captured) is not dict
            or set(locations) != set(SOURCE_SUFFIXES) or set(captured) != set(SOURCE_SUFFIXES)):
        raise ValueError('the complete captured source set is required')
    for name in SOURCE_SUFFIXES:
        item, raw = locations[name], captured[name]
        if (type(item) is not dict or set(item) != {'path', 'url'} or type(raw) is not bytes
                or _ordinary_source(item['path']) != raw):
            raise ValueError('source changed while protected')
    return True


def capture_public_files(paths, specifications, *, fetch=_fetch_source):
    """Capture caller-pinned immutable public artifacts without executing them.

    The caller supplies fixed, independently reviewed specifications; this
    helper does not establish that arbitrary specifications are authoritative.
    """
    if (type(paths) is not dict or type(specifications) is not dict or set(paths) != set(specifications)
            or not 0 < len(paths) <= 64 or not callable(fetch)):
        raise ValueError('the exact public input set is required')
    captured, seen = {}, set()
    for name, specification in specifications.items():
        if (type(name) is not str or type(specification) is not tuple or len(specification) != 3
                or type(paths[name]) is not str):
            raise ValueError('public input specification differs')
        url, digest, limit = specification
        if (type(url) is not str or re.fullmatch(re.escape(RAW_PREFIX) +
                r'[0-9a-f]{40}/[A-Za-z0-9._/-]+', url) is None
                or any(part in ('', '.', '..') for part in url[len(RAW_PREFIX) + 41:].split('/'))
                or type(digest) is not str or _HEX64.fullmatch(digest) is None
                or type(limit) is not int or not 0 < limit <= _MAX_SOURCE):
            raise ValueError('immutable public input identity differs')
        normalized = os.path.normcase(os.path.abspath(paths[name]))
        if normalized in seen:
            raise ValueError('public input paths must be distinct')
        seen.add(normalized)
        raw = _ordinary_source(paths[name], limit)
        if hashlib.sha256(raw).hexdigest() != digest or fetch(url) != raw:
            raise ValueError('local or published public input differs')
        captured[name] = raw
    return captured


def fixed_public_specs(plan, child):
    """Describe the exact published freeze, runtime and 16 public references."""
    if (type(plan) is not dict or type(plan.get('reference_sha256')) is not dict
            or set(plan['reference_sha256']) != set(child._ROLES)
            or any(type(value) is not str or _HEX64.fullmatch(value) is None
                   for value in plan['reference_sha256'].values())):
        raise ValueError('the fixed public reference set differs')
    result_dir = 'results/pretrained-number-injection-discovery-v1-20261028/'
    study = 'experiments/pretrained_number_injection_cpu_v1/'
    def url(commit, path):
        return RAW_PREFIX + commit + '/' + path
    specifications = {
        'freeze': (url('517d3a9352d8ce369bdf5b0e0577a2a690323986', result_dir + 'injection-prediction-freeze.json'),
                   child._FREEZE, 2 * 1024 * 1024),
        'prediction': (url(child._REFERENCE_COMMIT, result_dir + 'public/prediction.json'),
                       child._PREDICTION, 2 * 1024 * 1024),
        'discovery_runtime': (url('7ac8915d8e7782fb207247e45132ba5ea4375238', study + 'runtime.json'),
                              child._DISCOVERY_RUNTIME, 1024 * 1024),
        'heldout_runtime': (url('db008b2a2e7401bde63596b1350661e29a09bd7e', study + 'heldout-runtime-v1.json'),
                            child._HELDOUT_RUNTIME, 1024 * 1024),
        'discovery_plan': (url('afdd759f140c5d6101de9e2b7f3c9f05f1820bbb', study + 'plan.json'),
                           child._DISCOVERY_PLAN, 2 * 1024 * 1024),
        'site_commitment': (url('a23f66f3abf6e5dcf055c3f8b4dbb551ab8f9a29', study + 'site-commitment.txt'),
                            '26d4e9b45bd85c6e238d87b0afd49347b1a1f448a86273e029b1fa7b89e7e68e', 128),
        'source_supplement': (url('a1e17689af3808913ba160be638946bc010f01f9',
                                  'docs/PRETRAINED_NUMBER_INJECTION_HELDOUT_SUPPLEMENT.md'),
                              child._SOURCE_SUPPLEMENT, 1024 * 1024),
        'runtime_deviation': (url('83c519ac21f4b873a2e313bf05f9191544ca7647',
                                  'docs/PRETRAINED_NUMBER_INJECTION_RUNTIME_DEVIATION.md'),
                              child._DEVIATION, 1024 * 1024),
    }
    for role in child._ROLES:
        specifications['reference_' + role] = (
            url(child._REFERENCE_COMMIT, result_dir + 'public/observation-00-' + role + '.json'),
            plan['reference_sha256'][role], 512 * 1024)
    return specifications


def verify_fixed_freeze(raw, child):
    expected = {'schema': 'mlg-number-prediction-freezes-v1', 'protocol_sha256': child._PROTOCOL,
                'checkpoint_revision': child._CHECKPOINT,
                'tracks': {'injection': {'prediction_sha256': child._PREDICTION,
                                         'runtime_sha256': child._DISCOVERY_RUNTIME,
                                         'source_commit': child._SOURCE}}}
    if (type(raw) is not bytes or raw != canonical(expected)
            or hashlib.sha256(raw).hexdigest() != child._FREEZE):
        raise ValueError('the published prediction freeze differs')
    return True


def capture_fixed_public_inputs(paths, plan, child, *, fetch=_fetch_source):
    """Read back the frozen public artifacts and both disclosed runtimes."""
    captured = capture_public_files(paths, fixed_public_specs(plan, child), fetch=fetch)
    verify_fixed_freeze(captured['freeze'], child)
    if captured['site_commitment'] != (child._SITE_COMMITMENT + '\n').encode('ascii'):
        raise ValueError('the published site commitment differs')
    child._check_runtime(captured['discovery_runtime'], captured['heldout_runtime'])
    return captured


def capture_frozen_repository(root, runtime, *, fetch=_fetch_source):
    """Verify the frozen launcher before executing its clean-tree verifier."""
    if (type(root) is not str or not Path(root).is_absolute() or '..' in Path(root).parts
            or type(runtime) is not dict or runtime.get('source_commit') != DISCOVERY_SOURCE
            or type(runtime.get('repository_source_sha256')) is not str
            or _HEX64.fullmatch(runtime['repository_source_sha256']) is None
            or type(runtime.get('source_tree')) is not str
            or re.fullmatch(r'[0-9a-f]{40}', runtime['source_tree']) is None or not callable(fetch)):
        raise ValueError('the frozen repository anchor differs')
    path = Path(root) / 'scripts' / 'pretrained_number_cpu_launcher.py'
    raw = _ordinary_source(str(path), 1024 * 1024)
    url = RAW_PREFIX + DISCOVERY_SOURCE + '/scripts/pretrained_number_cpu_launcher.py'
    if hashlib.sha256(raw).hexdigest() != FROZEN_LAUNCHER_SHA256 or fetch(url) != raw:
        raise ValueError('the frozen launcher differs from its published source')
    launcher = load_captured('_mlg_replay_frozen_launcher', path, raw, FROZEN_LAUNCHER_SHA256)
    record = launcher._verify_unloaded_source(Path(root), DISCOVERY_SOURCE)
    record_bytes = canonical(record)
    if (hashlib.sha256(record_bytes).hexdigest() != runtime['repository_source_sha256']
            or record.get('git_tree') != runtime['source_tree']):
        raise ValueError('the frozen checkout differs from the discovery runtime')
    return launcher, record_bytes


def check_assets(cache_dir, manifest):
    """Hash the seven pinned offline assets without loading a model.

    The manifest must come from the independently verified published runtime;
    this read-only check does not itself protect files from later changes.
    """
    if (type(cache_dir) is not str or not Path(cache_dir).is_absolute() or '..' in Path(cache_dir).parts
            or type(manifest) is not dict or set(manifest) != {'schema', 'model_id', 'revision', 'assets'}
            or manifest['schema'] != 'resource-bootstrap-assets-v1'
            or manifest['model_id'] != 'distilbert/distilgpt2' or manifest['revision'] != CHECKPOINT_REVISION
            or type(manifest['assets']) is not list or len(manifest['assets']) != len(ASSET_NAMES)):
        raise ValueError('the fixed offline asset manifest differs')
    snapshot = Path(cache_dir) / 'models--distilbert--distilgpt2' / 'snapshots' / CHECKPOINT_REVISION
    assets, total, paths = set(), 0, []
    for asset in manifest['assets']:
        if (type(asset) is not dict or set(asset) != {'name', 'size', 'sha256'}
                or type(asset['name']) is not str or asset['name'] not in ASSET_NAMES or asset['name'] in assets
                or type(asset['size']) is not int or not 0 < asset['size'] <= 1024**3
                or type(asset['sha256']) is not str or _HEX64.fullmatch(asset['sha256']) is None):
            raise ValueError('an offline asset identity differs')
        assets.add(asset['name'])
        path = snapshot / asset['name']
        for component in (path, *path.parents):
            info = component.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('an offline asset path is linked')
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != asset['size']:
            raise ValueError('an offline asset size or identity differs')
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                count += len(block)
                if count > asset['size']:
                    raise ValueError('an offline asset grew while reading')
                digest.update(block)
        after = path.lstat()
        if (count != asset['size'] or digest.hexdigest() != asset['sha256']
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError('an offline asset changed or has a different digest')
        total += count
        if total > 1024**3:
            raise ValueError('offline assets exceed their bounded inventory')
        paths.append(path)
    if assets != ASSET_NAMES:
        raise ValueError('the fixed offline assets are incomplete')
    return tuple(paths)


def check_runtime_environment(environment, workspace, cache_dir, manifest):
    """Check the exact curated child environment against the published runtime."""
    numeric = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE'}
    controls = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
    system_names = {'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'COMSPEC', 'PATHEXT',
                    'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH'}
    cache_names = {'MLG_STUDY_WORKSPACE', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'}
    fixed = {**numeric, **controls}
    if (type(environment) is not dict or type(manifest) is not dict
            or any(type(key) is not str or type(value) is not str for key, value in environment.items())
            or set(environment) - (set(fixed) | system_names | cache_names)
            or any(environment.get(key) != value for key, value in fixed.items())
            or any(not environment.get(key) for key in cache_names | {'PATH'})
            or canonical(manifest.get('environment')) != canonical(fixed)):
        raise ValueError('the prospective runtime environment differs')
    system = {key: environment[key] for key in sorted(system_names) if key in environment}
    if (type(manifest.get('system_environment_sha256')) is not str
            or hashlib.sha256(canonical(system)).hexdigest() != manifest['system_environment_sha256']):
        raise ValueError('the prospective system environment fingerprint differs')
    if (type(workspace) is not str or type(cache_dir) is not str
            or not Path(workspace).is_absolute() or not Path(cache_dir).is_absolute()):
        raise ValueError('the prospective cache paths differ')
    try:
        root = Path(workspace).resolve(strict=True)
        home = Path(environment['HF_HOME']).resolve(strict=True)
        cache = Path(cache_dir).resolve(strict=True)
        configured = tuple(Path(environment[name]).resolve(strict=True)
                           for name in ('MLG_STUDY_WORKSPACE', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'))
    except (OSError, RuntimeError) as error:
        raise ValueError('the prospective cache paths are unavailable') from error
    if (not root.is_dir() or not home.is_dir() or not cache.is_dir()
            or configured != (root, cache, cache) or home == root or not home.is_relative_to(root)
            or cache == home or not cache.is_relative_to(home)):
        raise ValueError('the prospective offline cache differs')
    return True


def claim_row_sha256(row):
    if (type(row) is not list or len(row) != 5 or type(row[0]) is not str or row[0] != 'injection_heldout'
            or type(row[1]) is not str or not Path(row[1]).is_absolute()
            or type(row[2]) is not int or row[2] != 72 * 1024**2
            or type(row[3]) is not int or row[3] != 192 * 1024**2
            or type(row[4]) is not int or not 12 * 1024**3 <= row[4] < 2**63):
        raise ValueError('the fixed injection claim row differs')
    return hashlib.sha256(json.dumps(row, ensure_ascii=True, separators=(',', ':')).encode('ascii')).hexdigest()


def _database_identity(path, identity):
    if (type(path) is not str or not Path(path).is_absolute() or '..' in Path(path).parts
            or type(identity) is not tuple or len(identity) != 2
            or any(type(value) is not int or value <= 0 for value in identity)):
        raise ValueError('the protected ledger identity is invalid')
    path = Path(path)
    for component in (path, *path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('the protected ledger path is linked')
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != identity):
        raise ValueError('the protected ledger identity changed')
    return path


def _read_claim_row(ledger, phase):
    database = sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True, timeout=30)
    try:
        rows = database.execute('SELECT phase, CASE WHEN LENGTH(CAST(output AS BLOB)) <= 32768 '
                                'THEN output END, logical, disk, free FROM claims '
                                'WHERE phase=? LIMIT 2', ('injection_heldout',)).fetchall()
    except BaseException as primary:
        try:
            database.close()
        except BaseException as cleanup:
            if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise primary from cleanup
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary from cleanup
        raise
    else:
        database.close()
    if len(rows) != 1:
        raise ValueError('the one-use phase row is absent or duplicated')
    row = list(rows[0])
    claim_row_sha256(row)
    if row[1] != phase:
        raise ValueError('the one-use phase output differs')
    return row


def guarded_claim_row(ledger_path, phase_path, guard, *, db_identity, state_sha256,
                      history_sha256, scope_sha256, original_claims_sha256, expected_row_sha256=None):
    """Read the row, verify it through the independently bound v6 guard, and reread."""
    digests = (state_sha256, history_sha256, scope_sha256, original_claims_sha256)
    if (type(phase_path) is not str or not Path(phase_path).is_absolute() or '..' in Path(phase_path).parts
            or any(type(value) is not str or _HEX64.fullmatch(value) is None for value in digests)
            or (expected_row_sha256 is not None and
                (type(expected_row_sha256) is not str or _HEX64.fullmatch(expected_row_sha256) is None))):
        raise ValueError('the protected claim anchor differs')
    ledger = _database_identity(ledger_path, db_identity)
    row = _read_claim_row(ledger, phase_path)
    row_sha256 = claim_row_sha256(row)
    if expected_row_sha256 is not None and row_sha256 != expected_row_sha256:
        raise ValueError('the observed phase row differs')
    anchor = guard.ClaimAnchor(ledger_path=ledger_path, db_dev=db_identity[0], db_ino=db_identity[1],
        phase_path=phase_path, state_sha256=state_sha256, history_sha256=history_sha256,
        scope_sha256=scope_sha256, original_claims_sha256=original_claims_sha256,
        heldout_claim_sha256=row_sha256)
    if guard.verify_existing_claim(ledger_path, 'injection_heldout', output_path=phase_path, anchor=anchor) is not True:
        raise ValueError('the anchored one-use claim is unverified')
    _database_identity(ledger_path, db_identity)
    if _read_claim_row(ledger, phase_path) != row:
        raise ValueError('the phase row changed during verification')
    return row


def _process_identity(value):
    if (type(value) is not tuple or len(value) != 2 or type(value[0]) is not int
            or not 0 < value[0] <= 0xffffffff or type(value[1]) is not int
            or not 0 < value[1] < 2**64):
        raise ValueError('the retained process identity differs')
    return value


def _canonical_object(raw, limit):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError('the bounded private record differs')
    try:
        value = json.loads(raw, object_pairs_hook=_unique)
        if type(value) is not dict or canonical(value) != raw:
            raise ValueError('the private record is not canonical')
        return value
    except (TypeError, UnicodeError, RecursionError) as error:
        raise ValueError('the private record is not canonical') from error


def _write_once(path, raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 8192:
        raise ValueError('the private terminal is invalid or oversized')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(descriptor, view)
            if type(count) is not int or not 0 < count <= len(view):
                raise OSError('the private terminal write was incomplete')
            view = view[count:]
        os.fsync(descriptor)
    except BaseException as primary:
        try:
            os.close(descriptor)
        except BaseException as cleanup:
            if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise primary from cleanup
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary from cleanup
        raise
    else:
        os.close(descriptor)
    if _ordinary_source(str(path), 8192) != raw:
        raise ValueError('the private terminal did not read back exactly')


def verify_claimant_terminal(raw, *, parent_identity, plan_sha256, config_sha256, driver_sha256,
                             intent_sha256, receipt_sha256, claim_row_sha256):
    """Check a terminal against retained-handle and independently verified inputs."""
    pid, birth = _process_identity(parent_identity)
    digests = (plan_sha256, config_sha256, driver_sha256, intent_sha256, receipt_sha256, claim_row_sha256)
    if any(type(value) is not str or _HEX64.fullmatch(value) is None for value in digests):
        raise ValueError('the claimant terminal identity differs')
    _canonical_object(raw, 8192)
    expected = {'schema': 'mlg-injection-replay-claimant-terminal-v1', 'status': 'preclaimed',
        'parent_pid': pid, 'parent_birth_100ns': birth, 'plan_sha256': plan_sha256,
        'config_sha256': config_sha256, 'driver_sha256': driver_sha256, 'intent_sha256': intent_sha256,
        'receipt_sha256': receipt_sha256, 'claim_row_sha256': claim_row_sha256}
    if raw != canonical(expected):
        raise ValueError('the claimant terminal differs')
    return True


def claim_once(*, workspace, plan_bytes, plan_sha256, config_sha256, driver_sha256,
               parent_identity, ledger_identity, state_sha256, history_sha256, scope_sha256,
               original_claims_sha256, claim, child, guard, ledger_module, owner, authority):
    """Claim the single phase and save a private terminal after final checks.

    The dedicated caller must first source-bind these modules, protect all
    inputs, verify the published plan and no-model rehearsal, and supply a
    real authority callback. An independent observer must retain and reap this
    process: a terminal alone cannot prove a successful return.
    """
    if (type(workspace) is not str or not Path(workspace).is_absolute() or type(plan_bytes) is not bytes
            or not 0 < len(plan_bytes) <= 2 * 1024 * 1024 or not callable(authority)
            or any(type(value) is not str or _HEX64.fullmatch(value) is None for value in
                   (plan_sha256, config_sha256, driver_sha256, state_sha256, history_sha256,
                    scope_sha256, original_claims_sha256))
            or hashlib.sha256(plan_bytes).hexdigest() != plan_sha256):
        raise ValueError('the one-use claimant inputs differ')
    root = Path(workspace)
    parent = root / 'work' / 'number-injection-heldout-cpu-v1'
    ledger_path = root / 'work' / 'agreement-study-ledger-v5' / 'ledger.sqlite3'
    phase = parent / ('injection-heldout-' + child._FREEZE)
    pilot = parent / 'runtime-compatibility-replay-v1'
    terminal_path = parent / 'replay-claimant-terminal.json'
    if os.path.lexists(terminal_path) or authority() is not True:
        raise ValueError('the claimant authority or fresh terminal is unavailable')
    def row_check(path, output):
        return guarded_claim_row(str(path), str(output), guard, db_identity=ledger_identity,
            state_sha256=state_sha256, history_sha256=history_sha256, scope_sha256=scope_sha256,
            original_claims_sha256=original_claims_sha256)
    receipt = claim.claim_replay_once(workspace=root, plan_bytes=plan_bytes, plan_sha256=plan_sha256,
        parent_identity=parent_identity, ledger_identity=ledger_identity, verify_authority=authority,
        verify_claim_row=row_check, claim_phase=ledger_module.claim_phase)
    receipt_raw = _ordinary_source(str(parent / 'replay-claim-receipt.json'), 65536)
    intent_raw = _ordinary_source(str(parent / 'replay-claim-intent.json'), 65536)
    mini_config = {'attempt_dir': str(pilot / 'attempt'), 'postcheck_file': str(pilot / 'child-postcheck.json')}
    checked = child._check_claim_receipt(receipt_raw, plan_sha256, mini_config)
    row = guarded_claim_row(str(ledger_path), str(phase), guard, db_identity=ledger_identity,
        state_sha256=state_sha256, history_sha256=history_sha256, scope_sha256=scope_sha256,
        original_claims_sha256=original_claims_sha256, expected_row_sha256=checked['injection_claim_row_sha256'])
    if (type(receipt) is not dict or canonical(receipt) != receipt_raw
            or hashlib.sha256(intent_raw).hexdigest() != checked['intent_sha256']
            or claim_row_sha256(row) != checked['injection_claim_row_sha256'] or authority() is not True):
        raise ValueError('the one-use claim could not be reconciled')
    pid, birth = _process_identity(owner._current_process_identity())
    terminal = {'schema': 'mlg-injection-replay-claimant-terminal-v1', 'status': 'preclaimed',
        'parent_pid': pid, 'parent_birth_100ns': birth, 'plan_sha256': plan_sha256,
        'config_sha256': config_sha256, 'driver_sha256': driver_sha256,
        'intent_sha256': hashlib.sha256(intent_raw).hexdigest(),
        'receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(),
        'claim_row_sha256': checked['injection_claim_row_sha256']}
    _write_once(terminal_path, canonical(terminal))
    if (authority() is not True or _ordinary_source(str(parent / 'replay-claim-receipt.json'), 65536) != receipt_raw
            or _ordinary_source(str(parent / 'replay-claim-intent.json'), 65536) != intent_raw
            or guarded_claim_row(str(ledger_path), str(phase), guard, db_identity=ledger_identity,
                state_sha256=state_sha256, history_sha256=history_sha256, scope_sha256=scope_sha256,
                original_claims_sha256=original_claims_sha256,
                expected_row_sha256=terminal['claim_row_sha256']) != row):
        raise ValueError('the claim changed after the terminal was written')
    return terminal


def verify_exit_chain(exit_bytes, outer_bytes, terminal_bytes, *, purpose, observer_identity,
                      parent_identity, plan_sha256, config_sha256, parent_command_sha256,
                      parent_source_sha256, observer_command_sha256, observer_source_sha256,
                      terminal_check):
    """Check saved records against independently observed process identities.

    The caller must obtain both identities and the successful observer exit
    from retained process handles. This check cannot authenticate fabricated
    records or establish source provenance, publication, or scientific success.
    """
    observer_pid, observer_birth = _process_identity(observer_identity)
    parent_pid, parent_birth = _process_identity(parent_identity)
    digests = (plan_sha256, config_sha256, parent_command_sha256, parent_source_sha256,
               observer_command_sha256, observer_source_sha256)
    if (purpose not in ('validation', 'claim', 'replay') or not callable(terminal_check)
            or any(type(value) is not str or _HEX64.fullmatch(value) is None for value in digests)
            or type(terminal_bytes) is not bytes or not 0 < len(terminal_bytes) <= 2 * 1024 * 1024):
        raise ValueError('the observed process chain identity differs')
    _canonical_object(outer_bytes, 8192)
    expected_outer = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
        'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
        'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
        'parent_pid': parent_pid, 'parent_birth_100ns': parent_birth,
        'config_sha256': config_sha256, 'operation_sha256': plan_sha256,
        'command_sha256': parent_command_sha256, 'supervisor_source_sha256': parent_source_sha256,
        'supervisor_terminal_sha256': hashlib.sha256(terminal_bytes).hexdigest()}
    if outer_bytes != canonical(expected_outer) or terminal_check(terminal_bytes, parent_pid, parent_birth) is not True:
        raise ValueError('the observed parent terminal or cleanup differs')
    _canonical_object(exit_bytes, 8192)
    expected_exit = {'schema': 'mlg-injection-replay-observer-exit-v1', 'purpose': purpose,
        'status': 'completed', 'exit_code': 0, 'observer_reaped': True, 'observer_handle_closed': True,
        'job_drained': True, 'job_closed': True, 'active_job_members': 0,
        'observer_pid': observer_pid, 'observer_birth_100ns': observer_birth,
        'plan_sha256': plan_sha256, 'observer_command_sha256': observer_command_sha256,
        'observer_source_sha256': observer_source_sha256,
        'outer_observation_sha256': hashlib.sha256(outer_bytes).hexdigest()}
    if exit_bytes != canonical(expected_exit):
        raise ValueError('the observer process exit differs')
    return {'status': 'saved_chain_consistent', 'purpose': purpose}


def make_observed_exit(result, result_bytes, inner_bytes, terminal_bytes, *, purpose, plan_sha256,
                       parent_config_sha256, parent_command_sha256, parent_source_sha256,
                       observer_config_sha256, observer_command_sha256, observer_source_sha256,
                       terminal_check):
    """Build a private exit receipt after a separately retained observer returns.

    ``result`` must be the actual return from the verified outer observer, not
    a record reconstructed from disk. This helper checks consistency but does
    not itself retain a process handle or attest the caller's provenance.
    """
    outer = _canonical_object(result_bytes, 8192)
    inner = _canonical_object(inner_bytes, 8192)
    if type(result) is not dict or canonical(result) != result_bytes:
        raise ValueError('the observed supervisor result differs from its saved record')
    observer_identity = _process_identity((outer.get('parent_pid'), outer.get('parent_birth_100ns')))
    parent_identity = _process_identity((inner.get('parent_pid'), inner.get('parent_birth_100ns')))
    expected_outer = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
        'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
        'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
        'parent_pid': observer_identity[0], 'parent_birth_100ns': observer_identity[1],
        'config_sha256': observer_config_sha256, 'operation_sha256': plan_sha256,
        'command_sha256': observer_command_sha256, 'supervisor_source_sha256': observer_source_sha256,
        'supervisor_terminal_sha256': hashlib.sha256(inner_bytes).hexdigest()}
    if result_bytes != canonical(expected_outer):
        raise ValueError('the observer did not exit and clean up')
    receipt = {'schema': 'mlg-injection-replay-observer-exit-v1', 'purpose': purpose,
        'status': 'completed', 'exit_code': 0, 'observer_reaped': True, 'observer_handle_closed': True,
        'job_drained': True, 'job_closed': True, 'active_job_members': 0,
        'observer_pid': observer_identity[0], 'observer_birth_100ns': observer_identity[1],
        'plan_sha256': plan_sha256, 'observer_command_sha256': observer_command_sha256,
        'observer_source_sha256': observer_source_sha256,
        'outer_observation_sha256': hashlib.sha256(inner_bytes).hexdigest()}
    verify_exit_chain(canonical(receipt), inner_bytes, terminal_bytes, purpose=purpose,
        observer_identity=observer_identity, parent_identity=parent_identity, plan_sha256=plan_sha256,
        config_sha256=parent_config_sha256, parent_command_sha256=parent_command_sha256,
        parent_source_sha256=parent_source_sha256, observer_command_sha256=observer_command_sha256,
        observer_source_sha256=observer_source_sha256, terminal_check=terminal_check)
    return receipt


def module_graph(frozen_root, record_bytes, record_sha256, locations, captured, identities):
    """Capture the frozen package plus the separately bound operational graph.

    The caller must independently bind the record digest to the published
    discovery runtime and verify the detached Git tree before this helper.
    """
    if (type(frozen_root) is not str or not Path(frozen_root).is_absolute()
            or type(record_bytes) is not bytes or not 0 < len(record_bytes) <= 2 * 1024 * 1024
            or type(record_sha256) is not str or _HEX64.fullmatch(record_sha256) is None
            or hashlib.sha256(record_bytes).hexdigest() != record_sha256
            or any(type(item) is not dict or set(item) != set(SOURCE_SUFFIXES)
                   for item in (locations, captured, identities))):
        raise ValueError('frozen source graph identities differ')
    try:
        record = json.loads(record_bytes, object_pairs_hook=_unique)
        if (canonical(record) != record_bytes or type(record) is not dict
                or set(record) != {'git_commit', 'git_tree', 'files'}
                or record['git_commit'] != DISCOVERY_SOURCE or type(record['git_tree']) is not str
                or re.fullmatch(r'[0-9a-f]{40}', record['git_tree']) is None
                or type(record['files']) is not dict or not 0 < len(record['files']) <= 4096):
            raise ValueError('frozen source inventory differs')
    except (TypeError, UnicodeError, RecursionError) as error:
        raise ValueError('frozen source inventory differs') from error
    root = Path(frozen_root)
    entries, total = {}, 0
    for name, digest in record['files'].items():
        relative = PurePosixPath(name) if type(name) is str else None
        if (relative is None or not name or relative.is_absolute() or relative.as_posix() != name
                or '..' in relative.parts or '\\' in name or ':' in name
                or type(digest) is not str or _HEX64.fullmatch(digest) is None):
            raise ValueError('frozen source inventory contains an invalid path')
        if not name.startswith('looking_glass/') or not name.endswith('.py'):
            continue
        path = root.joinpath(*relative.parts)
        raw = _ordinary_source(str(path))
        total += len(raw)
        if total > 64 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('frozen package source differs')
        parts = list(relative.parts)
        is_package = parts[-1] == '__init__.py'
        parts = parts[:-1] if is_package else parts[:-1] + [parts[-1][:-3]]
        if not parts or not all(part.isidentifier() for part in parts):
            raise ValueError('frozen Python module path differs')
        module = '.'.join(parts)
        if module in entries:
            raise ValueError('duplicate frozen Python module')
        entries[module] = (str(path), raw, digest, is_package)
    if 'looking_glass' not in entries or entries['looking_glass'][3] is not True:
        raise ValueError('frozen project package is missing')
    empty_digest = hashlib.sha256(b'').hexdigest()
    for name in ('scripts', 'tools'):
        entries[name] = (str(root / name / '__init__.py'), b'', empty_digest, True)
    for name, suffix in SOURCE_SUFFIXES.items():
        item, raw, digest = locations[name], captured[name], identities[name]
        if (type(item) is not dict or set(item) != {'path', 'url'} or type(raw) is not bytes
                or type(digest) is not str or _HEX64.fullmatch(digest) is None
                or hashlib.sha256(raw).hexdigest() != digest or _ordinary_source(item['path']) != raw):
            raise ValueError('operational source changed during graph capture')
        module = suffix[:-3].replace('/', '.')
        entries[module] = (item['path'], raw, digest, False)
    return entries


def capture_preflight(*, plan_path, plan_url, source_locations, public_paths, frozen_root, cache_dir,
                      fetch=_fetch_source):
    """Capture the complete public/source preflight without a claim or model.

    The caller must subsequently acquire and retain file leases, repeat these
    observations under protection, and independently observe process cleanup.
    """
    plan, plan_bytes = capture_plan(plan_path, plan_url, fetch=fetch)
    sources = capture_sources(source_locations, plan['sources'], fetch=fetch)
    own_path = os.path.normcase(os.path.abspath(__file__))
    if (os.path.normcase(os.path.abspath(source_locations['driver']['path'])) != own_path
            or _ordinary_source(__file__) != sources['driver']):
        raise ValueError('the executing driver differs from the published plan')
    child = load_captured('_mlg_replay_plan_child', source_locations['child']['path'], sources['child'],
                          plan['sources']['child'])
    if canonical(child.validate_plan(plan_bytes, child_source_sha256=plan['sources']['child'])) != plan_bytes:
        raise ValueError('the fixed replay plan differs')
    public = capture_fixed_public_inputs(public_paths, plan, child, fetch=fetch)
    reference_paths = [Path(public_paths['reference_' + role]) for role in child._ROLES]
    if (not reference_paths or any(path.name != 'observation-00-' + role + '.json'
            or os.path.normcase(str(path.parent)) != os.path.normcase(str(reference_paths[0].parent))
            for path, role in zip(reference_paths, child._ROLES))):
        raise ValueError('the fixed public reference directory differs')
    runtime = child._json(public['discovery_runtime'])
    launcher, record = capture_frozen_repository(frozen_root, runtime, fetch=fetch)
    graph = module_graph(frozen_root, record, runtime['repository_source_sha256'], source_locations,
                         sources, plan['sources'])
    assets = check_assets(cache_dir, runtime['assets'])
    locations_bytes, public_paths_bytes = canonical(source_locations), canonical(public_paths)
    return {'plan': plan, 'plan_path': plan_path, 'plan_url': plan_url, 'plan_url_captured': plan_url,
            'plan_bytes': plan_bytes,
            'source_locations': json.loads(locations_bytes), 'source_locations_bytes': locations_bytes,
            'sources': sources, 'public_paths': json.loads(public_paths_bytes),
            'public_paths_bytes': public_paths_bytes, 'public': public,
            'frozen_root': frozen_root, 'frozen_record': record, 'launcher': launcher,
            'child': child, 'graph': graph, 'cache_dir': cache_dir, 'asset_manifest': runtime['assets'],
            'asset_paths': assets,
            'reference_dir': str(reference_paths[0].parent)}


def recheck_preflight(context, finder):
    """Repeat local checks under the caller's retained source and asset leases."""
    if (type(context) is not dict or finder.assert_intact() is not True
            or context['plan_url'] != context['plan_url_captured']
            or canonical(context['plan']) != context['plan_bytes']
            or canonical(context['source_locations']) != context['source_locations_bytes']
            or canonical(context['public_paths']) != context['public_paths_bytes']
            or _ordinary_source(context['plan_path'], 2 * 1024 * 1024) != context['plan_bytes']
            or recheck_sources(context['source_locations'], context['sources']) is not True
            or set(context['sources']) != set(SOURCE_SUFFIXES)
            or any(hashlib.sha256(raw).hexdigest() != context['plan']['sources'].get(name)
                   for name, raw in context['sources'].items())):
        raise ValueError('the captured replay preflight changed')
    specifications = fixed_public_specs(context['plan'], context['child'])
    if set(context['public']) != set(context['public_paths']) or set(context['public']) != set(specifications):
        raise ValueError('the captured public input set changed')
    for name, raw in context['public'].items():
        _, digest, limit = specifications[name]
        if (type(raw) is not bytes or not raw or len(raw) > limit or hashlib.sha256(raw).hexdigest() != digest
                or _ordinary_source(context['public_paths'][name], len(raw)) != raw):
            raise ValueError('a captured public input changed')
    record = context['launcher']._verify_unloaded_source(Path(context['frozen_root']), DISCOVERY_SOURCE)
    if (canonical(record) != context['frozen_record']
            or check_assets(context['cache_dir'], context['asset_manifest']) != context['asset_paths']):
        raise ValueError('the frozen repository or assets changed')
    return True


def bind_replay_config(context, child, values, prefix):
    """Bind the child configuration to the captured plan and public paths."""
    if (type(context) is not dict or type(values) is not dict or type(prefix) is not str
            or set(values) != (child.CONFIG_PATHS - {'pycache_prefix'}) | {'schema', 'plan_url'}):
        raise ValueError('the replay child configuration differs')
    expected = {'schema': child.CONFIG_SCHEMA, 'plan_url': context['plan_url'],
                'frozen_root': context['frozen_root'], 'cache_dir': context['cache_dir'],
                'replay_plan_file': context['plan_path'], 'reference_dir': context['reference_dir'],
                'producer_source_file': context['source_locations']['producer']['path']}
    for name in ('discovery_runtime', 'heldout_runtime', 'discovery_plan', 'runtime_deviation'):
        expected[name + '_file'] = context['public_paths'][name]
    if any(type(values.get(key)) is not str or values[key] != value for key, value in expected.items()):
        raise ValueError('the child configuration is not bound to the captured inputs')
    config = dict(values, pycache_prefix=prefix)
    raw = canonical(config)
    if len(raw) > 8192 or canonical(child.parse_config(raw)) != raw:
        raise ValueError('the bounded child configuration differs')
    return config


def lease_inputs(context, workspace, *, extra_files=()):
    """List the exact existing files whose leases must cover a replay launch."""
    if (type(context) is not dict or type(workspace) is not str or not Path(workspace).is_absolute()
            or '..' in Path(workspace).parts or type(extra_files) not in (tuple, list)
            or any(type(path) is not str for path in extra_files)):
        raise ValueError('the replay lease inventory arguments differ')
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir() or os.path.normcase(str(root)) != os.path.normcase(os.path.abspath(workspace)):
        raise ValueError('the replay workspace is not canonical')
    def checked(path):
        path = Path(path)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('a replay lease path is not absolute')
        for component in (path, *path.parents):
            info = component.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('a replay lease path is linked')
        info = path.lstat()
        resolved = path.resolve(strict=True)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not resolved.is_relative_to(root)
                or os.path.normcase(str(resolved)) != os.path.normcase(os.path.abspath(str(path)))):
            raise ValueError('a replay lease file is unavailable or outside the workspace')
        return resolved
    record = _canonical_object(context['frozen_record'], 2 * 1024 * 1024)
    if (record.get('git_commit') != DISCOVERY_SOURCE or type(record.get('files')) is not dict
            or not 0 < len(record['files']) <= 4096):
        raise ValueError('the frozen lease inventory differs')
    frozen_root = Path(context['frozen_root'])
    frozen_files = []
    for name in record['files']:
        relative = PurePosixPath(name) if type(name) is str else None
        if (relative is None or not name or relative.is_absolute() or relative.as_posix() != name
                or '..' in relative.parts or '\\' in name or ':' in name):
            raise ValueError('a frozen lease path differs')
        frozen_files.append(checked(frozen_root.joinpath(*relative.parts)))
    operation = [item['path'] for item in context['source_locations'].values()]
    operation.extend(context['public_paths'].values())
    operation.extend((context['plan_path'], *extra_files))
    def unique(values):
        result = {}
        for value in values:
            path = checked(value)
            result[os.path.normcase(str(path))] = path
        if not result or len(result) > 4096:
            raise ValueError('the replay lease file count differs')
        return tuple(result[key] for key in sorted(result))
    return unique(frozen_files), unique(operation), unique(context['asset_paths'])


def _execution_modules(finder):
    """Load every execution primitive from the active captured-source graph."""
    names = {'owner': 'tools.injection_runtime_replay_owner',
             'child': 'tools.injection_runtime_replay_child',
             'adapter': 'tools.injection_heldout_owner',
             'guard': 'scripts.v2_source_guards',
             'supervisor': 'looking_glass.natural_heldout_supervisor',
             'cpu_job': 'scripts.number_cpu_outer',
             'outer_observer': 'looking_glass.injection_outer_observer'}
    if type(finder) is not _BoundImports or finder.assert_intact() is not True:
        raise ValueError('the captured execution graph is not active')
    result = {key: importlib.import_module(name) for key, name in names.items()}
    if (finder.assert_intact() is not True or any(sys.modules.get(name) is not result[key]
            or getattr(getattr(result[key], '__spec__', None), 'loader', None) is not finder
            for key, name in names.items())):
        raise ValueError('a replay execution primitive is not source-bound')
    return result


def _fresh_workspace_path(path, root):
    if type(path) is not str or not Path(path).is_absolute() or '..' in Path(path).parts:
        raise ValueError('a fresh workspace path is required')
    target = Path(path)
    if os.path.lexists(target):
        raise ValueError('a replay output already exists')
    for component in (target.parent, *target.parent.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('a replay output parent is linked')
    parent = target.parent.resolve(strict=True)
    if (not parent.is_dir() or not parent.is_relative_to(root)
            or os.path.normcase(str(parent)) != os.path.normcase(os.path.abspath(str(target.parent)))):
        raise ValueError('a replay output parent is outside the protected workspace')
    return target


def check_replay_output_paths(workspace, config_path, output, terminal_path, postcheck, attempt):
    """Require separate, absent outputs under the protected workspace."""
    if type(workspace) is not str or not Path(workspace).is_absolute():
        raise ValueError('the replay workspace differs')
    root = Path(workspace).resolve(strict=True)
    paths = tuple(_fresh_workspace_path(path, root) for path in
                  (config_path, output, terminal_path, postcheck, attempt))
    normalized = tuple(os.path.normcase(str(path)) for path in paths)
    if len(set(normalized)) != len(paths):
        raise ValueError('the replay output paths overlap')
    return True


def verify_parent_command(raw, context, owner, cache_parent, purpose, config_path, *, role='parent'):
    """Reconstruct the source-bound direct-base parent command from saved bytes."""
    if (type(raw) is not bytes or not 0 < len(raw) <= 8192
            or purpose not in ('validation', 'claim', 'replay') or role not in ('parent', 'observer', 'operator')
            or (role != 'parent' and purpose != 'validation')):
        raise ValueError('the saved replay-parent command differs')
    try:
        command = json.loads(raw, object_pairs_hook=_unique)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError('the saved replay-parent command is invalid') from error
    if (type(command) is not list or len(command) != 12 or canonical(command) != raw
            or any(type(value) is not str or not value for value in command)
            or not command[5].startswith('pycache_prefix=')):
        raise ValueError('the saved replay-parent command differs')
    prefix = Path(command[5][len('pycache_prefix='):])
    if (not prefix.is_absolute() or prefix.parent != Path(cache_parent)
            or not prefix.name.startswith('v2-cache-') or '..' in prefix.parts):
        raise ValueError('the replay-parent bytecode prefix differs')
    expected = [str(sys._base_executable), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                '-c', owner._BOOTSTRAP, context['source_locations']['driver']['path'],
                context['plan']['sources']['driver'], '--' + purpose + '-' + role, config_path]
    if command != expected:
        raise ValueError('the replay-parent command is not source-bound')
    return command


def private_replay_inputs(config_values, validation_only, extra_files):
    """Lease the site and claim files only for the actual private replay."""
    if (type(config_values) is not dict or type(validation_only) is not bool
            or type(extra_files) not in (tuple, list)):
        raise ValueError('the private replay lease arguments differ')
    required = () if validation_only else tuple(config_values.get(key) for key in
        ('site_nonce_file', 'site_record_file', 'claim_receipt_file'))
    paths = (*extra_files, *required)
    if any(type(path) is not str or not Path(path).is_absolute() or '..' in Path(path).parts for path in paths):
        raise ValueError('a private replay lease path differs')
    return paths


def run_protected_replay(context, finder, *, workspace, cache_parent, config_values, config_path,
                         output, terminal_path, environment, authority, validation_only,
                         extra_files=(), timeout=7200):
    """Hold all source/asset/config handles through one gated child and reap.

    The caller must independently bind the publication, ACL, budget, claim,
    and observed claimant exit in ``authority``. The returned terminal is not
    an observation of this parent process's own exit or a scientific verdict.
    """
    if (type(validation_only) is not bool or not callable(authority) or type(config_values) is not dict
            or type(environment) is not dict or type(timeout) not in (int, float)
            or isinstance(timeout, bool) or not 0 < timeout <= 7200):
        raise ValueError('the protected replay invocation differs')
    modules = _execution_modules(finder)
    owner, guard, child = modules['owner'], modules['guard'], modules['child']
    root = Path(workspace).resolve(strict=True)
    check_replay_output_paths(workspace, config_path, output, terminal_path,
                              config_values.get('postcheck_file'), config_values.get('attempt_dir'))
    cache_parent = Path(cache_parent)
    if (not cache_parent.is_absolute() or not cache_parent.is_dir()
            or not cache_parent.resolve(strict=True).is_relative_to(root)
            or os.path.normcase(str(cache_parent.resolve(strict=True))) != os.path.normcase(os.path.abspath(str(cache_parent)))):
        raise ValueError('the protected bytecode parent differs')
    protected_extra = private_replay_inputs(config_values, validation_only, extra_files)
    frozen, operational, assets = lease_inputs(context, workspace, extra_files=protected_extra)
    leases = [guard.SourceFileLease(root, files) for files in (frozen, operational, assets)]
    cache = guard.PrivateEmptyCache(cache_parent)
    leases.append(cache)
    values_bytes, environment_bytes = canonical(config_values), canonical(environment)
    child_environment = json.loads(environment_bytes)
    state = {}
    def precheck(acquired):
        if (recheck_preflight(context, finder) is not True
                or check_runtime_environment(child_environment, workspace, context['cache_dir'],
                    _canonical_object(context['public']['heldout_runtime'], 1024 * 1024)) is not True
                or canonical(config_values) != values_bytes or canonical(environment) != environment_bytes):
            raise ValueError('the protected replay inputs changed')
        config = bind_replay_config(context, child, config_values, str(cache.path))
        config_raw = canonical(config)
        _write_once(config_path, config_raw)
        config_lease = guard.SourceFileLease(root, [Path(config_path)])
        acquired.append(config_lease)
        config_lease.__enter__()
        handles = tuple(handle for lease in acquired for handle in lease.handles())
        if (not 0 < len(handles) <= 8193 or len(set(handles)) != len(handles)
                or _ordinary_source(config_path, 8192) != config_raw
                or authority('before_launch', json.loads(config_raw)) is not True
                or canonical(environment) != environment_bytes or canonical(config_values) != values_bytes
                or recheck_preflight(context, finder) is not True):
            raise ValueError('the protected replay authority differs before launch')
        state.update({'config': config, 'config_raw': config_raw})
        return True
    def launch(acquired):
        return owner.launch_child(child=child, child_path=context['source_locations']['child']['path'],
            child_sha256=context['plan']['sources']['child'], config_path=config_path,
            expected_config=state['config_raw'], plan_bytes=context['plan_bytes'],
            base_python=sys._base_executable, output=output, environment=child_environment,
            guard=guard, adapter=modules['adapter'], supervisor=modules['supervisor'],
            cpu_job_type=modules['cpu_job'].WindowsJob, acquired=acquired,
            validation_only=validation_only, timeout=timeout)
    def postcheck(result):
        if (recheck_preflight(context, finder) is not True
                or canonical(environment) != environment_bytes or canonical(child_environment) != environment_bytes
                or canonical(config_values) != values_bytes
                or _ordinary_source(config_path, 8192) != state['config_raw']
                or authority('after_child', json.loads(state['config_raw'])) is not True):
            raise ValueError('the protected replay authority changed after the child')
        terminal = owner.check_supervisor_record(child, result['config'], result['plan_sha256'],
            context['plan']['sources']['child'], output, result['command'], result['verdict'],
            validation_only=validation_only)
        _write_once(terminal_path, canonical(terminal))
        if (recheck_preflight(context, finder) is not True
                or _ordinary_source(config_path, 8192) != state['config_raw']
                or authority('after_child', json.loads(state['config_raw'])) is not True):
            raise ValueError('the protected replay changed after terminal creation')
        return terminal
    return owner.run_guarded(leases, precheck, launch, postcheck, retained=modules)


def observe_replay_parent(context, finder, *, workspace, cache_parent, purpose, config_path,
                          terminal_path, output, command_path, environment, authority, terminal_check,
                          extra_files=(), timeout=7200, role='parent'):
    """Retain and reap one gated replay parent under the reviewed outer Job.

    The trusted caller must supply the exact parent config and independently
    verify its semantics, budget and prerequisites in ``authority``. The
    returned observation does not establish this observer's own clean exit.
    """
    if (purpose not in ('validation', 'claim', 'replay') or role not in ('parent', 'observer', 'operator')
            or (role != 'parent' and purpose != 'validation') or type(environment) is not dict
            or not callable(authority) or not callable(terminal_check) or type(timeout) is not int
            or not 1 <= timeout <= 7200 or type(extra_files) not in (tuple, list)):
        raise ValueError('the replay-parent observation arguments differ')
    modules = _execution_modules(finder)
    owner, guard, observer = modules['owner'], modules['guard'], modules['outer_observer']
    root = Path(workspace).resolve(strict=True)
    outer_path = _fresh_workspace_path(output, root)
    terminal = _fresh_workspace_path(terminal_path, root)
    command_file = _fresh_workspace_path(command_path, root)
    if (len({os.path.normcase(str(path)) for path in (outer_path, terminal, command_file)}) != 3
            or observer._QUARANTINED):
        raise ValueError('the observer output or prior custody differs')
    config_raw = _ordinary_source(config_path, 65536)
    plan_raw = context['plan_bytes']
    if type(plan_raw) is not bytes or not plan_raw or len(plan_raw) > 2 * 1024 * 1024:
        raise ValueError('the published replay plan is unavailable')
    frozen, operational, assets = lease_inputs(context, workspace, extra_files=(*extra_files, config_path))
    leases = [guard.SourceFileLease(root, files) for files in (frozen, operational, assets)]
    cache = guard.PrivateEmptyCache(cache_parent)
    leases.append(cache)
    environment_raw = canonical(environment)
    child_environment = json.loads(environment_raw)
    state = {}
    def source_check():
        return (canonical(environment) == environment_raw and canonical(child_environment) == environment_raw
                and _ordinary_source(config_path, 65536) == config_raw
                and (state.get('command_raw') is None or
                     _ordinary_source(command_path, 8192) == state['command_raw'])
                and recheck_preflight(context, finder) is True
                and check_runtime_environment(child_environment, workspace, context['cache_dir'],
                    _canonical_object(context['public']['heldout_runtime'], 1024 * 1024)) is True
                and authority() is True)
    def precheck(acquired):
        if source_check() is not True:
            raise ValueError('the replay-parent authority differs')
        source = context['source_locations']['driver']['path']
        source_sha256 = context['plan']['sources']['driver']
        if (hashlib.sha256(_ordinary_source(source)).hexdigest() != source_sha256
                or _ordinary_source(context['plan_path'], 2 * 1024 * 1024) != plan_raw):
            raise ValueError('the replay-parent source or plan differs')
        command = [str(sys._base_executable), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(cache.path),
                   '-c', owner._BOOTSTRAP, source, source_sha256, '--' + purpose + '-' + role, config_path]
        if sum(len(part) for part in command) > 16000:
            raise ValueError('the replay-parent command exceeds its bound')
        command_raw = canonical(command)
        if len(command_raw) > 8192:
            raise ValueError('the saved replay-parent command exceeds its bound')
        _write_once(command_path, command_raw)
        command_lease = guard.SourceFileLease(root, [command_file])
        acquired.append(command_lease)
        command_lease.__enter__()
        state['command_raw'] = command_raw
        verify_parent_command(command_raw, context, owner, str(cache_parent), purpose, config_path, role=role)
        state['command'] = command
        return True
    def launch(acquired):
        def protect(process):
            handles = tuple(handle for lease in acquired for handle in lease.handles())
            duplicated = guard.duplicate_to_process(handles, process)
            return (type(duplicated) is tuple and len(duplicated) == len(handles)
                    and all(type(handle) is int and handle > 0 for handle in duplicated))
        def check_terminal(raw, observed_purpose, pid, birth):
            return observed_purpose == 'execution' and terminal_check(raw, pid, birth) is True
        # The outer Job permits breakaway. Even if the observer reaped its
        # direct parent, an abnormal exit cannot prove that all helpers or
        # separately supervised descendants have stopped. Keep the owner's
        # rooted protections on every exception; do not infer cleanup from the
        # outer observer's own quarantine list.
        result = observer.observe_supervisor(purpose='execution', command=state['command'], output=output,
            environment=child_environment, config_path=config_path, config_sha256=hashlib.sha256(config_raw).hexdigest(),
            operation_path=context['plan_path'], operation_sha256=hashlib.sha256(plan_raw).hexdigest(),
            command_sha256=hashlib.sha256(canonical(state['command'])).hexdigest(),
            supervisor_source_sha256=context['plan']['sources']['driver'], terminal_path=terminal_path,
            start_token=START_TOKEN, timeout=timeout, source_check=source_check,
            protect_process=protect, terminal_check=check_terminal)
        return {'cleanup_verified': True, 'observation': result}
    def postcheck(result):
        observation = result['observation']
        observed_raw = _ordinary_source(str(Path(output) / 'observation.json'), 8192)
        if (type(observation) is not dict or canonical(observation) != observed_raw
                or source_check() is not True):
            raise ValueError('the replay-parent observation changed')
        return observation
    return owner.run_guarded(leases, precheck, launch, postcheck, retained=modules)


def load_captured(name, path, raw, expected_sha256):
    """Compile only captured, hash-verified Python bytes, never cached bytecode."""
    if (type(name) is not str or not all(part.isidentifier() for part in name.split('.'))
            or name in sys.modules or type(raw) is not bytes or not 0 < len(raw) <= _MAX_SOURCE
            or type(expected_sha256) is not str or _HEX64.fullmatch(expected_sha256) is None
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError('captured source identity differs')
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class _BoundImports:
    """An explicit, source-only module graph for one isolated parent process."""
    def __init__(self, entries, roots):
        if (type(roots) is not tuple or not roots or len(set(roots)) != len(roots)
                or any(type(root) is not str or not root.isidentifier() for root in roots)):
            raise ValueError('invalid project import roots')
        if any(name.split('.')[0] in roots for name in sys.modules):
            raise ValueError('a project module was imported before source verification')
        baseline = [importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                    importlib.machinery.PathFinder]
        if sys.meta_path != baseline:
            raise ValueError('the isolated import finder list differs')
        base = Path(sys.base_prefix).resolve(strict=True)
        if (not sys.path or any(type(path) is not str or not path or not Path(path).is_absolute()
                or not Path(path).resolve(strict=False).is_relative_to(base) for path in sys.path)):
            raise ValueError('the isolated import path differs')
        if type(entries) is not dict or not entries or len(entries) > 4096:
            raise ValueError('a bounded source import graph is required')
        checked = {}
        for name, entry in entries.items():
            if (type(name) is not str or not all(part.isidentifier() for part in name.split('.'))
                    or name.split('.')[0] not in roots or type(entry) is not tuple or len(entry) != 4):
                raise ValueError('a source import entry is invalid')
            path, raw, digest, is_package = entry
            if (type(path) is not str or not Path(path).is_absolute() or type(raw) is not bytes
                    or len(raw) > _MAX_SOURCE or type(digest) is not str or _HEX64.fullmatch(digest) is None
                    or hashlib.sha256(raw).hexdigest() != digest or type(is_package) is not bool):
                raise ValueError('a source import entry differs from its digest')
            checked[name] = (path, raw, digest, is_package)
        for root in roots:
            if root not in checked or checked[root][3] is not True:
                raise ValueError('each source import root must be a package')
        for name in checked:
            parent = name.rpartition('.')[0]
            if parent and (parent not in checked or checked[parent][3] is not True):
                raise ValueError('a source import parent package is missing')
        self._entries = MappingProxyType(checked)
        self.roots = roots
        self.baseline = baseline
        self.path = tuple(sys.path)
        self.active = False

    @property
    def entries(self):
        return self._entries

    def __enter__(self):
        if self.active or sys.meta_path != self.baseline or tuple(sys.path) != self.path:
            raise ValueError('the source import context changed before entry')
        if any(name.split('.')[0] in self.roots for name in sys.modules):
            raise ValueError('a project module was imported before source verification')
        self.active = True
        sys.meta_path.insert(0, self)
        return self

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in self.roots:
            return None
        if fullname not in self.entries:
            raise ModuleNotFoundError('the requested project source is not bound', name=fullname)
        source_path, _, _, is_package = self.entries[fullname]
        spec = importlib.util.spec_from_loader(fullname, self, origin=source_path, is_package=is_package)
        if spec is None:
            raise ImportError('cannot establish the bound project import')
        spec.has_location = True
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        if name not in self.entries or module.__spec__.loader is not self:
            raise ImportError('the project module is not bound to this loader')
        path, raw, digest, _ = self.entries[name]
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ImportError('captured project source changed')
        exec(compile(raw, path, 'exec', dont_inherit=True, optimize=0), module.__dict__)

    def assert_intact(self):
        if (not self.active or sys.meta_path != [self, *self.baseline] or tuple(sys.path) != self.path):
            raise ValueError('the verified source import context changed')
        for name, module in tuple(sys.modules.items()):
            if name.split('.')[0] in self.roots:
                if (name not in self.entries or getattr(getattr(module, '__spec__', None), 'loader', None) is not self):
                    raise ValueError('an unbound project module was loaded')
        return True

    def __exit__(self, exc_type, exc, traceback):
        error = None
        try:
            self.assert_intact()
        except BaseException as caught:
            error = caught
        finally:
            sys.meta_path[:] = self.baseline
            for name in tuple(sys.modules):
                if name.split('.')[0] in self.roots:
                    del sys.modules[name]
            self.active = False
        if error is not None:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise exc from error
            raise error from exc
        return False


def run_bound_imports(entries, callback, *, roots=('looking_glass', 'scripts', 'tools')):
    """Run once under a pre-rooted, closed captured-source import graph.

    The enclosing ``finally`` also removes the finder if interruption occurs
    after installation but before ``__enter__`` returns to its caller.
    """
    if not callable(callback):
        raise ValueError('a source-bound callback is required')
    finder = _BoundImports(entries, roots)
    try:
        finder.__enter__()
        return callback(finder)
    finally:
        if finder.active:
            finder.__exit__(*sys.exc_info())


def parse_validation_parent_config(raw, config_path):
    """Parse the path-only no-model parent configuration."""
    value = _canonical_object(raw, 65536)
    keys = {'schema', 'workspace', 'plan_path', 'plan_url', 'source_locations', 'public_paths',
            'frozen_root', 'cache_dir', 'run_dir', 'parent_identity', 'ledger_identity'}
    if (set(value) != keys or value['schema'] != 'mlg-injection-replay-validation-parent-config-v1'
            or type(value['source_locations']) is not dict or type(value['public_paths']) is not dict
            or any(type(value[name]) is not str or not Path(value[name]).is_absolute()
                   or '..' in Path(value[name]).parts for name in
                   ('workspace', 'plan_path', 'frozen_root', 'cache_dir', 'run_dir'))
            or type(value['plan_url']) is not str
            or re.fullmatch(re.escape(RAW_PREFIX) + r'[0-9a-f]{40}' + re.escape(PLAN_PATH), value['plan_url']) is None
            or any(type(value[name]) is not list or len(value[name]) != 2
                   or any(type(item) is not int or item <= 0 for item in value[name])
                   for name in ('parent_identity', 'ledger_identity'))):
        raise ValueError('the validation-parent configuration differs')
    root, run = Path(value['workspace']), Path(value['run_dir'])
    diagnostic = root / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    if (not root.is_dir() or not run.is_dir() or run.parent != diagnostic
            or not re.fullmatch(r'validation-[a-z0-9-]{1,48}', run.name)
            or Path(config_path) != run / 'parent-config.json'):
        raise ValueError('the validation-parent diagnostic path differs')
    for path in (root, diagnostic, run):
        for part in (path, *path.parents):
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('the validation-parent directory is linked')
        if os.path.normcase(str(path.resolve(strict=True))) != os.path.normcase(os.path.abspath(str(path))):
            raise ValueError('the validation-parent directory is not canonical')
    return value


def check_unclaimed_validation(workspace, ledger_identity):
    """Verify that a no-model rehearsal precedes the one-use injection claim."""
    if type(workspace) is not str or not Path(workspace).is_absolute():
        raise ValueError('the validation workspace differs')
    root = Path(workspace)
    ledger = _database_identity(str(root / 'work' / 'agreement-study-ledger-v5' / 'ledger.sqlite3'), ledger_identity)
    parent = root / 'work' / 'number-injection-heldout-cpu-v1'
    paths = (parent / ('injection-heldout-' + 'c327e191214cfb661b429c6d0a66900ca773c148782e1cc88be669c54deaee98'),
             parent / 'runtime-compatibility-replay-v1', parent / 'replay-claim-intent.json',
             parent / 'replay-claim-receipt.json', parent / 'replay-claimant-terminal.json')
    if any(os.path.lexists(path) for path in paths):
        raise ValueError('the injection claim or its attempt already exists')
    database = sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True, timeout=30)
    try:
        rows = database.execute('SELECT 1 FROM claims WHERE phase=? LIMIT 1', ('injection_heldout',)).fetchall()
    except BaseException as primary:
        try:
            database.close()
        except BaseException as cleanup:
            if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise primary from cleanup
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary from cleanup
        raise
    else:
        database.close()
    if rows or _database_identity(str(ledger), ledger_identity) != ledger or any(os.path.lexists(path) for path in paths):
        raise ValueError('the injection claim appeared during validation')
    return True


def _assert_validation_parent_isolated():
    flags = sys.flags
    if (not (flags.isolated and flags.ignore_environment and flags.no_site and flags.no_user_site
             and flags.safe_path and sys.dont_write_bytecode and flags.optimize == 0)
            or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()
            or Path(sys.executable).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)):
        raise ValueError('the validation parent is not isolated')
    prefix = Path(sys.pycache_prefix)
    for part in (prefix, *prefix.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('the validation parent bytecode prefix is linked')
    if not prefix.is_dir() or any(prefix.iterdir()):
        raise ValueError('the validation parent bytecode prefix is not empty')


def validation_child_values(context, config, child):
    """Construct the fixed no-model child paths without reading private inputs."""
    workspace, run = Path(config['workspace']), Path(config['run_dir'])
    private = workspace / 'work' / 'number-injection-cpu-study-v1' / 'private-custody'
    parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
    return {'schema': child.CONFIG_SCHEMA, 'plan_url': config['plan_url'],
        'frozen_root': config['frozen_root'], 'site_packages': str(workspace / 'work' / '.venv' / 'Lib' / 'site-packages'),
        'cache_dir': config['cache_dir'], 'heldout_runtime_file': config['public_paths']['heldout_runtime'],
        'discovery_runtime_file': config['public_paths']['discovery_runtime'],
        'discovery_plan_file': config['public_paths']['discovery_plan'],
        'runtime_deviation_file': config['public_paths']['runtime_deviation'],
        'replay_plan_file': config['plan_path'], 'producer_source_file': config['source_locations']['producer']['path'],
        'site_nonce_file': str(private / 'nonce.bin'), 'site_record_file': str(private / 'site-record.json'),
        'claim_receipt_file': str(parent / 'replay-claim-receipt.json'), 'reference_dir': context['reference_dir'],
        'attempt_dir': str(run / 'attempt'), 'postcheck_file': str(run / 'child-postcheck.json')}


def _validation_supervisor(config):
    output = Path(config['run_dir']) / 'child-supervisor'
    try:
        verdict = json.loads(_ordinary_source(str(output / 'supervisor.json'), 2 * 1024 * 1024),
                             object_pairs_hook=_unique)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError('the saved child supervisor is invalid') from error
    if (type(verdict) is not dict or type(verdict.get('child_pid')) is not int
            or not 0 < verdict['child_pid'] <= 0xffffffff):
        raise ValueError('the saved validation child identity differs')
    return output, verdict


def check_validation_parent_terminal(context, config, child, owner, raw, pid, birth):
    """Rebuild and validate the nested no-model child receipt after parent exit."""
    _process_identity((pid, birth))
    run = Path(config['run_dir'])
    config_path = str(run / 'child-config.json')
    config_raw = _ordinary_source(config_path, 8192)
    observed = child.parse_config(config_raw)
    prefix = Path(observed['pycache_prefix'])
    if (not prefix.is_absolute() or prefix.parent != run or '..' in prefix.parts
            or not prefix.name.startswith('v2-cache-')):
        raise ValueError('the saved child bytecode prefix differs')
    expected = bind_replay_config(context, child, validation_child_values(context, config, child),
                                  observed['pycache_prefix'])
    if canonical(expected) != config_raw:
        raise ValueError('the observed child configuration differs')
    source_sha256 = context['plan']['sources']['child']
    command = owner.child_command(sys._base_executable, observed['pycache_prefix'],
        context['source_locations']['child']['path'], source_sha256, config_path, validation_only=True)
    output, verdict = _validation_supervisor(config)
    if verdict['child_pid'] == pid:
        raise ValueError('the validation parent and child identities overlap')
    terminal = owner.check_supervisor_record(child, observed, hashlib.sha256(context['plan_bytes']).hexdigest(),
        source_sha256, output, command, verdict, validation_only=True, parent_identity=(pid, birth))
    if type(raw) is not bytes or raw != canonical(terminal):
        raise ValueError('the observed validation-parent terminal differs')
    return True


def make_validation_observer_terminal(context, config, child, owner, observation_raw, pid, birth):
    """Bind a clean nested validation observation to this observer identity."""
    _process_identity((pid, birth))
    run = Path(config['run_dir'])
    config_path = str(run / 'parent-config.json')
    config_raw = _ordinary_source(config_path, 65536)
    parent_raw = _ordinary_source(str(run / 'parent-terminal.json'), 8192)
    command_raw = _ordinary_source(str(run / 'parent-command.json'), 8192)
    verify_parent_command(command_raw, context, owner, str(run), 'validation', config_path)
    observation = _canonical_object(observation_raw, 8192)
    parent_pid, parent_birth = _process_identity((observation.get('parent_pid'), observation.get('parent_birth_100ns')))
    _, supervisor = _validation_supervisor(config)
    if pid in (parent_pid, supervisor['child_pid']):
        raise ValueError('the observer and nested process identities overlap')
    expected = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
        'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
        'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
        'parent_pid': parent_pid, 'parent_birth_100ns': parent_birth,
        'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'operation_sha256': hashlib.sha256(context['plan_bytes']).hexdigest(),
        'command_sha256': hashlib.sha256(command_raw).hexdigest(),
        'supervisor_source_sha256': context['plan']['sources']['driver'],
        'supervisor_terminal_sha256': hashlib.sha256(parent_raw).hexdigest()}
    if (canonical(expected) != observation_raw
            or check_validation_parent_terminal(context, config, child, owner, parent_raw, parent_pid, parent_birth) is not True):
        raise ValueError('the nested validation observation differs')
    return {'schema': 'mlg-injection-replay-validation-observer-terminal-v1',
        'status': 'validated_no_model_forward', 'observer_pid': pid, 'observer_birth_100ns': birth,
        'plan_sha256': hashlib.sha256(context['plan_bytes']).hexdigest(),
        'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'parent_observation_sha256': hashlib.sha256(observation_raw).hexdigest(),
        'parent_terminal_sha256': hashlib.sha256(parent_raw).hexdigest(),
        'parent_command_sha256': hashlib.sha256(command_raw).hexdigest(),
        'driver_sha256': context['plan']['sources']['driver'], 'forward_count': 0}


def make_validation_exit_receipt(context, config, child, owner, outer, outer_raw, operator_pid, operator_birth):
    """Bind the retained observer result; the operator's own exit is separate."""
    _process_identity((operator_pid, operator_birth))
    _canonical_object(outer_raw, 8192)
    if type(outer) is not dict or canonical(outer) != outer_raw:
        raise ValueError('the retained validation observer result differs')
    run = Path(config['run_dir'])
    config_path = str(run / 'parent-config.json')
    config_raw = _ordinary_source(config_path, 65536)
    command_raw = _ordinary_source(str(run / 'observer-command.json'), 8192)
    verify_parent_command(command_raw, context, owner, str(run), 'validation', config_path, role='observer')
    observer_raw = _ordinary_source(str(run / 'observer-terminal.json'), 8192)
    parent_observation = _ordinary_source(str(run / 'parent-observer' / 'observation.json'), 8192)
    observer_pid, observer_birth = _process_identity((outer.get('parent_pid'), outer.get('parent_birth_100ns')))
    inner = _canonical_object(parent_observation, 8192)
    parent_pid, _ = _process_identity((inner.get('parent_pid'), inner.get('parent_birth_100ns')))
    _, supervisor = _validation_supervisor(config)
    if operator_pid in (observer_pid, parent_pid, supervisor['child_pid']):
        raise ValueError('the operator and nested process identities overlap')
    expected = {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
        'status': 'completed', 'exit_code': 0, 'parent_reaped': True, 'parent_handle_closed': True,
        'outer_job_drained': True, 'outer_job_closed': True, 'active_job_members': 0,
        'parent_pid': observer_pid, 'parent_birth_100ns': observer_birth,
        'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'operation_sha256': hashlib.sha256(context['plan_bytes']).hexdigest(),
        'command_sha256': hashlib.sha256(command_raw).hexdigest(),
        'supervisor_source_sha256': context['plan']['sources']['driver'],
        'supervisor_terminal_sha256': hashlib.sha256(observer_raw).hexdigest()}
    terminal = make_validation_observer_terminal(context, config, child, owner, parent_observation,
                                                 observer_pid, observer_birth)
    if canonical(expected) != outer_raw or canonical(terminal) != observer_raw:
        raise ValueError('the observer exit or its nested evidence differs')
    return {'schema': 'mlg-injection-replay-validation-exit-v1', 'status': 'awaiting_operator_exit',
        'operator_pid': operator_pid, 'operator_birth_100ns': operator_birth,
        'observer_pid': observer_pid, 'observer_birth_100ns': observer_birth,
        'plan_sha256': hashlib.sha256(context['plan_bytes']).hexdigest(),
        'config_sha256': hashlib.sha256(config_raw).hexdigest(),
        'observer_command_sha256': hashlib.sha256(command_raw).hexdigest(),
        'observer_terminal_sha256': hashlib.sha256(observer_raw).hexdigest(),
        'outer_observation_sha256': hashlib.sha256(outer_raw).hexdigest(),
        'parent_observation_sha256': hashlib.sha256(parent_observation).hexdigest(),
        'driver_sha256': context['plan']['sources']['driver'], 'forward_count': 0}


def check_validation_exit_receipt(context, config, child, owner, raw, operator_pid, operator_birth):
    """Check an operator terminal against the retained live process identity."""
    run = Path(config['run_dir'])
    outer_raw = _ordinary_source(str(run / 'observer-observer' / 'observation.json'), 8192)
    outer = _canonical_object(outer_raw, 8192)
    inner = _canonical_object(_ordinary_source(str(run / 'parent-observer' / 'observation.json'), 8192), 8192)
    _, supervisor = _validation_supervisor(config)
    if os.getpid() in (operator_pid, outer.get('parent_pid'), inner.get('parent_pid'), supervisor['child_pid']):
        raise ValueError('the root observer and nested process identities overlap')
    expected = make_validation_exit_receipt(context, config, child, owner, outer, outer_raw,
                                             operator_pid, operator_birth)
    if type(raw) is not bytes or raw != canonical(expected):
        raise ValueError('the observed validation operator terminal differs')
    return True


def run_validation_parent(config_path):
    """Run the exact source-bound, no-claim and no-model validation child."""
    _assert_validation_parent_isolated()
    config_raw = _ordinary_source(config_path, 65536)
    config = parse_validation_parent_config(config_raw, config_path)
    workspace, run = Path(config['workspace']), Path(config['run_dir'])
    diagnostic = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    context = capture_preflight(plan_path=config['plan_path'], plan_url=config['plan_url'],
        source_locations=config['source_locations'], public_paths=config['public_paths'],
        frozen_root=config['frozen_root'], cache_dir=config['cache_dir'])
    def execute(finder):
        modules = _execution_modules(finder)
        owner, child = modules['owner'], modules['child']
        claim = importlib.import_module('tools.injection_replay_claim')
        if (getattr(getattr(claim, '__spec__', None), 'loader', None) is not finder
                or finder.assert_intact() is not True):
            raise ValueError('the validation accounting source is not bound')
        parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
        def authority(stage, _config):
            if (stage not in ('before_launch', 'after_child') or _ordinary_source(config_path, 65536) != config_raw
                    or owner.verify_private_parent(parent, tuple(config['parent_identity'])) is not True
                    or check_unclaimed_validation(str(workspace), tuple(config['ledger_identity'])) is not True):
                return False
            total, _ = claim._diagnostic_bytes(diagnostic)
            return total <= (1024 * 1024 if stage == 'before_launch' else 4 * 1024 * 1024)
        values = validation_child_values(context, config, child)
        return run_protected_replay(context, finder, workspace=str(workspace), cache_parent=str(run),
            config_values=values, config_path=str(run / 'child-config.json'), output=str(run / 'child-supervisor'),
            terminal_path=str(run / 'parent-terminal.json'), environment=dict(os.environ), authority=authority,
            validation_only=True, extra_files=(config_path,))
    return run_bound_imports(context['graph'], execute)


def run_validation_observer(config_path):
    """Retain and verify the gated no-model parent and its nested child."""
    _assert_validation_parent_isolated()
    config_raw = _ordinary_source(config_path, 65536)
    config = parse_validation_parent_config(config_raw, config_path)
    workspace, run = Path(config['workspace']), Path(config['run_dir'])
    diagnostic = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    context = capture_preflight(plan_path=config['plan_path'], plan_url=config['plan_url'],
        source_locations=config['source_locations'], public_paths=config['public_paths'],
        frozen_root=config['frozen_root'], cache_dir=config['cache_dir'])
    def execute(finder):
        modules = _execution_modules(finder)
        owner, child = modules['owner'], modules['child']
        claim = importlib.import_module('tools.injection_replay_claim')
        if (getattr(getattr(claim, '__spec__', None), 'loader', None) is not finder
                or finder.assert_intact() is not True):
            raise ValueError('the validation accounting source is not bound')
        parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
        def authority():
            if (_ordinary_source(config_path, 65536) != config_raw
                    or owner.verify_private_parent(parent, tuple(config['parent_identity'])) is not True
                    or check_unclaimed_validation(str(workspace), tuple(config['ledger_identity'])) is not True):
                return False
            total, _ = claim._diagnostic_bytes(diagnostic)
            return total <= 4 * 1024 * 1024
        total, _ = claim._diagnostic_bytes(diagnostic)
        if total > 1024 * 1024 or authority() is not True:
            raise ValueError('the no-model observation has insufficient reserved space')
        observation = observe_replay_parent(context, finder, workspace=str(workspace), cache_parent=str(run),
            purpose='validation', config_path=config_path, terminal_path=str(run / 'parent-terminal.json'),
            output=str(run / 'parent-observer'), command_path=str(run / 'parent-command.json'),
            environment=dict(os.environ), authority=authority,
            terminal_check=lambda raw, pid, birth: check_validation_parent_terminal(
                context, config, child, owner, raw, pid, birth))
        observed_raw = canonical(observation)
        evidence = [config_path, str(run / 'parent-command.json'), str(run / 'parent-terminal.json'),
            str(run / 'parent-observer' / 'observation.json'), str(run / 'child-config.json'),
            str(run / 'child-postcheck.json'), str(run / 'child-supervisor' / 'supervisor.json'),
            str(run / 'child-supervisor' / 'stdout.txt'), str(run / 'child-supervisor' / 'stderr.txt')]
        frozen, operational, assets = lease_inputs(context, str(workspace), extra_files=evidence)
        guard = modules['guard']
        leases = [guard.SourceFileLease(workspace, files) for files in (frozen, operational, assets)]
        state = {}
        def precheck(_acquired):
            if (recheck_preflight(context, finder) is not True or authority() is not True
                    or _ordinary_source(str(run / 'parent-observer' / 'observation.json'), 8192) != observed_raw):
                raise ValueError('the validation observation changed before its terminal')
            pid, birth = _process_identity(owner._current_process_identity())
            state['terminal'] = make_validation_observer_terminal(context, config, child, owner, observed_raw, pid, birth)
            return True
        def postcheck(_result):
            terminal_raw = canonical(state['terminal'])
            _write_once(str(run / 'observer-terminal.json'), terminal_raw)
            if (recheck_preflight(context, finder) is not True or authority() is not True
                    or _ordinary_source(str(run / 'parent-observer' / 'observation.json'), 8192) != observed_raw
                    or make_validation_observer_terminal(context, config, child, owner, observed_raw,
                        state['terminal']['observer_pid'], state['terminal']['observer_birth_100ns']) != state['terminal']):
                raise ValueError('the validation observer terminal could not be reconciled')
            return state['terminal']
        return owner.run_guarded(leases, precheck, lambda acquired: {'cleanup_verified': True}, postcheck,
                                 retained=modules)
    return run_bound_imports(context['graph'], execute)


def run_validation_operator(config_path):
    """Observe the no-model observer; this operator's own exit is separate."""
    _assert_validation_parent_isolated()
    config_raw = _ordinary_source(config_path, 65536)
    config = parse_validation_parent_config(config_raw, config_path)
    workspace, run = Path(config['workspace']), Path(config['run_dir'])
    diagnostic = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    context = capture_preflight(plan_path=config['plan_path'], plan_url=config['plan_url'],
        source_locations=config['source_locations'], public_paths=config['public_paths'],
        frozen_root=config['frozen_root'], cache_dir=config['cache_dir'])
    def execute(finder):
        modules = _execution_modules(finder)
        owner, child, guard = modules['owner'], modules['child'], modules['guard']
        claim = importlib.import_module('tools.injection_replay_claim')
        if (getattr(getattr(claim, '__spec__', None), 'loader', None) is not finder
                or finder.assert_intact() is not True):
            raise ValueError('the validation accounting source is not bound')
        parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
        def authority():
            if (_ordinary_source(config_path, 65536) != config_raw
                    or owner.verify_private_parent(parent, tuple(config['parent_identity'])) is not True
                    or check_unclaimed_validation(str(workspace), tuple(config['ledger_identity'])) is not True):
                return False
            total, _ = claim._diagnostic_bytes(diagnostic)
            return total <= 4 * 1024 * 1024
        total, _ = claim._diagnostic_bytes(diagnostic)
        if total > 1024 * 1024 or authority() is not True:
            raise ValueError('the no-model operator has insufficient reserved space')
        def terminal_check(raw, pid, birth):
            inner = _ordinary_source(str(run / 'parent-observer' / 'observation.json'), 8192)
            expected = make_validation_observer_terminal(context, config, child, owner, inner, pid, birth)
            return type(raw) is bytes and raw == canonical(expected)
        outer = observe_replay_parent(context, finder, workspace=str(workspace), cache_parent=str(run),
            purpose='validation', role='observer', config_path=config_path,
            terminal_path=str(run / 'observer-terminal.json'), output=str(run / 'observer-observer'),
            command_path=str(run / 'observer-command.json'), environment=dict(os.environ),
            authority=authority, terminal_check=terminal_check)
        outer_raw = canonical(outer)
        evidence = [config_path, str(run / 'parent-command.json'), str(run / 'parent-terminal.json'),
            str(run / 'parent-observer' / 'observation.json'), str(run / 'child-config.json'),
            str(run / 'child-postcheck.json'), str(run / 'child-supervisor' / 'supervisor.json'),
            str(run / 'child-supervisor' / 'stdout.txt'), str(run / 'child-supervisor' / 'stderr.txt'),
            str(run / 'observer-command.json'), str(run / 'observer-terminal.json'),
            str(run / 'observer-observer' / 'observation.json')]
        frozen, operational, assets = lease_inputs(context, str(workspace), extra_files=evidence)
        leases = [guard.SourceFileLease(workspace, files) for files in (frozen, operational, assets)]
        state = {}
        def precheck(_acquired):
            if (recheck_preflight(context, finder) is not True or authority() is not True
                    or _ordinary_source(str(run / 'observer-observer' / 'observation.json'), 8192) != outer_raw):
                raise ValueError('the no-model observer exit changed')
            pid, birth = _process_identity(owner._current_process_identity())
            state['receipt'] = make_validation_exit_receipt(context, config, child, owner, outer, outer_raw, pid, birth)
            return True
        def postcheck(_result):
            receipt_raw = canonical(state['receipt'])
            _write_once(str(run / 'validation-exit.json'), receipt_raw)
            if (recheck_preflight(context, finder) is not True or authority() is not True
                    or _ordinary_source(str(run / 'observer-observer' / 'observation.json'), 8192) != outer_raw
                    or make_validation_exit_receipt(context, config, child, owner, outer, outer_raw,
                        state['receipt']['operator_pid'], state['receipt']['operator_birth_100ns']) != state['receipt']):
                raise ValueError('the no-model exit receipt could not be reconciled')
            return state['receipt']
        return owner.run_guarded(leases, precheck, lambda acquired: {'cleanup_verified': True}, postcheck,
                                 retained=modules)
    return run_bound_imports(context['graph'], execute)


def run_validation_root(config_path):
    """Retain the validation operator; the caller must observe this root's exit."""
    flags = sys.flags
    if (not (flags.isolated and flags.ignore_environment and flags.no_site and flags.no_user_site
             and flags.safe_path and sys.dont_write_bytecode and flags.optimize == 0)
            or Path(sys.executable).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)):
        raise ValueError('the source-first validation root is not isolated')
    config_raw = _ordinary_source(config_path, 65536)
    config = parse_validation_parent_config(config_raw, config_path)
    workspace, run = Path(config['workspace']), Path(config['run_dir'])
    diagnostic = workspace / 'work' / 'injection-heldout-runtime-rehearsal-v1'
    context = capture_preflight(plan_path=config['plan_path'], plan_url=config['plan_url'],
        source_locations=config['source_locations'], public_paths=config['public_paths'],
        frozen_root=config['frozen_root'], cache_dir=config['cache_dir'])
    def execute(finder):
        modules = _execution_modules(finder)
        owner, child = modules['owner'], modules['child']
        claim = importlib.import_module('tools.injection_replay_claim')
        if (getattr(getattr(claim, '__spec__', None), 'loader', None) is not finder
                or finder.assert_intact() is not True):
            raise ValueError('the validation accounting source is not bound')
        parent = workspace / 'work' / 'number-injection-heldout-cpu-v1'
        def authority():
            if (_ordinary_source(config_path, 65536) != config_raw
                    or owner.verify_private_parent(parent, tuple(config['parent_identity'])) is not True
                    or check_unclaimed_validation(str(workspace), tuple(config['ledger_identity'])) is not True):
                return False
            total, _ = claim._diagnostic_bytes(diagnostic)
            return total <= 4 * 1024 * 1024
        total, _ = claim._diagnostic_bytes(diagnostic)
        if total > 1024 * 1024 or authority() is not True:
            raise ValueError('the no-model root has insufficient reserved space')
        return observe_replay_parent(context, finder, workspace=str(workspace), cache_parent=str(run),
            purpose='validation', role='operator', config_path=config_path,
            terminal_path=str(run / 'validation-exit.json'), output=str(run / 'operator-observer'),
            command_path=str(run / 'operator-command.json'), environment=dict(os.environ), authority=authority,
            terminal_check=lambda raw, pid, birth: check_validation_exit_receipt(
                context, config, child, owner, raw, pid, birth))
    return run_bound_imports(context['graph'], execute)


def main(argv=None, stream=None):
    try:
        await_start(stream)
        args = list(sys.argv[1:] if argv is None else argv)
        if len(args) != 2 or args[0] not in ('--validation-parent', '--validation-observer',
                                             '--validation-operator', '--validation-root'):
            raise ValueError('the replay-parent mode is unavailable')
        if args[0] == '--validation-parent':
            run_validation_parent(args[1])
        elif args[0] == '--validation-observer':
            run_validation_observer(args[1])
        elif args[0] == '--validation-operator':
            run_validation_operator(args[1])
        else:
            run_validation_root(args[1])
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
