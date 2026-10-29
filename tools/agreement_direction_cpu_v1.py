"""Source-bound direct entry for the fixed public agreement-direction study.

The prelaunch and latest-inventory records are operator observations. A
candidate terminal still requires an observed zero process exit and an
independent saved-data audit. This entry never retries an attempt.
"""
import argparse
import datetime
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

PROTOCOL_COMMIT = 'a39121c2f3cf2602281252a8856d48efadcb2a98'
PROTOCOL_PATH = 'docs/AGREEMENT_DIRECTION_STUDY_V1.md'
PROTOCOL_SHA = 'f539ad55b347f88641de3b76565104dc7be43e0c5d15f6d8fefdb4173694bbea'
CASE_COMMIT = '3a5d5d2088471a702acea42c2f38d2cb2e0072bc'
CASE_PATH = 'experiments/distilgpt2_agreement_direction_cases_v1.json'
CASE_SHA = 'a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13'
ROWS_SHA = '4b4faf815ae0d3d0943efec5fbc318fd251dc6fea80b6e32c770b470f01dae1d'
DIRECTION_COMMIT = '15c0ef7f8141322aeb1fa31dc02cab5a68fc16b8'
MANIFEST_SHA = '91dc4170eff6ac60416e5b3f884f2a28f41f9a4c1ad7339b409e32baeadf3a0e'
DIRECTION_SHA = {'u': 'd1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0',
                 'v1': '4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb',
                 'v2': '4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02'}
DIRECTION_PATH = 'results/distilgpt2-agreement-direction-training-v1'
RESERVATION_COMMIT = 'b7068665df240942793e848ceb06332c01e3a48f'
RESERVATION_VERSION = 'distilgpt2-development-exposures-v20'
RESERVATION_SHA = '3df696303a06b2ec6ddf4113df1969897dbae89566375b7d3a9cecb85b3d59cd'
V21_SHA = 'a391c5f3a8ca33ad32e9378417b424338b12ea55257979977d2afd5a0c7f90ae'
V22_SHA = 'd44e8cd2db2a1815335dfa48e26b5a5b05e44938579c6ce4e2989afa0e6d822e'
LATEST_COMMIT = '0a167ee5b044f80d52da3813cbc733bf6b498bd3'
LATEST_VERSION = 'distilgpt2-development-exposures-v23'
LATEST_SHA = '2549f1430ada47da088d2a909f8aa57ad5c08bf7c6d09f187bd34bb76b99f658'
LATEST_MODULE = 'looking_glass/pretrained_exposures_v23.py'
NATURAL_RECEIPT_PATH = 'results/distilgpt2-agreement-direction-cases-v1/natural-comparison.json'
NATURAL_INPUT_PATH = 'results/distilgpt2-agreement-direction-cases-v1/natural-comparison-inputs.json'
NATURAL_RECEIPT_SHA = '2f288aa89494b2e597b64ba594166d9c8bc78d2a9bc5831377971d8116c4b8d6'
NATURAL_INPUT_SHA = '62d762605c3ca9fa78f53c6f186c6b1f2e94e08e1479d5f3f16058a805eb28fe'
NATURAL_PARENT_SHA = 'e2ca37df19aeda036e42e7c6f33ce3ac5eef224c6bfc2fbb53190b94f912c2d4'
MODEL_ID = 'distilbert/distilgpt2'
MODEL_COMMIT = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
WEIGHTS_SHA = 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
REQUIRED_ENV = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'MKL_CBWR': None, 'ATEN_CPU_CAPABILITY': None}
VERSIONS = {'python': '3.11.15', 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
            'tokenizers': '0.22.2', 'numpy': '2.4.6', 'huggingface_hub': '0.36.2'}
LIMITS = {'expected_forwards': 784, 'logical_raw_bytes': 512 * 1024**2,
          'physical_attempt_bytes': 1024**3, 'minimum_free_disk_bytes': 2 * 1024**3,
          'automatic_retry': False}


def _digest(value, length):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('nonfinite JSON number')
    return number


def read_bytes(path, limit):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('input must be an absolute ordinary file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('input exceeds its byte limit')
    return raw


def parse_json(raw):
    def invalid(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=_unique, parse_float=_finite_float, parse_constant=invalid)


def read_json(path, limit):
    return parse_json(read_bytes(path, limit))


def _require(actual, expected):
    if type(actual) is not dict or any(type(actual.get(k)) is not type(v) or actual.get(k) != v
                                       for k, v in expected.items()):
        raise ValueError('prelaunch fixed fields differ')


def validate_prelaunch(record):
    if type(record) is not dict or record.get('schema') != 'distilgpt2-agreement-direction-prelaunch-v1':
        raise ValueError('invalid prelaunch schema')
    owner = record.get('owner')
    if (type(owner) is not dict or not all(_digest(owner.get(k), n) for k, n in
            (('source_revision', 40), ('source_tree', 40), ('executor_sha256', 64), ('entry_script_sha256', 64)))):
        raise ValueError('invalid owner identity')
    _require(record.get('protocol'), {'revision': PROTOCOL_COMMIT, 'path': PROTOCOL_PATH, 'sha256': PROTOCOL_SHA})
    _require(record.get('cases'), {'revision': CASE_COMMIT, 'path': CASE_PATH,
                                   'sha256': CASE_SHA, 'ordered_rows_sha256': ROWS_SHA})
    _require(record.get('directions'), {'revision': DIRECTION_COMMIT, 'manifest_sha256': MANIFEST_SHA,
                                        **{name + '_sha256': value for name, value in DIRECTION_SHA.items()}})
    _require(record.get('checkpoint'), {'model_id': MODEL_ID, 'revision': MODEL_COMMIT,
                                        'weights_sha256': WEIGHTS_SHA, 'tokenizer_sha256': TOKENIZER_SHA})
    _require(record.get('limits'), LIMITS)
    runtime = record.get('runtime')
    _require(runtime, {**VERSIONS, 'cpu_threads': 1, 'cpu_interop_threads': 1, 'claim_cuda_visibility': ''})
    if runtime.get('environment') != REQUIRED_ENV:
        raise ValueError('runtime environment differs')
    exposure = record.get('exposure')
    _require(exposure, {'reservation_revision': RESERVATION_COMMIT,
                        'reservation_version': RESERVATION_VERSION, 'reservation_sha256': RESERVATION_SHA,
                        'latest_revision': LATEST_COMMIT, 'latest_version': LATEST_VERSION,
                        'latest_sha256': LATEST_SHA})
    custody = record.get('custody')
    if type(custody) is not dict:
        raise ValueError('custody observations are absent')
    _require(custody.get('natural'), {'case_sha256': CASE_SHA, 'status': 'clear',
                                      'comparison_receipt_sha256': NATURAL_RECEIPT_SHA,
                                      'comparison_parent_observation_sha256': NATURAL_PARENT_SHA,
                                      'scope': 'bidirectional full-row-prefix against existing sealed split'})
    _require(custody.get('injection'), {'status': 'no_protected_rows', 'direct_target_conflict': 'none reported',
                                        'scope': 'fixed L3 final/subject and L1 final attention outputs'})
    if not all(type(record.get(key)) is str and record[key] for key in
               ('independent_reader_status', 'recorded_utc', 'scope')):
        raise ValueError('prelaunch observations are incomplete')
    try:
        observed = datetime.datetime.fromisoformat(record['recorded_utc'].replace('Z', '+00:00'))
        if observed.utcoffset() != datetime.timedelta(0):
            raise ValueError('not UTC')
    except ValueError as error:
        raise ValueError('prelaunch time is not UTC') from error
    return record


def _related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def _known_sequences(records):
    """Preserve complete known rows and continuations across status updates."""
    if type(records) is not list:
        raise ValueError('invalid inventory records')
    sequences = set()
    for record in records:
        if type(record) is not dict:
            raise ValueError('invalid inventory record')
        row = record.get('token_ids')
        candidates = record.get('candidate_token_ids', [])
        if (type(row) is not list or not 1 <= len(row) <= 128
                or any(type(token) is not int or not 0 <= token < 50257 for token in row)
                or type(candidates) is not list or len(candidates) > 512
                or any(type(token) is not int or not 0 <= token < 50257 for token in candidates)):
            raise ValueError('invalid known row or continuations')
        prefix = tuple(row)
        sequences.add(prefix)
        sequences.update(prefix + (token,) for token in candidates)
        target = record.get('target_token_id')
        if target is not None:
            if type(target) is not int or not 0 <= target < 50257:
                raise ValueError('invalid known target')
            sequences.add(prefix + (target,))
    return sequences


def validate_exposure(payload, cases):
    records = payload.get('records') if type(payload) is dict else None
    if (type(records) is not list or len(records) > 100000 or type(cases) is not list or len(cases) != 40):
        raise ValueError('invalid latest exposure inventory')
    expected = {'planned-' + row['case_id']: row for row in cases}
    if len(expected) != 40:
        raise ValueError('duplicate case identities')
    seen, own = set(), set()
    proposed = [tuple(row['token_ids']) for row in cases]
    complete = [row + (answer,) for row in proposed for answer in (318, 389, 373, 547)]
    for item in records:
        if type(item) is not dict or type(item.get('record_id')) is not str or item['record_id'] in seen:
            raise ValueError('invalid or repeated inventory identity')
        seen.add(item['record_id'])
        tokens = item.get('token_ids')
        if (type(tokens) is not list or not 1 <= len(tokens) <= 128
                or any(type(token) is not int or not 0 <= token < 50257 for token in tokens)):
            raise ValueError('invalid inventory token row')
        if item['record_id'] in expected:
            case = expected[item['record_id']]
            _require(item, {'category': 'planned', 'case_role': case['role'], 'public_text': case['text'],
                            'token_ids': case['token_ids'], 'candidate_token_ids': [318, 389, 373, 547]})
            own.add(item['record_id'])
            continue
        known = [tuple(tokens)]
        answer = item.get('target_token_id')
        if answer is not None:
            if type(answer) is not int or not 0 <= answer < 50257:
                raise ValueError('invalid inventory continuation')
            known.append(tuple(tokens) + (answer,))
        candidates = item.get('candidate_token_ids', [])
        if (type(candidates) is not list or len(candidates) > 512
                or any(type(answer) is not int or not 0 <= answer < 50257 for answer in candidates)):
            raise ValueError('invalid inventory continuations')
        known.extend(tuple(tokens) + (answer,) for answer in candidates)
        if any(_related(row, prior) for row in proposed + complete for prior in known):
            raise ValueError('latest inventory has a known case collision')
    if own != set(expected):
        raise ValueError('latest inventory lacks an exact planned row')


def validate_natural_records(receipt, bundle, parent, cases):
    _require(receipt, {'schema': 'mlg-natural-external-input-comparison-v1', 'status': 'clear',
                       'comparison': 'bidirectional-complete-row-prefix', 'public_case_count': 40,
                       'public_input_file_sha256': NATURAL_INPUT_SHA})
    _require(bundle, {'cases': [{'input_ids': row['token_ids']} for row in cases]})
    _require(parent, {'schema': 'mlg-natural-comparison-parent-observation-v1', 'exit_code': 0,
                      'reaped': True, 'receipt_verified': True, 'receipt_sha256': NATURAL_RECEIPT_SHA,
                      'adapted_public_file_sha256': NATURAL_INPUT_SHA,
                      'original_public_file_sha256': CASE_SHA, 'ordered_public_rows_sha256': ROWS_SHA})


def validate_natural_comparison(root, cases, parent_path):
    receipt = read_bytes(Path(root) / NATURAL_RECEIPT_PATH, 256 * 1024)
    bundle = read_bytes(Path(root) / NATURAL_INPUT_PATH, 256 * 1024)
    parent = read_bytes(Path(parent_path), 256 * 1024)
    if (_hash(receipt) != NATURAL_RECEIPT_SHA or _hash(bundle) != NATURAL_INPUT_SHA
            or _hash(parent) != NATURAL_PARENT_SHA):
        raise ValueError('natural comparison observation bytes differ')
    validate_natural_records(parse_json(receipt), parse_json(bundle), parse_json(parent), cases)
    projection = {'schema': 'agreement-direction-natural-observation-projection-v1',
                  'original_parent_sha256': NATURAL_PARENT_SHA, 'receipt_sha256': NATURAL_RECEIPT_SHA,
                  'input_bundle_sha256': NATURAL_INPUT_SHA, 'case_sha256': CASE_SHA,
                  'ordered_rows_sha256': ROWS_SHA, 'exit_code': 0, 'reaped': True,
                  'scope': 'operator-observed comparator; not authenticated execution or chronology'}
    return receipt, bundle, projection


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > 256 * 1024:
        raise ValueError('output metadata exceeds its bound')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short metadata write')
        stream.flush()
        os.fsync(stream.fileno())


def _write_raw(path, raw):
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short evidence write')
        stream.flush()
        os.fsync(stream.fileno())


def _ordinary(path, root):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError('path is outside checkout')
    cursor = root
    for component in path.relative_to(root).parts:
        cursor = cursor / component
        info = cursor.lstat()
        if cursor.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('source path contains a reparse point')
    if not path.is_file():
        raise ValueError('source is not an ordinary file')


def _git(root, *args):
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    environment.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                        'GIT_GRAFT_FILE': os.devnull, 'GIT_NO_REPLACE_OBJECTS': '1',
                        'GIT_TERMINAL_PROMPT': '0', 'GIT_OPTIONAL_LOCKS': '0'})
    return subprocess.check_output(['git', '--no-replace-objects', '-C', str(root), *args],
                                   env=environment, stderr=subprocess.DEVNULL, timeout=60)


def source_inventory(root, revision, tree):
    root = Path(root).resolve(strict=True)
    def git(*args):
        return _git(root, *args)
    if git('rev-parse', '--verify', 'HEAD').strip().decode('ascii') != revision:
        raise ValueError('checkout revision differs')
    if git('show', '-s', '--format=%T', revision).strip().decode('ascii') != tree:
        raise ValueError('checkout tree differs')
    if git('status', '--porcelain=v1', '--untracked-files=all').strip():
        raise ValueError('checkout is not clean')
    for predecessor in (PROTOCOL_COMMIT, CASE_COMMIT, DIRECTION_COMMIT, RESERVATION_COMMIT):
        git('merge-base', '--is-ancestor', predecessor, revision)
    entries = git('ls-tree', '-rz', '--full-tree', revision).split(b'\0')
    sources = {}
    for entry in entries:
        if not entry:
            continue
        header, name = entry.split(b'\t', 1)
        mode, kind, blob = header.decode('ascii').split()
        name = name.decode('utf-8')
        if not ((name.startswith('looking_glass/') and name.endswith('.py'))
                or name == 'tools/agreement_direction_cpu_v1.py'):
            continue
        if mode not in ('100644', '100755') or kind != 'blob' or '..' in Path(name).parts:
            raise ValueError('unsupported source tree entry')
        path = root / name
        _ordinary(path, root)
        raw = read_bytes(path, 8 * 1024 * 1024)
        object_id = hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
        if object_id != blob:
            raise ValueError('source bytes differ from the committed blob')
        sources[name] = raw
    actual = {p.relative_to(root).as_posix() for p in (root / 'looking_glass').rglob('*.py')}
    if actual != {name for name in sources if name.startswith('looking_glass/')}:
        raise ValueError('project source inventory differs')
    return sources


def bind_latest_committed_inventory(root, sources, revision, expected_raw):
    """Rebuild only the explicitly reviewed latest inventory module chain."""
    _git(root, 'merge-base', '--is-ancestor', RESERVATION_COMMIT, LATEST_COMMIT)
    _git(root, 'merge-base', '--is-ancestor', LATEST_COMMIT, revision)
    module_blob = _git(root, 'show', LATEST_COMMIT + ':' + LATEST_MODULE)
    if LATEST_MODULE not in sources or sources[LATEST_MODULE] != module_blob:
        raise ValueError('latest inventory module differs from its committed bytes')
    from looking_glass import pretrained_exposures_v20 as v20
    from looking_glass import pretrained_exposures_v21 as v21
    from looking_glass import pretrained_exposures_v22 as v22
    from looking_glass import pretrained_exposures_v23 as v23
    if (v20.INVENTORY_VERSION != RESERVATION_VERSION or v20.INVENTORY_SHA256 != RESERVATION_SHA
            or v21.INVENTORY_SHA256 != V21_SHA or v21.PREDECESSOR_SHA256 != RESERVATION_SHA
            or v22.INVENTORY_SHA256 != V22_SHA or v22.PREDECESSOR_SHA256 != V21_SHA
            or v23.INVENTORY_VERSION != LATEST_VERSION or v23.INVENTORY_SHA256 != LATEST_SHA
            or v23.PREDECESSOR_SHA256 != V22_SHA):
        raise ValueError('inventory module identities differ')
    previous = v20.inventory_payload()
    latest = v23.inventory_payload()
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=True, allow_nan=False).encode('ascii')
    if (_hash(canonical(previous)) != RESERVATION_SHA or canonical(latest) != expected_raw
            or _hash(expected_raw) != LATEST_SHA
            or not _known_sequences(previous['records']) <= _known_sequences(latest['records'])):
        raise ValueError('latest inventory does not preserve the known v20 rows and continuations')
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise ValueError('inventory validation unexpectedly imported model dependencies')


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, raw, filename):
        self.raw, self.filename = raw, filename
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__file__ = self.filename
        exec(compile(self.raw, self.filename, 'exec', dont_inherit=True), module.__dict__)


class _SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, sources):
        self.root, self.sources = root, sources
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'looking_glass' and not fullname.startswith('looking_glass.'):
            return None
        if any(re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', part) is None for part in fullname.split('.')):
            raise ModuleNotFoundError('invalid project module')
        name = fullname.replace('.', '/')
        package = name + '/__init__.py'
        source = name + '.py'
        selected = package if package in self.sources else source
        if selected not in self.sources:
            raise ModuleNotFoundError('module is outside the captured project source inventory')
        return importlib.util.spec_from_loader(fullname, _SourceLoader(self.sources[selected], str(self.root / selected)),
                                               is_package=selected == package)


def _early_precheck(record):
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules)
            or any(name in sys.modules for name in ('torch', 'transformers'))):
        raise ValueError('entry requires -I -S -B before project and model imports')
    expected = {**REQUIRED_ENV, 'CUDA_VISIBLE_DEVICES': ''}
    if {name: os.environ.get(name) for name in expected} != expected:
        raise ValueError('pre-claim CPU and offline environment differs')
    if record['runtime']['claim_cuda_visibility'] != '':
        raise ValueError('prelaunch claim mask differs')


def _capture_local_sources(root, record):
    """Capture source before the grant; Git verification follows the claim."""
    root = Path(root).resolve(strict=True)
    sources, total = {}, 0
    for path in (root / 'looking_glass').rglob('*.py'):
        _ordinary(path, root)
        raw = read_bytes(path, 8 * 1024 * 1024)
        total += len(raw)
        if total > 128 * 1024 * 1024 or len(sources) >= 5000:
            raise ValueError('pre-claim source inventory exceeds its bound')
        sources[path.relative_to(root).as_posix()] = raw
    entry = root / 'tools/agreement_direction_cpu_v1.py'
    _ordinary(entry, root)
    sources['tools/agreement_direction_cpu_v1.py'] = read_bytes(entry, 8 * 1024 * 1024)
    if ('looking_glass/__init__.py' not in sources
            or _hash(sources.get('looking_glass/agreement_direction_study_v1.py', b''))
               != record['owner']['executor_sha256']
            or _hash(sources['tools/agreement_direction_cpu_v1.py']) != record['owner']['entry_script_sha256']):
        raise ValueError('pre-claim source digest differs')
    return sources


def _claim_cpu(study_root):
    from looking_glass.cpu_study_supervision import claim_cpu_study
    return claim_cpu_study(study_root)


def _install_venv(venv):
    venv = Path(venv)
    if not venv.is_absolute() or not venv.is_dir() or venv.is_symlink():
        raise ValueError('venv must be an absolute ordinary directory')
    raw = read_bytes(venv / 'pyvenv.cfg', 32 * 1024).decode('utf-8')
    pairs = [line.split('=', 1) for line in raw.splitlines() if '=' in line]
    homes = [value.strip() for key, value in pairs if key.strip().lower() == 'home']
    system = [value.strip().lower() for key, value in pairs if key.strip().lower() == 'include-system-site-packages']
    if (len(homes) != 1 or not Path(homes[0]).is_absolute() or system != ['false']
            or os.path.normcase(str(Path(homes[0]).resolve()))
               != os.path.normcase(str(Path(sys.executable).resolve().parent))):
        raise ValueError('venv home does not match the direct base interpreter')
    site = venv / ('Lib/site-packages' if os.name == 'nt' else
                   'lib/python%d.%d/site-packages' % sys.version_info[:2])
    if not site.is_dir() or site.is_symlink() or getattr(site.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('venv site-packages directory is unavailable or aliased')
    sys.path.append(str(site))


def _bootstrap(root, record, study_root, venv):
    _early_precheck(record)
    study_root = Path(study_root)
    if not study_root.is_absolute() or not study_root.is_dir() or study_root.is_symlink():
        raise ValueError('CPU study root must be an existing absolute directory')
    sources = _capture_local_sources(root, record)
    sys.meta_path.insert(0, _SourceFinder(root, sources))
    session, lease = _claim_cpu(study_root)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU visibility changed during claim')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    _install_venv(venv)
    return session, lease, sources


def _fixed_inputs(root, case_file, direction_dir, latest_file, record):
    root = Path(root)
    protocol = read_bytes(root / PROTOCOL_PATH, 256 * 1024)
    if _hash(protocol) != PROTOCOL_SHA:
        raise ValueError('protocol bytes differ')
    raw_cases = read_bytes(case_file, 2 * 1024 * 1024)
    if _hash(raw_cases) != CASE_SHA or _hash(read_bytes(root / CASE_PATH, 2 * 1024 * 1024)) != CASE_SHA:
        raise ValueError('case bytes differ')
    document = parse_json(raw_cases)
    _require(document, {'schema': 'distilgpt2-agreement-direction-cases-v1', 'model_id': MODEL_ID,
                        'model_revision': MODEL_COMMIT, 'expected_forwards': 784,
                        'answer_ids': [318, 389, 373, 547], 'ordered_input_rows_sha256': ROWS_SHA})
    cases = document.get('cases')
    if type(cases) is not list or len(cases) != 40:
        raise ValueError('case count differs')
    rows = json.dumps([row['token_ids'] for row in cases], separators=(',', ':')).encode('ascii')
    if _hash(rows) != ROWS_SHA:
        raise ValueError('ordered rows differ')
    direction_dir = Path(direction_dir)
    if not direction_dir.is_absolute() or not direction_dir.is_dir() or direction_dir.is_symlink():
        raise ValueError('direction directory is invalid')
    directions = {}
    for location in (root / DIRECTION_PATH, direction_dir):
        if _hash(read_bytes(location / 'manifest.json', 1024 * 1024)) != MANIFEST_SHA:
            raise ValueError('direction manifest differs')
        for name, digest in DIRECTION_SHA.items():
            raw = read_bytes(location / (name + '.f64le'), 768 * 8)
            if len(raw) != 768 * 8 or _hash(raw) != digest:
                raise ValueError('direction bytes differ')
            values = struct.unpack('<768d', raw)
            if not all(math.isfinite(value) for value in values):
                raise ValueError('direction is not finite')
            directions[name] = values
    raw_inventory = read_bytes(latest_file, 16 * 1024 * 1024)
    exposure = record['exposure']
    if _hash(raw_inventory) != exposure['latest_sha256']:
        raise ValueError('latest exposure digest differs')
    payload = parse_json(raw_inventory)
    if (json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                   allow_nan=False).encode('ascii') != raw_inventory
            or payload.get('version') != exposure['latest_version']):
        raise ValueError('latest exposure bytes are not canonical or versioned')
    validate_exposure(payload, cases)
    return cases, directions, raw_inventory


def _runtime_precheck(record):
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or 'torch' in sys.modules or 'transformers' in sys.modules):
        raise ValueError('entry requires isolated startup before model imports')
    for name, module in tuple(sys.modules.items()):
        if (name == 'looking_glass' or name.startswith('looking_glass.')) and not isinstance(
                getattr(module, '__loader__', None), _SourceLoader):
            raise ValueError('project module did not use the captured source loader')
    if {name: os.environ.get(name) for name in REQUIRED_ENV} != REQUIRED_ENV:
        raise ValueError('required CPU and offline environment differs')
    actual = {'python': '.'.join(str(item) for item in sys.version_info[:3])}
    for name in ('torch', 'transformers', 'tokenizers', 'numpy', 'huggingface_hub'):
        actual[name] = importlib.metadata.version(name.replace('_', '-'))
    if actual != VERSIONS or any(record['runtime'][key] != value for key, value in actual.items()):
        raise ValueError('dependency versions differ')
    return actual


def _verify_cpu_runtime(torch):
    probe = torch.tensor([0x00800000], dtype=torch.int32, device='cpu')
    product = int((probe.view(torch.float32) * 0.5).view(torch.int32).item())
    if (torch.cuda.is_initialized() or torch.get_default_dtype() != torch.float32
            or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.get_float32_matmul_precision() != 'highest' or product != 0x00400000):
        raise ValueError('CPU numeric runtime differs')
    return {'threads': 1, 'interop_threads': 1, 'deterministic_algorithms': True,
            'float32_matmul_precision': 'highest', 'calling_thread_denormal_product_bits': product}


def _verify_tokenizer(tokenizer, cases):
    answers = ((' is', 318), (' are', 389), (' was', 373), (' were', 547))
    for text, answer in answers:
        if tokenizer.encode(text, add_special_tokens=False) != [answer]:
            raise ValueError('answer tokenization differs')
    for row in cases:
        text, expected = row['text'], row['token_ids']
        if tokenizer.encode(text, add_special_tokens=False) != expected:
            raise ValueError('case tokenization differs')
        for answer, token in answers:
            if tokenizer.encode(text + answer, add_special_tokens=False) != expected + [token]:
                raise ValueError('continuation tokenization differs')


def _physical_bytes(root):
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(base) / name
            if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
                raise ValueError('attempt contains an aliased directory')
        for name in files:
            path = Path(base) / name
            info = path.lstat()
            if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('attempt contains an aliased file')
            total += info.st_size
    return total


def _checked_source(root, record):
    owner = record['owner']
    sources = source_inventory(root, owner['source_revision'], owner['source_tree'])
    for name, expected in (('looking_glass/agreement_direction_study_v1.py', owner['executor_sha256']),
                           ('tools/agreement_direction_cpu_v1.py', owner['entry_script_sha256'])):
        if name not in sources or _hash(sources[name]) != expected:
            raise ValueError('prelaunch source file digest differs')
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prelaunch', 'prelaunch-sha256', 'latest-exposure', 'natural-parent-observation', 'case-file',
                 'direction-dir', 'cache-dir', 'output-dir', 'study-root', 'venv', 'child-custody'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    prelaunch_raw = read_bytes(Path(args.prelaunch), 256 * 1024)
    if not _digest(args.prelaunch_sha256, 64) or _hash(prelaunch_raw) != args.prelaunch_sha256:
        raise ValueError('prelaunch digest differs')
    record = validate_prelaunch(parse_json(prelaunch_raw))
    child_custody = Path(args.child_custody)
    if (not child_custody.is_absolute() or not child_custody.parent.is_dir()
            or os.path.lexists(child_custody) or child_custody.resolve().is_relative_to(root)):
        raise ValueError('child custody path must be a new absolute file outside the checkout')
    session, lease, captured_sources = _bootstrap(root, record, Path(args.study_root), Path(args.venv))
    session.check(lease)
    _write(child_custody, {'schema': 'agreement-direction-supervised-child-v1',
                           'run_id': session.run_id, 'owner_pid': os.getpid(),
                           'prelaunch_sha256': args.prelaunch_sha256,
                           'mode': 'validate_only' if args.validate_only else 'study'})
    versions = _runtime_precheck(record)
    sources = _checked_source(root, record)
    if sources != captured_sources:
        raise ValueError('pre-claim source bytes differ from the committed checkout')
    cases, directions, inventory_raw = _fixed_inputs(root, Path(args.case_file), Path(args.direction_dir),
                                                     Path(args.latest_exposure), record)
    natural_receipt, natural_inputs, natural_parent = validate_natural_comparison(
        root, cases, Path(args.natural_parent_observation))
    bind_latest_committed_inventory(root, sources, record['owner']['source_revision'], inventory_raw)
    session.check(lease)
    cache, output = Path(args.cache_dir), Path(args.output_dir)
    if (not cache.is_absolute() or not cache.is_dir() or cache.is_symlink()
            or not output.is_absolute() or not output.parent.is_dir() or os.path.lexists(output)
            or output.resolve().is_relative_to(root) or output.resolve().is_relative_to(cache.resolve())):
        raise ValueError('cache or fresh output path is invalid')
    if shutil.disk_usage(output.parent).free < LIMITS['minimum_free_disk_bytes']:
        raise ValueError('insufficient free space before launch')
    binding = {'prelaunch_sha256': args.prelaunch_sha256, 'reservation_sha256': RESERVATION_SHA,
               'latest_exposure_sha256': record['exposure']['latest_sha256']}
    observation = {'source_revision': record['owner']['source_revision'], 'source_tree': record['owner']['source_tree'],
                   'source_sha256': {name: _hash(raw) for name, raw in sorted(sources.items())},
                   'binding': binding, 'runtime_versions': versions, 'forwards': 0,
                   'owner_pid': os.getpid(), 'owner_run_id': session.run_id}
    if args.validate_only:
        print(json.dumps({'status': 'no_model_validation', **observation}, sort_keys=True))
        session.attest_clean(lease)
        return 0

    output.mkdir()
    # This exclusive marker prevents a duplicate attempt under this output
    # parent. The operator remains responsible for the wider one-shot gate.
    claim = output.parent / ('agreement-direction-v1-' + args.prelaunch_sha256 + '.claim')
    failure = output / 'entry-failure.json'
    try:
        _write(claim, {'output': str(output), 'prelaunch_sha256': args.prelaunch_sha256})
        _write(output / 'admission.json', observation)
        _write_raw(output / 'prelaunch.json', prelaunch_raw)
        _write_raw(output / 'latest-exposure.json', inventory_raw)
        _write_raw(output / 'natural-comparison.json', natural_receipt)
        _write_raw(output / 'natural-comparison-inputs.json', natural_inputs)
        _write(output / 'natural-parent-projection.json', natural_parent)
        session.check(lease)
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.set_flush_denormal(False)
        runtime = _verify_cpu_runtime(torch)
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.agreement_direction_study_v1 import run_direction_study
        from looking_glass.pretrained import load_distilgpt2
        journal = AgreementJournal(output / 'attempt', {'scope': 'public prospective agreement-direction',
                                                         'external_binding': binding,
                                                         'source_revision': record['owner']['source_revision']})
        loaded = load_distilgpt2(cache_dir=str(cache), allow_download=False)
        if (loaded.metadata['model_id'] != MODEL_ID or loaded.metadata['model_revision'] != MODEL_COMMIT
                or loaded.metadata['weights_sha256'] != WEIGHTS_SHA
                or loaded.metadata['asset_sha256']['tokenizer.json'] != TOKENIZER_SHA
                or loaded.model.config.n_embd != 768 or loaded.model.config.vocab_size != 50257
                or len(loaded.model.transformer.h) != 6 or torch.cuda.is_initialized()):
            raise ValueError('loaded model or architecture differs')
        _verify_tokenizer(loaded.tokenizer, cases)
        if ({name: os.environ.get(name) for name in REQUIRED_ENV} != REQUIRED_ENV
                or _verify_cpu_runtime(torch) != runtime):
            raise ValueError('environment changed during loading')
        _write(output / 'model-admission.json', {'status': 'loaded_before_forward',
                                                'runtime': runtime, 'checkpoint': loaded.metadata,
                                                'case_sha256': CASE_SHA, 'binding': binding})
        import gc
        import weakref
        model_ref = weakref.ref(loaded.model)
        result = run_direction_study(loaded.model, journal, cases, directions, binding=binding,
                                     custody_check=lambda: session.check(lease))
        del loaded
        gc.collect()
        if (model_ref() is not None or torch.cuda.is_initialized()
                or _verify_cpu_runtime(torch) != runtime
                or {name: os.environ.get(name) for name in REQUIRED_ENV} != REQUIRED_ENV):
            raise ValueError('CPU model disposal was not verified')
        session.check(lease)
        if _checked_source(root, record) != sources:
            raise ValueError('source changed during the attempt')
        # Keep a bounded allowance for the claim, result and separately retained
        # direct-operator receipts; the operator accounts those external files.
        if _physical_bytes(output) > LIMITS['physical_attempt_bytes'] - 16 * 1024**2:
            raise ValueError('physical attempt bytes exceeded the limit')
        _write(output / 'entry-result.json', {'status': 'complete_candidate', 'binding': binding,
                                             'worker': result, 'physical_attempt_bytes_before_result': _physical_bytes(output),
                                             'owner_pid': os.getpid(), 'owner_run_id': session.run_id,
                                             'model_disposed': True,
                                             'external_receipt_allowance_bytes': 16 * 1024**2,
                                             'scope': 'candidate only; external exit and independent scientific audit required'})
        print(json.dumps({'status': 'complete_candidate', 'forwards': result['forwards'], 'binding': binding},
                         sort_keys=True))
        session.attest_clean(lease)
        return 0
    except BaseException as error:
        try:
            _write(failure, {'status': 'cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                             'error_type': type(error).__name__, 'binding': binding,
                             'scope': 'incomplete; retain all partial artifacts and do not retry'})
        except BaseException as write_error:
            if isinstance(write_error, (KeyboardInterrupt, SystemExit)):
                raise write_error from error
        raise


if __name__ == '__main__':
    raise SystemExit(main())
