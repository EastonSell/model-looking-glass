"""Prospective subject-dependence CPU entry; real execution remains disabled.

The published case, exposure, custodian and prelaunch identities must be
bound and independently reviewed before this entry can load a model or run.
"""

import argparse
import hashlib
import importlib
import json
import math
import datetime
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import types

ANSWERS = ((' is', 318), (' are', 389), (' was', 373), (' were', 547),
           (' the', 262), (' a', 257), (' in', 287), (' on', 319))
WIDTH = 768
VOCABULARY = 50257
PROTOCOL_REVISION = '54f0f8d3ac7e60efc3005f0f49415715ba34ff47'
PROTOCOL_SHA256 = 'bd39b4556023110d0de88fb21381a0a3ce1e17cad862cdd0259b67849f540714'
PROTOCOL_PATH = 'docs/AGREEMENT_SUBJECT_DEPENDENCE_V1.md'
CASE_REVISION = '4db27a4f6d96a97c28ad644b911b9c5eacd4f55d'
CASE_SHA256 = '5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e'
ROWS_SHA256 = 'd31073b25d816258ff1bcbfcb7237eac5346b160a5ed2de65fea220bdb806ff1'
CASE_PATH = 'experiments/distilgpt2_agreement_subject_dependence_cases_v1.json'
DIRECTION_PATH = 'results/distilgpt2-agreement-direction-training-v1'
REFERENCE_PATH = 'results/distilgpt2-agreement-subject-output-reference-v1/output-slope.f64le'
REFERENCE_MANIFEST_PATH = 'results/distilgpt2-agreement-subject-output-reference-v1/manifest.json'
NATURAL_RECEIPT_PATH = 'results/distilgpt2-agreement-subject-dependence-cases-v1/natural-support-comparison.json'
MODEL_ID = 'distilbert/distilgpt2'
MODEL_REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
TOKENIZER_SHA256 = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
WEIGHTS_SHA256 = 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
DIRECTION_REVISION = '15c0ef7f8141322aeb1fa31dc02cab5a68fc16b8'
DIRECTION_ENTRY_SHA256 = 'a40ee2c5e8d758de81343b6f7d5ffca24c006d44354f3ed1365d778b3b29b76e'
DIRECTION_WORKER_SHA256 = 'a2fbb9d730f99778fd16009f9d65c9dca65a8613b115cdf6df12b08cafae1972'
DIRECTION_SHA256 = {
    'u': 'd1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0',
    'v1': '4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb',
    'v2': '4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02',
}
DIRECTION_MANIFEST_SHA256 = '91dc4170eff6ac60416e5b3f884f2a28f41f9a4c1ad7339b409e32baeadf3a0e'
REFERENCE_SHA256 = 'bdb31d5b74b6cff17ed2a683415beab22e36b101220257d774909be6f9f804bd'
REFERENCE_MANIFEST_SHA256 = '69c7bdadb365c83bd4a591da405cc5e1dccd686c204a68f9b03264c604e82ed5'
NATURAL_RECEIPT_SHA256 = '61d1f8555ad901924364fb89c731b0870ff057d6caeebcf4b5d293e819be1753'
SUCCESSOR_SUPPORT_SHA256 = '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
NATURAL_SCOPE = ('public support comparison; sealed-split coverage conditional on source-bound preparation; '
                 'no private membership queried')
RESERVATION_REVISION = 'baff8872b35193bcaf767c9e260ad4820e36332e'
RESERVATION_SHA256 = 'ef42683a664dda450e82c69afedda34a8a34f2daf9f50f2db54f6c9d846f07e5'
RESERVATION_VERSION = 'distilgpt2-development-exposures-v27'
RESERVATION_MODULE = 'looking_glass/pretrained_exposures_v27.py'
REQUIRED_ENV = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'MKL_CBWR': None, 'ATEN_CPU_CAPABILITY': None}
VERSIONS = {'python': '3.11.15', 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
            'tokenizers': '0.22.2', 'numpy': '2.4.6', 'huggingface_hub': '0.36.2'}
LIMITS = {'expected_forwards': 720, 'logical_success_bytes': 189468480,
          'failed_return_reserve_bytes': 2492208, 'invalid_candidate_reserve_bytes': 6144,
          'per_forward_bytes': 4 * 1024**2, 'retained_tensor_bytes': 512 * 1024**2,
          'event_log_bytes': 64 * 1024**2, 'physical_attempt_bytes': 1024**3,
          'external_receipt_allowance_bytes': 16 * 1024**2,
          'minimum_free_disk_bytes': 2 * 1024**3, 'automatic_retry': False}
_OWNER_ROOTS = None


def _read_asset(path, limit):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('asset must be an absolute ordinary file')
    if getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('asset cannot be a reparse point')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('asset exceeds its byte bound')
    return data


def load_reviewed_helper(root):
    """Load the byte-pinned prior CPU entry's stdlib-only custody helpers."""
    path = Path(root) / 'tools/agreement_direction_cpu_v1.py'
    raw = _read_asset(path, 8 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != DIRECTION_ENTRY_SHA256:
        raise ValueError('reviewed CPU helper bytes differ')
    module = types.ModuleType('_reviewed_agreement_direction_cpu_v1')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def verify_source(root, record, captured, helper, entry_path):
    """Match the captured Python inventory and this entry to the committed tree."""
    owner = record['owner']
    revision, tree = owner['source_revision'], owner['source_tree']
    checked = helper.source_inventory(root, revision, tree)
    if checked != captured:
        raise ValueError('captured source inventory differs from the committed checkout')
    worker = checked.get('looking_glass/agreement_subject_executor_v1.py')
    if type(worker) is not bytes or hashlib.sha256(worker).hexdigest() != owner['executor_sha256']:
        raise ValueError('worker source differs from the prelaunch binding')
    raw_entry = _read_asset(entry_path, 8 * 1024 * 1024)
    if (hashlib.sha256(raw_entry).hexdigest() != owner['entry_script_sha256']
            or helper._git(root, 'show', revision + ':tools/agreement_subject_cpu_v1.py') != raw_entry):
        raise ValueError('entry source differs from the committed prelaunch binding')
    for predecessor in (PROTOCOL_REVISION, CASE_REVISION, DIRECTION_REVISION,
                        RESERVATION_REVISION, record['exposure']['latest_revision']):
        helper._git(root, 'merge-base', '--is-ancestor', predecessor, revision)
    return checked


def verify_published_file(root, helper, revision, relative_path, expected_sha256, limit):
    """Match an ordinary checkout file to both the recorded commit and digest."""
    root = Path(root)
    path = root / relative_path
    helper._ordinary(path, root)
    raw = _read_asset(path, limit)
    if (hashlib.sha256(raw).hexdigest() != expected_sha256
            or helper._git(root, 'show', revision + ':' + relative_path) != raw):
        raise ValueError('published artifact differs from its recorded commit')
    return raw


def capture_and_claim(root, record, study_root, venv, helper, entry_path):
    """Capture project bytes, then claim the reviewed CPU-only supervisor lease."""
    proxy = {'owner': {'executor_sha256': DIRECTION_WORKER_SHA256,
                       'entry_script_sha256': DIRECTION_ENTRY_SHA256},
             'runtime': record['runtime']}
    helper._early_precheck(proxy)
    captured = helper._capture_local_sources(root, proxy)
    worker = captured.get('looking_glass/agreement_subject_executor_v1.py')
    if type(worker) is not bytes or hashlib.sha256(worker).hexdigest() != record['owner']['executor_sha256']:
        raise ValueError('captured worker differs from the prelaunch binding')
    raw_entry = _read_asset(entry_path, 8 * 1024 * 1024)
    if hashlib.sha256(raw_entry).hexdigest() != record['owner']['entry_script_sha256']:
        raise ValueError('captured entry differs from the prelaunch binding')
    sys.meta_path.insert(0, helper._SourceFinder(root, captured))
    session, lease = helper._claim_cpu(study_root)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU visibility changed during claim')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    helper._install_venv(venv)
    return session, lease, captured


def claim_attempt(study_root, output, prelaunch_sha256, helper):
    """Persist a case-bound one-shot claim in the shared CPU custody root."""
    root = Path(study_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError('CPU study root must be an existing absolute directory')
    claim = root / ('agreement-subject-v1-' + CASE_SHA256 + '.claim')
    helper._write(claim, {'case_sha256': CASE_SHA256, 'output': str(output),
                          'prelaunch_sha256': prelaunch_sha256})
    return claim


def verify_binary_assets(direction_dir, reference_path, reference_manifest_path):
    """Read exact public direction/reference bytes; never derive them anew."""
    directory = Path(direction_dir)
    if not directory.is_absolute() or not directory.is_dir() or directory.is_symlink():
        raise ValueError('direction directory is invalid')
    if getattr(directory.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('direction directory cannot be a reparse point')
    manifest = _read_asset(directory / 'manifest.json', 1024 * 1024)
    if hashlib.sha256(manifest).hexdigest() != DIRECTION_MANIFEST_SHA256:
        raise ValueError('direction manifest differs')
    directions = {}
    for name, expected in DIRECTION_SHA256.items():
        raw = _read_asset(directory / (name + '.f64le'), WIDTH * 8)
        if len(raw) != WIDTH * 8 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('direction bytes differ')
        values = struct.unpack('<' + str(WIDTH) + 'd', raw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('direction is not finite')
        directions[name] = values
    reference = _read_asset(reference_path, VOCABULARY * 8)
    manifest = _read_asset(reference_manifest_path, 1024 * 1024)
    if (len(reference) != VOCABULARY * 8 or hashlib.sha256(reference).hexdigest() != REFERENCE_SHA256
            or hashlib.sha256(manifest).hexdigest() != REFERENCE_MANIFEST_SHA256):
        raise ValueError('reference bytes or manifest differ')
    return directions


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _require(actual, expected):
    if type(actual) is not dict or any(type(actual.get(key)) is not type(value)
                                       or actual.get(key) != value for key, value in expected.items()):
        raise ValueError('prelaunch fixed fields differ')


def _digest(value, length):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def _utc(value):
    if type(value) is not str:
        raise ValueError('observation time must be UTC')
    try:
        stamp = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.utcoffset() != datetime.timedelta(0):
            raise ValueError('not UTC')
    except ValueError as error:
        raise ValueError('observation time must be UTC') from error


def validate_prelaunch(record):
    """Validate fixed identities and recorded operator observations, not authorization."""
    if not (_digest(RESERVATION_REVISION, 40) and _digest(RESERVATION_SHA256, 64)
            and type(RESERVATION_VERSION) is str and RESERVATION_VERSION):
        raise RuntimeError('published reservation is not pinned')
    _require(record, {'schema': 'distilgpt2-agreement-subject-prelaunch-v1'})
    if record.get('purpose') not in ('rehearsal', 'study'):
        raise ValueError('prelaunch purpose differs')
    owner = record.get('owner')
    if (type(owner) is not dict or not all(_digest(owner.get(key), size) for key, size in
            (('source_revision', 40), ('source_tree', 40), ('executor_sha256', 64), ('entry_script_sha256', 64)))):
        raise ValueError('invalid owner identity')
    _require(record.get('protocol'), {'revision': PROTOCOL_REVISION, 'path': PROTOCOL_PATH, 'sha256': PROTOCOL_SHA256})
    _require(record.get('cases'), {'revision': CASE_REVISION, 'path': CASE_PATH, 'sha256': CASE_SHA256,
                                   'ordered_rows_sha256': ROWS_SHA256})
    _require(record.get('directions'), {'revision': DIRECTION_REVISION, 'manifest_sha256': DIRECTION_MANIFEST_SHA256,
                                       **{name + '_sha256': value for name, value in DIRECTION_SHA256.items()}})
    _require(record.get('reference'), {'sha256': REFERENCE_SHA256, 'manifest_sha256': REFERENCE_MANIFEST_SHA256})
    _require(record.get('checkpoint'), {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                                       'weights_sha256': WEIGHTS_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256})
    _require(record.get('limits'), LIMITS)
    runtime = record.get('runtime')
    _require(runtime, {**VERSIONS, 'cpu_threads': 1, 'cpu_interop_threads': 1, 'claim_cuda_visibility': ''})
    if runtime.get('environment') != REQUIRED_ENV:
        raise ValueError('prelaunch runtime environment differs')
    exposure = record.get('exposure')
    _require(exposure, {'reservation_revision': RESERVATION_REVISION, 'reservation_sha256': RESERVATION_SHA256,
                        'reservation_version': RESERVATION_VERSION})
    if (not _digest(exposure.get('latest_revision'), 40) or not _digest(exposure.get('latest_sha256'), 64)
            or type(exposure.get('latest_version')) is not str or not exposure['latest_version']
            or type(exposure.get('latest_module')) is not str
            or re.fullmatch(r'looking_glass/pretrained_exposures_v[0-9]+\.py', exposure['latest_module']) is None):
        raise ValueError('latest exposure identity differs')
    if (not _digest(exposure.get('operator_revision'), 40) or not _digest(exposure.get('operator_sha256'), 64)
            or not _digest(exposure.get('operator_observation_sha256'), 64)
            or type(exposure.get('operator_version')) is not str or not exposure['operator_version']):
        raise ValueError('operator exposure observation differs')
    _utc(exposure.get('operator_refreshed_utc'))
    custody = record.get('custody')
    if type(custody) is not dict:
        raise ValueError('custody observations are absent')
    natural, injection = custody.get('natural'), custody.get('injection')
    _require(natural, {'status': 'clear', 'case_sha256': CASE_SHA256,
                       'support_receipt_sha256': NATURAL_RECEIPT_SHA256, 'scope': NATURAL_SCOPE})
    _require(injection, {'status': 'no_protected_rows', 'direct_target_conflict': 'none reported'})
    for observation in (natural, injection):
        if not _digest(observation.get('observation_sha256'), 64):
            raise ValueError('custody observation hash is absent')
        _utc(observation.get('refreshed_utc'))
    if (record.get('independent_reader_status') not in ('reviewed', 'under_review')
            or type(record.get('scope')) is not str or not record['scope']):
        raise ValueError('independent reader or scope observation is absent')
    _utc(record.get('recorded_utc'))
    return record


def verify_case_document(raw):
    """Bind the published raw case file and its compact ordered token rows."""
    if type(raw) is not bytes or len(raw) > 2 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != CASE_SHA256:
        raise ValueError('case bytes differ from the published file')
    def invalid_constant(_value):
        raise ValueError('nonfinite JSON value')
    document = json.loads(raw, object_pairs_hook=_unique_json_object, parse_constant=invalid_constant)
    fixed = {'schema': 'distilgpt2-agreement-subject-dependence-cases-v1',
             'protocol_revision': PROTOCOL_REVISION, 'protocol_sha256': PROTOCOL_SHA256,
             'reference_sha256': REFERENCE_SHA256, 'model_id': MODEL_ID,
             'model_revision': MODEL_REVISION, 'tokenizer_sha256': TOKENIZER_SHA256,
             'answer_ids': [value for _, value in ANSWERS], 'expected_forwards': 720,
             'ordered_rows_sha256': ROWS_SHA256}
    if type(document) is not dict or any(type(document.get(key)) is not type(value)
                                         or document.get(key) != value for key, value in fixed.items()):
        raise ValueError('case document fixed fields differ')
    rows = document.get('cases')
    if type(rows) is not list or len(rows) != 80:
        raise ValueError('case count differs')
    for row in rows:
        tokens = row.get('token_ids') if type(row) is dict else None
        if (type(tokens) is not list or not 1 <= len(tokens) <= 12
                or any(type(token) is not int or not 0 <= token < VOCABULARY for token in tokens)):
            raise ValueError('case token rows are invalid')
    ordered = json.dumps([row['token_ids'] for row in rows], separators=(',', ':')).encode('ascii')
    if hashlib.sha256(ordered).hexdigest() != ROWS_SHA256:
        raise ValueError('ordered case rows differ')
    return rows


def verify_custody(record, support_receipt_path, natural_observation_path, injection_observation_path):
    """Check saved operator observations without claiming authenticated custody."""
    receipt = _read_asset(support_receipt_path, 256 * 1024)
    natural = _read_asset(natural_observation_path, 256 * 1024)
    injection = _read_asset(injection_observation_path, 256 * 1024)
    if (hashlib.sha256(receipt).hexdigest() != NATURAL_RECEIPT_SHA256
            or hashlib.sha256(natural).hexdigest() != record['custody']['natural']['observation_sha256']
            or hashlib.sha256(injection).hexdigest() != record['custody']['injection']['observation_sha256']):
        raise ValueError('custody receipt or observation digest differs')
    def invalid(_value):
        raise ValueError('nonfinite custody record')
    receipt_doc, natural_doc, injection_doc = (json.loads(raw, object_pairs_hook=_unique_json_object,
                                                           parse_constant=invalid)
                                               for raw in (receipt, natural, injection))
    _require(receipt_doc, {'schema': 'mlg-agreement-subject-public-support-comparison-v1', 'status': 'clear',
                           'proposal_file_sha256': CASE_SHA256, 'proposal_ordered_rows_sha256': ROWS_SHA256})
    _require(natural_doc, {'schema': 'agreement-subject-natural-refresh-v1', 'status': 'clear',
                           'case_sha256': CASE_SHA256, 'support_receipt_sha256': NATURAL_RECEIPT_SHA256,
                           'scope': NATURAL_SCOPE, 'successor_support_sha256': SUCCESSOR_SUPPORT_SHA256,
                           'refreshed_utc': record['custody']['natural']['refreshed_utc']})
    if natural_doc.get('successor_status') not in ('undrawn', 'drawn_covered_by_public_successor_inputs'):
        raise ValueError('natural successor coverage is unavailable')
    _require(injection_doc, {'schema': 'agreement-subject-injection-refresh-v1', 'status': 'no_protected_rows',
                             'direct_target_conflict': 'none reported', 'site': 'transformer.h.3.attn',
                             'refreshed_utc': record['custody']['injection']['refreshed_utc']})
    return receipt, natural, injection


def verify_tokenizer(tokenizer, cases):
    """Recheck every full input and every fixed continuation boundary."""
    for text, expected in ANSWERS:
        if tokenizer.encode(text, add_special_tokens=False) != [expected]:
            raise ValueError('answer tokenization differs')
    for row in cases:
        text, expected = row['text'], row['token_ids']
        if tokenizer.encode(text, add_special_tokens=False) != expected:
            raise ValueError('case tokenization differs')
        for answer, target in ANSWERS:
            if tokenizer.encode(text + answer, add_special_tokens=False) != expected + [target]:
                raise ValueError('continuation tokenization differs')


def _grouped_candidates(item):
    groups = item.get('candidate_token_groups', {})
    if (type(groups) is not dict or len(groups) > 128
            or any(type(key) is not str or not key or len(key) > 128 for key in groups)):
        raise ValueError('invalid inventory candidate group')
    values = []
    for group in groups.values():
        if (type(group) is not list or len(group) > 512
                or any(type(token) is not int or not 0 <= token < VOCABULARY for token in group)):
            raise ValueError('invalid inventory candidate group')
        values.extend(group)
        if len(values) > 512:
            raise ValueError('inventory candidate groups exceed their bound')
    return values


def known_sequences(records):
    """Expand base rows and all simple or grouped single-token continuations."""
    if type(records) is not list or len(records) > 100000:
        raise ValueError('invalid inventory records')
    sequences = set()
    for item in records:
        if type(item) is not dict:
            raise ValueError('invalid inventory record')
        tokens, candidates = item.get('token_ids'), item.get('candidate_token_ids', [])
        if (type(tokens) is not list or not 1 <= len(tokens) <= 128
                or any(type(token) is not int or not 0 <= token < VOCABULARY for token in tokens)
                or type(candidates) is not list or len(candidates) > 512
                or any(type(token) is not int or not 0 <= token < VOCABULARY for token in candidates)):
            raise ValueError('invalid inventory row or continuations')
        candidates = candidates + _grouped_candidates(item)
        target = item.get('target_token_id')
        if target is not None:
            if type(target) is not int or not 0 <= target < VOCABULARY:
                raise ValueError('invalid inventory target')
            candidates.append(target)
        base = tuple(tokens)
        sequences.add(base)
        sequences.update(base + (token,) for token in candidates)
    return sequences


def validate_exposure(payload, cases):
    """Check the 80 own planned records and all other known row prefixes."""
    records = payload.get('records') if type(payload) is dict else None
    if type(records) is not list or len(records) > 100000 or type(cases) is not list or len(cases) != 80:
        raise ValueError('invalid exposure inventory or cases')
    expected = {'planned-' + row['case_id']: row for row in cases}
    if len(expected) != 80:
        raise ValueError('duplicate planned case identities')
    candidates = [token for _, token in ANSWERS]
    proposed = [tuple(row['token_ids']) for row in cases]
    complete = proposed + [row + (token,) for row in proposed for token in candidates]
    proposed_set = set(complete)
    prefixes = {row[:size] for row in complete for size in range(1, len(row) + 1)}
    seen, own = set(), set()
    for item in records:
        if type(item) is not dict or type(item.get('record_id')) is not str or item['record_id'] in seen:
            raise ValueError('invalid or repeated inventory identity')
        seen.add(item['record_id'])
        tokens = item.get('token_ids')
        if (type(tokens) is not list or not 1 <= len(tokens) <= 128
                or any(type(token) is not int or not 0 <= token < VOCABULARY for token in tokens)):
            raise ValueError('invalid inventory token row')
        grouped = _grouped_candidates(item)
        if item['record_id'] in expected:
            row = expected[item['record_id']]
            required = {'category': 'planned', 'case_role': row['role'], 'public_text': row['text'],
                        'token_ids': row['token_ids'], 'candidate_token_ids': candidates}
            if grouped or any(type(item.get(key)) is not type(value) or item.get(key) != value
                              for key, value in required.items()):
                raise ValueError('planned record differs from its case')
            own.add(item['record_id'])
            continue
        other_candidates = item.get('candidate_token_ids', [])
        if (type(other_candidates) is not list or len(other_candidates) > 512
                or any(type(token) is not int or not 0 <= token < VOCABULARY for token in other_candidates)):
            raise ValueError('invalid inventory continuations')
        target = item.get('target_token_id')
        if target is not None and (type(target) is not int or not 0 <= target < VOCABULARY):
            raise ValueError('invalid inventory target')
        base = tuple(tokens)
        known = [base] + [base + (token,) for token in other_candidates + grouped]
        if target is not None:
            known.append(base + (target,))
        if any(row in prefixes or any(row[:size] in proposed_set for size in range(1, len(row) + 1))
               for row in known):
            raise ValueError('latest inventory has a known case collision')
    if own != set(expected):
        raise ValueError('latest inventory lacks an exact planned case')


def verify_latest_inventory(root, record, path, cases, helper, sources):
    """Bind the fresh inventory to committed modules and preserve known rows."""
    exposure = record['exposure']
    raw = _read_asset(path, 16 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != exposure['latest_sha256']:
        raise ValueError('latest inventory digest differs')
    def invalid(_value):
        raise ValueError('nonfinite inventory value')
    payload = json.loads(raw, object_pairs_hook=_unique_json_object, parse_constant=invalid)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=True, allow_nan=False).encode('ascii')
    if canonical(payload) != raw or payload.get('version') != exposure['latest_version']:
        raise ValueError('latest inventory is not canonical or versioned')
    validate_exposure(payload, cases)
    latest_module = exposure['latest_module']
    if not isinstance(RESERVATION_MODULE, str):
        raise ValueError('reservation inventory module is not pinned')
    helper._git(root, 'merge-base', '--is-ancestor', RESERVATION_REVISION, exposure['latest_revision'])
    for revision, path_name in ((RESERVATION_REVISION, RESERVATION_MODULE),
                                (exposure['latest_revision'], latest_module)):
        if (path_name not in sources or helper._git(root, 'show', revision + ':' + path_name) != sources[path_name]):
            raise ValueError('inventory source differs from its committed module')
    previous = importlib.import_module(RESERVATION_MODULE[:-3].replace('/', '.'))
    latest = importlib.import_module(latest_module[:-3].replace('/', '.'))
    if (previous.INVENTORY_VERSION != RESERVATION_VERSION or previous.INVENTORY_SHA256 != RESERVATION_SHA256
            or latest.INVENTORY_VERSION != exposure['latest_version']
            or latest.INVENTORY_SHA256 != exposure['latest_sha256']):
        raise ValueError('inventory module identity differs')
    previous_payload, latest_payload = previous.inventory_payload(), latest.inventory_payload()
    if (hashlib.sha256(canonical(previous_payload)).hexdigest() != RESERVATION_SHA256
            or canonical(latest_payload) != raw
            or not known_sequences(previous_payload['records']) <= known_sequences(latest_payload['records'])):
        raise ValueError('latest inventory does not preserve the reservation')
    return raw


def verify_operator_inventory(record, path, observation_path, reconstructed_raw, cases):
    """Check an operator-observed successor without claiming source reconstruction."""
    exposure = record['exposure']
    raw = _read_asset(path, 16 * 1024 * 1024)
    observation = _read_asset(observation_path, 256 * 1024)
    if (hashlib.sha256(raw).hexdigest() != exposure['operator_sha256']
            or hashlib.sha256(observation).hexdigest() != exposure['operator_observation_sha256']
            or hashlib.sha256(reconstructed_raw).hexdigest() != exposure['latest_sha256']):
        raise ValueError('operator inventory or observation digest differs')
    def invalid(_value):
        raise ValueError('nonfinite operator inventory value')
    payload = json.loads(raw, object_pairs_hook=_unique_json_object, parse_constant=invalid)
    reconstructed = json.loads(reconstructed_raw, object_pairs_hook=_unique_json_object, parse_constant=invalid)
    observed = json.loads(observation, object_pairs_hook=_unique_json_object, parse_constant=invalid)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=True, allow_nan=False).encode('ascii')
    if canonical(payload) != raw or payload.get('version') != exposure['operator_version']:
        raise ValueError('operator inventory is not canonical or versioned')
    validate_exposure(payload, cases)
    if not known_sequences(reconstructed['records']) <= known_sequences(payload['records']):
        raise ValueError('operator inventory does not preserve reconstructed known rows')
    _require(observed, {'schema': 'agreement-subject-exposure-refresh-v1', 'status': 'clear',
                        'case_sha256': CASE_SHA256, 'reconstructed_sha256': exposure['latest_sha256'],
                        'operator_revision': exposure['operator_revision'],
                        'operator_version': exposure['operator_version'],
                        'operator_sha256': exposure['operator_sha256'],
                        'refreshed_utc': exposure['operator_refreshed_utc']})
    return raw, observation


def run_owner(args, record, prelaunch_raw):
    """Run one supervised child; completion remains a candidate for its parent."""
    global _OWNER_ROOTS
    root = Path(__file__).resolve().parents[1]
    helper = load_reviewed_helper(root)
    child_custody = Path(args.child_custody)
    if (not child_custody.is_absolute() or not child_custody.parent.is_dir()
            or os.path.lexists(child_custody) or child_custody.resolve().is_relative_to(root)):
        raise ValueError('child custody path must be a new absolute file outside the checkout')
    session, lease, captured = capture_and_claim(root, record, Path(args.study_root), Path(args.venv),
                                                  helper, Path(__file__).resolve())
    _OWNER_ROOTS = (session, lease, captured)
    session.check(lease)
    helper._write(child_custody, {'schema': 'agreement-direction-supervised-child-v1',
                                  'run_id': session.run_id, 'owner_pid': os.getpid(),
                                  'prelaunch_sha256': args.prelaunch_sha256,
                                  'mode': 'validate_only' if args.validate_only else 'study'})
    versions = helper._runtime_precheck(record)
    sources = verify_source(root, record, captured, helper, Path(__file__).resolve())
    protocol = verify_published_file(root, helper, PROTOCOL_REVISION, PROTOCOL_PATH, PROTOCOL_SHA256, 256 * 1024)
    case_raw = verify_published_file(root, helper, CASE_REVISION, CASE_PATH, CASE_SHA256, 2 * 1024 * 1024)
    if _read_asset(args.case_file, 2 * 1024 * 1024) != case_raw:
        raise ValueError('supplied case file differs from its published copy')
    cases = verify_case_document(case_raw)
    support = verify_published_file(root, helper, CASE_REVISION, NATURAL_RECEIPT_PATH,
                                    NATURAL_RECEIPT_SHA256, 256 * 1024)
    reference = verify_published_file(root, helper, PROTOCOL_REVISION, REFERENCE_PATH,
                                      REFERENCE_SHA256, VOCABULARY * 8)
    reference_manifest = verify_published_file(root, helper, PROTOCOL_REVISION, REFERENCE_MANIFEST_PATH,
                                               REFERENCE_MANIFEST_SHA256, 1024 * 1024)
    direction_manifest = verify_published_file(root, helper, DIRECTION_REVISION, DIRECTION_PATH + '/manifest.json',
                                               DIRECTION_MANIFEST_SHA256, 1024 * 1024)
    direction_raw = {name: verify_published_file(root, helper, DIRECTION_REVISION,
                                                 DIRECTION_PATH + '/' + name + '.f64le', digest, WIDTH * 8)
                     for name, digest in DIRECTION_SHA256.items()}
    published_directions = verify_binary_assets(root / DIRECTION_PATH, root / REFERENCE_PATH,
                                                 root / REFERENCE_MANIFEST_PATH)
    directions = verify_binary_assets(Path(args.direction_dir), Path(args.reference_file),
                                      Path(args.reference_manifest))
    if directions != published_directions:
        raise ValueError('supplied directions differ from published values')
    inventory = verify_latest_inventory(root, record, Path(args.latest_exposure), cases, helper, sources)
    operator_inventory, exposure_observation = verify_operator_inventory(
        record, Path(args.operator_exposure), Path(args.operator_observation), inventory, cases)
    support_copy, natural, injection = verify_custody(record, root / NATURAL_RECEIPT_PATH,
                                                       Path(args.natural_observation),
                                                       Path(args.injection_observation))
    if support_copy != support:
        raise ValueError('published support receipt changed')
    session.check(lease)
    cache, output, study_root = Path(args.cache_dir), Path(args.output_dir), Path(args.study_root)
    if (not cache.is_absolute() or not cache.is_dir() or cache.is_symlink()
            or not output.is_absolute() or not output.parent.is_dir() or os.path.lexists(output)
            or output.resolve().is_relative_to(root) or output.resolve().is_relative_to(cache.resolve())):
        raise ValueError('cache or fresh output path is invalid')
    if shutil.disk_usage(output.parent).free < LIMITS['minimum_free_disk_bytes']:
        raise ValueError('insufficient free space before one-shot claim')
    binding = {'prelaunch_sha256': args.prelaunch_sha256, 'reservation_sha256': RESERVATION_SHA256,
               'latest_exposure_sha256': record['exposure']['operator_sha256']}
    observation = {'source_revision': record['owner']['source_revision'], 'source_tree': record['owner']['source_tree'],
                   'source_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(sources.items())},
                   'binding': binding, 'runtime_versions': versions, 'forwards': 0,
                   'owner_pid': os.getpid(), 'owner_run_id': session.run_id}
    if args.validate_only:
        print(json.dumps({'status': 'no_model_validation', **observation}, sort_keys=True))
        session.attest_clean(lease)
        return 0
    claim_attempt(study_root, output, args.prelaunch_sha256, helper)
    output.mkdir()
    try:
        helper._write(output / 'admission.json', observation)
        for name, raw in (('prelaunch.json', prelaunch_raw), ('protocol.md', protocol),
                          ('cases.json', case_raw), ('reconstructed-exposure.json', inventory),
                          ('latest-exposure.json', operator_inventory), ('exposure-refresh.json', exposure_observation),
                          ('natural-support-comparison.json', support), ('natural-refresh.json', natural),
                          ('injection-refresh.json', injection), ('output-slope.f64le', reference),
                          ('output-reference-manifest.json', reference_manifest),
                          ('direction-manifest.json', direction_manifest)):
            helper._write_raw(output / name, raw)
        for name, raw in direction_raw.items():
            helper._write_raw(output / (name + '.f64le'), raw)
        session.check(lease)
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.set_flush_denormal(False)
        runtime = helper._verify_cpu_runtime(torch)
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.agreement_subject_executor_v1 import run_subject_study
        from looking_glass.pretrained import load_distilgpt2
        journal = AgreementJournal(output / 'attempt', {'scope': 'public prospective agreement subject-dependence',
                                                        'external_binding': binding,
                                                        'source_revision': record['owner']['source_revision']},
                                   max_forward_bytes=LIMITS['per_forward_bytes'])
        loaded = load_distilgpt2(cache_dir=str(cache), allow_download=False)
        if (loaded.metadata['model_id'] != MODEL_ID or loaded.metadata['model_revision'] != MODEL_REVISION
                or loaded.metadata['weights_sha256'] != WEIGHTS_SHA256
                or loaded.metadata['asset_sha256']['tokenizer.json'] != TOKENIZER_SHA256
                or loaded.metadata['parameter_count'] != 81912576
                or loaded.model.config.n_embd != WIDTH or loaded.model.config.vocab_size != VOCABULARY
                or len(loaded.model.transformer.h) != 6 or torch.cuda.is_initialized()):
            raise ValueError('loaded model or architecture differs')
        verify_tokenizer(loaded.tokenizer, cases)
        if ({name: os.environ.get(name) for name in REQUIRED_ENV} != REQUIRED_ENV
                or helper._verify_cpu_runtime(torch) != runtime):
            raise ValueError('CPU environment changed during model loading')
        helper._write(output / 'model-admission.json', {'status': 'loaded_before_forward', 'runtime': runtime,
                                                        'checkpoint': loaded.metadata, 'case_sha256': CASE_SHA256,
                                                        'binding': binding})
        import gc
        import weakref
        model_ref = weakref.ref(loaded.model)
        result = run_subject_study(loaded.model, journal, cases, directions, binding=binding,
                                   custody_check=lambda: session.check(lease))
        del loaded
        gc.collect()
        if (model_ref() is not None or torch.cuda.is_initialized()
                or helper._verify_cpu_runtime(torch) != runtime
                or {name: os.environ.get(name) for name in REQUIRED_ENV} != REQUIRED_ENV):
            raise ValueError('CPU model disposal was not verified')
        session.check(lease)
        if verify_source(root, record, captured, helper, Path(__file__).resolve()) != sources:
            raise ValueError('source changed during the attempt')
        physical = helper._physical_bytes(output)
        if physical > LIMITS['physical_attempt_bytes'] - LIMITS['external_receipt_allowance_bytes']:
            raise ValueError('physical attempt bytes exceeded the limit')
        helper._write(output / 'entry-result.json', {
            'status': 'complete_candidate', 'binding': binding, 'worker': result,
            'physical_attempt_bytes_before_result': physical, 'owner_pid': os.getpid(),
            'owner_run_id': session.run_id, 'model_disposed': True,
            'external_receipt_allowance_bytes': LIMITS['external_receipt_allowance_bytes'],
            'scope': 'candidate only; external exit and independent scientific audit required'})
        print(json.dumps({'status': 'complete_candidate', 'forwards': result['forwards'], 'binding': binding},
                         sort_keys=True))
        session.attest_clean(lease)
        return 0
    except BaseException as error:
        try:
            helper._write(output / 'entry-failure.json', {
                'status': 'cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                'error_type': type(error).__name__, 'binding': binding,
                'scope': 'incomplete; retain all partial artifacts and do not retry'})
        except BaseException as write_error:
            if isinstance(write_error, (KeyboardInterrupt, SystemExit)):
                raise write_error from error
        raise


def main(argv=None):
    if not _digest(RESERVATION_REVISION, 40):
        raise RuntimeError('published case and planned reservation bindings are not pinned')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prelaunch', 'prelaunch-sha256', 'latest-exposure', 'operator-exposure', 'operator-observation',
                 'natural-observation',
                 'injection-observation', 'case-file', 'direction-dir', 'reference-file',
                 'reference-manifest', 'cache-dir', 'output-dir', 'study-root', 'venv', 'child-custody'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    raw = _read_asset(args.prelaunch, 256 * 1024)
    if not _digest(args.prelaunch_sha256, 64) or hashlib.sha256(raw).hexdigest() != args.prelaunch_sha256:
        raise ValueError('prelaunch digest differs')
    def invalid(_value):
        raise ValueError('nonfinite prelaunch JSON value')
    record = json.loads(raw, object_pairs_hook=_unique_json_object, parse_constant=invalid)
    validate_prelaunch(record)
    if record['purpose'] != ('rehearsal' if args.validate_only else 'study'):
        raise ValueError('prelaunch purpose does not match entry mode')
    return run_owner(args, record, raw)


if __name__ == '__main__':
    raise SystemExit(main())
