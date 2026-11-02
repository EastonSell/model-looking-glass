"""Independent saved-tensor replay for the prospective agreement study.

It checks recorded content and computes scientific summaries without importing
the model or owner. The caller must separately establish exact public source,
cases, directions, custody, exposure and a direct-child exit. A consistent
journal is not proof that a claimed process ran or that examples were blind.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import struct

from . import agreement_science_v1 as science


def strict_json(raw):
    """Parse UTF-8 JSON without duplicate keys or nonfinite numeric values."""
    if type(raw) is not bytes:
        raise ValueError('JSON input must be bytes')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def finite_float(text):
        result = float(text)
        if not math.isfinite(result):
            raise ValueError('nonfinite JSON number')
        return result
    def reject_constant(_):
        raise ValueError('nonfinite JSON constant')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                          parse_float=finite_float, parse_constant=reject_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('invalid JSON encoding or nesting') from exc


def validate_terminal(terminal, state_hash, *, event_bytes, tensor_bytes):
    if (type(state_hash) is not str or re.fullmatch(r'[0-9a-f]{64}', state_hash, re.ASCII) is None
            or type(event_bytes) is not int or not 0 < event_bytes <= 64 * 1024 * 1024
            or type(tensor_bytes) is not int or not 0 < tensor_bytes <= 512 * 1024 * 1024):
        raise ValueError('invalid terminal audit quantities')
    expected = {'status': 'complete_candidate', 'events': 809, 'event_log_bytes': event_bytes,
                'retained_tensor_bytes': tensor_bytes, 'storage_accounting_complete': True,
                'detail': {'status': 'complete_candidate', 'forwards': 784,
                           'model_state_sha256_before_after': state_hash,
                           'retained_tensor_bytes': tensor_bytes}}
    if not science.same_recorded_value(terminal, expected):
        raise ValueError('terminal candidate or storage accounting differs')
    return True


def _margin_vector(values):
    if len(values) <= 547:
        raise ValueError('vocabulary is too small for the fixed token IDs')
    return {'present': values[318] - values[389], 'past': values[373] - values[547],
            'the_vs_a': values[262] - values[257], 'in_vs_on': values[287] - values[319]}


def _require_equal(actual, expected, message):
    if not science.same_recorded_value(actual, expected):
        raise ValueError(message)


def validate_recorded_runtime(contract):
    """Reject contradictions in the real owner's recorded runtime contract."""
    fields = {'kind', 'expected_forwards', 'seed', 'environment', 'execution_configuration',
              'model_state_sha256', 'external_binding'}
    if (type(contract) is not dict or set(contract) != fields
            or contract['kind'] != 'run_contract'
            or type(contract['expected_forwards']) is not int or contract['expected_forwards'] != 784
            or type(contract['seed']) is not int or contract['seed'] != 20261030):
        raise ValueError('recorded run contract fields differ')
    environment = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'MKL_CBWR': None, 'ATEN_CPU_CAPABILITY': None}
    numeric = [['deterministic_algorithms', True], ['deterministic_warn_only', False],
               ['cpu_threads', 1], ['cpu_interop_threads', 1],
               ['float32_matmul_precision', 'highest'], ['calling_thread_denormal_product_bits', 0x00400000]]
    _require_equal(contract['environment'], environment, 'recorded environment differs')
    configuration = contract['execution_configuration']
    if (type(configuration) is not list or len(configuration) != 2
            or type(configuration[1]) is not list or not configuration[1]):
        raise ValueError('recorded execution configuration is malformed')
    _require_equal(configuration[0], numeric, 'recorded numeric runtime differs')
    root = configuration[1][0]
    if (type(root) is not list or len(root) != 3 or root[0] != ''
            or type(root[1]) is not list or type(root[2]) is not list):
        raise ValueError('recorded root model configuration is malformed')
    values = {}
    for field in root[2]:
        if (type(field) is not list or len(field) != 2 or type(field[0]) is not str
                or field[0] in values):
            raise ValueError('recorded root model configuration has duplicate or invalid fields')
        values[field[0]] = field[1]
    expected = {'use_cache': ['bool', False], '_attn_implementation': ['str', 'eager'],
                'n_layer': ['int', 6], 'n_embd': ['int', 768], 'vocab_size': ['int', 50257]}
    for name, value in expected.items():
        if name not in values or not science.same_recorded_value(values[name], value):
            raise ValueError('recorded root model configuration differs')
    return True


def audit_records(records, cases, tensor_dir, directions, *, expected_binding,
                  width=768, vocabulary=50257, allow_synthetic=False):
    """Verify all ordered witnesses and score them; synthetic widths are test-only."""
    if type(allow_synthetic) is not bool or type(width) is not int or type(vocabulary) is not int:
        raise ValueError('invalid audit configuration')
    if not allow_synthetic and (width != 768 or vocabulary != 50257 or type(expected_binding) is not dict):
        raise ValueError('the real study requires fixed dimensions and an independent binding')
    if not 1 <= width <= 768 or not 548 <= vocabulary <= 50257:
        raise ValueError('unsupported dimensions')
    if type(directions) is not dict or set(directions) != {'u', 'v1', 'v2'}:
        raise ValueError('three fixed directions are required')
    for values in directions.values():
        if (type(values) not in (list, tuple) or len(values) != width
                or any(type(value) is not float or not math.isfinite(value) for value in values)
                or abs(math.sqrt(math.fsum(value * value for value in values)) - 1.0) > 1e-12):
            raise ValueError('invalid unit direction')
    for left, right in (('u', 'v1'), ('u', 'v2'), ('v1', 'v2')):
        if abs(math.fsum(a * b for a, b in zip(directions[left], directions[right]))) > 1e-12:
            raise ValueError('directions are not orthogonal')
    science.validate_journal_layout(records, cases)
    contract = records[0]['event']
    if not allow_synthetic:
        validate_recorded_runtime(contract)
    _require_equal(contract.get('external_binding'), expected_binding, 'external binding differs')
    state_hash = contract.get('model_state_sha256')
    if type(state_hash) is not str or re.fullmatch(r'[0-9a-f]{64}', state_hash, re.ASCII) is None:
        raise ValueError('invalid recorded model-state hash')

    by_id = {case['case_id']: case for case in cases}
    captures, clean_raw, clean_values, geometries, centered = {}, {}, {}, {}, {}
    scores, doses, vocabulary_reports, seen = {}, {}, {}, {}
    delivery_reports, centering_reports = {}, []
    targets = (318, 389, 373, 547, 262, 257, 287, 319)

    def payload(reference, shape):
        raw = science.read_saved_tensor(tensor_dir, reference, shape)
        digest = reference['sha256']
        if digest in seen and seen[digest] != len(raw):
            raise ValueError('a repeated digest has inconsistent size')
        seen[digest] = len(raw)
        return raw

    for record in records[1:]:
        event, refs = record['event'], record['tensors']
        kind = event['kind']
        if kind == 'family_geometry':
            family = event['family_index']
            members = cases[4 * family:4 * family + 4]
            computed = science.family_geometry([captures[row['case_id']]['l3_final'] for row in members],
                                               directions['u'])
            actual = {key: value for key, value in event.items() if key not in ('kind', 'family_index')}
            _require_equal(actual, computed, 'saved family geometry differs from the captures')
            geometries[family] = computed
            for i, row in enumerate(members):
                doses[row['case_id']] = computed['doses'][i]
            doses[cases[32 + family]['case_id']] = math.fsum((computed['doses'][0], computed['doses'][2])) / 2.0
            continue
        if kind == 'centering_result':
            if set(event) != {'kind', 'family_index', 'singular_variant', 'achieved_u_gaps'}:
                raise ValueError('saved centering record fields differ')
            family, first = event['family_index'], event['singular_variant']
            left, right = cases[4 * family + first]['case_id'], cases[4 * family + first + 1]['case_id']
            gaps = {}
            for direction in ('u', 'v1', 'v2'):
                a = science.decode_float32(centered[left, direction], expected_count=width)
                b = science.decode_float32(centered[right, direction], expected_count=width)
                gaps['center_' + direction] = (math.fsum(x * y for x, y in zip(a, directions['u']))
                    - math.fsum(x * y for x, y in zip(b, directions['u'])))
            _require_equal(event.get('achieved_u_gaps'), gaps, 'saved centering gap differs')
            centering_reports.append({'family_index': family, 'singular_variant': first,
                                      'achieved_u_gaps': gaps})
            continue

        case, action = by_id[event['case_id']], event['action']
        name, length = case['case_id'], len(case['token_ids'])
        unhooked = action in ('clean', 'clean_repeat', 'clean_after')
        ordinary_fields = {'kind', 'index', 'case_id', 'role', 'family_index', 'subject_number', 'action',
                           'slot', 'input_ids', 'attention_mask', 'plan_metadata',
                           'all_sequence_logits_finite', 'dose'}
        if set(event) != (ordinary_fields if unhooked else ordinary_fields | {'site', 'donor'}):
            raise ValueError('forward event fields differ')
        logits_raw = payload(refs['last_logits'], (vocabulary,))
        logits = science.decode_float32(logits_raw, expected_count=vocabulary)
        if action == 'clean':
            clean_raw[name], clean_values[name] = logits_raw, logits
            captures[name] = {}
        elif name not in clean_raw:
            raise ValueError('forward precedes its clean baseline')
        if action in ('clean_repeat', 'clean_after') and logits_raw != clean_raw[name]:
            raise ValueError('clean repeat differs in full-logit bits')
        scores[name, action] = _margin_vector(logits)
        vocabulary_reports.setdefault(name, {})[action] = science.vocabulary_diagnostics(
            clean_values[name], logits, targets)

        if unhooked:
            if event['dose'] is not None or not science.same_recorded_value(event['plan_metadata'], {'steps': []}):
                raise ValueError('unhooked forward contains a plan or dose')
            continue
        site, plan = science.expected_hook_plan(action, sequence_length=length, width=width)
        _require_equal(event['site'], site, 'hook site differs')
        _require_equal(event['plan_metadata'], plan, 'hook plan differs')
        before = payload(refs['before'], (width,))
        after = payload(refs['after'], (width,))
        whole_before = payload(refs['whole_before'], (1, length, width))
        whole_after = payload(refs['whole_after'], (1, length, width))
        token = 1 if site['token_role'] == 'subject' else length - 1
        science.verify_hook_witness(before, after, whole_before, whole_after,
                                    width=width, sequence_length=length, token_index=token)

        if action.startswith('noop_'):
            if event['dose'] is not None or event['donor'] is not None or before != after or logits_raw != clean_raw[name]:
                raise ValueError('no-op witness or full logits differ')
            captures[name][action[5:]] = before
            continue

        family, variant = case['family_index'], case.get('variant')
        geometry = geometries[family]
        q = doses[name]
        donor = None
        if action in ('same_diff', 'opp_same', 'opp_diff'):
            other = variant ^ {'same_diff': 2, 'opp_same': 1, 'opp_diff': 3}[action]
            donor = cases[4 * family + other]['case_id']
            base = captures[name]['l3_final']
            landed, expected_dose = science.donor_delivery(base, captures[donor]['l3_final'], q)
        else:
            capture = ('l1_final' if action.startswith('l1_') else
                       'l3_subject' if action.startswith('subject_') else 'l3_final')
            base = captures[name][capture]
            if action.startswith('center_'):
                direction = action[7:]
                amount = -geometry['centering'][variant]
            elif action.startswith('l1_') or action.startswith('subject_'):
                direction, amount = 'u', q if action.endswith('plus') else -q
            else:
                direction, sign = action.rsplit('_', 1)
                amount = q if sign == 'plus' else -q
            landed, expected_dose = science.direction_delivery(base, directions[direction], amount)
            if action.startswith('center_'):
                centered[name, direction] = landed
        if event['donor'] != donor or before != base or after != landed:
            raise ValueError('donor identity, pre-action capture or landed bits differ')
        _require_equal(event['dose'], expected_dose, 'recorded dose diagnostics differ')
        if expected_dose['dose_valid'] is not True:
            raise ValueError('delivered dose is outside the fixed tolerance')
        if expected_dose['requested_norm'] == 0 and logits_raw != clean_raw[name]:
            raise ValueError('zero-dose full logits differ from clean')
        delivery_reports.setdefault(name, {})[action] = {'donor': donor, 'metrics': expected_dose}

    report = science.evaluate_study(cases, scores, doses)
    report['family_geometry'] = [{'family_index': family, **geometries[family]} for family in range(8)]
    report['centering_achieved'] = centering_reports
    report['deliveries'] = delivery_reports
    report['vocabulary'] = vocabulary_reports
    report['source_tensor_files'] = len(seen)
    report['source_tensor_bytes'] = sum(seen.values())
    report['dimensions'] = {'width': width, 'vocabulary': vocabulary}
    report['audit_mode'] = 'synthetic_test' if allow_synthetic else 'real_fixed_dimensions'
    report['scope'] = ('Saved-byte numerical consistency only; full-sequence finiteness and process/exposure '
                       'statements are recorded observations, not independently established execution.')
    return report, seen


def _bounded_file(path, limit):
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
                or info.st_size > limit):
            raise ValueError('record is not a bounded ordinary file')
        with path.open('rb') as stream:
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size or len(raw) > limit:
            raise ValueError('record changed or exceeds its bound')
        return raw
    except OSError as exc:
        raise ValueError('record could not be read') from exc


def pinned_file(path, expected_sha256, *, limit):
    if (type(expected_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', expected_sha256, re.ASCII) is None
            or type(limit) is not int or limit <= 0):
        raise ValueError('invalid independent file pin')
    raw = _bounded_file(Path(path), limit)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('file differs from its independent pin')
    return raw


_CASE_SHA = 'a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13'
_MANIFEST_SHA = '91dc4170eff6ac60416e5b3f884f2a28f41f9a4c1ad7339b409e32baeadf3a0e'
_PROTOCOL_SHA = 'f539ad55b347f88641de3b76565104dc7be43e0c5d15f6d8fefdb4173694bbea'
_VECTOR_SHA = {'u': 'd1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0',
               'v1': '4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb',
               'v2': '4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02'}


def load_frozen_inputs(case_path, direction_directory):
    """Load the independently pinned public case and three direction bytes."""
    case_raw = pinned_file(case_path, _CASE_SHA, limit=2 * 1024 * 1024)
    manifest_raw = pinned_file(Path(direction_directory) / 'manifest.json', _MANIFEST_SHA, limit=1024 * 1024)
    case, manifest = strict_json(case_raw), strict_json(manifest_raw)
    if (type(case) is not dict or case.get('schema') != 'distilgpt2-agreement-direction-cases-v1'
            or case.get('protocol_sha256') != _PROTOCOL_SHA
            or case.get('direction_manifest_sha256') != _MANIFEST_SHA
            or type(manifest) is not dict or manifest.get('schema') != 'distilgpt2-agreement-direction-training-v1'
            or not science.same_recorded_value(manifest.get('vectors_sha256'), _VECTOR_SHA)):
        raise ValueError('public case, protocol or direction binding differs')
    cases = case.get('cases')
    science.expected_schedule(cases)
    directions = {}
    for name, digest in _VECTOR_SHA.items():
        raw = pinned_file(Path(direction_directory) / (name + '.f64le'), digest, limit=768 * 8)
        if len(raw) != 768 * 8:
            raise ValueError('direction width differs')
        directions[name] = struct.unpack('<768d', raw)
    return cases, directions


def _ordinary_directory(path):
    try:
        info = path.lstat()
        return (stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
                and not (getattr(info, 'st_file_attributes', 0)
                         & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)))
    except OSError:
        return False


def audit_attempt(directory, cases, directions, *, expected_binding, expected_source_revision,
                  width=768, vocabulary=50257, allow_synthetic=False):
    """Audit a closed journal candidate; this does not verify the child exit."""
    root = Path(directory)
    if not _ordinary_directory(root):
        raise ValueError('attempt directory is absent or aliased')
    if {entry.name for entry in root.iterdir()} != {'events.jsonl', 'started.json', 'terminal.json', 'tensors'}:
        raise ValueError('attempt file inventory differs')
    tensors = root / 'tensors'
    if not _ordinary_directory(tensors):
        raise ValueError('tensor directory is absent or aliased')
    start_raw = _bounded_file(root / 'started.json', 256 * 1024)
    terminal_raw = _bounded_file(root / 'terminal.json', 256 * 1024)
    events_raw = _bounded_file(root / 'events.jsonl', 64 * 1024 * 1024)
    start, terminal = strict_json(start_raw), strict_json(terminal_raw)
    if type(start) is not dict or not events_raw.endswith(b'\n'):
        raise ValueError('start or event log is incomplete')
    if not allow_synthetic:
        expected = {'prelaunch_sha256', 'reservation_sha256', 'latest_exposure_sha256'}
        if (type(expected_binding) is not dict or set(expected_binding) != expected
                or any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value, re.ASCII) is None
                       for value in expected_binding.values())
                or type(expected_source_revision) is not str
                or re.fullmatch(r'[0-9a-f]{40}', expected_source_revision, re.ASCII) is None):
            raise ValueError('independent source and binding pins are required')
        _require_equal(start, {'scope': 'public prospective agreement-direction',
                               'external_binding': expected_binding,
                               'source_revision': expected_source_revision}, 'start binding differs')
    lines = events_raw.splitlines()
    if any(not line or len(line) > 256 * 1024 for line in lines):
        raise ValueError('invalid event line size')
    records = [strict_json(line) for line in lines]
    report, seen = audit_records(records, cases, tensors, directions, expected_binding=expected_binding,
                                 width=width, vocabulary=vocabulary, allow_synthetic=allow_synthetic)
    expected_names = {digest + '.f32le' for digest in seen}
    if {entry.name for entry in tensors.iterdir()} != expected_names:
        raise ValueError('tensor file inventory differs from the journal')
    validate_terminal(terminal, records[0]['event']['model_state_sha256'],
                      event_bytes=len(events_raw), tensor_bytes=sum(seen.values()))
    report['journal'] = {'status': 'consistent_candidate', 'events': len(records),
        'started_sha256': hashlib.sha256(start_raw).hexdigest(),
        'events_sha256': hashlib.sha256(events_raw).hexdigest(),
        'terminal_sha256': hashlib.sha256(terminal_raw).hexdigest()}
    return report
