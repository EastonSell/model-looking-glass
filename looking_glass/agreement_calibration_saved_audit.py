"""Independent CPU-only verification of saved agreement calibration evidence.

This module never imports Torch, opens a CUDA device, loads a checkpoint, or
approves a profile. It verifies consistency of the recorded observations; it
cannot establish that an owner executed faithfully or that memory is bounded.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess


GIB = 1024 ** 3
MIB = 1024 ** 2
TRACE = 2 * MIB
SEED = 20261028
TARGET = [464, 3797, 1474, 262, 9168]
DONOR = [464, 11875, 1474, 262, 9168]
KINDS = ('clean_capture',
         'layer1_attn_cross_patch', 'layer1_mlp_cross_patch',
         'layer2_attn_cross_patch', 'layer2_mlp_cross_patch',
         'layer3_attn_cross_patch', 'layer3_mlp_cross_patch',
         'layer4_attn_cross_patch', 'layer4_mlp_cross_patch',
         'null', 'self_patch', 'block_last_cross_patch', 'block_subject_cross_patch')
PHASES = ('started', 'baseline', 'baseline_check', 'after_observed', 'postcheck', 'completed')
SITES = tuple((f'transformer.h.{layer}.{part}', 4, [] if part == 'mlp' else [0])
              for layer in (1, 2, 3, 4) for part in ('attn', 'mlp')) + (
                  ('transformer.h.5', 4, [0]), ('transformer.h.5', 1, [0]))
_ACTIONS = {
    **{f'layer{layer}_{part}_cross_patch': (f'transformer.h.{layer}.{part}', 4, 'cross_patch')
       for layer in (1, 2, 3, 4) for part in ('attn', 'mlp')},
    'null': ('transformer.h.1.attn', 4, 'null'),
    'self_patch': ('transformer.h.1.attn', 4, 'self_patch'),
    'block_last_cross_patch': ('transformer.h.5', 4, 'cross_patch'),
    'block_subject_cross_patch': ('transformer.h.5', 1, 'cross_patch'),
}
_POLICY = {
    'schema': 'agreement-cuda-capture-v1', 'batch_size': 1, 'sequence_length': 5,
    'clean_sites': 10, 'action_witnesses': ['whole_before', 'before', 'after', 'whole_after'],
    'max_tensor_bytes': MIB, 'max_capture_bytes': 512 * 1024, 'max_source_bytes': 128 * 1024,
    'gpu_trace_allowance_bytes': TRACE, 'host_trace_allowance_bytes': TRACE,
    'full_output_bytes': 5 * 50257 * 4, 'state_hash_chunk_bytes': MIB,
    'measurement_scope': 'state-check-input-copy-plan-forward-offload-sync-disposal-state-check',
    'clone_custody': 'retain-plan-sources-and-assigned-captures-until-verified-drain',
    'calibration_kinds': list(KINDS), 'clean_bindings': SITES, 'seed': SEED,
}


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('unsupported JSON evidence') from error


def _hash(value):
    return hashlib.sha256(_json(value)).hexdigest()


POLICY_SHA256 = _hash(_POLICY)
_WARMUP_POLICY = {'schema': 'agreement-cuda-warmup-v1', 'input_ids': TARGET,
                  'subject_index': 1, 'attention_mask': [1] * 5,
                  'operation': 'ten-site-clean-capture-and-offload', 'seed': SEED,
                  'capture_policy_sha256': POLICY_SHA256}
WARMUP_SHA256 = _hash(_WARMUP_POLICY)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, fields, what):
    _need(type(value) is dict and set(value) == set(fields), 'invalid ' + what + ' fields')


def _same(actual, expected, what):
    _need(_json(actual) == _json(expected), what + ' differs')


def _case_sha(kind):
    if kind == 'warmup':
        return WARMUP_SHA256
    if kind == 'clean_capture':
        path, token, operation = None, None, 'capture'
    else:
        _need(kind in _ACTIONS, 'unknown calibration kind')
        path, token, operation = _ACTIONS[kind]
    cross = operation == 'cross_patch'
    return _hash({'schema': 'agreement-cuda-calibration-case-v1', 'kind': kind,
                  'target_input_ids': TARGET, 'donor_input_ids': DONOR if cross else None,
                  'executed_input_ids': DONOR if kind == 'clean_capture' else TARGET,
                  'subject_index': 1, 'module_path': path, 'operation': operation,
                  'batch_index': 0 if path else None, 'token_index': token, 'invocation': 0,
                  'donor_module_path': path if cross else None,
                  'donor_token_index': token if cross else None, 'attention_mask': [1] * 5,
                  'output_selector': ([] if path and path.endswith('.mlp') else [0] if path else None),
                  'capture_policy_sha256': POLICY_SHA256})


def _event(name, action, path, token, selector, order, preceding, *, whole=False, source=None):
    return {'name': name, 'operation': action, 'order': order, 'module_path': path,
            'module_aliases': [path], 'call_site_known': False, 'forward_index': 0,
            'invocation': 0, 'batch_index': None if whole else 0,
            'token_index': None if whole else token, 'output_selector': selector,
            'output_shape': [1, 5, 768], 'selected_shape': [1, 5, 768] if whole else [768],
            'dtype': 'torch.float32', 'device': 'cuda:0', 'source_capture': source,
            'preceding_same_site_actions': list(preceding)}


def _metadata(kind):
    if kind in ('warmup', 'clean_capture'):
        steps = [_event(f'capture_{index}', 'capture', path, token, selector, index,
                        ('capture_8',) if index == 9 else ())
                 for index, (path, token, selector) in enumerate(SITES)]
        return {'steps': steps, 'capture_bytes': 30720, 'source_bytes': 0}
    path, token, action = _ACTIONS[kind]
    selector = [] if path.endswith('.mlp') else [0]
    specs = [('whole_before', 'capture', True, None), ('before', 'capture', False, None),
             ('action', 'null' if action == 'null' else 'patch', False,
              'before' if action == 'self_patch' else None),
             ('after', 'capture', False, None), ('whole_after', 'capture', True, None)]
    steps = [_event(name, op, path, token, selector, index,
                    [item[0] for item in specs[:index]], whole=whole, source=source)
             for index, (name, op, whole, source) in enumerate(specs)]
    return {'steps': steps, 'capture_bytes': 36864,
            'source_bytes': 3072 if action == 'cross_patch' else 0}


def _snapshot(value, total, cap):
    _fields(value, ('device_free_bytes', 'device_total_bytes', 'process_allocated_bytes',
                    'process_reserved_bytes'), 'memory snapshot')
    _need(all(type(item) is int and item >= 0 for item in value.values()), 'invalid memory counter')
    free, size = value['device_free_bytes'], value['device_total_bytes']
    allocated, reserved = value['process_allocated_bytes'], value['process_reserved_bytes']
    _need(size == total and free <= size and allocated <= reserved <= min(size, cap),
          'inconsistent memory snapshot')
    return free, allocated, reserved


def _headroom(snapshot, budget, total, cap):
    _fields(budget, ('telemetry', 'reserve_bytes', 'committed_incremental_bytes', 'active_leases',
                     'gpu_trace_bytes', 'host_trace_bytes', 'max_gpu_trace_bytes',
                     'max_host_trace_bytes', 'admissible_estimate_bytes'), 'budget')
    numeric = {name: item for name, item in budget.items() if name != 'telemetry'}
    _need(all(type(item) is int and item >= 0 for item in numeric.values()), 'invalid budget counter')
    reserve = max(GIB, (15 * total + 99) // 100)
    committed = 7 * GIB // 2
    _need(budget['reserve_bytes'] == reserve and budget['committed_incremental_bytes'] == committed
          and budget['active_leases'] == 2 and budget['gpu_trace_bytes'] == TRACE
          and budget['host_trace_bytes'] == TRACE and budget['max_gpu_trace_bytes'] == TRACE
          and budget['max_host_trace_bytes'] == TRACE, 'incorrect calibration claims')
    for measurement in (snapshot, budget['telemetry']):
        free, _, reserved = _snapshot(measurement, total, cap)
        _need(free >= 3 * GIB and free >= reserve + committed
              and reserved + committed + reserve <= total, 'insufficient recorded headroom')
    telemetry_free, _, telemetry_reserved = _snapshot(budget['telemetry'], total, cap)
    admissible = max(0, min(total - telemetry_reserved - reserve - committed,
                            telemetry_free - reserve - committed))
    _need(budget['admissible_estimate_bytes'] == admissible, 'budget arithmetic differs')


def audit_stage_records(records, warm_key):
    """Check the 84 fixed stage records and independently reconstruct peaks.

    The caller must separately bind ``warm_key`` to the frozen source/runtime
    and reconcile these samples with both saved profiles and the reaped owner.
    """
    _need(type(records) is list and len(records) == 84 and type(warm_key) is dict,
          'the complete 84-stage record and workload key are required')
    samples, callbacks, increments = _validate_stage_prefix(records, warm_key)
    return {'sample_count': 14, 'stage_count': 84, 'samples': samples, 'callbacks': callbacks,
            'cold_observed_increment_bytes': increments[0],
            'warm_observed_increment_bytes': max(increments[1:])}


def audit_partial_stage_records(records, warm_key):
    """Validate an incomplete ordered stage prefix without inventing results.

    A missing or incomplete callback is unknown, not a passed action. This
    method derives no profile or device-safety conclusion from partial data.
    Source identity and the reaped owner must be reviewed separately.
    """
    _need(type(records) is list and len(records) <= 85 and type(warm_key) is dict,
          'an incomplete stage prefix and workload key are required')
    failed = bool(records and type(records[-1]) is dict and records[-1].get('phase') == 'failed')
    prefix = records[:-1] if failed else records
    count = len(prefix)
    _need(count <= 84 and (count < 84 or failed), 'a full nonfailed sequence is not an incomplete prefix')
    samples, _, _ = _validate_stage_prefix(prefix, warm_key)
    kinds = ('warmup', *KINDS)
    failure = None
    if failed:
        last = records[-1]
        _fields(last, ('phase', 'kind', 'error_type'), 'failed stage')
        if count == 0:
            allowed = ('preflight', kinds[0])
        elif count % 6:
            allowed = (kinds[count // 6],)
        elif count == 84:
            allowed = (kinds[-1],)
        else:
            allowed = (kinds[count // 6 - 1], kinds[count // 6])
        _need(type(last['kind']) is str and last['kind'] in allowed and type(last['error_type']) is str
              and 0 < len(last['error_type']) <= 128, 'invalid failed stage')
        failure = {'kind': last['kind'], 'error_type': last['error_type']}
    return {'status': 'incomplete_prefix_consistent_not_approved', 'stage_count': len(records),
            'completed_samples': len(samples), 'failure': failure,
            'next_kind': kinds[count // 6] if count < 84 else None,
            'next_phase': PHASES[count % 6] if count < 84 else None}


def _validate_stage_prefix(records, warm_key):
    fixed = {'warmup_state': 'warm', 'warmup_protocol_sha256': WARMUP_SHA256,
             'capture_policy_sha256': POLICY_SHA256, 'operation': 'forward',
             'capture_mode': 'direct', 'batch_size': 1, 'sequence_length': 5,
             'input_dtypes': [['attention_mask', 'torch.int64'], ['input_ids', 'torch.int64']],
             'output_limit_bytes': 5 * 50257 * 4, 'max_gpu_trace_bytes': TRACE,
             'max_host_trace_bytes': TRACE}
    for field, expected in fixed.items():
        _same(warm_key.get(field), expected, 'workload ' + field)
    total = warm_key.get('device_total_bytes')
    _need(type(total) is int and 7 * GIB <= total <= 9 * GIB, 'unexpected device total')
    cap = total // 4
    cold_key = deepcopy(warm_key)
    cold_key['warmup_state'] = 'cold'
    fingerprints = [_hash({'schema': 'workload-key-v1', 'key': key}) for key in (cold_key, warm_key)]
    samples, callbacks, increments = [], [], []
    for index, kind in enumerate(('warmup', *KINDS)):
        stages = records[index * 6:(index + 1) * 6]
        if not stages:
            break
        for stage, phase in zip(stages, PHASES):
            _need(type(stage) is dict and stage.get('phase') == phase and stage.get('kind') == kind,
                  'stage order or kind differs')
        start = stages[0]
        _fields(start, ('phase', 'kind', 'seed', 'case_sha256'), 'started stage')
        _same(start, {'phase': 'started', 'kind': kind, 'seed': SEED, 'case_sha256': _case_sha(kind)},
              'started stage')
        if len(stages) == 1:
            break
        before = stages[1]
        _fields(before, ('phase', 'kind', 'snapshot'), 'baseline')
        _snapshot(before['snapshot'], total, cap)
        if len(stages) == 2:
            break
        check = stages[2]
        _fields(check, ('phase', 'kind', 'budget'), 'baseline_check')
        _headroom(before['snapshot'], check['budget'], total, cap)
        if len(stages) == 3:
            break
        after = stages[3]
        _fields(after, ('phase', 'kind', 'snapshot'), 'after_observed')
        _snapshot(after['snapshot'], total, cap)
        if len(stages) == 4:
            break
        post = stages[4]
        _fields(post, ('phase', 'kind', 'budget'), 'postcheck')
        _headroom(after['snapshot'], post['budget'], total, cap)
        if len(stages) == 5:
            break
        completed = stages[5]
        _fields(completed, ('phase', 'kind', 'sample', 'callback'), 'completed stage')
        sample, callback = completed['sample'], completed['callback']
        _fields(sample, ('workload_sha256', 'sample_id', 'case_sha256', 'seed', 'status',
                         'before', 'after', 'peak_allocated_bytes', 'peak_reserved_bytes'), 'sample')
        _same({name: sample[name] for name in ('workload_sha256', 'sample_id', 'case_sha256', 'seed', 'status')},
              {'workload_sha256': fingerprints[0 if index == 0 else 1], 'sample_id': kind,
               'case_sha256': _case_sha(kind), 'seed': SEED, 'status': 'completed'}, 'sample identity')
        _same(sample['before'], before['snapshot'], 'sample baseline')
        _same(sample['after'], after['snapshot'], 'sample endpoint')
        free0, alloc0, reserved0 = _snapshot(sample['before'], total, cap)
        free1, alloc1, reserved1 = _snapshot(sample['after'], total, cap)
        peak_a, peak_r = sample['peak_allocated_bytes'], sample['peak_reserved_bytes']
        _need(type(peak_a) is int and type(peak_r) is int
              and peak_a >= max(alloc0, alloc1) and peak_r >= max(peak_a, reserved0, reserved1)
              and peak_r <= cap, 'invalid per-sample peaks')
        _need(type(callback) is dict, 'missing callback record')
        if index == 0:
            _fields(callback, ('sample_id', 'policy_sha256', 'seed', 'model_state_sha256', 'plan_metadata'),
                    'warmup callback')
            digest = callback['model_state_sha256']
            _need(type(digest) is str and len(digest) == 64
                  and all(char in '0123456789abcdef' for char in digest), 'invalid model-state digest')
            expected_callback = {'sample_id': kind, 'policy_sha256': WARMUP_SHA256, 'seed': SEED,
                                 'model_state_sha256': digest, 'plan_metadata': _metadata(kind)}
        else:
            expected_callback = {'sample_id': kind, 'case_sha256': _case_sha(kind), 'seed': SEED,
                                 'plan_metadata': _metadata(kind), 'capture_or_landing_checked': True}
        _same(callback, expected_callback, 'callback metadata')
        samples.append(sample)
        callbacks.append(callback)
        increments.append(max(0, peak_a - alloc0, peak_r - reserved0, free0 - free1))
    return samples, callbacks, increments


def _parse_canonical(raw, name, *, ensure_ascii=False):
    _need(type(raw) is bytes and 0 < len(raw) <= MIB, name + ' must be bounded bytes')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _need(key not in value, 'duplicate JSON field')
            value[key] = item
        return value
    def reject(_):
        raise ValueError('nonfinite JSON number')
    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('invalid ' + name) from error
    canonical = json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True, separators=(',', ':'),
                           allow_nan=False).encode('utf-8')
    _need(type(value) is dict and canonical == raw, name + ' is not canonical JSON')
    return value


def audit_profiles(stage_report, warm_key, profile_documents, evidence_documents, source_bytes,
                   expected_profile_ids, expected_estimates):
    """Reconcile both candidate profiles with independently checked stages.

    SHA-256 identities check retained bytes, not which bytes an owner executed.
    The returned estimates remain measured observations, never approvals or
    worst-case device bounds.
    """
    _need(type(stage_report) is dict and stage_report.get('stage_count') == 84
          and stage_report.get('sample_count') == 14 and type(stage_report.get('samples')) is list
          and len(stage_report['samples']) == 14, 'a complete stage audit is required')
    _need(type(source_bytes) is bytes and 0 < len(source_bytes) <= MIB, 'missing source manifest bytes')
    for collection in (profile_documents, evidence_documents, expected_profile_ids, expected_estimates):
        _need(type(collection) is list and len(collection) == 2, 'two ordered profile entries are required')
    _need(all(type(item) is str and len(item) == 64 and all(c in '0123456789abcdef' for c in item)
              for item in expected_profile_ids) and len(set(expected_profile_ids)) == 2,
          'invalid distinct candidate profile identities')
    _need(all(type(item) is int and 0 < item <= 3 * GIB // 2 for item in expected_estimates),
          'candidate estimate exceeds the provisional job commitment')
    cold_key = deepcopy(warm_key)
    cold_key['warmup_state'] = 'cold'
    samples = stage_report['samples']
    estimates, identifiers = [], []
    for index, (key, selected) in enumerate(((cold_key, samples[:1]), (warm_key, samples[1:]))):
        evidence = _parse_canonical(evidence_documents[index], 'profile evidence')
        _same(evidence, {'schema': 'resource-profile-evidence-v1', 'key': key, 'samples': selected},
              'profile evidence and stages')
        profile = _parse_canonical(profile_documents[index], 'profile')
        _fields(profile, ('schema', 'key', 'samples', 'provenance', 'margin_bytes', 'margin_bps'), 'profile')
        _need(profile['schema'] == 'resource-profile-v1', 'unexpected profile schema')
        _same(profile['key'], key, 'profile key')
        _same(profile['samples'], selected, 'profile samples')
        _same(profile['margin_bytes'], 512 * MIB, 'profile fixed margin')
        _same(profile['margin_bps'], 2000, 'profile percentage margin')
        provenance = profile['provenance']
        _fields(provenance, ('protocol', 'source_manifest_sha256', 'evidence_sha256',
                             'collected_utc', 'isolation', 'boundaries'), 'profile provenance')
        _need(provenance['protocol'] == 'resource-profile-v1'
              and provenance['source_manifest_sha256'] == hashlib.sha256(source_bytes).hexdigest()
              and provenance['evidence_sha256'] == hashlib.sha256(evidence_documents[index]).hexdigest()
              and provenance['isolation'] == 'cooperative-exclusive'
              and provenance['boundaries'] == 'synchronize-reset-synchronize',
              'profile provenance differs from saved evidence')
        try:
            timestamp = datetime.fromisoformat(provenance['collected_utc'].replace('Z', '+00:00'))
            valid_utc = timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp)
        except (ValueError, TypeError, AttributeError):
            valid_utc = False
        _need(valid_utc, 'profile collection time must identify UTC')
        identifier = hashlib.sha256(profile_documents[index]).hexdigest()
        _need(identifier == expected_profile_ids[index], 'candidate profile identity differs')
        # Fixed input tensors are two int64 matrices of [1,5]. The output cap
        # and device-trace allowance are additive logical declarations.
        declared = 80 + 5 * 50257 * 4 + TRACE
        observed = max(max(0,
                           item['peak_allocated_bytes'] - item['before']['process_allocated_bytes'],
                           item['peak_reserved_bytes'] - item['before']['process_reserved_bytes'],
                           item['before']['device_free_bytes'] - item['after']['device_free_bytes'])
                       for item in selected)
        basis = declared + observed
        estimate = basis + 512 * MIB + (basis * 2000 + 9999) // 10000
        _need(estimate == expected_estimates[index] and estimate <= 3 * GIB // 2,
              'profile estimate does not match independent arithmetic')
        estimates.append(estimate)
        identifiers.append(identifier)
    return {'profile_ids': identifiers, 'estimates_bytes': estimates}


def _read_file(path, limit=MIB, *, allow_empty=False):
    path = Path(path)
    try:
        before = path.lstat()
        _need(stat.S_ISREG(before.st_mode) and not path.is_symlink()
              and not getattr(before, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
              and (0 <= before.st_size if allow_empty else 0 < before.st_size) and before.st_size <= limit,
              'evidence is not a bounded regular file')
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        current = path.lstat()
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        _need(identity(before) == identity(opened) == identity(after) == identity(current)
              and len(raw) == before.st_size, 'evidence changed during read')
        return raw
    except OSError as error:
        raise ValueError('evidence file is unavailable') from error


def _line_json(raw, name):
    _need(type(raw) is bytes and raw.endswith(b'\n') and raw.count(b'\n') == 1,
          name + ' must be one complete JSON line')
    # The controller and journal writers use JSON's default ASCII escapes;
    # source/runtime/profile documents use unescaped UTF-8 instead.
    return _parse_canonical(raw[:-1], name, ensure_ascii=True)


def audit_journal(directory, stage_report, run_id, source_sha256, profile_ids):
    """Bind each closed journal callback to the corresponding checked stage."""
    directory = Path(directory)
    try:
        info = directory.lstat()
        _need(stat.S_ISDIR(info.st_mode) and not directory.is_symlink()
              and not getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400),
              'journal is not an ordinary directory')
        _need({item.name for item in directory.iterdir()} == {'started.json', 'events.jsonl', 'terminal.json', 'tensors'},
              'journal has missing or unexpected artifacts')
        tensors = directory / 'tensors'
        tensor_info = tensors.lstat()
        _need(stat.S_ISDIR(tensor_info.st_mode) and not tensors.is_symlink()
              and not getattr(tensor_info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
              and not any(tensors.iterdir()), 'calibration journal contains retained tensors')
    except OSError as error:
        raise ValueError('journal directory is unavailable') from error
    _need(type(stage_report) is dict and stage_report.get('stage_count') == 84
          and type(stage_report.get('callbacks')) is list and len(stage_report['callbacks']) == 14,
          'complete checked callback records are required')
    started = _line_json(_read_file(directory / 'started.json'), 'journal start')
    _same(started, {'schema': 'agreement-calibration-journal-v1', 'run_id': run_id,
                    'source_sha256': source_sha256}, 'journal start')
    events_raw = _read_file(directory / 'events.jsonl')
    _need(events_raw.endswith(b'\n') and events_raw.count(b'\n') == 14, 'journal event count differs')
    for index, (line, callback) in enumerate(zip(events_raw.splitlines(), stage_report['callbacks'])):
        event = _parse_canonical(line, 'journal event')
        kind = 'gpu_warmup' if index == 0 else 'gpu_calibration_action'
        _same(event, {'index': index, 'event': {'kind': kind, **callback}, 'tensors': {}},
              'journal callback')
    terminal = _line_json(_read_file(directory / 'terminal.json'), 'journal terminal')
    _same(terminal, {'status': 'complete_candidate', 'detail': {'samples': 14, 'profile_ids': profile_ids},
                     'events': 14, 'retained_tensor_bytes': 0, 'event_log_bytes': len(events_raw),
                     'storage_accounting_complete': True}, 'journal terminal')
    return {'event_count': 14, 'retained_tensor_bytes': 0,
            'event_sha256': hashlib.sha256(events_raw).hexdigest()}


_ASSETS = {
    'config.json': (762, '4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a'),
    'generation_config.json': (124, 'b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805'),
    'tokenizer.json': (1355256, '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'),
    'tokenizer_config.json': (26, '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f'),
    'vocab.json': (1042301, '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'),
    'merges.txt': (456318, '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),
    'model.safetensors': (352824413, 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'),
}
_MODEL = 'distilbert/distilgpt2'
_REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
_PROTOCOLS = ('docs/AGREEMENT_GPU_CALIBRATION_V1.md', 'docs/AGREEMENT_CUDA_ADAPTER.md',
              'docs/AGREEMENT_CALIBRATION_SAMPLING_DRAFT.md')
_EXTRA_SOURCE = set(_PROTOCOLS) | {'docs/TRUSTED_SERVICE_PROFILE_V1.md',
    'scripts/agreement_calibration_bootstrap.py', 'scripts/v2_source_guards.py'}
_VERSIONS = ('python_version', 'torch_version', 'transformers_version', 'numpy_version',
             'huggingface_hub_version', 'safetensors_version', 'tokenizers_version', 'cuda_runtime_version')
_MODEL_FIELDS = ('checkpoint_sha256', 'checkpoint_revision', 'model_config_sha256', 'source_sha256',
    'runtime_settings_sha256', 'torch_version', 'transformers_version', 'cuda_runtime_version',
    'driver_version', 'python_version', 'device', 'device_uuid', 'device_name', 'compute_capability',
    'device_total_bytes', 'model_dtype', 'compute_dtype', 'attention_backend', 'model_mode', 'grad_mode',
    'autocast', 'use_cache', 'compiled')


def _sha_string(value, size=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{' + str(size) + '}', value) is not None


def _source_items(items, allowed):
    _need(type(items) is list and len(items) <= 4096, 'source item inventory is invalid')
    result = {}
    for item in items:
        _fields(item, ('path', 'size', 'sha256'), 'source item')
        path = item['path']
        _need(type(path) is str and path in allowed and path not in result
              and type(item['size']) is int and 0 <= item['size'] <= 16 * MIB
              and _sha_string(item['sha256']), 'source item is invalid')
        result[path] = item['sha256']
    return result


def audit_identity(records, *, expected_commit, expected_device_uuid, expected_freeze_sha256):
    """Reconstruct the warm workload from prelaunch and owner identity records.

    The supplied freeze hash, published commit and physical UUID must be
    independently selected by the operator. This function checks recorded
    bindings; a separate Git-blob check and trusted execution are still needed.
    """
    _fields(records, ('run_manifest', 'controller_source', 'source_base', 'source', 'runtime', 'assets'),
            'identity artifacts')
    _need(_sha_string(expected_commit, 40) and _sha_string(expected_freeze_sha256)
          and type(expected_device_uuid) is str
          and re.fullmatch(r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                           expected_device_uuid) is not None, 'invalid independent identity anchors')
    _need(type(records['run_manifest']) is bytes
          and hashlib.sha256(records['run_manifest']).hexdigest() == expected_freeze_sha256,
          'prelaunch freeze digest differs')
    frozen = _line_json(records['run_manifest'], 'run manifest')
    controller = _line_json(records['controller_source'], 'controller source')
    base = _parse_canonical(records['source_base'], 'source base')
    source = _parse_canonical(records['source'], 'source manifest')
    runtime = _parse_canonical(records['runtime'], 'runtime manifest')
    assets = _parse_canonical(records['assets'], 'asset manifest')
    _fields(frozen, ('schema', 'run_id', 'source', 'checkpoint', 'runtime', 'device', 'workload', 'directory'),
            'run manifest')
    _need(frozen['schema'] == 'agreement-calibration-run-freeze-v1' and _sha_string(frozen['run_id'], 32),
          'invalid run identity')
    src = frozen['source']
    _fields(src, ('commit', 'sha256', 'checkout', 'source_base_sha256'), 'frozen source')
    _need(src['commit'] == expected_commit and type(src['checkout']) is str and Path(src['checkout']).is_absolute()
          and _sha_string(src['sha256']) and _sha_string(src['source_base_sha256'])
          and src['source_base_sha256'] == hashlib.sha256(records['source_base']).hexdigest(),
          'frozen source differs from the independent anchors')
    _fields(controller, ('schema', 'commit', 'files'), 'controller source')
    controller_canonical = json.dumps(controller, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    _need(controller['schema'] == 'agreement-controller-source-v1' and controller['commit'] == expected_commit
          and hashlib.sha256(controller_canonical).hexdigest() == src['sha256']
          and type(controller['files']) is dict and 1 <= len(controller['files']) <= 4096,
          'controller source inventory differs from the freeze')
    files = controller['files']
    for path, sha in files.items():
        _need(type(path) is str and '\\' not in path and not path.startswith('/')
              and all(part not in ('', '.', '..') for part in path.split('/'))
              and ((path.startswith('looking_glass/') and path.lower().endswith('.py')) or path in _EXTRA_SOURCE)
              and _sha_string(sha), 'invalid controller source path or digest')
    _need(_EXTRA_SOURCE <= files.keys() and 'looking_glass/__init__.py' in files,
          'required frozen source entries are missing')
    expected_assets = {'schema': 'resource-bootstrap-assets-v1', 'model_id': _MODEL, 'revision': _REVISION,
                       'assets': [{'name': name, 'size': size, 'sha256': sha}
                                  for name, (size, sha) in sorted(_ASSETS.items())]}
    _same(assets, expected_assets, 'pinned asset manifest')
    checkpoint = frozen['checkpoint']
    _fields(checkpoint, ('model_id', 'revision', 'cache_root', 'assets'), 'frozen checkpoint')
    _need(checkpoint['model_id'] == _MODEL and checkpoint['revision'] == _REVISION
          and type(checkpoint['cache_root']) is str and Path(checkpoint['cache_root']).is_absolute(),
          'frozen checkpoint identity differs')
    _same(checkpoint['assets'], expected_assets['assets'], 'frozen asset pins')
    _fields(base, ('schema', 'repository_files', 'python_executable', 'assets'), 'source base')
    _need(base['schema'] == 'trusted-service-source-base-v1', 'invalid source base schema')
    _same(base['assets'], assets, 'source-base assets')
    py = base['python_executable']
    _fields(py, ('size', 'sha256'), 'Python executable')
    _need(type(py['size']) is int and py['size'] > 0 and _sha_string(py['sha256']), 'invalid Python identity')
    package = {path: sha for path, sha in files.items()
               if path.startswith('looking_glass/') or path == 'docs/TRUSTED_SERVICE_PROFILE_V1.md'}
    _same(_source_items(base['repository_files'], set(package)), package, 'source-base file inventory')
    _fields(source, ('schema', 'base', 'dependency_origins', 'protocols'), 'source manifest')
    _need(source['schema'] == 'agreement-calibration-source-v1', 'invalid source manifest schema')
    _same(source['base'], base, 'owner and prelaunch source base')
    _same(_source_items(source['protocols'], set(_PROTOCOLS)), {p: files[p] for p in _PROTOCOLS},
          'protocol source inventory')
    versions = frozen['runtime']
    _fields(versions, _VERSIONS, 'frozen runtime')
    _need(all(type(value) is str and value.strip() == value and value for value in versions.values())
          and versions['torch_version'] == '2.8.0+cu128' and versions['cuda_runtime_version'] == '12.8',
          'invalid frozen dependency versions')
    origins = source['dependency_origins']
    names = {'torch': 'torch_version', 'transformers': 'transformers_version', 'numpy': 'numpy_version',
             'huggingface_hub': 'huggingface_hub_version', 'safetensors': 'safetensors_version',
             'tokenizers': 'tokenizers_version'}
    _need(type(origins) is list and len(origins) == len(names), 'dependency origin inventory differs')
    found = set()
    for item in origins:
        _fields(item, ('name', 'version', 'path', 'size', 'sha256'), 'dependency origin')
        name = item['name']
        _need(type(name) is str and name in names and name not in found and item['version'] == versions[names[name]]
              and type(item['path']) is str and Path(item['path']).is_absolute()
              and type(item['size']) is int and item['size'] > 0 and _sha_string(item['sha256']),
              'invalid dependency origin')
        found.add(name)
    device = frozen['device']
    _fields(device, ('uuid', 'name', 'total_bytes', 'driver_version'), 'frozen device')
    total = runtime.get('device_total_bytes')
    _need(device['uuid'] == expected_device_uuid and type(device['name']) is str and device['name']
          and type(device['driver_version']) is str and device['driver_version']
          and type(device['total_bytes']) is int and 7 * GIB <= device['total_bytes'] <= 9 * GIB
          and type(total) is int and 7 * GIB <= total <= 9 * GIB
          and abs(total - device['total_bytes']) <= MIB, 'frozen physical device differs')
    workload = {'seed': SEED, 'target_input_ids': TARGET, 'donor_input_ids': DONOR,
                'subject_index': 1, 'last_token_index': 4, 'batch_size': 1, 'sequence_length': 5}
    _same(frozen['workload'], workload, 'frozen public workload')
    directory = frozen['directory']
    _fields(directory, ('path', 'device', 'inode'), 'private run directory')
    _need(type(directory['path']) is str and Path(directory['path']).is_absolute()
          and all(type(directory[name]) is int and directory[name] >= 0 for name in ('device', 'inode')),
          'invalid private run directory identity')
    required_runtime = {
        'schema': 'agreement-calibration-runtime-v1',
        'execution_path': 'looking_glass.agreement_cuda_calibration.CudaAgreementCalibrator',
        'device': 'cuda:0', 'device_uuid': expected_device_uuid, 'device_name': device['name'],
        'driver_version': device['driver_version'], 'driver_version_provenance': 'controller-prelaunch-nvidia-smi',
        'subprocess_policy': 'python-audit-hook-denied', 'model_dtype': 'torch.float32',
        'compute_dtype': 'torch.float32', 'attention_backend': 'eager', 'model_mode': 'eval',
        'grad_mode': 'inference', 'autocast': 'disabled', 'use_cache': False, 'compiled': False,
        'allocator_backend': 'native', 'allocator_fraction': '0.25', 'allocator_limit_bytes': total // 4,
        'torch_threads': 1, 'torch_interop_threads': 1, 'deterministic_algorithms': True,
        'matmul_tf32': False, 'cudnn_tf32': False, 'cudnn_enabled': True, 'cudnn_benchmark': False,
        'cudnn_deterministic': True, 'float32_matmul_precision': 'highest', 'deterministic_warn_only': False,
        'calling_thread_denormal_product_bits': 0x00400000, 'cpu_autocast': False, **versions,
    }
    _fields(runtime, set(required_runtime) | {'device_total_bytes', 'compute_capability', 'torch_build_sha256',
        'python_import_paths', 'environment', 'effective_huggingface_settings'}, 'runtime manifest')
    for name, expected in required_runtime.items():
        _same(runtime.get(name), expected, 'runtime ' + name)
    capability = runtime.get('compute_capability')
    _need(type(capability) is list and len(capability) == 2
          and all(type(value) is int and value >= 0 for value in capability)
          and _sha_string(runtime.get('torch_build_sha256')), 'invalid runtime build identity')
    paths = runtime.get('python_import_paths')
    _need(type(paths) is list and paths and all(type(path) is str and Path(path).is_absolute() for path in paths),
          'invalid recorded import paths')
    environment = runtime.get('environment')
    hf_home = str(Path(checkpoint['cache_root']).parent)
    expected_environment = {'CUDA_VISIBLE_DEVICES': expected_device_uuid, 'HF_HOME': hf_home,
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTORCH_CUDA_ALLOC_CONF': 'backend:native',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'PYTORCH_ALLOC_CONF': None, 'PYTORCH_NO_CUDA_MEMORY_CACHING': None, 'CUDA_LAUNCH_BLOCKING': None}
    _same(environment, expected_environment, 'runtime environment')
    effective = {'hf_home': hf_home, 'hf_hub_cache': str(Path(hf_home) / 'hub'),
                 'transformers_cache': str(Path(hf_home) / 'hub'),
                 'hf_modules_cache': str(Path(hf_home) / 'modules'),
                 'hub_offline': True, 'transformers_offline': True, 'telemetry_disabled': True}
    _same(runtime.get('effective_huggingface_settings'), effective, 'effective offline settings')
    identity = {'checkpoint_sha256': _ASSETS['model.safetensors'][1], 'checkpoint_revision': _REVISION,
                'model_config_sha256': _ASSETS['config.json'][1],
                'source_sha256': hashlib.sha256(records['source']).hexdigest(),
                'runtime_settings_sha256': _hash(runtime)}
    for name in _MODEL_FIELDS:
        if name not in identity:
            identity[name] = runtime[name]
    model_id = _hash({'schema': 'model-runtime-identity-v1', 'identity': identity})
    warm_key = {'model_id': model_id, **identity, 'warmup_state': 'warm',
        'warmup_protocol_sha256': WARMUP_SHA256, 'operation': 'forward', 'capture_mode': 'direct',
        'capture_policy_sha256': POLICY_SHA256, 'batch_size': 1, 'sequence_length': 5,
        'input_dtypes': [['attention_mask', 'torch.int64'], ['input_ids', 'torch.int64']],
        'output_limit_bytes': 5 * 50257 * 4, 'max_gpu_trace_bytes': TRACE, 'max_host_trace_bytes': TRACE}
    return {'run_id': frozen['run_id'], 'freeze': frozen, 'controller_source': controller,
            'source_sha256': identity['source_sha256'],
            'runtime_sha256': hashlib.sha256(records['runtime']).hexdigest(),
            'asset_sha256': hashlib.sha256(records['assets']).hexdigest(), 'warm_key': warm_key}


def verify_dependency_origins(source_bytes, selected_site):
    """Rehash the six recorded entry files inside an independently selected site.

    These entry-file hashes do not cover the packages' other Python or native
    binaries and do not establish which bytes were loaded during execution.
    """
    source = _parse_canonical(source_bytes, 'source manifest')
    origins = source.get('dependency_origins')
    _need(type(origins) is list and len(origins) == 6, 'dependency origins are missing')
    try:
        site = Path(selected_site).resolve(strict=True)
        _need(site.is_dir(), 'selected dependency site is not a directory')
        names = set()
        for entry in origins:
            _fields(entry, ('name', 'version', 'path', 'size', 'sha256'), 'dependency origin')
            name, path = entry['name'], Path(entry['path'])
            _need(type(name) is str and name not in names and path.is_absolute()
                  and path.resolve(strict=True) == path and path.is_relative_to(site)
                  and type(entry['size']) is int and 0 < entry['size'] <= 16 * MIB
                  and _sha_string(entry['sha256']), 'dependency origin is outside the selected site')
            raw = _read_file(path, 16 * MIB)
            _need(len(raw) == entry['size'] and hashlib.sha256(raw).hexdigest() == entry['sha256'],
                  'dependency origin bytes changed')
            names.add(name)
        _need(names == {'torch', 'transformers', 'numpy', 'huggingface_hub', 'safetensors', 'tokenizers'},
              'dependency origin set differs')
        return len(names)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError('dependency origin could not be verified') from error


def verify_committed_sources(repo, git_executable, commit, controller_source, source_bytes):
    """Compare all selected Git blobs, checkout bytes and recorded file sizes.

    A clean byte comparison is not evidence that these were the bytes actually
    executed by the child. The caller must select the published commit and a
    trusted Git executable independently.
    """
    _need(_sha_string(commit, 40) and type(controller_source) is dict
          and controller_source.get('schema') == 'agreement-controller-source-v1'
          and controller_source.get('commit') == commit and type(controller_source.get('files')) is dict,
          'invalid committed-source identity')
    source = _parse_canonical(source_bytes, 'source manifest')
    base = source.get('base')
    _need(type(base) is dict and type(base.get('repository_files')) is list
          and type(source.get('protocols')) is list, 'source item inventory is missing')
    files = controller_source['files']
    _need(1 <= len(files) <= 4096 and _EXTRA_SOURCE <= files.keys(), 'source path set is incomplete')
    package = {name: digest for name, digest in files.items()
               if name.startswith('looking_glass/') or name == 'docs/TRUSTED_SERVICE_PROFILE_V1.md'}
    _same(_source_items(base['repository_files'], set(package)), package, 'source-base Git inventory')
    _same(_source_items(source['protocols'], set(_PROTOCOLS)), {name: files[name] for name in _PROTOCOLS},
          'protocol Git inventory')
    recorded_sizes = {item['path']: item['size'] for item in (*base['repository_files'], *source['protocols'])}
    try:
        root = Path(repo).resolve(strict=True)
        git = Path(git_executable).resolve(strict=True)
        _need(root.is_dir() and git.is_file(), 'Git or source checkout is unavailable')
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
        env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                    'GIT_CONFIG_SYSTEM': os.devnull, 'GIT_TERMINAL_PROMPT': '0',
                    'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_REPLACE_OBJECTS': '1'})
        def call(arguments, *, data=None):
            return subprocess.run([str(git), '--no-replace-objects', *arguments], cwd=root, env=env,
                                  input=data, capture_output=True, check=True, timeout=30).stdout
        tree = call(['ls-tree', '-r', '-z', '--full-tree', commit, '--', 'looking_glass', *_EXTRA_SOURCE])
        _need(len(tree) <= 4 * MIB, 'committed source inventory is oversized')
        objects = {}
        for entry in tree.split(b'\0'):
            if not entry:
                continue
            head, encoded = entry.split(b'\t', 1)
            mode, kind, oid = head.split(b' ', 2)
            name = encoded.decode('utf-8')
            selected = (name.startswith('looking_glass/') and name.lower().endswith('.py')) or name in _EXTRA_SOURCE
            if selected:
                _need(mode in (b'100644', b'100755') and kind == b'blob' and name not in objects
                      and re.fullmatch(b'[0-9a-f]{40}', oid) is not None, 'invalid committed source entry')
                objects[name] = oid
        _need(set(objects) == set(files), 'declared and committed source path sets differ')
        ordered = sorted(objects)
        query = b''.join(objects[name] + b'\n' for name in ordered)
        sizes = call(['cat-file', '--batch-check'], data=query).splitlines()
        _need(len(sizes) == len(ordered), 'Git object inventory is incomplete')
        lengths = []
        for name, line in zip(ordered, sizes):
            oid, kind, length = line.split(b' ')
            _need(oid == objects[name] and kind == b'blob' and length.isdigit(), 'invalid Git blob')
            size = int(length)
            _need(size <= 16 * MIB and (name not in recorded_sizes or recorded_sizes[name] == size),
                  'recorded source byte size differs')
            lengths.append(size)
        _need(sum(lengths) <= 32 * MIB, 'committed source data is oversized')
        payload = call(['cat-file', '--batch'], data=query)
        position = 0
        for name, size in zip(ordered, lengths):
            end = payload.find(b'\n', position)
            _need(end >= 0 and payload[position:end] == objects[name] + b' blob ' + str(size).encode(),
                  'Git blob header differs')
            start = end + 1
            raw = payload[start:start + size]
            _need(len(raw) == size and payload[start + size:start + size + 1] == b'\n'
                  and hashlib.sha256(raw).hexdigest() == files[name], 'Git blob digest differs')
            path = root / name
            _need(path.resolve(strict=True) == path and path.is_relative_to(root), 'checkout source path is aliased')
            _need(_read_file(path, 16 * MIB, allow_empty=True) == raw, 'checkout source bytes differ from Git')
            position = start + size + 1
        _need(position == len(payload), 'unexpected Git blob output')
        actual_python = set()
        def fail_walk(error):
            raise ValueError('source package could not be inventoried') from error
        for directory, dirs, names in os.walk(root / 'looking_glass', followlinks=False, onerror=fail_walk):
            dirs[:] = sorted(name for name in dirs if name != '__pycache__')
            for name in dirs:
                path = Path(directory) / name
                info = path.lstat()
                _need(stat.S_ISDIR(info.st_mode) and not path.is_symlink()
                      and not getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400),
                      'package source directory is aliased')
            for name in names:
                if name.lower().endswith('.py'):
                    actual_python.add((Path(directory) / name).relative_to(root).as_posix())
        _need(actual_python == {name for name in files if name.startswith('looking_glass/')},
              'checkout has missing or untracked package Python')
        return len(ordered)
    except (OSError, subprocess.SubprocessError, UnicodeError, IndexError) as error:
        raise ValueError('committed source verification failed') from error


def _utc(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    except (ValueError, TypeError, AttributeError):
        return False


def audit_reaped_records(identity, freeze_sha256, profiles, controller_bytes, records):
    """Link the controller's clean-reap observation to exact owner evidence.

    The producer's observations and post-exit NVIDIA query are self-reports;
    the query is not a system-wide process, UUID, or GPU-cleanup attestation.
    """
    _need(type(identity) is dict and _sha_string(freeze_sha256), 'validated identity is required')
    _fields(profiles, ('profile_ids', 'estimates_bytes'), 'checked profiles')
    _fields(records, ('started', 'owner', 'candidate', 'final', 'supervision',
                      'model_checks', 'preload_admission', 'cleanup'), 'reaped owner artifacts')
    run, frozen = identity['run_id'], identity['freeze']
    commit, source_digest = frozen['source']['commit'], frozen['source']['sha256']
    controller = _line_json(controller_bytes, 'controller result')
    expected_controller_fields = {'schema', 'run_id', 'source_commit', 'source_sha256', 'run_manifest_sha256',
        'private_run_dir', 'nvidia_smi_before', 'nvidia_smi_after', 'status', 'supervisor_error_type',
        'supervisor', 'owner_presence_after_exit', 'limitations', 'owner_evidence', 'source_protections_retained'}
    _fields(controller, expected_controller_fields, 'controller result')
    _need(controller['schema'] == 'agreement-calibration-controller-v1'
          and controller['status'] == 'awaiting_independent_audit' and controller['run_id'] == run
          and controller['source_commit'] == commit and controller['source_sha256'] == source_digest
          and controller['run_manifest_sha256'] == freeze_sha256
          and controller['private_run_dir'] == frozen['directory']['path']
          and controller['supervisor_error_type'] is None and controller['source_protections_retained'] is False,
          'controller did not retain a clean candidate for independent audit')
    _need(type(controller['limitations']) is list and controller['limitations']
          and all(type(item) is str and 0 < len(item) <= 256 for item in controller['limitations']),
          'controller limitations are missing')
    supervisor = controller['supervisor']
    _fields(supervisor, ('pid', 'run_id', 'exit_code', 'clean', 'timed_out'), 'supervisor')
    _need(type(supervisor['pid']) is int and supervisor['pid'] > 0
          and _sha_string(supervisor['run_id'], 32) and type(supervisor['exit_code']) is int
          and supervisor['exit_code'] == 0 and supervisor['clean'] is True and supervisor['timed_out'] is False,
          'supervisor did not report a clean, reaped owner')
    _same(controller['owner_presence_after_exit'], {'query': 'ok', 'owner_pid_listed': False},
          'post-exit owner presence')
    device = frozen['device']
    for name in ('nvidia_smi_before', 'nvidia_smi_after'):
        item = controller[name]
        _fields(item, ('uuid', 'device_name', 'total_bytes', 'free_bytes', 'used_bytes', 'driver_version', 'timestamp_utc'),
                name)
        _need(item['uuid'] == device['uuid'] and item['device_name'] == device['name']
              and item['total_bytes'] == device['total_bytes']
              and type(item['total_bytes']) is int and item['driver_version'] == device['driver_version']
              and all(type(item[field]) is int and 0 <= item[field] <= item['total_bytes']
                      for field in ('free_bytes', 'used_bytes'))
              and item['free_bytes'] + item['used_bytes'] <= item['total_bytes'] + 2 * MIB
              and _utc(item['timestamp_utc']),
              'recorded physical-device observations differ')
    before_time = datetime.fromisoformat(controller['nvidia_smi_before']['timestamp_utc'].replace('Z', '+00:00'))
    after_time = datetime.fromisoformat(controller['nvidia_smi_after']['timestamp_utc'].replace('Z', '+00:00'))
    _need(before_time <= after_time, 'post-exit timestamp precedes prelaunch observation')
    parsed = {name: _line_json(raw, name) for name, raw in records.items()}
    _same(parsed['started'], {'schema': 'agreement-calibration-start-v1', 'run_id': run,
          'source_commit': commit, 'source_sha256': source_digest, 'run_manifest_sha256': freeze_sha256,
          'started_utc': controller['nvidia_smi_before']['timestamp_utc']}, 'controller start')
    _same(parsed['supervision'], {'schema': 'agreement-calibration-supervision-v1',
          'calibration_run_id': run, 'supervisor': supervisor}, 'supervision sidecar')
    _same(parsed['owner'], {'schema': 'agreement-calibration-owner-v1', 'run_id': run,
          'supervisor_run_id': supervisor['run_id'], 'pid': supervisor['pid'],
          'source_commit': commit, 'source_sha256': source_digest}, 'owner start')
    candidate = parsed['candidate']
    _fields(candidate, ('schema', 'run_id', 'supervisor_run_id', 'pid', 'value'), 'owner candidate')
    _need(candidate['schema'] == 'agreement-calibration-owner-candidate-v1' and candidate['run_id'] == run
          and candidate['supervisor_run_id'] == supervisor['run_id'] and type(candidate['pid']) is int
          and candidate['pid'] == supervisor['pid'], 'owner candidate identifies another process')
    value = candidate['value']
    _fields(value, ('measurement', 'cleanup'), 'owner candidate value')
    expected_cleanup = {'runtime_uuid_verified': True, 'active_memory_leases': 0,
        'device_synchronized': True, 'model_weakref_dead': True,
        'allocator_allocated_bytes': 0, 'allocator_reserved_bytes': 0,
        'pytorch_cublas_cache_clear_completed': True, 'trace_gpu_bytes': 0, 'trace_host_bytes': 0}
    _same(value['cleanup'], expected_cleanup, 'owner cleanup')
    _same(parsed['final'], {'schema': 'agreement-calibration-owner-final-v1', 'run_id': run,
          'supervisor_run_id': supervisor['run_id'], 'pid': supervisor['pid'],
          'candidate_sha256': hashlib.sha256(records['candidate']).hexdigest(),
          'cleanup': expected_cleanup}, 'owner final')
    _same(parsed['model_checks'], {'cpu_checkpoint_verified': True, 'same_model_after_copy': True,
          'device_tie_verified': True, 'post_copy_synchronized': True}, 'model checks')
    measurement = value['measurement']
    _fields(measurement, ('schema', 'run_id', 'profile_ids', 'estimates_bytes', 'source_sha256',
        'runtime_sha256', 'asset_sha256', 'run_manifest_sha256', 'sample_count', 'stage_count',
        'workspace_cleanup', 'limitations'), 'measurement')
    _need(measurement['schema'] == 'agreement-calibration-candidate-v1' and measurement['run_id'] == run
          and measurement['source_sha256'] == identity['source_sha256']
          and measurement['runtime_sha256'] == identity['runtime_sha256']
          and measurement['asset_sha256'] == identity['asset_sha256']
          and measurement['run_manifest_sha256'] == freeze_sha256
          and type(measurement['sample_count']) is int and measurement['sample_count'] == 14
          and type(measurement['stage_count']) is int and measurement['stage_count'] == 84,
          'measurement identity or sample counts differ')
    _same(measurement['profile_ids'], profiles['profile_ids'], 'candidate profile IDs')
    _same(measurement['estimates_bytes'], profiles['estimates_bytes'], 'candidate profile estimates')
    _need(type(measurement['limitations']) is list and measurement['limitations']
          and all(type(item) is str and 0 < len(item) <= 256 for item in measurement['limitations']),
          'measurement limitations are missing')
    _same(controller['owner_evidence'], measurement, 'controller and owner measurement')
    workspace = parsed['cleanup']
    _fields(workspace, ('before', 'after', 'api'), 'workspace cleanup')
    _need(workspace['api'] == 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces',
          'cleanup implementation differs')
    for name in ('before', 'after'):
        counters = workspace[name]
        _fields(counters, ('allocated_bytes', 'reserved_bytes'), 'cleanup counters')
        _need(all(type(value) is int and 0 <= value <= identity['warm_key']['device_total_bytes']
                  for value in counters.values())
              and counters['allocated_bytes'] <= counters['reserved_bytes'], 'invalid cleanup counters')
    _same(workspace['after'], {'allocated_bytes': 0, 'reserved_bytes': 0}, 'post-cleanup allocator counters')
    _same(measurement['workspace_cleanup'], workspace, 'candidate and cleanup record')
    total = identity['warm_key']['device_total_bytes']
    preload = parsed['preload_admission']
    _need(type(preload) is dict and 'telemetry' in preload, 'preload admission is missing')
    _headroom(preload['telemetry'], preload, total, total // 4)
    return {'status': 'saved_records_consistent_not_approved', 'run_id': run, 'sample_count': 14,
            'owner_pid': supervisor['pid'], 'supervisor_run_id': supervisor['run_id'],
            'profile_ids': profiles['profile_ids'], 'estimates_bytes': profiles['estimates_bytes']}


def audit_directory(*, run_dir, controller_result, source_checkout, git_executable, site,
                    expected_commit, expected_device_uuid, expected_freeze_sha256,
                    freeze_observation='controller-reported'):
    """Audit a retained private attempt directly; never promote its profiles.

    The three expected values, Git executable, checkout and package site are
    selected by the operator. Freeze chronology is an operator declaration;
    the default merely reports the producer's source-defined ordering. These checks are retrospective:
    saved counters and cleanup are self-reports, and matching files do not
    attest execution, all dependency bytes, or worst-case GPU memory.
    """
    _need(type(freeze_observation) is str and freeze_observation in ('controller-reported', 'independent-prelaunch'),
          'unrecognized freeze observation')
    try:
        directory = Path(run_dir).absolute()
        info = directory.lstat()
        _need(stat.S_ISDIR(info.st_mode) and not directory.is_symlink()
              and not getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
              and directory.resolve(strict=True) == directory, 'private run directory is aliased')
        original_id = (info.st_dev, info.st_ino)
        names = {item.name for item in directory.iterdir()}
        _need(not any(name.casefold() == 'failure.json' for name in names), 'owner failure record exists')
        _need({name for name in names if name.lower().startswith('stage-')} ==
              {f'stage-{index:03}.json' for index in range(84)}, 'stage file set differs')
        files = {'run_manifest': 'run-manifest.json', 'controller_source': 'controller-source.json',
                 'source_base': 'controller-source-base.json', 'source': 'source-manifest.json',
                 'runtime': 'runtime-manifest.json', 'assets': 'asset-manifest.json'}
        identity_files = {key: _read_file(directory / name) for key, name in files.items()}
        identity = audit_identity(identity_files, expected_commit=expected_commit,
                                  expected_device_uuid=expected_device_uuid,
                                  expected_freeze_sha256=expected_freeze_sha256)
        frozen_dir = identity['freeze']['directory']
        _need(Path(frozen_dir['path']) == directory
              and (frozen_dir['device'], frozen_dir['inode']) == original_id,
              'run directory differs from the prelaunch freeze')
        source_root = Path(source_checkout).resolve(strict=True)
        _need(Path(identity['freeze']['source']['checkout']).resolve(strict=True) == source_root,
              'selected source checkout differs from prelaunch')
        source_count = verify_committed_sources(source_root, git_executable, expected_commit,
                                                identity['controller_source'], identity_files['source'])
        origin_count = verify_dependency_origins(identity_files['source'], site)
        stage_bytes = [_read_file(directory / f'stage-{index:03}.json') for index in range(84)]
        stages = [_line_json(raw, 'stage') for raw in stage_bytes]
        stage_report = audit_stage_records(stages, identity['warm_key'])
        profile_documents = [_read_file(directory / (name + '-profile.json')) for name in ('cold', 'warm')]
        evidence_documents = [_read_file(directory / (name + '-evidence.json')) for name in ('cold', 'warm')]
        owner_files = {'started': 'started.json', 'owner': 'owner.json', 'candidate': 'candidate.json',
                       'final': 'final.json', 'supervision': 'supervision.json', 'model_checks': 'model-checks.json',
                       'preload_admission': 'preload-admission.json', 'cleanup': 'cleanup.json'}
        saved_owner = {key: _read_file(directory / name) for key, name in owner_files.items()}
        candidate = _line_json(saved_owner['candidate'], 'owner candidate')
        _fields(candidate, ('schema', 'run_id', 'supervisor_run_id', 'pid', 'value'), 'owner candidate')
        _fields(candidate['value'], ('measurement', 'cleanup'), 'owner candidate value')
        measurement = candidate['value']['measurement']
        _need(type(measurement) is dict and 'profile_ids' in measurement and 'estimates_bytes' in measurement,
              'candidate profile identities are missing')
        profiles = audit_profiles(stage_report, identity['warm_key'], profile_documents, evidence_documents,
                                  identity_files['source'], measurement['profile_ids'], measurement['estimates_bytes'])
        journal = audit_journal(directory / 'journal', stage_report, identity['run_id'],
                                identity['source_sha256'], profiles['profile_ids'])
        controller_bytes = _read_file(controller_result)
        owner = audit_reaped_records(identity, expected_freeze_sha256, profiles, controller_bytes, saved_owner)
        current = directory.lstat()
        final_names = {item.name for item in directory.iterdir()}
        _need((current.st_dev, current.st_ino) == original_id
              and not any(name.casefold() == 'failure.json' for name in final_names)
              and {name for name in final_names if name.lower().startswith('stage-')} ==
                  {f'stage-{index:03}.json' for index in range(84)},
              'private run directory changed during audit')
        artifact_bytes = {**identity_files, **saved_owner, 'controller_result': controller_bytes,
                          'cold_profile': profile_documents[0], 'warm_profile': profile_documents[1],
                          'cold_evidence': evidence_documents[0], 'warm_evidence': evidence_documents[1]}
        return {'status': owner['status'], 'run_id': identity['run_id'], 'source_commit': expected_commit,
                'run_manifest_sha256': expected_freeze_sha256, 'freeze_observation': freeze_observation,
                'sample_count': stage_report['sample_count'],
                'stage_count': stage_report['stage_count'], 'journal_event_count': journal['event_count'],
                'journal_event_sha256': journal['event_sha256'], 'profile_ids': profiles['profile_ids'],
                'estimates_bytes': profiles['estimates_bytes'], 'committed_source_files_checked': source_count,
                'dependency_entry_files_checked': origin_count,
                'stage_stream_sha256': hashlib.sha256(b''.join(stage_bytes)).hexdigest(),
                'artifact_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in artifact_bytes.items()},
                'limitations': ['saved observations are self-reports and do not attest execution',
                    'freeze chronology label is supplied by the operator and is not independently established by this audit',
                    'asset pins are checked but cached asset bytes are not independently rehashed',
                    'dependency entry files alone do not attest native or transitive dependencies',
                    'sampled memory and cleanup do not prove worst-case bounds or global GPU quiescence']}
    except OSError as error:
        raise ValueError('saved attempt or selected source is unavailable') from error


def audit_incomplete_directory(*, run_dir, source_checkout, git_executable, site,
                               expected_commit, expected_device_uuid, expected_freeze_sha256):
    """Read a retained attempt's stage prefix without asserting owner completion.

    The controller and supervisor must be reviewed independently. The journal
    is fingerprinted but its callbacks and terminal are not validated here.
    This helper never derives or approves a resource profile.
    """
    try:
        directory = Path(run_dir).absolute()
        info = directory.lstat()
        _need(stat.S_ISDIR(info.st_mode) and not directory.is_symlink()
              and not getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
              and directory.resolve(strict=True) == directory, 'private run directory is aliased')
        original_id = (info.st_dev, info.st_ino)
        names = {item.name for item in directory.iterdir()}
        stage_names = {name for name in names if name.lower().startswith('stage-')}
        _need(len(stage_names) <= 85 and stage_names ==
              {f'stage-{index:03}.json' for index in range(len(stage_names))},
              'stage files are not a contiguous bounded prefix')
        identity_names = {'run_manifest': 'run-manifest.json', 'controller_source': 'controller-source.json',
                          'source_base': 'controller-source-base.json', 'source': 'source-manifest.json',
                          'runtime': 'runtime-manifest.json', 'assets': 'asset-manifest.json'}
        identity_files = {key: _read_file(directory / name) for key, name in identity_names.items()}
        identity = audit_identity(identity_files, expected_commit=expected_commit,
                                  expected_device_uuid=expected_device_uuid,
                                  expected_freeze_sha256=expected_freeze_sha256)
        frozen_dir = identity['freeze']['directory']
        _need(Path(frozen_dir['path']) == directory
              and (frozen_dir['device'], frozen_dir['inode']) == original_id,
              'run directory differs from the prelaunch freeze')
        source_root = Path(source_checkout).resolve(strict=True)
        _need(Path(identity['freeze']['source']['checkout']).resolve(strict=True) == source_root,
              'selected source checkout differs from prelaunch')
        source_count = verify_committed_sources(source_root, git_executable, expected_commit,
                                                identity['controller_source'], identity_files['source'])
        origin_count = verify_dependency_origins(identity_files['source'], site)
        stage_bytes = [_read_file(directory / f'stage-{index:03}.json') for index in range(len(stage_names))]
        stages = [_line_json(raw, 'stage') for raw in stage_bytes]
        if len(stages) == 84 and stages[-1].get('phase') != 'failed':
            complete = audit_stage_records(stages, identity['warm_key'])
            report = {'status': 'stage_sequence_consistent_owner_unverified_not_approved',
                      'stage_count': complete['stage_count'], 'completed_samples': complete['sample_count'],
                      'failure': None, 'next_kind': None, 'next_phase': None}
        else:
            report = audit_partial_stage_records(stages, identity['warm_key'])
        failure_names = [name for name in names if name.casefold() == 'failure.json']
        _need(len(failure_names) <= 1, 'multiple failure sidecars exist')
        failure_raw = _read_file(directory / failure_names[0]) if failure_names else None
        journal = directory / 'journal'
        events_sha, events_length, complete_lines = None, None, None
        if journal.exists():
            journal_info = journal.lstat()
            _need(stat.S_ISDIR(journal_info.st_mode) and not journal.is_symlink()
                  and not getattr(journal_info, 'st_file_attributes', 0) &
                          getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400), 'journal directory is aliased')
            events_path = journal / 'events.jsonl'
            if events_path.exists():
                events = _read_file(events_path, allow_empty=True)
                events_sha, events_length, complete_lines = (hashlib.sha256(events).hexdigest(),
                                                              len(events), events.count(b'\n'))
        current = directory.lstat()
        final_names = {item.name for item in directory.iterdir()}
        _need((current.st_dev, current.st_ino) == original_id
              and {name for name in final_names if name.lower().startswith('stage-')} == stage_names
              and {name for name in final_names if name.casefold() == 'failure.json'} == set(failure_names),
              'private run directory changed during audit')
        report.update({'run_id': identity['run_id'], 'source_commit': expected_commit,
            'run_manifest_sha256': expected_freeze_sha256, 'freeze_observation': 'controller-reported',
            'stage_stream_sha256': hashlib.sha256(b''.join(stage_bytes)).hexdigest(),
            'committed_source_files_checked': source_count, 'dependency_entry_files_checked': origin_count,
            'failure_sidecar_sha256': hashlib.sha256(failure_raw).hexdigest() if failure_raw is not None else None,
            'journal_observed_sha256': events_sha, 'journal_observed_bytes': events_length,
            'journal_newline_count': complete_lines, 'journal_validated': False, 'supervision_checked': False,
            'limitations': ['the saved stages are self-reports and do not attest execution',
                'freeze ordering is controller-reported rather than independently witnessed',
                'journal fingerprint is not a callback or terminal audit',
                'process exit, cleanup, retained aliases and marker recovery are not checked here',
                'source and dependency checks are retrospective and cached assets are not rehashed']})
        return report
    except OSError as error:
        raise ValueError('incomplete attempt or selected source is unavailable') from error


def main(argv=None):
    """Write a compact, path-free audit report to stdout; never mutate the attempt."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'controller-result', 'source-checkout', 'git-executable', 'site',
                 'expected-commit', 'expected-device-uuid', 'expected-freeze-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--freeze-observation', choices=('controller-reported', 'independent-prelaunch'),
                        default='controller-reported')
    args = vars(parser.parse_args(argv))
    try:
        report = audit_directory(**args)
    except ValueError as error:
        print('saved evidence audit failed: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
