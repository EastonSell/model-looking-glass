"""Sampled serial-versus-mixed forward checks with a frozen v1 protocol.

Identity, pre-registration, executor isolation and device leases are external
trust boundaries. A passing sample is not a proof of batch separability.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import re
import threading

import torch

from .resident_worker import RequestCancelled


PROTOCOL_VERSION = 'model-looking-glass-batch-validation-v1'
EVIDENCE_VERSION = 'model-looking-glass-batch-evidence-v1'
_DTYPES = {'bool': torch.bool, 'int32': torch.int32, 'int64': torch.int64,
           'float16': torch.float16, 'bfloat16': torch.bfloat16,
           'float32': torch.float32, 'float64': torch.float64}
_BYTES = {'bool': 1, 'int32': 4, 'int64': 8, 'float16': 2,
          'bfloat16': 2, 'float32': 4, 'float64': 8}
_BINDING_FIELDS = {
    'profile_id', 'certification_id', 'model_id', 'model_fingerprint', 'checkpoint',
    'checkpoint_sha256', 'adapter', 'adapter_sha256', 'runtime_id', 'backend',
    'device', 'model_dtype', 'output_dtype', 'input_schema', 'input_shape',
    'output_width', 'approved_rows', 'max_requests', 'max_per_owner',
    'max_batch_rows', 'max_token_positions', 'max_logical_input_bytes',
    'max_logical_output_bytes', 'max_logical_total_bytes', 'seed', 'serial_atol',
    'serial_rtol', 'mode', 'deterministic', 'batch_separable', 'rng_policy',
    'cache_policy', 'mutable_state_policy', 'tracing', 'intervention',
}


class NumericModeError(RuntimeError):
    """The caller's CPU floating-point mode cannot preserve denormal results."""


def _require_denormals():
    # Check both denormal inputs (DAZ) and underflowed outputs (FTZ). Integer
    # views inspect raw bits; floating comparisons would themselves flush them.
    for integer, floating, normal_bits in ((torch.int32, torch.float32, 0x00800000),
                                           (torch.int64, torch.float64, 0x0010000000000000)):
        values = torch.tensor([1, normal_bits], dtype=integer, device='cpu').view(floating)
        denormal_input = (values[0] * 2.0).view(integer).item()
        denormal_output = (values[1] * 0.5).view(integer).item()
        if denormal_input == 0 or denormal_output == 0:
            raise NumericModeError('CPU runtime must preserve float32 and float64 denormals')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _owned_json_dict(value, name):
    if type(value) is not dict:
        raise ValueError(f'{name} must be a dictionary')
    try:
        return json.loads(_canonical(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must contain finite JSON data') from exc


def _digest(value, name):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')


def _name(value, name):
    if type(value) is not str or not value.strip() or len(value) > 128:
        raise ValueError(f'{name} must be a nonempty string of at most 128 characters')


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _validate_binding(binding):
    if type(binding) is not dict or set(binding) != _BINDING_FIELDS:
        raise ValueError('binding fields do not match the v1 protocol')
    for name in ('profile_id', 'certification_id', 'model_id', 'checkpoint', 'adapter',
                 'runtime_id', 'backend'):
        _name(binding[name], name)
    for name in ('model_fingerprint', 'checkpoint_sha256', 'adapter_sha256'):
        _digest(binding[name], name)
    device = binding['device']
    if type(device) is not str or not (device == 'cpu' or re.fullmatch(r'cuda:[0-9]+', device)):
        raise ValueError('device must be cpu or an indexed CUDA device')
    if binding['model_dtype'] not in ('float16', 'bfloat16', 'float32', 'float64'):
        raise ValueError('unsupported model dtype')
    if binding['output_dtype'] not in ('float16', 'bfloat16', 'float32', 'float64'):
        raise ValueError('unsupported output dtype')
    if (binding['mode'] != 'eval' or binding['deterministic'] is not True
            or binding['batch_separable'] is not True or binding['rng_policy'] != 'none'
            or binding['cache_policy'] != 'disabled' or binding['mutable_state_policy'] != 'none'
            or binding['tracing'] is not False or binding['intervention'] is not False):
        raise ValueError('binding must declare the compatible eval execution contract')
    shape = binding['input_shape']
    if type(shape) is not list or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('input_shape must contain two positive integers')
    schema = binding['input_schema']
    if type(schema) is not list or not schema:
        raise ValueError('input_schema must be a nonempty list')
    allowed = {'input_ids': ('int32', 'int64'), 'attention_mask': ('bool', 'int32', 'int64'),
               'position_ids': ('int32', 'int64'), 'token_type_ids': ('int32', 'int64')}
    names = []
    for entry in schema:
        if type(entry) is not list or len(entry) != 2:
            raise ValueError('invalid input schema entry')
        name, dtype = entry
        if type(name) is not str or name not in allowed or dtype not in allowed[name] or name in names:
            raise ValueError('unsupported or duplicate input schema entry')
        names.append(name)
    if 'input_ids' not in names:
        raise ValueError('input_ids are required')
    for name in ('output_width', 'max_requests', 'max_per_owner', 'max_batch_rows',
                 'max_token_positions', 'max_logical_input_bytes', 'max_logical_output_bytes',
                 'max_logical_total_bytes'):
        _positive(binding[name], name)
    if type(binding['seed']) is not int or not 0 <= binding['seed'] < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    for name in ('serial_atol', 'serial_rtol'):
        value = binding[name]
        if type(value) not in (int, float):
            raise ValueError(f'{name} must be finite and nonnegative')
        try:
            converted = float(value)
            valid = math.isfinite(converted) and converted >= 0 and converted == value
        except (OverflowError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f'{name} must be finite and nonnegative')
    approved = binding['approved_rows']
    if (type(approved) is not list or not approved
            or any(type(row) is not int or row <= 0 or row > binding['max_batch_rows']
                   or row % shape[0] for row in approved)
            or len(set(approved)) != len(approved)):
        raise ValueError('approved_rows must contain distinct supported batch sizes')


def _check_group_limits(binding, count):
    rows = count * binding['input_shape'][0]
    positions = rows * binding['input_shape'][1]
    inputs = positions * sum(_BYTES[dtype] for _, dtype in binding['input_schema'])
    outputs = positions * binding['output_width'] * _BYTES[binding['output_dtype']]
    if (count > binding['max_requests'] or rows not in binding['approved_rows']
            or rows > binding['max_batch_rows'] or positions > binding['max_token_positions']
            or inputs > binding['max_logical_input_bytes']
            or outputs > binding['max_logical_output_bytes']
            or inputs + outputs > binding['max_logical_total_bytes']):
        raise ValueError('mixed group is outside the bound workload profile')


def binding_from_profile(profile, *, model_fingerprint, checkpoint_sha256, adapter_sha256, backend):
    """Convert the actual trusted microbatch profile into its v1 JSON binding."""
    _require_denormals()
    from .microbatch import BatchProfile, ForwardContract, _validate_profile
    if type(profile) is not BatchProfile or type(profile.contract) is not ForwardContract:
        raise ValueError('an exact microbatch BatchProfile and ForwardContract are required')
    _validate_profile(profile)
    contract = profile.contract

    def dtype_name(dtype):
        for name, value in _DTYPES.items():
            if dtype == value:
                return name
        raise ValueError('unsupported profile dtype')

    def tolerance(value):
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('profile tolerance cannot be represented') from exc
        if converted != value:
            raise ValueError('profile tolerance must be exactly representable as binary64')
        return converted

    binding = {
        'profile_id': profile.profile_id, 'certification_id': profile.certification_id,
        'model_id': contract.model_id, 'model_fingerprint': model_fingerprint,
        'checkpoint': contract.checkpoint, 'checkpoint_sha256': checkpoint_sha256,
        'adapter': contract.adapter, 'adapter_sha256': adapter_sha256,
        'runtime_id': contract.runtime_id, 'backend': backend, 'device': contract.device,
        'model_dtype': dtype_name(contract.model_dtype), 'output_dtype': dtype_name(profile.output_dtype),
        'input_schema': [[name, dtype_name(dtype)] for name, dtype in profile.input_schema],
        'input_shape': list(profile.input_shape), 'output_width': profile.output_width,
        'approved_rows': list(profile.approved_rows), 'max_requests': profile.max_requests,
        'max_per_owner': profile.max_per_owner, 'max_batch_rows': profile.max_batch_rows,
        'max_token_positions': profile.max_token_positions,
        'max_logical_input_bytes': profile.max_logical_input_bytes,
        'max_logical_output_bytes': profile.max_logical_output_bytes,
        'max_logical_total_bytes': profile.max_logical_total_bytes,
        'seed': contract.seed, 'serial_atol': tolerance(profile.serial_atol),
        'serial_rtol': tolerance(profile.serial_rtol), 'mode': contract.mode,
        'deterministic': contract.deterministic, 'batch_separable': contract.batch_separable,
        'rng_policy': contract.rng_policy, 'cache_policy': contract.cache_policy,
        'mutable_state_policy': contract.mutable_state_policy,
        'tracing': contract.tracing, 'intervention': contract.intervention,
    }
    _validate_binding(binding)
    return binding


def _input_records(binding, inputs):
    _require_denormals()
    _validate_binding(binding)
    if type(inputs) is not dict or set(inputs) != {name for name, _ in binding['input_schema']}:
        raise ValueError('inputs must be a plain dictionary matching the input schema')
    records = []
    for name, dtype in binding['input_schema']:
        value = inputs[name]
        if (type(value) is not torch.Tensor or value.device.type != 'cpu'
                or value.layout != torch.strided or value.dtype != _DTYPES[dtype]
                or list(value.shape) != binding['input_shape'] or value.requires_grad):
            raise ValueError(f'invalid CPU input tensor: {name}')
        records.append({'name': name, 'dtype': dtype, 'shape': list(value.shape),
                        'values': value.tolist()})
    return sorted(records, key=lambda item: item['name'])


def input_digest(binding, inputs):
    """Hash the complete request's names, dtypes, shapes and integer values."""
    return _sha(_input_records(binding, inputs))


def input_row_digests(binding, inputs):
    """Hash each complete input row independently for split disjointness."""
    records = _input_records(binding, inputs)
    sequence = binding['input_shape'][1]
    return [_sha([{'name': item['name'], 'dtype': item['dtype'], 'shape': [1, sequence],
                   'values': [item['values'][row]]} for item in records])
            for row in range(binding['input_shape'][0])]


def freeze_protocol(binding, groups, *, discovery_row_hashes):
    """Build the protocol to retain separately before evaluating held-out cases."""
    _require_denormals()
    _validate_binding(binding)
    if type(discovery_row_hashes) not in (list, tuple) or not discovery_row_hashes:
        raise ValueError('nonempty discovery row hashes are required')
    for digest in discovery_row_hashes:
        _digest(digest, 'discovery row hash')
    discovery = sorted(set(discovery_row_hashes))
    if type(groups) not in (list, tuple) or not 1 <= len(groups) <= 64:
        raise ValueError('one to 64 planned mixed groups are required')
    group_ids = set()
    manifest = []
    for group in groups:
        if type(group) is not dict or set(group) != {'group_id', 'cases'}:
            raise ValueError('invalid mixed group')
        group_id, cases = group['group_id'], group['cases']
        _name(group_id, 'group_id')
        if group_id in group_ids:
            raise ValueError('group IDs must be unique')
        group_ids.add(group_id)
        if type(cases) not in (list, tuple) or len(cases) < 2:
            raise ValueError('each group must contain at least two cases')
        _check_group_limits(binding, len(cases))
        seen, owners, hashes, records = set(), Counter(), set(), []
        for case in cases:
            if type(case) is not dict or set(case) != {'case_id', 'owner', 'inputs'}:
                raise ValueError('invalid case')
            case_id, owner = case['case_id'], case['owner']
            _name(case_id, 'case_id')
            _name(owner, 'owner')
            if case_id in seen:
                raise ValueError('case IDs must be unique within a group')
            seen.add(case_id)
            owners[owner] += 1
            digest = input_digest(binding, case['inputs'])
            row_digests = input_row_digests(binding, case['inputs'])
            if any(row in discovery for row in row_digests):
                raise ValueError('held-out input row overlaps discovery input')
            hashes.add(digest)
            records.append({'case_id': case_id, 'owner': owner, 'input_sha256': digest,
                            'row_sha256': row_digests})
        if len(owners) < 2 or len(hashes) < 2 or max(owners.values()) > binding['max_per_owner']:
            raise ValueError('a group needs distinct owners and varied inputs within the owner cap')
        manifest.append({'group_id': group_id, 'cases': records})
    protocol = {'version': PROTOCOL_VERSION, 'binding': json.loads(_canonical(binding)),
                'discovery_row_hashes': discovery, 'groups': manifest}
    protocol['protocol_sha256'] = _sha(protocol)
    return protocol


def _verify_protocol_record(protocol, expected_sha256):
    _require_denormals()
    _digest(expected_sha256, 'expected protocol hash')
    if (type(protocol) is not dict or set(protocol) !=
            {'version', 'binding', 'discovery_row_hashes', 'groups', 'protocol_sha256'}
            or protocol['version'] != PROTOCOL_VERSION
            or protocol['protocol_sha256'] != expected_sha256):
        raise ValueError('protocol identity does not match the frozen hash')
    try:
        unsigned = {key: value for key, value in protocol.items() if key != 'protocol_sha256'}
        if _sha(unsigned) != expected_sha256:
            raise ValueError('protocol has been changed')
    except (TypeError, OverflowError) as exc:
        raise ValueError('protocol is not finite JSON data') from exc
    binding = protocol['binding']
    _validate_binding(binding)
    discovery = protocol['discovery_row_hashes']
    if type(discovery) is not list or not discovery:
        raise ValueError('discovery input hashes are required')
    for digest in discovery:
        _digest(digest, 'discovery input hash')
    if discovery != sorted(set(discovery)):
        raise ValueError('discovery hashes must be sorted and unique')
    groups = protocol['groups']
    if type(groups) is not list or not 1 <= len(groups) <= 64:
        raise ValueError('one to 64 planned mixed groups are required')
    group_ids = set()
    for group in groups:
        if type(group) is not dict or set(group) != {'group_id', 'cases'}:
            raise ValueError('invalid mixed group')
        _name(group['group_id'], 'group_id')
        if group['group_id'] in group_ids:
            raise ValueError('group IDs must be unique')
        group_ids.add(group['group_id'])
        cases = group['cases']
        if type(cases) is not list or len(cases) < 2:
            raise ValueError('mixed groups require at least two cases')
        _check_group_limits(binding, len(cases))
        ids, owners, hashes = set(), Counter(), set()
        for case in cases:
            if type(case) is not dict or set(case) != {'case_id', 'owner', 'input_sha256', 'row_sha256'}:
                raise ValueError('invalid frozen case')
            _name(case['case_id'], 'case_id')
            _name(case['owner'], 'owner')
            _digest(case['input_sha256'], 'held-out input hash')
            row_hashes = case['row_sha256']
            if type(row_hashes) is not list or len(row_hashes) != binding['input_shape'][0]:
                raise ValueError('held-out row hashes do not match request shape')
            for row in row_hashes:
                _digest(row, 'held-out row hash')
            if case['case_id'] in ids or any(row in discovery for row in row_hashes):
                raise ValueError('duplicate case ID or discovery overlap')
            ids.add(case['case_id'])
            owners[case['owner']] += 1
            hashes.add(case['input_sha256'])
        if len(owners) < 2 or len(hashes) < 2 or max(owners.values()) > binding['max_per_owner']:
            raise ValueError('mixed group owner or input requirements not met')


def _valid_output(output, shape, dtype):
    return (type(output) is torch.Tensor and output.device.type == 'cpu'
            and output.layout == torch.strided and tuple(output.shape) == shape
            and output.dtype == _DTYPES[dtype] and not output.requires_grad
            and output.grad_fn is None and bool(torch.isfinite(output).all()))


def _within_tolerance(ref, got, atol, rtol, difference):
    if got == ref:
        return True
    if atol == 0 and rtol == 0:
        return False
    product = rtol * abs(ref)
    product_lo = max(0.0, math.nextafter(product, -math.inf))
    product_hi = math.nextafter(product, math.inf)
    bound_lo = max(0.0, math.nextafter(atol + product_lo, -math.inf))
    bound_hi = math.nextafter(atol + product_hi, math.inf)
    difference_lo = max(0.0, math.nextafter(difference, -math.inf))
    difference_hi = math.nextafter(difference, math.inf)
    if difference_hi <= bound_lo:
        return True
    if difference_lo > bound_hi:
        return False
    # Resolve close boundaries and overflow using exact binary64 rational values.
    exact_ref, exact_got = Fraction.from_float(ref), Fraction.from_float(got)
    exact_atol, exact_rtol = Fraction.from_float(atol), Fraction.from_float(rtol)
    return abs(exact_got - exact_ref) <= exact_atol + exact_rtol * abs(exact_ref)


def _comparison(reference, candidate, atol, rtol):
    _require_denormals()
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError('comparison outputs must have the same shape')
    # Scalar arithmetic runs on the checked calling thread rather than an
    # already-created intra-op worker with an independently inherited FP mode.
    # Chunking bounds temporary Python float objects; flattening a noncontiguous
    # tensor can still require a full CPU copy accounted for by the caller.
    reference = reference.reshape(-1)
    candidate = candidate.reshape(-1)
    if reference.numel() != candidate.numel():
        raise ValueError('comparison outputs must have the same number of elements')
    atol, rtol = float(atol), float(rtol)
    max_abs = max_relative = 0.0
    zero_reference = False
    close = True
    for start in range(0, reference.numel(), 4096):
        left = reference[start:start + 4096].tolist()
        right = candidate[start:start + 4096].tolist()
        for ref, got in zip(left, right, strict=True):
            difference = abs(got - ref)
            denominator = abs(ref)
            close = close and _within_tolerance(ref, got, atol, rtol, difference)
            if denominator == 0:
                ratio = 0.0 if difference == 0 else float('inf')
                zero_reference = zero_reference or difference != 0
            elif math.isinf(difference):
                ratio = abs(got * 0.5 - ref * 0.5) / (denominator * 0.5)
            else:
                ratio = difference / denominator
            max_abs = max(max_abs, difference)
            max_relative = max(max_relative, ratio)
    result = {
        'status': 'matched' if close else 'mismatch',
        'max_abs_error': max_abs if math.isfinite(max_abs) else None,
        'absolute_error_overflow': not math.isfinite(max_abs),
        'max_relative_error': max_relative if math.isfinite(max_relative) else None,
        'relative_error_nonfinite_reason': ('zero_reference' if zero_reference else 'overflow')
            if not math.isfinite(max_relative) else None,
    }
    _require_denormals()
    return result


def _cancelled(event):
    return event is not None and event.is_set()


def run_validation(protocol, binding, groups, execute, *, expected_protocol_sha256,
                   cancel_event=None):
    """Execute frozen cases serially and in mixed groups via a trusted callback.

    The callback receives CPU input dictionaries and request_id/seed keywords,
    and returns completed detached CPU logits. It owns model and device safety.
    """
    # Admission requires caller ownership while these plain-JSON records are
    # copied. All subsequent reads use these private snapshots, including
    # thresholds and the manifest consulted after an executor callback.
    _require_denormals()
    protocol = _owned_json_dict(protocol, 'protocol')
    binding = _owned_json_dict(binding, 'binding')
    _verify_protocol_record(protocol, expected_protocol_sha256)
    expected = freeze_protocol(binding, groups, discovery_row_hashes=protocol['discovery_row_hashes'])
    if _canonical(expected) != _canonical(protocol):
        raise ValueError('binding, cases or inputs differ from the frozen protocol')
    if not callable(execute):
        raise ValueError('execute must be callable')
    if cancel_event is not None and type(cancel_event) is not threading.Event:
        raise ValueError('cancel_event must be a threading.Event')
    results = []
    for group in protocol['groups']:
        results.append({'group_id': group['group_id'], 'cases': [
            {**case, 'status': 'not_run', 'max_abs_error': None,
             'absolute_error_overflow': False, 'max_relative_error': None,
             'relative_error_nonfinite_reason': None} for case in group['cases']]})
    report = {'version': EVIDENCE_VERSION, 'protocol_sha256': expected_protocol_sha256,
              'status': 'inconclusive', 'complete': False,
              'calls': {'serial': 0, 'mixed': 0}, 'groups': results,
              'tested_rows': [], 'error': None}
    stopped = False
    rows, sequence = binding['input_shape']
    width, dtype = binding['output_width'], binding['output_dtype']

    def stop(kind, stage, group_index, case_index=None, exc=None):
        nonlocal stopped
        stopped = True
        report['error'] = {'kind': kind, 'stage': stage, 'group_id': results[group_index]['group_id'],
                           'case_id': (results[group_index]['cases'][case_index]['case_id']
                                       if case_index is not None else None),
                           'exception_type': type(exc).__name__ if exc is not None else None}
        if case_index is not None:
            results[group_index]['cases'][case_index]['status'] = 'inconclusive'

    for gi, group in enumerate(groups):
        if _cancelled(cancel_event):
            stop('cancelled', 'before_group', gi)
            break
        try:
            with torch.inference_mode(False), torch.no_grad():
                snapshots = [{name: case['inputs'][name].detach().clone()
                              for name, _ in binding['input_schema']} for case in group['cases']]
            for ci, snapshot in enumerate(snapshots):
                if input_digest(binding, snapshot) != protocol['groups'][gi]['cases'][ci]['input_sha256']:
                    stop('input_changed', 'snapshot', gi, ci)
                    break
        except Exception as exc:
            stop('preparation_error', 'snapshot', gi, exc=exc)
        if stopped:
            break
        serial = []
        for ci, snapshot in enumerate(snapshots):
            if _cancelled(cancel_event):
                stop('cancelled', 'before_serial', gi, ci)
                break
            try:
                with torch.inference_mode(False), torch.no_grad():
                    call_inputs = {name: value.clone() for name, value in snapshot.items()}
                report['calls']['serial'] += 1
                output = execute(call_inputs, request_id=f'serial-{gi}-{ci}', seed=binding['seed'])
                _require_denormals()
                if _cancelled(cancel_event):
                    stop('cancelled', 'after_serial', gi, ci)
                    break
                if not _valid_output(output, (rows, sequence, width), dtype):
                    stop('invalid_output', 'serial', gi, ci)
                    break
                with torch.inference_mode(False), torch.no_grad():
                    serial.append(output.detach().clone())
            except RequestCancelled as exc:
                _require_denormals()
                stop('cancelled', 'serial', gi, ci, exc)
                break
            except Exception as exc:
                _require_denormals()
                stop('phase_error', 'serial', gi, ci, exc)
                break
        if stopped:
            break
        if _cancelled(cancel_event):
            stop('cancelled', 'before_mixed', gi)
            break
        try:
            with torch.inference_mode(False), torch.no_grad():
                combined = {name: torch.cat([snapshot[name] for snapshot in snapshots], dim=0)
                            for name, _ in binding['input_schema']}
            report['calls']['mixed'] += 1
            output = execute(combined, request_id=f'mixed-{gi}', seed=binding['seed'])
            _require_denormals()
            if _cancelled(cancel_event):
                stop('cancelled', 'after_mixed', gi)
                break
            if not _valid_output(output, (len(snapshots) * rows, sequence, width), dtype):
                stop('invalid_output', 'mixed', gi)
                break
            for ci, reference in enumerate(serial):
                if _cancelled(cancel_event):
                    stop('cancelled', 'before_comparison', gi, ci)
                    break
                current = output[ci * rows:(ci + 1) * rows]
                comparison = _comparison(reference, current, binding['serial_atol'], binding['serial_rtol'])
                if _cancelled(cancel_event):
                    stop('cancelled', 'after_comparison', gi, ci)
                    break
                results[gi]['cases'][ci].update(comparison)
            if stopped:
                break
        except RequestCancelled as exc:
            _require_denormals()
            stop('cancelled', 'mixed', gi, exc=exc)
            break
        except Exception as exc:
            _require_denormals()
            stop('phase_error', 'mixed', gi, exc=exc)
            break
    # This check is the cancellation linearization point for publication.
    if not stopped and _cancelled(cancel_event):
        stop('cancelled', 'before_publication', len(results) - 1)
    all_cases = [case for group in results for case in group['cases']]
    report['complete'] = (report['error'] is None and
                          all(case['status'] in ('matched', 'mismatch') for case in all_cases))
    report['tested_rows'] = sorted({len(group['cases']) * rows for group in results
                                    if all(case['status'] in ('matched', 'mismatch')
                                           for case in group['cases'])})
    if any(case['status'] == 'mismatch' for case in all_cases):
        report['status'] = 'failed'
    elif report['complete'] and report['error'] is None:
        report['status'] = 'passed'
    _require_denormals()
    report['evidence_sha256'] = _sha(report)
    return report


def _nonnegative(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, ValueError):
        return False


def _verify_evidence(report, protocol, protocol_sha256, evidence_sha256):
    _verify_protocol_record(protocol, protocol_sha256)
    _digest(evidence_sha256, 'expected evidence hash')
    fields = {'version', 'protocol_sha256', 'status', 'complete', 'calls', 'groups',
              'tested_rows', 'error', 'evidence_sha256'}
    if (type(report) is not dict or set(report) != fields or report.get('version') != EVIDENCE_VERSION
            or report.get('protocol_sha256') != protocol_sha256
            or report.get('evidence_sha256') != evidence_sha256):
        raise ValueError('evidence identity does not match')
    unsigned = {key: value for key, value in report.items() if key != 'evidence_sha256'}
    if _sha(unsigned) != evidence_sha256:
        raise ValueError('evidence has been changed')
    if type(report['complete']) is not bool or report['status'] not in ('passed', 'failed', 'inconclusive'):
        raise ValueError('invalid evidence status')
    counts = report['calls']
    groups = report['groups']
    manifest = protocol['groups']
    if (type(counts) is not dict or set(counts) != {'serial', 'mixed'}
            or any(type(value) is not int or value < 0 for value in counts.values())
            or counts['serial'] > sum(len(group['cases']) for group in manifest)
            or counts['mixed'] > len(manifest) or type(groups) is not list or len(groups) != len(manifest)):
        raise ValueError('invalid evidence call counts or group coverage')
    statuses = []
    metric_fields = {'case_id', 'owner', 'input_sha256', 'row_sha256', 'status', 'max_abs_error',
                     'absolute_error_overflow', 'max_relative_error', 'relative_error_nonfinite_reason'}
    for group, expected_group in zip(groups, manifest):
        if (type(group) is not dict or set(group) != {'group_id', 'cases'}
                or group['group_id'] != expected_group['group_id'] or type(group['cases']) is not list
                or len(group['cases']) != len(expected_group['cases'])):
            raise ValueError('evidence group does not match the frozen manifest')
        for case, expected_case in zip(group['cases'], expected_group['cases']):
            if type(case) is not dict or set(case) != metric_fields or any(
                    case[key] != expected_case[key] for key in ('case_id', 'owner', 'input_sha256', 'row_sha256')):
                raise ValueError('evidence case does not match the frozen manifest')
            status = case['status']
            if status not in ('matched', 'mismatch', 'not_run', 'inconclusive'):
                raise ValueError('invalid case status')
            statuses.append(status)
            absolute, overflow = case['max_abs_error'], case['absolute_error_overflow']
            relative, reason = case['max_relative_error'], case['relative_error_nonfinite_reason']
            if type(overflow) is not bool:
                raise ValueError('invalid error overflow flag')
            if status in ('matched', 'mismatch'):
                if ((overflow and absolute is not None) or
                        (not overflow and not _nonnegative(absolute)) or
                        (relative is None and reason not in ('zero_reference', 'overflow')) or
                        (relative is not None and (not _nonnegative(relative) or reason is not None))):
                    raise ValueError('invalid case error metrics')
                if (status == 'matched' and protocol['binding']['serial_atol'] == 0
                        and protocol['binding']['serial_rtol'] == 0 and (overflow or absolute != 0)):
                    raise ValueError('nonzero errors cannot match zero tolerances')
                if status == 'mismatch' and not overflow and absolute == 0:
                    raise ValueError('zero absolute error cannot be a mismatch')
                if not overflow and absolute == 0 and (relative != 0 or reason is not None):
                    raise ValueError('zero absolute error has zero relative error')
            elif absolute is not None or overflow or relative is not None or reason is not None:
                raise ValueError('uncompared case cannot carry error metrics')
    error = report['error']
    if error is not None:
        if (type(error) is not dict or set(error) != {'kind', 'stage', 'group_id', 'case_id', 'exception_type'}
                or error['kind'] not in ('cancelled', 'invalid_output', 'phase_error',
                                         'preparation_error', 'input_changed')
                or type(error['stage']) is not str
                or error['group_id'] not in {group['group_id'] for group in manifest}
                or (error['exception_type'] is not None and type(error['exception_type']) is not str)):
            raise ValueError('invalid evidence error')
        allowed_cases = {case['case_id'] for group in manifest if group['group_id'] == error['group_id']
                         for case in group['cases']}
        if error['case_id'] is not None and error['case_id'] not in allowed_cases:
            raise ValueError('evidence error references an unknown case')
    all_compared = all(status in ('matched', 'mismatch') for status in statuses)
    complete = error is None and all_compared
    tested_rows = sorted({len(group['cases']) * protocol['binding']['input_shape'][0]
                          for group in groups if all(case['status'] in ('matched', 'mismatch')
                                                    for case in group['cases'])})
    if report['tested_rows'] != tested_rows:
        raise ValueError('tested row counts do not match completed comparisons')
    if complete and (counts['serial'] != len(statuses) or counts['mixed'] != len(groups)):
        raise ValueError('complete evidence has incorrect model-call counts')
    expected_status = ('failed' if 'mismatch' in statuses else
                       'passed' if complete else 'inconclusive')
    if report['complete'] != complete or report['status'] != expected_status:
        raise ValueError('evidence completion or status is inconsistent')


def require_evidence(report, *, protocol, actual_binding, expected_protocol_sha256,
                     evidence_sha256, rows):
    """Reject evidence that is incomplete, failed, mismatched, or lacks rows.

    This verifies metadata and sampled comparisons, not the honesty of the
    executor or universal batch separability. Those are external requirements.
    """
    _require_denormals()
    report = _owned_json_dict(report, 'report')
    protocol = _owned_json_dict(protocol, 'protocol')
    actual_binding = _owned_json_dict(actual_binding, 'actual binding')
    _verify_evidence(report, protocol, expected_protocol_sha256, evidence_sha256)
    _validate_binding(actual_binding)
    if _canonical(actual_binding) != _canonical(protocol['binding']):
        raise ValueError('actual binding differs from the frozen profile')
    if (type(rows) is not int or rows <= 0 or report['status'] != 'passed'
            or not report['complete'] or rows not in report['tested_rows']):
        raise ValueError('no complete passing sampled evidence for this row count')


def save_evidence(path, report, *, protocol, expected_protocol_sha256):
    """Create a new JSON evidence file; return its externally retainable digest."""
    _require_denormals()
    report = _owned_json_dict(report, 'report')
    protocol = _owned_json_dict(protocol, 'protocol')
    digest = report.get('evidence_sha256')
    _verify_evidence(report, protocol, expected_protocol_sha256, digest)
    payload = _canonical(report) + b'\n'
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError('evidence exceeds the maximum file size')
    with open(path, 'xb') as stream:
        stream.write(payload)
    return digest


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def load_evidence(path, *, protocol, expected_protocol_sha256, evidence_sha256):
    """Read and verify a saved report against separately retained digests."""
    _require_denormals()
    protocol = _owned_json_dict(protocol, 'protocol')
    with open(path, 'rb') as stream:
        payload = stream.read(5 * 1024 * 1024 + 1)
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError('evidence exceeds the maximum file size')
    try:
        report = json.loads(payload, object_pairs_hook=_unique_pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid evidence JSON') from exc
    _verify_evidence(report, protocol, expected_protocol_sha256, evidence_sha256)
    return report
