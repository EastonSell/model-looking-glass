"""CPU-checkable pieces of the proposed agreement GPU calibration collector.

This module does not launch an owner or authorize a GPU workload. The final
collector must bind these checks to independently reconstructed live identity.
"""


_SITES = tuple((f'transformer.h.{layer}.{component}', 4, [] if component == 'mlp' else [0])
               for layer in (1, 2, 3, 4) for component in ('attn', 'mlp')) + (
                   ('transformer.h.5', 4, [0]), ('transformer.h.5', 1, [0]))
_ACTIONS = {
    **{f'layer{layer}_{component}_cross_patch': (f'transformer.h.{layer}.{component}', 4, 'patch', None)
       for layer in (1, 2, 3, 4) for component in ('attn', 'mlp')},
    'null': ('transformer.h.1.attn', 4, 'null', None),
    'self_patch': ('transformer.h.1.attn', 4, 'patch', 'before'),
    'block_last_cross_patch': ('transformer.h.5', 4, 'patch', None),
    'block_subject_cross_patch': ('transformer.h.5', 1, 'patch', None),
}
CALIBRATION_EXECUTION_READY = False
_SAMPLING_CUSTODY = [None] * 4


def _require_claims(calibrator):
    from .memory_budget import MemoryLease
    resident, job = calibrator._resident_lease, calibrator._memory_lease
    gib = 1024 ** 3
    if (type(resident) is not MemoryLease or type(job) is not MemoryLease or resident is job
            or resident._budget is not job._budget or resident.released or job.released
            or resident.estimated_incremental_bytes != 2 * gib
            or job.estimated_incremental_bytes != 3 * gib // 2
            or resident.trace_bytes('gpu') != 0 or resident.trace_bytes('host') != 0
            or job.trace_bytes('gpu') != 2 * 1024 ** 2 or job.trace_bytes('host') != 2 * 1024 ** 2):
        raise ValueError('fixed resident, job and trace commitments are not held')
    active = job._budget.active_leases()
    if len(active) != 2 or resident not in active or job not in active:
        raise ValueError('the fixed calibration claims are not the active budget')
    return job._budget


def _validate_headroom(key, recorded, budget):
    from .memory_budget import BudgetSnapshot, CudaMemorySnapshot
    from .resource_profile import _check_headroom
    if type(recorded) is not CudaMemorySnapshot or type(budget) is not BudgetSnapshot:
        raise ValueError('calibration telemetry has an invalid type')
    total = key.device_total_bytes
    minimum_reserve = max(1024 ** 3, (total * 15 + 99) // 100)
    numeric = (budget.reserve_bytes, budget.committed_incremental_bytes, budget.active_leases,
               budget.gpu_trace_bytes, budget.host_trace_bytes,
               budget.max_gpu_trace_bytes, budget.max_host_trace_bytes, budget.admissible_estimate_bytes)
    if (type(budget.telemetry) is not CudaMemorySnapshot or any(type(value) is not int or value < 0 for value in numeric)
            or recorded.device_total_bytes != total or budget.telemetry.device_total_bytes != total
            or budget.reserve_bytes < minimum_reserve or budget.committed_incremental_bytes != 7 * 1024 ** 3 // 2
            or budget.active_leases != 2 or budget.gpu_trace_bytes != 2 * 1024 ** 2
            or budget.host_trace_bytes != 2 * 1024 ** 2 or budget.max_gpu_trace_bytes < budget.gpu_trace_bytes
            or budget.max_host_trace_bytes < budget.host_trace_bytes
            or recorded.device_free_bytes < 3 * 1024 ** 3 or budget.telemetry.device_free_bytes < 3 * 1024 ** 3):
        raise ValueError('calibration telemetry or fixed commitments are inconsistent')
    _check_headroom(budget, recorded)
    _check_headroom(budget, budget.telemetry)


def measure_calibration(calibrator, key, write_stage):
    """Measure the fixed cold and warm callbacks under an already-owned device.

    The writer is responsible for durable private stage records. This helper
    does not construct live identity, launch or attest an owner, or approve a
    profile. It remains disabled until those independent components are ready.
    """
    if not CALIBRATION_EXECUTION_READY:
        raise RuntimeError('agreement calibration execution is disabled')
    from dataclasses import asdict, replace
    import re
    from .agreement_cuda_calibration import CudaAgreementCalibrator
    from .agreement_cuda_runner import (CALIBRATION_KINDS, WARMUP_POLICY_SHA256,
                                        _validate_workload_key, calibration_case_sha256)
    from .resource_profile import MemorySample

    if type(calibrator) is not CudaAgreementCalibrator:
        raise ValueError('a calibration owner is required')
    if _SAMPLING_CUSTODY[0] is not None:
        raise RuntimeError('the process has already attempted calibration sampling')
    # An otherwise-valid calibrator can already own CUDA resources. Root it
    # before validating any inputs so every subsequent failure is quarantined.
    _SAMPLING_CUSTODY[0] = calibrator
    kind = 'preflight'
    try:
        _SAMPLING_CUSTODY[1] = key
        samples = []
        _SAMPLING_CUSTODY[2] = samples
        if type(CALIBRATION_KINDS) is not tuple or CALIBRATION_KINDS != ('clean_capture', *tuple(_ACTIONS)):
            raise ValueError('the calibration action order differs from the fixed protocol')
        _validate_workload_key(key)
        if (calibrator._expected_key != key or calibrator.quarantined or calibrator._warmup_completed
                or type(calibrator._calibration_index) is not int or calibrator._calibration_index != 0
                or not callable(write_stage)):
            raise ValueError('a fresh, matching calibration owner and stage writer are required')
        cold_key = replace(key, warmup_state='cold')
        _require_claims(calibrator)
        for index, kind in enumerate(('warmup', *CALIBRATION_KINDS)):
            stage_key = cold_key if index == 0 else key
            case = WARMUP_POLICY_SHA256 if index == 0 else calibration_case_sha256(kind)
            calibrator._check_owner()
            write_stage({'phase': 'started', 'kind': kind, 'seed': 20261028, 'case_sha256': case})
            calibrator._check_owner()
            calibrator.synchronize_for_measurement()
            calibrator.reset_measurement_peak()
            calibrator.synchronize_for_measurement()
            before = calibrator.measurement_snapshot()
            calibrator._check_owner()
            write_stage({'phase': 'baseline', 'kind': kind, 'snapshot': asdict(before)})
            calibrator._check_owner()
            budget = _require_claims(calibrator).snapshot()
            calibrator._check_owner()
            write_stage({'phase': 'baseline_check', 'kind': kind, 'budget': asdict(budget)})
            _validate_headroom(key, before, budget)
            _require_claims(calibrator)
            calibrator._check_owner()
            result = calibrator.warmup() if index == 0 else calibrator.calibrate_action(kind)
            calibrator._check_owner()
            if index == 0:
                if (type(result) is not dict or set(result) != {'sample_id', 'policy_sha256', 'seed',
                        'model_state_sha256', 'plan_metadata'} or result['sample_id'] != 'warmup'
                        or result['policy_sha256'] != case or type(result['seed']) is not int
                        or result['seed'] != 20261028 or type(result['model_state_sha256']) is not str
                        or re.fullmatch(r'[0-9a-f]{64}', result['model_state_sha256']) is None
                        or calibrator._warmup_completed is not True):
                    raise ValueError('warmup returned inconsistent evidence')
            elif (type(result) is not dict or set(result) != {'sample_id', 'case_sha256', 'seed',
                    'plan_metadata', 'capture_or_landing_checked'} or result['sample_id'] != kind
                    or result['case_sha256'] != case or type(result['seed']) is not int
                    or result['seed'] != 20261028 or result['capture_or_landing_checked'] is not True
                    or type(calibrator._calibration_index) is not int or calibrator._calibration_index != index):
                raise ValueError('calibration action returned inconsistent evidence')
            validate_plan_metadata(kind, result['plan_metadata'])
            calibrator.synchronize_for_measurement()
            after = calibrator.measurement_snapshot()
            calibrator._check_owner()
            write_stage({'phase': 'after_observed', 'kind': kind, 'snapshot': asdict(after)})
            calibrator._check_owner()
            budget = _require_claims(calibrator).snapshot()
            calibrator._check_owner()
            write_stage({'phase': 'postcheck', 'kind': kind, 'budget': asdict(budget)})
            _validate_headroom(key, after, budget)
            _require_claims(calibrator)
            allocated, reserved = calibrator.measurement_peaks()
            calibrator._check_owner()
            sample = MemorySample(stage_key.fingerprint(), kind, case, 20261028, 'completed',
                                  before, after, allocated, reserved)
            if sample.before.device_total_bytes != key.device_total_bytes:
                raise ValueError('sample device total differs from the workload')
            write_stage({'phase': 'completed', 'kind': kind, 'sample': asdict(sample), 'callback': result})
            calibrator._check_owner()
            _require_claims(calibrator)
            samples.append(sample)
        return samples[0], tuple(samples[1:])
    except BaseException as error:
        _SAMPLING_CUSTODY[3] = error
        try:
            calibrator.quarantine_after_uncertain_callback()
        except BaseException:
            pass
        try:
            write_stage({'phase': 'failed', 'kind': kind, 'error_type': type(error).__name__})
        except BaseException:
            pass
        raise


def _same(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (actual.keys() == expected.keys()
                and all(_same(actual[key], value) for key, value in expected.items()))
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same(left, right) for left, right in zip(actual, expected))
    return actual == expected


def _event(name, operation, path, token, selector, *, whole, source, order, preceding):
    return {'name': name, 'operation': operation, 'order': order, 'module_path': path,
            'module_aliases': [path], 'call_site_known': False, 'forward_index': 0,
            'invocation': 0, 'batch_index': None if whole else 0,
            'token_index': None if whole else token, 'output_selector': selector,
            'output_shape': [1, 5, 768], 'selected_shape': [1, 5, 768] if whole else [768],
            'dtype': 'torch.float32', 'device': 'cuda:0', 'source_capture': source,
            'preceding_same_site_actions': list(preceding)}


def validate_plan_metadata(kind, metadata):
    """Reject actual plan records that do not match the fixed GPU case.

    This validates bounded metadata and logical payload counts. It cannot
    authenticate the owner or prove a resource bound.
    """
    if kind in ('warmup', 'clean_capture'):
        steps = [_event(f'capture_{i}', 'capture', path, token, selector, whole=False,
                        source=None, order=i, preceding=('capture_8',) if i == 9 else ())
                 for i, (path, token, selector) in enumerate(_SITES)]
        expected = {'steps': steps, 'capture_bytes': 30720, 'source_bytes': 0}
    elif type(kind) is str and kind in _ACTIONS:
        path, token, operation, source = _ACTIONS[kind]
        selector = [] if path.endswith('.mlp') else [0]
        specs = [('whole_before', 'capture', True, None), ('before', 'capture', False, None),
                 ('action', operation, False, source), ('after', 'capture', False, None),
                 ('whole_after', 'capture', True, None)]
        steps = [_event(name, action, path, token, selector, whole=whole, source=source_name,
                        order=i, preceding=tuple(item[0] for item in specs[:i]))
                 for i, (name, action, whole, source_name) in enumerate(specs)]
        expected = {'steps': steps, 'capture_bytes': 36864,
                    'source_bytes': 3072 if kind.endswith('cross_patch') else 0}
    else:
        raise ValueError('unknown calibration case')
    if not _same(metadata, expected):
        raise ValueError('actual calibration plan metadata differs from the fixed case')
