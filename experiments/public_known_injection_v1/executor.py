"""Bounded two-site action schedule; model and public cases are supplied by the caller."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

from hook_core import (SITES, bitwise_equal, check_landing, observe_clean,
                       perform_action, perform_null_self)
from journal import Journal


class ControlFailure(RuntimeError):
    """A retained required control failed; the attempt is incomplete."""


def _tensor_bytes(tensor):
    if tensor.dtype != torch.float32 or tensor.device.type != 'cpu':
        raise ValueError('expected CPU float32 tensor')
    return tensor.detach().contiguous().numpy().astype('<f4', copy=False).tobytes()


def _hook_free(model):
    return all(not module._forward_hooks and not module._forward_pre_hooks
               for module in model.modules())


def _finite(result):
    return all(bool(torch.isfinite(tensor).all()) for tensor in result.values())


def execute_phase(model, cases, directory, *, phase, vocab_size=50257,
                  raw_cap=75 * 1024 * 1024, metadata_cap=8 * 1024 * 1024,
                  minimum_free_bytes=1024 * 1024 * 1024, additional_space_checks=()):
    if phase not in ('discovery', 'held-out') or not cases:
        raise ValueError('invalid phase')
    if metadata_cap <= 65536:
        raise ValueError('metadata cap must reserve fixed failure/summary space')
    writer = Journal(directory, raw_cap=raw_cap, metadata_cap=metadata_cap - 65536)
    ordinal = 0
    last_result = None
    partial = {'attempted': False, 'saved': None, 'write_error': None}

    def save_partial(logits, error):
        partial['attempted'] = True
        try:
            data = _tensor_bytes(logits)
            if len(data) != vocab_size * 4:
                raise ValueError('partial logits outside fixed vocabulary bound')
            file = Path(directory) / 'partial-return.f32'
            with file.open('xb') as stream:
                stream.write(data)
                stream.flush()
            writer.raw_bytes += len(data)
            partial['saved'] = {'file': file.name, 'length': len(data),
                                'sha256': hashlib.sha256(data).hexdigest(),
                                'trigger_error': type(error).__name__}
        except BaseException as save_error:
            partial['write_error'] = type(save_error).__name__
            raise

    def action(case_id, kind, details, operation, validator):
        nonlocal ordinal, last_result
        token_count, hidden = tokens.shape[1], model.config.n_embd
        full_count = {'clean': 0, 'null-self': 2, 'injected': 2, 'patch': 4,
                      'injected-self': 4, 'clean-repeat': 0}[kind]
        vector_count = {'clean': 4, 'null-self': 1, 'injected': 4, 'patch': 4,
                        'injected-self': 5, 'clean-repeat': 4}[kind]
        action_bound = 4 * (vocab_size + full_count * token_count * hidden + vector_count * hidden)
        # The additional vocabulary vector reserves a separately saved partial
        # return if persistence fails after the model has returned.
        required_raw = action_bound + 4 * vocab_size
        if writer.raw_bytes + required_raw > writer.raw_cap:
            raise ValueError('insufficient raw budget before model forward')
        if writer.metadata_bytes + 48 * 1024 > writer.metadata_cap:
            raise ValueError('insufficient metadata budget before model forward')
        if shutil.disk_usage(Path(directory)).free < minimum_free_bytes + required_raw + 48 * 1024:
            raise ValueError('insufficient disk reserve before model forward')
        for path, minimum in additional_space_checks:
            if shutil.disk_usage(path).free < minimum:
                raise ValueError('insufficient auxiliary volume reserve before model forward')
        writer.start(ordinal, case_id, kind, details)
        last_result = None
        if not _hook_free(model):
            raise RuntimeError('unexpected pre-existing hook')
        last_result = operation(save_partial)
        if not _hook_free(model):
            raise RuntimeError('hooks remain after completed action')
        logits = last_result['logits']
        captures = last_result.get('captures', {f'tap_{i}': v for i, v in last_result.get('vectors', {}).items()})
        tensors = {'logits': logits, **captures}
        descriptors = []
        for name, tensor in tensors.items():
            descriptors.append(writer.record(name, _tensor_bytes(tensor), tuple(tensor.shape)))
        if tuple(logits.shape) != (vocab_size,) or not _finite(tensors):
            raise ValueError('nonfinite or wrong-sized recorded tensor')
        checks = validator(last_result)
        writer.complete(descriptors, last_result['events'], checks)
        ordinal += 1
        completed = last_result
        last_result = None
        if any(value is False for value in checks.values()):
            raise ControlFailure('a recorded required bitwise control failed')
        return completed

    try:
        if model.training or not _hook_free(model):
            raise ValueError('model must be an unhooked evaluation model')
        with torch.inference_mode():
            for case in cases:
                case_id = case['case_id']
                if case['split'] != phase:
                    raise ValueError('case belongs to another phase')
                tokens = torch.tensor([case['token_ids']], dtype=torch.long, device='cpu')
                if not 1 <= tokens.shape[1] <= 32:
                    raise ValueError('case token length outside bound')
                clean = action(case_id, 'clean', {},
                    lambda partial_cb: observe_clean(model, tokens, f'{phase}-{case_id}-clean', partial_cb),
                    lambda result: {'trace_valid': True})
                vectors = clean['vectors']
                null = action(case_id, 'null-self', {},
                    lambda partial_cb: perform_null_self(model, tokens, partial_cb),
                    lambda result: {'before_after_equal': bitwise_equal(result['captures']['before'], result['captures']['after']),
                                    'logits_equal_clean': bitwise_equal(result['logits'], clean['logits'])})
                for cell, site in (('A', 1), ('B', 4)):
                    injected = action(case_id, 'injected', {'cell': cell, 'site': site},
                        lambda partial_cb, site=site: perform_action(model, tokens, vectors, site, 'injected', on_partial=partial_cb),
                        lambda result: _validate_injected(result))
                    for candidate in SITES:
                        def validate_patch(result, site=site, candidate=candidate, injected=injected):
                            check_landing(result, 'patch', vectors, candidate)
                            checks = {'landing_valid': True, 'relationship': ('upstream' if candidate < site else
                                     'same' if candidate == site else 'downstream')}
                            if candidate < site:
                                checks['upstream_donor_equal'] = bitwise_equal(result['captures']['patch_before'][0, -1], vectors[candidate])
                                checks['upstream_logits_equal_injected'] = bitwise_equal(result['logits'], injected['logits'])
                            if candidate == site:
                                checks['same_site_logits_equal_clean'] = bitwise_equal(result['logits'], clean['logits'])
                            return checks
                        action(case_id, 'patch', {'cell': cell, 'site': site, 'candidate': candidate},
                            lambda partial_cb, site=site, candidate=candidate: perform_action(
                                model, tokens, vectors, site, 'patch', candidate, partial_cb), validate_patch)
                    action(case_id, 'injected-self', {'cell': cell, 'site': site},
                        lambda partial_cb, site=site: perform_action(model, tokens, vectors, site, 'self', on_partial=partial_cb),
                        lambda result, injected=injected: _validate_self(result, injected))
                action(case_id, 'clean-repeat', {},
                    lambda partial_cb: observe_clean(model, tokens, f'{phase}-{case_id}-repeat', partial_cb),
                    lambda result: {'logits_equal_clean': bitwise_equal(result['logits'], clean['logits']),
                                    'vectors_equal_clean': all(bitwise_equal(result['vectors'][i], vectors[i]) for i in SITES)})
        summary = {'phase': phase, 'cases': len(cases), 'completed_actions': ordinal,
                   'raw_bytes': writer.raw_bytes, 'metadata_bytes': writer.metadata_bytes,
                   'status': 'complete'}
        with (Path(directory) / 'phase-summary.json').open('xb') as out:
            out.write((json.dumps(summary, sort_keys=True, indent=2) + '\n').encode('utf-8'))
        return summary
    except BaseException as error:
        # If the helper returned but persistence failed, retain bounded logits
        # in a separate file where possible while preserving the first error.
        if last_result is not None and not partial['attempted']:
            try:
                save_partial(last_result['logits'], error)
            except BaseException:
                pass
        try:
            writer.fail(error, partial=partial['saved'], partial_error=partial['write_error'])
        except BaseException as failure_error:
            if hasattr(error, 'add_note'):
                error.add_note('failure metadata could not be saved: ' + type(failure_error).__name__)
        raise
    finally:
        writer.close()


def _validate_injected(result):
    check_landing(result, 'injected')
    return {'landing_valid': True}


def _validate_self(result, injected):
    check_landing(result, 'self')
    return {'landing_valid': True, 'logits_equal_injected': bitwise_equal(result['logits'], injected['logits'])}
