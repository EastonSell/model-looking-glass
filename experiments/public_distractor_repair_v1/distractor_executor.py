"""Fixed bounded two-position action schedule for the public study."""
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path
import torch
from distractor_core import (bitwise_equal, candidate_map, check_landing, ensure_round_to_nearest,
                             intervene, null_self, observe_clean)
from distractor_journal import Journal


class ControlFailure(RuntimeError):
    """A saved required control failed; no later action is allowed."""


def _tensor_bytes(tensor):
    if tensor.dtype != torch.float32 or tensor.device.type != 'cpu':
        raise ValueError('expected CPU float32 tensor')
    return tensor.detach().contiguous().numpy().astype('<f4', copy=False).tobytes()


def _hook_free(model):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


_COUNTS = {'clean': (0, 10), 'null-self': (2, 1), 'decoy': (2, 10),
           'target': (2, 10), 'combined': (4, 10), 'patch': (6, 12),
           'rotation': (6, 11), 'self': (6, 11), 'clean-repeat': (0, 10)}
_SCHEDULE = ('clean', 'null-self', 'decoy') + (('target', 'combined') + ('patch',)*4
             + ('rotation', 'self'))*3 + ('clean-repeat',)


def _bound(kind, hidden, vocab):
    full, vectors = _COUNTS[kind]
    return 4 * (2*vocab + full*32*hidden + vectors*hidden)


def execute_phase(model, cases, directory, *, phase, vocab_size=50257,
                  raw_cap=256*1024*1024, metadata_cap=16*1024*1024,
                  minimum_free_bytes=1024*1024*1024, additional_space_checks=()):
    if phase not in ('discovery', 'held-out') or not cases or model.training or not _hook_free(model):
        raise ValueError('invalid phase, cases, or evaluation model')
    if metadata_cap <= 65536:
        raise ValueError('metadata cap must reserve failure space')
    ensure_round_to_nearest()
    hidden = model.config.n_embd
    writer = Journal(directory, raw_cap=raw_cap, metadata_cap=metadata_cap-65536)
    ordinal = 0
    last_result = None
    partial = {'attempted': False, 'saved': None, 'write_error': None}

    def save_partial(logits, error):
        partial['attempted'] = True
        try:
            data = _tensor_bytes(logits)
            if len(data) != 2*vocab_size*4:
                raise ValueError('partial logits outside fixed vocabulary bound')
            file = Path(directory)/'partial-return.f32'
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
        required_raw = _bound(kind, hidden, vocab_size) + 2*vocab_size*4
        if writer.raw_bytes + required_raw > writer.raw_cap:
            raise ValueError('insufficient raw budget before model forward')
        if writer.metadata_bytes + 48*1024 > writer.metadata_cap:
            raise ValueError('insufficient metadata budget before model forward')
        if shutil.disk_usage(Path(directory)).free < minimum_free_bytes + required_raw + 48*1024:
            raise ValueError('insufficient disk reserve before model forward')
        for path, minimum in additional_space_checks:
            if shutil.disk_usage(path).free < minimum:
                raise ValueError('insufficient auxiliary volume reserve before model forward')
        ensure_round_to_nearest()
        writer.start(ordinal, case_id, kind, details)
        last_result = None
        if not _hook_free(model):
            raise RuntimeError('unexpected pre-existing hook')
        last_result = operation(save_partial)
        if not _hook_free(model):
            raise RuntimeError('hooks remain after completed action')
        logits = last_result['logits']
        captures = last_result.get('captures')
        if captures is None:
            captures = {f'tap_{layer}_{pos}': vector
                        for (layer,pos),vector in last_result['vectors'].items()}
        tensors = {'logits': logits, **captures}
        descriptors = [writer.record(name, _tensor_bytes(tensor), tuple(tensor.shape))
                       for name,tensor in tensors.items()]
        displacement_norm = None
        if (kind == 'patch' and candidate_map(details['layer'])[details['candidate']]['role'] == 'downstream'):
            ensure_round_to_nearest()
            index=tokens.shape[1]-2
            before=_tensor_bytes(captures['patch_before'][0,index])
            after=_tensor_bytes(captures['patch_after'][0,index])
            displacement=[a[0]-b[0] for a,b in zip(struct.iter_unpack('<f',after),struct.iter_unpack('<f',before))]
            if len(displacement)!=hidden or not all(math.isfinite(x) for x in displacement):
                raise ValueError('invalid realized downstream displacement')
            data=b''.join(struct.pack('<d',x) for x in displacement)
            displacement_norm=math.sqrt(math.fsum(x*x for x in displacement))
            descriptors.append(writer.record64('realized_displacement',data,(hidden,)))
            ensure_round_to_nearest()
        # Record witnesses and hook order before checking the controls so even
        # the evidence of a failing action is retained.
        writer._event({'type': 'raw-recorded', **writer.current, 'tensors': descriptors,
                       'hook_events': last_result['events'],
                       'rotation_info': last_result.get('rotation_info'),
                       'displacement_norm': displacement_norm})
        if tuple(logits.shape) != (2, vocab_size) or not all(bool(torch.isfinite(x).all()) for x in tensors.values()):
            raise ValueError('nonfinite or wrong-sized recorded tensor')
        checks = validator(last_result)
        if not checks or any(type(value) is not bool for value in checks.values()):
            raise ValueError('required checks must be boolean')
        if not all(checks.values()):
            writer._event({'type': 'failed-controls', **writer.current, 'checks': checks})
            raise ControlFailure('a recorded required control failed')
        writer._event({'type': 'complete', **writer.current, 'checks': checks})
        writer.current = None
        ordinal += 1
        complete = last_result
        last_result = None
        return complete

    try:
        phase_bound = len(cases)*sum(_bound(kind, hidden, vocab_size) for kind in _SCHEDULE) + 2*vocab_size*4
        if phase_bound > writer.raw_cap or len(cases)*len(_SCHEDULE)*48*1024 > writer.metadata_cap:
            raise ValueError('fixed 32-token schedule cannot fit declared phase caps')
        with torch.inference_mode():
            for case in cases:
                case_id = case['case_id']
                if case['split'] != phase:
                    raise ValueError('case belongs to another phase')
                tokens = torch.tensor([case['token_ids']], dtype=torch.long, device='cpu')
                if not 3 <= tokens.shape[1] <= 32:
                    raise ValueError('case token length outside bound')
                t = tokens.shape[1]-2
                clean = action(case_id,'clean',{},
                    lambda cb: observe_clean(model,tokens,f'{phase}-{case_id}-clean',cb),
                    lambda result: {'trace_complete': True})
                vectors = clean['vectors']
                action(case_id,'null-self',{}, lambda cb: null_self(model,tokens,cb),
                    lambda result: {'before_after_equal': bitwise_equal(result['captures']['before'],result['captures']['after']),
                                    'self_source_matches': bitwise_equal(result['captures']['self_source'],result['captures']['before'][0,t]),
                                    'logits_equal_clean': bitwise_equal(result['logits'],clean['logits'])})
                def validate_decoy(result):
                    check_landing(result,tokens,vectors,1,'decoy')
                    return {'landing_valid': True, 'scored_logits_equal_clean': bitwise_equal(result['logits'][0],clean['logits'][0])}
                action(case_id,'decoy',{},lambda cb: intervene(model,tokens,vectors,1,'decoy',on_partial=cb),validate_decoy)
                for layer in (1,2,3):
                    def validate_target(result,layer=layer):
                        check_landing(result,tokens,vectors,layer,'target')
                        return {'landing_valid': True}
                    target=action(case_id,'target',{'layer':layer},
                        lambda cb,layer=layer: intervene(model,tokens,vectors,layer,'target',on_partial=cb),validate_target)
                    def validate_combined(result,layer=layer,target=target):
                        check_landing(result,tokens,vectors,layer,'combined')
                        return {'landing_valid': True, 'scored_logits_equal_target': bitwise_equal(result['logits'][0],target['logits'][0])}
                    combined=action(case_id,'combined',{'layer':layer},
                        lambda cb,layer=layer: intervene(model,tokens,vectors,layer,'combined',on_partial=cb),validate_combined)
                    for q,item in candidate_map(layer).items():
                        def validate_patch(result,layer=layer,q=q,item=item,combined=combined):
                            check_landing(result,tokens,vectors,layer,'patch',q=q)
                            checks={'landing_valid': True}
                            if item['role']=='source':
                                checks['scored_logits_equal_clean']=bitwise_equal(result['logits'][0],clean['logits'][0])
                            if item['role'] in ('upstream','distractor'):
                                checks['scored_logits_equal_combined']=bitwise_equal(result['logits'][0],combined['logits'][0])
                            if item['role']=='upstream':
                                checks['upstream_donor_equal']=bitwise_equal(result['captures']['patch_before'][0,t],vectors[(item['layer'],'t')])
                            if item['role']=='downstream':
                                checks['downstream_base_equal_combined']=bitwise_equal(
                                    result['captures']['patch_before'][0,t],combined['captures'][f'tap_{layer+1}_t'])
                            return checks
                        action(case_id,'patch',{'layer':layer,'candidate':q},
                            lambda cb,layer=layer,q=q: intervene(model,tokens,vectors,layer,'patch',q=q,on_partial=cb),validate_patch)
                    base=combined['captures'][f'tap_{layer+1}_t']
                    def validate_rotation(result,layer=layer,base=base):
                        check_landing(result,tokens,vectors,layer,'rotation',base=base)
                        return {'landing_valid': True}
                    action(case_id,'rotation',{'layer':layer},
                        lambda cb,layer=layer,base=base: intervene(model,tokens,vectors,layer,'rotation',base=base,on_partial=cb),validate_rotation)
                    def validate_self(result,layer=layer,combined=combined):
                        check_landing(result,tokens,vectors,layer,'self')
                        return {'landing_valid': True,
                                'self_source_matches': bitwise_equal(result['captures']['self_source'],result['captures']['self_before'][0,t]),
                                'logits_equal_combined': bitwise_equal(result['logits'],combined['logits'])}
                    action(case_id,'self',{'layer':layer},
                        lambda cb,layer=layer: intervene(model,tokens,vectors,layer,'self',on_partial=cb),validate_self)
                action(case_id,'clean-repeat',{},
                    lambda cb: observe_clean(model,tokens,f'{phase}-{case_id}-repeat',cb),
                    lambda result: {'logits_equal_clean': bitwise_equal(result['logits'],clean['logits']),
                        'vectors_equal_clean': all(bitwise_equal(result['vectors'][key],vectors[key]) for key in vectors)})
        summary={'phase':phase,'cases':len(cases),'completed_actions':ordinal,'raw_bytes':writer.raw_bytes,
                 'metadata_bytes':writer.metadata_bytes,'status':'complete'}
        with (Path(directory)/'phase-summary.json').open('xb') as out:
            out.write((json.dumps(summary,sort_keys=True,indent=2)+'\n').encode('utf-8'))
        return summary
    except BaseException as error:
        if last_result is not None and not partial['attempted']:
            try:
                save_partial(last_result['logits'],error)
            except BaseException:
                pass
        try:
            writer.fail(error,partial=partial['saved'],partial_error=partial['write_error'])
        except BaseException as failure_error:
            if hasattr(error,'add_note'):
                error.add_note('failure metadata could not be saved: '+type(failure_error).__name__)
        raise
    finally:
        writer.close()
