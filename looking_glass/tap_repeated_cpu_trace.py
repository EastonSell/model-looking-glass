"""CPU-only real GPT-2 tap preflight for the repeated-runner development.

This helper exercises actual direct/gradient GPT2TraceSession hooks on a
caller-supplied CPU model. It is not a pretrained-model or GPU benchmark.
"""

import torch

from .tap_repeated_protocol import IDENTITY, PROTOCOL
from .transformer_trace import GPT2TraceSession


def cpu_trace_preflight(model, input_ids, *, session):
    """Compare real CPU hook outputs with torch.equal without timing them."""
    if type(session) is not int or session not in range(5):
        raise ValueError('session must be an integer from 0 through 4')
    if (not isinstance(input_ids, torch.Tensor) or input_ids.device.type != 'cpu'
            or input_ids.dtype != torch.int64 or tuple(input_ids.shape) != (1, 2)
            or input_ids.tolist() != [[15496, 995]]):
        raise ValueError('expected the fixed CPU token IDs')
    if (model.training or getattr(model.config, 'model_type', None) != 'gpt2'
            or getattr(model.config, '_attn_implementation', None) != 'eager'
            or any(value.device.type != 'cpu' for value in list(model.parameters()) + list(model.buffers()))):
        raise ValueError('expected a CPU, eager, evaluation-mode GPT-2 model')
    run_id = f'{PROTOCOL}-session-{session}'
    outputs, sessions = [], {}
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        baseline = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
        outputs.append(baseline)
        for mode in ('id', 'direct', 'gradient'):
            with GPT2TraceSession(model, run_id=run_id, mode=mode, targets=('block',), layers=(0,),
                                  samples=(0,), tokens=(1,), capture_device='source',
                                  max_events=2, max_trace_bytes=64 * 1024) as trace:
                output = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
            outputs.append(output)
            sessions[mode] = trace
    expected_output = (1, 2, model.config.vocab_size)
    if any(tuple(value.shape) != expected_output or value.dtype != torch.float32
           or value.device.type != 'cpu' or not bool(torch.isfinite(value).all()) for value in outputs):
        raise ValueError('CPU preflight output is invalid')
    if not all(torch.equal(baseline, value) for value in outputs[1:]):
        raise ValueError('CPU preflight model outputs differ')
    for mode, trace in sessions.items():
        if not trace.successful or len(trace.events) != 1:
            raise ValueError('CPU preflight trace did not produce exactly one event')
        event = trace.events[0]
        identity = [event.module_path, event.module_kind, event.layer_index, event.forward_index,
                    event.invocation, list(event.output_path), list(event.batch_positions), list(event.token_positions)]
        if (identity != IDENTITY or type(event.sequence) is not int or event.sequence != 0
                or event.run_id != run_id or event.capture_mode != mode):
            raise ValueError('CPU preflight trace attribution differs')
    id_value = sessions['id'].events[0].value
    direct, gradient = (sessions[mode].events[0].value for mode in ('direct', 'gradient'))
    hidden = model.config.n_embd
    expected_bytes = hidden * 4
    if (id_value is not None or not isinstance(direct, torch.Tensor) or not isinstance(gradient, torch.Tensor)
            or tuple(direct.shape) != (1, 1, hidden) or tuple(gradient.shape) != (1, 1, hidden)
            or direct.dtype != torch.float32 or gradient.dtype != torch.float32
            or direct.device.type != 'cpu' or gradient.device.type != 'cpu'
            or sessions['direct'].tensor_bytes != expected_bytes or sessions['gradient'].tensor_bytes != expected_bytes
            or not bool(torch.isfinite(direct).all()) or not bool(torch.isfinite(gradient).all())
            or not torch.equal(direct, gradient)):
        raise ValueError('CPU preflight captures differ or have invalid layouts')
    return {'cpu_only': True, 'gpu_eligible': False, 'pretrained_identity_verified': False,
            'run_id': run_id, 'modes': ['none', 'id', 'direct', 'gradient'],
            'event_identity': IDENTITY, 'captured_shape': [1, 1, hidden],
            'captured_bytes': expected_bytes, 'output_shape': list(expected_output),
            'outputs_equal': True, 'captures_equal': True, 'comparison': 'torch.equal',
            'guarantees_bitwise_identity': False,
            'matches_protocol_layout': hidden == 768 and model.config.vocab_size == 50257}
