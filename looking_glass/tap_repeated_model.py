"""Disabled, source-verified fixed-model setup for the repeated tap owner.

Importing this module does not import Torch or load a checkpoint. The future
supervised process must establish its physical lease, runtime binding, memory
admission and allocator cap before enabling this single-use setup.
"""

import hashlib
import struct
import weakref

from . import tap_repeated_cuda_backend as backend
from . import tap_repeated_owner as owner
from . import tap_repeated_timing as timing
from .tap_repeated_protocol import GIB


_MODEL_SETUP_ENABLED = False
_TARGET_DEVICE = 'cuda:0'
_INPUT = (15496, 995)
_PARAMETERS = 81912576


def _verify_model(model, *, device):
    """Check the fixed architecture and the actually resident parameters."""
    import torch
    from transformers import GPT2LMHeadModel
    if type(model) is not GPT2LMHeadModel:
        raise ValueError('the fixed GPT-2 model class differs')
    config = model.config
    if (config.model_type != 'gpt2' or config.n_layer != 6 or config.n_embd != 768
            or config.n_head != 12 or config.vocab_size != 50257 or config.n_positions != 1024
            or config._attn_implementation != 'eager' or config.use_cache is not False
            or model.training is not False or any(module.training for module in model.modules())
            or model.get_input_embeddings().weight is not model.get_output_embeddings().weight):
        raise ValueError('the fixed GPT-2 configuration differs')
    count = 0
    for parameter in model.parameters():
        if str(parameter.device) != device or parameter.dtype != torch.float32:
            raise ValueError('model parameters differ from the fixed device or dtype')
        count += parameter.numel()
    if count != _PARAMETERS or any(str(buffer.device) != device for buffer in model.buffers()):
        raise ValueError('model parameter count or buffer device differs')


def _verify_bundle(bundle):
    from .pretrained import ASSETS, LoadedPretrained, MODEL_ID, REVISION
    if type(bundle) is not LoadedPretrained or type(bundle.metadata) is not dict:
        raise ValueError('the verified checkpoint bundle is missing')
    metadata = bundle.metadata
    if (metadata.get('model_id') != MODEL_ID or metadata.get('model_revision') != REVISION
            or metadata.get('tokenizer_id') != MODEL_ID or metadata.get('tokenizer_revision') != REVISION
            or metadata.get('model_class') != 'GPT2LMHeadModel'
            or metadata.get('weights_sha256') != ASSETS['model.safetensors'][1]
            or metadata.get('asset_sha256') != {name: digest for name, (_, digest) in ASSETS.items()}
            or metadata.get('device') != 'cpu' or metadata.get('dtype') != 'float32'
            or metadata.get('attention_implementation') != 'eager'
            or metadata.get('use_cache') is not False
            or type(metadata.get('parameter_count')) is not int or metadata['parameter_count'] != _PARAMETERS):
        raise ValueError('the pinned checkpoint metadata differs')
    _verify_model(bundle.model, device='cpu')
    return {'model_id': MODEL_ID, 'model_revision': REVISION,
            'weights_sha256': ASSETS['model.safetensors'][1],
            'asset_sha256': dict(metadata['asset_sha256']),
            'parameter_count': _PARAMETERS, 'attention_implementation': 'eager',
            'dtype': 'float32', 'use_cache': False,
            'input_ids': list(_INPUT),
            'input_ids_sha256_le_i64': hashlib.sha256(struct.pack('<2q', *_INPUT)).hexdigest()}


class _FixedModelSetup:
    """Load the fixed offline checkpoint and retain its live model and input."""

    def __init__(self, sample):
        if _MODEL_SETUP_ENABLED is not True:
            raise RuntimeError('repeated tap model setup remains disabled')
        if (type(sample) is not timing._TimedSample or sample.state != 'ready'
                or sample._preflight_verified is not False
                or type(sample.runtime) is not backend._CudaRuntime
                or sample.runtime.state != 'ready' or sample.runtime.device != _TARGET_DEVICE):
            raise ValueError('the exact ready CUDA timing owner is required')
        self.sample, self.operations = sample, sample.operations
        self.state, self.failure, self.partial = 'new', None, None

    def _run(self, cache_dir):
        operations, roots, sample = self.operations, self.operations.roots, self.sample

        def require_empty():
            if any(getattr(roots, name) is not None for name in owner._SLOTS
                   if name not in ('default_stream', 'start_event', 'end_event')):
                raise RuntimeError('model setup starts with occupied owner roots')

        def load():
            from .pretrained import load_distilgpt2
            return load_distilgpt2(cache_dir=cache_dir, allow_download=False)

        def transfer():
            import torch
            return roots.model.to(device=_TARGET_DEVICE, dtype=torch.float32)

        def verify_device_model():
            if roots.converted is not roots.model:
                raise RuntimeError('model transfer returned a different model')
            _verify_model(roots.model, device=_TARGET_DEVICE)

        def input_tensor():
            import torch
            return torch.tensor([list(_INPUT)], dtype=torch.int64, device=_TARGET_DEVICE)

        def input_values():
            import torch
            value = roots.input_ids
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 2)
                    or value.dtype != torch.int64 or str(value.device) != _TARGET_DEVICE):
                raise ValueError('fixed input tensor shape, dtype or device differs')
            return value.tolist()

        operations.call(require_empty)
        operations.call(load, root='model_bundle')
        identity = operations.call(lambda: _verify_bundle(roots.model_bundle))
        self.partial.update(stage='checkpoint_verified', model=identity)
        self.partial['memory_before_transfer'] = sample._snapshot()
        if self.partial['memory_before_transfer']['free_bytes'] < 3 * GIB:
            raise ValueError('at least 3 GiB must remain free immediately before model transfer')
        operations.call(lambda: roots.model_bundle.model, root='model')
        operations.call(transfer, root='converted')
        operations.call(verify_device_model)
        self.partial['memory_after_transfer'] = sample._snapshot()
        operations.call(input_tensor, root='input_ids')
        if operations.call(input_values) != [list(_INPUT)]:
            raise ValueError('fixed input token IDs differ')
        sample._none_call(sample.runtime.synchronize)
        reference = operations.call(lambda: weakref.ref(roots.model_bundle))
        operations.clear_after_synchronization('model_bundle')

        def require_released():
            if reference() is not None:
                raise RuntimeError('checkpoint bundle remains reachable')

        operations.call(require_released)
        sample._none_call(sample.runtime.synchronize)
        self.partial['memory_after_input'] = sample._snapshot()
        self.partial['stage'] = 'complete'
        return {**identity, 'device': _TARGET_DEVICE}

    def run(self, cache_dir):
        if self.state != 'new':
            raise RuntimeError('model setup is single-use')
        kind = None
        try:
            self.state = 'running'
            self.partial = {'stage': 'setup'}
            result = self._run(cache_dir)
            self.state = 'complete'
            return result
        except BaseException as exc:
            self.failure = exc
            self.operations.unsafe = True
            if self.operations.roots.failure is None:
                self.operations.roots.failure = exc
            self.state = 'failed'
            kind = owner._failure_kind(exc)
        owner._raise_detached(kind)
