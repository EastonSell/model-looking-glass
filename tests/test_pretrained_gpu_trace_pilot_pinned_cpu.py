"""Optional offline fixed-checkpoint CPU composition; never initializes CUDA."""

import gc
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get('LG_PINNED_CACHE'), reason='offline checkpoint not supplied')
def test_pinned_checkpoint_ten_forward_composition_and_disposal_without_gc(tmp_path):
    import torch
    from looking_glass.pretrained import load_distilgpt2
    from looking_glass.pretrained_gpu_trace_pilot import _verify_model, _weakrefs_for_disposal
    from looking_glass.pretrained_gpu_trace_pilot_core import TOKENS, run_fixed_trace
    from looking_glass.pretrained_gpu_trace_pilot_runtime import PilotExecutor

    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    was_enabled, old_collect = gc.isenabled(), gc.collect
    gc.disable()
    def forbidden(*_args, **_kwargs):
        raise AssertionError('cyclic collection is forbidden during custody')
    gc.collect = forbidden
    try:
        bundle = load_distilgpt2(cache_dir=Path(os.environ['LG_PINNED_CACHE']),
                                 stage_dir=tmp_path, allow_download=False)
        model = bundle.model
        _verify_model(model, torch, device='cpu', metadata=bundle.metadata)
        ids = torch.tensor([TOKENS], dtype=torch.int64)
        mask = torch.ones_like(ids)
        gib = 1024 ** 3
        memory = {'free_bytes': 7 * gib, 'total_bytes': 8 * gib, 'allocated_bytes': 0,
                  'reserved_bytes': 0, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
        roots = {'model': model, 'bundle': bundle}
        executor = PilotExecutor(torch=torch, check=lambda: None, synchronize=lambda: None,
            snapshot=lambda: memory.copy(), reset_peaks=lambda: None, directory=tmp_path, roots=roots,
            device='cpu', vocab_size=50257)
        executor.begin('load')
        executor.end('load', {'model_loaded': True})
        executor.verify_inputs(ids, mask)
        result = run_fixed_trace(model, ids, mask, executor, study_id='pinned-cpu-test')
        assert executor.index == 7 and executor.forward_count == 10 and result['direct_gradient_equal'] is True
        assert all(result['modes'][mode]['tensor_bytes'] == 110592 for mode in ('direct', 'gradient'))
        refs = _weakrefs_for_disposal(roots)
        roots['reference'] = roots['reference_lease'] = roots['reference_handles'] = None
        roots['traces'].clear()
        roots['quarantined_events'].clear()
        roots['cpu_logits'] = roots['input_ids'] = roots['attention_mask'] = None
        roots['model'] = roots['bundle'] = None
        del bundle, model, ids, mask
        assert all(ref() is None for ref in refs)
        assert not torch.cuda.is_initialized()
    finally:
        gc.collect = old_collect
        if was_enabled:
            gc.enable()
