"""Bounded phase executor for the one-shot pretrained trace pilot.

Callbacks are supplied by the supervised owner. This module alone cannot
establish CUDA ownership, identity or safety.
"""

import hashlib
from pathlib import Path

from .bert_gpu_evidence import write_json_new

MIB = 1024 ** 2
PHASE_NAMES = ('load', 'baseline_cold', 'baseline_warm', 'reference', 'direct', 'gradient', 'id')
OUTPUT_BYTES = 1 * 5 * 50257 * 4
TRACE_BYTES = 131072
INPUT_BYTES = 2 * 5 * 8


def phase_estimate(name, before, after):
    if name not in PHASE_NAMES:
        raise ValueError('unknown fixed phase')
    for row in (before, after):
        if (type(row) is not dict or any(type(row.get(k)) is not int or row[k] < 0 for k in
                ('free_bytes', 'total_bytes', 'allocated_bytes', 'reserved_bytes',
                 'peak_allocated_bytes', 'peak_reserved_bytes'))):
            raise ValueError('invalid phase memory observation')
        if (row['total_bytes'] == 0 or row['free_bytes'] > row['total_bytes']
                or row['allocated_bytes'] > row['reserved_bytes']
                or row['peak_allocated_bytes'] < row['allocated_bytes']
                or row['peak_reserved_bytes'] < max(row['reserved_bytes'], row['peak_allocated_bytes'])):
            raise ValueError('inconsistent phase memory observation')
    if before['total_bytes'] != after['total_bytes']:
        raise ValueError('device total changed')
    declared = 0 if name == 'load' else INPUT_BYTES + OUTPUT_BYTES
    if name in ('reference', 'direct', 'gradient'):
        declared += TRACE_BYTES
    observed = max(0, after['peak_allocated_bytes'] - before['allocated_bytes'],
                   after['peak_reserved_bytes'] - before['reserved_bytes'],
                   before['free_bytes'] - after['free_bytes'])
    basis = declared + observed
    return {'declared_bytes': declared, 'observed_increment_bytes': observed,
            'basis_bytes': basis, 'margin_bytes': 32 * MIB + (basis + 9) // 10,
            'estimated_incremental_bytes': basis + 32 * MIB + (basis + 9) // 10}


class RetainingEvents(list):
    """Keep partial adapter events rooted if its failure path clears them."""

    def __init__(self, custody):
        super().__init__()
        self.custody = custody

    def clear(self):
        self.custody.extend(self)
        super().clear()


class PilotExecutor:
    def __init__(self, *, torch, check, synchronize, snapshot, reset_peaks, directory, roots,
                 device='cuda:0', vocab_size=50257, load_limit=512 * MIB, job_limit=1024 * MIB):
        self.torch, self.check, self.synchronize = torch, check, synchronize
        self.snapshot, self.reset_peaks, self.directory, self.roots = snapshot, reset_peaks, Path(directory), roots
        self.device, self.vocab_size = device, vocab_size
        self.limits = (load_limit, job_limit)
        self.index, self.current, self.before, self.forward_count = 0, None, None, 0
        self.phases = []
        self.roots['reference'] = None
        self.roots['reference_lease'] = None
        self.roots['reference_handles'] = None
        self.roots['traces'] = {}
        self.roots['quarantined_events'] = []
        self.roots['output'] = self.roots['device_logits'] = self.roots['cpu_logits'] = None
        self.roots['input_ids'] = self.roots['attention_mask'] = None
        self.roots['input_check_ids'] = self.roots['input_check_mask'] = None
        self.roots['pending'] = False

    def bind_reference(self, reference):
        if self.roots['reference'] is not None:
            raise RuntimeError('reference custody is already bound')
        self.roots['reference'] = reference

    def bind_reference_lease(self, lease, handles):
        self.roots['reference_lease'], self.roots['reference_handles'] = lease, handles

    def bind_trace(self, mode, trace):
        if mode not in ('direct', 'gradient', 'id') or mode in self.roots['traces'] or trace.events:
            raise RuntimeError('invalid trace custody transition')
        trace.events = RetainingEvents(self.roots['quarantined_events'])
        self.roots['traces'][mode] = trace

    def verify_inputs(self, input_ids, attention_mask):
        if (self.roots['pending'] or str(input_ids.device) != self.device
                or str(attention_mask.device) != self.device
                or input_ids.dtype != self.torch.int64 or attention_mask.dtype != self.torch.int64
                or tuple(input_ids.shape) != (1, 5) or tuple(attention_mask.shape) != (1, 5)):
            raise ValueError('fixed input shape, dtype or device differs')
        if self.roots['input_ids'] is None:
            self.roots['input_ids'], self.roots['attention_mask'] = input_ids, attention_mask
        elif self.roots['input_ids'] is not input_ids or self.roots['attention_mask'] is not attention_mask:
            raise RuntimeError('fixed input custody changed')
        self.check()
        self.roots['pending'] = True
        self.roots['input_check_ids'] = input_ids.detach().to('cpu', copy=True)
        self.check()
        self.synchronize()
        self.check()
        if self.roots['input_check_ids'].tolist() != [[464, 6766, 318, 4171, 13]]:
            raise ValueError('fixed token IDs differ')
        self.roots['input_check_mask'] = attention_mask.detach().to('cpu', copy=True)
        self.check()
        self.synchronize()
        self.check()
        if self.roots['input_check_mask'].tolist() != [[1, 1, 1, 1, 1]]:
            raise ValueError('fixed attention mask differs')
        self.roots['input_check_ids'] = self.roots['input_check_mask'] = None
        self.roots['pending'] = False

    def begin(self, phase):
        if (self.current is not None or self.index >= len(PHASE_NAMES)
                or phase != PHASE_NAMES[self.index] or self.roots['pending']):
            raise RuntimeError('phase start order differs')
        if phase != 'load' and self.roots['input_ids'] is not None:
            self.verify_inputs(self.roots['input_ids'], self.roots['attention_mask'])
        self.check()
        self.synchronize()
        self.before = self.snapshot()
        self.check()
        self.reset_peaks()
        self.check()
        self.current = phase
        write_json_new(self.directory / f'phase-{self.index + 1:02d}-start.json',
                       {'schema': 'pretrained-gpu-trace-phase-start-v1', 'phase': phase,
                        'before': self.before})

    def forward(self, phase, invoke):
        if self.current != phase or phase == 'load' or self.roots['pending']:
            raise RuntimeError('forward is outside its phase')
        if self.roots['input_ids'] is not None:
            self.verify_inputs(self.roots['input_ids'], self.roots['attention_mask'])
        self.check()
        self.roots['pending'] = True
        output = invoke()
        self.roots['output'] = output
        self.check()
        self.synchronize()
        self.check()
        logits = output.logits
        self.roots['device_logits'] = logits
        if (getattr(output, 'past_key_values', None) is not None
                or getattr(output, 'hidden_states', None) is not None
                or getattr(output, 'attentions', None) is not None
                or not isinstance(logits, self.torch.Tensor)
                or str(logits.device) != self.device or logits.dtype != self.torch.float32
                or tuple(logits.shape) != (1, 5, self.vocab_size) or not logits.is_contiguous()
                or logits.numel() * logits.element_size() != 5 * self.vocab_size * 4):
            raise ValueError('model output differs from the fixed no-cache contract')
        self.check()
        copied = logits.detach().to('cpu', copy=True)
        self.roots['cpu_logits'] = copied
        self.check()
        self.synchronize()
        self.check()
        if copied.device.type != 'cpu' or not bool(self.torch.isfinite(copied).all()):
            raise ValueError('offloaded logits are invalid')
        self.roots['pending'] = False
        if self.roots['input_ids'] is not None:
            self.verify_inputs(self.roots['input_ids'], self.roots['attention_mask'])
        digest = hashlib.sha256(copied.contiguous().numpy().astype('<f4', copy=False).tobytes()).hexdigest()
        next_index = self.forward_count + 1
        write_json_new(self.directory / f'forward-{next_index:03d}.json',
                       {'schema': 'pretrained-gpu-trace-forward-v1', 'phase': phase,
                        'index': next_index, 'logit_sha256': digest,
                        'shape': [1, 5, self.vocab_size], 'dtype': 'float32'})
        self.forward_count = next_index
        self.check()
        self.roots['device_logits'] = None
        self.roots['output'] = None
        self.roots['pending'] = False
        return copied

    def end(self, phase, proof):
        if self.current != phase or self.roots['pending'] or type(proof) is not dict:
            raise RuntimeError('phase completion is uncertain')
        if phase != 'load' and self.roots['input_ids'] is not None:
            self.verify_inputs(self.roots['input_ids'], self.roots['attention_mask'])
        self.check()
        self.synchronize()
        after = self.snapshot()
        estimate = phase_estimate(phase, self.before, after)
        limit = self.limits[0] if phase == 'load' else self.limits[1]
        accepted = estimate['estimated_incremental_bytes'] <= limit
        row = {'schema': 'pretrained-gpu-trace-phase-v1', 'phase': phase, 'before': self.before,
               'after': after, 'estimate': estimate, 'commitment_bytes': limit,
               'within_commitment': accepted, 'proof': proof}
        write_json_new(self.directory / f'phase-{self.index + 1:02d}.json', row)
        self.phases.append(row)
        if not accepted:
            raise RuntimeError('measured phase exceeds its proposed commitment')
        self.current, self.before = None, None
        self.index += 1
