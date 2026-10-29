"""Fixed, untimed five-token trace comparisons for one supervised pilot.

The caller supplies the bounded phase/forward executor and retains its model,
inputs, outputs and capture roots. This module cannot authorize CUDA access.
"""

import hashlib
import threading

import torch

from .model_ownership import ModelLease
from .transformer_trace import GPT2TraceSession, _note_cleanup

TOKENS = (464, 6766, 318, 4171, 13)
LAYERS = (0, 2, 5)
TARGETS = ('attn', 'mlp', 'block')
POSITIONS = (0, 1)
TRACE_CAP = 131072
PHASES = ('baseline_cold', 'baseline_warm', 'reference', 'direct', 'gradient', 'id')


class _PilotTraceSession(GPT2TraceSession):
    """Keep a cleanup exception available if the adapter annotates a body error."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pilot_close_failure = None

    def close(self):
        try:
            if self._owner_thread is not None and threading.current_thread() is not self._owner_thread:
                raise RuntimeError('trace session must be closed by its owning thread')
            if self._entering or self._model_forward_on_stack():
                raise RuntimeError('cannot close a trace while registering or during a forward')
            if self._active or self._frames or self._site_frames or self._normal_completed:
                self._invalidate()
                self._active = self._normal_completed = False
                self._shape = None
                self._frames.clear()
                self._site_frames.clear()
            failed, errors = [], []
            for handle in self._handles:
                try:
                    handle.remove()
                except BaseException as error:
                    failed.append(handle)
                    errors.append(error)
            self._handles[:] = failed
            if isinstance(self._untracked_handle, torch.utils.hooks.RemovableHandle):
                try:
                    self._untracked_handle.remove()
                    self._untracked_handle = None
                    self._registration_inflight = False
                except BaseException as error:
                    errors.append(error)
            if self._registration_uncertain is not None or self._registration_inflight:
                errors.append(RuntimeError('hook registration is uncertain; model ownership retained'))
            if errors:
                self._invalidate()
                selected = next((error for error in errors if isinstance(error, (KeyboardInterrupt, SystemExit))),
                                errors[0])
                if selected is not errors[0]:
                    raise selected from errors[0]
                raise selected
            try:
                self._lease.release()
            except BaseException as primary:
                self._invalidate()
                try:
                    self._lease.release()
                except BaseException as cleanup:
                    if (isinstance(cleanup, (KeyboardInterrupt, SystemExit))
                            and not isinstance(primary, (KeyboardInterrupt, SystemExit))):
                        raise cleanup from primary
                    _note_cleanup(primary, 'model lease release retry was uncertain')
                raise
        except BaseException as error:
            self.pilot_close_failure = error
            raise


def _bits_equal(left, right):
    if (left.dtype != torch.float32 or right.dtype != torch.float32 or left.shape != right.shape
            or left.device.type != 'cpu' or right.device.type != 'cpu'):
        return False
    return bool(torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)))


def _hash(value):
    if value.dtype != torch.float32 or value.device.type != 'cpu' or not bool(torch.isfinite(value).all()):
        raise ValueError('a finite CPU float32 tensor is required')
    return hashlib.sha256(value.detach().contiguous().numpy().astype('<f4', copy=False).tobytes()).hexdigest()


def _sites():
    return [(f'transformer.h.{layer}' + ('' if kind == 'block' else '.' + kind), kind, layer)
            for layer in LAYERS for kind in TARGETS]


def run_fixed_trace(model, input_ids, attention_mask, executor, *, study_id):
    """Run the exact phase order; executor performs each guarded forward and drain."""
    if (not isinstance(study_id, str) or not study_id or not isinstance(input_ids, torch.Tensor)
            or not isinstance(attention_mask, torch.Tensor) or tuple(input_ids.shape) != (1, 5)
            or tuple(attention_mask.shape) != (1, 5) or input_ids.dtype != torch.int64
            or attention_mask.dtype != torch.int64 or input_ids.device != attention_mask.device):
        raise ValueError('fixed run identity or input geometry differs')
    if input_ids.device.type == 'cpu':
        if input_ids.tolist() != [list(TOKENS)] or attention_mask.tolist() != [[1] * 5]:
            raise ValueError('fixed token IDs or attention mask differ')
    elif input_ids.device.type == 'cuda':
        executor.verify_inputs(input_ids, attention_mask)
    else:
        raise ValueError('unsupported input device')
    hidden, vocab = model.config.n_embd, model.config.vocab_size
    if (type(hidden) is not int or hidden < 1 or type(vocab) is not int or vocab < 1
            or 18 * 2 * hidden * 4 > TRACE_CAP or 3 * 18 * 2 * hidden * 4 > 3 * TRACE_CAP
            or 5 * vocab * 4 > 1005140 or 6 * 5 * vocab * 4 > 6030840):
        raise ValueError('model geometry exceeds the fixed trace or output allowance')
    expected = [(forward, path, kind, layer) for forward in range(2) for path, kind, layer in _sites()]
    reference = []
    executor.bind_reference(reference)
    owner_thread = threading.current_thread()
    forward_index = 0
    reference_failed = False
    handles = [None] * len(_sites())
    lease = ModelLease(model)
    executor.bind_reference_lease(lease, handles)

    def forward(phase):
        def invoke():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                         return_dict=True, output_hidden_states=False, output_attentions=False)
        logits = executor.forward(phase, invoke)
        if (tuple(logits.shape) != (1, 5, vocab) or logits.dtype != torch.float32
                or logits.device.type != 'cpu'):
            raise ValueError('returned logits differ from the fixed geometry')
        _hash(logits)
        return logits

    def hook_for(path, kind, layer):
        def capture(_module, _args, output):
            nonlocal reference_failed
            try:
                if threading.current_thread() is not owner_thread or len(reference) >= 18:
                    raise RuntimeError('reference event thread or capacity differs')
                lease.verify_tree()
                if isinstance(output, torch.Tensor):
                    tensor, selector = output, ()
                elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                    tensor, selector = output[0], (0,)
                else:
                    raise TypeError('unsupported reference output')
                if (tuple(tensor.shape) != (1, 5, hidden) or tensor.dtype != torch.float32
                        or tensor.device != input_ids.device
                        or selector != ((0,) if kind in ('attn', 'block') else ())):
                    raise ValueError('reference output geometry or selector differs')
                if (len(reference) + 1) * 2 * hidden * 4 > TRACE_CAP:
                    raise ValueError('reference trace allowance would be exceeded')
                value = tensor.detach()[:, list(POSITIONS), :].to('cpu', copy=True)
                _hash(value)
                reference.append((forward_index, path, kind, layer, selector, value))
            except BaseException:
                reference_failed = True
                raise
            return None
        return capture

    with torch.inference_mode():
        uncertain = False
        primary = None
        try:
            lease.acquire()
            lease.verify_tree()
            baseline = []
            for phase in PHASES[:2]:
                executor.begin(phase)
                baseline.append(forward(phase))
                executor.end(phase, {'logit_sha256': [_hash(baseline[-1])]})
            if not _bits_equal(*baseline):
                raise ValueError('repeated baseline logits differ in float32 bits')
            executor.begin('reference')
            for index, (path, kind, layer) in enumerate(_sites()):
                uncertain = True
                handle = model.get_submodule(path).register_forward_hook(hook_for(path, kind, layer))
                if not isinstance(handle, torch.utils.hooks.RemovableHandle):
                    raise RuntimeError('reference hook registration is uncertain')
                handles[index] = handle
                uncertain = False
            reference_logits = []
            for forward_index in range(2):
                lease.verify_tree()
                reference_logits.append(forward('reference'))
                lease.verify_tree()
                if reference_failed:
                    raise RuntimeError('reference capture failed')
        except BaseException as error:
            primary = error
            raise
        finally:
            # These are host-only hook removals; any uncertain forward still
            # fails the owner and keeps the physical lease and roots to exit.
            cleanup_errors = []
            for handle in handles:
                if handle is not None:
                    try:
                        handle.remove()
                    except BaseException as error:
                        cleanup_errors.append(error)
            if not uncertain and not cleanup_errors:
                try:
                    lease.release()
                except BaseException as error:
                    cleanup_errors.append(error)
            if uncertain:
                cleanup_errors.append(RuntimeError('reference hook registration is uncertain'))
            cancellations = [error for error in cleanup_errors if isinstance(error, (KeyboardInterrupt, SystemExit))]
            if cancellations and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise cancellations[0] from primary
            if primary is not None and cleanup_errors:
                _note_cleanup(primary, 'reference hook or lease cleanup was uncertain')
            elif cleanup_errors:
                raise cleanup_errors[0]
        if (len(reference) != 18 or [(f, p, k, l) for f, p, k, l, _, _ in reference] != expected
                or not all(_bits_equal(a, b) for a, b in zip(baseline, reference_logits))):
            raise ValueError('reference event order or logits differ')
        ref_hashes = [_hash(item[5]) for item in reference]
        executor.end('reference', {'logit_sha256': [_hash(x) for x in reference_logits],
                                   'value_sha256': ref_hashes, 'events': len(reference)})
        result = {'baseline_logit_sha256': [_hash(x) for x in baseline],
                  'reference_logit_sha256': [_hash(x) for x in reference_logits],
                  'reference_value_sha256': ref_hashes, 'modes': {}}
        direct_values = None
        for mode in ('direct', 'gradient', 'id'):
            executor.begin(mode)
            trace = _PilotTraceSession(model, run_id=f'{study_id}-{mode}', layers=LAYERS, targets=TARGETS,
                                     samples=(0,), tokens=POSITIONS, mode=mode,
                                     max_events=18, max_trace_bytes=TRACE_CAP, capture_device='cpu')
            executor.bind_trace(mode, trace)
            outputs = []
            try:
                with trace:
                    for _ in range(2):
                        outputs.append(forward(mode))
            except BaseException as primary:
                cleanup = trace.pilot_close_failure
                if (isinstance(cleanup, (KeyboardInterrupt, SystemExit))
                        and not isinstance(primary, (KeyboardInterrupt, SystemExit)) and cleanup is not primary):
                    raise cleanup from primary
                raise
            if (not trace.successful or len(trace.events) != 18
                    or not all(_bits_equal(a, b) for a, b in zip(baseline, outputs))):
                raise ValueError(f'{mode} trace or logits differ')
            events, values = [], []
            for event, ref, expect in zip(trace.events, reference, expected):
                _fwd, _path, kind, _layer = expect
                if ((event.forward_index, event.module_path, event.module_kind, event.layer_index) != expect
                        or event.sequence != len(events) or event.invocation != 0
                        or event.run_id != f'{study_id}-{mode}' or event.capture_mode != mode
                        or event.output_path != ref[4] or event.batch_positions != (0,)
                        or event.token_positions != POSITIONS or event.shape != (1, 5, hidden)
                        or event.dtype != 'torch.float32' or event.device != str(input_ids.device)
                        or event.output_path != ((0,) if kind in ('attn', 'block') else ())):
                    raise ValueError(f'{mode} event metadata differs')
                if mode == 'id':
                    if event.value is not None or event.saved_shape is not None:
                        raise ValueError('ID capture retained a value')
                    digest = None
                else:
                    if (event.saved_shape != (1, 2, hidden) or event.saved_device != 'cpu'
                            or not _bits_equal(event.value, ref[5])):
                        raise ValueError(f'{mode} capture bits differ from reference')
                    digest = _hash(event.value)
                    values.append(event.value)
                events.append({'metadata': event.metadata(), 'value_sha256': digest})
            expected_bytes = 0 if mode == 'id' else 18 * 2 * hidden * 4
            if trace.tensor_bytes != expected_bytes or expected_bytes > TRACE_CAP:
                raise ValueError(f'{mode} capture byte count differs')
            if mode == 'direct':
                direct_values = values
            elif mode == 'gradient' and not all(_bits_equal(a, b) for a, b in zip(direct_values, values)):
                raise ValueError('direct and gradient captures differ')
            proof = {'logit_sha256': [_hash(x) for x in outputs], 'event_count': len(events),
                     'tensor_bytes': trace.tensor_bytes, 'events': events,
                     'logits_equal_baseline': True, 'values_equal_reference': None if mode == 'id' else True}
            result['modes'][mode] = proof
            executor.end(mode, proof)
        result['direct_gradient_equal'] = True
        return result
