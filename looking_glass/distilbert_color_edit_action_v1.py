"""Single-owner CPU model actions with restoration and full raw witnesses."""
import hashlib
import math
import sys

from . import distilbert_color_edit_metrics_v1 as metrics
from . import distilbert_color_edit_records_v1 as records
from .bert_agreement_study import capture_model_state, verify_model_state, _require_denormals


def tensor_bytes(tensor):
    import torch
    if (sys.byteorder != 'little' or tensor.dtype != torch.float32 or
            tensor.device.type != 'cpu' or not tensor.is_contiguous()):
        raise ValueError('a contiguous CPU f32 tensor on a little-endian host is required')
    _require_denormals(torch)
    raw = tensor.detach().view(torch.uint8).numpy().tobytes()
    _require_denormals(torch)
    return raw


def _no_hooks(model, torch):
    for module in model.modules():
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise RuntimeError('model has an unowned hook')
    module = torch.nn.modules.module
    for name in ('_global_forward_hooks', '_global_forward_pre_hooks',
                 '_global_backward_hooks', '_global_backward_pre_hooks'):
        if getattr(module, name, {}):
            raise RuntimeError('global module hooks are not allowed')


class ActionEngine:
    """Run one action at a time and restore the exact original registered state."""

    def __init__(self, model, linear, bias, *, vocabulary, hidden, intermediate):
        import torch
        if (not isinstance(model, torch.nn.Module) or any(module.training for module in model.modules()) or
                not isinstance(linear, torch.nn.Linear) or linear.weight.shape != (hidden, intermediate) or
                bias.shape != (vocabulary,) or bias.dtype != torch.float32 or
                any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32
                    for parameter in model.parameters())):
            raise ValueError('invalid eager CPU f32 model or target shapes')
        _no_hooks(model, torch)
        _require_denormals(torch)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError('preexisting parameter gradients are not allowed')
        self.model, self.linear, self.bias = model, linear, bias
        self.vocabulary, self.hidden, self.intermediate = vocabulary, hidden, intermediate
        self.weight_original = linear.weight.detach().clone()
        self.bias_original = bias.detach().clone()
        self.state = capture_model_state(model)

    def run(self, journal, row, *, phase, condition, delta=None, parameter=None,
            gradient_targets=None, capture_site=False):
        import torch
        _no_hooks(self.model, torch)
        verify_model_state(self.model, self.state)
        _require_denormals(torch)
        metrics.require_denormals()
        ids, mask, position = row['input_ids'], row['attention_mask'], row['mask_index']
        if (type(ids) is not list or type(mask) is not list or not ids or len(ids) != len(mask)
                or any(type(x) is not int for x in ids + mask) or any(x != 1 for x in mask)
                or type(position) is not int or not 0 <= position < len(ids)):
            raise ValueError('invalid explicit unpadded input and attention mask')
        if gradient_targets is not None and (not capture_site or delta is not None or
                len(gradient_targets) != 2 or gradient_targets[0] == gradient_targets[1] or
                any(type(i) is not int or not 0 <= i < self.vocabulary for i in gradient_targets)):
            raise ValueError('invalid gradient action')
        if delta is None and parameter is not None or delta is not None and parameter not in ('weight', 'bias'):
            raise ValueError('delta and affected parameter must be specified together')
        if delta is not None:
            target = self.linear.weight if parameter == 'weight' else self.bias
            if (not isinstance(delta, torch.Tensor) or delta.dtype != torch.float32 or
                    delta.device.type != 'cpu' or delta.shape != target.shape or
                    not bool(torch.isfinite(delta).all()) or not bool(torch.any(delta != 0))):
                raise ValueError('update must be finite, nonzero and have the exact target shape')
        else:
            target = None
        action = journal.forwards
        binding = {'action': action, 'phase': phase, 'object': row['object'],
                   'case_id': row['id'], 'condition': condition, 'input_ids': list(ids),
                   'attention_mask': list(mask), 'mask_index': position}
        journal.start(binding)
        inputs = torch.tensor([ids], dtype=torch.int64, device='cpu')
        attention = torch.tensor([mask], dtype=torch.int64, device='cpu')
        before_inputs, before_attention = inputs.clone(), attention.clone()
        saved_flags = [p.requires_grad for p in self.model.parameters()]
        hook = None
        site = {}
        realized_norm = 0.0
        parameter_sha = None
        captured_logits = None
        returned_output = {'status': 'not-returned'}
        gradient = None
        failure = None
        failure_stage = 'before-forward'
        expected_weight, expected_bias = self.weight_original, self.bias_original
        try:
            if delta is not None:
                with torch.no_grad():
                    original = self.weight_original if parameter == 'weight' else self.bias_original
                    target.copy_(original + delta)
                    actual = target.detach().to(torch.float64) - original.to(torch.float64)
                    realized_norm = float(torch.linalg.vector_norm(actual).item())
                    parameter_sha = hashlib.sha256(tensor_bytes(target.detach().contiguous())).hexdigest()
                    if not math.isfinite(realized_norm) or realized_norm == 0:
                        raise RuntimeError('update rounded to a nonfinite or zero realized change')
                    if parameter == 'weight':
                        expected_weight = target.detach().clone()
                    else:
                        expected_bias = target.detach().clone()
            if gradient_targets is not None:
                for p in self.model.parameters():
                    p.requires_grad_(False)
                self.linear.weight.requires_grad_(True)
            if capture_site:
                def capture(_module, args, output):
                    if site or len(args) != 1 or not isinstance(output, torch.Tensor):
                        raise RuntimeError('unexpected final linear invocation')
                    if (tuple(args[0].shape) != (1, len(ids), self.intermediate) or
                            tuple(output.shape) != (1, len(ids), self.hidden)):
                        raise RuntimeError('unexpected final linear site shape')
                    site['input'] = args[0][0, position].detach().contiguous().clone()
                    site['output'] = output[0, position].detach().contiguous().clone()
                    if gradient_targets is not None:
                        def save_gradient(value):
                            if 'output_gradient' in site:
                                raise RuntimeError('site gradient returned more than once')
                            site['output_gradient'] = value[0, position].detach().contiguous().clone()
                        output.register_hook(save_gradient)
                hook = self.linear.register_forward_hook(capture)
            with torch.set_grad_enabled(gradient_targets is not None):
                failure_stage = 'forward'
                output = self.model(input_ids=inputs, attention_mask=attention).logits
                failure_stage = 'post-forward-validation'
                if not isinstance(output, torch.Tensor):
                    returned_output = {'status': 'unavailable-non-tensor'}
                else:
                    returned_output = {'shape': list(output.shape), 'dtype': str(output.dtype),
                                       'device': output.device.type, 'status': 'unavailable-unindexable'}
                    if output.device.type != 'cpu' or output.dtype != torch.float32:
                        returned_output['status'] = 'unavailable-device-or-dtype'
                    elif output.ndim == 3 and output.shape[0] > 0 and output.shape[1] > position:
                        if not 0 < output.shape[2] <= self.vocabulary:
                            returned_output['status'] = 'unavailable-oversized-or-empty'
                        else:
                            captured_logits = output[0, position].detach().contiguous().clone()
                            returned_output['status'] = 'saved'
                            returned_output['row_length'] = output.shape[2]
                if (not isinstance(output, torch.Tensor) or
                        tuple(output.shape) != (1, len(ids), self.vocabulary) or
                        output.device.type != 'cpu' or output.dtype != torch.float32 or
                        not output.is_contiguous()):
                    raise RuntimeError('model returned invalid full logits')
                if not bool(torch.isfinite(output).all()):
                    raise RuntimeError('model returned nonfinite full logits')
                if gradient_targets is not None:
                    first, second = gradient_targets
                    gradient = torch.autograd.grad(output[0, position, first] - output[0, position, second],
                                                   self.linear.weight, retain_graph=False)[0].detach().contiguous()
            if not torch.equal(inputs, before_inputs) or not torch.equal(attention, before_attention):
                raise RuntimeError('model mutated an input tensor')
            if (not torch.equal(self.linear.weight.detach(), expected_weight) or
                    not torch.equal(self.bias.detach(), expected_bias)):
                raise RuntimeError('weight or decoder bias changed during the forward')
            if capture_site and not {'input', 'output'}.issubset(site):
                raise RuntimeError('final linear site was not captured')
            if gradient is not None:
                if ('output_gradient' not in site or not bool(torch.isfinite(gradient).all()) or
                        any(not bool(torch.isfinite(v).all()) for v in site.values())):
                    raise RuntimeError('gradient witness is incomplete or nonfinite')
                grad64 = gradient.to(torch.float64)
                reconstructed = torch.outer(site['output_gradient'].to(torch.float64),
                                            site['input'].to(torch.float64))
                norm = float(torch.linalg.vector_norm(grad64).item())
                if not math.isfinite(norm) or norm == 0:
                    raise RuntimeError('gradient is zero or nonfinite')
                residual = float(torch.linalg.vector_norm(grad64 - reconstructed).item()) / norm
                if not math.isfinite(residual) or residual > 1e-5:
                    raise RuntimeError('rank-one gradient reconstruction failed')
            else:
                residual = None
        except BaseException as error:
            failure = error
        cleanup_failure = None
        try:
            if hook is not None:
                hook.remove()
        except BaseException as error:
            cleanup_failure = error
        try:
            with torch.no_grad():
                self.linear.weight.copy_(self.weight_original)
                self.bias.copy_(self.bias_original)
                for p, flag in zip(self.model.parameters(), saved_flags):
                    p.requires_grad_(flag)
        except BaseException as error:
            cleanup_failure = error
        try:
            _no_hooks(self.model, torch)
            verify_model_state(self.model, self.state)
            _require_denormals(torch)
            metrics.require_denormals()
        except BaseException as error:
            cleanup_failure = error
        if cleanup_failure is not None:
            failure = cleanup_failure
            failure_stage = 'restoration'
        if failure is not None:
            incomplete = {'kind': 'incomplete', **binding, 'failure_stage': failure_stage,
                          'failure_type': type(failure).__name__,
                          'returned_output': returned_output,
                          'actual_input_ids': inputs[0].tolist(),
                          'actual_attention_mask': attention[0].tolist()}
            try:
                if captured_logits is not None:
                    incomplete['returned_logits'] = records.encode_raw(tensor_bytes(captured_logits))
                for key, tensor in site.items():
                    incomplete['site_' + key] = records.encode_raw(tensor_bytes(tensor))
                if gradient is not None:
                    incomplete['gradient'] = records.encode_raw(tensor_bytes(gradient))
                journal.emit(incomplete)
            except BaseException as evidence_error:
                raise RuntimeError('failure witness could not be completely saved') from evidence_error
            raise failure
        done = {'kind': 'complete', **binding, 'logits': records.encode_raw(tensor_bytes(captured_logits)),
                'parameter': parameter, 'realized_norm': realized_norm, 'parameter_sha256': parameter_sha}
        if capture_site:
            done['site_input'] = records.encode_raw(tensor_bytes(site['input']))
            done['site_output'] = records.encode_raw(tensor_bytes(site['output']))
        if gradient is not None:
            done['gradient_targets'] = list(gradient_targets)
            done['gradient'] = records.encode_raw(tensor_bytes(gradient))
            done['site_output_gradient'] = records.encode_raw(tensor_bytes(site['output_gradient']))
            done['relative_rank_one_residual'] = residual
        journal.emit(done)
        return {'logits': captured_logits, 'gradient': gradient, 'site': site,
                'realized_norm': realized_norm, 'parameter_sha256': parameter_sha,
                'relative_rank_one_residual': residual}
