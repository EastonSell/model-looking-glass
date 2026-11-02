"""Bounded, fail-stop forward adapter for one exposed ViT pair.

The caller must hold the physical device lease and memory/trace claims. This
module does not acquire them or prove continuous ownership inside a native
call. Its CPU device option exists only for fabricated tests.
"""

from threading import get_ident


FULL_BYTES = 197 * 192 * 4
TOKEN_BYTES = 192 * 4
MAX_SOURCE_BYTES = FULL_BYTES
MAX_CAPTURE_BYTES = 2 * FULL_BYTES + 8 * TOKEN_BYTES
MAX_TENSOR_BYTES = FULL_BYTES


def _copy_to_cpu(value):
    return value.detach().to('cpu').contiguous()


def _transfer_to_device(value, device):
    return value.detach().to(device)


class ForwardRoots:
    """Objects held until a verified completion and per-forward drain."""

    __slots__ = ('model', 'pixels', 'replacement', 'plan', 'sources', 'capture_roots',
                 'output', 'failure')

    def __init__(self, model):
        self.model = model
        self.pixels = self.replacement = self.plan = self.sources = None
        self.capture_roots = self.output = self.failure = None


class ForwardAdapter:
    """Supply CPU-copied results to the fixed scientific runner."""

    def __init__(self, *, check, synchronize, device):
        if not callable(check) or not callable(synchronize) or device not in ('cpu', 'cuda:0'):
            raise ValueError('invalid owner callbacks or device')
        self.check = check
        self.synchronize = synchronize
        self.device = device
        self.thread = get_ident()
        self.pending = False
        self.failed = False
        self.roots = None

    def _check(self):
        if get_ident() != self.thread:
            raise RuntimeError('forward adapter is single-threaded')
        self.check()

    def __call__(self, model, pixels, evidence, label, *, path=None, tokens=None,
                 replacement=None, action='capture', reference=None, numeric_guard=None):
        if self.failed or self.pending or self.roots is not None:
            raise RuntimeError('forward adapter is unsafe or already active')
        self.pending = True
        self.roots = ForwardRoots(model)
        try:
            self._check()
            result = self._execute(model, pixels, evidence, label, path=path, tokens=tokens,
                                   replacement=replacement, action=action, reference=reference,
                                   numeric_guard=numeric_guard)
            self._check()
            self.synchronize()
            self._check()
            for name in ('output', 'capture_roots', 'sources', 'plan', 'replacement', 'pixels', 'model'):
                self._check()
                setattr(self.roots, name, None)
                self._check()
            self.synchronize()
            self._check()
            self.roots = None
            self.pending = False
            return result
        except BaseException as error:
            self.failed = True
            if self.roots is not None:
                self.roots.failure = error
            if isinstance(getattr(evidence, 'active', None), dict):
                evidence.active['return_observed'] = self.roots is not None and self.roots.output is not None
            raise

    def _execute(self, model, pixels, evidence, label, *, path, tokens, replacement,
                 action, reference, numeric_guard):
        import torch
        from .bert_gpu_exposed_pair import _RetainedPlanCaptures, _unexpected_hooks
        from .intervention_plan import ActivationPlan
        from .vision_coco_study import EARLY, LATE, _bits, _check_metadata, check_denormals, score

        if (type(pixels) is not torch.Tensor or pixels.device.type != 'cpu'
                or pixels.dtype != torch.float32 or tuple(pixels.shape) != (1, 3, 224, 224)):
            raise ValueError('expected one CPU float32 image')
        if not bool(torch.isfinite(pixels).all()):
            raise ValueError('nonfinite image')
        if replacement is not None and (type(replacement) is not torch.Tensor
                or replacement.device.type != 'cpu' or replacement.dtype != torch.float32):
            raise ValueError('replacement must be a CPU float32 tensor')
        if action not in ('capture', 'null', 'patch') or (action == 'patch') != (replacement is not None):
            raise ValueError('invalid action or replacement')
        if path is None and (tokens is not None or action != 'capture'):
            raise ValueError('clean forward cannot carry an intervention')
        if path is not None and (path not in (EARLY, LATE) or action not in ('null', 'patch')):
            raise ValueError('invalid intervention site or action')
        if tokens is not None:
            if (type(tokens) is not list or not all(type(token) is int for token in tokens)
                    or not ((len(tokens) == 1 and tokens[0] == 0)
                            or (len(tokens) == 4 and len(set(tokens)) == 4
                                and all(1 <= token < 197 for token in tokens)))):
                raise ValueError('invalid selected token inventory')
        if replacement is not None:
            expected_shape = (1, 197, 192) if tokens is None else (len(tokens), 192)
            if tuple(replacement.shape) != expected_shape:
                raise ValueError('invalid replacement geometry')
            if not bool(torch.isfinite(replacement).all()):
                raise ValueError('replacement must be finite')
        if _unexpected_hooks(model):
            raise RuntimeError('unexpected hooks before forward')
        evidence.reserve(label)
        check_denormals()
        if numeric_guard is not None:
            numeric_guard()
        original_pixels = _bits(pixels)
        self._check()
        self.roots.pixels = _transfer_to_device(pixels, self.device)
        self._check()
        if replacement is not None:
            self.roots.replacement = _transfer_to_device(replacement, self.device)
            self._check()
        if path is None:
            steps = [{'name': 'early', 'operation': 'capture', 'module_path': EARLY},
                     {'name': 'late', 'operation': 'capture', 'module_path': LATE}]
        else:
            site = {'module_path': path}
            steps = [dict(name='before_all', operation='capture', **site)]
            if tokens is None:
                step = dict(name='action', operation=action, **site)
                if action == 'patch':
                    step['value'] = self.roots.replacement
                steps.append(step)
            else:
                for index, token in enumerate(tokens):
                    selected = dict(site, batch_index=0, token_index=token)
                    steps.append(dict(name=f'before_{index}', operation='capture', **selected))
                    step = dict(name=f'action_{index}', operation=action, **selected)
                    if action == 'patch':
                        step['value'] = self.roots.replacement[index]
                    steps.extend((step, dict(name=f'after_{index}', operation='capture', **selected)))
            steps.append(dict(name='after_all', operation='capture', **site))
        self._check()
        plan = ActivationPlan.__new__(ActivationPlan)
        self.roots.plan = plan
        ActivationPlan.__init__(plan, model, steps, max_tensor_bytes=MAX_TENSOR_BYTES,
                                max_capture_bytes=MAX_CAPTURE_BYTES, max_source_bytes=MAX_SOURCE_BYTES,
                                max_steps=14, max_invocations=1)
        self.roots.sources = tuple(plan._sources.values())
        self.roots.capture_roots = _RetainedPlanCaptures()
        plan._captures = self.roots.capture_roots
        self._check()
        with torch.inference_mode(), torch.autocast(device_type=self.roots.pixels.device.type, enabled=False), plan:
            self.roots.output = model(pixel_values=self.roots.pixels, return_dict=True,
                                      output_hidden_states=False, output_attentions=False)
            self._check()
            self.synchronize()
            self._check()
            value = self.roots.output.logits
            if type(value) is not torch.Tensor or value.dtype != torch.float32 or not 0 < value.numel() <= 1000:
                raise ValueError('returned logits cannot be retained')
            self._check()
            logits = _copy_to_cpu(value)
            self._check()
            if type(logits) is not torch.Tensor or logits.device.type != 'cpu':
                raise ValueError('logit copy did not reach CPU')
            raw = _bits(logits)
            name = f'return-{evidence.submissions:03d}.bin'
            digest = evidence.write(name, raw)
            evidence.json(f'return-{evidence.submissions:03d}.json',
                          {'label': label, 'file': name, 'sha256': digest,
                           'shape': list(logits.shape), 'dtype': 'little-endian-float32'})
        self._check()
        captures = {}
        for name, value in plan.captures.items():
            self._check()
            captures[name] = _copy_to_cpu(value)
            self._check()
        if (logits.device.type != 'cpu' or any(type(value) is not torch.Tensor or value.device.type != 'cpu'
                                               for value in captures.values())
                or (replacement is not None and replacement.device.type != 'cpu')):
            raise ValueError('scientific handoff requires CPU tensors')
        tensors = {'logits': logits, **captures}
        if replacement is not None:
            tensors['requested'] = replacement
        row = evidence.save_forward_bytes(label, tensors)
        evidence.active['phase'] = 'validation'
        self._check()
        completed_pixels = _copy_to_cpu(self.roots.pixels)
        self._check()
        if (type(completed_pixels) is not torch.Tensor or completed_pixels.device.type != 'cpu'
                or completed_pixels.dtype != torch.float32
                or tuple(completed_pixels.shape) != (1, 3, 224, 224)
                or _bits(completed_pixels) != original_pixels or _bits(pixels) != original_pixels):
            raise ValueError('model modified input pixels')
        check_denormals()
        if numeric_guard is not None:
            numeric_guard()
        metadata = plan.metadata
        record_score, _ = score(logits, reference)
        if numeric_guard is not None:
            numeric_guard()
        if evidence.labels is not None:
            record_score['top1_label'] = evidence.labels[record_score['top1_id']]
        row = evidence.finish_forward(row, metadata, record_score)
        _check_metadata(metadata, steps)
        if tuple(logits.shape) != (1, 1000) or not bool(torch.isfinite(logits).all()):
            raise ValueError('invalid or nonfinite logits')
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
               for value in captures.values()):
            raise ValueError('invalid or nonfinite capture')
        if _unexpected_hooks(model):
            raise RuntimeError('hook cleanup is uncertain')
        return logits, captures, row
