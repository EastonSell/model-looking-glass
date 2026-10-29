"""Bounded serial-reference and two-client routing core for one resident owner.

The caller owns admission, device supervision, evidence storage and cleanup.
This module does not acquire a GPU, establish a profile, or attest execution.
"""
import math
import threading
import time

import torch

from .broker import Broker, RequestState
from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
from .resident_worker import ForwardRequest, ForwardResult


def compare_f32(left, right, *, atol=1e-6, rtol=1e-5, chunk=8192):
    """Compare bounded CPU f32 tensors using float64 arithmetic and fixed chunks."""
    if (type(left) is not torch.Tensor or type(right) is not torch.Tensor
            or left.device.type != 'cpu' or right.device.type != 'cpu'
            or left.dtype != torch.float32 or right.dtype != torch.float32
            or tuple(left.shape) != tuple(right.shape) or left.ndim not in (1, 3)
            or not left.is_contiguous() or not right.is_contiguous()
            or type(chunk) is not int or not 1 <= chunk <= 8192
            or type(atol) is not float or type(rtol) is not float
            or not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0
            or not 1 <= left.numel() <= 32 * 65536):
        raise ValueError('unsupported comparison inputs or tolerance')
    left, right = left.view(-1), right.view(-1)
    count = violations = 0
    maximum = 0.0
    for start in range(0, left.numel(), chunk):
        a = left[start:start + chunk].to(torch.float64)
        b = right[start:start + chunk].to(torch.float64)
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise ValueError('nonfinite saved logits')
        difference = (a - b).abs()
        maximum = max(maximum, float(difference.max()))
        violations += int((difference > (atol + rtol * b.abs())).sum())
        count += a.numel()
    return {'count': count, 'violations': violations, 'max_abs': maximum,
            'atol': atol, 'rtol': rtol}


def _validate_request(request, *, bucket, vocab, pad):
    if type(request) is not ForwardRequest or type(request.inputs) is not dict:
        raise ValueError('expected a fixed CPU forward request')
    if set(request.inputs) != {'input_ids', 'attention_mask'}:
        raise ValueError('expected IDs and attention mask')
    ids, mask = request.inputs['input_ids'], request.inputs['attention_mask']
    if any(type(x) is not torch.Tensor or x.device.type != 'cpu' or x.dtype != torch.int64
           or tuple(x.shape) != (1, bucket) for x in (ids, mask)):
        raise ValueError('request shape or dtype differs')
    ids, mask = ids.tolist()[0], mask.tolist()[0]
    active = sum(mask)
    if (any(not 0 <= value < vocab for value in ids)
            or any(value not in (0, 1) for value in mask)
            or not 1 <= active <= bucket or mask != [1] * active + [0] * (bucket - active)
            or any(value != pad for value in ids[active:])):
        raise ValueError('invalid token IDs or right-padding mask')
    return active - 1


class SerialRoutingPilot:
    """One retained broker and four full-logit outcomes; caller owns lifetime."""

    def __init__(self, worker, authority, *, bucket, vocab, pad, seed, cases,
                 on_stage, on_tensor, timeout=45):
        if (type(worker) is not LifetimeWorkspaceWorker or type(authority) is not OwnerCompletionAuthority
                or worker._authority is not authority or authority._broker is not None
                or type(bucket) is not int or bucket not in (16, 32)
                or type(vocab) is not int or not 1 <= vocab <= 65536
                or type(pad) is not int or not 0 <= pad < vocab
                or type(seed) is not int or not 0 <= seed < 2**63
                or type(cases) is not tuple or len(cases) != 2
                or not callable(on_stage) or not callable(on_tensor)
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 45):
            raise ValueError('invalid pilot owner or fixed workload')
        self.worker, self.authority = worker, authority
        self.bucket, self.vocab, self.pad, self.seed = bucket, vocab, pad, seed
        self.on_stage, self.on_tensor, self.timeout = on_stage, on_tensor, timeout
        self.requests = []
        self.positions = []
        for index, inputs in enumerate(cases):
            request = ForwardRequest('case-' + str(index), inputs, seed)
            # The caller must not mutate its case while snapshotting it.
            snapshot = worker.snapshot_request(request)
            self.positions.append(_validate_request(ForwardRequest(request.request_id, dict(snapshot.inputs), seed),
                                                    bucket=bucket, vocab=vocab, pad=pad))
            self.requests.append(snapshot)
        self.broker = None
        self.handles, self.outcomes, self.threads, self.errors = [], [], [], []
        self.started = self.finished = False

    def _request(self, index, request_id):
        snapshot = self.requests[index]
        return ForwardRequest(request_id, dict(snapshot.inputs), self.seed)

    def _accept(self, handle, *, team, request_id):
        outcome = handle.wait(timeout=self.timeout)
        if (outcome.state is not RequestState.SUCCEEDED or outcome.team_id != team
                or outcome.request_id != request_id or outcome.broker_id != self.broker.broker_id
                or type(outcome.job_id) is not int or type(handle.job_id) is not int
                or outcome.job_id != handle.job_id
                or type(outcome.result) is not ForwardResult or outcome.result.request_id != request_id):
            raise RuntimeError('broker outcome identity or status differs')
        tensor = outcome.result.logits
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                or tensor.dtype != torch.float32 or tuple(tensor.shape) != (1, self.bucket, self.vocab)
                or not tensor.is_contiguous() or tensor.is_inference() or tensor.requires_grad):
            raise RuntimeError('broker did not deliver the bounded full-logit tensor')
        self.outcomes.append(outcome)
        return tensor

    def run(self):
        if self.started:
            raise RuntimeError('pilot is single use')
        self.started = True
        try:
            self.broker = Broker(self.worker, max_pending=2, max_pending_per_team=1,
                                 completion_authority=self.authority)
            references = []
            for index, team in enumerate(('alpha', 'beta')):
                request_id = 'reference-' + team
                self.on_stage('before-' + request_id)
                handle = self.broker.submit(team, self._request(index, request_id))
                self.handles.append(handle)
                value = self._accept(handle, team=team, request_id=request_id)
                self.on_tensor(request_id, value)
                references.append(value)
                self.on_stage('after-' + request_id)
            selected = [references[i][0, self.positions[i], :].contiguous() for i in range(2)]
            separation = compare_f32(selected[0], selected[1])
            if separation['violations'] == 0:
                raise RuntimeError('reference rows are indistinguishable at the routing tolerance')
            barrier = threading.Barrier(2)
            observed = [None, None]

            def client(index, team):
                try:
                    barrier.wait(timeout=10)
                    request_id = 'client-' + team
                    handle = self.broker.submit(team, self._request(index, request_id))
                    self.handles.append(handle)
                    observed[index] = self._accept(handle, team=team, request_id=request_id)
                except BaseException as exc:
                    self.errors.append(exc)
                    self.authority.mark_unsafe()

            self.on_stage('before-two-clients')
            self.threads = [threading.Thread(target=client, args=(i, team), daemon=True)
                            for i, team in enumerate(('alpha', 'beta'))]
            for thread in self.threads:
                thread.start()
            deadline = time.monotonic() + self.timeout + 10
            for thread in self.threads:
                thread.join(max(0, deadline - time.monotonic()))
            if any(thread.is_alive() for thread in self.threads) or self.errors or any(x is None for x in observed):
                raise RuntimeError('client completion is missing or uncertain')
            self.on_stage('after-two-clients')
            comparisons = []
            for index, team in enumerate(('alpha', 'beta')):
                self.on_tensor('client-' + team, observed[index])
                full = compare_f32(observed[index], references[index])
                own = compare_f32(observed[index][0, self.positions[index], :].contiguous(), selected[index])
                other = compare_f32(observed[index][0, self.positions[index], :].contiguous(), selected[1 - index])
                if full['violations'] or own['violations'] or not other['violations']:
                    raise RuntimeError('client output does not match its own distinct serial reference')
                comparisons.append({'team': team, 'active_position': self.positions[index],
                                    'full': full, 'own_last': own, 'other_last': other})
            if len({handle.job_id for handle in self.handles}) != 4:
                raise RuntimeError('broker job identities repeat')
            self.finished = True
            return {'status': 'passed', 'bucket': self.bucket, 'vocab': self.vocab,
                    'reference_separation': separation, 'comparisons': comparisons,
                    'broker_jobs': len(self.handles)}
        except BaseException:
            self.authority.mark_unsafe()
            raise

    def close(self):
        if not self.finished or self.errors or any(thread.is_alive() for thread in self.threads):
            self.authority.mark_unsafe()
            raise RuntimeError('pilot cannot be closed after uncertain work')
        try:
            if self.broker.close(timeout=30) is not True or self.authority.unsafe:
                raise RuntimeError('broker did not quiesce cleanly')
            self.worker.close()
        except BaseException:
            self.authority.mark_unsafe()
            raise
