"""Fabricated control, schedule and retention checks; no checkpoint is loaded."""
import struct
from types import SimpleNamespace

import pytest
import torch

from looking_glass import copy_paired_continuation_v1 as m
from scripts import copy_paired_continuation_cpu_v1 as owner


class Projection(torch.nn.Module):
    def forward(self, x):
        return x.clone()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = Projection()
        self.transformer = SimpleNamespace(h=[None, None, None,
            SimpleNamespace(attn=SimpleNamespace(c_proj=self.projection))])

    def forward(self, input_ids, attention_mask, **kwargs):
        x = torch.arange(12*768, dtype=torch.float32).reshape(1, 12, 768)
        self.projection(x)
        logits = torch.zeros((1, 12, m.VOCAB), dtype=torch.float32)
        return SimpleNamespace(logits=logits, past_key_values=None, attentions=None, hidden_states=None)


def case(v):
    return {'case_id':f'fake-{v}', 'group':1, 'variant':v, 'token_ids':list(range(12)),
            'candidate_f_id':1, 'candidate_g_id':2, 'correct_id':1 if v in (0,2) else 2}


def weights():
    out = {}
    for h, scale in ((2,1.0), (10,2.0), (7,1.0)):
        x = [0.0] * (64*768); x[0] = scale
        out[h] = struct.pack('<%df' % len(x), *x)
    return out


def before(v, *, zero=False):
    b = bytearray(m.WITNESS_BYTES)
    struct.pack_into('<f', b, (10*768+2*64)*4, 99+v)
    if v in (1,3) and not zero:
        for h, value in ((2,4.0), (10,2.0), (7,4.0)):
            struct.pack_into('<f', b, (11*768+h*64)*4, value)
    return bytes(b)


def test_real_twelve_token_witness_and_returned_control_bits():
    model = FakeModel(); ids = torch.tensor([list(range(12))])
    obs = {}; clean = owner.run_forward(model, ids, 'clean', None, None, obs)
    noop = owner.run_forward(model, ids, 'noop', None, None, {})
    delivery = m.head_slice(noop['cproj_before'], 2)
    self_patch = owner.run_forward(model, ids, 'self', 2, delivery, {})
    cache = {'clean':clean['logits'], 'before':noop['cproj_before'], 'noop_output':noop['cproj_output']}
    owner.validate_payload(noop, 'noop', cache, None, None)
    owner.validate_payload(self_patch, 'self', cache, 2, delivery)
    assert len(noop['cproj_before']) == 12*768*4
    assert m.head_slice(noop['cproj_before'], 2) == delivery
    damaged = dict(self_patch); x = bytearray(damaged['cproj_after']); x[10*768*4] ^= 1
    damaged['cproj_after'] = bytes(x)
    with pytest.raises(RuntimeError, match='untouched input'):
        owner.validate_payload(damaged, 'self', cache, 2, delivery)


@pytest.mark.parametrize('zero,bad_actual', [(False,False), (True,False), (False,True)])
def test_unsupported_family_preserves_schedule_and_dose_outcomes(tmp_path, zero, bad_actual):
    batch = [case(v) for v in range(4)]
    caches = [{'clean_score':{'correct_id':c['correct_id'], 'other_id':3-c['correct_id'],
                    'correct_logit':0.0, 'other_logit':0.0, 'margin':0.0, 'top_unique_id':None}}
              for c in batch]
    actions = []; weight_raw = weights(); matrix = m.weights(weight_raw)
    def perform(c, kind, cache, head=None, delivery=None):
        v = c['variant']; actions.append((v,kind,head))
        if kind == 'noop':
            cache.update({'before':before(v,zero=zero), 'noop_output':bytes(m.WITNESS_BYTES)})
        elif kind == 'self':
            assert head == 2 and delivery == m.head_slice(cache['before'],2)
        elif kind == 'patch':
            src = m._values(m.head_slice(cache['before'],head)); dst = m._values(delivery)
            vector = m.project(tuple(y-x for x,y in zip(src,dst)),matrix[head])
            raw = bytearray(m.WITNESS_BYTES)
            for j, value in enumerate(vector): struct.pack_into('<f',raw,(11*768+j)*4,value)
            if bad_actual and v == 0 and head == 2:
                struct.pack_into('<f',raw,11*768*4,4.5)
            landed = bytearray(cache['before']); start = (11*768+64*head)*4
            landed[start:start+256] = delivery
            owner.validate_payload({'logits':bytes(m.ROW_BYTES), 'cproj_before':cache['before'],
                'cproj_after':bytes(landed), 'cproj_output':bytes(raw)}, 'patch', cache, head, delivery)
            cache.setdefault('arm_outputs',{})[head] = bytes(raw)
            cache.setdefault('patch_scores',{})[head] = dict(cache['clean_score'])
    with (tmp_path/'journal').open('wb') as stream:
        family = owner.execute_family(batch, caches, perform, stream, weight_raw)
    assert family['summary']['odds_status'] == 'unsupported'
    assert [x[1] for x in actions[:8]] == ['noop']*4 + ['self']*4
    assert [x[1] for x in actions[-4:]] == ['clean_after']*4
    if zero:
        assert len(actions) == 12 and family['summary']['dose_status'] == 'dose-ineligible'
    else:
        patches = [x for x in actions if x[1]=='patch']
        assert len(actions) == 24 and len(patches)==12
        assert family['summary']['dose_status'] == ('dose-ineligible' if bad_actual else 'eligible')
        assert patches == [(v,'patch',h) for v in range(4)
                           for h in (owner.ORDER[v%3:]+owner.ORDER[:v%3])]


def test_returned_payload_persists_before_control_failure(tmp_path, monkeypatch):
    c = case(0); cache = {'clean':bytes(m.ROW_BYTES)}; model = object()
    payload = {'logits':bytes(m.ROW_BYTES), 'cproj_before':bytes(m.WITNESS_BYTES),
               'cproj_after':b'\x01'+bytes(m.WITNESS_BYTES-1), 'cproj_output':bytes(m.WITNESS_BYTES)}
    def fake_forward(model,ids,kind,head,delivery,observation):
        observation.update({'model_call_return_observed':True,'payload':payload})
        return payload
    monkeypatch.setattr(owner,'run_forward',fake_forward)
    monkeypatch.setattr(owner.prior,'runtime',lambda _: {'fake':True})
    counts = {'submitted':0,'model_returns':0,'returned':0,'retained':0}; activity = {}; completed = []
    with (tmp_path/'journal').open('wb') as log:
        with pytest.raises(RuntimeError,match='no-op changed'):
            owner.perform_action(c,'noop',cache,None,None,model=model,runtime_before={'fake':True},
                                 output=tmp_path,log=log,counts=counts,activity=activity,completed=completed)
    raw = (tmp_path/'fake-0.f32').read_bytes()
    assert raw == b''.join(payload.values()) and counts['retained']==len(raw)
    assert counts['submitted']==counts['model_returns']==counts['returned']==1 and completed==[]
