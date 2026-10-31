"""Full-vocabulary readout diagnostics, never top-k renormalization."""
import math
import pytest
import torch
from looking_glass.app_interventions import _next_token_distribution


class Labels:
    def decode(self, ids): return f'token-{ids[0]}'


def report(first,second):
    return _next_token_distribution(torch.tensor(first,dtype=torch.float32).reshape(1,1,-1),
        torch.tensor(second,dtype=torch.float32).reshape(1,1,-1),Labels())


def test_full_vocab_normalization_not_selected_top_five():
    result=report([0.]*10,[0.]*10)
    assert len(result['rows'])==5
    assert [r['token_id'] for r in result['rows']]==list(range(5))
    assert all(r['baseline_probability']==pytest.approx(.1) for r in result['rows'])
    assert sum(r['baseline_probability'] for r in result['rows'])==pytest.approx(.5)
    assert result['total_variation']==0


def test_alignment_union_ranks_and_full_distribution_distance():
    first=[10,9,8,7,6,0,-1,-2,-3,-4]
    second=[0,-1,-2,-3,-4,10,9,8,7,6]
    result=report(first,second)
    assert [row['token_id'] for row in result['rows']]==list(range(10))
    def softmax(values):
        numerators=[math.exp(x-max(values)) for x in values]
        return [x/sum(numerators) for x in numerators]
    p,q=softmax(first),softmax(second)
    for row in result['rows']:
        i=row['token_id']
        assert row['baseline_logit']==first[i] and row['intervention_logit']==second[i]
        assert row['baseline_probability']==pytest.approx(p[i],rel=1e-14)
        assert row['intervention_probability']==pytest.approx(q[i],rel=1e-14)
        assert row['probability_delta']==pytest.approx(q[i]-p[i],rel=1e-14)
        assert row['baseline_rank']==i+1
        assert row['intervention_rank']==(i+6 if i<5 else i-4)
    assert result['total_variation']==pytest.approx(sum(abs(a-b) for a,b in zip(p,q))/2,rel=1e-14)


def test_exact_ties_use_ascending_token_ids_without_logit_perturbation():
    result=report([2,2,1,1,-0.,0.],[2,2,1,1,-0.,3.])
    assert [r['token_id'] for r in result['rows']]==[0,1,2,3,4,5]
    assert [r['baseline_rank'] for r in result['rows']]==[1,2,3,4,5,6]
    assert result['rows'][4]['baseline_logit']==0.
    assert math.copysign(1.,result['rows'][4]['baseline_logit'])==-1.


def test_huge_finite_logits_and_underflow_retain_finite_log_probabilities():
    limit=torch.finfo(torch.float32).max
    result=report([limit,-limit,0.],[-limit,limit,0.])
    assert result['total_variation']==1.
    rows={r['token_id']:r for r in result['rows']}
    assert rows[0]['baseline_probability']==1. and rows[0]['intervention_probability']==0.
    assert rows[0]['intervention_log_probability']< -1e38
    assert all(math.isfinite(r[k]) for r in result['rows'] for k in ('baseline_log_probability','intervention_log_probability'))


def test_noop_and_constant_shift_distribution_not_logit_identity():
    first=[-5.,-4.,0.,2.,1.,-1.]
    noop=report(first,first)
    shifted=report(first,[x+8 for x in first])
    assert noop['total_variation']==shifted['total_variation']==0.
    for row in shifted['rows']:
        assert row['intervention_logit']-row['baseline_logit']==8.
        assert row['baseline_log_probability']==row['intervention_log_probability']
        assert row['probability_delta']==0.


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_rejection(bad):
    with pytest.raises(ValueError,match='finite'):report([0.,bad],[0.,1.])


@pytest.mark.parametrize('kind',['dtype','rank','empty','mismatch','batch','meta'])
def test_exact_input_schema(kind):
    first=torch.zeros(1,1,3);second=first.clone()
    if kind=='dtype':first=first.double()
    elif kind=='rank':first=first[0]
    elif kind=='empty':first=second=torch.zeros(1,1,0)
    elif kind=='mismatch':second=second[:,:,:2]
    elif kind=='batch':first=second=torch.zeros(2,1,3)
    else:first=first.to('meta')
    with pytest.raises(ValueError):_next_token_distribution(first,second,Labels())


def test_full_readout_position_and_source_bits_unchanged():
    first=torch.tensor([[[3.,1.,0.],[2.,-0.,1.]]]);second=first.clone()
    before=first.view(torch.int32).clone()
    result=_next_token_distribution(first,second,Labels())
    assert result['readout_position']==1 and result['vocabulary_size']==3
    assert torch.equal(first.view(torch.int32),before)
    assert result['rows'][0]['baseline_logit']==2.


def test_label_cap_unavailable_and_resource_failure():
    class Long:
        def decode(self,ids):return '🙂'*1000
    first=torch.zeros(1,1,2)
    result=_next_token_distribution(first,first,Long())
    assert all(len(r['text'])==512 and r['text_truncated'] for r in result['rows'])
    class Bad:
        def decode(self,ids):raise ValueError('display failure')
    assert all(r['text_unavailable'] for r in _next_token_distribution(first,first,Bad())['rows'])
    class Exhausted:
        def decode(self,ids):raise MemoryError('no memory')
    with pytest.raises(MemoryError):_next_token_distribution(first,first,Exhausted())


def test_workspace_budget_is_admitted_before_model_work(tmp_path,monkeypatch):
    from looking_glass.app_runtime import LocalModelRuntime
    from looking_glass.app_storage import TraceStore
    import looking_glass.app_interventions as helper
    runtime=LocalModelRuntime();runtime.load_demo()
    store=TraceStore(tmp_path/'exports');store.publish(runtime.run('Hi',{'max_new_tokens':1}))
    source=store.intervention_source(0,0)
    request=helper.validate_request(runtime,source,'zero',None,None,128*1024**2)
    extra=helper._distribution_workspace_bytes(runtime.model.config.vocab_size)
    old_allowance=request['planned_bytes']-extra
    assert extra>=runtime.model.config.vocab_size*128
    monkeypatch.setattr(helper,'execute',lambda *a:pytest.fail('budget must reject before forwarding'))
    binding=runtime.capture_binding()
    with pytest.raises(ValueError,match='memory allowance'):
        runtime.intervene_fixed_forward(binding,source,'zero',max_pair_bytes=old_allowance)
    assert runtime.capture_binding() is binding
