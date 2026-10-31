"""Bounded numerical comparisons are not semantic alignment or causal evidence."""
from types import SimpleNamespace
import pytest
import torch
from looking_glass.app_reference import ReferenceVector
from looking_glass.app_storage import TraceStore


def source(values=None, *, prompt="first", token=0):
    if values is None: values = torch.tensor([0., -0., 1., 2.])
    return dict(reference_value=values, prompt=prompt,
        coordinate=dict(status="known", module_path="transformer.h.0.mlp", layer_index=0, stack="input", app_forward_index=0, batch_position=0, token_position=token, token_id=4),
        metadata=dict(module_path="transformer.h.0.mlp", layer_index=0, output_path=[], identity={"run_id":"one"}),
        model=dict(family="gpt2", dtype="float32", device="cpu", execution_numerics={"dtype":"float32", "attention_implementation":"eager"}),
        step=dict(app_forward_index=0, run_id="one", input_token_ids=[4], input_attention_mask=[1], use_cache=False))


def test_pin_is_atomic_cloned_and_metadata_is_independent():
    ref=ReferenceVector(); item=source(); before=item["reference_value"].clone()
    summary=ref.pin(item)
    assert summary["total_bytes"] == 16 and summary["width"] == 4
    item["reference_value"].fill_(99); item["prompt"]="changed"
    summary["coordinate"]["token_position"]=99
    assert ref.tensor()["values"] == before.tolist()
    assert ref.summary()["prompt"] == "first"
    assert ref.summary()["coordinate"]["token_position"] == 0
    bad=source(torch.tensor([float("nan")]))
    with pytest.raises(ValueError): ref.pin(bad)
    assert ref.tensor()["values"] == before.tolist()
    ref.clear(); assert ref.summary() is None
    with pytest.raises(ValueError): ref.tensor()


def test_compare_signed_zero_subnormal_large_values_and_float64_metrics():
    tiny=torch.tensor([1],dtype=torch.int32).view(torch.float32)[0].item()
    a=torch.tensor([0.,tiny,3e38,-3e38]); b=torch.tensor([-0.,0.,-3e38,3e38])
    ref=ReferenceVector(); ref.pin(source(a)); report=ref.compare(source(b,prompt="different",token=4),1,3)
    delta=b.double()-a.double()
    assert report["deltas"] == delta[1:].tolist()
    metrics=report["metrics"]
    assert metrics["raw_bit_changed_count"] == 4 and metrics["numeric_nonzero_count"] == 3
    assert metrics["max_abs_delta"] == delta.abs().max().item()
    assert metrics["l1"] == delta.abs().sum().item()
    assert metrics["l2"] == torch.linalg.vector_norm(delta).item()
    assert metrics["cosine"] == pytest.approx(-1.)
    assert report["reference"]["prompt"] == "first" and report["current"]["prompt"] == "different"
    assert torch.equal(a.view(torch.int32), torch.tensor([0.,tiny,3e38,-3e38]).view(torch.int32))
    assert ref.tensor(0,1)["raw_bits"] == ["0x00000000"]
    assert report["current_raw_bits"] == [f"0x{x & 0xffffffff:08x}" for x in b[1:].view(torch.int32).tolist()]


def test_zero_norm_and_identical_nonzero():
    ref=ReferenceVector(); ref.pin(source(torch.zeros(4)))
    result=ref.compare(source(torch.ones(4)))
    assert result["metrics"]["cosine"] is None and result["metrics"]["cosine_reason"]
    ref.pin(source(torch.ones(4))); assert ref.compare(source(torch.ones(4)))["metrics"]["cosine"] == 1.


@pytest.mark.parametrize("field,value", [("family","llama"),("device","cuda:0"),("execution_numerics",{"dtype":"different"})])
def test_model_signature_mismatch(field,value):
    ref=ReferenceVector(); ref.pin(source()); other=source(); other["model"][field]=value
    with pytest.raises(ValueError): ref.compare(other)


@pytest.mark.parametrize("field,value", [("module_path","transformer.h.0.attn"),("layer_index",1),("stack","decoder")])
def test_coordinate_signature_mismatch(field,value):
    ref=ReferenceVector(); ref.pin(source()); other=source(); other["coordinate"][field]=value
    with pytest.raises(ValueError): ref.compare(other)


def test_selector_width_dtype_and_metadata_limits():
    ref=ReferenceVector(); ref.pin(source())
    for other in [source(torch.ones(5)),source(torch.ones(4,dtype=torch.float64))]:
        with pytest.raises(ValueError): ref.compare(other)
    other=source(); other["metadata"]["output_path"]=[0]
    with pytest.raises(ValueError): ref.compare(other)
    for field in ("prompt","metadata"):
        other=source(); other[field] = "x"*(256*1024+1)
        with pytest.raises(ValueError): ref.pin(other)
    with pytest.raises(ValueError): ref.pin(source(torch.ones(262145)))
    assert ref.summary()["width"] == 4


@pytest.mark.parametrize("offset,limit", [(-1,1),(True,1),(5,1),(0,0),(0,4097)])
def test_page_bounds(offset,limit):
    ref=ReferenceVector(); ref.pin(source())
    with pytest.raises(ValueError): ref.tensor(offset,limit)
    with pytest.raises(ValueError): ref.compare(source(),offset,limit)


def test_workspace_preflight():
    ref=ReferenceVector(max_workspace_bytes=65536); ref.pin(source())
    with pytest.raises(ValueError,match="workspace"): ref.compare(source())


def test_store_reference_source_budget_preflights_and_exact_single_snapshot(tmp_path,monkeypatch):
    store=TraceStore(tmp_path)
    meta=dict(identity={"run_id":"one"}, app_forward_index=0,module_path="transformer.h.0.mlp",layer_index=0,
              output_path=[],batch_axis=0,token_axis=1,shape=[1,1,4],saved_shape=[1,1,4],batch_positions=[0],token_positions=[0])
    data=source(); store.publish(dict(model=data["model"],prompt="stored prompt",steps=[data["step"]],events=[SimpleNamespace(value=data["reference_value"].reshape(1,1,4),metadata=lambda:meta)]))
    original=store.intervention_source
    monkeypatch.setattr(store,"intervention_source",lambda *a: pytest.fail("must preflight before cloning"))
    with pytest.raises(ValueError): store.reference_source(0,0,max_bytes=15)
    monkeypatch.setattr(store,"intervention_source",original)
    saved=store.reference_source(0,0,max_bytes=16)
    assert saved["prompt"] == "stored prompt" and saved["metadata"] == meta
    assert set(saved)=={"coordinate","step","model","reference_value","metadata","prompt"}
    saved["reference_value"].zero_(); assert store.tensor(0)["values"] == [0.,-0.,1.,2.]
    store._run["prompt"]="x"*(256*1024+1)
    monkeypatch.setattr(store,"intervention_source",lambda *a: pytest.fail("must preflight metadata before cloning"))
    with pytest.raises(ValueError): store.reference_source(0,0,max_bytes=16)


def test_llama_reference_does_not_require_token_or_forward_alignment():
    first=source(); first["model"]["family"]="llama"
    first["coordinate"]["module_path"]=first["metadata"]["module_path"]="model.layers.0.self_attn"
    first["metadata"]["output_path"]=[0]
    ref=ReferenceVector(); ref.pin(first)
    current=source(prompt="different prompt",token=3); current["model"]=first["model"]
    current["coordinate"]["module_path"]=current["metadata"]["module_path"]="model.layers.0.self_attn"
    current["metadata"]["output_path"]=[0]
    current["coordinate"]["app_forward_index"]=7
    current["coordinate"]["token_id"]=9
    current["step"]["app_forward_index"]=7
    result=ref.compare(current,4,1)
    assert result["deltas"] == [] and result["total"] == 4
    assert result["current"]["coordinate"]["app_forward_index"] == 7
    assert result["reference"]["coordinate"]["app_forward_index"] == 0
    assert result["metrics"]["raw_bit_changed_count"] == 0


@pytest.mark.parametrize("family,path,selector", [("gpt2","arbitrary",[]),("gpt2","transformer.h.1.mlp",[]),("gpt2","transformer.h.0.mlp",[0]),("gpt2","transformer.h.0",[]),("llama","model.layers.0",[0]),("llama","model.layers.0.self_attn",[])])
def test_noncanonical_boundary_or_selector_rejected(family,path,selector):
    item=source(); item["model"]["family"]=family
    item["coordinate"]["module_path"]=item["metadata"]["module_path"]=path
    item["metadata"]["output_path"]=selector
    with pytest.raises(ValueError): ReferenceVector().pin(item)


def test_total_bytes_read_only_property_without_summary_copy(monkeypatch):
    ref=ReferenceVector(); assert ref.total_bytes == 0
    ref.pin(source()); monkeypatch.setattr(ref,"summary",lambda: pytest.fail("must not copy metadata"))
    assert ref.total_bytes == 16
    with pytest.raises(AttributeError): ref.total_bytes = 1
    ref.clear(); assert ref.total_bytes == 0


def test_pin_return_copy_failure_preserves_previous_pin(monkeypatch):
    import looking_glass.app_reference as module
    ref=ReferenceVector(); ref.pin(source()); before=ref.tensor()
    original=module.json.loads
    calls=0
    def fail_return_copy(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2: raise MemoryError("injected return-metadata copy failure")
        return original(*args,**kwargs)
    monkeypatch.setattr(module.json,"loads",fail_return_copy)
    with pytest.raises(MemoryError): ref.pin(source(torch.full((4,),9.),prompt="replacement"))
    assert ref.tensor() == before
    assert ref.summary()["prompt"] == "first"


def test_subnormal_fidelity_under_ftz_in_fresh_process():
    import inspect
    import subprocess
    import sys
    code = "import torch, math\n" + inspect.getsource(source) + '''
from looking_glass.app_reference import ReferenceVector
if not torch.set_flush_denormal(True):
    raise SystemExit(77)
a = torch.tensor([0, -2147483648, 1, -2147483647], dtype=torch.int32).view(torch.float32)
b = torch.tensor([-2147483648, 0, 0, 0], dtype=torch.int32).view(torch.float32)
assert a.double().tolist()[2:] == [0.0, -0.0], "FTZ prerequisite did not flush"
ref = ReferenceVector()
ref.pin(source(a))
tiny = 2.0 ** -149
page = ref.tensor()
assert page["values"] == [0.0, -0.0, tiny, -tiny], page
assert math.copysign(1.0, page["values"][1]) == -1.0
assert page["raw_bits"] == ["0x00000000", "0x80000000", "0x00000001", "0x80000001"]
report = ref.compare(source(b))
assert report["reference_values"] == [0.0, -0.0, tiny, -tiny]
assert report["current_values"] == [-0.0, 0.0, 0.0, 0.0]
assert math.copysign(1.0, report["current_values"][0]) == -1.0
assert report["deltas"][2:] == [-tiny, tiny]
assert report["metrics"]["numeric_nonzero_count"] == 2
assert report["metrics"]["raw_bit_changed_count"] == 4
assert report["metrics"]["max_abs_delta"] == tiny
assert report["metrics"]["l1"] == 2 * tiny
assert report["metrics"]["cosine"] is None
assert report["metrics"]["l2"] == math.sqrt(2.0) * tiny
self_report = ref.compare(source(a))
assert self_report["metrics"]["cosine"] is not None
assert abs(self_report["metrics"]["cosine"] - 1.0) < 1e-15
assert self_report["metrics"]["numeric_nonzero_count"] == 0
assert ref.tensor(4, 1)["values"] == []
assert a.view(torch.int32).tolist() == [0, -2147483648, 1, -2147483647]
assert a.double().tolist()[2:] == [0.0, -0.0], "reference API changed caller FTZ mode"
'''
    child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
    if child.returncode == 77: pytest.skip("FTZ mode unavailable on this CPU")
    assert child.returncode == 0, child.stdout + child.stderr


def test_workspace_budget_checked_before_decode(monkeypatch):
    import looking_glass.agreement_numeric as numeric
    ref=ReferenceVector(max_workspace_bytes=65536+32*4); ref.pin(source())
    monkeypatch.setattr(numeric,"decode_f32_le",lambda *_: pytest.fail("must preflight decode workspace"))
    with pytest.raises(ValueError,match="workspace"): ref.compare(source())
