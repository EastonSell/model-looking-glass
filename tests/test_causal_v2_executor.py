"""CPU development fixtures only; no v2 split, site schedule, or held-out inputs."""
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _isolated(code):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(ROOT)}
    with tempfile.TemporaryDirectory() as prefix:
        return subprocess.run([sys.executable, "-B", "-X", f"pycache_prefix={prefix}", "-c", code],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)


class DevelopmentExecutorTests(unittest.TestCase):
    def test_tap_group_and_cpu_autocast_are_rejected_before_forward(self):
        code = r'''
import torch
from looking_glass.causal_v2_executor import DevelopmentInvalid, _check_model
from looking_glass.model import TinyResidual
m = TinyResidual(width=8, depth=4).eval()
m.blocks[1].output_tap.group = "dense"
try:
    _check_model(m)
except DevelopmentInvalid:
    pass
else:
    raise AssertionError("changed sparse tap group was accepted")
m.blocks[1].output_tap.group = "sparse"
with torch.autocast("cpu", dtype=torch.bfloat16):
    try:
        _check_model(m)
    except DevelopmentInvalid:
        pass
    else:
        raise AssertionError("CPU autocast was accepted")
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cache_shape_mutation_is_rejected_without_any_forward(self):
        code = r'''
import torch
from looking_glass.causal_v2_executor import (DevelopmentBudget, DevelopmentClean, DevelopmentTarget,
    DevelopmentInvalid, _CACHE_TOKEN, _snapshot_sha256, _target_sha256, _validate_clean, _validate_target, input_sha256)
from looking_glass.causal_v2_model_identity import state_sha256
from looking_glass.causal_v2_runtime import establish_numeric_policy
from looking_glass.model import TinyResidual
from looking_glass.probes import TraceEvent
policy=establish_numeric_policy(); model=TinyResidual(width=8,depth=4).eval(); budget=DevelopmentBudget(40)
x=torch.tensor([[2.,-.25,.5,.125,-.75,.2,-.3,.4]])
def events(run): return tuple(TraceEvent(run,f'block.{i}.output',i,0,(1,8),'torch.float32','cpu',torch.zeros((1,8))) for i in range(4))
first,second=events('a'),events('b'); clean_logits=torch.zeros((1,2)); repeat_logits=clean_logits.clone()
clean=DevelopmentClean(first,second,clean_logits,repeat_logits,input_sha256(x),state_sha256(model),
   _snapshot_sha256(first,second,clean_logits,repeat_logits),model,budget,policy,_CACHE_TOKEN)
path='blocks.2.activation_tap'
metadata={'module_path':path,'module_aliases':[path],'call_site_known':False,'forward_index':0,
          'invocation':0,'batch_index':None,'token_index':None,'output_shape':[1,8],
          'selected_shape':[1,8],'dtype':'torch.float32','device':'cpu'}
target_value=torch.zeros((1,8))
target=DevelopmentTarget('block.2.activation',target_value,metadata,
                         _target_sha256('block.2.activation',target_value,metadata),clean,_CACHE_TOKEN)
target.value.resize_(2,4)
try: _validate_target(target,'block.2.activation',clean)
except DevelopmentInvalid: pass
else: raise AssertionError('reshaped target accepted')
target.value.resize_(1,8)
target.value.data=target.value.view(torch.int32)
try: _validate_target(target,'block.2.activation',clean)
except DevelopmentInvalid: pass
else: raise AssertionError('wrong target dtype accepted')
target.value.data=target.value.view(torch.float32)
clean.clean_trace[0].value.resize_(2,4)
try: _validate_clean(clean,model,x,budget,policy)
except DevelopmentInvalid: pass
else: raise AssertionError('reshaped trace accepted')
clean.clean_trace[0].value.resize_(1,8)
clean.clean_logits.resize_(2,1)
try: _validate_clean(clean,model,x,budget,policy)
except DevelopmentInvalid: pass
else: raise AssertionError('reshaped logits accepted')
assert budget.submissions==budget.root_forwards==0
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_input_identity_uses_explicit_big_endian_float32_bits(self):
        code = r'''
import hashlib,struct,torch
from looking_glass.causal_v2_executor import DevelopmentInvalid,input_sha256
bits=[0x40000000,0xbe800000,0x3f000000,0x3e000000,0xbf400000,0x3e4ccccd,0xbe99999a,0x3ecccccd]
x=torch.tensor([[2.,-.25,.5,.125,-.75,.2,-.3,.4]],dtype=torch.float32,device='cpu')
expected=hashlib.sha256(b'MLG-v2-input-f32be-v1\0'+struct.pack('>II8I',1,8,*bits)).hexdigest()
assert input_sha256(x)==expected
positive=x.clone(); positive.view(torch.int32)[0,0]=0
negative=x.clone(); negative.view(torch.int32)[0,0]=-2147483648
assert input_sha256(positive)!=input_sha256(negative)
for bad in (torch.zeros(8), torch.full((1,8),float('nan')), x.refine_names('batch','hidden')):
    try: input_sha256(bad)
    except DevelopmentInvalid: pass
    else: raise AssertionError('unsupported input identity accepted')
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reuse_rejects_cross_input_model_and_target_mutation_before_forward(self):
        code = r'''
import torch
from looking_glass.causal_v2_executor import (DevelopmentBudget, DevelopmentInvalid,
    prepare_development_clean, prepare_development_target, run_development_active)
from looking_glass.causal_v2_runtime import establish_numeric_policy
from looking_glass.model import TinyResidual
policy=establish_numeric_policy()
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(41977)
    model=TinyResidual(width=8,depth=4).eval()
x=torch.tensor([[2.,-.25,.5,.125,-.75,.2,-.3,.4]],dtype=torch.float32,device='cpu')
budget=DevelopmentBudget(40)
clean=prepare_development_clean(model,x,budget,policy)
target=prepare_development_target(model,x,'block.2.activation',clean,budget,policy)
def reject(model_arg,input_arg,target_arg):
    try: run_development_active(model_arg,input_arg,'block.2.activation',.25,clean,target_arg,budget,policy)
    except DevelopmentInvalid: pass
    else: raise AssertionError('mutated or cross-bound source was accepted')
    assert budget.submissions==budget.root_forwards==3
other=x.clone(); other[0,0]=-2.
reject(model,other,target)
original=target.value.clone(); target.value[0,0] += 1
reject(model,x,target)
target.value.copy_(original)
old=target.metadata['invocation']; target.metadata['invocation']=False
reject(model,x,target)
target.metadata['invocation']=old
with torch.no_grad():
    original_weight=model.head.weight.clone(); model.head.weight[0,0] += 1
reject(model,x,target)
with torch.no_grad(): model.head.weight.copy_(original_weight)
other_model=TinyResidual(width=8,depth=4).eval(); other_model.load_state_dict(model.state_dict())
reject(other_model,x,target)
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reuses_clean_and_target_captures_across_two_strengths(self):
        code = r'''
import torch
from looking_glass.causal_v2_executor import (DevelopmentBudget, DevelopmentInvalid,
    prepare_development_clean, prepare_development_target, run_development_active)
from looking_glass.causal_v2_runtime import establish_numeric_policy, float32_exact_equal
from looking_glass.model import TinyResidual
policy=establish_numeric_policy()
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(41977)
    model=TinyResidual(width=8,depth=4).eval()
x=torch.tensor([[2.,-.25,.5,.125,-.75,.2,-.3,.4]],dtype=torch.float32,device='cpu')
budget=DevelopmentBudget(40)
clean=prepare_development_clean(model,x,budget,policy)
target=prepare_development_target(model,x,'block.2.activation',clean,budget,policy)
assert budget.submissions==budget.root_forwards==3
low=run_development_active(model,x,'block.2.activation',.25,clean,target,budget,policy)
high=run_development_active(model,x,'block.2.activation',.75,clean,target,budget,policy)
assert budget.submissions==budget.root_forwards==35
assert float32_exact_equal(low['clean_logits'],high['clean_logits'])
assert len(low['plans'])==len(high['plans'])==13
assert len(clean.input_sha256)==len(clean.model_state_sha256)==64
clean.clean_trace[0].value[0,0] += 1
try: run_development_active(model,x,'block.2.activation',.25,clean,target,budget,policy)
except DevelopmentInvalid: pass
else: raise AssertionError('mutated clean snapshot accepted')
assert budget.submissions==budget.root_forwards==35
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fresh_cache_prefix_ignores_timestamp_valid_stale_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture_module.py"
            source.write_text("VALUE='old'\n", encoding="utf-8")
            original = source.stat()
            cache = root / "__pycache__"
            cache.mkdir()
            py_compile.compile(str(source), cfile=str(cache / f"fixture_module.{sys.implementation.cache_tag}.pyc"),
                               doraise=True)
            source.write_text("VALUE='new'\n", encoding="utf-8")
            os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(root)}
            code = "import fixture_module; print(fixture_module.VALUE)"
            stale = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, env=env,
                                   capture_output=True, text=True, timeout=15)
            fresh = subprocess.run([sys.executable, "-B", "-X", f"pycache_prefix={root / 'empty'}",
                                    "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual((stale.returncode, stale.stdout.strip()), (0, "old"))
            self.assertEqual((fresh.returncode, fresh.stdout.strip()), (0, "new"))

    def test_local_and_global_backward_hooks_are_rejected_before_forward(self):
        code = r'''
import torch
from looking_glass.causal_v2_executor import DevelopmentInvalid, _check_hooks
from looking_glass.model import TinyResidual
model=TinyResidual(width=8,depth=4).eval()
hooks=[lambda: model.register_full_backward_hook(lambda module,inputs,outputs: None),
       lambda: torch.nn.modules.module.register_module_full_backward_hook(lambda module,inputs,outputs: None)]
for register in hooks:
    handle=register()
    try:
        try: _check_hooks(model)
        except DevelopmentInvalid: pass
        else: raise AssertionError('backward hook was accepted')
    finally: handle.remove()
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_sparse_boundary_before_hidden_target_must_match(self):
        code = r'''
import types,torch
from looking_glass.causal_v2_executor import DevelopmentInvalid, _check_upstream
from looking_glass.causal_v2_runtime import establish_numeric_policy
policy=establish_numeric_policy()
clean=[types.SimpleNamespace(value=torch.zeros((1,8))) for _ in range(4)]
changed=[types.SimpleNamespace(value=e.value.clone()) for e in clean]
changed[1].value[0,0]=1.0
try: _check_upstream(clean,changed,3,policy)
except DevelopmentInvalid: pass
else: raise AssertionError('pre-target mismatch was accepted')
_check_upstream(clean,changed,1,policy)
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_private_capture_metadata_requires_exact_typed_coordinates(self):
        code = r'''
from looking_glass.causal_v2_executor import DevelopmentInvalid, _check_capture_metadata
path='blocks.2.activation_tap'
value={'module_path':path,'module_aliases':[path],'call_site_known':False,
       'forward_index':0,'invocation':0,'batch_index':None,'token_index':None,
       'output_shape':[1,8],'selected_shape':[1,8],'dtype':'torch.float32','device':'cpu'}
_check_capture_metadata(value,path)
for field,bad in [('invocation',False),('output_shape',[True,8]),('module_path','blocks.1.activation_tap')]:
    changed=dict(value); changed[field]=bad
    try: _check_capture_metadata(changed,path)
    except DevelopmentInvalid: pass
    else: raise AssertionError('invalid private capture metadata accepted')
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_aliased_model_tree_is_rejected_before_forward(self):
        code = r'''
import torch
from looking_glass.causal_v2_executor import DevelopmentInvalid, _check_model
from looking_glass.model import TinyResidual
model=TinyResidual(width=8,depth=4).eval()
model.extra_alias=model.blocks[0].output_tap
try: _check_model(model)
except DevelopmentInvalid: pass
else: raise AssertionError('aliased module tree was accepted')
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fabricated_conditions_deliver_all_ordered_controls(self):
        code = r'''
import torch
from looking_glass.causal_v2_executor import DevelopmentBudget, run_development_condition
from looking_glass.causal_v2_runtime import establish_numeric_policy, float32_exact_equal
from looking_glass.model import TinyResidual
policy=establish_numeric_policy()
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(41977)
    model=TinyResidual(width=8, depth=4).eval()
for sign in (1,-1):
    x=torch.tensor([[2.0*sign,-0.25,0.5,0.125,-0.75,0.2,-0.3,0.4]],dtype=torch.float32,device='cpu')
    for block in (1,2,3):
        for operation in ('branch.0','activation','merge'):
            for strength in (0.25,0.75):
                budget=DevelopmentBudget(19)
                r=run_development_condition(model,x,f'block.{block}.{operation}',strength,budget,policy)
                c=r['condition']; clean=r['clean_logits']; perturbed=c.perturbed_logits
                assert budget.submissions==budget.root_forwards==19
                assert len(r['plans'])==13
                assert sum(p['metadata']['capture_bytes'] for p in r['plans'])==16*32
                assert len(r['private_integrity']['clean_target_bits'])==8
                assert len(r['private_integrity']['perturbed_target_bits'])==8
                assert r['private_integrity']['clean_capture_metadata']['module_path'].startswith(f'blocks.{block}.')
                assert r['private_integrity']['perturbed_capture_metadata']['invocation']==0
                for site in (1,2,3):
                    assert float32_exact_equal(c.clean_to_perturbed_logits[site-1],clean if site>=block else perturbed)
                    assert float32_exact_equal(c.perturbed_to_clean_logits[site-1],perturbed if site>=block else clean)
                    assert float32_exact_equal(c.null_logits[site-1],perturbed)
                    assert float32_exact_equal(c.identity_logits[site-1],perturbed)
                assert float32_exact_equal(c.upstream_logits,perturbed)
                assert float32_exact_equal(c.clean_verify_logits,clean)
                assert all(len(event.value.view(torch.int32).reshape(-1))==8 for event in c.perturbed_trace)
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_budget_invalid_fixture_and_foreign_hook_fail_without_leaks(self):
        code = r'''
import torch
from looking_glass.causal_v2_executor import DevelopmentBudget, DevelopmentInvalid, run_development_condition
from looking_glass.causal_v2_runtime import establish_numeric_policy
from looking_glass.model import TinyResidual
policy=establish_numeric_policy()
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(41977)
    model=TinyResidual(width=8,depth=4).eval()
x=torch.tensor([[2.,-.25,.5,.125,-.75,.2,-.3,.4]],dtype=torch.float32,device='cpu')
budget=DevelopmentBudget(2)
try: run_development_condition(model,x,'block.2.activation',.25,budget,policy)
except DevelopmentInvalid: pass
else: raise AssertionError('budget was not enforced')
assert budget.submissions==budget.root_forwards==2
assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
with torch.no_grad(): assert model(x).shape==(1,2)
for bad in (torch.zeros(1,8), x.clone().fill_(float('nan'))):
    budget=DevelopmentBudget(19)
    try: run_development_condition(model,bad,'block.2.activation',.25,budget,policy)
    except DevelopmentInvalid: pass
    else: raise AssertionError('non-development fixture was accepted')
    assert budget.submissions==budget.root_forwards==0
handle=model.blocks[2].output_tap.register_forward_hook(lambda module,args,result: result)
try:
    budget=DevelopmentBudget(19)
    try: run_development_condition(model,x,'block.2.activation',.25,budget,policy)
    except DevelopmentInvalid: pass
    else: raise AssertionError('foreign hook accepted')
    assert budget.submissions==budget.root_forwards==0
finally: handle.remove()
assert not torch.cuda.is_initialized()
'''
        result = _isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
