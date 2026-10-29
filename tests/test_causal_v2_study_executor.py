"""Development-only callback exercises; no reserved v2 cases are generated."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StudyExecutorDevelopmentTests(unittest.TestCase):
    def test_fixed_cpu_runtime_drift_poisoned_before_model_or_submission(self):
        code = r'''
import os
from unittest.mock import patch
import torch
from looking_glass.causal_v2_custodian_run import ForwardBudget
from looking_glass.causal_v2_runtime import establish_numeric_policy
from looking_glass.causal_v2_runtime_preflight import inspect_backend_settings
from looking_glass.causal_v2_study_executor import StudyExecutionError, StudyExecutor
policy=establish_numeric_policy(); original=torch.get_default_dtype()
for setting in ('dtype','mask','cuda'):
    executor=StudyExecutor(policy,backend_settings=inspect_backend_settings()); budget=ForwardBudget(5)
    try:
        if setting=='dtype': torch.set_default_dtype(torch.float64)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'0' if setting=='mask' else '-1'}), \
             patch('torch.cuda.is_initialized', return_value=(setting=='cuda')), \
             patch('looking_glass.causal_v2_study_executor.TinyResidual',
                   side_effect=AssertionError('model construction was not blocked')):
            try: executor.begin_model('discovery',31201,'a'*64,budget)
            except StudyExecutionError: pass
            else: raise AssertionError(f'{setting} drift was accepted')
            assert executor._failed and budget.submissions==budget.root_forwards==0
    finally:
        torch.set_default_dtype(original)
    assert executor.close() is None
assert not torch.cuda.is_initialized()
'''
        self._run(code)

    def test_fixed_cpu_runtime_drift_after_call_is_detected(self):
        code = r'''
import os, torch
from unittest.mock import patch
from looking_glass.causal_v2_executor import DevelopmentBudget, _forward
from looking_glass.causal_v2_runtime import establish_numeric_policy
from looking_glass.causal_v2_runtime_preflight import inspect_backend_settings
from looking_glass.causal_v2_study_executor import StudyExecutionError, StudyExecutor
policy=establish_numeric_policy(); executor=StudyExecutor(policy,backend_settings=inspect_backend_settings())
class FakeModel:
    def __call__(self,*args,**kwargs):
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        return torch.zeros((1,2),device='cpu')
budget=DevelopmentBudget(5)
try:
    with patch('looking_glass.causal_v2_executor._check_hooks'):
        try: _forward(FakeModel(),None,budget,policy,backend_check=executor._check_backend)
        except StudyExecutionError: pass
        else: raise AssertionError('post-call CPU mask drift was accepted')
    assert budget.submissions==budget.root_forwards==1
finally:
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
assert not torch.cuda.is_initialized()
'''
        self._run(code)

    def test_backend_snapshot_is_required_and_changes_poison_callback(self):
        code = r'''
import torch
from looking_glass.causal_v2_custodian_run import ForwardBudget
from looking_glass.causal_v2_runtime import establish_numeric_policy
from looking_glass.causal_v2_runtime_preflight import inspect_backend_settings
from looking_glass.causal_v2_study_executor import StudyExecutionError, StudyExecutor
policy=establish_numeric_policy(); snapshot=inspect_backend_settings()
bad={**snapshot,'mkldnn_enabled': not snapshot['mkldnn_enabled']}
try: StudyExecutor(policy,backend_settings=bad)
except StudyExecutionError: pass
else: raise AssertionError('incorrect backend attestation accepted')
executor=StudyExecutor(policy,backend_settings=snapshot); budget=ForwardBudget(100)
snapshot['thread_environment']['OMP_NUM_THREADS']='mutated caller copy'
executor._check_backend()  # The validated snapshot is owned, not aliased.
torch.set_float32_matmul_precision('medium' if snapshot['float32_matmul_precision']!='medium' else 'highest')
try:
    try: executor.begin_model('discovery',31201,'a'*64,budget)
    except StudyExecutionError: pass
    else: raise AssertionError('changed backend accepted')
    assert executor._failed and budget.submissions==budget.root_forwards==0
finally:
    torch.set_float32_matmul_precision(snapshot['float32_matmul_precision'])
assert executor.close() is None and not torch.cuda.is_initialized()
'''
        self._run(code)

    def test_input_construction_failure_is_counted_and_poisoned(self):
        code = r'''
import os,sys,threading,types
from unittest.mock import patch
from looking_glass.causal_v2_executor import DevelopmentBudget
custodian=types.ModuleType('looking_glass.causal_v2_custodian_run'); custodian.ForwardBudget=DevelopmentBudget
sys.modules[custodian.__name__]=custodian
from looking_glass import causal_v2_study_executor as study
budget=DevelopmentBudget(100); executor=object.__new__(study.StudyExecutor)
executor._owner_pid=os.getpid(); executor._owner_thread=threading.current_thread()
executor._closed=executor._failed=executor._active=False
executor._check_backend=lambda:None
executor._model=object(); executor._budget=budget; executor._nodes=[]; executor._case_bytes=None
executor._seed=17; executor._phase='discovery'; executor._state_sha='a'*64
executor._policy=types.SimpleNamespace(verify=lambda:None); executor._clean={}
assignment={'cell_id':'v2-m17-c00','model_seed':17,'injection_node':'block.1.branch.0'}
cases=[{'case_id':'fake-case','index':0}]; raw=tuple(study.canonical_json(c) for c in cases)
with patch.object(study,'_CASE_COUNT',1), \
     patch.object(executor,'_validate_cell',return_value=(raw,((2.,0.,0.,0.,0.,0.,0.,0.),))), \
     patch.object(study,'_tensor_from_values',side_effect=RuntimeError('fabricated input failure'),create=True), \
     patch.object(study,'_prepare_clean',side_effect=AssertionError('input was not constructed')):
    try: executor.run_discovery(assignment,cases,object(),budget)
    except RuntimeError: pass
    else: raise AssertionError('input failure was accepted')
assert budget.submissions==1 and budget.root_forwards==0 and executor._failed
'''
        self._run(code)

    def test_selection_snapshot_is_bounded_before_serialization(self):
        code = r'''
import sys,types
from looking_glass.causal_v2_executor import DevelopmentBudget
custodian=types.ModuleType('looking_glass.causal_v2_custodian_run'); custodian.ForwardBudget=DevelopmentBudget
sys.modules[custodian.__name__]=custodian
from looking_glass import causal_v2_study_executor as study
small={'first_changed':None,'measurements':[{'value':[0.0,1.0]}]}
copy=study._bounded_selection(small)
assert copy==small and copy is not small and copy['measurements'] is not small['measurements']
cycle=[]; cycle.append(cycle)
for bad in ({'value':'x'*(600*1024)}, {'value':[0]*257}, {'value':cycle}, {'value':float('nan')}):
    try: study._bounded_selection(bad)
    except study.StudyExecutionError: pass
    else: raise AssertionError('unbounded or invalid selection was accepted')
'''
        self._run(code)

    def test_full_fabricated_selection_fits_snapshot_bound(self):
        code = r'''
import sys,types
import torch
from looking_glass.causal_v2_executor import DevelopmentBudget
custodian=types.ModuleType('looking_glass.causal_v2_custodian_run'); custodian.ForwardBudget=DevelopmentBudget
sys.modules[custodian.__name__]=custodian
from looking_glass import causal_v2_study_executor as study
from looking_glass.causal_v2_runtime import establish_numeric_policy
from looking_glass.causal_v2_selection import DiscoveryCase,DiscoveryCondition,freeze_discovery
from looking_glass.probes import TraceEvent
policy=establish_numeric_policy()
def trace(run,changed):
    result=[]
    for i in range(4):
        bits=torch.tensor([0x3f6dcbaa,0x3eaaaaab,0x3f123456,0x3dabcdef,
                           0x3f345678,0x00800001,0x00000001,0x3f111111],dtype=torch.int32)
        if changed and i>=2: bits[0]=0x3f6dcba9
        result.append(TraceEvent(run,f'block.{i}.output',i,0,(1,8),'torch.float32','cpu',bits.view(torch.float32).reshape(1,8)))
    return tuple(result)
rows=[]
for i in range(16):
    clean=trace(f'c{i}',False); repeat=trace(f'r{i}',False)
    original=torch.tensor([[.123456789,-.234567891]],dtype=torch.float32)
    changed=torch.tensor([[.345678912,-.456789123]],dtype=torch.float32)
    conditions=[]
    for strength in (.25,.75):
        conditions.append(DiscoveryCondition(strength,trace(f'p{i}-{strength}',True),changed,
          (changed,original,original),(original,changed,changed),(changed,)*3,(changed,)*3,changed,original))
    rows.append(DiscoveryCase(i,1 if i%2==0 else -1,clean,repeat,original,original,tuple(conditions)))
selection=freeze_discovery(tuple(rows),numeric_policy=policy,pairing_verified=True,private_controls_verified=True)
result=study._bounded_selection(selection)
assert result==selection and len(study.canonical_json(result)) < study._MAX_SELECTION_BYTES
assert not torch.cuda.is_initialized()
'''
        self._run(code)

    def test_reentrant_failure_prevents_outer_callback_success(self):
        code = r'''
import os,sys,threading,types
from looking_glass.causal_v2_executor import DevelopmentBudget
custodian=types.ModuleType('looking_glass.causal_v2_custodian_run'); custodian.ForwardBudget=DevelopmentBudget
sys.modules[custodian.__name__]=custodian
from looking_glass import causal_v2_study_executor as study
class Fake:
    _check_owner=study.StudyExecutor._check_owner
    def _check_backend(self): pass
    @study._guarded
    def call(self,nested=False):
        if not nested:
            try: self.call(True)
            except study.StudyExecutionError: pass
        return 'success'
executor=Fake(); executor._owner_pid=os.getpid(); executor._owner_thread=threading.current_thread()
executor._closed=executor._failed=executor._active=False
try: executor.call()
except study.StudyExecutionError: pass
else: raise AssertionError('reentrant failure was masked by outer callback')
assert executor._failed is True
'''
        self._run(code)

    def test_prediction_references_are_ordered_and_share_one_digest(self):
        code = r'''
import copy,sys,types
from unittest.mock import patch
from looking_glass.causal_v2_executor import DevelopmentBudget
custodian=types.ModuleType('looking_glass.causal_v2_custodian_run'); custodian.ForwardBudget=DevelopmentBudget
sys.modules[custodian.__name__]=custodian
from looking_glass import causal_v2_study_executor as study
cases=[{'case_id':'fake-0','model_seed':17,'index':0},{'case_id':'fake-1','model_seed':17,'index':1}]
assignment={'cell_id':'v2-m17-c00','model_seed':17,'injection_node':'private'}
groups=[]
for case in cases:
    for strength in (.25,.75):
        rows=[{'model_seed':17,'cell_id':assignment['cell_id'],'case_id':case['case_id'],'strength':strength,
               'site':site,'polarity':1 if case['index']==0 else -1,'polarity_prediction':0.,
               'constant_prediction':0.,'zero_prediction':0.} for site in (1,2,3)]
        groups.append({'reference':f"sha256:{'a'*64}:{3*len(groups)}:3",'rows':rows})
with patch.object(study,'_CASE_COUNT',2), patch.object(study,'_MODEL_SEEDS',(17,)), \
     patch.object(study,'_CELLS_PER_MODEL',2):
    assert study._prediction_refs(groups,cases,assignment)==tuple(item['reference'] for item in groups)
    for change in ('offset','digest','leading-zero','shifted-all'):
        invalid=copy.deepcopy(groups)
        if change=='shifted-all':
            for i, group in enumerate(invalid): group['reference']=f"sha256:{'a'*64}:{12+3*i}:3"
        else: invalid[1]['reference']=(f"sha256:{'b'*64}:3:3" if change=='digest' else
                                 f"sha256:{'a'*64}:6:3" if change=='offset' else
                                 f"sha256:{'a'*64}:03:3")
        try: study._prediction_refs(invalid,cases,assignment)
        except study.StudyExecutionError: pass
        else: raise AssertionError(f'bad prediction reference accepted: {change}')
'''
        self._run(code)

    def test_signed_zero_duplicate_case_inputs_are_rejected_before_forward(self):
        code = r'''
import sys,types
from unittest.mock import patch
import torch
from looking_glass.causal_v2_executor import DevelopmentBudget
custodian=types.ModuleType('looking_glass.causal_v2_custodian_run'); custodian.ForwardBudget=DevelopmentBudget
sys.modules[custodian.__name__]=custodian
from looking_glass import causal_v2_study_executor as study
executor=object.__new__(study.StudyExecutor)
executor._model=object(); executor._budget=object(); executor._nodes=[]; executor._case_bytes=None
executor._seed=17; executor._phase='discovery'; executor._state_sha='a'*64; executor._policy=None
assignment={'cell_id':'v2-m17-c00','model_seed':17,'injection_node':'block.1.branch.0'}
cell=types.SimpleNamespace(cell_id=assignment['cell_id'],model_seed=17,phase='discovery',model_state_sha256='a'*64)
cases=[{'index':i} for i in range(3)]
values=[(1.,0.,0.,0.,0.,0.,0.,0.),(-1.,0.,0.,0.,0.,0.,0.,0.),(1.,-0.,0.,0.,0.,0.,0.,0.)]
with patch.object(study,'_CASE_COUNT',3), patch.object(study,'_case_values',side_effect=values), \
     patch.object(study,'state_sha256',return_value='a'*64):
    try: executor._validate_cell(assignment,cases,cell,executor._budget)
    except study.StudyExecutionError: pass
    else: raise AssertionError('signed-zero duplicate inputs were accepted')
assert not torch.cuda.is_initialized()
'''
        self._run(code)

    def test_assignment_mutation_cannot_mix_private_targets(self):
        code = r'''
import sys,types
from unittest.mock import patch
import torch
from looking_glass.causal_v2_executor import DevelopmentBudget,input_sha256
custodian=types.ModuleType('looking_glass.causal_v2_custodian_run'); custodian.ForwardBudget=DevelopmentBudget
sys.modules[custodian.__name__]=custodian
from looking_glass import causal_v2_study_executor as study
budget=DevelopmentBudget(100)
executor=object.__new__(study.StudyExecutor)
executor._model=object(); executor._budget=budget; executor._nodes=[]; executor._case_bytes=None
executor._seed=17; executor._phase='discovery'; executor._state_sha='a'*64
executor._policy=types.SimpleNamespace(verify=lambda:None); executor._clean={}
assignment={'cell_id':'v2-m17-c00','model_seed':17,'injection_node':'block.1.branch.0'}
cases=[{'case_id':'dev-case','index':0}]; values=((2.,0.,0.,0.,0.,0.,0.,0.),)
raw=tuple(study.canonical_json(case) for case in cases)
seen=[]
def clean(model,value,budget,policy,**kwargs):
    budget.submissions+=1; budget.root_forwards+=2
    trace=(types.SimpleNamespace(run_id='clean-id'),)
    return types.SimpleNamespace(input_sha256=input_sha256(value),clean_trace=trace,repeat_trace=trace,
                                 clean_logits=None,repeat_logits=None)
def target(model,value,node,*args,**kwargs):
    seen.append(node); assignment['injection_node']='block.2.activation'
    budget.root_forwards+=1
    return object()
def active(model,value,node,strength,clean,*args,**kwargs):
    seen.append(node)
    assert node=='block.1.branch.0', 'mixed target reached active execution'
    budget.submissions+=15; budget.root_forwards+=16
    condition=types.SimpleNamespace(perturbed_trace=(types.SimpleNamespace(run_id=f'changed-{strength}'),))
    return {'condition':condition,'clean_trace':clean.clean_trace}
with patch.object(study,'_CASE_COUNT',1), \
     patch.object(executor,'_validate_cell',return_value=(raw,values)), \
     patch.object(study,'_prepare_clean',side_effect=clean), \
     patch.object(study,'_prepare_target',side_effect=target), \
     patch.object(study,'_run_active',side_effect=active), \
     patch.object(study,'append_executor_result'), patch.object(study,'state_sha256',return_value='a'*64), \
     patch.object(study,'freeze_discovery',return_value={}):
    try: executor._run_cell(assignment,cases,object(),budget,None)
    except study.StudyExecutionError: pass
    else: raise AssertionError('changed caller assignment was accepted')
assert all(node=='block.1.branch.0' for node in seen)
assert executor._nodes==[]
assert not torch.cuda.is_initialized()
'''
        self._run(code)

    def test_heldout_prediction_mutation_invalidates_owned_snapshot(self):
        code = r'''
import sys,types
from unittest.mock import patch
import torch
from looking_glass.causal_v2_executor import DevelopmentBudget,input_sha256
custodian=types.ModuleType('looking_glass.causal_v2_custodian_run'); custodian.ForwardBudget=DevelopmentBudget
sys.modules[custodian.__name__]=custodian
from looking_glass import causal_v2_study_executor as study
budget=DevelopmentBudget(100); executor=object.__new__(study.StudyExecutor)
executor._model=object(); executor._budget=budget; executor._nodes=[]; executor._case_bytes=None
executor._seed=17; executor._phase='heldout'; executor._state_sha='a'*64
executor._policy=types.SimpleNamespace(verify=lambda:None); executor._clean={}
assignment={'cell_id':'v2-m17-c00','model_seed':17,'injection_node':'block.1.branch.0'}
cases=[{'case_id':'fake-case','model_seed':17,'index':0}]
groups=[]
for strength in (.25,.75):
    rows=[{'model_seed':17,'cell_id':assignment['cell_id'],'case_id':'fake-case','strength':strength,
           'site':site,'polarity':1,'polarity_prediction':0.,'constant_prediction':0.,'zero_prediction':0.}
          for site in (1,2,3)]
    groups.append({'reference':f"sha256:{'a'*64}:{3*len(groups)}:3",'rows':rows})
expected=[group['reference'] for group in groups]; saved=[]
values=((2.,0.,0.,0.,0.,0.,0.,0.),); raw=tuple(study.canonical_json(c) for c in cases)
def clean(model,value,budget,policy,**kwargs):
    budget.submissions+=1; budget.root_forwards+=2
    trace=(types.SimpleNamespace(run_id='clean-id'),)
    return types.SimpleNamespace(input_sha256=input_sha256(value),clean_trace=trace,repeat_trace=trace,
                                 clean_logits=None,repeat_logits=None)
def target(*args,**kwargs):
    groups[0]['reference']=f"sha256:{'b'*64}:0:3"
    budget.root_forwards+=1
    return object()
def active(model,value,node,strength,clean,*args,**kwargs):
    budget.submissions+=15; budget.root_forwards+=16
    condition=types.SimpleNamespace(perturbed_trace=(types.SimpleNamespace(run_id=f'changed-{strength}'),))
    return {'condition':condition,'clean_trace':clean.clean_trace}
def append(cell,result,**kwargs): saved.append(kwargs['prediction_ref'])
with patch.object(study,'_CASE_COUNT',1), patch.object(study,'_MODEL_SEEDS',(17,)), \
     patch.object(executor,'_validate_cell',return_value=(raw,values)), \
     patch.object(study,'_prepare_clean',side_effect=clean), patch.object(study,'_prepare_target',side_effect=target), \
     patch.object(study,'_run_active',side_effect=active), patch.object(study,'append_executor_result',side_effect=append), \
     patch.object(study,'state_sha256',return_value='a'*64):
    try: executor._run_cell(assignment,cases,object(),budget,groups)
    except study.StudyExecutionError: pass
    else: raise AssertionError('mutated prediction record was accepted')
assert saved==expected and executor._nodes==[]
assert not torch.cuda.is_initialized()
'''
        self._run(code)

    def _run(self, code):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(ROOT)}
        with tempfile.TemporaryDirectory() as prefix:
            process = subprocess.run([sys.executable, "-B", "-X", f"pycache_prefix={prefix}", "-c", code],
                                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_model_phase_reuses_clean_observations_across_cells(self):
        code = r'''
from unittest.mock import patch
import torch
from looking_glass.causal_v2_custodian_run import ForwardBudget
from looking_glass.causal_v2_executor import _check_input
from looking_glass.causal_v2_evidence_adapter import CellEvidence
from looking_glass.causal_v2_model_identity import state_sha256
from looking_glass.causal_v2_runtime import establish_numeric_policy
from looking_glass.causal_v2_runtime_preflight import inspect_backend_settings
from looking_glass.model import TinyResidual

# The two inputs are inventoried reviewer vectors outside the v2 domain.
from looking_glass import causal_v2_study_executor as study
policy=establish_numeric_policy()
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(41977)
    expected=state_sha256(TinyResidual(width=8,depth=4).eval())
class Writer:
    def publish_chunk(self,*args): raise AssertionError('test does not publish')
writer=Writer()
budget=ForwardBudget(200)
import atexit
atexit.register(lambda: print('development_root_forwards=',budget.root_forwards))
cases=[{'case_id':f'v2-m41977-d{i:02d}','model_seed':41977,'split':'discovery',
        'index':i,'values':[first,-.25,.5,.125,-.75,.2,-.3,.4]}
       for i,first in enumerate((2.,-2.))]
def fake_case_values(case, phase, seed, index):
    assert case['index']==index and case['model_seed']==seed and phase=='discovery'
    return tuple(case['values'])
def freeze(items, **kwargs):
    assert len(items)==2
    assert all(not hasattr(item,'injection_node') for item in items)
    return {'first_changed': 2, 'magnitude': 2}
with patch.object(study,'_MODEL_SEEDS',(41977,)), patch.object(study,'_CASE_COUNT',2), \
     patch.object(study,'_CELLS_PER_MODEL',2), \
     patch.object(study,'_NODES',('block.1.branch.0','block.2.activation')), \
     patch.object(study,'_study_input_check',_check_input), \
     patch.object(study,'_case_values',side_effect=fake_case_values), \
     patch.object(study,'freeze_discovery',side_effect=freeze):
    executor=study.StudyExecutor(policy,backend_settings=inspect_backend_settings())
    executor.begin_model('discovery',41977,expected,budget)
    for index,target in enumerate(study._NODES):
        cell_id=f'v2-m41977-c{index:02d}'
        assignment={'cell_id':cell_id,'model_seed':41977,'injection_node':target}
        cell=CellEvidence(writer,name=f'cell-{index:02d}-d.json',cell_id=cell_id,
                          phase='discovery',model_seed=41977,model_state_sha256=expected)
        result=executor.run_discovery(assignment,cases,cell,budget)
        assert result=={'first_changed':2,'magnitude':2}
        assert len(cell._records)==4
        assert budget.submissions==budget.root_forwards==(70 if index==0 else 136)
    executor.end_model()
    assert executor.close() is None  # An unfinished two-phase run cannot claim success.
    assert executor.close() is None
assert not torch.cuda.is_initialized()
'''
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(ROOT)}
        with tempfile.TemporaryDirectory() as prefix:
            process = subprocess.run([sys.executable, "-B", "-X", f"pycache_prefix={prefix}", "-c", code],
                                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
