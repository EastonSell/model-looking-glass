"""Interventions preserve the original run and require private process-local provenance."""
import copy
from pathlib import Path
import pytest
from looking_glass.app import AppState
from test_app_http import FakeRuntime
from test_app_storage import coordinate_result


class ExperimentRuntime(FakeRuntime):
    def __init__(self):
        super().__init__(); self.binding = None; self.calls = 0; self.last_budget = None
    def load(self, path, **kwargs):
        self.binding = object()
        self.loaded = {"family": "gpt2", "state": "ready", "name": path}
        return self.info()
    def capture_binding(self):
        return self.binding
    def unload(self):
        self.loaded = self.binding = None
    def run(self, prompt, settings, image_bytes=None):
        result = coordinate_result("gpt2", positions=[0])
        result["steps"][0]["use_cache"] = False
        result["steps"][0]["input_attention_mask"] = [1] * len(result["steps"][0]["input_token_ids"])
        return result
    def intervene_fixed_forward(self, binding, source, operation, feature_index=None, value=None, max_pair_bytes=0):
        assert binding is self.binding
        self.calls += 1; self.last_budget = max_pair_bytes
        result = self.run("", {})
        result.update(run_kind="intervention_comparison", experiment={"operation": operation}, answer="fixed-forward comparison")
        return result


@pytest.fixture
def experiment_state(tmp_path):
    state = AppState(runtime=ExperimentRuntime(), export_root=tmp_path / "exports", max_trace_bytes=1024)
    state.post("load", {"path": "fixture"})
    state.post("run", {"prompt": "x", "settings": {"max_trace_bytes": 1024}})
    return state


def test_separate_comparison_preserves_source_and_admits_remaining_budget(experiment_state):
    state = experiment_state
    original = state.store.tensor(0); summary = state.store.summary()
    assert state.status()["run"]["intervention_available"]
    result = state.post("intervene", {"event": 0, "vector": 0, "operation": "zero", "capture_id": state.status()["run"]["capture_id"]})
    assert result["run_kind"] == "intervention_comparison"
    assert state.runtime.last_budget == 1024 - summary["total_bytes"]
    assert state.store.tensor(0) == original and state.store.summary() == summary
    saved = state.post("save-experiment", {"name": "comparison", "experiment_id": result["experiment_id"]})
    assert Path(saved["path"]).is_dir()
    assert state.store.tensor(0) == original


def test_imported_or_reloaded_run_cannot_reuse_private_binding(experiment_state):
    state = experiment_state
    saved = state.post("save", {"name": "original"})
    state.post("open", {"path": saved["path"]})
    assert not state.status()["run"]["intervention_available"]
    with pytest.raises(ValueError, match="live|import|fresh"):
        state.post("intervene", {"event": 0, "vector": 0, "operation": "zero", "capture_id": state.status()["run"]["capture_id"]})
    assert state.runtime.calls == 0
    state.post("run", {"prompt": "x", "settings": {"max_trace_bytes": 1024}})
    state.runtime.load("same-path-new-instance")
    assert not state.status()["run"]["intervention_available"]
    with pytest.raises(ValueError): state.post("intervene", {"event": 0, "vector": 0, "operation": "zero", "capture_id": state.status()["run"]["capture_id"]})
    assert state.runtime.calls == 0


def test_unload_retains_both_results_but_disables_further_intervention(experiment_state):
    state=experiment_state
    state.post("intervene", {"event": 0, "vector": 0, "operation": "zero", "capture_id": state.status()["run"]["capture_id"]})
    original=state.store.summary(); comparison=state.experiment_store.summary()
    status=state.post("unload", {})
    assert state.store.summary()==original and state.experiment_store.summary()==comparison
    assert not status["run"]["intervention_available"]
    with pytest.raises(ValueError): state.post("intervene", {"event":0,"vector":0,"operation":"zero","capture_id":state.status()["run"]["capture_id"]})


def test_failed_intervention_keeps_source_and_no_stale_comparison(experiment_state):
    state=experiment_state; original=state.store.tensor(0)
    state.post("intervene", {"event":0,"vector":0,"operation":"zero","capture_id":state.status()["run"]["capture_id"]})
    def failure(*args, **kwargs):
        state.runtime.binding=None; state.runtime.loaded["state"]="quarantined"
        raise RuntimeError("synthetic uncertain owner")
    state.runtime.intervene_fixed_forward=failure
    with pytest.raises(RuntimeError): state.post("intervene", {"event":0,"vector":0,"operation":"zero","capture_id":state.status()["run"]["capture_id"]})
    assert state.store.tensor(0)==original and state.experiment_store.summary() is None
    assert not state.status()["run"]["intervention_available"]


def test_zero_remaining_budget_rejects_before_runtime(experiment_state):
    state=experiment_state; state.max_trace_bytes=state.store.summary()["total_bytes"]
    with pytest.raises(ValueError, match="budget|allowance|remaining"):
        state.post("intervene", {"event":0,"vector":0,"operation":"zero","capture_id":state.status()["run"]["capture_id"]})
    assert state.runtime.calls==0


@pytest.mark.parametrize("action,data", [("clear",{}),("load",{"path":"new"}),("run",{"prompt":"x","settings":{"max_trace_bytes":1024}})])
def test_source_replacement_invalidates_comparison(experiment_state, action, data):
    state=experiment_state;state.post("intervene",{"event":0,"vector":0,"operation":"zero","capture_id":state.status()["run"]["capture_id"]})
    state.post(action,data)
    assert state.experiment_store.summary() is None


def test_stale_capture_revision_rejects_before_runtime(experiment_state):
    state=experiment_state; stale=state.status()["run"]["capture_id"]
    state.post("run", {"prompt":"new", "settings":{"max_trace_bytes":1024}})
    assert state.status()["run"]["capture_id"] != stale
    with pytest.raises(ValueError, match="stale"):
        state.post("intervene", {"event":0,"vector":0,"operation":"zero","capture_id":stale})
    assert state.runtime.calls==0


@pytest.mark.parametrize('family', ['gpt2', 'llama'])
@pytest.mark.parametrize('operation', ['zero', 'add', 'replace'])
def test_real_http_fixed_forward_experiment_and_saved_witnesses(tmp_path, operation, family):
    import json, threading
    from looking_glass.app import create_server
    from looking_glass.app_runtime import LocalModelRuntime
    from test_app_runtime import local_checkpoint
    from test_app_http import call
    checkpoint=tmp_path/'checkpoint'
    if family == 'llama':
        from test_app_llama_http import llama_checkpoint, LLAMA3_SCALING
        llama_checkpoint(checkpoint, 2, LLAMA3_SCALING)
    else:
        local_checkpoint(checkpoint, 'gpt2')
    state=AppState(runtime=LocalModelRuntime(),export_root=tmp_path/'exports')
    httpd=create_server(state);thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    endpoint=(state,httpd)
    try:
        assert call(endpoint,'/api/load','POST',{'path':str(checkpoint),'device':'cpu'})[0]==200
        code,body,_=call(endpoint,'/api/run','POST',{'prompt':'hello world','settings':{'max_new_tokens':1}})
        assert code==200,body
        run=json.loads(body);assert run['intervention_available']
        original=json.loads(call(endpoint,'/api/tensor?event=0&vector=0&limit=4096')[1])
        request={'capture_id':run['capture_id'],'event':0,'vector':0,'operation':operation}
        if operation!='zero':request.update(feature_index=0,value=0.25)
        assert call(endpoint,'/api/intervene','POST',request,token=False)[0]==403
        code,body,_=call(endpoint,'/api/intervene','POST',request)
        assert code==200,body.decode()
        comparison=json.loads(body);assert comparison['run_kind']=='intervention_comparison'
        assert comparison['event_count']==4 and comparison['experiment']['operation']==operation
        assert all(comparison['experiment']['controls'].values())
        distribution=comparison['experiment']['next_token_distribution']
        assert distribution['normalization']=='full vocabulary' and distribution['computation_dtype']=='float64'
        assert distribution['readout_position']==comparison['experiment']['readout_position']
        assert 0 <= distribution['total_variation'] <= 1
        rows=distribution['rows'];assert 1 <= len(rows) <= 10 and len({row['token_id'] for row in rows})==len(rows)
        assert all(0 <= row['baseline_probability'] <= 1 and 0 <= row['intervention_probability'] <= 1 for row in rows)
        assert json.loads(call(endpoint,'/api/tensor?event=0&vector=0&limit=4096')[1])==original
        assert json.loads(call(endpoint,'/api/experiment')[1])==comparison
        assert call(endpoint,'/api/experiment',token=False)[0]==403
        assert call(endpoint,'/api/experiment-events',token=False)[0]==403
        assert call(endpoint,'/api/experiment-tensor?event=0',token=False)[0]==403
        assert call(endpoint,'/api/experiment-tensor?event=0')[0]==400
        assert call(endpoint,'/api/experiment-tensor?event=0&experiment_id=stale')[0]==400
        for route in ('events','vectors','tensor'):
            assert call(endpoint,f'/api/{route}?event=0&capture_id=stale')[0]==400
        assert call(endpoint,'/api/save','POST',{'name':'stale','capture_id':'old'})[0]==400
        assert not state.store.export_root.exists()
        witnesses=[]
        for index in range(4):
            code,body,_=call(endpoint,f'/api/experiment-tensor?event={index}&limit=4096&experiment_id={comparison["experiment_id"]}')
            assert code==200,body
            witness=json.loads(body);assert witness['coordinate']['status']=='known'
            witnesses.append(witness)
        assert witnesses[0]['raw_bits']==witnesses[1]['raw_bits']==witnesses[2]['raw_bits']==original['raw_bits']
        saved=json.loads(call(endpoint,'/api/save-experiment','POST',{'name':'experiment','experiment_id':comparison['experiment_id']})[1])
        assert json.loads(call(endpoint,'/api/tensor?event=0&vector=0&limit=4096')[1])==original
        code,body,_=call(endpoint,'/api/open','POST',{'path':saved['path']})
        assert code==200 and not json.loads(body)['intervention_available']
        assert json.loads(body)['experiment']==comparison['experiment']
        for index in range(4):assert json.loads(call(endpoint,f'/api/tensor?event={index}&limit=4096')[1])==witnesses[index]
        assert call(endpoint,'/api/intervene','POST',request)[0]==400
    finally:
        httpd.shutdown();thread.join(5);httpd.server_close();assert not thread.is_alive()


def test_stale_comparison_save_rejects_without_export(experiment_state):
    state=experiment_state;request={"event":0,"vector":0,"operation":"zero","capture_id":state.status()["run"]["capture_id"]}
    first=state.post("intervene",request);second=state.post("intervene",request)
    assert first["experiment_id"] != second["experiment_id"]
    with pytest.raises(ValueError,match="stale"):
        state.post("save-experiment",{"name":"wrong","experiment_id":first["experiment_id"]})
    assert not state.store.export_root.exists()
    with pytest.raises(ValueError,match="stale"):
        state.post("save",{"name":"wrong-original","capture_id":"old"})
    assert not state.store.export_root.exists()
