"""Explicit CPU fixed-prefix intervention acceptance on the pinned SmolLM2 assets.

No downloads. Outputs a compact JSON record; raw witnesses are saved only under
an explicitly supplied new export root. This is not a semantic/quality benchmark.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import sys
import threading

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--export-root', type=Path, required=True, help='New directory, never overwritten')
parser.add_argument('--implementation-root', type=Path, default=Path(__file__).resolve().parents[2])
args = parser.parse_args()
if not __debug__:
    raise RuntimeError('Assertions are required; do not use -O')
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
root = args.implementation_root.resolve(strict=True)
sys.path.insert(0, str(root))
import torch
import transformers
from looking_glass import app, app_runtime, app_storage, app_interventions, llama_trace
from looking_glass.model_ownership import ModelLease
assert all(Path(m.__file__).resolve().is_relative_to(root) for m in (app, app_runtime, app_storage, app_interventions, llama_trace))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
record_path = Path(__file__).with_name('record.json')
record_bytes = record_path.read_bytes()
assert hashlib.sha256(record_bytes).hexdigest() == 'd2425a1bd3848f1378570be7262c138c883be076070dab533ad19b5a279cfe32'
source = json.loads(record_bytes)['source']
checkpoint = args.checkpoint.resolve(strict=True)
expected = source['asset_manifest']

def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def assets():
    names = {p.name for p in checkpoint.iterdir() if not (p.name == '.cache' and p.is_dir() and not p.is_symlink())}
    assert names == set(expected), 'Exact pinned checkpoint inventory required'
    for name, row in expected.items():
        path = checkpoint / name
        assert path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], name

assets()
implementation_files = ['app.py', 'app_runtime.py', 'app_storage.py', 'app_interventions.py', 'llama_trace.py', 'intervention_plan.py', 'model_ownership.py']
implementation = {name: sha(root / 'looking_glass' / name) for name in implementation_files}
state = app.AppState(export_root=args.export_root)
assert not state.store.export_root.exists(), 'New export root required'
server = app.create_server(state)
worker = threading.Thread(target=server.serve_forever, daemon=True)
worker.start()

def call(path, body=None, expected_status=200):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=120)
    headers = {'X-MLG-Token': state.token}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    connection.request('GET' if body is None else 'POST', path, body=None if body is None else json.dumps(body), headers=headers)
    response = connection.getresponse(); payload = response.read(); code = response.status; connection.close()
    assert code == expected_status, (path, code, payload[:1000])
    return json.loads(payload)

def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def parameters(model):
    return {name: tensor_hash(value) for name, value in model.named_parameters()}

def capture_hashes():
    return [tensor_hash(value) for value in state.store._values]

def bits_equal(a, b):
    return a.dtype == b.dtype == torch.float32 and a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))

def independent(model, ids, mask, path, token, operation, feature, amount):
    """Separate plain forward + one direct hook; does not call ActivationPlan."""
    kwargs = dict(input_ids=ids.clone(), attention_mask=mask.clone(), use_cache=False,
                  past_key_values=None, logits_to_keep=0, output_attentions=False,
                  output_hidden_states=False, return_dict=True)
    snapshots = []
    def edit(_module, _inputs, output):
        full = output[0] if type(output) is tuple else output
        assert full.shape == (1, ids.shape[1], 576) and full.dtype == torch.float32
        changed = full.clone()
        if operation == 'zero': changed[0, token] = 0
        elif operation == 'add': changed[0, token, feature] = full[0, token, feature] + amount
        else: changed[0, token, feature] = amount
        snapshots.append((full[0, token].clone(), changed[0, token].clone()))
        return (changed, *output[1:]) if type(output) is tuple else changed
    with torch.inference_mode(), ModelLease(model):
        baseline = model(**kwargs).logits.detach().clone()
        handle = model.get_submodule(path).register_forward_hook(edit)
        try: treatment = model(**kwargs).logits.detach().clone()
        finally: handle.remove()
        restoration = model(**kwargs).logits.detach().clone()
    assert bits_equal(baseline, restoration) and len(snapshots) == 1
    assert torch.equal(kwargs['input_ids'], ids) and torch.equal(kwargs['attention_mask'], mask)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    delta = treatment.double() - baseline.double()
    return baseline, treatment, snapshots[0], dict(max_absolute_change=float(delta.abs().max()), l2_change=float(delta.norm()))

trials = []
try:
    call('/api/load', {'path': str(checkpoint), 'device': 'cpu'})
    model = state.runtime.model
    assert model.lm_head.weight is model.model.embed_tokens.weight
    original_parameters = parameters(model)
    rotary = tensor_hash(model.model.rotary_emb.inv_freq)
    run = call('/api/run', {'prompt': 'Gravity is', 'settings': {'max_new_tokens': 1}})
    assert run['intervention_available'] and run['event_count'] == 90 and run['total_bytes'] == 622080
    capture = run['capture_id']; before = capture_hashes()
    step = run['steps'][0]
    assert step['input_token_ids'] == [22007, 6463, 314] and step['input_attention_mask'] == [1, 1, 1]
    ids = torch.tensor([step['input_token_ids']], dtype=torch.int64)
    mask = torch.tensor([step['input_attention_mask']], dtype=torch.int64)
    cases = [('zero_attention', 'self_attn', 'zero', None, None),
             ('add_mlp_feature', 'mlp', 'add', 0, 1.0),
             ('replace_block_feature', '', 'replace', 0, 0.0),
             ('noop_attention_feature', 'self_attn', 'add', 0, 0.0)]
    saved = []
    for name, suffix, operation, feature, amount in cases:
        path = 'model.layers.15' + ('.' + suffix if suffix else '')
        rows = call('/api/events?q=' + path + '&forward=0&token_position=2&stack=input&capture_id=' + capture)['events']
        row, = [item for item in rows if item['metadata']['module_path'] == path]
        event = row['index']; vector = row['matched_vector_index']; assert vector == 2
        original = call(f'/api/tensor?event={event}&vector={vector}&limit=4096&capture_id={capture}')
        baseline, treatment, (ind_before, ind_after), metrics = independent(model, ids, mask, path, 2, operation, feature, amount)
        request = dict(capture_id=capture, event=event, vector=vector, operation=operation)
        if operation != 'zero': request.update(feature_index=feature, value=amount)
        result = call('/api/intervene', request)
        experiment = result['experiment']
        assert result['event_count'] == 4 and result['run_kind'] == 'intervention_comparison'
        control_keys = {'null_full_logits_bitwise_equal', 'restoration_full_logits_bitwise_equal',
                        'source_vector_bitwise_equal', 'other_token_positions_at_selected_site_bitwise_equal',
                        'static_rotary_state_unchanged', 'fixed_input_ids_mask_unchanged'}
        if operation != 'zero': control_keys.add('other_features_at_selected_vector_bitwise_equal')
        assert set(experiment['controls']) == control_keys and all(v is True for v in experiment['controls'].values())
        assert experiment['input_prefix'] == step['input_token_ids']
        assert experiment['full_logits']['max_absolute_change'] == metrics['max_absolute_change']
        assert experiment['full_logits']['l2_change'] == metrics['l2_change']
        assert experiment['comparison']['baseline_next_token_id'] == int(baseline[0, -1].argmax())
        assert experiment['comparison']['intervention_next_token_id'] == int(treatment[0, -1].argmax())
        witnesses = []
        for index in range(4):
            witness = call(f'/api/experiment-tensor?event={index}&limit=4096&experiment_id=' + result['experiment_id'])
            assert witness['coordinate']['status'] == 'known' and witness['coordinate']['module_path'] == path
            assert witness['coordinate']['token_id'] == 314 and len(witness['raw_bits']) == 576
            witnesses.append(witness)
        expected_before = [f'0x{x & 0xffffffff:08x}' for x in ind_before.view(torch.int32).tolist()]
        expected_after = [f'0x{x & 0xffffffff:08x}' for x in ind_after.view(torch.int32).tolist()]
        assert original['raw_bits'] == expected_before
        assert all(witnesses[index]['raw_bits'] == expected_before for index in range(3))
        assert witnesses[3]['raw_bits'] == expected_after
        if name.startswith('noop'):
            assert experiment['activation_changed'] is False and experiment['full_logits']['bitwise_equal']
        else:
            assert experiment['activation_changed'] is True and not experiment['full_logits']['bitwise_equal']
        assert call('/api/status')['run'] == run and capture_hashes() == before
        assert call(f'/api/tensor?event={event}&vector={vector}&limit=4096&capture_id={capture}') == original
        assert tensor_hash(model.model.rotary_emb.inv_freq) == rotary
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        if not saved: assert not state.store.export_root.exists()
        exported = call('/api/save-experiment', {'name': name, 'experiment_id': result['experiment_id']})
        saved.append((exported['path'], witnesses))
        trials.append({'name': name, 'module_path': path, 'token_position': 2, 'token_id': 314,
                       'operation': operation, 'feature_index': feature, 'requested_value': amount,
                       'independent_direct_hook_full_logit_metrics_equal': True,
                       'independent_direct_hook_witness_bits_equal': True,
                       'experiment': experiment})
    assert parameters(model) == original_parameters
    call('/api/unload', {})
    assert not call('/api/status')['run']['intervention_available'] and capture_hashes() == before
    call('/api/intervene', dict(capture_id=capture, event=0, vector=0, operation='zero'), expected_status=400)
    for path, witnesses in saved:
        reopened = call('/api/open', {'path': path})
        assert reopened['run_kind'] == 'intervention_comparison' and not reopened['intervention_available']
        for index, wanted in enumerate(witnesses):
            assert call(f'/api/tensor?event={index}&limit=4096&capture_id=' + reopened['capture_id']) == wanted
        call('/api/intervene', dict(capture_id=reopened['capture_id'], event=0, vector=0, operation='zero'), expected_status=400)
    assets()
    assert implementation == {name: sha(root / 'looking_glass' / name) for name in implementation_files}
    print(json.dumps({'schema': 'mlg-smollm2-fixed-prefix-interventions-v1', 'status': 'passed',
        'model': source['repository'], 'revision': source['revision'],
        'historical_asset_record_sha256': hashlib.sha256(record_bytes).hexdigest(),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': 'cpu', 'dtype': 'float32', 'threads': 1, 'cuda_executed': False},
        'implementation_sha256': implementation, 'script_sha256': sha(Path(__file__)),
        'prompt': 'Gravity is', 'input_ids': step['input_token_ids'], 'attention_mask': step['input_attention_mask'],
        'original_events': 90, 'original_vectors': 270, 'original_tensor_bytes': 622080,
        'checks': {'all_original_capture_hashes_unchanged_after_each_trial': True, 'all_model_parameter_hashes_unchanged': True,
            'tied_weight_identity': True, 'rotary_bits_unchanged': True, 'checkpoint_files_unchanged': True,
            'all_16_witness_vectors_exact_after_export_reopen': True, 'imported_and_unloaded_interventions_rejected': True},
        'trials': trials, 'scope': 'Four specified interventions on one three-token prefix, same-device CPU float32. No generated edited continuation, GPU validation, semantic explanation, global causal claim, quality benchmark, or checkpoint mutation.'}, indent=2))
finally:
    server.shutdown(); worker.join(5); server.server_close(); assert not worker.is_alive()
