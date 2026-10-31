"""Offline independent CPU observation validation for one pinned public checkpoint.
Supply the matching public checkpoint locally; this script performs no download.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import argparse
parser = argparse.ArgumentParser(description="Offline verification of one exact public SmolLM2 revision; no downloads")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--implementation-root", type=Path, default=Path(__file__).resolve().parents[2], help="Checkout or separately installed package directory")
args = parser.parse_args()
if not __debug__: raise RuntimeError("Run without -O; verification assertions are required")
REPO = args.implementation_root.resolve(strict=True)
CHECKPOINT = args.checkpoint.resolve(strict=True)
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[name] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(REPO))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def bits_equal(torch, first, second):
    return (first.shape == second.shape and first.dtype == second.dtype == torch.float32
            and first.device.type == second.device.type == 'cpu'
            and torch.equal(first.contiguous().view(torch.int32), second.contiguous().view(torch.int32)))


EXPECTED_ASSETS = {'README.md': {'bytes': 6340, 'sha256': 'd1ba68cae64a89b6b434b11526e6e2271ee5ffd2c914ec35ed515f9d84c6085c'}, 'config.json': {'bytes': 704, 'sha256': '1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843'}, 'generation_config.json': {'bytes': 111, 'sha256': '2056c988e990b0d13670f63f2f3b87b3b6d07edaf7a3416998ba27dab2d8a059'}, 'merges.txt': {'bytes': 466391, 'sha256': '0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510'}, 'model.safetensors': {'bytes': 269060552, 'sha256': '80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'}, 'special_tokens_map.json': {'bytes': 831, 'sha256': 'e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3'}, 'tokenizer.json': {'bytes': 2104556, 'sha256': '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'}, 'tokenizer_config.json': {'bytes': 3658, 'sha256': '4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424'}, 'vocab.json': {'bytes': 800662, 'sha256': '82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79'}}

def check_assets(folder):
    # The Hub's local-download bookkeeping is not a model/tokenizer input.
    names={p.name for p in folder.iterdir() if not (p.name==".cache" and p.is_dir() and not p.is_symlink())}
    if names != set(EXPECTED_ASSETS):
        raise RuntimeError("Checkpoint inventory differs from the exact pinned assets")
    for name, expected in EXPECTED_ASSETS.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise RuntimeError("Missing or wrong-sized pinned asset: " + name)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected["sha256"]:
            raise RuntimeError("Pinned asset hash mismatch: " + name)

def main():
    check_assets(CHECKPOINT)
    import torch
    import transformers
    from transformers import LlamaForCausalLM, AutoTokenizer
    from looking_glass import llama_trace
    if not Path(llama_trace.__file__).resolve().is_relative_to(REPO):
        raise RuntimeError("Wrong implementation import origin")
    trace_llama_causal_lm = llama_trace.trace_llama_causal_lm
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not (CHECKPOINT / 'config.json').is_file():
        raise RuntimeError('local checkpoint metadata is missing')
    config_json = json.loads((CHECKPOINT / 'config.json').read_text())
    required = dict(model_type='llama', hidden_size=576, num_hidden_layers=30,
                    num_attention_heads=9, num_key_value_heads=3, tie_word_embeddings=True)
    if any(config_json.get(key) != value for key, value in required.items()) or config_json.get('rope_scaling') is not None:
        raise RuntimeError('checkpoint metadata differs from the pinned validation scope')
    inputs = [CHECKPOINT/name for name in sorted(EXPECTED_ASSETS)]
    asset_hashes = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in inputs}
    if not any(p.suffix == '.safetensors' for p in inputs):
        raise RuntimeError('local safetensors checkpoint is missing')
    sources = ['looking_glass/llama_trace.py', 'looking_glass/intervention_plan.py',
        'looking_glass/model_ownership.py', 'looking_glass/trace_schema.py']
    source_hashes = {path:digest(REPO/path) for path in sources}
    started = time.perf_counter()
    model, loading = LlamaForCausalLM.from_pretrained(str(CHECKPOINT), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32,
        attn_implementation='eager', output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('checkpoint does not load exactly')
    model = model.to(device='cpu', dtype=torch.float32).eval()
    model.requires_grad_(False)
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise RuntimeError('declared tied input/output weights are not aliased')
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), local_files_only=True,
        trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('fast tokenizer is required')
    load_seconds = time.perf_counter() - started
    paths = [(layer, role, f'model.layers.{layer}{suffix}')
        for layer in range(30) for role, suffix in (('attention', '.self_attn'), ('mlp', '.mlp'), ('block', ''))]
    assert len(paths) == 90
    rotary_bits = model.model.rotary_emb.inv_freq.view(torch.int32).clone()
    results = []
    for prompt_index, prompt in enumerate(('Gravity is', 'The capital of France is')):
        encoded = tokenizer(prompt, return_tensors='pt', truncation=False)
        ids, mask = encoded['input_ids'], encoded['attention_mask']
        if ids.shape[0] != 1 or not 0 < ids.shape[1] <= 32:
            raise RuntimeError('prompt exceeds bounded validation scope')
        ids_before, mask_before = ids.clone(), mask.clone()
        timer = time.perf_counter()
        with torch.inference_mode():
            plain = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=0,
                output_attentions=False, output_hidden_states=False, return_dict=True).logits
        plain_seconds = time.perf_counter() - timer
        independent, handles, seen = {}, [], []
        try:
            for layer, role, path in paths:
                def hook(_module, _args, output, path=path, role=role):
                    if path in independent:
                        raise RuntimeError('independent hook invoked more than once')
                    if role == 'attention':
                        if type(output) is not tuple:
                            raise RuntimeError('attention output is not tuple')
                        value = output[0]
                    else:
                        if type(output) is not torch.Tensor:
                            raise RuntimeError('MLP/block output is not tensor')
                        value = output
                    independent[path] = value.detach().clone()
                    seen.append(path)
                handles.append(model.get_submodule(path).register_forward_hook(hook))
            timer = time.perf_counter()
            with torch.inference_mode():
                hooked = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=0,
                    output_attentions=False, output_hidden_states=False, return_dict=True).logits
            hooked_seconds = time.perf_counter() - timer
        finally:
            for handle in handles:
                handle.remove()
        if not bits_equal(torch, plain, hooked) or seen != [path for _, _, path in paths]:
            raise RuntimeError('independent hooks changed logits or boundary order')
        timer = time.perf_counter()
        traced = trace_llama_causal_lm(model, ids, mask, run_id=f'pretrained-independent-{prompt_index}',
            layers=tuple(range(30)), max_events=90, max_trace_bytes=8 * 1024**2,
            max_tensor_bytes=1024**2, max_logits_bytes=32 * 1024**2)
        trace_seconds = time.perf_counter() - timer
        if (not bits_equal(torch, plain, traced.output.logits) or len(traced.events) != 90
                or traced.tensor_bytes != 90 * ids.shape[1] * 576 * 4):
            raise RuntimeError('adapter logits/event count mismatch')
        for sequence, (record, (layer, role, path)) in enumerate(zip(traced.events, paths, strict=True)):
            event = record.event
            expected_shape = (1, ids.shape[1], 576)
            if (event.sequence != sequence or event.module_path != path or event.module_kind != f'llama_{role}'
                    or event.layer_index != layer or event.output_path != ((0,) if role == 'attention' else ())
                    or event.invocation != 0 or event.forward_index != 0
                    or event.shape != expected_shape or event.saved_shape != expected_shape
                    or event.batch_positions != (0,) or event.token_positions != tuple(range(ids.shape[1]))
                    or event.device != 'cpu' or event.saved_device != 'cpu'
                    or not bits_equal(torch, record.value, independent[path])):
                raise RuntimeError('capture bits or coordinates differ from independent boundary')
        clean = all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        stable = torch.equal(model.model.rotary_emb.inv_freq.view(torch.int32), rotary_bits)
        if not clean or not stable or not torch.equal(ids, ids_before) or not torch.equal(mask, mask_before):
            raise RuntimeError('hooks, rotary state or input mutation detected')
        next_id = int(plain[0, -1].argmax())
        results.append(dict(prompt=prompt, input_token_ids=ids[0].tolist(), input_attention_mask=mask[0].tolist(),
            next_token_id=next_id, next_token_text=tokenizer.decode([next_id], skip_special_tokens=False),
            full_logits_shape=list(plain.shape), boundary_events=90,
            captured_bytes=traced.tensor_bytes, logits_bytes=plain.numel()*4,
            checks=dict(full_logits_bitwise_equal=True, all_boundary_bits_equal=True,
                exact_inventory_order_selectors_coordinates=True, no_residual_hooks=True,
                rotary_bits_unchanged=True, inputs_unchanged=True),
            seconds=dict(plain_forward=plain_seconds, independent_hooks=hooked_seconds, adapter=trace_seconds)))
        del traced, plain, hooked, independent
    check_assets(CHECKPOINT)
    final_inputs = [CHECKPOINT/name for name in sorted(EXPECTED_ASSETS)]
    if asset_hashes != {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in final_inputs}:
        raise RuntimeError('local checkpoint assets changed during validation')
    if source_hashes != {path:digest(REPO/path) for path in sources}:
        raise RuntimeError('implementation changed during validation')
    report = dict(status='passed', source=dict(repository='HuggingFaceTB/SmolLM2-135M', revision=REVISION,
        revision_note='Requested pinned acquisition; local asset hashes below are execution evidence, not independent remote provenance attestation.', assets=asset_hashes),
        implementation_sha256=source_hashes, implementation_unchanged=True,
        environment=dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
            device='cpu', dtype='float32', threads=torch.get_num_threads(), cuda_executed=False),
        config={key:getattr(model.config,key) for key in (*required, 'vocab_size','intermediate_size','max_position_embeddings','rope_theta','rope_scaling')},
        tied_weights=model.lm_head.weight is model.model.embed_tokens.weight,
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        checkpoint_assets_unchanged=True, load_seconds=load_seconds, prompts=results,
        scope='Actual local pretrained CPU same-model hook/adapter transparency only; no GPU, between-device equivalence, model quality or universal interpretability claim. Wall times are observations, not a benchmark.')
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
