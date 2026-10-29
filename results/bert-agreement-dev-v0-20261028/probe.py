"""Development-only DistilBERT masked agreement probe; no held-out inputs."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
    raise RuntimeError('CPU-only development probe requires CUDA_VISIBLE_DEVICES=-1')

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL = 'distilbert/distilbert-base-uncased'
REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
ASSETS = {
    'config.json': (483, '69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83'),
    'tokenizer_config.json': (48, 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10'),
    'vocab.txt': (231508, '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'),
    'tokenizer.json': (466062, 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'),
    'model.safetensors': (267954768, '5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063'),
}
WORK = Path(__file__).resolve().parent
CACHE = WORK / 'hf-cache' / 'hub'
OUTPUT = WORK / 'bert-dev-probe-v0.json'
SEED = 2026102810


def file_digest(path):
    digest = hashlib.sha256()
    count = 0
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
            count += len(block)
    return count, digest.hexdigest()


def process_memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    info = Counters()
    info.cb = ctypes.sizeof(info)
    current = ctypes.windll.kernel32.GetCurrentProcess()
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not get_info(current, ctypes.byref(info), info.cb):
        raise OSError('unable to query process memory')
    return {'working_set_bytes': int(info.WorkingSetSize),
            'peak_working_set_bytes': int(info.PeakWorkingSetSize)}


def capture(model, inputs, layer_index, position, replacement=None):
    captured = []
    def hook(module, args, output):
        if type(output) is not tuple or len(output) != 1 or type(output[0]) is not torch.Tensor:
            raise RuntimeError('unexpected DistilBERT transformer block output')
        hidden = output[0]
        captured.append(hidden[:, position, :].detach().clone())
        if replacement is None:
            return None
        altered = hidden.clone()
        altered[:, position, :] = replacement
        return (altered,)
    handle = model.distilbert.transformer.layer[layer_index].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            logits = model(**inputs).logits.detach().clone()
    finally:
        handle.remove()
    if len(captured) != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError('missing/duplicate hook invocation or nonfinite logits')
    return logits, captured[0]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.random.default_generator.manual_seed(SEED)
    if torch.cuda.is_initialized():
        raise RuntimeError('PyTorch CUDA must remain uninitialized')
    before = process_memory()
    with tempfile.TemporaryDirectory(prefix='bert-dev-stage-', dir=WORK) as stage:
        for name, expected in ASSETS.items():
            cached = hf_hub_download(MODEL, name, revision=REVISION, cache_dir=str(CACHE),
                                     local_files_only=True, token=False)
            if file_digest(cached) != expected:
                raise ValueError('cached asset mismatch: ' + name)
            dest = Path(stage) / name
            shutil.copyfile(cached, dest)
            if file_digest(dest) != expected:
                raise ValueError('staged asset mismatch: ' + name)
        tokenizer = AutoTokenizer.from_pretrained(stage, local_files_only=True, trust_remote_code=False)
        model, info = AutoModelForMaskedLM.from_pretrained(
            stage, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            attn_implementation='eager', dtype=torch.float32, output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError('checkpoint weight load was incomplete')
        model.eval()
        if any(module.training or module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
            raise RuntimeError('model evaluation or hook state was invalid')
        after_load = process_memory()
        expected_ids = {'is': 2003, 'are': 2024, '[MASK]': 103}
        if any(tokenizer.convert_tokens_to_ids(k) != v for k, v in expected_ids.items()):
            raise RuntimeError('token identity mismatch')
        prompts = [
            ('farmer-singular', 'The farmer near the nurses [MASK] ready.', 'farmer-plural'),
            ('farmer-plural', 'The farmers near the nurses [MASK] ready.', 'farmer-singular'),
            ('pilot-singular', 'The pilot near the soldiers [MASK] ready.', 'pilot-plural'),
            ('pilot-plural', 'The pilots near the soldiers [MASK] ready.', 'pilot-singular'),
        ]
        encoded = {}
        for name, text, other in prompts:
            inputs = tokenizer(text, return_tensors='pt')
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
            if len(tokens) != 10 or tokens[2] not in ('farmer', 'farmers', 'pilot', 'pilots') or tokens[6] != '[MASK]':
                raise RuntimeError('development token positions differ from the planned layout')
            encoded[name] = (text, other, inputs, tokens)
        baselines, vectors = {}, {}
        for name, (_, _, inputs, _) in encoded.items():
            logits, vector = capture(model, inputs, 2, 2)
            baselines[name] = logits
            vectors[name] = vector
        rows = []
        for name, (text, donor_name, inputs, tokens) in encoded.items():
            own_logits, _ = capture(model, inputs, 2, 2, vectors[name])
            patched, _ = capture(model, inputs, 2, 2, vectors[donor_name])
            donor_inputs = encoded[donor_name][2]
            donor_logits, donor_mask = capture(model, donor_inputs, 5, 6)
            final_mask_patch, _ = capture(model, inputs, 5, 6, donor_mask)
            if not torch.equal(own_logits, baselines[name]):
                raise RuntimeError('identity patch altered baseline logits')
            def margin(value):
                return float(value[0, 6, 2003] - value[0, 6, 2024])
            rows.append({'name': name, 'text': text, 'tokens': tokens, 'donor': donor_name,
                         'baseline_margin_is_minus_are': margin(baselines[name]),
                         'donor_baseline_margin': margin(donor_logits),
                         'subject_layer_2_patch_margin': margin(patched),
                         'subject_patch_margin_change': margin(patched) - margin(baselines[name]),
                         'self_patch_bitwise_equal': True,
                         'final_mask_patch_matches_donor_logits': bool(torch.equal(final_mask_patch[0, 6], donor_logits[0, 6]))})
        if torch.cuda.is_initialized():
            raise RuntimeError('PyTorch CUDA became initialized')
        if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
            raise RuntimeError('a temporary hook was not removed')
        result = {'version': 'bert-agreement-dev-probe-v0', 'status': 'complete',
                  'scope': 'exploratory development only; not held-out evidence',
                  'model': MODEL, 'revision': REVISION, 'assets': ASSETS,
                  'seed': SEED, 'device': 'cpu', 'torch_version': str(torch.__version__),
                  'attention': 'eager', 'threads': torch.get_num_threads(),
                  'torch_cuda_initialized': False, 'memory_before_load': before,
                  'memory_after_load': after_load, 'memory_after_forwards': process_memory(),
                  'source_sha256': file_digest(__file__)[1], 'rows': rows}
        with open(OUTPUT, 'x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'output': str(OUTPUT), 'status': result['status'], 'rows': len(rows)}))


if __name__ == '__main__':
    attempt = WORK / 'bert-dev-probe-v0-attempt.json'
    with open(attempt, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump({'status': 'started', 'scope': 'development only',
                   'source_sha256': file_digest(__file__)[1]}, stream, sort_keys=True)
        stream.write('\n')
    try:
        main()
    except BaseException as exc:
        if not OUTPUT.exists():
            failure = WORK / 'bert-dev-probe-v0-failure.json'
            with open(failure, 'x', encoding='utf-8', newline='\n') as stream:
                json.dump({'status': 'incomplete', 'exception_type': type(exc).__name__,
                           'message': str(exc)[:1000]}, stream, sort_keys=True)
                stream.write('\n')
        raise
