"""Execute one model of the frozen cross-model subject-state comparison.

The caller must arrange public exposure reservation and appropriate device
coordination before launch. Each attempt gets a new output directory. A
failure leaves the partial journal and is never labeled complete.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import weakref


def _bootstrap_direct_script():
    """Avoid stale local bytecode; installed dependencies remain trusted."""
    if (__spec__ is not None or not sys.flags.isolated or not sys.flags.no_site
            or not sys.dont_write_bytecode or not sys.pycache_prefix
            or sys._xoptions.get('pycache_prefix') != sys.pycache_prefix):
        raise RuntimeError('run the source file directly using -I -S -B and a fresh pycache prefix')
    if Path(sys.executable).resolve() != Path(sys._base_executable).resolve():
        raise RuntimeError('the launched interpreter must be the base executable')
    prefix = Path(sys.pycache_prefix)
    if not prefix.is_absolute() or not prefix.is_dir() or prefix.is_symlink() or any(prefix.iterdir()):
        raise RuntimeError('pycache prefix must be a fresh empty ordinary directory')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--site', type=Path, required=True)
    known, _ = parser.parse_known_args()
    if not known.site.is_absolute() or not known.site.is_dir() or known.site.is_symlink():
        raise RuntimeError('site must be an absolute installed dependency directory')
    sys.path.insert(0, str(known.site.resolve()))


if __name__ == '__main__':
    _bootstrap_direct_script()

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

CASE_SHA = 'b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503'
ASSET_SHA = {
    'gpt': {'config.json': '4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a',
            'generation_config.json': 'b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805',
            'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
            'model.safetensors': 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
            'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
            'tokenizer.json': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
            'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'},
    'bert': {'config.json': '69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83',
             'model.safetensors': '5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063',
             'tokenizer_config.json': 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10',
             'tokenizer.json': 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98',
             'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'}}
TARGETS = {'gpt': {'is': 318, 'are': 389, 'was': 373, 'were': 547},
           'bert': {'is': 2003, 'are': 2024, 'was': 2001, 'were': 2020}}


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_assets(snapshot, model_key):
    expected = ASSET_SHA[model_key]
    if {path.name for path in snapshot.iterdir()} != set(expected):
        raise ValueError('model snapshot has missing or additional assets')
    actual = {name: sha256_file(snapshot / name) for name in expected}
    if actual != expected:
        raise ValueError('model, tokenizer or auxiliary asset differs')
    return actual


def verify_source(commit, case_path):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('source commit must be a full lowercase Git SHA')
    root = Path(__file__).resolve().parents[1]
    sources = {'scripts/cross_model_agreement_v1.py': Path(__file__).resolve(),
               'docs/CROSS_MODEL_AGREEMENT_V1.md': root / 'docs/CROSS_MODEL_AGREEMENT_V1.md',
               'experiments/cross_model_agreement_cases_v1.json': case_path.resolve()}
    hashes = {}
    for relative, path in sources.items():
        committed = subprocess.run(['git', '-C', str(root), 'show', commit + ':' + relative],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=30).stdout
        actual = path.read_bytes()
        if actual != committed:
            raise ValueError(f'local source differs from pinned commit: {relative}')
        hashes[relative] = hashlib.sha256(committed).hexdigest()
    return hashes


def replacement(base, donor, q):
    """Scale in float64, then round once to the source's float32 dtype."""
    base64, donor64 = np.asarray(base, dtype=np.float64), np.asarray(donor, dtype=np.float64)
    direction = donor64 - base64
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0 or not math.isfinite(q) or q <= 0 or q > norm + 1e-12:
        raise ValueError('invalid or insufficient donor displacement')
    intended = (q / norm) * direction
    value = np.asarray(base64 + intended, dtype='<f4')
    realized = value.astype(np.float64) - base64
    actual = float(np.linalg.norm(realized))
    direction_error = float(np.linalg.norm(realized - intended)) / q
    if (not math.isfinite(actual) or actual <= 0 or abs(actual - q) > 1e-4 * q
            or not math.isfinite(direction_error) or direction_error > 1e-4):
        raise ValueError('rounded replacement does not realize the requested dose')
    return value, actual


def random_replacement(base, q, seed):
    direction = np.random.Generator(np.random.PCG64(seed)).standard_normal(base.shape)
    return replacement(base, np.asarray(base, dtype=np.float64) + direction, q)


def donor_indices(cases, case):
    key = lambda c: (c['frame'], c['attractor_number'], c['lemma'], c['number'])
    index = {key(candidate): candidate['index'] for candidate in cases}
    opposite = (case['frame'], case['attractor_number'], case['lemma'], 1 - case['number'])
    same = (case['frame'], case['attractor_number'], case['same_number_donor_lemma'], case['number'])
    return index[opposite], index[same]


def same_bits(a, b):
    return (np.asarray(a, dtype='<f4').shape == np.asarray(b, dtype='<f4').shape
            and np.asarray(a, dtype='<f4').tobytes() == np.asarray(b, dtype='<f4').tobytes())


def require_subnormals():
    """Check this thread; this does not attest other native worker threads."""
    with np.errstate(under='ignore'):
        a = np.array([1], dtype=np.uint32).view(np.float32) * np.float32(1)
        b = np.array([1], dtype=np.uint64).view(np.float64) * np.float64(1)
        c = np.array([np.finfo(np.float32).tiny], np.float32) / np.float32(2)
        d = np.array([np.finfo(np.float64).tiny], np.float64) / np.float64(2)
    if not (a.view(np.uint32)[0] == 1 and b.view(np.uint64)[0] == 1
            and c.view(np.uint32)[0] == 1 << 22 and d.view(np.uint64)[0] == 1 << 51):
        raise RuntimeError('float32 or float64 subnormals are flushed on this thread')


def check_manifest(manifest, model_key, tokenizer):
    if (manifest.get('schema') != 'cross-model-agreement-cases-v1'
            or manifest.get('targets') != TARGETS or len(manifest.get('cases', ())) != 32):
        raise ValueError('manifest identity differs')
    for index, case in enumerate(manifest['cases']):
        row = case[model_key]
        encoded = tokenizer(row['text'], add_special_tokens=(model_key == 'bert'), truncation=False)
        if (case['index'] != index or encoded['input_ids'] != row['input_ids']
                or encoded['attention_mask'] != row['attention_mask']
                or len(row['input_ids']) > 32 or not 0 <= row['subject_position'] < len(row['input_ids'])
                or not 0 <= row['output_position'] < len(row['input_ids'])):
            raise ValueError('manifest row does not retokenize')
        if model_key == 'bert' and row['input_ids'][row['output_position']] != tokenizer.mask_token_id:
            raise ValueError('BERT output site is not the mask')
        donor_indices(manifest['cases'], case)


class Journal:
    def __init__(self, directory, vocab):
        self.starts = (directory / 'starts.jsonl').open('xb', buffering=0)
        self.events = (directory / 'events.jsonl').open('xb', buffering=0)
        self.samples = (directory / 'memory.jsonl').open('xb', buffering=0)
        self.retention_errors = (directory / 'retention-errors.jsonl').open('xb', buffering=0)
        self.logits = (directory / 'logits.f32').open('xb', buffering=0)
        self.vectors = (directory / 'witness.f32').open('xb', buffering=0)
        self.vocab = vocab
        self.count = 0

    def start(self, case, row, arm):
        event = {'event_index': self.count, 'case_id': case['case_id'], 'case_index': case['index'],
                 'arm': arm, 'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
                 'subject_position': row['subject_position'], 'output_position': row['output_position']}
        data = (json.dumps(event, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
        if self.starts.write(data) != len(data):
            raise OSError('short forward-start write')
        os.fsync(self.starts.fileno())

    def write(self, event, logits, before, after):
        if self.count >= 192:
            raise ValueError('fixed trace storage limit reached')
        logits = np.asarray(logits, dtype='<f4')
        before = np.asarray(before, dtype='<f4')
        after = np.asarray(after, dtype='<f4')
        if logits.shape != (self.vocab,) or before.shape != (768,) or after.shape != (768,):
            raise ValueError('malformed saved tensor')
        event['logits_offset'] = self.logits.tell()
        event['witness_offset'] = self.vectors.tell()
        for stream, data in ((self.logits, logits.tobytes()),
                             (self.vectors, before.tobytes() + after.tobytes())):
            if stream.write(data) != len(data):
                raise OSError('short tensor write')
        event['event_index'] = self.count
        data = (json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        if self.events.write(data) != len(data):
            raise OSError('short journal write')
        self.count += 1

    def sample(self, memory):
        data = (json.dumps({'event_index': self.count - 1, 'memory': memory},
                           sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
        if self.samples.write(data) != len(data):
            raise OSError('short memory sample write')

    def retention_error(self, event, error):
        data = (json.dumps({'event_index': self.count, 'case_id': event['case_id'], 'arm': event['arm'],
                            'reason': type(error).__name__}, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
        if self.retention_errors.write(data) != len(data):
            raise OSError('short retention-error write')

    def close(self):
        for stream in (self.starts, self.events, self.samples, self.retention_errors, self.logits, self.vectors):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()


def forward(model, block, row, replacement_value, device, *, journal=None, event=None):
    """Capture and optionally replace one selected block-output vector."""
    witness = []
    position = row['subject_position']

    def hook(_module, _inputs, output):
        if not isinstance(output, tuple) or not output or output[0].ndim != 3:
            raise ValueError('unexpected block output')
        hidden = output[0]
        if hidden.shape[0] != 1 or hidden.shape[2] != 768 or position >= hidden.shape[1]:
            raise ValueError('unexpected block output shape')
        before = hidden[0, position].detach().to('cpu').numpy().copy()
        if replacement_value is None:
            after = before.copy()
            result = output
        else:
            patched = hidden.clone()
            patched[0, position] = torch.as_tensor(replacement_value, device=hidden.device, dtype=hidden.dtype)
            after = patched[0, position].detach().to('cpu').numpy().copy()
            result = (patched,) + output[1:]
        witness.append((before, after))
        return result

    handle = block.register_forward_hook(hook)
    output = logits = None
    finite = None
    count_at_entry = journal.count if journal is not None else None
    try:
        ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
        mask = torch.tensor([row['attention_mask']], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, return_dict=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            logits = output.logits[0, row['output_position']].detach().to('cpu').numpy().copy()
        if len(witness) != 1:
            raise ValueError('block invocation count differs')
        finite = bool(torch.isfinite(output.logits).all().item())
        if journal is not None:
            journal.write({**event, 'hook_count': 1, 'full_logits_finite': finite,
                           'returned_status': 'returned'}, logits, witness[0][0], witness[0][1])
        return logits, witness[0][0], witness[0][1], finite
    finally:
        try:
            if journal is not None and journal.count == count_at_entry and output is not None:
                try:
                    if len(witness) != 1:
                        raise ValueError('returned model output lacks a unique captured witness')
                    if logits is None:
                        logits = output.logits[0, row['output_position']].detach().to('cpu').numpy().copy()
                    journal.write({**event, 'hook_count': 1, 'full_logits_finite': finite,
                                   'returned_status': 'interrupted'}, logits, witness[0][0], witness[0][1])
                except BaseException as error:
                    try:
                        journal.retention_error(event, error)
                    except BaseException:
                        pass
        finally:
            handle.remove()


def memory_sample(device):
    if device.type != 'cuda':
        return None
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {'cuda_free': free, 'cuda_total': total,
            'allocated': torch.cuda.memory_allocated(device),
            'reserved': torch.cuda.memory_reserved(device),
            'peak_allocated': torch.cuda.max_memory_allocated(device),
            'peak_reserved': torch.cuda.max_memory_reserved(device)}


def run(model, block, manifest, model_key, device, journal):
    cases = manifest['cases']
    captures, clean_logits = [], []
    for case in cases:
        require_subnormals()
        row = case[model_key]
        journal.start(case, row, 'clean')
        event = {'case_id': case['case_id'], 'case_index': case['index'], 'arm': 'clean'}
        logits, before, after, finite = forward(model, block, row, None, device, journal=journal, event=event)
        require_subnormals()
        journal.sample(memory_sample(device))
        if not finite or not np.isfinite(logits).all() or not np.isfinite(before).all():
            raise ValueError('nonfinite clean tensors')
        if not same_bits(before, after):
            raise ValueError('clean capture changed source')
        captures.append(before)
        clean_logits.append(logits)
    for case in cases:
        require_subnormals()
        index = case['index']
        base, baseline = captures[index], clean_logits[index]
        opposite_index, same_index = donor_indices(cases, case)
        opp_norm = float(np.linalg.norm(captures[opposite_index].astype(np.float64) - base.astype(np.float64)))
        same_norm = float(np.linalg.norm(captures[same_index].astype(np.float64) - base.astype(np.float64)))
        q = min(2.0, opp_norm, same_norm)
        if not math.isfinite(q) or q <= 0:
            raise ValueError('nonfinite or zero displacement')
        opposite, opp_actual = replacement(base, captures[opposite_index], q)
        same, same_actual = replacement(base, captures[same_index], q)
        random, random_actual = random_replacement(base, q, 20261029 + index)
        arms = (('self', base, index, 0.0), ('opposite', opposite, opposite_index, opp_actual),
                ('same', same, same_index, same_actual), ('random', random, None, random_actual),
                ('clean_after', None, None, 0.0))
        for arm, value, donor_index, actual in arms:
            require_subnormals()
            journal.start(case, case[model_key], arm)
            event = {'case_id': case['case_id'], 'case_index': index, 'arm': arm,
                     'donor_index': donor_index, 'requested_q': q,
                     'opposite_raw_norm': opp_norm, 'same_raw_norm': same_norm,
                     'actual_l2': actual, 'random_seed': 20261029 + index if arm == 'random' else None}
            logits, before, after, finite = forward(model, block, case[model_key], value, device,
                                                    journal=journal, event=event)
            require_subnormals()
            journal.sample(memory_sample(device))
            if not finite or not np.isfinite(logits).all() or not np.isfinite(before).all() or not np.isfinite(after).all():
                raise ValueError('nonfinite returned tensors')
            if not same_bits(before, base):
                raise ValueError('clean source changed between forwards')
            if value is not None and not same_bits(after, value):
                raise ValueError('replacement did not land')
            if arm in ('self', 'clean_after') and not same_bits(logits, baseline):
                raise ValueError('clean, self and clean-after logits differ')
    if journal.count != 192:
        raise ValueError('unexpected event count')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('gpt', 'bert'), required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASE_SHA:
        raise ValueError('case or source identity differs')
    manifest = json.loads(raw)
    source_hashes = verify_source(args.source_commit, args.cases)
    asset_hashes = verify_assets(args.model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    check_manifest(manifest, args.model, tokenizer)
    args.output.mkdir(parents=False, exist_ok=False)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.manual_seed(20261029)
    np.random.seed(20261029)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.set_flush_denormal(False) is not True:
        raise RuntimeError('cannot request subnormal-preserving arithmetic')
    require_subnormals()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    metadata = {'model': args.model, 'device': str(device), 'source_commit': args.source_commit,
                'case_sha256': CASE_SHA, 'assets_sha256': asset_hashes, 'sources_sha256': source_hashes,
                'python': sys.version,
                'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__,
                'base_interpreter_direct': True, 'isolated': bool(sys.flags.isolated),
                'seed': 20261029, 'pid': os.getpid(), 'start_unix': time.time(),
                'initial_memory': memory_sample(device)}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    model_type = AutoModelForCausalLM if args.model == 'gpt' else AutoModelForMaskedLM
    journal = Journal(args.output, 50257 if args.model == 'gpt' else 30522)
    model = block = None
    try:
        model = model_type.from_pretrained(args.model_dir, local_files_only=True, use_safetensors=True,
                                           torch_dtype=torch.float32, attn_implementation='eager').to(device)
        model.eval()
        block = model.transformer.h[3] if args.model == 'gpt' else model.distilbert.transformer.layer[3]
        run(model, block, manifest, args.model, device, journal)
        journal.close()
        if verify_assets(args.model_dir, args.model) != asset_hashes or verify_source(args.source_commit, args.cases) != source_hashes:
            raise ValueError('asset or source changed during execution')
        reference = weakref.ref(model)
        del block
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        require_subnormals()
        result = {'status': 'complete', 'events': 192, 'model_weakref_dead': reference() is None,
                  'final_memory': memory_sample(device), 'end_unix': time.time(),
                  'metadata_sha256': sha256_file(args.output / 'metadata.json'),
                  'starts_sha256': sha256_file(args.output / 'starts.jsonl'),
                  'events_sha256': sha256_file(args.output / 'events.jsonl'),
                  'memory_sha256': sha256_file(args.output / 'memory.jsonl'),
                  'retention_errors_sha256': sha256_file(args.output / 'retention-errors.jsonl'),
                  'logits_sha256': sha256_file(args.output / 'logits.f32'),
                  'witness_sha256': sha256_file(args.output / 'witness.f32')}
        (args.output / 'complete.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException:
        try:
            journal.close()
        except Exception:
            pass
        (args.output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
