"""Durable evidence primitives for a bounded DistilBERT agreement study.

This module does not authorize a model execution or certify provenance. The
caller must verify the frozen protocol, model, inputs, runtime and teardown.
"""

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from dataclasses import dataclass
import tempfile
from threading import current_thread
import re
import subprocess
import sys
from datetime import datetime, timezone

from .pretrained import _copy_verified

MODEL_ID = 'distilbert/distilbert-base-uncased'
REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
ASSETS = {
    'config.json': (483, '69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83'),
    'tokenizer_config.json': (48, 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10'),
    'vocab.txt': (231508, '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'),
    'tokenizer.json': (466062, 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'),
    'model.safetensors': (267954768, '5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063'),
}
SOURCE_PATHS = (
    'scripts/bert_agreement_bootstrap.py', 'scripts/bert_agreement_cpu_evaluation.py', 'docs/BERT_AGREEMENT_V1.md',
    'docs/CPU_STUDY_SUPERVISION.md', 'docs/SUPERVISOR.md', 'docs/BERT_AGREEMENT_EXPOSURES_V2.md',
    'docs/BERT_AGREEMENT_EXPOSURES_V3.md', 'experiments/bert_agreement_cpu_evaluation_v1.json',
    'experiments/bert_agreement_public_cases_v1.json', 'experiments/bert_agreement_exposures_v2.json',
    'experiments/bert_agreement_exposures_v3.json',
    'looking_glass/__init__.py', 'looking_glass/bert_agreement_cases.py',
    'looking_glass/bert_agreement_exposures_v2.py', 'looking_glass/bert_agreement_exposures_v3.py',
    'looking_glass/bert_evaluation_ledger.py', 'looking_glass/bert_agreement_runner.py',
    'looking_glass/bert_agreement_audit.py', 'looking_glass/bert_agreement_study.py',
    'looking_glass/pretrained.py', 'looking_glass/intervention_plan.py',
    'looking_glass/interventions.py', 'looking_glass/model_ownership.py',
    'looking_glass/cpu_study_supervision.py', 'looking_glass/supervisor.py',
    'looking_glass/device_lease.py', 'looking_glass/device_identity.py',
)
_CPU_CUSTODY = {}
_CONTROLLER_CUSTODY = {}
EVALUATION_ENABLED = False  # Enable only in the independently reviewed frozen revision.


def stage_verified_assets(snapshot, destination):
    """Copy exactly the pinned files to a new empty stage, checking every byte."""
    snapshot, destination = Path(snapshot), Path(destination)
    if not snapshot.is_dir() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('snapshot and an empty staging directory are required')
    for name, (size, digest) in ASSETS.items():
        _copy_verified(snapshot / name, destination / name, name, size, digest)


def verify_tokenizer_rows(tokenizer, manifest):
    """Check every frozen text/ID/mask pair against the actual tokenizer."""
    if (type(manifest) is not dict or type(manifest.get('rows')) is not list
            or not 1 <= len(manifest['rows']) <= 128 or tokenizer.mask_token_id != 103
            or tokenizer.convert_tokens_to_ids('is') != 2003
            or tokenizer.convert_tokens_to_ids('are') != 2024
            or tokenizer.encode('is', add_special_tokens=False) != [2003]
            or tokenizer.encode('are', add_special_tokens=False) != [2024]):
        raise ValueError('tokenizer or manifest identity is invalid')
    for row in manifest['rows']:
        if type(row) is not dict or type(row.get('text')) is not str:
            raise ValueError('manifest contains an invalid text row')
        encoded = tokenizer(row['text'], add_special_tokens=True, padding=False,
                            truncation=False, return_attention_mask=True)
        if (encoded['input_ids'] != row.get('input_ids')
                or encoded['attention_mask'] != row.get('attention_mask')):
            raise ValueError('actual tokenizer does not reproduce the frozen row')
    return len(manifest['rows'])


def read_journal(path):
    """Re-read a closed, bounded JSONL file and reject partial/ambiguous rows."""
    with Path(path).open('rb') as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if not raw or len(raw) > 16 * 1024 * 1024 or not raw.endswith(b'\n'):
        raise ValueError('journal is empty, oversized or incomplete')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON object key')
            result[key] = value
        return result
    records = []
    for line in raw.splitlines():
        if not line or len(line) > 65536:
            raise ValueError('invalid journal line size')
        try:
            value = json.loads(line.decode('ascii'), object_pairs_hook=unique,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError('invalid journal JSON') from error
        if type(value) is not dict:
            raise ValueError('journal row must be an object')
        records.append(value)
    return records, hashlib.sha256(raw).hexdigest()


def audit_event_journal(manifest, path, *, split, family, mask_id, hidden_size, expected_sha256):
    """Reconcile a closed journal with the entire frozen family schedule."""
    from .bert_agreement_audit import audit_family_events

    if (type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None
            or (split == 'development' and (type(family) is not int or not 1 <= family <= 4))
            or (split == 'evaluation' and family is not None)
            or split not in ('development', 'evaluation')):
        raise ValueError('invalid journal audit identity')
    records, digest = read_journal(path)
    if digest != expected_sha256:
        raise ValueError('journal digest changed after close')
    wanted = [family] if split == 'development' else list(range(1, 9))
    groups = {number: [] for number in wanted}
    index = 0
    for record in records:
        if (record.get('split') != split or type(record.get('family')) is not int
                or record['family'] not in groups):
            raise ValueError('journal contains an unexpected family or split')
        number = record['family']
        while index < len(wanted) and wanted[index] != number and groups[wanted[index]]:
            index += 1
        if index >= len(wanted) or wanted[index] != number:
            raise ValueError('journal families are missing or out of order')
        groups[number].append(record)
    audited = [audit_family_events(manifest, groups[number], split=split, family=number,
                                   mask_id=mask_id, hidden_size=hidden_size) for number in wanted]
    summary = aggregate_families(manifest, audited) if split == 'evaluation' else None
    return {'sha256': digest, 'families': audited, 'summary': summary}


def verify_git_sources(repo, commit, paths, *, git_exe='git'):
    """Compare source bytes to an ancestor commit, even if Git hides a change."""
    if (type(commit) is not str or re.fullmatch('[0-9a-f]{40}', commit) is None
            or type(paths) not in (list, tuple) or not paths or len(paths) > 64
            or any(type(path) is not str for path in paths) or len(set(paths)) != len(paths)):
        raise ValueError('invalid source commit or source path list')
    root = Path(repo).resolve(strict=True)
    def git(*args):
        try:
            return subprocess.run([git_exe, *args], cwd=root, capture_output=True,
                                  check=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError('unable to verify committed source') from error
    git('merge-base', '--is-ancestor', commit, 'HEAD')
    if git('status', '--porcelain', '--untracked-files=no').strip():
        raise RuntimeError('tracked checkout contains uncommitted changes')
    hashes = {}
    for relative in paths:
        path = PurePosixPath(relative)
        if (path.is_absolute() or path.as_posix() != relative or '\\' in relative
                or any(part in ('', '.', '..') for part in path.parts)):
            raise ValueError('source path must be a safe relative path')
        local = root.joinpath(*path.parts)
        resolved = local.resolve(strict=True)
        if local.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
            raise RuntimeError('source path is not a regular checkout file')
        committed = git('show', commit + ':' + relative)
        if len(committed) > 16 * 1024 * 1024 or local.read_bytes() != committed:
            raise RuntimeError('source bytes differ from the committed revision')
        hashes[relative] = hashlib.sha256(committed).hexdigest()
    return hashes


def validate_owner_result(candidate, *, run_id, pid, exit_code, source_commit, journal_sha256):
    """Tie a completed child candidate to its reaped process and closed log."""
    if (type(candidate) is not dict or type(run_id) is not str
            or re.fullmatch('[0-9a-f]{32}', run_id) is None
            or type(pid) is not int or pid <= 0 or type(exit_code) is not int or exit_code != 0
            or type(source_commit) is not str or re.fullmatch('[0-9a-f]{40}', source_commit) is None
            or type(journal_sha256) is not str or re.fullmatch('[0-9a-f]{64}', journal_sha256) is None
            or candidate.get('schema') != 'bert-agreement-owner-v1' or candidate.get('status') != 'complete'
            or candidate.get('run_id') != run_id or type(candidate.get('pid')) is not int
            or candidate['pid'] != pid or candidate.get('source_commit') != source_commit
            or candidate.get('journal_sha256') != journal_sha256):
        raise ValueError('owner result does not match its reaped process and journal')
    return True


def aggregate_families(manifest, families):
    """Recompute fixed held-out family decisions without excluding model errors."""
    from .bert_agreement_runner import SITES, decode_float32_bits

    if (type(manifest) is not dict or type(manifest.get('rows')) is not list
            or type(manifest.get('targets')) is not list or type(families) not in (list, tuple)
            or len(families) != 8):
        raise ValueError('eight complete evaluation families are required')
    rows = {}
    for row in manifest['rows']:
        if type(row) is not dict or type(row.get('row_id')) is not str or row['row_id'] in rows:
            raise ValueError('invalid or duplicate manifest row')
        rows[row['row_id']] = row
    sites = tuple(name for name, _, _ in SITES)
    order = (('singular', 'singular'), ('singular', 'plural'),
             ('plural', 'singular'), ('plural', 'plural'))
    family_means = {site: [] for site in sites}
    cell_records = {f'{a}_{b}': [] for a, b in order}
    for number, result in enumerate(families, 1):
        targets = [row for row in manifest['targets']
                   if row.get('split') == 'evaluation' and row.get('family') == number]
        if (type(result) is not dict or result.get('split') != 'evaluation'
                or type(result.get('family')) is not int or result['family'] != number
                or type(result.get('forward_count')) is not int or result['forward_count'] != 140
                or type(result.get('cases')) is not list or len(result['cases']) != 4
                or type(result.get('controls')) is not list or len(targets) != 4):
            raise ValueError('invalid evaluation family inventory')
        expected_controls = set()
        effects = {site: [] for site in sites}
        for case, target, factors in zip(result['cases'], targets, order):
            target_id, case_id = target.get('target_row_id'), target.get('case_id')
            row = rows.get(target_id)
            if (type(case) is not dict or row is None or case.get('target_row_id') != target_id
                    or case.get('case_id') != case_id or row.get('split') != 'evaluation'
                    or row.get('family') != number
                    or (row.get('subject_number'), row.get('distractor_number')) != factors
                    or type(case.get('effects')) is not dict or set(case['effects']) != set(sites)):
                raise ValueError('case identity or factorial order is invalid')
            for site in sites:
                value = case['effects'][site]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('case effect must be finite')
                effects[site].append(float(value))
                expected_controls.add(('self_patch', case_id, site))
                if site != 'candidate':
                    expected_controls.add(('clean_matches_candidate', target_id, site))
            expected_controls.update((('final_subject', case_id, 'final_subject'),
                                      ('final_mask', case_id, 'final_mask'),
                                      ('final_clean', case_id, 'candidate')))
            margin = math.fsum((decode_float32_bits(case['baseline_is_bits']),
                                -decode_float32_bits(case['baseline_are_bits'])))
            correct = margin > 0 if factors[0] == 'singular' else margin < 0
            cell_records[f'{factors[0]}_{factors[1]}'].append((margin, correct))
        actual_controls = []
        for control in result['controls']:
            if type(control) is not dict or control.get('passed') is not True:
                raise ValueError('a required control failed')
            actual_controls.append((control.get('name'), control.get('case_id'), control.get('site')))
        if len(actual_controls) != len(expected_controls) or set(actual_controls) != expected_controls:
            raise ValueError('required controls are missing or duplicated')
        for site in sites:
            family_means[site].append(math.fsum(effects[site]) / 4.0)
    means = {site: math.fsum(values) / 8.0 for site, values in family_means.items()}
    comparisons = sites[1:]
    candidate_minus_site = {site: math.fsum((means['candidate'], -means[site])) for site in comparisons}
    family_candidate_minus_site = {
        site: [math.fsum((candidate, -other)) for candidate, other in zip(family_means['candidate'],
                                                                           family_means[site])]
        for site in comparisons}
    mean_family_candidate_minus_site = {site: math.fsum(values) / 8.0
                                        for site, values in family_candidate_minus_site.items()}
    cells = {name: {'n': len(values), 'correct': sum(correct for _, correct in values),
                    'mean_margin': math.fsum(margin for margin, _ in values) / len(values)}
             for name, values in cell_records.items()}
    positive = sum(value > 0 for value in family_means['candidate'])
    return {'site_means': means, 'family_means': family_means,
            'candidate_minus_site': candidate_minus_site,
            'family_candidate_minus_site': family_candidate_minus_site,
            'mean_family_candidate_minus_site': mean_family_candidate_minus_site,
            'positive_candidate_families': positive,
            'directional_hypothesis_passed': means['candidate'] > .1 and positive >= 6,
            'site_specificity_passed': candidate_minus_site['block2_distractor'] > .1,
            'baseline': {'n': 32, 'correct': sum(cell['correct'] for cell in cells.values()), 'cells': cells}}


@dataclass
class LoadedMaskedBert:
    model: object
    tokenizer: object
    metadata: dict


def load_pinned_distilbert(cache_dir):
    """Load the exact pinned CPU masked model from local verified assets only."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('-1', ''):
        raise RuntimeError('CPU study requires CUDA visibility to be masked before dependency import')
    import torch
    import transformers

    if torch.cuda.is_initialized():
        raise RuntimeError('PyTorch CUDA was initialized before CPU model load')
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError('the CPU study requires deterministic algorithms and one thread')
    cache = Path(cache_dir).resolve(strict=True)
    snapshot = cache / 'models--distilbert--distilbert-base-uncased' / 'snapshots' / REVISION
    with tempfile.TemporaryDirectory(prefix='bert-agreement-stage-', dir=cache) as staged:
        stage_verified_assets(snapshot, staged)
        with torch.device('cpu'):
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                staged, local_files_only=True, trust_remote_code=False, use_fast=True)
            model, info = transformers.AutoModelForMaskedLM.from_pretrained(
                staged, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                attn_implementation='eager', dtype=torch.float32, output_loading_info=True)
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError('checkpoint load reported missing, unexpected or mismatched weights')
        model.eval()
        if (type(model) is not transformers.DistilBertForMaskedLM
                or model.config.model_type != 'distilbert' or model.config.n_layers != 6
                or model.config.dim != 768 or model.config.vocab_size != 30522
                or model.config._attn_implementation != 'eager'
                or tokenizer.mask_token_id != 103
                or tokenizer.convert_tokens_to_ids('is') != 2003
                or tokenizer.convert_tokens_to_ids('are') != 2024):
            raise RuntimeError('loaded model or tokenizer differs from the fixed study contract')
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU model load initialized PyTorch CUDA')
        metadata = {'model': MODEL_ID, 'revision': REVISION,
                    'assets': {name: {'size': size, 'sha256': digest}
                               for name, (size, digest) in ASSETS.items()},
                    'model_class': type(model).__name__, 'tokenizer_class': type(tokenizer).__name__,
                    'torch_version': str(torch.__version__), 'transformers_version': str(transformers.__version__),
                    'dtype': 'float32', 'device': 'cpu', 'attention_implementation': model.config._attn_implementation,
                    'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
                    'pytorch_cuda_initialized': False}
        return LoadedMaskedBert(model, tokenizer, metadata)


@dataclass(frozen=True)
class ModelStateWitness:
    """Strong references, structural metadata and digest for a CPU model."""

    modules: tuple
    tensors: tuple
    sha256: str


def capture_model_state(model):
    """Hash registered CPU parameters/buffers in bounded chunks, including aliases."""
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError('model must be a torch module')
    modules = tuple((name, module, bool(module.training))
                    for name, module in model.named_modules(remove_duplicate=False))
    tensors = []
    digest = hashlib.sha256()
    for kind, entries in (('parameter', model.named_parameters(remove_duplicate=False)),
                          ('buffer', model.named_buffers(remove_duplicate=False))):
        for name, tensor in entries:
            if tensor.device.type != 'cpu' or not tensor.is_contiguous():
                raise ValueError('model state must be contiguous and resident on CPU')
            storage = tensor.untyped_storage()
            metadata = (kind, name, tuple(tensor.shape), tuple(tensor.stride()),
                        str(tensor.dtype), str(tensor.device), tensor.storage_offset(),
                        storage.nbytes(), storage.data_ptr(), bool(tensor.requires_grad))
            header = json.dumps(metadata[:-2] + metadata[-1:], separators=(',', ':')).encode('utf-8')
            digest.update(len(header).to_bytes(4, 'big'))
            digest.update(header)
            raw = tensor.detach().reshape(-1).view(torch.uint8)
            for offset in range(0, raw.numel(), 1024 * 1024):
                digest.update(memoryview(raw[offset:offset + 1024 * 1024].numpy()))
            tensors.append((metadata, tensor, storage))
    return ModelStateWitness(modules, tuple(tensors), digest.hexdigest())


def verify_model_state(model, witness):
    """Reject changes to module/registration identity, mode or stored values."""
    if type(witness) is not ModelStateWitness:
        raise ValueError('a model state witness is required')
    current = capture_model_state(model)
    if (len(current.modules) != len(witness.modules) or len(current.tensors) != len(witness.tensors)
            or current.sha256 != witness.sha256):
        raise RuntimeError('model state or registration changed')
    for (name, module, mode), (old_name, old_module, old_mode) in zip(current.modules, witness.modules):
        if name != old_name or module is not old_module or mode != old_mode:
            raise RuntimeError('model module identity or mode changed')
    for (metadata, tensor, storage), (old_metadata, old_tensor, old_storage) in zip(current.tensors, witness.tensors):
        if metadata != old_metadata or tensor is not old_tensor or storage is not old_storage:
            raise RuntimeError('model tensor identity or storage changed')
    return True


class EventJournal:
    """Exclusive, size-bounded JSON-lines evidence with failure latching."""

    def __init__(self, path, *, max_bytes=16 * 1024 * 1024):
        if type(max_bytes) is not int or not 1 <= max_bytes <= 16 * 1024 * 1024:
            raise ValueError('journal cap must be between 1 and 16 MiB')
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.byte_count = 0
        self._digest = hashlib.sha256()
        self._failed = False
        self._owner = current_thread()
        self._stream = self.path.open('xb', buffering=0)

    def emit(self, record):
        self._check_owner()
        if self._failed or self._stream is None:
            raise RuntimeError('journal is closed or its attempt has failed')
        try:
            if type(record) is not dict:
                raise ValueError('journal records must be plain objects')
            raw = (json.dumps(record, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')
            if len(raw) > 65536 or self.byte_count + len(raw) > self.max_bytes:
                raise ValueError('journal evidence exceeds its byte cap')
            if self._stream.write(raw) != len(raw):
                raise OSError('journal write was incomplete')
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._digest.update(raw)
            self.byte_count += len(raw)
        except BaseException:
            self._failed = True
            raise

    @property
    def sha256(self):
        self._check_owner()
        if self._failed or self._stream is not None:
            raise RuntimeError('journal digest requires a successful close')
        return self._digest.hexdigest()

    def _check_owner(self):
        if current_thread() is not self._owner:
            raise RuntimeError('journal is restricted to its creating thread')

    def close(self):
        self._check_owner()
        stream = self._stream
        if stream is not None:
            try:
                stream.close()
            except BaseException:
                self._failed = True
                raise
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, _exception_type, _exception, _traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if _exception is not None:
                if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                    raise cleanup_error from _exception
                raise _exception.with_traceback(_traceback) from cleanup_error
            raise
        return False


def _write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False) + '\n').encode('ascii')
    if len(raw) > 1024 * 1024:
        raise ValueError('status record exceeds its byte limit')
    destination = Path(path)
    fd, temp = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp', dir=destination.parent)
    primary = None
    try:
        with os.fdopen(fd, 'wb', buffering=0) as stream:
            if stream.write(raw) != len(raw):
                raise OSError('status write was incomplete')
            stream.flush()
            os.fsync(stream.fileno())
        # The link is the publication point, but an interrupted publisher must
        # still propagate cancellation. Consumers also check the process exit.
        os.link(temp, destination)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            os.unlink(temp)
        except BaseException as cleanup_error:
            if primary is not None:
                if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                    raise cleanup_error from primary
                raise primary.with_traceback(primary.__traceback__) from cleanup_error
            raise
    return hashlib.sha256(raw).hexdigest()


def _same_json(left, right):
    """Compare JSON values without equating bools with integer coordinates."""
    return (json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False)
            == json.dumps(right, sort_keys=True, separators=(',', ':'), allow_nan=False))


def _read_json(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024 or not raw.endswith(b'\n'):
        raise ValueError('missing, oversized or incomplete status record')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate status key')
            result[key] = value
        return result
    result = json.loads(raw.decode('ascii'), object_pairs_hook=unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite status')))
    if type(result) is not dict:
        raise ValueError('status record must be an object')
    return result


def _artifact_inventory(directory, names):
    """Hash bounded, completed evidence; caller prevents concurrent writers."""
    records = {}
    for name, limit in names.items():
        digest, size = hashlib.sha256(), 0
        with (Path(directory) / name).open('rb') as stream:
            while True:
                chunk = stream.read(min(1024 * 1024, limit + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError('evidence artifact exceeds its byte limit')
                digest.update(chunk)
        if size == 0:
            raise ValueError('evidence artifact is empty')
        records[name] = {'bytes': size, 'sha256': digest.hexdigest()}
    return records


def _require_denormals(torch):
    # Integer views preserve the sentinel bits while checking DAZ and FTZ on
    # the calling thread; this is not an attestation of every native thread.
    for integer, floating, normal in ((torch.int32, torch.float32, 0x00800000),
                                      (torch.int64, torch.float64, 0x0010000000000000)):
        values = torch.tensor([1, normal], dtype=integer, device='cpu').view(floating)
        if ((values[0] * 2.0).view(integer).item() != 2
                or (values[1] * 0.5).view(integer).item() != normal // 2):
            raise RuntimeError('the calling thread must preserve denormals')


def audit_cpu_evaluation_journal(manifest, path, expected_sha256):
    """Audit a CPU journal without loading a model, recording local runtime."""
    import platform
    if 'transformers' in sys.modules:
        raise RuntimeError('controller journal audit must not import model dependencies')
    import torch
    if (str(torch.__version__) != '2.8.0+cu128' or torch.cuda.is_initialized()
            or not torch.set_flush_denormal(False)):
        raise RuntimeError('controller audit runtime or CUDA state differs')
    _require_denormals(torch)
    audited = audit_event_journal(manifest, path, split='evaluation', family=None, mask_id=103,
                                  hidden_size=768, expected_sha256=expected_sha256)
    _require_denormals(torch)
    if torch.cuda.is_initialized() or 'transformers' in sys.modules:
        raise RuntimeError('controller audit changed CUDA or model-dependency state')
    return audited, {'python': platform.python_version(), 'torch': str(torch.__version__),
                     'cuda_initialized_before_after': False, 'denormals_preserved_calling_thread': True,
                     'model_forwarded': False}


def _reported_machine(platform_module):
    """Record an unavailable OS machine string explicitly, without guessing."""
    value = platform_module.machine()
    if type(value) is not str:
        raise RuntimeError('platform machine identity must be a string')
    return value or 'unknown'


def _cpu_study_binding(root):
    from .cpu_study_supervision import CPU_STUDY_SUPERVISION_CONTRACT
    canonical = os.path.normcase(str(Path(root).resolve(strict=True)))
    return {'contract': CPU_STUDY_SUPERVISION_CONTRACT,
            'custody_root_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest()}


def _checked_manifest(repo, commit, git_exe, split):
    from .bert_agreement_cases import verify_manifest_bytes
    from .bert_agreement_exposures_v2 import verify_snapshot_bytes

    hashes = verify_git_sources(repo, commit, SOURCE_PATHS, git_exe=git_exe)
    manifest_raw = (Path(repo) / 'experiments/bert_agreement_public_cases_v1.json').read_bytes()
    exposure_raw = (Path(repo) / 'experiments/bert_agreement_exposures_v2.json').read_bytes()
    protocol = (Path(repo) / 'docs/BERT_AGREEMENT_V1.md').read_bytes()
    if (hashlib.sha256(manifest_raw).hexdigest() != hashes['experiments/bert_agreement_public_cases_v1.json']
            or hashlib.sha256(exposure_raw).hexdigest() != hashes['experiments/bert_agreement_exposures_v2.json']
            or hashlib.sha256(protocol).hexdigest() != hashes['docs/BERT_AGREEMENT_V1.md']):
        raise RuntimeError('protocol, manifest or exposure snapshot changed after source verification')
    checked = verify_manifest_bytes(manifest_raw)
    exposure = verify_snapshot_bytes(exposure_raw)
    if (exposure.snapshot_sha256 != hashes['experiments/bert_agreement_exposures_v2.json']
            or exposure.public_evaluation_rows != 64 or exposure.exact_development_evaluation_overlap != 0):
        raise RuntimeError('exposure snapshot does not match the public case inventory')
    lines = protocol.splitlines()
    frozen = b'Status: **frozen before evaluation.**'
    if split == 'evaluation' and (len(lines) < 3 or lines[2] != frozen or lines.count(frozen) != 1):
        raise RuntimeError('the evaluation protocol has not been frozen')
    return hashes, json.loads(manifest_raw), checked.manifest_sha256


def _evaluation_binding(repo, commit, hashes, manifest, manifest_sha, custody_root, ledger_root,
                        controller_custody_root):
    """Derive the ledger identity from verified sources and canonical cases."""
    from .bert_agreement_cases import build_manifest, manifest_digest
    from .bert_agreement_exposures_v3 import build_snapshot, verify_snapshot_bytes
    from .bert_evaluation_ledger import MAX_ATTEMPTS, root_identity

    profile_path = 'experiments/bert_agreement_cpu_evaluation_v1.json'
    exposure_path = 'experiments/bert_agreement_exposures_v3.json'
    profile_raw = (Path(repo) / profile_path).read_bytes()
    exposure_raw = (Path(repo) / exposure_path).read_bytes()
    if (hashlib.sha256(profile_raw).hexdigest() != hashes[profile_path]
            or hashlib.sha256(exposure_raw).hexdigest() != hashes[exposure_path]):
        raise RuntimeError('evaluation profile or exposure snapshot changed after verification')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate evaluation profile key')
            result[key] = value
        return result
    profile = json.loads(profile_raw, object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    checked = verify_snapshot_bytes(exposure_raw)
    ledger, ledger_sha = root_identity(ledger_root)
    custody = Path(custody_root).resolve(strict=True)
    cpu = _cpu_study_binding(custody)
    controller_custody = Path(controller_custody_root).resolve(strict=True)
    controller_cpu = _cpu_study_binding(controller_custody)
    checkout = Path(repo).resolve(strict=True)
    if (type(profile) is not dict or set(profile) != {'schema', 'ledger_root_sha256', 'custody_root_sha256',
            'controller_custody_root_sha256',
            'manifest_sha256', 'exposure_sha256', 'evaluation_rows', 'maximum_attempts'}
            or profile['schema'] != 'bert-agreement-cpu-evaluation-v1'
            or profile['ledger_root_sha256'] != ledger_sha
            or profile['custody_root_sha256'] != cpu['custody_root_sha256']
            or profile['controller_custody_root_sha256'] != controller_cpu['custody_root_sha256']
            or profile['manifest_sha256'] != manifest_sha or manifest_sha != manifest_digest()
            or not _same_json(manifest, build_manifest()) or profile['exposure_sha256'] != checked.snapshot_sha256
            or type(profile['evaluation_rows']) is not int or profile['evaluation_rows'] != 64
            or type(profile['maximum_attempts']) is not int or profile['maximum_attempts'] != MAX_ATTEMPTS
            or not custody.is_dir() or not controller_custody.is_dir()
            or any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                   for a, b in ((ledger, custody), (ledger, checkout), (custody, checkout),
                                (controller_custody, ledger), (controller_custody, custody),
                                (controller_custody, checkout)))):
        raise RuntimeError('evaluation profile does not match the frozen roots or inputs')
    def rows(data):
        return [{'row_id': item['row_id'], 'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']}
                for item in data if item['split'] == 'evaluation']
    evaluation_rows = rows(manifest['rows'])
    if checked.public_evaluation_rows != 64 or checked.exact_development_evaluation_overlap != 0 or not _same_json(
            evaluation_rows, rows(build_snapshot()['rows'])):
        raise RuntimeError('evaluation rows differ from the cited exposure snapshot')
    return {'source_commit': commit, 'source_sha256': hashes,
            'protocol_sha256': hashes['docs/BERT_AGREEMENT_V1.md'], 'manifest_sha256': manifest_sha,
            'exposure_sha256': checked.snapshot_sha256, 'ledger_root_sha256': ledger_sha,
            'cpu_study': cpu, 'controller_cpu_study': controller_cpu,
            'model': {'model': MODEL_ID, 'revision': REVISION,
                                       'assets': {name: {'size': size, 'sha256': digest}
                                                  for name, (size, digest) in ASSETS.items()}},
            'rows': evaluation_rows}


def _require_bootstrap():
    if (os.environ.get('LOOKING_GLASS_BERT_BOOTSTRAP_PID') != str(os.getpid())
            or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('-1', '')
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or not sys.pycache_prefix or any(Path(sys.pycache_prefix).iterdir())):
        raise RuntimeError('the source-isolated, CPU-masked bootstrap is required')


def _require_checkout(repo):
    actual = Path(__file__).resolve(strict=True).parent.parent
    bootstrap = os.environ.get('LOOKING_GLASS_BERT_ROOT')
    if (not bootstrap or Path(bootstrap).resolve(strict=True) != actual
            or Path(repo).resolve(strict=True) != actual):
        raise RuntimeError('requested checkout differs from the executing source')
    return actual


def validate_cpu_runtime_and_model(runtime, model, state_sha):
    """Enforce the fixed CPU model and runtime evidence for a completed run."""
    expected_model = {'model': MODEL_ID, 'revision': REVISION,
                      'assets': {name: {'size': size, 'sha256': sha} for name, (size, sha) in ASSETS.items()},
                      'model_class': 'DistilBertForMaskedLM', 'tokenizer_class': 'DistilBertTokenizerFast',
                      'torch_version': '2.8.0+cu128', 'transformers_version': '4.57.3',
                      'dtype': 'float32', 'device': 'cpu', 'attention_implementation': 'eager',
                      'parameter_count': 66985530, 'pytorch_cuda_initialized': False}
    if (type(runtime) is not dict or runtime.get('pytorch_cuda_initialized') is not False
            or runtime.get('denormals_preserved_calling_thread') is not True
            or runtime.get('deterministic_algorithms') is not True
            or type(runtime.get('intraop_threads')) is not int or runtime['intraop_threads'] != 1
            or type(runtime.get('interop_threads')) is not int or runtime['interop_threads'] != 1
            or runtime.get('torch') != '2.8.0+cu128' or runtime.get('transformers') != '4.57.3'
            or runtime.get('numpy') != '2.4.6' or type(runtime.get('seed')) is not int
            or runtime['seed'] != 2026102810 or any(type(runtime.get(k)) is not str or not runtime[k]
                                                     for k in ('python', 'platform', 'machine'))
            or runtime.get('python_hash_seed') != 'isolated interpreter; OS-randomized'
            or type(model) is not dict or not _same_json(model, expected_model)
            or type(state_sha) is not str or re.fullmatch('[0-9a-f]{64}', state_sha) is None):
        raise RuntimeError('owner runtime or model evidence differs from the fixed contract')


def _owner_main(*, repo, cache, run_dir, commit, git_exe, split, family, run_id, custody_root=None,
                ledger_root=None, controller_custody_root=None):
    """Run one CPU owner; the supervising controller decides final status."""
    stage = 'startup'
    try:
        _require_bootstrap()
        _require_checkout(repo)
        if split == 'evaluation' and (not EVALUATION_ENABLED or custody_root is None or ledger_root is None
                                       or controller_custody_root is None):
            raise RuntimeError('evaluation requires the reviewed frozen custody and ledger')
        cpu_binding = None
        if custody_root is not None:
            stage = 'custody_claim'
            from .cpu_study_supervision import claim_cpu_study
            session, lease = claim_cpu_study(custody_root)
            _CPU_CUSTODY.update(session=session, lease=lease)
            cpu_binding = _cpu_study_binding(custody_root)
        if 'torch' in sys.modules or 'transformers' in sys.modules:
            raise RuntimeError('model dependencies were imported before owner checks')
        if split not in ('development', 'evaluation') or (split == 'development' and
                (type(family) is not int or not 1 <= family <= 4)) or (split == 'evaluation' and family is not None):
            raise ValueError('invalid study selection')
        stage = 'source_preflight'
        hashes, manifest, manifest_sha = _checked_manifest(repo, commit, git_exe, split)
        evaluation_ledger = None
        if split == 'evaluation':
            from .bert_evaluation_ledger import reservation_binding, verify_owner_claim
            binding = _evaluation_binding(repo, commit, hashes, manifest, manifest_sha, custody_root, ledger_root,
                                          controller_custody_root)
            session.check(lease)
            reservation, reservation_sha = verify_owner_claim(
                ledger_root, binding['ledger_root_sha256'], run_dir, run_id, binding,
                supervisor_pid=session.supervisor_pid, supervisor_birth=session.supervisor_birth)
            evaluation_ledger = reservation_binding(reservation, reservation_sha)
        owner_started = {'schema': 'bert-agreement-owner-start-v1', 'run_id': run_id,
                         'pid': os.getpid(), 'source_commit': commit, 'manifest_sha256': manifest_sha,
                         'split': split, 'family': family}
        if cpu_binding is not None:
            owner_started['cpu_study'] = cpu_binding
        if evaluation_ledger is not None:
            owner_started['evaluation_ledger'] = evaluation_ledger
            owner_started['cpu_owner_session_run_id'] = session.run_id
        _write_json(Path(run_dir) / 'owner-started.json', owner_started)
        stage = 'model_load'
        import gc
        import platform
        import numpy
        import torch
        import transformers
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise RuntimeError('unable to request denormal-preserving CPU mode')
        _require_denormals(torch)
        torch.random.default_generator.manual_seed(2026102810)
        if (str(torch.__version__) != '2.8.0+cu128' or str(transformers.__version__) != '4.57.3'
                or str(numpy.__version__) != '2.4.6'):
            raise RuntimeError('runtime versions differ from the fixed CPU protocol')
        loaded = load_pinned_distilbert(cache)
        if verify_tokenizer_rows(loaded.tokenizer, manifest) != 96:
            raise RuntimeError('the complete token inventory was not checked')
        witness = capture_model_state(loaded.model)
        state_sha, model_metadata = witness.sha256, loaded.metadata
        runtime = {'python': platform.python_version(), 'platform': platform.system(),
                   'machine': _reported_machine(platform), 'torch': str(torch.__version__),
                   'transformers': str(transformers.__version__), 'numpy': str(numpy.__version__),
                   'intraop_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                   'denormals_preserved_calling_thread': True, 'pytorch_cuda_initialized': False,
                   'python_hash_seed': 'isolated interpreter; OS-randomized', 'seed': 2026102810}
        from .bert_agreement_runner import _has_hooks, run_family
        if _has_hooks(loaded.model):
            raise RuntimeError('model has preexisting hooks')
        stage = 'forwards'
        live = []
        selected = [family] if split == 'development' else list(range(1, 9))
        with EventJournal(Path(run_dir) / 'events.jsonl') as journal:
            for number in selected:
                _require_denormals(torch)
                live.append(run_family(loaded.model, manifest, split=split, family=number,
                                       is_id=2003, are_id=2024, mask_id=103, emit=journal.emit))
                _require_denormals(torch)
        journal_sha = journal.sha256
        stage = 'verification'
        if torch.cuda.is_initialized() or _has_hooks(loaded.model):
            raise RuntimeError('CPU CUDA state or hook cleanup changed')
        verify_model_state(loaded.model, witness)
        audited = audit_event_journal(manifest, Path(run_dir) / 'events.jsonl', split=split,
                                      family=family, mask_id=103, hidden_size=768,
                                      expected_sha256=journal_sha)
        if not _same_json(audited['families'], live):
            raise RuntimeError('closed evidence does not reproduce live family results')
        if verify_git_sources(repo, commit, SOURCE_PATHS, git_exe=git_exe) != hashes:
            raise RuntimeError('committed sources changed during execution')
        _require_denormals(torch)
        del witness, loaded
        gc.collect()
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU owner initialized CUDA during teardown')
        candidate = {'schema': 'bert-agreement-owner-v1', 'status': 'complete',
                     'run_id': run_id, 'pid': os.getpid(), 'source_commit': commit,
                     'source_sha256': hashes, 'manifest_sha256': manifest_sha,
                     'split': split, 'family': family, 'journal_sha256': journal_sha,
                     'runtime': runtime, 'model': model_metadata,
                     'model_state_sha256_before_after': state_sha, 'audited': audited}
        if cpu_binding is not None:
            candidate['cpu_study'] = cpu_binding
        if evaluation_ledger is not None:
            candidate['evaluation_ledger'] = evaluation_ledger
            candidate['cpu_owner_session_run_id'] = session.run_id
        _write_json(Path(run_dir) / 'owner-candidate.json', candidate)
        if custody_root is not None:
            session.attest_clean(lease)
        return 0
    except BaseException as error:
        try:
            _write_json(Path(run_dir) / 'owner-failure.json',
                        {'schema': 'bert-agreement-owner-failure-v1', 'status': 'incomplete',
                         'run_id': run_id, 'pid': os.getpid(), 'stage': stage,
                         'error_type': type(error).__name__})
        except (KeyboardInterrupt, SystemExit) as cancellation:
            try:
                _write_json(Path(run_dir) / 'owner-cancelled.json',
                            {'schema': 'bert-agreement-owner-cancel-v1', 'status': 'incomplete',
                             'run_id': run_id, 'pid': os.getpid(), 'stage': stage,
                             'error_type': type(cancellation).__name__,
                             'previous_error_type': type(error).__name__})
            except Exception:
                pass
            raise cancellation from error
        except Exception:
            pass
        return 1


def _run_cpu_controller(*, repo, cache, output_dir, commit, split, family=None, timeout=600, custody_root=None,
                        ledger_root=None, ledger_context=None, controller_custody_root=None):
    """Run an isolated CPU owner, then audit its closed evidence after exit."""
    import secrets
    import shutil

    _require_bootstrap()
    if split == 'evaluation' and (not EVALUATION_ENABLED or ledger_context is None):
        raise RuntimeError('evaluation requires the reviewed frozen custody and ledger')
    if (split not in ('development', 'evaluation') or (split == 'development' and
            (type(family) is not int or not 1 <= family <= 4)) or (split == 'evaluation' and family is not None)
            or type(timeout) is not int or not 1 <= timeout <= 1200):
        raise ValueError('invalid study selection or timeout')
    root, cache_path = Path(repo).resolve(strict=True), Path(cache).resolve(strict=True)
    destination = Path(output_dir)
    if ledger_context is None:
        if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
            raise ValueError('output must be a new absolute directory with an existing parent')
        destination.mkdir()
        run_id = secrets.token_hex(16)
    else:
        if (split != 'evaluation' or not destination.is_absolute() or destination.is_symlink()
                or not destination.is_dir() or destination != ledger_context['directory']):
            raise ValueError('evaluation output must be its reserved attempt directory')
        run_id = ledger_context['reservation']['run_id']
    child = None
    owner_pid = None
    exit_code = None
    manifest_sha = None
    owner_state = 'not_started'
    cleanup_error_type = None
    wait_error_type = None
    owner_failure = None
    owner_failure_read_error_type = None
    owner_cancellation = None
    owner_cancellation_read_error_type = None
    supervision = None
    cpu_binding = None
    evaluation_ledger = None
    stage = 'startup'
    try:
        if 'transformers' in sys.modules:
            raise RuntimeError('the controller cannot import model dependencies')
        if 'torch' in sys.modules:
            # A previous complete attempt is re-audited after the outer CPU
            # claim. It is the only permitted source of this Torch import.
            audits = None if ledger_context is None else ledger_context['reservation']['prior_audits']
            torch = sys.modules['torch']
            if (not audits or str(torch.__version__) != '2.8.0+cu128'
                    or torch.cuda.is_initialized()):
                raise RuntimeError('unexpected controller Torch state')
            _require_denormals(torch)
        controller_started = {'schema': 'bert-agreement-controller-start-v1',
                              'run_id': run_id, 'source_commit': commit, 'split': split, 'family': family,
                              'started_utc': datetime.now(timezone.utc).isoformat()}
        _write_json(destination / 'started.json', controller_started)
        stage = 'preflight'
        _require_checkout(root)
        git_exe = shutil.which('git')
        site = os.environ.get('LOOKING_GLASS_BERT_SITE')
        if not git_exe or not site or not Path(site).is_absolute() or not Path(site).is_dir():
            raise RuntimeError('Git and the bootstrap dependency directory are required')
        git_exe = str(Path(git_exe).resolve(strict=True))
        hashes, manifest, manifest_sha = _checked_manifest(root, commit, git_exe, split)
        if custody_root is not None:
            custody = Path(custody_root)
            if not custody.is_absolute() or not custody.is_dir():
                raise ValueError('CPU study custody root must be an existing absolute directory')
            custody = custody.resolve(strict=True)
            output_root = destination.resolve(strict=True)
            if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                   for a, b in ((custody, root), (custody, output_root))):
                raise ValueError('CPU study custody root must be separate from checkout and attempt')
            cpu_binding = _cpu_study_binding(custody)
        if ledger_context is not None:
            from .bert_evaluation_ledger import reservation_binding
            ledger_context['ledger']._check()
            binding = _evaluation_binding(root, commit, hashes, manifest, manifest_sha, custody_root, ledger_root,
                                          controller_custody_root)
            if not _same_json(binding, ledger_context['binding']):
                raise RuntimeError('evaluation binding changed after reservation')
            evaluation_ledger = reservation_binding(ledger_context['reservation'], ledger_context['reservation_sha256'])
        preflight = {'schema': 'bert-agreement-preflight-v1', 'run_id': run_id, 'source_commit': commit,
                     'source_sha256': hashes, 'manifest_sha256': manifest_sha, 'split': split, 'family': family}
        if cpu_binding is not None:
            preflight['cpu_study'] = cpu_binding
        if evaluation_ledger is not None:
            preflight['evaluation_ledger'] = evaluation_ledger
        _write_json(destination / 'preflight.json', preflight)
        prefix = destination / 'owner-pycache'
        prefix.mkdir()
        working = destination / 'owner-working'
        temporary = working / 'tmp'
        temporary.mkdir(parents=True)
        environment = {key: os.environ[key] for key in
                       ('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT') if key in os.environ}
        system_root = Path(environment.get('SYSTEMROOT', 'C:/Windows'))
        environment.update({'PATH': str(system_root / 'System32') if os.name == 'nt' else '/usr/bin:/bin',
                            'TEMP': str(temporary), 'TMP': str(temporary),
                            'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                            'OPENBLAS_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                            'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_DISABLE_TELEMETRY': '1'})
        command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                   str(root / 'scripts/bert_agreement_bootstrap.py'), '--bootstrap-site', site,
                   '--worker', '--repo', str(root), '--cache', str(cache_path),
                   '--output', str(destination), '--commit', commit, '--git', git_exe,
                   '--split', split, '--family', str(family or 0), '--run-id', run_id]
        if custody_root is not None:
            command.extend(('--custody-root', str(custody)))
        if evaluation_ledger is not None:
            command.extend(('--ledger-root', str(ledger_root)))
            command.extend(('--controller-custody-root', str(controller_custody_root)))
            ledger_context['ledger']._check()
            ledger_context['controller_session'].check(ledger_context['controller_lease'])
        stage = 'owner_start'
        owner_state = 'possibly_active'
        if custody_root is not None:
            from .cpu_study_supervision import run_cpu_study
            from .supervisor import SupervisionResult
            observed = run_cpu_study(command, study_root=custody, timeout=timeout,
                                     cwd=working, env=environment, inherit_environment=False)
            if (type(observed) is not SupervisionResult or type(observed.pid) is not int or observed.pid <= 0
                    or (observed.exit_code is not None and type(observed.exit_code) is not int)
                    or type(observed.clean) is not bool or type(observed.timed_out) is not bool
                    or type(observed.run_id) is not str or re.fullmatch('[0-9a-f]{32}', observed.run_id) is None
                    or type(observed.reason) is not str):
                raise RuntimeError('CPU study supervision result is malformed')
            owner_pid, exit_code = observed.pid, observed.exit_code
            supervision = {'scope': 'cooperative-cpu-study-custody-only',
                           'run_id': observed.run_id, 'pid': observed.pid,
                           'exit_code': observed.exit_code, 'clean': observed.clean,
                           'timed_out': observed.timed_out, 'reason': observed.reason}
            owner_state = 'reaped' if exit_code is not None else 'possibly_active'
        else:
            # Legacy development-only path: on ambiguous Popen return the
            # recorded owner state stays possibly active and blocks a retry.
            child = subprocess.Popen(command, cwd=working, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            owner_pid = child.pid
            try:
                stage = 'owner_wait'
                exit_code = child.wait(timeout=timeout)
                owner_state = 'reaped'
            except BaseException as wait_error:
                wait_error_type = type(wait_error).__name__
                try:
                    if child.poll() is None:
                        child.kill()
                    exit_code = child.wait(timeout=10)
                    owner_state = 'reaped'
                except BaseException as cleanup_error:
                    cleanup_error_type = type(cleanup_error).__name__
                    if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                        raise cleanup_error from wait_error
                raise
        if exit_code != 0:
            if owner_state != 'reaped':
                raise RuntimeError('CPU owner termination remains uncertain')
            if (destination / 'owner-cancelled.json').exists():
                try:
                    record = _read_json(destination / 'owner-cancelled.json')
                    if (set(record) != {'schema', 'status', 'run_id', 'pid', 'stage',
                                        'error_type', 'previous_error_type'}
                            or record['schema'] != 'bert-agreement-owner-cancel-v1'
                            or record['status'] != 'incomplete' or record['run_id'] != run_id
                            or type(record['pid']) is not int or record['pid'] != owner_pid
                            or record['stage'] not in ('startup', 'custody_claim', 'source_preflight',
                                                       'model_load', 'forwards', 'verification')
                            or record['error_type'] not in ('KeyboardInterrupt', 'SystemExit')
                            or type(record['previous_error_type']) is not str
                            or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', record['previous_error_type']) is None):
                        raise ValueError('invalid owner cancellation record')
                    owner_cancellation = record
                except Exception as cancellation_error:
                    owner_cancellation_read_error_type = type(cancellation_error).__name__
            try:
                record = _read_json(destination / 'owner-failure.json')
                if (set(record) != {'schema', 'status', 'run_id', 'pid', 'stage', 'error_type'}
                        or record['schema'] != 'bert-agreement-owner-failure-v1'
                        or record['status'] != 'incomplete' or record['run_id'] != run_id
                        or type(record['pid']) is not int or record['pid'] != owner_pid
                        or record['stage'] not in ('startup', 'custody_claim', 'source_preflight',
                                                   'model_load', 'forwards', 'verification')
                        or type(record['error_type']) is not str
                        or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', record['error_type']) is None):
                    raise ValueError('invalid owner failure record')
                owner_failure = record
            except Exception as failure_error:
                owner_failure_read_error_type = type(failure_error).__name__
            raise RuntimeError('the owner exited without a successful result')
        if supervision is not None and (supervision['clean'] is not True or supervision['timed_out'] is not False):
            raise RuntimeError('the CPU study owner did not establish clean custody')
        stage = 'evidence_validation'
        candidate = _read_json(destination / 'owner-candidate.json')
        controller_audit_runtime = None
        if split == 'evaluation':
            audited, controller_audit_runtime = audit_cpu_evaluation_journal(
                manifest, destination / 'events.jsonl', candidate.get('journal_sha256'))
        else:
            audited = audit_event_journal(manifest, destination / 'events.jsonl', split=split, family=family,
                                          mask_id=103, hidden_size=768, expected_sha256=candidate.get('journal_sha256'))
        validate_owner_result(candidate, run_id=run_id, pid=owner_pid, exit_code=exit_code,
                              source_commit=commit, journal_sha256=audited['sha256'])
        started = _read_json(destination / 'owner-started.json')
        expected_started = {'schema': 'bert-agreement-owner-start-v1', 'run_id': run_id,
                            'pid': owner_pid, 'source_commit': commit, 'manifest_sha256': manifest_sha,
                            'split': split, 'family': family}
        if cpu_binding is not None:
            expected_started['cpu_study'] = cpu_binding
        if evaluation_ledger is not None:
            expected_started['evaluation_ledger'] = evaluation_ledger
            expected_started['cpu_owner_session_run_id'] = supervision['run_id']
        if (not _same_json(started, expected_started)
                or not _same_json(_read_json(destination / 'started.json'), controller_started)
                or not _same_json(_read_json(destination / 'preflight.json'), preflight)
                or candidate.get('source_sha256') != hashes or candidate.get('manifest_sha256') != manifest_sha
                or not _same_json(candidate.get('cpu_study'), cpu_binding)
                or not _same_json(candidate.get('evaluation_ledger'), evaluation_ledger)
                or (evaluation_ledger is not None and candidate.get('cpu_owner_session_run_id') != supervision['run_id'])
                or candidate.get('split') != split or not _same_json(candidate.get('family'), family)
                or not _same_json(candidate.get('audited'), audited) or any(prefix.iterdir())
                or verify_git_sources(root, commit, SOURCE_PATHS, git_exe=git_exe) != hashes):
            raise RuntimeError('owner evidence, source or bytecode prefix changed')
        stage = 'runtime_validation'
        runtime, model = candidate.get('runtime'), candidate.get('model')
        validate_cpu_runtime_and_model(runtime, model, candidate.get('model_state_sha256_before_after'))
        stage = 'evidence_inventory'
        artifact_limits = {'started.json': 1024 * 1024,
            'preflight.json': 1024 * 1024, 'owner-started.json': 1024 * 1024,
            'owner-candidate.json': 1024 * 1024, 'events.jsonl': 16 * 1024 * 1024}
        if evaluation_ledger is not None:
            artifact_limits['reservation.json'] = 1024 * 1024
        artifacts = _artifact_inventory(destination, artifact_limits)
        if artifacts['events.jsonl']['sha256'] != audited['sha256']:
            raise RuntimeError('closed journal changed during evidence inventory')
        result = {'schema': 'bert-agreement-controller-result-v1', 'status': 'complete',
                  'run_id': run_id, 'owner_pid': owner_pid, 'owner_exit_code': exit_code, 'owner_state': owner_state,
                  'cpu_supervision': supervision, 'cpu_study': cpu_binding,
                  'source_commit': commit, 'source_sha256': hashes, 'manifest_sha256': manifest_sha,
                  'split': split, 'family': family, 'journal_sha256': audited['sha256'],
                  'runtime': runtime, 'model': model, 'artifacts': artifacts,
                  'model_state_sha256_before_after': candidate['model_state_sha256_before_after'],
                  'summary': audited['summary'], 'families': audited['families'],
                  'finished_utc': datetime.now(timezone.utc).isoformat(),
                  'limitations': ['public fixtures are held out only from local development',
                                  'full mask logits are represented by hashes, not retained tensors',
                                  'source hashes and event consistency do not attest faithful execution',
                                  'one checkpoint and correlated families do not establish generalization']}
        if evaluation_ledger is not None:
            result['evaluation_ledger'] = evaluation_ledger
            result['controller_audit_runtime'] = controller_audit_runtime
            result['evaluation_interpretation'] = {
                'designation': 'replay' if evaluation_ledger['prior_complete'] else 'first-complete-primary',
                'prior_attempt_may_have_exposed_rows': evaluation_ledger['prior_may_have_exposed'],
                'unknown_history_and_pretraining': True}
        stage = 'terminal_publication'
        _write_json(destination / 'result.json', result)
        return result
    except BaseException as error:
        result = {'schema': 'bert-agreement-controller-result-v1', 'status': 'incomplete',
                  'run_id': run_id, 'source_commit': commit, 'manifest_sha256': manifest_sha,
                  'split': split, 'family': family, 'owner_pid': owner_pid,
                  'cpu_supervision': supervision, 'cpu_study': cpu_binding,
                  'owner_exit_code': exit_code, 'owner_state': owner_state, 'stage': stage,
                  'error_type': type(error).__name__, 'wait_error_type': wait_error_type,
                  'cleanup_error_type': cleanup_error_type,
                  'cancelled': isinstance(error, (KeyboardInterrupt, SystemExit)) or owner_cancellation is not None or
                               (owner_failure is not None and owner_failure['error_type'] in ('KeyboardInterrupt', 'SystemExit')),
                  'timed_out': isinstance(error, subprocess.TimeoutExpired) or
                               (supervision is not None and supervision['timed_out']),
                  'owner_failure': owner_failure, 'owner_failure_read_error_type': owner_failure_read_error_type,
                  'owner_cancellation': owner_cancellation,
                  'owner_cancellation_read_error_type': owner_cancellation_read_error_type,
                  'finished_utc': datetime.now(timezone.utc).isoformat()}
        if evaluation_ledger is not None:
            result['evaluation_ledger'] = evaluation_ledger
        try:
            _write_json(destination / 'controller-failure.json', result)
        except (KeyboardInterrupt, SystemExit) as cancellation:
            cancelled_result = dict(result, error_type=type(cancellation).__name__, cancelled=True,
                                    previous_error_type=type(error).__name__)
            try:
                _write_json(destination / 'controller-cancelled.json', cancelled_result)
            except Exception:
                pass
            raise cancellation from error
        except Exception:
            pass
        try:
            if not (destination / 'result.json').exists():
                _write_json(destination / 'result.json', result)
        except (KeyboardInterrupt, SystemExit) as cancellation:
            cancelled_result = dict(result, error_type=type(cancellation).__name__, cancelled=True,
                                    previous_error_type=type(error).__name__)
            try:
                _write_json(destination / 'controller-cancelled.json', cancelled_result)
            except Exception:
                pass
            raise cancellation from error
        except Exception:
            pass
        return result


def run_cpu_controller(*, repo, cache, output_dir, commit, split, family=None, timeout=600,
                       custody_root=None, ledger_root=None, controller_custody_root=None):
    """Serialize each evaluation reservation through its complete controller attempt."""
    if split != 'evaluation':
        if ledger_root is not None or controller_custody_root is not None:
            raise ValueError('development attempts do not use the evaluation ledger')
        return _run_cpu_controller(repo=repo, cache=cache, output_dir=output_dir, commit=commit,
                                   split=split, family=family, timeout=timeout, custody_root=custody_root)
    if (not EVALUATION_ENABLED or custody_root is None or ledger_root is None
            or controller_custody_root is None):
        raise RuntimeError('evaluation requires the reviewed frozen custody and ledger')
    if family is not None or type(timeout) is not int or not 1 <= timeout <= 1200:
        raise ValueError('evaluation must use all families and a bounded timeout')
    if not Path(cache).resolve(strict=True).is_dir():
        raise ValueError('checkpoint cache must be an existing directory')
    import shutil
    from .bert_evaluation_ledger import EvaluationLedger, verify_reservation
    from .cpu_study_supervision import claim_cpu_study

    _require_bootstrap()
    # The controller is itself a separately supervised direct child. Claim
    # before source preflight, reservation, or any model-capable dependency.
    session, controller_lease = claim_cpu_study(controller_custody_root)
    _CONTROLLER_CUSTODY.update(session=session, lease=controller_lease)
    checkout = _require_checkout(repo)
    cache_path = Path(cache).resolve(strict=True)
    for study_root in (Path(custody_root).resolve(strict=True), Path(controller_custody_root).resolve(strict=True),
                       Path(ledger_root).resolve(strict=True)):
        if (study_root == cache_path or study_root.is_relative_to(cache_path)
                or cache_path.is_relative_to(study_root)):
            raise ValueError('cache and study custody roots must be separate')
    git_exe = shutil.which('git')
    if not git_exe:
        raise RuntimeError('Git is required for evaluation preflight')
    hashes, manifest, manifest_sha = _checked_manifest(checkout, commit, git_exe, split)
    binding = _evaluation_binding(checkout, commit, hashes, manifest, manifest_sha, custody_root, ledger_root,
                                  controller_custody_root)
    destination = Path(output_dir)
    root = Path(ledger_root).resolve(strict=True)
    if (not destination.is_absolute() or destination.parent.resolve(strict=True) != root
            or re.fullmatch('[0-9a-f]{32}', destination.name) is None or destination.exists()):
        raise ValueError('evaluation output must name a new attempt within the frozen ledger root')
    ledger = EvaluationLedger.acquire(root, binding['ledger_root_sha256'], binding)
    known_inner_quarantine = False
    try:
        session.check(controller_lease)
        marker = Path(custody_root).resolve(strict=True) / 'cpu-study.unsafe.json'
        # Avoid consuming an attempt for a quarantine already visible while
        # the ledger lock is held. This is not atomic admission against other
        # users of the owner root; its supervisor remains authoritative.
        known_inner_quarantine = os.path.lexists(marker) or os.path.lexists(str(marker) + '.attestation')
        if not known_inner_quarantine:
            directory, reservation_sha, _ = ledger.reserve(destination.name, controller_supervision_run_id=session.run_id)
            reservation, checked_sha = verify_reservation(directory, destination.name, binding)
            if checked_sha != reservation_sha:
                raise RuntimeError('evaluation reservation changed during publication')
            context = {'ledger': ledger, 'directory': directory, 'reservation': reservation,
                       'reservation_sha256': reservation_sha, 'binding': binding,
                       'controller_session': session, 'controller_lease': controller_lease}
            result = _run_cpu_controller(repo=repo, cache=cache, output_dir=directory, commit=commit, split=split,
                                         family=family, timeout=timeout, custody_root=custody_root,
                                         ledger_root=root, ledger_context=context,
                                         controller_custody_root=controller_custody_root)
    finally:
        ledger.release()
    if known_inner_quarantine:
        # No reservation or inner launch occurred in this controller.
        session.attest_clean(controller_lease)
        raise RuntimeError('inner CPU owner is quarantined; no evaluation attempt was reserved')
    # Only attest after the inner owner is known reaped or never started and
    # the ledger lock is released. Retain the outer lease until process exit.
    if result.get('owner_state') not in ('not_started', 'reaped'):
        raise RuntimeError('inner owner state is uncertain; retain controller quarantine')
    session.attest_clean(controller_lease)
    return result


def record_cpu_evaluation_exit(*, repo, commit, output_dir, run_id, supervision, custody_root, ledger_root,
                               controller_custody_root):
    """Record the local launcher's observed controller exit, without a model load."""
    if (not EVALUATION_ENABLED or custody_root is None or ledger_root is None
            or controller_custody_root is None):
        raise RuntimeError('evaluation requires the reviewed frozen custody and ledger')
    import shutil
    from .bert_evaluation_ledger import record_controller_exit

    _require_bootstrap()
    checkout = _require_checkout(repo)
    git_exe = shutil.which('git')
    if not git_exe:
        raise RuntimeError('Git is required for exit evidence')
    hashes, manifest, manifest_sha = _checked_manifest(checkout, commit, git_exe, 'evaluation')
    binding = _evaluation_binding(checkout, commit, hashes, manifest, manifest_sha, custody_root, ledger_root,
                                  controller_custody_root)
    destination = Path(output_dir)
    root = Path(ledger_root).resolve(strict=True)
    if (not destination.is_absolute() or destination.parent.resolve(strict=True) != root
            or destination.name != run_id):
        raise ValueError('exit evidence must name the reserved attempt')
    return record_controller_exit(root, binding['ledger_root_sha256'], run_id, binding, supervision)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--split', choices=('development', 'evaluation'), required=True)
    parser.add_argument('--family', type=int, default=0)
    parser.add_argument('--custody-root', help='existing persistent CPU study custody directory')
    parser.add_argument('--ledger-root', help='existing frozen evaluation attempt ledger')
    parser.add_argument('--controller-custody-root', help='distinct supervised controller custody directory')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--git', help=argparse.SUPPRESS)
    parser.add_argument('--run-id', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    family = args.family or None
    if args.worker:
        return _owner_main(repo=args.repo, cache=args.cache, run_dir=args.output, commit=args.commit,
                           git_exe=args.git, split=args.split, family=family, run_id=args.run_id,
                           custody_root=args.custody_root, ledger_root=args.ledger_root,
                           controller_custody_root=args.controller_custody_root)
    result = run_cpu_controller(repo=args.repo, cache=args.cache, output_dir=args.output,
                                commit=args.commit, split=args.split, family=family, custody_root=args.custody_root,
                                ledger_root=args.ledger_root, controller_custody_root=args.controller_custody_root)
    print(json.dumps({'status': result['status'], 'run_id': result['run_id']}, sort_keys=True))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
