"""Fixed exposed-pair FLAN-T5 CUDA development smoke; never a GPU service."""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from .device_lease import DeviceLease
from .flan_t5_device_owner import FlanT5CudaOwner
from .supervisor import OwnerSession

_CASE_HASH = 'e00d77cf45885b10e19fe689b8cdb13d2d36dd038c7ab2c1701f9367eb2fe9fc'
_ASSET_HASH = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
_REVISION = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
_A = [30355, 15, 1566, 12, 2968, 10, 484, 1]
_B = [30355, 15, 1566, 12, 2968, 10, 629, 1]
_TARGET = [4675, 1]
_LABELS = ('capture-a', 'capture-b', 'repeat-a', 'self-patch', 'donor-patch', 'null-a')
_LIMIT = 16 * 1024 ** 2
_CUSTODY = {'entered': False, 'session': None, 'lease': None, 'owner': None,
            'evidence': None, 'failure': None, 'diagnostic_failure': None}


def _fixed_rows(repo):
    path = Path(repo) / 'docs/FLAN_T5_PUBLIC_CASES_V1.json'
    with path.open('rb') as stream:
        raw = stream.read(1024 ** 2 + 1)
    if len(raw) > 1024 ** 2 or hashlib.sha256(raw).hexdigest() != _CASE_HASH:
        raise ValueError('fixed public case manifest differs')
    manifest = json.loads(raw)
    if (manifest.get('revision') != _REVISION or manifest.get('asset_manifest_sha256') != _ASSET_HASH
            or manifest.get('model') != 'google/flan-t5-small'):
        raise ValueError('fixed public case identity differs')
    pair = manifest['pairs'][1]
    a, b, target = pair['sources'][0], pair['sources'][1], pair['targets'][0]
    if (a['input_ids'] != _A or b['input_ids'] != _B or target['input_ids'] != _TARGET
            or a['attention_mask'] != [1] * 8 or b['attention_mask'] != [1] * 8
            or a.get('word_position') != 6 or b.get('word_position') != 6):
        raise ValueError('fixed public rows differ')
    return a, b, target


class SmokeEvidence:
    """Create-only, bounded private files; partial writes remain evidence."""

    def __init__(self, root, *, run_id):
        root = Path(root)
        if (not root.is_absolute() or not root.is_dir() or type(run_id) is not str
                or re.fullmatch('[0-9a-f]{32}', run_id) is None):
            raise ValueError('an existing absolute attempt directory and run ID are required')
        self.root = root
        self.run_id = run_id
        self.used = 0
        self.memory_count = 0
        self.forward_count = 0
        self.phase = 'startup'

    def _bytes(self, name, raw):
        if (type(name) is not str or re.fullmatch('[A-Za-z0-9_.-]+', name) is None
                or type(raw) is not bytes or len(raw) > 2 * 1024 ** 2
                or self.used + len(raw) > _LIMIT):
            raise ValueError('evidence name or size violates its fixed bound')
        self.used += len(raw)
        with (self.root / name).open('xb') as stream:
            if stream.write(raw) != len(raw):
                raise OSError('short evidence write')
            stream.flush()
            os.fsync(stream.fileno())
        return hashlib.sha256(raw).hexdigest()

    def json(self, name, value):
        raw = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')
        return self._bytes(name, raw)

    def record_stage(self, stage, snapshot):
        if self.memory_count >= 64:
            raise ValueError('memory evidence exceeds its fixed count')
        index = self.memory_count
        self.memory_count += 1
        self.json(f'memory-{index:03d}.json', {'schema': 'flan-t5-gpu-memory-v1',
            'run_id': self.run_id, 'index': index, 'phase': self.phase,
            'stage': stage, 'memory': snapshot,
            'peak_scope': 'cumulative-since-pre-model-load-reset'})

    def begin(self, label):
        if self.forward_count >= len(_LABELS) or label != _LABELS[self.forward_count]:
            raise RuntimeError('forward order differs from the fixed smoke')
        index = self.forward_count
        self.forward_count += 1
        self.phase = label
        self.json(f'forward-{index:03d}-started.json', {'schema': 'flan-t5-gpu-forward-v1',
            'run_id': self.run_id, 'index': index, 'label': label})

    def save(self, label, observation):
        import torch
        from .flan_t5_device_observation import Observation
        expected_source = _B if label == 'capture-b' else _A
        if (label != self.phase or type(observation) is not Observation or sys.byteorder != 'little'
                or tuple(observation.source_ids) != tuple(expected_source)
                or tuple(observation.source_mask) != (1,) * 8
                or tuple(observation.decoder_ids) != (0, 4675)
                or tuple(observation.decoder_mask) != (1, 1)):
            raise ValueError('observation input identity differs')
        specs = (('logits', observation.logits, (1, 2, 32128)),
                 ('state', observation.state, (1, 8, 512)),
                 ('before', observation.before, (1, 8, 512)),
                 ('after', observation.after, (1, 8, 512)))
        tensor_files = {}
        for name, tensor, shape in specs:
            if tensor is None and name == 'after' and label in ('capture-a', 'capture-b', 'repeat-a'):
                continue
            if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                    or tensor.dtype != torch.float32 or not tensor.is_contiguous()
                    or tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all())):
                raise ValueError('observation tensor differs from its bounded CPU layout')
            raw = tensor.detach().view(torch.int32).numpy().tobytes()
            filename = f'{label}-{name}.bin'
            digest = self._bytes(filename, raw)
            tensor_files[name] = {'file': filename, 'bytes': len(raw), 'shape': list(shape),
                                  'dtype': 'float32-little-endian-bits', 'sha256': digest}
        self.json(label + '.json', {'schema': 'flan-t5-gpu-observation-v1',
            'run_id': self.run_id, 'label': label, 'source_ids': list(observation.source_ids),
            'source_mask': list(observation.source_mask), 'decoder_ids': list(observation.decoder_ids),
            'decoder_mask': list(observation.decoder_mask), 'metadata': observation.metadata,
            'tensors': tensor_files})


def _same_bits(left, right):
    import torch
    return (type(left) is torch.Tensor and type(right) is torch.Tensor
            and left.device.type == right.device.type == 'cpu'
            and left.dtype == right.dtype == torch.float32 and left.shape == right.shape
            and torch.equal(left.view(torch.int32), right.view(torch.int32)))


def _run_sequence(owner, source_a, source_b, target, *, save, begin=lambda label: None):
    """Run six fixed observations; preserve all owner roots on failure."""
    if not callable(save) or not callable(begin):
        raise TypeError('durable observation and start writers are required')
    def mark(label):
        if begin(label) is not None:
            raise RuntimeError('start writer must return None or raise')
    def persist(label, observation):
        if save(label, observation) is not None:
            raise RuntimeError('observation writer must return None or raise')
    mark('capture-a')
    a_handle = owner.observe(source_a, target, action='capture')
    a = a_handle.value
    persist('capture-a', a)
    mark('capture-b')
    b_handle = owner.observe(source_b, target, action='capture')
    b = b_handle.value
    persist('capture-b', b)
    if _same_bits(a.before[0, 6], b.before[0, 6]):
        raise ValueError('the fixed donor vector does not differ')

    mark('repeat-a')
    repeat_handle = owner.observe(source_a, target, action='capture')
    repeat = repeat_handle.value
    persist('repeat-a', repeat)
    if (not _same_bits(repeat.logits, a.logits) or not _same_bits(repeat.state, a.state)
            or not _same_bits(repeat.before, a.before)):
        raise ValueError('repeat capture differs from baseline')
    del repeat
    repeat_handle.drop()
    repeat_handle.retire()

    mark('self-patch')
    self_handle = owner.observe(source_a, target, action='patch', token_index=6, value=a.before[0, 6])
    self_observation = self_handle.value
    persist('self-patch', self_observation)
    if (not _same_bits(self_observation.logits, a.logits)
            or not _same_bits(self_observation.state, a.state)
            or not _same_bits(self_observation.before, a.before)
            or not _same_bits(self_observation.after, a.before)):
        raise ValueError('self patch differs from baseline')
    del self_observation
    self_handle.drop()
    self_handle.retire()

    mark('donor-patch')
    donor_handle = owner.observe(source_a, target, action='patch', token_index=6, value=b.before[0, 6])
    donor = donor_handle.value
    persist('donor-patch', donor)
    if (not _same_bits(donor.before, a.before)
            or not _same_bits(donor.after[0, 6], b.before[0, 6])
            or _same_bits(donor.logits, a.logits)):
        raise ValueError('donor patch did not land or change the saved logits')
    del donor
    donor_handle.drop()
    donor_handle.retire()

    mark('null-a')
    null_handle = owner.observe(source_a, target, action='null', token_index=6)
    null = null_handle.value
    persist('null-a', null)
    if (not _same_bits(null.logits, a.logits) or not _same_bits(null.state, a.state)
            or not _same_bits(null.before, a.before) or not _same_bits(null.after, a.before)):
        raise ValueError('null control differs from baseline')
    del null
    null_handle.drop()
    null_handle.retire()
    del a, b
    a_handle.drop()
    b_handle.drop()
    gc.collect()
    a_handle.retire()
    b_handle.retire()
    return {'forwards': 6, 'repeat_exact': True, 'self_patch_exact': True,
            'null_exact': True, 'donor_differs': True, 'donor_changes_logits': True}


def _attempt_root(path, session, repo):
    path = Path(path)
    if not path.is_absolute() or session.identity is None:
        raise ValueError('an absolute private attempt and device identity are required')
    lexical = Path(os.path.abspath(path))
    for item in (lexical, *lexical.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('attempt path cannot traverse a link or reparse point')
    root = lexical.resolve(strict=True)
    work = (session.identity.workspace_root / 'work').resolve(strict=True)
    checkout = Path(repo).resolve(strict=True)
    if (not root.is_dir() or root == work or not root.is_relative_to(work)
            or root == checkout or root.is_relative_to(checkout) or checkout.is_relative_to(root)):
        raise ValueError('attempt must be a separate directory under workspace work')
    return root


def _staged_asset_accounting(directory):
    """Recheck the staged snapshot and count it separately from record bytes."""
    from .flan_t5_feasibility import verify_assets
    directory = Path(directory)
    manifest = verify_assets(directory)
    files = manifest.get('files') if type(manifest) is dict else None
    if (type(files) is not dict or len(files) != 7
            or any(type(item) is not dict or type(item.get('bytes')) is not int
                   or item['bytes'] <= 0 for item in files.values())):
        raise ValueError('staged asset accounting differs from the fixed manifest')
    payload = sum(item['bytes'] for item in files.values())
    if payload > 512 * 1024 ** 2:
        raise ValueError('staged assets exceed the fixed disk budget')
    with (directory / 'asset-manifest.json').open('rb') as stream:
        raw = stream.read(1024 ** 2 + 1)
    if len(raw) > 1024 ** 2 or hashlib.sha256(raw).hexdigest() != _ASSET_HASH:
        raise ValueError('staged asset manifest differs')
    return {'file_count': len(files), 'payload_bytes': payload,
            'manifest_bytes': len(raw), 'total_bytes': payload + len(raw)}


def owner_main(run_dir, asset_dir, source_commit, git_executable):
    """Run only as the exact child of the existing device supervisor."""
    evidence = None
    try:
        if _CUSTODY['entered'] is not False:
            raise RuntimeError('the FLAN-T5 smoke owner is single-use')
        _CUSTODY['entered'] = True
        session = OwnerSession.from_environment()
        _CUSTODY['session'] = session
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _CUSTODY['lease'] = lease
        OwnerSession.check(session, lease)
        repo = Path(__file__).resolve().parents[1]
        run = _attempt_root(run_dir, session, repo)
        assets = Path(asset_dir)
        if not assets.is_absolute() or not assets.is_dir():
            raise ValueError('pinned asset directory must be absolute')
        assets = assets.resolve(strict=True)
        if assets == run or assets.is_relative_to(run) or run.is_relative_to(assets):
            raise ValueError('asset and attempt directories must be separate')
        directory = run / 'owner'
        directory.mkdir()
        evidence = SmokeEvidence(directory, run_id=session.run_id)
        _CUSTODY['evidence'] = evidence
        evidence.json('owner-started.json', {'schema': 'flan-t5-gpu-smoke-owner-v1',
            'status': 'started', 'run_id': session.run_id, 'pid': os.getpid(),
            'source_commit': source_commit, 'case_manifest_sha256': _CASE_HASH,
            'asset_manifest_sha256': _ASSET_HASH, 'max_forwards': 6})
        owner = FlanT5CudaOwner(session, lease, repo=repo, git_executable=git_executable,
                                source_commit=source_commit, record_stage=evidence.record_stage)
        _CUSTODY['owner'] = owner
        owner.start()
        source_a, source_b, target = _fixed_rows(repo)
        evidence.phase = 'model-load'
        owner.load_model(assets, directory / 'staged-assets')
        model_state_sha256 = owner.witness.sha256
        controls = _run_sequence(owner, source_a, source_b, target,
                                 save=evidence.save, begin=evidence.begin)
        if evidence.forward_count != 6:
            raise RuntimeError('the fixed forward inventory is incomplete')
        evidence.phase = 'cleanup'
        cleanup = owner.finish()
        OwnerSession.check(session, lease)
        staged_assets = _staged_asset_accounting(owner.stage_directory)
        OwnerSession.check(session, lease)
        evidence.json('owner-final.json', {'schema': 'flan-t5-gpu-smoke-owner-v1',
            'status': 'owner-clean-pending-supervisor', 'run_id': session.run_id,
            'pid': os.getpid(), 'source_commit': source_commit, 'source_sha256': owner.source_hashes,
            'case_manifest_sha256': _CASE_HASH, 'asset_manifest_sha256': _ASSET_HASH,
            'revision': _REVISION, 'model_state_sha256': model_state_sha256,
            'runtime_versions': owner.runtime_versions, 'dependency_entry_files': owner.dependency_origins,
            'device_uuid': session.identity.gpu_uuid,
            'device_name': owner.device_name, 'device_total_bytes': owner.device_total,
            'allocator_cap_bytes': owner.allocator_cap, 'cleanup': cleanup,
            'memory_record_count': evidence.memory_count, 'forward_count': evidence.forward_count,
            'controls': controls, 'record_bytes_before_final': evidence.used,
            'staged_assets': staged_assets})
        OwnerSession.check(session, lease)
        OwnerSession.attest_clean(session, lease)
        return 0, None
    except BaseException as error:
        _CUSTODY['failure'] = error
        reported_error = error
        if evidence is not None:
            try:
                evidence.json('owner-failure.json', {'schema': 'flan-t5-gpu-smoke-owner-v1',
                    'status': 'quarantined', 'run_id': evidence.run_id,
                    'phase': evidence.phase, 'forward_starts': evidence.forward_count,
                    'memory_records': evidence.memory_count, 'error_type': type(error).__name__})
            except BaseException as diagnostic_error:
                _CUSTODY['diagnostic_failure'] = diagnostic_error
                if (isinstance(diagnostic_error, (KeyboardInterrupt, SystemExit))
                        and not isinstance(error, (KeyboardInterrupt, SystemExit))):
                    reported_error = diagnostic_error
        return (130 if isinstance(reported_error, KeyboardInterrupt) else 1), type(reported_error).__name__


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--asset-dir', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--git-executable', required=True, type=Path)
    args = parser.parse_args(argv)
    code, error_type = owner_main(args.run_dir, args.asset_dir, args.source_commit, args.git_executable)
    print(json.dumps({'status': 'owner-clean-pending-supervisor' if code == 0 else 'incomplete',
                      'error_type': error_type, 'model_dependencies_preloaded': any(
                          name in sys.modules for name in ('torch', 'transformers', 'numpy'))}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
