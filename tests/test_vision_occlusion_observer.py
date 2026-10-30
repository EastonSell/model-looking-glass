"""The observer records direct-process outcomes and prevents a phase retry."""

import json
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def test_preflight_does_not_consume_a_one_shot_phase_claim(tmp_path):
    from tools.observe_vision_coco_occlusion import claim_phase

    assert claim_phase(tmp_path, 'preflight', 'a' * 40) is None
    claim = claim_phase(tmp_path, 'screen', 'a' * 40)
    assert json.loads(claim.read_bytes()) == {'mode': 'screen', 'revision': 'a' * 40}
    with pytest.raises(FileExistsError):
        claim_phase(tmp_path, 'screen', 'a' * 40)


def test_model_and_hub_caches_are_placed_with_phase_temporary_files(tmp_path):
    from tools.observe_vision_coco_occlusion import cache_environment, child_environment

    env = cache_environment(tmp_path)
    assert env['TEMP'] == env['TMP'] == env['TMPDIR'] == str(tmp_path)
    for name in ('HF_HOME', 'HF_HUB_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME',
                 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'HF_ASSETS_CACHE',
                 'HF_MODULES_CACHE', 'HF_DATASETS_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TORCH_EXTENSIONS_DIR'):
        assert Path(env[name]).is_relative_to(tmp_path)
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
    assert 'HOME' not in env and 'USERPROFILE' not in env
    child = child_environment({'TRANSFORMERS_CACHE': 'C:/legacy',
                               'PYTORCH_TRANSFORMERS_CACHE': 'C:/legacy'}, tmp_path)
    assert 'TRANSFORMERS_CACHE' not in child and 'PYTORCH_TRANSFORMERS_CACHE' not in child


@pytest.mark.skipif(sys.platform != "win32", reason="requires distinct Windows drive semantics")
def test_owner_rejects_caches_or_bytecode_on_source_volume(tmp_path):
    from tools.observe_vision_coco_occlusion import cache_environment
    from tools.run_vision_coco_occlusion import validate_cache_placement

    temporary = tmp_path / 'temporary'
    temporary.mkdir()
    prefix = tmp_path / 'bytecode'
    prefix.mkdir()
    source = Path(('Z:' if temporary.drive.upper() != 'Z:' else 'Y:') + '/source')
    env = cache_environment(temporary)
    validate_cache_placement(tmp_path / 'run', prefix, source, env)
    with pytest.raises(ValueError, match='cache volume'):
        validate_cache_placement(tmp_path / 'run', prefix, source, {**env, 'TORCH_HOME': str(source / 'torch')})
    with pytest.raises(ValueError, match='cache volume'):
        validate_cache_placement(tmp_path / 'run', source / 'bytecode', source, env)
    with pytest.raises(ValueError, match='cache volume'):
        validate_cache_placement(tmp_path / 'run', prefix, source, {**env, 'TMPDIR': str(source)})
    with pytest.raises(ValueError, match='cache volume'):
        validate_cache_placement(tmp_path / 'run', prefix, source, {**env, 'TRANSFORMERS_CACHE': str(source)})



def test_owner_rejects_cache_placement_on_same_drive(tmp_path):
    from tools.observe_vision_coco_occlusion import cache_environment
    from tools.run_vision_coco_occlusion import validate_cache_placement

    temporary = tmp_path / "temporary"
    temporary.mkdir()
    bytecode = tmp_path / "bytecode"
    bytecode.mkdir()
    # The guard deliberately requires distinct Windows volumes. On POSIX,
    # all native Path.drive values are empty and must not accidentally pass.
    with pytest.raises(ValueError, match="cache volume"):
        validate_cache_placement(tmp_path / "run", bytecode, tmp_path / "source",
                                 cache_environment(temporary))


def test_supervisor_waits_for_real_direct_process_and_reports_failure(tmp_path):
    from tools.observe_vision_coco_occlusion import supervise

    with (tmp_path / 'out').open('wb') as stdout, (tmp_path / 'err').open('wb') as stderr:
        result = supervise([sys.executable, '-c', 'raise SystemExit(3)'], cwd=tmp_path,
                           env=None, stdout=stdout, stderr=stderr, timeout=10)
    assert result[:5] == (3, False, False, True, None)


def test_timeout_and_cancellation_preserve_direct_process_cleanup(monkeypatch):
    from tools import observe_vision_coco_occlusion as observer

    class Child:
        def __init__(self, *args, **kwargs):
            self.waits = 0
            self.killed = False

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired('fake', timeout)
            return 9

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    monkeypatch.setattr(observer.subprocess, 'Popen', Child)
    result = observer.supervise(['fake'], cwd='.', env={}, stdout=None, stderr=None, timeout=1)
    assert result[:5] == (9, True, True, True, 'TimeoutExpired')

    class Interrupted(Child):
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt()
            return 9

    monkeypatch.setattr(observer.subprocess, 'Popen', Interrupted)
    result = observer.supervise(['fake'], cwd='.', env={}, stdout=None, stderr=None, timeout=1)
    assert result[:5] == (9, False, True, True, 'KeyboardInterrupt')
    assert isinstance(result[5], KeyboardInterrupt)


def test_terminal_identity_and_exact_count_types_are_required(tmp_path):
    from tools.observe_vision_coco_occlusion import terminal_complete

    terminal = {'schema': 'vision-occlusion-v1-terminal', 'status': 'complete', 'mode': 'preflight',
                'source_revision': 'a' * 40, 'manifest_sha256': 'b' * 64,
                'submissions': 0, 'saved_forwards': 0}
    path = tmp_path / 'terminal.json'
    (tmp_path / 'started.json').write_text(json.dumps({
        'schema': 'vision-occlusion-v1-start', 'mode': 'preflight',
        'manifest_sha256': 'b' * 64, 'source': {'revision': 'a' * 40}}), encoding='utf-8')
    path.write_text(json.dumps(terminal), encoding='utf-8')
    assert terminal_complete(path, 'preflight', 'a' * 40, 'b' * 64)
    for changed in ({'source_revision': 'c' * 40}, {'manifest_sha256': 'c' * 64},
                    {'submissions': False}, {'saved_forwards': 1}, {'status': 'incomplete'}):
        path.write_text(json.dumps({**terminal, **changed}), encoding='utf-8')
        assert not terminal_complete(path, 'preflight', 'a' * 40, 'b' * 64)
    path.write_text('{', encoding='utf-8')
    assert not terminal_complete(path, 'preflight', 'a' * 40, 'b' * 64)


def test_published_discovery_handoff_binds_all_archive_parts_terminal_and_receipt():
    from tools.run_vision_coco_occlusion import verify_public_discovery

    ids = [328437, 142719, 476736, 503893]
    source = {'revision': 'a' * 40, 'sources': {}}
    cases = [{'image_id': item, 'dose': {'status': 'matchable'},
              'input_sha256': {name: hashlib.sha256(f'{item}-{name}'.encode()).hexdigest()
                               for name in ('clean', 'animal', 'background')}} for item in ids]
    start = {'schema': 'vision-occlusion-v1-start', 'mode': 'discovery', 'manifest_sha256': 'b' * 64,
             'source': source, 'runtime': {'fake': True}, 'model_state_sha256': 'c' * 64,
             'selected_ids': ids}
    terminal = {'schema': 'vision-occlusion-v1-terminal', 'mode': 'discovery', 'status': 'complete',
                'manifest_sha256': 'b' * 64, 'source_revision': source['revision'], 'runtime': {'fake': True},
                'model_state_sha256_before_after': 'c' * 64, 'cases': cases, 'selected': 4,
                'submissions': 72, 'saved_forwards': 72}
    encode = lambda value: (json.dumps(value, sort_keys=True) + '\n').encode()
    files = {'started.json': encode(start), 'terminal.json': encode(terminal)}
    for case in cases:
        ident = case['image_id']
        files[f'case-{ident}.json'] = encode(case)
        for name in ('clean', 'animal', 'background'):
            files[f'input-{ident}-{name}.bin'] = f'{ident}-{name}'.encode()
    for i in range(1, 73):
        raw = f'fake-forward-{i}'.encode()
        files[f'forward-{i:03d}.bin'] = raw
        files[f'forward-{i:03d}.json'] = encode({'submission': i, 'sha256': hashlib.sha256(raw).hexdigest()})
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    data = stream.getvalue()
    parts = [data[:len(data) // 2], data[len(data) // 2:]]
    index = {'schema': 'vision-occlusion-v1-discovery-archive-index',
             'archive_sha256': hashlib.sha256(data).hexdigest(), 'archive_bytes': len(data),
             'parts': [{'file': f'discovery-evidence.part-{i:03d}', 'bytes': len(raw),
                        'sha256': hashlib.sha256(raw).hexdigest()} for i, raw in enumerate(parts)],
             'members': [{'file': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                         for name, raw in sorted(files.items())]}
    observer = {'schema': 'vision-occlusion-v1-direct-observer', 'mode': 'discovery',
                'source_revision': source['revision'], 'exit_code': 0, 'timed_out': False,
                'reaped': True, 'forced_stop': False, 'status': 'complete',
                'terminal_sha256': hashlib.sha256(files['terminal.json']).hexdigest()}
    index_raw, observer_raw = encode(index), encode(observer)
    readback = {'schema': 'vision-occlusion-v1-discovery-public-readback', 'status': 'pass',
                'public_commit': 'd' * 40, 'archive_sha256': index['archive_sha256'],
                'index_sha256': hashlib.sha256(index_raw).hexdigest(),
                'parts': [{**part, 'git_blob': hashlib.sha1(b'blob ' + str(len(raw)).encode() + bytes([0]) + raw).hexdigest()}
                          for part, raw in zip(index['parts'], parts)]}
    readback_raw = encode(readback)
    freeze = {'schema': 'vision-occlusion-v1-discovery-freeze', 'status': 'complete',
              'manifest_sha256': 'b' * 64, 'selected_ids': ids,
              'terminal_sha256': observer['terminal_sha256'], 'archive_sha256': index['archive_sha256'],
              'index_sha256': hashlib.sha256(index_raw).hexdigest(),
              'observer_sha256': hashlib.sha256(observer_raw).hexdigest()}
    freeze_raw = encode(freeze)
    assert verify_public_discovery(freeze_raw, index_raw, observer_raw, readback_raw, data,
                                   manifest_sha='b' * 64, model_state='c' * 64) == terminal
    with pytest.raises(ValueError, match='archive'):
        verify_public_discovery(freeze_raw, index_raw, observer_raw, readback_raw, data[:-1] + b'X',
                                manifest_sha='b' * 64, model_state='c' * 64)
    broken = {**observer, 'reaped': False}
    with pytest.raises(ValueError, match='observer'):
        verify_public_discovery(freeze_raw, index_raw, encode(broken), readback_raw, data,
                                manifest_sha='b' * 64, model_state='c' * 64)
    with pytest.raises(ValueError, match='readback'):
        verify_public_discovery(freeze_raw, index_raw, observer_raw,
                                encode({**readback, 'parts': []}), data,
                                manifest_sha='b' * 64, model_state='c' * 64)


def test_bounded_archive_read_rejects_before_accepting_oversized_input(tmp_path):
    from tools.run_vision_coco_occlusion import read_bounded

    path = tmp_path / 'archive'
    path.write_bytes(b'12345')
    assert read_bounded(path, 5) == b'12345'
    with pytest.raises(ValueError, match='bounded'):
        read_bounded(path, 4)
