"""Fabricated CPU-only tests for the v4 owner and structural receipt."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from looking_glass.vision_coco_recognized_handoff import MODEL_STATE_SHA256
from looking_glass.vision_coco_recognized_screen import aggregate_recognized
from looking_glass.vision_coco_same_site_study import run_same_site
from looking_glass.vision_coco_study import Evidence

OWNER = Path(__file__).resolve().parents[1] / 'tools/run_vision_coco_recognized.py'


def runtime():
    return {'device': 'cpu', 'cuda_visible_devices': '', 'cuda_initialized': False,
            'python': '3.11.15', 'torch': '2.8.0+cu128', 'transformers': '4.57.3',
            'numpy': '2.4.6', 'pillow': '11.3.0', 'threads': 1, 'interop_threads': 1,
            'deterministic_algorithms': True, 'float32_matmul_precision': 'highest',
            'mkldnn_enabled': True, 'default_dtype': 'torch.float32', 'processor': 'ViTImageProcessor',
            'denormals_preserved': True, 'mkl_cbwr': 'COMPATIBLE', 'binary64_rounding': 'nearest-even'}


def test_preflight_receipt_rejects_missing_runtime_and_boolean_counts(tmp_path):
    owner = runpy.run_path(str(OWNER))
    source = {'revision': 'f' * 40, 'sources': {}}
    started = {'schema': 'vision-recognized-v4-start', 'mode': 'preflight', 'source': source,
               'manifest_sha256': owner['MANIFEST_SHA'], 'model_state_sha256': MODEL_STATE_SHA256,
               'runtime': runtime()}
    terminal = {'schema': 'vision-recognized-v4-terminal', 'status': 'complete', 'mode': 'preflight',
                'source_revision': source['revision'], 'manifest_sha256': owner['MANIFEST_SHA'],
                'model_state_sha256_before_after': MODEL_STATE_SHA256, 'runtime': runtime(),
                'submissions': 0, 'saved_forwards': 0}
    (tmp_path / 'started.json').write_text(json.dumps(started))
    (tmp_path / 'terminal.json').write_text(json.dumps(terminal))
    owner['verify_terminal'](tmp_path, terminal, source, 'preflight', {}, None)
    terminal['submissions'] = False
    with pytest.raises(ValueError, match='preflight'):
        owner['verify_terminal'](tmp_path, terminal, source, 'preflight', {}, None)
    terminal['submissions'] = 0
    del started['runtime']['mkldnn_enabled']
    del terminal['runtime']['mkldnn_enabled']
    (tmp_path / 'started.json').write_text(json.dumps(started))
    with pytest.raises(ValueError, match='runtime'):
        owner['verify_terminal'](tmp_path, terminal, source, 'preflight', {}, None)


def test_malformed_terminal_still_preserves_direct_child_receipt(tmp_path, monkeypatch):
    from looking_glass import cpu_study_supervision
    owner = runpy.run_path(str(OWNER))
    controller = owner['controller']
    controller.__globals__['source_snapshot'] = lambda revision: {'revision': revision, 'sources': {}}
    @dataclasses.dataclass
    class FakeResult:
        exit_code: int = 0
        clean: bool = True
        timed_out: bool = False
        pid: int = 123
    out, custody, assets, pillow = (tmp_path / name for name in ('out', 'custody', 'assets', 'pillow'))
    for directory in (custody, assets, pillow):
        directory.mkdir()
    def child(*args, **kwargs):
        data = out / 'data'
        data.mkdir()
        (data / 'terminal.json').write_text('{')
        return FakeResult()
    monkeypatch.setattr(cpu_study_supervision, 'run_cpu_study', child)
    args = SimpleNamespace(output=out, custody=custody, assets=assets, pillow=pillow, site_packages=pillow,
                           mode='screen', screen_data=None, revision='f' * 40, timeout=60)
    assert controller(args) == 1
    receipt = json.loads((out / 'supervisor.json').read_bytes())
    assert receipt['status'] == 'incomplete' and receipt['exit_code'] == 0
    assert receipt['failure_detail'].endswith('JSONDecodeError')


def test_supervisor_exception_retains_incomplete_receipt(tmp_path, monkeypatch):
    from looking_glass import cpu_study_supervision
    owner = runpy.run_path(str(OWNER))
    controller = owner['controller']
    controller.__globals__['source_snapshot'] = lambda revision: {'revision': revision, 'sources': {}}
    out, custody, assets, pillow = (tmp_path / name for name in ('out', 'custody', 'assets', 'pillow'))
    for directory in (custody, assets, pillow):
        directory.mkdir()
    def fail(*args, **kwargs):
        raise OSError('fabricated')
    monkeypatch.setattr(cpu_study_supervision, 'run_cpu_study', fail)
    args = SimpleNamespace(output=out, custody=custody, assets=assets, pillow=pillow, site_packages=pillow,
                           mode='screen', screen_data=None, revision='f' * 40, timeout=60)
    with pytest.raises(OSError, match='fabricated'):
        controller(args)
    receipt = json.loads((out / 'supervisor.json').read_bytes())
    assert receipt['status'] == 'incomplete' and receipt['error_type'] == 'OSError'
    claim = custody / 'vision-recognized-v4-screen.claim.json'
    assert claim.is_file()
    args.output = tmp_path / 'second-out'
    with pytest.raises(FileExistsError):
        controller(args)
    receipt = json.loads((args.output / 'supervisor.json').read_bytes())
    assert receipt['status'] == 'incomplete' and receipt['error_type'] == 'FileExistsError'


def test_post_claim_missing_systemroot_retains_incomplete_receipt(tmp_path, monkeypatch):
    if os.name != 'nt':
        pytest.skip('Windows startup environment')
    owner = runpy.run_path(str(OWNER))
    controller = owner['controller']
    controller.__globals__['source_snapshot'] = lambda revision: {'revision': revision, 'sources': {}}
    out, custody, assets, pillow = (tmp_path / name for name in ('out', 'custody', 'assets', 'pillow'))
    for directory in (custody, assets, pillow):
        directory.mkdir()
    monkeypatch.delenv('SystemRoot', raising=False)
    args = SimpleNamespace(output=out, custody=custody, assets=assets, pillow=pillow, site_packages=pillow,
                           mode='screen', screen_data=None, revision='f' * 40, timeout=60)
    with pytest.raises(ValueError, match='SystemRoot'):
        controller(args)
    receipt = json.loads((out / 'supervisor.json').read_bytes())
    assert receipt['status'] == 'incomplete' and receipt['error_type'] == 'ValueError'
    assert (custody / 'vision-recognized-v4-screen.claim.json').is_file()


def test_parent_rejects_ambient_import_shadow_before_import(tmp_path):
    shadow = tmp_path / 'shadow'
    shadow.mkdir()
    marker = tmp_path / 'executed'
    (shadow / 'json.py').write_text(f'from pathlib import Path; Path({str(marker)!r}).touch()\n')
    env = dict(os.environ, PYTHONPATH=str(shadow))
    positive = subprocess.run([sys.executable, '-B', '-c', 'import json'], env=env, capture_output=True)
    assert positive.returncode == 0 and marker.exists()
    marker.unlink()
    result = subprocess.run([sys.executable, '-B', str(OWNER), '--help'], env=env, capture_output=True)
    assert result.returncode != 0 and not marker.exists()
    assert b'requires isolated' in result.stderr
    cache = tmp_path / 'clean-cache'
    cache.mkdir()
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-X', f'pycache_prefix={cache}',
                             str(OWNER), '--help'], env=env, capture_output=True)
    assert result.returncode == 0 and not marker.exists()


def test_parent_bound_screen_handoff_is_rechecked_without_git(tmp_path):
    owner = runpy.run_path(str(OWNER))
    recheck = owner['recheck']
    recheck.__globals__['ROOT'] = tmp_path
    recheck.__globals__['SOURCES'] = ()
    names = [owner['PUBLIC_SCREEN'], owner['PUBLIC_SCREEN_ARCHIVE'], owner['PUBLIC_SCREEN_RECEIPT'],
             owner['PUBLIC_SCREEN_OBSERVER']]
    handoff = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fabricated')
        handoff[name] = hashlib.sha256(b'fabricated').hexdigest()
    recheck({'sources': {}, 'public_handoff': handoff})
    (tmp_path / names[-1]).write_bytes(b'changed')
    with pytest.raises(ValueError, match='handoff changed'):
        recheck({'sources': {}, 'public_handoff': handoff})


def test_custody_inside_assets_is_rejected_before_source_or_lease(tmp_path):
    owner = runpy.run_path(str(OWNER))
    assets = tmp_path / 'assets'
    custody = assets / 'custody'
    custody.mkdir(parents=True)
    pillow = tmp_path / 'pillow'
    pillow.mkdir()
    args = SimpleNamespace(output=tmp_path / 'out', custody=custody, assets=assets, pillow=pillow, site_packages=pillow,
                           mode='screen', screen_data=None, revision='f' * 40, timeout=60)
    with pytest.raises(ValueError, match='separate'):
        owner['controller'](args)
    assert not (custody / 'cpu-study.lock').exists()
    assert not args.output.exists()


def test_fabricated_intervention_schedule_and_missing_witness(tmp_path):
    owner = runpy.run_path(str(OWNER))
    class Layer(torch.nn.Module):
        def forward(self, value):
            return value + .01 * value.mean(dim=1, keepdim=True)
    class FakeVit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = torch.nn.Module()
            self.vit.encoder = torch.nn.Module()
            self.vit.encoder.layer = torch.nn.ModuleList([Layer() for _ in range(12)])
        def forward(self, pixel_values):
            value = pixel_values
            for layer in self.vit.encoder.layer:
                value = layer(value)
            scalar = value[:, 0, :1]
            logits = torch.zeros((1, 1000), dtype=torch.float32)
            logits[:, 151:269], logits[:, 281:286] = scalar, -scalar
            return SimpleNamespace(logits=logits)
    assert torch.set_flush_denormal(False)
    pixels = {1: torch.full((1, 197, 192), -5.), 2: torch.full((1, 197, 192), 5.),
              3: torch.full((1, 197, 192), -4.), 4: torch.full((1, 197, 192), 4.)}
    pair = {'pair_id': 'pair-00', 'split': 'heldout', 'box_tokens': [1, 2, 15, 16],
            'cat': {'image_id': 1}, 'dog': {'image_id': 2},
            'cat_donor': {'image_id': 3}, 'dog_donor': {'image_id': 4}}
    evidence = Evidence(tmp_path / 'data', 64)
    evidence.json('started.json', {'fabricated': True})
    seen = []
    rows = run_same_site(FakeVit().eval(), [pair], pixels, evidence,
                         check_clean=lambda image_id, logits: seen.append(image_id))
    assert seen == [1, 2, 3, 4]
    terminal = {'pairs': rows, 'aggregate': aggregate_recognized(rows),
                'submissions': evidence.submissions, 'saved_forwards': len(evidence.records)}
    evidence.json('terminal.json', terminal)
    owner['verify_intervention_schedule'](evidence.root, terminal, [pair])
    action = evidence.root / 'forward-006.json'
    row = json.loads(action.read_bytes())
    # Forward six is null; removing a token witness must fail closed.
    row['layout'].pop('before_0')
    action.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='token-level'):
        owner['verify_intervention_schedule'](evidence.root, terminal, [pair])
