"""Fabricated discovery-owner ordering checks, without model calls."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_all_discovery_initials_are_verified_before_any_patch(monkeypatch):
    import tools.run_vision_coco_cls_occlusion_discovery as owner

    images = [{'image_id': value} for value in (1, 2, 3, 4)]
    prior = {'inputs': {row['image_id']: {} for row in images},
             'logits': {row['image_id']: {} for row in images}}
    events = []
    def capture(_model, rows, _pixels, inputs, logits, _evidence, **_kwargs):
        events.append(('initials', [row['image_id'] for row in rows]))
        assert inputs == prior['inputs'] and logits == prior['logits']
        return {row['image_id']: 'captured' for row in rows}
    def case(_model, image, value, _evidence, **_kwargs):
        events.append(('case', image['image_id']))
        assert value == 'captured'
        return {'image_id': image['image_id']}
    monkeypatch.setattr(owner, 'capture_initials', capture)
    monkeypatch.setattr(owner, 'run_case', case)
    assert len(owner.run_discovery(None, images, {}, prior, None)) == 4
    assert events == [('initials', [1, 2, 3, 4]), ('case', 1), ('case', 2), ('case', 3), ('case', 4)]

    def reject(*_args, **_kwargs):
        raise ValueError('last initial differs')
    events.clear()
    monkeypatch.setattr(owner, 'capture_initials', reject)
    with pytest.raises(ValueError, match='last initial'):
        owner.run_discovery(None, images, {}, prior, None)
    assert not events


def test_observer_requires_complete_fixed_case_schedule(tmp_path):
    from tools.observe_vision_coco_cls_occlusion_discovery import IDS, MANIFEST_SHA, terminal_complete

    revision = 'a' * 40
    started = {'schema': 'vision-cls-occlusion-v1-start', 'mode': 'preflight',
               'source': {'revision': revision}, 'manifest_sha256': MANIFEST_SHA, 'selected_ids': IDS}
    terminal = {'schema': 'vision-cls-occlusion-v1-terminal', 'status': 'complete', 'mode': 'preflight',
                'source_revision': revision, 'manifest_sha256': MANIFEST_SHA, 'submissions': 0, 'saved_forwards': 0}
    (tmp_path / 'started.json').write_text(json.dumps(started))
    path = tmp_path / 'terminal.json'
    path.write_text(json.dumps(terminal))
    assert terminal_complete(path, 'preflight', revision)
    started['mode'] = terminal['mode'] = 'discovery'
    terminal.update(selected=4, cases=[{'image_id': ident, 'dose': {'status': 'matchable'}} for ident in IDS],
                    submissions=92, saved_forwards=92)
    (tmp_path / 'started.json').write_text(json.dumps(started))
    path.write_text(json.dumps(terminal))
    assert terminal_complete(path, 'discovery', revision)
    terminal['submissions'] = terminal['saved_forwards'] = 91
    path.write_text(json.dumps(terminal))
    assert not terminal_complete(path, 'discovery', revision)
    terminal['submissions'] = terminal['saved_forwards'] = 92
    terminal['cases'][-1]['image_id'] = 999
    path.write_text(json.dumps(terminal))
    assert not terminal_complete(path, 'discovery', revision)


def test_isolated_observer_rejects_preexisting_bytecode_prefix(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'stale.pyc').write_bytes(b'preexisting bytecode')
    script = Path(__file__).resolve().parents[1] / 'tools/observe_vision_coco_cls_occlusion_discovery.py'
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={cache}',
                             str(script), '--help'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'empty bytecode prefix' in result.stderr
