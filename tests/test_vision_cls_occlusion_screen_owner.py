"""Fabricated fresh-screen phase sequencing; no model is loaded."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def test_fresh_screen_keeps_all_clean_rows_before_mask_selection(tmp_path, monkeypatch):
    import tools.run_vision_coco_cls_occlusion_screen as owner

    images = [{'image_id': i, 'category_id': 17 if i < 32 else 18, 'rank': i % 32}
              for i in range(64)]
    observed = [{'image_id': i, 'logit_sha256': hashlib.sha256(bytes([i]) * 4000).hexdigest()}
                for i in range(64)]
    for i in range(64):
        (tmp_path / f'forward-{i + 1:03d}.bin').write_bytes(bytes([i]) * 4000)
    events, saved = [], {}
    def clean(_model, rows, _pixels, _evidence, **_kwargs):
        events.append(('clean', len(rows)))
        return observed, [], {}
    def masks(_model, rows, baseline, _pixels, clean_logits, _evidence, **_kwargs):
        events.append(('masks', len(rows)))
        assert baseline == observed and clean_logits[63] == bytes([63]) * 4000
        return {0: {'animal_drop': .2, 'background_drop': .1}}
    def select(rows, baseline, masked):
        events.append(('select', len(rows)))
        assert baseline == observed and set(masked) == {0}
        return [rows[0]], {'cat': {'selected': 1}}, [{'image_id': 0}]
    def save(name, value):
        saved[name] = value
        return 'result-hash'
    monkeypatch.setattr(owner, 'run_clean_screen', clean)
    monkeypatch.setattr(owner, 'run_response_masks', masks)
    monkeypatch.setattr(owner, 'select_response_screen', select)
    evidence = SimpleNamespace(root=tmp_path, json=save)
    result = owner.run_fresh_screen(None, images, {}, evidence, 'a' * 64)
    assert events == [('clean', 64), ('masks', 64), ('select', 64)]
    assert result['sha256'] == 'result-hash'
    assert saved['screen-result.json']['selected_ids'] == [0]
    assert saved['screen-result.json']['masked'] == {'0': {'animal_drop': .2, 'background_drop': .1}}


def test_screen_observer_checks_forward_count_and_empty_prefix(tmp_path):
    from tools.observe_vision_coco_cls_occlusion_screen import MANIFEST_SHA, terminal_complete

    revision = 'a' * 40
    started = {'schema': 'vision-cls-occlusion-v1-screen-start', 'mode': 'screen',
               'source': {'revision': revision}, 'manifest_sha256': MANIFEST_SHA, 'image_ids': list(range(64))}
    terminal = {'schema': 'vision-cls-occlusion-v1-screen-terminal', 'status': 'complete', 'mode': 'screen',
                'source_revision': revision, 'manifest_sha256': MANIFEST_SHA, 'submissions': 70, 'saved_forwards': 70,
                'counts': {'cat': {'recognized_geometry': 2}, 'dog': {'recognized_geometry': 1}},
                'selected_ids': [1, 2], 'decision': 'pending', 'screen_result_sha256': 'c' * 64}
    (tmp_path / 'started.json').write_text(json.dumps(started))
    path = tmp_path / 'terminal.json'
    path.write_text(json.dumps(terminal))
    assert terminal_complete(path, 'screen', revision)
    terminal['submissions'] = terminal['saved_forwards'] = 69
    path.write_text(json.dumps(terminal))
    assert not terminal_complete(path, 'screen', revision)
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'stale.pyc').write_bytes(b'preexisting bytecode')
    script = Path(__file__).resolve().parents[1] / 'tools/observe_vision_coco_cls_occlusion_screen.py'
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={cache}',
                             str(script), '--help'], capture_output=True, text=True)
    assert result.returncode != 0 and 'empty bytecode prefix' in result.stderr
