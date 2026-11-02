"""Fabricated checks for global baseline ordering and dose denominators."""

import pytest
import json

import looking_glass.vision_coco_cls_occlusion_evaluation as module


def fixture():
    selected = [{'image_id': i, 'category_id': 17 if i < 2 else 18} for i in range(4)]
    screen = {'selected': selected, 'inputs': {i: {'value': i} for i in range(4)},
              'logits': {i: {'value': i} for i in range(4)}}
    rows = [{'image_id': i, 'category_id': selected[i]['category_id'], 'occlusion_contrast': .1}
            for i in range(4)]
    return screen, rows, {i: i for i in range(4)}


def test_all_baselines_precede_any_case_and_limited_is_not_failure(monkeypatch):
    screen, rows, pixels = fixture()
    events = []
    statuses = ['matchable', 'limited-dose', 'dose-ineligible', 'matchable']
    def capture(_model, images, *_args, **_kwargs):
        events.append(('all_initials', [image['image_id'] for image in images]))
        return {image['image_id']: image['image_id'] for image in images}
    def case(_model, image, initial, _evidence, **_kwargs):
        i = image['image_id']
        events.append(('case', i))
        return {'image_id': i, 'category_id': image['category_id'], 'dose': {'status': statuses[i]},
                'comparisons': {'positive': i in (0, 1)}}
    monkeypatch.setattr(module, 'capture_initials', capture)
    monkeypatch.setattr(module, 'baseline_contrast', lambda image, initial: .1)
    monkeypatch.setattr(module, 'run_case', case)
    result = module.run_evaluation(None, screen, rows, pixels, None)
    assert events == [('all_initials', [0, 1, 2, 3])] + [('case', i) for i in range(4)]
    assert result['aggregate'] == {'status': 'inconclusive', 'evaluable': 2, 'positive': 1, 'cats': 1, 'dogs': 1}
    assert result['by_species']['cat']['limited-dose'] == 1
    assert result['by_species']['cat']['evaluable'] == 1
    assert result['by_species']['dog']['dose-ineligible'] == 1


def test_last_initial_or_baseline_mismatch_stops_before_any_patch(monkeypatch):
    screen, rows, pixels = fixture()
    events = []
    monkeypatch.setattr(module, 'run_case', lambda *_a, **_k: events.append('patch'))
    def reject_last(*_args, **_kwargs):
        events.append('initials')
        raise ValueError('last masked input differs')
    monkeypatch.setattr(module, 'capture_initials', reject_last)
    with pytest.raises(ValueError, match='last masked'):
        module.run_evaluation(None, screen, rows, pixels, None)
    assert events == ['initials']
    events.clear()
    monkeypatch.setattr(module, 'capture_initials', lambda *_a, **_k: {i: i for i in range(4)})
    monkeypatch.setattr(module, 'baseline_contrast', lambda image, initial: .2 if initial == 3 else .1)
    with pytest.raises(ValueError, match='baseline differs'):
        module.run_evaluation(None, screen, rows, pixels, None)
    assert events == []


def test_duplicate_or_missing_screen_rows_rejected_before_capture(monkeypatch):
    screen, rows, pixels = fixture()
    events = []
    monkeypatch.setattr(module, 'capture_initials', lambda *_a, **_k: events.append('initials'))
    with pytest.raises(ValueError, match='screen-row inventory'):
        module.run_evaluation(None, screen, rows + [rows[0]], pixels, None)
    with pytest.raises(ValueError, match='screen-row inventory'):
        module.run_evaluation(None, screen, rows[:-1], pixels, None)
    assert events == []


def test_evaluation_observer_rejects_wrong_count_or_case_order(tmp_path):
    from tools.observe_vision_coco_cls_occlusion_evaluation import MANIFEST_SHA, terminal_complete

    revision = 'a' * 40
    ids = [11, 22]
    started = {'schema': 'vision-cls-occlusion-v1-evaluation-start', 'mode': 'evaluation',
               'manifest_sha256': MANIFEST_SHA, 'source': {'revision': revision}, 'selected_ids': ids}
    terminal = {'schema': 'vision-cls-occlusion-v1-evaluation-terminal', 'status': 'complete',
                'mode': 'evaluation', 'manifest_sha256': MANIFEST_SHA, 'source_revision': revision,
                'selected_ids': ids, 'selected': 2, 'aggregate': {'status': 'inconclusive'},
                'submissions': 39, 'saved_forwards': 39,
                'cases': [{'image_id': 11, 'dose': {'status': 'matchable'}},
                          {'image_id': 22, 'dose': {'status': 'dose-ineligible'}}]}
    (tmp_path / 'started.json').write_text(json.dumps(started))
    path = tmp_path / 'terminal.json'
    path.write_text(json.dumps(terminal))
    assert terminal_complete(path, 'evaluation', revision)
    terminal['submissions'] = terminal['saved_forwards'] = 38
    path.write_text(json.dumps(terminal))
    assert not terminal_complete(path, 'evaluation', revision)
    terminal['submissions'] = terminal['saved_forwards'] = 39
    terminal['cases'].reverse()
    path.write_text(json.dumps(terminal))
    assert not terminal_complete(path, 'evaluation', revision)


def test_evaluation_observer_rejects_nonempty_bytecode_prefix(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'stale.pyc').write_bytes(b'preexisting bytecode')
    script = Path(__file__).resolve().parents[1] / 'tools/observe_vision_coco_cls_occlusion_evaluation.py'
    result = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={cache}',
                             str(script), '--help'], capture_output=True, text=True)
    assert result.returncode != 0 and 'empty bytecode prefix' in result.stderr
