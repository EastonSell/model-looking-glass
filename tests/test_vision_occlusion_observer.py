"""The observer records direct-process outcomes and prevents a phase retry."""

import json
import subprocess
import sys

import pytest


def test_preflight_does_not_consume_a_one_shot_phase_claim(tmp_path):
    from tools.observe_vision_coco_occlusion import claim_phase

    assert claim_phase(tmp_path, 'preflight', 'a' * 40) is None
    claim = claim_phase(tmp_path, 'screen', 'a' * 40)
    assert json.loads(claim.read_bytes()) == {'mode': 'screen', 'revision': 'a' * 40}
    with pytest.raises(FileExistsError):
        claim_phase(tmp_path, 'screen', 'a' * 40)


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
