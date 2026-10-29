import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('launch_v2', Path(__file__).with_name('year_label_v2_direct_launch.py'))
launch_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch_v2)


def _setup(tmp_path, monkeypatch):
    attempt = tmp_path/'attempt'
    prefix = attempt/'private-prefix'
    source, prior, cache = tmp_path/'source', tmp_path/'prior', tmp_path/'cache'
    for path in (prefix, source, prior, cache):
        path.mkdir(parents=True)
    prefix_sha = hashlib.sha256(os.path.normcase(str(prefix.resolve())).encode()).hexdigest()
    public = {'schema': 'distilgpt2-year-label-control-launch-v2',
              'source_revision': 'b256a6846ad2386d0c0c306003026bbd9b2893a8',
              'source_tree': 'f23aa6799a145b9168677563da91d5ec78abc9b7',
              'intended_execution': {'pycache_prefix_sha256': prefix_sha}}
    raw = json.dumps(public).encode()
    (attempt/'pre-run.json').write_bytes(raw)
    private = {'source_root': str(source), 'prior_attempt_dir': str(prior), 'cache_dir': str(cache),
               'pycache_prefix': str(prefix), 'output_dir': str(attempt/'run')}
    (attempt/'private-launch.json').write_text(json.dumps(private), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['launcher', str(attempt), '--pre-run-sha256', hashlib.sha256(raw).hexdigest()])
    return attempt, prefix


@pytest.mark.parametrize('interrupt', [False, True])
def test_one_child_is_reaped_and_receipt_includes_observed_state(tmp_path, monkeypatch, interrupt):
    attempt, prefix = _setup(tmp_path, monkeypatch)
    calls, receipts = [], []
    class FakeProcess:
        pid = 42
        def __init__(self):
            self.waits, self.terminated = 0, False
        def wait(self):
            self.waits += 1
            if interrupt and self.waits == 1:
                raise KeyboardInterrupt
            return 143 if interrupt else 0
        def poll(self):
            return None
        def terminate(self):
            self.terminated = True
    child = FakeProcess()
    def popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        kwargs['stdout'].write(b'child output')
        return child
    def run(cmd, **kwargs):
        receipts.append((cmd, kwargs))
        return SimpleNamespace(stdout=b'{"receipt_sha256":"synthetic"}')
    monkeypatch.setattr(launch_v2.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch_v2.subprocess, 'run', run)
    assert launch_v2.main() == (1 if interrupt else 0)
    assert len(calls) == len(receipts) == 1
    cmd, kwargs = calls[0]
    assert '-B' in cmd and '-X' in cmd and f'pycache_prefix={prefix}' in cmd
    assert '--prior-attempt-dir' in cmd and '--output-dir' in cmd
    assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '-1' and kwargs['env']['HF_HUB_OFFLINE'] == '1'
    assert ('--interrupted' in receipts[0][0]) is interrupt
    assert child.terminated is interrupt and (attempt/'stdout.txt').read_bytes() == b'child output'


def test_nonempty_prefix_stops_before_starting_a_child(tmp_path, monkeypatch):
    _, prefix = _setup(tmp_path, monkeypatch)
    (prefix/'unexpected.pyc').write_bytes(b'x')
    def reject(*args, **kwargs):
        raise AssertionError('no child should start')
    monkeypatch.setattr(launch_v2.subprocess, 'Popen', reject)
    with pytest.raises(ValueError, match='empty-prefix'):
        launch_v2.main()


def test_broken_start_progress_stream_reaps_child_and_writes_interrupted_receipt(tmp_path, monkeypatch):
    import builtins
    attempt, _ = _setup(tmp_path, monkeypatch)
    calls, receipts = [], []
    class FakeProcess:
        pid = 42
        terminated = False
        waits = 0
        def poll(self):
            return None
        def terminate(self):
            self.terminated = True
        def wait(self):
            self.waits += 1
            return 143
    child = FakeProcess()
    def popen(*args, **kwargs):
        calls.append(1)
        return child
    def run(cmd, **kwargs):
        receipts.append(cmd)
        return SimpleNamespace(stdout=b'{"receipt_sha256":"synthetic"}')
    original_print = builtins.print
    def broken_start(*args, **kwargs):
        if args and 'direct_child_started' in str(args[0]):
            raise BrokenPipeError('synthetic progress failure')
        return original_print(*args, **kwargs)
    monkeypatch.setattr(launch_v2.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch_v2.subprocess, 'run', run)
    monkeypatch.setattr(builtins, 'print', broken_start)
    assert launch_v2.main() == 1
    assert len(calls) == len(receipts) == 1 and child.terminated and child.waits == 1
    assert '--interrupted' in receipts[0]
    assert (attempt/'stdout.txt').exists() and (attempt/'stderr.txt').exists()


def test_spawn_lost_return_is_uncertain_and_never_retried(tmp_path, monkeypatch):
    attempt, _ = _setup(tmp_path, monkeypatch)
    calls = []
    def interrupted_spawn(*args, **kwargs):
        calls.append(1)
        raise KeyboardInterrupt
    def no_receipt(*args, **kwargs):
        raise AssertionError('no receipt can certify an unknown child')
    monkeypatch.setattr(launch_v2.subprocess, 'Popen', interrupted_spawn)
    monkeypatch.setattr(launch_v2.subprocess, 'run', no_receipt)
    with pytest.raises(KeyboardInterrupt):
        launch_v2.main()
    assert len(calls) == 1 and not (attempt/'supervisor.json').exists()
    assert (attempt/'stdout.txt').exists() and (attempt/'stderr.txt').exists()
