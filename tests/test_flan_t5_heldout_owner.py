"""Synthetic process and accounting tests; no checkpoint or held-out inputs."""
import io
import json
import subprocess
import sys
import types

import pytest

from looking_glass import flan_t5_heldout_owner as owner


def fake_inventory(root):
    entries = sorted(({'path': p.relative_to(root).as_posix(), 'size': p.stat().st_size}
                      for p in root.rglob('*') if p.is_file()), key=lambda x: x['path'])
    return {'bytes': sum(e['size'] for e in entries), 'files': len(entries), 'entries': entries}


def install_inventory(monkeypatch):
    module = types.ModuleType('looking_glass.flan_t5_heldout_custody')
    module.inventory_custody = fake_inventory
    monkeypatch.setitem(sys.modules, module.__name__, module)


def prepare(tmp_path):
    custody, asset = tmp_path / 'custody', tmp_path / 'asset'
    custody.mkdir()
    asset.mkdir()
    observations = []
    for index in range(4):
        path = custody / f'observation-{index}.json'
        path.write_text('{}')
        observations.append(path)
    return custody, asset, tmp_path / 'attempt', observations


def test_owner_records_child_exit_and_self_inclusive_inventory(monkeypatch, tmp_path):
    install_inventory(monkeypatch)
    custody, asset, output, observations = prepare(tmp_path)
    source = {'commit': 'a' * 40, 'source_sha256': {}}
    monkeypatch.setattr(owner, 'source_identity', lambda: source)

    class Child:
        def __init__(self, command, **kwargs):
            assert command[1:3] == ['-B', '-X']
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '-1'
            output.mkdir()
            (output / 'completed.json').write_text(json.dumps({
                'schema': 'mlg-flant5-heldout-response-v1', 'status': 'complete', 'source': source}))
            self.stdout, self.stderr = io.BytesIO(b'finished\n'), io.BytesIO(b'')
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(owner.subprocess, 'Popen', Child)
    result = owner.run_owner(asset, output, custody, *observations)
    assert result['status'] == 'complete'
    receipt = json.loads((custody / 'owner-completed.json').read_bytes())
    assert receipt['child_exit_code'] == 0
    assert receipt['final_custody_inventory'] == fake_inventory(custody)
    assert (custody / 'owner.stdout.bin').read_bytes() == b'finished\n'
    with pytest.raises(ValueError, match='already exists'):
        owner.run_owner(asset, tmp_path / 'different-attempt', custody, *observations)


def test_oversized_child_output_is_retained_as_incomplete(monkeypatch, tmp_path):
    install_inventory(monkeypatch)
    custody, asset, output, observations = prepare(tmp_path)
    source = {'commit': 'a' * 40, 'source_sha256': {}}
    monkeypatch.setattr(owner, 'source_identity', lambda: source)

    class Child:
        def __init__(self, command, **kwargs):
            output.mkdir()
            (output / 'completed.json').write_text(json.dumps({
                'schema': 'mlg-flant5-heldout-response-v1', 'status': 'complete', 'source': source}))
            self.stdout, self.stderr = io.BytesIO(b'x' * (owner.FILE_LIMIT + 1)), io.BytesIO(b'')
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(owner.subprocess, 'Popen', Child)
    result = owner.run_owner(asset, output, custody, *observations)
    receipt = json.loads((custody / 'owner-completed.json').read_bytes())
    assert result['status'] == 'incomplete'
    assert receipt['streams_exceeded_bound']
    assert receipt['stream_states'][0]['observed_bytes'] == owner.FILE_LIMIT + 1
    assert len((custody / 'owner.stdout.bin').read_bytes()) == owner.FILE_LIMIT
    assert receipt['final_custody_inventory'] == fake_inventory(custody)


def test_failed_spawn_is_retained_without_a_retry(monkeypatch, tmp_path):
    install_inventory(monkeypatch)
    custody, asset, output, observations = prepare(tmp_path)
    monkeypatch.setattr(owner, 'source_identity', lambda: {'commit': 'a' * 40, 'source_sha256': {}})
    monkeypatch.setattr(owner.subprocess, 'Popen', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('secret')))
    with pytest.raises(OSError):
        owner.run_owner(asset, output, custody, *observations)
    receipt = json.loads((custody / 'owner-completed.json').read_bytes())
    assert receipt['status'] == 'incomplete' and receipt['error_type'] == 'OSError'
    assert b'secret' not in (custody / 'owner-completed.json').read_bytes()
    assert receipt['final_custody_inventory'] == fake_inventory(custody)


def test_live_child_timeout_is_stopped_reaped_and_retained(monkeypatch, tmp_path):
    install_inventory(monkeypatch)
    custody, asset, output, observations = prepare(tmp_path)
    monkeypatch.setattr(owner, 'source_identity', lambda: {'commit': 'a' * 40, 'source_sha256': {}})
    monkeypatch.setattr(owner, 'OWNER_TIMEOUT_SECONDS', 0)

    class Child:
        def __init__(self, command, **kwargs):
            self.stdout, self.stderr = io.BytesIO(b''), io.BytesIO(b'')
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(owner.subprocess, 'Popen', Child)
    result = owner.run_owner(asset, output, custody, *observations)
    receipt = json.loads((custody / 'owner-completed.json').read_bytes())
    assert result['status'] == 'incomplete' and receipt['timed_out']
    assert receipt['stop']['terminate_attempted'] and receipt['stop']['reaped']
    with pytest.raises(ValueError, match='already exists'):
        owner.run_owner(asset, tmp_path / 'another-attempt', custody, *observations)


def test_stop_escalates_and_reports_unconfirmed_reap():
    class StubbornChild:
        def __init__(self, reap_after_kill):
            self.calls = []
            self.reap_after_kill = reap_after_kill

        def poll(self):
            return None

        def terminate(self):
            self.calls.append('terminate')

        def kill(self):
            self.calls.append('kill')

        def wait(self, timeout=None):
            self.calls.append('wait')
            if self.reap_after_kill and 'kill' in self.calls:
                return -9
            raise subprocess.TimeoutExpired('fabricated', timeout)

    child = StubbornChild(True)
    result = owner._stop_child(child)
    assert result['terminate_attempted'] and result['kill_attempted'] and result['reaped']
    assert child.calls == ['terminate', 'wait', 'kill', 'wait']
    child = StubbornChild(False)
    result = owner._stop_child(child)
    assert not result['reaped'] and result['error_type'] == 'TimeoutExpired'
