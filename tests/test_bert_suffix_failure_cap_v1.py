"""Exercise the suffix owner's failure-only storage ceiling without a model."""
from pathlib import Path
import sys
import pytest
from scripts import bert_agreement_suffix_v1 as owner


def size(path):
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def test_real_failure_handler_keeps_attempt_under_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'OUTPUT_CAP', 4096)
    monkeypatch.setattr(owner, 'TAIL_RESERVE', 1)
    monkeypatch.setattr(owner, 'EARLY_SOURCE', {'fake':'digest'})
    monkeypatch.setattr(owner, 'EARLY_GIT', 'fake-git-digest')
    monkeypatch.setattr(owner, 'verify_source', lambda *args: owner.EARLY_SOURCE)
    monkeypatch.setattr(owner, 'file_sha', lambda *args: owner.EARLY_GIT)
    monkeypatch.setattr(owner.common, 'verify_assets', lambda *args: {})
    monkeypatch.setattr(owner.AutoTokenizer, 'from_pretrained', lambda *a, **kw: object())
    monkeypatch.setattr(owner, 'check_manifest', lambda *args: None)
    monkeypatch.setattr(owner, 'require_policy', lambda *args: None)
    monkeypatch.setattr(owner.common, 'memory_sample', lambda *args: None)
    monkeypatch.setattr(owner.torch, 'set_num_threads', lambda *args: None)
    monkeypatch.setattr(owner.torch, 'set_num_interop_threads', lambda *args: None)
    def fail(*args, **kwargs):
        raise RuntimeError('X' * 10000)
    monkeypatch.setattr(owner.AutoModelForMaskedLM, 'from_pretrained', fail)
    attempt = tmp_path / 'attempt'
    cases = Path(__file__).resolve().parents[1] / 'experiments/bert_agreement_suffix_cases_v1.json'
    monkeypatch.setattr(sys, 'argv', ['probe', '--site', str(tmp_path), '--git', str(tmp_path/'git'),
        '--model-dir', str(tmp_path), '--cases', str(cases), '--output', str(attempt),
        '--device', 'cpu', '--source-commit', 'a'*40])
    with pytest.raises(RuntimeError, match='XXXXX'):
        owner.main()
    assert (attempt/'failure.txt').is_file()
    assert size(attempt) <= owner.OUTPUT_CAP
    assert b'truncated' in (attempt/'failure.txt').read_bytes()
    assert not (attempt/'complete.json').exists()


def test_no_remaining_space_still_creates_failure_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'OUTPUT_CAP', 32)
    (tmp_path/'prior').write_bytes(b'X'*32)
    owner.save_failure(tmp_path, 'failure text')
    assert (tmp_path/'failure.txt').is_file()
    assert (tmp_path/'failure.txt').stat().st_size == 0
    assert size(tmp_path) == 32


def test_small_message_is_preserved_and_large_unicode_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'OUTPUT_CAP', 50)
    owner.save_failure(tmp_path, 'a short error\n')
    assert (tmp_path/'failure.txt').read_text(encoding='utf-8') == 'a short error\n'
    other = tmp_path/'other'
    other.mkdir()
    owner.save_failure(other, '🧪' * 1000)
    assert size(other) <= 50
    assert b'truncated' in (other/'failure.txt').read_bytes()
