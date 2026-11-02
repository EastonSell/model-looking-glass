"""Source-first replay entry tests; no model or protected input is used."""
import hashlib
import importlib.util
import os
from pathlib import Path
import py_compile

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'tools/audit_agreement_invariant_v1.py'


def entry():
    if not ENTRY.exists():
        pytest.fail('source-first invariant reader entry is missing')
    spec = importlib.util.spec_from_file_location('_invariant_entry_test', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_reader_loads_and_changed_source_does_not_execute(tmp_path):
    helper = entry()
    reader = helper.load_reader(ROOT / 'looking_glass/agreement_invariant_science_v1.py')
    assert callable(reader.audit_attempt)
    bad = tmp_path / 'reader.py'
    bad.write_text('raise RuntimeError("must not execute")', encoding='utf-8')
    with pytest.raises(ValueError):
        helper.load_reader(bad)


def test_source_first_loading_ignores_same_size_stale_bytecode(tmp_path, monkeypatch):
    helper = entry()
    source = tmp_path / 'reader.py'
    source.write_bytes(b"VALUE = 'evil'\n")
    py_compile.compile(str(source), doraise=True)
    timestamp = int(source.stat().st_mtime)
    source.write_bytes(b"VALUE = 'good'\n")
    os.utime(source, (timestamp, timestamp))
    monkeypatch.setattr(helper, 'READER_SHA256', hashlib.sha256(source.read_bytes()).hexdigest())
    assert helper.load_reader(source).VALUE == 'good'


def test_oversized_source_is_rejected_before_execution(tmp_path):
    source = tmp_path / 'reader.py'
    source.write_bytes(b'#' * (1024 * 1024 + 1))
    with pytest.raises(ValueError):
        entry().load_reader(source)
