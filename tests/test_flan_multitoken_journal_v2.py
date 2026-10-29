"""Versioned journal budgets and retained-byte checks without a model."""

import hashlib
import json
import struct
from unittest import mock

import pytest

from looking_glass.flan_multitoken_journal_v2 import (
    EvidenceJournalV2, LimitsV2, JournalError, require_protocol_allocation,
    MAX_COMPONENT_BYTES, METADATA_BYTES, FAILURE_BYTES,
)


def small_limits(subject="story", **changes):
    values = dict(subject=subject, total_bytes=400_000, metadata_bytes=200_000,
                  failure_bytes=10_000, forward_bytes=1024, max_forwards=4, free_bytes=0)
    values.update(changes)
    return LimitsV2(**values)


@pytest.mark.parametrize("subject,cap,calls", [
    ("story", 5 * 1024**3 // 4, 3516),
    ("concert", 5 * 1024**3 // 4, 3516),
    ("painting", 3 * 1024**3 // 4, 1758),
])
def test_production_allocation_includes_the_pinned_pass_profile(subject, cap, calls):
    limits = LimitsV2.for_subject(subject)
    assert limits.total_bytes == cap
    assert limits.max_forwards == calls
    assert limits.metadata_bytes == METADATA_BYTES and limits.failure_bytes == FAILURE_BYTES
    assert limits.forward_bytes == MAX_COMPONENT_BYTES
    assert require_protocol_allocation(limits) < cap
    with pytest.raises(JournalError, match="allocation"):
        require_protocol_allocation(small_limits(subject))


def test_binary64_source_and_residual_are_retained_as_exact_independent_bytes(tmp_path):
    journal = EvidenceJournalV2(tmp_path / "attempt", {"schema": "fake"}, limits=small_limits())
    source = struct.pack("<2d", 1.25, -0.0)
    residual = struct.pack("<2d", -2.5, 1e-300)
    source_ref = journal.vector64(source)
    residual_ref = journal.vector64(residual)
    assert source_ref["path"] != residual_ref["path"]
    assert source_ref["sha256"] == hashlib.sha256(source).hexdigest()
    assert journal.read_vector64(source_ref) == source
    assert journal.read_vector64(residual_ref) == residual
    journal.record("orthogonal", {"source": source_ref, "residual": residual_ref})
    journal.finish("complete_candidate", {"synthetic": True})


def test_rounded_nonfinite_float32_bytes_survive_for_incomplete_disposition(tmp_path):
    journal = EvidenceJournalV2(tmp_path / "attempt", {}, limits=small_limits())
    raw = struct.pack("<I", 0x7f800000)
    ref = journal.tensor((1,), raw)
    journal.record("landing", {"bits": ref, "status": "nonfinite"})
    journal.finish("incomplete", {"reason": "nonfinite landing"})
    assert (journal.directory / ref["path"]).read_bytes() == raw
    assert json.loads((journal.directory / "terminal.json").read_text())["status"] == "incomplete"


def test_corrupt_vector_and_uncertain_write_cannot_be_accepted(tmp_path):
    journal = EvidenceJournalV2(tmp_path / "attempt", {}, limits=small_limits())
    ref = journal.vector64(bytes(16))
    (journal.directory / ref["path"]).write_bytes(bytes([1]) * 16)
    with pytest.raises(JournalError, match="changed"):
        journal.read_vector64(ref)
    with mock.patch("looking_glass.flan_multitoken_journal_v2._write_new", side_effect=OSError("fake")):
        with pytest.raises(OSError):
            journal.vector64(bytes(16))
    with pytest.raises(JournalError):
        journal.begin_component({"kind": "decoder"}, reserve_bytes=16)
    journal.finish("incomplete", {"reason": "write failure"})
    assert json.loads((journal.directory / "terminal.json").read_text())["storage_accounting_complete"] is False


def test_space_and_component_limits_reject_before_start(tmp_path):
    with mock.patch("looking_glass.flan_multitoken_journal_v2.shutil.disk_usage", return_value=(100, 99, 1)):
        with pytest.raises(JournalError, match="free"):
            EvidenceJournalV2(tmp_path / "no-space", {}, limits=small_limits(free_bytes=2))
    assert not (tmp_path / "no-space").exists()
    journal = EvidenceJournalV2(tmp_path / "attempt", {}, limits=small_limits(forward_bytes=8))
    with pytest.raises(JournalError, match="reservation"):
        journal.begin_component({"kind": "decoder"}, reserve_bytes=16)
    assert journal.component_count == 0
