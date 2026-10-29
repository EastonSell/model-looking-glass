"""Public known-exposure checks; no model or private cases are accessed."""

from pathlib import Path

import pytest

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_exposure as exposure


ROOT = Path(__file__).resolve().parents[1]


def test_published_snapshot_contains_all_exact_public_proposals():
    result = exposure.verify_public_reservations(ROOT)
    assert result["canonical_payload_sha256"] == exposure.INVENTORY_SHA256
    assert result["inventory_commit_label"] == exposure.INVENTORY_COMMIT
    assert result["public_evaluation_rows"] == 13
    assert result["combined_public_rows_checked"] == 26
    assert result["known_row_collisions"] == 0
    assert result["complete_history_certified"] is False
    assert result["private_split_check_performed"] is False


def test_other_checkout_inventory_import_is_rejected(tmp_path):
    with pytest.raises(v1.StudyError):
        exposure.verify_public_reservations(tmp_path)


def test_changed_public_case_cannot_reuse_the_old_reservation(monkeypatch):
    cases = list(v2.EVALUATION_CASES)
    name, role, text, ids, label = cases[0]
    cases[0] = (name, role, text, ids + (1,), label)
    monkeypatch.setattr(v2, "EVALUATION_CASES", tuple(cases))
    with pytest.raises(v1.StudyError):
        exposure.verify_public_reservations(ROOT)


def test_changed_inventory_digest_is_rejected(monkeypatch):
    monkeypatch.setattr(exposure.inventory_v4, "inventory_digest", lambda: "0" * 64)
    with pytest.raises(v1.StudyError):
        exposure.verify_public_reservations(ROOT)
