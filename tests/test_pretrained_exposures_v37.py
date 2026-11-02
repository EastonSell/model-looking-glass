"""Model-free regression checks for the planned-only public inventory."""
import hashlib

import pytest

from looking_glass import pretrained_exposures_v36 as v36
from looking_glass import pretrained_exposures_v37 as v37


def test_exact_predecessor_and_planned_cases():
    old = v36.inventory_payload()
    current = v37.inventory_payload()
    assert current["records"][:len(old["records"])] == old["records"]
    assert len(current["records"]) == len(old["records"]) + 32
    assert hashlib.sha256(v37.canonical_inventory_bytes()).hexdigest() == v37.INVENTORY_SHA256
    for (ident, text, ids, target), row in zip(v37.COPY_OUTPUT_DOSE_CASES, current["records"][len(old["records"]):]):
        assert row["record_id"] == "planned-" + ident and row["category"] == "planned"
        assert row["public_text"] == text and row["token_ids"] == list(ids) and row["target_token_id"] == target


def test_existing_input_prefix_rejected():
    old = v36.inventory_payload()
    old["records"].append({"record_id": "fabricated-prefix", "token_ids": list(v37.COPY_OUTPUT_DOSE_CASES[0][2][:5])})
    with pytest.raises(ValueError, match="known prefix relation"):
        v37._build_payload(old)


def test_duplicate_identity_rejected():
    old = v36.inventory_payload()
    old["records"].append(dict(old["records"][0]))
    with pytest.raises(ValueError, match="predecessor identities"):
        v37._build_payload(old)
