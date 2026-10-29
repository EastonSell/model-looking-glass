"""Verify the additive planned-input successor without model execution."""
import copy
import hashlib

import pytest

from looking_glass import pretrained_exposures_v30 as old, pretrained_exposures_v31 as new


@pytest.fixture(scope="module")
def predecessor():
    payload, digest = old._checked_inventory()
    assert digest == new.PREDECESSOR_SHA256
    return payload


def test_additive_canonical_inventory_and_exact_inputs(predecessor):
    payload = new._build_payload(predecessor)
    assert payload["records"][:915] == predecessor["records"]
    assert len(payload["records"]) == 949
    assert len(new._prefixes(payload["records"])) == 5789
    assert hashlib.sha256(new._canonical(payload)).hexdigest() == new.INVENTORY_SHA256
    added = payload["records"][915:]
    assert tuple(tuple(row["token_ids"]) for row in added[:32]) == new.COPY_PROMPT_DOSE_ROWS
    assert tuple((row["record_id"], tuple(row["token_ids"])) for row in added[32:]) == new.SYNTHETIC_SERVICE_CASES
    assert all(row["category"] == "planned" for row in added)
    assert all(row["attention_mask"] == [1] * len(row["token_ids"]) for row in added[32:])


def test_predecessor_input_collision_is_rejected(predecessor):
    modified = copy.deepcopy(predecessor)
    modified["records"].append({"record_id": "fixture-collision", "category": "planned",
                                "token_ids": list(new.COPY_PROMPT_DOSE_ROWS[0])})
    with pytest.raises(ValueError, match="predecessor"):
        new._build_payload(modified)


def test_new_input_collision_is_rejected(predecessor, monkeypatch):
    monkeypatch.setattr(new, "SYNTHETIC_SERVICE_CASES", (
        ("fixture-service-a", new.COPY_PROMPT_DOSE_ROWS[0]),
        ("fixture-service-b", (50256, 2001))))
    with pytest.raises(ValueError, match="identities or rows"):
        new._build_payload(predecessor)
