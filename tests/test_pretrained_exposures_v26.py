"""Fixed public reservation: model-free predecessor, metadata and collisions."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from looking_glass import pretrained_exposures_v25 as prior
from looking_glass import pretrained_exposures_v26 as current


@pytest.fixture(scope="module")
def predecessor():
    payload = prior.inventory_payload()
    assert hashlib.sha256(current._canonical(payload)).hexdigest() == current.PREDECESSOR_SHA256
    return payload


def test_exact_public_case_reservation_and_predecessor_preserved(predecessor):
    case_bytes = (Path(__file__).resolve().parents[1] / current.CASE_PATH).read_bytes()
    assert hashlib.sha256(case_bytes).hexdigest() == current.CASE_SHA256
    rows = json.loads(case_bytes)["cases"]
    assert len(rows) == len(current.FULL_TOP_CASES) == 48
    payload = current._build_payload(predecessor)
    assert payload["records"][:len(predecessor["records"])] == predecessor["records"]
    added = payload["records"][len(predecessor["records"]):]
    for case, record in zip(rows, added):
        labels = case["accepted"] if case["role"] == "locality" else [case["factual"], case["counterfactual"]]
        assert record["record_id"] == "planned-" + case["case_id"]
        assert record["category"] == "planned"
        assert record["token_ids"] == case["token_ids"]
        assert record["public_text"] == case["prompt"]
        assert record["candidate_token_ids"] == [label["token_id"] for label in labels]
        assert record["reference_commit"] == current.CASES_COMMIT
    assert hashlib.sha256(current._canonical(payload)).hexdigest() == current.INVENTORY_SHA256
    assert len(current._prefixes(payload["records"])) == current.FORBIDDEN_PREFIX_COUNT


def test_foreign_duplicate_or_prefix_is_rejected(predecessor):
    for row in (current.FULL_TOP_ROWS[0], current.FULL_TOP_ROWS[0][:-1]):
        bad = copy.deepcopy(predecessor)
        bad["records"].append({"record_id": "foreign-collision", "token_ids": list(row)})
        with pytest.raises(ValueError, match="prefix"):
            current._build_payload(bad)


def test_repeated_reservation_identity_is_rejected(predecessor):
    bad = copy.deepcopy(predecessor)
    bad["records"].append({"record_id": "planned-" + current.FULL_TOP_CASES[0][0], "token_ids": [50256]})
    with pytest.raises(ValueError, match="identities"):
        current._build_payload(bad)
