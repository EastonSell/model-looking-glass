"""Public-only checks for the corrected fresh-context planned reservation."""

import copy
import hashlib

import pytest

from looking_glass import pretrained_exposures_v43 as prior
from looking_glass import pretrained_exposures_v44 as current


@pytest.fixture(scope="module")
def predecessor():
    payload = prior.inventory_payload()
    assert hashlib.sha256(current._canonical(payload)).hexdigest() == current.PREDECESSOR_SHA256
    return payload


def test_v44_preserves_public_records_and_adds_only_fixed_planned_cases(predecessor):
    payload = current._build_payload(predecessor)
    assert hashlib.sha256(current._canonical(payload)).hexdigest() == current.INVENTORY_SHA256
    assert payload["records"][:-16] == predecessor["records"]
    assert len(current._prefixes(payload["records"])) == current.FORBIDDEN_PREFIX_COUNT
    for i, record in enumerate(payload["records"][-16:], 1):
        assert record["record_id"] == f"planned-agreement-head-v4-{i:02d}"
        assert record["category"] == "planned"
        assert record["candidate_token_ids"] == [318, 389, 373, 547]
        assert record["reference_commit"] == current.CASE_COMMIT


@pytest.mark.parametrize("mode", ["duplicate", "prefix"])
def test_v44_rejects_duplicate_identity_or_known_prefix(predecessor, mode):
    changed = copy.deepcopy(predecessor)
    if mode == "duplicate":
        changed["records"].append(copy.deepcopy(changed["records"][0]))
        expected = "identities repeat"
    else:
        case = current._build_payload(predecessor)["records"][-16]
        changed["records"].append({"record_id": "synthetic-prefix", "token_ids": case["token_ids"][:2]})
        expected = "prefix relation"
    with pytest.raises(ValueError, match=expected):
        current._build_payload(changed)
