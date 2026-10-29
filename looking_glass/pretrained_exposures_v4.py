"""Reserve public follow-on edit inputs without changing v3 evidence."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v3 as v3


INVENTORY_VERSION = "distilgpt2-development-exposures-v4"
PREDECESSOR_SHA256 = "76baf0c8996bf2872a3a1b8fd0fe2734e15525ce0f7014e54811145cb98ce912"
INVENTORY_SHA256 = "6c065926d08cf08aa98d2c2725299d6bdc25e919246851c34bd9c7d63e8a4af1"
TARGET = v3.TARGET
PUBLIC_DISCOVERY = v3.PUBLIC_DISCOVERY

# Public proposal supplied after tokenizer-only alignment checks; no forward
# or finalized protocol is asserted by reserving these exact rows.
PUBLIC_EDIT_FOLLOWON = (
    ("target", (464, 2260, 3139, 286, 4881, 318)),
    ("target", (28572, 338, 3139, 318, 262, 1748, 286)),
    ("target", (28572, 338, 5852, 286, 1230, 318, 287)),
    ("target", (464, 4141, 3139, 1748, 318)),
    ("locality", (464, 3139, 286, 2807, 318)),
    ("locality", (464, 3139, 286, 3284, 318)),
    ("locality", (464, 3139, 286, 3340, 318)),
    ("locality", (464, 3139, 286, 19101, 318)),
    ("locality", (464, 6697, 286, 1588, 318)),
    ("locality", (2025, 1711, 4909, 24742)),
    ("locality", (32, 22950, 468, 1115)),
    ("locality", (464, 4252, 16736, 287, 262)),
    ("locality", (818, 4881, 11, 262, 45718, 3303, 318)),
)


def inventory_payload():
    if v3.inventory_digest() != PREDECESSOR_SHA256:
        raise ValueError("predecessor exposure inventory does not match")
    rows = [tokens for _, tokens in PUBLIC_EDIT_FOLLOWON]
    predecessor_forbidden = v3._forbidden_rows()
    predecessor_full = {tuple(record["token_ids"]) for record in v3.inventory_payload()["records"]}
    if len(rows) != 13 or len(set(rows)) != 13 or any(row in predecessor_forbidden for row in rows):
        raise ValueError("public follow-on reservations do not match their expected split")
    if any(row[:len(old)] == old for row in rows for old in predecessor_full):
        raise ValueError("public follow-on reservation extends a previous full row")
    if any(left[:len(right)] == right or right[:len(left)] == left
           for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("public follow-on reservations contain a full-row prefix relation")
    payload = copy.deepcopy(v3.inventory_payload())
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v3.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_public_edit_followon_rows": 13, "model_forward_asserted": False,
                         "final_protocol_published_at_snapshot": False}
    for index, (role, tokens) in enumerate(PUBLIC_EDIT_FOLLOWON, 1):
        payload["records"].append({
            "record_id": f"planned-parameter-edit-followon-{role}-{index:02}",
            "category": "planned",
            "token_ids": list(tokens),
            "provenance": ("Public non-blind follow-on proposal, supplied after pinned-tokenizer-only "
                           "alignment checks. No model forward or frozen protocol is asserted."),
            "references": [],
            "public_text": "",
            "reference_commit": "",
        })
    return payload


def canonical_inventory_bytes():
    return json.dumps(inventory_payload(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def inventory_digest():
    digest = hashlib.sha256(canonical_inventory_bytes()).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError("development-exposure inventory does not match its pinned digest")
    return digest


def _forbidden_rows():
    return {tuple(row["token_ids"][:length]) for row in inventory_payload()["records"]
            for length in range(1, len(row["token_ids"]) + 1)}


def check_known_splits(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                       expected_model_identity, cases_per_split=8):
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    if type(cases_per_split) is not int or cases_per_split not in (8, 16):
        raise ValueError("cases per split must be exactly 8 or 16")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _forbidden_rows()
    if any(row in forbidden for row in discovery + heldout):
        raise ValueError("a candidate token sequence is in the development exclusion inventory")
    return v1.KnownSplitCheck(INVENTORY_VERSION, expected_inventory_sha256, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _forbidden_rows()
    if any(row in forbidden for row in heldout):
        raise ValueError("a held-out token sequence is in the development exclusion inventory")
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, expected_inventory_sha256, 32, len(forbidden),
                                    sum(row in forbidden for row in discovery))


def _validate_known_check(check):
    count = len(_forbidden_rows())
    if (type(check) not in (v1.KnownSplitCheck, v2.ExposedDiscoveryCheck) or
            check.inventory_version != INVENTORY_VERSION or check.inventory_sha256 != inventory_digest() or
            type(check.candidate_count) is not int or type(check.forbidden_row_count) is not int or
            check.forbidden_row_count != count or check.complete_exposure_history_certified is not False or
            check.preseal_ready is not False):
        raise ValueError("a matching known-split check is required")
    if type(check) is v1.KnownSplitCheck and check.scope == "known-rows-only" and check.candidate_count in (16, 32):
        return count, "known-prefixes-excluded-from-both", 0
    if (type(check) is v2.ExposedDiscoveryCheck and check.scope == "known-prefix-holdout-only" and
            check.candidate_count == 32 and type(check.known_or_reserved_discovery_overlap_count) is int and
            0 <= check.known_or_reserved_discovery_overlap_count <= 16):
        return count, "known-prefix-discovery-reuse-disclosed", check.known_or_reserved_discovery_overlap_count
    raise ValueError("a matching known-split check is required")


def require_preseal_readiness(check):
    _validate_known_check(check)
    raise v1.ExposureInventoryIncomplete("known-row checking alone does not authorize a study seal")


def prepare_preseal_disclosure(check, *, protocol_sha256, acknowledge_prior_exposure_uncertainty=False):
    count, policy, overlap = _validate_known_check(check)
    if acknowledge_prior_exposure_uncertainty is not True:
        raise v1.ExposureInventoryIncomplete("prior exposure uncertainty must be explicitly disclosed")
    if type(protocol_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None:
        raise ValueError("a canonical protocol SHA-256 is required")
    return v2.PresealDisclosure(INVENTORY_VERSION, check.inventory_sha256, protocol_sha256,
                                 check.candidate_count // 2, count, policy, overlap)
