"""Reserve four public broker-check inputs without changing v4 evidence."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v4 as v4


INVENTORY_VERSION = "distilgpt2-development-exposures-v5"
PREDECESSOR_SHA256 = "6c065926d08cf08aa98d2c2725299d6bdc25e919246851c34bd9c7d63e8a4af1"
INVENTORY_SHA256 = "1c597a028952a40ac48614c0e263a87a034d4f6a3df2b4cbffcdb6047adf6586"
TARGET = v4.TARGET
PUBLIC_DISCOVERY = v4.PUBLIC_DISCOVERY
PLAN_COMMIT = "332c1c6511a34f9658a1af5e0f2dd83a93f2eeaa"
PLAN_PATH = "docs/GROUPED_GPU_DEV_PLAN.md"

# Public synthetic broker-check rows, planned but not forwarded at this snapshot.
PUBLIC_BROKER_CHECK = tuple((50256, number, number + 100, number + 200)
                            for number in (107, 108, 109, 110))


def inventory_payload():
    predecessor = v4.inventory_payload()
    canonical = json.dumps(predecessor, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    if (v4.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(canonical).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    rows = list(PUBLIC_BROKER_CHECK)
    predecessor_forbidden = {tuple(record["token_ids"][:length])
                             for record in predecessor["records"]
                             for length in range(1, len(record["token_ids"]) + 1)}
    predecessor_full = {tuple(record["token_ids"]) for record in predecessor["records"]}
    if len(rows) != 4 or len(set(rows)) != 4 or any(row in predecessor_forbidden for row in rows):
        raise ValueError("public broker reservations do not match their expected split")
    if any(row[:len(old)] == old for row in rows for old in predecessor_full):
        raise ValueError("public broker reservation extends a previous full row")
    if any(left[:len(right)] == right or right[:len(left)] == left
           for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("public broker reservations contain a full-row prefix relation")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v4.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_public_broker_check_rows": 4, "model_forward_asserted": False}
    for index, tokens in enumerate(PUBLIC_BROKER_CHECK, 1):
        payload["records"].append({
            "record_id": f"planned-grouped-broker-check-{index:02}",
            "category": "planned",
            "token_ids": list(tokens),
            "provenance": ("Public synthetic broker-check row in the grouped GPU development plan. "
                           "Held out from that plan's proposed policy construction; no forward is asserted."),
            "references": [PLAN_PATH],
            "public_text": "",
            "reference_commit": PLAN_COMMIT,
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
