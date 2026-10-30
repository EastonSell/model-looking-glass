"""Reserve the published public copying follow-up before any forward."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v6 as v6
from . import pretrained_exposures_v7 as v7
from . import pretrained_exposures_v8 as v8


INVENTORY_VERSION = "distilgpt2-development-exposures-v9"
PREDECESSOR_SHA256 = "22b1392ed43568fcac574b55ae6cf0212766771d8439ac39e599f7aa112b298f"
INVENTORY_SHA256 = "e0c394786b316ea61b9f3697063eb86c7cf290d898c3ed84549880df8af91c4d"
TARGET = v8.TARGET
PUBLIC_DISCOVERY = v8.PUBLIC_DISCOVERY
FOLLOWUP_COMMIT = "939d59540b642c83bc999bcf45220ff2155f41b2"
FOLLOWUP_PATHS = (
    "docs/DISTILGPT2_COPY_HEAD_FOLLOWUP_V2.md",
    "experiments/distilgpt2_copy_head_followup_v2.json",
)
FOLLOWUP_PROTOCOL_SHA256 = "70616bb01aa7ccb8440da9dc503c274a3d5cb7b24dd3965fd985de6c645ca68c"
FOLLOWUP_CASES_SHA256 = "3831bfd1a48aaa3fcb67fbdb3cdc10f4422149b1fed90fd5267d3c73525255a8"
FOLLOWUP_GROUPS = (
    (5118, 4898, 4171, 8801, 8509, 6131),
    (43836, 6953, 6147, 8946, 32410, 12788),
    (7815, 25996, 47565, 41236, 21219, 8701),
    (20698, 48710, 13026, 21027, 8848, 4074),
    (6701, 19401, 17026, 31855, 8278, 10326),
    (16161, 3084, 1492, 20450, 5509, 6512),
    (2156, 8429, 36953, 3797, 7545, 11311),
    (5405, 9379, 10586, 10211, 20922, 5888),
)
FOLLOWUP_ROWS = tuple(
    tuple(group[i] for i in permutation + (0, 1, 2, 3, 4))
    for group in FOLLOWUP_GROUPS for permutation in v6.PUBLIC_COPY_PERMUTATIONS
)


def inventory_payload():
    predecessor = v8.inventory_payload()
    canonical = json.dumps(predecessor, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    if (v8.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(canonical).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v8.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_public_copying_followup_rows": 32}
    rows = list(FOLLOWUP_ROWS)
    tokens = [token for group in FOLLOWUP_GROUPS for token in group]
    previous_tokens = {token for group in v6.PUBLIC_COPY_GROUPS + v7.PUBLIC_EVALUATION_GROUPS for token in group}
    if (len(FOLLOWUP_GROUPS) != 8 or any(len(group) != 6 for group in FOLLOWUP_GROUPS)
            or len(tokens) != 48 or len(set(tokens)) != 48 or set(tokens) & previous_tokens):
        raise ValueError("public copying follow-up groups do not satisfy the fixed token constraints")
    expected_rows = [tuple(group[i] for i in permutation + (0, 1, 2, 3, 4))
                     for group in FOLLOWUP_GROUPS for permutation in v6.PUBLIC_COPY_PERMUTATIONS]
    if len(rows) != 32 or rows != expected_rows or len(set(rows)) != 32:
        raise ValueError("public copying follow-up rows do not match the fixed permutations")
    old_rows = [tuple(row["token_ids"]) for row in payload["records"]]
    if any(left[:len(right)] == right or right[:len(left)] == left
           for left in rows for right in old_rows):
        raise ValueError("public copying follow-up has a full-row prefix relation with a predecessor")
    if any(left[:len(right)] == right or right[:len(left)] == left
           for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("public copying follow-up contains a full-row prefix relation")
    old_ids = {row["record_id"] for row in payload["records"]}
    for index, tokens in enumerate(rows):
        group, variant = divmod(index, 4)
        record_id = f"planned-copy-followup-{group + 1:02d}-{variant}"
        if record_id in old_ids:
            raise ValueError("public copying follow-up record ID already exists")
        payload["records"].append({
            "record_id": record_id,
            "category": "planned", "token_ids": list(tokens),
            "target_token_id": FOLLOWUP_GROUPS[group][5],
            "provenance": ("Public non-blind copying-head follow-up and fixed controls. "
                           f"Case inventory SHA-256 {FOLLOWUP_CASES_SHA256}. "
                           "No model forward is asserted by this reservation."),
            "references": list(FOLLOWUP_PATHS), "public_text": "",
            "reference_commit": FOLLOWUP_COMMIT,
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
