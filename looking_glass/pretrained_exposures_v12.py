"""Reserve the public copying factorial cases before any forward."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v6 as v6
from . import pretrained_exposures_v11 as v11


INVENTORY_VERSION = "distilgpt2-development-exposures-v12"
PREDECESSOR_SHA256 = "84b1a95a41670798ff55c9f08272890c00a939a74a4cad7ad9b95fbf4bcdaa25"
INVENTORY_SHA256 = "af374aaceb708b3b38e06b3955ca975467b048162ee105cbfcc9452a8a600e49"
TARGET = v11.TARGET
PUBLIC_DISCOVERY = v11.PUBLIC_DISCOVERY
PROTOCOL_COMMIT = "9dfc2952d3b21c85f0f7afc6bc6d902ed210634c"
CASES_COMMIT = "7a5a7d6eaea4c82f02f7f4e05b6e5d41c85ecb49"
CASES_PATHS = (
    "docs/DISTILGPT2_COPY_HEAD_FACTORIAL_V1.md",
    "experiments/distilgpt2_copy_head_factorial_v1.json",
)
PROTOCOL_SHA256 = "467b566fd149bc7e489c0be265c01b63c3f005804bf30c7873560df930aa9d92"
CASES_SHA256 = "69f8463da5f919fb99aef05aa4401de439248100e99ca90899f8d9f98af90d24"
COMPLETE_ROWS_SHA256 = "8ecee93b9ec137be5523f20751e4f6d1db231fafa384d32270be9849cb09e632"
FACTORIAL_GROUPS = (
    (7051, 42775, 19516, 15191, 14237, 39459),
    (21670, 17644, 41266, 30089, 9891, 8887),
    (15900, 9875, 12967, 26917, 22045, 39610),
    (14093, 18364, 22740, 19236, 17182, 6333),
    (40334, 1618, 14015, 42644, 32859, 18662),
    (7779, 1323, 17536, 18757, 19612, 26450),
    (10147, 13209, 6877, 6297, 29144, 32263),
    (18447, 20710, 29461, 6508, 7480, 24808),
)
FACTORIAL_ROWS = tuple(
    tuple(group[i] for i in permutation + (0, 1, 2, 3, 4))
    for group in FACTORIAL_GROUPS for permutation in v6.PUBLIC_COPY_PERMUTATIONS
)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def inventory_payload():
    predecessor = v11.inventory_payload()
    canonical = json.dumps(predecessor, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    if (v11.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(canonical).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v11.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_public_copying_factorial_rows": 32}
    rows = list(FACTORIAL_ROWS)
    tokens = [token for group in FACTORIAL_GROUPS for token in group]
    expected = [tuple(group[i] for i in permutation + (0, 1, 2, 3, 4))
                for group in FACTORIAL_GROUPS for permutation in v6.PUBLIC_COPY_PERMUTATIONS]
    if (len(FACTORIAL_GROUPS) != 8 or any(len(group) != 6 for group in FACTORIAL_GROUPS)
            or len(tokens) != 48 or len(set(tokens)) != 48
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError("public factorial groups do not satisfy the fixed token constraints")
    if len(rows) != 32 or rows != expected or len(set(rows)) != 32:
        raise ValueError("public factorial rows do not match the fixed permutations")
    old_rows = [tuple(row["token_ids"]) for row in payload["records"]]
    complete_rows = [row + (FACTORIAL_GROUPS[index // 4][5],)
                     for index, row in enumerate(rows)]
    if any(_related(left, right) for left in complete_rows for right in old_rows):
        raise ValueError("public factorial input-plus-target has a full-row prefix relation with a predecessor")
    if any(_related(left, right) for i, left in enumerate(complete_rows)
           for right in complete_rows[i + 1:]):
        raise ValueError("public factorial complete rows contain a full-row prefix relation")
    old_prefixes = {row[:length] for row in old_rows for length in range(1, len(row) + 1)}
    if any(row[:length] in old_prefixes for row in complete_rows
           for length in range(1, len(row) + 1)):
        raise ValueError("public factorial row has a prefix in the known inventory")
    if any((token,) in old_prefixes for token in tokens):
        raise ValueError("public factorial group has a known one-token prefix")
    old_ids = {row["record_id"] for row in payload["records"]}
    for index, row in enumerate(rows):
        group, variant = divmod(index, 4)
        record_id = f"planned-copy-factorial-{group + 1:02d}-{variant}"
        if record_id in old_ids:
            raise ValueError("public factorial record ID already exists")
        payload["records"].append({
            "record_id": record_id,
            "category": "planned", "token_ids": list(row),
            "target_token_id": FACTORIAL_GROUPS[group][5],
            "provenance": ("Public non-blind copying-head factorial and fixed controls. "
                           f"Case inventory SHA-256 {CASES_SHA256}. "
                           "No model forward is asserted by this reservation."),
            "references": list(CASES_PATHS), "public_text": "",
            "reference_commit": CASES_COMMIT,
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


def _has_known_relation(row, forbidden, complete_rows):
    return row in forbidden or any(row[:len(known)] == known for known in complete_rows)


def check_known_splits(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                       expected_model_identity, cases_per_split=8):
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    if type(cases_per_split) is not int or cases_per_split not in (8, 16):
        raise ValueError("cases per split must be exactly 8 or 16")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _forbidden_rows()
    complete_rows = {tuple(row["token_ids"]) for row in inventory_payload()["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in discovery + heldout):
        raise ValueError("a candidate token sequence is in the development exclusion inventory")
    return v1.KnownSplitCheck(INVENTORY_VERSION, expected_inventory_sha256, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _forbidden_rows()
    complete_rows = {tuple(row["token_ids"]) for row in inventory_payload()["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in heldout):
        raise ValueError("a held-out token sequence is in the development exclusion inventory")
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, expected_inventory_sha256, 32, len(forbidden),
                                    sum(_has_known_relation(row, forbidden, complete_rows) for row in discovery))


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
