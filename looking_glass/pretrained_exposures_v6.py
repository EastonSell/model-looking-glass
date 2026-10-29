"""Reserve public copying-screen inputs without changing v5 evidence."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v5 as v5


INVENTORY_VERSION = "distilgpt2-development-exposures-v6"
PREDECESSOR_SHA256 = "1c597a028952a40ac48614c0e263a87a034d4f6a3df2b4cbffcdb6047adf6586"
INVENTORY_SHA256 = "76803775e6929b4a20463a50bcae839a727b1a875a0d433f7c27c9d1d69f0681"
TARGET = v5.TARGET
PUBLIC_DISCOVERY = v5.PUBLIC_DISCOVERY
PLAN_COMMIT = "55d11a53eb34a13eb6b14e19a5bd0c47172edda5"
PLAN_PATHS = ("docs/DISTILGPT2_COPYING_DEV_V1.md", "scripts/copying_dev_v1.py")

# Fixed public copying-screen source rows and controls; no forward is asserted.
PUBLIC_COPY_GROUPS = (
    (17180, 7850, 2647, 4077, 8223, 45725),
    (8465, 6279, 11376, 4512, 26839, 18873),
    (8222, 4936, 4324, 9151, 7872, 3348),
    (15121, 21613, 10162, 18355, 9294, 14032),
    (22746, 9082, 7988, 10912, 16669, 30623),
    (7374, 19751, 15317, 47750, 3113, 3491),
    (21190, 17972, 15224, 6891, 7022, 19132),
    (13385, 10701, 7543, 28699, 41207, 15061),
)
PUBLIC_COPY_PERMUTATIONS = ((0, 1, 2, 3, 4, 5), (4, 2, 0, 3, 1, 5),
                            (1, 3, 4, 2, 0, 5), (3, 0, 4, 1, 2, 5))
PUBLIC_COPY_ROWS = tuple(tuple(group[i] for i in permutation + (0, 1, 2, 3, 4))
                         for group in PUBLIC_COPY_GROUPS for permutation in PUBLIC_COPY_PERMUTATIONS)


def inventory_payload():
    if v5.inventory_digest() != PREDECESSOR_SHA256:
        raise ValueError("predecessor exposure inventory does not match")
    rows = list(PUBLIC_COPY_ROWS)
    predecessor_forbidden = v5._forbidden_rows()
    predecessor_full = {tuple(record["token_ids"]) for record in v5.inventory_payload()["records"]}
    if len(rows) != 32 or len(set(rows)) != 32 or any(row in predecessor_forbidden for row in rows):
        raise ValueError("public copying reservations do not match their expected split")
    if any(row[:len(old)] == old for row in rows for old in predecessor_full):
        raise ValueError("public copying reservation extends a previous full row")
    if any(left[:len(right)] == right or right[:len(left)] == left
           for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("public copying reservations contain a full-row prefix relation")
    payload = copy.deepcopy(v5.inventory_payload())
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v5.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_public_copying_rows": 32, "model_forward_asserted": False}
    for index, tokens in enumerate(PUBLIC_COPY_ROWS):
        group, variant = divmod(index, 4)
        payload["records"].append({
            "record_id": f"planned-copy-screen-{group + 1:02d}-{variant}",
            "category": "planned",
            "token_ids": list(tokens),
            "target_token_id": PUBLIC_COPY_GROUPS[group][5],
            "provenance": ("Public non-blind repeated-sequence development screen and fixed controls. "
                           "No model forward is asserted by this reservation."),
            "references": list(PLAN_PATHS),
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
