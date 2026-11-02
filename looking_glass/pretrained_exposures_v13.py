"""Record the completed public copying factorial without adding cases."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v12 as v12


INVENTORY_VERSION = "distilgpt2-development-exposures-v13"
PREDECESSOR_SHA256 = "af374aaceb708b3b38e06b3955ca975467b048162ee105cbfcc9452a8a600e49"
INVENTORY_SHA256 = "27ffe8d2ab312ccf2a62a07dab835d3cafce7364ca8abadbb1e73ec835e9aca8"
TARGET = v12.TARGET
PUBLIC_DISCOVERY = v12.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v12.FACTORIAL_GROUPS
FACTORIAL_ROWS = v12.FACTORIAL_ROWS
EVIDENCE_COMMIT = "cab7ce24057692f0611a37e799cfa3b57e842ce5"
EVIDENCE_PATHS = (
    "results/distilgpt2-copy-head-factorial-v1-20261028/run/manifest.json",
    "results/distilgpt2-copy-head-factorial-v1-20261028/run/result.json",
    "results/distilgpt2-copy-head-factorial-v1-20261028/run/attempt.jsonl",
    "results/distilgpt2-copy-head-factorial-v1-20261028/supervisor.json",
)
MANIFEST_SHA256 = "3fe5f868917b89912b96200ce01029812192c9703c5c39244d6b78485b5605bb"
RESULT_SHA256 = "7f62ed7a014978f71c09ed2e5973d1ce38f544eff0f2a1652cc9fd75e2fb125b"
ATTEMPT_SHA256 = "4e56988a89dddc9aeddcc67c69957e473402140a7a4ed6244cdaa75b5776adfa"
SUPERVISOR_SHA256 = "35626526739498d468c9e91d1c4f39e3f79603cd9015fc7fbd386ba3917f086d"


def inventory_payload():
    predecessor = v12.inventory_payload()
    canonical = json.dumps(predecessor, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    if (v12.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(canonical).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v12.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_public_copying_factorial_rows": 32}
    matched = []
    old_ids = {row["record_id"] for row in payload["records"]}
    for row in payload["records"]:
        if row["record_id"].startswith("planned-copy-factorial-"):
            if row["category"] != "planned":
                raise ValueError("factorial reservation was not planned in the predecessor")
            matched.append((row["record_id"], tuple(row["token_ids"]), row["target_token_id"]))
            new_id = row["record_id"].replace("planned-", "observed-", 1)
            if new_id in old_ids:
                raise ValueError("observed factorial record ID already exists")
            row["record_id"] = new_id
            row["category"] = "observed-saved"
            row["provenance"] = (
                "Public non-blind CPU copying-head factorial, 32 distinct rows and seven actions per row. "
                f"Case inventory SHA-256 {v12.CASES_SHA256}. "
                "Saved manifest, result, attempt log and process observation SHA-256 "
                f"{MANIFEST_SHA256}, {RESULT_SHA256}, {ATTEMPT_SHA256}, {SUPERVISOR_SHA256}."
            )
            row["references"] = list(EVIDENCE_PATHS)
            row["reference_commit"] = EVIDENCE_COMMIT
    expected = [(f"planned-copy-factorial-{i // 4 + 1:02d}-{i % 4}", tokens,
                 FACTORIAL_GROUPS[i // 4][5]) for i, tokens in enumerate(FACTORIAL_ROWS)]
    if len(expected) != 32 or matched != expected:
        raise ValueError("predecessor factorial rows do not match the completed public run")
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
