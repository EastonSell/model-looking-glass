"""Record the completed public copying follow-up without adding cases."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v9 as v9
from . import pretrained_exposures_v10 as v10


INVENTORY_VERSION = "distilgpt2-development-exposures-v11"
PREDECESSOR_SHA256 = "390550772b895b7ec2b1945f2df7683b3eb23ff831de5df2f55f8bb540f87588"
INVENTORY_SHA256 = "84b1a95a41670798ff55c9f08272890c00a939a74a4cad7ad9b95fbf4bcdaa25"
TARGET = v10.TARGET
PUBLIC_DISCOVERY = v10.PUBLIC_DISCOVERY
FOLLOWUP_GROUPS = v9.FOLLOWUP_GROUPS
FOLLOWUP_ROWS = v9.FOLLOWUP_ROWS
EVIDENCE_COMMIT = "0712ec0bdb9b1b9490bee48919f86d11dbf10869"
EVIDENCE_PATHS = (
    "results/distilgpt2-copy-head-followup-v2-20261028/run/manifest.json",
    "results/distilgpt2-copy-head-followup-v2-20261028/run/result.json",
    "results/distilgpt2-copy-head-followup-v2-20261028/run/attempt.jsonl",
    "results/distilgpt2-copy-head-followup-v2-20261028/supervisor.json",
)
MANIFEST_SHA256 = "39758a7d1670e31d0e159a0bf738a9094646efe8b638ffdee6519f28d50385fb"
RESULT_SHA256 = "4a6c9251a5752714525836df2c29c9e1a8c8659430720a79da47e5f4f3fc18f1"
ATTEMPT_SHA256 = "fa28d4921fcf91734c451d2bb3a5e31e27310fd3058d694389f436444398912b"
SUPERVISOR_SHA256 = "2ffc6adc91d51b3b6b91db587ad011d11eb72efc8d1eb614d0868fc1cff38ea9"


def inventory_payload():
    if v10.inventory_digest() != PREDECESSOR_SHA256:
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(v10.inventory_payload())
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v10.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_public_copying_followup_rows": 32}
    matched = []
    old_ids = {row["record_id"] for row in payload["records"]}
    for row in payload["records"]:
        if row["record_id"].startswith("planned-copy-followup-"):
            if row["category"] != "planned":
                raise ValueError("copying follow-up reservation was not planned in the predecessor")
            matched.append((row["record_id"], tuple(row["token_ids"]), row["target_token_id"]))
            new_id = row["record_id"].replace("planned-", "observed-", 1)
            if new_id in old_ids:
                raise ValueError("observed copying follow-up record ID already exists")
            row["record_id"] = new_id
            row["category"] = "observed-saved"
            row["provenance"] = (
                "Public non-blind CPU copying-head follow-up, 32 distinct rows and nine actions per row. "
                f"Case inventory SHA-256 {v9.FOLLOWUP_CASES_SHA256}. "
                "Saved manifest, result, attempt log and process observation SHA-256 "
                f"{MANIFEST_SHA256}, {RESULT_SHA256}, {ATTEMPT_SHA256}, {SUPERVISOR_SHA256}."
            )
            row["references"] = list(EVIDENCE_PATHS)
            row["reference_commit"] = EVIDENCE_COMMIT
    expected = [(f"planned-copy-followup-{i // 4 + 1:02d}-{i % 4}", tokens,
                 FOLLOWUP_GROUPS[i // 4][5]) for i, tokens in enumerate(FOLLOWUP_ROWS)]
    if len(expected) != 32 or matched != expected:
        raise ValueError("predecessor copying follow-up rows do not match the completed public run")
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
