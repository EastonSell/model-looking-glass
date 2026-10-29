"""Record the completed public copying-head evaluation without adding cases."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v7 as v7


INVENTORY_VERSION = "distilgpt2-development-exposures-v8"
PREDECESSOR_SHA256 = "e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344"
INVENTORY_SHA256 = "22b1392ed43568fcac574b55ae6cf0212766771d8439ac39e599f7aa112b298f"
TARGET = v7.TARGET
PUBLIC_DISCOVERY = v7.PUBLIC_DISCOVERY
PUBLIC_EVALUATION_ROWS = v7.PUBLIC_EVALUATION_ROWS
EVIDENCE_COMMIT = "ebe5caaae8394d210265ab65ab84c295b6463580"
EVIDENCE_PATHS = (
    "results/distilgpt2-copy-head-eval-v1-20261028/run/manifest.json",
    "results/distilgpt2-copy-head-eval-v1-20261028/run/result.json",
    "results/distilgpt2-copy-head-eval-v1-20261028/run/attempt.jsonl",
    "results/distilgpt2-copy-head-eval-v1-20261028/supervisor.json",
)
MANIFEST_SHA256 = "0398bd998fae4ad8ef473264fca4c8b5dcf2c63424e086e4f8c989a2f7cb819b"
RESULT_SHA256 = "e36732e0538f7b5a069fed337fd86eb725744663b3801d1d953907d0edb93a02"
ATTEMPT_SHA256 = "88026a17a65ddf7d303479bceba120517b456b90da3fefd0b5c6dd1090d26766"
SUPERVISOR_SHA256 = "2935b0dfbecf10a8171cd62d05a0856f86994c7cce975d9090022a2d74610c9a"


def inventory_payload():
    if v7.inventory_digest() != PREDECESSOR_SHA256:
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(v7.inventory_payload())
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v7.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_public_copying_evaluation_rows": 32}
    matched = []
    for row in payload["records"]:
        if row["record_id"].startswith("planned-copy-evaluation-"):
            if row["category"] != "planned":
                raise ValueError("copying evaluation reservation was not planned in the predecessor")
            matched.append(tuple(row["token_ids"]))
            row["record_id"] = row["record_id"].replace("planned-", "observed-", 1)
            row["category"] = "observed-saved"
            row["provenance"] = (
                "Public non-blind CPU copying-head evaluation, 32 distinct rows and five actions per row. "
                "Saved manifest, result, attempt log and process observation SHA-256 "
                f"{MANIFEST_SHA256}, {RESULT_SHA256}, {ATTEMPT_SHA256}, {SUPERVISOR_SHA256}."
            )
            row["references"] = list(EVIDENCE_PATHS)
            row["reference_commit"] = EVIDENCE_COMMIT
    if matched != list(PUBLIC_EVALUATION_ROWS):
        raise ValueError("predecessor copying evaluation rows do not match the completed public evaluation")
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
