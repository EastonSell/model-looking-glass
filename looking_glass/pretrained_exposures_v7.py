"""Record the public copying screen and reserve a public follow-on."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v6 as v6


INVENTORY_VERSION = "distilgpt2-development-exposures-v7"
PREDECESSOR_SHA256 = "76803775e6929b4a20463a50bcae839a727b1a875a0d433f7c27c9d1d69f0681"
INVENTORY_SHA256 = "e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344"
TARGET = v6.TARGET
PUBLIC_DISCOVERY = v6.PUBLIC_DISCOVERY
EVIDENCE_COMMIT = "357579d87c74a515b60f53fa92819ab010e16dc1"
EVIDENCE_PATHS = (
    "results/distilgpt2-copying-dev-v1-20261028/run/manifest.json",
    "results/distilgpt2-copying-dev-v1-20261028/run/result.json",
    "results/distilgpt2-copying-dev-v1-20261028/run/attempt.jsonl",
)
EVALUATION_PLAN_COMMIT = "947332a52aa9db197cd8624f839ed8c50c68154a"
EVALUATION_PLAN_PATHS = (
    "docs/DISTILGPT2_COPY_HEAD_EVAL_V1.md",
    "experiments/distilgpt2_copying_head_eval_v1.json",
)
EVALUATION_CASES_SHA256 = "34bbb68a5acb8615e856a20b77bc3ec93787cc82afa11c9dabb83d7c209baca9"
MANIFEST_SHA256 = "ab403154ed12f1776b993cc5d8ed45525e0406e1835af14d041bf596497f8843"
RESULT_SHA256 = "8914c014bb57e37bd662f73f9f87f1c2431ad6f04a084a2920e44b68b89f0e5d"
ATTEMPT_SHA256 = "d96e2595041702878e2bcde77d07054bbf93e698e6859a8682fdeb4123cf8bf2"
PUBLIC_EVALUATION_GROUPS = (
    (5440, 7696, 8598, 38283, 40562, 10441),
    (22206, 15554, 15291, 28774, 21657, 27223),
    (9215, 9197, 10000, 44539, 24240, 18744),
    (4676, 2214, 10047, 26241, 18002, 23612),
    (24535, 31176, 47434, 27944, 33039, 31812),
    (12498, 22101, 37252, 8824, 26428, 15027),
    (33083, 47969, 13447, 19272, 6729, 30777),
    (43272, 31733, 14335, 1660, 3869, 28520),
)
PUBLIC_EVALUATION_ROWS = tuple(
    tuple(group[i] for i in permutation + (0, 1, 2, 3, 4))
    for group in PUBLIC_EVALUATION_GROUPS for permutation in v6.PUBLIC_COPY_PERMUTATIONS
)


def inventory_payload():
    predecessor = v6.inventory_payload()
    canonical = json.dumps(predecessor, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    if (v6.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(canonical).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v6.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_public_copying_rows": 32, "planned_public_copying_evaluation_rows": 32}
    matched = []
    for row in payload["records"]:
        if row["record_id"].startswith("planned-copy-screen-"):
            if row["category"] != "planned":
                raise ValueError("copying reservation was not planned in the predecessor")
            matched.append(tuple(row["token_ids"]))
            row["record_id"] = row["record_id"].replace("planned-", "observed-", 1)
            row["category"] = "observed-saved"
            row["provenance"] = (
                "Public non-blind CPU copying screen, 32 distinct rows and one repeat. "
                f"Source {v6.PLAN_COMMIT}; manifest, result and attempt-log SHA-256 "
                f"{MANIFEST_SHA256}, {RESULT_SHA256}, {ATTEMPT_SHA256}."
            )
            row["references"] = list(EVIDENCE_PATHS)
            row["reference_commit"] = EVIDENCE_COMMIT
    if matched != list(v6.PUBLIC_COPY_ROWS):
        raise ValueError("predecessor copying rows do not match the fixed public screen")
    rows = list(PUBLIC_EVALUATION_ROWS)
    evaluation_tokens = [token for group in PUBLIC_EVALUATION_GROUPS for token in group]
    screen_tokens = {token for group in v6.PUBLIC_COPY_GROUPS for token in group}
    if len(evaluation_tokens) != 48 or len(set(evaluation_tokens)) != 48 or set(evaluation_tokens) & screen_tokens:
        raise ValueError("public copying evaluation token groups are not disjoint from the screen")
    old_full = {tuple(row["token_ids"]) for row in payload["records"]}
    old_forbidden = {tuple(row["token_ids"][:length]) for row in predecessor["records"]
                     for length in range(1, len(row["token_ids"]) + 1)}
    if len(rows) != 32 or len(set(rows)) != 32 or any(row in old_forbidden for row in rows):
        raise ValueError("public copying evaluation reservations do not match their expected split")
    if any(row[:len(old)] == old for row in rows for old in old_full):
        raise ValueError("public copying evaluation extends a previous full row")
    if any(left[:len(right)] == right or right[:len(left)] == left
           for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("public copying evaluation contains a full-row prefix relation")
    for index, tokens in enumerate(rows):
        group, variant = divmod(index, 4)
        payload["records"].append({
            "record_id": f"planned-copy-evaluation-{group + 1:02d}-{variant}",
            "category": "planned", "token_ids": list(tokens),
            "target_token_id": PUBLIC_EVALUATION_GROUPS[group][5],
            "provenance": ("Public non-blind copying follow-on evaluation and fixed controls. "
                           f"Case inventory SHA-256 {EVALUATION_CASES_SHA256}. "
                           "No model forward is asserted by this reservation."),
            "references": list(EVALUATION_PLAN_PATHS), "public_text": "",
            "reference_commit": EVALUATION_PLAN_COMMIT,
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
