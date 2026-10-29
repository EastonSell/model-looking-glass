"""Reserve the public projected-edit cases in the v9 exposure inventory.

This module records known public rows only. Its inventory does not certify
prior exposure, pretraining, private held-out disjointness, or execution.
"""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v9 as v9


INVENTORY_VERSION = "distilgpt2-development-exposures-v10"
PREDECESSOR_SHA256 = "e0c394786b316ea61b9f3697063eb86c7cf290d898c3ed84549880df8af91c4d"
INVENTORY_SHA256 = "390550772b895b7ec2b1945f2df7683b3eb23ff831de5df2f55f8bb540f87588"
TARGET = v9.TARGET
PUBLIC_DISCOVERY = v9.PUBLIC_DISCOVERY
CASES_COMMIT = "66e08b2970594cd08cf5131f4c4b13900593b23d"
CASES_PATH = "experiments/targeted_edit_projected_public_cases_v1.json"
CASES_SHA256 = "f7554a81ac7143c54a56d557fb04cb19419bac30e7dcd44c951eb63ddaff8ee8"
PROTOCOL_PATH = "docs/TARGETED_EDIT_PROJECTED_CPU_V1.md"
PROTOCOL_SHA256 = "4bf403abba087c52d21e6f0cc0cd01b7f36a9713eb7da446dfea113ba363a6d8"

# Case ID, split, public prompt, complete prompt token row, expected token.
PROJECTED_CASES = (('projected-target-1',
  'target',
  'The capital of the French Republic is',
  (464, 3139, 286, 262, 4141, 2066, 318),
  3576),
 ('projected-target-2',
  'target',
  "France's seat of government is located in",
  (28572, 338, 5852, 286, 1230, 318, 5140, 287),
  3576),
 ('projected-target-3',
  'target',
  'The French government is based in',
  (464, 4141, 1230, 318, 1912, 287),
  3576),
 ('projected-target-4',
  'target',
  'France has its national capital at',
  (28572, 468, 663, 2260, 3139, 379),
  3576),
 ('projected-locality-1',
  'locality',
  'The capital of Austria is',
  (464, 3139, 286, 17322, 318),
  23825),
 ('projected-locality-2',
  'locality',
  'The capital of Greece is',
  (464, 3139, 286, 10315, 318),
  21891),
 ('projected-locality-3',
  'locality',
  'The capital of Norway is',
  (464, 3139, 286, 15238, 318),
  39636),
 ('projected-locality-4',
  'locality',
  'The capital of Ireland is',
  (464, 3139, 286, 7517, 318),
  18220),
 ('projected-locality-5',
  'locality',
  'Water freezes at zero degrees',
  (19184, 44389, 379, 6632, 7370),
  34186),
 ('projected-locality-6',
  'locality',
  'The color of a ripe lemon is',
  (464, 3124, 286, 257, 29036, 18873, 318),
  7872),
 ('projected-locality-7', 'locality', 'A year has twelve', (32, 614, 468, 14104), 1933),
 ('projected-locality-8',
  'locality',
  'The opposite of empty is',
  (464, 6697, 286, 6565, 318),
  1336),
 ('projected-locality-9',
  'locality',
  'The currency used in France is the',
  (464, 7395, 973, 287, 4881, 318, 262),
  11063))


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def inventory_payload():
    if v9.inventory_digest() != PREDECESSOR_SHA256:
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(v9.inventory_payload())
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v9.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_public_projected_edit_rows": 13}
    if (len(PROJECTED_CASES) != 13
            or sum(case[1] == "target" for case in PROJECTED_CASES) != 4
            or sum(case[1] == "locality" for case in PROJECTED_CASES) != 9):
        raise ValueError("projected edit cases do not match the fixed split")
    old_rows = [tuple(row["token_ids"]) for row in payload["records"]]
    rows = [case[3] for case in PROJECTED_CASES]
    if any(_related(left, right) for left in rows for right in old_rows):
        raise ValueError("projected edit row has a full-row prefix relation with a predecessor")
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("projected edit rows contain a full-row prefix relation")
    old_ids = {row["record_id"] for row in payload["records"]}
    new_ids = [f"planned-{case[0]}" for case in PROJECTED_CASES]
    if len(set(new_ids)) != len(new_ids) or any(record_id in old_ids for record_id in new_ids):
        raise ValueError("projected edit record ID is duplicated")
    for (case_id, split, prompt, tokens, expected_token), record_id in zip(PROJECTED_CASES, new_ids):
        payload["records"].append({
            "record_id": record_id,
            "category": "planned",
            "token_ids": list(tokens),
            "target_token_id": expected_token,
            "provenance": (
                f"Public non-blind projected-edit {split} case {case_id}. "
                f"Case record SHA-256 {CASES_SHA256}. "
                "No projected-edit model forward is asserted by this reservation."
            ),
            "references": [PROTOCOL_PATH, CASES_PATH],
            "public_text": prompt,
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
    # A candidate may be a recorded prefix or extend a complete known row.
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
