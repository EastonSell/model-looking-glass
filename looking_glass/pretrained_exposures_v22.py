"""Reconcile four public grouped-pilot inputs with saved evidence."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v21 as v21

INVENTORY_VERSION = "distilgpt2-development-exposures-v22"
PREDECESSOR_SHA256 = "a391c5f3a8ca33ad32e9378417b424338b12ea55257979977d2afd5a0c7f90ae"
INVENTORY_SHA256 = "d44e8cd2db2a1815335dfa48e26b5a5b05e44938579c6ce4e2989afa0e6d822e"
FORBIDDEN_PREFIX_COUNT = 3888
TARGET = v21.TARGET
PUBLIC_DISCOVERY = v21.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v21.FACTORIAL_GROUPS
FACTORIAL_ROWS = v21.FACTORIAL_ROWS
RECIPIENT_CASES = v21.RECIPIENT_CASES
INJECTION_RECORD_IDS = v21.INJECTION_RECORD_IDS
PAIRS_DOSE_GROUPS = v21.PAIRS_DOSE_GROUPS
PAIRS_DOSE_ROWS = v21.PAIRS_DOSE_ROWS
YEAR_TRANSFER_CASES = v21.YEAR_TRANSFER_CASES
YEAR_TRANSFER_ROWS = v21.YEAR_TRANSFER_ROWS
GROUPED_REPORTED_ROWS = v21.GROUPED_REPORTED_ROWS
SCREENED_CASES = v21.SCREENED_CASES
SCREENED_ROWS = v21.SCREENED_ROWS
AGREEMENT_CASES = v21.AGREEMENT_CASES
AGREEMENT_ROWS = v21.AGREEMENT_ROWS
AGREEMENT_ANSWER_IDS = v21.AGREEMENT_ANSWER_IDS
EXPLICIT_YEAR_DIRECTION_CASES = v21.EXPLICIT_YEAR_DIRECTION_CASES
EXPLICIT_YEAR_DIRECTION_ROWS = v21.EXPLICIT_YEAR_DIRECTION_ROWS

SOURCE_COMMIT = "05ca62b99aa3930eaa91654b2d312024885e20e7"
PROTOCOL_PATH = "experiments/grouped_gpu_dev_v3.json"
PROTOCOL_SHA256 = "011e214dbdc57e29869f41e38082cd1fa4baaf15747169d0d778e6f27b33a6f1"
EVIDENCE_COMMIT = "729764e0a310a42a28a05ef7a8eb72bff08d4002"
EVIDENCE_PATHS = (
    "results/grouped-gpu-resource-v3-20261029/protocol-source.json",
    "results/grouped-gpu-resource-v3-20261029/result-public.json",
    "results/grouped-gpu-resource-v3-20261029/operational-review.json",
    "results/grouped-gpu-resource-v3-20261029/public-replay.json",
    "results/grouped-gpu-resource-v3-20261029/source-audit-05ca.json",
)
EVIDENCE_SHA256S = (
    "15d5d71d8bac12dbbffafca0a9afe23c42a1df4e8e8281026b91711e5b107c2e",
    "1872485fc7007a5c3c748043749e169037594d07991e1d4f37816b74868c1fcf",
    "41626cbd1d22ecc81834d7d22288aff48707bf25f49ea0bba35292b249cde728",
    "56127d3324ed5f1c85ec630746cd315a0e499a7df6ad03b5d68f4c15541a86da",
    "2da7e34902325487e24db277b0bd85746c2fe65485956b814a5e7729c06f727c",
)
ORIGINAL_RESULT_SHA256 = "218e5e7144667f309e82532699c6bac05ecb31da6616c5319da099dd64eefbba"
GROUPED_POLICY_ROWS = tuple((50256, i, i + 100, i + 200) for i in range(103, 107))
GROUPED_BROKER_OBSERVED_ROWS = tuple((50256, i, i + 100, i + 200) for i in range(107, 111))
_canonical = v21._canonical
_prefixes = v21._prefixes


def inventory_payload():
    predecessor = v21.inventory_payload()
    if (v21.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v21.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_grouped_broker_rows": 4}
    expected = {f"planned-grouped-broker-check-{i:02d}": row
                for i, row in enumerate(GROUPED_BROKER_OBSERVED_ROWS, 1)}
    policy = {f"observed-batch-{i}": row for i, row in zip(range(103, 107), GROUPED_POLICY_ROWS)}
    found, policy_found = set(), set()
    existing_ids = {record["record_id"] for record in payload["records"]}
    for record in payload["records"]:
        record_id = record["record_id"]
        if record_id in policy:
            if (record_id in policy_found or record.get("category") != "observed-saved"
                    or tuple(record["token_ids"]) != policy[record_id]):
                raise ValueError("prior policy input record differs")
            policy_found.add(record_id)
        if not record_id.startswith("planned-grouped-broker-check-"):
            continue
        if (record_id not in expected or record_id in found or record.get("category") != "planned"
                or tuple(record["token_ids"]) != expected[record_id]):
            raise ValueError("grouped broker input reservation differs")
        found.add(record_id)
        observed_id = record_id.replace("planned-", "observed-", 1)
        if observed_id in existing_ids:
            raise ValueError("observed record ID already exists")
        record["record_id"] = observed_id
        record["category"] = "observed-saved"
        record["provenance"] = (
            "Public synthetic GPU grouped-pilot input; reported completed in the saved public projection. "
            "Excluded only from this run's policy construction, not a blind example. "
            "Raw logits and the original private terminal were withheld; evidence is a public projection "
            "and saved-record/operator observation. Evidence SHA-256: " + ", ".join(EVIDENCE_SHA256S) + "."
        )
        record["references"] = list(EVIDENCE_PATHS)
        record["reference_commit"] = EVIDENCE_COMMIT
    if found != set(expected) or policy_found != set(policy):
        raise ValueError("grouped pilot input inventory is incomplete")
    payload["additional_observations"].append({
        "kind": "public-grouped-gpu-resource-v3-policy-inputs",
        "record_ids": list(policy),
        "evidence_commit": EVIDENCE_COMMIT,
        "references": list(EVIDENCE_PATHS),
        "evidence_sha256": list(EVIDENCE_SHA256S),
        "scope": "Additional reported use of the four already observed public policy inputs; "
                 "the saved projection is not independent execution attestation.",
    })
    if _prefixes(payload["records"]) != _prefixes(predecessor["records"]):
        raise ValueError("a status-only update changed the known input prefixes")
    if len(_prefixes(payload["records"])) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("known input prefix count changed")
    return payload

def canonical_inventory_bytes():
    return _canonical(inventory_payload())


def _checked_inventory():
    payload = inventory_payload()
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError("development-exposure inventory does not match its pinned digest")
    return payload, digest


def inventory_digest():
    return _checked_inventory()[1]


def _forbidden_rows():
    return _prefixes(inventory_payload()["records"])


def _has_known_relation(row, forbidden, complete_rows):
    return row in forbidden or any(row[:len(known)] == known for known in complete_rows)


def check_known_splits(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                       expected_model_identity, cases_per_split=8):
    payload, digest = _checked_inventory()
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest:
        raise ValueError("expected exposure inventory digest does not match")
    if type(cases_per_split) is not int or cases_per_split not in (8, 16):
        raise ValueError("cases per split must be exactly 8 or 16")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _prefixes(payload["records"])
    complete_rows = {tuple(row["token_ids"]) for row in payload["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in discovery + heldout):
        raise ValueError("a candidate token sequence is in the development exclusion inventory")
    return v1.KnownSplitCheck(INVENTORY_VERSION, digest, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    payload, digest = _checked_inventory()
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest:
        raise ValueError("expected exposure inventory digest does not match")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _prefixes(payload["records"])
    complete_rows = {tuple(row["token_ids"]) for row in payload["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in heldout):
        raise ValueError("a held-out token sequence is in the development exclusion inventory")
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, digest, 32, len(forbidden),
                                    sum(_has_known_relation(row, forbidden, complete_rows) for row in discovery))


def _validate_known_check(check):
    payload, digest = _checked_inventory()
    count = len(_prefixes(payload["records"]))
    if (type(check) not in (v1.KnownSplitCheck, v2.ExposedDiscoveryCheck)
            or check.inventory_version != INVENTORY_VERSION or check.inventory_sha256 != digest
            or type(check.candidate_count) is not int or type(check.forbidden_row_count) is not int
            or check.forbidden_row_count != count or check.complete_exposure_history_certified is not False
            or check.preseal_ready is not False):
        raise ValueError("a matching known-split check is required")
    if type(check) is v1.KnownSplitCheck and check.scope == "known-rows-only" and check.candidate_count in (16, 32):
        return count, "known-prefixes-excluded-from-both", 0
    if (type(check) is v2.ExposedDiscoveryCheck and check.scope == "known-prefix-holdout-only"
            and check.candidate_count == 32 and type(check.known_or_reserved_discovery_overlap_count) is int
            and 0 <= check.known_or_reserved_discovery_overlap_count <= 16):
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
