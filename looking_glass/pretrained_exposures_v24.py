"""Reconcile the completed public explicit-year inputs."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v23 as v23

INVENTORY_VERSION = "distilgpt2-development-exposures-v24"
PREDECESSOR_SHA256 = "2549f1430ada47da088d2a909f8aa57ad5c08bf7c6d09f187bd34bb76b99f658"
INVENTORY_SHA256 = "877fa0376925be34ac3de14ff066e5465a67953660bce81011e9c8fd87810bd3"
FORBIDDEN_PREFIX_COUNT = 3888
TARGET = v23.TARGET
PUBLIC_DISCOVERY = v23.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v23.FACTORIAL_GROUPS
FACTORIAL_ROWS = v23.FACTORIAL_ROWS
RECIPIENT_CASES = v23.RECIPIENT_CASES
INJECTION_RECORD_IDS = v23.INJECTION_RECORD_IDS
PAIRS_DOSE_GROUPS = v23.PAIRS_DOSE_GROUPS
PAIRS_DOSE_ROWS = v23.PAIRS_DOSE_ROWS
YEAR_TRANSFER_CASES = v23.YEAR_TRANSFER_CASES
YEAR_TRANSFER_ROWS = v23.YEAR_TRANSFER_ROWS
GROUPED_REPORTED_ROWS = v23.GROUPED_REPORTED_ROWS
GROUPED_POLICY_ROWS = v23.GROUPED_POLICY_ROWS
GROUPED_BROKER_OBSERVED_ROWS = v23.GROUPED_BROKER_OBSERVED_ROWS
SCREENED_CASES = v23.SCREENED_CASES
SCREENED_ROWS = v23.SCREENED_ROWS
SCREENED_OBSERVED_CASES = v23.SCREENED_OBSERVED_CASES
SCREENED_PLANNED_LOCALITY_CASES = v23.SCREENED_PLANNED_LOCALITY_CASES
AGREEMENT_CASES = v23.AGREEMENT_CASES
AGREEMENT_ROWS = v23.AGREEMENT_ROWS
AGREEMENT_ANSWER_IDS = v23.AGREEMENT_ANSWER_IDS
EXPLICIT_YEAR_DIRECTION_CASES = v23.EXPLICIT_YEAR_DIRECTION_CASES
EXPLICIT_YEAR_DIRECTION_ROWS = v23.EXPLICIT_YEAR_DIRECTION_ROWS
CANDIDATE_GROUP_KEYS = ("above_ids", "middle_ids", "local_above_ids", "local_below_ids")

SOURCE_COMMIT = "4059fa97d78033b1f382c09b4b9b93d8fb4885bd"
EVIDENCE_COMMIT = "10ddb8961d1d88b65cd3a4b9625f91eb82c5a65f"
CASE_SHA256 = "419d74c23333407f9a9dc0b6f7d7b70764ab5d1078e2b0a4c84d37ba194df8a9"
RAW_DIRECTORY = "results/distilgpt2-explicit-year-direction-v1-20261029/run/"
EVIDENCE_PATHS = (
    "experiments/distilgpt2_explicit_year_direction_v1.json",
    RAW_DIRECTORY + "manifest.json",
    RAW_DIRECTORY + "attempt.jsonl",
    RAW_DIRECTORY + "result.json",
    "results/distilgpt2-explicit-year-direction-v1-20261029/supervisor.json",
)
EVIDENCE_SHA256S = (
    CASE_SHA256,
    "4bd0e0f9d781bd7ab048320f7275832be9c386a4182aecdb3508b03ad83a1035",
    "2c0a41848aaadf62f34bdc2acbe27a4119dd5452f8edf6c0e16594d2f488d1d5",
    "d6ae653ca2e96864e8332bc6ad9b6fbd0c5b546570971fd4b274172ac8089bc3",
    "a792c872f2a5e6776daea6bf780722b023e94387d305d75e4fe2e90e42f1a926",
)
_canonical = v23._canonical
_prefixes = v23._prefixes


def inventory_payload():
    predecessor = v23.inventory_payload()
    if (v23.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v23.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_explicit_year_direction_rows": 48}
    if len(EXPLICIT_YEAR_DIRECTION_CASES) != 48:
        raise ValueError("explicit-year case count differs")
    expected = {"planned-" + case_id: (text, row, groups)
                for case_id, text, row, groups in EXPLICIT_YEAR_DIRECTION_CASES}
    found = set()
    existing_ids = {record["record_id"] for record in payload["records"]}
    for record in payload["records"]:
        record_id = record["record_id"]
        if not record_id.startswith("planned-year-direction-"):
            continue
        if (record_id not in expected or record_id in found or record.get("category") != "planned"
                or (record.get("public_text"), tuple(record["token_ids"])) != expected[record_id][:2]
                or record.get("candidate_token_groups") !=
                   {name: list(group) for name, group in zip(CANDIDATE_GROUP_KEYS, expected[record_id][2])}):
            raise ValueError("explicit-year input reservation differs")
        found.add(record_id)
        observed_id = record_id.replace("planned-", "observed-", 1)
        if observed_id in existing_ids:
            raise ValueError("observed record ID already exists")
        record["record_id"] = observed_id
        record["category"] = "observed-saved"
        record["provenance"] = (
            "Public non-blind explicit-year CPU run; the saved journal reports 336 completed actions "
            "across these 48 prompts. Direct-process and model reports are cooperative observations, "
            "not authenticated execution or descendant-process attestation. "
            "Evidence SHA-256: " + ", ".join(EVIDENCE_SHA256S) + "."
        )
        record["references"] = list(EVIDENCE_PATHS)
        record["reference_commit"] = EVIDENCE_COMMIT
    if found != set(expected):
        raise ValueError("explicit-year public case inventory is incomplete")
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
