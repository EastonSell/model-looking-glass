"""Reconcile the completed public screened baseline prompts."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v22 as v22

INVENTORY_VERSION = "distilgpt2-development-exposures-v23"
PREDECESSOR_SHA256 = "d44e8cd2db2a1815335dfa48e26b5a5b05e44938579c6ce4e2989afa0e6d822e"
INVENTORY_SHA256 = "2549f1430ada47da088d2a909f8aa57ad5c08bf7c6d09f187bd34bb76b99f658"
FORBIDDEN_PREFIX_COUNT = 3888
TARGET = v22.TARGET
PUBLIC_DISCOVERY = v22.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v22.FACTORIAL_GROUPS
FACTORIAL_ROWS = v22.FACTORIAL_ROWS
RECIPIENT_CASES = v22.RECIPIENT_CASES
INJECTION_RECORD_IDS = v22.INJECTION_RECORD_IDS
PAIRS_DOSE_GROUPS = v22.PAIRS_DOSE_GROUPS
PAIRS_DOSE_ROWS = v22.PAIRS_DOSE_ROWS
YEAR_TRANSFER_CASES = v22.YEAR_TRANSFER_CASES
YEAR_TRANSFER_ROWS = v22.YEAR_TRANSFER_ROWS
GROUPED_REPORTED_ROWS = v22.GROUPED_REPORTED_ROWS
GROUPED_POLICY_ROWS = v22.GROUPED_POLICY_ROWS
GROUPED_BROKER_OBSERVED_ROWS = v22.GROUPED_BROKER_OBSERVED_ROWS
SCREENED_CASES = v22.SCREENED_CASES
SCREENED_ROWS = v22.SCREENED_ROWS
AGREEMENT_CASES = v22.AGREEMENT_CASES
AGREEMENT_ROWS = v22.AGREEMENT_ROWS
AGREEMENT_ANSWER_IDS = v22.AGREEMENT_ANSWER_IDS
EXPLICIT_YEAR_DIRECTION_CASES = v22.EXPLICIT_YEAR_DIRECTION_CASES
EXPLICIT_YEAR_DIRECTION_ROWS = v22.EXPLICIT_YEAR_DIRECTION_ROWS
SCREENED_OBSERVED_CASES = tuple(case for case in SCREENED_CASES if case[0].startswith("screen-target-"))
SCREENED_PLANNED_LOCALITY_CASES = tuple(case for case in SCREENED_CASES if case[0].startswith("screen-locality-"))

SOURCE_COMMIT = "8f2dc4dcb8a2bcd10531ecaffbf893f794fe2e26"
EVIDENCE_COMMIT = "fc6496e89b9130805f484555378f29b74dd7ba76"
CASE_SHA256 = "fc3f00a0b4f7884a3c00011186420d16322009edf75b6b09e4296953dad198e3"
EVIDENCE_PATHS = (
    "experiments/targeted_edit_screened_public_cases_v1.json",
    "experiments/targeted_edit_screened_screen_run_v1.json",
    "results/targeted-edit-screened-matched-v1-20261029/screen/phase-record.json",
    "results/targeted-edit-screened-matched-v1-20261029/screen/child-result.json",
    "results/targeted-edit-screened-matched-v1-20261029/screen/screen/observations.jsonl",
    "results/targeted-edit-screened-matched-v1-20261029/screen/screen/screen-selection.json",
    "results/targeted-edit-screened-matched-v1-20261029/screen/screen/screen.f32",
)
EVIDENCE_SHA256S = (
    CASE_SHA256,
    "1de77fa0078b71d75606d39001280267adb6ae5d3d3558fdea36283995e9b653",
    "c8e4c87377d57ada69d50b717524fd082a8b5d464b66baf39bc2af5a1e9c53b5",
    "cc9d26c1fc26e23d6506c536d401111c43202f9a10afd1c91caf350dacebc286",
    "48456dedec8359e402e51abe5c52c5942b5728a4cf83574516a3283af905d0fb",
    "9ea861d89c8ed78c038947dbdb331d6dba1ea8267901ce06e5492dab32e7c11f",
    "252c7e7c377d12bac7b0263f67e42d8127a3ed3f87e0a76faca334a7ae570132",
)
_canonical = v22._canonical
_prefixes = v22._prefixes


def inventory_payload():
    predecessor = v22.inventory_payload()
    if (v22.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v22.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_screened_target_rows": 32}
    if len(SCREENED_OBSERVED_CASES) != 32 or len(SCREENED_PLANNED_LOCALITY_CASES) != 12:
        raise ValueError("screened target or locality case count differs")
    expected = {"planned-" + case_id: (text, row) for case_id, text, row in SCREENED_OBSERVED_CASES}
    locality = {"planned-" + case_id: (text, row) for case_id, text, row in SCREENED_PLANNED_LOCALITY_CASES}
    found, locality_found = set(), set()
    existing_ids = {record["record_id"] for record in payload["records"]}
    for record in payload["records"]:
        record_id = record["record_id"]
        if record_id in locality:
            if (record_id in locality_found or record.get("category") != "planned"
                    or (record.get("public_text"), tuple(record["token_ids"])) != locality[record_id]):
                raise ValueError("screened locality reservation differs")
            locality_found.add(record_id)
        if not record_id.startswith("planned-screen-target-"):
            continue
        if (record_id not in expected or record_id in found or record.get("category") != "planned"
                or (record.get("public_text"), tuple(record["token_ids"])) != expected[record_id]):
            raise ValueError("screened target reservation differs")
        found.add(record_id)
        observed_id = record_id.replace("planned-", "observed-", 1)
        if observed_id in existing_ids:
            raise ValueError("observed record ID already exists")
        record["record_id"] = observed_id
        record["category"] = "observed-saved"
        record["provenance"] = (
            "Public non-blind screened baseline on the pinned CPU model; the saved journal reports "
            "one completed baseline result for each of 32 target prompts. "
            "The 12 locality prompts remain planned. Process records are cooperative observations, "
            "not authenticated execution. Evidence SHA-256: " + ", ".join(EVIDENCE_SHA256S) + "."
        )
        record["references"] = list(EVIDENCE_PATHS)
        record["reference_commit"] = EVIDENCE_COMMIT
    if found != set(expected) or locality_found != set(locality):
        raise ValueError("screened public case inventory is incomplete")
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
