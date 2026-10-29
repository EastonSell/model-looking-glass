"""Add an operator-reported grouped CPU exposure to the known inventory."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v17 as v17

INVENTORY_VERSION = "distilgpt2-development-exposures-v18"
PREDECESSOR_SHA256 = "e5734470ccd3b4e2655c91608caf8ffa158fa8650ff0bdd7f5aeaa1028046f3a"
INVENTORY_SHA256 = "3fa3eda9508ca314505aa658b1c02d04dbc98681bfc776337cd9ca1c9c384f5a"
FORBIDDEN_PREFIX_COUNT = 3275
TARGET = v17.TARGET
PUBLIC_DISCOVERY = v17.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v17.FACTORIAL_GROUPS
FACTORIAL_ROWS = v17.FACTORIAL_ROWS
RECIPIENT_CASES = v17.RECIPIENT_CASES
INJECTION_RECORD_IDS = v17.INJECTION_RECORD_IDS
PAIRS_DOSE_GROUPS = v17.PAIRS_DOSE_GROUPS
PAIRS_DOSE_ROWS = v17.PAIRS_DOSE_ROWS
YEAR_TRANSFER_CASES = v17.YEAR_TRANSFER_CASES
YEAR_TRANSFER_ROWS = v17.YEAR_TRANSFER_ROWS

RETROSPECTIVE_PATH = "results/distilgpt2-grouped-unreserved-cpu-v1/retrospective.json"
RETROSPECTIVE_SHA256 = "e087d03ae0cc2c3d0c0c71a241b4dd583fedea4f609364c96ad1ec7c8412e5e4"
YEAR_EVIDENCE_COMMIT = "01e632d7b535cf006e00fb1a659c6e136d338c69"
YEAR_CASE_SHA256 = "d6177125f8ac835676eebe59c5f65e379cb20d1d782eeeb17edf2e8700ac23d5"
YEAR_EVIDENCE_PATHS = (
    "results/distilgpt2-year-transfer-v1-20261029/run/manifest.json",
    "results/distilgpt2-year-transfer-v1-20261029/run/result.json",
    "results/distilgpt2-year-transfer-v1-20261029/run/attempt.jsonl",
    "results/distilgpt2-year-transfer-v1-20261029/supervisor.json",
)
YEAR_EVIDENCE_SHA256S = (
    "b412d7a29f573e43e62b810b2bd63a9d92f1f2c698a2beb52bb8347e980cbb1d",
    "fad27de70b301e7171c6fc55c8b600e79f9d7a59c6d4f8feb1ee66362d5e926f",
    "9e94fc4c30f563c1679e0fafd577133127ac413b72aa318064c2dd8987292c2b",
    "1bd457a772a4985bf223624ea5248e8343771bde97495b240d9717748d997a42",
)
GROUPED_REPORTED_ROWS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
    (21, 22, 23, 24),
    (25, 26, 27, 28),
    (29, 30, 31, 32),
)
_canonical = v17._canonical
_prefixes = v17._prefixes
_has_known_relation = v17._has_known_relation


def inventory_payload():
    predecessor = v17.inventory_payload()
    if (v17.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v17.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"operator_reported_grouped_cpu_rows": 8,
                         "observed_year_transfer_rows": 48}
    expected_year = {"planned-" + case_id: (text, tokens)
                     for case_id, text, tokens in YEAR_TRANSFER_CASES}
    if len(expected_year) != 48 or not re.fullmatch(r"[0-9a-f]{40}", YEAR_EVIDENCE_COMMIT):
        raise ValueError("the year-transfer evidence identity is not fixed")
    prior_ids = {r["record_id"] for r in payload["records"]}
    found = set()
    for record in payload["records"]:
        record_id = record["record_id"]
        if not record_id.startswith("planned-year-transfer-"):
            continue
        if (record_id not in expected_year or record_id in found or record.get("category") != "planned"
                or (record.get("public_text"), tuple(record["token_ids"])) != expected_year[record_id]):
            raise ValueError("the predecessor year-transfer reservation differs")
        observed_id = record_id.replace("planned-", "observed-", 1)
        if observed_id in prior_ids:
            raise ValueError("observed year-transfer record ID already exists")
        found.add(record_id)
        record["record_id"] = observed_id
        record["category"] = "observed-saved"
        record["provenance"] = ("Public non-blind CPU year-transfer run; 48 rows and seven recorded actions "
                                "per row. Saved evidence SHA-256 values: " + ", ".join(YEAR_EVIDENCE_SHA256S) + ".")
        record["references"] = list(YEAR_EVIDENCE_PATHS)
        record["reference_commit"] = YEAR_EVIDENCE_COMMIT
    if found != set(expected_year):
        raise ValueError("the completed year-transfer case inventory is incomplete")
    rows = GROUPED_REPORTED_ROWS
    if (type(rows) is not tuple or len(rows) != 8 or len(set(rows)) != 8
            or any(type(row) is not tuple or len(row) != 4
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row)
                   for row in rows)):
        raise ValueError("the operator-reported rows are invalid")
    prior = [r for r in payload["records"] if r["record_id"] == "synthetic-gpt2-control-1"]
    if (len(prior) != 1 or tuple(prior[0]["token_ids"]) != rows[0]
            or prior[0]["category"] != "synthetic-only"):
        raise ValueError("the existing synthetic-only record does not match the first reported row")
    prior_ids = {r["record_id"] for r in payload["records"]}
    for i, row in enumerate(rows, 1):
        record_id = f"observed-grouped-unreserved-{i:02d}"
        if record_id in prior_ids:
            raise ValueError("operator-reported exposure record ID already exists")
        payload["records"].append({
            "record_id": record_id, "category": "observed-reported", "token_ids": list(row),
            "public_text": "",
            "provenance": "Operator-reported pinned-model CPU grouped test; execution source bytes, "
                          "per-forward logs and logits were not retained. Retrospective SHA-256 "
                          f"{RETROSPECTIVE_SHA256}.",
            "references": [RETROSPECTIVE_PATH], "reference_commit": "",
        })
    if len(_prefixes(payload["records"])) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("operator-reported exposure prefix count changed")
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
