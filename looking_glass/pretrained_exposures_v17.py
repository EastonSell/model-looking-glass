"""Reconcile two completed public studies with the known-exposure inventory."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v10 as v10
from . import pretrained_exposures_v16 as v16

INVENTORY_VERSION = "distilgpt2-development-exposures-v17"
PREDECESSOR_SHA256 = "275a16d49a386b1de247bff0f9031e92105d1b76f2493352090bf27c37d58843"
INVENTORY_SHA256 = "e5734470ccd3b4e2655c91608caf8ffa158fa8650ff0bdd7f5aeaa1028046f3a"
FORBIDDEN_PREFIX_COUNT = 3247
TARGET = v16.TARGET
PUBLIC_DISCOVERY = v16.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v16.FACTORIAL_GROUPS
FACTORIAL_ROWS = v16.FACTORIAL_ROWS
RECIPIENT_CASES = v16.RECIPIENT_CASES
INJECTION_RECORD_IDS = v16.INJECTION_RECORD_IDS
PAIRS_DOSE_GROUPS = v16.PAIRS_DOSE_GROUPS
PAIRS_DOSE_ROWS = v16.PAIRS_DOSE_ROWS
YEAR_TRANSFER_CASES = v16.YEAR_TRANSFER_CASES
YEAR_TRANSFER_ROWS = v16.YEAR_TRANSFER_ROWS

PAIRS_EVIDENCE_COMMIT = "28d7c0321cda85d1139c93b25f8605b874feae96"
PAIRS_EVIDENCE_PATHS = (
    "results/distilgpt2-copy-head-pairs-dose-v1-20261028/run/manifest.json",
    "results/distilgpt2-copy-head-pairs-dose-v1-20261028/run/result.json",
    "results/distilgpt2-copy-head-pairs-dose-v1-20261028/run/attempt.jsonl",
    "results/distilgpt2-copy-head-pairs-dose-v1-20261028/supervisor.json",
)
PAIRS_EVIDENCE_SHA256S = (
    "93ee4d22d0c4d040e50c25804cd36db60492698e601736e04282909e1dcafa35",
    "b382cf60edd79c9b9eccbd992dd1fffe33d11a3c202728a7a44127ef256946cf",
    "2e05be2c6f5cba20f8e62d3f0e03d189cdc5fe4555763c5ce5733a60bbc9a1e3",
    "60423637baf78178b0e4fa23438f41e5f6310423fae4ce7437ee10e3d1508d9d",
)
PROJECTED_EVIDENCE_COMMIT = "e704cb82ed6078d76ba5ab5bd32378eff106924a"
PROJECTED_EVIDENCE_PATHS = (
    "experiments/targeted_edit_projected_run_v1.json",
    "results/targeted-edit-projected-cpu-v1-20261028/evaluation/phase-record.json",
    "results/targeted-edit-projected-cpu-v1-20261028/evaluation/evaluation/observations.jsonl",
    "results/targeted-edit-projected-cpu-v1-20261028/evaluation/child-result.json",
)
PROJECTED_EVIDENCE_SHA256S = (
    "813149fca657f015cf1c70821f4bcd7d5a154e89f84369ca56af25973aff8961",
    "3a91f9e5c1783b5e94b9dd0b18ff8059894aa84962e10315c6ef3840960d630a",
    "e84965fae4be92552af676c247ac5ceda08f2a510cfb48f26bac02d72431a439",
    "d5de75fb1f64879c21a8896d5fcf9fd415becf2010e400e67299c921c321d7ca",
)
_canonical = v16._canonical
_prefixes = v16._prefixes
_has_known_relation = v16._has_known_relation


def inventory_payload():
    predecessor = v16.inventory_payload()
    if (v16.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v16.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_pairs_dose_rows": 32, "observed_projected_edit_rows": 13}
    pair_expected = {
        f"planned-copy-pairs-dose-{index // 4 + 1:02d}-{index % 4}":
            (row, PAIRS_DOSE_GROUPS[index // 4][5])
        for index, row in enumerate(PAIRS_DOSE_ROWS)
    }
    projected_expected = {
        "planned-" + case_id: (tokens, target, text)
        for case_id, _split, text, tokens, target in v10.PROJECTED_CASES
    }
    if len(pair_expected) != 32 or len(projected_expected) != 13:
        raise ValueError("fixed completed case identities changed")
    existing_ids = {row["record_id"] for row in payload["records"]}
    pairs_found, projected_found = set(), set()
    for row in payload["records"]:
        record_id = row["record_id"]
        if record_id.startswith("planned-copy-pairs-dose-"):
            if (record_id not in pair_expected or record_id in pairs_found or row.get("category") != "planned"
                    or (tuple(row["token_ids"]), row.get("target_token_id")) != pair_expected[record_id]):
                raise ValueError("the predecessor pairs/dose reservation differs")
            pairs_found.add(record_id)
            paths, hashes, commit = PAIRS_EVIDENCE_PATHS, PAIRS_EVIDENCE_SHA256S, PAIRS_EVIDENCE_COMMIT
            provenance = "Public non-blind CPU pairs/dose run; 32 rows, 18 recorded actions per row. "
        elif record_id.startswith("planned-projected-"):
            if (record_id not in projected_expected or record_id in projected_found or row.get("category") != "planned"
                    or (tuple(row["token_ids"]), row.get("target_token_id"), row.get("public_text"))
                       != projected_expected[record_id]):
                raise ValueError("the predecessor projected reservation differs")
            projected_found.add(record_id)
            paths, hashes, commit = PROJECTED_EVIDENCE_PATHS, PROJECTED_EVIDENCE_SHA256S, PROJECTED_EVIDENCE_COMMIT
            provenance = "Public non-blind CPU projected-edit evaluation; 13 rows, six recorded conditions per row. "
        else:
            continue
        observed_id = record_id.replace("planned-", "observed-", 1)
        if observed_id in existing_ids:
            raise ValueError("observed record ID already exists")
        row["record_id"] = observed_id
        row["category"] = "observed-saved"
        row["provenance"] = provenance + "Saved evidence SHA-256 values: " + ", ".join(hashes) + "."
        row["references"] = list(paths)
        row["reference_commit"] = commit
    if pairs_found != set(pair_expected) or projected_found != set(projected_expected):
        raise ValueError("completed public case inventory is incomplete")
    if _prefixes(payload["records"]) != _prefixes(predecessor["records"]):
        raise ValueError("a status-only update changed the known input prefixes")
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
