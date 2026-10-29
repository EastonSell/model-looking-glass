"""Reserve the public year-transfer inputs without asserting a model run."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v15 as v15

INVENTORY_VERSION = "distilgpt2-development-exposures-v16"
PREDECESSOR_SHA256 = "6c6ed34755dd9daebdde9a2de07f65c0c5d20301656416d873f321fa2c146d34"
INVENTORY_SHA256 = "275a16d49a386b1de247bff0f9031e92105d1b76f2493352090bf27c37d58843"
FORBIDDEN_PREFIX_COUNT = 3247
TARGET = v15.TARGET
PUBLIC_DISCOVERY = v15.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v15.FACTORIAL_GROUPS
FACTORIAL_ROWS = v15.FACTORIAL_ROWS
RECIPIENT_CASES = v15.RECIPIENT_CASES
INJECTION_RECORD_IDS = v15.INJECTION_RECORD_IDS
PAIRS_DOSE_GROUPS = v15.PAIRS_DOSE_GROUPS
PAIRS_DOSE_ROWS = v15.PAIRS_DOSE_ROWS

PROTOCOL_COMMIT = "38ffcab52256e3952315b7bd9b58d6f10d7921c2"
PROTOCOL_PATH = "docs/DISTILGPT2_YEAR_TRANSFER_V1.md"
PROTOCOL_SHA256 = "6f31697b09b2b68bf701494b4ceedde185b9f8501aea61caf50b647edac1fee7"
CASES_COMMIT = "421f008a03f94da27e737d08040413265c19cbdd"
CASE_PATH = "experiments/distilgpt2_year_transfer_v1.json"
CASE_SHA256 = "d6177125f8ac835676eebe59c5f65e379cb20d1d782eeeb17edf2e8700ac23d5"

YEAR_TRANSFER_CASES = (('year-transfer-01-war-forward-low',
  'The war lasted from the year 1705 to the year 17',
  (464, 1175, 15436, 422, 262, 614, 1596, 2713, 284, 262, 614, 1596)),
 ('year-transfer-01-war-forward-high',
  'The war lasted from the year 1714 to the year 17',
  (464, 1175, 15436, 422, 262, 614, 1596, 1415, 284, 262, 614, 1596)),
 ('year-transfer-01-war-reverse-low',
  'The war lasted until the year 1705, after starting in the year 17',
  (464, 1175, 15436, 1566, 262, 614, 1596, 2713, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-01-war-reverse-high',
  'The war lasted until the year 1714, after starting in the year 17',
  (464, 1175, 15436, 1566, 262, 614, 1596, 1415, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-01-war-neutral-low',
  'The war was recorded in the year 1705 beside the year 17',
  (464, 1175, 373, 6264, 287, 262, 614, 1596, 2713, 13970, 262, 614, 1596)),
 ('year-transfer-01-war-neutral-high',
  'The war was recorded in the year 1714 beside the year 17',
  (464, 1175, 373, 6264, 287, 262, 614, 1596, 1415, 13970, 262, 614, 1596)),
 ('year-transfer-01-reign-forward-low',
  'The reign lasted from the year 1705 to the year 17',
  (464, 13580, 15436, 422, 262, 614, 1596, 2713, 284, 262, 614, 1596)),
 ('year-transfer-01-reign-forward-high',
  'The reign lasted from the year 1714 to the year 17',
  (464, 13580, 15436, 422, 262, 614, 1596, 1415, 284, 262, 614, 1596)),
 ('year-transfer-01-reign-reverse-low',
  'The reign lasted until the year 1705, after starting in the year 17',
  (464, 13580, 15436, 1566, 262, 614, 1596, 2713, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-01-reign-reverse-high',
  'The reign lasted until the year 1714, after starting in the year 17',
  (464, 13580, 15436, 1566, 262, 614, 1596, 1415, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-01-reign-neutral-low',
  'The reign was recorded in the year 1705 beside the year 17',
  (464, 13580, 373, 6264, 287, 262, 614, 1596, 2713, 13970, 262, 614, 1596)),
 ('year-transfer-01-reign-neutral-high',
  'The reign was recorded in the year 1714 beside the year 17',
  (464, 13580, 373, 6264, 287, 262, 614, 1596, 1415, 13970, 262, 614, 1596)),
 ('year-transfer-02-war-forward-low',
  'The war lasted from the year 1727 to the year 17',
  (464, 1175, 15436, 422, 262, 614, 1596, 1983, 284, 262, 614, 1596)),
 ('year-transfer-02-war-forward-high',
  'The war lasted from the year 1736 to the year 17',
  (464, 1175, 15436, 422, 262, 614, 1596, 2623, 284, 262, 614, 1596)),
 ('year-transfer-02-war-reverse-low',
  'The war lasted until the year 1727, after starting in the year 17',
  (464, 1175, 15436, 1566, 262, 614, 1596, 1983, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-02-war-reverse-high',
  'The war lasted until the year 1736, after starting in the year 17',
  (464, 1175, 15436, 1566, 262, 614, 1596, 2623, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-02-war-neutral-low',
  'The war was recorded in the year 1727 beside the year 17',
  (464, 1175, 373, 6264, 287, 262, 614, 1596, 1983, 13970, 262, 614, 1596)),
 ('year-transfer-02-war-neutral-high',
  'The war was recorded in the year 1736 beside the year 17',
  (464, 1175, 373, 6264, 287, 262, 614, 1596, 2623, 13970, 262, 614, 1596)),
 ('year-transfer-02-reign-forward-low',
  'The reign lasted from the year 1727 to the year 17',
  (464, 13580, 15436, 422, 262, 614, 1596, 1983, 284, 262, 614, 1596)),
 ('year-transfer-02-reign-forward-high',
  'The reign lasted from the year 1736 to the year 17',
  (464, 13580, 15436, 422, 262, 614, 1596, 2623, 284, 262, 614, 1596)),
 ('year-transfer-02-reign-reverse-low',
  'The reign lasted until the year 1727, after starting in the year 17',
  (464, 13580, 15436, 1566, 262, 614, 1596, 1983, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-02-reign-reverse-high',
  'The reign lasted until the year 1736, after starting in the year 17',
  (464, 13580, 15436, 1566, 262, 614, 1596, 2623, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-02-reign-neutral-low',
  'The reign was recorded in the year 1727 beside the year 17',
  (464, 13580, 373, 6264, 287, 262, 614, 1596, 1983, 13970, 262, 614, 1596)),
 ('year-transfer-02-reign-neutral-high',
  'The reign was recorded in the year 1736 beside the year 17',
  (464, 13580, 373, 6264, 287, 262, 614, 1596, 2623, 13970, 262, 614, 1596)),
 ('year-transfer-03-war-forward-low',
  'The war lasted from the year 1749 to the year 17',
  (464, 1175, 15436, 422, 262, 614, 1596, 2920, 284, 262, 614, 1596)),
 ('year-transfer-03-war-forward-high',
  'The war lasted from the year 1758 to the year 17',
  (464, 1175, 15436, 422, 262, 614, 1596, 3365, 284, 262, 614, 1596)),
 ('year-transfer-03-war-reverse-low',
  'The war lasted until the year 1749, after starting in the year 17',
  (464, 1175, 15436, 1566, 262, 614, 1596, 2920, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-03-war-reverse-high',
  'The war lasted until the year 1758, after starting in the year 17',
  (464, 1175, 15436, 1566, 262, 614, 1596, 3365, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-03-war-neutral-low',
  'The war was recorded in the year 1749 beside the year 17',
  (464, 1175, 373, 6264, 287, 262, 614, 1596, 2920, 13970, 262, 614, 1596)),
 ('year-transfer-03-war-neutral-high',
  'The war was recorded in the year 1758 beside the year 17',
  (464, 1175, 373, 6264, 287, 262, 614, 1596, 3365, 13970, 262, 614, 1596)),
 ('year-transfer-03-reign-forward-low',
  'The reign lasted from the year 1749 to the year 17',
  (464, 13580, 15436, 422, 262, 614, 1596, 2920, 284, 262, 614, 1596)),
 ('year-transfer-03-reign-forward-high',
  'The reign lasted from the year 1758 to the year 17',
  (464, 13580, 15436, 422, 262, 614, 1596, 3365, 284, 262, 614, 1596)),
 ('year-transfer-03-reign-reverse-low',
  'The reign lasted until the year 1749, after starting in the year 17',
  (464, 13580, 15436, 1566, 262, 614, 1596, 2920, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-03-reign-reverse-high',
  'The reign lasted until the year 1758, after starting in the year 17',
  (464, 13580, 15436, 1566, 262, 614, 1596, 3365, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-03-reign-neutral-low',
  'The reign was recorded in the year 1749 beside the year 17',
  (464, 13580, 373, 6264, 287, 262, 614, 1596, 2920, 13970, 262, 614, 1596)),
 ('year-transfer-03-reign-neutral-high',
  'The reign was recorded in the year 1758 beside the year 17',
  (464, 13580, 373, 6264, 287, 262, 614, 1596, 3365, 13970, 262, 614, 1596)),
 ('year-transfer-04-war-forward-low',
  'The war lasted from the year 1771 to the year 17',
  (464, 1175, 15436, 422, 262, 614, 1596, 4869, 284, 262, 614, 1596)),
 ('year-transfer-04-war-forward-high',
  'The war lasted from the year 1780 to the year 17',
  (464, 1175, 15436, 422, 262, 614, 1596, 1795, 284, 262, 614, 1596)),
 ('year-transfer-04-war-reverse-low',
  'The war lasted until the year 1771, after starting in the year 17',
  (464, 1175, 15436, 1566, 262, 614, 1596, 4869, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-04-war-reverse-high',
  'The war lasted until the year 1780, after starting in the year 17',
  (464, 1175, 15436, 1566, 262, 614, 1596, 1795, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-04-war-neutral-low',
  'The war was recorded in the year 1771 beside the year 17',
  (464, 1175, 373, 6264, 287, 262, 614, 1596, 4869, 13970, 262, 614, 1596)),
 ('year-transfer-04-war-neutral-high',
  'The war was recorded in the year 1780 beside the year 17',
  (464, 1175, 373, 6264, 287, 262, 614, 1596, 1795, 13970, 262, 614, 1596)),
 ('year-transfer-04-reign-forward-low',
  'The reign lasted from the year 1771 to the year 17',
  (464, 13580, 15436, 422, 262, 614, 1596, 4869, 284, 262, 614, 1596)),
 ('year-transfer-04-reign-forward-high',
  'The reign lasted from the year 1780 to the year 17',
  (464, 13580, 15436, 422, 262, 614, 1596, 1795, 284, 262, 614, 1596)),
 ('year-transfer-04-reign-reverse-low',
  'The reign lasted until the year 1771, after starting in the year 17',
  (464, 13580, 15436, 1566, 262, 614, 1596, 4869, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-04-reign-reverse-high',
  'The reign lasted until the year 1780, after starting in the year 17',
  (464, 13580, 15436, 1566, 262, 614, 1596, 1795, 11, 706, 3599, 287, 262, 614, 1596)),
 ('year-transfer-04-reign-neutral-low',
  'The reign was recorded in the year 1771 beside the year 17',
  (464, 13580, 373, 6264, 287, 262, 614, 1596, 4869, 13970, 262, 614, 1596)),
 ('year-transfer-04-reign-neutral-high',
  'The reign was recorded in the year 1780 beside the year 17',
  (464, 13580, 373, 6264, 287, 262, 614, 1596, 1795, 13970, 262, 614, 1596)))
YEAR_TRANSFER_ROWS = tuple(row for _, _, row in YEAR_TRANSFER_CASES)


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _prefixes(records):
    return {tuple(row["token_ids"][:length]) for row in records
            for length in range(1, len(row["token_ids"]) + 1)}


def inventory_payload():
    predecessor = v15.inventory_payload()
    if (v15.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v15.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_public_year_transfer_rows": 48}
    cases = YEAR_TRANSFER_CASES
    if (type(cases) is not tuple or len(cases) != 48
            or any(type(case) is not tuple or len(case) != 3 for case in cases)):
        raise ValueError("expected the 48 fixed public cases")
    ids = [case[0] for case in cases]
    rows = [case[2] for case in cases]
    if (len(set(ids)) != 48 or len(set(rows)) != 48
            or any(type(case_id) is not str or re.fullmatch(
                r"year-transfer-0[1-4]-(war|reign)-(forward|reverse|neutral)-(low|high)",
                case_id) is None for case_id in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _ in cases)
            or any(type(row) is not tuple or len(row) not in (12, 13, 15)
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row)
                   for row in rows)):
        raise ValueError("year-transfer case identities or input tokens are invalid")
    old_rows = [tuple(row["token_ids"]) for row in payload["records"]]
    old_rows.extend(tuple(row["token_ids"] + [row["target_token_id"]])
                    for row in payload["records"] if type(row.get("target_token_id")) is int)
    if any(_related(candidate, prior) for candidate in rows for prior in old_rows):
        raise ValueError("year-transfer input has a known full-row prefix relation")
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("year-transfer inputs contain a full-row prefix relation")
    old_ids = {row["record_id"] for row in payload["records"]}
    for case_id, text, row in cases:
        record_id = "planned-" + case_id
        if record_id in old_ids:
            raise ValueError("year-transfer record ID already exists")
        payload["records"].append({
            "record_id": record_id, "category": "planned", "token_ids": list(row),
            "public_text": text,
            "provenance": "Public non-blind year-transfer cases; planned only and no model forward is asserted. "
                          f"Case inventory SHA-256 {CASE_SHA256}.",
            "references": [PROTOCOL_PATH, CASE_PATH], "reference_commit": CASES_COMMIT,
        })
    if len(_prefixes(payload["records"])) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("year-transfer forbidden-prefix count changed")
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
