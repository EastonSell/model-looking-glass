"""Reserve the fixed public screened-edit prompts in the known inventory."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v18 as v18

INVENTORY_VERSION = "distilgpt2-development-exposures-v19"
PREDECESSOR_SHA256 = "3fa3eda9508ca314505aa658b1c02d04dbc98681bfc776337cd9ca1c9c384f5a"
INVENTORY_SHA256 = "e2eb185b32e74a4442ec68c43e53bd851573f60b7d7423b0d83195950f21b175"
FORBIDDEN_PREFIX_COUNT = 3522
TARGET = v18.TARGET
PUBLIC_DISCOVERY = v18.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v18.FACTORIAL_GROUPS
FACTORIAL_ROWS = v18.FACTORIAL_ROWS
RECIPIENT_CASES = v18.RECIPIENT_CASES
INJECTION_RECORD_IDS = v18.INJECTION_RECORD_IDS
PAIRS_DOSE_GROUPS = v18.PAIRS_DOSE_GROUPS
PAIRS_DOSE_ROWS = v18.PAIRS_DOSE_ROWS
YEAR_TRANSFER_CASES = v18.YEAR_TRANSFER_CASES
YEAR_TRANSFER_ROWS = v18.YEAR_TRANSFER_ROWS
GROUPED_REPORTED_ROWS = v18.GROUPED_REPORTED_ROWS

PROTOCOL_COMMIT = "3d9a5a3c5cb6a6a905e90ef029fe627a147de4e6"
PROTOCOL_PATH = "docs/TARGETED_EDIT_SCREENED_MATCHED_V1.md"
PROTOCOL_SHA256 = "6a2bb800923e4d694819bb6fc8164e6b58698d0d12bd33d88397387b81783cde"
CASES_COMMIT = "e27798622cd56eff1db0866e112bbde1f3926971"
CASE_PATH = "experiments/targeted_edit_screened_public_cases_v1.json"
CASE_SHA256 = "fc3f00a0b4f7884a3c00011186420d16322009edf75b6b09e4296953dad198e3"
RECEIPT_PATH = "experiments/targeted_edit_screened_natural_receipt_v1.json"
RECEIPT_SHA256 = "bbd78ce2cb63db2a67cfc85cb030ac8cefea2f24c88c4105bff30d7392ef2c2b"
COMPARISON_INPUT_SHA256 = "48574a625548ee16fc6d58ccb6249ab9455b8fc464411700700c39c199d0256b"

SCREENED_CASES = (('screen-target-s1-01',
  'The recognized national capital of France is',
  (464, 8018, 2260, 3139, 286, 4881, 318)),
 ('screen-target-s1-02', 'The official capital of France is', (464, 1743, 3139, 286, 4881, 318)),
 ('screen-target-s1-03', 'The current capital of France is', (464, 1459, 3139, 286, 4881, 318)),
 ('screen-target-s1-04', 'The present capital of France is', (464, 1944, 3139, 286, 4881, 318)),
 ('screen-target-s1-05', 'The capital of France today is', (464, 3139, 286, 4881, 1909, 318)),
 ('screen-target-s1-06', 'The capital of France at present is', (464, 3139, 286, 4881, 379, 1944, 318)),
 ('screen-target-s1-07', 'The capital of the French nation is', (464, 3139, 286, 262, 4141, 3277, 318)),
 ('screen-target-s1-08', 'The national capital city of France is', (464, 2260, 3139, 1748, 286, 4881, 318)),
 ('screen-target-s2-01', "France's current capital city is", (28572, 338, 1459, 3139, 1748, 318)),
 ('screen-target-s2-02', "France's national capital is", (28572, 338, 2260, 3139, 318)),
 ('screen-target-s2-03', "France's official capital is", (28572, 338, 1743, 3139, 318)),
 ('screen-target-s2-04', "The French Republic's capital city is", (464, 4141, 2066, 338, 3139, 1748, 318)),
 ('screen-target-s2-05', "The French nation's capital city is", (464, 4141, 3277, 338, 3139, 1748, 318)),
 ('screen-target-s2-06', 'France has its capital city in', (28572, 468, 663, 3139, 1748, 287)),
 ('screen-target-s2-07', 'France identifies its capital city as', (28572, 21079, 663, 3139, 1748, 355)),
 ('screen-target-s2-08', 'France lists its national capital as', (28572, 8341, 663, 2260, 3139, 355)),
 ('screen-target-s3-01',
  "The city that serves as France's capital is",
  (464, 1748, 326, 9179, 355, 4881, 338, 3139, 318)),
 ('screen-target-s3-02',
  "The city designated as France's capital is",
  (464, 1748, 11032, 355, 4881, 338, 3139, 318)),
 ('screen-target-s3-03',
  'The city named as the capital of France is',
  (464, 1748, 3706, 355, 262, 3139, 286, 4881, 318)),
 ('screen-target-s3-04',
  'The city recognized as the capital of France is',
  (464, 1748, 8018, 355, 262, 3139, 286, 4881, 318)),
 ('screen-target-s3-05',
  'The city that is the capital of France is',
  (464, 1748, 326, 318, 262, 3139, 286, 4881, 318)),
 ('screen-target-s3-06',
  "The city holding the title of France's capital is",
  (464, 1748, 4769, 262, 3670, 286, 4881, 338, 3139, 318)),
 ('screen-target-s3-07',
  'The capital city of the country of France is called',
  (464, 3139, 1748, 286, 262, 1499, 286, 4881, 318, 1444)),
 ('screen-target-s3-08',
  "France's capital city bears the name",
  (28572, 338, 3139, 1748, 13062, 262, 1438)),
 ('screen-target-s4-01',
  "To name France's capital, one would say",
  (2514, 1438, 4881, 338, 3139, 11, 530, 561, 910)),
 ('screen-target-s4-02',
  "The answer to the question 'What is France's capital?' is",
  (464, 3280, 284, 262, 1808, 705, 2061, 318, 4881, 338, 3139, 8348, 318)),
 ('screen-target-s4-03',
  'When asked to name the capital of France, answer',
  (2215, 1965, 284, 1438, 262, 3139, 286, 4881, 11, 3280)),
 ('screen-target-s4-04',
  'On a list of countries and capitals, France is paired with',
  (2202, 257, 1351, 286, 2678, 290, 44590, 11, 4881, 318, 20312, 351)),
 ('screen-target-s4-05', "A map lists France's capital as", (32, 3975, 8341, 4881, 338, 3139, 355)),
 ('screen-target-s4-06',
  'The capital listed for the country of France is',
  (464, 3139, 5610, 329, 262, 1499, 286, 4881, 318)),
 ('screen-target-s4-07',
  'Fill in the blank: the capital of France is',
  (33762, 287, 262, 9178, 25, 262, 3139, 286, 4881, 318)),
 ('screen-target-s4-08',
  'For the nation of France, the capital is',
  (1890, 262, 3277, 286, 4881, 11, 262, 3139, 318)),
 ('screen-locality-other-country-capital-01',
  'The national capital of the United Kingdom is',
  (464, 2260, 3139, 286, 262, 1578, 7526, 318)),
 ('screen-locality-other-country-capital-02',
  'The national capital of Portugal is',
  (464, 2260, 3139, 286, 19101, 318)),
 ('screen-locality-other-country-capital-03',
  'The national capital of Poland is',
  (464, 2260, 3139, 286, 12873, 318)),
 ('screen-locality-other-country-capital-04',
  'The national capital of Sweden is',
  (464, 2260, 3139, 286, 10710, 318)),
 ('screen-locality-other-france-fact-01',
  'The Louvre Museum, formerly a royal palace, is in',
  (464, 4768, 43933, 9594, 11, 15734, 257, 15100, 20562, 11, 318, 287)),
 ('screen-locality-other-france-fact-02', "France's official language is", (28572, 338, 1743, 3303, 318)),
 ('screen-locality-other-france-fact-03',
  'Metropolitan France uses the currency called the',
  (9171, 14823, 4881, 3544, 262, 7395, 1444, 262)),
 ('screen-locality-other-france-fact-04',
  'Metropolitan France lies on the continent of',
  (9171, 14823, 4881, 7363, 319, 262, 15549, 286)),
 ('screen-locality-unrelated-fact-01',
  'The number of sides on a triangle is',
  (464, 1271, 286, 5389, 319, 257, 22950, 318)),
 ('screen-locality-unrelated-fact-02',
  'Water consists chemically of hydrogen and',
  (19184, 10874, 49966, 286, 17669, 290)),
 ('screen-locality-unrelated-fact-03',
  'A full hour consists of sixty',
  (32, 1336, 1711, 10874, 286, 24742)),
 ('screen-locality-unrelated-fact-04',
  'The number of days in a fortnight is',
  (464, 1271, 286, 1528, 287, 257, 46327, 318)))
SCREENED_ROWS = tuple(row for _, _, row in SCREENED_CASES)
_canonical = v18._canonical
_prefixes = v18._prefixes
_has_known_relation = v18._has_known_relation


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def inventory_payload():
    predecessor = v18.inventory_payload()
    if (v18.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v18.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_screened_edit_rows": 44}
    cases = SCREENED_CASES
    if (type(cases) is not tuple or len(cases) != 44
            or any(type(case) is not tuple or len(case) != 3 for case in cases)):
        raise ValueError("expected the 44 fixed public screened prompts")
    ids = [case[0] for case in cases]
    rows = [case[2] for case in cases]
    if (len(set(ids)) != 44 or len(set(rows)) != 44
            or any(type(case_id) is not str or re.fullmatch(r"screen-(target|locality)-[a-z0-9-]+", case_id) is None
                   for case_id in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 32
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)):
        raise ValueError("screened case identities or prompt tokens are invalid")
    old_rows = [tuple(row["token_ids"]) for row in payload["records"]]
    old_rows.extend(tuple(row["token_ids"] + [row["target_token_id"]])
                    for row in payload["records"] if type(row.get("target_token_id")) is int)
    if any(_related(candidate, prior) for candidate in rows for prior in old_rows):
        raise ValueError("screened prompt has a known full-row prefix relation")
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("screened prompts contain a full-row prefix relation")
    prior_ids = {row["record_id"] for row in payload["records"]}
    for case_id, text, row in cases:
        record_id = "planned-" + case_id
        if record_id in prior_ids:
            raise ValueError("screened prompt record ID already exists")
        payload["records"].append({
            "record_id": record_id, "category": "planned", "token_ids": list(row),
            "public_text": text,
            "provenance": "Public non-blind screened-edit prompt; planned only and no model forward is asserted. "
                          f"Case inventory SHA-256 {CASE_SHA256}.",
            "references": [PROTOCOL_PATH, CASE_PATH, RECEIPT_PATH], "reference_commit": CASES_COMMIT,
        })
    if len(_prefixes(payload["records"])) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("screened prompt forbidden-prefix count changed")
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
