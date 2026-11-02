"""Reserve the public agreement-direction prompts in the known inventory."""
from collections import Counter
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v19 as v19

INVENTORY_VERSION = "distilgpt2-development-exposures-v20"
PREDECESSOR_SHA256 = "e2eb185b32e74a4442ec68c43e53bd851573f60b7d7423b0d83195950f21b175"
INVENTORY_SHA256 = "3df696303a06b2ec6ddf4113df1969897dbae89566375b7d3a9cecb85b3d59cd"
FORBIDDEN_PREFIX_COUNT = 3620
TARGET = v19.TARGET
PUBLIC_DISCOVERY = v19.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v19.FACTORIAL_GROUPS
FACTORIAL_ROWS = v19.FACTORIAL_ROWS
RECIPIENT_CASES = v19.RECIPIENT_CASES
INJECTION_RECORD_IDS = v19.INJECTION_RECORD_IDS
PAIRS_DOSE_GROUPS = v19.PAIRS_DOSE_GROUPS
PAIRS_DOSE_ROWS = v19.PAIRS_DOSE_ROWS
YEAR_TRANSFER_CASES = v19.YEAR_TRANSFER_CASES
YEAR_TRANSFER_ROWS = v19.YEAR_TRANSFER_ROWS
GROUPED_REPORTED_ROWS = v19.GROUPED_REPORTED_ROWS
SCREENED_CASES = v19.SCREENED_CASES
SCREENED_ROWS = v19.SCREENED_ROWS

PROTOCOL_COMMIT = "a39121c2f3cf2602281252a8856d48efadcb2a98"
PROTOCOL_PATH = "docs/AGREEMENT_DIRECTION_STUDY_V1.md"
PROTOCOL_SHA256 = "f539ad55b347f88641de3b76565104dc7be43e0c5d15f6d8fefdb4173694bbea"
CASES_COMMIT = "3a5d5d2088471a702acea42c2f38d2cb2e0072bc"
CASE_PATH = "experiments/distilgpt2_agreement_direction_cases_v1.json"
CASE_SHA256 = "a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13"
ORDERED_ROWS_SHA256 = "4b4faf815ae0d3d0943efec5fbc318fd251dc6fea80b6e32c770b470f01dae1d"
COMPARISON_INPUT_PATH = "results/distilgpt2-agreement-direction-cases-v1/natural-comparison-inputs.json"
COMPARISON_INPUT_SHA256 = "62d762605c3ca9fa78f53c6f186c6b1f2e94e08e1479d5f3f16058a805eb28fe"
RECEIPT_PATH = "results/distilgpt2-agreement-direction-cases-v1/natural-comparison.json"
RECEIPT_SHA256 = "2f288aa89494b2e597b64ba594166d9c8bc78d2a9bc5831377971d8116c4b8d6"

AGREEMENT_CASES = (('agreement-direction-01-0', 'target', 'The chef near the fence', (464, 21221, 1474, 262, 13990)),
 ('agreement-direction-01-1', 'target', 'The chefs near the fence', (464, 36236, 1474, 262, 13990)),
 ('agreement-direction-01-2', 'target', 'The man near the fence', (464, 582, 1474, 262, 13990)),
 ('agreement-direction-01-3', 'target', 'The men near the fence', (464, 1450, 1474, 262, 13990)),
 ('agreement-direction-02-0', 'target', 'The chef near the fences', (464, 21221, 1474, 262, 35771)),
 ('agreement-direction-02-1', 'target', 'The chefs near the fences', (464, 36236, 1474, 262, 35771)),
 ('agreement-direction-02-2', 'target', 'The man near the fences', (464, 582, 1474, 262, 35771)),
 ('agreement-direction-02-3', 'target', 'The men near the fences', (464, 1450, 1474, 262, 35771)),
 ('agreement-direction-03-0', 'target', 'The nurse beside the tree', (464, 15849, 13970, 262, 5509)),
 ('agreement-direction-03-1', 'target', 'The nurses beside the tree', (464, 20669, 13970, 262, 5509)),
 ('agreement-direction-03-2', 'target', 'The woman beside the tree', (464, 2415, 13970, 262, 5509)),
 ('agreement-direction-03-3', 'target', 'The women beside the tree', (464, 1466, 13970, 262, 5509)),
 ('agreement-direction-04-0', 'target', 'The nurse beside the trees', (464, 15849, 13970, 262, 7150)),
 ('agreement-direction-04-1', 'target', 'The nurses beside the trees', (464, 20669, 13970, 262, 7150)),
 ('agreement-direction-04-2', 'target', 'The woman beside the trees', (464, 2415, 13970, 262, 7150)),
 ('agreement-direction-04-3', 'target', 'The women beside the trees', (464, 1466, 13970, 262, 7150)),
 ('agreement-direction-05-0', 'target', 'The pilot behind the window', (464, 8022, 2157, 262, 4324)),
 ('agreement-direction-05-1', 'target', 'The pilots behind the window', (464, 14982, 2157, 262, 4324)),
 ('agreement-direction-05-2', 'target', 'The child behind the window', (464, 1200, 2157, 262, 4324)),
 ('agreement-direction-05-3', 'target', 'The children behind the window', (464, 1751, 2157, 262, 4324)),
 ('agreement-direction-06-0', 'target', 'The pilot behind the windows', (464, 8022, 2157, 262, 9168)),
 ('agreement-direction-06-1', 'target', 'The pilots behind the windows', (464, 14982, 2157, 262, 9168)),
 ('agreement-direction-06-2', 'target', 'The child behind the windows', (464, 1200, 2157, 262, 9168)),
 ('agreement-direction-06-3', 'target', 'The children behind the windows', (464, 1751, 2157, 262, 9168)),
 ('agreement-direction-07-0', 'target', 'The guard outside the door', (464, 4860, 2354, 262, 3420)),
 ('agreement-direction-07-1', 'target', 'The guards outside the door', (464, 10942, 2354, 262, 3420)),
 ('agreement-direction-07-2', 'target', 'The person outside the door', (464, 1048, 2354, 262, 3420)),
 ('agreement-direction-07-3', 'target', 'The people outside the door', (464, 661, 2354, 262, 3420)),
 ('agreement-direction-08-0', 'target', 'The guard outside the doors', (464, 4860, 2354, 262, 8215)),
 ('agreement-direction-08-1', 'target', 'The guards outside the doors', (464, 10942, 2354, 262, 8215)),
 ('agreement-direction-08-2', 'target', 'The person outside the doors', (464, 1048, 2354, 262, 8215)),
 ('agreement-direction-08-3', 'target', 'The people outside the doors', (464, 661, 2354, 262, 8215)),
 ('agreement-direction-context-01', 'context', 'Near the fence', (40640, 262, 13990)),
 ('agreement-direction-context-02', 'context', 'Near the fences', (40640, 262, 35771)),
 ('agreement-direction-context-03', 'context', 'Beside the tree', (33, 274, 485, 262, 5509)),
 ('agreement-direction-context-04', 'context', 'Beside the trees', (33, 274, 485, 262, 7150)),
 ('agreement-direction-context-05', 'context', 'Behind the window', (34163, 262, 4324)),
 ('agreement-direction-context-06', 'context', 'Behind the windows', (34163, 262, 9168)),
 ('agreement-direction-context-07', 'context', 'Outside the door', (30815, 262, 3420)),
 ('agreement-direction-context-08', 'context', 'Outside the doors', (30815, 262, 8215)))
AGREEMENT_ROWS = tuple(row for _, _, _, row in AGREEMENT_CASES)
AGREEMENT_ANSWER_IDS = (318, 389, 373, 547)
_canonical = v19._canonical
_prefixes = v19._prefixes
_has_known_relation = v19._has_known_relation


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def inventory_payload():
    predecessor = v19.inventory_payload()
    if (v19.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v19.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_agreement_direction_rows": 40}
    cases = AGREEMENT_CASES
    if (type(cases) is not tuple or len(cases) != 40
            or any(type(case) is not tuple or len(case) != 4 for case in cases)):
        raise ValueError("expected the 40 fixed public agreement prompts")
    ids, rows = [case[0] for case in cases], [case[3] for case in cases]
    if (len(set(ids)) != 40 or len(set(rows)) != 40
            or Counter(case[1] for case in cases) != {"target": 32, "context": 8}
            or any(type(case_id) is not str or re.fullmatch(r"agreement-direction-[a-z0-9-]+", case_id) is None
                   for case_id in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, _, text, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 32
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)
            or type(AGREEMENT_ANSWER_IDS) is not tuple or len(AGREEMENT_ANSWER_IDS) != 4
            or len(set(AGREEMENT_ANSWER_IDS)) != 4
            or any(type(token) is not int or not 0 <= token < 50257 for token in AGREEMENT_ANSWER_IDS)):
        raise ValueError("agreement case identities, roles or tokens are invalid")
    old_rows = [tuple(row["token_ids"]) for row in payload["records"]]
    old_rows.extend(tuple(row["token_ids"] + [row["target_token_id"]])
                    for row in payload["records"] if type(row.get("target_token_id")) is int)
    complete = [row + (answer,) for row in rows for answer in AGREEMENT_ANSWER_IDS]
    if any(_related(candidate, prior) for candidate in rows + complete for prior in old_rows):
        raise ValueError("agreement input or candidate continuation has a known full-row prefix relation")
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("agreement prompts contain a full-row prefix relation")
    if any(_related(left, right) for i, left in enumerate(complete) for right in complete[i + 1:]):
        raise ValueError("agreement candidate continuations contain a full-row prefix relation")
    prior_ids = {row["record_id"] for row in payload["records"]}
    for case_id, role, text, row in cases:
        record_id = "planned-" + case_id
        if record_id in prior_ids:
            raise ValueError("agreement prompt record ID already exists")
        payload["records"].append({
            "record_id": record_id, "category": "planned", "token_ids": list(row),
            "case_role": role, "candidate_token_ids": list(AGREEMENT_ANSWER_IDS), "public_text": text,
            "provenance": "Public non-blind agreement-direction prompt; planned only and no model forward is asserted. "
                          f"Case inventory SHA-256 {CASE_SHA256}.",
            "references": [PROTOCOL_PATH, CASE_PATH, COMPARISON_INPUT_PATH, RECEIPT_PATH],
            "reference_commit": CASES_COMMIT,
        })
    if len(_prefixes(payload["records"])) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("agreement prompt forbidden-prefix count changed")
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
