"""Reserve public copying-head and synthetic service inputs; no forward asserted."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v30 as prior

INVENTORY_VERSION = "distilgpt2-development-exposures-v31"
PREDECESSOR_SHA256 = "921025a7f9c1246c5f10e97f2b987ca4971462241b2d05ebbc525d4aa0736c79"
INVENTORY_SHA256 = "2d710a40a49a84b7d664cd851fccd2527ea318c5a8c911111fe7275b7567e565"
FORBIDDEN_PREFIX_COUNT = 5789
COPY_CASES_COMMIT = "115d9bb42aadc481482de302d2df11c9c2d30789"
COPY_CASES_SHA256 = "b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69"
SYNTHETIC_PUBLIC_CHECK_SHA256 = "9ff6f36cb71960398a1113e5731e224d83e5483748cdd307ae719a6dc9dbbe39"
SYNTHETIC_CASES_COMMIT = "63886558b85212ec59d73f04c78e5e722f1e36f6"
SYNTHETIC_CASES_SHA256 = "47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c"
_canonical = prior._canonical
_prefixes = prior._prefixes
for _name in ("TARGET", "PUBLIC_DISCOVERY", "FACTORIAL_GROUPS", "FACTORIAL_ROWS",
              "RECIPIENT_CASES", "INJECTION_RECORD_IDS", "PAIRS_DOSE_GROUPS", "PAIRS_DOSE_ROWS",
              "YEAR_TRANSFER_CASES", "YEAR_TRANSFER_ROWS", "GROUPED_REPORTED_ROWS",
              "GROUPED_POLICY_ROWS", "GROUPED_BROKER_OBSERVED_ROWS", "SCREENED_CASES",
              "SCREENED_ROWS", "SCREENED_OBSERVED_CASES", "SCREENED_PLANNED_LOCALITY_CASES",
              "AGREEMENT_CASES", "AGREEMENT_ROWS", "AGREEMENT_ANSWER_IDS",
              "EXPLICIT_YEAR_DIRECTION_CASES", "EXPLICIT_YEAR_DIRECTION_ROWS",
              "YEAR_LABEL_CONTROL_CASES", "YEAR_LABEL_CONTROL_ROWS", "FULL_TOP_CASES", "FULL_TOP_ROWS",
              "AGREEMENT_SUBJECT_FOLLOWUP_ANSWER_IDS", "AGREEMENT_SUBJECT_FOLLOWUP_CASES",
              "AGREEMENT_SUBJECT_FOLLOWUP_ROWS", "ANSWER_FORMAT_CASES", "ANSWER_FORMAT_ROWS",
              "TWO_ATTRACTOR_CASES", "TWO_ATTRACTOR_ROWS"):
    globals()[_name] = getattr(prior, _name)

COPY_PROMPT_DOSE_CASES = (('copy-prompt-dose-01-0',
  ' marsh delta harbor reef plateau beach marsh delta harbor reef plateau',
  (22397, 25979, 25451, 25088, 38008, 10481, 22397, 25979, 25451, 25088, 38008),
  10481),
 ('copy-prompt-dose-01-1',
  ' plateau harbor marsh reef delta beach marsh delta harbor reef plateau',
  (38008, 25451, 22397, 25088, 25979, 10481, 22397, 25979, 25451, 25088, 38008),
  10481),
 ('copy-prompt-dose-01-2',
  ' delta reef plateau harbor marsh beach marsh delta harbor reef plateau',
  (25979, 25088, 38008, 25451, 22397, 10481, 22397, 25979, 25451, 25088, 38008),
  10481),
 ('copy-prompt-dose-01-3',
  ' reef marsh plateau delta harbor beach marsh delta harbor reef plateau',
  (25088, 22397, 38008, 25979, 25451, 10481, 22397, 25979, 25451, 25088, 38008),
  10481),
 ('copy-prompt-dose-02-0',
  ' pear plum lime lettuce cereal bacon pear plum lime lettuce cereal',
  (25286, 22802, 28738, 39406, 33158, 21385, 25286, 22802, 28738, 39406, 33158),
  21385),
 ('copy-prompt-dose-02-1',
  ' cereal lime pear lettuce plum bacon pear plum lime lettuce cereal',
  (33158, 28738, 25286, 39406, 22802, 21385, 25286, 22802, 28738, 39406, 33158),
  21385),
 ('copy-prompt-dose-02-2',
  ' plum lettuce cereal lime pear bacon pear plum lime lettuce cereal',
  (22802, 39406, 33158, 28738, 25286, 21385, 25286, 22802, 28738, 39406, 33158),
  21385),
 ('copy-prompt-dose-02-3',
  ' lettuce pear cereal plum lime bacon pear plum lime lettuce cereal',
  (39406, 25286, 33158, 22802, 28738, 21385, 25286, 22802, 28738, 39406, 33158),
  21385),
 ('copy-prompt-dose-03-0',
  ' seal trout insect ant bee butterfly seal trout insect ant bee',
  (13810, 49411, 12833, 1885, 20697, 35113, 13810, 49411, 12833, 1885, 20697),
  35113),
 ('copy-prompt-dose-03-1',
  ' bee insect seal ant trout butterfly seal trout insect ant bee',
  (20697, 12833, 13810, 1885, 49411, 35113, 13810, 49411, 12833, 1885, 20697),
  35113),
 ('copy-prompt-dose-03-2',
  ' trout ant bee insect seal butterfly seal trout insect ant bee',
  (49411, 1885, 20697, 12833, 13810, 35113, 13810, 49411, 12833, 1885, 20697),
  35113),
 ('copy-prompt-dose-03-3',
  ' ant seal bee trout insect butterfly seal trout insect ant bee',
  (1885, 13810, 20697, 49411, 12833, 35113, 13810, 49411, 12833, 1885, 20697),
  35113),
 ('copy-prompt-dose-04-0',
  ' shovel drill axe saw wrench rake shovel drill axe saw wrench',
  (33677, 16007, 23272, 2497, 34561, 47885, 33677, 16007, 23272, 2497, 34561),
  47885),
 ('copy-prompt-dose-04-1',
  ' wrench axe shovel saw drill rake shovel drill axe saw wrench',
  (34561, 23272, 33677, 2497, 16007, 47885, 33677, 16007, 23272, 2497, 34561),
  47885),
 ('copy-prompt-dose-04-2',
  ' drill saw wrench axe shovel rake shovel drill axe saw wrench',
  (16007, 2497, 34561, 23272, 33677, 47885, 33677, 16007, 23272, 2497, 34561),
  47885),
 ('copy-prompt-dose-04-3',
  ' saw shovel wrench drill axe rake shovel drill axe saw wrench',
  (2497, 33677, 34561, 16007, 23272, 47885, 33677, 16007, 23272, 2497, 34561),
  47885),
 ('copy-prompt-dose-05-0',
  ' bass opera orchestra festival chorus composer bass opera orchestra festival chorus',
  (12702, 27296, 40095, 10876, 26441, 26777, 12702, 27296, 40095, 10876, 26441),
  26777),
 ('copy-prompt-dose-05-1',
  ' chorus orchestra bass festival opera composer bass opera orchestra festival chorus',
  (26441, 40095, 12702, 10876, 27296, 26777, 12702, 27296, 40095, 10876, 26441),
  26777),
 ('copy-prompt-dose-05-2',
  ' opera festival chorus orchestra bass composer bass opera orchestra festival chorus',
  (27296, 10876, 26441, 40095, 12702, 26777, 12702, 27296, 40095, 10876, 26441),
  26777),
 ('copy-prompt-dose-05-3',
  ' festival bass chorus opera orchestra composer bass opera orchestra festival chorus',
  (10876, 12702, 26441, 27296, 40095, 26777, 12702, 27296, 40095, 10876, 26441),
  26777),
 ('copy-prompt-dose-06-0',
  ' van sedan helicopter carriage cruiser tram van sedan helicopter carriage cruiser',
  (5719, 38988, 15392, 25739, 35239, 29957, 5719, 38988, 15392, 25739, 35239),
  29957),
 ('copy-prompt-dose-06-1',
  ' cruiser helicopter van carriage sedan tram van sedan helicopter carriage cruiser',
  (35239, 15392, 5719, 25739, 38988, 29957, 5719, 38988, 15392, 25739, 35239),
  29957),
 ('copy-prompt-dose-06-2',
  ' sedan carriage cruiser helicopter van tram van sedan helicopter carriage cruiser',
  (38988, 25739, 35239, 15392, 5719, 29957, 5719, 38988, 15392, 25739, 35239),
  29957),
 ('copy-prompt-dose-06-3',
  ' carriage van cruiser sedan helicopter tram van sedan helicopter carriage cruiser',
  (25739, 5719, 35239, 38988, 15392, 29957, 5719, 38988, 15392, 25739, 35239),
  29957),
 ('copy-prompt-dose-07-0',
  ' trousers robe bracelet tie necklace ring trousers robe bracelet tie necklace',
  (37191, 33192, 42893, 9839, 38987, 5858, 37191, 33192, 42893, 9839, 38987),
  5858),
 ('copy-prompt-dose-07-1',
  ' necklace bracelet trousers tie robe ring trousers robe bracelet tie necklace',
  (38987, 42893, 37191, 9839, 33192, 5858, 37191, 33192, 42893, 9839, 38987),
  5858),
 ('copy-prompt-dose-07-2',
  ' robe tie necklace bracelet trousers ring trousers robe bracelet tie necklace',
  (33192, 9839, 38987, 42893, 37191, 5858, 37191, 33192, 42893, 9839, 38987),
  5858),
 ('copy-prompt-dose-07-3',
  ' tie trousers necklace robe bracelet ring trousers robe bracelet tie necklace',
  (9839, 37191, 38987, 33192, 42893, 5858, 37191, 33192, 42893, 9839, 38987),
  5858),
 ('copy-prompt-dose-08-0',
  ' desk sofa couch frame drawer stove desk sofa couch frame drawer',
  (6915, 34902, 18507, 5739, 33451, 27635, 6915, 34902, 18507, 5739, 33451),
  27635),
 ('copy-prompt-dose-08-1',
  ' drawer couch desk frame sofa stove desk sofa couch frame drawer',
  (33451, 18507, 6915, 5739, 34902, 27635, 6915, 34902, 18507, 5739, 33451),
  27635),
 ('copy-prompt-dose-08-2',
  ' sofa frame drawer couch desk stove desk sofa couch frame drawer',
  (34902, 5739, 33451, 18507, 6915, 27635, 6915, 34902, 18507, 5739, 33451),
  27635),
 ('copy-prompt-dose-08-3',
  ' frame desk drawer sofa couch stove desk sofa couch frame drawer',
  (5739, 6915, 33451, 34902, 18507, 27635, 6915, 34902, 18507, 5739, 33451),
  27635))
COPY_PROMPT_DOSE_ROWS = tuple(row for _, _, row, _ in COPY_PROMPT_DOSE_CASES)
SYNTHETIC_SERVICE_CASES = (
    ("planned-service-synthetic-mask-16", (50256, *range(1001, 1016))),
    ("planned-service-synthetic-mask-32", (50256, *range(2001, 2032))),
)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload["records"]
    old_ids = [record["record_id"] for record in records]
    if len(old_ids) != len(set(old_ids)):
        raise ValueError("predecessor record identities repeat")
    cases = COPY_PROMPT_DOSE_CASES
    ids = ["planned-" + case_id for case_id, _, _, _ in cases] + [item[0] for item in SYNTHETIC_SERVICE_CASES]
    rows = list(COPY_PROMPT_DOSE_ROWS) + [item[1] for item in SYNTHETIC_SERVICE_CASES]
    if (len(cases) != 32 or len(rows) != 34 or len(set(ids)) != 34 or len(set(rows)) != 34
            or set(ids).intersection(old_ids)
            or any(re.fullmatch(r"copy-prompt-dose-[0-9]{2}-[0-3]", case_id) is None
                   or type(text) is not str or not text or len(row) != 11
                   or type(target) is not int or not 0 <= target < 50257
                   for case_id, text, row, target in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 32
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)):
        raise ValueError("planned input identities or rows are invalid")
    old_rows = [tuple(record["token_ids"]) for record in records]
    old_rows.extend(tuple(record["token_ids"] + [record["target_token_id"]])
                    for record in records if type(record.get("target_token_id")) is int)
    old_rows.extend(tuple(record["token_ids"] + [token]) for record in records
                    for token in record.get("candidate_token_ids", ()))
    old_rows.extend(tuple(record["token_ids"] + [token]) for record in records
                    for group in record.get("candidate_token_groups", {}).values() for token in group)
    if any(_related(row, old) for row in rows for old in old_rows):
        raise ValueError("a planned input relates to a predecessor input or continuation")
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("planned inputs contain a full-row prefix relation")
    for case_id, text, row, target in cases:
        records.append({"record_id": "planned-" + case_id, "category": "planned",
                        "token_ids": list(row), "target_token_id": target, "public_text": text,
                        "provenance": "Public per-prompt matched-dose copying input; no model forward asserted. "
                                      "Case SHA-256 " + COPY_CASES_SHA256 + ".",
                        "references": ["docs/DISTILGPT2_COPY_HEAD_PROMPT_DOSE_V1.md",
                                       "experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json"],
                        "reference_commit": COPY_CASES_COMMIT})
    for record_id, row in SYNTHETIC_SERVICE_CASES:
        records.append({"record_id": record_id, "category": "planned", "token_ids": list(row),
                        "public_text": "", "attention_mask": [1] * len(row),
                        "provenance": "Public synthetic all-one-mask CPU service input; "
                                      "no model forward asserted. Public-scope receipt SHA-256 "
                                      + SYNTHETIC_PUBLIC_CHECK_SHA256 + ". Case SHA-256 "
                                      + SYNTHETIC_CASES_SHA256 + ".",
                        "references": ["experiments/short_cpu_service_reference_inputs_v1.json",
                                       "docs/SHORT_SERVICE_SYNTHETIC_PUBLIC_SUPPORT_V1.json"],
                        "reference_commit": SYNTHETIC_CASES_COMMIT})
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": prior.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_copy_prompt_dose_rows": 32, "planned_service_synthetic_rows": 2,
                         "model_forward_asserted": False}
    return payload


def inventory_payload():
    predecessor, digest = prior._checked_inventory()
    if digest != PREDECESSOR_SHA256:
        raise ValueError("predecessor exposure inventory changed")
    payload = _build_payload(predecessor)
    if len(_prefixes(payload["records"])) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("planned input prefix count changed")
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
    complete = {tuple(row["token_ids"]) for row in payload["records"]}
    if any(_has_known_relation(row, forbidden, complete) for row in discovery + heldout):
        raise ValueError("candidate has a known input prefix relation")
    return v1.KnownSplitCheck(INVENTORY_VERSION, digest, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    payload, digest = _checked_inventory()
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest:
        raise ValueError("expected exposure inventory digest does not match")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _prefixes(payload["records"])
    complete = {tuple(row["token_ids"]) for row in payload["records"]}
    if any(_has_known_relation(row, forbidden, complete) for row in heldout):
        raise ValueError("held-out candidate has a known input prefix relation")
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, digest, 32, len(forbidden),
                                    sum(_has_known_relation(row, forbidden, complete) for row in discovery))


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
