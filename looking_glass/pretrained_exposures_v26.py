"""Reserve the fixed full-top study inputs in the public exposure inventory."""
import copy
import hashlib
import re
from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v25 as v25

INVENTORY_VERSION = "distilgpt2-development-exposures-v26"
PREDECESSOR_SHA256 = "32cc4bcbe4b482ded122715c5f96c6c5c765a76a1ef3532360cdf9890827f144"
INVENTORY_SHA256 = "8fd80cb0b24225100628489dcd0fe2b7efd11e87dfcbb624cf039e54a8a3a62d"
FORBIDDEN_PREFIX_COUNT = 4475
CASES_COMMIT = "f0fd89bd6e3bb41d7e6d457037178fd30b6a88b6"
CASE_PATH = "experiments/targeted_edit_full_top_cases_v1.json"
CASE_SHA256 = "4057696ddef9e3b0d09db02f46e6454561a61ac1a07f8b8a13b92718944f4261"
PROTOCOL_PATH = "docs/TARGETED_EDIT_FULL_TOP_CPU_V1.md"
SCOPE_PATH = "docs/TARGETED_EDIT_FULL_TOP_SCOPE_V1.md"
FRAME_PATH = "experiments/targeted_edit_full_top_public_frame_v1.json"
for _name in ('TARGET', 'PUBLIC_DISCOVERY', 'FACTORIAL_GROUPS', 'FACTORIAL_ROWS', 'RECIPIENT_CASES', 'INJECTION_RECORD_IDS', 'PAIRS_DOSE_GROUPS', 'PAIRS_DOSE_ROWS', 'YEAR_TRANSFER_CASES', 'YEAR_TRANSFER_ROWS', 'GROUPED_REPORTED_ROWS', 'GROUPED_POLICY_ROWS', 'GROUPED_BROKER_OBSERVED_ROWS', 'SCREENED_CASES', 'SCREENED_ROWS', 'SCREENED_OBSERVED_CASES', 'SCREENED_PLANNED_LOCALITY_CASES', 'AGREEMENT_CASES', 'AGREEMENT_ROWS', 'AGREEMENT_ANSWER_IDS', 'EXPLICIT_YEAR_DIRECTION_CASES', 'EXPLICIT_YEAR_DIRECTION_ROWS', 'YEAR_LABEL_CONTROL_CASES', 'YEAR_LABEL_CONTROL_ROWS'):
    globals()[_name] = getattr(v25, _name)
_canonical = v25._canonical
_prefixes = v25._prefixes
FULL_TOP_CASES = (('full-top-france-discovery-01',
  'The capital of Spain is Madrid. The capital of France is',
  (464, 3139, 286, 8602, 318, 14708, 13, 383, 3139, 286, 4881, 318),
  (6342, 3576)),
 ('full-top-france-efficacy-calibration-01',
  'Name of the national capital of France:',
  (5376, 286, 262, 2260, 3139, 286, 4881, 25),
  (6342, 3576)),
 ('full-top-france-efficacy-calibration-02',
  'Country and capital: Spain - Madrid; France -',
  (33921, 290, 3139, 25, 8602, 532, 14708, 26, 4881, 532),
  (6342, 3576)),
 ('full-top-france-evaluation-01',
  'Q: What is the national capital of France?\nA:',
  (48, 25, 1867, 318, 262, 2260, 3139, 286, 4881, 30, 198, 32, 25),
  (6342, 3576)),
 ('full-top-france-evaluation-02',
  "In a geography quiz, France's national capital is",
  (818, 257, 27876, 38964, 11, 4881, 338, 2260, 3139, 318),
  (6342, 3576)),
 ('full-top-france-evaluation-03',
  "France's national capital city is",
  (28572, 338, 2260, 3139, 1748, 318),
  (6342, 3576)),
 ('full-top-france-evaluation-04',
  'Country: France\nCapital city:',
  (33921, 25, 4881, 198, 39315, 1748, 25),
  (6342, 3576)),
 ('full-top-france-evaluation-05',
  'Capital cities: Spain is Madrid; France is',
  (39315, 4736, 25, 8602, 318, 14708, 26, 4881, 318),
  (6342, 3576)),
 ('full-top-france-evaluation-06',
  "Madrid is Spain's capital. France's capital is",
  (18454, 6058, 318, 8602, 338, 3139, 13, 4881, 338, 3139, 318),
  (6342, 3576)),
 ('full-top-japan-discovery-01',
  'The capital of Spain is Madrid. The capital of Japan is',
  (464, 3139, 286, 8602, 318, 14708, 13, 383, 3139, 286, 2869, 318),
  (11790, 6342)),
 ('full-top-japan-efficacy-calibration-01',
  'Name of the national capital of Japan:',
  (5376, 286, 262, 2260, 3139, 286, 2869, 25),
  (11790, 6342)),
 ('full-top-japan-efficacy-calibration-02',
  'Country and capital: Spain - Madrid; Japan -',
  (33921, 290, 3139, 25, 8602, 532, 14708, 26, 2869, 532),
  (11790, 6342)),
 ('full-top-japan-evaluation-01',
  'Q: What is the national capital of Japan?\nA:',
  (48, 25, 1867, 318, 262, 2260, 3139, 286, 2869, 30, 198, 32, 25),
  (11790, 6342)),
 ('full-top-japan-evaluation-02',
  "In a geography quiz, Japan's national capital is",
  (818, 257, 27876, 38964, 11, 2869, 338, 2260, 3139, 318),
  (11790, 6342)),
 ('full-top-japan-evaluation-03',
  "Japan's national capital city is",
  (16504, 338, 2260, 3139, 1748, 318),
  (11790, 6342)),
 ('full-top-japan-evaluation-04',
  'Country: Japan\nCapital city:',
  (33921, 25, 2869, 198, 39315, 1748, 25),
  (11790, 6342)),
 ('full-top-japan-evaluation-05',
  'Capital cities: Spain is Madrid; Japan is',
  (39315, 4736, 25, 8602, 318, 14708, 26, 2869, 318),
  (11790, 6342)),
 ('full-top-japan-evaluation-06',
  "Madrid is Spain's capital. Japan's capital is",
  (18454, 6058, 318, 8602, 338, 3139, 13, 2869, 338, 3139, 318),
  (11790, 6342)),
 ('full-top-italy-discovery-01',
  'The capital of Spain is Madrid. The capital of Italy is',
  (464, 3139, 286, 8602, 318, 14708, 13, 383, 3139, 286, 8031, 318),
  (10598, 3576)),
 ('full-top-italy-efficacy-calibration-01',
  'Name of the national capital of Italy:',
  (5376, 286, 262, 2260, 3139, 286, 8031, 25),
  (10598, 3576)),
 ('full-top-italy-efficacy-calibration-02',
  'Country and capital: Spain - Madrid; Italy -',
  (33921, 290, 3139, 25, 8602, 532, 14708, 26, 8031, 532),
  (10598, 3576)),
 ('full-top-italy-evaluation-01',
  'Q: What is the national capital of Italy?\nA:',
  (48, 25, 1867, 318, 262, 2260, 3139, 286, 8031, 30, 198, 32, 25),
  (10598, 3576)),
 ('full-top-italy-evaluation-02',
  "In a geography quiz, Italy's national capital is",
  (818, 257, 27876, 38964, 11, 8031, 338, 2260, 3139, 318),
  (10598, 3576)),
 ('full-top-italy-evaluation-03',
  "Italy's national capital city is",
  (45001, 338, 2260, 3139, 1748, 318),
  (10598, 3576)),
 ('full-top-italy-evaluation-04',
  'Country: Italy\nCapital city:',
  (33921, 25, 8031, 198, 39315, 1748, 25),
  (10598, 3576)),
 ('full-top-italy-evaluation-05',
  'Capital cities: Spain is Madrid; Italy is',
  (39315, 4736, 25, 8602, 318, 14708, 26, 8031, 318),
  (10598, 3576)),
 ('full-top-italy-evaluation-06',
  "Madrid is Spain's capital. Italy's capital is",
  (18454, 6058, 318, 8602, 338, 3139, 13, 8031, 338, 3139, 318),
  (10598, 3576)),
 ('full-top-uk-discovery-01',
  'The capital of Spain is Madrid. The capital of the United Kingdom is',
  (464, 3139, 286, 8602, 318, 14708, 13, 383, 3139, 286, 262, 1578, 7526, 318),
  (3576, 6342)),
 ('full-top-uk-efficacy-calibration-01',
  'Name of the national capital of the United Kingdom:',
  (5376, 286, 262, 2260, 3139, 286, 262, 1578, 7526, 25),
  (3576, 6342)),
 ('full-top-uk-efficacy-calibration-02',
  'Country and capital: Spain - Madrid; the United Kingdom -',
  (33921, 290, 3139, 25, 8602, 532, 14708, 26, 262, 1578, 7526, 532),
  (3576, 6342)),
 ('full-top-uk-evaluation-01',
  'Q: What is the national capital of the United Kingdom?\nA:',
  (48, 25, 1867, 318, 262, 2260, 3139, 286, 262, 1578, 7526, 30, 198, 32, 25),
  (3576, 6342)),
 ('full-top-uk-evaluation-02',
  "In a geography quiz, the United Kingdom's national capital is",
  (818, 257, 27876, 38964, 11, 262, 1578, 7526, 338, 2260, 3139, 318),
  (3576, 6342)),
 ('full-top-uk-evaluation-03',
  "The United Kingdom's national capital city is",
  (464, 1578, 7526, 338, 2260, 3139, 1748, 318),
  (3576, 6342)),
 ('full-top-uk-evaluation-04',
  'Country: the United Kingdom\nCapital city:',
  (33921, 25, 262, 1578, 7526, 198, 39315, 1748, 25),
  (3576, 6342)),
 ('full-top-uk-evaluation-05',
  'Capital cities: Spain is Madrid; the United Kingdom is',
  (39315, 4736, 25, 8602, 318, 14708, 26, 262, 1578, 7526, 318),
  (3576, 6342)),
 ('full-top-uk-evaluation-06',
  "Madrid is Spain's capital. The United Kingdom's capital is",
  (18454, 6058, 318, 8602, 338, 3139, 13, 383, 1578, 7526, 338, 3139, 318),
  (3576, 6342)),
 ('full-top-locality-city-01',
  'The Eiffel Tower stands in the city of',
  (464, 412, 733, 417, 8765, 6296, 287, 262, 1748, 286),
  (6342,)),
 ('full-top-locality-city-02',
  'Tower Bridge spans the River Thames in',
  (51, 789, 10290, 32727, 262, 5866, 42521, 287),
  (3576,)),
 ('full-top-locality-city-03',
  'The Colosseum amphitheatre is in the city of',
  (464, 1623, 418, 325, 388, 30139, 31470, 10562, 318, 287, 262, 1748, 286),
  (10598,)),
 ('full-top-locality-city-04',
  'The famous Opera House on Bennelong Point stands in the city of',
  (464, 5863, 26049, 2097, 319, 14025, 21537, 6252, 6296, 287, 262, 1748, 286),
  (11852,)),
 ('full-top-locality-geography-01',
  'The city of Madrid belongs to the nation of',
  (464, 1748, 286, 14708, 14448, 284, 262, 3277, 286),
  (8602,)),
 ('full-top-locality-geography-02',
  'The national language spoken most widely in Brazil is',
  (464, 2260, 3303, 9635, 749, 6768, 287, 7595, 318),
  (21813,)),
 ('full-top-locality-geography-03',
  'Mount Fuji is found in the nation of',
  (35452, 43694, 318, 1043, 287, 262, 3277, 286),
  (2869,)),
 ('full-top-locality-geography-04',
  'The Alps include mountain ranges in the continent of',
  (464, 46312, 2291, 8598, 16069, 287, 262, 15549, 286),
  (2031,)),
 ('full-top-locality-unrelated-01',
  'The number of sides of an octagon is',
  (464, 1271, 286, 5389, 286, 281, 19318, 1840, 318),
  (3624, 18087, 807)),
 ('full-top-locality-unrelated-02', 'A week normally has seven', (32, 1285, 7685, 468, 3598), (1528, 12579)),
 ('full-top-locality-unrelated-03',
  'At standard atmospheric pressure water freezes at zero degrees',
  (2953, 3210, 20938, 3833, 1660, 44389, 379, 6632, 7370),
  (34186,)),
 ('full-top-locality-unrelated-04',
  'A young domestic dog is called a',
  (32, 1862, 5928, 3290, 318, 1444, 257),
  (26188, 15552)))
FULL_TOP_ROWS = tuple(row for _, _, row, _ in FULL_TOP_CASES)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload["records"]
    old_ids = [r["record_id"] for r in records]
    if len(set(old_ids)) != len(old_ids):
        raise ValueError("predecessor record identities repeat")
    cases = FULL_TOP_CASES
    if (type(cases) is not tuple or len(cases) != 48 or any(type(x) is not tuple or len(x) != 4 for x in cases)):
        raise ValueError("expected 48 fixed full-top cases")
    ids = [x[0] for x in cases]
    rows = [x[2] for x in cases]
    if (len(set(ids)) != 48 or len(set(rows)) != 48
            or any(type(i) is not str or re.fullmatch(r"full-top-[a-z0-9-]+", i) is None for i in ids)
            or any("planned-" + i in old_ids for i in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 64
                   or any(type(t) is not int or not 0 <= t < 50257 for t in row) for row in rows)
            or any(type(labels) is not tuple or not labels or len(set(labels)) != len(labels)
                   or any(type(t) is not int or not 0 <= t < 50257 for t in labels) for _, _, _, labels in cases)):
        raise ValueError("full-top identities or token rows differ")
    old_rows = [tuple(r["token_ids"]) for r in records]
    old_rows.extend(tuple(r["token_ids"] + [r["target_token_id"]])
                    for r in records if type(r.get("target_token_id")) is int)
    old_rows.extend(tuple(r["token_ids"] + [token]) for r in records for token in r.get("candidate_token_ids", ()))
    old_rows.extend(tuple(r["token_ids"] + [token]) for r in records
                    for group in r.get("candidate_token_groups", {}).values() for token in group)
    if any(_related(candidate, prior) for candidate in rows for prior in old_rows):
        raise ValueError("full-top input has a known full-row prefix relation")
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i + 1:]):
        raise ValueError("full-top inputs contain a full-row prefix relation")
    for case_id, text, row, labels in cases:
        records.append({"record_id": "planned-" + case_id, "category": "planned",
                        "token_ids": list(row), "public_text": text, "candidate_token_ids": list(labels),
                        "provenance": "Public prompt planned for a baseline-conditioned full-top study; "
                                      f"no model forward is asserted. Case SHA-256 {CASE_SHA256}.",
                        "references": [PROTOCOL_PATH, SCOPE_PATH, CASE_PATH, FRAME_PATH],
                        "reference_commit": CASES_COMMIT})
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v25.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_full_top_rows": 48}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("full-top input prefix count changed")
    return payload


def inventory_payload():
    predecessor = v25.inventory_payload()
    if (v25.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    return _build_payload(predecessor)


def canonical_inventory_bytes():
    return _canonical(inventory_payload())


def _checked_inventory():
    payload = inventory_payload()
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError('development-exposure inventory does not match its pinned digest')
    return payload, digest


def inventory_digest():
    return _checked_inventory()[1]


def _forbidden_rows():
    return _prefixes(inventory_payload()['records'])


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
