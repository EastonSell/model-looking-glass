"""Append 32 fixed public two-attractor patch cases; no forward is asserted."""
import copy
import hashlib
import re
from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v31 as prior

INVENTORY_VERSION = "distilgpt2-development-exposures-v32"
PREDECESSOR_SHA256 = "2d710a40a49a84b7d664cd851fccd2527ea318c5a8c911111fe7275b7567e565"
INVENTORY_SHA256 = "4a4ed0127d18a7c5a8f6472af619b82895cb4efa6c07314a48a463510b1878b2"
FORBIDDEN_PREFIX_COUNT = 5881
CASES_COMMIT = "64d72f4ea3818ea2a33edd9db5073e37417cd5c0"
CASE_PATH = "experiments/distilgpt2_two_attractor_patch_cases_v1.json"
CASE_SHA256 = "6d0cbba21d15f9ab05df106617141c25a460f07be7ee57c53470f969eb79e518"
REFERENCES = ("docs/PUBLIC_TWO_ATTRACTOR_PATCH_V1.md", CASE_PATH)
for _name in ('TARGET', 'PUBLIC_DISCOVERY', 'FACTORIAL_GROUPS', 'FACTORIAL_ROWS', 'RECIPIENT_CASES', 'INJECTION_RECORD_IDS', 'PAIRS_DOSE_GROUPS', 'PAIRS_DOSE_ROWS', 'YEAR_TRANSFER_CASES', 'YEAR_TRANSFER_ROWS', 'GROUPED_REPORTED_ROWS', 'GROUPED_POLICY_ROWS', 'GROUPED_BROKER_OBSERVED_ROWS', 'SCREENED_CASES', 'SCREENED_ROWS', 'SCREENED_OBSERVED_CASES', 'SCREENED_PLANNED_LOCALITY_CASES', 'AGREEMENT_CASES', 'AGREEMENT_ROWS', 'AGREEMENT_ANSWER_IDS', 'EXPLICIT_YEAR_DIRECTION_CASES', 'EXPLICIT_YEAR_DIRECTION_ROWS', 'YEAR_LABEL_CONTROL_CASES', 'YEAR_LABEL_CONTROL_ROWS', 'FULL_TOP_CASES', 'FULL_TOP_ROWS', 'AGREEMENT_SUBJECT_FOLLOWUP_ANSWER_IDS', 'AGREEMENT_SUBJECT_FOLLOWUP_CASES', 'AGREEMENT_SUBJECT_FOLLOWUP_ROWS', 'ANSWER_FORMAT_CASES', 'ANSWER_FORMAT_ROWS', 'TWO_ATTRACTOR_CASES', 'TWO_ATTRACTOR_ROWS', 'COPY_PROMPT_DOSE_CASES', 'COPY_PROMPT_DOSE_ROWS', 'SYNTHETIC_SERVICE_CASES'):
    globals()[_name] = getattr(prior, _name)
_canonical = prior._canonical
_prefixes = prior._prefixes
TWO_ATTRACTOR_PATCH_CASES = (('i1-o0-n000', 'The plate near the crate beside the shelf', (464, 7480, 1474, 262, 27021, 13970, 262, 18316)),
 ('i1-o0-n001', 'The plate near the crate beside the shelves', (464, 7480, 1474, 262, 27021, 13970, 262, 20726)),
 ('i1-o0-n010', 'The plate near the crates beside the shelf', (464, 7480, 1474, 262, 40382, 13970, 262, 18316)),
 ('i1-o0-n011', 'The plate near the crates beside the shelves', (464, 7480, 1474, 262, 40382, 13970, 262, 20726)),
 ('i1-o0-n100', 'The plates near the crate beside the shelf', (464, 13375, 1474, 262, 27021, 13970, 262, 18316)),
 ('i1-o0-n101', 'The plates near the crate beside the shelves', (464, 13375, 1474, 262, 27021, 13970, 262, 20726)),
 ('i1-o0-n110', 'The plates near the crates beside the shelf', (464, 13375, 1474, 262, 40382, 13970, 262, 18316)),
 ('i1-o0-n111',
  'The plates near the crates beside the shelves',
  (464, 13375, 1474, 262, 40382, 13970, 262, 20726)),
 ('i1-o1-n000', 'The plate near the shelf beside the crate', (464, 7480, 1474, 262, 18316, 13970, 262, 27021)),
 ('i1-o1-n001', 'The plate near the shelf beside the crates', (464, 7480, 1474, 262, 18316, 13970, 262, 40382)),
 ('i1-o1-n010', 'The plate near the shelves beside the crate', (464, 7480, 1474, 262, 20726, 13970, 262, 27021)),
 ('i1-o1-n011', 'The plate near the shelves beside the crates', (464, 7480, 1474, 262, 20726, 13970, 262, 40382)),
 ('i1-o1-n100', 'The plates near the shelf beside the crate', (464, 13375, 1474, 262, 18316, 13970, 262, 27021)),
 ('i1-o1-n101', 'The plates near the shelf beside the crates', (464, 13375, 1474, 262, 18316, 13970, 262, 40382)),
 ('i1-o1-n110', 'The plates near the shelves beside the crate', (464, 13375, 1474, 262, 20726, 13970, 262, 27021)),
 ('i1-o1-n111',
  'The plates near the shelves beside the crates',
  (464, 13375, 1474, 262, 20726, 13970, 262, 40382)),
 ('i2-o0-n000', 'The box near the crate beside the shelf', (464, 3091, 1474, 262, 27021, 13970, 262, 18316)),
 ('i2-o0-n001', 'The box near the crate beside the shelves', (464, 3091, 1474, 262, 27021, 13970, 262, 20726)),
 ('i2-o0-n010', 'The box near the crates beside the shelf', (464, 3091, 1474, 262, 40382, 13970, 262, 18316)),
 ('i2-o0-n011', 'The box near the crates beside the shelves', (464, 3091, 1474, 262, 40382, 13970, 262, 20726)),
 ('i2-o0-n100', 'The boxes near the crate beside the shelf', (464, 10559, 1474, 262, 27021, 13970, 262, 18316)),
 ('i2-o0-n101', 'The boxes near the crate beside the shelves', (464, 10559, 1474, 262, 27021, 13970, 262, 20726)),
 ('i2-o0-n110', 'The boxes near the crates beside the shelf', (464, 10559, 1474, 262, 40382, 13970, 262, 18316)),
 ('i2-o0-n111', 'The boxes near the crates beside the shelves', (464, 10559, 1474, 262, 40382, 13970, 262, 20726)),
 ('i2-o1-n000', 'The box near the shelf beside the crate', (464, 3091, 1474, 262, 18316, 13970, 262, 27021)),
 ('i2-o1-n001', 'The box near the shelf beside the crates', (464, 3091, 1474, 262, 18316, 13970, 262, 40382)),
 ('i2-o1-n010', 'The box near the shelves beside the crate', (464, 3091, 1474, 262, 20726, 13970, 262, 27021)),
 ('i2-o1-n011', 'The box near the shelves beside the crates', (464, 3091, 1474, 262, 20726, 13970, 262, 40382)),
 ('i2-o1-n100', 'The boxes near the shelf beside the crate', (464, 10559, 1474, 262, 18316, 13970, 262, 27021)),
 ('i2-o1-n101', 'The boxes near the shelf beside the crates', (464, 10559, 1474, 262, 18316, 13970, 262, 40382)),
 ('i2-o1-n110', 'The boxes near the shelves beside the crate', (464, 10559, 1474, 262, 20726, 13970, 262, 27021)),
 ('i2-o1-n111', 'The boxes near the shelves beside the crates', (464, 10559, 1474, 262, 20726, 13970, 262, 40382)))
TWO_ATTRACTOR_PATCH_ROWS = tuple(row for _, _, row in TWO_ATTRACTOR_PATCH_CASES)

def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload["records"]
    old_ids = [r["record_id"] for r in records]
    if len(set(old_ids)) != len(old_ids):
        raise ValueError("predecessor record identities repeat")
    cases = TWO_ATTRACTOR_PATCH_CASES
    if (type(cases) is not tuple or len(cases) != 32
            or any(type(x) is not tuple or len(x) != 3 for x in cases)):
        raise ValueError("expected 32 fixed patch cases")
    ids = [x[0] for x in cases]
    rows = [x[2] for x in cases]
    if (len(set(ids)) != 32 or len(set(rows)) != 32
            or any(type(i) is not str or re.fullmatch(r"i[1-2]-o[01]-n[01]{3}", i) is None for i in ids)
            or any("planned-two-attractor-patch-" + i in old_ids for i in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _ in cases)
            or any(type(row) is not tuple or len(row) != 8
                   or any(type(t) is not int or not 0 <= t < 50257 for t in row) for row in rows)):
        raise ValueError("patch identities or token rows differ")
    old_rows = [tuple(r["token_ids"]) for r in records]
    old_rows.extend(tuple(r["token_ids"] + [r["target_token_id"]]) for r in records if type(r.get("target_token_id")) is int)
    old_rows.extend(tuple(r["token_ids"] + [token]) for r in records for token in r.get("candidate_token_ids", ()))
    old_rows.extend(tuple(r["token_ids"] + [token]) for r in records
                    for group in r.get("candidate_token_groups", {}).values() for token in group)
    if any(_related(candidate, old) for candidate in rows for old in old_rows):
        raise ValueError("patch input has a known full-row prefix relation")
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i+1:]):
        raise ValueError("patch inputs contain a full-row prefix relation")
    for case_id, text, row in cases:
        records.append({"record_id":"planned-two-attractor-patch-" + case_id,"category":"planned","case_role":"evaluation",
                        "token_ids":list(row),"public_text":text,"candidate_token_ids":[318,389,373,547],
                        "provenance":"Public prompt planned for a two-attractor activation-patching follow-up; "
                                     f"no model forward is asserted. Case SHA-256 {CASE_SHA256}.",
                        "references":list(REFERENCES),"reference_commit":CASES_COMMIT})
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version":prior.INVENTORY_VERSION,"sha256":PREDECESSOR_SHA256}
    payload["update"] = {"planned_two_attractor_patch_rows":32}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("patch input prefix count changed")
    return payload


def inventory_payload():
    predecessor = prior.inventory_payload()
    if (prior.INVENTORY_SHA256 != PREDECESSOR_SHA256
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
