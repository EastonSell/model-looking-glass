"""Reserve 16 corrected public agreement-head inputs; no forward is asserted."""
import copy
import hashlib
import json
import re
from pathlib import Path

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v41 as public_sequences
from . import pretrained_exposures_v43 as prior

INVENTORY_VERSION = "distilgpt2-development-exposures-v44"
PREDECESSOR_SHA256 = "61a80aafd9d4aea075774776d830b017f8e4c5e1a4fd0f2643963d2075a6e4ee"
INVENTORY_SHA256 = "78d783dc37c1ace4c39300b62e67652d76f3323de79830628ecd64e46aa0bfa0"
FORBIDDEN_PREFIX_COUNT = 9165
CASE_COMMIT = "e179b439645d9f778ab338be89427828d3940c12"
CASE_PATH = "experiments/distilgpt2_agreement_heads_fresh_context_v4_1_cases.json"
CASE_SHA256 = "bfc93dbcf33f06c8266b0fabf351dfa9f17ab98d6501ca7380e2cd171bddc8b4"
REFERENCES = ("docs/DISTILGPT2_AGREEMENT_HEADS_FRESH_CONTEXT_V4_1.md", CASE_PATH)
ANSWER_IDS = (318, 389, 373, 547)
ANSWER_SUFFIXES = (" is", " are", " was", " were")
for _name in dir(prior):
    if _name.isupper() and _name not in globals():
        globals()[_name] = getattr(prior, _name)
_canonical, _prefixes = prior._canonical, prior._prefixes


def _related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload["records"]
    old_ids = {record["record_id"] for record in records}
    if len(old_ids) != len(records):
        raise ValueError("predecessor identities repeat")
    raw = (Path(__file__).resolve().parents[1] / CASE_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASE_SHA256:
        raise ValueError("fixed corrected case bytes differ")
    manifest = json.loads(raw)
    cases = manifest["cases"]
    if (manifest.get("schema") != "distilgpt2-agreement-head-fresh-context-cases-v4-1"
            or manifest.get("protocol_commit") != "9ee1e5ce84f4fa0cd31fdba12bc86160ee75dfdd"
            or manifest.get("model_forward_asserted") is not False or len(cases) != 16
            or len({case["case_id"] for case in cases}) != 16):
        raise ValueError("fixed corrected case identity differs")
    groups = []
    for index, case in enumerate(cases):
        row = tuple(case["token_ids"])
        identity = f"planned-agreement-head-v4-{index + 1:02d}"
        if (identity in old_ids or case["frame"] != index // 4
                or case["case_id"] != f"fresh-head-v4-1-{index // 4 + 1:02d}-{index % 4 + 1:02d}"
                or type(case["text"]) is not str or not case["text"] or len(case["text"]) > 256
                or not 1 <= len(row) <= 12 or any(type(v) is not int or not 0 <= v < 50257 for v in row)):
            raise ValueError("case order, identity, text or token row differs")
        continuations = case["continuations"]
        if (type(continuations) is not list or len(continuations) != len(ANSWER_IDS)
                or any(item.get("suffix") != suffix or item.get("token_id") != token
                       or item.get("tokens") != list(row) + [token]
                       for item, suffix, token in zip(continuations, ANSWER_SUFFIXES, ANSWER_IDS, strict=True))):
            raise ValueError("fixed candidate continuations differ")
        groups.append([row] + [row + (token,) for token in ANSWER_IDS])
    old = public_sequences._public_sequences(records)
    if any(_related(row, previous) for group in groups for row in group for previous in old):
        raise ValueError("corrected case or continuation has a known public prefix relation")
    if any(_related(left, right) for i, group in enumerate(groups) for other in groups[i+1:]
           for left in group for right in other):
        raise ValueError("distinct corrected cases share a public prefix relation")
    for index, case in enumerate(cases):
        records.append({"record_id": f"planned-agreement-head-v4-{index + 1:02d}", "category": "planned",
                        "token_ids": list(case["token_ids"]), "candidate_token_ids": list(ANSWER_IDS),
                        "public_text": case["text"], "reference_commit": CASE_COMMIT,
                        "references": list(REFERENCES),
                        "provenance": "Corrected public fresh-context agreement-head proposal; "
                                      "no model forward is asserted. " f"Case SHA-256 {CASE_SHA256}."})
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": prior.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_agreement_head_fresh_context_rows": 16}
    if FORBIDDEN_PREFIX_COUNT and len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("corrected case prefix count differs")
    return payload


def inventory_payload():
    predecessor = prior.inventory_payload()
    if (prior.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError('predecessor exposure inventory does not match')
    return _build_payload(predecessor)


def canonical_inventory_bytes():
    return _canonical(inventory_payload())


def _checked_inventory():
    payload = inventory_payload()
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError('exposure inventory does not match its pinned digest')
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
    if (type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest
            or type(cases_per_split) is not int or cases_per_split not in (8, 16)):
        raise ValueError('expected inventory or split size differs')
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _prefixes(payload['records'])
    complete_rows = {tuple(row['token_ids']) for row in payload['records']}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in discovery + heldout):
        raise ValueError('candidate sequence is in the development exclusion inventory')
    return v1.KnownSplitCheck(INVENTORY_VERSION, digest, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    payload, digest = _checked_inventory()
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest:
        raise ValueError('expected inventory digest differs')
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _prefixes(payload['records'])
    complete_rows = {tuple(row['token_ids']) for row in payload['records']}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in heldout):
        raise ValueError('held-out sequence is in the development exclusion inventory')
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, digest, 32, len(forbidden),
                                    sum(_has_known_relation(row, forbidden, complete_rows) for row in discovery))


def _validate_known_check(check):
    payload, digest = _checked_inventory()
    count = len(_prefixes(payload['records']))
    if (type(check) not in (v1.KnownSplitCheck, v2.ExposedDiscoveryCheck)
            or check.inventory_version != INVENTORY_VERSION or check.inventory_sha256 != digest
            or type(check.candidate_count) is not int or type(check.forbidden_row_count) is not int
            or check.forbidden_row_count != count or check.complete_exposure_history_certified is not False
            or check.preseal_ready is not False):
        raise ValueError('a matching known-split check is required')
    if type(check) is v1.KnownSplitCheck and check.scope == 'known-rows-only' and check.candidate_count in (16, 32):
        return count, 'known-prefixes-excluded-from-both', 0
    if (type(check) is v2.ExposedDiscoveryCheck and check.scope == 'known-prefix-holdout-only'
            and check.candidate_count == 32 and type(check.known_or_reserved_discovery_overlap_count) is int
            and 0 <= check.known_or_reserved_discovery_overlap_count <= 16):
        return count, 'known-prefix-discovery-reuse-disclosed', check.known_or_reserved_discovery_overlap_count
    raise ValueError('a matching known-split check is required')


def require_preseal_readiness(check):
    _validate_known_check(check)
    raise v1.ExposureInventoryIncomplete('known-row checking alone does not authorize a study seal')


def prepare_preseal_disclosure(check, *, protocol_sha256, acknowledge_prior_exposure_uncertainty=False):
    count, policy, overlap = _validate_known_check(check)
    if acknowledge_prior_exposure_uncertainty is not True:
        raise v1.ExposureInventoryIncomplete('prior exposure uncertainty must be explicitly disclosed')
    if type(protocol_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', protocol_sha256) is None:
        raise ValueError('a canonical protocol SHA-256 is required')
    return v2.PresealDisclosure(INVENTORY_VERSION, check.inventory_sha256, protocol_sha256,
                                check.candidate_count // 2, count, policy, overlap)
