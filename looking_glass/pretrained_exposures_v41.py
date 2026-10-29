"""Reserve 192 public copying-screen inputs; no forward is asserted."""
import copy
import hashlib
import json
from pathlib import Path
import re
from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v40 as prior

INVENTORY_VERSION = "distilgpt2-development-exposures-v41"
PREDECESSOR_SHA256 = "685391b672c356dd627923681c4a93fc9a310c03a40fdd22e0b29af2ae9f7be3"
INVENTORY_SHA256 = "110ebcf674168e794544c257d158b3c3242f98bcf108375cdc073bb145e2bfad"
FORBIDDEN_PREFIX_COUNT = 9011
CASE_COMMIT = "8efc8e019be0929ca6344eca3f94de1f5c5cbdd7"
CASE_PATH = "experiments/distilgpt2_copy_behavior_screen_cases_v1.json"
CASE_SHA256 = "96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734"
REFERENCES = ("docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_V1.md", CASE_PATH)
for _name in dir(prior):
    if _name.isupper() and _name not in globals():
        globals()[_name] = getattr(prior, _name)
_canonical = prior._canonical
_prefixes = prior._prefixes


def _related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def _public_sequences(records):
    result = []
    for record in records:
        base = tuple(record["token_ids"])
        bases = [base]
        if "attention_mask" in record:
            mask = record["attention_mask"]
            if len(mask) != len(base) or any(type(x) is not int or x not in (0, 1) for x in mask):
                raise ValueError("invalid predecessor mask")
            active = tuple(v for v, present in zip(base, mask) if present)
            if active and active != base:
                bases.append(active)
        choices = []
        if "target_token_id" in record:
            choices.append(record["target_token_id"])
        choices.extend(record.get("candidate_token_ids", ()))
        choices.extend(v for group in record.get("candidate_token_groups", {}).values() for v in group)
        if any(type(v) is not int or not 0 <= v < 50257 for v in choices):
            raise ValueError("invalid predecessor continuation")
        for item in bases:
            result.append(item)
            result.extend(item + (v,) for v in choices)
        if "scored_prefix_token_ids" in record:
            prefix = tuple(record["scored_prefix_token_ids"])
            candidates = record.get("scored_candidate_token_ids", ())
            if not prefix or any(type(v) is not int or not 0 <= v < 50257 for v in prefix + tuple(candidates)):
                raise ValueError("invalid scored prefix or candidate")
            result.append(prefix)
            result.extend(prefix + (v,) for v in candidates)
        elif "scored_candidate_token_ids" in record:
            raise ValueError("scored candidates without prefix")
    return result


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload["records"]
    old_ids = {r["record_id"] for r in records}
    if len(old_ids) != len(records):
        raise ValueError("predecessor identities repeat")
    raw = (Path(__file__).resolve().parents[1] / CASE_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASE_SHA256:
        raise ValueError("case bytes differ")
    cases = json.loads(raw)["cases"]
    if len(cases) != 192 or len({c["case_id"] for c in cases}) != 192:
        raise ValueError("case count or identities differ")
    rows = []
    for index, case in enumerate(cases):
        row, target = tuple(case["token_ids"]), case["target_id"]
        if (case["group"] != index // 8 + 1 or case["variant"] != index % 8
                or case["split"] != ("development" if case["group"] % 2 else "held-out")
                or "planned-" + case["case_id"] in old_ids or len(row) != 11
                or any(type(v) is not int or not 0 <= v < 50257 for v in row + (target,))
                or type(case["text"]) is not str or not case["text"]):
            raise ValueError("case order, split, identity or tokens differ")
        rows.append((row, row + (target,)))
    old = _public_sequences(records)
    if any(_related(row, other) for group in rows for row in group for other in old):
        raise ValueError("copy case or continuation overlaps a known public sequence")
    if any(_related(left, right) for i, group in enumerate(rows) for other in rows[i+1:]
           for left in group for right in other):
        raise ValueError("distinct copying inputs share a prefix relation")
    for case in cases:
        records.append({"record_id": "planned-" + case["case_id"], "category": "planned",
                        "token_ids": list(case["token_ids"]), "target_token_id": case["target_id"],
                        "public_text": case["text"], "reference_commit": CASE_COMMIT,
                        "references": list(REFERENCES),
                        "provenance": "Public copying-behavior proposal; no model forward is asserted. "
                                      f"Case SHA-256 {CASE_SHA256}."})
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": prior.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"planned_copy_behavior_screen_rows": 192}
    if FORBIDDEN_PREFIX_COUNT and len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError("copying screen prefix count changed")
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
