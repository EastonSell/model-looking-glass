"""Record public greater-than exposures and reserve pairs/dose inputs."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v6 as v6
from . import pretrained_exposures_v14 as v14

INVENTORY_VERSION = "distilgpt2-development-exposures-v15"
PREDECESSOR_SHA256 = "236e4ab86b006998cbf39096edf78222cc8506c3f2b81d5e7ae70b2a32b30d86"
INVENTORY_SHA256 = "6c6ed34755dd9daebdde9a2de07f65c0c5d20301656416d873f321fa2c146d34"
FORBIDDEN_PREFIX_COUNT = 2959
TARGET = v14.TARGET
PUBLIC_DISCOVERY = v14.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v14.FACTORIAL_GROUPS
FACTORIAL_ROWS = v14.FACTORIAL_ROWS
RECIPIENT_CASES = v14.RECIPIENT_CASES
INJECTION_RECORD_IDS = v14.INJECTION_RECORD_IDS
PROTOCOL_COMMIT = "8be210ce88a536dc9efcd769e40a8175b9beddfa"
CASES_COMMIT = "66ac1d16da70cc7f8c39399181e983e4bc944e32"
CASES_PATHS = ("docs/DISTILGPT2_COPY_HEAD_PAIRS_DOSE_V1.md",
               "experiments/distilgpt2_copy_head_pairs_dose_v1.json")
PROTOCOL_SHA256 = "811c18baef640e08ccff7f6cade26d041672edda969ddbc7a6babdb60d70bddb"
CASES_SHA256 = "3e5d40fff832a6976f5e4f8d3bda916ab203ed93936463a1a0726e88dcab1106"
COMPLETE_ROWS_SHA256 = "66451dc7cc822b933a9ad5b49348760e65ed35566cf359c8405b6954e8a3e688"
PAIRS_DOSE_GROUPS = (
    (13066, 4252, 33779, 39385, 19999, 25872),
    (26210, 31377, 19450, 38999, 10717, 8478),
    (6842, 17481, 21831, 20096, 37246, 37735),
    (8566, 32132, 16853, 26600, 10601, 47009),
    (4928, 20562, 12505, 10580, 9351, 36856),
    (7771, 20422, 30666, 31861, 19783, 1085),
    (2342, 11845, 13342, 3975, 1994, 5793),
    (6290, 32405, 21682, 4771, 2344, 6388),
)
PAIRS_DOSE_ROWS = tuple(tuple(group[i] for i in permutation + (0, 1, 2, 3, 4))
                        for group in PAIRS_DOSE_GROUPS for permutation in v6.PUBLIC_COPY_PERMUTATIONS)
PAIRS_DOSE_TEXTS = (
    ' orbit sun astronaut cosmos meteor eclipse orbit sun astronaut cosmos meteor',
    ' meteor astronaut orbit cosmos sun eclipse orbit sun astronaut cosmos meteor',
    ' sun cosmos meteor astronaut orbit eclipse orbit sun astronaut cosmos meteor',
    ' cosmos orbit meteor sun astronaut eclipse orbit sun astronaut cosmos meteor',
    ' oak maple olive thorn stem branch oak maple olive thorn stem',
    ' stem olive oak thorn maple branch oak maple olive thorn stem',
    ' maple thorn stem olive oak branch oak maple olive thorn stem',
    ' thorn oak stem maple olive branch oak maple olive thorn stem',
    ' bear wolf fox deer goose raven bear wolf fox deer goose',
    ' goose fox bear deer wolf raven bear wolf fox deer goose',
    ' wolf deer goose fox bear raven bear wolf fox deer goose',
    ' deer bear goose wolf fox raven bear wolf fox deer goose',
    ' cream yogurt jam vinegar flour oats cream yogurt jam vinegar flour',
    ' flour jam cream vinegar yogurt oats cream yogurt jam vinegar flour',
    ' yogurt vinegar flour jam cream oats cream yogurt jam vinegar flour',
    ' vinegar cream flour yogurt jam oats cream yogurt jam vinegar flour',
    ' church palace temple tower cabin cottage church palace temple tower cabin',
    ' cabin temple church tower palace cottage church palace temple tower cabin',
    ' palace tower cabin temple church cottage church palace temple tower cabin',
    ' tower church cabin palace temple cottage church palace temple tower cabin',
    ' steel brass nickel zinc tin lead steel brass nickel zinc tin',
    ' tin nickel steel zinc brass lead steel brass nickel zinc tin',
    ' brass zinc tin nickel steel lead steel brass nickel zinc tin',
    ' zinc steel tin brass nickel lead steel brass nickel zinc tin',
    ' watch calendar globe map key lock watch calendar globe map key',
    ' key globe watch map calendar lock watch calendar globe map key',
    ' calendar map key globe watch lock watch calendar globe map key',
    ' map watch key calendar globe lock watch calendar globe map key',
    ' rain hail frost ice wind storm rain hail frost ice wind',
    ' wind frost rain ice hail storm rain hail frost ice wind',
    ' hail ice wind frost rain storm rain hail frost ice wind',
    ' ice rain wind hail frost storm rain hail frost ice wind',
)
GREATER_SOURCE_COMMIT = "47d42b8137203fa8e5ffdc60942a1adece6c0feb"
GREATER_EVIDENCE_COMMIT = "b211bcbfc85f59167daee2d4c5ad5d822e32e834"
GREATER_EVIDENCE_PATHS = (
    "experiments/distilgpt2_greater_than_screen_v1.json",
    "results/distilgpt2-greater-than-screen-v1-20261028/run/manifest.json",
    "results/distilgpt2-greater-than-screen-v1-20261028/run/result.json",
    "results/distilgpt2-greater-than-screen-v1-20261028/run/attempt.jsonl",
    "results/distilgpt2-greater-than-screen-v1-20261028/run/last_logits.f32",
    "results/distilgpt2-greater-than-screen-v1-20261028/supervisor.json",
)
GREATER_CASES_SHA256 = "399994789ee4d3eff27dd6c9cce23ae27ab7761e4bc33caf58de4528194b01d2"
GREATER_MANIFEST_SHA256 = "5590f02a6becd05daffb0915088bbe66e41b3cc8dce2310030fb87162d63ee1c"
GREATER_RESULT_SHA256 = "65e48fdd8bfdb2a69455037e5ad059731106793014c366b06896252bb35970ef"
GREATER_JOURNAL_SHA256 = "f280ef79bdd6b0d6aea66d8c35e0868d4a18193bc5ebb7654f20b25f2b82ac3a"
GREATER_LOGITS_SHA256 = "e2cd020bc219d1e042fea90ecd098d95c0ccbf653e3526cfab1d6c39cb9b4737"
GREATER_RECEIPT_SHA256 = "469b61cf70916434d2e96779ba842ba1dd9e2c50be9fd511a76ac8d3894bcd48"
GREATER_THAN_CASES = (
    ('greater-12-war-forward', 'The war lasted from the year 1712 to the year 17', (464, 1175, 15436, 422, 262, 614, 1596, 1065, 284, 262, 614, 1596)),
    ('greater-12-war-reverse', 'The war lasted until the year 1712, after starting in the year 17', (464, 1175, 15436, 1566, 262, 614, 1596, 1065, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-12-war-neutral', 'The war was recorded in the year 1712 beside the year 17', (464, 1175, 373, 6264, 287, 262, 614, 1596, 1065, 13970, 262, 614, 1596)),
    ('greater-12-reign-forward', 'The reign lasted from the year 1712 to the year 17', (464, 13580, 15436, 422, 262, 614, 1596, 1065, 284, 262, 614, 1596)),
    ('greater-12-reign-reverse', 'The reign lasted until the year 1712, after starting in the year 17', (464, 13580, 15436, 1566, 262, 614, 1596, 1065, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-12-reign-neutral', 'The reign was recorded in the year 1712 beside the year 17', (464, 13580, 373, 6264, 287, 262, 614, 1596, 1065, 13970, 262, 614, 1596)),
    ('greater-21-war-forward', 'The war lasted from the year 1721 to the year 17', (464, 1175, 15436, 422, 262, 614, 1596, 2481, 284, 262, 614, 1596)),
    ('greater-21-war-reverse', 'The war lasted until the year 1721, after starting in the year 17', (464, 1175, 15436, 1566, 262, 614, 1596, 2481, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-21-war-neutral', 'The war was recorded in the year 1721 beside the year 17', (464, 1175, 373, 6264, 287, 262, 614, 1596, 2481, 13970, 262, 614, 1596)),
    ('greater-21-reign-forward', 'The reign lasted from the year 1721 to the year 17', (464, 13580, 15436, 422, 262, 614, 1596, 2481, 284, 262, 614, 1596)),
    ('greater-21-reign-reverse', 'The reign lasted until the year 1721, after starting in the year 17', (464, 13580, 15436, 1566, 262, 614, 1596, 2481, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-21-reign-neutral', 'The reign was recorded in the year 1721 beside the year 17', (464, 13580, 373, 6264, 287, 262, 614, 1596, 2481, 13970, 262, 614, 1596)),
    ('greater-32-war-forward', 'The war lasted from the year 1732 to the year 17', (464, 1175, 15436, 422, 262, 614, 1596, 2624, 284, 262, 614, 1596)),
    ('greater-32-war-reverse', 'The war lasted until the year 1732, after starting in the year 17', (464, 1175, 15436, 1566, 262, 614, 1596, 2624, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-32-war-neutral', 'The war was recorded in the year 1732 beside the year 17', (464, 1175, 373, 6264, 287, 262, 614, 1596, 2624, 13970, 262, 614, 1596)),
    ('greater-32-reign-forward', 'The reign lasted from the year 1732 to the year 17', (464, 13580, 15436, 422, 262, 614, 1596, 2624, 284, 262, 614, 1596)),
    ('greater-32-reign-reverse', 'The reign lasted until the year 1732, after starting in the year 17', (464, 13580, 15436, 1566, 262, 614, 1596, 2624, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-32-reign-neutral', 'The reign was recorded in the year 1732 beside the year 17', (464, 13580, 373, 6264, 287, 262, 614, 1596, 2624, 13970, 262, 614, 1596)),
    ('greater-43-war-forward', 'The war lasted from the year 1743 to the year 17', (464, 1175, 15436, 422, 262, 614, 1596, 3559, 284, 262, 614, 1596)),
    ('greater-43-war-reverse', 'The war lasted until the year 1743, after starting in the year 17', (464, 1175, 15436, 1566, 262, 614, 1596, 3559, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-43-war-neutral', 'The war was recorded in the year 1743 beside the year 17', (464, 1175, 373, 6264, 287, 262, 614, 1596, 3559, 13970, 262, 614, 1596)),
    ('greater-43-reign-forward', 'The reign lasted from the year 1743 to the year 17', (464, 13580, 15436, 422, 262, 614, 1596, 3559, 284, 262, 614, 1596)),
    ('greater-43-reign-reverse', 'The reign lasted until the year 1743, after starting in the year 17', (464, 13580, 15436, 1566, 262, 614, 1596, 3559, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-43-reign-neutral', 'The reign was recorded in the year 1743 beside the year 17', (464, 13580, 373, 6264, 287, 262, 614, 1596, 3559, 13970, 262, 614, 1596)),
    ('greater-54-war-forward', 'The war lasted from the year 1754 to the year 17', (464, 1175, 15436, 422, 262, 614, 1596, 4051, 284, 262, 614, 1596)),
    ('greater-54-war-reverse', 'The war lasted until the year 1754, after starting in the year 17', (464, 1175, 15436, 1566, 262, 614, 1596, 4051, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-54-war-neutral', 'The war was recorded in the year 1754 beside the year 17', (464, 1175, 373, 6264, 287, 262, 614, 1596, 4051, 13970, 262, 614, 1596)),
    ('greater-54-reign-forward', 'The reign lasted from the year 1754 to the year 17', (464, 13580, 15436, 422, 262, 614, 1596, 4051, 284, 262, 614, 1596)),
    ('greater-54-reign-reverse', 'The reign lasted until the year 1754, after starting in the year 17', (464, 13580, 15436, 1566, 262, 614, 1596, 4051, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-54-reign-neutral', 'The reign was recorded in the year 1754 beside the year 17', (464, 13580, 373, 6264, 287, 262, 614, 1596, 4051, 13970, 262, 614, 1596)),
    ('greater-65-war-forward', 'The war lasted from the year 1765 to the year 17', (464, 1175, 15436, 422, 262, 614, 1596, 2996, 284, 262, 614, 1596)),
    ('greater-65-war-reverse', 'The war lasted until the year 1765, after starting in the year 17', (464, 1175, 15436, 1566, 262, 614, 1596, 2996, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-65-war-neutral', 'The war was recorded in the year 1765 beside the year 17', (464, 1175, 373, 6264, 287, 262, 614, 1596, 2996, 13970, 262, 614, 1596)),
    ('greater-65-reign-forward', 'The reign lasted from the year 1765 to the year 17', (464, 13580, 15436, 422, 262, 614, 1596, 2996, 284, 262, 614, 1596)),
    ('greater-65-reign-reverse', 'The reign lasted until the year 1765, after starting in the year 17', (464, 13580, 15436, 1566, 262, 614, 1596, 2996, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-65-reign-neutral', 'The reign was recorded in the year 1765 beside the year 17', (464, 13580, 373, 6264, 287, 262, 614, 1596, 2996, 13970, 262, 614, 1596)),
    ('greater-76-war-forward', 'The war lasted from the year 1776 to the year 17', (464, 1175, 15436, 422, 262, 614, 1596, 4304, 284, 262, 614, 1596)),
    ('greater-76-war-reverse', 'The war lasted until the year 1776, after starting in the year 17', (464, 1175, 15436, 1566, 262, 614, 1596, 4304, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-76-war-neutral', 'The war was recorded in the year 1776 beside the year 17', (464, 1175, 373, 6264, 287, 262, 614, 1596, 4304, 13970, 262, 614, 1596)),
    ('greater-76-reign-forward', 'The reign lasted from the year 1776 to the year 17', (464, 13580, 15436, 422, 262, 614, 1596, 4304, 284, 262, 614, 1596)),
    ('greater-76-reign-reverse', 'The reign lasted until the year 1776, after starting in the year 17', (464, 13580, 15436, 1566, 262, 614, 1596, 4304, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-76-reign-neutral', 'The reign was recorded in the year 1776 beside the year 17', (464, 13580, 373, 6264, 287, 262, 614, 1596, 4304, 13970, 262, 614, 1596)),
    ('greater-87-war-forward', 'The war lasted from the year 1787 to the year 17', (464, 1175, 15436, 422, 262, 614, 1596, 5774, 284, 262, 614, 1596)),
    ('greater-87-war-reverse', 'The war lasted until the year 1787, after starting in the year 17', (464, 1175, 15436, 1566, 262, 614, 1596, 5774, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-87-war-neutral', 'The war was recorded in the year 1787 beside the year 17', (464, 1175, 373, 6264, 287, 262, 614, 1596, 5774, 13970, 262, 614, 1596)),
    ('greater-87-reign-forward', 'The reign lasted from the year 1787 to the year 17', (464, 13580, 15436, 422, 262, 614, 1596, 5774, 284, 262, 614, 1596)),
    ('greater-87-reign-reverse', 'The reign lasted until the year 1787, after starting in the year 17', (464, 13580, 15436, 1566, 262, 614, 1596, 5774, 11, 706, 3599, 287, 262, 614, 1596)),
    ('greater-87-reign-neutral', 'The reign was recorded in the year 1787 beside the year 17', (464, 13580, 373, 6264, 287, 262, 614, 1596, 5774, 13970, 262, 614, 1596)),
)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def inventory_payload():
    predecessor = v14.inventory_payload()
    canonical = json.dumps(predecessor, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    if (v14.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(canonical).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v14.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_public_greater_than_rows": 48,
                         "planned_public_pairs_dose_rows": 32}
    if (len(GREATER_THAN_CASES) != 48 or len({case[0] for case in GREATER_THAN_CASES}) != 48
            or len({case[2] for case in GREATER_THAN_CASES}) != 48):
        raise ValueError("greater-than cases must have 48 distinct IDs and inputs")
    old_rows = [tuple(row["token_ids"]) for row in payload["records"]]
    gt_rows = [case[2] for case in GREATER_THAN_CASES]
    if any(_related(row, prior) for row in gt_rows for prior in old_rows):
        raise ValueError("greater-than input has a known full-row prefix relation")
    if any(_related(left, right) for index, left in enumerate(gt_rows) for right in gt_rows[index + 1:]):
        raise ValueError("greater-than inputs contain a full-row prefix relation")
    old_ids = {row["record_id"] for row in payload["records"]}
    for case_id, text, ids in GREATER_THAN_CASES:
        record_id = "observed-" + case_id
        if (record_id in old_ids or type(text) is not str or not text or len(ids) not in (12, 13, 15)
                or any(type(token) is not int or not 0 <= token < 50257 for token in ids)):
            raise ValueError("greater-than case identity or unpadded tokens are invalid")
        payload["records"].append({
            "record_id": record_id, "category": "observed-saved", "token_ids": list(ids),
            "public_text": text,
            "provenance": "Public non-blind CPU greater-than baseline; complete 48-case forward. "
                f"Cases SHA-256 {GREATER_CASES_SHA256}, manifest SHA-256 {GREATER_MANIFEST_SHA256}, "
                f"result SHA-256 {GREATER_RESULT_SHA256}, journal SHA-256 {GREATER_JOURNAL_SHA256}, "
                f"logits SHA-256 {GREATER_LOGITS_SHA256}, process observation SHA-256 {GREATER_RECEIPT_SHA256}.",
            "references": list(GREATER_EVIDENCE_PATHS), "reference_commit": GREATER_EVIDENCE_COMMIT,
        })
    rows = list(PAIRS_DOSE_ROWS)
    tokens = [token for group in PAIRS_DOSE_GROUPS for token in group]
    expected = [tuple(group[i] for i in permutation + (0, 1, 2, 3, 4))
                for group in PAIRS_DOSE_GROUPS for permutation in v6.PUBLIC_COPY_PERMUTATIONS]
    if (len(PAIRS_DOSE_GROUPS) != 8 or any(len(group) != 6 for group in PAIRS_DOSE_GROUPS)
            or len(tokens) != 48 or len(set(tokens)) != 48
            or any(type(token) is not int or not 0 <= token < 50257 for token in tokens)):
        raise ValueError("public pairs/dose groups do not satisfy fixed token constraints")
    if len(rows) != 32 or rows != expected or len(set(rows)) != 32 or len(PAIRS_DOSE_TEXTS) != 32:
        raise ValueError("public pairs/dose rows do not match the fixed permutations")
    old_rows = [tuple(row["token_ids"]) for row in payload["records"]]
    complete = [row + (PAIRS_DOSE_GROUPS[index // 4][5],) for index, row in enumerate(rows)]
    if any(_related(left, right) for left in complete for right in old_rows):
        raise ValueError("public pairs/dose input-plus-target has a known full-row prefix relation")
    if any(_related(left, right) for index, left in enumerate(complete) for right in complete[index + 1:]):
        raise ValueError("public pairs/dose complete rows contain a full-row prefix relation")
    prefixes = {row[:n] for row in old_rows for n in range(1, len(row) + 1)}
    if any(row[:n] in prefixes for row in complete for n in range(1, len(row) + 1)):
        raise ValueError("public pairs/dose row has a prefix in the known inventory")
    if any((token,) in prefixes for token in tokens):
        raise ValueError("public pairs/dose group has a known one-token prefix")
    for index, row in enumerate(rows):
        group, variant = divmod(index, 4)
        record_id = f"planned-copy-pairs-dose-{group + 1:02d}-{variant}"
        if record_id in old_ids:
            raise ValueError("public pairs/dose record ID already exists")
        payload["records"].append({
            "record_id": record_id, "category": "planned", "token_ids": list(row),
            "target_token_id": PAIRS_DOSE_GROUPS[group][5], "public_text": PAIRS_DOSE_TEXTS[index],
            "provenance": "Public non-blind pairs/dose copying cases; no model forward is asserted. "
                f"Case inventory SHA-256 {CASES_SHA256}.",
            "references": list(CASES_PATHS), "reference_commit": CASES_COMMIT,
        })
    return payload


def canonical_inventory_bytes():
    return json.dumps(inventory_payload(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def inventory_digest():
    digest = hashlib.sha256(canonical_inventory_bytes()).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError("development-exposure inventory does not match its pinned digest")
    return digest


def _forbidden_rows():
    return {tuple(row["token_ids"][:length]) for row in inventory_payload()["records"]
            for length in range(1, len(row["token_ids"]) + 1)}


def _has_known_relation(row, forbidden, complete_rows):
    return row in forbidden or any(row[:len(known)] == known for known in complete_rows)


def check_known_splits(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                       expected_model_identity, cases_per_split=8):
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    if type(cases_per_split) is not int or cases_per_split not in (8, 16):
        raise ValueError("cases per split must be exactly 8 or 16")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _forbidden_rows()
    complete_rows = {tuple(row["token_ids"]) for row in inventory_payload()["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in discovery + heldout):
        raise ValueError("a candidate token sequence is in the development exclusion inventory")
    return v1.KnownSplitCheck(INVENTORY_VERSION, expected_inventory_sha256, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _forbidden_rows()
    complete_rows = {tuple(row["token_ids"]) for row in inventory_payload()["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in heldout):
        raise ValueError("a held-out token sequence is in the development exclusion inventory")
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, expected_inventory_sha256, 32, len(forbidden),
                                    sum(_has_known_relation(row, forbidden, complete_rows) for row in discovery))


def _validate_known_check(check):
    count = len(_forbidden_rows())
    if (type(check) not in (v1.KnownSplitCheck, v2.ExposedDiscoveryCheck) or
            check.inventory_version != INVENTORY_VERSION or check.inventory_sha256 != inventory_digest() or
            type(check.candidate_count) is not int or type(check.forbidden_row_count) is not int or
            check.forbidden_row_count != count or check.complete_exposure_history_certified is not False or
            check.preseal_ready is not False):
        raise ValueError("a matching known-split check is required")
    if type(check) is v1.KnownSplitCheck and check.scope == "known-rows-only" and check.candidate_count in (16, 32):
        return count, "known-prefixes-excluded-from-both", 0
    if (type(check) is v2.ExposedDiscoveryCheck and check.scope == "known-prefix-holdout-only" and
            check.candidate_count == 32 and type(check.known_or_reserved_discovery_overlap_count) is int and
            0 <= check.known_or_reserved_discovery_overlap_count <= 16):
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
