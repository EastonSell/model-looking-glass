"""Record additional public injection use of eight already-observed rows."""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v13 as v13

INVENTORY_VERSION = "distilgpt2-development-exposures-v14"
PREDECESSOR_SHA256 = "27ffe8d2ab312ccf2a62a07dab835d3cafce7364ca8abadbb1e73ec835e9aca8"
INVENTORY_SHA256 = "236e4ab86b006998cbf39096edf78222cc8506c3f2b81d5e7ae70b2a32b30d86"
TARGET = v13.TARGET
PUBLIC_DISCOVERY = v13.PUBLIC_DISCOVERY
FACTORIAL_GROUPS = v13.FACTORIAL_GROUPS
FACTORIAL_ROWS = v13.FACTORIAL_ROWS
EVIDENCE_COMMIT = "40cf34fceaae66cad3e9934415e25816b1cd4c4d"
STUDY_SOURCE_COMMIT = "e84f66ae63b6f20e6df81789f2e4741b13598fb1"
PUBLIC_DIRECTORY = "results/pretrained-number-injection-discovery-v1-20261028/public"
EVIDENCE_PATHS = (
    "experiments/pretrained_number_injection_cpu_v1/plan.json",
    "results/pretrained-number-injection-discovery-v1-20261028/audit/result.json",
    "results/pretrained-number-injection-discovery-v1-20261028/audit/number-injection-public-provenance-review-v1.json",
    "results/pretrained-number-injection-discovery-v1-20261028/audit/operator-note.json",
    PUBLIC_DIRECTORY + "/prediction.json",
)
PLAN_SHA256 = "e59d98638e474ed9f915d24fa527edd6357caa0e62c8bdabdda0433c20c683f8"
AUDIT_SHA256 = "77e17994255be6ac4a3aadbe2d1335884a4efc71bb2822e7de9f0cdb2b5e6018"
PROVENANCE_SHA256 = "3b806322f1270acf4e4f22b70fa105bfdf8649a9ccad0ce027e7e98ba885d149"
OPERATOR_SHA256 = "2ee79c62c0188f95f531b63c410e935073878ff511bc03851152a1d2c0ef7327"
PREDICTION_SHA256 = "11376d8aae517db76d6a33c1ae5dc1f4b0c747a9d5216c9b02a17b1b0b300107"
ARCHIVE_SHA256 = "082fee918cffe437c5fe49732c1cf9045fba9ace39608ceca5c68efabe61c523"
INJECTION_RECORD_IDS = tuple(f"observed-agreement-discovery-{index + 1:02d}"
                             for index in (0, 1, 4, 5, 8, 9, 12, 13))


RECIPIENT_SOURCE_COMMIT = "32164cd570dea1f4bc0022c5b7f81fe17f8c2797"
RECIPIENT_EVIDENCE_COMMIT = "f29d26fec983de3d01f18319014ea8245e98f6b6"
RECIPIENT_EVIDENCE_PATHS = (
    "experiments/distilgpt2_recipient_screen_v1.json",
    "results/distilgpt2-recipient-screen-v1-20261028/run/manifest.json",
    "results/distilgpt2-recipient-screen-v1-20261028/run/result.json",
    "results/distilgpt2-recipient-screen-v1-20261028/run/attempt.jsonl",
    "results/distilgpt2-recipient-screen-v1-20261028/run/last_logits.f32",
    "results/distilgpt2-recipient-screen-v1-20261028/supervisor.json",
)
RECIPIENT_CASES_SHA256 = "df9036ece115a42851a43f2d7833865fa1258ab542e322beff16d549df27d345"
RECIPIENT_MANIFEST_SHA256 = "064bc1a2cf7e0a8201a476582ebde860d9345f67db131f34ec16f9f9cbb077e4"
RECIPIENT_RESULT_SHA256 = "3848582148a04d027012e862004c783152bc18255d783f028325f7c68be9b344"
RECIPIENT_JOURNAL_SHA256 = "c43e12ef31dbddb2753fe5d067ff1e95223d62754c34d551572dc00fc477a413"
RECIPIENT_LOGITS_SHA256 = "2c5d551f5dfb76db9adc32c2413e3d71d94576e912fff31dea9b94536644e724"
RECIPIENT_RECEIPT_SHA256 = "fbf7e2f93074c502e13ebb2ceac53547985d6952070c98d6d09a457fbe71b9cf"
RECIPIENT_CASES = (
    ('recipient-01-0-0-0', 'When John and Mary went to the store, John gave a book to', (2215, 1757, 290, 5335, 1816, 284, 262, 3650, 11, 1757, 2921, 257, 1492, 284), 5335),
    ('recipient-01-0-0-1', 'When John and Mary went to the store, Mary gave a book to', (2215, 1757, 290, 5335, 1816, 284, 262, 3650, 11, 5335, 2921, 257, 1492, 284), 1757),
    ('recipient-01-0-1-0', 'When Mary and John went to the store, John gave a book to', (2215, 5335, 290, 1757, 1816, 284, 262, 3650, 11, 1757, 2921, 257, 1492, 284), 5335),
    ('recipient-01-0-1-1', 'When Mary and John went to the store, Mary gave a book to', (2215, 5335, 290, 1757, 1816, 284, 262, 3650, 11, 5335, 2921, 257, 1492, 284), 1757),
    ('recipient-01-1-0-0', 'After John and Mary went to the park, John handed a gift to', (3260, 1757, 290, 5335, 1816, 284, 262, 3952, 11, 1757, 10158, 257, 6979, 284), 5335),
    ('recipient-01-1-0-1', 'After John and Mary went to the park, Mary handed a gift to', (3260, 1757, 290, 5335, 1816, 284, 262, 3952, 11, 5335, 10158, 257, 6979, 284), 1757),
    ('recipient-01-1-1-0', 'After Mary and John went to the park, John handed a gift to', (3260, 5335, 290, 1757, 1816, 284, 262, 3952, 11, 1757, 10158, 257, 6979, 284), 5335),
    ('recipient-01-1-1-1', 'After Mary and John went to the park, Mary handed a gift to', (3260, 5335, 290, 1757, 1816, 284, 262, 3952, 11, 5335, 10158, 257, 6979, 284), 1757),
    ('recipient-02-0-0-0', 'When Michael and Sarah went to the store, Michael gave a book to', (2215, 3899, 290, 10490, 1816, 284, 262, 3650, 11, 3899, 2921, 257, 1492, 284), 10490),
    ('recipient-02-0-0-1', 'When Michael and Sarah went to the store, Sarah gave a book to', (2215, 3899, 290, 10490, 1816, 284, 262, 3650, 11, 10490, 2921, 257, 1492, 284), 3899),
    ('recipient-02-0-1-0', 'When Sarah and Michael went to the store, Michael gave a book to', (2215, 10490, 290, 3899, 1816, 284, 262, 3650, 11, 3899, 2921, 257, 1492, 284), 10490),
    ('recipient-02-0-1-1', 'When Sarah and Michael went to the store, Sarah gave a book to', (2215, 10490, 290, 3899, 1816, 284, 262, 3650, 11, 10490, 2921, 257, 1492, 284), 3899),
    ('recipient-02-1-0-0', 'After Michael and Sarah went to the park, Michael handed a gift to', (3260, 3899, 290, 10490, 1816, 284, 262, 3952, 11, 3899, 10158, 257, 6979, 284), 10490),
    ('recipient-02-1-0-1', 'After Michael and Sarah went to the park, Sarah handed a gift to', (3260, 3899, 290, 10490, 1816, 284, 262, 3952, 11, 10490, 10158, 257, 6979, 284), 3899),
    ('recipient-02-1-1-0', 'After Sarah and Michael went to the park, Michael handed a gift to', (3260, 10490, 290, 3899, 1816, 284, 262, 3952, 11, 3899, 10158, 257, 6979, 284), 10490),
    ('recipient-02-1-1-1', 'After Sarah and Michael went to the park, Sarah handed a gift to', (3260, 10490, 290, 3899, 1816, 284, 262, 3952, 11, 10490, 10158, 257, 6979, 284), 3899),
    ('recipient-03-0-0-0', 'When Daniel and Emma went to the store, Daniel gave a book to', (2215, 7806, 290, 18966, 1816, 284, 262, 3650, 11, 7806, 2921, 257, 1492, 284), 18966),
    ('recipient-03-0-0-1', 'When Daniel and Emma went to the store, Emma gave a book to', (2215, 7806, 290, 18966, 1816, 284, 262, 3650, 11, 18966, 2921, 257, 1492, 284), 7806),
    ('recipient-03-0-1-0', 'When Emma and Daniel went to the store, Daniel gave a book to', (2215, 18966, 290, 7806, 1816, 284, 262, 3650, 11, 7806, 2921, 257, 1492, 284), 18966),
    ('recipient-03-0-1-1', 'When Emma and Daniel went to the store, Emma gave a book to', (2215, 18966, 290, 7806, 1816, 284, 262, 3650, 11, 18966, 2921, 257, 1492, 284), 7806),
    ('recipient-03-1-0-0', 'After Daniel and Emma went to the park, Daniel handed a gift to', (3260, 7806, 290, 18966, 1816, 284, 262, 3952, 11, 7806, 10158, 257, 6979, 284), 18966),
    ('recipient-03-1-0-1', 'After Daniel and Emma went to the park, Emma handed a gift to', (3260, 7806, 290, 18966, 1816, 284, 262, 3952, 11, 18966, 10158, 257, 6979, 284), 7806),
    ('recipient-03-1-1-0', 'After Emma and Daniel went to the park, Daniel handed a gift to', (3260, 18966, 290, 7806, 1816, 284, 262, 3952, 11, 7806, 10158, 257, 6979, 284), 18966),
    ('recipient-03-1-1-1', 'After Emma and Daniel went to the park, Emma handed a gift to', (3260, 18966, 290, 7806, 1816, 284, 262, 3952, 11, 18966, 10158, 257, 6979, 284), 7806),
    ('recipient-04-0-0-0', 'When Peter and Lisa went to the store, Peter gave a book to', (2215, 5613, 290, 15378, 1816, 284, 262, 3650, 11, 5613, 2921, 257, 1492, 284), 15378),
    ('recipient-04-0-0-1', 'When Peter and Lisa went to the store, Lisa gave a book to', (2215, 5613, 290, 15378, 1816, 284, 262, 3650, 11, 15378, 2921, 257, 1492, 284), 5613),
    ('recipient-04-0-1-0', 'When Lisa and Peter went to the store, Peter gave a book to', (2215, 15378, 290, 5613, 1816, 284, 262, 3650, 11, 5613, 2921, 257, 1492, 284), 15378),
    ('recipient-04-0-1-1', 'When Lisa and Peter went to the store, Lisa gave a book to', (2215, 15378, 290, 5613, 1816, 284, 262, 3650, 11, 15378, 2921, 257, 1492, 284), 5613),
    ('recipient-04-1-0-0', 'After Peter and Lisa went to the park, Peter handed a gift to', (3260, 5613, 290, 15378, 1816, 284, 262, 3952, 11, 5613, 10158, 257, 6979, 284), 15378),
    ('recipient-04-1-0-1', 'After Peter and Lisa went to the park, Lisa handed a gift to', (3260, 5613, 290, 15378, 1816, 284, 262, 3952, 11, 15378, 10158, 257, 6979, 284), 5613),
    ('recipient-04-1-1-0', 'After Lisa and Peter went to the park, Peter handed a gift to', (3260, 15378, 290, 5613, 1816, 284, 262, 3952, 11, 5613, 10158, 257, 6979, 284), 15378),
    ('recipient-04-1-1-1', 'After Lisa and Peter went to the park, Lisa handed a gift to', (3260, 15378, 290, 5613, 1816, 284, 262, 3952, 11, 15378, 10158, 257, 6979, 284), 5613),
    ('recipient-05-0-0-0', 'When Alice and Claire went to the store, Alice gave a book to', (2215, 14862, 290, 20758, 1816, 284, 262, 3650, 11, 14862, 2921, 257, 1492, 284), 20758),
    ('recipient-05-0-0-1', 'When Alice and Claire went to the store, Claire gave a book to', (2215, 14862, 290, 20758, 1816, 284, 262, 3650, 11, 20758, 2921, 257, 1492, 284), 14862),
    ('recipient-05-0-1-0', 'When Claire and Alice went to the store, Alice gave a book to', (2215, 20758, 290, 14862, 1816, 284, 262, 3650, 11, 14862, 2921, 257, 1492, 284), 20758),
    ('recipient-05-0-1-1', 'When Claire and Alice went to the store, Claire gave a book to', (2215, 20758, 290, 14862, 1816, 284, 262, 3650, 11, 20758, 2921, 257, 1492, 284), 14862),
    ('recipient-05-1-0-0', 'After Alice and Claire went to the park, Alice handed a gift to', (3260, 14862, 290, 20758, 1816, 284, 262, 3952, 11, 14862, 10158, 257, 6979, 284), 20758),
    ('recipient-05-1-0-1', 'After Alice and Claire went to the park, Claire handed a gift to', (3260, 14862, 290, 20758, 1816, 284, 262, 3952, 11, 20758, 10158, 257, 6979, 284), 14862),
    ('recipient-05-1-1-0', 'After Claire and Alice went to the park, Alice handed a gift to', (3260, 20758, 290, 14862, 1816, 284, 262, 3952, 11, 14862, 10158, 257, 6979, 284), 20758),
    ('recipient-05-1-1-1', 'After Claire and Alice went to the park, Claire handed a gift to', (3260, 20758, 290, 14862, 1816, 284, 262, 3952, 11, 20758, 10158, 257, 6979, 284), 14862),
    ('recipient-06-0-0-0', 'When Anna and Susan went to the store, Anna gave a book to', (2215, 11735, 290, 14919, 1816, 284, 262, 3650, 11, 11735, 2921, 257, 1492, 284), 14919),
    ('recipient-06-0-0-1', 'When Anna and Susan went to the store, Susan gave a book to', (2215, 11735, 290, 14919, 1816, 284, 262, 3650, 11, 14919, 2921, 257, 1492, 284), 11735),
    ('recipient-06-0-1-0', 'When Susan and Anna went to the store, Anna gave a book to', (2215, 14919, 290, 11735, 1816, 284, 262, 3650, 11, 11735, 2921, 257, 1492, 284), 14919),
    ('recipient-06-0-1-1', 'When Susan and Anna went to the store, Susan gave a book to', (2215, 14919, 290, 11735, 1816, 284, 262, 3650, 11, 14919, 2921, 257, 1492, 284), 11735),
    ('recipient-06-1-0-0', 'After Anna and Susan went to the park, Anna handed a gift to', (3260, 11735, 290, 14919, 1816, 284, 262, 3952, 11, 11735, 10158, 257, 6979, 284), 14919),
    ('recipient-06-1-0-1', 'After Anna and Susan went to the park, Susan handed a gift to', (3260, 11735, 290, 14919, 1816, 284, 262, 3952, 11, 14919, 10158, 257, 6979, 284), 11735),
    ('recipient-06-1-1-0', 'After Susan and Anna went to the park, Anna handed a gift to', (3260, 14919, 290, 11735, 1816, 284, 262, 3952, 11, 11735, 10158, 257, 6979, 284), 14919),
    ('recipient-06-1-1-1', 'After Susan and Anna went to the park, Susan handed a gift to', (3260, 14919, 290, 11735, 1816, 284, 262, 3952, 11, 14919, 10158, 257, 6979, 284), 11735),
    ('recipient-07-0-0-0', 'When Paul and Mark went to the store, Paul gave a book to', (2215, 3362, 290, 2940, 1816, 284, 262, 3650, 11, 3362, 2921, 257, 1492, 284), 2940),
    ('recipient-07-0-0-1', 'When Paul and Mark went to the store, Mark gave a book to', (2215, 3362, 290, 2940, 1816, 284, 262, 3650, 11, 2940, 2921, 257, 1492, 284), 3362),
    ('recipient-07-0-1-0', 'When Mark and Paul went to the store, Paul gave a book to', (2215, 2940, 290, 3362, 1816, 284, 262, 3650, 11, 3362, 2921, 257, 1492, 284), 2940),
    ('recipient-07-0-1-1', 'When Mark and Paul went to the store, Mark gave a book to', (2215, 2940, 290, 3362, 1816, 284, 262, 3650, 11, 2940, 2921, 257, 1492, 284), 3362),
    ('recipient-07-1-0-0', 'After Paul and Mark went to the park, Paul handed a gift to', (3260, 3362, 290, 2940, 1816, 284, 262, 3952, 11, 3362, 10158, 257, 6979, 284), 2940),
    ('recipient-07-1-0-1', 'After Paul and Mark went to the park, Mark handed a gift to', (3260, 3362, 290, 2940, 1816, 284, 262, 3952, 11, 2940, 10158, 257, 6979, 284), 3362),
    ('recipient-07-1-1-0', 'After Mark and Paul went to the park, Paul handed a gift to', (3260, 2940, 290, 3362, 1816, 284, 262, 3952, 11, 3362, 10158, 257, 6979, 284), 2940),
    ('recipient-07-1-1-1', 'After Mark and Paul went to the park, Mark handed a gift to', (3260, 2940, 290, 3362, 1816, 284, 262, 3952, 11, 2940, 10158, 257, 6979, 284), 3362),
    ('recipient-08-0-0-0', 'When David and Robert went to the store, David gave a book to', (2215, 3271, 290, 5199, 1816, 284, 262, 3650, 11, 3271, 2921, 257, 1492, 284), 5199),
    ('recipient-08-0-0-1', 'When David and Robert went to the store, Robert gave a book to', (2215, 3271, 290, 5199, 1816, 284, 262, 3650, 11, 5199, 2921, 257, 1492, 284), 3271),
    ('recipient-08-0-1-0', 'When Robert and David went to the store, David gave a book to', (2215, 5199, 290, 3271, 1816, 284, 262, 3650, 11, 3271, 2921, 257, 1492, 284), 5199),
    ('recipient-08-0-1-1', 'When Robert and David went to the store, Robert gave a book to', (2215, 5199, 290, 3271, 1816, 284, 262, 3650, 11, 5199, 2921, 257, 1492, 284), 3271),
    ('recipient-08-1-0-0', 'After David and Robert went to the park, David handed a gift to', (3260, 3271, 290, 5199, 1816, 284, 262, 3952, 11, 3271, 10158, 257, 6979, 284), 5199),
    ('recipient-08-1-0-1', 'After David and Robert went to the park, Robert handed a gift to', (3260, 3271, 290, 5199, 1816, 284, 262, 3952, 11, 5199, 10158, 257, 6979, 284), 3271),
    ('recipient-08-1-1-0', 'After Robert and David went to the park, David handed a gift to', (3260, 5199, 290, 3271, 1816, 284, 262, 3952, 11, 3271, 10158, 257, 6979, 284), 5199),
    ('recipient-08-1-1-1', 'After Robert and David went to the park, Robert handed a gift to', (3260, 5199, 290, 3271, 1816, 284, 262, 3952, 11, 5199, 10158, 257, 6979, 284), 3271),
)

def inventory_payload():
    predecessor = v13.inventory_payload()
    canonical = json.dumps(predecessor, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    if (v13.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(canonical).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(predecessor)
    if "additional_observations" in payload:
        raise ValueError("predecessor already contains additional-observation metadata")
    matches = {row["record_id"]: row for row in payload["records"]
               if row["record_id"] in INJECTION_RECORD_IDS}
    if len(matches) != len(INJECTION_RECORD_IDS):
        raise ValueError("prior agreement discovery records are missing or duplicated")
    if sum(row["record_id"] in INJECTION_RECORD_IDS for row in payload["records"]) != len(INJECTION_RECORD_IDS):
        raise ValueError("prior agreement discovery records are duplicated")
    for index, record_id in zip((0, 1, 4, 5, 8, 9, 12, 13), INJECTION_RECORD_IDS):
        row = matches[record_id]
        text, tokens = v2.PUBLIC_DISCOVERY[index]
        if (row["category"] != "observed-saved" or tuple(row["token_ids"]) != tokens
                or row["public_text"] != text):
            raise ValueError("prior agreement discovery identity does not match")
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v13.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"additional_public_injection_uses": 8,
                         "observed_public_recipient_baseline_rows": 64}
    payload["additional_observations"] = [{
        "kind": "public-number-injection-discovery",
        "record_ids": list(INJECTION_RECORD_IDS),
        "evidence_commit": EVIDENCE_COMMIT,
        "references": list(EVIDENCE_PATHS),
        "plan_sha256": PLAN_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "audit_sha256": AUDIT_SHA256,
        "provenance_sha256": PROVENANCE_SHA256,
        "operator_sha256": OPERATOR_SHA256,
        "prediction_sha256": PREDICTION_SHA256,
    }]
    if (len(RECIPIENT_CASES) != 64 or len({case[0] for case in RECIPIENT_CASES}) != 64
            or len({case[2] for case in RECIPIENT_CASES}) != 64):
        raise ValueError("recipient cases must have 64 distinct IDs and inputs")
    old_rows = [tuple(row["token_ids"]) for row in payload["records"]]
    rows = [case[2] for case in RECIPIENT_CASES]
    def related(left, right):
        return left[:len(right)] == right or right[:len(left)] == left
    if any(related(row, prior) for row in rows for prior in old_rows):
        raise ValueError("recipient input has a known full-row prefix relation")
    if any(related(left, right) for index, left in enumerate(rows) for right in rows[index + 1:]):
        raise ValueError("recipient inputs contain a full-row prefix relation")
    old_ids = {row["record_id"] for row in payload["records"]}
    for case_id, text, tokens, target in RECIPIENT_CASES:
        record_id = "observed-" + case_id
        if (record_id in old_ids or type(text) is not str or not text or type(target) is not int
                or len(tokens) != 14 or any(type(token) is not int or not 0 <= token < 50257 for token in tokens)):
            raise ValueError("recipient case identity or unpadded tokens are invalid")
        payload["records"].append({
            "record_id": record_id, "category": "observed-saved", "token_ids": list(tokens),
            "target_token_id": target, "public_text": text,
            "provenance": "Public, non-blind CPU recipient baseline; complete 64-case forward. "
                f"Cases SHA-256 {RECIPIENT_CASES_SHA256}, manifest SHA-256 {RECIPIENT_MANIFEST_SHA256}, "
                f"result SHA-256 {RECIPIENT_RESULT_SHA256}, journal SHA-256 {RECIPIENT_JOURNAL_SHA256}, "
                f"logits SHA-256 {RECIPIENT_LOGITS_SHA256}, process observation SHA-256 {RECIPIENT_RECEIPT_SHA256}.",
            "references": list(RECIPIENT_EVIDENCE_PATHS), "reference_commit": RECIPIENT_EVIDENCE_COMMIT,
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
