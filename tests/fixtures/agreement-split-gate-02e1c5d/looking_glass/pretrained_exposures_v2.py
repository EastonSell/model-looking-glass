"""Immutable successor to the known DistilGPT2 development-exposure inventory.

This version records one previously planned row as an actual CPU smoke and
reserves public inputs announced by concurrent study teams. It does not certify
an exhaustive exposure history or authorize a study seal.
"""
from dataclasses import dataclass, field, replace
import hashlib
import json
import re

from . import pretrained_exposures as v1


INVENTORY_VERSION = "distilgpt2-development-exposures-v2"
PREDECESSOR_VERSION = "distilgpt2-development-exposures-v1"
PREDECESSOR_SHA256 = "6f21bffeacbe2323c21e53378161443eae8ba28b7210f30e70fd353497e87c8f"
INVENTORY_SHA256 = "e64bc0b875fb295f03d7ad4d5180f6978d1d632487a98ea2824df01844ea4fb1"
SMOKE_SOURCE_COMMIT = "3f8de5d7fb299e4878dbf75f86bb58f2249d6a37"
SMOKE_PUBLICATION_COMMIT = "c6f5bf6bd17c440432af14655cc0ff505b45337e"
SMOKE_ATTEMPT_SHA256 = "2efe8d012a2a8f305dd6dcc8ac0b7c9e51bc00026537a9538d00b285fa3f314d"
SMOKE_RESULT_SHA256 = "8e5594edf340646b20498106c698bc5a25e80048fa6b564c1e1af0dab74f6d91"
REPLAY_ATTEMPT_SHA256 = "087146900792b1a2269e72d8c65c6e8af40e45d5b7514a5abc51823be3c5e23d"
REPLAY_RESULT_SHA256 = "5c0fc0b8508e281ce621c2227e46ac824d237b4fd45fa8b46621c6c36528c608"
SMOKE_LOCAL_DIRECTORY = "work/distilgpt2-intervention-dev-attempts/pinned-cf59716b06b64d6ab8ed7210e45bf632/attempt"
TARGET = v1.TARGET

# Public discovery material announced before any forward. These tokens were
# reconstructed offline using the pinned tokenizer; they are reservations,
# not evidence that the pinned model has executed them.
PUBLIC_DISCOVERY = (
    ("The cat near the windows", (464, 3797, 1474, 262, 9168)),
    ("The cats near the windows", (464, 11875, 1474, 262, 9168)),
    ("The dog near the windows", (464, 3290, 1474, 262, 9168)),
    ("The dogs near the windows", (464, 6844, 1474, 262, 9168)),
    ("The teacher in the room", (464, 4701, 287, 262, 2119)),
    ("The teachers in the room", (464, 7799, 287, 262, 2119)),
    ("The doctor in the room", (464, 6253, 287, 262, 2119)),
    ("The doctors in the room", (464, 7519, 287, 262, 2119)),
    ("The artist beside the rivers", (464, 6802, 13970, 262, 18180)),
    ("The artists beside the rivers", (464, 7912, 13970, 262, 18180)),
    ("The writer beside the rivers", (464, 6260, 13970, 262, 18180)),
    ("The writers beside the rivers", (464, 8786, 13970, 262, 18180)),
    ("The farmer behind the house", (464, 18739, 2157, 262, 2156)),
    ("The farmers behind the house", (464, 9818, 2157, 262, 2156)),
    ("The sailor behind the house", (464, 43272, 2157, 262, 2156)),
    ("The sailors behind the house", (464, 29996, 2157, 262, 2156)),
)

# A former proposal was amended after a tokenizer-only alignment check. Keep
# its rows as conservative references, not as forwards or current plans.
SUPERSEDED_DISCOVERY = (
    ("The baker beside the rivers", (464, 46412, 13970, 262, 18180)),
    ("The bakers beside the rivers", (464, 275, 3979, 13970, 262, 18180)),
)

# These are deliberately public parameter-edit proposals, not blind data.
# The study owner supplied the exact IDs after tokenizer-only encoding.
PUBLIC_PARAMETER_EDIT = (
    ("discovery", (464, 3139, 1748, 286, 4881, 318)),
    ("evaluation", (464, 3139, 286, 4881, 318)),
    ("evaluation", (28572, 468, 663, 3139, 287)),
    ("evaluation", (818, 4881, 11, 262, 3139, 1748, 318)),
    ("evaluation", (28572, 338, 3139, 1748, 318)),
    ("locality", (464, 3139, 286, 8031, 318)),
    ("locality", (464, 3139, 1748, 286, 8602, 318)),
    ("locality", (16504, 338, 3139, 318)),
    ("locality", (464, 3139, 286, 4486, 318)),
    ("locality", (464, 4387, 9151, 318, 262)),
    ("locality", (7571, 5556, 734, 21767)),
    ("locality", (464, 6697, 286, 3024, 318)),
    ("locality", (32, 1285, 468, 3598)),
)


@dataclass(frozen=True)
class ExposedDiscoveryCheck:
    inventory_version: str
    inventory_sha256: str
    candidate_count: int
    forbidden_row_count: int
    known_or_reserved_discovery_overlap_count: int
    scope: str = field(default="known-prefix-holdout-only", init=False)
    complete_exposure_history_certified: bool = field(default=False, init=False)
    preseal_ready: bool = field(default=False, init=False)


@dataclass(frozen=True)
class PresealDisclosure:
    inventory_version: str
    inventory_sha256: str
    protocol_sha256: str
    candidates_per_split: int
    forbidden_prefix_count: int
    discovery_policy: str
    known_or_reserved_discovery_overlap_count: int
    known_prefix_comparison: str = field(default="trusted-caller-report", init=False)
    prior_exposure_history: str = field(default="not-independently-certified-complete", init=False)
    sealed_manifest_verified: bool = field(default=False, init=False)
    independent_review_verified: bool = field(default=False, init=False)
    execution_authorized: bool = field(default=False, init=False)


def _records():
    if v1.INVENTORY_VERSION != PREDECESSOR_VERSION or v1.inventory_digest() != PREDECESSOR_SHA256:
        raise ValueError("predecessor exposure inventory does not match")
    matches = [record for record in v1.RECORDS if record.record_id == "planned-intervention-smoke"]
    if (len(matches) != 1 or matches[0].category != "planned" or
            matches[0].token_ids != (50256, 50256, 50256, 50256)):
        raise ValueError("predecessor planned smoke record does not match")
    updated = replace(
        matches[0], record_id="observed-intervention-smoke", category="observed-saved",
        provenance=("Pinned-model CPU intervention smoke and same-host independent replay; raw attempt/result "
                    f"SHA-256 values are {SMOKE_ATTEMPT_SHA256}, {SMOKE_RESULT_SHA256}, "
                    f"{REPLAY_ATTEMPT_SHA256}, and {REPLAY_RESULT_SHA256}; no GPU or causal-study input."),
        references=("docs/DISTILGPT2_INTERVENTION_DEV_V1.md", "looking_glass/distilgpt2_intervention_dev.py",
                    "results/distilgpt2-intervention-dev-cpu-v1-20261028/README.md",
                    "results/distilgpt2-intervention-dev-cpu-v1-20261028/publication.json",
                    "results/distilgpt2-intervention-dev-cpu-v1-20261028/pinned/result.json",
                    "results/distilgpt2-intervention-dev-cpu-v1-20261028/independent-replay/result.json"),
        reference_commit=SMOKE_PUBLICATION_COMMIT,
    )
    previous = tuple(updated if record is matches[0] else record for record in v1.RECORDS)
    planned = tuple(v1.Exposure(f"planned-agreement-discovery-{index:02}", "planned", ids,
                                "Public agreement discovery prefix announced before model forwards; tokenized offline with the pinned tokenizer.",
                                (), public_text=text)
                    for index, (text, ids) in enumerate(PUBLIC_DISCOVERY, 1))
    superseded = tuple(v1.Exposure(f"reference-superseded-agreement-{index:02}", "reference-only", ids,
                                   "Public preliminary proposal, superseded after tokenizer-only alignment; not a recorded model forward.",
                                   (), public_text=text)
                       for index, (text, ids) in enumerate(SUPERSEDED_DISCOVERY, 1))
    parameter_edit = tuple(v1.Exposure(f"planned-parameter-edit-{role}-{index:02}", "planned", ids,
                                       "Public parameter-edit input reported after tokenizer-only encoding; no model forward is asserted by this reservation.",
                                       ())
                           for index, (role, ids) in enumerate(PUBLIC_PARAMETER_EDIT, 1))
    return previous + planned + superseded + parameter_edit


def inventory_payload():
    """Return a fresh public payload linked to the exact v1 predecessor."""
    records = _records()
    result = v1.inventory_payload()
    result["version"] = INVENTORY_VERSION
    result["predecessor"] = {"version": PREDECESSOR_VERSION, "sha256": PREDECESSOR_SHA256}
    result["update"] = {"previous_record_id": "planned-intervention-smoke",
                        "record_id": "observed-intervention-smoke",
                        "reason": "previously planned public CPU smoke executed",
                        "reserved_public_discovery_rows": len(PUBLIC_DISCOVERY),
                        "superseded_public_reference_rows": len(SUPERSEDED_DISCOVERY),
                        "reserved_public_parameter_edit_rows": len(PUBLIC_PARAMETER_EDIT)}
    result["records"] = [{"record_id": item.record_id, "category": item.category,
                          "token_ids": list(item.token_ids), "provenance": item.provenance,
                          "references": list(item.references), "public_text": item.public_text,
                          "reference_commit": item.reference_commit or
                          (v1.REFERENCE_SOURCE_COMMIT if item.references else "")}
                         for item in records]
    return result


def canonical_inventory_bytes():
    return json.dumps(inventory_payload(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def inventory_digest():
    digest = hashlib.sha256(canonical_inventory_bytes()).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError("development-exposure inventory does not match its pinned digest")
    return digest


def _snapshot_split(value, cases_per_split):
    if type(value) not in (tuple, list) or len(value) != cases_per_split:
        raise ValueError("candidate split has the wrong number of token rows")
    rows = tuple(v1._snapshot_split(value[offset:offset + v1.SPLIT_SIZE])
                 for offset in range(0, cases_per_split, v1.SPLIT_SIZE))
    result = tuple(row for part in rows for row in part)
    if len(result) != cases_per_split:
        raise ValueError("candidate split has the wrong number of token rows")
    return result


def _forbidden_rows():
    return {record.token_ids[:length] for record in _records()
            for length in range(1, len(record.token_ids) + 1)}


def _identity(expected_model_identity):
    if (type(expected_model_identity) is not v1.ModelIdentity or
            any(type(getattr(expected_model_identity, name)) is not str
                for name in expected_model_identity.__dataclass_fields__) or
            expected_model_identity != TARGET):
        raise ValueError("expected model and tokenizer identity does not match")


def _candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split):
    discovery = _snapshot_split(discovery_token_ids, cases_per_split)
    heldout = _snapshot_split(heldout_token_ids, cases_per_split)
    if len(set(discovery + heldout)) != 2 * cases_per_split:
        raise ValueError("candidate splits contain a duplicate token sequence")
    if any(left[:len(right)] == right or right[:len(left)] == left
           for left in discovery for right in heldout):
        raise ValueError("discovery and held-out rows have a full-row prefix relationship")
    return discovery, heldout


def check_known_splits(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                       expected_model_identity, cases_per_split=8):
    """Check 8+8 or 16+16 rows; exceptions and tracebacks stay private."""
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    if type(cases_per_split) is not int or cases_per_split not in (8, 16):
        raise ValueError("cases per split must be exactly 8 or 16")
    _identity(expected_model_identity)
    discovery, heldout = _candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _forbidden_rows()
    if any(row in forbidden for row in discovery + heldout):
        raise ValueError("a candidate token sequence is in the development exclusion inventory")
    return v1.KnownSplitCheck(INVENTORY_VERSION, expected_inventory_sha256,
                              2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    """Permit public discovery reuse; exclude known prefixes from 16 held-out rows."""
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    _identity(expected_model_identity)
    discovery, heldout = _candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _forbidden_rows()
    if any(row in forbidden for row in heldout):
        raise ValueError("a held-out token sequence is in the development exclusion inventory")
    return ExposedDiscoveryCheck(INVENTORY_VERSION, expected_inventory_sha256, 32, len(forbidden),
                                  sum(row in forbidden for row in discovery))


def _validate_known_check(check):
    forbidden_count = len(_forbidden_rows())
    if (type(check) not in (v1.KnownSplitCheck, ExposedDiscoveryCheck) or
            check.inventory_version != INVENTORY_VERSION or check.inventory_sha256 != inventory_digest() or
            type(check.candidate_count) is not int or type(check.forbidden_row_count) is not int or
            check.forbidden_row_count != forbidden_count or check.complete_exposure_history_certified is not False or
            check.preseal_ready is not False):
        raise ValueError("a matching known-split check is required")
    if type(check) is v1.KnownSplitCheck and check.scope == "known-rows-only" and check.candidate_count in (16, 32):
        return forbidden_count, "known-prefixes-excluded-from-both", 0
    if (type(check) is ExposedDiscoveryCheck and check.scope == "known-prefix-holdout-only" and
            check.candidate_count == 32 and type(check.known_or_reserved_discovery_overlap_count) is int and
            0 <= check.known_or_reserved_discovery_overlap_count <= 16):
        return forbidden_count, "known-prefix-discovery-reuse-disclosed", check.known_or_reserved_discovery_overlap_count
    raise ValueError("a matching known-split check is required")


def require_preseal_readiness(check):
    """This inventory check alone never authorizes a study seal."""
    _validate_known_check(check)
    raise v1.ExposureInventoryIncomplete("known-row checking alone does not authorize a study seal")


def prepare_preseal_disclosure(check, *, protocol_sha256, acknowledge_prior_exposure_uncertainty=False):
    """Return a trusted-caller disclosure, not an attestation or authorization."""
    forbidden_count, discovery_policy, overlap_count = _validate_known_check(check)
    if acknowledge_prior_exposure_uncertainty is not True:
        raise v1.ExposureInventoryIncomplete("prior exposure uncertainty must be explicitly disclosed")
    if type(protocol_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None:
        raise ValueError("a canonical protocol SHA-256 is required")
    return PresealDisclosure(INVENTORY_VERSION, check.inventory_sha256, protocol_sha256,
                             check.candidate_count // 2, forbidden_count, discovery_policy, overlap_count)
