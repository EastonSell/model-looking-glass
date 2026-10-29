"""Known development inputs for the pinned DistilGPT2 study, not a blind split.

The inventory lists saved or operator-reported exposures and conservative
reservations; it does not certify a complete history.
This module does not generate, tokenize, seal, or run held-out cases. Python
tracebacks can retain private inputs and must stay inside the custodian.
"""
from dataclasses import dataclass, field
import hashlib
import json
import re


INVENTORY_VERSION = "distilgpt2-development-exposures-v1"
REFERENCE_SOURCE_COMMIT = "740692931da8deb5dc5c39a9692a2183cea32898"
# Pinned below after canonical payload construction; a change requires a new
# inventory version and an independently recorded digest.
INVENTORY_SHA256 = "6f21bffeacbe2323c21e53378161443eae8ba28b7210f30e70fd353497e87c8f"
VOCAB_SIZE = 50257
MAX_TOKENS = 32
SPLIT_SIZE = 8


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    model_revision: str
    weights_sha256: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_sha256: str


TARGET = ModelIdentity(
    "distilbert/distilgpt2",
    "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
    "distilbert/distilgpt2",
    "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
)


@dataclass(frozen=True)
class Exposure:
    record_id: str
    category: str
    token_ids: tuple[int, ...]
    provenance: str
    references: tuple[str, ...]
    public_text: str = ""
    reference_commit: str = ""


# Actual pinned-weight CPU forwards, supported by saved records or explicitly
# marked operator reports. Modified-model diagnostics also used pinned weights
# and are included conservatively. The inventory is not exhaustive.
_OBSERVED = (
    Exposure("observed-trace-smoke", "observed-saved", (464, 6766, 318, 4171, 13),
             "Pinned-model trace reference and same-host replays; also earlier unsaved loader and adapter checks.",
             ("results/pretrained-trace-cpu-v1-20261028/README.md", "tests/test_pretrained.py"),
             "The sky is blue."),
    Exposure("observed-throughput-a", "observed-saved", (464, 1402, 6512, 11406, 319),
             "Pinned-model CPU throughput smoke; token IDs reconstructed from the saved prompt using the pinned offline tokenizer.",
             ("results/throughput-cpu-smoke-v1-20261028/measurements/manifest.json",
              "results/throughput-cpu-smoke-v1-20261028/measurements/partial_windows.jsonl"),
             "The small bird landed on"),
    Exposure("observed-throughput-b", "observed-saved", (32, 6016, 1657, 4120, 287),
             "Pinned-model CPU throughput smoke; token IDs reconstructed from the saved prompt using the pinned offline tokenizer.",
             ("results/throughput-cpu-smoke-v1-20261028/measurements/manifest.json",
              "results/throughput-cpu-smoke-v1-20261028/measurements/partial_windows.jsonl"),
             "A bright light appeared in"),
    Exposure("observed-service-the", "observed-saved", (262,),
             "Pinned-model CPU service and CLI serial-reference and client checks.",
             ("docs/PRETRAINED_SERVICE_SMOKE.md", "docs/PRETRAINED_CLI_SMOKE.md",
              "evidence/pretrained-service-cpu-v1.json", "evidence/pretrained-cli-cpu-v1.json")),
    Exposure("observed-service-a", "observed-saved", (257,),
             "Pinned-model CPU service and CLI serial-reference and client checks.",
             ("docs/PRETRAINED_SERVICE_SMOKE.md", "docs/PRETRAINED_CLI_SMOKE.md",
              "evidence/pretrained-service-cpu-v1.json", "evidence/pretrained-cli-cpu-v1.json")),
    Exposure("observed-bootstrap", "observed-saved", (15496, 995),
             "Pinned-model CPU bootstrap preflight and independent replay; also an unsaved verifier challenge.",
             ("docs/BOOTSTRAP_CPU_PREFLIGHT.md", "evidence/bootstrap-pretrained-cpu-v1.json",
              "evidence/bootstrap-pretrained-cpu-independent-v1.json")),
    Exposure("observed-verifier-diagnostic", "observed-reported", (464, 2068, 318, 257),
             "Operator-reported unsaved CPU debug forward before and after first-layer attention-scaling mutation; pinned weights, modified model state.",
             ()),
    Exposure("observed-stride-diagnostic", "observed-reported", (262, 257),
             "Operator-reported unsaved CPU verifier probe before and after a temporary stride mutation; pinned weights, modified model state.",
             ()),
    Exposure("observed-capture-preflight", "observed-reported", (15496, 11, 616, 1438, 318, 257, 1332, 13),
             "Pinned-model CPU capture preflight, input reconstructed from the bound source; local work/benchmark-preflight-9d289b-20261028-01/cpu_preflight.json SHA-256 becf33ace3373c4f880c03e6c8b6f993b6ab7222c7c291c50dbaa631b3a12b27; two earlier preflights reported; no GPU timing run.",
             ("docs/CUDA_BENCHMARK_V1.md", "looking_glass/cuda_benchmark.py"),
             reference_commit="9d289b8715f6ffe2cde52ccd13924c79873e065c"),
) + tuple(
    Exposure(f"observed-batch-{i}", "observed-saved", (50256, i, i + 100, i + 200),
             "Pinned-model CPU sampled batch validation and fresh-process replay; synthetic development inputs, not causal-study held-out cases.",
             ("docs/DISTILGPT2_BATCH_CPU_V1.md", "results/distilgpt2-batch-cpu-v1-20261028/RESULTS.md",
              "results/distilgpt2-batch-cpu-v1-20261028/frozen.json",
              "results/distilgpt2-batch-cpu-v1-20261028/run/summary.json"),
             reference_commit="c67a7ed036a9744ac6e67c94c73127d50f8eae27")
    for i in range(103, 107)
)

# Planned and reference-only rows were not reported as target-model forwards
# at snapshot time. They are reserved conservatively, not evidence of runs.
_RESERVED = (
    Exposure("planned-intervention-smoke", "planned", (50256, 50256, 50256, 50256),
             "Proposed pinned-model CPU intervention smoke, not yet confirmed executed at snapshot time.", ()),
    Exposure("planned-throughput-train", "planned", (464, 4512, 5284, 379, 262),
             "Additional fixed throughput prompt not yet confirmed executed at snapshot time; token IDs reconstructed with the pinned offline tokenizer.",
             ("looking_glass/throughput_benchmark.py",), "The train arrived at the"),
    Exposure("planned-throughput-garden", "planned", (32, 5897, 11376, 6348, 13970),
             "Additional fixed throughput prompt not yet confirmed executed at snapshot time; token IDs reconstructed with the pinned offline tokenizer.",
             ("looking_glass/throughput_benchmark.py",), "A quiet garden grew beside"),
) + tuple(
    Exposure(f"reference-batch-{i}", "reference-only", (50256, i, i + 100, i + 200),
             "Public batch-validation discovery row; used only for content disjointness and not forwarded in the recorded run.",
             ("docs/DISTILGPT2_BATCH_CPU_V1.md",),
             reference_commit="c67a7ed036a9744ac6e67c94c73127d50f8eae27")
    for i in range(101, 103)
)

# These are artificial software-test rows, not observations of the target
# pretrained model. They are reserved as a conservative collision policy.
_SYNTHETIC = (
    Exposure("synthetic-custodian-1", "synthetic-only", (11, 12, 13), "Untrained arithmetic custodian test.", ("tests/test_pretrained_custodian.py",)),
    Exposure("synthetic-custodian-2", "synthetic-only", (1, 2), "Untrained arithmetic custodian test.", ("tests/test_pretrained_custodian.py",)),
    Exposure("synthetic-custodian-3", "synthetic-only", (1,), "Untrained arithmetic custodian test.", ("tests/test_pretrained_custodian.py",)),
    Exposure("synthetic-custodian-4", "synthetic-only", (4, 5), "Untrained arithmetic custodian test.", ("tests/test_pretrained_custodian.py",)),
    Exposure("synthetic-gpt2-control-1", "synthetic-only", (1, 2, 3, 4), "Random-weight GPT-2 control.", ("tests/test_intervention_plan.py",)),
    Exposure("synthetic-gpt2-control-2", "synthetic-only", (4, 3, 2, 1), "Random-weight GPT-2 trace control.", ("tests/test_transformer_trace.py",)),
    Exposure("synthetic-plan-reference-1", "synthetic-only", (1, 7, 11, 21, 31, 10), "Random-weight GPT-2 ordered-plan reference.", ("tests/test_gpt2_plan_reference.py",)),
    Exposure("synthetic-plan-reference-2", "synthetic-only", (2, 9, 13, 17, 23, 35), "Random-weight GPT-2 ordered-plan reference.", ("tests/test_gpt2_plan_reference.py",)),
)
RECORDS = _OBSERVED + _RESERVED + _SYNTHETIC


class ExposureInventoryIncomplete(ValueError):
    """The known-row check needs a scoped protocol and uncertainty disclosure."""


@dataclass(frozen=True)
class KnownSplitCheck:
    inventory_version: str
    inventory_sha256: str
    candidate_count: int
    forbidden_row_count: int
    scope: str = field(default="known-rows-only", init=False)
    complete_exposure_history_certified: bool = field(default=False, init=False)
    preseal_ready: bool = field(default=False, init=False)


@dataclass(frozen=True)
class PresealDisclosure:
    inventory_version: str
    inventory_sha256: str
    protocol_sha256: str
    candidates_per_split: int
    forbidden_prefix_count: int
    known_prefix_comparison: str = field(default="trusted-caller-report", init=False)
    prior_exposure_history: str = field(default="not-independently-certified-complete", init=False)
    sealed_manifest_verified: bool = field(default=False, init=False)
    independent_review_verified: bool = field(default=False, init=False)
    execution_authorized: bool = field(default=False, init=False)


def inventory_payload():
    """Return a fresh public payload; edits to it cannot mutate the inventory."""
    return {
        "version": INVENTORY_VERSION,
        "coverage": "saved or operator-reported exposures plus conservative reservations; history not independently certified complete",
        "exclusion_rule": "every nonempty prefix of each observed, planned, reference-only or synthetic row",
        "reference_source_commit": REFERENCE_SOURCE_COMMIT,
        "identity": {name: getattr(TARGET, name) for name in TARGET.__dataclass_fields__},
        "records": [{"record_id": item.record_id, "category": item.category,
                     "token_ids": list(item.token_ids), "provenance": item.provenance,
                     "references": list(item.references), "public_text": item.public_text,
                     "reference_commit": item.reference_commit or
                     (REFERENCE_SOURCE_COMMIT if item.references else "")}
                    for item in RECORDS],
    }


def canonical_inventory_bytes():
    return json.dumps(inventory_payload(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def inventory_digest():
    """Return the pinned digest, rejecting any in-process inventory drift."""
    digest = hashlib.sha256(canonical_inventory_bytes()).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError("development-exposure inventory does not match its pinned digest")
    return digest


def _snapshot_split(value):
    if type(value) not in (tuple, list) or len(value) != SPLIT_SIZE:
        raise ValueError("each candidate split must contain eight token rows")
    result = []
    for row in value:
        if type(row) not in (tuple, list) or not 1 <= len(row) <= MAX_TOKENS:
            raise ValueError("a candidate token row has an invalid shape")
        snapshot = tuple(row)
        if not 1 <= len(snapshot) <= MAX_TOKENS:
            raise ValueError("a candidate token row has an invalid shape")
        if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in snapshot):
            raise ValueError("a candidate token row has invalid token IDs")
        result.append(snapshot)
    if len(result) != SPLIT_SIZE:
        raise ValueError("each candidate split must contain eight token rows")
    return tuple(result)


def check_known_splits(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                       expected_model_identity):
    """Check exact token sequences; return values and error messages omit cases.

    The custodian must independently verify the real tokenizer and keep both
    input containers quiescent during copying and keep exception objects and
    tracebacks private. This check cannot establish that the known inventory
    contains every historical model exposure.
    """
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    if (type(expected_model_identity) is not ModelIdentity or
            any(type(getattr(expected_model_identity, name)) is not str
                for name in expected_model_identity.__dataclass_fields__) or
            expected_model_identity != TARGET):
        raise ValueError("expected model and tokenizer identity does not match")
    candidates = _snapshot_split(discovery_token_ids) + _snapshot_split(heldout_token_ids)
    if len(set(candidates)) != 2 * SPLIT_SIZE:
        raise ValueError("candidate splits contain a duplicate token sequence")
    forbidden = {record.token_ids[:length] for record in RECORDS
                 for length in range(1, len(record.token_ids) + 1)}
    if any(candidate in forbidden for candidate in candidates):
        # Do not put a private token row or its position in an exception.
        raise ValueError("a candidate token sequence is in the development exclusion inventory")
    return KnownSplitCheck(INVENTORY_VERSION, expected_inventory_sha256, len(candidates), len(forbidden))


def _validate_known_check(check):
    forbidden_count = len({record.token_ids[:length] for record in RECORDS
                           for length in range(1, len(record.token_ids) + 1)})
    if (type(check) is not KnownSplitCheck or check.inventory_version != INVENTORY_VERSION or
            check.inventory_sha256 != inventory_digest() or check.candidate_count != 2 * SPLIT_SIZE or
            check.forbidden_row_count != forbidden_count or check.scope != "known-rows-only" or
            check.complete_exposure_history_certified is not False or check.preseal_ready is not False):
        raise ValueError("a matching known-split check is required")
    return forbidden_count


def require_preseal_readiness(check):
    """This inventory check alone never authorizes a study seal."""
    _validate_known_check(check)
    raise ExposureInventoryIncomplete("known-row checking alone does not authorize a study seal")


def prepare_preseal_disclosure(check, *, protocol_sha256, acknowledge_prior_exposure_uncertainty=False):
    """Prepare a scoped report, not an attestation or execution authorization.

    A trusted caller must check the actual protocol, make and verify a fresh
    commitment, preserve the disclosure and obtain the separate review. The
    caller-supplied protocol digest and known-split object are not authenticated.
    """
    forbidden_count = _validate_known_check(check)
    if acknowledge_prior_exposure_uncertainty is not True:
        raise ExposureInventoryIncomplete("prior exposure uncertainty must be explicitly disclosed")
    if type(protocol_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None:
        raise ValueError("a canonical protocol SHA-256 is required")
    return PresealDisclosure(INVENTORY_VERSION, check.inventory_sha256, protocol_sha256,
                             SPLIT_SIZE, forbidden_count)
