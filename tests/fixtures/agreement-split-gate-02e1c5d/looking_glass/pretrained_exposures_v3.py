"""Observed-exposure successor to the immutable DistilGPT2 inventory v2.

The 29 newly observed input rows were already reserved in v2. No model or
tokenizer is loaded here, and no complete exposure-history claim is made.
"""
import copy
import hashlib
import json
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2


INVENTORY_VERSION = "distilgpt2-development-exposures-v3"
PREDECESSOR_SHA256 = "e64bc0b875fb295f03d7ad4d5180f6978d1d632487a98ea2824df01844ea4fb1"
INVENTORY_SHA256 = "76baf0c8996bf2872a3a1b8fd0fe2734e15525ce0f7014e54811145cb98ce912"
TARGET = v2.TARGET
PUBLIC_DISCOVERY = v2.PUBLIC_DISCOVERY
AGREEMENT_EVIDENCE_COMMIT = "abc1b3500928246152b113bb2821bd1028a596b9"
AGREEMENT_MANIFEST_COMMIT = "0e5e3ae68d8dc4fba51e4aadf378e4c3dac27e96"
AGREEMENT_MANIFEST_SHA256 = "1b1adf4d72163bfa3e0399ed8ee92eba6b5c0eb82a47edf28cecde5ffb1deafe"
AGREEMENT_ATTEMPT_SHA256 = "5636bb89f071232d17a8499e97174f727a6900257b188f92f1cc157392cf20a1"
AGREEMENT_RESULT_SHA256 = "96bcb687dbdd9bb4017a8506b6acaffeb5a3da43478cf2da551a4a5cb0c723c8"
AGREEMENT_LOGITS_SHA256 = "7c262f3fe81236888a8f5fdb6d31f7835666674be9ffbf36f8ce2ca6a375eb6c"
EDIT_SOURCE_COMMIT = "1eee5d8589386fbf8256afac19b99518805f4bfc"
EDIT_MANIFEST_COMMIT = "ce025e994d28969384bd7fbd07b9e9e8e220c8f7"
EDIT_EVIDENCE_COMMIT = "4beaaf17eadc9715ae355d802fea3194b234aa98"
EDIT_MANIFEST_SHA256 = "93d953494cca7f2fcf0e7cd8baec35fa9ca1b68945e1e44f183e75e10ed03d1d"
EDIT_TERMINAL_SHA256 = "4affff192e0bdaa304dc606bf480865fa36a1e012cdb733a0ba9066fe02e92dd"
EDIT_OBSERVATIONS_SHA256 = "9fec61b36d704e38946f7528769d5817f0e18dd2f39ba11de97ed4698f8aaff2"
EDIT_REPLAY_TERMINAL_SHA256 = "ad1acde50c867a2c1ad13c7829b8dda2123ddb9af1e525a5632d8608cba4461f"
GPU_EVIDENCE_COMMIT = "082b1c350df8ce82352326dd1490cdd3a5aa7747"
GPU_V1_SHA256 = "6d79de374088adab40a1bb8641d78669cbf89f419462c1b8b6c73ebd8a8b1e61"
GPU_V2_SHA256 = "9c9af77278053954ed8ef0a80d163d16fa4d78243c76ead205df0909b2f4ede4"
THROUGHPUT_SOURCE_COMMIT = "c480e7e987ef7127ae61658ead3018e15ce9ae84"
THROUGHPUT_EVIDENCE_COMMIT = "271d6339fd19fe3c1ffaab2bdee801a0bac78765"
THROUGHPUT_WARMUP_SHA256 = "ac297babc41fba05e9b7d1bec84cf3aaeb97369a5fb05f6afc8bb835b9a1e48a"
THROUGHPUT_RAW_SHA256 = "060d57896dc597d5bd537dc24d14347074bb60abaf49d6f3406f8774fbd23d21"
THROUGHPUT_RESULT_SHA256 = "8f1072f08c4d6131f99294002a1f6a404081bfae82916ea9dd0306df2c29f39a"


def inventory_payload():
    if v2.inventory_digest() != PREDECESSOR_SHA256:
        raise ValueError("predecessor exposure inventory does not match")
    payload = copy.deepcopy(v2.inventory_payload())
    payload["version"] = INVENTORY_VERSION
    payload["predecessor"] = {"version": v2.INVENTORY_VERSION, "sha256": PREDECESSOR_SHA256}
    payload["update"] = {"observed_public_agreement_rows": 16, "observed_public_edit_rows": 13,
                         "additional_gpu_provenance_rows": 3,
                         "token_rows_changed": False}
    agreement = edit = gpu = 0
    for row in payload["records"]:
        name = row["record_id"]
        if name.startswith("planned-agreement-discovery-"):
            agreement += 1
            row["record_id"] = name.replace("planned-", "observed-", 1)
            row["category"] = "observed-saved"
            row["provenance"] = (
                "Public pinned-model CPU agreement preflight; complete result, attempt log and logits "
                f"SHA-256 values: {AGREEMENT_RESULT_SHA256}, {AGREEMENT_ATTEMPT_SHA256}, "
                f"{AGREEMENT_LOGITS_SHA256}. Frozen manifest {AGREEMENT_MANIFEST_COMMIT} "
                f"has SHA-256 {AGREEMENT_MANIFEST_SHA256}."
            )
            row["references"] = [
                "results/distilgpt2-agreement-preflight-v1-20261028/frozen.json",
                "results/distilgpt2-agreement-preflight-v1-20261028/run/attempt.jsonl",
                "results/distilgpt2-agreement-preflight-v1-20261028/run/result.json",
                "results/distilgpt2-agreement-preflight-v1-20261028/run/last_logits.f32",
            ]
            row["reference_commit"] = AGREEMENT_EVIDENCE_COMMIT
        elif name.startswith("planned-parameter-edit-"):
            edit += 1
            row["record_id"] = name.replace("planned-", "observed-", 1)
            row["category"] = "observed-saved"
            row["provenance"] = (
                "Saved CPU targeted-edit reference and replay, each covering this public input "
                "under original, gradient and random conditions. "
                f"Source {EDIT_SOURCE_COMMIT}; frozen manifest {EDIT_MANIFEST_COMMIT} SHA-256 "
                f"{EDIT_MANIFEST_SHA256}; reference terminal, observation and replay-terminal SHA-256 "
                f"{EDIT_TERMINAL_SHA256}, {EDIT_OBSERVATIONS_SHA256}, {EDIT_REPLAY_TERMINAL_SHA256}. "
                "Both terminal records report a negative decision."
            )
            row["references"] = [
                "results/targeted-edit-cpu-v1-20261028/frozen.json",
                "results/targeted-edit-cpu-v1-20261028/reference/terminal.json",
                "results/targeted-edit-cpu-v1-20261028/reference/observations.jsonl",
                "results/targeted-edit-cpu-v1-20261028/replay/terminal.json",
                "results/targeted-edit-cpu-v1-20261028/replay/observations.jsonl",
            ]
            row["reference_commit"] = EDIT_EVIDENCE_COMMIT
        elif name == "observed-bootstrap":
            gpu += 1
            row["provenance"] += (
                " Also used in saved GPU v1/v2 correctness pilots. V1 reported failed cleanup and "
                f"quarantine; v2 reported a clean supervised exit. Result SHA-256 {GPU_V1_SHA256} "
                f"and {GPU_V2_SHA256}."
            )
            row["references"] += ["docs/GPU_ONESHOT_RESULTS_V2.md", "results/gpu-oneshot-v1-20261029.json",
                                  "results/gpu-oneshot-v2-20261029.json"]
            row["reference_commit"] = GPU_EVIDENCE_COMMIT
        elif name in ("observed-throughput-a", "observed-throughput-b"):
            gpu += 1
            row["provenance"] += (
                " Also used in a saved quick GPU shared-load run with a public redacted derivative. "
                f"Source {THROUGHPUT_SOURCE_COMMIT}; original warmup, raw and result SHA-256 "
                f"{THROUGHPUT_WARMUP_SHA256}, {THROUGHPUT_RAW_SHA256}, {THROUGHPUT_RESULT_SHA256}."
            )
            row["references"] += [
                "results/throughput-gpu-quick-dev-v1-20261029/README.md",
                "results/throughput-gpu-quick-dev-v1-20261029/redaction_manifest.json",
                "results/throughput-gpu-quick-dev-v1-20261029/evidence/owner/measurements/warmup_samples.jsonl",
                "results/throughput-gpu-quick-dev-v1-20261029/evidence/owner/measurements/raw_samples.jsonl",
                "results/throughput-gpu-quick-dev-v1-20261029/evidence/owner/measurements/results.json",
            ]
            row["reference_commit"] = THROUGHPUT_EVIDENCE_COMMIT
    if (agreement, edit, gpu) != (16, 13, 3):
        raise ValueError("predecessor exposure rows do not match the expected snapshot")
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


def check_known_splits(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                       expected_model_identity, cases_per_split=8):
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    if type(cases_per_split) is not int or cases_per_split not in (8, 16):
        raise ValueError("cases per split must be exactly 8 or 16")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _forbidden_rows()
    if any(row in forbidden for row in discovery + heldout):
        raise ValueError("a candidate token sequence is in the development exclusion inventory")
    return v1.KnownSplitCheck(INVENTORY_VERSION, expected_inventory_sha256, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != inventory_digest():
        raise ValueError("expected exposure inventory digest does not match")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _forbidden_rows()
    if any(row in forbidden for row in heldout):
        raise ValueError("a held-out token sequence is in the development exclusion inventory")
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, expected_inventory_sha256, 32, len(forbidden),
                                    sum(row in forbidden for row in discovery))


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
