"""Public-side, preregistered calculations for the DistilGPT2 causal study.

These helpers do not load a model or access a custodian's private record.
Execution and public artifact publication require the separately reviewed,
exclusive-lease composite runner; calculations alone are not a study result.
"""
from datetime import datetime
from dataclasses import dataclass
import base64
import hashlib
import json
import math
import os
import re

import numpy as np
import torch

from . import pretrained_protocol as protocol
from . import registry


ABS_TOL = 1e-5
REL_TOL = 1e-5
VOCAB_SIZE = 50257
WIDTH = 768
N_CASES = 8
FORWARD_BYTES = 1024 * 1024
STUDY_BYTES = 256 * 1024 * 1024


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _equal_json(actual, expected):
    try:
        return _canonical(actual) == _canonical(expected)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False


def _hex(value, length):
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def validate_public_manifest(public):
    """Validate only public v1 structure; this does not open a commitment.

    Tokenizer, checkpoint, checkout and runtime binding must be verified by a
    trusted loader and supervisor separately before any real forward.
    """
    fields = {"protocol", "protocol_sha256", "source_commit", "created_utc_self_reported",
              "model", "runtime", "seed", "discovery", "discovery_data_sha256",
              "heldout_count", "heldout_data_commitment", "injection", "metric",
              "random_baseline_layer", "site_commitment"}
    if type(public) is not dict or set(public) != fields:
        raise ValueError("unexpected public manifest fields")
    try:
        if len(_canonical(public)) > 1024 * 1024:
            raise ValueError("public manifest exceeds its byte limit")
    except (TypeError, OverflowError, RecursionError):
        raise ValueError("invalid public manifest encoding") from None
    expected = {
        "protocol": protocol.PROTOCOL_ID,
        "model": {"id": protocol.MODEL_ID, "revision": protocol.REVISION,
                  "sha256": protocol.ASSETS},
        "seed": protocol.SEED,
        "heldout_count": N_CASES,
        "injection": {"eligible_layers": list(protocol.ELIGIBLE_SITES), "module": "mlp_output",
                      "token": "last_input_token", "invocation": 0, "delta_l2_nominal": 0.5},
        "metric": {"logit_token_ids": [262, 257], "absolute_tolerance": ABS_TOL,
                   "relative_tolerance": REL_TOL},
        "random_baseline_layer": random_baseline(),
    }
    if any(not _equal_json(public[key], value) for key, value in expected.items()):
        raise ValueError("public manifest conflicts with the fixed protocol")
    if not (_hex(public["protocol_sha256"], 64) and _hex(public["source_commit"], 40)
            and _hex(public["heldout_data_commitment"], 64)
            and _hex(public["site_commitment"], 64)):
        raise ValueError("public manifest has invalid digest fields")
    timestamp = public["created_utc_self_reported"]
    try:
        if type(timestamp) is not str or datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%dT%H:%M:%SZ") != timestamp:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("public manifest time must be an ISO UTC timestamp") from None
    runtime = public["runtime"]
    keys = {"python", "torch", "transformers", "device", "driver", "deterministic_settings"}
    if type(runtime) is not dict or set(runtime) != keys or any(
            type(item) is not str or not item.strip() for item in runtime.values()):
        raise ValueError("public manifest has invalid runtime fields")
    rows = public["discovery"]
    if type(rows) is not list or len(rows) != N_CASES:
        raise ValueError("public manifest needs eight discovery rows")
    ids_seen = set()
    for row, prompt in zip(rows, protocol.DISCOVERY_PROMPTS):
        if (type(row) is not dict or set(row) != {"text", "token_ids"} or
                type(row["text"]) is not str or row["text"] != prompt):
            raise ValueError("discovery text does not match protocol")
        ids = row["token_ids"]
        if (type(ids) is not list or not 1 <= len(ids) <= 32 or
                any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids)):
            raise ValueError("invalid discovery token IDs")
        ids_seen.add(tuple(ids))
    if len(ids_seen) != N_CASES or public["discovery_data_sha256"] != protocol.canonical_sha256(rows):
        raise ValueError("discovery token sequences or digest are invalid")
    return protocol.canonical_sha256(public)


def random_baseline():
    return 1 + hashlib.sha256(b"MLG-v1-random-baseline\0" +
                              protocol.SEED.to_bytes(8, "big")).digest()[0] % 4


def injection_delta():
    """Return the fixed nominal additive vector in float32 on CPU."""
    signs = []
    for chunk in range((WIDTH + 255) // 256):
        digest = hashlib.sha256(b"MLG-v1-direction\0" + protocol.SEED.to_bytes(8, "big") +
                                chunk.to_bytes(4, "big")).digest()
        signs.extend(1 if byte & (1 << bit) else -1
                     for byte in digest for bit in range(7, -1, -1))
    return torch.tensor(signs[:WIDTH], dtype=torch.float32) * (0.5 / math.sqrt(WIDTH))


def verify_injection_integrity(pre, post):
    """Check private pre/post values and return the realized float32 delta.

    The caller must retain these vectors privately until reveal and separately
    verify that the plan modified only the authorized module/token/invocation.
    A zero downstream effect is allowed; an unchanged selected vector is not.
    """
    before, after = _array(pre, WIDTH), _array(post, WIDTH)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.add(before, injection_delta().numpy(), dtype=np.float32)
        realized = np.subtract(after, before, dtype=np.float32)
    if (not np.isfinite(expected).all() or not np.isfinite(realized).all() or
            not np.array_equal(after.view(np.uint32), expected.view(np.uint32)) or not np.any(realized != 0)):
        raise ValueError("injection did not realize the prescribed finite float32 addition")
    return {"realized_delta": torch.from_numpy(realized.copy()),
            "realized_norm": float(np.linalg.norm(realized.astype(np.float64)))}


def rank_sites(first_differences, l2_differences):
    """Score blinded discovery observations and fixed comparison selectors."""
    if type(first_differences) not in (tuple, list) or len(first_differences) != N_CASES:
        raise ValueError("exactly eight discovery first differences are required")
    if type(l2_differences) not in (tuple, list) or len(l2_differences) != N_CASES:
        raise ValueError("exactly eight discovery magnitude rows are required")
    counts = {site: 0 for site in protocol.ELIGIBLE_SITES}
    for first in first_differences:
        if first is not None:
            if type(first) is not int or first not in counts:
                raise ValueError("first difference must be an eligible layer or None")
            counts[first] += 1
    mags = []
    for row in l2_differences:
        if type(row) not in (tuple, list) or len(row) != 6:
            raise ValueError("each magnitude row must cover all six layers")
        clean = []
        for value in row:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("magnitudes must be finite nonnegative numbers")
            clean.append(float(value))
        mags.append(clean)
    means = {site: math.fsum(row[site] for row in mags) / N_CASES
             for site in protocol.ELIGIBLE_SITES}
    ranking = sorted(counts, key=lambda site: (-counts[site], site))
    magnitude = min(counts, key=lambda site: (-means[site], site))
    return {"ranking": ranking, "selected": ranking[0] if counts[ranking[0]] else None,
            "counts": {str(site): counts[site] for site in counts},
            "mean_l2": {str(site): means[site] for site in means},
            "magnitude": magnitude, "random": random_baseline(), "earliest": 1}


def score_revealed_site(selectors, site):
    """Score one disclosed site without treating an abstention as a ranked win."""
    if type(site) is not int or site not in protocol.ELIGIBLE_SITES or type(selectors) is not dict:
        raise ValueError("invalid revealed site or selectors")
    ranking, selected, counts = selectors.get("ranking"), selectors.get("selected"), selectors.get("counts")
    if (type(ranking) is not list or any(type(item) is not int for item in ranking) or
            sorted(ranking) != list(protocol.ELIGIBLE_SITES) or
            (selected is not None and type(selected) is not int) or type(counts) is not dict or
            set(counts) != {str(i) for i in protocol.ELIGIBLE_SITES} or
            any(type(value) is not int or not 0 <= value <= N_CASES for value in counts.values()) or
            sum(counts.values()) > N_CASES or
            ranking != sorted(protocol.ELIGIBLE_SITES, key=lambda item: (-counts[str(item)], item)) or
            selected != (ranking[0] if counts[str(ranking[0])] else None)):
        raise ValueError("invalid discovery ranking")
    means = selectors.get("mean_l2")
    if (type(means) is not dict or set(means) != {str(i) for i in protocol.ELIGIBLE_SITES} or
            any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in means.values()) or
            type(selectors.get("magnitude")) is not int or
            selectors["magnitude"] != min(protocol.ELIGIBLE_SITES, key=lambda i: (-means[str(i)], i)) or
            type(selectors.get("random")) is not int or selectors["random"] != random_baseline() or
            type(selectors.get("earliest")) is not int or selectors["earliest"] != 1):
        raise ValueError("invalid frozen baseline selectors")
    rank = ranking.index(site) + 1 if selected is not None else None
    return {"site": site, "rank": rank, "top_1": selected == site,
            "reciprocal_rank": 1.0 / rank if rank is not None else 0.0,
            "baseline_top_1": {name: selectors[name] == site for name in ("magnitude", "random", "earliest")}}


def _finite_values(values, count, name):
    if type(values) not in (tuple, list) or len(values) != count:
        raise ValueError(f"{name} must have exactly {count} values")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(value) for value in values]


def summarize_effects(actual, *, prediction):
    """Describe eight per-prompt effects and compare the fixed zero predictor."""
    actual = _finite_values(actual, N_CASES, "actual")
    prediction = _finite_values([prediction], 1, "prediction")[0]
    errors = [value - prediction for value in actual]
    selected = [value for value in actual if abs(value) > ABS_TOL]
    sign_correct = sum(abs(prediction) > ABS_TOL and (prediction > 0) == (value > 0)
                       for value in selected)
    try:
        result = {"effects": actual, "mean_effect": math.fsum(actual) / N_CASES,
                "prediction": prediction, "mae": math.fsum(map(abs, errors)) / N_CASES,
                "rmse": math.sqrt(math.fsum(value * value for value in errors) / N_CASES),
                "zero_predictor_mae": math.fsum(map(abs, actual)) / N_CASES,
                "zero_predictor_rmse": math.sqrt(math.fsum(value * value for value in actual) / N_CASES),
                "sign_eligible": len(selected), "sign_excluded": N_CASES - len(selected),
                "sign_accuracy": sign_correct / len(selected) if selected else None,
                "zero_predictor_sign_accuracy": 0.0 if selected else None}
        if any(type(value) is float and not math.isfinite(value) for value in result.values()):
            raise ValueError("effect summary exceeds finite numeric range")
        return result
    except (OverflowError, ValueError) as exc:
        raise ValueError("effect summary exceeds finite numeric range") from exc


def summarize_restoration(denominators, numerators):
    """Aggregate full-logit squared residuals; negative restoration is valid."""
    den = _finite_values(denominators, N_CASES, "denominators")
    num = _finite_values(numerators, N_CASES, "numerators")
    if any(value < 0 for value in den + num):
        raise ValueError("squared residuals cannot be negative")
    try:
        denominator, numerator = math.fsum(den), math.fsum(num)
        if not math.isfinite(denominator) or not math.isfinite(numerator):
            raise ValueError("aggregate residual exceeds finite numeric range")
        restoration = 1 - numerator / denominator if denominator > 1e-8 else None
        if restoration is not None and not math.isfinite(restoration):
            raise ValueError("restoration exceeds finite numeric range")
    except OverflowError as exc:
        raise ValueError("aggregate residual exceeds finite numeric range") from exc
    return {"denominator": denominator, "numerator": numerator,
            "denominator_per_prompt": den, "numerator_per_prompt": num,
            "restoration": restoration}


@dataclass(frozen=True)
class TraceEvent:
    """One sanitized, selected observation; never include composite-plan metadata."""
    layer: int
    module_path: str
    invocation: int
    batch_index: int
    token_index: int
    output_selector: tuple
    source_shape: tuple
    source_dtype: str
    source_device: str
    forward_index: int
    sequence: int
    value: torch.Tensor


@dataclass(frozen=True)
class Observation:
    events: tuple
    logits: torch.Tensor
    injection_integrity_verified: bool
    private_tensor_bytes: int
    extra_tensor_bytes: int = 0


@dataclass(frozen=True)
class DiscoveryCase:
    index: int
    conditions: dict
    post_clean: dict


def _array(tensor, size):
    if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu" or
            tensor.dtype != torch.float32 or tuple(tensor.shape) != (size,) or
            tensor.requires_grad):
        raise ValueError("trace tensors must be detached CPU float32 vectors of the specified shape")
    result = tensor.numpy()
    if not np.isfinite(result).all():
        raise ValueError("trace tensor contains a nonfinite value")
    return result


def validate_observation(observation, *, token_index, injected, source_device=None):
    """Check one public, sanitized forward and return its charged payload bytes.

    The custodian is trusted to enforce and truthfully report private snapshot
    sizes and injection integrity; those cannot be inferred from a public trace.
    This checks storage payload, not CUDA allocator or process memory limits.
    """
    if (type(observation) is not Observation or type(token_index) is not int or
            not 0 <= token_index < 32 or type(injected) is not bool):
        raise ValueError("invalid observation or token index")
    if (type(observation.events) is not tuple or len(observation.events) != 7 or
            type(observation.injection_integrity_verified) is not bool or
            observation.injection_integrity_verified != injected or
            type(observation.private_tensor_bytes) is not int or observation.private_tensor_bytes < 0 or
            type(observation.extra_tensor_bytes) is not int or observation.extra_tensor_bytes < 0 or
            observation.private_tensor_bytes != (3 * WIDTH * 4 if injected else 0)):
        raise ValueError("invalid capture or private integrity accounting")
    for index, event in enumerate(observation.events):
        layer = index if index < 6 else 5
        path = f"transformer.h.{layer}.mlp" if index < 6 else "transformer.h.5"
        selector = () if index < 6 else (0,)
        if (type(event) is not TraceEvent or type(event.layer) is not int or event.layer != layer or
                type(event.module_path) is not str or event.module_path != path or
                type(event.invocation) is not int or event.invocation != 0 or
                type(event.batch_index) is not int or event.batch_index != 0 or
                type(event.token_index) is not int or event.token_index != token_index or
                type(event.output_selector) is not tuple or event.output_selector != selector or
                any(type(element) is not int for element in event.output_selector) or
                type(event.source_shape) is not tuple or event.source_shape != (1, token_index + 1, WIDTH) or
                any(type(element) is not int for element in event.source_shape) or
                type(event.source_dtype) is not str or event.source_dtype != "torch.float32" or
                type(event.source_device) is not str or
                not event.source_device or (source_device is not None and event.source_device != source_device) or
                type(event.forward_index) is not int or event.forward_index != 0 or
                type(event.sequence) is not int or event.sequence != index):
            raise ValueError("unexpected, repeated, or out-of-order trace event")
        if event.source_device != observation.events[0].source_device:
            raise ValueError("trace events have inconsistent source devices")
        _array(event.value, WIDTH)
    _array(observation.logits, VOCAB_SIZE)
    total = ((7 * WIDTH + VOCAB_SIZE) * 4 + observation.private_tensor_bytes +
             observation.extra_tensor_bytes)
    if total > FORWARD_BYTES:
        raise ValueError("forward capture exceeds the 1 MiB payload cap")
    return total


def _stats(left, right):
    a = left.numpy().astype(np.float64)
    b = right.numpy().astype(np.float64)
    delta = a - b
    return {"different": bool(np.any(np.abs(delta) > ABS_TOL + REL_TOL * np.maximum(np.abs(a), np.abs(b)))),
            "l2": float(np.linalg.norm(delta)), "max_abs": float(np.max(np.abs(delta))),
            "squared_l2": float(np.dot(delta, delta)),
            "left_l2": float(np.linalg.norm(a)), "right_l2": float(np.linalg.norm(b))}


def _assert_same(actual, expected, name):
    for left, right in zip(actual.events, expected.events):
        if _stats(left.value, right.value)["different"]:
            raise ValueError(f"{name} control changed a measured vector")
    if _stats(actual.logits, expected.logits)["different"]:
        raise ValueError(f"{name} control changed full logits")


def _metric(logits):
    # Use the recorded float32 logits with float64 arithmetic for the contrast.
    return float(logits[262]) - float(logits[257])


def analyze_discovery_case(case, *, token_index, source_device=None):
    """Validate a complete blinded case supplied by a trusted custodial runner.

    This performs no model forward and cannot verify the runner's hook plan,
    custody, actual checkpoint or exclusive device lease.
    """
    if type(case) is not DiscoveryCase or type(case.index) is not int or not 0 <= case.index < N_CASES:
        raise ValueError("invalid discovery case")
    patches = {f"patch_{site}" for site in protocol.ELIGIBLE_SITES}
    reverse = {f"reverse_{site}" for site in protocol.ELIGIBLE_SITES}
    clean_only = {"clean", "clean_repeat"}
    intervention = {"injected", "null", "self_patch", "upstream", "late"} | patches | reverse
    if type(case.conditions) is not dict or set(case.conditions) != clean_only | intervention:
        raise ValueError("discovery case is missing or has extra conditions")
    if type(case.post_clean) is not dict or set(case.post_clean) != intervention:
        raise ValueError("every intervention requires a post-clean control")
    payload = 0
    for name, observation in case.conditions.items():
        injected = name not in clean_only | reverse
        # Fixed site-independent capture scheme: one nominal injection source
        # for every injected forward and one separate patch/self-patch snapshot.
        expected_extra = (int(injected) + int(name in patches | reverse | {"upstream", "late", "self_patch"})) * WIDTH * 4
        payload += validate_observation(observation, token_index=token_index,
                                        injected=injected, source_device=source_device)
        if observation.extra_tensor_bytes != expected_extra:
            raise ValueError("condition violates the fixed, site-independent snapshot charge")
    for observation in case.post_clean.values():
        payload += validate_observation(observation, token_index=token_index, injected=False,
                                        source_device=source_device)
        if observation.extra_tensor_bytes != 0:
            raise ValueError("post-clean snapshots have unexpected extra capture bytes")
    clean, repeated, corrupt = (case.conditions[name] for name in ("clean", "clean_repeat", "injected"))
    _assert_same(repeated, clean, "repeated clean")
    for name, observation in case.post_clean.items():
        _assert_same(observation, clean, f"post-clean after {name}")
    for name in ("null", "self_patch", "upstream"):
        _assert_same(case.conditions[name], corrupt, name)
    if _stats(case.conditions["late"].logits, clean.logits)["different"]:
        raise ValueError("late downstream rescue failed to restore full logits")
    differences = [_stats(a.value, b.value) for a, b in zip(clean.events[:6], corrupt.events[:6])]
    if differences[0]["different"]:
        raise ValueError("an upstream layer differs before the eligible injection sites")
    first_observed = next((layer for layer in range(1, 6) if differences[layer]["different"]), None)
    first = first_observed if first_observed in protocol.ELIGIBLE_SITES else None
    clean_metric, corrupt_metric = _metric(clean.logits), _metric(corrupt.logits)
    effects = {str(site): _metric(case.conditions[f"patch_{site}"].logits) - corrupt_metric
               for site in protocol.ELIGIBLE_SITES}
    reverse_effects = {str(site): _metric(case.conditions[f"reverse_{site}"].logits) - clean_metric
                       for site in protocol.ELIGIBLE_SITES}
    denominator = _stats(corrupt.logits, clean.logits)["squared_l2"]
    residuals = {str(site): _stats(case.conditions[f"patch_{site}"].logits, clean.logits)["squared_l2"]
                 for site in protocol.ELIGIBLE_SITES}
    reverse_residuals = {
        str(site): {"versus_clean": _stats(case.conditions[f"reverse_{site}"].logits, clean.logits)["squared_l2"],
                    "versus_injected": _stats(case.conditions[f"reverse_{site}"].logits, corrupt.logits)["squared_l2"]}
        for site in protocol.ELIGIBLE_SITES}
    return {"index": case.index, "token_index": token_index, "first_difference": first,
            "first_observed_layer": first_observed,
            "layer_differences": differences, "clean_metric": clean_metric,
            "injected_metric": corrupt_metric, "effects": effects,
            "reverse_effects": reverse_effects, "denominator": denominator,
            "patch_residuals": residuals, "reverse_residuals": reverse_residuals,
            "tensor_payload_bytes": payload}


class StudyExecutionError(RuntimeError):
    """A development run failed; potentially secret backend details are withheld."""


class StudyPublicationUncertain(RuntimeError):
    """A completion marker might have been published but could not be verified."""


class StudyCancelled(KeyboardInterrupt):
    """A development phase was cancelled before its completion marker."""

    def __init__(self, stage, terminal_state):
        super().__init__(f"cancellation before {stage} commit; recorded outcome: {terminal_state}")
        self.stage = stage
        self.terminal_state = terminal_state


class StudyCancellationUncertain(KeyboardInterrupt):
    """Cancellation was requested but publication state could not be verified."""

    def __init__(self, stage):
        super().__init__(f"{stage} cancellation has unresolved publication state; inspect the registry")
        self.stage = stage


class StudyCommittedCancellation(KeyboardInterrupt):
    """Cancellation was requested after the named phase had already committed."""

    def __init__(self, stage, artifact_sha256):
        super().__init__(f"{stage} committed before cancellation; this caller started no later phase")
        self.stage = stage
        self.artifact_sha256 = artifact_sha256


def _cancelled_terminal_state(run):
    try:
        state = run.status()["state"]
        return state if state in {"incomplete", "failed", "cancelled", "succeeded"} else "unverified"
    except BaseException:
        return "unverified"


_CONDITIONS = {
    "clean": {"base": "clean", "operation": "none"},
    "clean_repeat": {"base": "clean", "operation": "none"},
    "injected": {"base": "injected", "operation": "none"},
    "null": {"base": "injected", "operation": "null", "site": "mlp.1"},
    "self_patch": {"base": "injected", "operation": "patch", "site": "mlp.1",
                   "source": "same-run pre-patch capture"},
    "upstream": {"base": "injected", "operation": "patch", "site": "mlp.0",
                 "source": "same-input clean"},
    "late": {"base": "injected", "operation": "patch", "site": "block.5",
             "source": "same-input clean"},
}
for _site in protocol.ELIGIBLE_SITES:
    _CONDITIONS[f"patch_{_site}"] = {"base": "injected", "operation": "patch",
                                     "site": f"mlp.{_site}", "source": "same-input clean"}
    _CONDITIONS[f"reverse_{_site}"] = {"base": "clean", "operation": "patch",
                                       "site": f"mlp.{_site}", "source": "same-input injected"}


def _snapshot_case(case, *, token_index, source_device):
    """Copy a quiescent callback result into private, bounded owned CPU tensors.

    A backend must stop mutating the submitted result until this copy finishes;
    a tensor copy is not an atomic snapshot of concurrent writes. Subsequent
    backend mutations cannot change the analyzed and serialized owned copy.
    """
    if (type(case) is not DiscoveryCase or type(case.index) is not int or not 0 <= case.index < N_CASES or
            type(case.conditions) is not dict or type(case.post_clean) is not dict):
        raise ValueError("invalid callback case")
    conditions, post_clean = case.conditions.copy(), case.post_clean.copy()
    expected = set(_CONDITIONS)
    intervention = expected - {"clean", "clean_repeat"}
    if set(conditions) != expected or set(post_clean) != intervention:
        raise ValueError("callback case has incorrect condition or post-clean keys")
    # Validate all sizes and reject unbounded tensors before allocating copies.
    total = 0
    for name, observation in conditions.items():
        total += validate_observation(observation, token_index=token_index,
                                      injected=name not in {"clean", "clean_repeat"} and
                                      not name.startswith("reverse_"), source_device=source_device)
    for observation in post_clean.values():
        total += validate_observation(observation, token_index=token_index, injected=False,
                                      source_device=source_device)
    if total > (len(conditions) + len(post_clean)) * FORWARD_BYTES:
        raise ValueError("case snapshot exceeds its bounded forward budget")
    copies = {}
    def own(tensor):
        key = id(tensor)
        if key not in copies:
            copies[key] = tensor.detach().clone()
        return copies[key]
    def observation_copy(observation):
        events = tuple(TraceEvent(event.layer, event.module_path, event.invocation, event.batch_index,
                                  event.token_index, event.output_selector, event.source_shape,
                                  event.source_dtype, event.source_device, event.forward_index,
                                  event.sequence, own(event.value)) for event in observation.events)
        return Observation(events, own(observation.logits), observation.injection_integrity_verified,
                           observation.private_tensor_bytes, observation.extra_tensor_bytes)
    return DiscoveryCase(case.index, {name: observation_copy(value) for name, value in conditions.items()},
                         {name: observation_copy(value) for name, value in post_clean.items()})


def _tensor_reference(tensor, pool):
    """Deduplicate public float32 payloads by their little-endian byte hash."""
    raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
    digest = hashlib.sha256(raw).hexdigest()
    if digest not in pool:
        pool[digest] = {"dtype": "float32-le", "count": len(raw) // 4,
                        "base64": base64.b64encode(raw).decode("ascii")}
    return digest


def _observation_record(observation, pool):
    return {"events": [{"layer": event.layer, "module_path": event.module_path,
                         "invocation": event.invocation, "invocation_count": 1,
                         "batch_index": event.batch_index, "token_index": event.token_index,
                         "output_selector": list(event.output_selector), "selected_shape": [WIDTH],
                         "source_shape": list(event.source_shape), "source_dtype": event.source_dtype,
                         "source_device": event.source_device, "forward_index": event.forward_index,
                         "sequence": event.sequence,
                         "tensor_sha256": _tensor_reference(event.value, pool)}
                        for event in observation.events],
            "logits": {"shape": [VOCAB_SIZE], "tensor_sha256": _tensor_reference(observation.logits, pool)},
            "injection_integrity_verified_by_custodian": observation.injection_integrity_verified,
            "private_tensor_bytes": observation.private_tensor_bytes,
            "extra_tensor_bytes": observation.extra_tensor_bytes}


def _case_record(case, analysis, pool):
    return {"analysis": analysis,
            "conditions": {name: _observation_record(case.conditions[name], pool)
                           for name in sorted(case.conditions)},
            "post_clean": {name: _observation_record(case.post_clean[name], pool)
                           for name in sorted(case.post_clean)}}


def _development_run(path, public, public_digest):
    mib = 1024 * 1024
    run = registry.create_run(
        path, protocol={"id": protocol.PROTOCOL_ID, "sha256": public["protocol_sha256"]},
        inputs={"public_manifest_sha256": public_digest}, model=public["model"],
        code={"source_commit_self_reported": public["source_commit"]},
        metadata={"mode": "development-only", "status": "not a recorded study"},
        artifact_limits={"public_manifest.json": mib, "predictions.json": 86 * mib,
                         "partial_discovery.json": 86 * mib, "evaluation.json": 80 * mib,
                         "reveal.json": mib, "failure.json": 4096, "evaluation_claim.json": 4096,
                         "discovery_complete.json": 4096, "evaluation_complete.json": 4096,
                         "discovery_warning.json": 4096, "evaluation_warning.json": 4096},
        required_artifacts=["public_manifest.json", "predictions.json", "discovery_complete.json",
                            "evaluation_claim.json", "evaluation.json", "evaluation_complete.json", "reveal.json"],
        max_total_bytes=255 * mib)
    phase = run.path / ".phase"
    phase.mkdir()
    (phase / ".lock").touch(exist_ok=False)
    return run


def _completion_marker(stage, public_digest, predictions_digest, *, claim_digest=None, evaluation_digest=None):
    result = {"protocol": protocol.PROTOCOL_ID, "mode": "development-only", "stage": stage,
              "public_manifest_sha256": public_digest, "predictions_sha256": predictions_digest}
    if stage == "evaluation":
        result.update({"evaluation_claim_sha256": claim_digest, "evaluation_sha256": evaluation_digest})
    return result


def _completion_matches(run, marker_name, marker_bytes, predecessors):
    status = run.status()
    if status["state"] == "uninitialized":
        return False
    artifacts = status["artifacts"] if status["state"] == "incomplete" else status["outcome"]["artifacts"]
    expected = dict(predecessors)
    expected[marker_name] = hashlib.sha256(marker_bytes).hexdigest()
    return all(name in artifacts and artifacts[name]["sha256"] == digest and
               registry.sha256_file(run.path / "artifacts" / name) == digest
               for name, digest in expected.items())


def _publish_phase(run, *, stage, data_name, raw, marker, predecessors):
    """Publish a data artifact and its completion marker under an OS file lock.

    Marker publication is the phase linearization point. If its write reports
    an error after publication, exact marker and predecessor hashes are checked
    before reconciling the phase as committed. A warning is retained best effort.
    """
    marker_name, warning_name = f"{stage}_complete.json", f"{stage}_warning.json"
    marker_bytes = _canonical(marker)
    warning, cancelled = False, False
    def unresolved(*errors):
        if any(isinstance(error, KeyboardInterrupt) for error in errors):
            raise StudyCancellationUncertain(stage) from None
        raise StudyPublicationUncertain("completion publication could not be verified") from None
    try:
        with registry._lock(run.path / ".phase", run.lock_timeout):
            run.write_artifact(data_name, raw)
            run.write_artifact(marker_name, marker_bytes)
    except BaseException as error:
        # A failure before marker publication is an ordinary failed phase. If
        # lock cleanup fails after the marker, avoid contradicting the commit.
        try:
            (run.path / "artifacts" / marker_name).stat()
            exists = True
        except FileNotFoundError:
            exists = False
        except BaseException as verification_error:
            unresolved(error, verification_error)
        if not exists:
            raise
        try:
            matched = _completion_matches(run, marker_name, marker_bytes, predecessors)
        except BaseException as verification_error:
            unresolved(error, verification_error)
        if not matched:
            unresolved(error)
        warning = True
        cancelled = isinstance(error, KeyboardInterrupt)
        try:
            run.write_artifact(warning_name, _canonical({"stage": stage,
                                                         "warning": "completion publication reported an error after commit"}))
        except BaseException as warning_error:
            cancelled = cancelled or isinstance(warning_error, KeyboardInterrupt)
    if cancelled:
        raise StudyCommittedCancellation(stage, predecessors[data_name])
    return warning


def _reconcile_outer_error(run, stage, marker, predecessors, error):
    """Reconcile an error that can occur immediately after a helper commits."""
    cancelled = isinstance(error, KeyboardInterrupt)
    marker_name = f"{stage}_complete.json"
    try:
        (run.path / "artifacts" / marker_name).stat()
    except FileNotFoundError:
        return "absent", cancelled
    except BaseException as check_error:
        return "uncertain", cancelled or isinstance(check_error, KeyboardInterrupt)
    try:
        matched = _completion_matches(run, marker_name, _canonical(marker), predecessors)
    except BaseException as check_error:
        return "uncertain", cancelled or isinstance(check_error, KeyboardInterrupt)
    if not matched:
        return "uncertain", cancelled
    try:
        run.write_artifact(f"{stage}_warning.json", _canonical({"stage": stage,
                             "warning": "completion publication reported an error after commit"}))
    except BaseException as warning_error:
        cancelled = cancelled or isinstance(warning_error, KeyboardInterrupt)
    return "committed", cancelled


def run_development_discovery(path, public, callback):
    """Exercise the publication order using a synthetic, development-only callback.

    This API cannot execute or attest a preregistered real study. The callback
    is supplied by trusted development code and receives public discovery data
    only; Python cannot prevent it from running arbitrary code or accessing
    same-account private files. No private reveal is accepted here.
    """
    public_digest = validate_public_manifest(public)
    public_bytes = _canonical(public)
    public_snapshot = json.loads(public_bytes)
    if not callable(callback):
        raise ValueError("development callback must be callable")
    run = _development_run(path, public_snapshot, public_digest)
    records, pool, metrics = [], {}, []
    failed, uncertain, uncertain_cancel, committed_cancel, cancelled = False, False, False, None, False
    commit_context, reconciled_result, cancellation_seen = None, None, False
    try:
        run.write_artifact("public_manifest.json", public_bytes)
        for index, row in enumerate(public_snapshot["discovery"]):
            if registry.sha256_file(run.path / "artifacts" / "public_manifest.json") != public_digest:
                raise ValueError("public manifest changed before a development callback")
            case = callback(index, tuple(row["token_ids"]))
            if type(case) is not DiscoveryCase or case.index != index:
                raise ValueError("development case does not match the requested index")
            case = _snapshot_case(case, token_index=len(row["token_ids"]) - 1,
                                  source_device=public_snapshot["runtime"]["device"])
            measured = analyze_discovery_case(case, token_index=len(row["token_ids"]) - 1,
                                              source_device=public_snapshot["runtime"]["device"])
            if sum(item["tensor_payload_bytes"] for item in metrics) + measured["tensor_payload_bytes"] > STUDY_BYTES:
                raise ValueError("complete-study tensor payload cap exceeded")
            records.append(_case_record(case, measured, pool))
            metrics.append(measured)
        if registry.sha256_file(run.path / "artifacts" / "public_manifest.json") != public_digest:
            raise ValueError("public manifest changed during discovery")
        selectors = rank_sites([item["first_difference"] for item in metrics],
                               [[entry["l2"] for entry in item["layer_differences"]] for item in metrics])
        predictions = {str(site): math.fsum(item["effects"][str(site)] for item in metrics) / N_CASES
                       for site in protocol.ELIGIBLE_SITES}
        frozen = {"protocol": protocol.PROTOCOL_ID, "mode": "development-only",
                  "public_manifest_sha256": public_digest, "selectors": selectors,
                  "predictions": predictions, "zero_predictor": 0.0,
                  "conditions": _CONDITIONS,
                  "arithmetic": "float32 stored values; float64 differences and summaries",
                  "tensor_payload_bytes": sum(item["tensor_payload_bytes"] for item in metrics),
                  "cases": records, "tensors": pool}
        raw = _canonical(frozen)
        predictions_digest = hashlib.sha256(raw).hexdigest()
        marker = _completion_marker("discovery", public_digest, predictions_digest)
        result = {"public_manifest_sha256": public_digest, "predictions_sha256": predictions_digest,
                  "completed_cases": len(records), "state": "incomplete", "mode": "development-only"}
        predecessors = {"public_manifest.json": public_digest, "predictions.json": predictions_digest}
        commit_context = (marker, predecessors, result, predictions_digest)
        result["publication_warning"] = _publish_phase(
            run, stage="discovery", data_name="predictions.json", raw=raw, marker=marker,
            predecessors=predecessors)
        return result
    except BaseException as error:
        if isinstance(error, StudyCommittedCancellation):
            committed_cancel = (error.stage, error.artifact_sha256)
        elif commit_context is not None:
            marker, predecessors, result, artifact_digest = commit_context
            state, cancellation_seen = _reconcile_outer_error(run, "discovery", marker, predecessors, error)
            if state == "committed" and cancellation_seen:
                committed_cancel = ("discovery", artifact_digest)
            elif state == "committed":
                result["publication_warning"] = True
                reconciled_result = result
            elif state == "uncertain" and cancellation_seen:
                uncertain_cancel = True
            elif state == "uncertain":
                uncertain = True
            else:
                failed = True
        else:
            failed = True
        cancelled = isinstance(error, KeyboardInterrupt) or cancellation_seen
        # Preserve only sanitized, completed cases; exception strings can contain
        # hidden plan metadata and must remain within the custodian boundary.
        if failed and not uncertain and not uncertain_cancel and committed_cancel is None:
            try:
                partial = {"protocol": protocol.PROTOCOL_ID, "mode": "development-only",
                           "state": "partial", "public_manifest_sha256": public_digest,
                           "cases": records, "tensors": pool}
                run.write_artifact("partial_discovery.json", _canonical(partial))
            except BaseException as partial_error:
                cancelled = cancelled or isinstance(partial_error, KeyboardInterrupt)
            try:
                run.finalize("cancelled" if cancelled else "failed",
                             {"stage": "discovery", "completed_cases": len(records),
                              "reason": "interrupted" if cancelled else "invalid or failed"})
            except BaseException as finalization_error:
                cancelled = cancelled or isinstance(finalization_error, KeyboardInterrupt)
    # Raising outside the handler avoids attaching a backend exception as context.
    if failed:
        if cancelled:
            raise StudyCancelled("discovery", _cancelled_terminal_state(run))
        raise StudyExecutionError("development discovery did not complete; inspect the registry outcome")
    if uncertain:
        raise StudyPublicationUncertain("development discovery publication is unresolved; inspect the registry")
    if uncertain_cancel:
        raise StudyCancellationUncertain("discovery")
    if committed_cancel is not None:
        raise StudyCommittedCancellation(*committed_cancel)
    if reconciled_result is not None:
        return reconciled_result


def _checked_development_stage(path, public_digest, predictions_digest, *, needs_evaluation=False,
                               evaluation_digest=None):
    """Check externally held hashes and canonical artifacts before entering a phase."""
    if not _hex(public_digest, 64) or not _hex(predictions_digest, 64):
        raise ValueError("external manifest and prediction digests are required")
    if needs_evaluation and not _hex(evaluation_digest, 64):
        raise ValueError("external evaluation digest is required")
    run = registry.open_run(path)
    status = run.status()
    if status["state"] != "incomplete" or status["manifest"]["metadata"].get("mode") != "development-only":
        raise ValueError("run is not an incomplete development run")
    expected_files = {"public_manifest.json": public_digest, "predictions.json": predictions_digest}
    discovery_entry = status["artifacts"].get("discovery_complete.json", {})
    discovery_digest = discovery_entry.get("sha256")
    if not _hex(discovery_digest, 64):
        raise ValueError("discovery completion marker is missing")
    expected_files["discovery_complete.json"] = discovery_digest
    if needs_evaluation:
        claim_entry = status["artifacts"].get("evaluation_claim.json", {})
        claim_digest = claim_entry.get("sha256")
        if not _hex(claim_digest, 64):
            raise ValueError("evaluation claim is missing")
        expected_files["evaluation_claim.json"] = claim_digest
        expected_files["evaluation.json"] = evaluation_digest
        complete_digest = status["artifacts"].get("evaluation_complete.json", {}).get("sha256")
        if not _hex(complete_digest, 64):
            raise ValueError("evaluation completion marker is missing")
        expected_files["evaluation_complete.json"] = complete_digest
    for warning_name in ("discovery_warning.json", "evaluation_warning.json"):
        if warning_name in status["artifacts"]:
            if warning_name == "evaluation_warning.json" and not needs_evaluation:
                raise ValueError("unexpected evaluation warning before evaluation")
            expected_files[warning_name] = status["artifacts"][warning_name]["sha256"]
    if set(status["artifacts"]) != set(expected_files):
        raise ValueError("run is missing a required stage or contains extra artifacts")
    parsed = {}
    for name, digest in expected_files.items():
        if status["artifacts"][name]["sha256"] != digest:
            raise ValueError("an externally held artifact digest does not match")
        raw = (run.path / "artifacts" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("artifact changed during stage validation")
        try:
            value = json.loads(raw)
            if _canonical(value) != raw:
                raise ValueError
        except (UnicodeError, TypeError, ValueError, RecursionError):
            raise ValueError("artifact is not canonical JSON") from None
        parsed[name] = value
    public, frozen = parsed["public_manifest.json"], parsed["predictions.json"]
    if validate_public_manifest(public) != public_digest:
        raise ValueError("invalid public manifest")
    if not _equal_json(parsed["discovery_complete.json"],
                       _completion_marker("discovery", public_digest, predictions_digest)):
        raise ValueError("discovery completion does not bind its predecessors")
    for stage in ("discovery", "evaluation"):
        name = f"{stage}_warning.json"
        if name in parsed and not _equal_json(parsed[name], {"stage": stage,
                  "warning": "completion publication reported an error after commit"}):
            raise ValueError("invalid publication warning")
    if (type(frozen) is not dict or frozen.get("mode") != "development-only" or
            frozen.get("protocol") != protocol.PROTOCOL_ID or
            frozen.get("public_manifest_sha256") != public_digest or
            type(frozen.get("tensor_payload_bytes")) is not int or
            not 0 <= frozen["tensor_payload_bytes"] <= STUDY_BYTES or
            type(frozen.get("cases")) is not list or len(frozen["cases"]) != N_CASES or
            type(frozen.get("predictions")) is not dict or
            set(frozen["predictions"]) != {str(site) for site in protocol.ELIGIBLE_SITES}):
        raise ValueError("invalid frozen development predictions")
    for value in frozen["predictions"].values():
        _finite_values([value], 1, "prediction")
    score_revealed_site(frozen.get("selectors"), 1)
    if needs_evaluation:
        evaluation = parsed["evaluation.json"]
        claim = parsed["evaluation_claim.json"]
        if claim != {"protocol": protocol.PROTOCOL_ID, "mode": "development-only", "stage": "evaluation",
                     "public_manifest_sha256": public_digest, "predictions_sha256": predictions_digest}:
            raise ValueError("invalid evaluation ownership claim")
        if not _equal_json(parsed["evaluation_complete.json"],
                           _completion_marker("evaluation", public_digest, predictions_digest,
                                              claim_digest=claim_digest, evaluation_digest=evaluation_digest)):
            raise ValueError("evaluation completion does not bind its predecessors")
        if (type(evaluation) is not dict or evaluation.get("mode") != "development-only" or
                evaluation.get("state") != "complete" or evaluation.get("protocol") != protocol.PROTOCOL_ID or
                evaluation.get("public_manifest_sha256") != public_digest or
                evaluation.get("predictions_sha256") != predictions_digest or
                evaluation.get("evaluation_claim_sha256") != claim_digest):
            raise ValueError("invalid development evaluation")
    return run, parsed


def run_development_evaluation(path, public_digest, predictions_digest, callback):
    """Exercise evaluation gating on fake cases, never on reserved real inputs.

    Callback receives only an artificial integer case index and returns
    ``(last_token_index, DiscoveryCase)``. This is not a real held-out runner.
    External digest arguments must be retained independently from the mutable
    local run directory; they provide consistency, not trusted timestamps.
    """
    if not callable(callback):
        raise ValueError("development callback must be callable")
    run, parsed = _checked_development_stage(path, public_digest, predictions_digest)
    public, frozen = parsed["public_manifest.json"], parsed["predictions.json"]
    claim = {"protocol": protocol.PROTOCOL_ID, "mode": "development-only", "stage": "evaluation",
             "public_manifest_sha256": public_digest, "predictions_sha256": predictions_digest}
    claim_bytes = _canonical(claim)
    # Registry publication is an atomic, process-safe create-if-absent. This
    # is outside the failure handler so a loser cannot finalize the winner.
    with registry._lock(run.path / ".phase", run.lock_timeout):
        run.write_artifact("evaluation_claim.json", claim_bytes)
    claim_digest = hashlib.sha256(claim_bytes).hexdigest()
    records, pool, metrics = [], {}, []
    failed, uncertain, uncertain_cancel, committed_cancel, cancelled = False, False, False, None, False
    commit_context, reconciled_result, cancellation_seen = None, None, False
    try:
        for index in range(N_CASES):
            # Recheck externally held digests before every development callback.
            for name, digest in (("public_manifest.json", public_digest), ("predictions.json", predictions_digest)):
                if registry.sha256_file(run.path / "artifacts" / name) != digest:
                    raise ValueError("a frozen artifact changed before evaluation")
            token_index, case = callback(index)
            if type(case) is not DiscoveryCase or case.index != index:
                raise ValueError("development case does not match the requested index")
            case = _snapshot_case(case, token_index=token_index, source_device=public["runtime"]["device"])
            measured = analyze_discovery_case(case, token_index=token_index,
                                              source_device=public["runtime"]["device"])
            total = frozen["tensor_payload_bytes"] + sum(item["tensor_payload_bytes"] for item in metrics)
            if total + measured["tensor_payload_bytes"] > STUDY_BYTES:
                raise ValueError("complete-study tensor payload cap exceeded")
            records.append(_case_record(case, measured, pool))
            metrics.append(measured)
        for name, digest in (("public_manifest.json", public_digest), ("predictions.json", predictions_digest)):
            if registry.sha256_file(run.path / "artifacts" / name) != digest:
                raise ValueError("a frozen artifact changed during evaluation")
        effects, restoration = {}, {}
        for site in protocol.ELIGIBLE_SITES:
            key = str(site)
            effects[key] = summarize_effects([item["effects"][key] for item in metrics],
                                             prediction=frozen["predictions"][key])
            restoration[key] = summarize_restoration([item["denominator"] for item in metrics],
                                                     [item["patch_residuals"][key] for item in metrics])
        evaluation = {"protocol": protocol.PROTOCOL_ID, "mode": "development-only", "state": "complete",
                      "public_manifest_sha256": public_digest, "predictions_sha256": predictions_digest,
                      "evaluation_claim_sha256": claim_digest,
                      "selectors": frozen["selectors"], "metrics": effects, "restoration": restoration,
                      "tensor_payload_bytes": frozen["tensor_payload_bytes"] +
                                              sum(item["tensor_payload_bytes"] for item in metrics),
                      "cases": records, "tensors": pool}
        raw = _canonical(evaluation)
        evaluation_digest = hashlib.sha256(raw).hexdigest()
        marker = _completion_marker("evaluation", public_digest, predictions_digest,
                                    claim_digest=claim_digest, evaluation_digest=evaluation_digest)
        result = {"state": "incomplete", "mode": "development-only",
                  "evaluation_sha256": evaluation_digest, "completed_cases": len(records)}
        predecessors = {"public_manifest.json": public_digest, "predictions.json": predictions_digest,
                        "evaluation_claim.json": claim_digest, "evaluation.json": evaluation_digest}
        commit_context = (marker, predecessors, result, evaluation_digest)
        result["publication_warning"] = _publish_phase(
            run, stage="evaluation", data_name="evaluation.json", raw=raw, marker=marker,
            predecessors=predecessors)
        return result
    except BaseException as error:
        if isinstance(error, StudyCommittedCancellation):
            committed_cancel = (error.stage, error.artifact_sha256)
        elif commit_context is not None:
            marker, predecessors, result, artifact_digest = commit_context
            state, cancellation_seen = _reconcile_outer_error(run, "evaluation", marker, predecessors, error)
            if state == "committed" and cancellation_seen:
                committed_cancel = ("evaluation", artifact_digest)
            elif state == "committed":
                result["publication_warning"] = True
                reconciled_result = result
            elif state == "uncertain" and cancellation_seen:
                uncertain_cancel = True
            elif state == "uncertain":
                uncertain = True
            else:
                failed = True
        else:
            failed = True
        cancelled = isinstance(error, KeyboardInterrupt) or cancellation_seen
        if failed and not uncertain and not uncertain_cancel and committed_cancel is None:
            try:
                partial = {"protocol": protocol.PROTOCOL_ID, "mode": "development-only", "state": "partial",
                           "public_manifest_sha256": public_digest, "predictions_sha256": predictions_digest,
                           "evaluation_claim_sha256": claim_digest,
                           "cases": records, "tensors": pool}
                run.write_artifact("evaluation.json", _canonical(partial))
            except BaseException as partial_error:
                cancelled = cancelled or isinstance(partial_error, KeyboardInterrupt)
            try:
                run.finalize("cancelled" if cancelled else "failed",
                             {"stage": "evaluation", "completed_cases": len(records),
                              "reason": "interrupted" if cancelled else "invalid or failed"})
            except BaseException as finalization_error:
                cancelled = cancelled or isinstance(finalization_error, KeyboardInterrupt)
    if failed:
        if cancelled:
            raise StudyCancelled("evaluation", _cancelled_terminal_state(run))
        raise StudyExecutionError("development evaluation did not complete; inspect the registry outcome")
    if uncertain:
        raise StudyPublicationUncertain("development evaluation publication is unresolved; inspect the registry")
    if uncertain_cancel:
        raise StudyCancellationUncertain("evaluation")
    if committed_cancel is not None:
        raise StudyCommittedCancellation(*committed_cancel)
    if reconciled_result is not None:
        return reconciled_result


def publish_development_reveal(path, public_digest, predictions_digest, evaluation_digest, reveal):
    """Verify and publish a synthetic reveal after a completed development evaluation."""
    run, _ = _checked_development_stage(path, public_digest, predictions_digest,
                                        needs_evaluation=True, evaluation_digest=evaluation_digest)
    raw, score, predecessors = None, None, None
    warning, uncertain, committed_cancel, requested_cancel = False, False, False, False
    details = {"stage": "synthetic development only", "heldout_cases": N_CASES}
    try:
        with registry._lock(run.path / ".phase", run.lock_timeout):
            run, parsed = _checked_development_stage(path, public_digest, predictions_digest,
                                                     needs_evaluation=True, evaluation_digest=evaluation_digest)
            public, frozen = parsed["public_manifest.json"], parsed["predictions.json"]
            if not protocol.verify_reveal(public, reveal):
                raise ValueError("development reveal does not match its public commitments")
            score = score_revealed_site(frozen["selectors"], reveal["site"])
            payload = {"protocol": protocol.PROTOCOL_ID, "mode": "development-only",
                       "public_manifest_sha256": public_digest, "predictions_sha256": predictions_digest,
                       "evaluation_sha256": evaluation_digest, "reveal": reveal,
                       "revealed_site_score": score}
            raw = _canonical(payload)
            predecessors = {"public_manifest.json": public_digest, "predictions.json": predictions_digest,
                            "evaluation.json": evaluation_digest, "reveal.json": hashlib.sha256(raw).hexdigest()}
            for name in ("discovery_complete.json", "evaluation_claim.json", "evaluation_complete.json"):
                predecessors[name] = hashlib.sha256(_canonical(parsed[name])).hexdigest()
            run.write_artifact("reveal.json", raw)
            run.finalize("succeeded", details)
    except BaseException as error:
        if raw is None:
            raise
        requested_cancel = isinstance(error, KeyboardInterrupt)
        try:
            status = run.status()
            if status["state"] != "succeeded" or not _equal_json(status["outcome"]["details"], details):
                uncertain = True
            else:
                inventory = status["outcome"]["artifacts"]
                warning = all(name in inventory and inventory[name]["sha256"] == digest and
                              registry.sha256_file(run.path / "artifacts" / name) == digest
                              for name, digest in predecessors.items())
                uncertain = not warning
        except BaseException as verification_error:
            requested_cancel = requested_cancel or isinstance(verification_error, KeyboardInterrupt)
            uncertain = True
        if warning:
            committed_cancel = requested_cancel
            # The terminal registry is immutable. This sanitized, best-effort
            # sidecar is not part of its terminal artifact digest.
            try:
                with (run.path / ".phase" / "reveal_warning.json").open("xb") as stream:
                    stream.write(_canonical({"stage": "reveal",
                                             "warning": "terminal publication reported an error after commit"}))
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException as warning_error:
                committed_cancel = committed_cancel or isinstance(warning_error, KeyboardInterrupt)
    if uncertain:
        if requested_cancel:
            raise StudyCancellationUncertain("reveal")
        raise StudyPublicationUncertain("development reveal publication is unresolved; inspect the registry")
    if committed_cancel:
        raise StudyCommittedCancellation("reveal", hashlib.sha256(raw).hexdigest())
    return {"state": "succeeded", "mode": "development-only", "reveal_sha256": hashlib.sha256(raw).hexdigest(),
            "revealed_site_score": score, "publication_warning": warning}
