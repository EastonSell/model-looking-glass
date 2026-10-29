"""Site-blind discovery scoring for the public DistilGPT2 number study.

This module only accepts the eight prescribed public discovery inputs. It does
not run a model, choose or accept a hidden injection site, draw a held-out
split, attest private controls, or publish evidence. The custodian must save the
raw observation bytes and independently verify the private action witnesses.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import numpy as np

from .agreement_numeric import compare_f32_le, decode_f32_le
from .pretrained_exposures_v2 import PUBLIC_DISCOVERY


PROTOCOL_SHA256 = "95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee"
CHECKPOINT_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
_WIDTH_BYTES = 768 * 4
_LOGIT_BYTES = 50257 * 4
_SITES = (1, 2, 3, 4)
_REQUEST = re.compile(r"[0-9a-f]{32}\Z")


class DiscoveryInvalid(ValueError):
    """A discovery case is incomplete; the caller must preserve its evidence."""


@dataclass(frozen=True)
class Boundary:
    module_path: str
    capture_order: int
    invocation: int
    batch_index: int
    token_index: int
    output_selector: tuple
    output_shape: tuple
    selected_shape: tuple
    dtype: str
    device: str


@dataclass(frozen=True)
class Donor:
    request_id: str
    input_sha256: str
    module_path: str
    invocation: int
    batch_index: int
    token_index: int
    output_selector: tuple
    selected_shape: tuple
    dtype: str
    device: str


@dataclass(frozen=True)
class Action:
    operation: str
    module_path: str
    invocation: int
    batch_index: int
    token_index: int
    output_selector: tuple
    output_shape: tuple
    selected_shape: tuple
    dtype: str
    device: str
    donor: Donor | None


@dataclass(frozen=True)
class Observation:
    request_id: str
    input_sha256: str
    condition: str
    attention: tuple
    mlp: tuple
    block: bytes
    logits: bytes
    trace: tuple
    action: Action | None


@dataclass(frozen=True)
class DiscoveryCase:
    index: int
    clean: Observation
    repeat: Observation
    injected: Observation
    forward: tuple
    reverse: tuple
    null: Observation
    self_patch: Observation
    upstream: Observation
    downstream: Observation
    verify_clean: Observation


def _input_sha256(ids):
    return hashlib.sha256(b"MLG-number-input-v1\0" + bytes([len(ids)]) +
                          b"".join(value.to_bytes(4, "big") for value in ids)).hexdigest()


def _same_typed(actual, expected):
    """Compare schema values without accepting bool as int or list as tuple."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is tuple:
        return len(actual) == len(expected) and all(_same_typed(a, b) for a, b in zip(actual, expected))
    if hasattr(expected, "__dataclass_fields__"):
        return all(_same_typed(getattr(actual, key), getattr(expected, key))
                   for key in expected.__dataclass_fields__)
    return actual == expected


def _finite(raw, length):
    if type(raw) is not bytes or len(raw) != length:
        raise DiscoveryInvalid("observation payload has the wrong type or size")
    bits = np.frombuffer(raw, dtype="<u4")
    if np.any((bits & np.uint32(0x7f800000)) == np.uint32(0x7f800000)):
        raise DiscoveryInvalid("observation contains a nonfinite value")


def _within(left, right):
    return compare_f32_le(left, right)["within_tolerance"]


def _require_within(left, right):
    if not _within(left, right):
        raise DiscoveryInvalid("a required discovery control failed")


def _boundary(path, order, length, device):
    selector = (0,) if path.endswith(".attn") or path == "transformer.h.5" else ()
    return Boundary(path, order, 0, 0, length - 1, selector, (1, length, 768), (768,),
                    "torch.float32", device)


def _observation(observation, *, condition, ids, device, seen):
    if (type(observation) is not Observation or type(observation.request_id) is not str or
            _REQUEST.fullmatch(observation.request_id) is None or observation.request_id in seen or
            type(observation.input_sha256) is not str or observation.input_sha256 != _input_sha256(ids) or
            type(observation.condition) is not str or observation.condition != condition):
        raise DiscoveryInvalid("observation identity or condition does not match")
    seen.add(observation.request_id)
    if (type(observation.attention) is not tuple or len(observation.attention) != 6 or
            type(observation.mlp) is not tuple or len(observation.mlp) != 6 or
            type(observation.trace) is not tuple or len(observation.trace) != 13):
        raise DiscoveryInvalid("observation trace is incomplete")
    paths = [f"transformer.h.{layer}.{kind}" for layer in range(6) for kind in ("attn", "mlp")]
    paths.append("transformer.h.5")
    for index, (actual, path) in enumerate(zip(observation.trace, paths)):
        if not _same_typed(actual, _boundary(path, index, len(ids), device)):
            raise DiscoveryInvalid("observation trace coordinates do not match")
    for raw in observation.attention + observation.mlp + (observation.block,):
        _finite(raw, _WIDTH_BYTES)
    _finite(observation.logits, _LOGIT_BYTES)


def _check_action(observation, *, path, donor, ids, device, operation="patch"):
    selector = (0,) if path == "transformer.h.5" else ()
    source = None if donor is None else Donor(donor.request_id, donor.input_sha256, path, 0, 0,
                                               len(ids) - 1, selector, (768,), "torch.float32", device)
    expected = Action(operation, path, 0, 0, len(ids) - 1, selector,
                      (1, len(ids), 768), (768,), "torch.float32", device, source)
    if not _same_typed(observation.action, expected):
        raise DiscoveryInvalid("action or donor coordinates do not match")


def _same_trace(left, right):
    for a, b in zip(left.attention + left.mlp + (left.block,), right.attention + right.mlp + (right.block,)):
        _require_within(a, b)
    _require_within(left.logits, right.logits)


def _margin(raw, first=318, second=389):
    # Decode binary32 as exact binary64, avoiding FTZ-sensitive float32 casts.
    a = float(decode_f32_le(raw[4 * first:4 * first + 4])[0])
    b = float(decode_f32_le(raw[4 * second:4 * second + 4])[0])
    return math.fsum((a, -b))


def _norm(left, right):
    a, b = decode_f32_le(left), decode_f32_le(right)
    return math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _jsonable(value):
    if type(value) is tuple:
        return [_jsonable(item) for item in value]
    if type(value) is list:
        return [_jsonable(item) for item in value]
    if type(value) is dict:
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _inventory(role, observation):
    """Bind separately retained raw bytes and public coordinates by SHA-256."""
    return {"role": role, "request_id": observation.request_id,
            "input_sha256": observation.input_sha256, "condition": observation.condition,
            "attention_sha256": [hashlib.sha256(raw).hexdigest() for raw in observation.attention],
            "mlp_sha256": [hashlib.sha256(raw).hexdigest() for raw in observation.mlp],
            "block_sha256": hashlib.sha256(observation.block).hexdigest(),
            "logits_sha256": hashlib.sha256(observation.logits).hexdigest(),
            "present_margin": _margin(observation.logits),
            "past_margin": _margin(observation.logits, 373, 547),
            "trace": [_jsonable(asdict(item)) for item in observation.trace],
            "action": None if observation.action is None else _jsonable(asdict(observation.action))}


def freeze_discovery(cases, *, private_controls_verified=False, execution_device):
    """Return deterministic prediction bytes for exactly eight public cases.

    ``private_controls_verified`` is a custodian assertion, not proof of the
    hidden injection's site, action integrity, prior publication, or blindness.
    The caller must save and publish all raw observations and failed controls
    separately. This function accepts no private site and never draws a split.
    """
    if (private_controls_verified is not True or type(execution_device) is not str or
            execution_device not in ("cpu", "cuda:0")):
        raise DiscoveryInvalid("custodian attestation and supported execution device are required")
    if type(cases) is not tuple or len(cases) != 8:
        raise DiscoveryInvalid("exactly eight ordered public discovery cases are required")
    seen, votes = set(), {site: 0 for site in _SITES}
    norms, effects = {site: [] for site in _SITES}, {site: [] for site in _SITES}
    records = []
    for index, case in enumerate(cases):
        if type(case) is not DiscoveryCase or type(case.index) is not int or case.index != index:
            raise DiscoveryInvalid("discovery case ordering is invalid")
        if type(case.forward) is not tuple or len(case.forward) != 4 or type(case.reverse) is not tuple or len(case.reverse) != 4:
            raise DiscoveryInvalid("candidate patch results are incomplete")
        text, ids = PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2]
        entries = [("clean", case.clean, "clean"), ("repeat", case.repeat, "clean"),
                   ("injected", case.injected, "injected")]
        entries += [(f"forward_{site}", case.forward[site - 1], "clean_to_injected") for site in _SITES]
        entries += [(f"reverse_{site}", case.reverse[site - 1], "injected_to_clean") for site in _SITES]
        entries += [("null", case.null, "null"), ("self", case.self_patch, "self"),
                    ("upstream", case.upstream, "upstream"), ("downstream", case.downstream, "downstream"),
                    ("verify_clean", case.verify_clean, "clean")]
        for _, observation, condition in entries:
            _observation(observation, condition=condition, ids=ids, device=execution_device, seen=seen)
        for observation in (case.clean, case.repeat, case.injected, case.verify_clean):
            if observation.action is not None:
                raise DiscoveryInvalid("unmodified observation includes an action")
        for site in _SITES:
            path = f"transformer.h.{site}.mlp"
            forward, reverse = case.forward[site - 1], case.reverse[site - 1]
            _check_action(forward, path=path, donor=case.clean, ids=ids, device=execution_device)
            _check_action(reverse, path=path, donor=case.injected, ids=ids, device=execution_device)
            if forward.mlp[site] != case.clean.mlp[site] or reverse.mlp[site] != case.injected.mlp[site]:
                raise DiscoveryInvalid("candidate patch did not reproduce its donor bits")
        _check_action(case.null, path="transformer.h.1.mlp", donor=None, ids=ids, device=execution_device, operation="null")
        _check_action(case.self_patch, path="transformer.h.1.mlp", donor=case.self_patch, ids=ids, device=execution_device)
        _check_action(case.upstream, path="transformer.h.0.mlp", donor=case.clean, ids=ids, device=execution_device)
        _check_action(case.downstream, path="transformer.h.5", donor=case.clean, ids=ids, device=execution_device)
        if case.upstream.mlp[0] != case.clean.mlp[0] or case.downstream.block != case.clean.block:
            raise DiscoveryInvalid("control patch did not reproduce its donor bits")
        _same_trace(case.clean, case.repeat)
        _same_trace(case.clean, case.verify_clean)
        for observation in (case.null, case.self_patch):
            _require_within(observation.mlp[1], case.injected.mlp[1])
            _require_within(observation.logits, case.injected.logits)
        _require_within(case.upstream.logits, case.injected.logits)
        _require_within(case.downstream.logits, case.clean.logits)
        for left, right in ((case.clean.attention[0], case.injected.attention[0]),
                            (case.clean.mlp[0], case.injected.mlp[0]),
                            (case.clean.attention[1], case.injected.attention[1])):
            _require_within(left, right)
        first = next((site for site in _SITES if not _within(case.clean.mlp[site], case.injected.mlp[site])), None)
        if first is not None:
            votes[first] += 1
        clean_margin, injected_margin = _margin(case.clean.logits), _margin(case.injected.logits)
        forward_margins = [_margin(observation.logits) for observation in case.forward]
        reverse_margins = [_margin(observation.logits) for observation in case.reverse]
        current_effects = [math.fsum((margin, -injected_margin)) for margin in forward_margins]
        reverse_effects = [math.fsum((margin, -clean_margin)) for margin in reverse_margins]
        current_norms = [_norm(case.clean.mlp[site], case.injected.mlp[site]) for site in _SITES]
        for site in _SITES:
            norms[site].append(current_norms[site - 1])
            effects[site].append(current_effects[site - 1])
        records.append({"index": index, "family_index": index // 2, "number": "singular" if index % 2 == 0 else "plural",
                        "text": text, "input_ids": list(ids), "first_changed": first,
                        "l2": current_norms, "clean_margin": clean_margin, "injected_margin": injected_margin,
                        "clean_correct": bool(clean_margin > 0 if index % 2 == 0 else clean_margin < 0),
                        "clean_past_margin": _margin(case.clean.logits, 373, 547),
                        "injected_past_margin": _margin(case.injected.logits, 373, 547),
                        "forward_margins": forward_margins, "reverse_margins": reverse_margins,
                        "forward_effects": current_effects, "reverse_effects": reverse_effects,
                        "observations": [_inventory(role, observation) for role, observation, _ in entries]})
    ranking = sorted(_SITES, key=lambda site: (-votes[site], site))
    mean_l2 = {site: math.fsum(norms[site]) / 8 for site in _SITES}
    random = 1 + hashlib.sha256(b"MLG-number-v1-injection-random\0" + (20261028).to_bytes(8, "big")).digest()[0] % 4
    result = {"schema": "mlg-number-injection-discovery-prediction-v1", "status": "complete",
              "protocol_sha256": PROTOCOL_SHA256, "checkpoint_revision": CHECKPOINT_REVISION,
              "execution_device": execution_device, "private_controls": "caller-attested",
              "first_changed": ranking[0] if votes[ranking[0]] else None, "ranking": ranking,
              "votes": {str(site): votes[site] for site in _SITES},
              "mean_l2": {str(site): mean_l2[site] for site in _SITES},
              "earliest": 1, "random": random,
              "magnitude": max(_SITES, key=lambda site: (mean_l2[site], -site)),
              "clean_accuracy": {"correct": sum(row["clean_correct"] for row in records),
                                 "incorrect": sum(not row["clean_correct"] for row in records),
                                 "ties": sum(row["clean_margin"] == 0 for row in records)},
              "predictions": {str(site): math.fsum(effects[site]) / 8 for site in _SITES},
              "cases": records}
    return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")
