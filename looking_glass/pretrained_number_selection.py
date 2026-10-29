"""Site-blind discovery and evaluation scoring for the number study.

This module does not run a model, choose or accept a hidden injection site,
draw a held-out split, attest private controls, or publish evidence. The
custodian must save raw observation bytes and verify private action witnesses.
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
# Every finite binary32 number has magnitude at most F. A 768-vector L2
# difference is at most 2*sqrt(768)*F < 64*F; margins and effects are smaller.
_MAX_ANALYSIS_SCALAR = 64.0 * (2.0 ** 128 - 2.0 ** 104)


class DiscoveryInvalid(ValueError):
    """A discovery case is incomplete; the caller must preserve its evidence."""


class EvaluationInvalid(ValueError):
    """Evaluation is incomplete; the caller must preserve its evidence."""


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


@dataclass(frozen=True)
class EvaluationCase(DiscoveryCase):
    """One custodian-provided case from the committed held-out record."""
    text: str
    input_ids: tuple


def _input_sha256(ids):
    return hashlib.sha256(b"MLG-number-input-v1\0" + bytes([len(ids)]) +
                          b"".join(value.to_bytes(4, "big") for value in ids)).hexdigest()


def _same_typed(actual, expected):
    """Compare schema values without accepting bool as int or list as tuple."""
    if type(actual) is not type(expected):
        return False
    if type(expected) in (tuple, list):
        return len(actual) == len(expected) and all(_same_typed(a, b) for a, b in zip(actual, expected))
    if type(expected) is dict:
        return (actual.keys() == expected.keys() and
                all(_same_typed(actual[key], value) for key, value in expected.items()))
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


def _score_case(case, index, text, ids, execution_device, seen):
    if type(case.forward) is not tuple or len(case.forward) != 4 or type(case.reverse) is not tuple or len(case.reverse) != 4:
        raise DiscoveryInvalid("candidate patch results are incomplete")
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
    clean_margin, injected_margin = _margin(case.clean.logits), _margin(case.injected.logits)
    forward_margins = [_margin(observation.logits) for observation in case.forward]
    reverse_margins = [_margin(observation.logits) for observation in case.reverse]
    current_effects = [math.fsum((margin, -injected_margin)) for margin in forward_margins]
    reverse_effects = [math.fsum((margin, -clean_margin)) for margin in reverse_margins]
    current_norms = [_norm(case.clean.mlp[site], case.injected.mlp[site]) for site in _SITES]
    return {"index": index, "family_index": index // 2, "number": "singular" if index % 2 == 0 else "plural",
                    "text": text, "input_ids": list(ids), "first_changed": first,
                    "l2": current_norms, "clean_margin": clean_margin, "injected_margin": injected_margin,
                    "clean_correct": bool(clean_margin > 0 if index % 2 == 0 else clean_margin < 0),
                    "clean_past_margin": _margin(case.clean.logits, 373, 547),
                    "injected_past_margin": _margin(case.injected.logits, 373, 547),
                    "forward_margins": forward_margins, "reverse_margins": reverse_margins,
                    "forward_effects": current_effects, "reverse_effects": reverse_effects,
                    "observations": [_inventory(role, observation) for role, observation, _ in entries]}


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
        text, ids = PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2]
        record = _score_case(case, index, text, ids, execution_device, seen)
        if record["first_changed"] is not None:
            votes[record["first_changed"]] += 1
        for site in _SITES:
            norms[site].append(record["l2"][site - 1])
            effects[site].append(record["forward_effects"][site - 1])
        records.append(record)
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


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _float_list(value):
    return (type(value) is list and len(value) == 4 and
            all(type(item) is float and math.isfinite(item) and abs(item) <= _MAX_ANALYSIS_SCALAR
                for item in value))


def _load_prediction(raw, digest, execution_device):
    """Parse owned immutable bytes and check selection-level consistency.

    The caller, not this parser, must prove that these bytes are the prediction
    committed in the published aggregate and that the raw observations exist.
    """
    if (type(raw) is not bytes or not 1 <= len(raw) <= 2 * 1024 * 1024 or
            not _digest(digest) or hashlib.sha256(raw).hexdigest() != digest):
        raise EvaluationInvalid("prediction bytes or digest are invalid")
    try:
        parsed = json.loads(raw.decode("utf-8"))
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as error:
        raise EvaluationInvalid("prediction JSON is invalid") from error
    if canonical != raw or type(parsed) is not dict:
        raise EvaluationInvalid("prediction JSON is not canonical")
    keys = {"schema", "status", "protocol_sha256", "checkpoint_revision", "execution_device",
            "private_controls", "first_changed", "ranking", "votes", "mean_l2", "earliest",
            "random", "magnitude", "clean_accuracy", "predictions", "cases"}
    if (set(parsed) != keys or parsed["schema"] != "mlg-number-injection-discovery-prediction-v1" or
            parsed["status"] != "complete" or parsed["protocol_sha256"] != PROTOCOL_SHA256 or
            parsed["checkpoint_revision"] != CHECKPOINT_REVISION or
            parsed["execution_device"] != execution_device or
            parsed["private_controls"] != "caller-attested"):
        raise EvaluationInvalid("prediction identity or schema does not match")
    cases = parsed["cases"]
    site_keys = {str(site) for site in _SITES}
    if (type(cases) is not list or len(cases) != 8 or
            any(type(parsed[key]) is not dict or set(parsed[key]) != site_keys
                for key in ("votes", "mean_l2", "predictions"))):
        raise EvaluationInvalid("prediction cases or sites are incomplete")
    votes = {site: 0 for site in _SITES}
    norms, effects = {site: [] for site in _SITES}, {site: [] for site in _SITES}
    record_keys = {"index", "family_index", "number", "text", "input_ids", "first_changed", "l2",
                   "clean_margin", "injected_margin", "clean_correct", "clean_past_margin",
                   "injected_past_margin", "forward_margins", "reverse_margins", "forward_effects",
                   "reverse_effects", "observations"}
    for index, record in enumerate(cases):
        text, ids = PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2]
        if (type(record) is not dict or set(record) != record_keys or
                type(record["index"]) is not int or record["index"] != index or
                type(record["family_index"]) is not int or record["family_index"] != index // 2 or
                record["number"] != ("singular" if index % 2 == 0 else "plural") or
                type(record["text"]) is not str or record["text"] != text or
                type(record["input_ids"]) is not list or
                any(type(token) is not int for token in record["input_ids"]) or record["input_ids"] != list(ids) or
                type(record["observations"]) is not list or len(record["observations"]) != 16 or
                any(not _float_list(record[name]) for name in
                    ("l2", "forward_margins", "reverse_margins", "forward_effects", "reverse_effects")) or
                any(value < 0 for value in record["l2"]) or
                any(type(record[name]) is not float or not math.isfinite(record[name]) or
                    abs(record[name]) > _MAX_ANALYSIS_SCALAR for name in
                    ("clean_margin", "injected_margin", "clean_past_margin", "injected_past_margin")) or
                type(record["clean_correct"]) is not bool):
            raise EvaluationInvalid("prediction case is malformed")
        first = record["first_changed"]
        if first is not None:
            if type(first) is not int or first not in _SITES:
                raise EvaluationInvalid("prediction vote is invalid")
            votes[first] += 1
        expected_correct = (record["clean_margin"] > 0 if index % 2 == 0 else record["clean_margin"] < 0)
        if record["clean_correct"] is not expected_correct:
            raise EvaluationInvalid("prediction clean outcome is inconsistent")
        for site in _SITES:
            i = site - 1
            if (record["forward_effects"][i] != math.fsum((record["forward_margins"][i], -record["injected_margin"])) or
                    record["reverse_effects"][i] != math.fsum((record["reverse_margins"][i], -record["clean_margin"]))):
                raise EvaluationInvalid("prediction effects are inconsistent")
            norms[site].append(record["l2"][i])
            effects[site].append(record["forward_effects"][i])
    ranking = sorted(_SITES, key=lambda site: (-votes[site], site))
    mean_l2 = {site: math.fsum(norms[site]) / 8 for site in _SITES}
    expected_accuracy = {"correct": sum(row["clean_correct"] for row in cases),
                         "incorrect": sum(not row["clean_correct"] for row in cases),
                         "ties": sum(row["clean_margin"] == 0 for row in cases)}
    random = 1 + hashlib.sha256(b"MLG-number-v1-injection-random\0" + (20261028).to_bytes(8, "big")).digest()[0] % 4
    if (not _same_typed(parsed["ranking"], ranking) or
            not _same_typed(parsed["first_changed"], ranking[0] if votes[ranking[0]] else None) or
            not _same_typed(parsed["votes"], {str(site): votes[site] for site in _SITES}) or
            not _same_typed(parsed["mean_l2"], {str(site): mean_l2[site] for site in _SITES}) or
            not _same_typed(parsed["predictions"], {str(site): math.fsum(effects[site]) / 8 for site in _SITES}) or
            not _same_typed(parsed["clean_accuracy"], expected_accuracy) or
            not _same_typed(parsed["earliest"], 1) or not _same_typed(parsed["random"], random) or
            not _same_typed(parsed["magnitude"], max(_SITES, key=lambda site: (mean_l2[site], -site)))):
        raise EvaluationInvalid("prediction selections or means are inconsistent")
    return parsed


def revealed_site_rank(discovery_bytes, *, prediction_sha256, execution_device, revealed_layer):
    """Return the revealed site's 1-based position in the frozen ranking.

    The custodian must first verify the site commitment and publication order,
    and bind the expected prediction digest to the published freeze. This pure
    lookup does not establish those facts or alter the discovery selection.
    """
    if (type(execution_device) is not str or execution_device not in ("cpu", "cuda:0") or
            type(revealed_layer) is not int or revealed_layer not in _SITES):
        raise EvaluationInvalid("supported device and revealed layer are required")
    prediction = _load_prediction(discovery_bytes, prediction_sha256, execution_device)
    return prediction["ranking"].index(revealed_layer) + 1


def _squared_logits(cases, site):
    for case in cases:
        clean = decode_f32_le(case.clean.logits)
        other = decode_f32_le(case.injected.logits if site is None else case.forward[site - 1].logits)
        yield from ((float(a) - float(b)) ** 2 for a, b in zip(other, clean))


def score_evaluation(cases, *, discovery_bytes, prediction_sha256, prediction_freeze_sha256,
                     heldout_commitment_sha256, custody_verified=False,
                     private_controls_verified=False, execution_device):
    """Score eight committed cases without changing frozen discovery selections.

    Commitment order, raw evidence and private intervention controls are
    caller-attested; the hashes alone cannot prove publication or execution.
    Returns canonical JSON bytes, to be kept private until the protocol reveal.
    """
    if (custody_verified is not True or private_controls_verified is not True or
            type(execution_device) is not str or execution_device not in ("cpu", "cuda:0") or
            not _digest(prediction_freeze_sha256) or not _digest(heldout_commitment_sha256)):
        raise EvaluationInvalid("custodian attestations and commitment digests are required")
    prediction = _load_prediction(discovery_bytes, prediction_sha256, execution_device)
    if type(cases) is not tuple or len(cases) != 8:
        raise EvaluationInvalid("exactly eight ordered committed cases are required")
    seen, seen_inputs, records = set(), set(), []
    for index, case in enumerate(cases):
        if (type(case) is not EvaluationCase or type(case.index) is not int or case.index != index or
                type(case.text) is not str or not case.text or len(case.text) > 4096 or
                type(case.input_ids) is not tuple or not 1 <= len(case.input_ids) <= 32 or
                any(type(token) is not int or not 0 <= token < 50257 for token in case.input_ids) or
                case.input_ids in seen_inputs):
            raise EvaluationInvalid("committed case identity or ordering is invalid")
        seen_inputs.add(case.input_ids)
        try:
            records.append(_score_case(case, index, case.text, case.input_ids, execution_device, seen))
        except DiscoveryInvalid as error:
            raise EvaluationInvalid("committed case or control failed") from error
    denominator = math.fsum(_squared_logits(cases, None))
    sites = {}
    for site in _SITES:
        predicted = prediction["predictions"][str(site)]
        effects = [record["forward_effects"][site - 1] for record in records]
        reverse = [record["reverse_effects"][site - 1] for record in records]
        errors = [math.fsum((effect, -predicted)) for effect in effects]
        numerator = math.fsum(_squared_logits(cases, site))
        sites[str(site)] = {
            "predicted_effect": predicted, "effects": effects, "reverse_effects": reverse,
            "errors": errors, "zero_errors": effects,
            "mae": math.fsum(abs(error) for error in errors) / 8,
            "rmse": math.sqrt(math.fsum(error * error for error in errors) / 8),
            "zero_mae": math.fsum(abs(effect) for effect in effects) / 8,
            "zero_rmse": math.sqrt(math.fsum(effect * effect for effect in effects) / 8),
            "families": [{"family_index": family, "case_indices": [2 * family, 2 * family + 1],
                          "effects": effects[2 * family:2 * family + 2],
                          "errors": errors[2 * family:2 * family + 2],
                          "zero_errors": effects[2 * family:2 * family + 2]}
                         for family in range(4)],
            "restoration": {"numerator": numerator, "denominator": denominator,
                            "value": None if denominator <= 1e-8 else 1 - numerator / denominator}}
    result = {"schema": "mlg-number-injection-evaluation-v1", "status": "complete",
              "protocol_sha256": PROTOCOL_SHA256, "checkpoint_revision": CHECKPOINT_REVISION,
              "execution_device": execution_device, "private_controls": "caller-attested",
              "custody": "caller-attested", "prediction_sha256": prediction_sha256,
              "prediction_freeze_sha256": prediction_freeze_sha256,
              "heldout_commitment_sha256": heldout_commitment_sha256,
              "selection": {name: prediction[name] for name in
                            ("first_changed", "ranking", "votes", "earliest", "random", "magnitude")},
              "clean_accuracy": {"correct": sum(record["clean_correct"] for record in records),
                                 "incorrect": sum(not record["clean_correct"] for record in records),
                                 "ties": sum(record["clean_margin"] == 0 for record in records)},
              "sites": sites, "cases": records}
    return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")
