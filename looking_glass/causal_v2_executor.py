"""Synthetic v2 execution kernel and its restricted development interface.

The public development helpers accept only the two already inventoried reviewer
vectors with first coordinate +/-2. Private shared-kernel helpers also serve a
separate trusted study callback; neither interface generates cases, manages
custody, seals predictions, or authorizes a study run.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib

import torch
from torch import nn
from torch.nn.modules import module as module_hooks

from .causal_trace_v2 import compare_sparse_pair
from .causal_v2_custody import canonical_json
from .causal_v2_model_identity import state_sha256
from .causal_v2_runtime import NumericPolicy, float32_exact_equal
from .causal_v2_selection import DiscoveryCondition
from .intervention_plan import ActivationPlan
from .interventions import ActivationCapture
from .model import ResidualBlock, TinyResidual
from .probes import TapPoint, TraceSession


class DevelopmentInvalid(ValueError):
    """A development condition or its instrumentation failed validation."""


class DevelopmentBudget:
    """Bound submissions before hook preparation and count root forward calls."""
    def __init__(self, limit):
        if type(limit) is not int or not 0 < limit <= 60000:
            raise DevelopmentInvalid("invalid forward budget")
        self.limit = limit
        self.submissions = 0
        self.root_forwards = 0

    def reserve(self):
        if self.submissions != self.root_forwards or self.submissions >= self.limit:
            raise DevelopmentInvalid("development submission is pending or budget exhausted")
        self.submissions += 1

    def root(self):
        if self.root_forwards >= self.submissions:
            raise DevelopmentInvalid("root forward has no submission")
        self.root_forwards += 1


_CACHE_TOKEN = object()
_STUDY_CACHE_TOKEN = object()


@dataclass(frozen=True, eq=False)
class DevelopmentClean:
    clean_trace: tuple
    repeat_trace: tuple
    clean_logits: torch.Tensor
    repeat_logits: torch.Tensor
    input_sha256: str
    model_state_sha256: str
    _payload_sha256: str = field(repr=False)
    _model: object = field(repr=False)
    _budget: object = field(repr=False)
    _policy: object = field(repr=False)
    _token: object = field(repr=False)


@dataclass(frozen=True, eq=False)
class DevelopmentTarget:
    target: str
    value: torch.Tensor
    metadata: dict
    _payload_sha256: str = field(repr=False)
    _clean: object = field(repr=False)
    _token: object = field(repr=False)


def _check_hooks(model):
    if torch.is_autocast_enabled("cpu"):
        raise DevelopmentInvalid("CPU autocast is unsupported")
    if (module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks or
            module_hooks._global_backward_hooks or module_hooks._global_backward_pre_hooks):
        raise DevelopmentInvalid("global forward or backward hooks are unsupported")
    for module in model.modules():
        if (module._forward_hooks or module._forward_pre_hooks or
                module._backward_hooks or module._backward_pre_hooks):
            raise DevelopmentInvalid("pre-existing forward or backward hooks are unsupported")
        if getattr(module.forward, "__func__", None) is not type(module).forward:
            raise DevelopmentInvalid("custom module call paths are unsupported")


def _check_model(model):
    if type(model) is not TinyResidual or model.width != 8 or model.depth != 4:
        raise DevelopmentInvalid("expected the fixed development model architecture")
    if type(model.blocks) is not nn.ModuleList or len(model.blocks) != 4 or type(model.head) is not nn.Linear:
        raise DevelopmentInvalid("invalid development model tree")
    for i, block in enumerate(model.blocks):
        if (type(block) is not ResidualBlock or type(block.branches) is not nn.ModuleList or
                type(block.branch_taps) is not nn.ModuleList or len(block.branches) != 2 or
                len(block.branch_taps) != 2 or any(type(branch) is not nn.Linear for branch in block.branches) or
                type(block.projection) is not nn.Linear):
            raise DevelopmentInvalid("invalid development block")
        points = list(block.branch_taps) + [block.activation_tap, block.merge_tap, block.output_tap]
        names = [f"block.{i}.branch.0", f"block.{i}.branch.1", f"block.{i}.activation",
                 f"block.{i}.merge", f"block.{i}.output"]
        groups = ("dense", "dense", "dense", "dense", "sparse")
        if any(type(point) is not TapPoint or point.node_id != name or point.group != group
               for point, name, group in zip(points, names, groups)):
            raise DevelopmentInvalid("invalid development tap labels")
    modules = [module for _, module in model.named_modules(remove_duplicate=False)]
    if len({id(module) for module in modules}) != len(modules) or any(module.training for module in modules):
        raise DevelopmentInvalid("model must have distinct modules and be in evaluation mode")
    if any(tensor.device.type != "cpu" or tensor.dtype != torch.float32
           for tensor in model.state_dict().values()):
        raise DevelopmentInvalid("model state must be CPU float32")
    _check_hooks(model)


def _check_input(value, policy):
    if (type(value) is not torch.Tensor or tuple(value.shape) != (1, 8) or
            value.dtype != torch.float32 or value.device.type != "cpu"):
        raise DevelopmentInvalid("expected one CPU float32 development vector")
    candidates = [torch.tensor([[first, -.25, .5, .125, -.75, .2, -.3, .4]],
                               dtype=torch.float32, device="cpu") for first in (2., -2.)]
    if not any(_equal(value, candidate, policy) for candidate in candidates):
        raise DevelopmentInvalid("only retained development exclusion vectors are allowed")


def _equal(left, right, policy):
    policy.verify()
    result = float32_exact_equal(left, right)
    policy.verify()
    return result


def _require_equal(left, right, description, policy):
    if not _equal(left, right, policy):
        raise DevelopmentInvalid(description)


def _bits(value):
    return [int(bit) & 0xffffffff for bit in value.detach().view(torch.int32).reshape(-1).tolist()]


def input_sha256(value):
    """Bind a [1,8] CPU float32 input's exact bits in a defined byte order.

    This development identity is SHA-256 of the ASCII domain tag and NUL,
    followed by two unsigned 32-bit big-endian dimensions and eight unsigned
    IEEE binary32 bit patterns in big-endian order. Signed zeros stay distinct.
    It does not bind a model, source revision or custody record.
    """
    if (type(value) is not torch.Tensor or tuple(value.shape) != (1, 8) or
            value.layout != torch.strided or value.is_nested or value.is_neg() or value.is_conj() or
            value.has_names() or value.dtype != torch.float32 or value.device.type != "cpu"):
        raise DevelopmentInvalid("input identity requires a CPU float32 [1,8] tensor")
    bits = _bits(value)
    if any((bit & 0x7f800000) == 0x7f800000 for bit in bits):
        raise DevelopmentInvalid("input identity requires finite values")
    data = b"MLG-v2-input-f32be-v1\0" + (1).to_bytes(4, "big") + (8).to_bytes(4, "big")
    return hashlib.sha256(data + b"".join(bit.to_bytes(4, "big") for bit in bits)).hexdigest()


def _snapshot_sha256(clean_trace, repeat_trace, clean, repeat):
    rows = []
    for stream in (clean_trace, repeat_trace):
        rows.append([{"metadata": event.metadata(), "bits": _bits(event.value)} for event in stream])
    return hashlib.sha256(canonical_json({"traces": rows, "logits": [_bits(clean), _bits(repeat)]})).hexdigest()


def _target_sha256(target, value, metadata):
    return hashlib.sha256(canonical_json({"target": target, "bits": _bits(value), "metadata": metadata})).hexdigest()


def _validate_cached_tensor(value, shape, policy):
    if (type(value) is not torch.Tensor or tuple(value.shape) != shape or value.layout != torch.strided or
            value.is_nested or value.is_neg() or value.is_conj() or value.has_names() or
            value.dtype != torch.float32 or value.device.type != "cpu"):
        raise DevelopmentInvalid("cached tensor shape, dtype or layout differs")
    if not _equal(value, value, policy):
        raise DevelopmentInvalid("cached tensor is not finite")


def _validate_clean(clean, model, x, budget, policy, *, scope=_CACHE_TOKEN, input_check=_check_input):
    if (type(clean) is not DevelopmentClean or clean._token is not scope or
            clean._model is not model or clean._budget is not budget or clean._policy is not policy):
        raise DevelopmentInvalid("clean observation does not belong to this execution")
    policy.verify()
    _check_model(model)
    input_check(x, policy)
    _validate_cached_tensor(clean.clean_logits, (1, 2), policy)
    _validate_cached_tensor(clean.repeat_logits, (1, 2), policy)
    _require_equal(clean.clean_logits, clean.repeat_logits, "cached clean logits differ", policy)
    try:
        pair = _compare(clean.clean_trace, clean.repeat_trace, policy)
    except Exception as error:
        raise DevelopmentInvalid("cached trace payload or metadata differs") from error
    if pair["status"] != "no_difference_observed":
        raise DevelopmentInvalid("cached clean traces differ")
    if (input_sha256(x) != clean.input_sha256 or state_sha256(model) != clean.model_state_sha256 or
            _snapshot_sha256(clean.clean_trace, clean.repeat_trace, clean.clean_logits, clean.repeat_logits)
            != clean._payload_sha256):
        raise DevelopmentInvalid("clean observation or its execution identity changed")
    policy.verify()


def _validate_target(target_capture, target, clean, *, scope=_CACHE_TOKEN):
    if (type(target_capture) is not DevelopmentTarget or target_capture._token is not scope or
            target_capture._clean is not clean or target_capture.target != target):
        raise DevelopmentInvalid("private target capture or its identity changed")
    _, path = _target_spec(target)
    _check_capture_metadata(target_capture.metadata, path)
    _validate_cached_tensor(target_capture.value, (1, 8), clean._policy)
    if _target_sha256(target_capture.target, target_capture.value, target_capture.metadata) != target_capture._payload_sha256:
        raise DevelopmentInvalid("private target capture or its identity changed")


def _compare(reference, candidate, policy):
    policy.verify()
    result = compare_sparse_pair(reference, candidate, reference_complete=True,
                                 candidate_complete=True, pairing_verified=True)
    policy.verify()
    return result


def _check_upstream(reference, candidate, block, policy):
    for index in range(block):
        _require_equal(reference[index].value, candidate[index].value,
                       "private upstream trace integrity failed", policy)


def _forward(model, x, budget, policy, make_session=None, *, perturb=None, reserved=False):
    if type(reserved) is not bool:
        raise DevelopmentInvalid("invalid reservation state")
    if not reserved:
        budget.reserve()  # Counts even if checks, hook preparation or model work fail.
    elif (budget.submissions != budget.root_forwards + 1 or getattr(budget, "_pending", True) is not True):
        raise DevelopmentInvalid("forward has no pending reservation")
    _check_hooks(model)
    policy.verify()
    session = make_session() if make_session is not None else nullcontext()
    with session:
        with torch.no_grad():
            budget.root()
            output = model(x, route=0, perturb=perturb).detach().clone()
    policy.verify()
    _check_hooks(model)
    if tuple(output.shape) != (1, 2) or not _equal(output, output, policy):
        raise DevelopmentInvalid("invalid development logits")
    return output, session


def _trace(model, x, budget, policy, perturb=None, *, reserved=False):
    run_id = f"v2-observation-{budget.submissions if reserved else budget.submissions + 1:05d}"
    output, session = _forward(model, x, budget, policy,
        lambda: TraceSession(model, mode="direct", placement="sparse", run_id=run_id,
                             max_events=4, max_tensor_bytes=128), perturb=perturb, reserved=reserved)
    if len(session.events) != 4 or session.tensor_bytes != 128:
        raise DevelopmentInvalid("incomplete sparse trace")
    return output, tuple(session.events)


def _capture(model, x, path, budget, policy, perturb=None, *, reserved=False):
    output, session = _forward(model, x, budget, policy,
        lambda: ActivationCapture(model, path, invocation=0, max_tensor_bytes=32, max_invocations=1),
        perturb=perturb, reserved=reserved)
    value = session.value
    if (tuple(value.shape) != (1, 8) or value.numel() * value.element_size() != 32 or
            not _equal(value, value, policy)):
        raise DevelopmentInvalid("invalid private target capture")
    metadata = session.metadata
    _check_capture_metadata(metadata, path)
    return output, value, metadata


def _check_capture_metadata(metadata, path):
    expected = {"module_path": path, "module_aliases": [path], "call_site_known": False,
                "forward_index": 0, "invocation": 0, "batch_index": None, "token_index": None,
                "output_shape": [1, 8], "selected_shape": [1, 8],
                "dtype": "torch.float32", "device": "cpu"}
    if canonical_json(metadata) != canonical_json(expected):
        raise DevelopmentInvalid("private capture metadata differs")


def _check_plan_metadata(metadata, path, names, operations, source_capture, capture_bytes, source_bytes):
    if (type(metadata) is not dict or set(metadata) != {"steps", "capture_bytes", "source_bytes"} or
            type(metadata["capture_bytes"]) is not int or metadata["capture_bytes"] != capture_bytes or
            type(metadata["source_bytes"]) is not int or metadata["source_bytes"] != source_bytes or
            type(metadata["steps"]) is not list or len(metadata["steps"]) != len(names)):
        raise DevelopmentInvalid("invalid plan metadata or byte accounting")
    for index, (event, name, operation) in enumerate(zip(metadata["steps"], names, operations)):
        expected = {"name": name, "operation": operation, "order": index, "module_path": path,
                    "module_aliases": [path], "call_site_known": False, "forward_index": 0,
                    "invocation": 0, "batch_index": None, "token_index": None,
                    "output_selector": [], "output_shape": [1, 8], "selected_shape": [1, 8],
                    "dtype": "torch.float32", "device": "cpu",
                    "source_capture": source_capture if operation == "patch" else None,
                    "preceding_same_site_actions": list(names[:index])}
        # Canonical typed JSON comparison avoids Python's False == 0 aliases.
        if canonical_json(event) != canonical_json(expected):
            raise DevelopmentInvalid("unexpected plan event metadata")


def _plan(model, x, site, kind, source, trace_value, budget, policy, perturb):
    budget.reserve()  # Source snapshots and plan arguments are submission preparation.
    path = f"blocks.{site}.output_tap"
    base = {"module_path": path, "invocation": 0}
    owned = source.detach().clone() if source is not None else None
    if kind in ("forward", "reverse", "upstream"):
        argument = owned.clone()
        steps = [dict(base, name="patch", operation="patch", value=argument),
                 dict(base, name="post_action", operation="capture")]
        names, operations, capture_bytes, source_bytes = ("patch", "post_action"), ("patch", "capture"), 32, 32
        source_reference = None
    elif kind == "null":
        argument = None
        steps = [dict(base, name="null", operation="null"), dict(base, name="post_action", operation="capture")]
        names, operations, capture_bytes, source_bytes = ("null", "post_action"), ("null", "capture"), 32, 0
        source_reference = None
    elif kind == "self":
        argument = None
        steps = [dict(base, name="self_source", operation="capture"),
                 dict(base, name="self_patch", operation="patch", source_capture="self_source"),
                 dict(base, name="post_action", operation="capture")]
        names, operations, capture_bytes, source_bytes = (("self_source", "self_patch", "post_action"),
                                                           ("capture", "patch", "capture"), 64, 0)
        source_reference = "self_source"
    else:
        raise DevelopmentInvalid("unsupported development plan")
    output, plan = _forward(model, x, budget, policy,
        lambda: ActivationPlan(model, steps, max_tensor_bytes=32, max_capture_bytes=capture_bytes,
                               max_source_bytes=32, max_steps=len(steps), max_invocations=1),
        perturb=perturb, reserved=True)
    metadata = plan.metadata
    _check_plan_metadata(metadata, path, names, operations, source_reference, capture_bytes, source_bytes)
    captures = plan.captures
    expected_captures = {"self_source", "post_action"} if kind == "self" else {"post_action"}
    if set(captures) != expected_captures:
        raise DevelopmentInvalid("plan capture inventory differs")
    if owned is not None:
        _require_equal(source, owned, "source snapshot changed during plan", policy)
        _require_equal(argument, owned, "plan argument changed during execution", policy)
        _require_equal(captures["post_action"], owned, "patch delivery failed", policy)
    elif kind == "null":
        _require_equal(captures["post_action"], trace_value, "null state differs from trace", policy)
    else:
        _require_equal(captures["self_source"], trace_value, "self source differs from trace", policy)
        _require_equal(captures["post_action"], captures["self_source"], "self patch delivery failed", policy)
    return output, {"kind": kind, "site": site, "metadata": deepcopy(metadata),
                    "captures": {name: _bits(captures[name]) for name in sorted(captures)},
                    "source_identity": {"forward": "clean_sparse_trace", "reverse": "perturbed_sparse_trace",
                                        "upstream": "clean_sparse_trace", "null": None,
                                        "self": "same_forward_self_source"}[kind],
                    "comparison_identity": ("perturbed_sparse_trace" if kind in ("null", "self") else
                                            "clean_sparse_trace" if kind in ("forward", "upstream") else
                                            "perturbed_sparse_trace"),
                    "base_identity": "clean" if kind == "reverse" else "perturbed"}


def _prepare_clean(model, x, budget, policy, *, budget_type, scope, input_check, first_reserved=False):
    """Private shared kernel; the caller supplies its fixed input policy and budget type."""
    try:
        if type(budget) is not budget_type or type(policy) is not NumericPolicy:
            raise DevelopmentInvalid("development budget and numeric policy are required")
        policy.verify()
        _check_model(model)
        input_check(x, policy)
        initial_input, initial_state = input_sha256(x), state_sha256(model)
        clean, clean_trace = _trace(model, x, budget, policy, reserved=first_reserved)
        repeat, repeat_trace = _trace(model, x, budget, policy)
        _require_equal(clean, repeat, "repeated clean logits differ", policy)
        clean_pair = _compare(clean_trace, repeat_trace, policy)
        if clean_pair["status"] != "no_difference_observed":
            raise DevelopmentInvalid("repeated clean sparse trace differs")
        if input_sha256(x) != initial_input or state_sha256(model) != initial_state:
            raise DevelopmentInvalid("model or input changed during clean observation")
        result = DevelopmentClean(clean_trace, repeat_trace, clean, repeat, initial_input, initial_state,
                                  _snapshot_sha256(clean_trace, repeat_trace, clean, repeat),
                                  model, budget, policy, scope)
        _validate_clean(result, model, x, budget, policy, scope=scope, input_check=input_check)
        return result
    except DevelopmentInvalid:
        raise
    except Exception as error:
        raise DevelopmentInvalid("development clean preparation failed") from error


def prepare_development_clean(model, x, budget, policy):
    """Capture a reusable clean/repeat pair using two development forwards."""
    return _prepare_clean(model, x, budget, policy, budget_type=DevelopmentBudget,
                          scope=_CACHE_TOKEN, input_check=_check_input)


def _target_spec(target):
    targets = {f"block.{block}.{op}": (block, op) for block in (1, 2, 3)
               for op in ("branch.0", "activation", "merge")}
    if type(target) is not str or target not in targets:
        raise DevelopmentInvalid("invalid development injection")
    block, op = targets[target]
    path = f"blocks.{block}.branch_taps.0" if op == "branch.0" else f"blocks.{block}.{op}_tap"
    return block, path


def _prepare_target(model, x, target, clean, budget, policy, *, scope, input_check, first_reserved=False):
    try:
        _validate_clean(clean, model, x, budget, policy, scope=scope, input_check=input_check)
        _, path = _target_spec(target)
        capture_logits, clean_target, clean_metadata = _capture(model, x, path, budget, policy,
                                                                reserved=first_reserved)
        _require_equal(capture_logits, clean.clean_logits, "clean target capture logits differ", policy)
        _validate_clean(clean, model, x, budget, policy, scope=scope, input_check=input_check)
        return DevelopmentTarget(target, clean_target, clean_metadata,
                                 _target_sha256(target, clean_target, clean_metadata), clean, scope)
    except DevelopmentInvalid:
        raise
    except Exception as error:
        raise DevelopmentInvalid("development target preparation failed") from error


def prepare_development_target(model, x, target, clean, budget, policy):
    """Capture one private clean target for reuse across strengths."""
    return _prepare_target(model, x, target, clean, budget, policy, scope=_CACHE_TOKEN, input_check=_check_input)


def _run_active(model, x, target, strength, clean, target_capture, budget, policy, *, scope, input_check,
                first_reserved=False):
    try:
        _validate_clean(clean, model, x, budget, policy, scope=scope, input_check=input_check)
        _validate_target(target_capture, target, clean, scope=scope)
        block, path = _target_spec(target)
        if type(strength) is not float or strength not in (.25, .75):
            raise DevelopmentInvalid("invalid development strength")
        clean_trace, repeat_trace = clean.clean_trace, clean.repeat_trace
        clean_logits, repeat_logits = clean.clean_logits, clean.repeat_logits
        clean_target, clean_metadata = target_capture.value, target_capture.metadata
        delta = strength if x[0, 0].item() > 0 else -strength
        perturb = (target, delta)
        changed, changed_trace = _trace(model, x, budget, policy, perturb, reserved=first_reserved)
        compared = _compare(clean_trace, changed_trace, policy)
        if compared["status"] == "upstream_difference":
            raise DevelopmentInvalid("upstream observation changed")
        _check_upstream(clean_trace, changed_trace, block, policy)
        capture_logits, changed_target, changed_metadata = _capture(model, x, path, budget, policy, perturb)
        _require_equal(capture_logits, changed, "perturbed target capture logits differ", policy)
        expected = clean_target.clone()
        expected[0, 0] = expected[0, 0] + delta
        _require_equal(changed_target, expected, "native injection did not land", policy)
        if _equal(clean_target, changed_target, policy):
            raise DevelopmentInvalid("native injection was masked at its target")
        plans = []
        results = {kind: [] for kind in ("forward", "reverse", "null", "self")}
        for kind in ("forward", "reverse", "null", "self"):
            for site in (1, 2, 3):
                source = (clean_trace[site].value if kind == "forward" else
                          changed_trace[site].value if kind == "reverse" else None)
                trace_value = changed_trace[site].value
                output, evidence = _plan(model, x, site, kind, source, trace_value, budget, policy,
                                         None if kind == "reverse" else perturb)
                expected_logits = (clean_logits if site >= block else changed) if kind == "forward" else (
                    (changed if site >= block else clean_logits) if kind == "reverse" else changed)
                _require_equal(output, expected_logits, "development control logits differ", policy)
                results[kind].append(output)
                plans.append(evidence)
        upstream, evidence = _plan(model, x, 0, "upstream", clean_trace[0].value,
                                   changed_trace[0].value, budget, policy, perturb)
        _require_equal(upstream, changed, "upstream control differs", policy)
        plans.append(evidence)
        verification, _ = _forward(model, x, budget, policy)
        _require_equal(verification, clean_logits, "post-hook clean logits differ", policy)
        condition = DiscoveryCondition(strength, changed_trace, changed, tuple(results["forward"]),
                                       tuple(results["reverse"]), tuple(results["null"]), tuple(results["self"]),
                                       upstream, verification)
        _validate_clean(clean, model, x, budget, policy, scope=scope, input_check=input_check)
        _validate_target(target_capture, target, clean, scope=scope)
        return {"clean_trace": clean_trace, "repeat_trace": repeat_trace, "clean_logits": clean_logits,
                "repeat_logits": repeat_logits, "condition": condition, "plans": plans,
                "private_integrity": {"target": target, "clean_target_bits": _bits(clean_target),
                                      "perturbed_target_bits": _bits(changed_target), "delta": delta,
                                      "clean_capture_metadata": deepcopy(clean_metadata),
                                      "perturbed_capture_metadata": deepcopy(changed_metadata)},
                "comparison": compared, "input_sha256": clean.input_sha256,
                "model_state_sha256": clean.model_state_sha256}
    except DevelopmentInvalid:
        raise
    except Exception as error:
        raise DevelopmentInvalid("development active condition failed") from error


def run_development_active(model, x, target, strength, clean, target_capture, budget, policy):
    """Execute the 16-forward development condition with validated reusable inputs."""
    return _run_active(model, x, target, strength, clean, target_capture, budget, policy,
                       scope=_CACHE_TOKEN, input_check=_check_input)


def run_development_condition(model, x, target, strength, budget, policy):
    """Run the standalone 19-forward development exercise with private evidence.

    The caller must not expose ``private_integrity`` or ``target`` to selection.
    This function cannot enforce confidentiality, phase gates or provenance.
    """
    _target_spec(target)
    if type(strength) is not float or strength not in (.25, .75):
        raise DevelopmentInvalid("invalid development strength")
    clean = prepare_development_clean(model, x, budget, policy)
    private_target = prepare_development_target(model, x, target, clean, budget, policy)
    return run_development_active(model, x, target, strength, clean, private_target, budget, policy)
