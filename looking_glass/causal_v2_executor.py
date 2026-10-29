"""Development-only execution of one synthetic v2-style condition.

Only the two already inventoried reviewer vectors with first coordinate +/-2
are accepted. This module does not generate study cases, manage custody, seal
predictions, execute held-out work, or authorize a study run.
"""
from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import nn
from torch.nn.modules import module as module_hooks

from .causal_trace_v2 import compare_sparse_pair
from .causal_v2_custody import canonical_json
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
        if self.submissions >= self.limit:
            raise DevelopmentInvalid("development submission budget exhausted")
        self.submissions += 1

    def root(self):
        if self.root_forwards >= self.submissions:
            raise DevelopmentInvalid("root forward has no submission")
        self.root_forwards += 1


def _check_hooks(model):
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
        if any(type(point) is not TapPoint or point.node_id != name for point, name in zip(points, names)):
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


def _forward(model, x, budget, policy, make_session=None, *, perturb=None):
    budget.reserve()  # Counts even if checks, hook preparation or model work fail.
    _check_hooks(model)
    policy.verify()
    session = make_session() if make_session is not None else nullcontext()
    with session:
        budget.root()
        with torch.no_grad():
            output = model(x, route=0, perturb=perturb).detach().clone()
    policy.verify()
    _check_hooks(model)
    if tuple(output.shape) != (1, 2) or not _equal(output, output, policy):
        raise DevelopmentInvalid("invalid development logits")
    return output, session


def _trace(model, x, budget, policy, perturb=None):
    run_id = f"development-{budget.submissions + 1:05d}"
    output, session = _forward(model, x, budget, policy,
        lambda: TraceSession(model, mode="direct", placement="sparse", run_id=run_id,
                             max_events=4, max_tensor_bytes=128), perturb=perturb)
    if len(session.events) != 4 or session.tensor_bytes != 128:
        raise DevelopmentInvalid("incomplete sparse trace")
    return output, tuple(session.events)


def _capture(model, x, path, budget, policy, perturb=None):
    output, session = _forward(model, x, budget, policy,
        lambda: ActivationCapture(model, path, invocation=0, max_tensor_bytes=32, max_invocations=1),
        perturb=perturb)
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
        perturb=perturb)
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


def run_development_condition(model, x, target, strength, budget, policy):
    """Run 19 CPU forwards on retained development vectors; return private evidence.

    The caller must not expose ``private_integrity`` or ``target`` to selection.
    This function cannot enforce confidentiality, phase gates or provenance.
    """
    try:
        if type(budget) is not DevelopmentBudget or type(policy) is not NumericPolicy:
            raise DevelopmentInvalid("development budget and numeric policy are required")
        policy.verify()
        _check_model(model)
        _check_input(x, policy)
        targets = {f"block.{block}.{op}": (block, op) for block in (1, 2, 3)
                   for op in ("branch.0", "activation", "merge")}
        if type(target) is not str or target not in targets or type(strength) is not float or strength not in (.25, .75):
            raise DevelopmentInvalid("invalid development injection")
        block, op = targets[target]
        path = f"blocks.{block}.branch_taps.0" if op == "branch.0" else f"blocks.{block}.{op}_tap"
        delta = strength if x[0, 0].item() > 0 else -strength
        perturb = (target, delta)
        clean, clean_trace = _trace(model, x, budget, policy)
        repeat, repeat_trace = _trace(model, x, budget, policy)
        _require_equal(clean, repeat, "repeated clean logits differ", policy)
        clean_pair = _compare(clean_trace, repeat_trace, policy)
        if clean_pair["status"] != "no_difference_observed":
            raise DevelopmentInvalid("repeated clean sparse trace differs")
        capture_logits, clean_target, clean_metadata = _capture(model, x, path, budget, policy)
        _require_equal(capture_logits, clean, "clean target capture logits differ", policy)
        changed, changed_trace = _trace(model, x, budget, policy, perturb)
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
                expected_logits = (clean if site >= block else changed) if kind == "forward" else (
                    (changed if site >= block else clean) if kind == "reverse" else changed)
                _require_equal(output, expected_logits, "development control logits differ", policy)
                results[kind].append(output)
                plans.append(evidence)
        upstream, evidence = _plan(model, x, 0, "upstream", clean_trace[0].value,
                                   changed_trace[0].value, budget, policy, perturb)
        _require_equal(upstream, changed, "upstream control differs", policy)
        plans.append(evidence)
        verification, _ = _forward(model, x, budget, policy)
        _require_equal(verification, clean, "post-hook clean logits differ", policy)
        condition = DiscoveryCondition(strength, changed_trace, changed, tuple(results["forward"]),
                                       tuple(results["reverse"]), tuple(results["null"]), tuple(results["self"]),
                                       upstream, verification)
        policy.verify()
        return {"clean_trace": clean_trace, "repeat_trace": repeat_trace, "clean_logits": clean,
                "repeat_logits": repeat, "condition": condition, "plans": plans,
                "private_integrity": {"target": target, "clean_target_bits": _bits(clean_target),
                                      "perturbed_target_bits": _bits(changed_target), "delta": delta,
                                      "clean_capture_metadata": clean_metadata,
                                      "perturbed_capture_metadata": changed_metadata},
                "comparison": compared}
    except DevelopmentInvalid:
        raise
    except Exception as error:
        raise DevelopmentInvalid("development condition failed") from error
