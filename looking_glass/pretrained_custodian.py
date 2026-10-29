"""CPU-only development custodian for synthetic injected-change studies.

No checkpoint, private study split or GPU worker is created here. This trusted
in-process helper is not a security boundary or an approved recorded-v1 runner.
"""
import hashlib
import json
from threading import Event, Lock

import torch
from torch import nn

from .intervention_plan import ActivationPlan
from . import pretrained_causal_experiment as study
from . import pretrained_protocol as protocol


class CustodianExecutionError(RuntimeError):
    """A development custodian failed without disclosing private plan details."""


class CustodianCancelled(KeyboardInterrupt):
    """A development custodian was interrupted and quarantined."""


class DevelopmentCustodian:
    """Assemble the v1 intervention controls on an exclusively owned CPU model.

    This object and ``private_evidence()`` are custodian-private. Public cases
    contain only sanitized observations; they do not include plan metadata,
    private pre/post vectors, or the synthetic site label. Cooperative Python
    model ownership is not process isolation or a GPU lease.
    """

    def __init__(self, model, *, synthetic_site, max_study_bytes=study.STUDY_BYTES):
        if type(synthetic_site) is not int or synthetic_site not in protocol.ELIGIBLE_SITES:
            raise ValueError("synthetic site must be an eligible integer layer")
        if type(max_study_bytes) is not int or not 0 < max_study_bytes <= study.STUDY_BYTES:
            raise ValueError("invalid total capture payload cap")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a CPU module")
        self._model = model
        self._site = synthetic_site
        self._max_study_bytes = max_study_bytes
        self._charged = 0
        self._private = []
        self._failures = []
        self._private_metadata_bytes = 0
        self._current = None
        self._case_runs = 0
        self._active_plan = None
        self._poisoned = Event()
        self._lock = Lock()
        self._binding = self._model_binding()

    @property
    def poisoned(self):
        return self._poisoned.is_set()

    def private_evidence(self):
        """Return custodian-private snapshots; never send this to the investigator."""
        return tuple(dict(row) for row in self._private)

    def private_failures(self):
        """Return bounded, best-effort custodian-private failure diagnostics."""
        return tuple(dict(row) for row in self._failures)

    def _model_binding(self):
        model = self._model
        config = getattr(model, "config", None)
        if (config is None or type(getattr(config, "n_layer", None)) is not int or config.n_layer != 6 or
                type(getattr(config, "n_embd", None)) is not int or config.n_embd != study.WIDTH or
                type(getattr(config, "vocab_size", None)) is not int or config.vocab_size != study.VOCAB_SIZE or
                getattr(config, "use_cache", None) is not False or
                getattr(config, "_attn_implementation", None) != "eager" or
                getattr(model, "is_gradient_checkpointing", False) is not False):
            raise ValueError("development model configuration is incompatible")
        if torch.is_autocast_enabled("cpu"):
            raise ValueError("CPU autocast is incompatible with the float32 protocol")
        global_hooks = torch.nn.modules.module
        if any(getattr(global_hooks, name, {}) for name in (
                "_global_forward_hooks", "_global_forward_pre_hooks",
                "_global_backward_hooks", "_global_backward_pre_hooks")):
            raise ValueError("global module hooks are incompatible with the development custodian")
        modules = list(model.named_modules(remove_duplicate=False))
        if len({id(module) for _, module in modules}) != len(modules):
            raise ValueError("aliased modules are not supported")
        for _, module in modules:
            if (module.training or module._forward_hooks or module._forward_pre_hooks or
                    module._backward_hooks or module._backward_pre_hooks):
                raise ValueError("development model must be in eval mode with no existing hooks")
        targets = [model.get_submodule(f"transformer.h.{layer}.mlp") for layer in range(6)]
        target_block = model.get_submodule("transformer.h.5")
        if len({id(module) for module in targets + [target_block]}) != 7:
            raise ValueError("selected modules must be distinct")
        tensors = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
        binding = []
        for name, tensor in tensors:
            if (tensor.device.type != "cpu" or tensor.is_complex() or
                    (tensor.is_floating_point() and tensor.dtype != torch.float32)):
                raise ValueError("development model tensors must be CPU float32 or nonfloating")
            binding.append((name, id(tensor), tensor._version, tuple(tensor.shape), str(tensor.dtype)))
        return (id(model), id(config), tuple((name, id(module)) for name, module in modules), tuple(binding))

    def _check_model(self):
        if self._model_binding() != self._binding:
            raise RuntimeError("development model changed during the attempt")

    @staticmethod
    def _step(name, operation, path, token, **extra):
        return {"name": name, "operation": operation, "module_path": path,
                "invocation": 0, "batch_index": 0, "token_index": token, **extra}

    @staticmethod
    def _cancel(cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise KeyboardInterrupt()

    @staticmethod
    def _same(left, right):
        left = left.to(dtype=torch.float64)
        right = right.to(dtype=torch.float64)
        return bool(torch.all(torch.abs(left - right) <=
                              study.ABS_TOL + study.REL_TOL * torch.maximum(torch.abs(left), torch.abs(right))))

    @staticmethod
    def _tensor_bytes(value):
        return value.detach().numpy().astype("<f4", copy=False).tobytes()

    @classmethod
    def _private_capture_bytes(cls, captures, name):
        value = captures[name]
        if (type(value) is not torch.Tensor or value.device.type != "cpu" or
                value.dtype != torch.float32 or tuple(value.shape) != (study.WIDTH,)):
            raise ValueError("invalid private capture")
        return cls._tensor_bytes(value)

    def _forward(self, condition, ids, clean, injected, *, index, case_run, label, cancel_event):
        self._current = {"case_index": index, "case_run": case_run, "condition": condition, "forward_label": label,
                         "captures_available": False}
        self._cancel(cancel_event)
        self._check_model()
        token = ids.shape[1] - 1
        is_reverse = condition.startswith("reverse_")
        corrupt = condition not in ("clean", "clean_repeat") and not is_reverse
        patch = condition.startswith("patch_") or is_reverse or condition in ("upstream", "late", "self_patch")
        post_action = condition not in ("clean", "clean_repeat", "injected")
        extra_bytes = (int(corrupt) + int(patch) + int(post_action)) * study.WIDTH * 4
        private_bytes = 3 * study.WIDTH * 4 if corrupt else 0
        charge = (7 * study.WIDTH + study.VOCAB_SIZE) * 4 + private_bytes + extra_bytes
        if charge > study.FORWARD_BYTES or self._charged + charge > self._max_study_bytes:
            raise ValueError("development capture payload limit exceeded")
        # Reserve before model execution; do not refund failed attempts.
        self._charged += charge
        if condition.startswith("patch_") or is_reverse:
            candidate = int(condition.rsplit("_", 1)[1])
            if candidate not in protocol.ELIGIBLE_SITES:
                raise ValueError("invalid candidate")
        elif condition == "upstream":
            candidate = 0
        else:
            candidate = None
        source = injected if is_reverse else clean
        steps = []
        for layer in range(6):
            path = f"transformer.h.{layer}.mlp"
            if corrupt and layer == self._site:
                steps.extend([self._step("private_before", "capture", path, token),
                              self._step("private_inject", "add", path, token, value=study.injection_delta()),
                              self._step("private_after", "capture", path, token)])
            steps.append(self._step(f"observe_{layer}", "capture", path, token))
            if condition == "null" and layer == 1:
                steps.append(self._step("control_null", "null", path, token))
                steps.append(self._step("private_control_after", "capture", path, token))
            elif condition == "self_patch" and layer == 1:
                steps.append(self._step("self_source", "capture", path, token))
                steps.append(self._step("control_self", "patch", path, token, source_capture="self_source"))
                steps.append(self._step("private_control_after", "capture", path, token))
            elif candidate == layer:
                steps.append(self._step("candidate_patch", "patch", path, token, value=source.events[layer].value))
                steps.append(self._step("private_control_after", "capture", path, token))
        path = "transformer.h.5"
        steps.append(self._step("observe_block", "capture", path, token))
        if condition == "late":
            steps.append(self._step("late_patch", "patch", path, token, value=clean.events[6].value))
            steps.append(self._step("private_control_after", "capture", path, token))
        plan = ActivationPlan(self._model, steps, max_tensor_bytes=study.FORWARD_BYTES,
                              max_capture_bytes=study.FORWARD_BYTES, max_source_bytes=study.FORWARD_BYTES,
                              max_steps=32, max_invocations=1)
        self._active_plan = plan
        expected_ids = ids.detach().clone()
        attention_mask = torch.ones_like(ids)
        with plan:
            with torch.no_grad():
                output = self._model(input_ids=ids, attention_mask=attention_mask,
                                     use_cache=False, return_dict=True)
                if not torch.equal(ids, expected_ids) or not bool(torch.all(attention_mask == 1)):
                    raise ValueError("development model modified its input")
                logits = getattr(output, "logits", None)
                if (type(logits) is not torch.Tensor or logits.device.type != "cpu" or
                        logits.dtype != torch.float32 or tuple(logits.shape) != (1, ids.shape[1], study.VOCAB_SIZE) or
                        getattr(output, "past_key_values", None) is not None):
                    raise ValueError("development output is incompatible")
                saved_logits = logits[0, token].detach().clone()
        metadata, captures = plan.metadata, plan.captures
        self._current["captures_available"] = True
        evidence = {"site": self._site, "case_index": index, "case_run": case_run,
                    "condition": condition, "forward_label": label, "token_index": token,
                    "injection_integrity_verified": False}
        # Record bounded snapshots before checking metadata or the model again,
        # so a later validation failure can still be diagnosed privately.
        self._private.append(evidence)
        if corrupt:
            for name, field in (("private_before", "pre"), ("private_after", "post")):
                raw = self._private_capture_bytes(captures, name)
                evidence[f"{field}_f32le"] = raw
                evidence[f"{field}_sha256"] = hashlib.sha256(raw).hexdigest()
        if post_action:
            raw_after = self._private_capture_bytes(captures, "private_control_after")
            evidence.update({"control_after_f32le": raw_after,
                             "control_after_sha256": hashlib.sha256(raw_after).hexdigest(),
                             "control_integrity_verified": False})
        self._check_model()
        if len(metadata["steps"]) != len(steps):
            raise RuntimeError("plan step coverage mismatch")
        preceding_by_site = {}
        for order, (expected, actual) in enumerate(zip(steps, metadata["steps"])):
            for key in ("name", "operation", "module_path", "invocation", "batch_index", "token_index"):
                if actual.get(key) != expected[key] or type(actual.get(key)) is not type(expected[key]):
                    raise RuntimeError("plan metadata does not match the prescribed operations")
            preceding = preceding_by_site.setdefault(expected["module_path"], [])
            if (type(actual.get("order")) is not int or actual["order"] != order or
                    type(actual.get("forward_index")) is not int or actual["forward_index"] != 0 or
                    type(actual.get("preceding_same_site_actions")) is not list or
                    any(type(item) is not str for item in actual["preceding_same_site_actions"]) or
                    actual["preceding_same_site_actions"] != preceding):
                raise RuntimeError("plan order or forward metadata mismatch")
            preceding.append(expected["name"])
            expected_selector = [0] if expected["module_path"] == "transformer.h.5" else []
            selector = actual.get("output_selector")
            output_shape = actual.get("output_shape")
            selected_shape = actual.get("selected_shape")
            if (actual.get("source_capture") != expected.get("source_capture") or
                    type(selector) is not list or any(type(item) is not int for item in selector) or
                    selector != expected_selector or
                    type(output_shape) is not list or any(type(item) is not int for item in output_shape) or
                    output_shape != [1, ids.shape[1], study.WIDTH] or
                    type(selected_shape) is not list or any(type(item) is not int for item in selected_shape) or
                    selected_shape != [study.WIDTH] or
                    actual.get("dtype") != "torch.float32" or actual.get("device") != "cpu"):
                raise RuntimeError("plan output or patch source metadata mismatch")
        capture_bytes = (7 + 2 * int(corrupt) + int(condition == "self_patch") + int(post_action)) * study.WIDTH * 4
        source_bytes = (int(corrupt) + int(patch and condition != "self_patch")) * study.WIDTH * 4
        if (type(metadata["capture_bytes"]) is not int or type(metadata["source_bytes"]) is not int or
                metadata["capture_bytes"] != capture_bytes or metadata["source_bytes"] != source_bytes):
            raise RuntimeError("plan byte accounting is inconsistent")
        source_description = None
        if post_action:
            if condition in ("null", "self_patch"):
                observed = captures["observe_1"]
                reference = captures["self_source"] if condition == "self_patch" else observed
                if not self._same(observed, injected.events[1].value):
                    raise ValueError("control pre-action reference mismatch")
                source_kind, source_forward, source_path, source_selector = "same-run", label, "transformer.h.1.mlp", []
            else:
                target_index = 6 if condition == "late" else candidate
                reference = source.events[target_index].value
                source_kind = "injected" if is_reverse else "clean"
                source_forward = source_kind
                source_path = source.events[target_index].module_path
                source_selector = list(source.events[target_index].output_selector)
            source_description = {"kind": source_kind, "case_index": index, "case_run": case_run,
                                  "forward_label": source_forward, "module_path": source_path,
                                  "batch_index": 0, "token_index": token, "invocation": 0,
                                  "output_selector": source_selector,
                                  "tensor_sha256": hashlib.sha256(self._tensor_bytes(reference)).hexdigest()}
        private_metadata = {"mode": "development-only", "synthetic_site": self._site,
                            "case_index": index, "case_run": case_run, "forward_label": label,
                            "condition": condition, "base": "injected" if corrupt else "clean",
                            "input_sha256": hashlib.sha256(ids.numpy().astype("<i8", copy=False).tobytes()).hexdigest(),
                            "input_shape": list(ids.shape), "source": source_description, "plan": metadata}
        metadata_bytes = json.dumps(private_metadata, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(metadata_bytes) > 65536 or self._private_metadata_bytes + len(metadata_bytes) > 16 * 1024 * 1024:
            raise ValueError("private plan metadata budget exceeded")
        self._private_metadata_bytes += len(metadata_bytes)
        evidence.update({"plan_metadata_json": metadata_bytes,
                         "plan_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest()})
        if post_action:
            after = captures["private_control_after"]
            # Null and replacement operations copy values; require exact bits,
            # including signed zero, at the site immediately after the action.
            if condition == "self_patch" and self._tensor_bytes(reference) != self._tensor_bytes(observed):
                raise ValueError("self-patch source differs from the same-site observation")
            if self._tensor_bytes(after) != self._tensor_bytes(reference):
                raise ValueError("post-action tensor does not match its prescribed source")
            evidence["control_integrity_verified"] = True
        selected = [entry for entry in metadata["steps"] if entry["name"].startswith("observe_")]
        if len(selected) != 7:
            raise RuntimeError("sanitized trace coverage mismatch")
        events = []
        for sequence, entry in enumerate(selected):
            layer = sequence if sequence < 6 else 5
            events.append(study.TraceEvent(layer=layer, module_path=entry["module_path"],
                                           invocation=entry["invocation"], batch_index=entry["batch_index"],
                                           token_index=entry["token_index"],
                                           output_selector=tuple(entry["output_selector"]),
                                           source_shape=tuple(entry["output_shape"]), source_dtype=entry["dtype"],
                                           source_device=entry["device"], forward_index=entry["forward_index"], sequence=sequence,
                                           value=captures[entry["name"]].detach().clone()))
        if corrupt:
            pre, post = captures["private_before"], captures["private_after"]
            integrity = study.verify_injection_integrity(pre, post)
            raw_delta = self._tensor_bytes(integrity["realized_delta"])
            evidence.update({"realized_delta_f32le": raw_delta,
                             "realized_delta_sha256": hashlib.sha256(raw_delta).hexdigest(),
                             "realized_norm": integrity["realized_norm"], "injection_integrity_verified": True})
        observation = study.Observation(tuple(events), saved_logits, corrupt, private_bytes, extra_bytes)
        study.validate_observation(observation, token_index=token, injected=corrupt, source_device="cpu")
        self._active_plan = None
        self._cancel(cancel_event)
        return observation

    def run_case(self, *, index, input_ids, cancel_event=None):
        """Run one development case with all prescribed controls and fresh clean runs."""
        if not self._lock.acquire(blocking=False):
            raise CustodianExecutionError("development custodian is busy")
        if self.poisoned:
            self._lock.release()
            raise CustodianExecutionError("development custodian is quarantined")
        failed, cancelled = False, False
        try:
            self._case_runs += 1
            case_run = self._case_runs
            safe_index = index if type(index) is int and 0 <= index < study.N_CASES else None
            self._current = {"case_index": safe_index, "case_run": case_run, "condition": "preparing",
                             "forward_label": "preparing", "captures_available": False}
            if cancel_event is not None and type(cancel_event) is not Event:
                raise ValueError("cancel_event must be a threading.Event")
            self._cancel(cancel_event)
            if (type(index) is not int or not 0 <= index < study.N_CASES or
                    type(input_ids) is not torch.Tensor or input_ids.device.type != "cpu" or
                    input_ids.dtype != torch.long or input_ids.ndim != 2 or input_ids.shape[0] != 1 or
                    not 1 <= input_ids.shape[1] <= 32 or bool(torch.any(input_ids < 0)) or
                    bool(torch.any(input_ids >= study.VOCAB_SIZE))):
                raise ValueError("invalid development input")
            ids = input_ids.detach().clone()
            def forward(condition, label, clean=None, injected=None):
                return self._forward(condition, ids, clean, injected, index=index, case_run=case_run,
                                     label=label, cancel_event=cancel_event)
            clean = forward("clean", "clean")
            conditions = {"clean": clean, "clean_repeat": forward("clean_repeat", "clean_repeat")}
            post_clean = {}
            corrupt = forward("injected", "injected", clean)
            conditions["injected"] = corrupt
            post_clean["injected"] = forward("clean", "post_clean:injected")
            names = ["null", "self_patch", "upstream", "late"] + [
                f"patch_{site}" for site in protocol.ELIGIBLE_SITES] + [
                f"reverse_{site}" for site in protocol.ELIGIBLE_SITES]
            for name in names:
                conditions[name] = forward(name, name, clean, corrupt)
                post_clean[name] = forward("clean", f"post_clean:{name}")
            case = study.DiscoveryCase(index, conditions, post_clean)
            study.analyze_discovery_case(case, token_index=ids.shape[1] - 1, source_device="cpu")
            self._cancel(cancel_event)
            return case
        except BaseException as error:
            # Latch before any cleanup or potentially allocating diagnostic work.
            self._poisoned.set()
            cancelled, failed = isinstance(error, KeyboardInterrupt), True
            try:
                diagnostic = dict(self._current or {})
                diagnostic.update({"error_type": type(error).__name__[:128],
                                   "error_message": str(error)[:2048],
                                   "cancelled": cancelled})
                self._failures.append(diagnostic)
            except BaseException as diagnostic_error:
                cancelled = cancelled or isinstance(diagnostic_error, KeyboardInterrupt)
        finally:
            self._lock.release()
        if cancelled:
            raise CustodianCancelled("development custodian was interrupted and quarantined")
        if failed:
            raise CustodianExecutionError("development custodian failed and is quarantined")
