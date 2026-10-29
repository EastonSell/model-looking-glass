"""CPU-only primitives for the frozen known-injection number-study protocol.

These arithmetic and instrumentation primitives do not allocate a split,
commit a site, or authorize execution of an experiment by themselves.
"""
import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import asdict, dataclass
from threading import Lock, current_thread

import torch
from torch import nn

from .agreement_numeric import decode_f32_le
from .intervention_plan import ActivationPlan
from .model_ownership import ModelLease


_SEED = 20261028
_WIDTH = 768
_POSITIVE_DIRECTION_BITS = 0x3C93CD3A
_NEGATIVE_DIRECTION_BITS = 0xBC93CD3A


def _finite_components(bits):
    if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError("float32 bits must be a uint32 integer")
    exponent, fraction = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("float32 operands must be finite")
    mantissa = fraction if exponent == 0 else (1 << 23) | fraction
    power = -149 if exponent == 0 else exponent - 150
    return (-mantissa if bits & 0x80000000 else mantissa), power


def add_float32_bits(left, right):
    """Return correctly rounded IEEE binary32 sum as bits using integer math.

    Operands must be finite and rounding is nearest, ties-to-even. Overflow is
    represented as the appropriate infinity; a study rejects nonfinite output.
    Signed zeros follow that rounding mode. No host float32 conversion occurs.
    """
    a, a_power = _finite_components(left)
    b, b_power = _finite_components(right)
    power = min(a_power, b_power)
    total = (a << (a_power - power)) + (b << (b_power - power))
    if total == 0:
        return 0x80000000 if left == right == 0x80000000 else 0
    sign = 0x80000000 if total < 0 else 0
    magnitude = abs(total)
    exponent = magnitude.bit_length() - 1 + power
    if exponent < -126:
        # Every finite binary32 number is an exact multiple of 2**-149, so a
        # subnormal sum is also exact in that unit.
        return sign | (magnitude << (power + 149))
    shift = magnitude.bit_length() - 24
    if shift > 0:
        retained, discarded = divmod(magnitude, 1 << shift)
        halfway = 1 << (shift - 1)
        if discarded > halfway or (discarded == halfway and retained & 1):
            retained += 1
    else:
        retained = magnitude << -shift
    if retained == 1 << 24:
        retained >>= 1
        exponent += 1
    if exponent > 127:
        return sign | 0x7F800000
    return sign | ((exponent + 127) << 23) | (retained & 0x7FFFFF)


def injection_direction_bits():
    """Return the protocol's 768 exact signed float32 component bit patterns."""
    signs = []
    for chunk in range((_WIDTH + 255) // 256):
        digest = hashlib.sha256(b"MLG-v1-direction\0" + _SEED.to_bytes(8, "big") +
                                chunk.to_bytes(4, "big")).digest()
        for byte in digest:
            signs.extend(_POSITIVE_DIRECTION_BITS if byte & (1 << bit) else _NEGATIVE_DIRECTION_BITS
                         for bit in range(7, -1, -1))
    return tuple(signs[:_WIDTH])


def verify_cpu_float32_addition():
    """Reject a CPU whose current float32 addition does not match the protocol.

    The sentinels include subnormal input/output and nearest-even halfway cases;
    passing these cases is a runtime check, not a proof about every CPU operation.
    """
    cases = ((0x00000001, 0x00000000), (0x00000001, 0x00000001),
             (0x00800000, 0x807FFFFF), (0x3F800000, 0x33800000),
             (0x3F800001, 0x33800000), (0x80000000, 0x80000000))
    signed = lambda value: value if value < 1 << 31 else value - (1 << 32)
    left = torch.tensor([signed(a) for a, _ in cases], dtype=torch.int32, device="cpu").view(torch.float32)
    right = torch.tensor([signed(b) for _, b in cases], dtype=torch.int32, device="cpu").view(torch.float32)
    actual = [value & 0xFFFFFFFF for value in torch.add(left, right).view(torch.int32).tolist()]
    expected = [add_float32_bits(a, b) for a, b in cases]
    if actual != expected:
        raise RuntimeError("CPU float32 addition does not match required rounding or denormal behavior")


def verify_injection_bits(pre_bytes, post_bytes):
    """Verify exact CPU post-add bits against the fixed protocol direction.

    Inputs are immutable little-endian snapshots of the complete private
    selected vector. The caller must retain the snapshots and verify the
    action's site, invocation, and cleanup separately.
    """
    if (type(pre_bytes) is not bytes or type(post_bytes) is not bytes or
            len(pre_bytes) != 4 * _WIDTH or len(post_bytes) != 4 * _WIDTH):
        raise ValueError("injection snapshots must contain 768 float32 bit patterns")
    before = struct.unpack("<768I", pre_bytes)
    after = struct.unpack("<768I", post_bytes)
    changed = 0
    for left, actual, delta in zip(before, after, injection_direction_bits()):
        _finite_components(actual)
        expected = add_float32_bits(left, delta)
        if ((expected >> 23) & 0xFF) == 0xFF or actual != expected:
            raise ValueError("injection did not produce the exact finite float32 sum")
        changed += actual != left
    if not changed:
        raise ValueError("injection did not change any selected element")
    return {"changed_coordinates": changed}


def _tensor_from_bits(raw):
    if type(raw) is not bytes or len(raw) != 4 * _WIDTH:
        raise ValueError("donor must be one immutable 768-element float32 vector")
    values = struct.unpack("<768I", raw)
    for value in values:
        _finite_components(value)
    signed = [value if value < 1 << 31 else value - (1 << 32) for value in values]
    return torch.tensor(signed, dtype=torch.int32, device="cpu").view(torch.float32)


def verify_action_scope(pre_bytes, post_bytes, *, token, operation, donor=None):
    """Check full-output witnesses for one selected-token action on CPU.

    The caller must still bind these bytes to the planned module, invocation,
    tensor selector and actual forward. Whole-output witnesses remain private
    until the hidden site is revealed.
    """
    width_bytes = 4 * _WIDTH
    if (type(pre_bytes) is not bytes or type(post_bytes) is not bytes or
            len(pre_bytes) != len(post_bytes) or not len(pre_bytes) or
            len(pre_bytes) % width_bytes or len(pre_bytes) > 32 * width_bytes or
            type(token) is not int or not 0 <= token < len(pre_bytes) // width_bytes):
        raise ValueError("invalid full-output witnesses or token index")
    for (bits,) in struct.iter_unpack("<I", pre_bytes + post_bytes):
        _finite_components(bits)
    start, end = token * width_bytes, (token + 1) * width_bytes
    if pre_bytes[:start] != post_bytes[:start] or pre_bytes[end:] != post_bytes[end:]:
        raise ValueError("action changed an unselected token")
    before, after = pre_bytes[start:end], post_bytes[start:end]
    if operation == "add" and donor is None:
        return verify_injection_bits(before, after)
    if operation == "patch" and type(donor) is bytes and len(donor) == width_bytes:
        if after != donor:
            raise ValueError("patch did not land the exact donor bits")
        return {}
    if operation == "null" and donor is None and before == after:
        return {}
    raise ValueError("invalid action or action result")


def _plan_steps(*, site, token, condition, layer=None, donor=None):
    """Build private ordered steps for one protocol forward, without running it."""
    conditions = {"clean", "injected", "clean_to_injected", "injected_to_clean",
                  "upstream", "downstream", "null", "self"}
    if (type(site) is not int or site not in range(1, 5) or
            type(token) is not int or not 0 <= token < 32 or condition not in conditions):
        raise ValueError("invalid private site, token, or condition")
    if condition in {"clean_to_injected", "injected_to_clean"}:
        if type(layer) is not int or layer not in range(1, 5):
            raise ValueError("candidate layer must be between one and four")
    elif layer is not None:
        raise ValueError("candidate layer is only valid for a candidate patch")
    donor_conditions = {"clean_to_injected", "injected_to_clean", "upstream", "downstream"}
    source = _tensor_from_bits(donor) if condition in donor_conditions else None
    if condition not in donor_conditions and donor is not None:
        raise ValueError("unexpected donor for condition")
    inject = condition in {"injected", "clean_to_injected", "upstream", "downstream", "null", "self"}
    delta = _tensor_from_bits(struct.pack("<768I", *injection_direction_bits())) if inject else None

    def selected(name, operation, path, **extra):
        return {"name": name, "operation": operation, "module_path": path, "invocation": 0,
                "batch_index": 0, "token_index": token, **extra}

    def whole(name, path):
        return {"name": name, "operation": "capture", "module_path": path, "invocation": 0}

    steps = []
    for current in range(6):
        attention = f"transformer.h.{current}.attn"
        mlp = f"transformer.h.{current}.mlp"
        steps.append(selected(f"attention_{current}", "capture", attention))
        if inject and current == site:
            steps.extend((whole("inject_before", mlp), selected("inject_add", "add", mlp, value=delta),
                          whole("inject_after", mlp)))
        patch_here = ((condition in {"clean_to_injected", "injected_to_clean"} and current == layer)
                      or (condition == "upstream" and current == 0))
        control_here = condition in {"null", "self"} and current == 1
        if control_here and condition == "self":
            steps.append(selected("self_source", "capture", mlp))
        if patch_here or control_here:
            steps.append(whole("action_before", mlp))
            if patch_here:
                steps.append(selected("action_patch", "patch", mlp, value=source))
            elif condition == "self":
                steps.append(selected("action_patch", "patch", mlp, source_capture="self_source"))
            else:
                steps.append(selected("action_null", "null", mlp))
            steps.append(whole("action_after", mlp))
        steps.append(selected(f"mlp_{current}", "capture", mlp))
        if current == 5:
            block = "transformer.h.5"
            if condition == "downstream":
                steps.extend((whole("action_before", block),
                              selected("action_patch", "patch", block, value=source),
                              whole("action_after", block)))
            steps.append(selected("block_5", "capture", block))
    return steps


@dataclass(frozen=True, repr=False)
class InjectionObservation:
    """Selector-facing captures and the requested, site-independent action."""
    request_id: str
    input_sha256: str
    condition: str
    attention: tuple
    mlp: tuple
    block: bytes
    logits: bytes
    trace: tuple
    action: "PublicAction | None"
    _owner_token: object

    def mlp_donor(self, layer):
        if type(layer) is not int or not 0 <= layer < 6:
            raise ValueError("invalid donor layer")
        return self._donor(self.trace[2 * layer + 1], self.mlp[layer])

    def block_donor(self):
        return self._donor(self.trace[-1], self.block)

    def _donor(self, boundary, data):
        identity = DonorIdentity(self.request_id, self.input_sha256, boundary.module_path,
                                 boundary.invocation, boundary.batch_index, boundary.token_index,
                                 boundary.output_selector, boundary.selected_shape, boundary.dtype, boundary.device)
        return DonorSnapshot(identity, data, self._owner_token)


@dataclass(frozen=True)
class DonorIdentity:
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
class PublicAction:
    """Requested patch/null coordinates, with no injection or plan ordinals."""
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
    donor_identity: DonorIdentity | None


@dataclass(frozen=True, repr=False)
class DonorSnapshot:
    identity: DonorIdentity
    data: bytes
    _owner_token: object


@dataclass(frozen=True)
class PublicTraceBoundary:
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


@dataclass(frozen=True, repr=False)
class PrivateActionEvidence:
    """Custodian-only full-output witnesses and action metadata."""
    request_id: str
    input_sha256: str
    condition: str
    operation: str
    module_path: str
    token_index: int
    before: bytes
    after: bytes
    donor: bytes | None
    donor_identity: DonorIdentity | None
    realized_delta_f64_le: bytes | None
    metadata_json: str


@dataclass(frozen=True, repr=False)
class PrivateFailureEvidence:
    request_id: str | None
    input_sha256: str
    condition: str
    exception_type: str
    exception_message: str
    captures: tuple
    logits: bytes | None
    metadata_json: str | None


def _raw_tensor(tensor, shape):
    if (sys.byteorder != "little" or type(tensor) is not torch.Tensor or tensor.device.type != "cpu" or
            tensor.dtype is not torch.float32 or tuple(tensor.shape) != tuple(shape)):
        raise ValueError("captured tensor shape, dtype, or device is unexpected")
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _finite_raw(raw):
    for (bits,) in struct.iter_unpack("<I", raw):
        _finite_components(bits)


def _check_model_for_cpu(model):
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a CPU module")
    config = getattr(model, "config", None)
    if (config is None or type(getattr(config, "n_layer", None)) is not int or config.n_layer != 6 or
            type(getattr(config, "n_embd", None)) is not int or config.n_embd != _WIDTH or
            type(getattr(config, "vocab_size", None)) is not int or config.vocab_size != 50257 or
            getattr(config, "use_cache", None) is not False or
            getattr(config, "_attn_implementation", None) != "eager" or
            getattr(model, "is_gradient_checkpointing", False) is not False or torch.is_autocast_enabled("cpu") or
            torch.get_num_threads() != 1):
        raise ValueError("model execution settings do not match the CPU protocol")
    hooks = torch.nn.modules.module
    if any(getattr(hooks, name, {}) for name in ("_global_forward_hooks", "_global_forward_pre_hooks",
                                                  "_global_backward_hooks", "_global_backward_pre_hooks")):
        raise ValueError("unexpected global hooks")
    modules = tuple(model.named_modules(remove_duplicate=False))
    if len({id(module) for _, module in modules}) != len(modules):
        raise ValueError("aliased module paths are unsupported")
    for _, module in modules:
        if (module.training or module._forward_hooks or module._forward_pre_hooks or
                module._backward_hooks or module._backward_pre_hooks):
            raise ValueError("model must be in evaluation mode without hooks")
    for tensor in tuple(model.parameters()) + tuple(model.buffers()):
        if (tensor.device.type != "cpu" or tensor.is_complex() or
                (tensor.is_floating_point() and tensor.dtype is not torch.float32)):
            raise ValueError("model tensors must be CPU float32 or nonfloating")


def _typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (len(actual) == len(expected) and all(type(key) is str for key in actual) and
                all(key in actual and _typed_equal(actual[key], value) for key, value in expected.items()))
    if type(expected) in (list, tuple):
        return len(actual) == len(expected) and all(_typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _check_plan_metadata(steps, events, sequence_length):
    if type(events) is not list or len(steps) != len(events):
        raise ValueError("plan did not execute every requested step")
    previous_site, preceding = None, []
    for index, (step, event) in enumerate(zip(steps, events)):
        path = step["module_path"]
        site = (path, step["invocation"])
        if site != previous_site:
            preceding = []
        selected = "token_index" in step
        expected_selector = [0] if path.endswith(".attn") or path == "transformer.h.5" else []
        expected = {"name": step["name"], "operation": step["operation"], "order": index,
                    "module_path": path, "module_aliases": [path], "call_site_known": False,
                    "forward_index": 0, "invocation": 0, "batch_index": 0 if selected else None,
                    "token_index": step["token_index"] if selected else None, "output_selector": expected_selector,
                    "output_shape": [1, sequence_length, _WIDTH],
                    "selected_shape": [_WIDTH] if selected else [1, sequence_length, _WIDTH],
                    "dtype": "torch.float32", "device": "cpu", "source_capture": step.get("source_capture"),
                    "preceding_same_site_actions": preceding}
        if not _typed_equal(event, expected):
            raise ValueError("plan metadata did not match the declared ordered steps")
        preceding = preceding + [step["name"]]
        previous_site = site


def _check_witness_chain(raw, events, token):
    """Cross-check adjacent private actions and public post-action captures."""
    start, end = token * _WIDTH * 4, (token + 1) * _WIDTH * 4
    inject = events.get("inject_add")
    action = events.get("action_patch") or events.get("action_null")
    if inject is not None:
        path = inject["module_path"]
        if action is not None and action["module_path"] == path:
            if raw["inject_after"] != raw["action_before"]:
                raise ValueError("adjacent same-site witnesses differ")
        else:
            layer = int(path.split(".")[2])
            if raw["mlp_" + str(layer)] != raw["inject_after"][start:end]:
                raise ValueError("post-injection public capture differs from witness")
    if action is not None:
        path = action["module_path"]
        name = "block_5" if path == "transformer.h.5" else "mlp_" + path.split(".")[2]
        if raw[name] != raw["action_after"][start:end]:
            raise ValueError("post-action public capture differs from witness")
        if "self_source" in raw and raw["self_source"] != raw["action_before"][start:end]:
            raise ValueError("self-patch source differs from its immediate pre-action value")


class NumberInjectionExecutor:
    """Private, bounded CPU forward primitive; it is not a study custodian.

    The owner must verify the checkpoint and protocol, publish commitments and
    frozen predictions in order, durably save evidence, and enforce split
    custody separately. This helper does not provide process isolation.
    """
    def __init__(self, model, *, site, max_study_bytes=512 * 1024 * 1024):
        if type(site) is not int or site not in range(1, 5):
            raise ValueError("private site must be an eligible layer")
        if type(max_study_bytes) is not int or not 0 < max_study_bytes <= 512 * 1024 * 1024:
            raise ValueError("invalid study tensor-payload budget")
        self._model, self._site = model, site
        self._max_study_bytes = max_study_bytes
        self._lease = ModelLease(model)
        self._owner = None
        self._lifecycle_lock = Lock()
        self._entry_token = None
        self._used = False
        self._entering = False
        self._entered = False
        self._closing = False
        self._in_forward = False
        self._poisoned = False
        self._completed = []
        self._private = []
        self._private_failures = []
        self._failures = []
        self._retained = {}
        self._retained_bytes = 0
        self._owner_token = object()
        self._seen = set()
        self._completed_by_id = {}

    @property
    def poisoned(self):
        return self._poisoned

    @property
    def retained_tensor_bytes(self):
        return self._retained_bytes

    def completed(self):
        return tuple(self._completed)

    def private_evidence(self):
        """Return private witnesses; do not disclose them to the selector."""
        return tuple(self._private)

    def failures(self):
        return tuple(dict(item) for item in self._failures)

    def private_failures(self):
        """Return available failed-forward snapshots; keep them private."""
        return tuple(self._private_failures)

    def __enter__(self):
        token = object()
        with self._lifecycle_lock:
            if self._used or self._entering or self._entered or self._closing:
                raise RuntimeError("executor is single-use")
            self._used = True
            self._entry_token = token
            self._entering = True
            self._owner = current_thread()
        try:
            verify_cpu_float32_addition()
            _check_model_for_cpu(self._model)
            self._lease.acquire()
            _check_model_for_cpu(self._model)
            with self._lifecycle_lock:
                if self._entry_token is not token or not self._entering:
                    raise RuntimeError("executor entry ownership changed")
                self._entering, self._entered = False, True
            return self
        except BaseException as error:
            with self._lifecycle_lock:
                owns_entry = self._entry_token is token
                if owns_entry:
                    self._entering, self._entered = False, False
            if not owns_entry:
                raise
            try:
                self._lease.release()
            except BaseException as cleanup_error:
                raise error.with_traceback(error.__traceback__) from cleanup_error
            raise

    def __exit__(self, exc_type, exc, tb):
        with self._lifecycle_lock:
            if not self._entered or current_thread() is not self._owner or self._closing:
                raise RuntimeError("executor must exit on its owner thread")
            if self._in_forward:
                self._poisoned = True
                raise RuntimeError("cannot exit while a forward is active")
            self._entered = False
            self._closing = True
            if exc_type is not None:
                self._poisoned = True
        try:
            if self._poisoned:
                # A failed plan or uncertain cleanup must not return the model
                # to a cooperating owner. Discard the dedicated process.
                self._lease._quarantine()
            else:
                _check_model_for_cpu(self._model)
                self._lease.release()
        except BaseException as cleanup_error:
            try:
                self._lease._quarantine()
            except BaseException as quarantine_error:
                if quarantine_error is not cleanup_error:
                    cleanup_error.__cause__ = quarantine_error
            if exc is not None:
                raise exc.with_traceback(tb) from cleanup_error
            raise
        finally:
            with self._lifecycle_lock:
                self._closing = False
        return False

    def _keep(self, raw):
        if id(raw) not in self._retained:
            self._retained[id(raw)] = raw
            self._retained_bytes += len(raw)

    def _donor_data(self, donor, condition, layer, input_sha256):
        donor_conditions = {"clean_to_injected", "injected_to_clean", "upstream", "downstream"}
        if condition not in donor_conditions:
            if donor is not None:
                raise ValueError("unexpected donor")
            return None
        if type(donor) is not DonorSnapshot or donor._owner_token is not self._owner_token:
            raise ValueError("donor is not a completed capture from this executor")
        if type(donor.identity) is not DonorIdentity:
            raise ValueError("donor identity is malformed")
        source = self._completed_by_id.get(donor.identity.request_id)
        if source is None or source.input_sha256 != input_sha256 or donor.identity.input_sha256 != input_sha256:
            raise ValueError("donor input identity differs from the target")
        expected_condition = "injected" if condition == "injected_to_clean" else "clean"
        if source.condition != expected_condition:
            raise ValueError("donor came from an unexpected source condition")
        if condition == "downstream":
            expected = source.block_donor()
        else:
            expected_layer = 0 if condition == "upstream" else layer
            if type(expected_layer) is not int or not 0 <= expected_layer < 6:
                raise ValueError("invalid donor layer")
            expected = source.mlp_donor(expected_layer)
        if (not _typed_equal(asdict(donor.identity), asdict(expected.identity)) or donor.data is not expected.data):
            raise ValueError("donor boundary or immutable capture identity differs")
        return donor.data

    def forward(self, input_ids, *, request_id, condition, layer=None, donor=None):
        with self._lifecycle_lock:
            if (not self._entered or current_thread() is not self._owner or self._poisoned or
                    self._closing or self._in_forward):
                raise RuntimeError("executor is unavailable or owned by another thread")
            self._in_forward = True
        try:
            return self._forward_impl(input_ids, request_id=request_id, condition=condition, layer=layer, donor=donor)
        finally:
            with self._lifecycle_lock:
                self._in_forward = False

    def _forward_impl(self, input_ids, *, request_id, condition, layer=None, donor=None):
        phase = "request"
        raw, logits, metadata = {}, None, None
        input_sha256 = ""
        safe_request = request_id if type(request_id) is str and re.fullmatch(r"[0-9a-f]{32}", request_id) else None
        conditions = {"clean", "injected", "clean_to_injected", "injected_to_clean",
                      "upstream", "downstream", "null", "self"}
        safe_condition = condition if type(condition) is str and condition in conditions else "invalid"
        try:
            if safe_request is None or request_id in self._seen:
                raise ValueError("request identity must be a fresh 32-character lowercase hex string")
            self._seen.add(request_id)
            if safe_condition == "invalid":
                raise ValueError("condition must be a supported string")
            if type(input_ids) not in (list, tuple):
                raise ValueError("input IDs must be a short nonempty sequence")
            frozen_ids = tuple(input_ids)
            if (not 1 <= len(frozen_ids) <= 32 or
                    any(type(value) is not int or not 0 <= value < 50257 for value in frozen_ids)):
                raise ValueError("input IDs must be a short nonempty sequence")
            token, length = len(frozen_ids) - 1, len(frozen_ids)
            input_sha256 = hashlib.sha256(b"MLG-number-input-v1\0" + bytes([length]) +
                                         b"".join(value.to_bytes(4, "big") for value in frozen_ids)).hexdigest()
            phase = "donor"
            donor_data = self._donor_data(donor, condition, layer, input_sha256)
            steps = _plan_steps(site=self._site, token=token, condition=condition, layer=layer, donor=donor_data)
            capture_bytes = sum((4 * _WIDTH if "token_index" in step else length * 4 * _WIDTH)
                                for step in steps if step["operation"] == "capture")
            output_bytes = 50257 * 4
            source_bytes = sum(4 * _WIDTH for step in steps if "value" in step)
            realized_bytes = 8 * _WIDTH if any(step["name"] == "inject_add" for step in steps) else 0
            phase = "budget"
            if capture_bytes + output_bytes + source_bytes + realized_bytes > 1024 * 1024:
                raise ValueError("per-forward retained tensor-payload cap exceeded")
            donor_bytes = len(donor_data) if donor_data is not None and id(donor_data) not in self._retained else 0
            additional = capture_bytes + output_bytes + donor_bytes + realized_bytes
            if self._retained_bytes + additional > self._max_study_bytes:
                raise ValueError("study retained tensor-payload cap exceeded")
            phase = "prepare"
            verify_cpu_float32_addition()
            _check_model_for_cpu(self._model)
            self._lease.verify_tree()
            ids = torch.tensor([frozen_ids], dtype=torch.int64, device="cpu")
            mask = torch.ones_like(ids)
            entered = 0
            def matching_inputs():
                return (type(ids) is torch.Tensor and ids.device.type == "cpu" and ids.dtype is torch.int64
                        and tuple(ids.shape) == (1, length) and ids.tolist() == [list(frozen_ids)]
                        and type(mask) is torch.Tensor and mask.device.type == "cpu" and mask.dtype is torch.int64
                        and tuple(mask.shape) == (1, length) and mask.tolist() == [[1] * length])

            def check_entry(module, args, kwargs):
                nonlocal entered
                if (entered != 0 or module is not self._model or type(args) is not tuple or args
                        or type(kwargs) is not dict
                        or set(kwargs) != {"input_ids", "attention_mask", "use_cache", "return_dict"}
                        or kwargs["input_ids"] is not ids or kwargs["attention_mask"] is not mask
                        or kwargs["use_cache"] is not False or kwargs["return_dict"] is not True
                        or not matching_inputs()):
                    raise RuntimeError("model entry inputs differ from the fixed request")
                entered += 1

            plan = ActivationPlan(self._model, steps, max_tensor_bytes=1024 * 1024,
                                  max_capture_bytes=1024 * 1024, max_source_bytes=1024 * 1024,
                                  _parent_lease=self._lease)
            phase = "forward"
            with plan:
                handle = None
                try:
                    handle = self._model.register_forward_pre_hook(check_entry, with_kwargs=True)
                    if not callable(getattr(handle, "remove", None)):
                        raise TypeError("model-entry hook registration returned an invalid handle")
                    with torch.inference_mode():
                        result = self._model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                except BaseException as error:
                    if callable(getattr(handle, "remove", None)):
                        try:
                            handle.remove()
                        except BaseException as cleanup_error:
                            raise error.with_traceback(error.__traceback__) from cleanup_error
                    raise
                else:
                    handle.remove()
            phase = "snapshot"
            metadata = plan.metadata
            captures = plan.captures
            for step in steps:
                if step["operation"] != "capture":
                    continue
                name = step["name"]
                raw[name] = _raw_tensor(captures[name],
                    ([_WIDTH] if "token_index" in step else [1, length, _WIDTH]))
                self._keep(raw[name])
                _finite_raw(raw[name])
            phase = "input_contract"
            if entered != 1 or not matching_inputs() or getattr(result, "past_key_values", None) is not None:
                raise ValueError("model input, mask, or cache contract changed during the forward")
            all_logits = result.logits
            if (type(all_logits) is not torch.Tensor or all_logits.device.type != "cpu" or
                    all_logits.dtype is not torch.float32 or tuple(all_logits.shape) != (1, length, 50257)):
                raise ValueError("logit shape, dtype, or device is unexpected")
            logits = _raw_tensor(all_logits[0, -1], [50257])
            self._keep(logits)
            _finite_raw(logits)
            if donor_data is not None:
                self._keep(donor_data)
            phase = "runtime_after"
            verify_cpu_float32_addition()
            phase = "metadata"
            with self._lifecycle_lock:
                if self._poisoned:
                    raise RuntimeError("executor was invalidated during the forward")
            if (type(metadata) is not dict or set(metadata) != {"steps", "capture_bytes", "source_bytes"} or
                    type(metadata["capture_bytes"]) is not int or metadata["capture_bytes"] != capture_bytes or
                    type(metadata["source_bytes"]) is not int or metadata["source_bytes"] != source_bytes):
                raise ValueError("plan byte totals do not match its declared steps")
            _check_plan_metadata(steps, metadata["steps"], length)
            _check_model_for_cpu(self._model)
            event_by_name = {event["name"]: event for event in metadata["steps"]}
            _check_witness_chain(raw, event_by_name, token)
            private = []
            public_action = None
            for prefix, operation in (("inject", "add"), ("action", None)):
                before_name, after_name = prefix + "_before", prefix + "_after"
                if before_name not in raw:
                    continue
                action_name = "inject_add" if prefix == "inject" else (
                    "action_null" if condition == "null" else "action_patch")
                actual_operation = operation or ("null" if condition == "null" else "patch")
                actual_donor = raw.get("self_source") if condition == "self" and prefix == "action" else (
                    donor_data if prefix == "action" and actual_operation == "patch" else None)
                phase = "landing"
                verify_action_scope(raw[before_name], raw[after_name], token=token,
                                    operation=actual_operation, donor=actual_donor)
                realized = None
                if actual_operation == "add":
                    start, end = token * _WIDTH * 4, (token + 1) * _WIDTH * 4
                    left = decode_f32_le(raw[before_name][start:end])
                    right = decode_f32_le(raw[after_name][start:end])
                    realized = struct.pack("<768d", *(math.fsum((float(a), -float(b)))
                                                     for a, b in zip(right, left)))
                    self._keep(realized)
                proof = [event_by_name[before_name], event_by_name[action_name], event_by_name[after_name]]
                donor_identity = None
                if condition == "self" and prefix == "action":
                    event = event_by_name["self_source"]
                    donor_identity = DonorIdentity(request_id, input_sha256, event["module_path"], event["invocation"],
                        event["batch_index"], event["token_index"], tuple(event["output_selector"]),
                        tuple(event["selected_shape"]), event["dtype"], event["device"])
                elif donor is not None and prefix == "action":
                    donor_identity = donor.identity
                if prefix == "action":
                    event = event_by_name[action_name]
                    public_action = PublicAction(actual_operation, event["module_path"], event["invocation"],
                        event["batch_index"], event["token_index"], tuple(event["output_selector"]),
                        tuple(event["output_shape"]), tuple(event["selected_shape"]), event["dtype"],
                        event["device"], donor_identity)
                private.append(PrivateActionEvidence(request_id, input_sha256, condition, actual_operation,
                               event_by_name[action_name]["module_path"], token, raw[before_name], raw[after_name],
                               actual_donor, donor_identity, realized,
                               json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False)))
            public_names = [name for i in range(6) for name in (f"attention_{i}", f"mlp_{i}")] + ["block_5"]
            trace = tuple(PublicTraceBoundary(event_by_name[name]["module_path"], index,
                          event_by_name[name]["invocation"], event_by_name[name]["batch_index"],
                          event_by_name[name]["token_index"], tuple(event_by_name[name]["output_selector"]),
                          tuple(event_by_name[name]["output_shape"]), tuple(event_by_name[name]["selected_shape"]),
                          event_by_name[name]["dtype"], event_by_name[name]["device"])
                          for index, name in enumerate(public_names))
            observation = InjectionObservation(request_id, input_sha256, condition,
                                               tuple(raw[f"attention_{i}"] for i in range(6)),
                                               tuple(raw[f"mlp_{i}"] for i in range(6)),
                                               raw["block_5"], logits, trace, public_action, self._owner_token)
            self._completed.append(observation)
            self._completed_by_id[request_id] = observation
            self._private.extend(private)
            return observation
        except BaseException as exc:
            self._poisoned = True
            partial = bool(raw) or logits is not None
            try:
                # All exception details, names and metadata are custodian-only;
                # successful plan captures remain available after later failures.
                try:
                    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
                except (TypeError, ValueError):
                    metadata_json = None
                try:
                    message = str(exc)[:1024]
                except BaseException:
                    message = "exception message unavailable"
                self._private_failures.append(PrivateFailureEvidence(safe_request, input_sha256, safe_condition,
                                                                     type(exc).__name__[:128], message,
                                                                     tuple(raw.items()), logits, metadata_json))
                self._failures.append({"request_id": safe_request, "condition": safe_condition, "phase": phase,
                                       "completed_forwards": len(self._completed),
                                       "partial_forward_captures": "retained_privately" if partial else "discarded"})
            except BaseException as record_error:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise exc.with_traceback(exc.__traceback__) from record_error
                try:
                    raise exc.with_traceback(exc.__traceback__) from record_error
                except BaseException as original:
                    raise RuntimeError("ordered injection forward failed; failure recording also failed") from original
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                if isinstance(exc, SystemExit):
                    raise
                raise KeyboardInterrupt("ordered injection forward interrupted; details withheld") from None
            raise RuntimeError("ordered injection forward failed; private detail withheld") from None
