"""Bounded development-only DistilBERT CUDA intervention calibration.

The observation primitive also accepts CPU tensors for synthetic tests. A
supervised owner is responsible for the device lease, memory commitment,
checkpoint provenance, cancellation, and durable failure evidence.
"""

from dataclasses import dataclass
import base64
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import traceback
import weakref

from .device_lease import DeviceLease
from .device_identity import resolve_device_identity
from .bert_gpu_evidence import write_json_new as _write_json
from .gpu_oneshot import (_runtime_uuid, _smi_snapshot,
                          _owner_presence_after_exit, _verify_controller_import_mode)
from .memory_budget import CudaMemorySnapshot, MemoryBudget
from .supervisor import OwnerSession, run_supervised

_GIB = 1024 ** 3
_MIB = 1024 ** 2
_HEADROOM = 256 * _MIB
_PEAK_SCOPE = "cumulative_since_pre_model_load_reset"
MODEL_ID = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
ASSETS = {
    "config.json": (483, "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83"),
    "model.safetensors": (267954768, "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"),
    "tokenizer.json": (466062, "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98"),
    "tokenizer_config.json": (48, "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"),
    "vocab.txt": (231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
}
TARGET_IDS = (101, 1996, 7500, 2379, 1996, 11500, 103, 3201, 1012, 102)
DONOR_IDS = (101, 1996, 6617, 2379, 1996, 11500, 103, 3201, 1012, 102)
# Objects whose ownership is uncertain after an interrupted call remain rooted
# until this single-use supervised process exits.
_OWNER_CUSTODY = {"entered": False, "session": None, "device_lease": None,
                  "budget": None, "memory_lease": None, "model": None,
                  "converted_model": None, "failure": None}


def _file_identity(path, maximum):
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(min(1024 * 1024, maximum - count + 1))
            if not block:
                break
            count += len(block)
            if count > maximum:
                raise ValueError("checkpoint asset exceeds its expected size")
            digest.update(block)
    return count, digest.hexdigest()


def stage_assets(snapshot, stage):
    """Copy and rehash the exact expected assets into a new local directory."""
    snapshot, stage = Path(snapshot), Path(stage)
    for name, (size, expected) in ASSETS.items():
        source, destination = snapshot / name, stage / name
        if _file_identity(source, size) != (size, expected):
            raise ValueError("offline checkpoint asset does not match the pinned identity")
        digest, count = hashlib.sha256(), 0
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            while True:
                block = incoming.read(min(1024 * 1024, size - count + 1))
                if not block:
                    break
                count += len(block)
                if count > size:
                    raise ValueError("checkpoint asset changed while staging")
                digest.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if ((count, digest.hexdigest()) != (size, expected)
                or _file_identity(destination, size) != (size, expected)):
            raise ValueError("staged checkpoint asset does not match the pinned identity")


def verify_tokenizer_inputs(tokenizer):
    """Verify output-word IDs and the two explicitly exposed development rows."""
    if (not tokenizer.is_fast or tokenizer.mask_token_id != 103
            or tokenizer.convert_tokens_to_ids("is") != 2003
            or tokenizer.convert_tokens_to_ids("are") != 2024
            or tokenizer.encode("is", add_special_tokens=False) != [2003]
            or tokenizer.encode("are", add_special_tokens=False) != [2024]):
        raise RuntimeError("pinned tokenizer output tokens differ")
    for text, expected in (("The farmer near the nurses [MASK] ready.", TARGET_IDS),
                           ("The farmers near the nurses [MASK] ready.", DONOR_IDS)):
        encoded = tokenizer(text, add_special_tokens=True, return_attention_mask=True)
        if encoded["input_ids"] != list(expected) or encoded["attention_mask"] != [1] * 10:
            raise RuntimeError("public development input tokenization differs from the fixed case")


def _load_model(cache_dir, torch):
    """Stage the pinned local snapshot and load the model on CPU only."""
    from importlib.metadata import version
    import transformers

    cache = Path(cache_dir).resolve(strict=True)
    snapshot = cache / "models--distilbert--distilbert-base-uncased" / "snapshots" / REVISION
    with tempfile.TemporaryDirectory(prefix="bert-gpu-calibration-stage-", dir=cache.parent) as directory:
        stage = Path(directory)
        stage_assets(snapshot, stage)
        with torch.device("cpu"):
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                stage, local_files_only=True, trust_remote_code=False, use_fast=True)
            model, info = transformers.AutoModelForMaskedLM.from_pretrained(
                stage, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                attn_implementation="eager", dtype=torch.float32, output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("checkpoint load reported missing, unexpected or mismatched weights")
        model.eval()
        if (type(model) is not transformers.DistilBertForMaskedLM
                or type(tokenizer) is not transformers.DistilBertTokenizerFast or model.training
                or model.config.model_type != "distilbert" or model.config.n_layers != 6
                or model.config.dim != 768 or model.config.vocab_size != 30522
                or model.config._attn_implementation != "eager"):
            raise RuntimeError("loaded model or tokenizer differs from the pinned contract")
        if (sum(p.numel() for p in model.parameters()) != 66985530
                or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
                or any(b.device.type != "cpu" for b in model.buffers())
                or model.get_input_embeddings().weight is not model.vocab_projector.weight):
            raise RuntimeError("loaded CPU model state or tied parameters differ")
        verify_tokenizer_inputs(tokenizer)
        metadata = {"model": MODEL_ID, "revision": REVISION,
                    "assets": {name: {"size": size, "sha256": digest}
                               for name, (size, digest) in ASSETS.items()},
                    "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
                    "transformers_version": str(transformers.__version__),
                    "tokenizers_version": version("tokenizers"), "safetensors_version": version("safetensors"),
                    "attention_implementation": model.config._attn_implementation,
                    "parameter_count": 66985530, "cpu_tied_weight_verified": True}
        return model, metadata


def check_memory(snapshot, *, expected_total, cap, commitment):
    """Reject an inconsistent or insufficient synchronized device snapshot.

    A held commitment is conservatively counted again after allocations have
    materialized. These endpoint checks cannot prevent external transient use.
    """
    names = ("device_free_bytes", "device_total_bytes", "allocated_bytes", "reserved_bytes",
             "peak_allocated_bytes", "peak_reserved_bytes")
    if (type(snapshot) is not dict or any(type(snapshot.get(k)) is not int or snapshot[k] < 0 for k in names)
            or any(type(x) is not int for x in (expected_total, cap, commitment))
            or expected_total <= 0 or commitment <= 0 or cap <= _HEADROOM
            or cap > expected_total):
        raise ValueError("invalid synchronized memory observation or policy")
    free, total = snapshot["device_free_bytes"], snapshot["device_total_bytes"]
    allocated, reserved = snapshot["allocated_bytes"], snapshot["reserved_bytes"]
    peak_allocated, peak_reserved = snapshot["peak_allocated_bytes"], snapshot["peak_reserved_bytes"]
    if (total != expected_total or free > total or allocated > reserved or reserved > total
            or peak_allocated < allocated or peak_reserved < reserved
            or peak_allocated > peak_reserved or peak_reserved > total):
        raise ValueError("inconsistent synchronized memory counters")
    reserve = max(_GIB, (15 * total + 99) // 100)
    if (free < reserve + commitment or reserved + reserve + commitment > total
            or free < reserve + _HEADROOM or peak_reserved > cap - _HEADROOM):
        raise RuntimeError("insufficient device reserve or allocator headroom")
    return {"reserve_bytes": reserve, "commitment_bytes": commitment,
            "allocator_cap_bytes": cap, "allocator_headroom_bytes": _HEADROOM}


def reserve_owner_budget(telemetry, *, cap, custody=None):
    """Reserve the whole owner plus aggregate trace allowances before model work."""
    if type(cap) is not int or cap <= _HEADROOM:
        raise ValueError("invalid allocator cap")
    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=64 * 1024, max_host_trace_bytes=_MIB)
    if custody is not None:
        custody["budget"] = budget
    lease = budget.acquire(cap + _HEADROOM, max_gpu_trace_bytes=64 * 1024,
                           max_host_trace_bytes=_MIB)
    if custody is not None:
        custody["memory_lease"] = lease
    # Any interrupted claim leaves the child unverified and is reconciled by
    # terminating that owner, not by guessing which count update completed.
    lease.reserve_trace("gpu", 64 * 1024)
    lease.reserve_trace("host", _MIB)
    return budget, lease


@dataclass(frozen=True)
class Observation:
    before: object
    after: object
    before_bits: bytes
    after_bits: bytes | None
    mask_logits_bytes: bytes
    is_bits: int
    are_bits: int
    events: tuple


def _raw_float32(value):
    """Copy original float32 bits to little-endian bytes, with no float cast."""
    import torch

    if (type(value) is not torch.Tensor or value.dtype != torch.float32
            or value.device.type not in ("cpu", "cuda") or sys.byteorder != "little"):
        raise ValueError("expected a little-endian CPU or CUDA float32 tensor")
    return value.detach().view(torch.int32).to("cpu").contiguous().numpy().tobytes()


def _unexpected_hooks(model):
    import torch

    names = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
    if any(getattr(module, name, {}) for module in model.modules() for name in names):
        return True
    registry = torch.nn.modules.module
    return any(getattr(registry, name, {}) for name in (
        "_global_forward_pre_hooks", "_global_forward_hooks",
        "_global_backward_pre_hooks", "_global_backward_hooks"))


def observe(model, input_ids, *, is_id, are_id, source=None, self_patch=False, synchronize=None,
            check_owner=None):
    """Capture or replace the fixed block-2 subject vector in one forward.

    Input and externally supplied source are owned by the caller and must stay
    quiescent. The returned device captures must remain charged until disposed.
    This helper does not verify checkpoint identity, provenance, or admission.
    """
    import torch
    from transformers import DistilBertForMaskedLM
    from .intervention_plan import ActivationPlan

    if ((synchronize is not None and not callable(synchronize))
            or (check_owner is not None and not callable(check_owner))):
        raise TypeError("owner callbacks must be callable")

    def ownership():
        if check_owner is not None:
            check_owner()

    def guarded(function):
        ownership()
        result = function()
        ownership()
        return result

    ownership()

    def completion(stage):
        if synchronize is not None:
            synchronize(stage)
        elif input_ids.device.type == "cuda":
            torch.cuda.synchronize(input_ids.device)

    if type(model) is not DistilBertForMaskedLM or any(m.training for m in model.modules()):
        raise ValueError("expected an evaluation-mode DistilBERT masked language model")
    if _unexpected_hooks(model):
        raise RuntimeError("pre-existing local or global hooks are unsupported")
    if (type(input_ids) is not torch.Tensor or input_ids.dtype != torch.int64
            or input_ids.device.type not in ("cpu", "cuda") or tuple(input_ids.shape) != (1, 10)):
        raise ValueError("expected one ten-token int64 input row")
    if any(p.device != input_ids.device or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("model parameters and input must share a float32 execution device")
    if any(b.device != input_ids.device for b in model.buffers()):
        raise ValueError("model buffers and input must share the execution device")
    vocab, hidden = model.config.vocab_size, model.config.dim
    if (len(model.distilbert.transformer.layer) != 6 or type(is_id) is not int
            or type(are_id) is not int or is_id == are_id
            or not 0 <= is_id < vocab or not 0 <= are_id < vocab
            or guarded(lambda: bool((input_ids < 0).any()))
            or guarded(lambda: bool((input_ids >= vocab).any()))):
        raise ValueError("invalid fixed-site model or vocabulary coordinates")
    if type(self_patch) is not bool or (self_patch and source is not None):
        raise ValueError("self and external sources are mutually exclusive")
    if source is not None and (type(source) is not torch.Tensor or source.device != input_ids.device
                               or source.dtype != torch.float32 or tuple(source.shape) != (hidden,)
                               or not guarded(lambda: bool(torch.isfinite(source).all()))):
        raise ValueError("source must be a finite same-device hidden vector")
    ownership()

    path = "distilbert.transformer.layer.2"
    site = {"module_path": path, "invocation": 0, "batch_index": 0, "token_index": 2}
    steps = [{"name": "before", "operation": "capture", **site}]
    if self_patch or source is not None:
        patch = {"name": "patch", "operation": "patch", **site}
        patch["source_capture" if self_patch else "value"] = "before" if self_patch else source
        steps.extend((patch, {"name": "after", "operation": "capture", **site}))
    plan = ActivationPlan(model, steps, max_tensor_bytes=10 * hidden * 4,
                          max_capture_bytes=2 * hidden * 4, max_source_bytes=hidden * 4,
                          max_steps=3, max_invocations=1)
    ownership()
    with torch.inference_mode(), plan:
        attention_mask = guarded(lambda: torch.ones_like(input_ids))
        ownership()
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_attentions=False, output_hidden_states=False,
                       return_dict=True).logits
        ownership()
        if (type(logits) is not torch.Tensor or logits.dtype != torch.float32
                or logits.device != input_ids.device or tuple(logits.shape) != (1, 10, vocab)
                or not guarded(lambda: bool(torch.isfinite(logits).all()))):
            raise RuntimeError("invalid or nonfinite complete model logits")
        ownership()
        mask_bits = guarded(lambda: _raw_float32(logits[0, 6]))
        ownership()
        completion("output_live")
    del logits
    completion("output_disposed")
    captures, metadata = plan.captures, plan.metadata
    if _unexpected_hooks(model):
        raise RuntimeError("hook cleanup is uncertain; discard the owner")
    ownership()
    names = ["before", "patch", "after"] if len(steps) == 3 else ["before"]
    operations = ["capture", "patch", "capture"] if len(steps) == 3 else ["capture"]
    expected = []
    for index, (name, operation) in enumerate(zip(names, operations)):
        expected.append({"name": name, "operation": operation, "order": index,
                         "module_path": path, "module_aliases": [path], "call_site_known": False,
                         "forward_index": 0, "invocation": 0, "batch_index": 0,
                         "token_index": 2, "output_selector": [0], "output_shape": [1, 10, hidden],
                         "selected_shape": [hidden], "dtype": "torch.float32",
                         "device": str(input_ids.device),
                         "source_capture": "before" if self_patch and index == 1 else None,
                         "preceding_same_site_actions": names[:index]})
    expected_meta = {"steps": expected, "capture_bytes": hidden * 4 * (2 if len(steps) == 3 else 1),
                     "source_bytes": hidden * 4 if source is not None else 0}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(metadata) != canonical(expected_meta):
        raise RuntimeError("the recorded action metadata differs from the fixed site")
    before, after = captures["before"], captures.get("after")
    if (not guarded(lambda: bool(torch.isfinite(before).all()))
            or (after is not None and not guarded(lambda: bool(torch.isfinite(after).all())))):
        raise RuntimeError("nonfinite selected activation")
    ownership()
    before_bits = guarded(lambda: _raw_float32(before))
    ownership()
    after_bits = None if after is None else guarded(lambda: _raw_float32(after))
    ownership()
    expected_bits = before_bits if self_patch else (None if source is None else
                                                    guarded(lambda: _raw_float32(source)))
    if after is not None and after_bits != expected_bits:
        raise RuntimeError("the intended source did not land at the selected site")
    is_bits = int.from_bytes(mask_bits[is_id * 4:(is_id + 1) * 4], "little")
    are_bits = int.from_bytes(mask_bits[are_id * 4:(are_id + 1) * 4], "little")
    return Observation(before, after, before_bits, after_bits, mask_bits, is_bits, are_bits,
                       tuple(metadata["steps"]))


def _decode_float32(bits):
    exponent, fraction = (bits >> 23) & 0xff, bits & 0x7fffff
    if exponent == 0xff:
        raise ValueError("nonfinite selected logit")
    magnitude = (math.ldexp(float(fraction), -149) if exponent == 0
                 else math.ldexp(float((1 << 23) | fraction), exponent - 150))
    return -magnitude if bits >> 31 else magnitude


def run_sequence(model, target_ids, donor_ids, *, is_id, are_id, checkpoint, synchronize=None,
                 check_owner=None):
    """Measure a fixed capture/self/cross sequence while all captures remain live.

    The checkpoint callback synchronizes, samples memory and checks admission.
    It receives only the stage name and number of live observations. Returned
    evidence contains host bytes and metadata, never device tensor references.
    """
    if not callable(checkpoint):
        raise TypeError("a synchronized checkpoint callback is required")
    observations = []
    checkpoint("before_target", len(observations))
    observations.append(observe(model, target_ids, is_id=is_id, are_id=are_id, synchronize=synchronize,
                                check_owner=check_owner))
    checkpoint("after_target", len(observations))
    checkpoint("before_donor", len(observations))
    observations.append(observe(model, donor_ids, is_id=is_id, are_id=are_id, synchronize=synchronize,
                                check_owner=check_owner))
    checkpoint("after_donor", len(observations))
    checkpoint("before_self", len(observations))
    observations.append(observe(model, target_ids, is_id=is_id, are_id=are_id, self_patch=True,
                                synchronize=synchronize, check_owner=check_owner))
    checkpoint("after_self", len(observations))
    checkpoint("before_cross", len(observations))
    observations.append(observe(model, target_ids, is_id=is_id, are_id=are_id,
                                source=observations[1].before, synchronize=synchronize,
                                check_owner=check_owner))
    checkpoint("after_cross", len(observations))
    baseline, donor, same, cross = observations
    if same.mask_logits_bytes != baseline.mask_logits_bytes or same.before_bits != baseline.before_bits:
        raise RuntimeError("same-forward self-patch or repeated target baseline differs")
    if cross.before_bits != baseline.before_bits or cross.after_bits != donor.before_bits:
        raise RuntimeError("cross-patch source or repeated target baseline differs")
    if donor.before_bits == baseline.before_bits:
        raise RuntimeError("donor and target selected states are identical")
    rows = []
    for name, item in zip(("target", "donor", "self", "cross"), observations):
        rows.append({"stage": name, "is_bits": item.is_bits, "are_bits": item.are_bits,
                     "is_logit": _decode_float32(item.is_bits),
                     "are_logit": _decode_float32(item.are_bits),
                     "mask_logits_sha256": hashlib.sha256(item.mask_logits_bytes).hexdigest(),
                     "before_float32_le_base64": base64.b64encode(item.before_bits).decode("ascii"),
                     "after_float32_le_base64": (None if item.after_bits is None else
                                                    base64.b64encode(item.after_bits).decode("ascii")),
                     "events": list(item.events)})
    return {"forward_count": 4, "observations": rows, "self_mask_logits_bitwise_equal": True,
            "cross_source_landed": True, "source_differs_from_target": True}


def _require_pristine_dependencies():
    if any(name in sys.modules for name in ("torch", "transformers", "numpy")):
        raise RuntimeError("CUDA-capable dependencies were imported before lease acquisition")


def _verify_startup():
    _verify_controller_import_mode()
    site = Path(os.environ.get("LOOKING_GLASS_BERT_BOOTSTRAP_SITE", ""))
    if (not sys.flags.isolated or not sys.flags.no_site
            or os.environ.get("LOOKING_GLASS_BERT_BOOTSTRAP_PID") != str(os.getpid())
            or not site.is_absolute() or not site.is_dir()):
        raise RuntimeError("calibration requires its isolated source bootstrap")


def _import_torch():
    import torch
    return torch


def _guarded_call(session, lease, function, *args):
    OwnerSession.check(session, lease)
    result = function(*args)
    OwnerSession.check(session, lease)
    return result


def _guarded_snapshot(session, lease, torch):
    """Synchronize and recheck ownership around every CUDA telemetry call."""
    _guarded_call(session, lease, torch.cuda.synchronize, 0)
    free, total = _guarded_call(session, lease, torch.cuda.mem_get_info, 0)
    return {"device_free_bytes": free, "device_total_bytes": total,
            "allocated_bytes": _guarded_call(session, lease, torch.cuda.memory_allocated, 0),
            "reserved_bytes": _guarded_call(session, lease, torch.cuda.memory_reserved, 0),
            "peak_allocated_bytes": _guarded_call(session, lease, torch.cuda.max_memory_allocated, 0),
            "peak_reserved_bytes": _guarded_call(session, lease, torch.cuda.max_memory_reserved, 0)}


def measure_owned(session, lease, torch, cache_dir, *, record_stage):
    """Measure one bounded owner workload under an already held device lease."""
    if type(session) is not OwnerSession or type(lease) is not DeviceLease:
        raise ValueError("an actual supervised owner and device lease are required")
    if not callable(record_stage):
        raise TypeError("a stage recorder is required")
    OwnerSession.check(session, lease)
    if torch.cuda.is_initialized() is not False:
        raise RuntimeError("CUDA was initialized before the owner check")
    count = _guarded_call(session, lease, torch.cuda.device_count)
    if type(count) is not int or count != 1:
        raise RuntimeError("exactly one CUDA-visible device is required")
    properties = _guarded_call(session, lease, torch.cuda.get_device_properties, 0)
    runtime_uuid = _runtime_uuid(torch, properties.uuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid, device_count=count)
    if not 7 * _GIB <= properties.total_memory <= 9 * _GIB:
        raise RuntimeError("device total is outside the expected 8 GiB range")
    allocator_backend = _guarded_call(session, lease, torch.cuda.get_allocator_backend)
    if allocator_backend != "native":
        raise RuntimeError("the active CUDA allocator backend is not native")
    _guarded_call(session, lease, torch.cuda.set_per_process_memory_fraction, .25, 0)
    cap = int(.25 * properties.total_memory)
    if _guarded_call(session, lease, torch.cuda.get_per_process_memory_fraction, 0) != cap / properties.total_memory:
        raise RuntimeError("allocator fraction readback differs from the cap")
    _guarded_call(session, lease, torch.cuda.reset_peak_memory_stats, 0)
    initial = _guarded_snapshot(session, lease, torch)
    record_stage("before_model_load", {"memory": initial, "observations_live": 0,
                                       "peak_scope": _PEAK_SCOPE})
    if initial["device_total_bytes"] != properties.total_memory:
        raise RuntimeError("CUDA property and device telemetry totals disagree")
    if initial["device_free_bytes"] < 3 * _GIB:
        raise RuntimeError("insufficient device-wide free memory before model loading")
    check_memory(initial, expected_total=initial["device_total_bytes"], cap=cap,
                 commitment=cap + _HEADROOM)
    total = initial["device_total_bytes"]
    commitment = cap + _HEADROOM

    def telemetry():
        free, observed_total = _guarded_call(session, lease, torch.cuda.mem_get_info, 0)
        if observed_total != total:
            raise RuntimeError("CUDA telemetry total changed during admission")
        return CudaMemorySnapshot(free, observed_total,
                                  _guarded_call(session, lease, torch.cuda.memory_allocated, 0),
                                  _guarded_call(session, lease, torch.cuda.memory_reserved, 0))

    budget, memory_lease = reserve_owner_budget(telemetry, cap=cap, custody=_OWNER_CUSTODY)
    if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise RuntimeError("unsupported module conversion flags")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(17)
    _guarded_call(session, lease, torch.cuda.manual_seed, 17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    execution_config = {"device": "cuda:0", "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "overwrite_module_params_on_conversion": torch.__future__.get_overwrite_module_params_on_conversion(),
        "swap_module_params_on_conversion": torch.__future__.get_swap_module_params_on_conversion()}
    expected_config = {"device": "cuda:0", "threads": 1, "interop_threads": 1,
        "deterministic_algorithms": True, "matmul_tf32": False, "cudnn_tf32": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "overwrite_module_params_on_conversion": False, "swap_module_params_on_conversion": False}
    if json.dumps(execution_config, sort_keys=True) != json.dumps(expected_config, sort_keys=True):
        raise RuntimeError("runtime execution configuration differs from the calibration contract")

    def snapshot(stage, count=None):
        value = _guarded_snapshot(session, lease, torch)
        check_memory(value, expected_total=total, cap=cap, commitment=commitment)
        record_stage(stage, {"memory": value, "observations_live": count, "peak_scope": _PEAK_SCOPE})
        return value

    def synchronize(stage):
        _guarded_call(session, lease, torch.cuda.synchronize, 0)

    model, metadata = _load_model(cache_dir, torch)
    _OWNER_CUSTODY["model"] = model
    snapshot("before_model_copy")
    converted = _guarded_call(session, lease, model.to, "cuda:0")
    _OWNER_CUSTODY["converted_model"] = converted
    if (converted is not model or model.training
            or sum(p.numel() for p in model.parameters()) != 66985530
            or any(p.device.type != "cuda" or p.device.index != 0 or p.dtype != torch.float32
                   for p in model.parameters())
            or any(b.device.type != "cuda" or b.device.index != 0 for b in model.buffers())
            or model.get_input_embeddings().weight is not model.vocab_projector.weight
            or torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise RuntimeError("GPU model state, tied parameter, or conversion flags differ")
    snapshot("after_model_copy")
    target_cpu = torch.tensor([list(TARGET_IDS)], dtype=torch.long, device="cpu")
    donor_cpu = torch.tensor([list(DONOR_IDS)], dtype=torch.long, device="cpu")
    target = _guarded_call(session, lease, target_cpu.to, "cuda:0")
    donor = _guarded_call(session, lease, donor_cpu.to, "cuda:0")
    if any(t.device.type != "cuda" or t.device.index != 0 for t in (target, donor)):
        raise RuntimeError("development inputs were not transferred to cuda:0")
    sequence = run_sequence(model, target, donor, is_id=2003, are_id=2024,
                            checkpoint=snapshot, synchronize=synchronize,
                            check_owner=lambda: OwnerSession.check(session, lease))
    # run_sequence's observations and all device capture references are gone
    # when it returns. The returned evidence contains host copies only.
    references = {"model": weakref.ref(model), "target": weakref.ref(target),
                  "donor": weakref.ref(donor)}
    _OWNER_CUSTODY["converted_model"] = None
    _OWNER_CUSTODY["model"] = None
    del converted, model, target, donor, target_cpu, donor_cpu
    gc.collect()
    survivors = {name: ref() is not None for name, ref in references.items()}
    record_stage("cleanup_weakrefs", survivors)
    synchronize("after_disposal")
    _guarded_call(session, lease, torch.cuda.empty_cache)
    snapshot("cleanup_before_workspace_clear")
    if (torch.__version__ != "2.8.0+cu128"
            or not callable(getattr(torch._C, "_cuda_clearCublasWorkspaces", None))):
        raise RuntimeError("unsupported PyTorch cuBLAS workspace cleanup API")
    _guarded_call(session, lease, torch._C._cuda_clearCublasWorkspaces)
    synchronize("after_workspace_clear")
    _guarded_call(session, lease, torch.cuda.empty_cache)
    cleanup = snapshot("after_cleanup")
    if any(survivors.values()) or cleanup["allocated_bytes"] != 0 or cleanup["reserved_bytes"] != 0:
        raise RuntimeError("owner references or CUDA allocator bytes remain after disposal")
    active = len(budget.active_leases())
    if active != 1:
        raise RuntimeError("whole-owner commitment was not retained through cleanup")
    return {"target_ids": list(TARGET_IDS), "donor_ids": list(DONOR_IDS),
            "output_token_ids": {"is": 2003, "are": 2024}, "seed": 17,
            "sequence": sequence, "cleanup": cleanup, "cleanup_weakrefs": survivors,
            "budget_active_before_candidate_write": active, "runtime_uuid_verified": True,
            "device_name": properties.name, "device_total_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "allocator_fraction": .25, "allocator_cap_bytes": cap, "allocator_backend": allocator_backend,
            "whole_run_commitment_bytes": commitment,
            "gpu_trace_claim_bytes": 64 * 1024, "host_trace_claim_bytes": _MIB,
            "torch_version": str(torch.__version__), "cuda_build": str(torch.version.cuda),
            "python_version": sys.version.split()[0],
            "model_metadata": metadata, "execution_config": execution_config, "peak_scope": _PEAK_SCOPE,
            "workspace_cleanup": "PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces"}


def _release_owner_budget_after_candidate():
    """Release the claims only after result serialization and local disposal."""
    budget, memory_lease = _OWNER_CUSTODY["budget"], _OWNER_CUSTODY["memory_lease"]
    if type(budget) is not MemoryBudget or memory_lease not in budget.active_leases():
        raise RuntimeError("owner budget custody is not active")
    memory_lease.release_trace("gpu", 64 * 1024)
    memory_lease.release_trace("host", _MIB)
    memory_lease.release()
    if budget.active_leases():
        raise RuntimeError("memory commitments remain after candidate disposal")


def owner_main(run_dir, cache_dir, run_id):
    """A direct child retains its shared device lease until process exit."""
    try:
        if _OWNER_CUSTODY["entered"] is not False:
            raise RuntimeError("owner is single-use")
        _OWNER_CUSTODY["entered"] = True
        _verify_startup()
        _require_pristine_dependencies()
        session = OwnerSession.from_environment()
        _OWNER_CUSTODY["session"] = session
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _OWNER_CUSTODY["device_lease"] = lease
        OwnerSession.check(session, lease)
        if session.identity is None:
            raise RuntimeError("a physical GPU identity is required")
        _write_json(Path(run_dir) / "owner.json", {"schema": "bert-gpu-calibration-owner-v1",
                    "run_id": run_id, "supervisor_run_id": session.run_id, "pid": os.getpid()})
        torch = _import_torch()
        result = measure_owned(session, lease, torch, cache_dir,
                               record_stage=lambda name, value: _write_json(
                                   Path(run_dir) / (name + ".json"), value))
        _write_json(Path(run_dir) / "candidate.json", {"schema": "bert-gpu-calibration-candidate-v1",
                    "run_id": run_id, "supervisor_run_id": session.run_id,
                    "pid": os.getpid(), "measurement": result})
        _, candidate_hash = _file_identity(Path(run_dir) / "candidate.json", _MIB)
        del result
        gc.collect()
        _release_owner_budget_after_candidate()
        _write_json(Path(run_dir) / "final.json", {"schema": "bert-gpu-calibration-final-v1",
                    "run_id": run_id, "supervisor_run_id": session.run_id, "pid": os.getpid(),
                    "candidate_sha256": candidate_hash, "active_memory_leases": 0})
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as exc:
        if _OWNER_CUSTODY["failure"] is None:
            _OWNER_CUSTODY["failure"] = exc
        # Never attest an unverified cleanup; the supervisor's marker remains.
        try:
            _write_json(Path(run_dir) / "failure.json", {"schema": "bert-gpu-calibration-failure-v1",
                        "run_id": run_id, "status": "quarantined", "error_type": type(exc).__name__,
                        "private_traceback": traceback.format_exc()[-8192:]})
        except BaseException:
            pass
        return 1


_SOURCE_PATHS = ("looking_glass/__init__.py", "looking_glass/bert_gpu_calibration.py",
                 "looking_glass/bert_gpu_evidence.py", "docs/BERT_GPU_DIAGNOSTIC_WRITER_V2.md",
                 "looking_glass/gpu_oneshot.py", "looking_glass/device_identity.py",
                 "looking_glass/device_lease.py", "looking_glass/supervisor.py",
                 "looking_glass/memory_budget.py", "looking_glass/intervention_plan.py",
                 "looking_glass/interventions.py", "looking_glass/model_ownership.py",
                 "docs/BERT_GPU_CALIBRATION_V1.md", "scripts/bert_gpu_bootstrap.py")


def _source_identity(repo):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                            text=True, check=True, timeout=10).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                           capture_output=True, text=True, check=True, timeout=10).stdout
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or dirty:
        raise RuntimeError("calibration requires a clean committed checkout")
    hashes = {}
    for relative in _SOURCE_PATHS:
        stored = subprocess.run(["git", "show", commit + ":" + relative], cwd=repo,
                                capture_output=True, check=True, timeout=10).stdout
        current = (repo / relative).read_bytes()
        if current != stored:
            raise RuntimeError("committed calibration source bytes differ from checkout")
        hashes[relative] = hashlib.sha256(current).hexdigest()
    return commit, hashes


def _verify_offline_cache(cache):
    snapshot = Path(cache) / "models--distilbert--distilbert-base-uncased" / "snapshots" / REVISION
    for name, (size, expected) in ASSETS.items():
        if _file_identity(snapshot / name, size) != (size, expected):
            raise RuntimeError("offline checkpoint integrity mismatch")


def _read_json_capped(path, maximum):
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("serialized evidence exceeds its byte cap")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate evidence key")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError("nonfinite evidence")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
    if type(value) is not dict:
        raise ValueError("serialized evidence must be an object")
    return value


def _validate_measurement(value):
    if type(value) is not dict:
        raise ValueError("owner measurement is not an object")
    sequence, cleanup, weakrefs = value.get("sequence"), value.get("cleanup"), value.get("cleanup_weakrefs")
    if (value.get("target_ids") != list(TARGET_IDS) or value.get("donor_ids") != list(DONOR_IDS)
            or value.get("output_token_ids") != {"is": 2003, "are": 2024}
            or type(value.get("budget_active_before_candidate_write")) is not int
            or value["budget_active_before_candidate_write"] != 1
            or type(cleanup) is not dict or any(type(cleanup.get(key)) is not int or cleanup[key] != 0
                                               for key in ("allocated_bytes", "reserved_bytes"))
            or type(weakrefs) is not dict or set(weakrefs) != {"model", "target", "donor"}
            or any(flag is not False for flag in weakrefs.values()) or type(sequence) is not dict
            or type(sequence.get("forward_count")) is not int or sequence["forward_count"] != 4
            or any(sequence.get(key) is not True for key in
                   ("self_mask_logits_bitwise_equal", "cross_source_landed", "source_differs_from_target"))):
        raise ValueError("owner measurement failed the fixed workload or cleanup contract")
    rows = sequence.get("observations")
    if (type(rows) is not list or len(rows) != 4 or any(type(row) is not dict for row in rows)
            or [row.get("stage") for row in rows] != ["target", "donor", "self", "cross"]):
        raise ValueError("owner observations are incomplete or out of order")
    total, cap = value.get("device_total_bytes"), value.get("allocator_cap_bytes")
    metadata = value.get("model_metadata")
    expected_assets = {name: {"size": size, "sha256": digest}
                       for name, (size, digest) in ASSETS.items()}
    expected_config = {"device": "cuda:0", "threads": 1, "interop_threads": 1,
        "deterministic_algorithms": True, "matmul_tf32": False, "cudnn_tf32": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "overwrite_module_params_on_conversion": False, "swap_module_params_on_conversion": False}
    if (value.get("runtime_uuid_verified") is not True
            or value.get("device_name") != "NVIDIA GeForce RTX 2070 SUPER"
            or value.get("compute_capability") != [7, 5]
            or type(total) is not int or not 7 * _GIB <= total <= 9 * _GIB
            or type(cap) is not int or cap != int(.25 * total)
            or value.get("allocator_backend") != "native"
            or type(value.get("allocator_fraction")) is not float or value["allocator_fraction"] != .25
            or type(value.get("whole_run_commitment_bytes")) is not int
            or value["whole_run_commitment_bytes"] != cap + _HEADROOM
            or type(value.get("gpu_trace_claim_bytes")) is not int or value["gpu_trace_claim_bytes"] != 65536
            or type(value.get("host_trace_claim_bytes")) is not int or value["host_trace_claim_bytes"] != _MIB
            or type(value.get("seed")) is not int or value["seed"] != 17
            or value.get("torch_version") != "2.8.0+cu128" or value.get("cuda_build") != "12.8"
            or type(value.get("python_version")) is not str or not value["python_version"]
            or value.get("peak_scope") != _PEAK_SCOPE
            or json.dumps(value.get("execution_config"), sort_keys=True) != json.dumps(expected_config, sort_keys=True)
            or type(metadata) is not dict or metadata.get("model") != MODEL_ID
            or metadata.get("revision") != REVISION or metadata.get("assets") != expected_assets
            or metadata.get("model_class") != "DistilBertForMaskedLM"
            or metadata.get("tokenizer_class") != "DistilBertTokenizerFast"
            or metadata.get("transformers_version") != "4.57.3"
            or any(type(metadata.get(key)) is not str or not metadata[key]
                   for key in ("tokenizers_version", "safetensors_version"))
            or metadata.get("attention_implementation") != "eager"
            or type(metadata.get("parameter_count")) is not int or metadata["parameter_count"] != 66985530
            or metadata.get("cpu_tied_weight_verified") is not True):
        raise ValueError("model, runtime, device or budget identity differs")

    def vector(encoded):
        if type(encoded) is not str:
            raise ValueError("missing encoded activation vector")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid encoded activation vector") from exc
        if (len(raw) != 768 * 4 or base64.b64encode(raw).decode("ascii") != encoded
                or any(int.from_bytes(raw[i:i + 4], "little") & 0x7f800000 == 0x7f800000
                       for i in range(0, len(raw), 4))):
            raise ValueError("activation vector has invalid bytes")
        return raw

    path = "distilbert.transformer.layer.2"
    before, after = [], []
    for row in rows:
        stage = row["stage"]
        if not isinstance(row.get("mask_logits_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", row["mask_logits_sha256"]):
            raise ValueError("invalid mask-logit digest")
        for token in ("is", "are"):
            bits, decoded = row.get(token + "_bits"), row.get(token + "_logit")
            if type(bits) is not int or not 0 <= bits <= 0xffffffff or type(decoded) is not float:
                raise ValueError("invalid selected-logit representation")
            expected = _decode_float32(bits)
            if (not math.isfinite(decoded) or decoded != expected
                    or math.copysign(1.0, decoded) != math.copysign(1.0, expected)):
                raise ValueError("selected-logit value does not match its bits")
        before.append(vector(row.get("before_float32_le_base64")))
        encoded_after = row.get("after_float32_le_base64")
        after.append(None if encoded_after is None else vector(encoded_after))
        names = ["before"] if stage in ("target", "donor") else ["before", "patch", "after"]
        operations = ["capture"] if len(names) == 1 else ["capture", "patch", "capture"]
        expected_events = [{"name": name, "operation": operation, "order": index,
            "module_path": path, "module_aliases": [path], "call_site_known": False,
            "forward_index": 0, "invocation": 0, "batch_index": 0, "token_index": 2,
            "output_selector": [0], "output_shape": [1, 10, 768], "selected_shape": [768],
            "dtype": "torch.float32", "device": "cuda:0",
            "source_capture": "before" if stage == "self" and index == 1 else None,
            "preceding_same_site_actions": names[:index]}
            for index, (name, operation) in enumerate(zip(names, operations))]
        canonical = lambda data: json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical(row.get("events")) != canonical(expected_events):
            raise ValueError("recorded action coordinates differ")
    if (after[0] is not None or after[1] is not None or before[0] == before[1]
            or before[2] != before[0] or after[2] != before[0]
            or before[3] != before[0] or after[3] != before[1]
            or rows[2]["mask_logits_sha256"] != rows[0]["mask_logits_sha256"]
            or any(rows[2][token + "_bits"] != rows[0][token + "_bits"] for token in ("is", "are"))):
        raise ValueError("saved self or donor patch evidence does not agree")


_STAGE_COUNTS = (("before_model_load", 0), ("before_model_copy", None), ("after_model_copy", None),
                 ("before_target", 0), ("after_target", 1), ("before_donor", 1), ("after_donor", 2),
                 ("before_self", 2), ("after_self", 3), ("before_cross", 3), ("after_cross", 4),
                 ("cleanup_before_workspace_clear", None), ("after_cleanup", None))


def _read_stage_observations(run_dir):
    records = {}
    for stage in (*[name for name, _ in _STAGE_COUNTS], "cleanup_weakrefs"):
        try:
            records[stage] = _read_json_capped(Path(run_dir) / (stage + ".json"), 4096)
        except (OSError, ValueError):
            pass
    return records


def _validate_stage_observations(records, measurement):
    prior_allocated, prior_reserved = 0, 0
    for name, count in _STAGE_COUNTS:
        entry = records.get(name)
        if (type(entry) is not dict or set(entry) != {"memory", "observations_live", "peak_scope"}
                or entry["peak_scope"] != _PEAK_SCOPE
                or (count is None and entry["observations_live"] is not None)
                or (count is not None and (type(entry["observations_live"]) is not int
                                            or entry["observations_live"] != count))):
            raise ValueError("missing or inconsistent stage record")
        memory = entry["memory"]
        check_memory(memory, expected_total=measurement["device_total_bytes"],
                     cap=measurement["allocator_cap_bytes"],
                     commitment=measurement["whole_run_commitment_bytes"])
        if memory["peak_allocated_bytes"] < prior_allocated or memory["peak_reserved_bytes"] < prior_reserved:
            raise ValueError("cumulative peak counters decreased")
        prior_allocated, prior_reserved = memory["peak_allocated_bytes"], memory["peak_reserved_bytes"]
    if (records["before_model_load"]["memory"]["device_free_bytes"] < 3 * _GIB
            or records.get("cleanup_weakrefs") != measurement["cleanup_weakrefs"]
            or records["after_cleanup"]["memory"] != measurement["cleanup"]):
        raise ValueError("initial or final cleanup stage differs from the candidate")


def run_calibration(*, workspace_root, cache_dir, output, site_packages, smi):
    """Supervise one calibration; never accept an unverified child outcome."""
    _verify_startup()
    root = Path(workspace_root).resolve(strict=True)
    cache = Path(cache_dir).resolve(strict=True)
    site = Path(site_packages).resolve(strict=True)
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
        raise ValueError("output must be an absolute, new file")
    with tempfile.NamedTemporaryFile(prefix=".bert-calibration-write-test-", dir=destination.parent):
        pass
    repo = Path(__file__).resolve().parents[1]
    commit, source_hashes = _source_identity(repo)
    _verify_offline_cache(cache)
    gpu_uuid, before = _smi_snapshot(smi)
    environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC")
                   if key in os.environ}
    system_root = Path(environment.get("SYSTEMROOT", "C:/Windows"))
    environment.update({"PATH": str(system_root / "System32"), "CUDA_VISIBLE_DEVICES": gpu_uuid,
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "HF_HOME": str(cache.parent), "PYTHONNOUSERSITE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
    run_dir = Path(tempfile.mkdtemp(prefix="bert-gpu-calibration-", dir=root / "work"))
    prefix = Path(tempfile.mkdtemp(prefix="pycache-", dir=run_dir))
    environment["PYTHONPYCACHEPREFIX"] = str(prefix)
    run_id = secrets.token_hex(16)
    _write_json(run_dir / "started.json", {"schema": "bert-gpu-calibration-start-v1", "run_id": run_id,
               "source_commit": commit, "started_utc": before["timestamp_utc"]})
    bootstrap = repo / "scripts" / "bert_gpu_bootstrap.py"
    command = [sys._base_executable, "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
               str(bootstrap), "--bootstrap-site", str(site), "--owner", str(run_dir), str(cache), run_id]
    supervision = None
    supervisor_error = None
    try:
        supervision = run_supervised(command, device_lease_path=identity.lock_path,
                                     marker_path=identity.marker_path, identity=identity,
                                     timeout=300, lease_timeout=30, cwd=repo, env=environment,
                                     inherit_environment=False)
    except BaseException as exc:
        supervisor_error = type(exc).__name__
    supervision_evidence_error = None
    if supervision is not None:
        try:
            _write_json(run_dir / "supervision.json", {"run_id": run_id,
                        "supervisor_run_id": supervision.run_id, "pid": supervision.pid,
                        "exit_code": supervision.exit_code, "clean": supervision.clean,
                        "timed_out": supervision.timed_out, "reason": supervision.reason})
        except Exception as exc:
            supervision_evidence_error = type(exc).__name__
    after = None
    telemetry_error = None
    try:
        after_uuid, after = _smi_snapshot(smi)
        if after_uuid != gpu_uuid:
            raise RuntimeError("physical inventory changed after owner exit")
    except Exception as exc:
        after = None
        telemetry_error = type(exc).__name__
    presence = (_owner_presence_after_exit(smi, supervision.pid) if supervision is not None
                else {"query": "not_attempted", "owner_pid_listed": None})
    outcome = {"schema": "bert-gpu-calibration-result-v1", "run_id": run_id, "source_commit": commit,
               "source_sha256": source_hashes, "controller_fresh_pycache": True,
               "status": "quarantined", "supervisor": ({"exit_code": supervision.exit_code,
                   "clean": supervision.clean, "timed_out": supervision.timed_out}
                   if supervision is not None else None), "supervisor_error_type": supervisor_error,
               "supervision_evidence_error_type": supervision_evidence_error,
               "nvidia_smi_before": before, "nvidia_smi_after": after,
               "owner_presence_after_exit": presence, "post_telemetry_error_type": telemetry_error,
               "limitations": ["development-only shape-specific calibration; no held-out study",
                   "no latency, throughput or worst-case memory claim",
                   "the allocator cap excludes CUDA context and non-PyTorch allocations",
                   "the logical host-trace allowance is not a bound on total host RAM",
                   "point-in-time device memory observations can miss external pressure",
                   "the cooperative lease does not control unrelated GPU applications",
                   "Windows WDDM may not expose all GPU processes or their memory"]}
    partial = _read_stage_observations(run_dir)
    if partial:
        outcome["partial_stage_observations"] = partial
    if supervision is not None and supervision.clean and supervision.exit_code == 0 and not supervision.timed_out:
        outcome["status"] = "incomplete"
        try:
            candidate_path = run_dir / "candidate.json"
            candidate = _read_json_capped(candidate_path, _MIB)
            final = _read_json_capped(run_dir / "final.json", 16384)
            _, candidate_hash = _file_identity(candidate_path, _MIB)
            if (candidate.get("schema") != "bert-gpu-calibration-candidate-v1"
                    or final.get("schema") != "bert-gpu-calibration-final-v1"
                    or any(record.get("run_id") != run_id
                           or record.get("supervisor_run_id") != supervision.run_id
                           or type(record.get("pid")) is not int or record["pid"] != supervision.pid
                           for record in (candidate, final))
                    or final.get("candidate_sha256") != candidate_hash
                    or type(final.get("active_memory_leases")) is not int
                    or final["active_memory_leases"] != 0):
                raise RuntimeError("owner result or final cleanup evidence does not match the reaped child")
            _validate_measurement(candidate["measurement"])
            _validate_stage_observations(partial, candidate["measurement"])
            outcome["measurement"] = candidate["measurement"]
            if (telemetry_error is None and supervision_evidence_error is None
                    and presence["owner_pid_listed"] is not True):
                outcome["status"] = "success"
        except Exception as exc:
            outcome["candidate_error_type"] = type(exc).__name__
    else:
        try:
            failure = _read_json_capped(run_dir / "failure.json", 16384)
            if failure.get("schema") == "bert-gpu-calibration-failure-v1" and failure.get("run_id") == run_id:
                outcome["owner_error_type"] = failure.get("error_type")
        except (OSError, ValueError):
            pass
    try:
        _verify_startup()
    except Exception as exc:
        outcome["controller_fresh_pycache"] = False
        outcome["controller_import_error_type"] = type(exc).__name__
        if outcome["status"] == "success":
            outcome["status"] = "incomplete"
    _write_json(destination, outcome)
    return outcome


def main(argv=None):
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", nargs=3, metavar=("RUN_DIR", "CACHE_DIR", "RUN_ID"), help=argparse.SUPPRESS)
    parser.add_argument("--workspace-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.owner:
        return owner_main(*args.owner)
    if not args.workspace_root or not args.cache_dir or not args.output:
        parser.error("--workspace-root, --cache-dir and --output are required")
    result = run_calibration(workspace_root=args.workspace_root, cache_dir=args.cache_dir,
                             output=args.output, site_packages=os.environ["LOOKING_GLASS_BERT_BOOTSTRAP_SITE"],
                             smi=shutil.which("nvidia-smi"))
    print(json.dumps({"status": result["status"], "source_commit": result["source_commit"]}))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
