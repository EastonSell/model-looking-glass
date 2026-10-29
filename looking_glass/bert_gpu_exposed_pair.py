"""Bounded exposed-pair DistilBERT CUDA intervention replication.

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
TARGET_IDS = (101, 1996, 5276, 3875, 1996, 3063, 103, 3201, 1012, 102)
DONOR_IDS = (101, 1996, 12499, 3875, 1996, 3063, 103, 3201, 1012, 102)
TARGET_TEXT = "The painter beside the artist [MASK] ready."
DONOR_TEXT = "The painters beside the artist [MASK] ready."
TARGET_ROW_ID = "evaluation-f01-A-s-s"
DONOR_ROW_ID = "evaluation-f01-A-p-s"
MANIFEST_SHA256 = "20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5"
EXPOSED_PAIR_ENABLED = True
# Objects whose ownership is uncertain after an interrupted call remain rooted
# until this single-use supervised process exits.
_OWNER_CUSTODY = {"entered": False, "session": None, "device_lease": None,
                  "budget": None, "memory_lease": None, "model": None,
                  "converted_model": None, "target": None, "donor": None,
                  "target_cpu": None, "donor_cpu": None, "observations": None,
                  "observation_custodies": None, "cap": None, "commitment": None,
                  "drain_failed": False, "failure": None, "diagnostic_failure": None}


def _workspace_root(repo, requested):
    """Use the common workspace namespace for every supervised device entry."""
    repo, requested = Path(repo), Path(requested)
    if not repo.is_absolute() or not requested.is_absolute():
        raise ValueError("the checkout and workspace root must be absolute")
    repo = repo.resolve(strict=True)
    if not repo.is_dir():
        raise ValueError("the checkout must be a directory")
    requested = requested.resolve(strict=True)
    work_parents = [parent for parent in repo.parents if parent.name.casefold() == "work"]
    if len(work_parents) != 1 or work_parents[0].name != "work":
        raise ValueError("checkout must have exactly one canonical work ancestor")
    work = work_parents[0]
    if (work.parent / "work").resolve(strict=True) != work:
        raise ValueError("work ancestor is not canonical")
    expected = work.parent
    if requested != expected:
        raise ValueError("workspace root does not match the shared checkout layout")
    return expected


def _validate_workspace_spec(session):
    spec = getattr(session, "_identity_spec", None)
    if type(spec) is not dict or type(spec.get("workspace_root")) is not str:
        raise RuntimeError("a supervised physical GPU identity is required")
    _workspace_root(Path(__file__).resolve().parents[1], spec["workspace_root"])


def _validate_workspace_binding(session):
    if session.identity is None:
        raise RuntimeError("a physical GPU identity is required")
    _workspace_root(Path(__file__).resolve().parents[1], session.identity.workspace_root)


def _verify_public_pair(repo):
    """Bind the fixed rows to the previously published public case manifest."""
    path = Path(repo) / "experiments" / "bert_agreement_public_cases_v1.json"
    with path.open("rb") as stream:
        raw = stream.read(_MIB + 1)
    if len(raw) > _MIB or hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError("public case manifest differs from the fixed identity")
    manifest = json.loads(raw)
    rows = {row.get("row_id"): row for row in manifest.get("rows", [])}
    for row_id, text, ids in ((TARGET_ROW_ID, TARGET_TEXT, TARGET_IDS),
                              (DONOR_ROW_ID, DONOR_TEXT, DONOR_IDS)):
        row = rows.get(row_id)
        if (type(row) is not dict or row.get("text") != text
                or row.get("input_ids") != list(ids) or row.get("attention_mask") != [1] * 10):
            raise RuntimeError("fixed public row differs from the manifest")
    return MANIFEST_SHA256


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
    for text, expected in ((TARGET_TEXT, TARGET_IDS), (DONOR_TEXT, DONOR_IDS)):
        encoded = tokenizer(text, add_special_tokens=True, return_attention_mask=True)
        if encoded["input_ids"] != list(expected) or encoded["attention_mask"] != [1] * 10:
            raise RuntimeError("public development input tokenization differs from the fixed case")


def _load_model(cache_dir, torch):
    """Stage the pinned local snapshot and load the model on CPU only."""
    from importlib.metadata import version
    import transformers

    if str(transformers.__version__) != "4.57.3":
        raise RuntimeError("unsupported Transformers version before checkpoint loading")
    cache = Path(cache_dir).resolve(strict=True)
    snapshot = cache / "models--distilbert--distilbert-base-uncased" / "snapshots" / REVISION
    with tempfile.TemporaryDirectory(prefix="bert-gpu-exposed-pair-stage-", dir=cache.parent) as directory:
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
    if (total != expected_total or free > total or free + reserved > total
            or allocated > reserved or reserved > total
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


class _ObservationCustody:
    """Preallocated roots retained by the owner until a verified drain."""
    __slots__ = ("entered", "input_ids", "source", "attention_mask", "plan",
                 "plan_sources", "capture_roots", "captures", "logits")

    def __init__(self):
        self.entered = False
        self.input_ids = self.source = self.attention_mask = self.plan = None
        self.plan_sources = ()
        self.capture_roots = self.captures = self.logits = None


class _RetainedPlanCaptures(dict):
    """Keep capture clones rooted when plan cleanup follows a failed drain."""
    def clear(self):
        return None


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
            check_owner=None, custody=None):
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
    if custody is None:
        custody = _ObservationCustody()
    if type(custody) is not _ObservationCustody or custody.entered:
        raise ValueError("a fresh observation custody object is required")
    custody.entered = True
    custody.input_ids, custody.source = input_ids, source

    def ownership():
        if check_owner is not None:
            check_owner()

    def guarded(function):
        ownership()
        result = function()
        ownership()
        return result

    def rooted(name, function):
        ownership()
        result = function()
        setattr(custody, name, result)
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
    # Root the plan before its constructor clones a replacement. Its cleanup
    # clears source references even after a failed live-output drain.
    ownership()
    plan = ActivationPlan.__new__(ActivationPlan)
    custody.plan = plan
    ActivationPlan.__init__(plan, model, steps, max_tensor_bytes=10 * hidden * 4,
                            max_capture_bytes=2 * hidden * 4, max_source_bytes=hidden * 4,
                            max_steps=3, max_invocations=1)
    custody.plan_sources = tuple(plan._sources.values())
    capture_roots = _RetainedPlanCaptures()
    custody.capture_roots = capture_roots
    plan._captures = capture_roots
    ownership()
    with torch.inference_mode(), plan:
        attention_mask = rooted("attention_mask", lambda: torch.ones_like(input_ids))
        ownership()
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_attentions=False, output_hidden_states=False,
                       return_dict=True).logits
        custody.logits = logits
        ownership()
        if (type(logits) is not torch.Tensor or logits.dtype != torch.float32
                or logits.device != input_ids.device or tuple(logits.shape) != (1, 10, vocab)
                or not guarded(lambda: bool(torch.isfinite(logits).all()))):
            raise RuntimeError("invalid or nonfinite complete model logits")
        ownership()
        mask_bits = guarded(lambda: _raw_float32(logits[0, 6]))
        ownership()
        completion("output_live")
    custody.logits = None
    del logits
    completion("output_disposed")
    captures, metadata = plan.captures, plan.metadata
    custody.captures = captures
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
                 check_owner=None, observation_custodies=None):
    """Measure a fixed capture/self/cross sequence while all captures remain live.

    The checkpoint callback synchronizes, samples memory and checks admission.
    It receives only the stage name and number of live observations. Returned
    evidence contains host bytes and metadata, never device tensor references.
    """
    if not callable(checkpoint):
        raise TypeError("a synchronized checkpoint callback is required")
    owner_retains = observation_custodies is not None
    if observation_custodies is None:
        observation_custodies = tuple(_ObservationCustody() for _ in range(4))
    if (type(observation_custodies) is not tuple or len(observation_custodies) != 4
            or any(type(item) is not _ObservationCustody for item in observation_custodies)):
        raise ValueError("four fresh observation custody objects are required")
    observations = []
    if owner_retains:
        _OWNER_CUSTODY["observations"] = observations
    checkpoint("before_target", len(observations))
    observations.append(observe(model, target_ids, is_id=is_id, are_id=are_id, synchronize=synchronize,
                                check_owner=check_owner, custody=observation_custodies[0]))
    checkpoint("after_target", len(observations))
    checkpoint("before_donor", len(observations))
    observations.append(observe(model, donor_ids, is_id=is_id, are_id=are_id, synchronize=synchronize,
                                check_owner=check_owner, custody=observation_custodies[1]))
    checkpoint("after_donor", len(observations))
    checkpoint("before_self", len(observations))
    observations.append(observe(model, target_ids, is_id=is_id, are_id=are_id, self_patch=True,
                                synchronize=synchronize, check_owner=check_owner,
                                custody=observation_custodies[2]))
    checkpoint("after_self", len(observations))
    checkpoint("before_cross", len(observations))
    observations.append(observe(model, target_ids, is_id=is_id, are_id=are_id,
                                source=observations[1].before, synchronize=synchronize,
                                check_owner=check_owner, custody=observation_custodies[3]))
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


def summarize_pair(sequence):
    """Decode saved bits into the fixed singular-target margin comparison."""
    rows = sequence.get("observations") if type(sequence) is dict else None
    if (type(rows) is not list or len(rows) != 4
            or [row.get("stage") if type(row) is dict else None for row in rows]
               != ["target", "donor", "self", "cross"]):
        raise ValueError("the fixed four observations are required")
    margins = {}
    for row in rows:
        bits = (row.get("is_bits"), row.get("are_bits"))
        if any(type(value) is not int or not 0 <= value <= 0xffffffff for value in bits):
            raise ValueError("invalid selected float32 bits")
        margins[row["stage"]] = math.fsum((_decode_float32(bits[0]), -_decode_float32(bits[1])))
    movement = math.fsum((margins["target"], -margins["cross"]))
    if not math.isfinite(movement):
        raise ValueError("nonfinite margin movement")
    target, cross = margins["target"], margins["cross"]
    return {"margins": margins, "target_minus_cross": movement,
            "strict_sign_change": (target > 0 and cross < 0) or (target < 0 and cross > 0)}


def _require_pristine_dependencies():
    if any(name in sys.modules for name in ("torch", "transformers", "numpy")):
        raise RuntimeError("CUDA-capable dependencies were imported before lease acquisition")


def _verify_startup():
    _verify_controller_import_mode()
    site = Path(os.environ.get("LOOKING_GLASS_PAIR_BOOTSTRAP_SITE", ""))
    if (not sys.flags.isolated or not sys.flags.no_site
            or os.environ.get("LOOKING_GLASS_PAIR_BOOTSTRAP_PID") != str(os.getpid())
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
    if _OWNER_CUSTODY["drain_failed"] is not False:
        raise RuntimeError("an unverified drain quarantined the owner")
    # Mark the drain unverified before entering CUDA, so an interruption while
    # a failed call unwinds cannot leave a falsely healthy owner.
    _OWNER_CUSTODY["drain_failed"] = True
    _guarded_call(session, lease, torch.cuda.synchronize, 0)
    _OWNER_CUSTODY["drain_failed"] = False
    free, total = _guarded_call(session, lease, torch.cuda.mem_get_info, 0)
    return {"device_free_bytes": free, "device_total_bytes": total,
            "allocated_bytes": _guarded_call(session, lease, torch.cuda.memory_allocated, 0),
            "reserved_bytes": _guarded_call(session, lease, torch.cuda.memory_reserved, 0),
            "peak_allocated_bytes": _guarded_call(session, lease, torch.cuda.max_memory_allocated, 0),
            "peak_reserved_bytes": _guarded_call(session, lease, torch.cuda.max_memory_reserved, 0)}


def _check_pair_custody(session, device_lease, *, gpu_claim=64 * 1024, host_claim=_MIB,
                        pending_drain=False):
    """Check the held commitment and claims under the trusted single-owner model."""
    if type(pending_drain) is not bool or _OWNER_CUSTODY["drain_failed"] is not pending_drain:
        raise RuntimeError("an unverified drain quarantined the owner")
    OwnerSession.check(session, device_lease)
    if (_OWNER_CUSTODY["session"] is not session
            or _OWNER_CUSTODY["device_lease"] is not device_lease):
        raise RuntimeError("supervised device custody differs")
    budget, lease = _OWNER_CUSTODY["budget"], _OWNER_CUSTODY["memory_lease"]
    cap, commitment = _OWNER_CUSTODY["cap"], _OWNER_CUSTODY["commitment"]
    if (type(budget) is not MemoryBudget or budget.active_leases() != (lease,)
            or type(cap) is not int or cap <= 0 or commitment != cap + _HEADROOM
            or lease.estimated_incremental_bytes != commitment
            or lease.trace_bytes("gpu") != gpu_claim or lease.trace_bytes("host") != host_claim):
        raise RuntimeError("owner budget, commitment or trace custody was lost")
    OwnerSession.check(session, device_lease)


def _synchronize_held(session, device_lease, torch):
    """Premark a drain and clear it only after a guarded successful sync."""
    _check_pair_custody(session, device_lease)
    _OWNER_CUSTODY["drain_failed"] = True
    _check_pair_custody(session, device_lease, pending_drain=True)
    torch.cuda.synchronize(0)
    _check_pair_custody(session, device_lease, pending_drain=True)
    _OWNER_CUSTODY["drain_failed"] = False


def measure_owned(session, lease, torch, cache_dir, *, record_stage):
    """Measure one bounded owner workload under an already held device lease."""
    if EXPOSED_PAIR_ENABLED is not True:
        raise RuntimeError("exposed-pair GPU execution is disabled pending source review")
    if _OWNER_CUSTODY["drain_failed"] is not False:
        raise RuntimeError("an unverified drain quarantined the owner")
    if type(session) is not OwnerSession or type(lease) is not DeviceLease:
        raise ValueError("an actual supervised owner and device lease are required")
    if not callable(record_stage):
        raise TypeError("a stage recorder is required")
    OwnerSession.check(session, lease)
    _validate_workspace_binding(session)
    if (getattr(torch, "__version__", None) != "2.8.0+cu128"
            or getattr(getattr(torch, "version", None), "cuda", None) != "12.8"
            or not callable(getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None))):
        raise RuntimeError("unsupported PyTorch or CUDA build before device work")
    if torch.cuda.is_initialized() is not False:
        raise RuntimeError("CUDA was initialized before the owner check")
    count = _guarded_call(session, lease, torch.cuda.device_count)
    if type(count) is not int or count != 1:
        raise RuntimeError("exactly one CUDA-visible device is required")
    properties = _guarded_call(session, lease, torch.cuda.get_device_properties, 0)
    runtime_uuid = _runtime_uuid(torch, properties.uuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid, device_count=count)
    if (properties.name != "NVIDIA GeForce RTX 2070 SUPER"
            or properties.major != 7 or properties.minor != 5):
        raise RuntimeError("CUDA device name or capability differs from the fixed contract")
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
    _OWNER_CUSTODY["cap"], _OWNER_CUSTODY["commitment"] = cap, commitment

    def telemetry():
        free, observed_total = _guarded_call(session, lease, torch.cuda.mem_get_info, 0)
        if observed_total != total:
            raise RuntimeError("CUDA telemetry total changed during admission")
        return CudaMemorySnapshot(free, observed_total,
                                  _guarded_call(session, lease, torch.cuda.memory_allocated, 0),
                                  _guarded_call(session, lease, torch.cuda.memory_reserved, 0))

    budget, memory_lease = reserve_owner_budget(telemetry, cap=cap, custody=_OWNER_CUSTODY)

    def owner_check():
        _check_pair_custody(session, lease)

    def guarded(function, *args):
        owner_check()
        result = function(*args)
        owner_check()
        return result

    def rooted(name, function, *args):
        owner_check()
        result = function(*args)
        _OWNER_CUSTODY[name] = result
        owner_check()
        return result

    owner_check()
    if (torch.__future__.get_overwrite_module_params_on_conversion() is not False
            or torch.__future__.get_swap_module_params_on_conversion() is not False):
        raise RuntimeError("unsupported module conversion flags")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(17)
    guarded(torch.cuda.manual_seed, 17)
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
        synchronize(stage)
        free, observed_total = guarded(torch.cuda.mem_get_info, 0)
        value = {"device_free_bytes": free, "device_total_bytes": observed_total,
                 "allocated_bytes": guarded(torch.cuda.memory_allocated, 0),
                 "reserved_bytes": guarded(torch.cuda.memory_reserved, 0),
                 "peak_allocated_bytes": guarded(torch.cuda.max_memory_allocated, 0),
                 "peak_reserved_bytes": guarded(torch.cuda.max_memory_reserved, 0)}
        record_stage(stage, {"memory": value, "observations_live": count, "peak_scope": _PEAK_SCOPE})
        check_memory(value, expected_total=total, cap=cap, commitment=commitment)
        return value

    def synchronize(stage):
        _synchronize_held(session, lease, torch)

    model, metadata = _load_model(cache_dir, torch)
    _OWNER_CUSTODY["model"] = model
    owner_check()
    snapshot("before_model_copy")
    converted = rooted("converted_model", model.to, "cuda:0")
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
    _OWNER_CUSTODY["target_cpu"], _OWNER_CUSTODY["donor_cpu"] = target_cpu, donor_cpu
    target = rooted("target", target_cpu.to, "cuda:0")
    donor = rooted("donor", donor_cpu.to, "cuda:0")
    if any(t.device.type != "cuda" or t.device.index != 0 for t in (target, donor)):
        raise RuntimeError("development inputs were not transferred to cuda:0")
    custodies = tuple(_ObservationCustody() for _ in range(4))
    _OWNER_CUSTODY["observation_custodies"] = custodies
    sequence = run_sequence(model, target, donor, is_id=2003, are_id=2024,
                            checkpoint=snapshot, synchronize=synchronize,
                            check_owner=owner_check, observation_custodies=custodies)
    # All prior observation and plan roots remain live until this final drain.
    synchronize("sequence_live")
    _OWNER_CUSTODY["observations"] = _OWNER_CUSTODY["observation_custodies"] = None
    del custodies
    references = {"model": weakref.ref(model), "target": weakref.ref(target),
                  "donor": weakref.ref(donor)}
    _OWNER_CUSTODY["converted_model"] = _OWNER_CUSTODY["model"] = None
    _OWNER_CUSTODY["target"] = _OWNER_CUSTODY["donor"] = None
    _OWNER_CUSTODY["target_cpu"] = _OWNER_CUSTODY["donor_cpu"] = None
    del converted, model, target, donor, target_cpu, donor_cpu
    gc.collect()
    survivors = {name: ref() is not None for name, ref in references.items()}
    record_stage("cleanup_weakrefs", survivors)
    synchronize("after_disposal")
    guarded(torch.cuda.empty_cache)
    snapshot("cleanup_before_workspace_clear")
    if (torch.__version__ != "2.8.0+cu128"
            or not callable(getattr(torch._C, "_cuda_clearCublasWorkspaces", None))):
        raise RuntimeError("unsupported PyTorch cuBLAS workspace cleanup API")
    guarded(torch._C._cuda_clearCublasWorkspaces)
    synchronize("after_workspace_clear")
    guarded(torch.cuda.empty_cache)
    cleanup = snapshot("after_cleanup")
    if any(survivors.values()) or cleanup["allocated_bytes"] != 0 or cleanup["reserved_bytes"] != 0:
        raise RuntimeError("owner references or CUDA allocator bytes remain after disposal")
    active = len(budget.active_leases())
    if active != 1:
        raise RuntimeError("whole-owner commitment was not retained through cleanup")
    return {"target_row_id": TARGET_ROW_ID, "donor_row_id": DONOR_ROW_ID,
            "manifest_sha256": MANIFEST_SHA256,
            "target_ids": list(TARGET_IDS), "donor_ids": list(DONOR_IDS),
            "output_token_ids": {"is": 2003, "are": 2024}, "seed": 17,
            "sequence": sequence, "pair_analysis": summarize_pair(sequence),
            "cleanup": cleanup, "cleanup_weakrefs": survivors,
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


def _release_owner_budget_after_candidate(session, device_lease):
    """Release the claims only after result serialization and local disposal."""
    _check_pair_custody(session, device_lease)
    budget, memory_lease = _OWNER_CUSTODY["budget"], _OWNER_CUSTODY["memory_lease"]
    memory_lease.release_trace("gpu", 64 * 1024)
    _check_pair_custody(session, device_lease, gpu_claim=0)
    memory_lease.release_trace("host", _MIB)
    _check_pair_custody(session, device_lease, gpu_claim=0, host_claim=0)
    memory_lease.release()
    OwnerSession.check(session, device_lease)
    if budget.active_leases():
        raise RuntimeError("memory commitments remain after candidate disposal")


def owner_main(run_dir, cache_dir, run_id, source_commit, git_exe):
    """A direct child retains its shared device lease until process exit."""
    try:
        if EXPOSED_PAIR_ENABLED is not True:
            raise RuntimeError("exposed-pair GPU execution is disabled pending source review")
        if _OWNER_CUSTODY["entered"] is not False:
            raise RuntimeError("owner is single-use")
        _OWNER_CUSTODY["entered"] = True
        _verify_startup()
        _require_pristine_dependencies()
        session = OwnerSession.from_environment()
        _OWNER_CUSTODY["session"] = session
        _validate_workspace_spec(session)
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _OWNER_CUSTODY["device_lease"] = lease
        OwnerSession.check(session, lease)
        _validate_workspace_binding(session)
        repo = Path(__file__).resolve().parents[1]
        actual_commit, _ = _source_identity(repo, git_exe)
        if actual_commit != source_commit:
            raise RuntimeError("owner source commit differs from the controller")
        _verify_public_pair(repo)
        _write_json(Path(run_dir) / "owner.json", {"schema": "bert-gpu-exposed-pair-owner-v1",
                    "run_id": run_id, "supervisor_run_id": session.run_id, "pid": os.getpid(),
                    "source_commit": source_commit})
        torch = _import_torch()
        result = measure_owned(session, lease, torch, cache_dir,
                               record_stage=lambda name, value: _write_json(
                                   Path(run_dir) / (name + ".json"), value))
        _check_pair_custody(session, lease)
        _write_json(Path(run_dir) / "candidate.json", {"schema": "bert-gpu-exposed-pair-candidate-v1",
                    "run_id": run_id, "supervisor_run_id": session.run_id,
                    "pid": os.getpid(), "source_commit": source_commit, "measurement": result})
        _check_pair_custody(session, lease)
        _, candidate_hash = _file_identity(Path(run_dir) / "candidate.json", _MIB)
        _check_pair_custody(session, lease)
        del result
        gc.collect()
        _release_owner_budget_after_candidate(session, lease)
        OwnerSession.check(session, lease)
        _write_json(Path(run_dir) / "final.json", {"schema": "bert-gpu-exposed-pair-final-v1",
                    "run_id": run_id, "supervisor_run_id": session.run_id, "pid": os.getpid(),
                    "source_commit": source_commit, "candidate_sha256": candidate_hash,
                    "active_memory_leases": 0})
        OwnerSession.check(session, lease)
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as exc:
        if _OWNER_CUSTODY["failure"] is None:
            _OWNER_CUSTODY["failure"] = exc
        # Never attest an unverified cleanup; the supervisor's marker remains.
        try:
            _write_json(Path(run_dir) / "failure.json", {"schema": "bert-gpu-exposed-pair-failure-v1",
                        "run_id": run_id, "status": "quarantined", "error_type": type(exc).__name__,
                        "private_traceback": traceback.format_exc()[-8192:]})
        except BaseException as diagnostic_error:
            _OWNER_CUSTODY["diagnostic_failure"] = diagnostic_error
            if not isinstance(diagnostic_error, Exception):
                raise
        if isinstance(exc, Exception):
            return 1
        raise


_SOURCE_PATHS = ("looking_glass/__init__.py", "looking_glass/bert_gpu_exposed_pair.py",
                 "looking_glass/bert_gpu_evidence.py", "docs/BERT_GPU_DIAGNOSTIC_WRITER_V2.md",
                 "experiments/bert_agreement_public_cases_v1.json",
                 "looking_glass/gpu_oneshot.py", "looking_glass/device_identity.py",
                 "looking_glass/device_lease.py", "looking_glass/supervisor.py",
                 "looking_glass/memory_budget.py", "looking_glass/intervention_plan.py",
                 "looking_glass/interventions.py", "looking_glass/model_ownership.py",
                 "docs/BERT_GPU_EXPOSED_PAIR_V1.md", "scripts/bert_gpu_exposed_pair_bootstrap.py")


def _source_identity(repo, git_exe):
    git = Path(git_exe)
    if not git.is_absolute() or not git.resolve(strict=True).is_file():
        raise ValueError("an absolute Git executable is required")
    git = str(git.resolve(strict=True))
    environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT")
                   if key in os.environ}
    environment.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                        "GIT_CONFIG_COUNT": "0", "GIT_NO_REPLACE_OBJECTS": "1",
                        "GIT_TERMINAL_PROMPT": "0"})
    def git_run(*args, text=False):
        return subprocess.run([git, "--no-replace-objects", *args], cwd=repo,
                              env=environment, capture_output=True, text=text, check=True, timeout=10).stdout
    commit = git_run("rev-parse", "HEAD", text=True).strip()
    dirty = git_run("status", "--porcelain", "--untracked-files=no", text=True)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or dirty:
        raise RuntimeError("calibration requires a clean committed checkout")
    hashes = {}
    for relative in _SOURCE_PATHS:
        stored = git_run("show", commit + ":" + relative)
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
    canonical = lambda data: json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(value.get("pair_analysis")) != canonical(summarize_pair(sequence)):
        raise ValueError("saved pair analysis differs from selected float32 bits")
    if (value.get("target_row_id") != TARGET_ROW_ID or value.get("donor_row_id") != DONOR_ROW_ID
            or value.get("manifest_sha256") != MANIFEST_SHA256
            or value.get("target_ids") != list(TARGET_IDS) or value.get("donor_ids") != list(DONOR_IDS)
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


_PUBLIC_MEMORY_FIELDS = ("device_free_bytes", "device_total_bytes", "allocated_bytes", "reserved_bytes",
                         "peak_allocated_bytes", "peak_reserved_bytes")


def _public_memory(value):
    if type(value) is not dict or any(type(value.get(key)) is not int or value[key] < 0
                                      for key in _PUBLIC_MEMORY_FIELDS):
        raise ValueError("invalid public memory observation")
    return {key: value[key] for key in _PUBLIC_MEMORY_FIELDS}


def _public_measurement(value):
    """Copy only the fixed validated measurement fields into a publishable result."""
    _validate_measurement(value)
    scalar = ("target_row_id", "donor_row_id", "manifest_sha256", "target_ids", "donor_ids",
              "output_token_ids", "seed", "budget_active_before_candidate_write", "runtime_uuid_verified",
              "device_name", "device_total_bytes", "compute_capability", "allocator_fraction",
              "allocator_cap_bytes", "allocator_backend", "whole_run_commitment_bytes",
              "gpu_trace_claim_bytes", "host_trace_claim_bytes", "torch_version", "cuda_build",
              "execution_config", "peak_scope")
    result = {key: value[key] for key in scalar}
    versions = (value["python_version"], value["model_metadata"]["tokenizers_version"],
                value["model_metadata"]["safetensors_version"])
    if any(type(version) is not str or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version) is None
           for version in versions):
        raise ValueError("invalid runtime version in public measurement")
    result["python_version"] = versions[0]
    metadata_fields = ("model", "revision", "assets", "model_class", "tokenizer_class",
                       "transformers_version", "tokenizers_version", "safetensors_version",
                       "attention_implementation", "parameter_count", "cpu_tied_weight_verified")
    result["model_metadata"] = {key: value["model_metadata"][key] for key in metadata_fields}
    sequence_fields = ("forward_count", "self_mask_logits_bitwise_equal", "cross_source_landed",
                       "source_differs_from_target")
    row_fields = ("stage", "is_bits", "are_bits", "is_logit", "are_logit", "mask_logits_sha256",
                  "before_float32_le_base64", "after_float32_le_base64", "events")
    sequence = {key: value["sequence"][key] for key in sequence_fields}
    sequence["observations"] = [{key: row[key] for key in row_fields}
                                for row in value["sequence"]["observations"]]
    result["sequence"] = sequence
    result["pair_analysis"] = summarize_pair(sequence)
    result["cleanup"] = _public_memory(value["cleanup"])
    result["cleanup_weakrefs"] = {key: value["cleanup_weakrefs"][key] for key in ("model", "target", "donor")}
    cleanup = "PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces"
    if value.get("workspace_cleanup") != cleanup:
        raise ValueError("missing or mismatched workspace cleanup evidence")
    result["workspace_cleanup"] = cleanup
    return result


def _public_stage_observations(records):
    """Keep only bounded, typed fields from available partial stage records."""
    result = {}
    for name, count in _STAGE_COUNTS:
        entry = records.get(name)
        if (type(entry) is not dict or not {"memory", "observations_live", "peak_scope"}.issubset(entry)
                or entry.get("peak_scope") != _PEAK_SCOPE
                or type(entry.get("observations_live")) is not type(count)
                or entry.get("observations_live") != count):
            continue
        try:
            memory = _public_memory(entry.get("memory"))
        except ValueError:
            continue
        result[name] = {"memory": memory, "observations_live": count, "peak_scope": _PEAK_SCOPE}
    weakrefs = records.get("cleanup_weakrefs")
    if (type(weakrefs) is dict and all(type(weakrefs.get(key)) is bool for key in ("model", "target", "donor"))):
        result["cleanup_weakrefs"] = {key: weakrefs[key] for key in ("model", "target", "donor")}
    return result


def _public_smi_snapshot(value):
    if type(value) is not dict:
        return None
    result = {}
    for key in ("total_bytes", "free_bytes", "used_bytes"):
        if type(value.get(key)) is int and value[key] >= 0:
            result[key] = value[key]
    if value.get("device_name") == "NVIDIA GeForce RTX 2070 SUPER":
        result["device_name"] = value["device_name"]
    version = value.get("driver_version")
    if type(version) is str and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version):
        result["driver_version"] = version
    timestamp = value.get("timestamp_utc")
    if type(timestamp) is str and re.fullmatch(r"[0-9TZ:+.\-]{10,40}", timestamp):
        result["timestamp_utc"] = timestamp
    return result


def _public_presence(value):
    if type(value) is dict:
        query, listed = value.get("query"), value.get("owner_pid_listed")
        if query == "ok" and type(listed) is bool:
            return {"query": "ok", "owner_pid_listed": listed}
        if query in ("unavailable", "inconclusive", "not_attempted") and listed is None:
            return {"query": query, "owner_pid_listed": None}
    return {"query": "inconclusive", "owner_pid_listed": None}


def _public_error_type(exc):
    name = type(exc).__name__
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name) else "UnknownError"


def _public_supervisor(value):
    if value is None:
        return None
    return {"exit_code": value.exit_code if type(value.exit_code) is int else None,
            "clean": value.clean if type(value.clean) is bool else False,
            "timed_out": value.timed_out if type(value.timed_out) is bool else True}


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
    if EXPOSED_PAIR_ENABLED is not True:
        raise RuntimeError("exposed-pair GPU execution is disabled pending source review")
    _verify_startup()
    repo = Path(__file__).resolve().parents[1]
    root = _workspace_root(repo, workspace_root)
    cache = Path(cache_dir).resolve(strict=True)
    site = Path(site_packages).resolve(strict=True)
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
        raise ValueError("output must be an absolute, new file")
    with tempfile.NamedTemporaryFile(prefix=".bert-calibration-write-test-", dir=destination.parent):
        pass
    git_exe = os.environ.get("LOOKING_GLASS_PAIR_GIT_EXE", "")
    bootstrap_commit = os.environ.get("LOOKING_GLASS_PAIR_SOURCE_COMMIT", "")
    commit, source_hashes = _source_identity(repo, git_exe)
    if commit != bootstrap_commit:
        raise RuntimeError("bootstrap and controller source commits differ")
    _verify_public_pair(repo)
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
    run_dir = Path(tempfile.mkdtemp(prefix="bert-gpu-exposed-pair-", dir=root / "work"))
    prefix = Path(tempfile.mkdtemp(prefix="pycache-", dir=run_dir))
    environment["PYTHONPYCACHEPREFIX"] = str(prefix)
    run_id = secrets.token_hex(16)
    _write_json(run_dir / "started.json", {"schema": "bert-gpu-exposed-pair-start-v1", "run_id": run_id,
               "source_commit": commit, "started_utc": before["timestamp_utc"]})
    bootstrap = repo / "scripts" / "bert_gpu_exposed_pair_bootstrap.py"
    command = [sys._base_executable, "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
               str(bootstrap), "--bootstrap-site", str(site), "--source-commit", commit,
               "--git-executable", git_exe, "--owner", str(run_dir), str(cache),
               run_id, commit, git_exe]
    supervision = None
    supervisor_error = None
    try:
        supervision = run_supervised(command, device_lease_path=identity.lock_path,
                                     marker_path=identity.marker_path, identity=identity,
                                     timeout=300, lease_timeout=30, cwd=repo, env=environment,
                                     inherit_environment=False, require_zero_exit=True)
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        supervisor_error = _public_error_type(exc)
    supervision_evidence_error = None
    if supervision is not None:
        try:
            _write_json(run_dir / "supervision.json", {"run_id": run_id,
                        "supervisor_run_id": supervision.run_id, "pid": supervision.pid,
                        "exit_code": supervision.exit_code, "clean": supervision.clean,
                        "timed_out": supervision.timed_out, "reason": supervision.reason})
        except Exception as exc:
            supervision_evidence_error = _public_error_type(exc)
    after = None
    telemetry_error = None
    try:
        after_uuid, after = _smi_snapshot(smi)
        if after_uuid != gpu_uuid:
            raise RuntimeError("physical inventory changed after owner exit")
    except Exception as exc:
        after = None
        telemetry_error = _public_error_type(exc)
    presence = (_owner_presence_after_exit(smi, supervision.pid) if supervision is not None
                else {"query": "not_attempted", "owner_pid_listed": None})
    presence = _public_presence(presence)
    outcome = {"schema": "bert-gpu-exposed-pair-result-v1", "run_id": run_id, "source_commit": commit,
               "source_sha256": source_hashes, "controller_fresh_pycache": True,
               "status": "quarantined", "supervisor": _public_supervisor(supervision),
               "supervisor_error_type": supervisor_error,
               "supervision_evidence_error_type": supervision_evidence_error,
               "nvidia_smi_before": _public_smi_snapshot(before), "nvidia_smi_after": _public_smi_snapshot(after),
               "owner_presence_after_exit": presence, "post_telemetry_error_type": telemetry_error,
               "limitations": ["one previously exposed public pair; no held-out or lexical-control claim",
                   "CPU and GPU results need not be bit-identical",
                   "full mask logits are retained only as hashes and selected bits",
                   "no latency, throughput or worst-case memory claim",
                   "the allocator cap excludes CUDA context and non-PyTorch allocations",
                   "the logical host-trace allowance is not a bound on total host RAM",
                   "point-in-time device memory observations can miss external pressure",
                   "partial stage records in incomplete outcomes are typed but not admission-validated",
                   "the cooperative lease does not control unrelated GPU applications",
                   "supervision reaps the direct child and assumes no CUDA-using descendants",
                   "startup source hashes assume a trusted, quiescent checkout throughout the run",
                   "dependency and bytecode paths are trusted; interpreter and native libraries are not content-attested",
                   "Windows WDDM may not expose all GPU processes or their memory"]}
    partial = _read_stage_observations(run_dir)
    public_partial = _public_stage_observations(partial)
    if public_partial:
        outcome["partial_stage_observations"] = public_partial
    if (supervision is not None and supervision.clean is True and type(supervision.exit_code) is int
            and supervision.exit_code == 0 and supervision.timed_out is False):
        outcome["status"] = "incomplete"
        try:
            candidate_path = run_dir / "candidate.json"
            candidate = _read_json_capped(candidate_path, _MIB)
            final = _read_json_capped(run_dir / "final.json", 16384)
            _, candidate_hash = _file_identity(candidate_path, _MIB)
            if (candidate.get("schema") != "bert-gpu-exposed-pair-candidate-v1"
                    or final.get("schema") != "bert-gpu-exposed-pair-final-v1"
                    or any(record.get("run_id") != run_id
                           or record.get("supervisor_run_id") != supervision.run_id
                           or record.get("source_commit") != commit
                           or type(record.get("pid")) is not int or record["pid"] != supervision.pid
                           for record in (candidate, final))
                    or final.get("candidate_sha256") != candidate_hash
                    or type(final.get("active_memory_leases")) is not int
                    or final["active_memory_leases"] != 0):
                raise RuntimeError("owner result or final cleanup evidence does not match the reaped child")
            public_measurement = _public_measurement(candidate["measurement"])
            _validate_stage_observations(partial, candidate["measurement"])
            outcome["measurement"] = public_measurement
            if (telemetry_error is None and supervision_evidence_error is None
                    and presence.get("query") == "ok" and presence.get("owner_pid_listed") is False):
                outcome["status"] = "success"
        except Exception as exc:
            outcome["candidate_error_type"] = _public_error_type(exc)
    else:
        try:
            failure = _read_json_capped(run_dir / "failure.json", 16384)
            error_type = failure.get("error_type")
            if (failure.get("schema") == "bert-gpu-exposed-pair-failure-v1" and failure.get("run_id") == run_id
                    and type(error_type) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type)):
                outcome["owner_error_type"] = error_type
        except (OSError, ValueError):
            pass
    try:
        _verify_startup()
    except Exception as exc:
        outcome["controller_fresh_pycache"] = False
        outcome["controller_import_error_type"] = _public_error_type(exc)
        if outcome["status"] == "success":
            outcome["status"] = "incomplete"
    _write_json(destination, outcome)
    return outcome


def main(argv=None):
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", nargs=5,
                        metavar=("RUN_DIR", "CACHE_DIR", "RUN_ID", "SOURCE_COMMIT", "GIT_EXE"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--workspace-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.owner:
        return owner_main(*args.owner)
    if not args.workspace_root or not args.cache_dir or not args.output:
        parser.error("--workspace-root, --cache-dir and --output are required")
    result = run_calibration(workspace_root=args.workspace_root, cache_dir=args.cache_dir,
                             output=args.output, site_packages=os.environ["LOOKING_GLASS_PAIR_BOOTSTRAP_SITE"],
                             smi=shutil.which("nvidia-smi"))
    print(json.dumps({"status": result["status"], "source_commit": result["source_commit"]}))
    return 0 if result["status"] == "success" else 1


def _owner_process_main(argv=None):
    """Exit the direct owner without Python shutdown hooks after any outcome."""
    status = 1
    try:
        status = main(argv)
    finally:
        os._exit(status if type(status) is int and 0 <= status <= 255 else 1)


if __name__ == "__main__":
    if "--owner" in sys.argv[1:]:
        _owner_process_main()
    else:
        raise SystemExit(main())
