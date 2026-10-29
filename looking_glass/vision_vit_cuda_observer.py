"""Single-use, supervised CUDA observer for one previously exposed ViT pair."""

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import struct
import subprocess
import sys
import tempfile
import traceback
import weakref

from .bert_gpu_evidence import write_json_new
from .bert_gpu_exposed_pair import (_file_identity, _unexpected_hooks, _workspace_root,
    _validate_workspace_spec, _validate_workspace_binding, check_memory,
    _RetainedPlanCaptures, _public_smi_snapshot, _public_presence, _public_supervisor)
from .device_identity import resolve_device_identity
from .device_lease import DeviceLease
from .gpu_oneshot import (_runtime_uuid, _smi_snapshot, _owner_presence_after_exit,
                          _verify_controller_import_mode)
from .memory_budget import CudaMemorySnapshot, MemoryBudget
from .supervisor import OwnerSession, run_supervised

MIB = 1024 ** 2
HEADROOM = 256 * MIB
TRACE = 16 * MIB
MANIFEST_SHA = "8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207"
REFERENCE_SHA = "3df19e35e8550f456015f3af3a9ce5536d0387fbf19cca7c58f6dd25fa975fa0"
STATE_SHA = "d1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd"
TOKENS = (161, 162, 175, 176)
SITE = "vit.encoder.layer.2"
LABELS = ("cat-clean", "dog-clean", "cat-self", "cat-gpu-donor", "cat-cpu-donor", "cat-repeat")
_SOURCE_PATHS = ("looking_glass/__init__.py", "looking_glass/vision_vit_cuda_observer.py",
    "looking_glass/bert_gpu_exposed_pair.py", "looking_glass/bert_gpu_evidence.py",
    "looking_glass/gpu_oneshot.py", "looking_glass/device_identity.py", "looking_glass/device_lease.py",
    "looking_glass/supervisor.py", "looking_glass/memory_budget.py", "looking_glass/intervention_plan.py",
    "looking_glass/interventions.py", "looking_glass/model_ownership.py", "looking_glass/vision_coco_study.py",
    "experiments/vision_coco_spatial_v1.json", "docs/VISION_VIT_CUDA_OBSERVER_V1.md",
    "tools/prepare_vision_cuda_reference.py", "scripts/vision_vit_cuda_bootstrap.py")

# Preallocated roots are never released after an unverified CUDA operation.
_ROOT = {"entered": False, "pending": False, "session": None, "lease": None,
         "budget": None, "memory_lease": None, "model": None, "pixels": None,
         "observation": None, "submission": 0, "label": None,
         "failure": None, "diagnostic_failure": None}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _load_reference(directory):
    directory = Path(directory).resolve(strict=True)
    raw = (directory / "reference.json").read_bytes()
    if _sha(raw) != REFERENCE_SHA:
        raise ValueError("reference manifest does not match the fixed CPU reference")
    ref = json.loads(raw)
    if (ref.get("status") != "complete" or ref.get("model_forwards") != 0
            or ref.get("case_sha256") != MANIFEST_SHA or ref.get("tokens") != list(TOKENS)):
        raise ValueError("reference metadata differs")
    names = {"cat-pixels.f32", "dog-pixels.f32"} | {
        f"cpu-forward-{n:03d}.{ext}" for n in (1, 2, 24) for ext in ("json", "bin")}
    if set(ref["files"]) != names:
        raise ValueError("reference inventory differs")
    files = {}
    for name in sorted(names):
        item = ref["files"][name]
        if type(item.get("bytes")) is not int or not 0 < item["bytes"] <= MIB:
            raise ValueError("reference file exceeds its bound")
        data = (directory / name).read_bytes()
        if len(data) != item["bytes"] or _sha(data) != item["sha256"]:
            raise ValueError("reference bytes differ")
        files[name] = data
    donor = files["cpu-forward-002.bin"]
    requested = files["cpu-forward-024.bin"][312736:315808]
    selected = b"".join(donor[4000 + token * 768:4000 + (token + 1) * 768] for token in TOKENS)
    if len(requested) != 3072 or selected != requested:
        raise ValueError("saved CPU donor does not equal the original requested intervention")
    return files


def _source_identity(repo, git):
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT")
           if key in os.environ}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_COUNT": "0", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0"})
    def run(*args):
        return subprocess.run([str(git), "--no-replace-objects", *args], cwd=repo, env=env,
                              capture_output=True, check=True, timeout=15).stdout
    commit = run("rev-parse", "HEAD").decode("ascii").strip()
    if len(commit) != 40 or run("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("a clean committed source checkout is required")
    hashes = {}
    for name in _SOURCE_PATHS:
        raw = (repo / name).read_bytes()
        if len(raw) > 4 * MIB or raw != run("show", commit + ":" + name):
            raise RuntimeError("source bytes differ from the committed checkout")
        hashes[name] = _sha(raw)
    if hashes["experiments/vision_coco_spatial_v1.json"] != MANIFEST_SHA:
        raise RuntimeError("public case manifest differs")
    return commit, hashes


def _asset_inventory(repo, assets):
    manifest = json.loads((repo / "experiments/vision_coco_spatial_v1.json").read_bytes())
    expected = manifest["model_files"]
    if set(expected) != {"config.json", "preprocessor_config.json", "model.safetensors"}:
        raise ValueError("model asset inventory differs")
    for name, digest in expected.items():
        if _file_identity(Path(assets) / name, 32 * MIB)[1] != digest:
            raise ValueError("model asset differs from fixed manifest")
    return expected


def _bits(tensor):
    import torch
    if (type(tensor) is not torch.Tensor or tensor.dtype != torch.float32 or sys.byteorder != "little"
            or tensor.numel() > 197 * 192):
        raise ValueError("unsupported captured tensor")
    return tensor.detach().view(torch.int32).to("cpu").contiguous().numpy().tobytes()


def _write_bytes(path, data):
    if len(data) > MIB:
        raise ValueError("per-forward evidence exceeds one MiB")
    with Path(path).open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError("short evidence write")
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(data)


def _diff(a, b):
    if len(a) != len(b) or len(a) % 4:
        raise ValueError("comparison geometry differs")
    left = struct.iter_unpack("<f", a)
    right = struct.iter_unpack("<f", b)
    differences = [abs(x[0] - y[0]) for x, y in zip(left, right)]
    if not differences or any(not math.isfinite(x) for x in differences):
        raise ValueError("nonfinite comparison")
    return {"max_abs": max(differences), "mean_abs": math.fsum(differences) / len(differences),
            "count": len(differences)}


def _margin(logits):
    values = [x[0] for x in struct.iter_unpack("<f", logits)]
    if len(values) != 1000 or any(not math.isfinite(x) for x in values):
        raise ValueError("nonfinite or incomplete logits")
    def shifted(group):
        maximum = max(group)
        return maximum, math.log(math.fsum(math.exp(v - maximum) for v in group))
    dog, dog_sum = shifted(values[151:269])
    cat, cat_sum = shifted(values[281:286])
    return (dog - cat) + dog_sum - cat_sum


def _descriptive(tensors, reference, rows):
    cat, dog, _self_obs, gpu_patch, cpu_patch, _repeat = tensors
    cpu_cat = reference["cpu-forward-001.bin"]
    cpu_dog = reference["cpu-forward-002.bin"]
    cpu_patched = reference["cpu-forward-024.bin"]
    comparisons = {
        "cpu_gpu_cat_logits": _diff(cpu_cat[:4000], cat["logits"]),
        "cpu_gpu_dog_logits": _diff(cpu_dog[:4000], dog["logits"]),
        "cpu_gpu_cat_early": _diff(cpu_cat[4000:155296], cat["before_all"]),
        "cpu_gpu_dog_early": _diff(cpu_dog[4000:155296], dog["before_all"]),
        "cpu_gpu_cpu_donor_logits": _diff(cpu_patched[:4000], cpu_patch["logits"]),
        "gpu_native_vs_cpu_donor_logits": _diff(gpu_patch["logits"], cpu_patch["logits"]),
        "cpu_gpu_donor_tokens": _diff(cpu_patched[312736:315808], gpu_patch["requested"])}
    margins = {row["label"]: row["dog_minus_cat_log_odds"] for row in rows}
    margins.update({"cpu_cat": _margin(cpu_cat[:4000]), "cpu_dog": _margin(cpu_dog[:4000]),
                    "cpu_cat_cpu_donor": _margin(cpu_patched[:4000])})
    return comparisons, margins


class ObservationCustody:
    __slots__ = ("plan", "sources", "captures", "output", "pixels", "replacement")
    def __init__(self):
        self.plan = self.sources = self.captures = self.output = self.pixels = self.replacement = None


def observe(model, pixels, *, label, output, index, replacement=None, self_patch=False,
            synchronize=lambda: None, check=lambda: None, custody=None):
    """Run one bounded same-invocation observation; callbacks own CUDA lifetime."""
    import torch
    from .intervention_plan import ActivationPlan

    if label not in LABELS or type(index) is not int or not 1 <= index <= 6:
        raise ValueError("unexpected forward label or index")
    if type(self_patch) is not bool or (self_patch and replacement is not None):
        raise ValueError("invalid patch mode")
    custody = ObservationCustody() if custody is None else custody
    if type(custody) is not ObservationCustody:
        raise TypeError("invalid observation custody")
    custody.pixels, custody.replacement = pixels, replacement
    patch = self_patch or replacement is not None
    steps = [{"name": "before_all", "operation": "capture", "module_path": SITE}]
    if patch:
        for i, token in enumerate(TOKENS):
            site = {"module_path": SITE, "batch_index": 0, "token_index": token}
            steps.append(dict(name=f"before_{i}", operation="capture", **site))
            action = dict(name=f"action_{i}", operation="patch", **site)
            action["source_capture" if self_patch else "value"] = f"before_{i}" if self_patch else replacement[i]
            steps.extend((action, dict(name=f"after_{i}", operation="capture", **site)))
        steps.append({"name": "after_all", "operation": "capture", "module_path": SITE})
    if _unexpected_hooks(model):
        raise RuntimeError("unexpected hooks before observation")
    check()
    plan = ActivationPlan.__new__(ActivationPlan)
    custody.plan = plan
    ActivationPlan.__init__(plan, model, steps, max_tensor_bytes=197 * 192 * 4,
                            max_capture_bytes=400000, max_source_bytes=3072,
                            max_steps=14, max_invocations=1)
    custody.sources = tuple(plan._sources.values())
    plan._captures = _RetainedPlanCaptures()
    custody.captures = plan._captures
    with torch.inference_mode(), torch.autocast(device_type=pixels.device.type, enabled=False), plan:
        custody.output = model(pixel_values=pixels, return_dict=True)
        synchronize()
        # Preserve a bounded completed output before any shape, plan cleanup,
        # capture-property, metadata, scoring or intervention validation. A
        # failed synchronize or copy leaves the owner quarantined; it does not
        # initiate another CUDA read in a salvage handler.
        logits = custody.output.logits
        if (type(logits) is not torch.Tensor or logits.dtype != torch.float32
                or not 0 < logits.numel() <= 1000):
            raise ValueError("returned logits cannot be safely retained")
        returned = _bits(logits)
        returned_name = f"return-{index:03d}.bin"
        returned_sha = _write_bytes(Path(output) / returned_name, returned)
        write_json_new(Path(output) / f"return-{index:03d}.json", {"label": label, "submission": index,
            "file": returned_name, "sha256": returned_sha, "shape": list(logits.shape),
            "dtype": "little-endian-float32"})
    check()
    captures = plan.captures
    if tuple(logits.shape) != (1, 1000):
        raise ValueError("invalid complete logits shape")
    tensors = {"logits": logits, **captures}
    if replacement is not None:
        tensors["requested"] = replacement
    data, layout = [], {}
    for name, value in tensors.items():
        raw = _bits(value)
        layout[name] = {"offset": sum(map(len, data)), "bytes": len(raw), "shape": list(value.shape)}
        data.append(raw)
    packed = b"".join(data)
    digest = _write_bytes(Path(output) / f"forward-{index:03d}.bin", packed)
    # The bytes are durable before metadata, scoring or landing checks.
    raw_tensors = {name: packed[row["offset"]:row["offset"] + row["bytes"]] for name, row in layout.items()}
    metadata = plan.metadata
    if _unexpected_hooks(model):
        raise RuntimeError("hook cleanup is uncertain")
    events = metadata.get("steps")
    if type(events) is not list or len(events) != len(steps):
        raise ValueError("missing action metadata")
    for i, (event, step) in enumerate(zip(events, steps)):
        if any(event.get(k) != step.get(k) for k in ("name", "operation", "module_path", "token_index", "batch_index")):
            raise ValueError("action metadata differs")
        if (event.get("order") != i or event.get("invocation") != 0 or
                event.get("output_shape") != [1, 197, 192] or event.get("dtype") != "torch.float32"):
            raise ValueError("action geometry differs")
    if patch:
        before, after = raw_tensors["before_all"], raw_tensors["after_all"]
        expected = bytearray(before)
        requested = b"".join(raw_tensors[f"before_{i}"] for i in range(4)) if self_patch else raw_tensors["requested"]
        for i, token in enumerate(TOKENS):
            start = token * 768
            expected[start:start + 768] = requested[i * 768:(i + 1) * 768]
            if raw_tensors[f"before_{i}"] != before[start:start + 768] or raw_tensors[f"after_{i}"] != expected[start:start + 768]:
                raise ValueError("selected same-forward witness differs")
        if after != bytes(expected):
            raise ValueError("landing or untouched tokens differ")
    margin = _margin(raw_tensors["logits"])
    row = {"label": label, "submission": index, "file": f"forward-{index:03d}.bin",
           "sha256": digest, "dtype": "little-endian-float32", "layout": layout,
           "metadata": metadata, "dog_minus_cat_log_odds": margin}
    write_json_new(Path(output) / f"forward-{index:03d}.json", row)
    return raw_tensors, row


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(cpu.dtype).encode() + b"\0")
        digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _startup():
    _verify_controller_import_mode()
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or os.environ.get("LOOKING_GLASS_VISION_BOOTSTRAP_PID") != str(os.getpid())):
        raise RuntimeError("isolated source bootstrap is required")


def _require_pristine():
    if any(name in sys.modules for name in ("torch", "numpy", "transformers")):
        raise RuntimeError("model libraries were imported before the physical lease")


def _held(session, lease):
    OwnerSession.check(session, lease)
    if _ROOT["session"] is not session or _ROOT["lease"] is not lease:
        raise RuntimeError("device lease custody changed")


def _measure(session, lease, torch, assets, reference, run_dir):
    import transformers
    from transformers import ViTForImageClassification

    _held(session, lease)
    def guarded(function, *args):
        _held(session, lease)
        result = function(*args)
        _held(session, lease)
        return result
    if (torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8"
            or transformers.__version__ != "4.57.3"):
        raise RuntimeError("unexpected runtime")
    # From the first CUDA-facing setup call onward, a failed call or lost
    # owner check leaves completion unverified until an explicit guarded sync.
    _ROOT["pending"] = True
    if guarded(torch.cuda.is_initialized):
        raise RuntimeError("premature CUDA initialization")
    if guarded(torch.cuda.device_count) != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    properties = guarded(torch.cuda.get_device_properties, 0)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=_runtime_uuid(torch, properties.uuid),
                                        device_count=1)
    if (properties.name != "NVIDIA GeForce RTX 2070 SUPER" or (properties.major, properties.minor) != (7, 5)
            or not 7 * 1024 ** 3 <= properties.total_memory <= 9 * 1024 ** 3):
        raise RuntimeError("device differs from the reviewed execution contract")
    if guarded(torch.cuda.get_allocator_backend) != "native":
        raise RuntimeError("native CUDA allocator is required")
    guarded(torch.cuda.set_per_process_memory_fraction, .25, 0)
    cap = int(.25 * properties.total_memory)
    if guarded(torch.cuda.get_per_process_memory_fraction, 0) != cap / properties.total_memory:
        raise RuntimeError("allocator cap readback differs")
    guarded(torch.cuda.reset_peak_memory_stats, 0)
    guarded(torch.cuda.synchronize, 0)
    _ROOT["pending"] = False

    def check():
        _held(session, lease)
        budget, held = _ROOT["budget"], _ROOT["memory_lease"]
        if budget is not None and (budget.active_leases() != (held,) or held.trace_bytes("gpu") != TRACE
                                   or held.trace_bytes("host") != TRACE):
            raise RuntimeError("owner memory or trace commitment changed")
        _held(session, lease)

    def sync():
        check()
        torch.cuda.synchronize(0)
        check()

    def transaction(fn):
        check()
        if _ROOT["pending"] is not False:
            raise RuntimeError("unverified CUDA work remains")
        _ROOT["pending"] = True
        result = fn()
        sync()
        _ROOT["pending"] = False
        return result

    stages = []
    def snapshot(stage):
        def collect():
            sync()
            free, total = guarded(torch.cuda.mem_get_info, 0)
            return {"device_free_bytes": free, "device_total_bytes": total,
                    "allocated_bytes": guarded(torch.cuda.memory_allocated, 0),
                    "reserved_bytes": guarded(torch.cuda.memory_reserved, 0),
                    "peak_allocated_bytes": guarded(torch.cuda.max_memory_allocated, 0),
                    "peak_reserved_bytes": guarded(torch.cuda.max_memory_reserved, 0)}
        value = transaction(collect)
        check_memory(value, expected_total=properties.total_memory, cap=cap, commitment=cap + HEADROOM)
        write_json_new(Path(run_dir) / f"stage-{len(stages) + 1:02d}.json",
                       {"stage": stage, "memory": value, "peak_scope": "since_pre_model_reset"})
        stages.append(stage)
        return value

    initial = snapshot("before_model_load")
    def telemetry():
        free, total = guarded(torch.cuda.mem_get_info, 0)
        return CudaMemorySnapshot(free, total, guarded(torch.cuda.memory_allocated, 0),
                                  guarded(torch.cuda.memory_reserved, 0))
    budget = MemoryBudget(telemetry, max_gpu_trace_bytes=TRACE, max_host_trace_bytes=TRACE)
    _ROOT["budget"] = budget
    memory_lease = budget.acquire(cap + HEADROOM, max_gpu_trace_bytes=TRACE, max_host_trace_bytes=TRACE)
    _ROOT["memory_lease"] = memory_lease
    memory_lease.reserve_trace("gpu", TRACE)
    memory_lease.reserve_trace("host", TRACE)
    check()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(17)
    transaction(lambda: torch.cuda.manual_seed(17))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime = {"torch": str(torch.__version__), "transformers": transformers.__version__, "device": "cuda:0",
               "cuda_build": torch.version.cuda, "python": sys.version.split()[0], "seed": 17,
               "device_name": properties.name, "compute_capability": [properties.major, properties.minor],
               "device_total_bytes": properties.total_memory, "dtype": "float32", "attention": "eager",
               "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
               "cudnn_benchmark": torch.backends.cudnn.benchmark,
               "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "autocast_enabled": torch.is_autocast_enabled("cuda"),
               "float32_matmul_precision": torch.get_float32_matmul_precision(),
               "allocator": "native", "allocator_cap_bytes": cap, "commitment_bytes": cap + HEADROOM,
               "gpu_trace_claim_bytes": TRACE, "host_trace_claim_bytes": TRACE}
    if (runtime["threads"] != 1 or runtime["interop_threads"] != 1 or not runtime["deterministic_algorithms"]
            or runtime["matmul_tf32"] or runtime["cudnn_tf32"] or runtime["cudnn_benchmark"]
            or not runtime["cudnn_deterministic"] or runtime["autocast_enabled"]):
        raise RuntimeError("precision or determinism settings differ")

    def numeric_settings():
        return {"threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "autocast_enabled": torch.is_autocast_enabled("cuda"),
                "float32_matmul_precision": torch.get_float32_matmul_precision()}

    expected_numeric = {name: runtime[name] for name in numeric_settings()}
    model, info = ViTForImageClassification.from_pretrained(str(assets), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, output_loading_info=True, attn_implementation="eager")
    _ROOT["model"] = model
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("model checkpoint loaded incompletely")
    model.eval()
    cfg = model.config
    if ((cfg.hidden_size, cfg.num_hidden_layers, cfg.num_labels, cfg.patch_size, cfg.image_size) != (192, 12, 1000, 16, 224)
            or cfg._attn_implementation != "eager"):
        raise RuntimeError("model geometry differs")
    if _state_digest(model) != STATE_SHA or any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("CPU checkpoint state differs")
    snapshot("before_model_copy")
    transaction(lambda: model.to("cuda:0"))
    if any(p.dtype != torch.float32 or p.device.type != "cuda" or p.device.index != 0 for p in model.parameters()):
        raise RuntimeError("model transfer differs")
    snapshot("after_model_copy")

    def frombytes(data, shape):
        return torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(shape)
    pixel_cpu = {side: frombytes(reference[side + "-pixels.f32"], (1, 3, 224, 224)) for side in ("cat", "dog")}
    _ROOT["pixels"] = {}
    for side in ("cat", "dog"):
        # Retain the target immediately on return from transfer, before syncing.
        def transfer():
            _ROOT["pixels"][side] = pixel_cpu[side].to("cuda:0")
        transaction(transfer)
    pixels = _ROOT["pixels"]
    records, observed = [], []
    for index, label in enumerate(LABELS, 1):
        _ROOT["submission"], _ROOT["label"] = index, label
        side = "dog" if label == "dog-clean" else "cat"
        observation = ObservationCustody()
        _ROOT["observation"] = observation
        source = None
        if label in ("cat-gpu-donor", "cat-cpu-donor"):
            if label == "cat-gpu-donor":
                raw = observed[1]["before_all"]
                donor = b"".join(raw[t * 768:(t + 1) * 768] for t in TOKENS)
            else:
                donor = reference["cpu-forward-024.bin"][312736:315808]
            def source_transfer():
                observation.replacement = frombytes(donor, (4, 192)).to("cuda:0")
            transaction(source_transfer)
            source = observation.replacement
        def execute():
            return observe(model, pixels[side], label=label, output=run_dir, index=index,
                           replacement=source, self_patch=label == "cat-self", synchronize=sync,
                           check=check, custody=observation)
        values, record = transaction(execute)
        records.append(record)
        observed.append(values)
        # All GPU work and byte copies completed, so these roots can be released.
        _ROOT["observation"] = None
        observation = source = None
        snapshot(f"after_forward_{index}")
    cat, dog, self_obs, gpu_patch, cpu_patch, repeat = observed
    if any(item["before_all"] != cat["before_all"] for item in (self_obs, gpu_patch, cpu_patch, repeat)):
        raise RuntimeError("same-GPU cat pre-action states differ")
    if self_obs["after_all"] != cat["before_all"] or any(
            item["logits"] != cat["logits"] for item in (self_obs, repeat)):
        raise RuntimeError("same-GPU self or repeat control differs")
    comparisons, margins = _descriptive(observed, reference, records)
    if transaction(lambda: _state_digest(model)) != STATE_SHA or _unexpected_hooks(model):
        raise RuntimeError("model state or hooks changed")
    numeric_after = numeric_settings()
    if numeric_after != expected_numeric:
        raise RuntimeError("runtime numerical settings changed after inference")
    refs = {"model": weakref.ref(model), "cat_pixels": weakref.ref(pixels["cat"]),
            "dog_pixels": weakref.ref(pixels["dog"])}
    _ROOT["model"] = _ROOT["pixels"] = None
    del model, pixels, pixel_cpu, observation, source
    gc.collect()
    survivors = {name: ref() is not None for name, ref in refs.items()}
    transaction(torch.cuda.empty_cache)
    snapshot("after_disposal")
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if not callable(clear):
        raise RuntimeError("expected PyTorch workspace cleanup API unavailable")
    transaction(clear)
    transaction(torch.cuda.empty_cache)
    cleanup = snapshot("after_cleanup")
    if any(survivors.values()) or cleanup["allocated_bytes"] != 0 or cleanup["reserved_bytes"] != 0:
        raise RuntimeError("GPU tensors or allocator bytes remain after cleanup")
    return {"reference_sha256": REFERENCE_SHA, "model_state_sha256": STATE_SHA, "forward_count": 6,
            "forwards": records, "runtime": runtime, "comparisons": comparisons, "margins": margins,
            "numeric_settings_after": numeric_after,
            "same_gpu_self_repeat_and_landing": True, "stages": stages, "initial_memory": initial,
            "cleanup_memory": cleanup, "cleanup_weakrefs": survivors}


def _read_json(path, limit=MIB):
    data = Path(path).read_bytes()
    if not 0 < len(data) <= limit:
        raise ValueError("JSON record exceeds its bound")
    return json.loads(data)


def _saved_forward(directory, index, expected, device="cuda:0"):
    row = _read_json(Path(directory) / f"forward-{index:03d}.json")
    if (type(row.get("submission")) is not int or row["submission"] != index or row.get("label") != expected
            or row.get("file") != f"forward-{index:03d}.bin"
            or row.get("dtype") != "little-endian-float32"):
        raise ValueError("forward identity differs")
    raw = (Path(directory) / row["file"]).read_bytes()
    if not 0 < len(raw) <= MIB or _sha(raw) != row.get("sha256"):
        raise ValueError("forward bytes differ")
    layout = row["layout"]
    expected_shapes = {"logits": [1, 1000], "before_all": [1, 197, 192]}
    if index in (3, 4, 5):
        expected_shapes.update({"after_all": [1, 197, 192]})
        for i in range(4):
            expected_shapes.update({f"before_{i}": [192], f"after_{i}": [192]})
        if index in (4, 5):
            expected_shapes["requested"] = [4, 192]
    if set(layout) != set(expected_shapes):
        raise ValueError("forward tensor inventory differs")
    spans = []
    tensors = {}
    for name, shape in expected_shapes.items():
        item = layout[name]
        nbytes = math.prod(shape) * 4
        offset = item.get("offset")
        if (type(item.get("shape")) is not list or any(type(v) is not int for v in item["shape"])
                or item["shape"] != shape or type(item.get("bytes")) is not int or item["bytes"] != nbytes
                or type(offset) is not int
                or offset < 0 or offset + nbytes > len(raw)):
            raise ValueError("forward tensor layout differs")
        spans.append((offset, offset + nbytes))
        tensors[name] = raw[offset:offset + nbytes]
    spans.sort()
    if spans[0][0] != 0 or spans[-1][1] != len(raw) or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("forward tensor layout has gaps or overlaps")
    if row.get("dog_minus_cat_log_odds") != _margin(tensors["logits"]):
        raise ValueError("saved score differs from logits")
    returned = _read_json(Path(directory) / f"return-{index:03d}.json", 16384)
    returned_raw = (Path(directory) / f"return-{index:03d}.bin").read_bytes()
    if (returned.get("label") != expected or type(returned.get("submission")) is not int
            or returned["submission"] != index or returned.get("file") != f"return-{index:03d}.bin"
            or type(returned.get("shape")) is not list or any(type(v) is not int for v in returned["shape"])
            or returned["shape"] != [1, 1000]
            or returned.get("dtype") != "little-endian-float32" or returned.get("sha256") != _sha(returned_raw)
            or returned_raw != tensors["logits"]):
        raise ValueError("early returned-logit receipt differs")
    planned = [("before_all", "capture", None)]
    if index in (3, 4, 5):
        for i, token in enumerate(TOKENS):
            planned.extend(((f"before_{i}", "capture", token), (f"action_{i}", "patch", token),
                            (f"after_{i}", "capture", token)))
        planned.append(("after_all", "capture", None))
    events = row.get("metadata", {}).get("steps")
    if type(events) is not list or len(events) != len(planned):
        raise ValueError("saved action inventory differs")
    for order, ((name, operation, token), event) in enumerate(zip(planned, events)):
        if (type(event) is not dict or event.get("name") != name or event.get("operation") != operation
                or type(event.get("order")) is not int or event["order"] != order
                or event.get("module_path") != SITE or type(event.get("forward_index")) is not int
                or event["forward_index"] != 0 or type(event.get("invocation")) is not int
                or event["invocation"] != 0
                or (token is None and (event.get("batch_index") is not None or event.get("token_index") is not None))
                or (token is not None and (type(event.get("batch_index")) is not int or event["batch_index"] != 0
                    or type(event.get("token_index")) is not int or event["token_index"] != token))
                or type(event.get("output_shape")) is not list
                or any(type(v) is not int for v in event["output_shape"]) or event["output_shape"] != [1, 197, 192]
                or event.get("dtype") != "torch.float32" or event.get("device") != device
                or event.get("source_capture") != (name.replace("action_", "before_")
                    if index == 3 and operation == "patch" else None)):
            raise ValueError("saved action metadata differs")
    if index in (3, 4, 5):
        before, after = tensors["before_all"], tensors["after_all"]
        requested = (b"".join(tensors[f"before_{i}"] for i in range(4))
                     if index == 3 else tensors["requested"])
        result = bytearray(before)
        for i, token in enumerate(TOKENS):
            start = token * 768
            result[start:start + 768] = requested[i * 768:(i + 1) * 768]
            if (tensors[f"before_{i}"] != before[start:start + 768]
                    or tensors[f"after_{i}"] != result[start:start + 768]):
                raise ValueError("selected witness differs")
        if after != bytes(result):
            raise ValueError("saved landing or untouched coordinates differ")
    return row, tensors


def _validate_saved(directory, measurement, reference):
    rows, tensors = [], []
    device = measurement.get("runtime", {}).get("device")
    if device not in ("cuda:0", "cpu"):
        raise ValueError("saved device differs")
    root = Path(directory)
    for pattern, prefix, suffix in (("forward-*.json", "forward-", ".json"),
                                    ("forward-*.bin", "forward-", ".bin"),
                                    ("return-*.json", "return-", ".json"),
                                    ("return-*.bin", "return-", ".bin")):
        expected_names = {f"{prefix}{i:03d}{suffix}" for i in range(1, 7)}
        if {path.name for path in root.glob(pattern)} != expected_names:
            raise ValueError("closed forward or return inventory differs")
    if any(root.glob("*.partial")):
        raise ValueError("an unfinished partial record remains")
    for i, label in enumerate(LABELS, 1):
        row, value = _saved_forward(directory, i, label, device=device)
        rows.append(row)
        tensors.append(value)
    if (type(measurement.get("forward_count")) is not int or measurement["forward_count"] != 6
            or measurement.get("forwards") != rows):
        raise ValueError("candidate and closed forward records differ")
    cat, dog, self_obs, gpu_patch, cpu_patch, repeat = tensors
    if any(x["before_all"] != cat["before_all"] for x in (self_obs, gpu_patch, cpu_patch, repeat)):
        raise ValueError("saved cat pre-action witnesses differ")
    if (self_obs["after_all"] != cat["before_all"] or self_obs["logits"] != cat["logits"]
            or repeat["logits"] != cat["logits"]):
        raise ValueError("saved self/repeat control differs")
    donor = dog["before_all"]
    expected_gpu = b"".join(donor[t * 768:(t + 1) * 768] for t in TOKENS)
    if (gpu_patch["requested"] != expected_gpu
            or cpu_patch["requested"] != reference["cpu-forward-024.bin"][312736:315808]):
        raise ValueError("saved patch donor bytes differ")
    comparisons, margins = _descriptive(tensors, reference, rows)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if (canonical(measurement.get("comparisons")) != canonical(comparisons)
            or canonical(measurement.get("margins")) != canonical(margins)):
        raise ValueError("candidate comparisons differ from the saved bytes")
    if (measurement.get("reference_sha256") != REFERENCE_SHA or measurement.get("model_state_sha256") != STATE_SHA
            or measurement.get("same_gpu_self_repeat_and_landing") is not True):
        raise ValueError("candidate identity or controls differ")
    runtime, cleanup = measurement["runtime"], measurement["cleanup_memory"]
    expected_runtime_keys = {"torch", "transformers", "device", "cuda_build", "python", "seed", "device_name",
        "compute_capability", "device_total_bytes", "dtype", "attention", "threads", "interop_threads",
        "deterministic_algorithms", "matmul_tf32", "cudnn_tf32", "cudnn_benchmark", "cudnn_deterministic",
        "autocast_enabled", "float32_matmul_precision", "allocator", "allocator_cap_bytes", "commitment_bytes",
        "gpu_trace_claim_bytes", "host_trace_claim_bytes"}
    if (type(runtime) is not dict or set(runtime) != expected_runtime_keys
            or runtime.get("torch") != "2.8.0+cu128" or runtime.get("cuda_build") != "12.8"
            or runtime.get("transformers") != "4.57.3" or type(runtime.get("python")) is not str
            or not runtime["python"] or type(runtime.get("seed")) is not int or runtime["seed"] != 17
            or runtime.get("device_name") != "NVIDIA GeForce RTX 2070 SUPER"
            or runtime.get("compute_capability") != [7, 5]
            or any(type(v) is not int for v in runtime["compute_capability"])
            or type(runtime.get("device_total_bytes")) is not int
            or not 7 * 1024 ** 3 <= runtime["device_total_bytes"] <= 9 * 1024 ** 3
            or type(runtime.get("threads")) is not int or runtime["threads"] != 1
            or type(runtime.get("interop_threads")) is not int or runtime["interop_threads"] != 1
            or runtime.get("dtype") != "float32" or runtime.get("attention") != "eager"
            or runtime.get("autocast_enabled") is not False or runtime.get("matmul_tf32") is not False
            or runtime.get("cudnn_tf32") is not False or runtime.get("cudnn_benchmark") is not False
            or runtime.get("cudnn_deterministic") is not True or runtime.get("deterministic_algorithms") is not True
            or runtime.get("float32_matmul_precision") != "highest" or runtime.get("allocator") != "native"
            or type(runtime.get("allocator_cap_bytes")) is not int
            or runtime["allocator_cap_bytes"] != runtime["device_total_bytes"] // 4
            or type(runtime.get("commitment_bytes")) is not int
            or runtime["commitment_bytes"] != runtime["allocator_cap_bytes"] + HEADROOM
            or type(runtime.get("gpu_trace_claim_bytes")) is not int or runtime["gpu_trace_claim_bytes"] != TRACE
            or type(runtime.get("host_trace_claim_bytes")) is not int or runtime["host_trace_claim_bytes"] != TRACE
            or type(cleanup.get("allocated_bytes")) is not int or cleanup["allocated_bytes"] != 0
            or type(cleanup.get("reserved_bytes")) is not int or cleanup["reserved_bytes"] != 0
            or type(measurement.get("cleanup_weakrefs")) is not dict
            or set(measurement["cleanup_weakrefs"]) != {"model", "cat_pixels", "dog_pixels"}
            or any(value is not False for value in measurement["cleanup_weakrefs"].values())):
        raise ValueError("candidate precision or cleanup differs")
    numeric_names = {"threads", "interop_threads", "deterministic_algorithms", "matmul_tf32", "cudnn_tf32",
                     "cudnn_benchmark", "cudnn_deterministic", "autocast_enabled", "float32_matmul_precision"}
    expected_numeric = {name: runtime[name] for name in numeric_names}
    if canonical(measurement.get("numeric_settings_after")) != canonical(expected_numeric):
        raise ValueError("post-inference numerical settings differ")
    expected_stages = (["before_model_load", "before_model_copy", "after_model_copy"]
                       + [f"after_forward_{i}" for i in range(1, 7)] + ["after_disposal", "after_cleanup"])
    if measurement.get("stages") != expected_stages:
        raise ValueError("candidate stage inventory differs")
    if {path.name for path in root.glob("stage-*.json")} != {f"stage-{i:02d}.json" for i in range(1, 12)}:
        raise ValueError("saved memory stage inventory differs")
    for i, stage in enumerate(expected_stages, 1):
        saved = _read_json(Path(directory) / f"stage-{i:02d}.json", 16384)
        if saved.get("stage") != stage or saved.get("peak_scope") != "since_pre_model_reset":
            raise ValueError("saved memory stage identity differs")
        check_memory(saved["memory"], expected_total=runtime["device_total_bytes"],
                     cap=runtime["allocator_cap_bytes"], commitment=runtime["commitment_bytes"])
        if ((i == 1 and saved["memory"] != measurement["initial_memory"])
                or (i == len(expected_stages) and saved["memory"] != cleanup)):
            raise ValueError("candidate memory counters differ from the saved stage")
    return rows


def owner_main(run_dir, assets, run_id, commit, git):
    try:
        if _ROOT["entered"] is not False:
            raise RuntimeError("owner is single-use")
        _ROOT["entered"] = True
        _startup()
        _require_pristine()
        session = OwnerSession.from_environment()
        _ROOT["session"] = session
        _validate_workspace_spec(session)
        lease = DeviceLease.acquire(session.device_lease_path, timeout=30)
        _ROOT["lease"] = lease
        _held(session, lease)
        _validate_workspace_binding(session)
        repo = Path(__file__).resolve().parents[1]
        if _source_identity(repo, git)[0] != commit:
            raise RuntimeError("child source identity differs")
        _asset_inventory(repo, assets)
        reference = _load_reference(os.environ["LOOKING_GLASS_VISION_REFERENCE"])
        write_json_new(Path(run_dir) / "owner.json", {"schema": "vision-vit-cuda-owner-v1",
            "run_id": run_id, "supervisor_run_id": session.run_id, "pid": os.getpid(), "source_commit": commit})
        import torch
        result = _measure(session, lease, torch, assets, reference, run_dir)
        _held(session, lease)
        if _ROOT["pending"] or _source_identity(repo, git)[0] != commit:
            raise RuntimeError("source or CUDA completion became uncertain")
        _asset_inventory(repo, assets)
        _load_reference(os.environ["LOOKING_GLASS_VISION_REFERENCE"])
        write_json_new(Path(run_dir) / "candidate.json", {"schema": "vision-vit-cuda-candidate-v1",
            "run_id": run_id, "supervisor_run_id": session.run_id, "pid": os.getpid(),
            "source_commit": commit, "measurement": result})
        candidate_sha = _file_identity(Path(run_dir) / "candidate.json", MIB)[1]
        memory_lease = _ROOT["memory_lease"]
        memory_lease.release_trace("gpu", TRACE)
        memory_lease.release_trace("host", TRACE)
        memory_lease.release()
        if _ROOT["budget"].active_leases():
            raise RuntimeError("memory commitment remains")
        _held(session, lease)
        write_json_new(Path(run_dir) / "final.json", {"schema": "vision-vit-cuda-final-v1", "run_id": run_id,
            "supervisor_run_id": session.run_id, "pid": os.getpid(), "source_commit": commit,
            "candidate_sha256": candidate_sha, "active_memory_leases": 0})
        _held(session, lease)
        OwnerSession.attest_clean(session, lease)
        return 0
    except BaseException as exc:
        _ROOT["failure"] = exc
        try:
            observation = _ROOT["observation"]
            write_json_new(Path(run_dir) / "failure.json", {"schema": "vision-vit-cuda-failure-v1",
                "run_id": run_id, "status": "incomplete", "error_type": type(exc).__name__,
                "submission": _ROOT["submission"], "label": _ROOT["label"],
                "cuda_completion_unverified": _ROOT["pending"],
                "cleanup_attested": False,
                "model_return_observed": observation is not None and observation.output is not None,
                "detail": traceback.format_exc()[-8192:]}, max_bytes=16384)
        except BaseException as error:
            _ROOT["diagnostic_failure"] = error
        return 1


def controller_main(workspace_root, assets, reference, output, site, smi):
    _startup()
    _require_pristine()
    repo = Path(__file__).resolve().parents[1]
    root = _workspace_root(repo, workspace_root)
    assets = Path(assets).resolve(strict=True)
    reference_path = Path(reference).resolve(strict=True)
    site = Path(site).resolve(strict=True)
    destination = Path(output)
    if not destination.is_absolute() or destination.exists() or not destination.parent.is_dir():
        raise ValueError("output must be a new absolute directory")
    git = os.environ["LOOKING_GLASS_VISION_GIT_EXE"]
    commit, hashes = _source_identity(repo, git)
    if commit != os.environ["LOOKING_GLASS_VISION_SOURCE_COMMIT"]:
        raise RuntimeError("controller and bootstrap source identities differ")
    assets_hashes = _asset_inventory(repo, assets)
    ref = _load_reference(reference_path)
    gpu_uuid, before = _smi_snapshot(smi)
    environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC")
                   if key in os.environ}
    system_root = Path(environment.get("SYSTEMROOT", "C:/Windows"))
    environment.update({"PATH": str(system_root / "System32"), "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": str(root / "work" / "hf-cache"),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "LOOKING_GLASS_VISION_REFERENCE": str(reference_path)})
    identity = resolve_device_identity(root, gpu_uuid, physical_uuids=[gpu_uuid], environ=environment)
    destination.mkdir(exist_ok=False)
    prefix = Path(tempfile.mkdtemp(prefix="pycache-", dir=destination))
    environment["PYTHONPYCACHEPREFIX"] = str(prefix)
    run_id = secrets.token_hex(16)
    write_json_new(destination / "started.json", {"schema": "vision-vit-cuda-start-v1", "run_id": run_id,
        "source_commit": commit, "source_sha256": hashes, "asset_sha256": assets_hashes,
        "reference_sha256": REFERENCE_SHA, "started_utc": before["timestamp_utc"]})
    bootstrap = repo / "scripts" / "vision_vit_cuda_bootstrap.py"
    command = [sys._base_executable, "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix), str(bootstrap),
               "--bootstrap-site", str(site), "--source-commit", commit, "--git-executable", git,
               "--owner", str(destination), str(assets), run_id, commit, git]
    supervision = None
    error = None
    try:
        supervision = run_supervised(command, device_lease_path=identity.lock_path,
            marker_path=identity.marker_path, identity=identity, timeout=300, lease_timeout=30,
            cwd=repo, env=environment, inherit_environment=False, require_zero_exit=True)
    except BaseException as exc:
        error = type(exc).__name__
    if supervision is not None:
        write_json_new(destination / "supervision.json", {"run_id": run_id,
            "supervisor_run_id": supervision.run_id, "pid": supervision.pid, "exit_code": supervision.exit_code,
            "clean": supervision.clean, "timed_out": supervision.timed_out, "reason": supervision.reason})
    after, presence = None, {"query": "not_attempted", "owner_pid_listed": None}
    telemetry_error = None
    try:
        after_uuid, after = _smi_snapshot(smi)
        if after_uuid != gpu_uuid:
            raise RuntimeError("device inventory changed")
        if supervision is not None:
            presence = _public_presence(_owner_presence_after_exit(smi, supervision.pid))
    except Exception as exc:
        telemetry_error = type(exc).__name__
    outcome = {"schema": "vision-vit-cuda-result-v1", "run_id": run_id, "source_commit": commit,
        "status": "incomplete", "supervisor": _public_supervisor(supervision),
        "supervisor_error_type": error, "telemetry_error_type": telemetry_error,
        "nvidia_smi_before": _public_smi_snapshot(before), "nvidia_smi_after": _public_smi_snapshot(after),
        "owner_presence_after_exit": presence,
        "scope": "previously exposed pair, descriptive cross-runtime observation",
        "limits": ["direct-child supervision assumes no CUDA descendants",
                   "device lease and cleanup are cooperative; other GPU users are not controlled",
                   "memory observations are points in time, not worst-case or total host-memory bounds",
                   "source and receipts assume trusted quiescent files and native runtime, not execution attestation",
                   "CPU and GPU differences are descriptive, with no general performance claim"]}
    if (supervision is not None and supervision.clean is True and type(supervision.exit_code) is int
            and supervision.exit_code == 0 and supervision.timed_out is False
            and not (destination / "failure.json").exists()):
        try:
            candidate = _read_json(destination / "candidate.json")
            final = _read_json(destination / "final.json", 16384)
            digest = _file_identity(destination / "candidate.json", MIB)[1]
            owner = _read_json(destination / "owner.json", 16384)
            for record, schema in ((owner, "vision-vit-cuda-owner-v1"), (candidate, "vision-vit-cuda-candidate-v1"),
                                   (final, "vision-vit-cuda-final-v1")):
                if (record.get("schema") != schema or record.get("run_id") != run_id
                        or record.get("source_commit") != commit or record.get("supervisor_run_id") != supervision.run_id
                        or type(record.get("pid")) is not int or record["pid"] != supervision.pid):
                    raise ValueError("candidate identity does not match the reaped owner")
            if (final.get("candidate_sha256") != digest or type(final.get("active_memory_leases")) is not int
                    or final["active_memory_leases"] != 0):
                raise ValueError("candidate finalization differs")
            _validate_saved(destination, candidate["measurement"], ref)
            if candidate["measurement"]["runtime"].get("device") != "cuda:0":
                raise ValueError("completed owner did not record the CUDA execution device")
            if _source_identity(repo, git) != (commit, hashes):
                raise ValueError("source changed during observation")
            _asset_inventory(repo, assets)
            _load_reference(reference_path)
            if telemetry_error is not None or presence.get("query") != "ok" or presence.get("owner_pid_listed") is not False:
                raise ValueError("post-exit observation was unavailable or still listed the owner")
            outcome["status"] = "complete"
            outcome["candidate_sha256"] = digest
            outcome["forward_count"] = 6
        except Exception as exc:
            outcome["validation_error_type"] = type(exc).__name__
    write_json_new(destination / "outcome.json", outcome)
    return outcome


def main(argv=None):
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", nargs=5, help=argparse.SUPPRESS)
    parser.add_argument("--workspace-root")
    parser.add_argument("--assets")
    parser.add_argument("--reference")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.owner:
        return owner_main(*args.owner)
    if not all((args.workspace_root, args.assets, args.reference, args.output)):
        parser.error("workspace-root, assets, reference and output are required")
    result = controller_main(args.workspace_root, args.assets, args.reference, args.output,
        os.environ["LOOKING_GLASS_VISION_BOOTSTRAP_SITE"], shutil.which("nvidia-smi"))
    print(json.dumps({"status": result["status"], "source_commit": result["source_commit"]}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    if "--owner" in sys.argv[1:]:
        code = 1
        try:
            code = main()
        finally:
            os._exit(code if type(code) is int and 0 <= code <= 255 else 1)
    raise SystemExit(main())
