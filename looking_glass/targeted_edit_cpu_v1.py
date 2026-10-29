"""One preregistered CPU-only parameter-edit experiment; no weights are saved.

The public cases are held out from fitting, not blind. See the versioned
protocol for the decision rule and the limits of recorded attestations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import pretrained, pretrained_exposures, pretrained_exposures_v2


VERSION = "targeted-edit-cpu-v1"
SEED = 20261029
SITE = "transformer.h.5.mlp.c_proj.weight"
WEIGHT_SHAPE = (3072, 768)
VOCAB_SIZE = 50257
PARAMETER_COUNT = 81912576
RELATIVE_NORM = 0.01
EXPOSURE_SNAPSHOT = {
    "commit": "41158c397a60febb4c7464aa393261a2346e7b9a",
    "canonical_payload_sha256": "e64bc0b875fb295f03d7ad4d5180f6978d1d632487a98ea2824df01844ea4fb1",
    "pre_reservation_check": "no candidate complete row equalled a known row or its nonempty prefix",
    "scope": "public non-blind cases; unknown history and pretraining are not certified",
}
SOURCE_PATHS = (
    "docs/TARGETED_EDIT_CPU_V1.md",
    "looking_glass/__init__.py",
    "looking_glass/pretrained.py",
    "looking_glass/pretrained_exposures.py",
    "looking_glass/pretrained_exposures_v2.py",
    "looking_glass/targeted_edit_cpu_v1.py",
)
CASES = (
    ("discovery", "discovery", "The capital city of France is", (464, 3139, 1748, 286, 4881, 318), None),
    ("target-1", "target", "The capital of France is", (464, 3139, 286, 4881, 318), None),
    ("target-2", "target", "France has its capital in", (28572, 468, 663, 3139, 287), None),
    ("target-3", "target", "In France, the capital city is", (818, 4881, 11, 262, 3139, 1748, 318), None),
    ("target-4", "target", "France's capital city is", (28572, 338, 3139, 1748, 318), None),
    ("locality-1", "locality", "The capital of Italy is", (464, 3139, 286, 8031, 318), (" Rome", 10598)),
    ("locality-2", "locality", "The capital city of Spain is", (464, 3139, 1748, 286, 8602, 318), (" Madrid", 14708)),
    ("locality-3", "locality", "Japan's capital is", (16504, 338, 3139, 318), (" Tokyo", 11790)),
    ("locality-4", "locality", "The capital of Germany is", (464, 3139, 286, 4486, 318), (" Berlin", 11307)),
    ("locality-5", "locality", "The largest ocean is the", (464, 4387, 9151, 318, 262), (" Pacific", 8211)),
    ("locality-6", "locality", "Two plus two equals", (7571, 5556, 734, 21767), (" four", 1440)),
    ("locality-7", "locality", "The opposite of hot is", (464, 6697, 286, 3024, 318), (" cold", 4692)),
    ("locality-8", "locality", "A week has seven", (32, 1285, 468, 3598), (" days", 1528)),
)
CONDITIONS = ("original", "gradient", "random")
LONDON, PARIS = 3576, 6342
RULE = {
    "baseline_margin": "logit(London)-logit(Paris)<0",
    "minimum_originally_negative_targets": 3,
    "minimum_reversed_originally_negative_targets": 3,
    "mean_target_improvement_strictly_greater_than": 1.0,
    "random_control_advantage_at_least": 0.5,
    "all_locality_top_tokens_unchanged": True,
    "mean_locality_kl_at_most": 0.02, "max_locality_kl_at_most": 0.10,
    "mean_locality_nll_deterioration_at_most": 0.05,
}


class StudyError(RuntimeError):
    """The study cannot be validly executed or audited."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    with open(path, "rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _bounded_read(path, maximum):
    with open(path, "rb") as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise StudyError("artifact exceeds its declared bound")
    return data


def _decode_document(data):
    try:
        value = json.loads(data)
        canonical = _canonical(value)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        raise StudyError("invalid JSON artifact") from exc
    if type(value) is not dict or canonical != data:
        raise StudyError("JSON artifact is not canonical")
    return value


def _read_document(path, maximum=512_000):
    return _decode_document(_bounded_read(path, maximum))


def _tensor_bytes(tensor):
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise StudyError("expected a CPU float32 tensor")
    return tensor.detach().contiguous().numpy().astype("<f4", copy=False).tobytes()


def _tensor_sha(tensor):
    return _sha(_tensor_bytes(tensor))


def _git(repo, *args, check=True):
    return subprocess.run(["git", *args], cwd=repo, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _commit_name(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise StudyError("invalid commit identity")
    return value


def _cases_payload():
    return [{"case_id": name, "split": split, "prompt": text,
             "token_ids": list(ids), "expected": None if label is None else
             {"text": label[0], "token_id": label[1]}}
            for name, split, text, ids, label in CASES]


def _manifest(source_commit, files):
    return {
        "version": VERSION, "source_commit": source_commit,
        "source_sha256": files, "model_id": pretrained.MODEL_ID,
        "model_revision": pretrained.REVISION,
        "assets": {name: digest for name, (_, digest) in pretrained.ASSETS.items()},
        "target": {"text": " London", "token_id": LONDON},
        "competitor": {"text": " Paris", "token_id": PARIS},
        "site": SITE, "site_shape": list(WEIGHT_SHAPE), "parameter_count": PARAMETER_COUNT, "seed": SEED,
        "relative_norm": RELATIVE_NORM, "conditions": list(CONDITIONS),
        "cases": _cases_payload(), "success_rule": RULE,
        "exposure_snapshot": EXPOSURE_SNAPSHOT,
    }


def make_manifest(repo, source_commit):
    """Bind fixed cases to already committed source; does not load a model."""
    source_commit = _commit_name(source_commit)
    repo = Path(repo)
    files = {}
    for path in SOURCE_PATHS:
        blob = _git(repo, "show", f"{source_commit}:{path}")
        files[path] = _sha(blob)
    return _manifest(source_commit, files)


def _public_manifest(path, expected_sha256):
    if type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise StudyError("invalid manifest digest")
    raw = _bounded_read(path, 256_000)
    if _sha(raw) != expected_sha256:
        raise StudyError("manifest digest differs")
    value = _decode_document(raw)
    source = _commit_name(value.get("source_commit"))
    files = value.get("source_sha256")
    if (type(files) is not dict or set(files) != set(SOURCE_PATHS) or
            any(type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in files.values()) or _canonical(value) != _canonical(_manifest(source, files))):
        raise StudyError("manifest does not match the fixed protocol")
    return value, raw


def verify_manifest(repo, manifest_path, expected_sha256):
    """Verify the separately committed manifest and all bound checkout sources."""
    repo, path = Path(repo).resolve(), Path(manifest_path).resolve()
    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise StudyError("manifest must be in the checkout") from exc
    data, raw = _public_manifest(path, expected_sha256)
    source = _commit_name(data.get("source_commit"))
    if _canonical(make_manifest(repo, source)) != raw:
        raise StudyError("manifest does not match the fixed protocol and source")
    head = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    if source == head or subprocess.run(["git", "merge-base", "--is-ancestor", source, head],
                                        cwd=repo, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL).returncode:
        raise StudyError("manifest must be committed after the source commit")
    if subprocess.run(["git", "cat-file", "-e", f"{source}:{relative}"], cwd=repo,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise StudyError("manifest already existed at the source commit")
    if _git(repo, "show", f"{head}:{relative}") != raw:
        raise StudyError("manifest is not the committed checkout version")
    for name, digest in data["source_sha256"].items():
        if _file_sha(repo / name) != digest:
            raise StudyError("a source checkout file differs from its frozen blob")
    return data, head


def _json_new(path, value):
    """Publish one complete JSON document after its bytes have been flushed."""
    path = Path(path)
    pending = path.with_name(path.name + ".partial")
    if path.exists() or pending.exists():
        raise StudyError("artifact already exists")
    with open(pending, "xb") as handle:
        handle.write(_canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise StudyError("artifact already exists")
    os.replace(pending, path)


def _failure_terminal(output, value):
    """Never replace a terminal if publication may already have completed."""
    path = output / "terminal.json"
    if path.exists() or path.with_name(path.name + ".partial").exists():
        return
    try:
        _json_new(path, value)
    except KeyboardInterrupt as exc:
        try:
            exc.published_terminal = (value["status"] + "_verified" if
                                      _canonical(_read_document(path)) == _canonical(value) else "unknown")
        except Exception:
            exc.published_terminal = "unknown"
        raise
    except Exception:
        # The absent or partial terminal explicitly leaves the run incomplete.
        return


def _norm(tensor):
    return float(torch.linalg.vector_norm(tensor.detach().to(torch.float64)).item())


def _finite_number(value):
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _denormal_probe():
    """Probe two float32 hardware cases and NumPy's float64 conversion."""
    smallest = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    normal = torch.tensor([0x00800000], dtype=torch.int32).view(torch.float32)
    one_bits = int((smallest * torch.ones(1, dtype=torch.float32)).view(torch.int32)[0].item())
    half_bits = int((normal * torch.tensor([0.5], dtype=torch.float32)).view(torch.int32)[0].item())
    converted = float(np.array([1], dtype="<u4").view("<f4").astype("<f8")[0])
    result = {"subnormal_times_one_bits": one_bits, "normal_times_half_bits": half_bits,
              "numpy_cast_preserves_min_subnormal": converted == 2.0 ** -149}
    if result != {"subnormal_times_one_bits": 1, "normal_times_half_bits": 0x00400000,
                  "numpy_cast_preserves_min_subnormal": True}:
        raise StudyError("the denormal-preservation probe failed")
    return result


def _model_hash(model):
    """Hash named CPU state tensors, including tied names, without saving them."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise StudyError("unexpected state tensor")
        header = _canonical({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        view = memoryview(tensor.detach().numpy()).cast("B")
        digest.update(len(view).to_bytes(8, "little"))
        digest.update(view)
    return digest.hexdigest()


def _scale_outer(left, right, desired_norm):
    if (left.dtype != torch.float32 or right.dtype != torch.float32 or
            not torch.isfinite(left).all() or not torch.isfinite(right).all()):
        raise StudyError("invalid rank-one factors")
    outer = torch.outer(left, right)
    norm = _norm(outer)
    if not math.isfinite(norm) or norm == 0 or not math.isfinite(desired_norm) or desired_norm <= 0:
        raise StudyError("rank-one update has invalid norm")
    result = (outer.to(torch.float64) * (desired_norm / norm)).to(torch.float32)
    if not torch.isfinite(result).all():
        raise StudyError("rank-one update is nonfinite")
    return result


def _cleanup_candidate(handles, original_flags, weight):
    """Attempt every hook/gradient/flag cleanup even if one operation fails."""
    caught = None
    actions = [handle.remove for handle in handles]
    actions.append(lambda: setattr(weight, "grad", None))
    actions.extend(lambda parameter=parameter, flag=flag: parameter.requires_grad_(flag)
                   for parameter, flag in original_flags)
    for action in actions:
        for _ in range(2):
            try:
                action()
                break
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt) or caught is None:
                    caught = exc
    return caught


def _candidate(model, module, ids, target, competitor, *, expected_shape=None):
    """Derive the two updates from one discovery forward, without applying them."""
    weight = module.weight
    if expected_shape is not None and tuple(weight.shape) != tuple(expected_shape):
        raise StudyError("unexpected projection shape")
    captured = {}

    def before(_module, inputs):
        if "input" in captured or len(inputs) != 1 or inputs[0].ndim != 3:
            raise StudyError("unexpected projection input")
        captured["input"] = inputs[0][0, -1].detach().clone()

    def after(_module, _inputs, output):
        if "output" in captured or not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise StudyError("unexpected projection output")
        output.retain_grad()
        captured["output"] = output

    handles = []
    original_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
        for parameter, _ in original_flags:
            parameter.requires_grad_(False)
        weight.requires_grad_(True)
        weight.grad = None
        tokens = torch.tensor([ids], dtype=torch.int64)
        logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        if logits.shape[:2] != tokens.shape or not torch.isfinite(logits).all():
            raise StudyError("invalid discovery logits")
        margin = logits[0, -1, target] - logits[0, -1, competitor]
        discovery_row = logits[0, -1].detach().contiguous()
        discovery_sha = _tensor_sha(discovery_row)
        margin_float64 = float((discovery_row[target].to(torch.float64) -
                                discovery_row[competitor].to(torch.float64)).item())
        margin.backward()
        if set(captured) != {"input", "output"} or captured["output"].grad is None or weight.grad is None:
            raise StudyError("missing projection gradient")
        left, right = captured["input"], captured["output"].grad[0, -1].detach().clone()
        outer = torch.outer(left, right)
        if not torch.isfinite(weight.grad).all() or not torch.allclose(weight.grad, outer, rtol=1e-5, atol=1e-7):
            raise StudyError("projection gradient is not the specified outer product")
        gradient_norm = _norm(weight.grad)
        if not math.isfinite(gradient_norm) or gradient_norm == 0:
            raise StudyError("projection gradient is zero or nonfinite")
        residual = _norm(weight.grad.to(torch.float64) - outer.to(torch.float64)) / gradient_norm
        if not math.isfinite(residual) or residual > 1e-5:
            raise StudyError("projection gradient has excessive rank-one residual")
        original_norm = _norm(weight)
        planned = RELATIVE_NORM * original_norm
        gradient = _scale_outer(left, right, planned)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        random_left = torch.randn(weight.shape[0], generator=generator, dtype=torch.float32)
        random_right = torch.randn(weight.shape[1], generator=generator, dtype=torch.float32)
        random = _scale_outer(random_left, random_right, planned)
        return gradient, random, {
            "discovery_objective_float32": float(margin.detach().item()),
            "discovery_margin_float64": margin_float64,
            "discovery_logit_sha256": discovery_sha,
            "weight_norm": original_norm, "planned_delta_norm": planned,
            "gradient_norm": gradient_norm, "input_factor_norm": _norm(left),
            "output_factor_norm": _norm(right), "gradient_outer_relative_residual": residual,
        }
    finally:
        prior = sys.exception()
        cleanup_error = _cleanup_candidate(handles, original_flags, weight)
        if cleanup_error is not None and (prior is None or
                (isinstance(cleanup_error, KeyboardInterrupt) and not isinstance(prior, KeyboardInterrupt))):
            raise cleanup_error


def _candidate_attestation(original, candidates):
    """Compute actual post-addition norms and hashes without mutating the model."""
    original_norm = _norm(original)
    info = {}
    for name, delta in candidates.items():
        changed = original.clone()
        changed.add_(delta)
        actual = changed.to(torch.float64) - original.to(torch.float64)
        actual_norm = _norm(actual)
        info[name] = {"intended_delta_sha256": _tensor_sha(delta),
                      "realized_weight_sha256": _tensor_sha(changed),
                      "realized_delta_norm": actual_norm,
                      "realized_relative_norm": actual_norm / original_norm}
    a, b = info["gradient"]["realized_delta_norm"], info["random"]["realized_delta_norm"]
    if (not math.isfinite(a) or not math.isfinite(b) or a <= 0 or b <= 0 or
            not 0.999 <= a / b <= 1.001 or
            max(info[name]["realized_relative_norm"] for name in info) > 0.01001):
        raise StudyError("actual edit magnitudes are not within the declared bounds")
    return info


def _score(array):
    """Float64 score from one saved float32 vocabulary row."""
    if array.shape != (VOCAB_SIZE,) or not np.isfinite(array).all():
        raise StudyError("invalid saved logit row")
    z = array.astype(np.float64)
    maximum = float(np.max(z))
    logp = z - (maximum + float(np.log(np.sum(np.exp(z - maximum), dtype=np.float64))))
    return z, logp, int(np.argmax(z))


def analyze(logits):
    """Recompute all declared metrics and the decision from three arrays."""
    _denormal_probe()
    if set(logits) != set(CONDITIONS):
        raise StudyError("missing condition")
    expected_shape = (len(CASES), VOCAB_SIZE)
    if any(array.shape != expected_shape or array.dtype != np.dtype("<f4") for array in logits.values()):
        raise StudyError("invalid logit matrix")
    scores = {name: [_score(row) for row in logits[name]] for name in CONDITIONS}
    margins = {name: [float(z[LONDON] - z[PARIS]) for z, _, _ in scores[name][:5]]
               for name in CONDITIONS}
    baseline = margins["original"]
    summaries = {}
    for name in ("gradient", "random"):
        locality = []
        for index in range(5, len(CASES)):
            _, base_logp, base_top = scores["original"][index]
            _, edit_logp, edit_top = scores[name][index]
            label = CASES[index][4][1]
            kl = float(np.sum(np.exp(base_logp) * (base_logp - edit_logp), dtype=np.float64))
            locality.append({"case_id": CASES[index][0], "original_top": base_top,
                             "condition_top": edit_top, "kl": kl,
                             "original_nll": float(-base_logp[label]),
                             "condition_nll": float(-edit_logp[label])})
        diffs = [margins[name][i] - baseline[i] for i in range(1, 5)]
        summaries[name] = {
            "discovery_margin": margins[name][0],
            "target_margins": margins[name][1:], "target_changes": diffs,
            "mean_target_change": float(sum(diffs) / len(diffs)),
            "locality": locality,
            "mean_kl": float(sum(row["kl"] for row in locality) / len(locality)),
            "max_kl": float(max(row["kl"] for row in locality)),
            "mean_nll_deterioration": float(sum(row["condition_nll"] - row["original_nll"]
                                                for row in locality) / len(locality)),
            "all_top_preserved": all(row["original_top"] == row["condition_top"] for row in locality),
        }
    gradient, random = summaries["gradient"], summaries["random"]
    eligible = (baseline[0] < 0 and sum(value < 0 for value in baseline[1:]) >=
                RULE["minimum_originally_negative_targets"])
    criteria = {
        "discovery_reversal": gradient["discovery_margin"] > 0,
        "three_originally_negative_targets_reversed": sum(
            baseline[i] < 0 and margins["gradient"][i] > 0 for i in range(1, 5)) >= RULE["minimum_reversed_originally_negative_targets"],
        "mean_target_improvement": gradient["mean_target_change"] > RULE["mean_target_improvement_strictly_greater_than"],
        "random_control_advantage": gradient["mean_target_change"] - random["mean_target_change"] >= RULE["random_control_advantage_at_least"],
        "all_locality_top_preserved": gradient["all_top_preserved"],
        "mean_locality_kl": gradient["mean_kl"] <= RULE["mean_locality_kl_at_most"],
        "max_locality_kl": gradient["max_kl"] <= RULE["max_locality_kl_at_most"],
        "mean_locality_nll_deterioration": gradient["mean_nll_deterioration"] <= RULE["mean_locality_nll_deterioration_at_most"],
    }
    return {"version": VERSION, "baseline_margins": baseline,
            "conditions": summaries, "eligible": eligible, "criteria": criteria,
            "decision": "success" if eligible and all(criteria.values()) else
                        ("negative" if eligible else "ineligible")}


def _environment():
    import transformers
    from transformers.models.gpt2 import modeling_gpt2

    def path_hash(value):
        return _sha(os.path.normcase(os.path.abspath(value)).encode("utf-8"))

    return {
        "python_version": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "python_executable_path_sha256": path_hash(sys.executable),
        "python_prefix_path_sha256": path_hash(sys.prefix),
        "python_base_prefix_path_sha256": path_hash(sys.base_prefix),
        "python_import_path_sha256": [path_hash(p or os.getcwd()) for p in sys.path],
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "transformers_version": transformers.__version__,
        "torch_build_sha256": _sha(torch.__config__.show().encode("utf-8")),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "gpt2_source_sha256": _file_sha(modeling_gpt2.__file__),
        "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "cuda_visibility": "disabled", "cuda_initialized": torch.cuda.is_initialized(),
        "bytecode_attested": False, "denormal_probe": _denormal_probe(),
    }


def _verify_tokenizer(tokenizer):
    for _, split, text, ids, label in CASES:
        actual = tokenizer.encode(text, add_special_tokens=False)
        if actual != list(ids) or not 1 <= len(actual) <= 32:
            raise StudyError("prompt tokenization differs from the frozen case")
        if label is not None and tokenizer.encode(label[0], add_special_tokens=False) != [label[1]]:
            raise StudyError("locality label tokenization differs")
        continuations = (label,) if split == "locality" else ((" London", LONDON), (" Paris", PARIS))
        for continuation, token_id in continuations:
            if tokenizer.encode(text + continuation, add_special_tokens=False) != list(ids) + [token_id]:
                raise StudyError("continuation does not preserve the frozen prompt token prefix")
    if tokenizer.encode(" London", add_special_tokens=False) != [LONDON]:
        raise StudyError("target tokenization differs")
    if tokenizer.encode(" Paris", add_special_tokens=False) != [PARIS]:
        raise StudyError("competitor tokenization differs")


def _verify_exposure(repo):
    """Check the published public-case reservations before any model forward."""
    for module, name in ((pretrained_exposures, "pretrained_exposures.py"),
                         (pretrained_exposures_v2, "pretrained_exposures_v2.py")):
        if Path(module.__file__).resolve() != repo / "looking_glass" / name:
            raise StudyError("exposure inventory import differs from checkout")
    if pretrained_exposures_v2.inventory_digest() != EXPOSURE_SNAPSHOT["canonical_payload_sha256"]:
        raise StudyError("exposure inventory digest differs")
    records = pretrained_exposures_v2.inventory_payload()["records"]
    prefix = "planned-parameter-edit-"
    reserved = {record["record_id"]: record for record in records if record["record_id"].startswith(prefix)}
    if len(reserved) != len(CASES):
        raise StudyError("public case reservations are incomplete")
    for index, (_name, split, _text, ids, _label) in enumerate(CASES, 1):
        role = "evaluation" if split == "target" else split
        record = reserved.get(f"{prefix}{role}-{index:02}")
        if record is None or record["category"] != "planned" or record["token_ids"] != list(ids):
            raise StudyError("public case reservation differs")
    others = [tuple(record["token_ids"]) for record in records if not record["record_id"].startswith(prefix)]
    forbidden = {row[:length] for row in others for length in range(1, len(row) + 1)}
    if any(ids in forbidden for _name, _split, _text, ids, _label in CASES):
        raise StudyError("a study row collides with the known exposure inventory")
    return {"canonical_payload_sha256": EXPOSURE_SNAPSHOT["canonical_payload_sha256"],
            "commit": EXPOSURE_SNAPSHOT["commit"], "planned_rows_checked": len(CASES),
            "known_other_prefix_collision_count": 0, "complete_history_certified": False}


def _logit_row(model, ids):
    tokens = torch.tensor([ids], dtype=torch.int64)
    with torch.no_grad():
        logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        if logits.shape != (1, len(ids), VOCAB_SIZE) or logits.dtype != torch.float32:
            raise StudyError("unexpected model output")
        row = logits[0, -1].detach().contiguous()
        if not torch.isfinite(row).all():
            raise StudyError("nonfinite model output")
        return _tensor_bytes(row)


def _run_conditions(model, weight, original, candidates, attestations, discovery, output):
    """Append each completed raw row and a matching durable journal entry."""
    journal_path = output / "observations.jsonl"
    with open(journal_path, "xb") as journal:
        for condition in CONDITIONS:
            with torch.no_grad():
                weight.copy_(original)
                if condition != "original":
                    weight.add_(candidates[condition])
                    if _tensor_sha(weight) != attestations[condition]["realized_weight_sha256"]:
                        raise StudyError("applied edit differs from precommit")
                elif _tensor_sha(weight) != _tensor_sha(original):
                    raise StudyError("original condition differs from snapshot")
            with open(output / f"{condition}.f32", "xb") as raw:
                for index, (name, split, _text, ids, _label) in enumerate(CASES):
                    _denormal_probe()
                    row = _logit_row(model, ids)
                    denormal_bits = _denormal_probe()
                    if len(row) != VOCAB_SIZE * 4:
                        raise StudyError("unexpected serialized row size")
                    if condition == "original" and index == 0:
                        saved = np.frombuffer(row, dtype="<f4").astype(np.float64)
                        saved_margin = float(saved[LONDON] - saved[PARIS])
                        if (_sha(row) != discovery["discovery_logit_sha256"] or
                                saved_margin != discovery["discovery_margin_float64"]):
                            raise StudyError("original discovery output differs from the fitted forward")
                    if raw.write(row) != len(row):
                        raise StudyError("short raw-logit write")
                    raw.flush()
                    os.fsync(raw.fileno())
                    entry = {"condition": condition, "case_id": name, "split": split,
                             "row": index, "offset": index * VOCAB_SIZE * 4,
                             "size": len(row), "sha256": _sha(row),
                             "denormal_probe": denormal_bits}
                    encoded = _canonical(entry)
                    if journal.write(encoded) != len(encoded):
                        raise StudyError("short journal write")
                    journal.flush()
                    os.fsync(journal.fileno())
            with torch.no_grad():
                weight.copy_(original)
                if _tensor_sha(weight) != _tensor_sha(original):
                    raise StudyError("condition rollback failed")


def _read_outputs(output):
    """Validate ordered raw rows and return independent metric inputs."""
    output = Path(output)
    try:
        lines = _bounded_read(output / "observations.jsonl", 1_000_000).splitlines(keepends=True)
    except OSError as exc:
        raise StudyError("observation journal is unavailable") from exc
    if len(lines) != len(CONDITIONS) * len(CASES):
        raise StudyError("observation journal is incomplete")
    matrices = {}
    row_size = VOCAB_SIZE * 4
    position = 0
    for condition in CONDITIONS:
        data = _bounded_read(output / f"{condition}.f32", len(CASES) * row_size)
        if len(data) != len(CASES) * row_size:
            raise StudyError("logit artifact has wrong size")
        for index, (name, split, _text, _ids, _label) in enumerate(CASES):
            start = index * row_size
            expected = {"condition": condition, "case_id": name, "split": split,
                        "row": index, "offset": start, "size": row_size,
                        "sha256": _sha(data[start:start + row_size]),
                        "denormal_probe": {"subnormal_times_one_bits": 1,
                                           "normal_times_half_bits": 0x00400000,
                                           "numpy_cast_preserves_min_subnormal": True}}
            if lines[position] != _canonical(expected):
                raise StudyError("observation journal differs from raw data")
            position += 1
        matrices[condition] = np.frombuffer(data, dtype="<f4").reshape(len(CASES), VOCAB_SIZE)
    return matrices


def _restore_once(model, module, weight, original, state_before, original_flags):
    with torch.no_grad():
        weight.copy_(original)
    error = _cleanup_candidate([], original_flags, weight)
    if error is not None:
        raise error
    return (_tensor_sha(weight) == _tensor_sha(original), _model_hash(model) == state_before,
            all(parameter.requires_grad == flag for parameter, flag in original_flags),
            all(parameter.grad is None for parameter, _flag in original_flags),
            not module._forward_hooks and not module._forward_pre_hooks)


def _restore_snapshot(model, module, weight, original, state_before, original_flags, caught):
    """Retry one interrupted rollback and preserve any cancellation signal."""
    names = ("weight_restored", "state_restored", "flags_restored", "grads_cleared", "hooks_restored")
    cleanup = {"attempted": True, **{name: None for name in names}}
    for _ in range(2):
        try:
            results = _restore_once(model, module, weight, original, state_before, original_flags)
            cleanup.update(dict(zip(names, results)))
            break
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt) or caught is None:
                caught = exc
    if caught is None and any(cleanup[name] is not True for name in names):
        caught = StudyError("model restoration was not verified")
    return cleanup, caught


def _reconcile_published(output, manifest_path, manifest_sha256, exc):
    try:
        audit(output, manifest_path, manifest_sha256)
        exc.published_terminal = "complete_verified"
    except KeyboardInterrupt as interrupted:
        interrupted.published_terminal = "unknown"
        exc = interrupted
    except Exception:
        exc.published_terminal = "unknown"
    return exc


def run(repo, manifest_path, manifest_sha256, cache_dir, output):
    """Run a single study; any failure or interruption remains a separate record."""
    repo = Path(repo).resolve()
    if Path(__file__).resolve() != repo / "looking_glass" / "targeted_edit_cpu_v1.py":
        raise StudyError("runner import does not match checkout")
    if Path(pretrained.__file__).resolve() != repo / "looking_glass" / "pretrained.py":
        raise StudyError("loader import does not match checkout")
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    run_id = uuid.uuid4().hex
    _json_new(output / "started.json", {"version": VERSION, "run_id": run_id,
                                        "manifest_sha256": manifest_sha256, "started_utc": started})
    phase, model, original, weight, module, state_before, original_flags = "verify", None, None, None, None, None, None
    caught = None
    cleanup = {"attempted": False, "weight_restored": None, "state_restored": None,
               "flags_restored": None, "grads_cleared": None, "hooks_restored": None}
    try:
        manifest, head = verify_manifest(repo, manifest_path, manifest_sha256)
        if (sys.byteorder != "little" or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or
                os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
            raise StudyError("CPU-only offline environment is required")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise StudyError("denormal preservation is not available")
        _denormal_probe()
        if torch.cuda.is_initialized():
            raise StudyError("CUDA was already initialized")
        environment = _environment()
        exposure = _verify_exposure(repo)
        phase = "load"
        loaded = pretrained.load_distilgpt2(cache_dir=cache_dir, allow_download=False)
        model = loaded.model
        if (model.training or any(module.training for module in model.modules()) or model.config.use_cache or
                getattr(model.config, "_attn_implementation", None) != "eager" or
                type(model).__name__ != "GPT2LMHeadModel" or
                type(loaded.tokenizer).__name__ != "GPT2TokenizerFast" or
                sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT or
                any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
                    for parameter in model.parameters()) or
                loaded.metadata["asset_sha256"] != manifest["assets"]):
            raise StudyError("checkpoint state differs from the protocol")
        _verify_tokenizer(loaded.tokenizer)
        state_before = _model_hash(model)
        module = model.transformer.h[5].mlp.c_proj
        weight = module.weight
        if (tuple(weight.shape) != WEIGHT_SHAPE or weight.dtype != torch.float32 or
                weight.device.type != "cpu" or not torch.isfinite(weight).all() or
                module._forward_hooks or module._forward_pre_hooks):
            raise StudyError("model site differs from the protocol")
        original_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        if any(parameter.grad is not None for parameter, _flag in original_flags):
            raise StudyError("model already has parameter gradients")
        original = weight.detach().clone()
        original_sha = _tensor_sha(original)
        phase = "discovery"
        _denormal_probe()
        gradient, random, facts = _candidate(model, module, CASES[0][3], LONDON, PARIS,
                                             expected_shape=WEIGHT_SHAPE)
        facts["denormal_probe_after_discovery"] = _denormal_probe()
        if _tensor_sha(weight) != original_sha or _model_hash(model) != state_before:
            raise StudyError("discovery changed model state")
        candidates = {"gradient": gradient, "random": random}
        attestations = _candidate_attestation(original, candidates)
        phase = "precommit"
        _json_new(output / "precommit.json", {
            "version": VERSION, "run_id": run_id, "manifest_sha256": manifest_sha256,
            "source_commit": manifest["source_commit"], "checkout_commit": head,
            "model": {**loaded.metadata, "evaluation_mode": True}, "environment": environment,
            "exposure_check": exposure,
            "model_state_sha256": state_before, "weight_sha256": original_sha,
            "site": SITE, "site_shape": list(weight.shape), "seed": SEED,
            "discovery": facts, "candidates": attestations,
        })
        phase = "evaluation"
        _run_conditions(model, weight, original, candidates, attestations, facts, output)
        if torch.cuda.is_initialized():
            raise StudyError("CUDA unexpectedly initialized")
    except BaseException as exc:
        caught = exc
    finally:
        if weight is not None and original is not None:
            cleanup, caught = _restore_snapshot(model, module, weight, original, state_before, original_flags, caught)
        elif caught is None:
            caught = StudyError("model restoration was not verified")
    if caught is not None:
        terminal = {"version": VERSION, "run_id": run_id,
                    "status": "cancelled" if isinstance(caught, KeyboardInterrupt) else "failed",
                    "phase": phase, "error_type": type(caught).__name__, "cleanup": cleanup}
        _failure_terminal(output, terminal)
        raise caught
    try:
        phase = "analysis"
        _denormal_probe()
        report = analyze(_read_outputs(output))
        report["denormal_probe_after_analysis"] = _denormal_probe()
        report["run_id"] = run_id
        report["manifest_sha256"] = manifest_sha256
        _json_new(output / "report.json", report)
        artifacts = {name: _file_sha(output / name) for name in (
            "started.json", "precommit.json", "observations.jsonl", "original.f32",
            "gradient.f32", "random.f32", "report.json")}
        _json_new(output / "terminal.json", {"version": VERSION, "run_id": run_id,
                  "status": "complete", "decision": report["decision"], "cleanup": cleanup,
                  "finished_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts})
    except BaseException as exc:
        if (output / "terminal.json").exists():
            # An interruption may follow a successful atomic rename. Keep the
            # committed terminal authoritative only if its artifacts verify.
            exc = _reconcile_published(output, manifest_path, manifest_sha256, exc)
        else:
            _failure_terminal(output, {"version": VERSION, "run_id": run_id,
                              "status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                              "phase": phase, "error_type": type(exc).__name__, "cleanup": cleanup})
        raise exc
    return report


def audit(output, manifest_path, manifest_sha256):
    """Recompute saved-output metrics and identity links, without attesting execution."""
    if not torch.set_flush_denormal(False):
        raise StudyError("denormal preservation is not available")
    _denormal_probe()
    output = Path(output)
    manifest, _ = _public_manifest(manifest_path, manifest_sha256)
    terminal = _read_document(output / "terminal.json")
    if terminal.get("version") != VERSION or terminal.get("status") != "complete":
        raise StudyError("run is incomplete or failed")
    if set(terminal) != {"version", "run_id", "status", "decision", "cleanup", "finished_utc", "artifacts"}:
        raise StudyError("terminal fields differ")
    required = ("started.json", "precommit.json", "observations.jsonl", "original.f32",
                "gradient.f32", "random.f32", "report.json")
    artifacts = terminal.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != set(required):
        raise StudyError("terminal artifact list differs")
    max_sizes = {name: 512_000 for name in required}
    max_sizes["observations.jsonl"] = 1_000_000
    max_sizes.update({f"{name}.f32": len(CASES) * VOCAB_SIZE * 4 for name in CONDITIONS})
    for name in required:
        digest = artifacts[name]
        if (type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None or
                _sha(_bounded_read(output / name, max_sizes[name])) != digest):
            raise StudyError("artifact digest differs")
    started = _read_document(output / "started.json")
    precommit = _read_document(output / "precommit.json")
    report = _read_document(output / "report.json")
    run_id = terminal.get("run_id")
    if (type(run_id) is not str or re.fullmatch(r"[0-9a-f]{32}", run_id) is None or
            any(record.get("run_id") != run_id for record in (started, precommit, report)) or
            any(record.get("version") != VERSION for record in (started, precommit, report)) or
            any(record.get("manifest_sha256") != manifest_sha256 for record in (started, precommit, report))):
        raise StudyError("run or manifest identities differ")
    if _canonical(terminal.get("cleanup")) != _canonical({"attempted": True,
            "weight_restored": True, "state_restored": True, "flags_restored": True,
            "grads_cleared": True, "hooks_restored": True}):
        raise StudyError("rollback was not reported as verified")
    try:
        start = datetime.fromisoformat(started["started_utc"])
        finish = datetime.fromisoformat(terminal["finished_utc"])
        if start.utcoffset() is None or finish.utcoffset() is None or finish < start:
            raise ValueError("invalid timestamp order")
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyError("run timestamps differ") from exc
    if (precommit.get("source_commit") != manifest["source_commit"] or
            precommit.get("site") != SITE or _canonical(precommit.get("site_shape")) != _canonical(list(WEIGHT_SHAPE)) or
            type(precommit.get("seed")) is not int or precommit["seed"] != SEED):
        raise StudyError("precommit identity differs from the frozen manifest")
    _commit_name(precommit.get("checkout_commit"))
    for name in ("model_state_sha256", "weight_sha256"):
        if type(precommit.get(name)) is not str or re.fullmatch(r"[0-9a-f]{64}", precommit[name]) is None:
            raise StudyError("precommit state identity is invalid")
    model = precommit.get("model")
    fixed_model = {"model_id": pretrained.MODEL_ID, "model_revision": pretrained.REVISION,
                   "tokenizer_id": pretrained.MODEL_ID, "tokenizer_revision": pretrained.REVISION,
                   "model_class": "GPT2LMHeadModel", "tokenizer_class": "GPT2TokenizerFast",
                   "weights_sha256": pretrained.ASSETS["model.safetensors"][1],
                   "asset_sha256": manifest["assets"], "device": "cpu", "dtype": "float32",
                   "attention_implementation": "eager", "use_cache": False,
                   "parameter_count": PARAMETER_COUNT, "evaluation_mode": True}
    if (type(model) is not dict or set(model) != set(fixed_model) | {"torch_version", "transformers_version"} or
            any(_canonical(model.get(key)) != _canonical(value) for key, value in fixed_model.items())):
        raise StudyError("precommit checkpoint identity differs")
    exposure = precommit.get("exposure_check")
    expected_exposure = {"canonical_payload_sha256": EXPOSURE_SNAPSHOT["canonical_payload_sha256"],
                         "commit": EXPOSURE_SNAPSHOT["commit"], "planned_rows_checked": len(CASES),
                         "known_other_prefix_collision_count": 0, "complete_history_certified": False}
    if _canonical(exposure) != _canonical(expected_exposure):
        raise StudyError("precommit exposure check differs")
    environment = precommit.get("environment")
    environment_fields = {"python_version", "platform", "machine", "processor",
                          "python_executable_path_sha256", "python_prefix_path_sha256",
                          "python_base_prefix_path_sha256", "python_import_path_sha256",
                          "torch_version", "numpy_version", "transformers_version", "torch_build_sha256",
                          "torch_cpu_capability", "gpt2_source_sha256", "intraop_threads", "interop_threads",
                          "deterministic_algorithms", "mkldnn_enabled", "omp_num_threads", "mkl_num_threads",
                          "cuda_visibility", "cuda_initialized", "bytecode_attested", "denormal_probe"}
    if (type(environment) is not dict or set(environment) != environment_fields or
            environment.get("cuda_visibility") != "disabled" or
            environment.get("cuda_initialized") is not False or
            type(environment.get("intraop_threads")) is not int or environment["intraop_threads"] != 1 or
            type(environment.get("interop_threads")) is not int or environment["interop_threads"] != 1 or
            environment.get("deterministic_algorithms") is not True or
            type(environment.get("mkldnn_enabled")) is not bool or
            environment.get("bytecode_attested") is not False or
            _canonical(environment.get("denormal_probe")) != _canonical(_denormal_probe())):
        raise StudyError("precommit execution settings differ")
    for field in ("python_version", "platform", "machine", "torch_version", "numpy_version",
                  "transformers_version", "torch_cpu_capability"):
        if type(environment[field]) is not str or not environment[field]:
            raise StudyError("runtime identity is incomplete")
    if (type(environment["processor"]) is not str or
            any(environment[field] is not None and type(environment[field]) is not str
                for field in ("omp_num_threads", "mkl_num_threads")) or
            model.get("torch_version") != environment["torch_version"] or
            model.get("transformers_version") != environment["transformers_version"]):
        raise StudyError("runtime identity is inconsistent")
    for field in ("python_executable_path_sha256", "python_prefix_path_sha256",
                  "python_base_prefix_path_sha256", "torch_build_sha256", "gpt2_source_sha256"):
        if type(environment[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", environment[field]) is None:
            raise StudyError("runtime identity digest is invalid")
    paths = environment["python_import_path_sha256"]
    if (type(paths) is not list or not paths or any(type(path) is not str or
            re.fullmatch(r"[0-9a-f]{64}", path) is None for path in paths)):
        raise StudyError("runtime import path fingerprints are invalid")
    candidates = precommit.get("candidates")
    if type(candidates) is not dict or set(candidates) != {"gradient", "random"}:
        raise StudyError("precommit candidate identities differ")
    for candidate in candidates.values():
        if type(candidate) is not dict:
            raise StudyError("invalid candidate record")
        for field in ("intended_delta_sha256", "realized_weight_sha256"):
            if type(candidate.get(field)) is not str or re.fullmatch(r"[0-9a-f]{64}", candidate[field]) is None:
                raise StudyError("invalid candidate digest")
        for field in ("realized_delta_norm", "realized_relative_norm"):
            value = candidate.get(field)
            if not _finite_number(value) or value <= 0:
                raise StudyError("invalid candidate norm")
        if candidate["realized_relative_norm"] > 0.01001:
            raise StudyError("candidate norm exceeds the bound")
    if not 0.999 <= candidates["gradient"]["realized_delta_norm"] / candidates["random"]["realized_delta_norm"] <= 1.001:
        raise StudyError("candidate norms differ")
    discovery = precommit.get("discovery")
    discovery_numbers = ("discovery_objective_float32", "discovery_margin_float64", "weight_norm",
                         "planned_delta_norm", "gradient_norm", "input_factor_norm", "output_factor_norm",
                         "gradient_outer_relative_residual")
    if (type(discovery) is not dict or
            any(not _finite_number(discovery.get(field)) for field in discovery_numbers) or
            any(discovery[field] <= 0 for field in ("weight_norm", "planned_delta_norm", "gradient_norm",
                                                    "input_factor_norm", "output_factor_norm")) or
            not 0 <= discovery["gradient_outer_relative_residual"] <= 1e-5 or
            not math.isclose(discovery["planned_delta_norm"], RELATIVE_NORM * discovery["weight_norm"], rel_tol=1e-12) or
            _canonical(discovery.get("denormal_probe_after_discovery")) != _canonical(_denormal_probe())):
        raise StudyError("precommit discovery measurements differ")
    for candidate in candidates.values():
        if not math.isclose(candidate["realized_relative_norm"],
                            candidate["realized_delta_norm"] / discovery["weight_norm"], rel_tol=1e-12):
            raise StudyError("candidate norm ratio differs")
    matrices = _read_outputs(output)
    result = analyze(matrices)
    if (discovery.get("discovery_logit_sha256") != _sha(matrices["original"][0].tobytes()) or
            discovery["discovery_margin_float64"] != result["baseline_margins"][0] or
            discovery["discovery_objective_float32"] != float(np.float32(result["baseline_margins"][0]))):
        raise StudyError("original discovery output differs from the fitted forward")
    if set(report) != set(result) | {"run_id", "manifest_sha256", "denormal_probe_after_analysis"}:
        raise StudyError("saved report fields differ")
    if (report.get("denormal_probe_after_analysis") != _denormal_probe() or
            _canonical(result) != _canonical({key: report.get(key) for key in result}) or
            terminal.get("decision") != result["decision"]):
        raise StudyError("saved metrics differ from recomputed metrics")
    return {"status": "matching_saved_outputs", "decision": result["decision"],
            "execution_attested": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="create a manifest from a committed source revision")
    freeze.add_argument("--source-commit", required=True)
    freeze.add_argument("--output", required=True)
    launch = commands.add_parser("run", help="run once using a committed frozen manifest")
    launch.add_argument("--manifest", required=True)
    launch.add_argument("--manifest-sha256", required=True)
    launch.add_argument("--cache-dir", required=True)
    launch.add_argument("--output", required=True)
    check = commands.add_parser("audit", help="recompute saved-output metrics without a model")
    check.add_argument("--output", required=True)
    check.add_argument("--manifest", required=True)
    check.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent.parent
    try:
        if args.command == "freeze":
            data = make_manifest(repo, args.source_commit)
            _json_new(args.output, data)
            result = {"manifest_sha256": _file_sha(args.output)}
        elif args.command == "run":
            result = run(repo, args.manifest, args.manifest_sha256, args.cache_dir, args.output)
        else:
            result = audit(args.output, args.manifest, args.manifest_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except KeyboardInterrupt as exc:
        print(json.dumps({"status": "cancelled", "published_terminal": getattr(exc, "published_terminal", None)}))
        return 130
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "published_terminal": getattr(exc, "published_terminal", None)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
