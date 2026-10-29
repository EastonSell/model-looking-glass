"""Two-stage projected-edit computations for an already admitted CPU owner.

This module neither launches a process nor establishes source, checkpoint,
publication, or cleanup provenance. The caller exclusively owns the model,
the fixed rows and output directories; it must supervise and reap the owner,
publish the development inventory before evaluation, and retain partial runs.
No evaluation input is forwarded until the saved development is reconstructed
and all original model, activation, gradient and edit identities agree.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_projected_capture as capture
from . import targeted_edit_projected_development as development
from . import targeted_edit_projected_math as projected
from . import targeted_edit_projected_score as score
from . import targeted_edit_v2_owner as owner


class ExecutionError(RuntimeError):
    """A phase is incomplete or differs from the fixed development record."""


class _Prepared:
    """Private single-use, in-process handoff, not publication evidence."""
    def __init__(self, guard, candidate, updates):
        self.guard = guard
        self.candidate = candidate
        self.candidate_bytes = v1._canonical(candidate)
        self.updates = updates
        self.used = False


def _read(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ExecutionError("saved artifact exceeds its bound")
    return data


def _candidate(directory, expected_sha256):
    """Read every candidate artifact and bind it to its recorded bytes."""
    directory = Path(directory)
    raw = _read(directory / "pre-evaluation-candidate.json", 1_000_000)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ExecutionError("development candidate digest differs")
    try:
        candidate = v1._decode_document(raw)
        inventory = candidate["artifacts"]
        if (candidate["status"] != "private-candidate" or type(inventory) is not dict
                or not 1 <= len(inventory) <= 100):
            raise ExecutionError("development inventory is invalid")
        total = 0
        for name, item in inventory.items():
            if (type(name) is not str or Path(name).name != name or name in ("", ".", "..")
                    or type(item) is not dict or type(item.get("bytes")) is not int
                    or not 0 <= item["bytes"] <= 10_000_000):
                raise ExecutionError("development artifact entry is invalid")
            total += item["bytes"]
            if total > 50_000_000:
                raise ExecutionError("development inventory exceeds its bound")
            data = _read(directory / name, item["bytes"])
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item.get("sha256"):
                raise ExecutionError("development artifact differs")
    except (KeyError, TypeError, ValueError, OSError) as error:
        if isinstance(error, ExecutionError):
            raise
        raise ExecutionError("development candidate is invalid") from error
    return candidate


def _same_tensor(tensor, candidate, name):
    """Compare the fresh CPU float32 tensor with the saved artifact identity."""
    item = candidate["artifacts"][name]
    if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32 or list(tensor.shape) != item.get("shape")
            or item.get("dtype") != "float32-le" or v1._tensor_sha(tensor) != item.get("sha256")):
        raise ExecutionError("reconstructed development tensor differs: " + name)


def _saved_tensor(directory, candidate, name):
    item = candidate["artifacts"][name]
    if item.get("dtype") != "float32-le" or type(item.get("shape")) is not list:
        raise ExecutionError("saved tensor schema differs")
    shape = item["shape"]
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ExecutionError("saved tensor shape differs")
    data = _read(Path(directory) / name, item["bytes"])
    values = np.frombuffer(data, dtype="<f4")
    if values.size != int(np.prod(shape, dtype=np.int64)) or not np.isfinite(values).all():
        raise ExecutionError("saved tensor size or values differ")
    return torch.from_numpy(values.copy().reshape(shape))


def _check_update(guard, candidate, condition, delta):
    item = candidate["construction"]["conditions"][condition]
    if (type(delta) is not torch.Tensor or delta.device.type != "cpu"
            or delta.dtype != torch.float32 or delta.shape != guard.original.shape
            or not bool(torch.isfinite(delta).all())
            or v1._tensor_sha(delta) != item["intended_delta_sha256"]):
        raise ExecutionError("intended update differs from development")
    before = v1._denormal_probe()
    with torch.no_grad():
        applied = guard.original + delta
    after = v1._denormal_probe()
    if (before != after or v1._tensor_sha(guard.original) != guard.weight_sha
            or v1._tensor_sha(applied) != item["realized_weight_sha256"]):
        raise ExecutionError("realized update differs from development")
    return applied


def _check_saved_measurements(guard, directory, candidate, updates, target, competitor):
    """Recompute the saved discovery and calibration values used by scoring."""
    x = _saved_tensor(directory, candidate, "calibration-inputs.f32")
    original = _saved_tensor(directory, candidate, "calibration-original-outputs.f32")
    initial_probe = v1._denormal_probe()
    for condition in projected.CONDITIONS:
        delta = updates[condition]
        applied = _check_update(guard, candidate, condition, delta)
        with torch.no_grad():
            actual = applied.to(torch.float64) - guard.original.to(torch.float64)
            algebraic = torch.max(torch.abs(x.to(torch.float64) @ actual), dim=1).values.tolist()
            condition_output = _saved_tensor(directory, candidate, "calibration-" + condition + "-outputs.f32")
            if condition_output.shape != original.shape:
                raise ExecutionError("calibration output shape differs")
            difference = condition_output.to(torch.float64) - original.to(torch.float64)
            observed = torch.max(torch.abs(difference), dim=1).values.tolist()
        saved = candidate["calibration_shifts"][condition]
        if (v1._canonical(saved["max_output_abs_by_token"]) != v1._canonical(observed)
                or saved["max_output_abs"] != max(observed)
                or v1._canonical(saved["max_algebraic_abs_by_token"]) != v1._canonical(algebraic)
                or saved["max_algebraic_abs"] != max(algebraic)):
            raise ExecutionError("saved calibration measurement differs")
        row = _saved_tensor(directory, candidate, "discovery-" + condition + "-logits.f32")
        if row.ndim != 1 or max(target, competitor) >= row.numel():
            raise ExecutionError("saved discovery row shape differs")
        margin = float(row[target].to(torch.float64) - row[competitor].to(torch.float64))
        saved_change = candidate["discovery_changes"][condition]
        if (saved_change["margin"] != margin
                or saved_change["change"] != margin - candidate["discovery_margin"]):
            raise ExecutionError("saved discovery measurement differs")
    if v1._denormal_probe() != initial_probe:
        raise ExecutionError("numeric mode changed while checking development")


def reconstruct(guard, directory, expected_sha256, discovery_ids, calibration_rows, target, competitor):
    """Recreate development from the original model before new-case forwards.

    A failure fences the model even when cleanup appears successful; the
    isolated owner must exit without further forwards. The committed candidate
    and all files are caller-bound to the independently reviewed publication.
    """
    try:
        candidate = _candidate(directory, expected_sha256)
        if (candidate["discovery_token_ids"] != list(discovery_ids)
                or candidate["calibration_token_ids"] != [list(row) for row in calibration_rows]
                or candidate["target_token_id"] != target or candidate["competitor_token_id"] != competitor
                or candidate["model_state_sha256"] != guard.state_sha
                or candidate["weight_sha256"] != guard.weight_sha
                or candidate["buffer_state_sha256"] != guard.buffer_sha
                or v1._canonical(candidate["runtime_settings"]) != v1._canonical(guard.runtime_settings)):
            raise ExecutionError("original model or development inputs differ")
        guard.verify()
        before = v1._denormal_probe()
        discovery = capture.capture_discovery(guard.model, guard.module, discovery_ids, target, competitor)
        after = v1._denormal_probe()
        guard.verify()
        for key, name in (("logits", "discovery-logits.f32"),
                          ("site_inputs", "discovery-grad-inputs.f32"),
                          ("site_outputs", "discovery-grad-outputs.f32"),
                          ("input_factor", "input-factor.f32"),
                          ("output_factor", "output-factor.f32"),
                          ("gradient", "full-gradient.f32")):
            _same_tensor(discovery[key], candidate, name)
        if (discovery["margin"] != candidate["discovery_margin"]
                or before != candidate["discovery_denormal_probe_before"]
                or after != candidate["discovery_denormal_probe_after"]):
            raise ExecutionError("discovery measurement differs")
        observed = {}
        def baseline_rows():
            observed["calibration"], observed["probes"] = development._calibration_rows(guard, calibration_rows)
            observed["discovery"], observed["discovery_probe"] = development._no_grad_discovery(
                guard, discovery_ids, discovery["logits"].numel())
        guard.condition(None, guard.weight_sha, baseline_rows)
        for key, name in (("inputs", "calibration-inputs.f32"),
                          ("outputs", "calibration-original-outputs.f32"),
                          ("logits", "calibration-original-logits.f32")):
            _same_tensor(observed["calibration"][key], candidate, name)
        for key, name in (("logits", "discovery-original-logits.f32"),
                          ("inputs", "discovery-original-inputs.f32"),
                          ("outputs", "discovery-original-outputs.f32")):
            _same_tensor(observed["discovery"][key], candidate, name)
        guard.commit_condition()
        built = projected.prepare_projected_edits(
            guard.original, observed["calibration"]["inputs"], discovery["input_factor"],
            discovery["output_factor"], discovery["gradient"])
        if v1._canonical(built["attestation"]) != v1._canonical(candidate["construction"]):
            raise ExecutionError("reconstructed edits differ from development")
        _check_saved_measurements(guard, directory, candidate, built["updates"], target, competitor)
        guard.verify()
        return _Prepared(guard, candidate, built["updates"])
    except BaseException:
        guard.poisoned = True
        raise


def evaluate(prepared, public_cases, output):
    """Write all six conditions and provisional scores; no final outcome claim."""
    if not isinstance(prepared, _Prepared) or prepared.used:
        raise ExecutionError("a fresh reconstructed development handoff is required")
    prepared.used = True
    guard, candidate, updates = prepared.guard, prepared.candidate, prepared.updates
    try:
        if (v1._canonical(candidate) != prepared.candidate_bytes or type(updates) is not dict
                or set(updates) != set(projected.CONDITIONS)):
            raise ExecutionError("the reconstructed handoff changed")
        guard.verify()
        for condition in projected.CONDITIONS:
            _check_update(guard, candidate, condition, updates[condition])
        cases = score._cases(public_cases)
        output = Path(output)
        output.mkdir(exist_ok=False)
        logits = {}
        with (output / "observations.jsonl").open("xb") as journal:
            for condition in score.CONDITIONS:
                applied = guard.weight_sha if condition == "original" else candidate["construction"]["conditions"][condition]["realized_weight_sha256"]
                if condition != "original":
                    _check_update(guard, candidate, condition, updates[condition])
                addition_probes = {"before": v1._denormal_probe()}
                matrix = []
                with (output / (condition + ".f32")).open("xb") as raw:
                    def rows():
                        addition_probes["after"] = v1._denormal_probe()
                        for index, (case_id, _role, _prompt, ids, _label) in enumerate(cases):
                            before = v1._denormal_probe()
                            row = v1._logit_row(guard.model, ids)
                            after = v1._denormal_probe()
                            if len(row) != v1.VOCAB_SIZE * 4:
                                raise ExecutionError("evaluation logit row has the wrong size")
                            owner._append(raw, row)
                            owner._append(journal, v1._canonical({"condition": condition, "case_id": case_id,
                                "index": index, "offset": index * len(row), "bytes": len(row),
                                "sha256": hashlib.sha256(row).hexdigest(),
                                "denormal_probe_before": before, "denormal_probe_after": after}))
                            matrix.append(np.frombuffer(row, dtype="<f4").copy())
                    guard.condition(None if condition == "original" else updates[condition], applied, rows)
                owner._json_new(output / ("state-" + condition + ".json"), {
                    "condition": condition, "applied_weight_sha256": applied,
                    "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                    "buffer_state_sha256": guard.buffer_sha, "runtime_settings": guard.runtime_settings,
                    "addition_denormal_probe_before": addition_probes["before"],
                    "addition_denormal_probe_after": addition_probes["after"], "cleanup": guard.last_cleanup})
                guard.commit_condition()
                logits[condition] = np.stack(matrix)
        report = score.evaluate(logits, public_cases, candidate)
        guard.verify()
        owner._json_new(output / "provisional-report.json", report)
        return report
    except BaseException:
        guard.poisoned = True
        raise
