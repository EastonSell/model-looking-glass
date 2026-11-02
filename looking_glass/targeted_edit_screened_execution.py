"""Private, single-use screened evaluation for an already supervised owner.

The owner must independently bind the published discovery and screen records,
the current public and protected exposure checks, source and checkpoint, and
reap an isolated process. This module checks saved bytes and model state but
cannot authenticate publication, execution or cleanup. On failure the guard
is fenced and the isolated process must exit without replaying the attempt.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_projected_execution as prior
from . import targeted_edit_screened_math as projected
from . import targeted_edit_screened_score as score
from . import targeted_edit_screened_screen as screen
from . import targeted_edit_screened_selection as selection
from . import targeted_edit_v2_owner as owner


class ExecutionError(RuntimeError):
    """The screened handoff is incomplete or differs from frozen bytes."""


class _Prepared:
    def __init__(self, guard, candidate, updates, attestations, document, offsets, baseline, match):
        self.guard, self.candidate, self.updates, self.attestations = guard, candidate, updates, attestations
        self.candidate_bytes = v1._canonical(candidate)
        self.document, self.offsets, self.baseline, self.match = document, tuple(offsets), tuple(baseline), match
        self.attestation_bytes = v1._canonical(attestations)
        self.match_bytes = v1._canonical(match)
        self.baseline_hashes = tuple(hashlib.sha256(row).hexdigest() for row in baseline)
        self.frozen_offsets = tuple(offsets)
        self.used = False


def _bytes(path, limit):
    return prior._read(path, limit)


def _update(guard, value, attestation):
    if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.float32
            or value.shape != guard.original.shape or not bool(torch.isfinite(value).all())
            or v1._tensor_sha(value) != attestation["intended_delta_sha256"]):
        raise ExecutionError("intended update differs from frozen construction")
    before = v1._denormal_probe()
    with torch.no_grad():
        applied = guard.original + value
        realized = applied.to(torch.float64) - guard.original.to(torch.float64)
    after = v1._denormal_probe()
    if (before != after or v1._tensor_sha(guard.original) != guard.weight_sha
            or v1._tensor_sha(applied) != attestation["realized_weight_sha256"]
            or projected._planned_sha(realized) != attestation["realized_delta_float64_sha256"]):
        raise ExecutionError("realized update differs from frozen construction")


def _screen(guard, directory, expected_sha, document, candidate_sha):
    directory = Path(directory)
    raw = _bytes(directory / "screen-selection.json", 1_000_000)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ExecutionError("screen selection digest differs")
    record = json.loads(raw)
    if (record.get("status") != "private-screen" or record.get("case_sha256") != screen.CASE_SHA256
            or record.get("identity", {}).get("candidate_sha256") != candidate_sha):
        raise ExecutionError("screen binding differs")
    for name in ("screen.f32", "observations.jsonl", "state-original.json"):
        item = record["artifacts"][name]
        data = _bytes(directory / name, item["bytes"])
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ExecutionError("screen artifact differs")
        if name == "screen.f32":
            matrix = data
        elif name == "state-original.json":
            state = json.loads(data)
    if (state["model_state_sha256"] != guard.state_sha or state["weight_sha256"] != guard.weight_sha
            or state["buffer_state_sha256"] != guard.buffer_sha or state["runtime_settings"] != guard.runtime_settings
            or state["cleanup"] != "verified"):
        raise ExecutionError("screen original state differs")
    fixed = screen._targets(document)
    size = v1.VOCAB_SIZE * 4
    if len(matrix) != 32 * size:
        raise ExecutionError("screen matrix size differs")
    rows, margins = [], []
    for i in range(32):
        row = matrix[i * size:(i + 1) * size]
        values = np.frombuffer(row, dtype="<f4").astype(np.float64)
        if not np.isfinite(values).all():
            raise ExecutionError("screen row is nonfinite")
        margins.append(float(values[v1.LONDON] - values[v1.PARIS]))
        rows.append(row)
    chosen = selection.select_screen(fixed, margins)
    recorded = record["selection"]
    for key, value in chosen.items():
        if key == "rows":
            if len(value) != len(recorded[key]):
                raise ExecutionError("screen row count differs")
            for i, (expected, actual) in enumerate(zip(value, recorded[key])):
                if any(actual.get(k) != v for k, v in expected.items()):
                    raise ExecutionError("screen selection row differs")
                if (actual.get("logit_sha256") != hashlib.sha256(rows[i]).hexdigest()
                        or actual.get("logit_byte_offset") != i * size or actual.get("logit_bytes") != size):
                    raise ExecutionError("screen row hash or offset differs")
        elif recorded.get(key) != value:
            raise ExecutionError("screen round-robin selection differs")
    offsets = chosen["selected_offsets"]
    return offsets, [rows[i] for i in offsets]


def reconstruct(guard, candidate_dir, candidate_sha, screen_dir, screen_sha, document):
    """Reconstruct frozen directions and validate every selected screen row."""
    try:
        guard.verify()
        candidate = prior._candidate(candidate_dir, candidate_sha)
        if (candidate.get("version") != "targeted-edit-screened-development-v1"
                or candidate["model_state_sha256"] != guard.state_sha
                or candidate["weight_sha256"] != guard.weight_sha
                or candidate["buffer_state_sha256"] != guard.buffer_sha
                or candidate["runtime_settings"] != guard.runtime_settings
                or (candidate["target_token_id"], candidate["competitor_token_id"]) != (v1.LONDON, v1.PARIS)):
            raise ExecutionError("frozen development identity differs")
        offsets, baseline = _screen(guard, screen_dir, screen_sha, document, candidate_sha)
        saved = lambda name: prior._saved_tensor(candidate_dir, candidate, name)
        x, h, g, gradient = (saved(name) for name in
                            ("calibration-inputs.f32", "input-factor.f32", "output-factor.f32", "full-gradient.f32"))
        built = projected.prepare_projected_edits(guard.original, x, h, g, gradient)
        grid = projected.prepare_unprojected_grid(guard.original, x, h, g, gradient, built)
        if (v1._canonical(built["attestation"]) != v1._canonical(candidate["construction"])
                or v1._canonical(grid["attestation"]) != v1._canonical(candidate["grid_construction"])):
            raise ExecutionError("reconstructed construction differs")
        original = saved("discovery-logits.f32")
        margin = float(original[v1.LONDON].to(torch.float64) - original[v1.PARIS].to(torch.float64))
        if margin != candidate["discovery_margin"]:
            raise ExecutionError("saved discovery baseline differs")
        measured = []
        for i, dose in enumerate(projected.DOSE_GRID, 1):
            condition = "unprojected" if dose == projected.DOSE else f"unprojected-grid-{i:02d}"
            row = saved("discovery-" + condition + "-logits.f32")
            gain = float(row[v1.LONDON].to(torch.float64) - row[v1.PARIS].to(torch.float64)) - margin
            measured.append({"dose": dose, "condition": condition, "gain": gain})
        row = saved("discovery-projected-logits.f32")
        projected_gain = float(row[v1.LONDON].to(torch.float64) - row[v1.PARIS].to(torch.float64)) - margin
        match = projected.select_discovery_match(projected_gain,
                    [{"dose": item["dose"], "gain": item["gain"]} for item in measured])
        if (v1._canonical(measured) != v1._canonical(candidate["grid_measurements"])
                or v1._canonical(match) != v1._canonical(candidate["discovery_match"])):
            raise ExecutionError("saved discovery match differs")
        updates = dict(built["updates"])
        attestations = dict(built["attestation"]["conditions"])
        if match["available"] and not match["shared_equal_norm_arm"]:
            dose = match["selected_dose"]
            updates["matched"] = grid["updates"][dose]
            attestations["matched"] = next(item for item in grid["attestation"]["candidates"] if item["dose"] == dose)
        for name, value in updates.items():
            _update(guard, value, attestations[name])
        guard.verify()
        return _Prepared(guard, candidate, updates, attestations, document, offsets, baseline, match)
    except BaseException:
        guard.poisoned = True
        raise


def evaluate(prepared, output):
    """Write full logits and a provisional decision, reusing screen bytes."""
    if not isinstance(prepared, _Prepared) or prepared.used:
        raise ExecutionError("a fresh screened handoff is required")
    prepared.used = True
    guard = prepared.guard
    try:
        expected_updates = set(projected.CONDITIONS)
        if prepared.match["available"] and not prepared.match["shared_equal_norm_arm"]:
            expected_updates.add("matched")
        if (v1._canonical(prepared.candidate) != prepared.candidate_bytes
                or v1._canonical(prepared.attestations) != prepared.attestation_bytes
                or v1._canonical(prepared.match) != prepared.match_bytes
                or set(prepared.updates) != expected_updates or set(prepared.attestations) != expected_updates
                or prepared.offsets != prepared.frozen_offsets
                or tuple(hashlib.sha256(row).hexdigest() for row in prepared.baseline) != prepared.baseline_hashes):
            raise ExecutionError("the frozen handoff changed")
        for name, value in prepared.updates.items():
            _update(guard, value, prepared.attestations[name])
        fixed = score.cases(prepared.document, list(prepared.offsets))
        count = len(prepared.offsets)
        names = score.BASE_CONDITIONS + (("matched",) if prepared.match["available"] else ())
        output = Path(output)
        output.mkdir(exist_ok=False)
        matrices, states = {}, {}
        with (output / "observations.jsonl").open("xb") as journal:
            for name in names:
                if name == "matched" and prepared.match["shared_equal_norm_arm"]:
                    matrices[name] = matrices["unprojected"].copy()
                    reused = _bytes(output / "unprojected.f32", len(fixed) * v1.VOCAB_SIZE * 4)
                    owner._write_new(output / "matched.f32", reused)
                    for i, case in enumerate(fixed):
                        data = reused[i * v1.VOCAB_SIZE * 4:(i + 1) * v1.VOCAB_SIZE * 4]
                        probe = v1._denormal_probe()
                        owner._append(journal, v1._canonical({"condition": name, "case_id": case[0],
                            "index": i, "offset": i * len(data), "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(), "source": "unprojected",
                            "screen_offset": None, "denormal_probe_before": probe, "denormal_probe_after": probe}))
                    states[name] = {**states["unprojected"], "condition": name, "reused_condition": "unprojected"}
                    owner._json_new(output / "state-matched.json", states[name])
                    continue
                update = None if name == "original" else prepared.updates[name]
                if v1._canonical(prepared.attestations) != prepared.attestation_bytes:
                    raise ExecutionError("the frozen update attestations changed")
                applied = guard.weight_sha if name == "original" else prepared.attestations[name]["realized_weight_sha256"]
                if update is not None:
                    _update(guard, update, prepared.attestations[name])
                addition = {"before": v1._denormal_probe()}
                rows = []
                with (output / (name + ".f32")).open("xb") as raw:
                    def forward_rows():
                        addition["after"] = v1._denormal_probe()
                        for i, case in enumerate(fixed):
                            before = v1._denormal_probe()
                            reused = name == "original" and i < count
                            data = prepared.baseline[i] if reused else v1._logit_row(guard.model, case[4])
                            after = v1._denormal_probe()
                            if type(data) is not bytes or len(data) != v1.VOCAB_SIZE * 4:
                                raise ExecutionError("evaluation row size differs")
                            values = np.frombuffer(data, dtype="<f4")
                            if not np.isfinite(values).all():
                                raise ExecutionError("evaluation row is nonfinite")
                            owner._append(raw, data)
                            owner._append(journal, v1._canonical({"condition": name, "case_id": case[0],
                                "index": i, "offset": i * len(data), "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "source": "screen" if reused else "forward",
                                "screen_offset": prepared.offsets[i] if reused else None,
                                "denormal_probe_before": before, "denormal_probe_after": after}))
                            rows.append(values.copy())
                    guard.condition(update, applied, forward_rows)
                state = {"condition": name, "applied_weight_sha256": applied,
                         "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                         "buffer_state_sha256": guard.buffer_sha, "runtime_settings": guard.runtime_settings,
                         "addition_denormal_probe_before": addition["before"],
                         "addition_denormal_probe_after": addition["after"], "cleanup": guard.last_cleanup}
                states[name] = state
                owner._json_new(output / ("state-" + name + ".json"), state)
                guard.commit_condition()
                matrices[name] = np.stack(rows)
        result = score.evaluate(matrices, prepared.document, list(prepared.offsets), prepared.match["available"])
        result["discovery_match"] = dict(prepared.match)
        guard.verify()
        owner._json_new(output / "provisional-report.json", result)
        return result
    except BaseException:
        guard.poisoned = True
        raise
