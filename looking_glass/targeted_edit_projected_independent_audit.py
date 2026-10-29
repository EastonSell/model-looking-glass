"""Independently inspect saved projected-study evidence without a model.

The reader checks recorded data and attestations, not executed-code authenticity,
remote publication timing, or protection from concurrent local modification.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from datetime import datetime

CONDITIONS = ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance")
VOCAB_SIZE = 50257
LONDON, PARIS = 3576, 6342
DENORMAL_PROBE = {"subnormal_times_one_bits": 1, "normal_times_half_bits": 4194304,
                  "numpy_cast_preserves_min_subnormal": True}
V1_RUNTIME = {"python_version": "3.11.15", "platform": "Windows-10-10.0.26200-SP0",
              "machine": "AMD64", "processor": "AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD",
              "torch_version": "2.8.0+cu128", "numpy_version": "2.4.6", "transformers_version": "4.57.3",
              "torch_build_sha256": "9835a6eed581f616d1e294cc42162b701425ea24bf4723678b8233e78f30e8c6",
              "torch_cpu_capability": "AVX2",
              "gpt2_source_sha256": "b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80",
              "intraop_threads": 1, "interop_threads": 1, "deterministic_algorithms": True,
              "mkldnn_enabled": True, "omp_num_threads": None, "mkl_num_threads": None,
              "cuda_visibility": "disabled", "cuda_initialized": False, "bytecode_attested": False,
              "denormal_probe": DENORMAL_PROBE}


class EvidenceError(ValueError):
    """Saved evidence is incomplete, inconsistent, or outside its scope."""


def _ordinary_path(path):
    try:
        lexical = Path(os.path.abspath(path))
        for component in (lexical, *lexical.parents):
            info = component.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise EvidenceError("redirected evidence path is unsupported")
        return lexical
    except OSError as error:
        raise EvidenceError("evidence path is unavailable") from error


def read_ordinary(path, maximum):
    """Read a bounded ordinary file under the documented quiescence assumption."""
    if type(maximum) is not int or maximum < 0:
        raise EvidenceError("invalid evidence bound")
    lexical = _ordinary_path(path)
    try:
        if not stat.S_ISREG(lexical.lstat().st_mode):
            raise EvidenceError("evidence path is not an ordinary file")
        with lexical.open("rb") as stream:
            data = stream.read(maximum + 1)
    except OSError as error:
        raise EvidenceError("evidence file could not be read") from error
    if len(data) > maximum:
        raise EvidenceError("evidence file exceeds its bound")
    return data


def snapshot_files(directory, inventory, *, optional_empty_dirs=()):
    """Read an exact bounded inventory, rejecting extra files before any file read."""
    if type(inventory) is not dict or not 1 <= len(inventory) <= 100:
        raise EvidenceError("artifact inventory count differs")
    root = _ordinary_path(directory)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise EvidenceError("artifact root is not an ordinary directory")
    expected_dirs, total = set(), 0
    for name, item in inventory.items():
        if type(name) is not str or not name or "\\" in name or ":" in name:
            raise EvidenceError("artifact name differs")
        parts = name.split("/")
        if any(part in ("", ".", "..") or re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None for part in parts):
            raise EvidenceError("artifact name differs")
        expected_dirs.update("/".join(parts[:index]) for index in range(1, len(parts)))
        if type(item) is not dict or type(item.get("bytes")) is not int or not 0 <= item["bytes"] <= 10_000_000:
            raise EvidenceError("artifact size differs")
        _digest(item.get("sha256"))
        total += item["bytes"]
    if total > 100_000_000:
        raise EvidenceError("artifact inventory exceeds its bound")
    if (type(optional_empty_dirs) is not tuple
            or any(type(name) is not str or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None
                   for name in optional_empty_dirs)):
        raise EvidenceError("optional directory list differs")
    found_files, found_dirs = set(), set()
    def scan(path):
        for child in path.iterdir():
            checked = _ordinary_path(child)
            info = checked.lstat()
            name = checked.relative_to(root).as_posix()
            if stat.S_ISDIR(info.st_mode):
                found_dirs.add(name)
                if name not in expected_dirs and name not in optional_empty_dirs:
                    raise EvidenceError("unlisted artifact directory")
                scan(checked)
            elif stat.S_ISREG(info.st_mode):
                found_files.add(name)
            else:
                raise EvidenceError("unsupported artifact entry")
    try:
        scan(root)
    except OSError as error:
        raise EvidenceError("artifact directory could not be read") from error
    if found_files != set(inventory) or not expected_dirs <= found_dirs:
        raise EvidenceError("artifact file set differs")
    result = {}
    for name, item in inventory.items():
        data = read_ordinary(root.joinpath(*name.split("/")), item["bytes"])
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise EvidenceError("artifact bytes differ from the recorded inventory")
        result[name] = data
    return result


def read_phase_snapshot(directory, expected_sha256, *, phase):
    """Read a complete phase with its exact protocol-defined artifact inventory.

    The caller must ensure the directory is quiescent. A clean saved record is a
    cooperative attestation; it does not independently authenticate execution.
    """
    _digest(expected_sha256)
    if phase not in ("development", "evaluation"):
        raise EvidenceError("phase differs")
    raw = read_ordinary(Path(directory) / "phase-record.json", 1_000_000)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise EvidenceError("phase record hash differs")
    record = parse_document(raw)
    if (record.get("version") != "targeted-edit-projected-phase-v1"
            or record.get("status") != "complete" or record.get("phase") != phase):
        raise EvidenceError("phase is not recorded complete")
    common = {"started.json", "child-result.json", "manifest-review-attestation.json",
              "public-exposure.json", "public-exposure-attestation.json"}
    if phase == "development":
        candidate_files = {
            "discovery-logits.f32", "discovery-grad-inputs.f32", "discovery-grad-outputs.f32",
            "input-factor.f32", "output-factor.f32", "full-gradient.f32",
            "discovery-original-logits.f32", "discovery-original-inputs.f32", "discovery-original-outputs.f32",
            "calibration-inputs.f32", "calibration-original-outputs.f32", "calibration-original-logits.f32",
            "state-original.json", "factor-calibration_basis.f64", "factor-projected_input.f64",
            "factor-random_input.f64", "factor-random_output.f64", "factor-projected_random_input.f64",
            "factor-nuisance_basis.f64", "factor-nuisance_input.f64"}
        for name in CONDITIONS[1:]:
            candidate_files.update({f"calibration-{name}-outputs.f32", f"calibration-{name}-logits.f32",
                                    f"discovery-{name}-logits.f32", f"discovery-{name}-inputs.f32",
                                    f"discovery-{name}-outputs.f32", f"state-{name}.json"})
        required = common | {"candidate/pre-evaluation-candidate.json"} | {
            "candidate/" + name for name in candidate_files}
    else:
        required = common | {"review-attestation.json", "evaluation/observations.jsonl",
                             "evaluation/provisional-report.json"}
        required |= {f"evaluation/{name}.f32" for name in CONDITIONS}
        required |= {f"evaluation/state-{name}.json" for name in CONDITIONS}
    inventory = record.get("artifacts")
    if type(inventory) is not dict or set(inventory) != required:
        raise EvidenceError("phase artifact inventory differs")
    all_files = {**inventory, "phase-record.json": {"bytes": len(raw), "sha256": expected_sha256}}
    return record, snapshot_files(directory, all_files, optional_empty_dirs=("pycache",))


def decode_float32_bits(bits):
    """Decode finite IEEE binary32 without the caller's float32 FTZ/DAZ mode."""
    if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF:
        raise EvidenceError("invalid binary32 word")
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF:
        raise EvidenceError("nonfinite saved logit")
    value = (math.ldexp(fraction, -149) if exponent == 0
             else math.ldexp((1 << 23) | fraction, exponent - 150))
    return -value if bits & 0x80000000 else value


def score_row(raw):
    """Decode one row and compute a stable, shifted binary64 log-softmax."""
    if type(raw) is not bytes or not raw or len(raw) % 4:
        raise EvidenceError("saved row is not a nonempty binary32 array")
    values = [decode_float32_bits(bits) for (bits,) in struct.iter_unpack("<I", raw)]
    top = max(range(len(values)), key=values.__getitem__)
    maximum = values[top]
    shifted = [value - maximum for value in values]
    normalizer = math.log(math.fsum(math.exp(value) for value in shifted))
    return values, [value - normalizer for value in shifted], top


def verify_journal(journal, matrices, case_ids, *, row_bytes):
    """Check every ordered row against the frozen IDs and actual matrix bytes."""
    if (type(journal) is not bytes or len(journal) > 2_000_000
            or type(matrices) is not dict or set(matrices) != set(CONDITIONS)
            or type(case_ids) is not list or len(case_ids) != 13
            or any(type(case_id) is not str or not case_id for case_id in case_ids)
            or len(set(case_ids)) != 13 or type(row_bytes) is not int or row_bytes <= 0 or row_bytes % 4):
        raise EvidenceError("journal inputs differ from the fixed protocol")
    lines = journal.splitlines(keepends=True)
    if len(lines) != len(CONDITIONS) * len(case_ids):
        raise EvidenceError("journal row count differs")
    position = 0
    for condition in CONDITIONS:
        data = matrices[condition]
        if type(data) is not bytes or len(data) != len(case_ids) * row_bytes:
            raise EvidenceError("condition matrix size differs")
        for index, case_id in enumerate(case_ids):
            offset = index * row_bytes
            row = data[offset:offset + row_bytes]
            expected = {"condition": condition, "case_id": case_id, "index": index,
                        "offset": offset, "bytes": row_bytes, "sha256": hashlib.sha256(row).hexdigest(),
                        "denormal_probe_before": DENORMAL_PROBE, "denormal_probe_after": DENORMAL_PROBE}
            encoded = (json.dumps(expected, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
            if lines[position] != encoded:
                raise EvidenceError("journal row identity, hash, or numeric probe differs")
            position += 1
    return position


def _finite(value, *, nonnegative=False):
    if type(value) not in (int, float):
        raise EvidenceError("numeric evidence has an invalid type")
    try:
        value = float(value)
    except OverflowError as error:
        raise EvidenceError("numeric evidence is outside the finite range") from error
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise EvidenceError("numeric evidence is nonfinite or negative")
    return value


def numerical_decision(summaries, candidate):
    """Apply the prospective eligibility and success rules to derived metrics."""
    try:
        edited, random, unprojected = (summaries["projected"], summaries["projected_random"],
                                       summaries["unprojected"])
        discovery = _finite(candidate["discovery_margin"])
        shifts = candidate["calibration_shifts"]
        projected_shift = _finite(shifts["projected"]["max_output_abs"], nonnegative=True)
        unprojected_shift = _finite(shifts["unprojected"]["max_output_abs"], nonnegative=True)
        negative, reversed_count = edited["originally_negative_targets"], edited["reversed_targets"]
        preserved = edited["all_top_preserved"]
        if (type(negative) is not int or not 0 <= negative <= 4
                or type(reversed_count) is not int or not 0 <= reversed_count <= negative
                or type(preserved) is not bool):
            raise EvidenceError("target or locality counts differ")
        gain = _finite(edited["mean_target_change"])
        random_gain = _finite(random["mean_target_change"])
        mean_kl = _finite(edited["mean_kl"])
        max_kl = _finite(edited["max_kl"])
        nll = _finite(edited["mean_nll_deterioration"])
        unprojected_kl = _finite(unprojected["mean_kl"])
    except (KeyError, TypeError) as error:
        raise EvidenceError("required numerical evidence is missing") from error
    eligibility = {"baseline_discovery_negative": discovery < 0,
                   "baseline_three_targets_negative": negative >= 3,
                   "unprojected_calibration_shift": unprojected_shift > 1e-6}
    criteria = {"at_least_two_targets_reversed": reversed_count >= 2,
                "mean_target_gain": gain > 0.5,
                "random_control_advantage": gain - random_gain >= 0.5,
                "all_locality_top_preserved": preserved,
                "mean_locality_kl": mean_kl <= 0.02,
                "max_locality_kl": max_kl <= 0.10,
                "mean_locality_nll_deterioration": nll <= 0.05,
                "calibration_shift_ratio": projected_shift <= 0.1 * unprojected_shift,
                "locality_kl_advantage": unprojected_kl - mean_kl >= 1e-6}
    decision = "ineligible" if not all(eligibility.values()) else (
        "success" if all(criteria.values()) else "negative")
    return {"eligibility": eligibility, "criteria": criteria, "decision": decision}


def score_matrices(matrices, cases, candidate):
    """Independently derive every target/locality metric from the six raw matrices."""
    if (type(matrices) is not dict or set(matrices) != set(CONDITIONS)
            or type(cases) is not list or len(cases) != 13):
        raise EvidenceError("fixed evaluation inputs differ")
    row_bytes = VOCAB_SIZE * 4
    if any(type(data) is not bytes or len(data) != 13 * row_bytes for data in matrices.values()):
        raise EvidenceError("evaluation matrix size differs")
    try:
        ids = [case["case_id"] for case in cases]
        if (any(type(value) is not str or not value for value in ids) or len(set(ids)) != 13
                or [case["split"] for case in cases] != ["target"] * 4 + ["locality"] * 9):
            raise EvidenceError("evaluation case identities differ")
        labels = []
        for case in cases[4:]:
            ids_for_label = case["continuations"][0]["ids"]
            if (type(ids_for_label) is not list or len(ids_for_label) != 1
                    or type(ids_for_label[0]) is not int or not 0 <= ids_for_label[0] < VOCAB_SIZE):
                raise EvidenceError("locality continuation differs")
            labels.append(ids_for_label[0])
    except (KeyError, TypeError, IndexError) as error:
        raise EvidenceError("evaluation case schema differs") from error
    summaries = {name: {"discovery": None, "targets": [], "locality": []} for name in CONDITIONS}
    for index, case in enumerate(cases):
        start = index * row_bytes
        original_raw = matrices["original"][start:start + row_bytes]
        base_z, base_logp, base_top = score_row(original_raw)
        for name in CONDITIONS:
            raw = matrices[name][start:start + row_bytes]
            same = raw == original_raw
            edit_z, edit_logp, edit_top = (base_z, base_logp, base_top) if same else score_row(raw)
            kl = 0.0 if same else math.fsum(math.exp(a) * (a - b) for a, b in zip(base_logp, edit_logp))
            if index < 4:
                base_margin = base_z[LONDON] - base_z[PARIS]
                edit_margin = edit_z[LONDON] - edit_z[PARIS]
                row = {"case_id": case["case_id"], "original_margin": base_margin,
                       "condition_margin": edit_margin, "change": edit_margin - base_margin,
                       "originally_negative": base_margin < 0,
                       "reversed": base_margin < 0 and edit_margin > 0,
                       "original_top": base_top, "condition_top": edit_top, "kl": kl,
                       "original_london_nll": -base_logp[LONDON], "condition_london_nll": -edit_logp[LONDON],
                       "original_paris_nll": -base_logp[PARIS], "condition_paris_nll": -edit_logp[PARIS]}
                summaries[name]["targets"].append(row)
            else:
                label = labels[index - 4]
                summaries[name]["locality"].append({"case_id": case["case_id"], "original_top": base_top,
                    "condition_top": edit_top, "kl": kl, "original_nll": -base_logp[label],
                    "condition_nll": -edit_logp[label]})
    for summary in summaries.values():
        targets, locality = summary["targets"], summary["locality"]
        summary.update({"mean_target_change": math.fsum(row["change"] for row in targets) / 4,
                        "originally_negative_targets": sum(row["originally_negative"] for row in targets),
                        "reversed_targets": sum(row["reversed"] for row in targets),
                        "all_top_preserved": all(row["original_top"] == row["condition_top"] for row in locality),
                        "mean_kl": math.fsum(row["kl"] for row in locality) / 9,
                        "max_kl": max(row["kl"] for row in locality),
                        "mean_nll_deterioration": math.fsum(row["condition_nll"] - row["original_nll"]
                                                            for row in locality) / 9})
    return {"conditions": summaries, **numerical_decision(summaries, candidate)}


def compare_record(recorded, derived):
    """Compare recorded metrics, tolerating only small float reduction differences."""
    if type(recorded) is not type(derived):
        raise EvidenceError("recorded metric type differs")
    if type(derived) is dict:
        if set(recorded) != set(derived):
            raise EvidenceError("recorded metric fields differ")
        for key in derived:
            compare_record(recorded[key], derived[key])
    elif type(derived) is list:
        if len(recorded) != len(derived):
            raise EvidenceError("recorded metric row count differs")
        for actual, wanted in zip(recorded, derived):
            compare_record(actual, wanted)
    elif type(derived) is float:
        if (not math.isfinite(recorded) or not math.isfinite(derived)
                or not math.isclose(recorded, derived, rel_tol=5e-13, abs_tol=1e-10)):
            raise EvidenceError("recorded numerical metric differs")
    elif type(derived) not in (str, int, bool, type(None)) or recorded != derived:
        raise EvidenceError("recorded metric value differs")


def calibration_token_maxima(original, edited, *, width):
    """Recompute each token's largest float32 output change in binary64."""
    if (type(original) is not bytes or type(edited) is not bytes or not original
            or len(original) != len(edited) or type(width) is not int or width <= 0
            or len(original) % (4 * width)):
        raise EvidenceError("calibration output shape differs")
    before = struct.iter_unpack("<I", original)
    after = struct.iter_unpack("<I", edited)
    maxima, current = [], 0.0
    for index, ((left,), (right,)) in enumerate(zip(before, after), 1):
        change = abs(decode_float32_bits(right) - decode_float32_bits(left))
        current = max(current, change)
        if index % width == 0:
            maxima.append(current)
            current = 0.0
    return maxima


def _margin(raw, target, competitor):
    if (type(raw) is not bytes or not raw or len(raw) % 4
            or type(target) is not int or type(competitor) is not int
            or min(target, competitor) < 0 or max(target, competitor) >= len(raw) // 4):
        raise EvidenceError("discovery row or target IDs differ")
    values = [decode_float32_bits(bits) for (bits,) in struct.iter_unpack("<I", raw)]
    return values[target] - values[competitor]


def verify_development_measurements(candidate, artifacts, *, target, competitor, width):
    """Recompute the saved discovery and per-token calibration measurements."""
    try:
        for left, right in (("discovery-logits.f32", "discovery-original-logits.f32"),
                            ("discovery-grad-inputs.f32", "discovery-original-inputs.f32"),
                            ("discovery-grad-outputs.f32", "discovery-original-outputs.f32")):
            if artifacts[left] != artifacts[right]:
                raise EvidenceError("gradient and no-gradient discovery bytes differ")
        baseline = _margin(artifacts["discovery-logits.f32"], target, competitor)
        if _canonical(candidate["discovery_margin"]) != _canonical(baseline):
            raise EvidenceError("saved discovery margin differs")
        original_outputs = artifacts["calibration-original-outputs.f32"]
        maxima = {}
        for name in CONDITIONS[1:]:
            if artifacts["discovery-" + name + "-inputs.f32"] != artifacts["discovery-original-inputs.f32"]:
                raise EvidenceError("edited discovery input differs")
            margin = _margin(artifacts["discovery-" + name + "-logits.f32"], target, competitor)
            expected = {"margin": margin, "change": margin - baseline}
            if _canonical(candidate["discovery_changes"][name]) != _canonical(expected):
                raise EvidenceError("saved discovery change differs")
            per_token = calibration_token_maxima(original_outputs,
                artifacts["calibration-" + name + "-outputs.f32"], width=width)
            saved = candidate["calibration_shifts"][name]
            if (_canonical(saved["max_output_abs_by_token"]) != _canonical(per_token)
                    or _canonical(saved["max_output_abs"]) != _canonical(max(per_token))):
                raise EvidenceError("saved calibration output shifts differ")
            maxima[name] = max(per_token)
    except (KeyError, TypeError) as error:
        raise EvidenceError("development measurements are incomplete") from error
    return {"discovery_margin": baseline, "calibration_maxima": maxima}


def _recorded_time(value):
    if type(value) is not str:
        raise EvidenceError("recorded time is not a string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise EvidenceError("recorded time is invalid") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise EvidenceError("recorded time has no timezone")
    return result


def verify_phase_record(record, started, child, manifest, candidate, *, phase, manifest_sha256,
                        candidate_sha256, development_record_sha256=None):
    """Check a saved parent/child phase record, not the truth of its attestations."""
    if phase not in ("development", "evaluation"):
        raise EvidenceError("unknown study phase")
    _digest(manifest_sha256)
    _digest(candidate_sha256)
    if phase == "evaluation":
        _digest(development_record_sha256)
    try:
        source = manifest["source_commit"]
        if type(source) is not str or re.fullmatch(r"[0-9a-f]{40}", source) is None:
            raise EvidenceError("manifest source identity differs")
        supervision = record["supervisor"]
        if (record["version"] != "targeted-edit-projected-phase-v1" or record["status"] != "complete"
                or record["phase"] != phase or record["source_commit"] != source
                or record["manifest_sha256"] != manifest_sha256 or record["owner_reaped"] is not True
                or record["job_drained"] is not True or type(supervision) is not dict
                or supervision["clean"] is not True or supervision["timed_out"] is not False
                or type(supervision["exit_code"]) is not int or supervision["exit_code"] != 0
                or type(supervision["pid"]) is not int or supervision["pid"] <= 0
                or type(supervision["run_id"]) is not str
                or re.fullmatch(r"[0-9a-f]{32}", supervision["run_id"]) is None):
            raise EvidenceError("phase did not record clean supervision and cleanup")
        if (started["phase"] != phase or started["source_commit"] != source
                or type(record["checkout_commit"]) is not str
                or re.fullmatch(r"[0-9a-f]{40}", record["checkout_commit"]) is None
                or started["checkout_commit"] != record["checkout_commit"]
                or started["manifest_sha256"] != manifest_sha256
                or (phase == "evaluation" and (started["candidate_sha256"] != candidate_sha256
                    or started["development_record_sha256"] != development_record_sha256))
                or hashlib.sha256(_canonical(started)).hexdigest() != record["started_sha256"]
                or hashlib.sha256(_canonical(child)).hexdigest() != record["child_result_sha256"]
                or _canonical(record["child"]) != _canonical(child)):
            raise EvidenceError("phase start or child record binding differs")
        if (child["phase"] != phase or child["manifest_sha256"] != manifest_sha256
                or child["candidate_sha256"] != candidate_sha256 or child["model_disposed"] is not True
                or child["no_model_loaded"] is not False or type(child["owner_pid"]) is not int
                or child["owner_pid"] != supervision["pid"] or child["owner_run_id"] != supervision["run_id"]
                or _canonical(child["runtime_settings"]) != _canonical(candidate["runtime_settings"])
                or _canonical(child["runtime"]) != _canonical(V1_RUNTIME)
                or _canonical(child["post_work_checkpoint_assets"]) != _canonical(manifest["checkpoint"]["asset_sha256"])):
            raise EvidenceError("child identity, runtime, or checkpoint binding differs")
        for key in ("intraop_threads", "interop_threads", "deterministic_algorithms", "mkldnn_enabled",
                    "omp_num_threads", "mkl_num_threads", "cuda_initialized"):
            if _canonical(child["runtime"][key]) != _canonical(child["runtime_settings"][key]):
                raise EvidenceError("recorded runtime settings contradict each other")
        if child["runtime_settings"]["cuda_visibility"] != "-1":
            raise EvidenceError("recorded CPU visibility differs")
        imported, source_hashes = child["imported_sources"], manifest["source_sha256"]
        names = ("__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity", "pretrained",
                 "targeted_edit_cpu_v1", "targeted_edit_v2_edits", "targeted_edit_v2_owner",
                 "targeted_edit_projected_capture", "targeted_edit_projected_development",
                 "targeted_edit_projected_math", "targeted_edit_projected_execution", "targeted_edit_projected_score")
        required = {"looking_glass/" + name + ".py" for name in names}
        if (type(imported) is not dict or type(source_hashes) is not dict or not required <= set(imported)
                or any(source_hashes.get(name) != value for name, value in imported.items())):
            raise EvidenceError("recorded project import identities differ")
        for value in imported.values():
            _digest(value)
        times = [_recorded_time(value) for value in (manifest["created_utc"], started["started_utc"],
                                                       child["finished_utc"], record["finished_utc"])]
        if any(left > right for left, right in zip(times, times[1:])):
            raise EvidenceError("recorded phase times are contradictory")
    except (KeyError, TypeError) as error:
        raise EvidenceError("phase supervision evidence is incomplete") from error


def verify_manifest(manifest, case_bytes):
    """Check this study's exact public cases and pinned checkpoint identity."""
    case_sha = "f7554a81ac7143c54a56d557fb04cb19419bac30e7dcd44c951eb63ddaff8ee8"
    if type(case_bytes) is not bytes or hashlib.sha256(case_bytes).hexdigest() != case_sha:
        raise EvidenceError("fixed public case bytes differ")
    try:
        cases = json.loads(case_bytes)["tokenizer_preflight"]["cases"]
        expected_development = {"discovery_ids": [464, 3139, 1748, 286, 4881, 318],
            "calibration_ids": [[464, 4387, 9151, 318, 262], [7571, 5556, 734, 21767],
                                [464, 6697, 286, 3024, 318], [32, 1285, 468, 3598]],
            "target_id": 3576, "competitor_id": 6342}
        expected_checkpoint = {"model_id": "distilbert/distilgpt2",
            "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
            "tokenizer_id": "distilbert/distilgpt2", "tokenizer_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
            "model_class": "GPT2LMHeadModel", "tokenizer_class": "GPT2TokenizerFast", "device": "cpu", "dtype": "float32",
            "evaluation_mode": True, "use_cache": False, "attention_implementation": "eager", "parameter_count": 81912576,
            "torch_version": "2.8.0+cu128", "transformers_version": "4.57.3",
            "weights_sha256": "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
            "asset_sha256": {"config.json": "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a",
                "generation_config.json": "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805",
                "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
                "model.safetensors": "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b",
                "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
                "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
                "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"}}
        if (manifest["version"] != "targeted-edit-projected-run-manifest-v1"
                or manifest["source_commit"] != "fc2d8bc12cea349763a310956f503ae540cd3d81"
                or manifest["cases_sha256"] != case_sha or _canonical(manifest["evaluation_cases"]) != _canonical(cases)
                or manifest["reference_sha256"] != "cb666aa911db020b8904dab6f08f989e506d85e875d7cac65e72d3b94cef2fa5"
                or manifest["original_model_state_sha256"] != "6a69517e85aeecec31aa49914e78d95e1615a674216ca82ab6ea82f555cf2c9f"
                or manifest["original_weight_sha256"] != "26b478a67c89b984ecc5c039fe5de9a481d2638e52ff91769d77f1d16b069263"
                or _canonical(manifest["development"]) != _canonical(expected_development)
                or _canonical(manifest["checkpoint"]) != _canonical(expected_checkpoint)):
            raise EvidenceError("fixed manifest identity differs")
        if (type(cases) is not list or len(cases) != 13
                or [case["split"] for case in cases] != ["target"] * 4 + ["locality"] * 9):
            raise EvidenceError("fixed evaluation cases differ")
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        if isinstance(error, EvidenceError):
            raise
        raise EvidenceError("fixed manifest is incomplete") from error
    return cases


def _canonical(value):
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError) as error:
        raise EvidenceError("evidence is not finite canonical JSON") from error


def parse_document(raw):
    """Require one canonical JSON object with no duplicate or nonfinite values."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError("duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(_value):
        raise EvidenceError("nonfinite JSON constant")
    if type(raw) is not bytes:
        raise EvidenceError("JSON evidence is not bytes")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise EvidenceError("invalid JSON evidence") from error
    if type(value) is not dict or _canonical(value) != raw:
        raise EvidenceError("JSON evidence is not a canonical object")
    return value


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EvidenceError("invalid SHA-256 identity")
    return value


def verify_condition_states(states, candidate, manifest):
    """Bind six saved restoration/control attestations to the development record.

    Hash matches are recorded assertions; the actual weights are not saved here.
    """
    if type(states) is not dict or set(states) != set(CONDITIONS):
        raise EvidenceError("condition state set differs")
    try:
        model = _digest(candidate["model_state_sha256"])
        weight = _digest(candidate["weight_sha256"])
        buffers = _digest(candidate["buffer_state_sha256"])
        if model != manifest["original_model_state_sha256"] or weight != manifest["original_weight_sha256"]:
            raise EvidenceError("development baseline differs from manifest")
        runtime = candidate["runtime_settings"]
        runtime_keys = {"float32_matmul_precision", "mkldnn_deterministic", "deterministic_warn_only",
                        "intraop_threads", "interop_threads", "deterministic_algorithms", "mkldnn_enabled",
                        "cpu_autocast_enabled", "inference_mode_enabled", "cuda_visibility", "cuda_initialized",
                        "hf_hub_offline", "transformers_offline", "omp_num_threads", "mkl_num_threads"}
        if type(runtime) is not dict or set(runtime) != runtime_keys:
            raise EvidenceError("runtime settings are incomplete")
        if (runtime["float32_matmul_precision"] not in ("highest", "high", "medium")
                or any(type(runtime[key]) is not bool for key in ("mkldnn_deterministic", "mkldnn_enabled"))
                or any(type(runtime[key]) is not int or runtime[key] != 1 for key in ("intraop_threads", "interop_threads"))
                or runtime["deterministic_algorithms"] is not True
                or any(runtime[key] is not False for key in ("deterministic_warn_only", "cpu_autocast_enabled",
                                                              "inference_mode_enabled", "cuda_initialized"))
                or runtime["cuda_visibility"] != "-1" or runtime["hf_hub_offline"] != "1"
                or runtime["transformers_offline"] != "1"
                or any(runtime[key] is not None and type(runtime[key]) is not str
                       for key in ("omp_num_threads", "mkl_num_threads"))):
            raise EvidenceError("runtime settings do not describe the CPU protocol")
        construction = candidate["construction"]
        controls = construction["conditions"]
        if type(controls) is not dict or set(controls) != set(CONDITIONS[1:]):
            raise EvidenceError("development control set differs")
        planned = _finite(construction["planned_delta_norm"], nonnegative=True)
        weight_norm = _finite(construction["weight_norm"], nonnegative=True)
        if (planned == 0 or weight_norm == 0 or type(construction["relative_dose"]) is not float
                or construction["relative_dose"] != 0.001
                or not math.isclose(planned, 0.001 * weight_norm, rel_tol=1e-12)):
            raise EvidenceError("planned dose differs from the fixed weight-relative dose")
        norms = {}
        for name in CONDITIONS[1:]:
            item = controls[name]
            _digest(item["intended_delta_sha256"])
            _digest(item["realized_weight_sha256"])
            norm = _finite(item["realized_delta_norm"], nonnegative=True)
            ratio = _finite(item["realized_to_planned"], nonnegative=True)
            if norm == 0 or not 0.999 <= ratio <= 1.001 or not math.isclose(ratio, norm / planned, rel_tol=1e-12):
                raise EvidenceError("recorded control dose differs")
            norms[name] = norm
        for name in CONDITIONS[1:]:
            ratio = _finite(controls[name]["realized_to_projected"], nonnegative=True)
            if not 0.999 <= ratio <= 1.001 or not math.isclose(ratio, norms[name] / norms["projected"], rel_tol=1e-12):
                raise EvidenceError("recorded control dose does not match projected")
        keys = {"condition", "applied_weight_sha256", "model_state_sha256", "weight_sha256",
                "buffer_state_sha256", "runtime_settings", "addition_denormal_probe_before",
                "addition_denormal_probe_after", "cleanup"}
        for name in CONDITIONS:
            state = states[name]
            applied = weight if name == "original" else controls[name]["realized_weight_sha256"]
            if (type(state) is not dict or set(state) != keys or state["condition"] != name
                    or state["applied_weight_sha256"] != applied or state["model_state_sha256"] != model
                    or state["weight_sha256"] != weight or state["buffer_state_sha256"] != buffers
                    or _canonical(state["runtime_settings"]) != _canonical(runtime)
                    or state["cleanup"] != "verified"
                    or _canonical(state["addition_denormal_probe_before"]) != _canonical(DENORMAL_PROBE)
                    or _canonical(state["addition_denormal_probe_after"]) != _canonical(DENORMAL_PROBE)):
                raise EvidenceError("condition restoration, runtime, or control identity differs")
    except (KeyError, TypeError) as error:
        raise EvidenceError("condition state evidence is incomplete") from error
    return len(CONDITIONS)


def verify_development_states(states, candidate):
    """Check the saved development restoration and per-forward mode attestations."""
    if type(states) is not dict or set(states) != set(CONDITIONS):
        raise EvidenceError("development state set differs")
    try:
        for key in ("discovery_denormal_probe_before", "discovery_denormal_probe_after"):
            if _canonical(candidate[key]) != _canonical(DENORMAL_PROBE):
                raise EvidenceError("development discovery probe differs")
        model = _digest(candidate["model_state_sha256"])
        weight = _digest(candidate["weight_sha256"])
        buffers = _digest(candidate["buffer_state_sha256"])
        controls = candidate["construction"]["conditions"]
        expected_forward = {"calibration": [{"before": DENORMAL_PROBE, "after": DENORMAL_PROBE} for _ in range(4)],
                            "discovery": {"before": DENORMAL_PROBE, "after": DENORMAL_PROBE}}
        keys = {"condition", "applied_weight_sha256", "model_state_sha256", "weight_sha256",
                "buffer_state_sha256", "addition_denormal_probe_before", "addition_denormal_probe_after",
                "forward_denormal_probes", "cleanup"}
        for name in CONDITIONS:
            state = states[name]
            applied = weight if name == "original" else _digest(controls[name]["realized_weight_sha256"])
            if (type(state) is not dict or set(state) != keys or state["condition"] != name
                    or state["applied_weight_sha256"] != applied or state["model_state_sha256"] != model
                    or state["weight_sha256"] != weight or state["buffer_state_sha256"] != buffers
                    or state["cleanup"] != "verified"
                    or _canonical(state["addition_denormal_probe_before"]) != _canonical(DENORMAL_PROBE)
                    or _canonical(state["addition_denormal_probe_after"]) != _canonical(DENORMAL_PROBE)
                    or _canonical(state["forward_denormal_probes"]) != _canonical(expected_forward)):
                raise EvidenceError("development restoration, control or numeric probes differ")
    except (KeyError, TypeError) as error:
        raise EvidenceError("development state evidence is incomplete") from error
    return len(CONDITIONS)


def parse_attestation(raw):
    """Read noncanonical JSON attestations while rejecting ambiguous values."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError("duplicate attestation key")
            result[key] = value
        return result
    def nonfinite(_value):
        raise EvidenceError("nonfinite attestation value")
    if type(raw) is not bytes:
        raise EvidenceError("attestation is not bytes")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise EvidenceError("invalid attestation") from error
    if type(value) is not dict:
        raise EvidenceError("attestation is not an object")
    return value


def verify_public_exposure(manifest, started, files, cases):
    """Compare the recorded public inventory and its receipt, without claiming completeness."""
    try:
        exposure = manifest["exposure"]
        inventory_raw = files["public-exposure.json"]
        receipt_raw = files["public-exposure-attestation.json"]
        inventory, receipt = parse_attestation(inventory_raw), parse_attestation(receipt_raw)
        digest = hashlib.sha256(inventory_raw).hexdigest()
        summary = {key: value for key, value in exposure.items() if key != "known_complete_rows"}
        source_path = "looking_glass/pretrained_exposures_v15.py"
        records = inventory["records"]
        if (digest != _digest(exposure["canonical_sha256"])
                or digest != started["public_exposure_file_sha256"] or digest != receipt["inventory_sha256"]
                or hashlib.sha256(receipt_raw).hexdigest() != started["public_exposure_attestation_sha256"]
                or _canonical(started["current_public_exposure"]) != _canonical(summary)
                or inventory["version"] != exposure["version"]
                or type(records) is not list or type(exposure["record_count"]) is not int
                or len(records) != exposure["record_count"] or type(exposure["projected_rows"]) is not int
                or exposure["projected_rows"] != 13 or type(exposure["known_relations"]) is not int
                or exposure["known_relations"] != 0 or receipt["source_path"] != source_path
                or receipt["source_sha256"] != _digest(manifest["source_sha256"][source_path])
                or type(receipt["reviewed_commit"]) is not str
                or re.fullmatch(r"[0-9a-f]{40}", receipt["reviewed_commit"]) is None
                or _recorded_time(receipt["observed_utc"]) > _recorded_time(started["started_utc"])):
            raise EvidenceError("recorded exposure binding differs")
        known = {tuple(record["token_ids"]) for record in records}
        if type(cases) is not list or len(cases) != 13 or any(tuple(case["token_ids"]) not in known for case in cases):
            raise EvidenceError("fixed public cases are not all reserved in the recorded inventory")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, EvidenceError):
            raise
        raise EvidenceError("recorded exposure evidence is incomplete") from error


def audit_saved(manifest_bytes, case_bytes, development_directory, evaluation_directory, *,
                manifest_sha256, candidate_sha256, development_record_sha256, evaluation_record_sha256):
    """Reconstruct the saved study result without a model, assuming quiescent inputs.

    Callers must independently obtain the four expected digests. This verifies
    saved bytes and cooperative process/cleanup assertions, not actual execution,
    Git ancestry, remote publication, chronology, or unseen exposure history.
    """
    for value in (manifest_sha256, candidate_sha256, development_record_sha256, evaluation_record_sha256):
        _digest(value)
    if type(manifest_bytes) is not bytes or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise EvidenceError("manifest hash differs")
    manifest = parse_document(manifest_bytes)
    cases = verify_manifest(manifest, case_bytes)
    dev, dev_files = read_phase_snapshot(development_directory, development_record_sha256, phase="development")
    ev, ev_files = read_phase_snapshot(evaluation_directory, evaluation_record_sha256, phase="evaluation")
    def require(condition, message):
        if not condition:
            raise EvidenceError(message)
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()
    try:
        dev_started = parse_document(dev_files["started.json"])
        dev_child = parse_document(dev_files["child-result.json"])
        ev_started = parse_document(ev_files["started.json"])
        ev_child = parse_document(ev_files["child-result.json"])
        candidate_raw = dev_files["candidate/pre-evaluation-candidate.json"]
        require(sha(candidate_raw) == candidate_sha256, "development candidate hash differs")
        candidate = parse_document(candidate_raw)
        require(candidate["version"] == "targeted-edit-projected-development-v1"
                and candidate["status"] == "private-candidate", "development candidate status differs")
        verify_phase_record(dev, dev_started, dev_child, manifest, candidate, phase="development",
                            manifest_sha256=manifest_sha256, candidate_sha256=candidate_sha256)
        verify_phase_record(ev, ev_started, ev_child, manifest, candidate, phase="evaluation",
                            manifest_sha256=manifest_sha256, candidate_sha256=candidate_sha256,
                            development_record_sha256=development_record_sha256)
        require(all(dev_child[key] == candidate[key] for key in
                    ("model_state_sha256", "weight_sha256", "buffer_state_sha256", "discovery_margin")),
                "development child binding differs")
        fixed = manifest["development"]
        require(_canonical(candidate["discovery_token_ids"]) == _canonical(fixed["discovery_ids"])
                and _canonical(candidate["calibration_token_ids"]) == _canonical(fixed["calibration_ids"])
                and type(candidate["target_token_id"]) is int and candidate["target_token_id"] == fixed["target_id"]
                and type(candidate["competitor_token_id"]) is int and candidate["competitor_token_id"] == fixed["competitor_id"],
                "development input binding differs")
        for record, started, files in ((dev, dev_started, dev_files), (ev, ev_started, ev_files)):
            observation = files["manifest-review-attestation.json"]
            require(sha(observation) == record["manifest_review_attestation_sha256"]
                    == started["manifest_review_attestation_sha256"]
                    and parse_attestation(observation)["manifest_sha256"] == manifest_sha256,
                    "manifest observation differs")
            verify_public_exposure(manifest, started, files, cases)
        review_raw = ev_files["review-attestation.json"]
        review = parse_attestation(review_raw)
        reviewed_commit = "0995c739e10d692a50c80eb3fc8cdc81d32e39ba"
        require(sha(review_raw) == ev["review_attestation_sha256"] == ev_started["review_attestation_sha256"]
                and _canonical(review) == _canonical(ev_started["review_attestation"])
                and review["reviewed_commit"] == reviewed_commit == ev["checkout_commit"]
                and review["candidate_sha256"] == candidate_sha256
                and review["development_record_sha256"] == development_record_sha256
                and review["manifest_sha256"] == manifest_sha256, "development review binding differs")
        times = [_recorded_time(value) for value in (dev["finished_utc"], review["observed_utc"],
                                                      ev_started["started_utc"])]
        require(all(left <= right for left, right in zip(times, times[1:])), "recorded times are contradictory")
        artifacts = candidate["artifacts"]
        names = {name[len("candidate/"):] for name in dev["artifacts"] if name.startswith("candidate/")}
        require(type(artifacts) is dict and len(artifacts) == 50
                and names == set(artifacts) | {"pre-evaluation-candidate.json"}, "candidate inventory differs")
        saved = {}
        for name, item in artifacts.items():
            data = dev_files["candidate/" + name]
            require(type(item["bytes"]) is int and len(data) == item["bytes"] and sha(data) == item["sha256"],
                    "candidate artifact identity differs")
            saved[name] = data
        require(sha(saved["discovery-logits.f32"]) == "af42750f9e1cfc595e6f942e08290f543ffbd34f31388f51d8a4156b86fed43e",
                "discovery row does not match the v1 reference")
        measurements = verify_development_measurements(candidate, saved, target=LONDON, competitor=PARIS, width=768)
        require(all(len(candidate["calibration_shifts"][name]["max_output_abs_by_token"]) == 18
                    for name in CONDITIONS[1:]), "calibration token count differs")
        dev_states = {name: parse_document(saved["state-" + name + ".json"]) for name in CONDITIONS}
        verify_development_states(dev_states, candidate)
        states = {name: parse_document(ev_files["evaluation/state-" + name + ".json"]) for name in CONDITIONS}
        verify_condition_states(states, candidate, manifest)
        matrices = {name: ev_files["evaluation/" + name + ".f32"] for name in CONDITIONS}
        count = verify_journal(ev_files["evaluation/observations.jsonl"], matrices,
                               [case["case_id"] for case in cases], row_bytes=VOCAB_SIZE * 4)
        derived = score_matrices(matrices, cases, candidate)
        report_raw = ev_files["evaluation/provisional-report.json"]
        report = parse_document(report_raw)
        require(sha(report_raw) == ev_child["report_sha256"]
                and report["version"] == "targeted-edit-projected-score-v1", "saved report identity differs")
        for key in ("conditions", "eligibility", "criteria"):
            compare_record(report[key], derived[key])
        compare_record(report["discovery_margin"], measurements["discovery_margin"])
        compare_record(report["projected_calibration_max_abs"], measurements["calibration_maxima"]["projected"])
        compare_record(report["unprojected_calibration_max_abs"], measurements["calibration_maxima"]["unprojected"])
        require(report["provisional_numerical_decision"] == derived["decision"] == ev_child["provisional_numerical_decision"]
                and _canonical(report["denormal_probe_before"]) == _canonical(DENORMAL_PROBE)
                and _canonical(report["denormal_probe_after"]) == _canonical(DENORMAL_PROBE),
                "recorded decision or scoring mode differs")
    except (KeyError, IndexError, TypeError) as error:
        raise EvidenceError("saved evidence is incomplete or malformed") from error
    return {"schema": "projected-independent-saved-output-audit-v1", "status": "matching_saved_outputs",
            "manifest_sha256": manifest_sha256, "development_record_sha256": development_record_sha256,
            "candidate_sha256": candidate_sha256, "evaluation_record_sha256": evaluation_record_sha256,
            "evaluation_report_sha256": sha(report_raw), "verified_journal_rows": count,
            "verified_condition_states": len(states), "development_measurements": measurements,
            "decision": derived["decision"], "eligibility": derived["eligibility"], "criteria": derived["criteria"],
            "conditions": derived["conditions"],
            "protocol_deviation": "Targets 1 and 4 are direct capital prompts; targets 2 and 3 concern government location or seat.",
            "limitations": ["Recorded process, restoration and weight identities are attestations, not independent execution proof.",
                            "Git and reported times do not authenticate publication chronology or unseen prior exposure.",
                            "Original and edited weight bytes were not saved; applied-weight identities cannot be reconstructed here.",
                            "The public non-blind cases do not establish general capability preservation."]}


def main(argv=None):
    """Emit one JSON result; failures never carry a numerical decision."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Inspect quiescent projected-study saved evidence without a model")
    for flag in ("manifest", "cases", "development", "evaluation", "manifest-sha256", "candidate-sha256",
                 "development-sha256", "evaluation-sha256"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_saved(read_ordinary(args.manifest, 1_000_000), read_ordinary(args.cases, 1_000_000),
                             args.development, args.evaluation, manifest_sha256=args.manifest_sha256,
                             candidate_sha256=args.candidate_sha256, development_record_sha256=args.development_sha256,
                             evaluation_record_sha256=args.evaluation_sha256)
    except Exception:
        sys.stdout.buffer.write(_canonical({"status": "incomplete", "error": "saved evidence could not be verified"}))
        return 1
    sys.stdout.buffer.write(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
