"""Audit saved full-v1 CPU records without rerunning the experiment or loading torch.

Consistent files are not proof that the recorded experiment actually ran. Hash
comparisons establish byte agreement with a supplied source tree, not provenance.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


class AuditError(ValueError):
    """The saved records are incomplete, malformed, or mutually inconsistent."""


CONFIG = {"widths": [16, 64], "seeds": [0, 1, 2, 3], "depth": 4,
          "batch": 8, "warmup": 25, "repeats": 100}
SOURCES = {"__init__.py", "experiment.py", "model.py", "probes.py", "PROTOCOL.md"}
MODES = ("direct", "gradient", "id")
PLACEMENTS = ("sparse", "dense")
TIMING_FIELDS = ("width", "seed", "mode", "placement", "median_us", "p90_us",
                 "samples", "events", "recorded_tensor_bytes")
RAW_FIELDS = ("width", "seed", "mode", "placement", "sample", "nanoseconds")
CHECKS = ("output", "loss", "input_gradient", "gradients", "state")


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def _key(row, fields):
    return tuple(row[field] for field in fields)


def _rows_by_key(rows, fields, expected, label):
    found = {}
    exemplar = next(iter(expected))
    for row in rows:
        key = _key(row, fields)
        _require(all(type(value) is type(sample) for value, sample in zip(key, exemplar)),
                 f"invalid {label} case identifier: {key}")
        _require(key not in found, f"duplicate {label} case: {key}")
        found[key] = row
    _require(set(found) == expected, f"{label} cases do not match the full-v1 design")
    return found


def _bool(value, label):
    _require(type(value) is bool, f"{label} must be a boolean")
    return value


def _int(value, label):
    _require(type(value) is int and value >= 0, f"{label} must be a nonnegative integer")
    return value


def _number(value, label):
    _require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")
    return value


def _read_csv(path, fields):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(fields)
                 and set(reader.fieldnames) == set(fields), f"invalid CSV header: {path.name}")
        rows = list(reader)
    _require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
             f"malformed CSV row: {path.name}")
    return rows


def _source_hashes(hashes, source_dir):
    _require(isinstance(hashes, dict) and SOURCES <= set(hashes) and all(
        isinstance(name, str) and Path(name).name == name and
        (name == "PROTOCOL.md" or name.endswith(".py")) for name in hashes),
        "invalid or incomplete recorded source hash set")
    result = {}
    for name, recorded in sorted(hashes.items()):
        _require(isinstance(recorded, str) and len(recorded) == 64
                 and all(c in "0123456789abcdef" for c in recorded), f"invalid source hash: {name}")
        path = source_dir / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        result[name] = {"recorded": recorded, "actual": actual,
                        "status": "missing" if actual is None else "match" if actual == recorded else "mismatch"}
    return result


def _audit(run_dir, source_dir):
    def reject_constant(value):
        raise AuditError(f"nonfinite JSON value: {value}")

    with (run_dir / "results.json").open(encoding="utf-8") as stream:
        data = json.load(stream, parse_constant=reject_constant)
    _require(data["protocol_mode"] == "full protocol v1" and
             json.dumps(data["config"], sort_keys=True) == json.dumps(CONFIG, sort_keys=True),
             "only the frozen full protocol v1 is supported")
    env = data["environment"]
    _require(env["device"] == "cpu" and env["dtype"] == "float32" and
             type(env["threads"]) is int and env["threads"] == 1,
             "recorded environment contradicts the CPU float32 protocol")
    hashes = _source_hashes(data["source_sha256"], source_dir)
    widths, seeds = CONFIG["widths"], CONFIG["seeds"]
    placements = {(w, s, p) for w in widths for s in seeds for p in PLACEMENTS}
    instrumented = {(w, s, m, p) for w in widths for s in seeds for m in MODES for p in PLACEMENTS}
    timing_expected = instrumented | {(w, s, "none", "none") for w in widths for s in seeds}

    transparency = _rows_by_key(data["transparency"], ("width", "seed", "mode", "placement"),
                                instrumented, "transparency")
    transparency_passed = 0
    for row in transparency.values():
        checks = [_bool(row[field], f"transparency {field}") for field in CHECKS]
        transparency_passed += all(checks)
    _require(_bool(data["all_transparent"], "all_transparent") == (transparency_passed == len(transparency)),
             "all_transparent disagrees with recorded cases")

    reconstruction = _rows_by_key(data["reconstruction"], ("width", "seed", "placement"),
                                   placements, "reconstruction")
    reconstruction_passed = 0
    reconstruction_elements = 0
    for (width, _, placement), row in reconstruction.items():
        reconstruction_passed += _bool(row["equal"], "reconstruction equal")
        elements = _int(row["tensor_elements"], "reconstruction tensor_elements")
        _require(elements == 8 * width * (4 if placement == "sparse" else 16),
                 "reconstruction element count disagrees with the design")
        reconstruction_elements += elements
    _require(_bool(data["channels_equal"], "channels_equal") == (reconstruction_passed == len(reconstruction)),
             "channels_equal disagrees with recorded cases")

    nodes = [f"block.{i}.{suffix}" for i in range(4) for suffix in ("branch.0", "activation", "merge")]
    loc_expected = {(w, s, p, n) for w, s, p in placements for n in nodes}
    localization = _rows_by_key(data["localization"], ("width", "seed", "placement", "injected_at"),
                                 loc_expected, "localization")
    loc_summary = {p: {"trials": 0, "detected": 0, "exact_first_point": 0,
                       "first_point_after_injection": 0} for p in PLACEMENTS}
    for (_, _, placement, injected), row in localization.items():
        detected = _bool(row["detected"], "localization detected")
        exact = _bool(row["exact_first_point"], "localization exact_first_point")
        interval = row["observed_interval"]
        _require(detected == (interval is not None), "localization detection disagrees with interval")
        if detected:
            dense_order = [f"block.{i}.{suffix}" for i in range(4)
                           for suffix in ("branch.0", "activation", "merge", "output")]
            observed = ([f"block.{i}.output" for i in range(4)] if placement == "sparse" else dense_order)
            first = interval["first_unequal"]
            _require(first in observed, "localization first point is not observed")
            _require(dense_order.index(first) >= dense_order.index(injected),
                     "localization first point precedes the injection")
            index = observed.index(first)
            _require(interval["last_equal"] == (observed[index - 1] if index else None),
                     "localization interval is inconsistent with observation order")
            _require(exact == (first == injected), "localization exact flag disagrees with interval")
        else:
            _require(not exact, "undetected localization cannot be exact")
        summary = loc_summary[placement]
        summary["trials"] += 1
        summary["detected"] += detected
        summary["exact_first_point"] += exact
        summary["first_point_after_injection"] += detected and dense_order.index(first) > dense_order.index(injected)

    timing = _rows_by_key(data["timing"], ("width", "seed", "mode", "placement"),
                          timing_expected, "timing")
    for (width, _, mode, placement), row in timing.items():
        raw = row["raw_ns"]
        _require(isinstance(raw, list) and len(raw) == CONFIG["repeats"], "wrong raw timing sample count")
        for sample in raw:
            _int(sample, "raw timing")
        ordered = sorted(raw)
        _require(_number(row["median_us"], "median") == statistics.median(ordered) / 1000,
                 "timing median disagrees with raw samples")
        _require(_number(row["p90_us"], "p90") == ordered[int(.9 * (len(ordered) - 1))] / 1000,
                 "timing p90 disagrees with raw samples")
        _require(_int(row["samples"], "samples") == len(raw), "timing sample count mismatch")
        events = 0 if mode == "none" else (4 if placement == "sparse" else 16)
        value_bytes = 0 if mode in ("none", "id") else 8 * width * events * 4
        _require(_int(row["events"], "events") == events and
                 _int(row["recorded_tensor_bytes"], "recorded_tensor_bytes") == value_bytes,
                 "timing events or value bytes disagree with the design")

    summary_csv = _read_csv(run_dir / "timing.csv", TIMING_FIELDS)
    for row in summary_csv:
        for field in ("width", "seed", "samples", "events", "recorded_tensor_bytes"):
            row[field] = int(row[field])
        for field in ("median_us", "p90_us"):
            row[field] = float(row[field])
    csv_summary = _rows_by_key(summary_csv, TIMING_FIELDS[:4], timing_expected, "summary CSV")
    for key, row in csv_summary.items():
        _require(all(row[f] == timing[key][f] for f in TIMING_FIELDS), "summary CSV disagrees with JSON")

    raw_csv = _read_csv(run_dir / "raw_timing.csv", RAW_FIELDS)
    seen = set()
    for row in raw_csv:
        width, seed, index, ns = (int(row[field]) for field in ("width", "seed", "sample", "nanoseconds"))
        key = (width, seed, row["mode"], row["placement"])
        sample_key = (*key, index)
        _require(key in timing and sample_key not in seen and 0 <= index < CONFIG["repeats"],
                 "raw CSV contains an unexpected or duplicate sample")
        seen.add(sample_key)
        _require(ns == timing[key]["raw_ns"][index], "raw CSV disagrees with JSON")
    _require(len(seen) == len(timing) * CONFIG["repeats"], "raw CSV sample count mismatch")

    aggregates = {}
    for width in widths:
        width_report = {"baseline_median_us": statistics.median(timing[(width, s, "none", "none")]["median_us"] for s in seeds)}
        for placement in PLACEMENTS:
            modes = {m: [timing[(width, s, m, placement)]["median_us"] for s in seeds] for m in MODES}
            _require(all(v > 0 for v in modes["direct"]), "paired ratio requires positive direct medians")
            width_report[placement] = {f"{m}_median_us": statistics.median(modes[m]) for m in MODES}
            width_report[placement]["gradient_over_direct_paired_median"] = statistics.median(
                g / d for g, d in zip(modes["gradient"], modes["direct"]))
        aggregates[str(width)] = width_report

    controls = {key: _bool(data[key], key) for key in
                ("identical_branches_value_ambiguous", "identical_branches_metadata_distinct")}
    return {"scope": "saved full-v1 CPU records; no model rerun or provenance proof",
            "recorded_environment": env, "source_hashes": hashes,
            "transparency_passed": transparency_passed, "transparency_cases": len(transparency),
            "reconstruction_passed": reconstruction_passed, "reconstruction_cases": len(reconstruction),
            "reconstruction_elements": reconstruction_elements, "reported_branch_controls": controls,
            "localization": loc_summary, "timing_cells": len(timing), "raw_samples": len(seen),
            "timing_aggregates": aggregates}


def audit_run(run_dir, source_dir=None):
    """Check preserved records and return recomputed aggregates and source comparisons.

    Source mismatches are reported separately: a newer checkout can still audit
    historical files. Invalid/inconsistent saved data raises :class:`AuditError`.
    """
    try:
        return _audit(Path(run_dir), Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent)
    except AuditError:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise AuditError(f"invalid or unreadable saved records: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory containing results.json and both timing CSVs")
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="package source directory to compare against recorded hashes")
    args = parser.parse_args(argv)
    try:
        report = audit_run(args.run_dir, args.source_dir)
    except AuditError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
