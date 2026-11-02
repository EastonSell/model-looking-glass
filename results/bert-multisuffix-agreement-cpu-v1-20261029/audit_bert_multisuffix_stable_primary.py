"""Independent, model-free numerical recheck of the fixed primary comparison.

The input is an already-audited, quiescent saved-byte bundle.  This is not an
execution attestation.  Its sign intervals depend on the floating-point and
libm assumptions documented in bert_multisuffix_stable_pair; inconclusive
intervals remain unresolved rather than being interpreted as negative.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys

from bert_multisuffix_stable_pair import SLACK, pair_difference

VOCAB = 30522
IDS = (2003, 2024)
VARIANTS = ("terminal", "period", "present", "outside", "awake")
CASE_HASH = "bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e"
LOGITS_HASH = "623c173a2c0244a3b3aac7978a2fb83a41cf8d1d9de892945b3f8b111d5d2149"
SOURCE = "42519c13a5b33ce3601b89ba65d67b07ebd2f78d"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def interval(result):
    value, err = result["difference"], result["absolute_error_bound"]
    if result["sign"] == "zero":
        return 0.0, 0.0
    if value is None or err is None:
        return None
    return math.nextafter(value - err, -math.inf), math.nextafter(value + err, math.inf)


def mean_interval(intervals):
    if any(x is None for x in intervals):
        return None
    if all(x == (0.0, 0.0) for x in intervals):
        return 0.0, 0.0
    lows, highs = [x[0] for x in intervals], [x[1] for x in intervals]
    # Conservatively allow rounding in the interval endpoint accumulation too.
    scale = max(math.fsum(abs(x) for x in lows), math.fsum(abs(x) for x in highs))
    margin = SLACK * scale
    lo = math.nextafter((math.fsum(lows) - margin) / len(intervals), -math.inf)
    hi = math.nextafter((math.fsum(highs) + margin) / len(intervals), math.inf)
    return lo, hi


def sign_bounds(intervals):
    positives = sum(v is not None and v[0] > 0 for v in intervals)
    possible = sum(v is None or v[1] > 0 for v in intervals)
    return positives, possible


def criterion(intervals, groups):
    mean = mean_interval(intervals)
    lower, upper = sign_bounds(intervals)
    group_intervals = {k: mean_interval(v) for k, v in groups.items()}
    gmin, gmax = sign_bounds(list(group_intervals.values()))
    if mean is not None and mean[0] > 0 and lower >= 24 and gmin >= 6:
        result = True
    elif (mean is not None and mean[1] <= 0) or upper < 24 or gmax < 6:
        result = False
    else:
        result = None
    return {"mean_interval": mean, "positive_case_count_interval": [lower, upper],
            "group_mean_intervals": group_intervals, "positive_group_count_interval": [gmin, gmax],
            "criterion": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("attempt", "cases", "source-report", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    args = p.parse_args()
    attempt, cases_path, report_path = args.attempt.resolve(), args.cases.resolve(), args.source_report.resolve()
    output = args.output.resolve()
    if output.is_relative_to(attempt) or output == report_path:
        raise ValueError("output must be separate from the closed records")
    logits_path = attempt / "logits.f32"
    before = {"cases": sha(cases_path), "logits": sha(logits_path), "source_report": sha(report_path)}
    if before["cases"] != CASE_HASH or before["logits"] != LOGITS_HASH:
        raise ValueError("unexpected input bytes")
    cases = json.loads(cases_path.read_bytes())["cases"]
    old = json.loads(report_path.read_bytes())
    if (len(cases) != 32 or old.get("source_commit") != SOURCE or old.get("synthetic_unattested") is not False
            or old.get("file_sha256", {}).get("logits.f32") != LOGITS_HASH):
        raise ValueError("source report or case count differs")
    if logits_path.stat().st_size != 1088 * VOCAB * 4:
        raise ValueError("saved full-vocabulary logits length differs")
    with logits_path.open("rb") as f:
        raw = f.read(160 * VOCAB * 4)
    if len(raw) != 160 * VOCAB * 4 or array("f").itemsize != 4:
        raise ValueError("unsupported float32 input")
    logits = array("f")
    logits.frombytes(raw)
    if sys.byteorder != "little":
        logits.byteswap()
    if len(old["cases"]) != 32:
        raise ValueError("original report case count differs")
    results = {name: [] for name in VARIANTS[2:]}
    rows = []
    for i, case in enumerate(cases):
        prior = old["cases"][i]
        if (case["index"] != i or prior["case_id"] != case["case_id"] or prior["frame"] != case["frame"]
                or prior["lemma"] != case["lemma"]):
            raise ValueError("case/report order differs")
        def vec(variant_index):
            start = (5 * i + variant_index) * VOCAB
            return logits[start:start + VOCAB]
        baseline = vec(1)
        row = {"case_id": case["case_id"], "frame": case["frame"], "lemma": case["lemma"], "variants": {}}
        for j, name in enumerate(VARIANTS[2:], 2):
            measured = pair_difference(baseline, vec(j), IDS)
            bounds = interval(measured)
            row["variants"][name] = {**measured, "interval": bounds}
            results[name].append((case, bounds))
        rows.append(row)
    summary = {}
    for name, samples in results.items():
        groups = {f"{f}:{l}": [bound for case, bound in samples if case["frame"] == f and case["lemma"] == l]
                  for f in (0, 1) for l in range(4)}
        if any(len(v) != 4 for v in groups.values()):
            raise ValueError("unexpected frame/lemma structure")
        summary[name] = criterion([x[1] for x in samples], groups)
        summary[name]["original_rounded_criterion"] = old["summary"]["suffix_vs_period"][name]["criterion"]
    if before != {"cases": sha(cases_path), "logits": sha(logits_path), "source_report": sha(report_path)}:
        raise ValueError("input changed during numerical replay")
    report = {"schema": "multisuffix-stable-primary-replay-v1", "source_commit": SOURCE,
              "inputs_sha256": before, "helper_sha256": sha(Path(__file__).with_name("bert_multisuffix_stable_pair.py")),
              "script_sha256": sha(Path(__file__)), "cases": rows, "summary": summary,
              "scope": "Independent calculation from previously checked saved CPU float32 logits; no model, execution or chronology attestation. Error intervals assume IEEE binary64 round-to-nearest and exp/expm1 errors at most 16 ulps; not a platform-independent libm proof. Unknown signs remain unresolved."}
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as f:
        if f.write(data) != len(data):
            raise OSError("short output write")
    print("stable_primary_replay", hashlib.sha256(data).hexdigest())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
