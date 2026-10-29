"""Recompute all fixed paired clean masses from saved float32 logits with Decimal.

This model-free numerical cross-check is separate from the source-bound reader.
It uses 70 decimal digits and direct softmax, without the expm1 formula. It is
not an exact-arithmetic proof or an execution/source-chronology attestation.
"""
from array import array
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys

base = Path("E:/Codex/model-looking-glass/work")
logit_path = base / "bert-multisuffix-cpu-v1-20261029-01/logits.f32"
case_path = base / "bert-multisuffix-v1/experiments/bert_multisuffix_agreement_cases_v1.json"
stable_path = Path("work/reviews/bert-multisuffix-actual-stable-primary-v1.json")
output = Path("work/reviews/bert-multisuffix-actual-decimal-primary-v1.json")
expected = {"logits": "623c173a2c0244a3b3aac7978a2fb83a41cf8d1d9de892945b3f8b111d5d2149",
            "cases": "bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e",
            "stable": "25d3c63005b39397d901eb2164c5e0ea1ed0f7398b2f80be31d1eefc8394e926"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprints():
    return {"logits": sha(logit_path), "cases": sha(case_path), "stable": sha(stable_path)}


if fingerprints() != expected:
    raise ValueError("unexpected input hashes")
cases = json.loads(case_path.read_bytes())["cases"]
stable = json.loads(stable_path.read_bytes())
if len(cases) != 32 or len(stable["cases"]) != 32:
    raise ValueError("case count")
with logit_path.open("rb") as f:
    raw = f.read(160 * 30522 * 4)
if len(raw) != 160 * 30522 * 4 or array("f").itemsize != 4:
    raise ValueError("float32 size")
values = array("f")
values.frombytes(raw)
if sys.byteorder != "little":
    values.byteswap()


def mass(i, v):
    start = (i * 5 + v) * 30522
    vector = values[start:start + 30522]
    mx = max(vector)
    weighted = [(Decimal.from_float(float(x)) - Decimal.from_float(float(mx))).exp() for x in vector]
    return (weighted[2003] + weighted[2024]) / sum(weighted, Decimal(0))


rows = []
with localcontext() as ctx:
    ctx.prec = 70
    for i, case in enumerate(cases):
        if case["index"] != i or stable["cases"][i]["case_id"] != case["case_id"]:
            raise ValueError("case order")
        period = mass(i, 1)
        row = {"case_id": case["case_id"], "frame": case["frame"], "lemma": case["lemma"], "variants": {}}
        for vi, variant in enumerate(("present", "outside", "awake"), 2):
            delta = mass(i, vi) - period
            bounds = stable["cases"][i]["variants"][variant]["interval"]
            if bounds is None or not Decimal.from_float(bounds[0]) <= delta <= Decimal.from_float(bounds[1]):
                raise ValueError("Decimal value outside stable interval: " + case["case_id"] + " " + variant)
            row["variants"][variant] = str(delta)
        rows.append(row)
        if (i + 1) % 8 == 0:
            print("checked", i + 1, "of 32", flush=True)
    summary = {}
    for variant in ("present", "outside", "awake"):
        vals = [Decimal(row["variants"][variant]) for row in rows]
        grouped = {f"{f}:{l}": [Decimal(row["variants"][variant]) for row in rows if row["frame"] == f and row["lemma"] == l]
                   for f in (0, 1) for l in range(4)}
        if any(len(v) != 4 for v in grouped.values()):
            raise ValueError("group shape")
        mean = sum(vals, Decimal(0)) / 32
        means = {k: sum(v, Decimal(0)) / 4 for k, v in grouped.items()}
        positive = sum(v > 0 for v in vals)
        group_positive = sum(v > 0 for v in means.values())
        criterion = mean > 0 and positive >= 24 and group_positive >= 6
        if criterion != stable["summary"][variant]["criterion"]:
            raise ValueError("primary rule differs: " + variant)
        summary[variant] = {"mean": str(mean), "positive_cases": positive, "positive_groups": group_positive,
                            "criterion": criterion, "group_means": {k: str(v) for k, v in means.items()}}
if fingerprints() != expected:
    raise ValueError("input changed during replay")
report = {"schema": "multisuffix-decimal-primary-replay-v1", "inputs_sha256": expected,
          "script_sha256": sha(Path(__file__)), "decimal_precision": 70, "cases": rows, "summary": summary,
          "scope": "Independent direct-softmax calculation from saved float32 logits; Decimal finite precision, not formal exact arithmetic or evidence of model execution, chronology or cleanup."}
data = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
with output.open("xb") as f:
    if f.write(data) != len(data):
        raise OSError("short write")
print("decimal_primary_replay", hashlib.sha256(data).hexdigest(), flush=True)
