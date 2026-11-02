"""Reconstruct the baseline-screened copying intervention from closed, pinned saved bytes.

This related-code replay adapts the public earlier output-dose saved-byte reader.
It loads no model.  It checks saved-byte consistency, not faithful execution,
source protection, process cleanup, a trusted clock, or unobserved exposure.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

if not __debug__ or sys.byteorder != "little":
    raise RuntimeError("requires assertions and a little-endian host")

CASE_SHA = "96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734"
HEADS = (2, 10, 7)
ORDER = ("out_h2", "out_h10", "out_h7")
WIDTH = 768
SLICE_BYTES = 64 * WIDTH * 4
WITNESS_BYTES = 11 * WIDTH * 4
LOGITS_BYTES = 50257 * 4


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned(path, digest):
    raw = path.read_bytes()
    assert sha(raw) == digest, str(path)
    return raw


def float32(bits):
    sign = -1.0 if bits >> 31 else 1.0
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    assert exponent != 255
    significand = fraction if exponent == 0 else (1 << 23) | fraction
    return sign * math.ldexp(float(significand), -149 if exponent == 0 else exponent - 150)


def vals(raw):
    assert len(raw) % 4 == 0
    return tuple(float32(x[0]) for x in struct.iter_unpack("<I", raw))


def norm(values):
    return math.sqrt(math.fsum(x * x for x in values))


def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return (type(a) in (int, float) and type(b) in (int, float)
                and math.isfinite(a) and math.isfinite(b)
                and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def round_f32(x):
    """Independently find adjacent float32 values; break ties to even."""
    assert math.isfinite(x)
    sign = 0x80000000 if math.copysign(1.0, x) < 0 else 0
    x = abs(x)
    if x == 0:
        return sign
    low, high = 0, 0x7f7fffff
    assert x <= float32(high)
    while low <= high:
        middle = (low + high) // 2
        if float32(middle) <= x:
            low = middle + 1
        else:
            high = middle - 1
    lower = high
    if float32(lower) == x:
        return sign | lower
    upper = lower + 1
    dl, du = x - float32(lower), float32(upper) - x
    return sign | (lower if dl < du or (dl == du and lower % 2 == 0) else upper)


def project(vector, weight):
    assert len(vector) == 64 and len(weight) == 64 * WIDTH
    result = tuple(math.fsum(vector[i] * weight[i * WIDTH + j] for i in range(64))
                   for j in range(WIDTH))
    assert all(math.isfinite(x) for x in result)
    return result


def slice_head(raw, head):
    assert len(raw) == WITNESS_BYTES
    start = (10 * WIDTH + 64 * head) * 4
    return raw[start:start + 256]


def delivery(original, fraction):
    before = vals(original)
    before_bits = tuple(x[0] for x in struct.iter_unpack("<I", original))
    after_bits = tuple(round_f32((1.0 - fraction) * x) for x in before)
    after = struct.pack("<64I", *after_bits)
    delta = tuple(y - x for x, y in zip(before, vals(after)))
    return {"fraction": fraction, "source": original, "delivered": after,
            "delta": delta, "input_removed_l2": norm(delta),
            "changed_elements": sum(a != b for a, b in zip(before_bits, after_bits)),
            "changed_bits": sum((a ^ b).bit_count() for a, b in zip(before_bits, after_bits))}


def stats(raw, target):
    v = vals(raw)
    assert len(v) == 50257 and type(target) is int and 0 <= target < len(v)
    maximum = max(v); z = math.log(math.fsum(math.exp(x-maximum) for x in v))
    ties, first = sum(x == maximum for x in v), v.index(maximum)
    return v, {"maximum": maximum, "logz_shift": z, "target_logit": v[target],
               "target_log_probability": (v[target]-maximum)-z, "target_rank": 1+sum(x>v[target] for x in v),
               "top_unique_id": first if ties == 1 else None, "top_representative_id": first, "top_tie_count": ties}


def compare(c, cs, a, ars):
    zc, za, mc, ma = cs["logz_shift"], ars["logz_shift"], cs["maximum"], ars["maximum"]
    delta = math.fsum((ars["target_logit"], -cs["target_logit"], mc, -ma, zc, -za))
    kl, tv = [], []
    for x, y in zip(c, a):
        p, q = math.exp((x-mc)-zc), math.exp((y-ma)-za)
        kl.append(p*math.fsum((x, -y, -mc, ma, -zc, za))); tv.append(abs(p-q))
    top = ("tied/indeterminate" if cs["top_unique_id"] is None or ars["top_unique_id"] is None
           else "changed" if cs["top_unique_id"] != ars["top_unique_id"] else "unchanged")
    return {"target_logit_delta": ars["target_logit"]-cs["target_logit"],
            "target_log_probability_delta": delta, "kl_clean_to_arm": math.fsum(kl),
            "total_variation": math.fsum(tv)/2, "top_change": top}


def cosines(vectors):
    out = {}
    for a, b in ((2, 10), (2, 7), (10, 7)):
        na, nb = norm(vectors[a]), norm(vectors[b])
        out[f"{a}-{b}"] = None if na <= 2**-8 or nb <= 2**-8 else math.fsum(x*y for x,y in zip(vectors[a],vectors[b]))/(na*nb)
    return out


def verified_checkpoint_rows(path):
    """Hash the full safetensors and capture its actual 64-row Conv1D blocks."""
    expected = "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"
    with path.open("rb") as stream:
        prefix = stream.read(8)
        assert len(prefix) == 8
        n = struct.unpack("<Q", prefix)[0]
        assert 0 < n <= 16*1024*1024
        header = stream.read(n)
        assert len(header) == n
        tensor = json.loads(header)["transformer.h.3.attn.c_proj.weight"]
        assert tensor["dtype"] == "F32" and tensor["shape"] == [WIDTH, WIDTH]
        start, stop = tensor["data_offsets"]
        assert type(start) is int and type(stop) is int and start >= 0 and stop-start == WIDTH*WIDTH*4
        base = 8+n+start
        windows = {h: (base+h*SLICE_BYTES, base+(h+1)*SLICE_BYTES) for h in HEADS}
        pieces = {h: bytearray() for h in HEADS}
        offset = 0; hasher = hashlib.sha256(); header_again = bytearray()
        stream.seek(0)
        while True:
            block = stream.read(1024*1024)
            if not block:
                break
            end = offset+len(block); hasher.update(block)
            if offset < 8+n:
                header_again.extend(block[:min(len(block), 8+n-offset)])
            for h, (left, right) in windows.items():
                lo, hi = max(left, offset), min(right, end)
                if lo < hi:
                    pieces[h].extend(block[lo-offset:hi-offset])
            offset = end
    assert bytes(header_again) == prefix+header and hasher.hexdigest() == expected
    assert offset >= 8+n+stop and all(len(pieces[h]) == SLICE_BYTES for h in HEADS)
    return b"".join(pieces[h] for h in HEADS)


SCREEN_SHA = {
    "screen.f32": "7d54f4f40b9bcdf56dd6a34823ceb4da61098c0f2c6ed84c90396272bc7328aa",
    "manifest.json": "4b920bd5f29b54c77914ec8fc402784ceb4b1219da386d900926727547442549",
    "attempt.jsonl": "f5a7857dec21cdc21d855ed4c07686c8b976162e41beae8256c6ab4f0877ade0",
    "result.json": "fcee5b994d9f8d70c0d9c35eae62da8827f5cb92008c9db709c8beb9a733aa99",
}
SCREEN_AUDIT_SHA = "584c0ccf1ad11c5ad4d857bfe73c3a5d70b138fd1c85b5bee6aac6152ba6b186"


def output_plan(inputs, matrix):
    """Build the fixed 24-way plan independently from saved no-op inputs."""
    assert len(inputs) == 8
    originals = [{h: slice_head(raw, h) for h in HEADS} for raw in inputs]
    norms = [{h: norm(project(vals(row[h]), matrix[h])) for h in HEADS} for row in originals]
    q = min(norms[v][h] for v in range(8) for h in HEADS)
    minima = [[v, h] for v in range(8) for h in HEADS if norms[v][h] == q]
    descriptions, plans = None, None
    if q > 2**-8:
        descriptions, plans, predicted = [], [], []
        for v in range(8):
            row, desc = {}, {}
            for h in HEADS:
                p = delivery(originals[v][h], q / norms[v][h])
                p["predicted"] = project(p["delta"], matrix[h])
                p["predicted_output_l2"] = norm(p["predicted"])
                predicted.append(p["predicted_output_l2"])
                desc[str(h)] = {"fraction": p["fraction"], "source_bits_hex": p["source"].hex(),
                                "delivered_bits_hex": p["delivered"].hex(),
                                "input_removed_l2": p["input_removed_l2"],
                                "predicted_output_l2": p["predicted_output_l2"],
                                "changed_elements": p["changed_elements"], "changed_bits": p["changed_bits"]}
                row[h] = p
            descriptions.append(desc)
            plans.append(row)
        valid = all(abs(n-q) <= 1e-4*q for n in predicted) and max(predicted)-min(predicted) <= 1e-4*q
        reason = None if valid else "predicted rounded dose mismatch"
    else:
        valid, reason = False, "dose floor"
    return {"status": "eligible" if valid else "dose-ineligible", "reason": reason, "q_out": q,
            "source_output_norms": [{str(h): row[h] for h in HEADS} for row in norms],
            "minima": minima, "plans": descriptions}, plans


def family_summary(group, baseline_status, assessment, q, comparisons):
    """Reconstruct all descriptive effects and the unrounded strict sign."""
    from fractions import Fraction
    if assessment != "eligible":
        return {"group": group, "baseline_status": baseline_status, "dose_status": "dose-ineligible",
                "status": "dose-ineligible", "q_out": q, "rotations": None, "D": None, "contrast": None,
                "contrast_terms": None, "contrast_sign": None}
    assert len(comparisons) == 8
    names = ("h2", "h10", "h7")
    terms = {h: [] for h in names}
    logits = {h: [] for h in names}
    rotations = []
    for rotation in range(2):
        row = {}
        for h in names:
            p = [comparisons[i][h]["target_log_probability_delta"] for i in range(4*rotation, 4*rotation+4)]
            l = [comparisons[i][h]["target_logit_delta"] for i in range(4*rotation, 4*rotation+4)]
            pt = [p[0], p[0], p[0], -p[1], -p[2], -p[3]]
            lt = [l[0], l[0], l[0], -l[1], -l[2], -l[3]]
            terms[h].extend(pt)
            logits[h].extend(lt)
            row[h] = {"target_log_probability_D": math.fsum(pt)/3, "target_logit_D": math.fsum(lt)/3}
        rotations.append(row)
    contrast_terms = terms["h10"] + [-x for x in terms["h2"]]
    exact = sum((Fraction(x) for x in contrast_terms), Fraction())
    numerator = math.fsum(contrast_terms)
    assert (numerator > 0) == (exact > 0) and (numerator < 0) == (exact < 0)
    difference = lambda a, b: math.fsum(terms[a] + [-x for x in terms[b]])/6
    return {"group": group, "baseline_status": baseline_status, "dose_status": "eligible", "status": "evaluable",
            "q_out": q, "rotations": rotations, "D": {h: math.fsum(terms[h])/6 for h in names},
            "target_logit_D": {h: math.fsum(logits[h])/6 for h in names}, "contrast": numerator/6,
            "contrast_terms": contrast_terms, "contrast_sign": 1 if exact > 0 else -1 if exact < 0 else 0,
            "h10_minus_h7": difference("h10", "h7"), "h2_minus_h7": difference("h2", "h7")}


def decision(families):
    """Apply the fixed eligibility and sign rule to raw reconstructed terms."""
    from fractions import Fraction
    def describe(rows):
        if not rows:
            return {"count": 0, "positive": 0, "mean_contrast": None, "mean_sign": None}
        terms = [x for row in rows for x in row["contrast_terms"]]
        exact = sum((Fraction(x) for x in terms), Fraction())
        total = math.fsum(terms)
        assert (total > 0) == (exact > 0) and (total < 0) == (exact < 0)
        return {"count": len(rows), "positive": sum(row["contrast_sign"] > 0 for row in rows),
                "mean_contrast": total/(6*len(rows)), "mean_sign": 1 if exact > 0 else -1 if exact < 0 else 0}
    evaluable = [f for f in families if f["status"] == "evaluable"]
    supported = [f for f in evaluable if f["baseline_status"] == "supported"]
    unsupported = [f for f in evaluable if f["baseline_status"] == "unsupported"]
    primary = describe(supported)
    status = ("ineligible" if primary["count"] < 6 else
              "criterion-met" if primary["mean_sign"] > 0 and 4*primary["positive"] >= 3*primary["count"] else
              "criterion-not-met")
    return {"primary_status": status, "primary": primary, "all_dose_eligible": describe(evaluable),
            "baseline_unsupported_dose_eligible": describe(unsupported),
            "dose_ineligible_groups": [f["group"] for f in families if f["status"] == "dose-ineligible"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path)
    parser.add_argument("--screen-audit", type=Path)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--synthetic-weights-sha256", help="fixture only; requires explicit synthetic result and manifest")
    parser.add_argument("--synthetic-baseline", type=Path, help="fixture-only invented baseline matrix")
    parser.add_argument("--synthetic-baseline-sha256", help="fixture-only baseline digest")
    parser.add_argument("--synthetic-one-family", action="store_true", help="test-only partial mismatch fixture")
    args = parser.parse_args()

    case_raw = pinned(args.cases, CASE_SHA)
    cases = json.loads(case_raw)["cases"]
    assert len(cases) == 192 and len({c["case_id"] for c in cases}) == 192
    assert all(c["group"] == i//8+1 and c["variant"] == i%8 and
               c["split"] == ("development" if c["group"] % 2 else "held-out")
               for i, c in enumerate(cases))
    held = [c for c in cases if c["split"] == "held-out"]
    result_raw = pinned(args.run/"result.json", args.result_sha256)
    journal_raw = pinned(args.run/"attempt.jsonl", args.journal_sha256)
    result = json.loads(result_raw)
    assert result["schema"] == "distilgpt2-copy-behavior-intervention-result-v1" and result["status"] == "complete"
    assert not (args.run/"failure.json").exists()
    manifest_raw = (args.run/"manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    synthetic = args.synthetic_weights_sha256 is not None
    if synthetic:
        assert args.checkpoint is None and args.screen_dir is None and args.screen_audit is None
        assert args.synthetic_baseline is not None and args.synthetic_baseline_sha256
        assert result.get("synthetic_fixture") is True and manifest.get("synthetic_fixture") is True
        screen_raw = pinned(args.synthetic_baseline, args.synthetic_baseline_sha256)
        assert manifest["baseline_raw_sha256"] == sha(screen_raw)
        # These are injected labels for the fake denominator fixture, not scores of the invented rows.
        support = {g: ("supported" if g in {2, 10, 12, 16, 20} else "unsupported") for g in range(1, 25)}
        screen_audit_sha = None
    else:
        assert not args.synthetic_one_family
        assert args.checkpoint is not None and args.screen_dir is not None and args.screen_audit is not None
        assert args.synthetic_baseline is None and args.synthetic_baseline_sha256 is None
        assert result.get("synthetic_fixture") is not True and manifest.get("synthetic_fixture") is not True
        screen_records = {name: pinned(args.screen_dir/name, expected) for name, expected in SCREEN_SHA.items()}
        screen_raw = screen_records["screen.f32"]
        audit_raw = pinned(args.screen_audit, SCREEN_AUDIT_SHA)
        audit = json.loads(audit_raw)
        assert audit["cases_sha256"] == CASE_SHA and audit["raw_sha256"] == SCREEN_SHA["screen.f32"]
        assert audit["row_count"] == 192 and audit["synthetic_fixture"] is False
        families = audit["summary"]["families"]
        assert [f["group"] for f in families] == list(range(1, 25))
        support = {f["group"]: f["baseline_status"] for f in families}
        assert all(s in ("supported", "unsupported") for s in support.values())
        assert manifest["baseline_hashes"] == SCREEN_SHA
        assert manifest["baseline_independent_audit_sha256"] == SCREEN_AUDIT_SHA
        screen_audit_sha = sha(audit_raw)
    if args.synthetic_one_family:
        assert result.get("partial_fixture") is True and len(result["families"]) == 1
        assert result["families"][0]["group"] == 2 and result["decision"] is None
        held = held[:8]
    else:
        assert result.get("partial_fixture") is not True
    assert len(screen_raw) == 192*LOGITS_BYTES
    assert manifest["schema"] == "distilgpt2-copy-behavior-intervention-run-v1"
    assert manifest["cases_sha256"] == CASE_SHA and result["manifest_sha256"] == sha(manifest_raw)
    if not synthetic:
        assert manifest["weights_sha256"] == "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"

    raw_weights = (args.run/"projection-weights.f32").read_bytes()
    assert len(raw_weights) == 3*SLICE_BYTES
    if synthetic:
        assert sha(raw_weights) == args.synthetic_weights_sha256
    else:
        assert raw_weights == verified_checkpoint_rows(args.checkpoint)
    matrix = {h: vals(raw_weights[i*SLICE_BYTES:(i+1)*SLICE_BYTES]) for i, h in enumerate(HEADS)}
    assert result["weight_sha256"] == sha(raw_weights)
    assert result["weight_head_sha256"] == {str(h): sha(raw_weights[i*SLICE_BYTES:(i+1)*SLICE_BYTES]) for i,h in enumerate(HEADS)}
    assert result["state_sha256_before"] == result["state_sha256_after"]
    assert result["buffers_sha256_before"] == result["buffers_sha256_after"]
    assert {p.name for p in args.run.glob("*.f32")} == {c["case_id"]+".f32" for c in held} | {"projection-weights.f32"}
    events = [json.loads(line) for line in journal_raw.splitlines()]
    position = 0
    records, action_reports = [], []
    offsets = {c["case_id"]: 0 for c in held}
    runtime = events[position]
    position += 1
    assert runtime["event"] == "runtime_ready" and runtime["weight_order"] == list(HEADS)
    assert runtime["weight_sha256"] == sha(raw_weights) and runtime["weight_head_sha256"] == result["weight_head_sha256"]
    assert runtime["actual_runtime"] == result["runtime"]

    def action(case, kind):
        nonlocal position
        start, event, complete = events[position:position+3]
        position += 3
        assert [(e["event"], e["case_id"], e["action"]) for e in (start, event, complete)] == [
            (e, case["case_id"], kind) for e in ("start", "raw_recorded", "complete")]
        rec = {k: v for k, v in complete.items() if k != "event"}
        assert event["raw"] == rec["raw"]
        assert (rec["group"], rec["variant"], rec["target_id"]) == (case["group"], case["variant"], case["target_id"])
        payload = {}
        with (args.run/(case["case_id"]+".f32")).open("rb") as stream:
            for name, segment in sorted(rec["raw"].items(), key=lambda item: item[1]["offset"]):
                assert segment["offset"] == offsets[case["case_id"]]
                stream.seek(segment["offset"])
                raw = stream.read(segment["bytes"])
                assert len(raw) == segment["bytes"] and sha(raw) == segment["sha256"]
                offsets[case["case_id"]] += len(raw)
                payload[name] = raw
        expected = {"logits"} if kind in ("clean", "clean_after") else {"logits", "cproj_before", "cproj_after", "cproj_output"}
        assert set(payload) == expected and len(payload["logits"]) == LOGITS_BYTES
        for name in expected - {"logits"}:
            assert len(payload[name]) == WITNESS_BYTES
            vals(payload[name])
        vector, score = stats(payload["logits"], case["target_id"])
        assert close(score, rec["score"])
        records.append(rec)
        return payload, vector, score, rec

    reconstructed = []
    group_numbers = [2] if args.synthetic_one_family else list(range(2, 25, 2))
    for index, group in enumerate(group_numbers):
        batch = held[8*index:8*index+8]
        assert [c["group"] for c in batch] == [group]*8 and [c["variant"] for c in batch] == list(range(8))
        caches = []
        for case in batch:
            clean, values, score, _ = action(case, "clean")
            baseline_index = 8*(group-1)+case["variant"]
            assert clean["logits"] == screen_raw[baseline_index*LOGITS_BYTES:(baseline_index+1)*LOGITS_BYTES]
            noop, _, _, _ = action(case, "noop")
            assert clean["logits"] == noop["logits"] and noop["cproj_before"] == noop["cproj_after"]
            caches.append({"clean": clean, "values": values, "score": score, "noop": noop,
                           "comparisons": {}, "vectors": {}})
        plan, plans = output_plan([c["noop"]["cproj_before"] for c in caches], matrix)
        event = events[position]
        position += 1
        assert event["event"] == "family_plan" and event["group"] == group and close(plan, event["plan"])
        for variant, (case, cache) in enumerate(zip(batch, caches)):
            if plan["status"] == "eligible":
                shift = (8*(group-1)+variant) % 3
                for kind in ORDER[shift:]+ORDER[:shift]:
                    head = int(kind.split("h")[1])
                    wanted = plans[variant][head]
                    payload, vector, score, rec = action(case, kind)
                    before = cache["noop"]["cproj_before"]
                    begin = (10*WIDTH+64*head)*4
                    assert payload["cproj_before"] == before
                    assert payload["cproj_after"] == before[:begin]+wanted["delivered"]+before[begin+256:]
                    noop_output = cache["noop"]["cproj_output"]
                    assert payload["cproj_output"][:10*WIDTH*4] == noop_output[:10*WIDTH*4]
                    pre = vals(noop_output)[10*WIDTH:]
                    post = vals(payload["cproj_output"])[10*WIDTH:]
                    actual = tuple(b-a for a,b in zip(pre,post))
                    dose = {"input_removed_l2": wanted["input_removed_l2"],
                            "predicted_output_l2": norm(wanted["predicted"]), "actual_output_l2": norm(actual),
                            "predicted_vector_residual_l2": norm(a-b for a,b in zip(actual,wanted["predicted"]))}
                    comparison = compare(cache["values"], cache["score"], vector, score)
                    assert close(dose, rec["dose"]) and close(comparison, rec["comparison"])
                    cache["comparisons"]["h"+str(head)] = comparison
                    cache["vectors"][head] = actual
                    action_reports.append({"case_id":case["case_id"], "action":kind, "dose":dose,
                                           "comparison":comparison, "score":score})
            after, _, _, _ = action(case, "clean_after")
            assert after["logits"] == cache["clean"]["logits"]
        if plan["status"] == "eligible":
            norms, residuals, rows, pair_cosines = [], [], [], []
            for variant, cache in enumerate(caches):
                row, vectors = {}, cache["vectors"]
                assert set(vectors) == set(HEADS)
                for h in HEADS:
                    n = norm(vectors[h])
                    residual = norm(a-b for a,b in zip(vectors[h], plans[variant][h]["predicted"]))
                    norms.append(n)
                    residuals.append(residual)
                    row[str(h)] = {"actual_output_l2":n, "relative_error":abs(n-plan["q_out"])/plan["q_out"],
                                   "predicted_vector_residual_l2":residual}
                rows.append(row)
                pair_cosines.append(cosines(vectors))
            q = plan["q_out"]
            valid = (all(abs(n-q)<=1e-4*q for n in norms) and all(r<=1e-4*q for r in residuals)
                     and max(norms)-min(norms)<=1e-4*q)
            assessment = {"status":"eligible" if valid else "dose-ineligible",
                          "reason":None if valid else "actual output dose mismatch", "q_out":q,
                          "variants":rows, "pair_cosines":pair_cosines, "actual_range":max(norms)-min(norms)}
        else:
            assessment = {"status":"dose-ineligible", "reason":plan["reason"], "q_out":plan["q_out"]}
        summary = family_summary(group, support[group], assessment["status"], plan["q_out"],
                                 [c["comparisons"] for c in caches])
        family = {"group":group, "baseline_status":support[group], "output_plan":plan,
                  "output_assessment":assessment, "summary":summary,
                  "clean_target_ranks":[c["score"]["target_rank"] for c in caches]}
        complete = events[position]
        position += 1
        assert complete["event"] == "family_complete" and close(family, {k:v for k,v in complete.items() if k != "event"})
        assert close(family, result["families"][index])
        reconstructed.append(family)

    assert position == len(events) and records == result["records"]
    assert all((args.run/(case_id+".f32")).stat().st_size == size for case_id,size in offsets.items())
    planned_count = sum(f["output_plan"]["status"] == "eligible" for f in reconstructed)
    expected_forwards = len(held)*3 + 24*planned_count
    expected_bytes = len(held)*(3*LOGITS_BYTES+3*WITNESS_BYTES) + 24*planned_count*(LOGITS_BYTES+3*WITNESS_BYTES)
    assert len(records) == expected_forwards
    assert result["counts"] == {"submitted":expected_forwards, "model_returns":expected_forwards,
                                "returned":expected_forwards, "retained":expected_bytes}
    assert sum(offsets.values()) == expected_bytes <= 154720512
    outcome = None if args.synthetic_one_family else decision([f["summary"] for f in reconstructed])
    assert close(outcome, result["decision"])
    print(json.dumps({"status":"synthetic_partial_pass" if args.synthetic_one_family else "complete_saved_byte_pass", "synthetic_fixture":synthetic,
        "cases_sha256":sha(case_raw), "journal_sha256":sha(journal_raw), "result_sha256":sha(result_raw),
        "manifest_sha256":sha(manifest_raw), "baseline_matrix_sha256":sha(screen_raw),
        "baseline_audit_sha256":screen_audit_sha, "weight_sha256":sha(raw_weights),
        "checkpoint_rows_verified":not synthetic, "forwards":len(records), "action_bytes":sum(offsets.values()),
        "families":reconstructed, "actions":action_reports, "decision":outcome,
        "scope":"Related-code saved-byte replay of fixed held-out cases, full logits, 24-way dose and landing,"
                " controls and decisions; no model execution, process, clock, private exposure or provenance attestation."
                " Synthetic support labels are injected for denominator tests only."},
        sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


