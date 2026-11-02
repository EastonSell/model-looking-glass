"""Reconstruct the family-matched copying run from closed, pinned saved bytes.

This related-code replay builds on our earlier independent prompt-dose reader.
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

CASE_SHA = "e45fe8cfe8b28a4f4308a4d3b2206f35cebe8154c7b4301ac1356c28258b135e"
HEADS = (2, 10, 7)
ORDER = ("out_h2", "out_h10", "out_h7", "in_h2", "in_h10", "in_h7")
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


def input_plan(raw):
    vals(raw)
    originals = {h: slice_head(raw, h) for h in HEADS}
    norms = {h: norm(vals(originals[h])) for h in HEADS}
    q = min(norms.values())
    plans = None
    if q > 2**-20:
        plans = {h: delivery(originals[h], q / norms[h]) for h in HEADS}
    achieved = [plans[h]["input_removed_l2"] for h in HEADS] if plans else []
    valid = bool(plans) and all(abs(x - q) <= 1e-5*q for x in achieved) and max(achieved)-min(achieved) <= 1e-5*q
    reason = None if valid else "realized dose mismatch" if plans else "dose floor"
    descriptions = {}
    for h in HEADS:
        item = {"source_l2": norms[h], "source_bits_hex": originals[h].hex()}
        if plans:
            p = plans[h]
            item.update({"fraction": p["fraction"], "achieved": p["input_removed_l2"],
                         "changed_elements": p["changed_elements"], "changed_bits": p["changed_bits"],
                         "delivered_bits_hex": p["delivered"].hex()})
        descriptions[str(h)] = item
    return {"status": "eligible" if valid else "dose-ineligible", "reason": reason,
            "q": q, "heads": descriptions,
            "plan_unavailable": None if plans else "plan unavailable: dose floor"}, plans


def output_plan(inputs, matrix):
    assert len(inputs) == 4
    originals = [{h: slice_head(raw, h) for h in HEADS} for raw in inputs]
    source_norms = [{h: norm(project(vals(row[h]), matrix[h])) for h in HEADS} for row in originals]
    q = min(source_norms[v][h] for v in range(4) for h in HEADS)
    minima = [[v, h] for v in range(4) for h in HEADS if source_norms[v][h] == q]
    descriptions, plans = None, None
    if q > 2**-8:
        plans, descriptions, predicted = [], [], []
        for v in range(4):
            row, desc = {}, {}
            for h in HEADS:
                p = delivery(originals[v][h], q/source_norms[v][h])
                p["predicted"] = project(p["delta"], matrix[h])
                p["predicted_output_l2"] = norm(p["predicted"])
                predicted.append(p["predicted_output_l2"])
                desc[str(h)] = {"fraction": p["fraction"], "source_bits_hex": p["source"].hex(),
                                "delivered_bits_hex": p["delivered"].hex(),
                                "input_removed_l2": p["input_removed_l2"],
                                "predicted_output_l2": p["predicted_output_l2"],
                                "changed_elements": p["changed_elements"], "changed_bits": p["changed_bits"]}
                row[h] = p
            plans.append(row); descriptions.append(desc)
        valid = all(abs(x-q) <= 1e-4*q for x in predicted) and max(predicted)-min(predicted) <= 1e-4*q
        reason = None if valid else "predicted rounded dose mismatch"
    else:
        valid, reason = False, "dose floor"
    return {"status": "eligible" if valid else "dose-ineligible", "reason": reason, "q_out": q,
            "source_output_norms": [{str(h): row[h] for h in HEADS} for row in source_norms],
            "minima": minima, "plans": descriptions}, plans


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


def family_summary(rows):
    if any(row["status"] != "eligible" for row in rows):
        return {"status": "dose-ineligible", "contrast": None}
    d, logits, deltas = {}, {}, {}
    for arm in ("h2", "h10", "h7"):
        p = [r["comparisons"][arm]["target_log_probability_delta"] for r in rows]
        l = [r["comparisons"][arm]["target_logit_delta"] for r in rows]
        d[arm] = math.fsum((p[0], -math.fsum(p[1:])/3))
        logits[arm] = math.fsum((l[0], -math.fsum(l[1:])/3))
        deltas[arm] = {"target_log_probability": p, "target_logit": l}
    normalized = [(r["comparisons"]["h10"]["target_log_probability_delta"]-r["comparisons"]["h2"]["target_log_probability_delta"])/r["q"] for r in rows]
    return {"status": "evaluable", "D": d, "target_logit_D": logits, "case_deltas": deltas,
            "contrast": d["h10"]-d["h2"], "target_logit_contrast": logits["h10"]-logits["h2"],
            "h10_minus_h7": d["h10"]-d["h7"], "h2_minus_h7": d["h2"]-d["h7"],
            "dose_normalized_contrast": math.fsum((normalized[0], -math.fsum(normalized[1:])/3))}


def decision(families):
    good = [f for f in families if f["status"] == "evaluable"]
    n, positive = len(good), sum(f["contrast"] > 0 for f in good)
    mean = math.fsum(f["contrast"] for f in good)/n if n else None
    return {"status": "ineligible" if n < 6 else "criterion-met" if mean > 0 and 4*positive >= 3*n else "criterion-not-met",
            "evaluable_families": n, "strictly_positive": positive, "mean_contrast": mean}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--synthetic-weights-sha256", help="test-only: requires explicitly marked fabricated result")
    args = parser.parse_args()
    case_raw = pinned(args.cases, CASE_SHA)
    cases = json.loads(case_raw)["cases"]
    assert len(cases) == 32 and len({c["case_id"] for c in cases}) == 32
    assert all(c["group"] == i//4+1 and c["variant"] == i%4 for i,c in enumerate(cases))
    result_raw = pinned(args.run/"result.json", args.result_sha256)
    journal_raw = pinned(args.run/"attempt.jsonl", args.journal_sha256)
    result = json.loads(result_raw)
    assert result["schema"] == "distilgpt2-copy-output-dose-result-v1" and result["status"] == "complete"
    assert not (args.run/"failure.json").exists()
    synthetic = args.synthetic_weights_sha256 is not None
    if synthetic:
        assert args.checkpoint is None and result.get("synthetic_fixture") is True
        manifest_raw = None
    else:
        assert args.checkpoint is not None and result.get("synthetic_fixture") is not True
        manifest_raw = (args.run/"manifest.json").read_bytes(); manifest = json.loads(manifest_raw)
        assert manifest["schema"] == "distilgpt2-copy-output-dose-run-v1"
        assert manifest["cases_sha256"] == sha(case_raw) and result["manifest_sha256"] == sha(manifest_raw)
        assert manifest["weights_sha256"] == "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"
    raw_weights = (args.run/"projection-weights.f32").read_bytes()
    assert len(raw_weights) == 3*SLICE_BYTES
    if synthetic:
        assert sha(raw_weights) == args.synthetic_weights_sha256
    else:
        assert raw_weights == verified_checkpoint_rows(args.checkpoint)
    matrix = {h: vals(raw_weights[i*SLICE_BYTES:(i+1)*SLICE_BYTES]) for i,h in enumerate(HEADS)}
    assert result["weight_sha256"] == sha(raw_weights)
    assert result["weight_head_sha256"] == {str(h): sha(raw_weights[i*SLICE_BYTES:(i+1)*SLICE_BYTES]) for i,h in enumerate(HEADS)}
    assert result["state_sha256_before"] == result["state_sha256_after"]
    assert result["buffers_sha256_before"] == result["buffers_sha256_after"]
    assert {p.name for p in args.run.glob("*.f32")} == {c["case_id"]+".f32" for c in cases} | {"projection-weights.f32"}
    events = [json.loads(line) for line in journal_raw.splitlines()]
    position = 0; records = []; offsets = {c["case_id"]: 0 for c in cases}; action_reports = []
    runtime = events[position]; position += 1
    assert runtime["event"] == "runtime_ready" and runtime["weight_order"] == list(HEADS)
    assert runtime["weight_sha256"] == sha(raw_weights) and runtime["weight_head_sha256"] == result["weight_head_sha256"]

    def action(case, kind):
        nonlocal position
        start, event, complete = events[position:position+3]; position += 3
        assert [(e["event"], e["case_id"], e["action"]) for e in (start,event,complete)] == [
            (e, case["case_id"], kind) for e in ("start", "raw_recorded", "complete")]
        rec = {k:v for k,v in complete.items() if k != "event"}
        assert event["raw"] == rec["raw"]
        assert (rec["group"], rec["variant"], rec["target_id"]) == (case["group"], case["variant"], case["target_id"])
        payload = {}
        with (args.run/(case["case_id"]+".f32")).open("rb") as stream:
            for name, segment in sorted(rec["raw"].items(), key=lambda item: item[1]["offset"]):
                assert segment["offset"] == offsets[case["case_id"]]
                stream.seek(segment["offset"]); raw = stream.read(segment["bytes"])
                assert len(raw) == segment["bytes"] and sha(raw) == segment["sha256"]
                offsets[case["case_id"]] += len(raw); payload[name] = raw
        expected = {"logits"} if kind in ("clean", "clean_after") else {"logits", "cproj_before", "cproj_after", "cproj_output"}
        assert set(payload) == expected and len(payload["logits"]) == LOGITS_BYTES
        for name in expected-{"logits"}:
            assert len(payload[name]) == WITNESS_BYTES
            vals(payload[name])
        vector, score = stats(payload["logits"], case["target_id"])
        assert close(score, rec["score"])
        records.append(rec)
        return payload, vector, score, rec

    families = []
    for group in range(8):
        batch = cases[4*group:4*group+4]; caches = []
        for case in batch:
            clean, cv, cs, _ = action(case, "clean")
            noop, _, _, _ = action(case, "noop")
            assert clean["logits"] == noop["logits"] and noop["cproj_before"] == noop["cproj_after"]
            caches.append({"clean": clean, "values": cv, "score": cs, "noop": noop,
                           "comparisons": {}, "vectors": {"out": {}, "in": {}}})
        out, out_plans = output_plan([c["noop"]["cproj_before"] for c in caches], matrix)
        input_data = [input_plan(c["noop"]["cproj_before"]) for c in caches]
        event = events[position]; position += 1
        assert event["event"] == "dose_plan" and event["group"] == group+1
        assert close(out, event["output"]) and close([p[0] for p in input_data], event["input"])
        all_actual_norms, all_residuals, assessment_rows, output_cosines = [], [], [], []
        for variant, (case, cache, (inp, in_plans)) in enumerate(zip(batch,caches,input_data)):
            shift = (4*group+variant) % 6
            for kind in ORDER[shift:]+ORDER[:shift]:
                scheme, head_text = kind.split("_"); head = int(head_text[1:])
                selected = out if scheme == "out" else inp
                if selected["status"] != "eligible":
                    skipped = events[position]; position += 1
                    assert skipped == {"event": "skipped", "case_id": case["case_id"], "action": kind, "reason": selected["reason"]}
                    continue
                plan = out_plans[variant][head] if scheme == "out" else in_plans[head]
                payload, vector, score, rec = action(case,kind)
                before = cache["noop"]["cproj_before"]
                begin = (10*WIDTH+64*head)*4
                assert payload["cproj_before"] == before
                assert payload["cproj_after"] == before[:begin]+plan["delivered"]+before[begin+256:]
                noop_output = cache["noop"]["cproj_output"]
                assert payload["cproj_output"][:10*WIDTH*4] == noop_output[:10*WIDTH*4]
                base_output = vals(noop_output)[10*WIDTH:]; edited_output = vals(payload["cproj_output"])[10*WIDTH:]
                actual = tuple(b-a for a,b in zip(base_output,edited_output))
                predicted = project(plan["delta"],matrix[head])
                dose = {"input_removed_l2": plan["input_removed_l2"], "predicted_output_l2": norm(predicted),
                        "actual_output_l2": norm(actual), "predicted_vector_residual_l2": norm(a-b for a,b in zip(actual,predicted))}
                comparison = compare(cache["values"],cache["score"],vector,score)
                assert close(dose,rec["dose"]) and close(comparison,rec["comparison"])
                cache["comparisons"][kind] = comparison; cache["vectors"][scheme][head] = actual
                action_reports.append({"case_id":case["case_id"],"action":kind,"dose":dose,"comparison":comparison,"score":score})
            after, _, _, _ = action(case,"clean_after")
            assert after["logits"] == cache["clean"]["logits"]
        if out["status"] == "eligible":
            for v, cache in enumerate(caches):
                desc = {}; vectors = cache["vectors"]["out"]
                assert set(vectors) == set(HEADS)
                for h in HEADS:
                    actual = vectors[h]; n = norm(actual)
                    residual = norm(a-b for a,b in zip(actual,out_plans[v][h]["predicted"]))
                    all_actual_norms.append(n); all_residuals.append(residual)
                    desc[str(h)] = {"actual_output_l2":n, "relative_error":abs(n-out["q_out"])/out["q_out"],
                                    "predicted_vector_residual_l2":residual}
                assessment_rows.append(desc); output_cosines.append(cosines(vectors))
            q = out["q_out"]
            valid = (all(abs(n-q)<=1e-4*q for n in all_actual_norms)
                     and all(r<=1e-4*q for r in all_residuals)
                     and max(all_actual_norms)-min(all_actual_norms)<=1e-4*q)
            assessment = {"status":"eligible" if valid else "dose-ineligible",
                          "reason":None if valid else "actual output dose mismatch", "q_out":q,
                          "variants":assessment_rows, "pair_cosines":output_cosines,
                          "actual_range":max(all_actual_norms)-min(all_actual_norms)}
        else:
            assessment = {"status":"dose-ineligible", "reason":out["reason"], "q_out":out["q_out"]}
        regimes = {"output":[],"input":[]}
        for cache,(inp,_) in zip(caches,input_data):
            for label,scheme,status,q in (("output","out",assessment["status"],out["q_out"]),
                                           ("input","in",inp["status"],inp["q"])):
                regimes[label].append({"status":status,"q":q,"comparisons":{
                    "h"+str(h):cache["comparisons"].get(scheme+"_h"+str(h)) for h in HEADS}})
        summaries = {label:family_summary(rows) for label,rows in regimes.items()}
        clean = [c["score"]["target_log_probability"] for c in caches]
        family = {"group":group+1,"output_plan":out,"input_plans":[p[0] for p in input_data],
                  "output_assessment":assessment,"summaries":summaries,
                  "input_output_cosines":[cosines(c["vectors"]["in"]) if set(c["vectors"]["in"]) == set(HEADS) else None for c in caches],
                  "clean_target_ranks":[c["score"]["target_rank"] for c in caches],
                  "clean_repeat_minus_shuffle":math.fsum((clean[0],-math.fsum(clean[1:])/3))}
        complete = events[position]; position += 1
        assert complete["event"] == "family_complete" and close(family,{k:v for k,v in complete.items() if k != "event"})
        assert close(family,result["families"][group]); families.append(family)

    assert position == len(events) and records == result["records"]
    assert all((args.run/(case_id+".f32")).stat().st_size == size for case_id,size in offsets.items())
    in_count = sum(p["status"] == "eligible" for f in families for p in f["input_plans"])
    out_count = sum(f["output_plan"]["status"] == "eligible" for f in families)
    expected_forwards = 96+3*in_count+12*out_count
    expected_bytes = 32*(3*LOGITS_BYTES+3*WITNESS_BYTES)+(3*in_count+12*out_count)*(LOGITS_BYTES+3*WITNESS_BYTES)
    assert len(records) == expected_forwards == result["submitted_forwards"] == result["model_call_returns_observed"] == result["returned_forwards"]
    assert sum(offsets.values()) == expected_bytes == result["raw_payload_bytes"] <= 80604288
    decisions = {label:decision([f["summaries"][label] for f in families]) for label in ("output","input")}
    assert close(decisions,result["decisions"])
    print(json.dumps({"status":"complete_saved_byte_pass", "synthetic_fixture":synthetic,
                      "cases_sha256":sha(case_raw),"journal_sha256":sha(journal_raw),"result_sha256":sha(result_raw),
                      "manifest_sha256":sha(manifest_raw) if manifest_raw is not None else None,
                      "weight_sha256":sha(raw_weights),"checkpoint_rows_verified":not synthetic,
                      "forwards":len(records),"action_bytes":sum(offsets.values()),"families":families,
                      "actions":action_reports,"decisions":decisions,
                      "scope":"Related-code saved-byte replay, including dose/landing/score/decision reconstruction; no model execution, process or exposure attestation."},
                     sort_keys=True,allow_nan=False))


if __name__ == "__main__":
    main()
