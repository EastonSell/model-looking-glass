"""Replay a closed, complete year/label v2 attempt from its saved bytes.

This reader adapts the author's earlier v1 saved-byte replay. Its arithmetic
is separate from the other reviewer's reader, but it is related to the
author's earlier implementation. Run only after the producer has exited and
its files are quiescent. This is not an execution or provenance attestation.
"""
import argparse
import array
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

if not __debug__ or sys.byteorder != "little":
    raise RuntimeError("Requires assertions enabled and little-endian host")
probe = array.array("f")
probe.frombytes(struct.pack("<II", 1, 0x80000001))
if probe.itemsize != 4 or list(probe) != [2.0**-149, -(2.0**-149)]:
    raise RuntimeError("Float32 subnormals are not preserved")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", required=True, type=Path, help="closed complete run directory")
parser.add_argument("--cases", required=True, type=Path, help="pinned public case JSON")
parser.add_argument("--journal-sha256", required=True, help="expected closed journal SHA-256")
args = parser.parse_args()
if len(args.journal_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.journal_sha256):
    raise ValueError("journal SHA-256 must be 64 lowercase hexadecimal characters")
root = args.run
case_file = args.cases
cases_raw = case_file.read_bytes()
assert hashlib.sha256(cases_raw).hexdigest() == "07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08"
full = json.loads(cases_raw)["cases"]
rows = full[12:]
assert len(full) == 48 and len(rows) == 36
byid = {r["case_id"]: r for r in rows}
assert len(byid) == 36
actions = ("self", "year_final", "label_final", "year_at_zero", "rotated_year_final")
conditions = ("after", "before", "near")
labels = (("alpha", "beta"), ("red", "blue"))
rowbytes = 768 * 4
digits = (405,486,2999,3070,3023,2713,3312,2998,2919,2931,940,1157,1065,1485,1415,1314,1433,1558,
1507,1129,1238,2481,1828,1954,1731,1495,2075,1983,2078,1959,1270,3132,2624,2091,2682,2327,
2623,2718,2548,2670,1821,3901,3682,3559,2598,2231,3510,2857,2780,2920,1120,4349,4309,4310,
4051,2816,3980,3553,3365,3270,1899,5333,5237,5066,2414,2996,2791,3134,3104,3388,2154,4869,
4761,4790,4524,2425,4304,3324,3695,3720,1795,6659,6469,5999,5705,5332,4521,5774,3459,4531,
3829,6420,5892,6052,5824,3865,4846,5607,4089,2079)
numeric = set(digits)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def floats(raw):
    assert len(raw) % 4 == 0
    v = array.array("f")
    assert v.itemsize == 4
    v.frombytes(raw)
    assert all(math.isfinite(x) for x in v)
    return v


def norm(values):
    return math.sqrt(math.fsum(x*x for x in values))


def band(v, left, right):
    a, b = [v[i] for i in left], [v[i] for i in right]
    u, w = max(a), max(b)
    return ((u-w) + math.log(math.fsum(math.exp(x-u) for x in a))) - math.log(math.fsum(math.exp(x-w) for x in b))


def score(raw, row):
    v = floats(raw)
    assert len(v) == 50257
    m = max(v)
    weights = [math.exp(float(x)-m) for x in v]
    total = math.fsum(weights)
    return {"band_score": band(v, row["middle_ids"], row["above_ids"]),
            "local_score": band(v, row["local_above_ids"], row["local_below_ids"]),
            "aligned_01_99_full_mass": math.fsum(weights[i] for i in digits[1:])/total,
            "zero_00_next_token_mass": weights[digits[0]]/total,
            "other_vocabulary_mass": math.fsum(x for i, x in enumerate(weights) if i not in numeric)/total,
            "top_token_id": v.index(m)}


for row in rows:
    assert row["family_index"] in (1, 2, 3)
    for key, position in (("year_donor_case_id", 10), ("label_donor_case_id", 3)):
        donor = byid[row[key]]
        assert donor["family_index"] == row["family_index"] and donor[key] == row["case_id"]
        assert [i for i, (a, b) in enumerate(zip(row["input_ids"], donor["input_ids"])) if a != b] == [position]
    assert row["rotation"] == (row["family_index"] + 2*row["condition_index"] + row["label_index"] + 3*row["anchor_index"]) % 5

manifest_raw = (root/"manifest.json").read_bytes()
manifest = json.loads(manifest_raw)
result_raw = (root/"result.json").read_bytes()
result = json.loads(result_raw)
journal_raw = (root/"attempt.jsonl").read_bytes()
assert digest(journal_raw) == args.journal_sha256
events = [json.loads(line) for line in journal_raw.splitlines()]
assert manifest["cases"] == rows and result["manifest_sha256"] == digest(manifest_raw)
assert result["status"] == "complete" and result["interventions_performed"] is True
expected = [(r["case_id"], a) for r in rows for a in ("clean", "noop")]
for row in rows:
    s = (row["family_index"] + 2*row["condition_index"] + row["label_index"] + 3*row["anchor_index"]) % 5
    expected += [(row["case_id"], a) for a in actions[s:] + actions[:s] + ("clean_after",)]
assert len(expected) == 288
assert [(e["event"], e["case_id"], e["action"]) for e in events if e["event"] in ("start", "raw_recorded", "complete")] == [
    (stage, case, action) for case, action in expected for stage in ("start", "raw_recorded", "complete")]
records = [{k: v for k, v in e.items() if k != "event"} for e in events if e["event"] == "complete"]
assert records == result["records"]
assert [e["raw"] for e in events if e["event"] == "raw_recorded"] == [r["raw"] for r in records]

seen, offsets = {}, {key: 0 for key in byid}
for record in records:
    row, action = byid[record["case_id"]], record["action"]
    payload = {}
    with (root/(row["case_id"] + ".f32")).open("rb") as stream:
        for name, part in sorted(record["raw"].items(), key=lambda item: item[1]["offset"]):
            assert part["offset"] == offsets[row["case_id"]]
            stream.seek(part["offset"])
            raw = stream.read(part["bytes"])
            assert len(raw) == part["bytes"] and digest(raw) == part["sha256"]
            offsets[row["case_id"]] += len(raw)
            payload[name] = raw
    assert set(payload) == ({"logits"} if action in ("clean", "clean_after") else {"logits", "block_before", "block_after"})
    measured = score(payload["logits"], row)
    assert measured == record["scores"]
    seen[row["case_id"], action] = record, payload, measured
assert sum(offsets.values()) == result["raw_payload_bytes"] == 77802624
assert all((root/(key+".f32")).stat().st_size == count for key, count in offsets.items())
assert {p.name for p in root.glob("*.f32")} == {key+".f32" for key in byid}
for row in rows:
    key = row["case_id"]
    clean = seen[key, "clean"][1]["logits"]
    noop = seen[key, "noop"][1]
    assert noop["logits"] == clean == seen[key, "self"][1]["logits"] == seen[key, "clean_after"][1]["logits"]
    assert noop["block_before"] == noop["block_after"] and len(noop["block_before"]) == len(row["input_ids"])*rowbytes

clean = {r["case_id"]: seen[r["case_id"], "clean"][2] for r in rows}
families, all_differences = [], []
for offset, index in enumerate((1, 2, 3)):
    group = rows[offset*12:(offset+1)*12]
    cond = {}
    for c in conditions:
        cond[c] = {}
        for label in labels[index % 2]:
            pair = [r for r in group if r["condition"] == c and r["label"] == label]
            assert [r["year_role"] for r in pair] == ["low", "high"]
            low, high = (clean[r["case_id"]] for r in pair)
            gap = low["band_score"] - high["band_score"]
            all_differences.append(gap)
            cond[c][label] = {"low": low, "high": high, "low_minus_high": gap}
    families.append({"family_index": index, "conditions": cond})
clean_summary = {"families": families,
                 "mean_band_score": math.fsum(clean[r["case_id"]]["band_score"] for r in rows)/36,
                 "mean_local_score": math.fsum(clean[r["case_id"]]["local_score"] for r in rows)/36,
                 "mean_low_minus_high": math.fsum(all_differences)/18}
assert clean_summary == result["clean_summary"]
assert [e["summary"] for e in events if e["event"] == "clean_summary"] == [clean_summary]
assert [e["pairs_checked"] for e in events if e["event"] == "position_zero_check"] == [27]

summaries, doses = [], []
for offset, family_index in enumerate((1, 2, 3)):
    group = rows[offset*12:(offset+1)*12]
    labs = labels[family_index % 2]
    def find(c, l, role):
        found = [r for r in group if (r["condition"], r["label"], r["year_role"]) == (c, l, role)]
        assert len(found) == 1
        return found[0]
    raw = {(c, l, role): seen[find(c, l, role)["case_id"], "noop"][1]["block_before"]
           for c in conditions for l in labs for role in ("low", "high")}
    vec = {k: floats(v[-rowbytes:]) for k, v in raw.items()}
    norms = {}
    for c in conditions:
        assert len({raw[c,l,role][:rowbytes] for l in labs for role in ("low", "high")}) == 1
        for l in labs:
            norms[f"{c}:year:{l}"] = norm(b-a for a,b in zip(vec[c,l,"low"],vec[c,l,"high"]))
        for role in ("low", "high"):
            norms[f"{c}:label:{role}"] = norm(b-a for a,b in zip(vec[c,labs[0],role],vec[c,labs[1],role]))
    q = min(norms.values())/2
    assert q > 2**-20
    info = {"family_index": family_index, "dose": q, "norms": norms,
            "fractions": {k: q/v for k,v in norms.items()}}
    doses.append(info)
    targets = []
    for case in group:
        c,l,role = case["condition"], case["label"], case["year_role"]
        li, ri = labs.index(l), ("low", "high").index(role)
        source = vec[c,l,role]
        year_donor = vec[c,l,("low", "high")[1-ri]]
        label_donor = vec[c,labs[1-li],role]
        yd = tuple(q/norms[f"{c}:year:{l}"]*(b-a) for a,b in zip(source,year_donor))
        ld = tuple(q/norms[f"{c}:label:{role}"]*(b-a) for a,b in zip(source,label_donor))
        rot = tuple(yd[(i-1)%768] for i in range(768))
        before = raw[c,l,role]
        end = len(case["input_ids"])-1
        choices = {"self": (end, None), "year_final": (end, yd), "label_final": (end, ld),
                   "year_at_zero": (0, yd), "rotated_year_final": (end, rot)}
        for action, (pos, delta) in choices.items():
            record, payload, _ = seen[case["case_id"],action]
            start = pos*rowbytes
            before_site = before[start:start+rowbytes]
            after_site = (before_site if delta is None else
                          struct.pack("<768f", *(a+b for a,b in zip(floats(before_site),delta))))
            assert payload["block_before"] == before and payload["block_after"] == before[:start]+after_site+before[start+rowbytes:]
            detail = record["intervention"]
            assert detail["position"] == pos and detail["expected_before_hex"] == before_site.hex() and detail["expected_after_hex"] == after_site.hex()
            changed = sum(a != b for a,b in zip(struct.iter_unpack("<I",before_site),struct.iter_unpack("<I",after_site)))
            assert detail["changed_bits"] == detail["observed_changed_bits"] == changed
            if delta is None:
                assert detail["certificate"] is detail["observed_certificate"] is None and detail["requested"] == 0.0
            else:
                assert detail["requested"] == q and detail["displacement_f64_hex"] == struct.pack("<768d",*delta).hex()
                assert detail["certificate"] == detail["observed_certificate"]
                src, dst = floats(before_site), floats(after_site)
                d = tuple(Fraction.from_float(x) for x in delta)
                r = tuple(Fraction.from_float(float(z))-Fraction.from_float(float(b)) for b,z in zip(src,dst))
                D2, R2, E2 = (sum((x*x for x in term), Fraction()) for term in (d,r,tuple(x-y for x,y in zip(r,d))))
                cert = detail["certificate"]
                for key, value in (("D2",D2),("R2",R2),("E2",E2)):
                    assert cert[key] == {"numerator":str(value.numerator),"denominator":str(value.denominator)}
                fq = Fraction.from_float(q)
                assert (fq*(1-Fraction(1,10**12)))**2 <= D2 <= (fq*(1+Fraction(1,10**12)))**2
                assert (fq*(1-Fraction(1,5000)))**2 <= R2 <= (fq*(1+Fraction(1,5000)))**2 and E2 <= (fq/5000)**2
                assert detail["achieved"] == detail["observed_achieved"] == math.sqrt(float(R2))
        key = case["case_id"]
        effects = {a: seen[key,a][2]["band_score"]-clean[key]["band_score"] for a in actions}
        year = -effects["year_final"] if role == "low" else effects["year_final"]
        controls = ("label_final", "year_at_zero", "rotated_year_final")
        checks = {a: year > abs(effects[a]) + 0.1 for a in controls}
        targets.append({"case_id":key,"condition":c,"label":l,"year_role":role,"effects":effects,
                        "year_donorward":year,"margins":{a:year-abs(effects[a]) for a in controls},
                        "checks":checks,"pass":all(checks.values())})
    summaries.append({"family_index":family_index,"targets":targets,"pass":all(t["pass"] for t in targets)})
assert doses == result["family_doses"]
assert [{k:v for k,v in e.items() if k != "event"} for e in events if e["event"] == "family_dose"] == doses
targets = [t for family in summaries for t in family["targets"]]
means = {c:{a:math.fsum(t["effects"][a] for t in targets if t["condition"] == c)/12 for a in actions} for c in conditions}
donor = {c:math.fsum(t["year_donorward"] for t in targets if t["condition"] == c)/12 for c in conditions}
count = sum(f["pass"] for f in summaries)
summary = {"families":summaries,"mean_effect_by_condition":means,"mean_year_donorward_by_condition":donor,
           "pattern_count":count,"pattern_supported":count == 3}
assert summary == result["intervention_summary"]
print(json.dumps({"status":"complete_saved_byte_pass", "forwards":288, "raw_bytes":sum(offsets.values()),
                  "cases_sha256":digest(cases_raw), "journal_sha256":digest(journal_raw),
                  "manifest_sha256":digest(manifest_raw), "result_sha256":digest(result_raw),
                  "target_passes":[sum(t["pass"] for t in f["targets"]) for f in summaries],
                  "family_doses":doses, "intervention_summary":summary,
                  "scope":"Saved-byte consistency only; not independent execution, source, process, or external exposure attestation. Adapted from this author's earlier v1 replay; separately written from the other reviewer's replay."},
                 sort_keys=True, allow_nan=False))
