"""Audit a closed full-top baseline screen from public cases and saved bytes.

Never loads a model. Process/source/custody fields are checked as reported
records, not authenticated proof of execution, timing or unknown exposure.
"""
import argparse
import hashlib
import json
import math
import pathlib
from datetime import datetime

import numpy as np


VOCAB = 50257
ROW_BYTES = VOCAB * 4
CASE_SHA = "4057696ddef9e3b0d09db02f46e6454561a61ac1a07f8b8a13b92718944f4261"
MANIFEST_SHA = "458c934bbe74ad01fd0d6abe6d4686aa1e47cbb993f4b76e04d451d07b3e56bb"
SOURCE = "9d10c8e221a39589173fba9ad1726022e9e19950"
CHECKOUT = "4f963a53ab5a8aed26469b6e4189a28a98f53a7c"
PROBE = {"subnormal_times_one_bits": 1, "normal_times_half_bits": 0x00400000,
         "numpy_cast_preserves_min_subnormal": True}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate JSON key " + key)
        out[key] = value
    return out


def read_json(raw):
    return json.loads(raw, object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def unpack(raw):
    bits = np.frombuffer(raw, dtype="<u4")
    if len(bits) != VOCAB:
        raise ValueError("vocabulary length")
    exponent = (bits >> 23) & 255
    if np.any(exponent == 255):
        raise ValueError("nonfinite saved logit")
    fraction = bits & 0x7fffff
    significand = np.where(exponent == 0, fraction, fraction | 0x800000).astype(np.float64)
    power = np.where(exponent == 0, -149, exponent.astype(np.int32) - 150)
    return np.where((bits >> 31) != 0, -np.ldexp(significand, power), np.ldexp(significand, power))


def metrics(raw, factual, alternative):
    values = unpack(raw)
    maximum = float(np.max(values))
    tied = np.flatnonzero(values == maximum)
    representative = int(tied[0])
    unique = representative if len(tied) == 1 else None
    second = float(np.partition(values, -2)[-2])
    shifted = values - maximum
    with np.errstate(under="ignore"):
        denominator = math.fsum(float(x) for x in np.exp(shifted))
    if not math.isfinite(denominator) or denominator < 1:
        raise ValueError("invalid finite logit normalization")
    logdenom = math.log(denominator)
    return {"unique_top_id": unique, "top_tie_count": len(tied),
            "top_representative_id": representative, "top_gap": maximum - second,
            "factual_logit": float(values[factual]), "alternative_logit": float(values[alternative]),
            "factual_logp": float(shifted[factual]) - logdenom,
            "alternative_logp": float(shifted[alternative]) - logdenom}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--cases", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--phase-sha", required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    attempt = args.attempt
    manifest_raw, case_raw = args.manifest.read_bytes(), args.cases.read_bytes()
    if sha(manifest_raw) != MANIFEST_SHA or sha(case_raw) != CASE_SHA:
        raise ValueError("pinned manifest or case mismatch")
    manifest, cases = read_json(manifest_raw), read_json(case_raw)
    phase_raw = (attempt / "phase-record.json").read_bytes()
    if sha(phase_raw) != args.phase_sha:
        raise ValueError("phase digest mismatch")
    phase = read_json(phase_raw)
    if (phase["version"] != "targeted-edit-full-top-screen-phase-v1" or phase["phase"] != "screen"
            or phase["status"] != "complete" or phase["source_commit"] != SOURCE
            or phase["checkout_commit"] != CHECKOUT or phase["manifest_sha256"] != MANIFEST_SHA
            or phase["owner_reaped"] is not True or phase["job_drained"] is not True):
        raise ValueError("phase identity or teardown record mismatch")
    sup, child = phase["supervisor"], phase["child"]
    if (sup["clean"] is not True or type(sup["exit_code"]) is not int or sup["exit_code"] != 0
            or sup["timed_out"] is not False or type(sup["pid"]) is not int
            or sup["pid"] != child["owner_pid"] or sup["run_id"] != child["owner_run_id"]
            or child["phase"] != "screen" or child["manifest_sha256"] != MANIFEST_SHA
            or child["model_disposed"] is not True or child["no_model_loaded"] is not False):
        raise ValueError("child or process reports differ")
    for name in ("model_state_sha256", "weight_sha256", "buffer_state_sha256"):
        expected = {"model_state_sha256": "original_model_state_sha256", "weight_sha256": "original_weight_sha256",
                    "buffer_state_sha256": "baseline_buffer_state_sha256"}[name]
        if child[name] != manifest[expected]:
            raise ValueError("child model-state binding " + name)
    if (child["runtime_settings"] != manifest["baseline_runtime_settings"]
            or child["post_work_checkpoint_assets"] != manifest["checkpoint"]["asset_sha256"]
            or any(manifest["source_sha256"].get(k) != v for k, v in child["imported_sources"].items())):
        raise ValueError("child runtime, assets or source report differs")
    expected = {"started.json", "child-result.json", "manifest-review-attestation.json", "public-exposure.json",
                "public-exposure-attestation.json", "screen/screen.f32", "screen/observations.jsonl",
                "screen/state-original.json", "screen/screen-selection.json", "screen/screen-top-tokens.json"}
    if set(phase["artifacts"]) != expected:
        raise ValueError("phase artifact inventory differs")
    files = {p.relative_to(attempt).as_posix() for p in attempt.rglob("*") if p.is_file()}
    if files != expected | {"phase-record.json"} or not (attempt / "pycache").is_dir() or any((attempt / "pycache").iterdir()):
        raise ValueError("unexpected files or nonempty bytecode")
    for name, item in phase["artifacts"].items():
        data = (attempt / name).read_bytes()
        if item != {"bytes": len(data), "sha256": sha(data)}:
            raise ValueError("artifact hash mismatch " + name)
    if (read_json((attempt / "child-result.json").read_bytes()) != child
            or phase["child_result_sha256"] != sha((attempt / "child-result.json").read_bytes())
            or phase["started_sha256"] != sha((attempt / "started.json").read_bytes())):
        raise ValueError("child or start digest differs")
    started = read_json((attempt / "started.json").read_bytes())
    if (started["phase"] != "screen" or started["source_commit"] != SOURCE or started["checkout_commit"] != CHECKOUT
            or started["manifest_sha256"] != MANIFEST_SHA):
        raise ValueError("start identity differs")
    for key, name in (("manifest_review_sha256", "manifest-review-attestation.json"),
                      ("public_exposure_file_sha256", "public-exposure.json"),
                      ("public_exposure_attestation_sha256", "public-exposure-attestation.json")):
        if started[key] != sha((attempt / name).read_bytes()):
            raise ValueError("start artifact binding " + name)
    review = read_json((attempt / "manifest-review-attestation.json").read_bytes())
    exposure_review = read_json((attempt / "public-exposure-attestation.json").read_bytes())
    exposure_raw = (attempt / "public-exposure.json").read_bytes()
    if (review["reviewer"] != "/root/review_trace_budget" or review["reviewed_commit"] != CHECKOUT
            or review["manifest_sha256"] != MANIFEST_SHA
            or "no-later-than" not in review["timing_scope"]
            or exposure_review["reviewer"] != "/root/review_trace_budget"
            or exposure_review["reviewed_commit"] != "27e1455796bb176cc2cf848db7dd481092300954"
            or exposure_review["inventory_sha256"] != sha(exposure_raw)
            or exposure_review["inventory_sha256"] != manifest["exposure"]["canonical_sha256"]
            or exposure_review["source_path"] != "looking_glass/pretrained_exposures_v26.py"
            or exposure_review["source_sha256"] != manifest["source_sha256"][exposure_review["source_path"]]
            or exposure_review["operator_observed_main"] != "8beca2d61936e0a443db7e65c45831052ae742f8"
            or "no-later-than" not in exposure_review["timing_scope"]):
        raise ValueError("reported independent source or inventory review differs")
    exposure = read_json(exposure_raw)
    public = started["current_public_exposure"]
    if (exposure["version"] != "distilgpt2-development-exposures-v26" or len(exposure["records"]) != 643
            or public["version"] != exposure["version"] or public["canonical_sha256"] != sha(exposure_raw)
            or (public["record_count"], public["own_reserved_rows"], public["other_row_comparisons"], public["known_relations"]) != (643, 48, 28560, 0)
            or manifest["exposure"]["known_complete_rows"] != [row["token_ids"] for row in exposure["records"]]):
        raise ValueError("current public inventory report differs")
    custodies = {row["custodian"]: row for row in started["custodian_observations"]["observations"]}
    if set(custodies) != {"natural", "injection"}:
        raise ValueError("custody report inventory differs")
    natural, injection = custodies["natural"], custodies["injection"]
    if (natural["reported_by"] != "/root/review_localization" or natural["status"] != "public_frame_clear"
            or natural["case_sha256"] != CASE_SHA or natural["commitment_sha256"] != manifest["natural_commitment_sha256"]
            or natural["frame_sha256"] != manifest["public_support_sha256"] or natural["report_sha256"] != manifest["public_frame_sha256"]
            or natural["original_split_unchanged"] is not True or natural["no_additional_relevant_splits_reported"] is not True
            or natural["successor_status"] != "no_successor_split_reported" or injection["reported_by"] != "/root/fixtures"
            or injection["status"] != "no_protected_rows_reported"
            or any("no private query" not in c["timing_scope"] or "no-later-than" not in c["timing_scope"] for c in (natural, injection))):
        raise ValueError("scoped custody report differs")
    started_at = datetime.fromisoformat(started["started_utc"])
    if any(datetime.fromisoformat(record["observed_utc"]) > started_at for record in (review, exposure_review, natural, injection)):
        raise ValueError("reported review or custody time follows reported start")
    selection = read_json((attempt / "screen/screen-selection.json").read_bytes())
    state = read_json((attempt / "screen/state-original.json").read_bytes())
    if (selection["case_sha256"] != CASE_SHA or selection["status"] != "private-screen"
            or selection["identity"] != {"source_commit": SOURCE, "manifest_sha256": MANIFEST_SHA}
            or selection["score_denormal_probe_before"] != PROBE or selection["score_denormal_probe_after"] != PROBE
            or child["selection_sha256"] != sha((attempt / "screen/screen-selection.json").read_bytes())):
        raise ValueError("selection identity or numeric mode differs")
    if set(selection["artifacts"]) != {"screen.f32", "observations.jsonl", "state-original.json"}:
        raise ValueError("selection artifact inventory differs")
    if any(phase["artifacts"]["screen/" + name] != item for name, item in selection["artifacts"].items()):
        raise ValueError("selection artifact binding differs")
    if (state["cleanup"] != "verified" or state["model_state_sha256"] != child["model_state_sha256"]
            or state["weight_sha256"] != child["weight_sha256"] or state["applied_weight_sha256"] != child["weight_sha256"]
            or state["buffer_state_sha256"] != child["buffer_state_sha256"] or state["runtime_settings"] != child["runtime_settings"]
            or state["addition_denormal_probe_before"] != PROBE or state["addition_denormal_probe_after"] != PROBE):
        raise ValueError("saved restoration record differs")
    matrix = (attempt / "screen/screen.f32").read_bytes()
    journals = (attempt / "screen/observations.jsonl").read_bytes().splitlines()
    rows, observations = cases["cases"][:36], selection["selection"]["observations"]
    tops = read_json((attempt / "screen/screen-top-tokens.json").read_bytes())
    if len(matrix) != 36 * ROW_BYTES or not (len(rows) == len(journals) == len(observations) == len(tops) == 36):
        raise ValueError("screen row inventory differs")
    tokenizer_raw = args.tokenizer.read_bytes()
    if sha(tokenizer_raw) != cases["tokenizer_sha256"]:
        raise ValueError("tokenizer hash differs")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    results = []
    for i, (case, line, saved, saved_top) in enumerate(zip(rows, journals, observations, tops)):
        record = read_json(line)
        offset = i * ROW_BYTES
        raw = matrix[offset:offset + ROW_BYTES]
        digest = sha(raw)
        if (record["index"] != i or record["case_id"] != case["case_id"] or record["offset"] != offset
                or record["bytes"] != ROW_BYTES or record["sha256"] != digest
                or record["forward_denormal_probe_before"] != PROBE or record["forward_denormal_probe_after"] != PROBE
                or saved["offset"] != i or saved["case_id"] != case["case_id"] or saved["role"] != case["role"]
                or saved["fact_id"] != case["fact_id"] or saved["fact_index"] != i // 9 or saved["logit_sha256"] != digest
                or saved["logit_byte_offset"] != offset or saved["logit_bytes"] != ROW_BYTES):
            raise ValueError("row journal mismatch " + str(i))
        numerical = metrics(raw, case["factual"]["token_id"], case["counterfactual"]["token_id"])
        for key in ("unique_top_id", "top_tie_count", "top_representative_id", "top_gap", "factual_logit", "alternative_logit"):
            if saved[key] != numerical[key]:
                raise ValueError("row score mismatch " + str(i) + " " + key)
        for key in ("factual_logp", "alternative_logp"):
            if not math.isclose(saved[key], numerical[key], rel_tol=1e-13, abs_tol=1e-12):
                raise ValueError("row log probability mismatch " + str(i) + " " + key)
        unique, rep = numerical["unique_top_id"], numerical["top_representative_id"]
        expected_top = {"case_id": case["case_id"], "unique_top_id": unique,
                        "unique_top_text": None if unique is None else tokenizer.decode([unique], skip_special_tokens=False),
                        "unique_top_raw_token": None if unique is None else tokenizer.id_to_token(unique),
                        "top_tie_count": numerical["top_tie_count"], "top_representative_id": rep,
                        "top_representative_text": tokenizer.decode([rep], skip_special_tokens=False),
                        "top_representative_raw_token": tokenizer.id_to_token(rep)}
        if saved_top != expected_top:
            raise ValueError("decoded top mismatch " + str(i))
        eligible = unique == case["factual"]["token_id"]
        if saved["eligible"] is not eligible:
            raise ValueError("row eligibility mismatch " + str(i))
        results.append({"case_id": case["case_id"], "fact_id": case["fact_id"], "role": case["role"],
                        "eligible": eligible, "row_sha256": digest, **numerical,
                        "decoded_top": expected_top["unique_top_text"]})
    chosen = None
    fact_summaries = []
    for i, fact in enumerate(cases["fact_order"]):
        block = results[i * 9:(i + 1) * 9]
        if [x["fact_id"] for x in block] != [fact] * 9:
            raise ValueError("fact order differs")
        eligible = [i * 9 + j for j in range(3, 9) if block[j]["eligible"]]
        qualifies = block[0]["eligible"] and all(x["eligible"] for x in block[1:3]) and len(eligible) >= 4
        fact_summaries.append({"fact_id": fact, "discovery_eligible": block[0]["eligible"],
                               "calibration_eligible_count": sum(x["eligible"] for x in block[1:3]),
                               "evaluation_eligible_count": len(eligible), "qualifies": qualifies})
        if chosen is None and qualifies:
            chosen = (i, eligible[:4])
    decision = selection["selection"]
    if len(decision["candidates"]) != 4:
        raise ValueError("candidate summary inventory differs")
    for i, expected in enumerate(fact_summaries):
        candidate = decision["candidates"][i]
        offsets = [9 * i + j for j in range(3, 9) if results[9 * i + j]["eligible"]]
        if (candidate["fact_index"] != i or candidate["discovery_eligible"] != expected["discovery_eligible"]
                or candidate["calibration_eligible_count"] != expected["calibration_eligible_count"]
                or candidate["evaluation_eligible_count"] != expected["evaluation_eligible_count"]
                or candidate["qualifies"] != expected["qualifies"] or candidate["eligible_evaluation_offsets"] != offsets):
            raise ValueError("candidate summary differs for fact " + str(i))
    expected_status = "eligible" if chosen is not None else "ineligible"
    if (decision["status"] != expected_status or decision["fact_index"] != (None if chosen is None else chosen[0])
            or decision["evaluation_offsets"] != ([] if chosen is None else chosen[1])
            or decision["discovery_offset"] != (None if chosen is None else 9 * chosen[0])
            or decision["calibration_offsets"] != ([] if chosen is None else [9 * chosen[0] + 1, 9 * chosen[0] + 2])
            or child["selection_status"] != expected_status):
        raise ValueError("baseline-only selection differs")
    report = {"schema": "full-top-screen-independent-record-v1", "phase_sha256": sha(phase_raw),
              "manifest_sha256": MANIFEST_SHA, "cases_sha256": CASE_SHA, "matrix_sha256": sha(matrix),
              "auditor_sha256": sha(pathlib.Path(__file__).read_bytes()), "public_exposure_sha256": sha(exposure_raw),
              "selection_status": expected_status, "selected_fact": None if chosen is None else cases["fact_order"][chosen[0]],
              "selected_evaluation_offsets": [] if chosen is None else chosen[1], "facts": fact_summaries, "rows": results,
              "scope": "Baseline-only public phrasings and saved-byte replay; process and custody are reported, not authenticated. No edited evaluation or private split comparison."}
    with args.report.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({"selection_status": expected_status, "selected_fact": report["selected_fact"],
                      "selected_evaluation_offsets": report["selected_evaluation_offsets"],
                      "report_sha256": sha(args.report.read_bytes())}, sort_keys=True))


if __name__ == "__main__":
    main()
