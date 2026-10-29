"""Reconstruct the factorial's additive-logit reference from saved bytes.

This pure calculation neither executes a model nor authenticates the saved
records. The caller must bind case identities, source, receipt and raw hashes.
"""
import math
import hashlib
import json
import re
import struct
from pathlib import Path


def _decode_float32(raw):
    if type(raw) is not bytes or not raw or len(raw) % 4:
        raise ValueError("expected nonempty little-endian float32 bytes")
    values = []
    for (bits,) in struct.iter_unpack("<I", raw):
        sign, exponent, fraction = bits >> 31, (bits >> 23) & 255, bits & 0x7FFFFF
        if exponent == 255:
            raise ValueError("saved logits must be finite")
        if exponent == 0:
            value = math.ldexp(float(fraction), -149)
        else:
            value = math.ldexp(float((1 << 23) | fraction), exponent - 150)
        values.append(-value if sign else value)
    return values


def _log_probability(values, target):
    if type(target) is not int or not 0 <= target < len(values):
        raise ValueError("target is outside the vocabulary")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("logits must be finite")
    maximum = max(values)
    answer = (values[target] - maximum) - math.log(
        math.fsum(math.exp(value - maximum) for value in values)
    )
    if not math.isfinite(answer):
        raise ValueError("target log probability is nonfinite")
    return answer


def analyze_family(cases):
    """Return per-prompt and paired reference outcomes for four fixed cases.

    Each case supplies ``target_id`` and little-endian float32 byte strings
    named ``clean``, ``h2``, ``h10`` and ``joint``. The first is repeated and
    the remaining three are the predeclared shuffles. Vocabulary size is
    intentionally generic here; the evidence reader must require 50,257.
    """
    if type(cases) not in (list, tuple) or len(cases) != 4:
        raise ValueError("a family must have one repeated and three shuffle cases")
    rows = []
    for case in cases:
        if type(case) is not dict or set(case) != {"target_id", "clean", "h2", "h10", "joint"}:
            raise ValueError("malformed saved case")
        decoded = {name: _decode_float32(case[name]) for name in ("clean", "h2", "h10", "joint")}
        if len({len(values) for values in decoded.values()}) != 1:
            raise ValueError("saved vocabularies differ in length")
        target = case["target_id"]
        scores = {name: _log_probability(values, target) for name, values in decoded.items()}
        additive = [(h2 - clean) + h10 for clean, h2, h10 in zip(
            decoded["clean"], decoded["h2"], decoded["h10"])]
        scores["additive"] = _log_probability(additive, target)
        deltas = {name: scores[name] - scores["clean"] for name in ("h2", "h10", "joint", "additive")}
        rows.append({"target_id": target, "log_probabilities": scores, "deltas": deltas,
                     "prompt_interaction": ((scores["joint"] - scores["h2"]) - scores["h10"]) + scores["clean"],
                     "additive_prompt_interaction": ((scores["additive"] - scores["h2"]) - scores["h10"]) + scores["clean"]})
    contrasts = {name: rows[0]["deltas"][name] - math.fsum(row["deltas"][name] for row in rows[1:]) / 3
                 for name in ("h2", "h10", "joint", "additive")}
    return {"cases": rows, "contrasts": contrasts,
            "interaction": (contrasts["joint"] - contrasts["h2"]) - contrasts["h10"],
            "additive_interaction": (contrasts["additive"] - contrasts["h2"]) - contrasts["h10"],
            "joint_minus_additive": contrasts["joint"] - contrasts["additive"]}


def analyze_study(families):
    """Analyze the fixed eight families in their saved order."""
    if type(families) not in (list, tuple) or len(families) != 8:
        raise ValueError("expected exactly eight ordered families")
    result = [analyze_family(cases) for cases in families]
    return {"families": result,
            "mean_interaction": math.fsum(row["interaction"] for row in result) / 8,
            "mean_additive_interaction": math.fsum(row["additive_interaction"] for row in result) / 8,
            "mean_joint_minus_additive": math.fsum(row["joint_minus_additive"] for row in result) / 8}


def extract_saved_families(cases, records, raw_files, *, vocabulary_size):
    """Bind ordered case/action records to hashed saved float32 logit bytes.

    This checks the factorial schedule and each recorded logit slice, not
    source authenticity, the receipt or intervention landing witnesses. The
    production caller must require vocabulary_size=50,257 and separately
    validate those other evidence boundaries.
    """
    if (type(cases) is not list or len(cases) != 32 or type(records) is not list or len(records) != 224
            or type(raw_files) is not dict or type(vocabulary_size) is not int or vocabulary_size <= 0):
        raise ValueError("unexpected saved case, record or vocabulary count")
    expected = []
    for group in range(8):
        family = cases[group * 4:group * 4 + 4]
        for variant, case in enumerate(family):
            case_id = f"copy-factorial-{group + 1:02d}-{variant}"
            if (type(case) is not dict or case.get("case_id") != case_id
                    or type(case.get("group")) is not int or case["group"] != group + 1
                    or type(case.get("variant")) is not int or case["variant"] != variant
                    or type(case.get("target_id")) is not int or not 0 <= case["target_id"] < vocabulary_size):
                raise ValueError("saved case identity or target differs")
        expected.extend((family[variant], action) for variant in range(4) for action in ("clean", "noop"))
        for variant, case in enumerate(family):
            arms = ("h10", "h2", "h7", "joint")
            rotation = (group + variant) % 4
            expected.extend((case, action) for action in (*arms[rotation:], *arms[:rotation], "clean_after"))
    ids = {case["case_id"] for case in cases}
    if set(raw_files) != ids or any(type(value) is not bytes for value in raw_files.values()):
        raise ValueError("saved raw files do not match the fixed case identities")
    previous_end = {case_id: 0 for case_id in ids}
    selected = {case_id: {} for case_id in ids}
    for record, (case, action) in zip(records, expected):
        if (type(record) is not dict or any(record.get(key) != case[key]
                for key in ("case_id", "group", "variant", "target_id")) or record.get("action") != action):
            raise ValueError("saved action order or identity differs")
        if any(type(record.get(key)) is not int for key in ("group", "variant", "target_id")):
            raise ValueError("saved action identity has invalid types")
        raw = record.get("raw")
        item = raw.get("logits") if type(raw) is dict else None
        if (type(item) is not dict or set(item) != {"offset", "bytes", "sha256"}
                or type(item["offset"]) is not int or type(item["bytes"]) is not int
                or item["bytes"] != 4 * vocabulary_size or item["offset"] < previous_end[case["case_id"]]
                or type(item["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
            raise ValueError("invalid or overlapping saved logit slice")
        end = item["offset"] + item["bytes"]
        data = raw_files[case["case_id"]][item["offset"]:end]
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("saved logit slice digest differs")
        previous_end[case["case_id"]] = end
        if action in ("clean", "noop", "clean_after", "h2", "h10", "joint"):
            selected[case["case_id"]][action] = data
    for saved in selected.values():
        if saved["clean"] != saved["noop"] or saved["clean"] != saved["clean_after"]:
            raise ValueError("clean, no-op and clean-after logits differ")
    return [[{"target_id": case["target_id"], **{name: selected[case["case_id"]][name]
             for name in ("clean", "h2", "h10", "joint")}} for case in cases[g * 4:g * 4 + 4]]
            for g in range(8)]


def _json_object(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    if type(raw) is not bytes:
        raise ValueError("expected saved JSON bytes")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid saved JSON") from error
    if type(value) is not dict:
        raise ValueError("expected a JSON object")
    return value


def analyze_saved(case_raw, manifest_raw, result_raw, pre_run_raw, receipt_raw, journal_raw, raw_files,
                  *, expected, vocabulary_size):
    """Bind a completed direct-process receipt and rederive reference scores.

    This is a saved-data consistency calculation; it does not authenticate the
    operator's receipt, model execution, source provenance or landing controls.
    """
    case_file, manifest, result = map(_json_object, (case_raw, manifest_raw, result_raw))
    pre, receipt = map(_json_object, (pre_run_raw, receipt_raw))
    if type(expected) is not dict or set(expected) != {"source_revision", "case_sha256", "protocol_sha256",
            "clarification_sha256", "clarification_revision", "v11", "v12"}:
        raise ValueError("expected source and evidence bindings are incomplete")
    if (type(expected["source_revision"]) is not str
            or re.fullmatch(r"[0-9a-f]{40}", expected["source_revision"]) is None
            or type(expected["clarification_revision"]) is not str
            or re.fullmatch(r"[0-9a-f]{40}", expected["clarification_revision"]) is None
            or any(type(expected[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", expected[key]) is None
                   for key in ("case_sha256", "protocol_sha256", "clarification_sha256", "v11", "v12"))):
        raise ValueError("expected source or digest has an invalid form")
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    if (receipt.get("schema") != "copying-head-factorial-process-observation-v1"
            or type(receipt.get("process_exit_code")) is not int or receipt["process_exit_code"] != 0
            or receipt.get("interrupted") is not False or receipt.get("failure_file_present") is not False
            or receipt.get("source_revision") != expected["source_revision"]
            or receipt.get("pre_run_sha256") != sha(pre_run_raw)):
        raise ValueError("direct-process receipt is incomplete or inconsistent")
    if (pre.get("schema") != "copying-head-factorial-launch-v1"
            or pre.get("source_revision") != expected["source_revision"]
            or pre.get("case_sha256") != expected["case_sha256"]
            or pre.get("clarification_sha256") != expected["clarification_sha256"]
            or sha(case_raw) != expected["case_sha256"]
            or case_file.get("schema") != "distilgpt2-copy-head-factorial-cases-v1"
            or manifest.get("schema") != "distilgpt2-copy-head-factorial-run-v1"
            or manifest.get("cases") != case_file.get("cases")
            or manifest.get("cases_sha256") != expected["case_sha256"]
            or manifest.get("protocol_sha256") != expected["protocol_sha256"]
            or manifest.get("source", {}).get("revision") != expected["source_revision"]
            or manifest.get("logit_reference_clarification") != {
                "revision": expected["clarification_revision"], "sha256": expected["clarification_sha256"]}
            or manifest.get("exposure_inventories") != {"v11": expected["v11"], "reservation": expected["v12"],
                "reservation_version": "distilgpt2-development-exposures-v12"}):
        raise ValueError("case, source, exposure or clarification binding differs")
    if (result.get("schema") != "distilgpt2-copy-head-factorial-result-v1" or result.get("status") != "complete"
            or result.get("manifest_sha256") != sha(manifest_raw)
            or type(result.get("submitted_forwards")) is not int or result["submitted_forwards"] != 224
            or type(result.get("returned_forwards")) is not int or result["returned_forwards"] != 224
            or type(result.get("raw_payload_bytes")) is not int
            or result["raw_payload_bytes"] != sum(len(data) for data in raw_files.values())):
        raise ValueError("saved result is incomplete or inconsistent")
    if type(journal_raw) is not bytes:
        raise ValueError("expected saved journal bytes")
    actual_inventory = {"manifest.json": sha(manifest_raw), "result.json": sha(result_raw),
                        "attempt.jsonl": sha(journal_raw)}
    actual_inventory.update({case_id + ".f32": sha(data) for case_id, data in raw_files.items()})
    if receipt.get("run_sha256") != actual_inventory:
        raise ValueError("receipt file inventory or digest differs")
    families = extract_saved_families(case_file["cases"], result.get("cases"), raw_files,
                                      vocabulary_size=vocabulary_size)
    computed = analyze_study(families)
    reported = result.get("summary", {}).get("families")
    if type(reported) is not list or len(reported) != 8:
        raise ValueError("saved family summary is missing")
    records = {(row["case_id"], row["action"]): row for row in result["cases"]}
    for group, family in enumerate(computed["families"]):
        saved = reported[group]
        if (type(saved) is not dict or type(saved.get("group")) is not int or saved["group"] != group + 1
                or type(saved.get("interaction")) not in (int, float)
                or not math.isfinite(saved["interaction"])
                or not math.isclose(saved["interaction"], family["interaction"], rel_tol=0, abs_tol=1e-12)):
            raise ValueError("saved observed interaction disagrees with logits")
        arms = saved.get("arms")
        if type(arms) is not dict:
            raise ValueError("saved arm summaries are missing")
        for action in ("h2", "h10", "joint"):
            arm = arms.get(action)
            if (type(arm) is not dict or type(arm.get("family_contrast")) not in (int, float)
                    or not math.isfinite(arm["family_contrast"])
                    or not math.isclose(arm["family_contrast"], family["contrasts"][action], rel_tol=0, abs_tol=1e-12)
                    or type(arm.get("case_deltas")) is not list or len(arm["case_deltas"]) != 4):
                raise ValueError("saved arm summary disagrees with logits")
            for variant, delta in enumerate(arm["case_deltas"]):
                if (type(delta) not in (int, float) or not math.isfinite(delta)
                        or not math.isclose(delta, family["cases"][variant]["deltas"][action],
                                            rel_tol=0, abs_tol=1e-12)):
                    raise ValueError("saved arm delta disagrees with logits")
        interactions = saved.get("case_interactions")
        if type(interactions) is not list or len(interactions) != 4:
            raise ValueError("saved prompt interactions are missing")
        for variant, interaction in enumerate(interactions):
            if (type(interaction) not in (int, float) or not math.isfinite(interaction)
                    or not math.isclose(interaction, family["cases"][variant]["prompt_interaction"],
                                        rel_tol=0, abs_tol=1e-12)):
                raise ValueError("saved prompt interaction disagrees with logits")
        for variant, row in enumerate(family["cases"]):
            case_id = f"copy-factorial-{group + 1:02d}-{variant}"
            for action in ("clean", "h2", "h10", "joint"):
                score = records[(case_id, action)].get("score", {}).get("target_log_probability")
                if (type(score) not in (int, float) or not math.isfinite(score)
                        or not math.isclose(score, row["log_probabilities"][action], rel_tol=0, abs_tol=1e-12)):
                    raise ValueError("saved target score disagrees with logits")
    return {"schema": "distilgpt2-copy-factorial-additive-reference-v1", "status": "pass",
            "scope": "saved-logit and receipt-hash consistency; journal events, model execution and landing are not verified",
            "case_sha256": sha(case_raw), "manifest_sha256": sha(manifest_raw), "result_sha256": sha(result_raw),
            "receipt_sha256": sha(receipt_raw), "source_revision": expected["source_revision"], **computed}


def analyze_directory(attempt, case_file, *, expected, vocabulary_size, expected_total_bytes):
    """Read a quiescent saved attempt with bounded files, then analyze it."""
    attempt, case_file = Path(attempt), Path(case_file)
    run = attempt / "run"
    names = {f"copy-factorial-{group:02d}-{variant}" for group in range(1, 9) for variant in range(4)}
    expected_names = {"manifest.json", "result.json", "attempt.jsonl"} | {name + ".f32" for name in names}
    if not run.is_dir() or {path.name for path in run.iterdir()} != expected_names:
        raise ValueError("saved run file inventory is incomplete or has extra files")

    def read(path, maximum):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
            raise ValueError("saved evidence file is missing or exceeds its bound")
        return path.read_bytes()

    files = {name: read(run / (name + ".f32"), 2 * 1024 * 1024) for name in names}
    if type(expected_total_bytes) is not int or expected_total_bytes <= 0 or sum(map(len, files.values())) != expected_total_bytes:
        raise ValueError("saved raw payload total differs")
    return analyze_saved(read(case_file, 1024 * 1024), read(run / "manifest.json", 2 * 1024 * 1024),
                         read(run / "result.json", 4 * 1024 * 1024), read(attempt / "pre-run.json", 65536),
                         read(attempt / "supervisor.json", 65536), read(run / "attempt.jsonl", 4 * 1024 * 1024),
                         files, expected=expected, vocabulary_size=vocabulary_size)
