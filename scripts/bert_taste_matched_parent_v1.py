"""One-shot, CPU-only parent observation for the public matched-dose taste exploration."""

import sys

if __name__ in ("__main__", "<run_path>") and (
        not sys.flags.isolated or not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError("run the parent isolated with no bytecode writes and a fresh prefix")

import argparse
import ast
import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "scripts/bert_taste_matched_dose_v1.py"
EXPECTED_SOURCE_PATHS = (
    "scripts/bert_taste_matched_dose_v1.py", "scripts/bert_taste_matched_parent_v1.py",
    "scripts/bert_taste_matched_independent.py",
    "docs/BERT_TASTE_MATCHED_DOSE_V1.md", "experiments/bert_taste_matched_dose_public_cases_v1.json",
    "experiments/bert_agreement_exposures_v6.json", "experiments/bert_matched_public_cases_v1.json",
    "experiments/bert_taste_public_exposures_v2.json", "scripts/bert_taste_patch_v1.py",
    "looking_glass/__init__.py", "looking_glass/pretrained.py", "looking_glass/bert_agreement_study.py",
    "looking_glass/bert_agreement_runner.py", "looking_glass/intervention_plan.py",
    "looking_glass/interventions.py", "looking_glass/model_ownership.py",
)
EXPECTED_ASSETS = {
    "config.json": {"sha256": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83", "size": 483},
    "model.safetensors": {"sha256": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063", "size": 267954768},
    "tokenizer.json": {"sha256": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98", "size": 466062},
    "tokenizer_config.json": {"sha256": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10", "size": 48},
    "vocab.txt": {"sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3", "size": 231508},
}


def _git(*args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1"})
    return subprocess.run(["git", "-c", "core.fsmonitor=false", *args], cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def source_snapshot():
    """Check the child's declared source bytes before child imports begin."""
    tree = ast.parse(CHILD.read_bytes())
    declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SOURCE_PATHS"
                            for target in node.targets)]
    if len(declarations) != 1:
        raise ValueError("child source list is ambiguous")
    names = ast.literal_eval(declarations[0])
    if (type(names) is not tuple or names != EXPECTED_SOURCE_PATHS or len(names) != len(set(names))
            or any(type(name) is not str or not name or ".." in Path(name).parts
                   or Path(name).is_absolute() for name in names)):
        raise ValueError("child source list is invalid")
    commit = _git("rev-parse", "HEAD").decode("ascii").strip()
    hashes = {}
    for name in names:
        data = (ROOT / name).read_bytes()
        if data != _git("show", f"{commit}:{name}"):
            raise RuntimeError("selected source differs from its Git blob")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {"commit": commit, "selected_source_sha256": hashes}


def _bounded(path, maximum):
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("saved evidence exceeds its bound")
    return data


def _schedule(frame):
    """Expected public journal identities, including the six dose plans."""
    result = []

    def forward(row, layer, position, mode, donor):
        identity = {"row_id": f"{frame}-{row}", "layer": layer, "position": position,
                    "mode": mode, "donor": donor}
        result.extend({"event": event, **identity} for event in ("forward_start", "forward_complete"))

    for layer in range(6):
        for row in ("sugar", "lemon"):
            forward(row, layer, 7, "capture_ingredient", None)
            forward(row, layer, 1, "capture_control", None)
        result.append({"event": "dose_plan", "row_id": f"{frame}-sugar", "frame": frame, "layer": layer})
        for row, other in (("sugar", "lemon"), ("lemon", "sugar")):
            for mode, position, is_self in (("ingredient_donor", 7, False), ("ingredient_self", 7, True),
                                            ("control_self", 1, True), ("ingredient_matched", 7, False),
                                            ("control_matched", 1, False)):
                forward(row, layer, position, mode, f"{frame}-{row if is_self else other}")
    for row in ("sugar", "lemon"):
        forward(row, 0, 7, "final_clean", None)
    return result


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate journal key")
        value[key] = item
    return value


def _reject_constant(_value):
    raise ValueError("nonfinite JSON constant")


def _finite_json(value):
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_finite_json(item) for item in value)
    if type(value) is dict:
        return all(_finite_json(item) for item in value.values())
    return value is None or type(value) in (str, int, bool)


def _same_typed(left, right):
    if type(left) is not type(right):
        return False
    if type(right) is dict:
        return left.keys() == right.keys() and all(_same_typed(left[key], value) for key, value in right.items())
    if type(right) is list:
        return len(left) == len(right) and all(_same_typed(a, b) for a, b in zip(left, right))
    return left == right


def _sha256(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _payload(value, size):
    if type(value) is not str:
        raise ValueError("missing numeric payload")
    raw = base64.b64decode(value, validate=True)
    if len(raw) != size or any(((bits >> 23) & 255) == 255 for (bits,) in struct.iter_unpack("<I", raw)):
        raise ValueError("numeric payload shape or finite values differ")
    return raw


def _decode_f32(bits):
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    if exponent == 255:
        raise ValueError("nonfinite float32")
    magnitude = math.ldexp(fraction, -149) if exponent == 0 else math.ldexp((1 << 23) | fraction, exponent - 150)
    return -magnitude if bits >> 31 else magnitude


def _values(raw):
    return [_decode_f32(bits) for (bits,) in struct.iter_unpack("<I", raw)]


def _distance(left, right):
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(_values(left), _values(right))))


def _metadata_matches(event, layer, position, mode):
    patched = mode in ("ingredient_donor", "ingredient_self", "control_self", "ingredient_matched", "control_matched")
    names = ["full_before", "before", "patch", "after", "full_after"] if patched else ["full_before", "before"]
    steps = event.get("events")
    if type(steps) is not list or len(steps) != len(names):
        return False
    path = f"distilbert.transformer.layer.{layer}"
    for index, name in enumerate(names):
        selected = name in ("before", "patch", "after")
        expected = {"name": name, "operation": "patch" if name == "patch" else "capture",
                    "order": index, "module_path": path, "module_aliases": [path],
                    "call_site_known": False, "forward_index": 0, "invocation": 0,
                    "batch_index": 0 if selected else None, "token_index": position if selected else None,
                    "output_selector": [0], "output_shape": [1, 10, 768],
                    "selected_shape": [768] if selected else [1, 10, 768],
                    "dtype": "torch.float32", "device": "cpu",
                    "source_capture": "before" if name == "patch" and mode.endswith("_self") else None,
                    "preceding_same_site_actions": names[:index]}
        if not _same_typed(steps[index], expected):
            return False
    return True


def _completion_matches(event, identity, rows, state):
    row, layer, position, mode = identity["row_id"], identity["layer"], identity["position"], identity["mode"]
    expected_entry = {"input_ids": rows[row]["input_ids"], "attention_mask": rows[row]["attention_mask"]}
    if not _same_typed(event.get("model_entry"), expected_entry):
        return False
    probe = {"smallest_times_one_bits": 1, "normal_half_bits": 4194304}
    if any(not _same_typed(event.get(key), probe) for key in ("denormal_before", "denormal_after", "denormal_after_scoring")):
        return False
    summary = event.get("summary")
    if type(summary) is not dict or not _sha256(summary.get("mask_logits_sha256")):
        return False
    for key in ("sweet_bits", "sour_bits"):
        value = summary.get(key)
        if type(value) is not int or not 0 <= value < 2 ** 32 or ((value >> 23) & 255) == 255:
            return False
    for key in ("sweet_logit", "sour_logit", "sweet_minus_sour", "sweet_probability", "sour_probability"):
        value = summary.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            return False
    if (summary["sweet_logit"] != _decode_f32(summary["sweet_bits"])
            or summary["sour_logit"] != _decode_f32(summary["sour_bits"])
            or summary["sweet_minus_sour"] != math.fsum((summary["sweet_logit"], -summary["sour_logit"]))):
        return False
    if not (0 <= summary["sweet_probability"] <= 1 and 0 <= summary["sour_probability"] <= 1):
        return False
    ids, tokens = summary.get("top_five_ids"), summary.get("top_five_tokens")
    if (type(ids) is not list or len(ids) != 5 or len(set(ids)) != 5
            or any(type(x) is not int or not 0 <= x < 30522 for x in ids)
            or type(tokens) is not list or len(tokens) != 5 or any(type(x) is not str for x in tokens)):
        return False
    before = _payload(event.get("before_float32_le_base64"), 768 * 4)
    if not _sha256(event.get("full_before_sha256")) or not _metadata_matches(event, layer, position, mode):
        return False
    full_key = (row, layer)
    if full_key in state["full_before"] and state["full_before"][full_key] != event["full_before_sha256"]:
        return False
    state["full_before"][full_key] = event["full_before_sha256"]
    capture = mode.startswith("capture_")
    matched = mode.endswith("_matched")
    patched = mode in ("ingredient_donor", "ingredient_self", "control_self", "ingredient_matched", "control_matched")
    baseline_raw = event.get("baseline_mask_logits_float32_le_base64")
    if mode == "capture_ingredient" and layer == 0:
        logits = _payload(baseline_raw, 30522 * 4)
        if (hashlib.sha256(logits).hexdigest() != summary["mask_logits_sha256"]
                or struct.unpack_from("<I", logits, 4086 * 4)[0] != summary["sweet_bits"]
                or struct.unpack_from("<I", logits, 14768 * 4)[0] != summary["sour_bits"]):
            return False
        state["baselines"][row] = summary["mask_logits_sha256"]
    elif baseline_raw is not None:
        return False
    baseline = state["baselines"].get(row)
    if baseline is None:
        return False
    if capture:
        state["captures"][(row, layer, position)] = before
    if (capture or mode == "final_clean" or mode.endswith("_self") or layer == 5) and summary["mask_logits_sha256"] != baseline:
        return False
    if patched:
        after = _payload(event.get("after_float32_le_base64"), 768 * 4)
        if (before != state["captures"].get((row, layer, position))
                or not _sha256(event.get("full_after_sha256")) or event.get("untouched_positions_equal") is not True):
            return False
        if mode.endswith("_self"):
            if (after != before or event["full_after_sha256"] != event["full_before_sha256"]
                    or event.get("source_float32_le_base64") is not None or event.get("donor_clean_float32_le_base64") is not None):
                return False
        else:
            source = _payload(event.get("source_float32_le_base64"), 768 * 4)
            donor = _payload(event.get("donor_clean_float32_le_base64"), 768 * 4)
            if (after != source or donor != state["captures"].get((identity["donor"], layer, position))
                    or (mode == "ingredient_donor" and source != donor)):
                return False
    elif any(event.get(key) is not None for key in ("after_float32_le_base64", "source_float32_le_base64",
                                                      "donor_clean_float32_le_base64", "full_after_sha256",
                                                      "untouched_positions_equal")):
        return False
    if matched:
        dose = event.get("dose")
        actual = event.get("actual_pre_post_norm")
        plan = state["plans"].get(layer)
        site = "ingredient" if position == 7 else "control"
        if (type(dose) is not dict or type(actual) not in (int, float) or not math.isfinite(actual) or actual <= 0
                or any(type(dose.get(key)) not in (int, float) or not math.isfinite(dose[key])
                       for key in ("fraction", "relative_error", "recipient_norm"))
                or type(dose.get("delivered_norm")) not in (int, float) or dose["delivered_norm"] != actual
                or type(dose.get("requested_norm")) not in (int, float) or not math.isfinite(dose["requested_norm"])
                or plan is None or dose["requested_norm"] != plan["radius"]
                or dose.get("fraction") != plan["fractions"][site]
                or actual != _distance(after, before)
                or abs(actual - dose["requested_norm"]) > 1e-6 + 1e-4 * dose["requested_norm"]
                or dose.get("relative_error") != abs(actual - plan["radius"]) / plan["radius"]):
            return False
        recipient_norm = _distance(before, bytes(len(before)))
        if (dose.get("recipient_norm") != recipient_norm
                or dose.get("recipient_norm_status") != ("zero" if recipient_norm == 0 else "positive")
                or dose.get("dose_to_recipient_norm") != (None if recipient_norm == 0 else actual / recipient_norm)):
            return False
    elif event.get("dose") is not None or event.get("actual_pre_post_norm") is not None:
        return False
    return True


def _journal_schedule_matches(data, frame, rows):
    """Reject missing, extra, malformed, reordered, or incomplete evidence."""
    if not data.endswith(b"\n"):
        return False
    lines, expected = data.splitlines(), _schedule(frame)
    if len(lines) != len(expected):
        return False
    state = {"baselines": {}, "captures": {}, "plans": {}, "full_before": {}}
    for line, identity in zip(lines, expected):
        event = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if type(event) is not dict or any(key not in event or type(event[key]) is not type(value)
                                          or event[key] != value for key, value in identity.items()):
            return False
        if identity["event"] == "forward_complete" and not _completion_matches(event, identity, rows, state):
            return False
        if identity["event"] == "dose_plan":
            distances, fractions, radius = event.get("distances"), event.get("fractions"), event.get("radius")
            if (type(distances) is not dict or type(fractions) is not dict
                    or set(distances) != {"ingredient", "control"}
                    or set(fractions) != {"ingredient", "control"}
                    or type(radius) not in (int, float) or not math.isfinite(radius) or radius < 1e-5
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x < 1e-5
                           for x in distances.values())
                    or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
                           for x in fractions.values())
                    or radius != min(distances.values())
                    or any(fractions[site] != radius / distances[site] for site in distances)):
                return False
            state["plans"][identity["layer"]] = {"radius": radius, "fractions": fractions}
    return True


def _audit_saved_bytes(output, before):
    """Run the verified stdlib auditor in a fresh isolated interpreter."""
    name = "scripts/bert_taste_matched_independent.py"
    source = _bounded(ROOT / name, 1024 * 1024)
    if hashlib.sha256(source).hexdigest() != before["selected_source_sha256"][name]:
        raise ValueError("saved-byte auditor source differs")
    cases = "experiments/bert_taste_matched_dose_public_cases_v1.json"
    prefix = Path(tempfile.mkdtemp(prefix="bert-taste-audit-pyc-", dir=output.parent))
    program = ("import sys; raw=sys.stdin.buffer.read(); ns={'__name__':'_saved_auditor'}; "
               "exec(compile(raw,'<verified-auditor>','exec'),ns); "
               "r=ns['audit_files'](sys.argv[1],sys.argv[2],sys.argv[3],manifest_sha256=sys.argv[4])\n"
               "if type(r.get('completed_forwards')) is not int or r['completed_forwards'] != 516:\n"
               " raise ValueError('saved-byte auditor did not verify the full schedule')")
    command = [sys.executable, "-I", "-B", "-X", f"pycache_prefix={prefix}", "-c", program,
               str(ROOT / cases), str(output / "result.json"), str(output), before["selected_source_sha256"][cases]]
    subprocess.run(command, cwd=ROOT, input=source, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=120, check=True)
    if any(prefix.rglob("*")):
        raise ValueError("saved-byte auditor prefix is not empty")


def assess_result(output, exit_code, before, after, prefix_empty):
    """Require the child's exit and exact result/journal bytes for completion."""
    receipt = {"status": "incomplete", "child_exit_code": exit_code,
               "child_reaped": True, "source_unchanged": before == after,
               "bytecode_prefix_empty": prefix_empty}
    if exit_code != 0 or before != after or not prefix_empty or (output / "incomplete.json").exists():
        return receipt
    try:
        raw = _bounded(output / "result.json", 2 * 1024 * 1024)
        result = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if type(result) is not dict or not _finite_json(result):
            return receipt
        started = json.loads(_bounded(output / "started.json", 256 * 1024),
                             object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not _same_typed(started, {"status": "started", "source": before,
                                     "cases_path": "experiments/bert_taste_matched_dose_public_cases_v1.json",
                                     "seed": 20261030}):
            return receipt
        journals = result["journals"]
        cases = json.loads(_bounded(ROOT / "experiments/bert_taste_matched_dose_public_cases_v1.json", 1024 * 1024),
                           object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        rows = {row["id"]: row for row in cases["rows"]}
        frame_names = ("coffee", "milk", "soup", "cake", "pudding", "cocoa")
        expected_rows = {frame + "-" + word for frame in frame_names for word in ("sugar", "lemon")}
        if (result["status"] != "complete" or type(result["completed_forwards"]) is not int
                or result["completed_forwards"] != 516 or not _same_typed(result["source"], before)
                or result.get("schema") != "bert-taste-matched-dose-result-v1"
                or type(result.get("seed")) is not int or result["seed"] != 20261030
                or set(rows) != expected_rows or len(cases["rows"]) != 12
                or type(result.get("baselines")) is not dict or set(result["baselines"]) != expected_rows
                or type(result.get("eligibility")) is not dict or set(result["eligibility"]) != set(frame_names)
                or any(type(x) is not bool for x in result["eligibility"].values())
                or type(result.get("effects")) is not list or len(result["effects"]) != 360
                or type(result.get("doses")) is not list or len(result["doses"]) != 36
                or type(result.get("comparisons")) is not list or len(result["comparisons"]) != 72
                or type(result.get("layer_summary")) is not list or len(result["layer_summary"]) != 6
                or set(journals) != {"coffee.jsonl", "milk.jsonl", "soup.jsonl", "cake.jsonl", "pudding.jsonl", "cocoa.jsonl"}):
            return receipt
        model = result.get("model")
        if (type(model) is not dict or model.get("model") != "distilbert/distilbert-base-uncased"
                or model.get("revision") != "12040accade4e8a0f71eabdb258fecc2e7e948be"
                or model.get("device") != "cpu" or model.get("dtype") != "float32"
                or model.get("model_class") != "DistilBertForMaskedLM"
                or model.get("tokenizer_class") != "DistilBertTokenizerFast"
                or model.get("attention_implementation") != "eager"
                or type(model.get("parameter_count")) is not int or model["parameter_count"] != 66985530
                or model.get("pytorch_cuda_initialized") is not False
                or not _same_typed(model.get("assets"), EXPECTED_ASSETS)
                or any(type(model.get(k)) is not str or not model[k] for k in ("torch_version", "transformers_version"))):
            return receipt
        runtime = result.get("runtime")
        if (type(runtime) is not dict or type(runtime.get("threads")) is not int or runtime["threads"] != 1
                or type(runtime.get("interop_threads")) is not int or runtime["interop_threads"] != 1
                or runtime.get("deterministic_algorithms") is not True or runtime.get("cuda_initialized") is not False
                or runtime.get("default_dtype") != "torch.float32" or type(runtime.get("mkldnn_enabled")) is not bool
                or runtime.get("omp_num_threads") != "1" or runtime.get("mkl_num_threads") != "1"
                or runtime.get("mkl_cbwr") != "AUTO" or runtime.get("aten_cpu_capability") is not None
                or not _same_typed(runtime.get("denormal_probe"), {"smallest_times_one_bits": 1, "normal_half_bits": 4194304})
                or runtime.get("rounding_probe") != "nearest-even-halfway-cases-pass"
                or not _sha256(runtime.get("torch_build_sha256"))
                or runtime.get("torch") != model.get("torch_version")
                or any(type(runtime.get(k)) is not str or not runtime[k] for k in ("python", "torch"))):
            return receipt
        if (not _sha256(result.get("model_state_sha256_before_after"))
                or not _same_typed(result.get("exposure"), {"prior_row_counts": [160, 64, 8],
                                   "exact_or_complete_prefix_collisions": 0,
                                   "complete_exposure_history_certified": False})):
            return receipt
        for name, expected in journals.items():
            data = _bounded(output / name, 4 * 1024 * 1024)
            if (not _same_typed(expected, {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                    or not _journal_schedule_matches(data, name.removesuffix(".jsonl"), rows)):
                return receipt
        _audit_saved_bytes(output, before)
    except Exception:
        return receipt
    receipt.update({"status": "complete", "result_sha256": hashlib.sha256(raw).hexdigest(),
                    "journals": journals, "source_commit": before["commit"]})
    return receipt


def _write_receipt(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        remaining = memoryview(raw)
        while remaining:
            written = stream.write(remaining)
            if type(written) is not int or not 0 < written <= len(remaining):
                raise OSError("receipt write made no progress")
            remaining = remaining[written:]
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    receipt_path = output.with_name(output.name + ".observer.json")
    if (output.exists() or receipt_path.exists() or not output.parent.is_dir()
            or output == ROOT or ROOT in output.parents):
        raise ValueError("a fresh output and receipt outside the checkout are required")
    before = source_snapshot()
    prefix = Path(tempfile.mkdtemp(prefix="bert-taste-matched-pyc-", dir=output.parent))
    if any(prefix.iterdir()):
        raise RuntimeError("fresh bytecode prefix is not empty")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "MKL_CBWR": "AUTO", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    env.pop("ATEN_CPU_CAPABILITY", None)
    command = [sys.executable, "-I", "-B", "-X", f"pycache_prefix={prefix}"]
    if args.preflight_only:
        command += ["-c", "import sys; assert sys.pycache_prefix and sys.dont_write_bytecode"]
    else:
        command += [str(CHILD), "--cache-dir", args.cache_dir, "--output", str(output)]
    try:
        child = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=900, check=False)
        after = source_snapshot()
        empty = not any(prefix.rglob("*"))
        if args.preflight_only:
            ok = child.returncode == 0 and before == after and empty
            receipt = {"status": "preflight_passed" if ok else "incomplete", "child_reaped": True,
                       "child_exit_code": child.returncode, "source_unchanged": before == after,
                       "bytecode_prefix_empty": empty, "source_commit": before["commit"]}
        else:
            receipt = assess_result(output, child.returncode, before, after, empty)
        _write_receipt(receipt_path, receipt)
        return 0 if receipt["status"] in ("complete", "preflight_passed") else 1
    except BaseException as exc:
        try:
            _write_receipt(receipt_path, {"status": "incomplete", "exception_type": type(exc).__name__,
                                          "child_reaped": False})
        except BaseException as secondary:
            if isinstance(secondary, (KeyboardInterrupt, SystemExit)) and not isinstance(
                    exc, (KeyboardInterrupt, SystemExit)):
                raise secondary from exc
        raise


if __name__ == "__main__":
    raise SystemExit(main())
