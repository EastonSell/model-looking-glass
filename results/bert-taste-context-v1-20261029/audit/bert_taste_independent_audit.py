"""Independent standard-library audit of saved public taste-study evidence.

No model, checkpoint, private input, or study implementation is imported.
This checks saved records, not execution provenance or full-output equality.
"""
import argparse
import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct


class AuditError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise AuditError(message)


def decode_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda _x: (_ for _ in ()).throw(AuditError("nonfinite JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError("invalid JSON") from exc


def bounded(path, maximum):
    item = Path(path).lstat()
    require(stat.S_ISREG(item.st_mode) and not stat.S_ISLNK(item.st_mode)
            and not getattr(item, "st_file_attributes", 0) & 0x400, "evidence is not an ordinary file")
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "saved evidence exceeds its bound")
    return raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def equal_typed(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def float_bits(bits):
    require(type(bits) is int and 0 <= bits <= 0xffffffff, "invalid float32 bits")
    exponent, fraction = (bits >> 23) & 0xff, bits & 0x7fffff
    require(exponent != 0xff, "nonfinite float32")
    magnitude = (math.ldexp(float(fraction), -149) if exponent == 0
                 else math.ldexp(float((1 << 23) | fraction), exponent - 150))
    return -magnitude if bits >> 31 else magnitude


def vector(record, field, size, sha_field):
    encoded = record.get(field)
    require(type(encoded) is str, "missing saved vector")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError, binascii.Error) as exc:
        raise AuditError("invalid base64 vector") from exc
    require(len(raw) == size * 4 and digest(raw) == record.get(sha_field), "vector bytes/hash differ")
    for (bits,) in struct.iter_unpack("<I", raw):
        float_bits(bits)
    return raw


def number_equal(actual, expected, message, *, tolerance=0.0):
    require(type(actual) in (int, float) and math.isfinite(actual), message)
    require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), message)


def schedule(family, layers):
    pair = [family + "-sugar", family + "-lemon"]
    for layer in range(layers):
        for row in pair:
            for position, role in ((7, "ingredient"), (1, "control")):
                yield (row, layer, position, "capture_" + role, None)
        for index, recipient in enumerate(pair):
            donor = pair[1 - index]
            for mode, position, source in (("ingredient_swap", 7, donor),
                                           ("ingredient_self", 7, recipient),
                                           ("control_swap", 1, donor)):
                yield (recipient, layer, position, mode, source)
    for row in pair:
        yield (row, 0, 7, "final_clean", None)


def audit_records(result, journal_bytes, *, families, layers, hidden, vocab, sweet, sour):
    require(type(result) is dict and result.get("status") == "complete", "missing complete child result")
    expected_journals = {family + ".jsonl" for family in families}
    require(set(journal_bytes) == expected_journals and set(result.get("journals", {})) == expected_journals,
            "journal inventory differs")
    expected_count = len(families) * (layers * 10 + 2)
    require(type(result.get("completed_forwards")) is int and result["completed_forwards"] == expected_count,
            "forward count differs")
    require(type(result.get("baselines")) is dict and type(result.get("effects")) is list,
            "missing result summaries")
    captures, full_clean, references, base_margin = {}, {}, {}, {}
    effects = []
    for family in families:
        name = family + ".jsonl"
        raw = journal_bytes[name]
        require(equal_typed(result["journals"][name], {"size": len(raw), "sha256": digest(raw)}), "journal hash differs")
        require(raw.endswith(b"\n"), "incomplete journal")
        lines = raw.splitlines()
        planned = list(schedule(family, layers))
        require(len(lines) == 2 * len(planned), "journal record count differs")
        for index, identity in enumerate(planned):
            start, record = decode_json(lines[2 * index]), decode_json(lines[2 * index + 1])
            row, layer, position, mode, donor = identity
            wanted = {"row_id": row, "layer": layer, "position": position, "mode": mode, "donor": donor}
            require(type(start) is dict and type(record) is dict, "journal row is not an object")
            require(start == {"event": "forward_start", **wanted}, "forward start differs")
            require(all(record.get(k) == v and type(record.get(k)) is type(v) for k, v in wanted.items())
                    and record.get("event") == "forward_complete", "forward completion differs")
            probe = {"smallest_times_one_bits": 1, "normal_half_bits": 0x00400000}
            require(record.get("denormal_before") == probe and record.get("denormal_after") == probe,
                    "denormal observation differs")
            require(all(type(record[key][name]) is int for key in ("denormal_before", "denormal_after")
                        for name in probe), "denormal observations have invalid types")
            patch_mode = mode in ("ingredient_swap", "ingredient_self", "control_swap")
            names = ["full_before", "before"] + (["patch", "after", "full_after"] if patch_mode else [])
            events = record.get("events")
            require(type(events) is list and len(events) == len(names), "intervention step inventory differs")
            for step_index, (event, step_name) in enumerate(zip(events, names)):
                require(type(event) is dict and event.get("name") == step_name
                        and event.get("operation") == ("patch" if step_name == "patch" else "capture")
                        and type(event.get("order")) is int and event["order"] == step_index
                        and event.get("module_path") == "distilbert.transformer.layer." + str(layer)
                        and event.get("module_aliases") == [event.get("module_path")]
                        and type(event.get("forward_index")) is int and event["forward_index"] == 0
                        and type(event.get("invocation")) is int and event["invocation"] == 0
                        and event.get("output_selector") == [0]
                        and event.get("output_shape") == [1, 10, hidden]
                        and event.get("dtype") == "torch.float32" and event.get("device") == "cpu"
                        and event.get("preceding_same_site_actions") == names[:step_index],
                        "intervention step metadata differs")
                whole = step_name in ("full_before", "full_after")
                require(event.get("batch_index") == (None if whole else 0)
                        and event.get("token_index") == (None if whole else position)
                        and (whole or (type(event.get("batch_index")) is int and type(event.get("token_index")) is int))
                        and event.get("selected_shape") == ([1, 10, hidden] if whole else [hidden])
                        and event.get("source_capture") == ("before" if step_name == "patch" and mode == "ingredient_self" else None),
                        "intervention coordinates differ")
            before = vector(record, "before_float32_le_base64", hidden, "before_sha256")
            full_before = record.get("full_before_sha256")
            require(type(full_before) is str and len(full_before) == 64
                    and all(ch in "0123456789abcdef" for ch in full_before), "missing full-output hash")
            selected = (row, layer, position)
            if mode.startswith("capture_"):
                captures[selected] = before
                if (row, layer) in full_clean:
                    require(full_clean[(row, layer)] == full_before, "clean whole-output hashes differ")
                else:
                    full_clean[(row, layer)] = full_before
                require(record.get("after_float32_le_base64") is None and record.get("source_float32_le_base64") is None
                        and record.get("after_sha256") is None and record.get("source_sha256") is None
                        and record.get("full_after_sha256") is None and record.get("untouched_positions_equal") is None,
                        "clean capture claims an edit")
            elif mode != "final_clean":
                require(before == captures[selected] and full_before == full_clean[(row, layer)],
                        "pre-patch capture differs from clean")
                after = vector(record, "after_float32_le_base64", hidden, "after_sha256")
                if mode == "ingredient_self":
                    require(record.get("source_float32_le_base64") is None and record.get("source_sha256") is None,
                            "self patch has an external source")
                    wanted_vector = before
                else:
                    wanted_vector = captures[(donor, layer, position)]
                    source = vector(record, "source_float32_le_base64", hidden, "source_sha256")
                    require(source == wanted_vector, "donor vector differs from clean capture")
                require(after == wanted_vector and record.get("untouched_positions_equal") is True,
                        "selected landing or owner witness differs")
                require(type(record.get("full_after_sha256")) is str and len(record["full_after_sha256"]) == 64
                        and all(ch in "0123456789abcdef" for ch in record["full_after_sha256"]),
                        "missing post-patch full-output hash")
                if mode == "ingredient_self":
                    require(record["full_after_sha256"] == full_before, "self-patch full-output hash differs")
            else:
                require(before == captures[selected] and full_before == full_clean[(row, layer)]
                        and record.get("after_float32_le_base64") is None
                        and record.get("source_float32_le_base64") is None
                        and record.get("after_sha256") is None and record.get("source_sha256") is None
                        and record.get("full_after_sha256") is None
                        and record.get("untouched_positions_equal") is None,
                        "final clean capture differs")
            summary = record.get("summary")
            require(type(summary) is dict, "missing mask summary")
            sweet_value, sour_value = float_bits(summary.get("sweet_bits")), float_bits(summary.get("sour_bits"))
            margin = math.fsum((sweet_value, -sour_value))
            number_equal(summary.get("sweet_logit"), sweet_value, "sweet logit differs")
            number_equal(summary.get("sour_logit"), sour_value, "sour logit differs")
            number_equal(summary.get("sweet_minus_sour"), margin, "margin differs")
            if mode == "capture_ingredient" and layer == 0:
                try:
                    full = base64.b64decode(record["baseline_mask_logits_float32_le_base64"], validate=True)
                except (ValueError, KeyError, TypeError, binascii.Error) as exc:
                    raise AuditError("invalid baseline logits") from exc
                require(len(full) == 4 * vocab and digest(full) == summary.get("mask_logits_sha256"),
                        "baseline logit bytes differ")
                vals = [float_bits(bits) for (bits,) in struct.iter_unpack("<I", full)]
                require(struct.unpack_from("<I", full, 4 * sweet)[0] == summary["sweet_bits"]
                        and struct.unpack_from("<I", full, 4 * sour)[0] == summary["sour_bits"],
                        "target bits differ from baseline row")
                maximum = max(vals)
                denom = math.fsum(math.exp(value - maximum) for value in vals)
                number_equal(summary.get("sweet_probability"), math.exp(vals[sweet] - maximum) / denom,
                             "baseline sweet probability differs", tolerance=1e-12)
                number_equal(summary.get("sour_probability"), math.exp(vals[sour] - maximum) / denom,
                             "baseline sour probability differs", tolerance=1e-12)
                tops = summary.get("top_five_ids")
                require(type(tops) is list and len(tops) == 5 and len(set(tops)) == 5
                        and all(type(item) is int and 0 <= item < vocab for item in tops), "invalid baseline top five")
                require([vals[item] for item in tops] == sorted(vals, reverse=True)[:5], "baseline top values differ")
                references[row] = digest(full)
                base_margin[row] = margin
                baseline_summary = dict(result["baselines"].get(row, {}))
                baseline_summary.pop("top_five_tokens", None)
                require(equal_typed(baseline_summary, summary), "published baseline summary differs")
            else:
                require(record.get("baseline_mask_logits_float32_le_base64") is None, "unexpected baseline row")
            if mode.startswith("capture_") or mode == "final_clean" or mode == "ingredient_self" or layer == layers - 1:
                require(summary.get("mask_logits_sha256") == references[row], "clean/control logits hash differs")
            if not mode.startswith("capture_") and mode != "final_clean":
                effect = math.fsum((margin, -base_margin[row]))
                effects.append({"recipient": row, "donor": donor, "layer": layer, "position": position,
                                "mode": mode, "baseline_margin": base_margin[row],
                                "patched_margin": margin, "effect": effect})
    require(set(result["baselines"]) == set(references), "baseline inventory differs")
    require(equal_typed(result["effects"], effects), "saved directional effects differ")
    summaries = []
    for layer in range(layers):
        entry = {"layer": layer}
        for mode in ("ingredient_swap", "ingredient_self", "control_swap"):
            directions = (("lemon_self", "-lemon"), ("sugar_self", "-sugar")) if mode == "ingredient_self" else (
                ("sugar_to_lemon", "-lemon"), ("lemon_to_sugar", "-sugar"))
            entry[mode] = {}
            for direction, suffix in directions:
                values = [item["effect"] for item in effects if item["layer"] == layer
                          and item["mode"] == mode and item["recipient"].endswith(suffix)]
                require(len(values) == len(families), "effect group differs")
                entry[mode][direction] = {"n": len(values), "mean_effect": math.fsum(values) / len(values)}
        summaries.append(entry)
    require(equal_typed(result.get("layer_summary"), summaries), "layer summary differs")
    return {"forwards_checked": expected_count, "directional_effects_checked": len(effects),
            "baseline_rows_checked": len(references)}


def audit(root, observer_path, commit):
    root = Path(root)
    observer_raw = bounded(observer_path, 65536)
    observer = decode_json(observer_raw)
    require(type(observer) is dict and observer.get("status") == "complete"
            and type(observer.get("child_exit_code")) is int and observer["child_exit_code"] == 0
            and observer.get("child_reaped") is True and observer.get("source_unchanged") is True
            and observer.get("bytecode_prefix_empty") is True and observer.get("source_commit") == commit,
            "observer outcome differs")
    require(not os.path.lexists(root / "incomplete.json"), "incomplete marker present")
    require({item.name for item in root.iterdir()} == {"started.json", "result.json", "tea.jsonl",
            "drink.jsonl", "sauce.jsonl", "water.jsonl"}, "saved result directory inventory differs")
    result_raw = bounded(root / "result.json", 2 * 1024 * 1024)
    require(digest(result_raw) == observer.get("result_sha256"), "observer/result hash differs")
    result = decode_json(result_raw)
    started = decode_json(bounded(root / "started.json", 65536))
    require(type(started) is dict and started.get("status") == "started"
            and type(started.get("seed")) is int and started["seed"] == 20261029
            and started.get("cases_path") == "experiments/bert_taste_public_cases_v1.json"
            and started.get("source") == result.get("source"), "start/result source binding differs")
    require(result.get("schema") == "bert-taste-context-result-v1"
            and result.get("source", {}).get("commit") == commit, "source or result identity differs")
    model, runtime = result.get("model", {}), result.get("runtime", {})
    require(model.get("model") == "distilbert/distilbert-base-uncased"
            and model.get("revision") == "12040accade4e8a0f71eabdb258fecc2e7e948be"
            and model.get("assets", {}).get("model.safetensors", {}).get("sha256") ==
            "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
            and model.get("device") == "cpu" and model.get("dtype") == "float32"
            and model.get("attention_implementation") == "eager", "recorded model identity differs")
    require(type(runtime.get("threads")) is int and runtime["threads"] == 1
            and type(runtime.get("interop_threads")) is int and runtime["interop_threads"] == 1
            and runtime.get("deterministic_algorithms") is True and runtime.get("cuda_initialized") is False,
            "recorded runtime differs")
    journals = {name + ".jsonl": bounded(root / (name + ".jsonl"), 4 * 1024 * 1024)
                for name in ("tea", "drink", "sauce", "water")}
    require(equal_typed(observer.get("journals"), result.get("journals")), "observer/journal inventory differs")
    checks = audit_records(result, journals, families=("tea", "drink", "sauce", "water"),
                           layers=6, hidden=768, vocab=30522, sweet=4086, sour=14768)
    return {"status": "saved_public_evidence_consistent", "source_commit": commit,
            "result_sha256": digest(result_raw), "observer_sha256": digest(observer_raw),
            "checks": checks,
            "limitations": ["No model execution, dependency identity, publication timing, or process-tree lifetime is authenticated.",
                            "Full-output untouched-coordinate equality and patched full-vocabulary summaries remain owner-observed.",
                            "The public prompts are exploratory and do not establish held-out generalization."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--observer", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    require(len(args.source_commit) == 40 and all(c in "0123456789abcdef" for c in args.source_commit),
            "expected an exact source commit")
    report = audit(args.result_dir, args.observer, args.source_commit)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with Path(args.report).open("xb") as stream:
        require(stream.write(raw) == len(raw), "report write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
