"""Independent model-free replay of the relative-dose agreement study."""

import hashlib
import json
import math
from pathlib import Path
import re
import struct


def _rounding_probe():
    one, half = 1.0, math.ldexp(1.0, -53)
    next_one = one + math.ldexp(1.0, -52)
    if (one + half != one or next_one + half != 1.0 + math.ldexp(1.0, -51)
            or one - math.ldexp(1.0, -54) != one or next_one * 1.5 != 1.5 + math.ldexp(1.0, -51)
            or one / 10.0 != float.fromhex("0x1.999999999999ap-4")
            or math.sqrt(2.0) != float.fromhex("0x1.6a09e667f3bcdp+0")
            or struct.pack("<f", 1.0 + 2.0 ** -24) != bytes.fromhex("0000803f")
            or struct.pack("<f", 1.0 + 3.0 * 2.0 ** -24) != bytes.fromhex("0200803f")):
        raise ValueError("replay arithmetic is not nearest-even under the fixed probes")


def _values(raw):
    if type(raw) is not bytes or not raw or len(raw) % 4:
        raise ValueError("invalid binary32 payload")
    out = []
    for (bits,) in struct.iter_unpack("<I", raw):
        sign, exponent, fraction = bits >> 31, (bits >> 23) & 255, bits & 0x7fffff
        if exponent == 255:
            raise ValueError("nonfinite binary32 payload")
        significand = fraction if exponent == 0 else (1 << 23) | fraction
        value = math.ldexp(float(significand), -149 if exponent == 0 else exponent - 150)
        out.append(-value if sign else value)
    return out


def _distance(a, b):
    if len(a) != len(b):
        raise ValueError("head-vector size differs")
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(_values(a), _values(b), strict=True)))


def _margin(raw, ids):
    values = _values(raw)
    return values[ids[0]] - values[ids[1]]


def _evaluation(details, candidate, comparator):
    by_key = {(item["head"], item["target"]): item for item in details}
    if len(by_key) != 32 or set(by_key) != {(h, i) for h in (candidate, comparator) for i in range(16)}:
        raise ValueError("missing evaluation cells")
    a = [by_key[candidate, i]["D"] for i in range(16)]
    b = [by_key[comparator, i]["D"] for i in range(16)]
    correct = [by_key[candidate, i]["sign"] * by_key[candidate, i]["clean_margin"] > 0 for i in range(16)]
    frames = []
    for frame in range(4):
        indices = range(4 * frame, 4 * frame + 4)
        mean_a = math.fsum(a[i] for i in indices) / 4
        mean_abs_b = math.fsum(abs(b[i]) for i in indices) / 4
        frames.append({"frame": frame, "correct": sum(correct[i] for i in indices),
                       "candidate_mean_D": mean_a, "comparator_mean_abs_D": mean_abs_b,
                       "criterion": mean_a > 0 and mean_a > mean_abs_b})
    mean_a, mean_abs_b = math.fsum(a) / 16, math.fsum(abs(value) for value in b) / 16
    baseline = sum(correct) >= 12 and all(item["correct"] >= 3 for item in frames)
    criterion = (mean_a >= 0.05 and mean_a > mean_abs_b and sum(value > 0 for value in a) >= 10
                 and sum(item["criterion"] for item in frames) >= 3)
    return {"baseline_correct": sum(correct), "baseline_adequate": baseline,
            "candidate_mean_D": mean_a, "comparator_mean_abs_D": mean_abs_b,
            "candidate_positive": sum(value > 0 for value in a), "candidate_zero": sum(value == 0 for value in a),
            "candidate_negative": sum(value < 0 for value in a), "frames": frames, "criterion": criterion,
            "status": "baseline_limited" if not baseline else
                      ("passes_fixed_rule" if criterion else "does_not_pass_fixed_rule")}


def audit(directory, rows, *, phase, heads, present_ids, secondary_ids=None,
          expected_width=768, expected_nheads=12, expected_vocab=50257):
    """Recompute one closed attempt; callers separately verify source and admission."""
    _rounding_probe()
    root = Path(directory)
    if (phase not in ("development", "evaluation") or len(rows) != 16 or len(heads) < 2
            or (phase == "evaluation" and len(heads) != 2) or len(set(heads)) != len(heads)):
        raise ValueError("invalid phase, row or head specification")
    if (any(type(value) is not int or value <= 0 for value in (expected_width, expected_nheads, expected_vocab))
            or expected_width % expected_nheads):
        raise ValueError("invalid expected model layout")
    terminal = json.loads((root / "terminal.json").read_bytes())
    started = json.loads((root / "started.json").read_bytes())
    raw_events = (root / "events.jsonl").read_bytes()
    if (started.get("phase") != phase or started.get("protocol") != "distilgpt2-agreement-heads-v2"
            or terminal.get("status") != "complete_candidate"
            or terminal.get("storage_accounting_complete") is not True
            or terminal.get("event_log_bytes") != len(raw_events) or len(raw_events) > 8 * 1024 * 1024
            or not raw_events.endswith(b"\n")):
        raise ValueError("attempt is not a bounded complete candidate")
    events = [json.loads(line) for line in raw_events.splitlines()]
    if terminal.get("events") != len(events) or len(events) > 1000:
        raise ValueError("event count differs")
    for index, record in enumerate(events):
        if (type(record) is not dict or set(record) != {"index", "event", "tensors"}
                or type(record["index"]) is not int or record["index"] != index):
            raise ValueError("event indices or shape differ")
    cursor, ordinal, unique = 0, 0, {}

    def next_record(event_type):
        nonlocal cursor
        if cursor >= len(events):
            raise ValueError("missing event")
        record = events[cursor]
        cursor += 1
        if record["event"].get("type") != event_type:
            raise ValueError("event order differs")
        return record

    def payload(reference):
        if (type(reference) is not dict or set(reference) != {"sha256", "shape", "dtype", "bytes"}
                or type(reference["sha256"]) is not str
                or re.fullmatch("[0-9a-f]{64}", reference["sha256"]) is None
                or reference["dtype"] != "float32-le" or type(reference["shape"]) is not list
                or not reference["shape"] or any(type(i) is not int or i <= 0 for i in reference["shape"])
                or type(reference["bytes"]) is not int or math.prod(reference["shape"]) * 4 != reference["bytes"]
                or reference["bytes"] > 1024 * 1024):
            raise ValueError("invalid tensor reference")
        data = (root / "tensors" / (reference["sha256"] + ".f32le")).read_bytes()
        if len(data) != reference["bytes"] or hashlib.sha256(data).hexdigest() != reference["sha256"]:
            raise ValueError("tensor bytes or digest differ")
        _values(data)
        unique[reference["sha256"]] = len(data)
        return data

    def read_forward(row, head, kind, donor=None):
        nonlocal ordinal
        _rounding_probe()
        identity = {"kind": kind, "row": row, "head": head, "donor": donor, "ordinal": ordinal}
        begin, complete = next_record("forward_begin"), next_record("forward_complete")
        if (begin["event"] != {"type": "forward_begin", "binary64_before": "nearest-even-probes-passed", **identity}
                or begin["tensors"] != {}):
            raise ValueError("forward announcement differs")
        event, refs = complete["event"], complete["tensors"]
        if any(event.get(key) != value for key, value in identity.items()):
            raise ValueError("forward identity differs")
        expected = {"before", "after", "projection_output", "last_logits"}
        if kind not in ("observe", "repeat"):
            expected.add("replacement")
        if set(refs) != expected:
            raise ValueError("forward tensor set differs")
        data = {name: payload(ref) for name, ref in refs.items()}
        point = event["point"]
        width, nheads = point["input_shape"][-1], point["heads"]
        if (type(width) is not int or type(nheads) is not int or (width, nheads) != (expected_width, expected_nheads)
                or point != {"run_id": f"{phase}-v2-{ordinal}", "module_path": "transformer.h.3.attn.c_proj",
                    "layer": 3, "batch": 0, "token": len(rows[row]) - 1, "invocation": 0,
                    "head": head, "heads": nheads, "head_width": width // nheads,
                    "feature_start": head * (width // nheads), "feature_end": (head + 1) * (width // nheads),
                    "input_shape": [1, len(rows[row]), width], "input_ids": list(rows[row]),
                    "attention_mask": [1] * len(rows[row]), "operation": "observe" if kind in ("observe", "repeat") else "patch",
                    "tensor_role": "projection_input", "input_argument": 0, "forward_index": 0,
                    "use_cache": False, "dtype": "float32", "device": "cpu"}
                or any(refs[name]["shape"] != [1, len(rows[row]), width]
                       for name in ("before", "after", "projection_output"))
                or refs["last_logits"]["shape"] != [expected_vocab]
                or max(present_ids + (() if secondary_ids is None else secondary_ids)) >= expected_vocab):
            raise ValueError("forward coordinates or shapes differ")
        if kind not in ("observe", "repeat") and refs["replacement"]["shape"] != [width // nheads]:
            raise ValueError("replacement width differs")
        postcheck = next_record("numeric_postcheck")
        if postcheck["event"] != {"type": "numeric_postcheck", "ordinal": ordinal,
                                  "binary64_after": "nearest-even-probes-passed"} or postcheck["tensors"]:
            raise ValueError("post-forward numeric probe record differs")
        _rounding_probe()
        ordinal += 1
        return event, data, width, nheads

    observations = []
    for row in range(16):
        event, data, width, nheads = read_forward(row, 0, "observe")
        if data["before"] != data["after"]:
            raise ValueError("observation changed input")
        observations.append((data, width, nheads))
    if len({(x[1], x[2]) for x in observations}) != 1:
        raise ValueError("head layout changed")
    width, nheads = observations[0][1:]
    if any(type(head) is not int or not 0 <= head < nheads for head in heads):
        raise ValueError("head outside observed layout")
    head_width = width // nheads

    def selected(row, head):
        start = ((len(rows[row]) - 1) * width + head * head_width) * 4
        return observations[row][0]["before"][start:start + head_width * 4]

    _rounding_probe()
    targets = [i for i in range(16) if phase == "evaluation" or i % 4 < 2]
    head_records, reconstructed = [], {}
    for head in heads:
        cells, proposed = [], {}
        for target in targets:
            group, local = (target // 4) * 4, target % 4
            source = selected(target, head)
            source_values = _values(source)
            norm = math.sqrt(math.fsum(x * x for x in source_values))
            q = 0.05 * norm
            donors = []
            for name, donor in (("opposite", group + (local ^ 1)), ("same", group + (local ^ 2))):
                donors.append({"kind": name, "donor": donor,
                               "distance": _distance(source, selected(donor, head)), "realized_q": None})
            status = "eligible"
            if q < 2.0 ** -20:
                status = "dose_below_floor"
            elif any(item["distance"] < q for item in donors):
                status = "insufficient_donor_distance"
            else:
                for item in donors:
                    fraction = q / item["distance"]
                    other = _values(selected(item["donor"], head))
                    expected = b"".join(struct.pack("<f", a + fraction * (b - a))
                                        for a, b in zip(source_values, other, strict=True))
                    realized = _distance(source, expected)
                    if realized == 0 or abs(realized - q) > 1e-5 * q:
                        status = "rounded_dose_mismatch"
                        break
                    item["realized_q"] = realized
                    proposed[(head, target, item["kind"])] = expected
                if status == "eligible" and max(item["realized_q"] for item in donors) / min(
                        item["realized_q"] for item in donors) > 1.00001:
                    status = "pair_dose_mismatch"
            cells.append({"target": target, "receiver_norm": norm, "requested_q": q,
                          "donors": donors, "status": status})
        eligible = all(cell["status"] == "eligible" for cell in cells)
        head_records.append({"head": head, "eligible": eligible, "targets": cells})
        if eligible:
            reconstructed.update(proposed)
    expected_plan = {"relative_requested": 0.05, "relative_error_limit": 1e-5,
                     "within_pair_ratio_limit": 1.00001, "heads": head_records}
    plan = next_record("dose_plan")
    if plan["tensors"] or plan["event"] != {"type": "dose_plan", **expected_plan}:
        raise ValueError("relative dose or saved eligibility plan differs")
    active = tuple(entry["head"] for entry in head_records if entry["eligible"])
    if len(active) < 2 or (phase == "evaluation" and active != heads):
        raise ValueError("dose_ineligible: required heads do not meet the fixed dose plan")
    cells = {(entry["head"], cell["target"]): cell for entry in head_records for cell in entry["targets"]}
    _rounding_probe()
    self_targets = targets if phase == "evaluation" else targets[:1]
    for head in active:
        for target in self_targets:
            event, data, _, _ = read_forward(target, head, "self")
            if (data["before"] != observations[target][0]["before"] or data["after"] != data["before"]
                    or data["replacement"] != selected(target, head)
                    or data["last_logits"] != observations[target][0]["last_logits"]):
                raise ValueError("self control differs")
    effects, details = {head: [] for head in active}, []
    for head in active:
        for target in targets:
            group, local = (target // 4) * 4, target % 4
            margins, secondary, doses = {}, {}, {}
            cell = cells[(head, target)]
            q = cell["requested_q"]
            for name, donor in (("opposite", group + (local ^ 1)), ("same", group + (local ^ 2))):
                event, data, _, _ = read_forward(target, head, name, donor)
                source, other = selected(target, head), selected(donor, head)
                distance = _distance(source, other)
                expected = reconstructed[(head, target, name)]
                realized = _distance(source, expected)
                offset = ((len(rows[target]) - 1) * width + head * head_width) * 4
                after = bytearray(observations[target][0]["before"])
                after[offset:offset + len(expected)] = expected
                if (data["before"] != observations[target][0]["before"] or data["replacement"] != expected
                        or data["after"] != bytes(after) or event["requested_q"] != q
                        or event["donor_distance"] != distance or event["realized_q"] != realized
                        or realized == 0 or abs(realized - q) > 1e-5 * q):
                    raise ValueError("patch, untouched bits or achieved dose differs")
                margins[name] = _margin(data["last_logits"], present_ids)
                if secondary_ids is not None:
                    secondary[name] = _margin(data["last_logits"], secondary_ids)
                doses[name] = realized
            sign = 1 if local % 2 == 0 else -1
            contrast = float(sign * (margins["same"] - margins["opposite"]))
            effects[head].append(contrast)
            details.append({"head": head, "target": target, "sign": sign,
                "clean_margin": _margin(observations[target][0]["last_logits"], present_ids),
                "opposite_margin": margins["opposite"], "same_margin": margins["same"],
                "D": contrast, "receiver_norm": cell["receiver_norm"], "requested_q": q,
                "realized_doses": doses,
                "realized_relative_doses": {name: value / cell["receiver_norm"] for name, value in doses.items()},
                "secondary": (None if secondary_ids is None else
                    {"clean_margin": _margin(observations[target][0]["last_logits"], secondary_ids),
                     "opposite_margin": secondary["opposite"], "same_margin": secondary["same"],
                     "D": float(sign * (secondary["same"] - secondary["opposite"]))})})
    for row in range(16):
        event, data, _, _ = read_forward(row, 0, "repeat")
        if any(data[name] != observations[row][0][name] for name in ("before", "after", "last_logits")):
            raise ValueError("repeat control differs")
    _rounding_probe()
    ranking = sorted([{"head": head, "values": effects[head],
                       "mean_D": math.fsum(effects[head]) / len(effects[head])} for head in active],
                     key=lambda item: (-item["mean_D"], item["head"]))
    summary_record = next_record("summary")
    summary = summary_record["event"].copy()
    summary.pop("type")
    expected_evaluation = _evaluation(details, heads[0], heads[1]) if phase == "evaluation" else None
    if (summary_record["tensors"] or summary.get("phase") != phase or summary.get("forward_count") != ordinal
            or summary.get("dose_plan") != expected_plan or summary.get("ranking") != ranking or summary.get("details") != details
            or summary.get("candidate") != (ranking[0]["head"] if phase == "development" else heads[0])
            or summary.get("comparator") != (ranking[1]["head"] if phase == "development" else heads[1])
            or (phase == "evaluation" and summary.get("evaluation") != expected_evaluation)
            or terminal.get("detail") != summary or cursor != len(events)
            or terminal.get("retained_tensor_bytes") != sum(unique.values())
            or sum(unique.values()) > 128 * 1024 * 1024):
        raise ValueError("summary, candidate, terminal or retained byte count differs")
    _rounding_probe()
    return {"phase": phase, "forward_count": ordinal, "dose_plan": expected_plan, "ranking": ranking,
            "candidate": summary["candidate"], "comparator": summary["comparator"],
            "details": details, "evaluation": expected_evaluation, "retained_tensor_bytes": sum(unique.values()),
            "expected_layout": {"width": expected_width, "heads": expected_nheads, "vocab": expected_vocab},
            "scope": "saved bytes and arithmetic; not execution, exposure or descendant-drain proof"}
