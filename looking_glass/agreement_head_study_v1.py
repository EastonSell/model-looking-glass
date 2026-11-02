"""Bounded CPU head-slice agreement comparisons on fixed paired inputs.

This worker does not authenticate its caller's model, source or exposure
admission. A supervising phase owner must bind those before invoking it.
"""

import math
import sys

import torch

from .agreement_journal import AgreementJournal
from .copy_head_factorial import binary64_rounding_probe, f32_bits_to_f64, f64_to_f32_bits
from .gpt2_head_slice import GPT2HeadSlicePoint, run_gpt2_head_slice


MIN_DOSE = 2.0 ** -20


def _bits(value):
    if (type(value) is not torch.Tensor or value.dtype != torch.float32 or value.device.type != "cpu"
            or value.numel() == 0 or sys.byteorder != "little"):
        raise ValueError("expected nonempty little-endian CPU float32 tensor")
    return value.detach().view(torch.int32).contiguous().numpy().tobytes()


def _values(value):
    return tuple(f32_bits_to_f64(int(x) & 0xffffffff)
                 for x in value.detach().view(torch.int32).contiguous().reshape(-1).tolist())


def _distance(left, right):
    if left.shape != right.shape:
        raise ValueError("head-vector shapes differ")
    result = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(_values(left), _values(right), strict=True)))
    if not math.isfinite(result):
        raise ValueError("nonfinite head-vector distance")
    return result


def _replacement(target, donor, q):
    distance = _distance(target, donor)
    if type(q) is not float or not math.isfinite(q) or q < MIN_DOSE or distance <= 0 or q > distance:
        raise ValueError("dose is nonfinite, too small, or would amplify the donor displacement")
    fraction = q / distance
    encoded = [f64_to_f32_bits(float(a + fraction * (b - a)))
               for a, b in zip(_values(target), _values(donor), strict=True)]
    # Construct from integer encodings to preserve exact float32 bits.
    signed = [bits if bits < 0x80000000 else bits - 0x100000000 for bits in encoded]
    result = torch.tensor(signed, dtype=torch.int32).view(torch.float32).reshape(target.shape)
    actual = _distance(target, result)
    if actual == 0 or abs(actual - q) > 1e-5 * q:
        raise ValueError("delivered dose is zero or outside relative tolerance")
    return result, distance, actual


def _rank_heads(values):
    if (type(values) is not dict or len(values) < 2 or any(type(head) is not int or head < 0 for head in values)
            or any(type(v) is not list or not v or any(type(x) is not float or not math.isfinite(x) for x in v)
                   for v in values.values()) or len({len(v) for v in values.values()}) != 1):
        raise ValueError("head values must contain equally sized finite lists for distinct heads")
    ranked = [{"head": head, "values": values[head], "mean_D": math.fsum(values[head]) / len(values[head])}
              for head in values]
    return sorted(ranked, key=lambda item: (-item["mean_D"], item["head"]))


def _margin(logits, pair):
    values = _values(logits)
    return values[pair[0]] - values[pair[1]]


def _slice(full, head, heads):
    width = full.shape[-1] // heads
    return full[0, -1, head * width:(head + 1) * width]


def _evaluation_rule(details, candidate, comparator):
    indexed = {(item["head"], item["target"]): item for item in details}
    if len(indexed) != 32 or set(indexed) != {(h, i) for h in (candidate, comparator) for i in range(16)}:
        raise ValueError("evaluation requires both heads on all 16 fixed targets")
    candidate_values = [indexed[candidate, i]["D"] for i in range(16)]
    control_values = [indexed[comparator, i]["D"] for i in range(16)]
    correct = [indexed[candidate, i]["sign"] * indexed[candidate, i]["clean_margin"] > 0 for i in range(16)]
    frames = []
    for frame in range(4):
        indices = range(4 * frame, 4 * frame + 4)
        mean = math.fsum(candidate_values[i] for i in indices) / 4
        control_abs = math.fsum(abs(control_values[i]) for i in indices) / 4
        frames.append({"frame": frame, "correct": sum(correct[i] for i in indices),
                       "candidate_mean_D": mean, "comparator_mean_abs_D": control_abs,
                       "criterion": mean > 0 and mean > control_abs})
    mean = math.fsum(candidate_values) / 16
    control_abs = math.fsum(abs(value) for value in control_values) / 16
    baseline = sum(correct) >= 12 and all(frame["correct"] >= 3 for frame in frames)
    criterion = (mean >= 0.05 and mean > control_abs and sum(value > 0 for value in candidate_values) >= 10
                 and sum(frame["criterion"] for frame in frames) >= 3)
    return {"baseline_correct": sum(correct), "baseline_adequate": baseline,
            "candidate_mean_D": mean, "comparator_mean_abs_D": control_abs,
            "candidate_positive": sum(value > 0 for value in candidate_values),
            "candidate_zero": sum(value == 0 for value in candidate_values),
            "candidate_negative": sum(value < 0 for value in candidate_values), "frames": frames,
            "criterion": criterion, "status": "baseline_limited" if not baseline else
                       ("passes_fixed_rule" if criterion else "does_not_pass_fixed_rule")}


def _validate_rows(model, rows, heads, present_ids):
    if (type(rows) not in (list, tuple) or len(rows) != 16
            or any(type(row) not in (list, tuple) or not row or len(row) > 12
                   or any(type(i) is not int or not 0 <= i < model.config.vocab_size for i in row) for row in rows)
            or type(heads) is not tuple or len(heads) < 2 or len(set(heads)) != len(heads)
            or any(type(h) is not int or not 0 <= h < model.config.n_head for h in heads)
            or type(present_ids) is not tuple or len(present_ids) != 2
            or any(type(i) is not int or not 0 <= i < model.config.vocab_size for i in present_ids)
            or present_ids[0] == present_ids[1]):
        raise ValueError("invalid rows, heads or scored-token IDs")
    for start in range(0, 16, 4):
        group = rows[start:start + 4]
        if len({len(row) for row in group}) != 1:
            raise ValueError("quartet row lengths differ")
        variable = [i for i in range(len(group[0])) if len({row[i] for row in group}) > 1]
        if len(variable) != 1 or len({row[variable[0]] for row in group}) != 4:
            raise ValueError("quartet must vary at exactly one distinct subject-token position")


def _run(model, rows, directory, *, heads, present_ids, secondary_ids, metadata, phase):
    _validate_rows(model, rows, heads, present_ids)
    if phase not in ("development", "evaluation") or type(metadata) is not dict:
        raise ValueError("invalid phase or run metadata")
    if phase == "evaluation" and len(heads) != 2:
        raise ValueError("evaluation requires candidate and comparator heads")
    if (secondary_ids is not None and (type(secondary_ids) is not tuple or len(secondary_ids) != 2
            or any(type(i) is not int or not 0 <= i < model.config.vocab_size for i in secondary_ids)
            or secondary_ids[0] == secondary_ids[1])):
        raise ValueError("invalid secondary scored-token IDs")
    journal = AgreementJournal(directory, {"phase": phase, **metadata}, max_total_bytes=128 * 1024 * 1024,
                               max_forward_bytes=1024 * 1024, max_events=1000,
                               max_event_log_bytes=8 * 1024 * 1024)
    count = 0

    def call(row_index, head, kind, replacement=None, *, donor=None, q=None, distance=None, actual=None):
        nonlocal count
        binary64_rounding_probe()
        identity = {"kind": kind, "row": row_index, "head": head, "donor": donor, "ordinal": count}
        journal.record({"type": "forward_begin", "binary64_before": "nearest-even-probes-passed", **identity}, {})
        ids = torch.tensor([rows[row_index]], dtype=torch.long)
        point = GPT2HeadSlicePoint(layer=3, token=len(rows[row_index]) - 1, head=head)
        result = run_gpt2_head_slice(model, ids, point=point, run_id=f"{phase}-{count}", replacement=replacement,
                                     max_input_bytes=1024, max_tensor_bytes=64 * 1024,
                                     max_capture_bytes=192 * 1024, max_source_bytes=4096,
                                     max_logits_bytes=4 * 1024 * 1024)
        last = result.logits[0, -1].detach().contiguous()
        tensors = {"before": (tuple(result.before.shape), _bits(result.before)),
                   "after": (tuple(result.after.shape), _bits(result.after)),
                   "projection_output": (tuple(result.projection_output.shape), _bits(result.projection_output)),
                   "last_logits": (tuple(last.shape), _bits(last))}
        if replacement is not None:
            tensors["replacement"] = (tuple(replacement.shape), _bits(replacement))
        journal.record({"type": "forward_complete", **identity, "point": result.metadata.to_dict(),
                        "requested_q": q, "donor_distance": distance, "realized_q": actual}, tensors)
        count += 1
        # Retain the returned witnesses before testing post-forward arithmetic.
        # A failed probe stops the attempt and leaves no success postcheck.
        binary64_rounding_probe()
        journal.record({"type": "numeric_postcheck", "ordinal": count - 1,
                        "binary64_after": "nearest-even-probes-passed"}, {})
        return result.before, result.after, last

    try:
        observations = []
        for index in range(16):
            before, after, last = call(index, 0, "observe")
            if _bits(before) != _bits(after):
                raise RuntimeError("observe changed projection-input bits")
            observations.append((before, last))
        binary64_rounding_probe()
        targets = [i for i in range(16) if phase == "evaluation" or i % 4 < 2]
        distances = []
        for head in heads:
            for target in targets:
                group, local = (target // 4) * 4, target % 4
                for name, donor in (("opposite", group + (local ^ 1)), ("same", group + (local ^ 2))):
                    d = _distance(_slice(observations[target][0], head, model.config.n_head),
                                  _slice(observations[donor][0], head, model.config.n_head))
                    distances.append({"head": head, "target": target, "kind": name, "donor": donor, "l2": d})
        q = min(item["l2"] for item in distances) / 2.0
        if not math.isfinite(q) or q < MIN_DOSE:
            raise ValueError("common dose is nonfinite or below the fixed floor")
        binary64_rounding_probe()
        journal.record({"type": "dose_plan", "q": q, "distances": distances}, {})
        self_targets = targets if phase == "evaluation" else targets[:1]
        for head in heads:
            for target in self_targets:
                selected = _slice(observations[target][0], head, model.config.n_head)
                before, after, last = call(target, head, "self", selected)
                if (_bits(before) != _bits(observations[target][0]) or _bits(after) != _bits(before)
                        or _bits(last) != _bits(observations[target][1])):
                    raise RuntimeError("self patch or self logits differ from observation")
        effects = {head: [] for head in heads}
        details = []
        for head in heads:
            for target in targets:
                group, local = (target // 4) * 4, target % 4
                margins, secondary = {}, {}
                doses = {}
                for name, donor in (("opposite", group + (local ^ 1)), ("same", group + (local ^ 2))):
                    binary64_rounding_probe()
                    source = _slice(observations[target][0], head, model.config.n_head)
                    other = _slice(observations[donor][0], head, model.config.n_head)
                    replacement, d, actual = _replacement(source, other, q)
                    before, after, last = call(target, head, name, replacement, donor=donor,
                                               q=q, distance=d, actual=actual)
                    if (_bits(before) != _bits(observations[target][0])
                            or _bits(_slice(after, head, model.config.n_head)) != _bits(replacement)):
                        raise RuntimeError("patch source or exact landing differs from the plan")
                    expected = before.view(torch.int32).clone()
                    width = expected.shape[-1] // model.config.n_head
                    expected[0, -1, head * width:(head + 1) * width] = replacement.view(torch.int32)
                    if not torch.equal(after.view(torch.int32), expected):
                        raise RuntimeError("patch changed unselected projection-input bits")
                    margins[name] = _margin(last, present_ids)
                    if secondary_ids is not None:
                        secondary[name] = _margin(last, secondary_ids)
                    doses[name] = actual
                sign = 1 if local % 2 == 0 else -1
                contrast = float(sign * (margins["same"] - margins["opposite"]))
                binary64_rounding_probe()
                effects[head].append(contrast)
                details.append({"head": head, "target": target, "sign": sign,
                                "clean_margin": _margin(observations[target][1], present_ids),
                                "opposite_margin": margins["opposite"], "same_margin": margins["same"],
                                "D": contrast, "realized_doses": doses,
                                "secondary": (None if secondary_ids is None else
                                    {"clean_margin": _margin(observations[target][1], secondary_ids),
                                     "opposite_margin": secondary["opposite"], "same_margin": secondary["same"],
                                     "D": float(sign * (secondary["same"] - secondary["opposite"]))})})
        for index in range(16):
            before, after, last = call(index, 0, "repeat")
            if (_bits(before) != _bits(observations[index][0]) or _bits(after) != _bits(before)
                    or _bits(last) != _bits(observations[index][1])):
                raise RuntimeError("repeat differs from clean observation")
        binary64_rounding_probe()
        ranking = _rank_heads(effects)
        summary = {"phase": phase, "forward_count": count, "q": q, "ranking": ranking,
                   "candidate": ranking[0]["head"] if phase == "development" else heads[0],
                   "comparator": ranking[1]["head"] if phase == "development" else heads[1],
                   "details": details}
        if phase == "evaluation":
            summary["evaluation"] = _evaluation_rule(details, heads[0], heads[1])
        binary64_rounding_probe()
        journal.record({"type": "summary", **summary}, {})
        journal.finish("complete", summary)
        return summary
    except BaseException as error:
        try:
            journal.finish("failed", {"error_type": type(error).__name__, "completed_forwards": count})
        except BaseException as cleanup:
            if hasattr(error, "add_note"):
                error.add_note(f"Could not publish failure terminal: {type(cleanup).__name__}")
        raise


def run_development(model, rows, directory, *, heads=tuple(range(12)), present_ids=(318, 389),
                    secondary_ids=(373, 547), metadata):
    """Run the supplied fixed public development quartets with one owned CPU model."""
    return _run(model, rows, directory, heads=heads, present_ids=present_ids, secondary_ids=secondary_ids, metadata=metadata,
                phase="development")


def run_evaluation(model, rows, directory, *, candidate, comparator, metadata,
                   present_ids=(318, 389), secondary_ids=(373, 547)):
    """Run fixed evaluation quartets after external source/exposure admission."""
    return _run(model, rows, directory, heads=(candidate, comparator), present_ids=present_ids,
                secondary_ids=secondary_ids, metadata=metadata, phase="evaluation")
