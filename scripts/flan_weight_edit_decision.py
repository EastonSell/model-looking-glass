"""Fixed scientific decisions and model-free substitution checks."""

import math


class SubstitutionMismatch(RuntimeError):
    pass


def primary_outcome(cases, *, calibration_eligible):
    if (type(calibration_eligible) is not bool or len(cases) != 4
            or any(type(row.get("baseline_eligible")) is not bool
                   or row.get("edited_label") not in (None, "positive", "negative") for row in cases)):
        raise ValueError("four complete target observations are required")
    eligible = [row for row in cases if row["baseline_eligible"]]
    reversed_count = sum(row["edited_label"] == "negative" for row in eligible)
    if not calibration_eligible or len(eligible) < 3:
        status = "ineligible"
    else:
        status = "positive" if reversed_count >= 3 else "negative"
    return {"status": status, "baseline_eligible": len(eligible), "negative_labels": reversed_count}


def evaluation_balanced(primary, control):
    if not primary or len(primary) != len(control):
        raise ValueError("paired eligible target effects are required")
    for reference, observed in zip(primary, control):
        if (reference.get("strict_label") not in ("positive", "negative")
                or observed.get("strict_label") != reference["strict_label"]):
            return False
        for name in ("margin_gain", "target_gain"):
            a, b = reference.get(name), observed.get(name)
            if type(a) not in (float, int) or type(b) not in (float, int) or not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("effects must be finite numbers")
            if abs(math.fsum((float(a), -float(b)))) > max(0.10, 0.10 * abs(float(a))):
                return False
    return True


def check_substitution(target, donor):
    if set(target["label_spans"]) != {"positive", "negative"} or set(donor["label_spans"]) != {"positive", "negative"}:
        raise ValueError("both label sequences are required")

    def compare(left, right):
        left_prefix = left["metadata"]["decoder_prefix"]
        right_prefix = right["metadata"]["decoder_prefix"]
        if left_prefix != right_prefix:
            raise ValueError("label decoder prefixes differ")
        if 1627 not in left_prefix and left["sha256"] != right["sha256"]:
            raise SubstitutionMismatch("same-prefix raw logits differ")

    for label in ("positive", "negative"):
        compare(target["label_spans"][label], donor["label_spans"][label])
    if (len(target["greedy_ids"]) != len(target["greedy_spans"])
            or len(donor["greedy_ids"]) != len(donor["greedy_spans"])):
        raise ValueError("greedy IDs and recorded spans differ")
    for index, (left, right) in enumerate(zip(target["greedy_spans"], donor["greedy_spans"])):
        a, b = left["metadata"]["decoder_prefix"], right["metadata"]["decoder_prefix"]
        if a != b or 1627 in a:
            break
        compare(left, right)
        if target["greedy_ids"][index] != donor["greedy_ids"][index]:
            raise SubstitutionMismatch("same logits have different recorded greedy IDs")
    return True
