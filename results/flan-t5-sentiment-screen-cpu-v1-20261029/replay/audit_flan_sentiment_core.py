"""Model-free checks for a fixed FLAN sentiment screen's saved observations.

This independent core reconstructs decoding and the feasibility rule; it does
not establish source identity, process execution, tokenizer identity, or model
state. A caller must bind those separately and supply the pinned tokenizer.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
import unicodedata

import torch

INSTRUCTIONS = ("Classify the sentiment of this review as positive or negative:",
                "Is this review positive or negative?")
SUBJECTS = ("book", "movie", "meal", "service")
PAIRS = (("wonderful", "awful"), ("excellent", "terrible"),
         ("amazing", "horrible"), ("enjoyable", "disappointing"))


def require_denormals():
    """Check the auditing thread rather than assuming its floating-point mode."""
    for fmt, positive, negative in (("<f", bytes.fromhex("01000000"), bytes.fromhex("01000080")),
                                     ("<d", bytes.fromhex("0100000000000000"),
                                      bytes.fromhex("0100000000000080"))):
        a, b = struct.unpack(fmt, positive)[0], struct.unpack(fmt, negative)[0]
        if not (a > 0. and b < 0. and struct.pack(fmt, a * 1.) == positive
                and struct.pack(fmt, b * 1.) == negative):
            raise ValueError("auditor subnormal preservation is unavailable")


def fixed_cases():
    result = []
    for instruction_index, instruction in enumerate(INSTRUCTIONS):
        for subject in SUBJECTS:
            for pair_index, adjectives in enumerate(PAIRS):
                for side, adjective in enumerate(adjectives):
                    result.append({"index": len(result), "instruction_index": instruction_index,
                                   "subject": subject, "pair_index": pair_index, "side": side,
                                   "adjective": adjective, "expected": ("positive", "negative")[side],
                                   "text": f"{instruction} The {subject} was {adjective}."})
    return result


def read_tensor(root, ref, *, shape):
    """Require the named byte payload and exact finite little-endian float32 values."""
    require_denormals()
    if (type(ref) is not dict or ref.get("dtype") != "<f4" or type(ref.get("shape")) is not list
            or any(type(x) is not int for x in ref["shape"]) or ref["shape"] != list(shape)
            or type(ref.get("path")) is not str or type(ref.get("bytes")) is not int
            or type(ref.get("sha256")) is not str):
        raise ValueError("tensor reference differs")
    relative = Path(ref["path"])
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "tensors":
        raise ValueError("tensor path differs")
    if not (relative.parts[1].endswith(".f32") and relative.parts[1][:-4].isdigit()):
        raise ValueError("tensor filename differs")
    raw = (Path(root) / relative).read_bytes()
    count = math.prod(shape)
    if (len(raw) != 4 * count or len(raw) != ref["bytes"]
            or hashlib.sha256(raw).hexdigest() != ref["sha256"]):
        raise ValueError("tensor size or hash differs")
    values = tuple(value[0] for value in struct.iter_unpack("<f", raw))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("tensor is nonfinite")
    return values


def strict_output(ids, tokenizer, expected):
    if (type(ids) is not list or not 1 <= len(ids) <= 4
            or any(type(value) is not int or value < 0 for value in ids)):
        raise ValueError("generated token IDs differ")
    eos_positions = [index for index, value in enumerate(ids) if value == 1]
    if eos_positions and eos_positions != [len(ids) - 1]:
        raise ValueError("generation continued after EOS")
    terminated = bool(eos_positions)
    content = ids[:-1] if terminated else ids
    decoded = tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    normalized = unicodedata.normalize("NFC", decoded).strip().casefold()
    no_specials = all(token not in tokenizer.all_special_ids for token in content)
    return {"decoded": decoded, "normalized": normalized, "terminated": terminated,
            "no_content_specials": no_specials,
            "correct": terminated and no_specials and normalized == expected}


def check_greedy(generated_ids, prefixes, logit_vectors):
    if not (type(generated_ids) is list and 1 <= len(generated_ids) <= 4
            and len(prefixes) == len(logit_vectors) == len(generated_ids)):
        raise ValueError("generation steps differ")
    for index, (emitted, prefix, logits) in enumerate(zip(generated_ids, prefixes, logit_vectors)):
        if (type(emitted) is not int or prefix != [0, *generated_ids[:index]] or not logits
                or any(not math.isfinite(x) for x in logits)):
            raise ValueError("generation prefix or logits differ")
        # Python max resolves ties by first index, which is the smallest token ID.
        wanted = max(range(len(logits)), key=logits.__getitem__)
        if emitted != wanted or (emitted == 1 and index != len(generated_ids) - 1):
            raise ValueError("generation is not raw-logit greedy decoding")
    if generated_ids[-1] != 1 and len(generated_ids) != 4:
        raise ValueError("generation stopped before EOS or cap")


def check_teacher_scores(logits, targets, saved_values, saved_score):
    """Replay the producer's CPU float32 log-softmax and diagnose float64 drift."""
    require_denormals()
    if (type(targets) is not list or len(targets) < 2 or targets[-1] != 1
            or len(logits) != len(targets) or len(saved_values) != len(targets)
            or any(type(x) is not int for x in targets)):
        raise ValueError("target IDs or score count differ")
    if any(len(row) != len(logits[0]) for row in logits):
        raise ValueError("teacher logit row widths differ")
    matrix = torch.tensor(logits, dtype=torch.float32, device="cpu")
    replay = torch.log_softmax(matrix, dim=-1)
    differences = []
    for index, (row, token, saved) in enumerate(zip(logits, targets, saved_values)):
        if (not row or not 0 <= token < len(row) or type(saved) not in (int, float)
                or not math.isfinite(saved) or any(not math.isfinite(x) for x in row)):
            raise ValueError("teacher logits or target score differ")
        expected = float(replay[index, token].item())
        if saved != expected:
            raise ValueError("teacher float32 log-probability differs")
        maximum = max(row)
        high_precision = row[token] - maximum - math.log(math.fsum(math.exp(x - maximum) for x in row))
        differences.append(abs(saved - high_precision))
    if type(saved_score) not in (int, float) or saved_score != math.fsum(saved_values):
        raise ValueError("teacher sequence reduction differs")
    require_denormals()
    return max(differences)


def feasibility(correct):
    if type(correct) is not list or len(correct) != 64 or any(type(value) is not bool for value in correct):
        raise ValueError("expected 64 Boolean results")
    cases = fixed_cases()
    groups = {"instruction_pairs": {str(i): 0 for i in range(2)},
              "subject_pairs": {name: 0 for name in SUBJECTS},
              "adjective_pairs": {str(i): 0 for i in range(4)}}
    completed = 0
    for i in range(0, 64, 2):
        if correct[i] and correct[i + 1]:
            c = cases[i]
            completed += 1
            groups["instruction_pairs"][str(c["instruction_index"])] += 1
            groups["subject_pairs"][c["subject"]] += 1
            groups["adjective_pairs"][str(c["pair_index"])] += 1
    labels = {label: sum(correct[i] for i, case in enumerate(cases) if case["expected"] == label)
              for label in ("positive", "negative")}
    feasible = (completed >= 24 and all(n >= 10 for n in groups["instruction_pairs"].values())
                and all(n >= 4 for n in groups["subject_pairs"].values())
                and all(n >= 4 for n in groups["adjective_pairs"].values()))
    return {"cases": 64, "complete_pairs": completed, "pair_denominator": 32, **groups,
            "correct_by_label": labels, "label_denominator": 32, "feasible": feasible}
