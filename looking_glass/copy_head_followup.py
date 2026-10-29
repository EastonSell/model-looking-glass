"""Fixed head and token-position controls for the public copying follow-up."""

from __future__ import annotations

import hashlib
import math

import numpy as np


# Names and arm order are fixed before any new public case is assembled.
ARMS = (("h10_final", 10, 10), ("h2_final", 2, 10), ("h7_final", 7, 10),
        ("h3_final", 3, 10), ("h1_final", 1, 10), ("h10_target", 10, 5))
ARM_MAP = {name: (head, position) for name, head, position in ARMS}
CONTROL_SEEDS = (20261030, 20261031)
CONTROL_HASHES = ("c8c677ecc975438c531a4924bde114be5ce725033bb74223bf4c3d8b4a94e89e",
                  "1a0b56b4d8e52e240475c432b3cdb636daf3690f34a0187b38a9c963441cf329")


def selected_controls():
    pool = [head for head in range(12) if head not in (2, 7, 10)]
    selected = []
    for seed in CONTROL_SEEDS:
        digest = hashlib.sha256(b"MLG-copy-expanded-control-v1\0" + seed.to_bytes(8, "big")).digest()
        selected.append((digest.hex(), pool.pop(int.from_bytes(digest, "big") % len(pool))))
    return tuple(selected)


def action_order(family_index: int, variant: int):
    if (type(family_index) is not int or not 0 <= family_index < 8
            or type(variant) is not int or not 0 <= variant < 4):
        raise ValueError("family and variant must be in their fixed ranges")
    if selected_controls() != ((CONTROL_HASHES[0], 3), (CONTROL_HASHES[1], 1)):
        raise RuntimeError("fixed control selection changed")
    shift = (family_index + 2 * variant) % len(ARMS)
    names = tuple(name for name, _, _ in ARMS)
    return ("clean", "noop", *names[shift:], *names[:shift], "clean_after")


def arm_witness(action, payload):
    if action not in ARM_MAP:
        return None
    head, position = ARM_MAP[action]
    before = np.frombuffer(payload["cproj_before"], dtype="<f4").reshape(1, 11, 768)
    old_bits = np.frombuffer(payload["cproj_before"], dtype="<u4").reshape(1, 11, 768)
    new_bits = np.frombuffer(payload["cproj_after"], dtype="<u4").reshape(1, 11, 768)
    start, end = 64 * head, 64 * (head + 1)
    values = before[0, position, start:end].astype("<f8")
    return {"layer": 3, "head": head, "position": position, "channels": [start, end],
            "pre_head_l2": math.sqrt(math.fsum(float(value) ** 2 for value in values)),
            "changed_float32_bit_patterns": int(np.count_nonzero(
                old_bits[0, position, start:end] != new_bits[0, position, start:end]))}


def summarize(records):
    if len(records) != 32 * 9:
        raise ValueError("summary requires all 288 fixed forward records")
    families = []
    for group in range(8):
        rows = []
        for variant in range(4):
            offset = (group * 4 + variant) * 9
            case = records[offset:offset + 9]
            if [row["action"] for row in case] != list(action_order(group, variant)):
                raise ValueError("records do not follow the fixed rotated arm order")
            rows.append({row["action"]: row for row in case})
        result = {"group": group + 1, "arms": {}, "candidate_minus_arm": {},
                  "clean_target_ranks": [row["clean"]["score"]["target_rank"] for row in rows],
                  "clean_target_top1": [row["clean"]["score"]["top_token_id"] == row["clean"]["target_id"]
                                        for row in rows]}
        for name, _, _ in ARMS:
            deltas = [row[name]["score"]["target_log_probability"]
                      - row["clean"]["score"]["target_log_probability"] for row in rows]
            shuffle_mean = math.fsum(deltas[1:]) / 3
            result["arms"][name] = {"case_deltas": deltas,
                                    "repeat_delta": deltas[0], "shuffle_mean_delta": shuffle_mean,
                                    "family_contrast": deltas[0] - shuffle_mean,
                                    "target_ranks": [row[name]["score"]["target_rank"] for row in rows],
                                    "target_top1": [row[name]["score"]["top_token_id"] == row[name]["target_id"]
                                                    for row in rows]}
        candidate = result["arms"]["h10_final"]["family_contrast"]
        for name, _, _ in ARMS[1:]:
            result["candidate_minus_arm"][name] = candidate - result["arms"][name]["family_contrast"]
        families.append(result)
    return {"families": families,
            "top1_clean": {"repeated": sum(row["clean_target_top1"][0] for row in families),
                           "shuffled": sum(sum(row["clean_target_top1"][1:]) for row in families)},
            "mean_contrast_by_arm": {name: math.fsum(
                row["arms"][name]["family_contrast"] for row in families) / 8 for name, _, _ in ARMS},
            "mean_repeat_delta_by_arm": {name: math.fsum(
                row["arms"][name]["repeat_delta"] for row in families) / 8 for name, _, _ in ARMS},
            "mean_shuffle_delta_by_arm": {name: math.fsum(
                row["arms"][name]["shuffle_mean_delta"] for row in families) / 8 for name, _, _ in ARMS},
            "top1_by_arm": {name: {"repeated": sum(row["arms"][name]["target_top1"][0] for row in families),
                                   "shuffled": sum(sum(row["arms"][name]["target_top1"][1:]) for row in families)}
                            for name, _, _ in ARMS},
            "mean_candidate_minus_arm": {name: math.fsum(
                row["candidate_minus_arm"][name] for row in families) / 8 for name, _, _ in ARMS[1:]}}
