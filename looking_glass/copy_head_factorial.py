"""Bit-defined matched-dose interventions for the public copying factorial."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np
import torch

from looking_glass.copy_head_intervention import HeadProjectionWitness, ProjectionWitnessError, _equal_float_bits


HEADS = (10, 2, 7)
ARMS = ("h10", "h2", "h7", "joint")
SHAPE_BYTES = 11 * 768 * 4
MIN_DOSE = 2.0 ** -20


def f32_bits_to_f64(bits: int) -> float:
    """Decode a finite IEEE binary32 bit pattern without a float32 cast."""
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError("invalid binary32 bits")
    sign, exponent, fraction = bits >> 31, (bits >> 23) & 0xff, bits & 0x7fffff
    if exponent == 0xff:
        raise ValueError("nonfinite binary32 value")
    significand = fraction if exponent == 0 else (1 << 23) | fraction
    value = math.ldexp(float(significand), -149 if exponent == 0 else exponent - 150)
    return -value if sign else value


def _round_shift_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    quotient, remainder = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    return quotient + (remainder > halfway or (remainder == halfway and quotient & 1))


def f64_to_f32_bits(value: float) -> int:
    """Round finite binary64 to binary32, nearest with ties to even."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("expected a finite binary64 value")
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    sign = (bits >> 32) & 0x80000000
    exponent, fraction = (bits >> 52) & 0x7ff, bits & ((1 << 52) - 1)
    if exponent == 0:
        return sign | _round_shift_even(fraction, 925)
    power, significand = exponent - 1023, (1 << 52) | fraction
    if power >= -126:
        rounded = _round_shift_even(significand, 29)
        if rounded == 1 << 24:
            rounded >>= 1
            power += 1
        if power > 127:
            return sign | 0x7f800000
        return sign | ((power + 127) << 23) | (rounded & 0x7fffff)
    return sign | _round_shift_even(significand, -(power + 97))


def binary64_rounding_probe() -> None:
    """Check representative native operations used by the fixed formula."""
    one, half_ulp = 1.0, math.ldexp(1.0, -53)
    next_one = math.ldexp(1.0, -52) + one
    if (one + half_ulp != one
            or next_one + half_ulp != 1.0 + math.ldexp(1.0, -51)
            or one - math.ldexp(1.0, -54) != one
            or next_one * 1.5 != 1.5 + math.ldexp(1.0, -51)
            or one / 10.0 != float.fromhex("0x1.999999999999ap-4")
            or math.sqrt(2.0) != float.fromhex("0x1.6a09e667f3bcdp+0")):
        raise RuntimeError("binary64 arithmetic probe failed")


def _bits(raw: bytes):
    if type(raw) is not bytes or len(raw) % 4:
        raise ValueError("expected binary32 bytes")
    return tuple(value[0] for value in struct.iter_unpack("<I", raw))


def _values(raw: bytes):
    return tuple(f32_bits_to_f64(value) for value in _bits(raw))


def _norm(values) -> float:
    norm = math.sqrt(math.fsum(value ** 2 for value in values))
    if not math.isfinite(norm):
        raise ValueError("nonfinite slice norm")
    return norm


def head_slice(projection_bytes: bytes, head: int) -> bytes:
    if type(projection_bytes) is not bytes or len(projection_bytes) != SHAPE_BYTES:
        raise ValueError("unexpected projection input bytes")
    if type(head) is not int or head not in HEADS:
        raise ValueError("unexpected head")
    start = (10 * 768 + 64 * head) * 4
    return projection_bytes[start:start + 64 * 4]


@dataclass(frozen=True)
class HeadDose:
    head: int
    norm: float
    fraction: float
    achieved: float
    changed_bits: int
    before: bytes
    delivered: bytes


@dataclass(frozen=True)
class FamilyDose:
    requested: float
    variants: tuple[dict[int, HeadDose], ...]


def plan_family(noop_inputs: tuple[bytes, ...]) -> FamilyDose:
    """Calculate the one fixed dose from all four no-op witnesses."""
    if type(noop_inputs) is not tuple or len(noop_inputs) != 4:
        raise ValueError("expected four no-op projection inputs")
    original = [{head: head_slice(raw, head) for head in HEADS} for raw in noop_inputs]
    norms = [{head: _norm(_values(row[head])) for head in HEADS} for row in original]
    requested = min(row[head] for row in norms for head in HEADS)
    if not math.isfinite(requested) or requested <= MIN_DOSE:
        raise ValueError("family dose is nonfinite or too small")
    variants = []
    for originals, row_norms in zip(original, norms):
        doses = {}
        for head in HEADS:
            before = originals[head]
            values = _values(before)
            fraction = requested / row_norms[head]
            scale = 1.0 - fraction
            after_bits = tuple(f64_to_f32_bits(scale * value) for value in values)
            delivered = struct.pack("<64I", *after_bits)
            after_values = _values(delivered)
            achieved = _norm(after - old for after, old in zip(after_values, values))
            if not math.isfinite(achieved) or abs(achieved - requested) > 1e-5 * requested:
                raise ValueError("realized slice dose does not match the requested dose")
            doses[head] = HeadDose(head, row_norms[head], fraction, achieved,
                                   sum(left != right for left, right in zip(_bits(before), after_bits)),
                                   before, delivered)
        variants.append(doses)
    return FamilyDose(requested, tuple(variants))


def joint_achieved(row: dict[int, HeadDose]) -> float:
    if type(row) is not dict or set(row) != set(HEADS):
        raise ValueError("expected a full variant dose")
    return _norm(after - before for head in (2, 10)
                 for after, before in zip(_values(row[head].delivered), _values(row[head].before)))


class EvidenceProjectionWitness(HeadProjectionWitness):
    """Capture bounded witnesses before rejecting nonfinite projection data."""
    def _capture_input(self, arguments):
        if type(arguments) is tuple and len(arguments) == 1:
            value = arguments[0]
            if (type(value) is torch.Tensor and value.device.type == "cpu" and value.dtype == torch.float32
                    and tuple(value.shape) == (1, self.sequence_length, self.width) and value.is_contiguous()):
                self.before = value.detach().clone().contiguous()

    def _pre(self, module, arguments):
        self._capture_input(arguments)
        return super()._pre(module, arguments)

    def _post(self, module, arguments, output):
        if (module is not self.projection or self.pre_calls != 1 or self.post_calls != 0
                or type(arguments) is not tuple or len(arguments) != 1):
            raise ProjectionWitnessError("unexpected projection post-hook call")
        actual = self._tensor(arguments[0])
        if self.after is None or not _equal_float_bits(actual, self.after):
            raise ProjectionWitnessError("projection received different input bits")
        if (type(output) is not torch.Tensor or output.device.type != "cpu" or output.dtype != torch.float32
                or tuple(output.shape) != (1, self.sequence_length, self.width) or not output.is_contiguous()):
            raise ProjectionWitnessError("unexpected projection output shape or type")
        self.output = output.detach().clone().contiguous()
        self.post_calls = 1
        return None


class MatchedProjectionWitness(EvidenceProjectionWitness):
    """Deliver precomputed exact slice bits and record projection input/output."""
    def __init__(self, projection, *, expected_before: bytes, deliveries: dict[int, bytes]):
        super().__init__(projection, sequence_length=11, width=768, heads=12, mode="noop")
        if type(expected_before) is not bytes or len(expected_before) != SHAPE_BYTES:
            raise ValueError("unexpected reference projection input")
        if (type(deliveries) is not dict or not deliveries or set(deliveries) not in
                ({10}, {2}, {7}, {2, 10})
                or any(type(head) is not int for head in deliveries)
                or any(type(raw) is not bytes or len(raw) != 64 * 4 for raw in deliveries.values())):
            raise ValueError("unexpected head deliveries")
        self.expected_before = expected_before
        self.deliveries = dict(deliveries)

    def _pre(self, module, arguments):
        if (module is not self.projection or self.pre_calls != 0
                or type(arguments) is not tuple or len(arguments) != 1):
            raise ProjectionWitnessError("unexpected projection pre-hook call")
        self._capture_input(arguments)
        value = self._tensor(arguments[0])
        raw = value.detach().contiguous().numpy().tobytes()
        if raw != self.expected_before:
            raise ProjectionWitnessError("pre-action input differs from its no-op")
        self.pre_calls = 1
        self.before = value.detach().clone().contiguous()
        delivered = value.clone().contiguous()
        expected = bytearray(raw)
        for head, data in self.deliveries.items():
            start = (10 * 768 + head * 64) * 4
            expected[start:start + len(data)] = data
            source = torch.from_numpy(np.frombuffer(data, dtype="<f4").copy())
            delivered[0, 10, 64 * head:64 * (head + 1)].copy_(source)
        self.after = delivered.detach().clone().contiguous()
        if self.after.numpy().tobytes() != bytes(expected):
            raise ProjectionWitnessError("delivered projection bits differ from the fixed slices")
        return (delivered,)


def action_order(family_index: int, variant: int) -> tuple[str, ...]:
    if (type(family_index) is not int or not 0 <= family_index < 8
            or type(variant) is not int or not 0 <= variant < 4):
        raise ValueError("family and variant are outside the fixed range")
    shift = (family_index + variant) % 4
    return ARMS[shift:] + ARMS[:shift]


def statistics(raw: bytes, target: int) -> dict:
    values = _values(raw)
    if len(values) != 50_257 or type(target) is not int or not 0 <= target < len(values):
        raise ValueError("invalid logit vector or target")
    maximum, target_value = max(values), values[target]
    logp = (target_value - maximum) - math.log(math.fsum(math.exp(value - maximum) for value in values))
    return {"target_logit": target_value, "target_log_probability": logp,
            "target_rank": 1 + sum(value > target_value for value in values),
            "top_token_id": values.index(maximum)}


def projection_diagnostics(outputs: dict[str, bytes]) -> dict:
    """Descriptive final-position c_proj output changes and joint residual."""
    if type(outputs) is not dict or set(outputs) != {"noop", *ARMS}:
        raise ValueError("expected all five projection outputs")
    vectors = {}
    for name, raw in outputs.items():
        if type(raw) is not bytes or len(raw) != SHAPE_BYTES:
            raise ValueError("unexpected projection output")
        vectors[name] = _values(raw[10 * 768 * 4:11 * 768 * 4])
    reference = vectors["noop"]
    changes = {name: _norm(after - before for after, before in zip(vectors[name], reference))
               for name in ARMS}
    residual = _norm(((joint - two) - ten) + clean for joint, two, ten, clean in
                     zip(vectors["joint"], vectors["h2"], vectors["h10"], reference))
    return {"final_output_delta_l2": changes, "joint_minus_singles_output_residual_l2": residual}


def summarize(records: list[dict]) -> dict:
    """Compute the fixed repeat/shuffle and log-probability interaction."""
    if type(records) is not list or len(records) != 224:
        raise ValueError("expected 224 records")
    families = []
    for family in range(8):
        segment = records[28 * family:28 * (family + 1)]
        expected = [(variant, action) for variant in range(4) for action in ("clean", "noop")]
        expected += [(variant, action) for variant in range(4)
                     for action in (*action_order(family, variant), "clean_after")]
        if [(row["variant"], row["action"]) for row in segment] != expected:
            raise ValueError("records do not follow the fixed family schedule")
        if any(row["group"] != family + 1 for row in segment):
            raise ValueError("records do not match the fixed family")
        rows = [{row["action"]: row for row in segment if row["variant"] == variant}
                for variant in range(4)]
        result = {"group": family + 1, "arms": {}, "comparisons": {},
                  "clean_target_ranks": [row["clean"]["score"]["target_rank"] for row in rows],
                  "clean_target_top1": [row["clean"]["score"]["top_token_id"] == row["clean"]["target_id"]
                                        for row in rows]}
        for name in ARMS:
            deltas = [row[name]["score"]["target_log_probability"]
                      - row["clean"]["score"]["target_log_probability"] for row in rows]
            shuffled = math.fsum(deltas[1:]) / 3
            result["arms"][name] = {"case_deltas": deltas, "repeat_delta": deltas[0],
                                    "shuffle_mean_delta": shuffled, "family_contrast": deltas[0] - shuffled,
                                    "target_ranks": [row[name]["score"]["target_rank"] for row in rows],
                                    "target_top1": [row[name]["score"]["top_token_id"] == row[name]["target_id"]
                                                    for row in rows]}
        d = {name: result["arms"][name]["family_contrast"] for name in ARMS}
        result["case_interactions"] = [
            ((row["joint"]["score"]["target_log_probability"]
              - row["h2"]["score"]["target_log_probability"])
             - row["h10"]["score"]["target_log_probability"])
            + row["clean"]["score"]["target_log_probability"] for row in rows]
        result["interaction"] = (d["joint"] - d["h2"]) - d["h10"]
        result["comparisons"] = {"h10_minus_h2": d["h10"] - d["h2"],
                                 "h10_minus_h7": d["h10"] - d["h7"],
                                 "h2_minus_h7": d["h2"] - d["h7"]}
        families.append(result)
    return {"families": families,
            "mean_contrast_by_arm": {name: math.fsum(row["arms"][name]["family_contrast"]
                                                       for row in families) / 8 for name in ARMS},
            "mean_repeat_delta_by_arm": {name: math.fsum(row["arms"][name]["repeat_delta"]
                                                           for row in families) / 8 for name in ARMS},
            "mean_shuffle_delta_by_arm": {name: math.fsum(row["arms"][name]["shuffle_mean_delta"]
                                                            for row in families) / 8 for name in ARMS},
            "mean_interaction": math.fsum(row["interaction"] for row in families) / 8,
            "mean_comparisons": {key: math.fsum(row["comparisons"][key] for row in families) / 8
                                 for key in ("h10_minus_h2", "h10_minus_h7", "h2_minus_h7")},
            "top1_clean": {"repeated": sum(row["clean_target_top1"][0] for row in families),
                           "shuffled": sum(sum(row["clean_target_top1"][1:]) for row in families)},
            "top1_by_arm": {name: {"repeated": sum(row["arms"][name]["target_top1"][0] for row in families),
                                   "shuffled": sum(sum(row["arms"][name]["target_top1"][1:]) for row in families)}
                            for name in ARMS}}
