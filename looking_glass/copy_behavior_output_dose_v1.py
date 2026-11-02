"""Family-matched projection-output plans, without model forwards."""
from __future__ import annotations

import math
import struct

from looking_glass.copy_head_factorial import (
    SHAPE_BYTES, _bits, _norm, _values, f64_to_f32_bits, head_slice,
)

HEADS = (2, 10, 7)
FLOOR = 2.0 ** -8
TOLERANCE = 1e-4
WIDTH = 768
HEAD_WIDTH = 64
WEIGHT_BYTES = HEAD_WIDTH * WIDTH * 4


def weights(raw: dict[int, bytes]) -> dict[int, tuple[float, ...]]:
    if set(raw) != set(HEADS) or any(type(raw[h]) is not bytes or len(raw[h]) != WEIGHT_BYTES for h in HEADS):
        raise ValueError("expected three exact 64 by 768 float32 head-row slices")
    return {h: _values(raw[h]) for h in HEADS}


def project(x, w):
    if len(x) != HEAD_WIDTH or len(w) != HEAD_WIDTH * WIDTH:
        raise ValueError("incorrect row-major projection geometry")
    result = tuple(math.fsum(x[j] * w[j * WIDTH + k] for j in range(HEAD_WIDTH)) for k in range(WIDTH))
    if any(not math.isfinite(z) for z in result):
        raise ValueError("nonfinite projection")
    return result


def plan_family(noop_inputs: tuple[bytes, ...], weight_slices: dict[int, bytes]) -> dict:
    if type(noop_inputs) is not tuple or len(noop_inputs) != 8:
        raise ValueError("expected eight no-op inputs")
    for raw in noop_inputs:
        if type(raw) is not bytes or len(raw) != SHAPE_BYTES:
            raise ValueError("invalid no-op input witness")
        _values(raw)
    matrix = weights(weight_slices)
    original = [{h: head_slice(raw, h) for h in HEADS} for raw in noop_inputs]
    values = [{h: _values(row[h]) for h in HEADS} for row in original]
    projected = [{h: _norm(project(row[h], matrix[h])) for h in HEADS} for row in values]
    q = min(projected[v][h] for v in range(8) for h in HEADS)
    if not math.isfinite(q):
        raise ValueError("nonfinite family dose")
    minima = [(v, h) for v in range(8) for h in HEADS if projected[v][h] == q]
    result = {"status": "dose-ineligible", "reason": "dose floor", "q_out": q,
              "source_output_norms": projected, "minima": minima, "plans": None}
    if q <= FLOOR:
        return result
    plans, predicted = [], []
    for variant in range(8):
        row = {}
        for head in HEADS:
            source = values[variant][head]
            fraction = q / projected[variant][head]
            bits = tuple(f64_to_f32_bits((1.0 - fraction) * value) for value in source)
            delivered = struct.pack("<64I", *bits)
            after = _values(delivered)
            delta = tuple(b - a for a, b in zip(source, after))
            projected_delta = project(delta, matrix[head])
            p = _norm(projected_delta)
            before_bits = _bits(original[variant][head])
            row[head] = {"fraction": fraction, "source": original[variant][head],
                         "delivered": delivered, "input_removed_l2": _norm(delta),
                         "predicted_output_l2": p, "predicted_vector": projected_delta,
                         "changed_elements": sum(a != b for a, b in zip(before_bits, bits)),
                         "changed_bits": sum((a ^ b).bit_count() for a, b in zip(before_bits, bits))}
            predicted.append(p)
        plans.append(row)
    valid = (all(abs(value - q) <= TOLERANCE * q for value in predicted)
             and max(predicted) - min(predicted) <= TOLERANCE * q)
    result.update({"status": "eligible" if valid else "dose-ineligible",
                   "reason": None if valid else "predicted rounded dose mismatch", "plans": plans})
    return result


def inspect_actual(noop_outputs: tuple[bytes, ...], arms: tuple[dict[int, bytes], ...], plan: dict) -> dict:
    if plan.get("status") != "eligible" or len(noop_outputs) != 8 or len(arms) != 8:
        raise ValueError("actual-dose assessment requires eight eligible output arms")
    q = plan["q_out"]
    norms, vectors, rows, residuals = [], [], [], []
    token_start = 10 * WIDTH * 4
    for variant, (noop, row) in enumerate(zip(noop_outputs, arms)):
        if type(noop) is not bytes or len(noop) != SHAPE_BYTES or set(row) != set(HEADS):
            raise ValueError("invalid output witness")
        before = _values(noop)[10 * WIDTH:]
        vec, desc = {}, {}
        for head in HEADS:
            raw = row[head]
            if type(raw) is not bytes or len(raw) != SHAPE_BYTES:
                raise ValueError("invalid output witness")
            if raw[:token_start] != noop[:token_start]:
                raise ValueError("an untouched token-position output changed")
            after = _values(raw)[10 * WIDTH:]
            delta = tuple(b - a for a, b in zip(before, after))
            n = _norm(delta)
            predicted = plan["plans"][variant][head]["predicted_vector"]
            residual = _norm(a - b for a, b in zip(delta, predicted))
            vec[head] = delta
            norms.append(n)
            residuals.append(residual)
            desc[str(head)] = {"actual_output_l2": n, "relative_error": abs(n - q) / q,
                               "predicted_vector_residual_l2": residual}
        vectors.append(vec)
        rows.append(desc)
    pair_cosines = []
    for variant, vec in enumerate(vectors):
        row = {}
        for first, second in ((2, 10), (2, 7), (10, 7)):
            a, b = vec[first], vec[second]
            na, nb = _norm(a), _norm(b)
            row[f"{first}-{second}"] = None if na <= FLOOR or nb <= FLOOR else math.fsum(x * y for x, y in zip(a, b)) / (na * nb)
        pair_cosines.append(row)
    valid = (all(abs(n - q) <= TOLERANCE * q for n in norms)
             and all(residual <= TOLERANCE * q for residual in residuals)
             and max(norms) - min(norms) <= TOLERANCE * q)
    return {"status": "eligible" if valid else "dose-ineligible",
            "reason": None if valid else "actual output dose mismatch",
            "q_out": q, "variants": rows, "pair_cosines": pair_cosines,
            "actual_range": max(norms) - min(norms)}
