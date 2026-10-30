"""Compare explicitly paired ViT trace coordinates, without causal inference."""

import math
from types import SimpleNamespace

import torch

from .bert_matched_numeric import f32_bits_to_f64
from .localization import compare_traces
from .trace_schema import GenerationContext, RunIdentity, TraceEvent
from .vit_trace import ViTTracePoint, ViTTraceRecord, ViTTraceResult


_MAX_EVENTS = 1024
_MAX_BYTES = 64 * 1024 * 1024
_MODULE_ORDER = {"attention": 0, "intermediate": 1, "block": 2}
_GEOMETRY_KEYS = {"pixel_shape", "patch_shape", "grid_shape", "sequence_length",
                  "cls_token_index", "patch_order", "box_coordinates"}


def _positive_ints(value, length):
    return (type(value) is list and len(value) == length
            and all(type(item) is int and item > 0 for item in value))


def _geometry(value):
    if type(value) is not dict or set(value) != _GEOMETRY_KEYS:
        return None
    pixels, patches, grid = value["pixel_shape"], value["patch_shape"], value["grid_shape"]
    if (not _positive_ints(pixels, 4) or not _positive_ints(patches, 2)
            or not _positive_ints(grid, 2) or type(value["sequence_length"]) is not int
            or type(value["cls_token_index"]) is not int or value["cls_token_index"] != 0
            or value["patch_order"] != "row-major"
            or value["box_coordinates"] != "processed-pixel-xyxy-exclusive"
            or pixels[2] != grid[0] * patches[0] or pixels[3] != grid[1] * patches[1]
            or value["sequence_length"] != 1 + grid[0] * grid[1]):
        return None
    return {key: list(item) if type(item) is list else item for key, item in value.items()}


def _point_order(point):
    return point.layer, _MODULE_ORDER[point.module], point.batch, point.token


def _validate_result(result, points, max_bytes):
    """Check the producer's declared geometry and all expected trace records."""
    if type(result) is not ViTTraceResult or type(result.records) is not tuple:
        return None, "expected a ViTTraceResult with a tuple of records"
    geometry = _geometry(result.image_metadata)
    if geometry is None:
        return None, "invalid processed-image geometry"
    if len(result.records) != len(points):
        return None, "record count differs from the declared policy"
    batch, _, _, _ = geometry["pixel_shape"]
    patch_h, patch_w = geometry["patch_shape"]
    rows, columns = geometry["grid_shape"]
    sequence = geometry["sequence_length"]
    run_id, total, widths = None, 0, {}
    for index, (point, record) in enumerate(zip(points, result.records, strict=True)):
        if type(record) is not ViTTraceRecord or type(record.event) is not TraceEvent:
            return None, f"record {index} is not a ViT trace record"
        event, value = record.event, record.value
        if run_id is None:
            run_id = event.identity.run_id
        path = f"vit.encoder.layer.{point.layer}"
        if point.module != "block":
            path += f".{point.module}"
        if (event.identity != RunIdentity(run_id) or len(run_id) > 128
                or event.sequence != index or event.forward_index != 0 or event.invocation != point.invocation
                or event.module_path != path or event.module_kind != f"vit_{point.module}"
                or event.layer_index != point.layer or event.output_path != ()
                or event.batch_positions != (point.batch,) or event.token_positions != (point.token,)
                or event.batch_axis != 0 or event.token_axis != 1
                or event.dtype != "torch.float32" or event.device != "cpu" or event.capture_mode != "direct"
                or event.generation != GenerationContext() or event.saved_dtype != "torch.float32"
                or event.saved_device != "cpu" or len(event.shape) != 3
                or event.shape[:2] != (batch, sequence) or event.shape[2] <= 0
                or event.saved_shape != (event.shape[2],)
                or point.batch >= batch or point.token >= sequence):
            return None, f"record {index} does not match its declared coordinate or metadata"
        width = event.shape[2]
        kind = "hidden" if point.module in ("attention", "block") else "intermediate"
        if kind in widths and widths[kind] != width:
            return None, f"record {index} has inconsistent feature width"
        widths[kind] = width
        if point.token == 0:
            expected = ("cls", None, None, None)
        else:
            row, column = divmod(point.token - 1, columns)
            expected = ("patch", row, column,
                        (column * patch_w, row * patch_h, (column + 1) * patch_w, (row + 1) * patch_h))
            if (type(record.patch_row) is not int or type(record.patch_column) is not int
                    or type(record.processed_pixel_box_xyxy) is not tuple
                    or len(record.processed_pixel_box_xyxy) != 4
                    or any(type(value) is not int for value in record.processed_pixel_box_xyxy)):
                return None, f"record {index} has invalid patch-grid coordinate types"
        if (record.token_kind, record.patch_row, record.patch_column, record.processed_pixel_box_xyxy) != expected:
            return None, f"record {index} has inconsistent CLS or patch-grid coordinates"
        if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.float32
                or value.layout != torch.strided or value.is_nested or value.is_neg() or value.is_conj()
                or any(name is not None for name in getattr(value, "names", ())) or tuple(value.shape) != (width,)):
            return None, f"record {index} has an invalid saved vector"
        total += value.numel() * value.element_size()
        if total > max_bytes:
            return None, "trace value byte budget exceeded"
    if type(result.tensor_bytes) is not int or result.tensor_bytes != total:
        return None, "retained tensor byte count differs from the records"
    return {"geometry": geometry, "run_id": run_id, "bytes": total, "records": result.records}, None


def _snapshot(value):
    # Copy integer bits first; float32 conversion can flush subnormals.
    bits = value.detach().view(torch.int32).contiguous().clone()
    if bool(((bits & 0x7f800000) == 0x7f800000).any()):
        raise ValueError("saved vector contains nonfinite float32 values")
    return bits


def _metrics(left_bits, right_bits):
    count, bit_changes, numeric_changes, maximum = left_bits.numel(), 0, 0, 0.0
    absolute_sums, squared_sums = [], []
    for start in range(0, count, 4096):
        left = left_bits[start:start + 4096].tolist()
        right = right_bits[start:start + 4096].tolist()
        differences = []
        for a, b in zip(left, right, strict=True):
            bit_changes += a != b
            difference = abs(f32_bits_to_f64(a & 0xffffffff) - f32_bits_to_f64(b & 0xffffffff))
            numeric_changes += difference != 0.0
            maximum = max(maximum, difference)
            differences.append(difference)
        absolute_sums.append(math.fsum(differences))
        squared_sums.append(math.fsum(value * value for value in differences))
    return {"l2_difference": math.sqrt(math.fsum(squared_sums)),
            "max_absolute_difference": maximum,
            "mean_absolute_difference": math.fsum(absolute_sums) / count,
            "numerically_changed_elements": numeric_changes,
            "bitwise_changed_elements": bit_changes}


def compare_vit_traces(reference, candidate, *, points, pairing_verified=False,
                       max_events=128, max_trace_bytes=8 * 1024 * 1024):
    """Compare identical declared ViT coordinates after a caller pairing attestation.

    Both results must contain the complete selected policy in trace execution
    order and the same processed geometry. Unverified or mismatched evidence is
    inconclusive and returns no partial metrics. The attestation is a caller
    statement about execution, model/runtime and intended input alignment; it
    does not establish semantic correspondence or causal origin.
    """
    if type(pairing_verified) is not bool:
        raise ValueError("pairing_verified must be a boolean")
    if (type(max_events) is not int or not 0 < max_events <= _MAX_EVENTS
            or type(max_trace_bytes) is not int or not 0 < max_trace_bytes <= _MAX_BYTES):
        raise ValueError("event or trace byte budget is outside the supported bounds")
    if (type(points) not in (tuple, list) or not points or len(points) > max_events
            or any(type(point) is not ViTTracePoint for point in points) or len(set(points)) != len(points)):
        raise ValueError("points must be a bounded nonempty list of distinct ViTTracePoint values")
    policy = tuple(sorted(points, key=_point_order))
    report = {"schema": "vit-paired-traces-v1", "status": "inconclusive",
              "pairing_verified_by_caller": pairing_verified, "issues": [], "comparisons": [],
              "causal_origin": "unknown"}
    if not pairing_verified:
        report["issues"].append("caller pairing attestation is required")
        return report
    left, left_error = _validate_result(reference, policy, max_trace_bytes)
    right, right_error = _validate_result(candidate, policy, max_trace_bytes)
    if left_error:
        report["issues"].append(f"reference: {left_error}")
    if right_error:
        report["issues"].append(f"candidate: {right_error}")
    if report["issues"]:
        return report
    if left["bytes"] + right["bytes"] > max_trace_bytes:
        report["issues"].append("combined trace value byte budget exceeded")
    if left["geometry"] != right["geometry"]:
        report["issues"].append("processed-image geometries differ")
    if left["run_id"] == right["run_id"]:
        report["issues"].append("run IDs must be distinct")
    for index, (a, b) in enumerate(zip(left["records"], right["records"], strict=True)):
        if a.event.shape != b.event.shape or a.event.saved_shape != b.event.saved_shape:
            report["issues"].append(f"record {index} source or saved shapes differ")
    if report["issues"]:
        return report
    comparisons = []
    try:
        for index, (point, a, b) in enumerate(zip(policy, left["records"], right["records"], strict=True)):
            a_bits, b_bits = _snapshot(a.value), _snapshot(b.value)
            node = f"vit-selected-{index}"
            def proxy(run_id, bits):
                return SimpleNamespace(run_id=run_id, node_id=node, sequence=0, occurrence=0,
                                       shape=tuple(bits.shape), dtype="torch.float32", device="cpu",
                                       value=bits.view(torch.float32))
            equality = compare_traces([proxy(left["run_id"], a_bits)], [proxy(right["run_id"], b_bits)],
                                      reference_complete=True, candidate_complete=True)
            comparison = equality["comparisons"][0]["result"]
            if comparison not in ("equal", "different"):
                raise ValueError("saved vector is unavailable for exact comparison")
            comparisons.append({"layer": point.layer, "module": point.module,
                "module_path": a.event.module_path, "batch": point.batch, "token": point.token,
                "invocation": point.invocation, "token_kind": a.token_kind,
                "patch_row": a.patch_row, "patch_column": a.patch_column,
                "processed_pixel_box_xyxy": (None if a.processed_pixel_box_xyxy is None
                                               else list(a.processed_pixel_box_xyxy)),
                "feature_width": a.event.shape[2], "numeric_equality": comparison,
                **_metrics(a_bits, b_bits)})
    except (TypeError, ValueError, RuntimeError) as error:
        report["issues"].append(f"comparison unavailable: {error}")
        return report
    report.update({"status": "compared", "comparisons": comparisons,
                   "reference_run_id": left["run_id"], "candidate_run_id": right["run_id"],
                   "image_geometry": left["geometry"],
                   "reference_tensor_bytes": left["bytes"], "candidate_tensor_bytes": right["bytes"]})
    return report
