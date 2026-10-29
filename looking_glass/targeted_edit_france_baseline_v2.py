"""Durable baseline core; this is not a model-loading or process supervisor.

A separate owner must bind published source, cases, assets, current exposure,
authorized custody and CPU resources before calling this code, and observe a
clean direct-process exit and cleanup afterwards. Caller-owned objects and
the output namespace must remain exclusive and quiescent. Never retry an
uncertain forward or infer execution from a saved record alone.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_france_top_edits_v2 as edits
from . import targeted_edit_france_top_v2 as science
from . import targeted_edit_v2_owner as custody


CASE_SHA256 = "c8dd418bea948c8ca2b4d1e39e696374a28bd2f911bb16707415c38170fbba82"
CASE_IDS = ("fit", "calibration-1", "calibration-2") + tuple(
    f"T{i:02d}" for i in range(1, 9)) + tuple(f"L{i:02d}" for i in range(1, 17))


class BaselineError(ValueError):
    """The saved baseline could not be completed under its fixed contract."""


def _validate(rows, locality, true_id, alternative_id, vocabulary, identity):
    if (type(vocabulary) is not int or vocabulary < 3 or type(rows) is not list or len(rows) != 27
            or type(identity) is not dict or type(true_id) is not int or type(alternative_id) is not int
            or true_id == alternative_id or not 0 <= true_id < vocabulary
            or not 0 <= alternative_id < vocabulary):
        raise BaselineError("baseline identity or shape differs")
    fixed_identity = json.loads(v1._canonical(identity))
    fixed = []
    for case_id, row in zip(CASE_IDS, rows):
        if (type(row) is not dict or set(row) != {"case_id", "token_ids", "attention_mask"}
                or row["case_id"] != case_id or type(row["token_ids"]) is not list
                or not 1 <= len(row["token_ids"]) <= 128
                or any(type(x) is not int or not 0 <= x < vocabulary for x in row["token_ids"])
                or type(row["attention_mask"]) is not list
                or len(row["attention_mask"]) != len(row["token_ids"])
                or any(type(x) is not int or x != 1 for x in row["attention_mask"])):
            raise BaselineError("baseline row, ID or mask differs")
        fixed.append({"case_id": case_id, "token_ids": list(row["token_ids"]),
                      "attention_mask": list(row["attention_mask"])})
    if len({tuple(row["token_ids"]) for row in fixed}) != 27:
        raise BaselineError("baseline input rows repeat")
    checked_locality = science._locality(locality, vocabulary)
    fixed_locality = [{"case_id": item["case_id"], "accepted_ids": list(item["accepted_ids"])}
                      for item in checked_locality]
    return fixed, fixed_locality, fixed_identity


def _snapshot(result, vocabulary):
    """Capture at most one bounded last-position row, including invalid bits."""
    try:
        logits = result.logits
        if type(logits) is not torch.Tensor:
            return {"tensor_type": type(logits).__name__}, None, None
        metadata = {"shape": list(logits.shape), "dtype": str(logits.dtype),
                    "device": str(logits.device)}
        if (logits.device.type != "cpu" or logits.dtype != torch.float32
                or logits.layout != torch.strided or logits.ndim != 3
                or logits.shape[0] != 1 or logits.shape[1] <= 0
                or not 0 < logits.shape[2] <= vocabulary):
            return metadata, None, logits
        last = logits[0, -1].detach().contiguous()
        data = last.numpy().astype("<f4", copy=False).tobytes()
        return metadata, data, logits
    except Exception as error:
        return {"snapshot_error_type": type(error).__name__}, None, None


def _event(handle, record):
    custody._append(handle, v1._canonical(record))


def _attempt_invalid(output, journal, *, index, case_id, returned, result, vocabulary,
                     entry, error):
    metadata, data, _ = _snapshot(result, vocabulary) if returned else ({}, None, None)
    record = {"kind": "forward_returned_invalid" if returned else "forward_failed",
              "index": index, "case_id": case_id, "error_type": type(error).__name__,
              "actual_entry": entry, "returned_metadata": metadata, "raw_last_row": None}
    if data is not None:
        name = f"invalid-{index:04d}.f32"
        custody._write_new(output / name, data)
        record["raw_last_row"] = {"file": name, "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()}
    _event(journal, record)


def _forward(guard, row, index, vocabulary, output, raw, journal):
    probe_before = edits._probe()
    _event(journal, {"kind": "forward_start", "index": index, "case_id": row["case_id"],
                     "planned_input_ids": row["token_ids"], "planned_attention_mask": row["attention_mask"],
                     "denormal_probe_before": probe_before})
    captured, result, returned, handle = [], None, False, None
    try:
        def observe(_module, args, kwargs):
            if captured or args or set(kwargs) != {"input_ids", "attention_mask", "use_cache"}:
                raise BaselineError("model entry call differs")
            ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
            if (type(ids) is not torch.Tensor or type(mask) is not torch.Tensor
                    or ids.device.type != "cpu" or mask.device.type != "cpu"
                    or ids.dtype != torch.int64 or mask.dtype != torch.int64
                    or ids.shape != (1, len(row["token_ids"])) or mask.shape != ids.shape
                    or kwargs["use_cache"] is not False):
                raise BaselineError("model entry tensors differ")
            captured.append({"input_ids": ids.detach().tolist()[0],
                             "attention_mask": mask.detach().tolist()[0], "use_cache": False})

        handle = guard.model.register_forward_pre_hook(observe, with_kwargs=True)
        tokens = torch.tensor([row["token_ids"]], dtype=torch.int64)
        mask = torch.tensor([row["attention_mask"]], dtype=torch.int64)
        call_error, call_traceback = None, None
        try:
            with torch.no_grad():
                result = guard.model(input_ids=tokens, attention_mask=mask, use_cache=False)
            returned = True
        except BaseException as error:
            call_error, call_traceback = error, error.__traceback__
        try:
            handle.remove()
            handle = None
        except BaseException as error:
            if call_error is None:
                call_error, call_traceback = error, error.__traceback__
            else:
                call_error.add_note("model-entry hook cleanup also failed: " + type(error).__name__)
        if call_error is not None:
            raise call_error.with_traceback(call_traceback)
        metadata, data, logits = _snapshot(result, vocabulary)
        after = edits._probe()
        if (len(captured) != 1 or captured[0]["input_ids"] != row["token_ids"]
                or captured[0]["attention_mask"] != row["attention_mask"]
                or logits is None or tuple(logits.shape) != (1, len(row["token_ids"]), vocabulary)
                or not bool(torch.isfinite(logits).all()) or data is None or len(data) != 4 * vocabulary):
            raise BaselineError("returned logits or observed model entry differ")
        science.prior._rows32(np.frombuffer(data, dtype="<f4").reshape(1, -1), 1)
        digest = hashlib.sha256(data).hexdigest()
        custody._append(raw, data)
        _event(journal, {"kind": "forward_complete", "index": index, "case_id": row["case_id"],
                         "actual_input_ids": captured[0]["input_ids"],
                         "actual_attention_mask": captured[0]["attention_mask"],
                         "offset": index * 4 * vocabulary, "bytes": len(data), "sha256": digest,
                         "returned_metadata": metadata, "denormal_probe_after": after})
        return digest
    except BaseException as error:
        if handle is not None:
            try:
                handle.remove()
            except BaseException as cleanup:
                error.add_note("model-entry hook cleanup also failed: " + type(cleanup).__name__)
        try:
            _attempt_invalid(output, journal, index=index, case_id=row["case_id"], returned=returned,
                             result=result, vocabulary=vocabulary,
                             entry=captured[0] if len(captured) == 1 else None, error=error)
        except BaseException as recording:
            error.add_note("invalid-return evidence could not be fully recorded: " + type(recording).__name__)
        raise


def run_rows(guard, rows, locality, true_id, alternative_id, output, *, vocabulary, identity):
    """Execute a validated 27-row screen under an already-authorized guard.

    `completed.json` is provisional until a separately supervised direct child
    exits and all source, custody and cleanup records are independently checked.
    On failure, retain the directory and do not invoke this function again.
    """
    fixed, locality, identity = _validate(rows, locality, true_id, alternative_id, vocabulary, identity)
    output = Path(output)
    if not output.is_absolute() or not output.parent.is_dir():
        raise BaselineError("a fresh absolute output under an existing parent is required")
    edits._probe()
    guard.verify()
    output.mkdir(exist_ok=False)
    phase, started = "started_record", False
    try:
        custody._json_new(output / "started.json", {"identity": identity, "rows": len(fixed),
                           "vocabulary": vocabulary, "scope": "owner-reported baseline; no execution attestation"})
        started = True
        phase = "forwards"
        row_hashes = []
        with (output / "screen.f32").open("xb") as raw, (output / "events.jsonl").open("xb") as journal:
            def run_forwards():
                for index, row in enumerate(fixed):
                    row_hashes.append(_forward(guard, row, index, vocabulary, output, raw, journal))
            guard.condition(None, guard.weight_sha, run_forwards)
        phase = "scoring"
        data = (output / "screen.f32").read_bytes()
        if len(data) != 27 * vocabulary * 4:
            raise BaselineError("saved matrix size differs")
        baseline = science.screen(np.frombuffer(data, dtype="<f4").reshape(27, vocabulary),
                                  true_id, alternative_id, locality)
        result = {"identity": identity, "baseline": baseline, "row_sha256": row_hashes,
                  "screen_sha256": hashlib.sha256(data).hexdigest(),
                  "journal_sha256": v1._file_sha(output / "events.jsonl"),
                  "original_state_sha256": guard.state_sha, "original_weight_sha256": guard.weight_sha,
                  "original_buffer_sha256": guard.buffer_sha,
                  "scope": "provisional saved baseline; source, custody, exit and cleanup are separate"}
        custody._json_new(output / "provisional.json", result)
        phase = "condition_acknowledgement"
        guard.commit_condition()
        guard.verify()
        custody._json_new(output / "completed.json", {"identity": identity,
                           "provisional_sha256": v1._file_sha(output / "provisional.json"),
                           "status": "provisional-complete", "cleanup": guard.last_cleanup})
        return result
    except BaseException as error:
        guard.poisoned = True
        try:
            custody._json_new(output / "incomplete.json", {"identity": identity, "phase": phase,
                               "started_record_written": started, "error_type": type(error).__name__,
                               "scope": "retain partial evidence; do not retry uncertain work"})
        except BaseException as recording:
            error.add_note("incomplete record could not be fully written: " + type(recording).__name__)
        raise


def run_fixed(guard, case_bytes, identity, output):
    """Bind the one public case document; authorization remains external."""
    if (type(case_bytes) is not bytes or len(case_bytes) > 200_000
            or hashlib.sha256(case_bytes).hexdigest() != CASE_SHA256):
        raise BaselineError("fixed case bytes differ")
    try:
        manifest = json.loads(case_bytes)
        if (manifest["schema"] != "france-full-top-fixed-cases-v3"
                or manifest["model"] != "distilbert/distilgpt2"
                or manifest["revision"] != "2290a62682d06624634c1f46a6ad5be0f47f38aa"):
            raise BaselineError("fixed case identity differs")
        source_rows = manifest["rows"]
        rows = [{key: row[key] for key in ("case_id", "token_ids", "attention_mask")}
                for row in source_rows]
        locality = [{"case_id": row["case_id"],
                     "accepted_ids": [item["token_id"] for item in row["continuations"]]}
                    for row in source_rows[11:]]
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise BaselineError("fixed case document is malformed") from error
    return run_rows(guard, rows, locality, 6342, 3576, output, vocabulary=50257, identity=identity)
