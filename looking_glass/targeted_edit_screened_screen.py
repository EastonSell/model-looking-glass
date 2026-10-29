"""Private baseline-only screen; publication and process custody are external.

The owner must first bind and independently review the committed discovery
record, current exposure/custodian checks and complete 44-row reservation.
This helper cannot authenticate those records or authorize a forward. The
caller owns the model and frozen inputs exclusively and must terminate its
isolated process after any failure or interrupted return.
"""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_screened_selection as selection
from . import targeted_edit_v2_owner as owner


CASE_SHA256 = "fc3f00a0b4f7884a3c00011186420d16322009edf75b6b09e4296953dad198e3"


class ScreenError(ValueError):
    """The saved screening attempt is incomplete or inconsistent."""


def _binding(value):
    if type(value) is not dict or set(value) != {"source_commit", "manifest_sha256", "development_record_sha256",
                                                "candidate_sha256"}:
        raise ScreenError("screen identity differs")
    for key, item in value.items():
        length = 40 if key == "source_commit" else 64
        if type(item) is not str or re.fullmatch("[0-9a-f]{" + str(length) + "}", item) is None:
            raise ScreenError("screen identity differs")
    return dict(value)


def _targets(case_bytes):
    if (type(case_bytes) is not bytes or len(case_bytes) > 100_000
            or hashlib.sha256(case_bytes).hexdigest() != CASE_SHA256):
        raise ScreenError("fixed case bytes differ")
    try:
        all_rows = json.loads(case_bytes)["cases"]
        if (type(all_rows) is not list or len(all_rows) != 44
                or any(row["role"] != "target" for row in all_rows[:32])
                or any(row["role"] != "locality" for row in all_rows[32:])):
            raise ScreenError("fixed case roles differ")
        rows = []
        for row in all_rows[:32]:
            ids = row["token_ids"]
            if (type(ids) is not list or not ids
                    or any(type(token) is not int or token < 0 or token >= v1.VOCAB_SIZE for token in ids)):
                raise ScreenError("a screen input row is invalid")
            rows.append({"case_id": row["case_id"], "role": row["role"], "stratum": row["stratum"],
                         "token_ids": tuple(ids)})
        selection.select_screen(rows, [0.0] * 32)
        return rows
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        if isinstance(exc, (ScreenError, selection.SelectionError)):
            raise
        raise ScreenError("fixed case document differs") from exc


def screen(guard, case_bytes, binding, output):
    """Save all 32 original rows and a private sign-only selection record."""
    try:
        fixed = _targets(case_bytes)
        identity = _binding(binding)
        guard.verify()
        output = Path(output)
        output.mkdir(exist_ok=False)
        row_size = v1.VOCAB_SIZE * 4
        margins, row_hashes = [], []
        add_probes = {"before": v1._denormal_probe()}
        with (output / "screen.f32").open("xb") as raw, (output / "observations.jsonl").open("xb") as journal:
            def rows():
                add_probes["after"] = v1._denormal_probe()
                for offset, case in enumerate(fixed):
                    before = v1._denormal_probe()
                    logits = v1._logit_row(guard.model, case["token_ids"])
                    after = v1._denormal_probe()
                    if type(logits) is not bytes or len(logits) != row_size:
                        raise ScreenError("a baseline row has the wrong size")
                    score_before = v1._denormal_probe()
                    values = np.frombuffer(logits, dtype="<f4").astype(np.float64)
                    if not np.isfinite(values).all():
                        raise ScreenError("a baseline row is nonfinite")
                    margin = float(values[v1.LONDON] - values[v1.PARIS])
                    score_after = v1._denormal_probe()
                    row_sha = hashlib.sha256(logits).hexdigest()
                    owner._append(raw, logits)
                    owner._append(journal, v1._canonical({"case_id": case["case_id"], "index": offset,
                        "offset": offset * row_size, "bytes": row_size, "sha256": row_sha,
                        "forward_denormal_probe_before": before, "forward_denormal_probe_after": after,
                        "score_denormal_probe_before": score_before, "score_denormal_probe_after": score_after}))
                    margins.append(margin)
                    row_hashes.append(row_sha)
            guard.condition(None, guard.weight_sha, rows)
        chosen = selection.select_screen(fixed, margins)
        for row in chosen["rows"]:
            row["logit_sha256"] = row_hashes[row["offset"]]
            row["logit_byte_offset"] = row["offset"] * row_size
            row["logit_bytes"] = row_size
        state = {"condition": "original", "applied_weight_sha256": guard.weight_sha,
                 "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                 "buffer_state_sha256": guard.buffer_sha, "runtime_settings": guard.runtime_settings,
                 "addition_denormal_probe_before": add_probes["before"],
                 "addition_denormal_probe_after": add_probes["after"], "cleanup": guard.last_cleanup}
        owner._json_new(output / "state-original.json", state)
        artifacts = {}
        for name in ("screen.f32", "observations.jsonl", "state-original.json"):
            data = (output / name).read_bytes()
            artifacts[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        record = {"version": "targeted-edit-screened-private-screen-v1", "status": "private-screen",
                  "identity": identity, "case_sha256": CASE_SHA256, "selection": chosen,
                  "artifacts": artifacts,
                  "scope": "saved baseline selection only; publication, execution and cleanup require the owner"}
        owner._json_new(output / "screen-selection.json", record)
        guard.commit_condition()
        guard.verify()
        return record
    except BaseException:
        guard.poisoned = True
        raise
