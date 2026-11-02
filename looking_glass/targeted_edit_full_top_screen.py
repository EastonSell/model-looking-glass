"""Private baseline screen for the full-top study; supervision is external.

The caller must first bind the reviewed source, manifest, exact public cases,
checkpoint, current exposure and relevant custody/reservation. This writer
cannot authorize a model forward or authenticate those records. Any failure
poisons the guard; preserve the partial output and terminate its child.
"""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_full_top as numerical
from . import targeted_edit_v2_owner as owner


CASE_SHA256 = "4057696ddef9e3b0d09db02f46e6454561a61ac1a07f8b8a13b92718944f4261"
PROTOCOL_SHA256 = "36f1a709a6951866d6835802be71c8a0b3ea1a8bbcbfab4c5c3a97be5392299f"
FACTS = ("france", "japan", "italy", "uk")
ROLES = ("discovery", "efficacy-calibration", "efficacy-calibration") + ("evaluation",) * 6


class ScreenError(ValueError):
    """The frozen screen, row or recorded identity is invalid."""


def _identity(value):
    if type(value) is not dict or set(value) != {"source_commit", "manifest_sha256"}:
        raise ScreenError("screen identity differs")
    for key, item in value.items():
        length = 40 if key == "source_commit" else 64
        if type(item) is not str or re.fullmatch("[0-9a-f]{" + str(length) + "}", item) is None:
            raise ScreenError("screen identity differs")
    return dict(value)


def _cases(raw):
    if type(raw) is not bytes or len(raw) > 200_000 or hashlib.sha256(raw).hexdigest() != CASE_SHA256:
        raise ScreenError("fixed case bytes differ")
    try:
        data = json.loads(raw)
        cases = data["cases"]
        if (data["protocol_sha256"] != PROTOCOL_SHA256 or data["model_forward"] is not False
                or data["fact_order"] != list(FACTS) or type(cases) is not list or len(cases) != 48):
            raise ScreenError("fixed case metadata differs")
        if len({row["case_id"] for row in cases}) != 48:
            raise ScreenError("case identifiers repeat")
        rows, pairs = [], []
        for fact, fact_id in enumerate(FACTS):
            token_pair = None
            for role, row in zip(ROLES, cases[9 * fact:9 * fact + 9]):
                ids = row["token_ids"]
                if (row["fact_id"] != fact_id or row["role"] != role or type(ids) is not list or not ids
                        or any(type(token) is not int or not 0 <= token < v1.VOCAB_SIZE for token in ids)):
                    raise ScreenError("target role or input differs")
                pair = []
                for label in ("factual", "counterfactual"):
                    item = row[label]
                    token = item["token_id"]
                    if (type(token) is not int or not 0 <= token < v1.VOCAB_SIZE
                            or item["full_token_ids"] != ids + [token]):
                        raise ScreenError("a target continuation differs")
                    pair.append(token)
                if pair[0] == pair[1] or (token_pair is not None and pair != token_pair):
                    raise ScreenError("a fact's token pair differs")
                token_pair = pair
                rows.append({"case_id": row["case_id"], "role": role, "fact_id": fact_id,
                             "token_ids": tuple(ids)})
            pairs.append(tuple(token_pair))
        if any(row["role"] != "locality" for row in cases[36:]):
            raise ScreenError("locality roles differ")
        return rows, pairs
    except (KeyError, IndexError, TypeError, ValueError, UnicodeError) as exc:
        if isinstance(exc, ScreenError):
            raise
        raise ScreenError("case document differs") from exc


def screen(guard, case_bytes, identity, output):
    """Save all 36 original full-vocabulary rows and fixed selection."""
    try:
        rows, pairs = _cases(case_bytes)
        identity = _identity(identity)
        guard.verify()
        output = Path(output)
        output.mkdir(exist_ok=False)
        row_size = v1.VOCAB_SIZE * 4
        row_hashes = []
        probes = {"before": v1._denormal_probe()}
        with (output / "screen.f32").open("xb") as raw, (output / "observations.jsonl").open("xb") as journal:
            def forward_rows():
                probes["after"] = v1._denormal_probe()
                for offset, case in enumerate(rows):
                    before = v1._denormal_probe()
                    logits = v1._logit_row(guard.model, case["token_ids"])
                    after = v1._denormal_probe()
                    if type(logits) is not bytes or len(logits) != row_size:
                        raise ScreenError("a baseline row has the wrong size")
                    # Validate the float32 bits before any row is marked complete.
                    numerical._rows32(np.frombuffer(logits, dtype="<f4").reshape(1, -1), 1)
                    digest = hashlib.sha256(logits).hexdigest()
                    owner._append(raw, logits)
                    owner._append(journal, v1._canonical({"case_id": case["case_id"], "index": offset,
                        "offset": offset * row_size, "bytes": row_size, "sha256": digest,
                        "forward_denormal_probe_before": before, "forward_denormal_probe_after": after}))
                    row_hashes.append(digest)
            guard.condition(None, guard.weight_sha, forward_rows)
        data = (output / "screen.f32").read_bytes()
        if len(data) != 36 * row_size:
            raise ScreenError("saved screen length differs")
        score_before = v1._denormal_probe()
        selected = numerical.select_screen(np.frombuffer(data, dtype="<f4").reshape(36, -1), pairs)
        score_after = v1._denormal_probe()
        for item, case, digest in zip(selected["observations"], rows, row_hashes):
            item.update({"case_id": case["case_id"], "role": case["role"], "fact_id": case["fact_id"],
                         "logit_sha256": digest, "logit_byte_offset": item["offset"] * row_size,
                         "logit_bytes": row_size})
        state = {"condition": "original", "applied_weight_sha256": guard.weight_sha,
                 "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                 "buffer_state_sha256": guard.buffer_sha, "runtime_settings": guard.runtime_settings,
                 "addition_denormal_probe_before": probes["before"],
                 "addition_denormal_probe_after": probes["after"], "cleanup": guard.last_cleanup}
        owner._json_new(output / "state-original.json", state)
        artifacts = {}
        for name in ("screen.f32", "observations.jsonl", "state-original.json"):
            raw = (output / name).read_bytes()
            artifacts[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        record = {"version": "targeted-edit-full-top-private-screen-v1", "status": "private-screen",
                  "identity": identity, "case_sha256": CASE_SHA256, "selection": selected,
                  "score_denormal_probe_before": score_before, "score_denormal_probe_after": score_after,
                  "artifacts": artifacts,
                  "scope": "saved original-model selection only; publication, execution and cleanup require the owner"}
        owner._json_new(output / "screen-selection.json", record)
        guard.commit_condition()
        guard.verify()
        return record
    except BaseException:
        guard.poisoned = True
        raise
