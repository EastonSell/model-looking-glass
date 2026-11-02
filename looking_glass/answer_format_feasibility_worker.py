"""Private saved-row writer for the two-phase answer-format feasibility study.

Publication, custody, source binding, supervision and clean teardown are the
caller's responsibilities. This writer never authorizes a model forward.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from . import answer_format_feasibility as numerical
from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_v2_owner as owner


CASE_SHA = "cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a"
PROTOCOL_SHA = "f60b240f1303fbf6ca785d2197e10d182aa1d34e6818ff8ed3e2f30a70c9307d"
SCOPE_SHA = "2530ea0517847731d9c82731a17d58d23504738cc3e85b52fe259fe9bf4ed330"


class StudyError(ValueError):
    """Incomplete or inconsistent evidence, not a numerical result."""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise StudyError("duplicate JSON key")
            result[key] = value
        return result
    def constant(_value):
        raise StudyError("nonfinite JSON value")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def _cases(raw):
    if type(raw) is not bytes or len(raw) > 300_000 or sha(raw) != CASE_SHA:
        raise StudyError("fixed case bytes differ")
    try:
        doc = _strict_json(raw)
        if (doc["protocol_sha256"] != PROTOCOL_SHA or doc["scope_sha256"] != SCOPE_SHA
                or doc["model_forward"] is not False or len(doc["cases"]) != 64
                or doc["seeds"] != numerical.SEEDS
                or doc["execution_permutations"] != {s: numerical.execution_order(s) for s in numerical.SEEDS}):
            raise StudyError("fixed case metadata differs")
        for stage in numerical.SEEDS:
            rows = [r for r in doc["cases"] if r["role"] == stage]
            numerical._validate_cases(rows, stage, v1.VOCAB_SIZE)
            for row in rows:
                ids = row["token_ids"]
                if (type(ids) is not list or not ids or any(type(i) is not int or not 0 <= i < v1.VOCAB_SIZE for i in ids)
                        or row["prompt"] != row["prefix_text"] + row["suffix_text"]
                        or ids != row["prefix_token_ids"] + row["suffix_token_ids"]):
                    raise StudyError("case input or prefix boundary differs")
                for label in numerical.LABELS:
                    item = row["labels"][label]
                    if item["full_token_ids"] != ids + [item["token_id"]]:
                        raise StudyError("case continuation boundary differs")
        if len({r["case_id"] for r in doc["cases"]}) != 64:
            raise StudyError("duplicate case ID")
        return doc
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        if isinstance(exc, StudyError):
            raise
        raise StudyError("invalid case document") from exc


def _identity(value):
    if type(value) is not dict or set(value) != {"source_commit", "manifest_sha256"}:
        raise StudyError("source identity differs")
    for key, item in value.items():
        size = 40 if key == "source_commit" else 64
        if type(item) is not str or re.fullmatch("[0-9a-f]{" + str(size) + "}", item) is None:
            raise StudyError("source identity differs")
    return dict(value)


def _with_hashes(score, matrix):
    size = v1.VOCAB_SIZE * 4
    for item in score["observations"]:
        offset = item["execution_index"] * size
        item["logit_byte_offset"] = offset
        item["logit_bytes"] = size
        item["logit_sha256"] = sha(matrix[offset:offset + size])
    return score


def _prior_development(doc, prior):
    if (type(prior) is not dict or set(prior) != {"matrix", "result", "matrix_sha256", "result_sha256"}
            or type(prior["matrix"]) is not bytes or type(prior["result"]) is not bytes
            or sha(prior["matrix"]) != prior["matrix_sha256"] or sha(prior["result"]) != prior["result_sha256"]
            or len(prior["matrix"]) != 32 * v1.VOCAB_SIZE * 4):
        raise StudyError("development bytes differ")
    saved = _strict_json(prior["result"])
    cases = [r for r in doc["cases"] if r["role"] == "development"]
    matrix = np.frombuffer(prior["matrix"], dtype="<f4").reshape(32, -1)
    replay = _with_hashes(numerical.summarize("development", matrix, cases), prior["matrix"])
    if (saved.get("stage") != "development" or saved.get("case_sha256") != CASE_SHA
            or v1._canonical(saved.get("score")) != v1._canonical(replay)
            or saved.get("artifacts", {}).get("matrix.f32", {}).get("sha256") != prior["matrix_sha256"]):
        raise StudyError("development selection does not replay")
    return replay


def run(guard, case_bytes, identity, stage, output, *, prior=None):
    """Save one complete phase; fence the guard on every failed boundary."""
    try:
        doc, identity = _cases(case_bytes), _identity(identity)
        if stage not in numerical.SEEDS:
            raise StudyError("phase differs")
        prior_before = None if stage == "development" else v1._denormal_probe()
        previous = None if stage == "development" else _prior_development(doc, prior)
        prior_after = None if stage == "development" else v1._denormal_probe()
        if stage == "development" and prior is not None:
            raise StudyError("development cannot consume prior selection")
        rows = [r for r in doc["cases"] if r["role"] == stage]
        order = numerical.execution_order(stage)
        guard.verify()
        output = Path(output)
        output.mkdir(exist_ok=False)
        row_size = v1.VOCAB_SIZE * 4
        probes = {"before": v1._denormal_probe()}
        with (output / "matrix.f32").open("xb") as matrix, (output / "journal.jsonl").open("xb") as journal:
            def forward_rows():
                probes["after"] = v1._denormal_probe()
                for execution_index, canonical_index in enumerate(order):
                    case = rows[canonical_index]
                    before = v1._denormal_probe()
                    raw = v1._logit_row(guard.model, tuple(case["token_ids"]))
                    after = v1._denormal_probe()
                    if type(raw) is not bytes or len(raw) != row_size:
                        raise StudyError("saved row size differs")
                    numerical.full_top._rows32(np.frombuffer(raw, dtype="<f4").reshape(1, -1), 1)
                    item = {"case_id": case["case_id"], "execution_index": execution_index,
                            "canonical_index": canonical_index, "offset": execution_index * row_size,
                            "bytes": row_size, "sha256": sha(raw),
                            "forward_denormal_probe_before": before, "forward_denormal_probe_after": after}
                    owner._append(matrix, raw)
                    owner._append(journal, v1._canonical(item))
            guard.condition(None, guard.weight_sha, forward_rows)
        matrix = (output / "matrix.f32").read_bytes()
        if len(matrix) != 32 * row_size:
            raise StudyError("saved matrix length differs")
        before = v1._denormal_probe()
        score = numerical.summarize(stage, np.frombuffer(matrix, dtype="<f4").reshape(32, -1), rows,
                                    development=previous)
        score = _with_hashes(score, matrix)
        after = v1._denormal_probe()
        state = {"condition": "original", "applied_weight_sha256": guard.weight_sha,
                 "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                 "buffer_state_sha256": guard.buffer_sha, "runtime_settings": guard.runtime_settings,
                 "addition_denormal_probe_before": probes["before"],
                 "addition_denormal_probe_after": probes["after"], "cleanup": guard.last_cleanup}
        owner._json_new(output / "state-original.json", state)
        artifacts = {}
        for name in ("matrix.f32", "journal.jsonl", "state-original.json"):
            raw = (output / name).read_bytes()
            artifacts[name] = {"bytes": len(raw), "sha256": sha(raw)}
        record = {"version": "answer-format-private-phase-v1", "stage": stage,
                  "identity": identity, "case_sha256": CASE_SHA, "score": score,
                  "score_denormal_probe_before": before, "score_denormal_probe_after": after,
                  "development_replay_denormal_probe_before": prior_before,
                  "development_replay_denormal_probe_after": prior_after,
                  "development_matrix_sha256": None if prior is None else prior["matrix_sha256"],
                  "development_result_sha256": None if prior is None else prior["result_sha256"],
                  "artifacts": artifacts,
                  "scope": "saved rows only; source, custody and teardown require the supervised owner"}
        owner._json_new(output / "results.json", record)
        guard.commit_condition()
        guard.verify()
        return record
    except BaseException:
        guard.poisoned = True
        raise

