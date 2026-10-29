"""Recompute a prospective v2 evaluation from bounded saved artifacts.

This checks internal identities, saved-logit arithmetic and recorded cleanup.
The caller must separately obtain the committed selection from the selection
gate and verify the committed phase witness, source and execution. Neither a
constructible snapshot nor a passing saved record proves those attestations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_cpu_v2 as v2
from . import targeted_edit_v2_edits as edits
from . import targeted_edit_v2_evidence as evidence
from . import targeted_edit_v2_gate as gate


PHASE = "evaluation"
CONDITIONS = v2.EVALUATION_CONDITIONS
DOCUMENTS = ("started.json", "owner.json", "metadata.json", "selection.json", "report.json",
             "observations.jsonl", "condition_states.jsonl")
CLEANUP_KEYS = (*evidence.CLEANUP_KEYS, "model_disposed")


def _same(left, right):
    return v1._canonical(left) == v1._canonical(right)


def _inspect(directory, *, snapshot, manifest_sha256, terminal_sha256,
             evaluation_witness_commit, evaluation_witness_sha256, witness_observed_utc):
    if (type(snapshot) is not gate.SelectionSnapshot or not evidence._digest(manifest_sha256) or
            not evidence._digest(terminal_sha256) or not evidence._commit(evaluation_witness_commit) or
            not evidence._digest(evaluation_witness_sha256) or not evidence._commit(snapshot.selection_commit) or
            not evidence._digest(snapshot.terminal_sha256)):
        raise v1.StudyError("pinned evaluation identities are incomplete")
    manifest = v1._decode_document(snapshot.manifest_bytes)
    selection = v1._decode_document(snapshot.selection_bytes)
    development = v1._decode_document(snapshot.metadata_bytes)
    if (v1._sha(snapshot.manifest_bytes) != manifest_sha256 or manifest.get("version") != v2.VERSION or
            not evidence._commit(manifest.get("source_commit")) or
            development.get("manifest_sha256") != manifest_sha256 or
            development.get("source_commit") != manifest["source_commit"] or
            development.get("phase") != "development" or not evidence._run_id(development.get("run_id"))):
        raise v1.StudyError("development snapshot identity differs")
    gate._check_metadata(manifest, selection, development, development["exposure"], development["custody"])
    root = Path(directory)
    evidence._regular(root, directory=True)
    try:
        names = {path.name for path in root.iterdir()}
    except OSError as exc:
        raise v1.StudyError("evaluation artifact directory could not be listed") from exc
    artifact_names = {*DOCUMENTS, *(f"{name}.f32" for name in CONDITIONS)}
    if names != artifact_names | {"terminal.json"}:
        raise v1.StudyError("evaluation inventory is incomplete or has extra files")
    files = {"terminal.json": evidence._read(root / "terminal.json", evidence.MAX_JSON)}
    if v1._sha(files["terminal.json"]) != terminal_sha256:
        raise v1.StudyError("evaluation terminal differs from its external pin")
    terminal = v1._decode_document(files["terminal.json"])
    run_id = terminal.get("run_id")
    common = {"version": v2.VERSION, "phase": PHASE, "run_id": run_id,
              "manifest_sha256": manifest_sha256, "source_commit": manifest["source_commit"],
              "selection_commit": snapshot.selection_commit, "selection_terminal_sha256": snapshot.terminal_sha256,
              "evaluation_witness_commit": evaluation_witness_commit,
              "evaluation_witness_sha256": evaluation_witness_sha256}
    terminal_keys = set(common) | {"status", "cleanup", "finished_utc", "artifacts"}
    if (not evidence._run_id(run_id) or run_id == development["run_id"] or set(terminal) != terminal_keys or
            any(terminal.get(key) != value for key, value in common.items()) or terminal.get("status") != "complete" or
            type(terminal.get("cleanup")) is not dict or set(terminal["cleanup"]) != set(CLEANUP_KEYS) or
            any(terminal["cleanup"][key] is not True for key in CLEANUP_KEYS) or
            type(terminal.get("artifacts")) is not dict or set(terminal["artifacts"]) != artifact_names or
            any(not evidence._digest(value) for value in terminal["artifacts"].values())):
        raise v1.StudyError("evaluation terminal is incomplete or cleanup was not verified")
    finished = evidence._utc(terminal["finished_utc"])
    row_bytes = v1.VOCAB_SIZE * 4
    matrix_bytes = len(v2.EVALUATION_CASES) * row_bytes
    for name in sorted(artifact_names):
        limit = (matrix_bytes if name.endswith(".f32") else
                 (evidence.MAX_JOURNAL if name.endswith(".jsonl") else evidence.MAX_JSON))
        raw = evidence._read(root / name, limit)
        if (v1._sha(raw) != terminal["artifacts"][name] or
                (name.endswith(".f32") and len(raw) != matrix_bytes)):
            raise v1.StudyError("evaluation artifact differs from its inventory or fixed size")
        files[name] = raw
    started = v1._decode_document(files["started.json"])
    owner = v1._decode_document(files["owner.json"])
    metadata = v1._decode_document(files["metadata.json"])
    selected = v1._decode_document(files["selection.json"])
    report = v1._decode_document(files["report.json"])
    def exact(record, extras):
        return type(record) is dict and set(record) == set(common) | set(extras) and all(
            record.get(key) == value for key, value in common.items())
    if (not exact(started, {"started_utc"}) or evidence._utc(started["started_utc"]) > finished or
            evidence._utc(witness_observed_utc) > evidence._utc(started["started_utc"]) or
            not exact(owner, {"supervisor_run_id", "pid"}) or
            not exact(selected, {"selection"}) or not _same(selected["selection"], selection) or
            not exact(report, {"result"})):
        raise v1.StudyError("evaluation record identities or reported times differ")
    extra = {"model", "environment", "compatibility", "controls", "buffer_state_sha256", "runtime_settings",
             "analysis_denormal_probe_before", "analysis_denormal_probe_after", "supervision"}
    if not exact(metadata, extra):
        raise v1.StudyError("evaluation metadata is incomplete")
    owner_info = evidence._check_supervision(metadata["supervision"])
    evidence._check_analysis_probes(metadata)
    evidence._check_runtime_settings(metadata["runtime_settings"], metadata["environment"])
    if (owner.get("supervisor_run_id") != owner_info["run_id"] or type(owner.get("pid")) is not int or
            owner["pid"] != owner_info["pid"] or owner_info["run_id"] == development["supervision"]["owner"]["run_id"] or
            any(not _same(metadata[key], development[key]) for key in
                ("model", "compatibility", "controls", "buffer_state_sha256", "runtime_settings")) or
            any(key not in metadata["environment"] or
                not _same(metadata["environment"][key], development["environment"][key])
                for key in edits.ENVIRONMENT_FIELDS)):
        raise v1.StudyError("evaluation owner, runtime or reconstructed edit attestation differs")
    state_sha = development["compatibility"]["model_state_sha256"]
    weight_sha = development["compatibility"]["weight_sha256"]
    buffer_sha = development["buffer_state_sha256"]
    applied = {"original": weight_sha,
               **{name: development["controls"][name]["realized_weight_sha256"]
                  for name in ("selected", "random", "opposite")},
               "v1_1_percent": manifest["v1_compatibility"]["candidates"]["gradient"]["realized_weight_sha256"]}
    if any(not evidence._digest(value) for value in (state_sha, weight_sha, buffer_sha, *applied.values())):
        raise v1.StudyError("evaluation state digests are incomplete")
    states = files["condition_states.jsonl"].splitlines(keepends=True)
    if len(states) != len(CONDITIONS):
        raise v1.StudyError("evaluation state journal has missing or extra rows")
    for index, condition in enumerate(CONDITIONS):
        expected = {**common, "condition": condition, "model_state_before_sha256": state_sha,
                    "weight_before_sha256": weight_sha, "applied_weight_sha256": applied[condition],
                    "model_state_after_sha256": state_sha, "weight_after_sha256": weight_sha,
                    "buffer_state_before_sha256": buffer_sha, "buffer_state_after_sha256": buffer_sha,
                    "runtime_settings_before": development["runtime_settings"],
                    "runtime_settings_after": development["runtime_settings"],
                    "denormal_probe_before": evidence.DENORMAL_PROBE, "denormal_probe_after": evidence.DENORMAL_PROBE,
                    "flags_restored": True, "grads_cleared": True, "hooks_restored": True}
        if not _same(v1._decode_document(states[index]), expected):
            raise v1.StudyError("evaluation state journal is incomplete or inconsistent")
    lines = files["observations.jsonl"].splitlines(keepends=True)
    if len(lines) != len(CONDITIONS) * len(v2.EVALUATION_CASES):
        raise v1.StudyError("evaluation observation journal has missing or extra rows")
    arrays = {}
    for condition_index, condition in enumerate(CONDITIONS):
        raw = files[f"{condition}.f32"]
        arrays[condition] = np.frombuffer(raw, dtype="<f4").reshape(len(v2.EVALUATION_CASES), v1.VOCAB_SIZE)
        for case_index, case in enumerate(v2.EVALUATION_CASES):
            offset = case_index * row_bytes
            expected = {**common, "condition": condition, "case_id": case[0], "offset": offset,
                        "bytes": row_bytes, "sha256": v1._sha(raw[offset:offset + row_bytes]),
                        "denormal_probe_before": evidence.DENORMAL_PROBE, "denormal_probe_after": evidence.DENORMAL_PROBE}
            if not _same(v1._decode_document(lines[condition_index * len(v2.EVALUATION_CASES) + case_index]), expected):
                raise v1.StudyError("evaluation journal does not bind ordered output rows")
    derived = v2.evaluate(arrays, selection)
    # The saved producer probes do not attest the auditor's current thread.
    # v2 probes before scoring; check it again after all metric calculations.
    v1._denormal_probe()
    if not _same(report["result"], derived):
        raise v1.StudyError("evaluation report differs from the saved logits")
    return {"status": "matching_saved_evaluation", "run_id": run_id,
            "terminal_sha256": terminal_sha256, "started_utc": started["started_utc"],
            "numerical_decision": derived["decision"], "result": derived,
            "scope": "saved bytes and recorded attestations; external provenance and execution not proved"}


def audit_evaluation(directory, *, snapshot, manifest_sha256, terminal_sha256,
                     evaluation_witness_commit, evaluation_witness_sha256, witness_observed_utc):
    """Check a complete saved evaluation; failed or partial attempts never pass.

    Callers must obtain ``snapshot`` from the real committed selection gate,
    independently verify the evaluation witness, and supply its true pinned
    commit, digest and observed time. This helper does not perform those Git
    checks or infer that the reported model, rollback or fetch actually occurred.
    """
    try:
        return _inspect(directory, snapshot=snapshot, manifest_sha256=manifest_sha256,
                        terminal_sha256=terminal_sha256, evaluation_witness_commit=evaluation_witness_commit,
                        evaluation_witness_sha256=evaluation_witness_sha256, witness_observed_utc=witness_observed_utc)
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as exc:
        raise v1.StudyError("invalid evaluation evidence") from exc
