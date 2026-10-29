"""Private evaluation artifact writers; no live entry point is enabled.

The caller must independently establish committed selection, source, witness,
exclusive model ownership and supervision. These helpers do not publish a
terminal or certify any execution or cleanup attestation.
"""

from pathlib import Path

from . import targeted_edit_v2_owner as owner


def _identity(identity, v2, evidence):
    keys = {"version", "phase", "run_id", "manifest_sha256", "source_commit",
            "selection_commit", "selection_terminal_sha256", "evaluation_witness_commit",
            "evaluation_witness_sha256"}
    if (type(identity) is not dict or set(identity) != keys or
            identity["version"] != v2.VERSION or identity["phase"] != "evaluation" or
            not evidence._run_id(identity["run_id"]) or
            any(not evidence._digest(identity[name]) for name in
                ("manifest_sha256", "selection_terminal_sha256", "evaluation_witness_sha256")) or
            any(not evidence._commit(identity[name]) for name in
                ("source_commit", "selection_commit", "evaluation_witness_commit"))):
        raise owner.OwnerError("evaluation identity is incomplete")
    return dict(identity)


def _snapshot(identity, snapshot, guard, v1, evidence, gate):
    """Bind execution to the already gate-verified snapshot before any row."""
    if (type(snapshot) is not gate.SelectionSnapshot or snapshot.selection_commit != identity["selection_commit"]
            or snapshot.terminal_sha256 != identity["selection_terminal_sha256"]
            or v1._sha(snapshot.manifest_bytes) != identity["manifest_sha256"]):
        raise owner.OwnerError("evaluation selection snapshot identity differs")
    try:
        manifest = v1._decode_document(snapshot.manifest_bytes)
        development = v1._decode_document(snapshot.metadata_bytes)
        selection = v1._decode_document(snapshot.selection_bytes)
        compatibility = development["compatibility"]
        controls = development["controls"]
        expected = {"original": compatibility["weight_sha256"],
                    **{name: controls[name]["realized_weight_sha256"]
                       for name in ("selected", "random", "opposite")},
                    "v1_1_percent": manifest["v1_compatibility"]["candidates"]["gradient"]["realized_weight_sha256"]}
        intended = {**{name: controls[name]["intended_delta_sha256"]
                       for name in ("selected", "random", "opposite")},
                    "v1_1_percent": manifest["v1_compatibility"]["candidates"]["gradient"]["intended_delta_sha256"]}
        if (manifest["source_commit"] != identity["source_commit"] or
                development["source_commit"] != identity["source_commit"] or
                development["manifest_sha256"] != identity["manifest_sha256"] or
                development["phase"] != "development" or
                not evidence._run_id(development["run_id"]) or development["run_id"] == identity["run_id"] or
                compatibility["model_state_sha256"] != guard.state_sha or
                compatibility["weight_sha256"] != guard.weight_sha or
                development["buffer_state_sha256"] != guard.buffer_sha or
                development["runtime_settings"] != guard.runtime_settings or
                v1._tensor_sha(guard.original) != guard.weight_sha or type(selection) is not dict):
            raise owner.OwnerError("evaluation model or development identity differs")
    except (KeyError, TypeError, AttributeError) as exc:
        raise owner.OwnerError("evaluation snapshot is incomplete") from exc
    return selection, expected, intended


def _evaluation_conditions(output, identity, snapshot, guard, updates, applied, v1, v2, evidence, gate):
    """Write fixed-order rows and fsynced restoration journals, but no terminal.

    ``snapshot`` must come from the committed-selection gate, and ``guard``
    must be the trusted exclusive owner's already-verified model guard.
    A failed or lost return leaves a partial attempt; no completed condition
    may be blindly replayed. Successful conditions stay fenced until their
    state record has been flushed and fsynced by this writer.
    """
    identity = _identity(identity, v2, evidence)
    guard.verify()
    _selection, expected, intended = _snapshot(identity, snapshot, guard, v1, evidence, gate)
    names = tuple(v2.EVALUATION_CONDITIONS)
    if (names != ("original", "selected", "random", "opposite", "v1_1_percent") or
            type(updates) is not dict or set(updates) != set(names[1:]) or
            type(applied) is not dict or set(applied) != set(names) or
            applied != expected or applied["original"] != guard.weight_sha or
            any(not evidence._digest(applied[name]) for name in names) or
            any(not evidence._digest(intended[name]) for name in names[1:])):
        raise owner.OwnerError("evaluation updates or applied digests are incomplete")
    # Validate every fixed edit before exposing even the original held-out row.
    # These sequential temporary tensors are outside any general host-memory cap.
    torch = guard.torch
    def check_update(name):
        with torch.no_grad():
            delta = updates[name]
            if (type(delta) is not torch.Tensor or delta.device.type != "cpu" or delta.dtype != torch.float32
                    or delta.layout != torch.strided or delta.shape != guard.original.shape
                    or not delta.is_contiguous() or delta.requires_grad or not bool(torch.isfinite(delta).all())):
                raise owner.OwnerError("evaluation update is invalid")
            if v1._tensor_sha(delta) != intended[name]:
                raise owner.OwnerError("evaluation intended update differs from the selected control")
            changed = guard.original + delta
            if not bool(torch.isfinite(changed).all()) or v1._tensor_sha(changed) != applied[name]:
                raise owner.OwnerError("evaluation update does not match the selected control")
    for name in names[1:]:
        check_update(name)
    guard.verify()
    output = Path(output)
    row_size = v1.VOCAB_SIZE * 4
    with open(output / "observations.jsonl", "xb") as journal, open(output / "condition_states.jsonl", "xb") as states:
        for condition in names:
            guard.verify()
            if condition != "original":
                check_update(condition)
            before = v1._denormal_probe()
            runtime_before = dict(guard.runtime_settings)
            with open(output / f"{condition}.f32", "xb") as raw:
                def run_rows():
                    for index, case in enumerate(v2.EVALUATION_CASES):
                        row_before = v1._denormal_probe()
                        row = v1._logit_row(guard.model, case[3])
                        row_after = v1._denormal_probe()
                        if type(row) is not bytes or len(row) != row_size:
                            raise owner.OwnerError("a serialized evaluation row has the wrong size")
                        owner._append(raw, row)
                        owner._append(journal, v1._canonical({**identity, "condition": condition,
                            "case_id": case[0], "offset": index * row_size, "bytes": row_size,
                            "sha256": v1._sha(row), "denormal_probe_before": row_before,
                            "denormal_probe_after": row_after}))
                guard.condition(None if condition == "original" else updates[condition], applied[condition], run_rows)
            after = v1._denormal_probe()
            runtime_after = owner._runtime_settings(guard.torch)
            if (guard.last_cleanup != "verified" or not guard.poisoned or runtime_after != runtime_before):
                raise owner.OwnerError("evaluation cleanup or runtime was not verified")
            owner._append(states, v1._canonical({**identity, "condition": condition,
                "model_state_before_sha256": guard.state_sha, "weight_before_sha256": guard.weight_sha,
                "applied_weight_sha256": applied[condition], "model_state_after_sha256": guard.state_sha,
                "weight_after_sha256": guard.weight_sha, "buffer_state_before_sha256": guard.buffer_sha,
                "buffer_state_after_sha256": guard.buffer_sha,
                "runtime_settings_before": runtime_before, "runtime_settings_after": runtime_after,
                "denormal_probe_before": before, "denormal_probe_after": after,
                "flags_restored": True, "grads_cleared": True, "hooks_restored": True}))
            guard.commit_condition()


def _evaluation_analysis(output, identity, snapshot, guard, v1, v2, evidence, gate, np):
    """Recompute the provisional report from saved rows and the selection bytes."""
    identity = _identity(identity, v2, evidence)
    selection, _expected, _intended = _snapshot(identity, snapshot, guard, v1, evidence, gate)
    output = Path(output)
    size = len(v2.EVALUATION_CASES) * v1.VOCAB_SIZE * 4
    arrays = {}
    for name in v2.EVALUATION_CONDITIONS:
        data = evidence._read(output / f"{name}.f32", size)
        if len(data) != size:
            raise owner.OwnerError("evaluation matrix is incomplete")
        arrays[name] = np.frombuffer(data, dtype="<f4").reshape(len(v2.EVALUATION_CASES), v1.VOCAB_SIZE)
    guard.verify()
    before = v1._denormal_probe()
    result = v2.evaluate(arrays, selection)
    after = v1._denormal_probe()
    guard.verify()
    owner._json_new(output / "selection.json", {**identity, "selection": selection})
    owner._json_new(output / "report.json", {**identity, "result": result})
    return {"analysis_denormal_probe_before": before, "analysis_denormal_probe_after": after}
