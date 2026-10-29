"""Immutable handoff between v2 selection evidence and a future CPU owner.

No model is loaded or forwarded here. A matching handoff does not itself
authorize a forward; the owner must verify source origins, custody, checkpoint,
fresh-process runtime and reconstructed candidates before touching new cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_cpu_v2 as v2
from . import targeted_edit_v2_edits as edits
from . import targeted_edit_v2_evidence as evidence
from . import targeted_edit_v2_exposure as exposure
from . import targeted_edit_v2_manifest as manifest_module
from . import targeted_edit_v2_custody as custody


@dataclass(frozen=True)
class SelectionSnapshot:
    """Private immutable bytes, not a certificate that execution occurred."""

    manifest_bytes: bytes
    selection_bytes: bytes
    metadata_bytes: bytes
    selection_commit: str
    terminal_sha256: str


def _same(left, right):
    try:
        return v1._canonical(left) == v1._canonical(right)
    except (TypeError, ValueError, RecursionError, OverflowError, UnicodeError) as exc:
        raise v1.StudyError("invalid frozen handoff data") from exc


def _attestation(value, *, dose, original_norm):
    keys = {"planned_norm", "intended_delta_sha256", "realized_weight_sha256",
            "realized_norm", "realized_relative_norm"}
    if (type(value) is not dict or set(value) != keys or
            not evidence._digest(value.get("intended_delta_sha256")) or
            not evidence._digest(value.get("realized_weight_sha256"))):
        raise v1.StudyError("invalid edit attestation")
    for key in ("planned_norm", "realized_norm", "realized_relative_norm"):
        number = value[key]
        if type(number) is not float or not math.isfinite(number) or number <= 0:
            raise v1.StudyError("invalid edit norm")
    if (value["planned_norm"] != dose * original_norm or
            value["realized_relative_norm"] != value["realized_norm"] / original_norm or
            not 0.999 <= value["realized_norm"] / value["planned_norm"] <= 1.001):
        raise v1.StudyError("saved edit norm differs from the fixed dose")


def _check_metadata(manifest, selection, metadata, public_exposure, custody_result):
    doses = dict(v2.GRID)
    if (type(selection) is not dict or type(selection.get("selected_condition")) is not str or
            selection["selected_condition"] not in doses or
            type(selection.get("selected_relative_norm")) is not float or
            selection["selected_relative_norm"] != doses[selection["selected_condition"]] or
            type(selection.get("development_qualified")) is not bool or
            (not selection["development_qualified"] and selection["selected_condition"] != v2.GRID[0][0]) or
            type(selection.get("point_prediction")) is not float or
            not math.isfinite(selection["point_prediction"]) or selection.get("sign_prediction") != "positive"):
        raise v1.StudyError("invalid selected dose or prediction")
    identity_keys = {"version", "phase", "run_id", "manifest_sha256", "source_commit"}
    required = identity_keys | {"model", "environment", "exposure", "custody", "compatibility", "grid", "controls"}
    if type(metadata) is not dict or set(metadata) != required:
        raise v1.StudyError("development metadata is incomplete")
    prior = manifest["v1_compatibility"]
    if (not _same(metadata["model"], prior["model"]) or not _same(metadata["exposure"], public_exposure) or
            not _same(metadata["custody"], custody_result)):
        raise v1.StudyError("development model, public reservations or custody report differ")
    environment = metadata["environment"]
    if type(environment) is not dict or any(key not in environment for key in edits.ENVIRONMENT_FIELDS):
        raise v1.StudyError("development runtime identity is incomplete")
    if not _same({key: environment[key] for key in edits.ENVIRONMENT_FIELDS}, prior["environment"]):
        raise v1.StudyError("development runtime differs from v1")
    expected_compatibility = {"status": "matching_recorded_v1",
                              "reference_precommit_sha256": prior["precommit_sha256"],
                              "model_state_sha256": prior["model_state_sha256"],
                              "weight_sha256": prior["weight_sha256"],
                              "discovery_logit_sha256": prior["discovery"]["discovery_logit_sha256"],
                              "gradient_delta_sha256": prior["candidates"]["gradient"]["intended_delta_sha256"],
                              "v1_random_delta_sha256": prior["candidates"]["random"]["intended_delta_sha256"]}
    if not _same(metadata["compatibility"], expected_compatibility):
        raise v1.StudyError("development v1 compatibility claim differs")
    original_norm = prior["discovery"]["weight_norm"]
    if type(original_norm) is not float or not math.isfinite(original_norm) or original_norm <= 0:
        raise v1.StudyError("invalid original weight norm")
    grid, controls = metadata["grid"], metadata["controls"]
    if type(grid) is not dict or set(grid) != {name for name, _ in v2.GRID}:
        raise v1.StudyError("development dose attestations are incomplete")
    for name, dose in v2.GRID:
        _attestation(grid[name], dose=dose, original_norm=original_norm)
    if type(controls) is not dict or set(controls) != {"selected", "random", "opposite"}:
        raise v1.StudyError("selected controls are incomplete")
    dose = selection["selected_relative_norm"]
    for value in controls.values():
        _attestation(value, dose=dose, original_norm=original_norm)
    if not _same(controls["selected"], grid[selection["selected_condition"]]):
        raise v1.StudyError("selected edit differs from its grid attestation")
    selected_norm = controls["selected"]["realized_norm"]
    if any(not 0.999 <= controls[name]["realized_norm"] / selected_norm <= 1.001
           for name in ("random", "opposite")):
        raise v1.StudyError("control norms differ from the selected edit")


def prepare_committed_selection(repo, *, manifest_path, manifest_commit, manifest_sha256,
                                custody_path, custody_commit, custody_sha256,
                                development_dir, selection_commit, terminal_sha256):
    """Recompute the committed selection and retain only canonical byte copies.

    The public inventory is checked; custody, import origins and model/control
    reconstruction remain separate obligations of the future owner.
    """
    manifest = manifest_module.verify_manifest(repo, manifest_path, manifest_commit=manifest_commit,
                                               expected_sha256=manifest_sha256)
    public_exposure = exposure.verify_public_reservations(repo)
    custody_result = custody.verify_committed_receipt(repo, manifest_path=manifest_path,
                         manifest_commit=manifest_commit, manifest_sha256=manifest_sha256,
                         receipt_path=custody_path, receipt_commit=custody_commit, receipt_sha256=custody_sha256)
    evidence._require_commit_object(repo, selection_commit)
    directory = evidence._relative(development_dir)
    if (custody_commit == selection_commit or
            evidence._git(repo, "merge-base", "--is-ancestor", custody_commit, selection_commit, check=False).returncode or
            not evidence._path_absent(repo, custody_commit, directory)):
        raise v1.StudyError("committed custody receipt must precede the selection")
    recorded = evidence.verify_committed_development(repo, directory=development_dir,
                    manifest_path=manifest_path, manifest_commit=manifest_commit,
                    selection_commit=selection_commit, manifest_sha256=manifest_sha256,
                    terminal_sha256=terminal_sha256)
    if (recorded.get("status") != "matching_committed_selection" or
            recorded.get("source_commit") != manifest.get("source_commit") or
            recorded.get("manifest_sha256") != manifest_sha256):
        raise v1.StudyError("committed development identity differs from the frozen manifest")
    started = evidence._utc(recorded.get("started_utc"))
    try:
        observations = [custody_result["public_inventory"]["observed_utc"],
                        *(row["observed_utc"] for row in custody_result["observations"])]
    except (KeyError, TypeError) as exc:
        raise v1.StudyError("custody observation times are incomplete") from exc
    if any(evidence._utc(value) > started for value in observations):
        raise v1.StudyError("custody observation is later than the recorded development start")
    selection, metadata = recorded["selection"], recorded["metadata"]
    _check_metadata(manifest, selection, metadata, public_exposure, custody_result)
    return SelectionSnapshot(v1._canonical(manifest), v1._canonical(selection), v1._canonical(metadata),
                             selection_commit, terminal_sha256)


def reconstruct_selected_controls(snapshot, *, reference_bytes, environment, model_metadata,
                                  model_state_sha256, original, v1_gradient, v1_random, discovery):
    """Check a fresh reconstruction against the immutable saved handoff.

    The caller exclusively owns input tensors and must independently validate
    checkpoint/source/custody before invoking any evaluation forward. Returned
    tensors remain caller-owned; the immutable selection is never exposed to a
    callback. The snapshot type itself does not prove it came from
    prepare_committed_selection; callers must use that result in the same
    trusted owner process.
    """
    if type(snapshot) is not SelectionSnapshot:
        raise v1.StudyError("invalid selection snapshot type")
    manifest = v1._decode_document(snapshot.manifest_bytes)
    selection = v1._decode_document(snapshot.selection_bytes)
    metadata = v1._decode_document(snapshot.metadata_bytes)
    try:
        _check_metadata(manifest, selection, metadata, metadata["exposure"], metadata["custody"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise v1.StudyError("invalid frozen selection snapshot") from exc
    compatibility = edits.verify_v1_compatibility(reference_bytes, environment=environment,
                        model_metadata=model_metadata, model_state_sha256=model_state_sha256,
                        original=original, v1_gradient=v1_gradient, v1_random=v1_random, discovery=discovery)
    if not _same(compatibility, metadata["compatibility"]):
        raise v1.StudyError("fresh v1 reconstruction differs from development")
    actual_grid = {}
    for name, dose in v2.GRID:
        delta = edits.construct_dose(original, v1_gradient, dose)
        actual_grid[name] = edits.attest(original, delta, dose)
    if not _same(actual_grid, metadata["grid"]):
        raise v1.StudyError("fresh grid reconstruction differs from development")
    updates, facts = edits.construct_controls(original, v1_gradient, selection["selected_relative_norm"])
    if not _same(facts, metadata["controls"]):
        raise v1.StudyError("fresh selected controls differ from development")
    if manifest.get("version") != v2.VERSION:
        raise v1.StudyError("snapshot manifest version differs")
    return updates
