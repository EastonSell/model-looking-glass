"""One-shot, cooperative custodian for the proposed synthetic-v2 study.

The executor, scorer and source launcher are separate trust boundaries. This
module orders and records their work; it does not establish source provenance,
protect secrets from the same account, or attest model work independently.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

from . import registry
from .causal_v2_custody import canonical_json, commitment
from .causal_v2_evidence_adapter import CELL_CHUNK_LIMIT, CellEvidence
from .v2_evidence import create_evidence


_MODEL_SEEDS = (31201, 31202, 31203, 31204, 31205, 31206)
_CELLS_PER_MODEL = 9
_CASES_PER_MODEL = 16
_EXPECTED_FORWARDS = 57408
_GIB = 1024 ** 3
_MIB = 1024 ** 2
_PUBLIC_LIMITS = {"pre_manifest.json": 512 * 1024,
                  "prediction_freeze.json": 32 * _MIB,
                  "prediction_commitment.json": 16 * 1024,
                  "results.json": 32 * _MIB,
                  "reveal.json": 16 * _MIB,
                  "failure.json": 16 * 1024}
_PUBLIC_RESERVE = 128 * _MIB
_PRIVATE_LIMITS = {"private_material.json": 4 * _MIB,
                   "prediction_record.json": 16 * _MIB,
                   "exposure_record.json": 16 * 1024}
_FAILURE_LIMIT = 64 * 1024


class CustodianError(RuntimeError):
    """The one-shot custodian cannot safely complete an attempt."""


class CommittedSuccessError(CustodianError):
    """Exact success is visible, but acknowledgement or durability is uncertain."""


class ForwardBudget:
    """Single-threaded submission and actual root-forward counter."""
    def __init__(self, limit=60000):
        if type(limit) is not int or not 0 < limit <= 60000:
            raise ValueError("invalid forward limit")
        self.limit = limit
        self.submissions = 0
        self.root_forwards = 0
        self._pending = False
        self._pid = os.getpid()
        self._thread = threading.current_thread()

    def _owner(self):
        if os.getpid() != self._pid or threading.current_thread() is not self._thread:
            raise CustodianError("forward budget belongs to another process or thread")

    def reserve(self):
        self._owner()
        if self._pending or self.submissions >= self.limit:
            raise CustodianError("forward submission is already pending or budget exhausted")
        self.submissions += 1
        self._pending = True

    def root(self):
        self._owner()
        if not self._pending or self.root_forwards >= self.limit:
            raise CustodianError("root forward lacks an unused submission")
        self.root_forwards += 1
        self._pending = False


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data, maximum):
    if type(data) is not bytes or len(data) > maximum:
        raise CustodianError("record is not bounded JSON bytes")
    try:
        value = json.loads(data)
        if canonical_json(value) != data:
            raise ValueError("noncanonical JSON")
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise CustodianError("record is not canonical JSON") from error


def _checked_preparation(prepared, expected_context, exclusion_bytes, pilot_bytes,
                         historical_cases, source_sha256, clarification_sha256,
                         runtime, created_utc):
    # These reviewed builders are separate: no caller-supplied builder can
    # replace preparation or freeze validation through the public interface.
    from .causal_v2_preparation import verify_prepared
    from .causal_v2_freeze import public_pre_manifest
    snapshot = verify_prepared(prepared, expected_context, exclusion_bytes, pilot_bytes,
                               historical_cases)
    manifest = public_pre_manifest(snapshot, source_sha256=source_sha256,
        clarification_sha256=clarification_sha256, runtime=runtime, created_utc=created_utc)
    return snapshot, manifest


def _build_freeze(snapshot, manifest, cells):
    from .causal_v2_freeze import build_freeze
    return build_freeze(snapshot, manifest, cells)


def _verify_discovery(snapshot, cells, chunks):
    # No study can proceed until the separately reviewed binding exists.
    from .causal_v2_freeze import verify_discovery_evidence
    verify_discovery_evidence(snapshot, cells, chunks)


def _read(path, maximum):
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise CustodianError("published evidence exceeds its declared bound")
    return data


def _sync_directory(path):
    # Windows offers no portable directory fsync through this Python API.
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _record(path, name, raw, maximum, expected):
    actual = _read(path / "artifacts" / name, maximum)
    if actual != raw:
        raise CustodianError("published bytes do not match intended bytes")
    expected[name] = {"bytes": len(actual), "sha256": _sha(actual)}


def _terminal_matches(path, state, details, expected):
    try:
        info = registry.verify_run(path)
        return (info["state"] == state
                and canonical_json(info["outcome"]["details"]) == canonical_json(details)
                and info["outcome"]["artifacts"] == expected)
    except Exception:
        return False


def _groups(record, digest, seed, assignment, cases, start):
    """Bind every immutable committed three-site group to its case and cell."""
    rows = record.get("predictions") if type(record) is dict else None
    if type(rows) is not list:
        raise CustodianError("prediction record has no ordered rows")
    result = []
    cursor = start
    for case in cases:
        for strength in (0.25, 0.75):
            group = rows[cursor:cursor + 3]
            if len(group) != 3:
                raise CustodianError("prediction group is incomplete")
            for site, row in enumerate(group, 1):
                if (type(row) is not dict or type(row.get("model_seed")) is not int
                        or row["model_seed"] != seed or row.get("cell_id") != assignment["cell_id"]
                        or row.get("case_id") != case["case_id"]
                        or type(row.get("strength")) is not float or row["strength"] != strength
                        or type(row.get("site")) is not int or row["site"] != site):
                    raise CustodianError("prediction group identity differs")
            result.append({"reference": f"sha256:{digest}:{cursor}:3", "rows": deepcopy(group)})
            cursor += 3
    return result, cursor


def _check_summary(raw, exposure, manifest, freeze, prediction, assignments, heldout, expected):
    """Bind the scorer's public label and inventory to the evidence it read."""
    value = _json(raw, _PUBLIC_LIMITS["results.json"])
    keys = {"schema", "status", "mode", "interpretation", "exposure_classification",
            "pre_manifest_sha256", "freeze_sha256", "prediction_sha256", "models",
            "equal_model_means", "cells", "conditions_count", "conditions", "repair_groups"}
    interpretation = {"unexposed": "confirmation_conditional_on_caller_attestations",
                      "unblinded": "exploratory_unblinded",
                      "unknown": "unconfirmed_exposure_unknown"}[exposure]
    if (type(value) is not dict or set(value) != keys
            or value["schema"] != "synthetic-v2-heldout-summary-v1"
            or value["status"] != "complete" or value["mode"] != "study"
            or value["exposure_classification"] != exposure or value["interpretation"] != interpretation
            or value["pre_manifest_sha256"] != _sha(manifest)
            or value["freeze_sha256"] != _sha(freeze)
            or value["prediction_sha256"] != _sha(prediction)
            or type(value["models"]) is not list or len(value["models"]) != len(_MODEL_SEEDS)
            or type(value["equal_model_means"]) is not dict
            or type(value["cells"]) is not list or len(value["cells"]) != len(assignments)
            or type(value["conditions_count"]) is not int
            or value["conditions_count"] != len(assignments) * _CASES_PER_MODEL * 2
            or type(value["conditions"]) is not list
            or len(value["conditions"]) != value["conditions_count"]
            or type(value["repair_groups"]) is not list or len(value["repair_groups"]) != len(assignments) * 12):
        raise CustodianError("scorer summary schema, identity or exposure classification differs")
    for model, seed in zip(value["models"], _MODEL_SEEDS):
        if type(model) is not dict or type(model.get("model_seed")) is not int or model["model_seed"] != seed:
            raise CustodianError("scorer model order differs")
    for index, (cell, assignment) in enumerate(zip(value["cells"], assignments)):
        name = f"cell-{index:02d}-h.json"
        if (type(cell) is not dict or type(cell.get("model_seed")) is not int
                or cell["model_seed"] != assignment["model_seed"]
                or cell.get("cell_id") != assignment["cell_id"]
                or cell.get("chunk_name") != name or cell.get("chunk_sha256") != expected[name]["sha256"]):
            raise CustodianError("scorer cell identity or chunk hash differs")
        cases = [case for case in heldout["cases"] if case["model_seed"] == assignment["model_seed"]]
        rows = value["conditions"][index * _CASES_PER_MODEL * 2:(index + 1) * _CASES_PER_MODEL * 2]
        for row_index, row in enumerate(rows):
            case_index, strength_index = divmod(row_index, 2)
            if (type(row) is not dict or type(row.get("model_seed")) is not int
                    or row["model_seed"] != assignment["model_seed"]
                    or row.get("cell_id") != assignment["cell_id"]
                    or row.get("case_id") != cases[case_index]["case_id"]
                    or type(row.get("case_index")) is not int or row["case_index"] != case_index
                    or type(row.get("strength")) is not float or row["strength"] != (0.25, 0.75)[strength_index]):
                raise CustodianError("scorer condition identity or order differs")


def _failure(primary, stage, budget, cell_id, phase, outcome, secondary=None):
    message = str(primary)
    if secondary is not None:
        message += f"\nAssociated error ({type(secondary).__name__}): {secondary}"
    result = {"schema": "causal-v2-failure-v1", "outcome": outcome, "stage": stage,
              "error_type": type(primary).__name__, "message": message,
              "submissions": budget.submissions, "root_forwards": budget.root_forwards}
    if cell_id is not None:
        result["cell_id"] = cell_id
    if phase is not None:
        result["phase"] = phase
    return canonical_json(result)


def _raise_primary(primary, cleanup):
    if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
        raise cleanup from primary
    raise primary from cleanup


def _preflight(prepared, expected_context, exclusion_bytes, pilot_bytes, historical_cases,
               source_sha256, clarification_sha256, runtime, created_utc):
    snapshot, manifest = _checked_preparation(prepared, expected_context, exclusion_bytes,
        pilot_bytes, historical_cases, source_sha256, clarification_sha256, runtime, created_utc)
    public = _json(snapshot.public, _PUBLIC_LIMITS["pre_manifest.json"])
    heldout = _json(snapshot.heldout, _PRIVATE_LIMITS["private_material.json"])
    schedule = _json(snapshot.schedule, _PRIVATE_LIMITS["private_material.json"])
    _json(manifest, _PUBLIC_LIMITS["pre_manifest.json"])
    states = public["context"]["model_state_hashes"]
    if public["mode"] != "study" or tuple(x["seed"] for x in states) != _MODEL_SEEDS:
        raise CustodianError("preparation is not the fixed study")
    if (public["heldout_commitment"] != commitment("heldout", snapshot.heldout_root, heldout)
            or public["schedule_commitment"] != commitment("schedule", snapshot.site_nonce, schedule)):
        raise CustodianError("private preparation commitments differ")
    assignments = schedule["assignments"]
    if len(assignments) != len(_MODEL_SEEDS) * _CELLS_PER_MODEL:
        raise CustodianError("schedule does not contain every cell")
    chunk_limits = {f"cell-{i:02d}-{suffix}.json": CELL_CHUNK_LIMIT
                    for i in range(len(assignments)) for suffix in ("d", "h")}
    identities = {"protocol": {"id": "synthetic-causal-v2", "pre_manifest_sha256": _sha(manifest)},
                  "inputs": {"public_preparation_sha256": _sha(snapshot.public)},
                  "model": {"states": states}, "code": {"source_sha256": source_sha256},
                  "metadata": {"scope": "one-shot cooperative custodian"}}
    return snapshot, manifest, public, heldout, schedule, states, assignments, chunk_limits, identities


def run_study_once(path, *, prepared, expected_context, exclusion_bytes, pilot_bytes,
                   historical_cases, source_sha256, clarification_sha256, runtime,
                   created_utc, executor, score, classify_exposure):
    """Execute one checked study attempt through trusted executor/scorer callbacks.

    The caller must establish the separate source/runtime boundary, provide a
    quiescent model executor, keep the private directory private, and supervise
    interruption. Neither this function nor a successful record proves those
    attestations. Failed or uncertain attempts must never be resumed.
    """
    root = Path(path)
    if not root.is_absolute():
        raise ValueError("study path must be absolute")
    if not callable(score) or not callable(classify_exposure):
        raise TypeError("scorer and exposure classifier must be callable")
    if (sum(_PUBLIC_LIMITS.values()) + max(_PUBLIC_LIMITS.values())
            + 2 * registry.MAX_METADATA_BYTES > _PUBLIC_RESERVE):
        raise CustodianError("public registry exceeds the reserved disk envelope")
    budget = ForwardBudget()
    private = public_run = None
    private_expected, public_expected = {}, {}
    private_safe = public_safe = True
    private_terminal = public_terminal = False
    active_cell = None
    stage, phase, cell_id = "initialize", None, None
    executor_closed = False

    def put_public(name, data):
        nonlocal public_safe
        public_safe = False
        public_run.write_artifact(name, data)
        _sync_directory(public_run.path / "artifacts")
        _sync_directory(public_run.path)
        if (public_run.path / ".pending").exists():
            raise CustodianError("public publication left pending evidence")
        _record(public_run.path, name, data, _PUBLIC_LIMITS[name], public_expected)
        public_safe = True

    def put_private(name, data):
        nonlocal private_safe
        private_safe = False
        private.publish_artifact(name, data)
        _record(private.path, name, data, _PRIVATE_LIMITS[name], private_expected)
        private_safe = True

    def save_cell(cell, complete):
        nonlocal private_safe
        private_safe = False
        if complete:
            cell.publish_complete()
        else:
            cell.mark_failed()
            cell.publish_incomplete()
        raw = _read(private.path / "artifacts" / cell.name, CELL_CHUNK_LIMIT)
        info = _json(raw, CELL_CHUNK_LIMIT)
        if info["state"] != ("complete" if complete else "incomplete"):
            raise CustodianError("published cell state differs")
        private_expected[cell.name] = {"bytes": len(raw), "sha256": _sha(raw)}
        private_safe = True

    try:
        snapshot, manifest, public, heldout, schedule, states, assignments, chunk_limits, identities = _preflight(
            prepared, expected_context, exclusion_bytes, pilot_bytes, historical_cases,
            source_sha256, clarification_sha256, runtime, created_utc)
        root.mkdir()
        private = create_evidence(root / "private", **identities, chunk_limits=chunk_limits,
            extra_limits=_PRIVATE_LIMITS, required_extra=list(_PRIVATE_LIMITS),
            failure_limit=_FAILURE_LIMIT, external_reserve_bytes=_PUBLIC_RESERVE,
            disk_limit_bytes=_GIB)
        private_safe = False
        if private.status()["pending_write"]:
            raise CustodianError("private initialization left pending evidence")
        _sync_directory(private.path)
        _sync_directory(root)
        private_safe = True
        public_safe = False
        public_run = registry.create_run(root / "public", **identities,
            artifact_limits=_PUBLIC_LIMITS,
            required_artifacts=[name for name in _PUBLIC_LIMITS if name != "failure.json"],
            max_total_bytes=sum(_PUBLIC_LIMITS.values()))
        _sync_directory(public_run.path)
        _sync_directory(root)
        if registry.verify_run(public_run.path)["pending_write"]:
            raise CustodianError("public initialization left pending evidence")
        public_safe = True
        stage = "pre_manifest"
        put_public("pre_manifest.json", manifest)
        material = canonical_json({"schema": "synthetic-v2-private-material-v1",
            "heldout": heldout, "schedule": schedule, "heldout_root": snapshot.heldout_root.hex(),
            "site_nonce": snapshot.site_nonce.hex(), "prediction_nonce": snapshot.prediction_nonce.hex()})
        put_private("private_material.json", material)
        cells = []
        prediction_raw = freeze_raw = commitment_raw = None
        for phase in ("discovery", "heldout"):
            if phase == "heldout":
                stage = "freeze"
                discovery_reads = 0
                def saved_discovery_chunks():
                    nonlocal discovery_reads
                    for index in range(len(assignments)):
                        name = f"cell-{index:02d}-d.json"
                        raw = _read(private.path / "artifacts" / name, CELL_CHUNK_LIMIT)
                        if {"bytes": len(raw), "sha256": _sha(raw)} != private_expected[name]:
                            raise CustodianError("discovery chunk changed after publication")
                        discovery_reads += 1
                        yield raw
                _verify_discovery(snapshot, cells, saved_discovery_chunks())
                if discovery_reads != len(assignments):
                    raise CustodianError("discovery verification did not consume all chunks")
                freeze_raw, prediction_raw, commitment_raw = _build_freeze(snapshot, manifest, cells)
                freeze = _json(freeze_raw, _PUBLIC_LIMITS["prediction_freeze.json"])
                predictions = _json(prediction_raw, _PRIVATE_LIMITS["prediction_record.json"])
                prediction_sha = _sha(prediction_raw)
                committed = _json(commitment_raw, _PUBLIC_LIMITS["prediction_commitment.json"])
                if (predictions.get("pre_manifest_sha256") != _sha(manifest)
                        or predictions.get("freeze_sha256") != _sha(freeze_raw)
                        or committed.get("pre_manifest_sha256") != _sha(manifest)
                        or committed.get("freeze_sha256") != _sha(freeze_raw)
                        or committed.get("commitment") != commitment("predictions", snapshot.prediction_nonce, predictions)
                        or type(freeze) is not dict):
                    raise CustodianError("freeze or prediction commitment differs")
                # Validate the entire concrete prediction inventory before the
                # first held-out model call, including groups in later cells.
                cursor = 0
                for model_index, seed in enumerate(_MODEL_SEEDS):
                    cases = [case for case in heldout["cases"] if case["model_seed"] == seed]
                    if len(cases) != _CASES_PER_MODEL:
                        raise CustodianError("held-out model case count differs")
                    for local in range(_CELLS_PER_MODEL):
                        assignment = assignments[model_index * _CELLS_PER_MODEL + local]
                        _, cursor = _groups(predictions, prediction_sha, seed, assignment, cases, cursor)
                if cursor != len(predictions["predictions"]):
                    raise CustodianError("prediction record has extra rows")
                put_private("prediction_record.json", prediction_raw)
                put_public("prediction_freeze.json", freeze_raw)
                put_public("prediction_commitment.json", commitment_raw)
                # Recheck all global commitments immediately before any held-out work.
                for name, raw in (("pre_manifest.json", manifest), ("prediction_freeze.json", freeze_raw),
                                  ("prediction_commitment.json", commitment_raw)):
                    _record(public_run.path, name, raw, _PUBLIC_LIMITS[name], public_expected)
                _record(private.path, "prediction_record.json", prediction_raw,
                        _PRIVATE_LIMITS["prediction_record.json"], private_expected)
            rows = public["discovery_cases"] if phase == "discovery" else heldout["cases"]
            prediction_cursor = 0
            for model_index, seed in enumerate(_MODEL_SEEDS):
                model_cases = [case for case in rows if case["model_seed"] == seed]
                if len(model_cases) != _CASES_PER_MODEL:
                    raise CustodianError("model case count differs")
                stage = "model_start"
                executor.begin_model(phase, seed, states[model_index]["sha256"], budget)
                for local in range(_CELLS_PER_MODEL):
                    index = model_index * _CELLS_PER_MODEL + local
                    assignment = assignments[index]
                    if assignment["model_seed"] != seed:
                        raise CustodianError("schedule model order differs")
                    cell_id = assignment["cell_id"]
                    stage = "cell"
                    name = f"cell-{index:02d}-{'d' if phase == 'discovery' else 'h'}.json"
                    active_cell = CellEvidence(private, name=name, cell_id=cell_id, phase=phase,
                                               model_seed=seed, model_state_sha256=states[model_index]["sha256"])
                    if phase == "discovery":
                        selection = executor.run_discovery(deepcopy(assignment), deepcopy(model_cases),
                                                           active_cell, budget)
                        cells.append({"cell_id": cell_id, "model_seed": seed, "selection": selection})
                    else:
                        groups, prediction_cursor = _groups(predictions, prediction_sha, seed, assignment,
                                                            model_cases, prediction_cursor)
                        executor.run_heldout(deepcopy(assignment), deepcopy(model_cases), groups,
                                             active_cell, budget)
                    save_cell(active_cell, True)
                    active_cell = None
                stage = "model_cleanup"
                executor.end_model()
            if phase == "heldout" and prediction_cursor != len(predictions["predictions"]):
                raise CustodianError("prediction record has extra rows")
        stage, phase, cell_id = "cleanup", None, None
        executor_closed = True
        receipt = executor.close()
        if (type(receipt) is not dict
                or set(receipt) != {"input_model_pairing_verified", "private_plan_metadata_verified"}
                or any(value is not True for value in receipt.values())):
            raise CustodianError("executor did not verify model/input and private-plan checks")
        if (budget.submissions != _EXPECTED_FORWARDS or budget.root_forwards != _EXPECTED_FORWARDS
                or budget._pending):
            raise CustodianError("forward count does not match the fixed successful plan")
        stage = "exposure_classification"
        exposure = classify_exposure()
        if (type(exposure) is not dict
                or set(exposure) != {"schema", "classification", "basis"}
                or exposure["schema"] != "synthetic-v2-exposure-v1"
                or type(exposure["classification"]) is not str
                or exposure["classification"] not in ("unexposed", "unblinded", "unknown")
                or type(exposure["basis"]) is not str or not 0 < len(exposure["basis"]) <= 4096):
            raise CustodianError("invalid exposure classification or basis")
        exposure_raw = canonical_json(exposure)
        exposure = _json(exposure_raw, _PRIVATE_LIMITS["exposure_record.json"])
        put_private("exposure_record.json", exposure_raw)
        stage = "scoring"
        heldout_reads = 0
        def saved_heldout_chunks():
            nonlocal heldout_reads
            for index in range(len(assignments)):
                name = f"cell-{index:02d}-h.json"
                raw = _read(private.path / "artifacts" / name, CELL_CHUNK_LIMIT)
                if {"bytes": len(raw), "sha256": _sha(raw)} != private_expected[name]:
                    raise CustodianError("held-out chunk changed after publication")
                heldout_reads += 1
                yield raw
        results = score(manifest, freeze_raw, prediction_raw, commitment_raw, snapshot.heldout,
            snapshot.schedule, snapshot.heldout_root, snapshot.site_nonce, snapshot.prediction_nonce,
            saved_heldout_chunks(), phase_order_verified=True,
            input_model_pairing_verified=receipt["input_model_pairing_verified"],
            private_plan_metadata_verified=receipt["private_plan_metadata_verified"],
            durable_evidence_verified=True,
            case_generation_verified=True, discovery_freeze_verified=True,
            exposure_classification=exposure["classification"])
        if heldout_reads != len(assignments):
            raise CustodianError("scorer did not consume all held-out chunks")
        _check_summary(results, exposure["classification"], manifest, freeze_raw, prediction_raw,
                       assignments, heldout, private_expected)
        put_public("results.json", results)
        reveal = canonical_json({"schema": "synthetic-v2-reveal-v1", "heldout": heldout,
            "schedule": schedule, "heldout_root": snapshot.heldout_root.hex(),
            "site_nonce": snapshot.site_nonce.hex(), "prediction_nonce": snapshot.prediction_nonce.hex(),
            "prediction_record": predictions, "exposure": {
                "schema": "synthetic-v2-exposure-reveal-v1", "classification": exposure["classification"],
                "private_record_sha256": _sha(exposure_raw)}})
        put_public("reveal.json", reveal)
        stage = "private_terminal"
        private_details = {"scope": "private evidence", "submissions": budget.submissions,
                           "root_forwards": budget.root_forwards}
        private_safe = False
        try:
            private.finalize("succeeded", private_details)
        finally:
            private_terminal = _terminal_matches(private.path, "succeeded", private_details, private_expected)
        if not private_terminal:
            raise CustodianError("private terminal state could not be verified")
        stage = "public_terminal"
        details = {"scope": "complete study", "submissions": budget.submissions,
                   "root_forwards": budget.root_forwards,
                   "private_outcome_sha256": _sha(_read(private.path / "outcome.json", registry.MAX_METADATA_BYTES))}
        public_safe = False
        try:
            public_run.finalize("succeeded", details)
            _sync_directory(public_run.path)
        except BaseException as error:
            public_terminal = _terminal_matches(public_run.path, "succeeded", details, public_expected)
            if public_terminal:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise error from CommittedSuccessError("success is visible; acknowledgement or directory durability is uncertain")
                raise CommittedSuccessError("success is visible; acknowledgement or directory durability is uncertain") from error
            raise
        public_terminal = _terminal_matches(public_run.path, "succeeded", details, public_expected)
        if not public_terminal:
            raise CustodianError("public terminal state could not be verified")
        return {"status": "succeeded", "submissions": budget.submissions,
                "root_forwards": budget.root_forwards, "path": str(root)}
    except BaseException as primary:
        if public_terminal:
            raise
        cleanup_error = None
        private_failure_status = "unavailable"
        def remember_cleanup(error):
            nonlocal cleanup_error
            if cleanup_error is None or isinstance(error, (KeyboardInterrupt, SystemExit)):
                cleanup_error = error
        if not executor_closed:
            executor_closed = True
            try:
                executor.close()
            except BaseException as error:
                remember_cleanup(error)
        if (active_cell is not None and private is not None and private_safe
                and active_cell.name not in private_expected):
            try:
                save_cell(active_cell, False)
            except BaseException as error:
                remember_cleanup(error)
        effective = cleanup_error if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)) else primary
        outcome = "cancelled" if isinstance(effective, (KeyboardInterrupt, SystemExit)) else "failed"
        if private is not None and private_safe and not private_terminal:
            try:
                associated = primary if effective is not primary else cleanup_error
                raw = _failure(effective, stage, budget, cell_id, phase, outcome, associated)
                if len(raw) > _FAILURE_LIMIT:
                    private_failure_status = "evidence_limit"
                    raise CustodianError("complete private failure exceeds its evidence limit")
                private_safe = False
                private.publish_failure(raw)
                _record(private.path, "failure.json", raw, _FAILURE_LIMIT, private_expected)
                details = {"scope": "private evidence", "submissions": budget.submissions,
                           "root_forwards": budget.root_forwards}
                try:
                    private.finalize(outcome, details)
                finally:
                    private_terminal = _terminal_matches(private.path, outcome, details, private_expected)
                    if private_terminal:
                        private_failure_status = "complete"
            except BaseException as error:
                remember_cleanup(error)
        if public_run is not None and public_safe and not public_terminal:
            try:
                effective = cleanup_error if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)) else primary
                outcome = "cancelled" if isinstance(effective, (KeyboardInterrupt, SystemExit)) else "failed"
                raw = canonical_json({"schema": "synthetic-v2-public-failure-v1", "outcome": outcome,
                    "stage": stage, "cell_id": cell_id, "phase": phase,
                    "private_failure_evidence": private_failure_status,
                    "submissions": budget.submissions, "root_forwards": budget.root_forwards})
                put_public("failure.json", raw)
                public_safe = False
                details = {"scope": "complete study", "submissions": budget.submissions,
                           "root_forwards": budget.root_forwards}
                try:
                    public_run.finalize(outcome, details)
                    _sync_directory(public_run.path)
                finally:
                    public_terminal = _terminal_matches(public_run.path, outcome, details, public_expected)
            except BaseException as error:
                remember_cleanup(error)
        if cleanup_error is not None:
            _raise_primary(primary, cleanup_error)
        raise
