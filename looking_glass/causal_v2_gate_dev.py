"""Development-only phase-ordering exercise; this cannot run a v2 study.

It accepts only fake identifiers and a no-argument callback. It does not
generate cases, validate real v2 schemas, isolate a secret, or prevent arbitrary
code in a callback from bypassing this cooperative object.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading

from . import registry
from .causal_v2_custody import canonical_json, commitment


_CELL = re.compile(r"dev-cell-[0-9]{2}\Z")
_CASE = re.compile(r"dev-case-[0-9]{2}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MODE = "synthetic-v2-phase-gate-development-only"
_CREATE_TOKEN = object()


class DevelopmentCommittedCleanupError(RuntimeError):
    """The exact success was published before a later cleanup error."""


@dataclass(frozen=True)
class DevelopmentSeal:
    nonce: bytes
    prediction_record: bytes
    public_sha256: str


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _identifiers(values, pattern, limit):
    if (type(values) is not list or not 1 <= len(values) <= limit
            or any(type(item) is not str or not pattern.fullmatch(item) for item in values)
            or values != sorted(set(values))):
        raise ValueError("expected sorted, unique development-only identifiers")


class DevelopmentGate:
    """One-process, one-thread fake gate with immutable registry evidence.

    A crash, publication error or callback failure invalidates this object;
    there is deliberately no API to resume the same attempt. This prototype
    has no real study inputs and does not enforce isolation against arbitrary
    callbacks or noncooperating same-account processes.
    """

    def __init__(self, run, public, *, _token=None):
        if _token is not _CREATE_TOKEN:
            raise RuntimeError("development attempts cannot be reopened")
        self.run = run
        self.public = public
        self.public_raw = canonical_json(public)
        self.public_sha256 = _digest(self.public_raw)
        self._pid = os.getpid()
        self._thread = threading.current_thread()
        self._stopped = False
        self._busy = False
        self._violated = False
        self._success_intent = None

    @classmethod
    def create(cls, path, *, cells, case_ids, exclusion_sha256):
        _identifiers(cells, _CELL, 54)
        _identifiers(case_ids, _CASE, 16)
        if type(exclusion_sha256) is not str or not _HASH.fullmatch(exclusion_sha256):
            raise ValueError("exclusion identity must be a lowercase SHA-256")
        public = {"mode": _MODE, "cells": list(cells), "case_ids": list(case_ids),
                  "exclusion_sha256": exclusion_sha256}
        limits = {"public_manifest.json": 16384, "freeze.json": 262144,
                  "prediction_record.json": 262144, "prediction_commitment.json": 4096, "evaluation_claim.json": 4096,
                  "evaluation_result.json": 4096}
        for index in range(len(cells)):
            limits[f"discovery_{index:02d}.json"] = 4096
        run = registry.create_run(path, protocol={"id": _MODE}, inputs={"public": public},
                                  model={"kind": "fake"}, code={"kind": "development-only"},
                                  metadata={"mode": _MODE}, artifact_limits=limits,
                                  required_artifacts=list(limits), max_total_bytes=1024 * 1024)
        phase = run.path / ".phase"
        try:
            phase.mkdir()
            (phase / ".lock").touch(exist_ok=False)
            run.write_artifact("public_manifest.json", canonical_json(public))
        except BaseException as error:
            try:
                run.finalize("cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                             {"stage": "development-setup"})
            except (KeyboardInterrupt, SystemExit) as interrupt:
                raise interrupt from error
            except BaseException:
                pass
            raise
        return cls(run, public, _token=_CREATE_TOKEN)

    def __reduce__(self):
        raise TypeError("a development gate cannot be serialized or resumed")

    def _check_owner(self):
        if os.getpid() != self._pid or threading.current_thread() is not self._thread:
            self._violated = True
            raise RuntimeError("development gate owner changed")
        if self._busy:
            self._violated = True
            raise RuntimeError("development attempt is busy")
        if self._stopped or self._violated:
            raise RuntimeError("development attempt is stopped or busy")

    def _terminal_failure(self, error):
        self._stopped = True
        desired = "cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        details = {"stage": "development-attempt"}
        try:
            status = self.run.status()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as failure:
            raise RuntimeError("development terminal state is unknown") from failure
        if status["state"] != "incomplete":
            raise RuntimeError("development terminal state is unknown")
        intended_artifacts = status["artifacts"]
        try:
            self.run.finalize(desired, details)
        except BaseException as terminal_error:
            try:
                after = self.run.status()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as failure:
                raise RuntimeError("development terminal state is unknown") from failure
            if (after["state"] != desired or after["outcome"]["details"] != details
                    or after["outcome"]["artifacts"] != intended_artifacts):
                raise RuntimeError("development terminal state is unknown") from terminal_error
            if isinstance(terminal_error, (KeyboardInterrupt, SystemExit)):
                raise terminal_error

    def _transition(self, action):
        self._check_owner()
        body_error = None
        try:
            self._busy = True
            with registry._lock(self.run.path / ".phase", self.run.lock_timeout):
                try:
                    return action()
                except BaseException as error:
                    body_error = error
                    raise
        except BaseException as outer_error:
            fresh_interrupt = (body_error is not None and outer_error is not body_error
                               and isinstance(outer_error, (KeyboardInterrupt, SystemExit)))
            primary = outer_error if fresh_interrupt or body_error is None else body_error
            if isinstance(primary, DevelopmentCommittedCleanupError):
                self._stopped = True
                if primary is not outer_error:
                    raise primary from outer_error
                raise
            if self._success_intent is not None:
                try:
                    status = self.run.status()
                    details, artifacts = self._success_intent
                    if (status["state"] == "succeeded" and status["outcome"]["details"] == details
                            and status["outcome"]["artifacts"] == artifacts):
                        self._stopped = True
                        if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                            raise primary from DevelopmentCommittedCleanupError(
                                "development evaluation committed as succeeded before interruption")
                        raise DevelopmentCommittedCleanupError(
                            "development evaluation committed as succeeded; cleanup failed") from primary
                except DevelopmentCommittedCleanupError:
                    raise
                except (KeyboardInterrupt, SystemExit) as interrupt:
                    if interrupt is primary:
                        raise
                    self._stopped = True
                    raise interrupt from primary
                except BaseException as verify_error:
                    self._stopped = True
                    unknown = RuntimeError("development terminal state is unknown")
                    unknown.__cause__ = verify_error
                    raise primary from unknown
            try:
                self._terminal_failure(primary)
            except (KeyboardInterrupt, SystemExit) as interrupt:
                if interrupt is primary:
                    raise
                raise interrupt from primary
            except BaseException as failure:
                raise primary from failure
            if fresh_interrupt:
                raise primary from body_error
            if primary is not outer_error:
                if outer_error.__context__ is primary:
                    outer_error.__context__ = None
                raise primary from outer_error
            raise
        finally:
            self._busy = False

    def _status(self, names):
        status = self.run.status()
        if (status["state"] != "incomplete" or status["manifest"]["metadata"].get("mode") != _MODE
                or set(status["artifacts"]) != set(names)):
            raise RuntimeError("development phase inventory is invalid")
        return status

    def _read(self, name):
        limit = self.run.status()["manifest"]["artifact_limits"][name]
        with (self.run.path / "artifacts" / name).open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise RuntimeError("development artifact exceeds its bound")
        try:
            value = json.loads(raw)
            if canonical_json(value) != raw:
                raise ValueError("noncanonical artifact")
            return value, raw
        except (ValueError, UnicodeError, RecursionError) as error:
            raise RuntimeError("invalid development artifact") from error

    def _receipt_names(self):
        return [f"discovery_{index:02d}.json" for index in range(len(self.public["cells"]))]

    def record_discovery(self, cell_id, score):
        def action():
            status = self.run.status()
            if status["state"] != "incomplete" or "freeze.json" in status["artifacts"]:
                raise RuntimeError("discovery is not active")
            if cell_id not in self.public["cells"]:
                raise ValueError("unknown development cell")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("development score must be finite")
            index = self.public["cells"].index(cell_id)
            self.run.write_artifact(f"discovery_{index:02d}.json",
                                    canonical_json({"cell_id": cell_id, "score": float(score)}))
        return self._transition(action)

    def _freeze_record(self):
        names = ["public_manifest.json"] + self._receipt_names()
        self._status(names)
        public, raw = self._read("public_manifest.json")
        if raw != self.public_raw or public != self.public:
            raise RuntimeError("development manifest changed")
        receipts = []
        for name, expected in zip(self._receipt_names(), self.public["cells"]):
            value, raw = self._read(name)
            if (type(value) is not dict or set(value) != {"cell_id", "score"}
                    or value["cell_id"] != expected or type(value["score"]) is not float
                    or not math.isfinite(value["score"])):
                raise RuntimeError("invalid development discovery receipt")
            receipts.append({"cell_id": expected, "score": value["score"], "sha256": _digest(raw)})
        return {"mode": _MODE, "public_manifest_sha256": self.public_sha256, "receipts": receipts}

    def freeze(self):
        def action():
            raw = canonical_json(self._freeze_record())
            self.run.write_artifact("freeze.json", raw)
            return _digest(raw)
        return self._transition(action)

    def _checked_freeze(self, extra=()):
        names = ["public_manifest.json"] + self._receipt_names() + ["freeze.json"] + list(extra)
        self._status(names)
        public, public_raw = self._read("public_manifest.json")
        if public_raw != self.public_raw or public != self.public:
            raise RuntimeError("development manifest changed")
        freeze, freeze_raw = self._read("freeze.json")
        if (type(freeze) is not dict or set(freeze) != {"mode", "public_manifest_sha256", "receipts"}
                or freeze["mode"] != _MODE or freeze["public_manifest_sha256"] != self.public_sha256
                or type(freeze["receipts"]) is not list or len(freeze["receipts"]) != len(self.public["cells"])):
            raise RuntimeError("invalid development freeze")
        for item, name, expected in zip(freeze["receipts"], self._receipt_names(), self.public["cells"]):
            receipt, raw = self._read(name)
            if (type(item) is not dict or set(item) != {"cell_id", "score", "sha256"}
                    or item["cell_id"] != expected or item["sha256"] != _digest(raw)
                    or type(item["score"]) is not float
                    or raw != canonical_json({"cell_id": expected, "score": item["score"]})):
                raise RuntimeError("development freeze no longer matches discovery")
        return freeze, freeze_raw

    def seal_predictions(self, nonce):
        def action():
            freeze, raw = self._checked_freeze()
            rows = [{"cell_id": item["cell_id"], "case_id": case_id, "prediction": item["score"]}
                    for item in freeze["receipts"] for case_id in self.public["case_ids"]]
            record = {"pre_manifest_sha256": self.public_sha256, "freeze_sha256": _digest(raw),
                      "predictions": rows}
            sealed = commitment("predictions", nonce, record)
            record_raw = canonical_json(record)
            published = canonical_json({"mode": _MODE, "pre_manifest_sha256": self.public_sha256,
                                        "freeze_sha256": _digest(raw), "prediction_record_sha256": _digest(record_raw),
                                        "commitment": sealed})
            # These are fake development predictions; real held-out predictions
            # must instead be durably retained in the private custodian record.
            self.run.write_artifact("prediction_record.json", record_raw)
            self.run.write_artifact("prediction_commitment.json", published)
            return DevelopmentSeal(nonce, record_raw, _digest(published))
        return self._transition(action)

    def evaluate(self, seal, callback):
        def action():
            if (type(seal) is not DevelopmentSeal or not callable(callback)
                    or type(seal.nonce) is not bytes or len(seal.nonce) != 32
                    or type(seal.prediction_record) is not bytes or len(seal.prediction_record) > 262144
                    or type(seal.public_sha256) is not str or not _HASH.fullmatch(seal.public_sha256)):
                raise ValueError("a development seal and callback are required")
            # Verify the exact frozen records before the one-shot evaluation claim.
            freeze, freeze_raw = self._checked_freeze(extra=("prediction_record.json", "prediction_commitment.json"))
            _, saved_record = self._read("prediction_record.json")
            commitment_record, commit_raw = self._read("prediction_commitment.json")
            if _digest(commit_raw) != seal.public_sha256 or saved_record != seal.prediction_record:
                raise RuntimeError("development commitment changed")
            try:
                record = json.loads(seal.prediction_record)
            except (ValueError, UnicodeError, RecursionError) as error:
                raise ValueError("invalid development prediction bytes") from error
            if canonical_json(record) != seal.prediction_record:
                raise ValueError("noncanonical development prediction bytes")
            expected_rows = [{"cell_id": item["cell_id"], "case_id": case_id, "prediction": item["score"]}
                             for item in freeze["receipts"] for case_id in self.public["case_ids"]]
            if (record.get("pre_manifest_sha256") != self.public_sha256
                    or record.get("freeze_sha256") != _digest(freeze_raw)
                    or canonical_json(record.get("predictions")) != canonical_json(expected_rows)
                    or commitment_record != {"mode": _MODE, "pre_manifest_sha256": self.public_sha256,
                                              "freeze_sha256": _digest(freeze_raw),
                                              "prediction_record_sha256": _digest(saved_record),
                                              "commitment": commitment("predictions", seal.nonce, record)}):
                raise RuntimeError("development prediction binding mismatch")
            claim_raw = canonical_json({"mode": _MODE,
                "public_manifest_sha256": self.public_sha256, "freeze_sha256": _digest(freeze_raw),
                "prediction_commitment_sha256": seal.public_sha256})
            self.run.write_artifact("evaluation_claim.json", claim_raw)
            callback()
            if self._violated:
                raise RuntimeError("invalid reentrant development transition")
            _, checked_freeze = self._checked_freeze(extra=("prediction_record.json", "prediction_commitment.json", "evaluation_claim.json"))
            _, checked_record = self._read("prediction_record.json")
            _, checked_commit = self._read("prediction_commitment.json")
            _, checked_claim = self._read("evaluation_claim.json")
            if (checked_freeze != freeze_raw or checked_record != saved_record
                    or checked_commit != commit_raw or checked_claim != claim_raw):
                raise RuntimeError("development phase evidence changed during evaluation")
            self.run.write_artifact("evaluation_result.json", canonical_json({"mode": _MODE, "callback_returned": True}))
            before = self.run.status()
            details = {"stage": "development-evaluation"}
            self._success_intent = (details, before["artifacts"])
            self.run.finalize("succeeded", details)
            self._stopped = True
        return self._transition(action)
