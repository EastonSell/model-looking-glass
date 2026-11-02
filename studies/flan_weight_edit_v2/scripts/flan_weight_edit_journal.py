"""Bounded evidence journal for supervised FLAN weight-change phases."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


class EvidenceLimit(RuntimeError):
    pass


class JournalFenced(RuntimeError):
    pass


class Journal:
    def __init__(self, directory, *, vocabulary, max_calls, max_bytes, reserve, external_bytes=None):
        if (any(type(x) is not int or x <= 0 for x in (vocabulary, max_calls, max_bytes, reserve))
                or reserve >= max_bytes):
            raise ValueError("invalid journal limits")
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.events = (self.root / "events.jsonl").open("xb")
        self.logits = (self.root / "logits.f32").open("xb")
        self.vocabulary, self.max_calls = vocabulary, max_calls
        self.max_bytes, self.reserve = max_bytes, reserve
        self.calls, self.used, self.event_index = 0, 0, 0
        self.last_record = None
        self.poisoned = False
        self.external_bytes = (lambda: 0) if external_bytes is None else external_bytes
        if not callable(self.external_bytes):
            raise ValueError("external byte accounting must be callable")

    def _within_budget(self, additional):
        external = self.external_bytes()
        if type(external) is not int or external < 0:
            raise ValueError("invalid external byte accounting")
        return self.used + additional + external + self.reserve <= self.max_bytes

    def _event_bytes(self, payload):
        return (json.dumps({"event_index": self.event_index, **payload}, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

    def _write_event(self, raw):
        if not self._within_budget(len(raw)):
            raise EvidenceLimit("journal byte limit reached")
        if self.events.write(raw) != len(raw):
            raise OSError("short journal write")
        self.events.flush()
        os.fsync(self.events.fileno())
        self.used += len(raw)
        self.event_index += 1

    def call(self, case_id, stage, function, *, save_logits, metadata=None):
        if self.poisoned:
            raise JournalFenced("a prior call or write left this journal incomplete")
        try:
            return self._call(case_id, stage, function, save_logits=save_logits, metadata=metadata)
        except BaseException:
            self.poisoned = True
            raise

    def record(self, kind, metadata):
        if self.poisoned:
            raise JournalFenced("a prior call or write left this journal incomplete")
        try:
            if not isinstance(kind, str) or not isinstance(metadata, dict):
                raise ValueError("invalid evidence record")
            self._write_event(self._event_bytes({"kind": kind, "metadata": metadata}))
        except BaseException:
            self.poisoned = True
            raise

    def fence(self):
        self.poisoned = True

    def _call(self, case_id, stage, function, *, save_logits, metadata=None):
        if self.calls >= self.max_calls:
            raise EvidenceLimit("model call limit reached")
        if (not isinstance(case_id, str) or not isinstance(stage, str) or type(save_logits) is not bool
                or (metadata is not None and not isinstance(metadata, dict))):
            raise ValueError("invalid call identity")
        metadata = {} if metadata is None else metadata
        index = self.calls
        self.calls += 1
        self._write_event(self._event_bytes({"kind": "forward_started", "call": index,
                                             "case_id": case_id, "stage": stage, "metadata": metadata}))
        value = function()
        record = {"kind": "forward_complete", "call": index, "case_id": case_id,
                  "stage": stage, "metadata": metadata}
        if save_logits:
            if hasattr(value, "detach"):
                if value.device.type != "cpu":
                    raise ValueError("saved logits must come from CPU")
                value_array = value.detach().numpy()
            else:
                value_array = np.asarray(value)
            if (value_array.dtype != np.float32 or value_array.ndim not in (1, 2)
                    or value_array.shape[-1] != self.vocabulary or value_array.size == 0
                    or not np.isfinite(value_array).all()):
                raise ValueError("saved logits must be finite float32 full-vocabulary rows")
            raw = np.ascontiguousarray(value_array, dtype="<f4").tobytes()
            record.update({"offset": self.logits.tell(), "bytes": len(raw),
                           "shape": list(value_array.shape), "sha256": hashlib.sha256(raw).hexdigest()})
            event = self._event_bytes(record)
            if not self._within_budget(len(raw) + len(event)):
                raise EvidenceLimit("logit byte limit reached")
            if self.logits.write(raw) != len(raw):
                raise OSError("short logit write")
            self.logits.flush()
            os.fsync(self.logits.fileno())
            self.used += len(raw)
            self._write_event(event)
        else:
            self._write_event(self._event_bytes(record))
        self.last_record = record.copy()
        return value

    def close(self):
        self.events.close()
        self.logits.close()
