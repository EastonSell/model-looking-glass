"""Checkpoint row candidates and scoped mutations for the fixed study."""

from contextlib import contextmanager
import hashlib
import json
import math

import numpy as np
import torch

import flan_weight_edit_guard as guard
import flan_weight_edit_numerical as science
import flan_weight_edit_study as study
import flan_weight_edit_worker as worker

TOKEN_IDS = {"wonderful": 1627, "awful": 16242, "excellent": 1287,
             "positive": 1465, "negative": 2841}


class Session:
    def __init__(self, model, tokenizer, journal, probe_case):
        self.model, self.tokenizer, self.journal, self.probe_case = model, tokenizer, journal, probe_case
        self.base_state = guard.state_hash(model)
        self.base_topology = guard.topology(model)
        self.base_modes = tuple((name, id(module), module.training)
                                for name, module in model.named_modules(remove_duplicate=False))
        self.rows = {}
        self.receipt = None
        self.reference_probe = None
        self._candidate_bytes = None
        self._receipt_sha = None
        self.last_probe_record = None
        self._active = None

    def _validate_topology(self):
        shared = self.model.shared.weight
        head = self.model.lm_head.weight
        if (self.model.encoder.embed_tokens.weight is not shared
                or self.model.decoder.embed_tokens.weight is not shared or head is shared
                or shared.dtype != torch.float32 or head.dtype != torch.float32
                or shared.device.type != "cpu" or head.device.type != "cpu"
                or shared.ndim != 2 or head.ndim != 2 or shared.shape[1] != 512 or head.shape[1] != 512):
            raise ValueError("checkpoint parameter topology differs")
        a, b = shared.untyped_storage(), head.untyped_storage()
        if max(a.data_ptr(), b.data_ptr()) < min(a.data_ptr() + a.nbytes(), b.data_ptr() + b.nbytes()):
            raise ValueError("shared and output-head storage overlap")

    def _validate_live_topology(self):
        self._validate_topology()
        if guard.topology(self.model) != self.base_topology:
            raise ValueError("model alias topology changed")
        modes = tuple((name, id(module), module.training)
                      for name, module in self.model.named_modules(remove_duplicate=False))
        if modes != self.base_modes or any(mode for _, _, mode in modes):
            raise ValueError("model training mode or module identity changed")
        if self._active is not None:
            parameter, index, expected_sha, _ = self._active
            if hashlib.sha256(parameter[index].detach().numpy().tobytes()).hexdigest() != expected_sha:
                raise ValueError("active candidate row changed")

    def _validate_case_state(self):
        self._validate_live_topology()
        if self._active is None:
            if guard.state_hash(self.model) != self.base_state:
                raise ValueError("clean model state changed")
        else:
            parameter, index, _, expected_unchanged = self._active
            if guard.state_hash(self.model, skip=(parameter, index)) != expected_unchanged:
                raise ValueError("unedited model state changed")

    def prepare(self, *, expected_receipt=None, expected_probe=None, before_probe=None):
        try:
            self._validate_live_topology()
            if guard.state_hash(self.model) != self.base_state or guard.topology(self.model) != self.base_topology:
                raise ValueError("model state changed before candidate construction")

            def construct():
                shared, head = self.model.shared.weight, self.model.lm_head.weight
                def get(parameter, row):
                    return parameter[row].detach().numpy().copy()
                wonderful = get(shared, TOKEN_IDS["wonderful"])
                awful = get(shared, TOKEN_IDS["awful"])
                excellent = get(shared, TOKEN_IDS["excellent"])
                negative = get(head, TOKEN_IDS["negative"])
                positive = get(head, TOKEN_IDS["positive"])
                try:
                    directions = science.directions(wonderful, awful, excellent, negative, positive)
                except science.DirectionIneligible as exc:
                    return {"status": "ineligible", "reason": "invalid-direction", "detail": str(exc)}
                self.rows[("primary", 1.0)] = science.donor_row(awful)
                self.rows[("same", 1.0)] = science.donor_row(excellent)
                for name in study.CONTROLS:
                    original = negative if name == "broad" else wonderful
                    for dose in study.DOSES:
                        self.rows[(name, dose)] = science.construct_row(original, directions[name], dose)
                entries = []
                for (name, dose), row in self.rows.items():
                    original = negative if name == "broad" else wonderful
                    direction = directions[{"primary": "target", "same": "same"}.get(name, name)]
                    realized = science.norm([math.fsum((float(a), -float(b))) for a, b in zip(row, original)])
                    raw = row.astype("<f4", copy=False).tobytes()
                    donor = positive if name == "broad" else excellent if name == "same" else awful
                    donor_token = TOKEN_IDS["positive"] if name == "broad" else TOKEN_IDS["excellent"] if name == "same" else TOKEN_IDS["awful"]
                    entries.append({"arm": name, "dose": dose, "sha256": hashlib.sha256(raw).hexdigest(),
                                    "original_sha256": hashlib.sha256(original.astype("<f4", copy=False).tobytes()).hexdigest(),
                                    "donor_sha256": hashlib.sha256(donor.astype("<f4", copy=False).tobytes()).hexdigest(),
                                    "donor_token": donor_token,
                                    "donor_storage": "lm_head" if name == "broad" else "shared",
                                    "bytes_hex": raw.hex(), "intended_norm": science.norm(direction) * dose,
                                    "realized_norm": realized})
                random_raw = directions["random"].astype("<f8", copy=False).tobytes()
                return {"state_sha256": self.base_state, "entries": entries,
                        "random_direction_sha256": hashlib.sha256(random_raw).hexdigest(),
                        "random_direction_hex": random_raw.hex()}

            self.receipt = worker._checked(self.journal, "candidate-construction", construct)
            if self.receipt.get("status") == "ineligible":
                self.journal.record("ineligible", self.receipt)
                return self.receipt.copy()
            raw = json.dumps(self.receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            receipt_sha = hashlib.sha256(raw).hexdigest()
            if expected_receipt is not None and receipt_sha != expected_receipt:
                raise ValueError("candidate receipt differs from frozen development")
            self._receipt_sha = receipt_sha
            self._candidate_bytes = {(item["arm"], item["dose"]): bytes.fromhex(item["bytes_hex"])
                                     for item in self.receipt["entries"]}
            self.journal.record("frozen_candidates", {"sha256": receipt_sha, "receipt": self.receipt})
            if before_probe is not None:
                before_probe()
            self.reference_probe = self._probe()
            if expected_probe is not None and self.reference_probe != expected_probe:
                raise ValueError("clean reference probe differs from frozen development")
            return {"candidate_sha256": receipt_sha, "probe_sha256": hashlib.sha256(self.reference_probe).hexdigest()}
        except BaseException:
            self.journal.fence()
            raise

    def _probe(self):
        self._validate_live_topology()
        ids = torch.tensor([self.probe_case["input_ids"]], dtype=torch.long)
        attention = torch.tensor([self.probe_case["attention_mask"]], dtype=torch.long)
        decoder = torch.tensor([[0]], dtype=torch.long)
        with torch.inference_mode():
            logits = worker._checked(self.journal, "clean-probe", lambda:
                self.journal.call(self.probe_case["id"], "clean-probe", lambda:
                    self.model(input_ids=ids, attention_mask=attention, decoder_input_ids=decoder,
                               decoder_attention_mask=torch.ones_like(decoder), use_cache=False,
                               return_dict=True).logits[0, -1].contiguous(),
                    save_logits=True, metadata={"arm": "clean-probe", "decoder_prefix": [0]}))
        self._validate_live_topology()
        return logits.detach().numpy().astype("<f4", copy=False).tobytes()

    def _check_candidates(self):
        if self._candidate_bytes is None or set(self.rows) != set(self._candidate_bytes):
            raise ValueError("candidate inventory differs from the frozen receipt")
        raw = json.dumps(self.receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != self._receipt_sha:
            raise ValueError("candidate inventory receipt changed")
        for key, expected in self._candidate_bytes.items():
            row = self.rows[key]
            if not isinstance(row, np.ndarray) or row.dtype != np.float32 or row.tobytes() != expected:
                raise ValueError("candidate inventory differs from the frozen receipt")

    @contextmanager
    def arm(self, name, dose):
        try:
            if self.reference_probe is None or (name, dose) not in self.rows:
                raise ValueError("arm is not among the frozen candidates")
            self._validate_live_topology()
            self._check_candidates()
            parameter = self.model.lm_head.weight if name == "broad" else self.model.shared.weight
            index = TOKEN_IDS["negative"] if name == "broad" else TOKEN_IDS["wonderful"]
            entry = next(item for item in self.receipt["entries"] if (item["arm"], item["dose"]) == (name, dose))
            original_sha = hashlib.sha256(parameter[index].detach().numpy().tobytes()).hexdigest()
            donor_parameter = self.model.lm_head.weight if entry["donor_storage"] == "lm_head" else self.model.shared.weight
            donor_sha = hashlib.sha256(donor_parameter[entry["donor_token"]].detach().numpy().tobytes()).hexdigest()
            if original_sha != entry["original_sha256"] or donor_sha != entry["donor_sha256"]:
                raise ValueError("original or donor checkpoint row changed")
            before_buffer = guard.buffer_hash(self.model)
            before_topology = guard.topology(self.model)
            unchanged = guard.state_hash(self.model, skip=(parameter, index))
            self.journal.record("arm_started", {"arm": name, "dose": dose, "original_sha256": original_sha,
                                                "donor_sha256": donor_sha, "state_sha256": self.base_state,
                                                "buffer_sha256": before_buffer})
            with guard.temporary_row(self.model, parameter, index, self.rows[(name, dose)],
                                     expected_state=self.base_state, expected_probe=self.reference_probe,
                                     restored_probe=self._probe):
                installed_sha = hashlib.sha256(parameter[index].detach().numpy().tobytes()).hexdigest()
                if installed_sha != entry["sha256"]:
                    raise ValueError("installed row differs from the frozen candidate")
                self.journal.record("arm_installed", {"arm": name, "dose": dose, "sha256": installed_sha})
                self._active = (parameter, index, installed_sha, unchanged)
                try:
                    yield
                finally:
                    self._active = None
            after_state = guard.state_hash(self.model)
            after_buffer = guard.buffer_hash(self.model)
            self._validate_live_topology()
            restored_sha = hashlib.sha256(parameter[index].detach().numpy().tobytes()).hexdigest()
            if (after_state != self.base_state or after_buffer != before_buffer
                    or guard.topology(self.model) != before_topology or restored_sha != original_sha):
                raise guard.RestorationError("arm completion state differs")
            self.journal.record("arm_completed", {"arm": name, "dose": dose, "installed_sha256": installed_sha,
                                                   "restored_row_sha256": restored_sha,
                                                   "pre_buffer_sha256": before_buffer,
                                                   "post_restore_state_sha256": after_state,
                                                   "post_probe_buffer_sha256": after_buffer,
                                                   "topology_unchanged": True,
                                                   "probe_span": self.journal.last_record.copy()})
        except BaseException:
            self.journal.fence()
            raise

    def evaluate(self, arm, case):
        try:
            if self.reference_probe is None:
                raise ValueError("session is not prepared")
            self._validate_case_state()
            outcome = worker.evaluate_case(self.model, self.tokenizer, case, self.journal, arm,
                                           labels={"positive": [1465, 1], "negative": [2841, 1]},
                                           validate=self._validate_live_topology)
            self._validate_case_state()
            self.journal.record("case_complete", {"arm": arm, "outcome": outcome})
            return outcome
        except BaseException:
            self.journal.fence()
            raise

    def score(self, name, function):
        try:
            return worker._checked(self.journal, name, function)
        except BaseException:
            self.journal.fence()
            raise
