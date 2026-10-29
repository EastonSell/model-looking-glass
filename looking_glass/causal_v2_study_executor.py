"""Private model callback for the proposed synthetic-v2 custodian.

This module does not prepare cases, authorize a study, publish a premanifest,
enforce secrecy, or supply a source-verified launcher. The separate custodian
must establish phase order and keep the arguments and evidence private. A
callback is a cooperative execution component, not an independent attestation.
"""
from copy import deepcopy
import json
import math
import os
import re
import threading

import torch

from .causal_v2_custodian_run import ForwardBudget
from .causal_v2_custody import canonical_json
from .causal_v2_evidence_encoder import append_executor_result
from .causal_v2_executor import (_STUDY_CACHE_TOKEN, _check_hooks, _check_model,
                                 _prepare_clean, _prepare_target, _run_active,
                                 input_sha256)
from .causal_v2_model_identity import state_sha256
from .causal_v2_runtime import NumericPolicy
from .causal_v2_selection import DiscoveryCase, freeze_discovery
from .model import TinyResidual


_MODEL_SEEDS = (31201, 31202, 31203, 31204, 31205, 31206)
_CASE_COUNT = 16
_CELLS_PER_MODEL = 9
_NODES = tuple(f"block.{block}.{op}" for block in (1, 2, 3)
               for op in ("branch.0", "activation", "merge"))
_STRENGTHS = (0.25, 0.75)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REFERENCE = re.compile(r"sha256:([0-9a-f]{64}):(0|[1-9][0-9]*):3\Z")
_PREDICTION_FIELDS = {"model_seed", "cell_id", "case_id", "strength", "site", "polarity",
                      "polarity_prediction", "constant_prediction", "zero_prediction"}
_MAX_SELECTION_BYTES = 512 * 1024


class StudyExecutionError(ValueError):
    """A private callback, model or supplied study record failed validation."""


def _study_input_check(value, policy):
    policy.verify()
    input_sha256(value)  # Validates shape, representation and finite float32 bits.
    values = value.tolist()[0]
    if (not 1 <= abs(values[0] * 16) <= 16 or not (values[0] * 16).is_integer()
            or any(not -16 <= item * 16 <= 15 or not (item * 16).is_integer()
                   for item in values[1:])):
        raise StudyExecutionError("input is outside the fixed synthetic domain")
    policy.verify()


def _case_values(case, phase, seed, index):
    suffix = "d" if phase == "discovery" else "h"
    if (type(case) is not dict or set(case) != {"case_id", "model_seed", "split", "index", "values"}
            or case["case_id"] != f"v2-m{seed}-{suffix}{index:02d}"
            or type(case["model_seed"]) is not int or case["model_seed"] != seed
            or case["split"] != phase or type(case["index"]) is not int or case["index"] != index
            or type(case["values"]) is not list or len(case["values"]) != 8
            or any(type(value) is not float or not math.isfinite(value) for value in case["values"])):
        raise StudyExecutionError("invalid ordered case record")
    values = case["values"]
    if (not 1 <= abs(values[0] * 16) <= 16 or not (values[0] * 16).is_integer()
            or (values[0] > 0) != (index % 2 == 0)
            or any(not -16 <= item * 16 <= 15 or not (item * 16).is_integer()
                   for item in values[1:])):
        raise StudyExecutionError("case values or polarity are outside the protocol")
    return tuple(values)


def _tensor_from_values(values):
    return torch.tensor([values], dtype=torch.float32, device="cpu")


def _bounded_json(value, maximum):
    """Check a finite upper bound before constructing canonical JSON bytes."""
    stack = [(value, 0)]
    estimate = nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 32 or nodes > 60000:
            raise StudyExecutionError("JSON evidence is too deep or complex")
        if type(item) is dict:
            if len(item) > 64 or any(type(key) is not str for key in item):
                raise StudyExecutionError("JSON mapping is too large or has invalid keys")
            estimate += 2 + 2 * len(item)
            for key, child in item.items():
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
        elif type(item) in (list, tuple):
            if len(item) > 256:
                raise StudyExecutionError("JSON sequence is too large")
            estimate += 2 + len(item)
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > 256 or not item.isascii():
                raise StudyExecutionError("JSON text is too large or non-ASCII")
            estimate += len(json.dumps(item, ensure_ascii=True))
        elif item is None or type(item) is bool:
            estimate += 5
        elif type(item) is int:
            if item.bit_length() > 64:
                raise StudyExecutionError("JSON integer is too large")
            estimate += 32
        elif type(item) is float:
            if not math.isfinite(item):
                raise StudyExecutionError("JSON number is nonfinite")
            estimate += 32
        else:
            raise StudyExecutionError("unsupported JSON evidence value")
        if estimate > maximum:
            raise StudyExecutionError("JSON evidence exceeds its byte bound")
    raw = canonical_json(value)
    if len(raw) > maximum:
        raise StudyExecutionError("JSON evidence exceeds its byte bound")
    return raw


def _bounded_selection(selection):
    if type(selection) is not dict:
        raise StudyExecutionError("selection must be a mapping")
    return json.loads(_bounded_json(selection, _MAX_SELECTION_BYTES))


def _prediction_group(group, case, assignment, strength):
    if (type(group) is not dict or set(group) != {"reference", "rows"}
            or type(group["reference"]) is not str or _REFERENCE.fullmatch(group["reference"]) is None
            or type(group["rows"]) is not list or len(group["rows"]) != 3):
        raise StudyExecutionError("invalid committed prediction group")
    for site, row in enumerate(group["rows"], 1):
        if (type(row) is not dict or set(row) != _PREDICTION_FIELDS
                or type(row["model_seed"]) is not int or row["model_seed"] != case["model_seed"]
                or row["cell_id"] != assignment["cell_id"] or row["case_id"] != case["case_id"]
                or type(row["strength"]) is not float or row["strength"] != strength
                or type(row["site"]) is not int or row["site"] != site
                or type(row["polarity"]) is not int or row["polarity"] != (1 if case["index"] % 2 == 0 else -1)
                or any(type(row[name]) is not float or not math.isfinite(row[name])
                       for name in ("polarity_prediction", "constant_prediction", "zero_prediction"))
                or row["zero_prediction"] != 0.0):
            raise StudyExecutionError("committed prediction identity or value differs")
    return group["reference"]


def _prediction_refs(groups, cases, assignment):
    if type(groups) is not list or len(groups) != 2 * _CASE_COUNT:
        raise StudyExecutionError("prediction groups are incomplete")
    references = tuple(_prediction_group(groups[2 * i + k], case, assignment, strength)
                       for i, case in enumerate(cases) for k, strength in enumerate(_STRENGTHS))
    parts = [_REFERENCE.fullmatch(reference).groups() for reference in references]
    first = int(parts[0][1])
    try:
        model_index = _MODEL_SEEDS.index(assignment["model_seed"])
        prefix = f"v2-m{assignment['model_seed']}-c"
        cell_text = assignment["cell_id"][len(prefix):]
        cell_index = int(cell_text)
        if (not assignment["cell_id"].startswith(prefix) or cell_text != f"{cell_index:02d}"
                or not 0 <= cell_index < _CELLS_PER_MODEL):
            raise ValueError("invalid cell index")
    except (KeyError, ValueError, TypeError) as error:
        raise StudyExecutionError("prediction cell identity differs") from error
    expected = (model_index * _CELLS_PER_MODEL + cell_index) * _CASE_COUNT * len(_STRENGTHS) * 3
    if first != expected or any(digest != parts[0][0] or int(offset) != first + 3 * index
                        for index, (digest, offset) in enumerate(parts)):
        raise StudyExecutionError("prediction references do not form one ordered record")
    return references


def _guarded(method):
    def wrapped(self, *args, **kwargs):
        entered = False
        try:
            self._check_owner()
            if self._active:
                raise StudyExecutionError("executor callback is already active")
            self._active = True
            entered = True
            result = method(self, *args, **kwargs)
            self._check_owner()
            return result
        except BaseException:
            self._failed = True
            raise
        finally:
            if entered:
                self._active = False
    return wrapped


class StudyExecutor:
    """Single-threaded callback; a separate reviewed custodian owns phase gates."""
    def __init__(self, numeric_policy):
        if type(numeric_policy) is not NumericPolicy:
            raise StudyExecutionError("checked numeric policy is required")
        numeric_policy.verify()
        self._policy = numeric_policy
        self._owner_pid, self._owner_thread = os.getpid(), threading.current_thread()
        self._order = tuple((phase, seed) for phase in ("discovery", "heldout") for seed in _MODEL_SEEDS)
        self._completed = 0
        self._model = self._budget = None
        self._failed = self._closed = self._active = False
        self._receipt = None

    def _check_owner(self):
        if (self._closed or self._failed or os.getpid() != self._owner_pid
                or threading.current_thread() is not self._owner_thread):
            raise StudyExecutionError("executor is unavailable")

    @_guarded
    def begin_model(self, phase, seed, expected_model_state_sha256, budget):
        if (self._model is not None or self._completed >= len(self._order)
                or (phase, seed) != self._order[self._completed] or type(seed) is not int
                or type(expected_model_state_sha256) is not str
                or _DIGEST.fullmatch(expected_model_state_sha256) is None or type(budget) is not ForwardBudget
                or (self._budget is not None and self._budget is not budget)):
            raise StudyExecutionError("invalid model phase, state identity or shared budget")
        self._policy.verify()
        if torch.get_default_dtype() != torch.float32:
            raise StudyExecutionError("default floating dtype differs")
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = TinyResidual(width=8, depth=4).eval()
        _check_model(model)
        if state_sha256(model) != expected_model_state_sha256:
            raise StudyExecutionError("model state does not match the premanifest")
        self._policy.verify()
        self._model, self._budget = model, budget
        self._phase, self._seed, self._state_sha = phase, seed, expected_model_state_sha256
        self._case_bytes = None
        self._clean = {}
        self._nodes = []

    def _validate_cell(self, assignment, cases, cell, budget):
        index = len(self._nodes)
        if (self._model is None or budget is not self._budget or index >= _CELLS_PER_MODEL
                or type(assignment) is not dict or set(assignment) != {"cell_id", "model_seed", "injection_node"}
                or assignment["cell_id"] != f"v2-m{self._seed}-c{index:02d}"
                or type(assignment["model_seed"]) is not int or assignment["model_seed"] != self._seed
                or type(assignment["injection_node"]) is not str or assignment["injection_node"] not in _NODES
                or assignment["injection_node"] in self._nodes
                or cell.cell_id != assignment["cell_id"] or cell.model_seed != self._seed
                or cell.phase != self._phase or cell.model_state_sha256 != self._state_sha
                or type(cases) not in (tuple, list) or len(cases) != _CASE_COUNT):
            raise StudyExecutionError("invalid private cell, case collection or evidence identity")
        raw = tuple(_bounded_json(case, 4096) for case in cases)
        inputs = tuple(_case_values(case, self._phase, self._seed, i) for i, case in enumerate(cases))
        # Split uniqueness treats both signed-zero encodings as one value;
        # the separately recorded input digest intentionally preserves bits.
        identities = tuple(tuple(0.0 if item == 0.0 else item for item in value) for value in inputs)
        if len(set(identities)) != _CASE_COUNT:
            raise StudyExecutionError("duplicate case inputs")
        if self._case_bytes is not None and raw != self._case_bytes:
            raise StudyExecutionError("model case records changed between cells")
        if state_sha256(self._model) != self._state_sha:
            raise StudyExecutionError("model state changed")
        return raw, inputs

    def _run_cell(self, assignment, cases, cell, budget, predictions):
        if (type(assignment) is not dict or type(cases) not in (tuple, list) or len(cases) != _CASE_COUNT
                or (predictions is not None and (type(predictions) is not list or len(predictions) != 2 * _CASE_COUNT))):
            raise StudyExecutionError("invalid cell input collection")
        supplied_assignment, supplied_cases, supplied_predictions = assignment, cases, predictions
        assignment_bytes = _bounded_json(assignment, 1024)
        supplied_case_bytes = tuple(_bounded_json(case, 4096) for case in cases)
        prediction_bytes = None if predictions is None else _bounded_json(predictions, 128 * 1024)
        assignment = json.loads(assignment_bytes)
        cases = tuple(json.loads(item) for item in supplied_case_bytes)
        predictions = None if prediction_bytes is None else json.loads(prediction_bytes)
        raw, inputs = self._validate_cell(assignment, cases, cell, budget)
        if raw != supplied_case_bytes:
            raise StudyExecutionError("case snapshot changed during validation")
        if self._phase == "heldout":
            refs = _prediction_refs(predictions, cases, assignment)
        elif predictions is None:
            refs = (None,) * (2 * _CASE_COUNT)
        else:
            raise StudyExecutionError("discovery must not receive held-out predictions")
        before = (budget.submissions, budget.root_forwards)
        discoveries = []
        for index, (case, values) in enumerate(zip(cases, inputs)):
            if index not in self._clean:
                budget.reserve()  # Input construction belongs to the first clean submission.
                value = _tensor_from_values(values)
                clean = _prepare_clean(self._model, value, budget, self._policy, budget_type=ForwardBudget,
                                       scope=_STUDY_CACHE_TOKEN, input_check=_study_input_check,
                                       first_reserved=True)
                self._clean[index] = (value, clean)
            saved_value, clean = self._clean[index]
            if input_sha256(saved_value) != clean.input_sha256:
                raise StudyExecutionError("case input no longer matches the retained trace")
            budget.reserve()
            target = _prepare_target(self._model, saved_value, assignment["injection_node"], clean, budget,
                                     self._policy, scope=_STUDY_CACHE_TOKEN, input_check=_study_input_check,
                                     first_reserved=True)
            conditions = []
            for offset, strength in enumerate(_STRENGTHS):
                budget.reserve()
                result = _run_active(self._model, saved_value, assignment["injection_node"], strength,
                                     clean, target, budget, self._policy, scope=_STUDY_CACHE_TOKEN,
                                     input_check=_study_input_check, first_reserved=True)
                condition = result["condition"]
                clean_run, changed_run = result["clean_trace"][0].run_id, condition.perturbed_trace[0].run_id
                references = {"clean_sparse_trace": clean_run, "perturbed_sparse_trace": changed_run,
                              "clean": clean_run + "/logits", "perturbed": changed_run + "/logits"}
                condition_id = f"{assignment['cell_id']}:{self._phase}:{index:02d}:{offset}"
                append_executor_result(cell, result, case_id=case["case_id"], case_index=index,
                                       condition_id=condition_id, prediction_ref=refs[2 * index + offset],
                                       references=references)
                conditions.append(condition)
            discoveries.append(DiscoveryCase(index, 1 if index % 2 == 0 else -1, clean.clean_trace,
                                              clean.repeat_trace, clean.clean_logits, clean.repeat_logits,
                                              tuple(conditions)))
        expected = (2 * _CASE_COUNT if self._case_bytes is None else 0) + 33 * _CASE_COUNT
        if ((budget.submissions - before[0], budget.root_forwards - before[1]) != (expected, expected)
                or _bounded_json(supplied_assignment, 1024) != assignment_bytes
                or tuple(_bounded_json(case, 4096) for case in supplied_cases) != supplied_case_bytes
                or (supplied_predictions is not None and _bounded_json(supplied_predictions, 128 * 1024) != prediction_bytes)
                or state_sha256(self._model) != self._state_sha):
            raise StudyExecutionError("execution count or retained identities changed")
        self._policy.verify()
        selection = (_bounded_selection(freeze_discovery(tuple(discoveries), numeric_policy=self._policy,
                                                          pairing_verified=True, private_controls_verified=True))
                     if self._phase == "discovery" else None)
        self._case_bytes = raw
        self._nodes.append(assignment["injection_node"])
        return selection

    @_guarded
    def run_discovery(self, assignment, cases, cell, budget):
        if self._model is None or self._phase != "discovery":
            raise StudyExecutionError("discovery phase is not active")
        return self._run_cell(assignment, cases, cell, budget, None)

    @_guarded
    def run_heldout(self, assignment, cases, predictions, cell, budget):
        if self._model is None or self._phase != "heldout":
            raise StudyExecutionError("held-out phase is not active")
        return self._run_cell(assignment, cases, cell, budget, predictions)

    @_guarded
    def end_model(self):
        if (self._model is None or len(self._nodes) != _CELLS_PER_MODEL
                or set(self._nodes) != set(_NODES) or len(self._clean) != _CASE_COUNT
                or state_sha256(self._model) != self._state_sha
                or self._budget.submissions != self._budget.root_forwards
                or getattr(self._budget, "_pending", False) is not False):
            raise StudyExecutionError("model phase is incomplete or changed")
        self._policy.verify()
        _check_model(self._model)
        self._clean.clear()
        self._model = None
        self._completed += 1

    def close(self):
        if os.getpid() != self._owner_pid or threading.current_thread() is not self._owner_thread:
            self._failed = True
            raise StudyExecutionError("executor belongs to another process or thread")
        if self._active:
            self._failed = True
            raise StudyExecutionError("executor callback is still active")
        if self._closed:
            return deepcopy(self._receipt)
        if self._model is not None:
            _check_hooks(self._model)
            self._clean.clear()
            self._model = None
        if not self._failed and self._completed == len(self._order):
            self._receipt = {"input_model_pairing_verified": True, "private_plan_metadata_verified": True}
        self._closed = True
        return deepcopy(self._receipt)
