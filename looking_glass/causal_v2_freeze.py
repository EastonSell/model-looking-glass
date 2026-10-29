"""Build public v2 prediction freezes and private concrete predictions.

These pure builders do not run models, publish bytes or enforce phase order.
The custodian must verify source/model provenance and persist the resulting
bytes and commitment before permitting any held-out model forward.
"""

import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timedelta

from .causal_v2_custody import _float32_bits, canonical_json, commitment, validate_nonce_set
from .causal_v2_preparation import (MODEL_SEEDS, NODES, PreparedSnapshot, _PILOT_SHA256,
                                    _V1_PROJECTION_SHA256, _context)
from .causal_trace_v2 import _binary32_value
from .pretrained_number_injection import add_float32_bits


_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SITES = (1, 2, 3)
_STRENGTHS = (0.25, 0.75)
_MAX_JSON = 64 * 1024 * 1024
_CLARIFICATIONS = ("numeric_clarification_sha256", "source_clarification_sha256",
                   "self_patch_clarification_sha256", "resource_clarification_sha256")
_CLARIFICATION_FILES = {
    "numeric_clarification_sha256": "docs/CAUSAL_CONFIRMATION_V2_NUMERIC_CLARIFICATION.md",
    "source_clarification_sha256": "docs/CAUSAL_CONFIRMATION_V2_SOURCE_CLARIFICATION.md",
    "self_patch_clarification_sha256": "docs/CAUSAL_CONFIRMATION_V2_SELF_PATCH_CLARIFICATION.md",
    "resource_clarification_sha256": "docs/CAUSAL_CONFIRMATION_V2_RESOURCE_CLARIFICATION.md",
}
_REQUIRED_SOURCE_FILES = frozenset({
    "docs/CAUSAL_CONFIRMATION_V2.md", *_CLARIFICATION_FILES.values(),
    "scripts/v2_source_launcher.py", "scripts/v2_source_bootstrap.py", "scripts/v2_source_guards.py",
    "looking_glass/__init__.py", "looking_glass/model.py", "looking_glass/probes.py",
    "looking_glass/model_ownership.py", "looking_glass/intervention_plan.py",
    "looking_glass/interventions.py", "looking_glass/agreement_numeric.py",
    "looking_glass/causal_trace_v2.py", "looking_glass/causal_v2_runtime.py",
    "looking_glass/causal_v2_selection.py", "looking_glass/causal_v2_model_identity.py",
    "looking_glass/causal_v2_custody.py", "looking_glass/causal_v2_preparation.py",
    "looking_glass/causal_v2_freeze.py", "looking_glass/causal_v2_scoring.py", "looking_glass/causal_v2_executor.py",
    "looking_glass/causal_v2_study_executor.py", "looking_glass/causal_v2_custodian_run.py",
    "scripts/run_synthetic_v2.py", "looking_glass/causal_v2_evidence_encoder.py",
    "looking_glass/causal_v2_evidence_adapter.py", "looking_glass/v2_evidence.py",
    "looking_glass/registry.py", "looking_glass/pretrained_number_injection.py",
})


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(raw):
    if type(raw) is not bytes or len(raw) > _MAX_JSON:
        raise ValueError("expected bounded canonical JSON bytes")
    try:
        value = json.loads(raw)
        if canonical_json(value) != raw:
            raise ValueError("JSON is not canonical")
        return value
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid canonical JSON") from error


def _finite(value):
    return type(value) is float and math.isfinite(value)


def _digest(value):
    return type(value) is str and _HEX.fullmatch(value) is not None


def _vector(values, size):
    if type(values) is not list or len(values) != size or any(type(value) is not float for value in values):
        raise ValueError("invalid float32 vector")
    bits = [_float32_bits(value) for value in values]
    return [0 if bit & 0x7fffffff == 0 else bit for bit in bits]


def _metric(values):
    _vector(values, 2)
    # The specified metric subtracts in float32, then converts to binary64.
    bits = add_float32_bits(_float32_bits(values[1]), _float32_bits(values[0]) ^ 0x80000000)
    if bits & 0x7f800000 == 0x7f800000:
        raise ValueError("nonfinite float32 logit contrast")
    return _binary32_value(bits)


def _trace(trace):
    if type(trace) is not list or len(trace) != 4:
        raise ValueError("incomplete sparse trace")
    keys = {"node_id", "sequence", "occurrence", "shape", "dtype", "device", "values"}
    bits = []
    for index, event in enumerate(trace):
        if (type(event) is not dict or set(event) != keys or event["node_id"] != f"block.{index}.output"
                or type(event["sequence"]) is not int or event["sequence"] != index
                or type(event["occurrence"]) is not int or event["occurrence"] != 0
                or type(event["shape"]) is not list or len(event["shape"]) != 2
                or any(type(size) is not int for size in event["shape"]) or event["shape"] != [1, 8]
                or event["dtype"] != "torch.float32" or event["device"] != "cpu"):
            raise ValueError("invalid sparse event metadata")
        bits.append(_vector(event["values"], 8))
    return bits


def _checked_snapshot(snapshot):
    """Validate the public schema and private commitments, not generation provenance."""
    if type(snapshot) is not PreparedSnapshot:
        raise ValueError("a verified preparation snapshot is required")
    public, heldout, schedule = _json(snapshot.public), _json(snapshot.heldout), _json(snapshot.schedule)
    keys = {"mode", "context", "context_sha256", "pilot_exclusions_sha256",
            "development_exclusions_sha256", "historical_exclusions_sha256", "historical_exclusions_count",
            "discovery_cases", "heldout_commitment", "schedule_commitment"}
    try:
        if type(public) is not dict or set(public) != keys or public["mode"] not in ("development", "study"):
            raise ValueError("invalid public preparation schema")
        context = public["context"]
        states = context["model_state_hashes"]
        if type(states) is not list or not states:
            raise ValueError("empty model-state inventory")
        seeds = tuple(item["seed"] for item in states)
        context_hash = _context(context, seeds)
        if (len(set(seeds)) != len(seeds) or public["context_sha256"] != context_hash
                or any(not _digest(public[key]) for key in ("pilot_exclusions_sha256",
                    "development_exclusions_sha256", "historical_exclusions_sha256",
                    "heldout_commitment", "schedule_commitment"))
                or type(public["historical_exclusions_count"]) is not int
                or public["historical_exclusions_count"] < 1):
            raise ValueError("invalid preparation identity")
        validate_nonce_set(snapshot.heldout_root, snapshot.site_nonce, snapshot.prediction_nonce)
        if (type(heldout) is not dict or set(heldout) != {"context_sha256", "cases"}
                or type(schedule) is not dict or set(schedule) != {"context_sha256", "assignments"}
                or heldout["context_sha256"] != context_hash or schedule["context_sha256"] != context_hash
                or public["heldout_commitment"] != commitment("heldout", snapshot.heldout_root, heldout)
                or public["schedule_commitment"] != commitment("schedule", snapshot.site_nonce, schedule)):
            raise ValueError("initial private commitments differ")
        discovery, cases, assignments = public["discovery_cases"], heldout["cases"], schedule["assignments"]
        if type(discovery) is not list or type(cases) is not list or len(cases) != len(discovery):
            raise ValueError("invalid split sizes")
        count, remainder = divmod(len(cases), len(seeds))
        if remainder or not 2 <= count <= 16 or count % 2:
            raise ValueError("invalid cases per model")
        if (public["mode"] == "study" and (seeds != MODEL_SEEDS or count != 16
                or public["pilot_exclusions_sha256"] != _PILOT_SHA256
                or public["historical_exclusions_sha256"] != _V1_PROJECTION_SHA256
                or public["historical_exclusions_count"] != 24)):
            raise ValueError("study seeds or historical inputs differ")
        if public["mode"] == "development" and set(seeds) & set(MODEL_SEEDS):
            raise ValueError("development uses a reserved study seed")
        if type(assignments) is not list or len(assignments) != len(seeds) * len(NODES):
            raise ValueError("invalid schedule count")
        prefix = "dev-v2" if public["mode"] == "development" else "v2"
        seen = set()
        for phase, rows, marker in (("discovery", discovery, "d"), ("heldout", cases, "h")):
            for seed_index, seed in enumerate(seeds):
                for index, case in enumerate(rows[seed_index * count:(seed_index + 1) * count]):
                    if (type(case) is not dict or set(case) != {"case_id", "model_seed", "split", "index", "values"}
                            or type(case["model_seed"]) is not int or case["model_seed"] != seed
                            or type(case["index"]) is not int or case["index"] != index or case["split"] != phase
                            or case["case_id"] != f"{prefix}-m{seed}-{marker}{index:02d}"):
                        raise ValueError("invalid split identity")
                    values = case["values"]
                    bits = tuple(_vector(values, 8))
                    if (bits in seen or (values[0] <= 0 if index % 2 == 0 else values[0] >= 0)
                            or not 1 <= abs(values[0]) * 16 <= 16
                            or any(value * 16 != int(value * 16) for value in values)
                            or any(not -16 <= value * 16 <= 15 for value in values[1:])):
                        raise ValueError("invalid or duplicate split values")
                    seen.add(bits)
        for seed_index, seed in enumerate(seeds):
            segment = assignments[seed_index * len(NODES):(seed_index + 1) * len(NODES)]
            observed = []
            for index, item in enumerate(segment):
                if (type(item) is not dict or set(item) != {"cell_id", "model_seed", "injection_node"}
                        or type(item["model_seed"]) is not int or item["model_seed"] != seed
                        or item["cell_id"] != f"{prefix}-m{seed}-c{index:02d}"):
                    raise ValueError("invalid schedule identity")
                observed.append(item["injection_node"])
            if set(observed) != set(NODES):
                raise ValueError("invalid schedule permutation")
    except (KeyError, TypeError, ZeroDivisionError, RecursionError) as error:
        raise ValueError("invalid prepared snapshot") from error
    return public, heldout, schedule


def public_pre_manifest(snapshot, *, source_sha256, clarification_sha256, runtime, created_utc):
    """Build a complete public manifest from a separately checked snapshot.

    Hashes and runtime are caller-supplied attestations: this builder does not
    establish that source bytes were executed or that model hashes are true.
    """
    prepared, _, _ = _checked_snapshot(snapshot)
    if (type(prepared) is not dict or type(source_sha256) is not dict or not source_sha256
            or any(type(path) is not str or not path or len(path) > 512 or path.startswith('/')
                   or re.fullmatch(r"[A-Za-z0-9_./-]+", path) is None
                   or any(part in ('', '.', '..') for part in path.split('/'))
                   or not _digest(value)
                   for path, value in source_sha256.items())
            or type(clarification_sha256) is not dict or set(clarification_sha256) != set(_CLARIFICATIONS)
            or not all(_digest(value) for value in clarification_sha256.values())
            or type(runtime) is not dict or set(runtime) != {"python_version", "torch_version", "system",
                "machine", "byteorder", "intraop_threads", "denormal_preserved", "torch_build_sha256"}
            or any(type(runtime[key]) is not str or re.fullmatch(r"[A-Za-z0-9_.+\-]{1,64}", runtime[key]) is None
                   for key in ("python_version", "torch_version", "system", "machine"))
            or runtime["byteorder"] not in ("little", "big")
            or type(runtime["intraop_threads"]) is not int or runtime["intraop_threads"] != 1
            or runtime["denormal_preserved"] is not True or not _digest(runtime["torch_build_sha256"])
            or type(created_utc) is not str or not created_utc or len(created_utc) > 128):
        raise ValueError("invalid source, clarification, runtime or clock metadata")
    try:
        timestamp = datetime.fromisoformat(created_utc)
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("UTC timestamp required")
    except ValueError as error:
        raise ValueError("invalid UTC timestamp") from error
    if not _REQUIRED_SOURCE_FILES.issubset(source_sha256):
        raise ValueError("source inventory omits required v2 implementation paths")
    if (prepared["mode"] == "study" and (runtime["python_version"] != "3.11.15"
            or runtime["torch_version"] != "2.8.0+cu128")):
        raise ValueError("study runtime differs from the protocol")
    if (source_sha256.get("docs/CAUSAL_CONFIRMATION_V2.md") != prepared["context"]["protocol_sha256"]
            or any(source_sha256.get(path) != clarification_sha256[field]
                   for field, path in _CLARIFICATION_FILES.items())):
        raise ValueError("protocol or clarification hashes differ from source inventory")
    manifest = dict(prepared)
    manifest.update({"schema": "synthetic-v2-pre-manifest-v1", "source_sha256": source_sha256,
                     "runtime": runtime, "created_utc": created_utc,
                     "settings": {"width": 8, "depth": 4, "route": 0,
                                  "device": "cpu", "dtype": "float32", "intraop_threads": 1,
                                  "strengths": list(_STRENGTHS), "polarity": "sign of x[0,0]",
                                  "input_identity": {"tag": "MLG-v2-input-f32be-v1", "shape": [1, 8],
                                      "encoding": "ASCII tag, NUL, two uint32be dimensions, eight uint32be float bits"}}})
    manifest.update(clarification_sha256)
    return canonical_json(manifest)


def _selector(selection, count):
    keys = {"first_changed", "magnitude", "first_votes", "mean_l2", "observation_misses",
            "no_injection", "random", "predictions", "measurements"}
    if type(selection) is not dict or set(selection) != keys:
        raise ValueError("invalid selection summary")
    rows = selection["measurements"]
    if type(rows) is not list or len(rows) != count:
        raise ValueError("incomplete discovery measurements")
    votes = {str(site): 0 for site in _SITES}
    norms = {site: [] for site in _SITES}
    effects = {site: {strength: {1: [], -1: []} for strength in _STRENGTHS} for site in _SITES}
    misses = 0
    for index, row in enumerate(rows):
        polarity = 1 if index % 2 == 0 else -1
        row_keys = {"case_index", "polarity", "clean_trace", "repeated_clean_trace", "clean_logits",
                    "repeated_clean_logits", "conditions"}
        if (type(row) is not dict or set(row) != row_keys or type(row.get("case_index")) is not int
                or row["case_index"] != index or type(row.get("polarity")) is not int
                or row["polarity"] != polarity or type(row.get("conditions")) is not list
                or len(row["conditions"]) != 2):
            raise ValueError("invalid discovery ordering")
        clean_bits = _trace(row["clean_trace"])
        if clean_bits != _trace(row["repeated_clean_trace"]):
            raise ValueError("clean repeat trace differs")
        clean_logits = _vector(row["clean_logits"], 2)
        if clean_logits != _vector(row["repeated_clean_logits"], 2):
            raise ValueError("clean repeat logits differ")
        for strength, condition in zip(_STRENGTHS, row["conditions"]):
            condition_keys = {"strength", "perturbed_trace", "first_unequal", "last_equal", "l2_by_block",
                              "perturbed_logits", "clean_to_perturbed_logits", "perturbed_to_clean_logits",
                              "null_logits", "identity_logits", "upstream_logits", "clean_verify_logits", "effects"}
            if (type(condition) is not dict or set(condition) != condition_keys
                    or type(condition.get("strength")) is not float
                    or condition["strength"] != strength
                    or type(condition.get("l2_by_block")) is not list
                    or len(condition["l2_by_block"]) != 4
                    or any(not _finite(value) or value < 0 for value in condition["l2_by_block"])
                    or type(condition.get("effects")) is not list or len(condition["effects"]) != 3
                    or any(not _finite(value) for value in condition["effects"])):
                raise ValueError("invalid discovery condition")
            perturbed_bits = _trace(condition["perturbed_trace"])
            changed = [index for index in range(4) if clean_bits[index] != perturbed_bits[index]]
            if changed and changed[0] == 0:
                raise ValueError("upstream sparse state changed")
            first_index = changed[0] if changed else None
            expected_first = None if first_index is None else f"block.{first_index}.output"
            expected_last = f"block.{3 if first_index is None else first_index - 1}.output"
            norms_by_block = [math.sqrt(math.fsum((a - b) ** 2 for a, b in
                zip(clean["values"], perturb["values"])))
                for clean, perturb in zip(row["clean_trace"], condition["perturbed_trace"])]
            if (condition["first_unequal"] != expected_first or condition["last_equal"] != expected_last
                    or canonical_json(condition["l2_by_block"]) != canonical_json(norms_by_block)):
                raise ValueError("sparse comparison differs from retained traces")
            base = _vector(condition["perturbed_logits"], 2)
            if clean_bits[3] == perturbed_bits[3] and clean_logits != base:
                raise ValueError("equal final states have different logits")
            for field in ("clean_to_perturbed_logits", "perturbed_to_clean_logits", "null_logits", "identity_logits"):
                if type(condition[field]) is not list or len(condition[field]) != 3:
                    raise ValueError("expected three site outputs")
                for value in condition[field]:
                    _vector(value, 2)
            if (any(_vector(value, 2) != base for field in ("null_logits", "identity_logits")
                    for value in condition[field])
                    or _vector(condition["upstream_logits"], 2) != base
                    or _vector(condition["clean_verify_logits"], 2) != clean_logits
                    or _vector(condition["clean_to_perturbed_logits"][2], 2) != clean_logits
                    or _vector(condition["perturbed_to_clean_logits"][2], 2) != base):
                raise ValueError("discovery control differs")
            base_metric = _metric(condition["perturbed_logits"])
            reconstructed = [_metric(value) - base_metric for value in condition["clean_to_perturbed_logits"]]
            if canonical_json(condition["effects"]) != canonical_json(reconstructed):
                raise ValueError("saved patch effects differ from retained logits")
            first = condition.get("first_unequal")
            if first is None:
                misses += 1
            elif type(first) is str and first in {f"block.{site}.output" for site in _SITES}:
                votes[first.split('.')[1]] += 1
            else:
                raise ValueError("invalid first observed block")
            for site in _SITES:
                norms[site].append(condition["l2_by_block"][site])
                effects[site][strength][polarity].append(condition["effects"][site - 1])
    expected_first = max(_SITES, key=lambda site: (votes[str(site)], -site)) if sum(votes.values()) else None
    means = {str(site): statistics.fmean(norms[site]) for site in _SITES}
    expected_magnitude = max(_SITES, key=lambda site: (means[str(site)], -site)) if any(means.values()) else None
    expected_predictions = {}
    for site in _SITES:
        expected_predictions[str(site)] = {}
        for strength in _STRENGTHS:
            positive, negative = effects[site][strength][1], effects[site][strength][-1]
            all_cases = [value for pair in zip(positive, negative) for value in pair]
            expected_predictions[str(site)][str(strength)] = {
                "positive": statistics.fmean(positive), "negative": statistics.fmean(negative),
                "constant": statistics.fmean(all_cases), "zero": 0.0}
    expected_no_injection = {"cases": count, "false_detections": 0,
                             "first_changed": None, "magnitude": None}
    expected = {"first_changed": expected_first, "magnitude": expected_magnitude,
                "first_votes": votes, "mean_l2": means, "observation_misses": misses,
                "no_injection": expected_no_injection,
                "random": {"sites": list(_SITES), "each_probability": "1/3"},
                "predictions": expected_predictions}
    if canonical_json({key: selection[key] for key in expected}) != canonical_json(expected):
        raise ValueError("selection or prediction means do not match discovery measurements")


def verify_discovery_evidence(snapshot, cells, chunks):
    """Match every selection observation to its persisted private cell chunk.

    ``chunks`` yields complete, canonical discovery chunks read back in the
    prepared cell order. Only the custodian may call this with private records;
    the selector receives no hidden target or private plan metadata. This
    establishes consistency of supplied bytes, not capture validity, secrecy,
    publication order, or execution. The caller must first verify preparation,
    source/model/runtime identities, and actual chunk publication.
    """
    from . import registry
    from .causal_v2_evidence_adapter import CELL_CHUNK_LIMIT, CellEvidence
    from .causal_v2_scoring import validate_saved_controls
    from .v2_evidence import _chunk

    public, _, schedule = _checked_snapshot(snapshot)
    assignments = schedule["assignments"]
    cases = public["discovery_cases"]
    states = {item["seed"]: item["sha256"] for item in public["context"]["model_state_hashes"]}
    count = len(cases) // len(states)
    if type(cells) is not list or len(cells) != len(assignments):
        raise ValueError("all ordered discovery cells are required")
    try:
        source = iter(chunks)
    except TypeError as error:
        raise ValueError("ordered discovery chunks are required") from error

    class _NoPublish:
        def publish_chunk(self, name, data):
            raise RuntimeError("verification cannot publish evidence")

    def saved_bits(tensor):
        # CellEvidence has already checked the tensor schema and finiteness.
        return [0 if bit & 0x7fffffff == 0 else bit for bit in tensor["bits"]]

    def input_digest(values):
        bits = [_float32_bits(value) for value in values]
        prefix = b"MLG-v2-input-f32be-v1\0" + (1).to_bytes(4, "big") + (8).to_bytes(4, "big")
        return _sha(prefix + b"".join(bit.to_bytes(4, "big") for bit in bits))

    condition_ids, target_baselines = set(), {}
    trace_owners, clean_trace_ids = {}, {}
    for index, (cell, assignment) in enumerate(zip(cells, assignments)):
        if (type(cell) is not dict or set(cell) != {"cell_id", "model_seed", "selection"}
                or type(cell["model_seed"]) is not int or cell["model_seed"] != assignment["model_seed"]
                or cell["cell_id"] != assignment["cell_id"]):
            raise ValueError("discovery cell identity or order differs")
        _selector(cell["selection"], count)
        name = f"cell-{index:02d}-d.json"
        try:
            raw = next(source)
        except StopIteration as error:
            raise ValueError("a discovery chunk is missing") from error
        # _chunk enforces the registry's canonical encoding and byte bound.
        chunk = _chunk(raw, name, CELL_CHUNK_LIMIT)
        records = chunk["records"]
        if chunk["state"] != "complete" or len(records) != count * 2:
            raise ValueError("a discovery chunk is incomplete")
        seed = assignment["model_seed"]
        model_cases = [case for case in cases if case["model_seed"] == seed]
        buffer = CellEvidence(_NoPublish(), name=name, cell_id=cell["cell_id"],
                              phase="discovery", model_seed=seed, model_state_sha256=states[seed])
        for position, record in enumerate(records):
            case_index, strength_index = divmod(position, 2)
            case, row = model_cases[case_index], cell["selection"]["measurements"][case_index]
            condition = row["conditions"][strength_index]
            # Reuse the writer's strict record, tensor and ordering validation.
            buffer.append(registry._json_bytes(record))
            if (record["case_id"] != case["case_id"] or record["input_sha256"] != input_digest(case["values"])
                    or record["private_integrity"]["target"] != assignment["injection_node"]
                    or record["private_integrity"]["delta"] != condition["strength"] * row["polarity"]
                    or record["condition_id"] in condition_ids):
                raise ValueError("private discovery identity differs")
            condition_ids.add(record["condition_id"])
            checked = validate_saved_controls(record, target=assignment["injection_node"],
                                               strength=condition["strength"], polarity=row["polarity"])
            target_key = (cell["cell_id"], case_index)
            target_bits = tuple(0 if bit & 0x7fffffff == 0 else bit for bit in checked["clean_target"])
            if target_key in target_baselines and target_baselines[target_key] != target_bits:
                raise ValueError("reused private target capture differs")
            target_baselines[target_key] = target_bits
            traces = record["traces"]
            trace_ids = {key: traces[key][0]["metadata"]["run_id"]
                         for key in ("clean", "repeat_clean", "perturbed")}
            if len(set(trace_ids.values())) != 3:
                raise ValueError("discovery trace pairing is invalid")
            clean_key = (seed, case_index)
            clean_pair = (trace_ids["clean"], trace_ids["repeat_clean"])
            if clean_key in clean_trace_ids and clean_trace_ids[clean_key] != clean_pair:
                raise ValueError("reused clean trace identity differs")
            clean_trace_ids[clean_key] = clean_pair
            for kind in ("clean", "repeat_clean"):
                owner = (seed, case_index, kind)
                run_id = trace_ids[kind]
                if run_id in trace_owners and trace_owners[run_id] != owner:
                    raise ValueError("discovery trace identity is reused")
                trace_owners[run_id] = owner
            if trace_ids["perturbed"] in trace_owners:
                raise ValueError("discovery trace identity is reused")
            trace_owners[trace_ids["perturbed"]] = (cell["cell_id"], position, "perturbed")
            for saved, observed in ((traces["clean"], row["clean_trace"]),
                                    (traces["repeat_clean"], row["repeated_clean_trace"]),
                                    (traces["perturbed"], condition["perturbed_trace"])):
                if [saved_bits(event["value"]) for event in saved] != _trace(observed):
                    raise ValueError("discovery sparse evidence differs")
            logits = record["logits"]
            for saved, observed in ((logits["clean"], row["clean_logits"]),
                                    (logits["repeat_clean"], row["repeated_clean_logits"]),
                                    (logits["perturbed"], condition["perturbed_logits"]),
                                    (logits["upstream"], condition["upstream_logits"]),
                                    (logits["post_hook_clean"], condition["clean_verify_logits"])):
                if saved_bits(saved) != _vector(observed, 2):
                    raise ValueError("discovery logit evidence differs")
            for key, field in (("forward", "clean_to_perturbed_logits"),
                               ("reverse", "perturbed_to_clean_logits"),
                               ("null", "null_logits"), ("self", "identity_logits")):
                if [saved_bits(value) for value in logits[key]] != [
                        _vector(value, 2) for value in condition[field]]:
                    raise ValueError("discovery control evidence differs")
            comparison = {"status": ("no_difference_observed" if condition["first_unequal"] is None
                                      else "difference_observed"),
                          "first_unequal": condition["first_unequal"],
                          "last_equal": condition["last_equal"], "l2_by_block": condition["l2_by_block"]}
            if canonical_json(record["comparison"]) != canonical_json(comparison):
                raise ValueError("discovery comparison evidence differs")
    try:
        next(source)
    except StopIteration:
        return
    raise ValueError("unexpected additional discovery chunk")


def build_freeze(snapshot, pre_manifest_bytes, cells):
    """Return public freeze, private per-case predictions and public commitment.

    Cell measurements must come from the separately reviewed selector. The
    checks recompute contrasts, effects, choices and means from retained logits
    and traces; they do not independently prove capture validity or execution.
    """
    if type(snapshot) is not PreparedSnapshot:
        raise ValueError("a verified preparation snapshot is required")
    public, manifest = _json(snapshot.public), _json(pre_manifest_bytes)
    try:
        clarifications = {name: manifest[name] for name in _CLARIFICATIONS}
        rebuilt_manifest = public_pre_manifest(snapshot, source_sha256=manifest["source_sha256"],
            clarification_sha256=clarifications, runtime=manifest["runtime"], created_utc=manifest["created_utc"])
        if rebuilt_manifest != pre_manifest_bytes:
            raise ValueError("pre-manifest differs from prepared inputs")
    except (KeyError, TypeError) as error:
        raise ValueError("invalid pre-manifest") from error
    heldout, schedule = _json(snapshot.heldout), _json(snapshot.schedule)
    try:
        context = public["context"]
        context_sha = _sha(canonical_json(context))
        validate_nonce_set(snapshot.heldout_root, snapshot.site_nonce, snapshot.prediction_nonce)
        if (public["context_sha256"] != context_sha or heldout["context_sha256"] != context_sha
                or schedule["context_sha256"] != context_sha
                or public["heldout_commitment"] != commitment("heldout", snapshot.heldout_root, heldout)
                or public["schedule_commitment"] != commitment("schedule", snapshot.site_nonce, schedule)):
            raise ValueError("initial private commitments differ")
        assignments, cases = schedule["assignments"], heldout["cases"]
        seeds = [item["seed"] for item in context["model_state_hashes"]]
    except (KeyError, TypeError, RecursionError) as error:
        raise ValueError("invalid prepared commitment records") from error
    if type(cells) is not list or len(cells) != len(assignments):
        raise ValueError("all discovery cells are required")
    if (type(seeds) is not list or not seeds or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds) or len(assignments) != len(seeds) * len(NODES)
            or type(cases) is not list):
        raise ValueError("invalid prepared seed or schedule count")
    count = len(cases) // len(seeds)
    if count < 2 or count % 2 or len(cases) != len(seeds) * count:
        raise ValueError("invalid held-out case count")
    prefix = "dev-v2" if public.get("mode") == "development" else "v2"
    for seed_index, seed in enumerate(seeds):
        segment = assignments[seed_index * len(NODES):(seed_index + 1) * len(NODES)]
        observed_nodes = []
        for index, assignment in enumerate(segment):
            if (type(assignment) is not dict or set(assignment) != {"cell_id", "model_seed", "injection_node"}
                    or type(assignment["model_seed"]) is not int or assignment["model_seed"] != seed
                    or assignment["cell_id"] != f"{prefix}-m{seed}-c{index:02d}"):
                raise ValueError("invalid private schedule identity")
            observed_nodes.append(assignment["injection_node"])
        if set(observed_nodes) != set(NODES):
            raise ValueError("private schedule is not a permutation")
        for index, case in enumerate(cases[seed_index * count:(seed_index + 1) * count]):
            if (type(case) is not dict or set(case) != {"case_id", "model_seed", "split", "index", "values"}
                    or type(case["model_seed"]) is not int or case["model_seed"] != seed
                    or type(case["index"]) is not int or case["index"] != index or case["split"] != "heldout"
                    or case["case_id"] != f"{prefix}-m{seed}-h{index:02d}"):
                raise ValueError("invalid held-out case identity")
            _vector(case["values"], 8)
            if (case["values"][0] <= 0 if index % 2 == 0 else case["values"][0] >= 0):
                raise ValueError("invalid held-out polarity")
    baselines = {}
    for cell, assignment in zip(cells, assignments):
        if (type(cell) is not dict or set(cell) != {"cell_id", "model_seed", "selection"}
                or type(cell["model_seed"]) is not int or cell["model_seed"] != assignment["model_seed"]
                or cell["cell_id"] != assignment["cell_id"]):
            raise ValueError("discovery cell order or identity differs")
        _selector(cell["selection"], count)
        for row in cell["selection"]["measurements"]:
            key = (cell["model_seed"], row["case_index"])
            fingerprint = (tuple(tuple(bits) for bits in _trace(row["clean_trace"])),
                           tuple(tuple(bits) for bits in _trace(row["repeated_clean_trace"])),
                           tuple(_vector(row["clean_logits"], 2)),
                           tuple(_vector(row["repeated_clean_logits"], 2)))
            if key in baselines and baselines[key] != fingerprint:
                raise ValueError("reused clean baseline differs across cells")
            baselines[key] = fingerprint
    freeze = {"schema": "synthetic-v2-prediction-freeze-v1", "pre_manifest_sha256": _sha(pre_manifest_bytes),
              "cells": cells, "metric": "float32(logits[0,1]-logits[0,0])",
              "patch": "same-input clean source into perturbed base, whole block output",
              "selection": "first-vote or mean-l2, ties lower; zero observations abstain",
              "random": "uniform expectation over blocks 1, 2, 3",
              "controls": "clean repeat, null, same-forward self, upstream, downstream, reverse and post-hook clean"}
    freeze_bytes = canonical_json(freeze)
    predictions = []
    cases_by_seed = {seed: [case for case in cases if case["model_seed"] == seed] for seed in seeds}
    for cell in cells:
        for case in cases_by_seed[cell["model_seed"]]:
            polarity = 1 if case["index"] % 2 == 0 else -1
            for strength in _STRENGTHS:
                for site in _SITES:
                    row = cell["selection"]["predictions"][str(site)][str(strength)]
                    predictions.append({"model_seed": cell["model_seed"], "cell_id": cell["cell_id"],
                                        "case_id": case["case_id"], "strength": strength, "site": site,
                                        "polarity": polarity,
                                        "polarity_prediction": row["positive" if polarity > 0 else "negative"],
                                        "constant_prediction": row["constant"], "zero_prediction": row["zero"]})
    record = {"pre_manifest_sha256": _sha(pre_manifest_bytes), "freeze_sha256": _sha(freeze_bytes),
              "predictions": predictions}
    private_bytes = canonical_json(record)
    public_commitment = canonical_json({"schema": "synthetic-v2-prediction-commitment-v1",
        "pre_manifest_sha256": record["pre_manifest_sha256"], "freeze_sha256": record["freeze_sha256"],
        "commitment": commitment("predictions", snapshot.prediction_nonce, record)})
    return freeze_bytes, private_bytes, public_commitment
