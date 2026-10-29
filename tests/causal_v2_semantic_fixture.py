"""Build only fabricated v2-shaped saved records; never run a model."""
import copy
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

from looking_glass import registry
from looking_glass.causal_v2_custody import canonical_json, commitment
from looking_glass.causal_v2_freeze import _REQUIRED_SOURCE_FILES, build_freeze, public_pre_manifest
from looking_glass.causal_v2_preparation import PreparedSnapshot
from looking_glass.causal_v2_scoring import score_heldout

_spec = importlib.util.spec_from_file_location("fabricated_scoring_fixture", Path(__file__).with_name("test_causal_v2_scoring.py"))
_scoring_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scoring_fixture)
heldout_fixture = _scoring_fixture.fixture


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _input_hash(values):
    return _sha(b"MLG-v2-input-f32be-v1\0" + struct.pack(">II", 1, 8)
                + b"".join(struct.pack(">f", value) for value in values))


def _values(tensor):
    return [struct.unpack(">f", struct.pack(">I", bit))[0] for bit in tensor["bits"]]


def _public_trace(events):
    keys = ("node_id", "sequence", "occurrence", "shape", "dtype", "device")
    return [{**{key: event["metadata"][key] for key in keys}, "values": _values(event["value"])}
            for event in events]


def _discovery_selection(records, block):
    rows = []
    for index in range(16):
        first, second = records[index * 2:index * 2 + 2]
        polarity = 1 if index % 2 == 0 else -1
        conditions = []
        for record in (first, second):
            strength = record["strength"]
            logits = record["logits"]
            effects = [-strength * polarity if site >= block else 0.0 for site in (1, 2, 3)]
            conditions.append({"strength": strength, "perturbed_trace": _public_trace(record["traces"]["perturbed"]),
                "first_unequal": record["comparison"]["first_unequal"],
                "last_equal": record["comparison"]["last_equal"],
                "l2_by_block": record["comparison"]["l2_by_block"],
                "perturbed_logits": _values(logits["perturbed"]),
                "clean_to_perturbed_logits": [_values(item) for item in logits["forward"]],
                "perturbed_to_clean_logits": [_values(item) for item in logits["reverse"]],
                "null_logits": [_values(item) for item in logits["null"]],
                "identity_logits": [_values(item) for item in logits["self"]],
                "upstream_logits": _values(logits["upstream"]),
                "clean_verify_logits": _values(logits["post_hook_clean"]), "effects": effects})
        rows.append({"case_index": index, "polarity": polarity,
            "clean_trace": _public_trace(first["traces"]["clean"]),
            "repeated_clean_trace": _public_trace(first["traces"]["repeat_clean"]),
            "clean_logits": _values(first["logits"]["clean"]),
            "repeated_clean_logits": _values(first["logits"]["repeat_clean"]), "conditions": conditions})
    predictions = {str(site): {str(strength): {
        "positive": -strength if site >= block else 0.0,
        "negative": strength if site >= block else 0.0, "constant": 0.0, "zero": 0.0}
        for strength in (0.25, 0.75)} for site in (1, 2, 3)}
    return {"first_changed": block, "magnitude": block,
            "first_votes": {str(site): 32 if site == block else 0 for site in (1, 2, 3)},
            "mean_l2": {str(site): 0.5 if site >= block else 0.0 for site in (1, 2, 3)},
            "observation_misses": 0,
            "no_injection": {"cases": 16, "false_detections": 0, "first_changed": None, "magnitude": None},
            "random": {"sites": [1, 2, 3], "each_probability": "1/3"},
            "predictions": predictions, "measurements": rows}


def build_fixture():
    original = heldout_fixture()
    old_manifest, old_heldout, schedule = (json.loads(original[i]) for i in (0, 4, 5))
    context = old_manifest["context"]
    seeds = [item["seed"] for item in context["model_state_hashes"]]
    discovery, heldout_cases = [], []
    for model_index, seed in enumerate(seeds):
        for index in range(16):
            for split, marker, dest in (("discovery", "d", discovery), ("heldout", "h", heldout_cases)):
                values = [0.0625 if index % 2 == 0 else -0.0625, (index - 8) / 16,
                          (model_index - 3) / 16, 0.0 if split == "discovery" else 0.0625]
                values += [0.0] * 4
                dest.append({"case_id": f"dev-v2-m{seed}-{marker}{index:02d}", "model_seed": seed,
                             "split": split, "index": index, "values": values})
    heldout = {"context_sha256": old_heldout["context_sha256"], "cases": heldout_cases}
    roots = (b"h" * 32, b"s" * 32, b"p" * 32)
    public = {"mode": "development", "context": context, "context_sha256": heldout["context_sha256"],
              "pilot_exclusions_sha256": "d" * 64, "development_exclusions_sha256": "e" * 64,
              "historical_exclusions_sha256": "f" * 64, "historical_exclusions_count": 1,
              "discovery_cases": discovery, "heldout_commitment": commitment("heldout", roots[0], heldout),
              "schedule_commitment": commitment("schedule", roots[1], schedule)}
    snapshot = PreparedSnapshot(canonical_json(public), canonical_json(heldout), canonical_json(schedule), *roots)
    clarifications = {name + "_clarification_sha256": str(index) * 64 for index, name in
                      enumerate(("numeric", "source", "self_patch", "resource"), 1)}
    source = {path: "f" * 64 for path in _REQUIRED_SOURCE_FILES}
    source["docs/CAUSAL_CONFIRMATION_V2.md"] = context["protocol_sha256"]
    source["results/synthetic-v2-exclusions-20261028/inputs.json"] = public["development_exclusions_sha256"]
    source["results/synthetic-v2-resource-pilot-20261028/artifacts/pilot_inputs.json"] = public["pilot_exclusions_sha256"]
    for name, label in (("numeric", "NUMERIC"), ("source", "SOURCE"),
                        ("self_patch", "SELF_PATCH"), ("resource", "RESOURCE")):
        source[f"docs/CAUSAL_CONFIRMATION_V2_{label}_CLARIFICATION.md"] = clarifications[name + "_clarification_sha256"]
    runtime = {"python_version": "3.11.15", "torch_version": "2.8.0+cu128", "system": "Windows",
               "machine": "AMD64", "byteorder": "little", "intraop_threads": 1,
               "denormal_preserved": True, "torch_build_sha256": "e" * 64, "interop_threads": 1,
               "float32_matmul_precision": "highest", "mkldnn_enabled": True,
               "mkldnn_deterministic": True, "deterministic_algorithms": True,
               "deterministic_warn_only": False, "cpu_capability": "AVX2",
               "thread_environment": {name: None for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "OMP_DYNAMIC", "MKL_DYNAMIC", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    manifest = public_pre_manifest(snapshot, source_sha256=source, clarification_sha256=clarifications,
                                   runtime=runtime, created_utc="2026-10-29T02:00:00+00:00")
    cells, discovery_chunks, heldout_objects = [], [], []
    for cell_index, raw in enumerate(original[-1]):
        chunk = json.loads(raw)
        chunk["name"] = f"cell-{cell_index:02d}-h.json"
        seed_index = cell_index // 9
        assignment = schedule["assignments"][cell_index]
        discovery_records = []
        for row in chunk["records"]:
            case_index = row["case_index"]
            row["input_sha256"] = _input_hash(heldout_cases[seed_index * 16 + case_index]["values"])
            copy_row = copy.deepcopy(row)
            copy_row.update(phase="discovery", case_id=discovery[seed_index * 16 + case_index]["case_id"],
                            input_sha256=_input_hash(discovery[seed_index * 16 + case_index]["values"]),
                            prediction_ref=None, condition_id="d-" + row["condition_id"])
            for trace in copy_row["traces"].values():
                for event in trace:
                    event["metadata"]["run_id"] = "d-" + event["metadata"]["run_id"]
            for plan in copy_row["plans"]:
                for key in ("source_id", "comparison_id", "base_id"):
                    if plan[key] not in (None, "same_forward_self_source"):
                        plan[key] = "d-" + plan[key]
            discovery_records.append(copy_row)
        block = int(assignment["injection_node"].split(".")[1])
        cells.append({"cell_id": assignment["cell_id"], "model_seed": assignment["model_seed"],
                      "selection": _discovery_selection(discovery_records, block)})
        discovery_chunks.append(registry._json_bytes({"name": f"cell-{cell_index:02d}-d.json",
            "records": discovery_records, "schema": "causal-v2-chunk-v1", "state": "complete"}))
        heldout_objects.append(chunk)
    freeze, prediction, public_commitment = build_freeze(snapshot, manifest, cells)
    prediction_hash = _sha(prediction)
    heldout_chunks = []
    for cell_index, chunk in enumerate(heldout_objects):
        for position, row in enumerate(chunk["records"]):
            row["prediction_ref"] = f"sha256:{prediction_hash}:{cell_index * 96 + position * 3}:3"
        heldout_chunks.append(registry._json_bytes(chunk))
    kwargs = {name: True for name in ("phase_order_verified", "input_model_pairing_verified",
        "private_plan_metadata_verified", "durable_evidence_verified", "case_generation_verified",
        "discovery_freeze_verified")}
    results = score_heldout(manifest, freeze, prediction, public_commitment, snapshot.heldout,
                            snapshot.schedule, *roots, iter(heldout_chunks),
                            exposure_classification="unblinded", **kwargs)
    exposure = {"schema": "synthetic-v2-exposure-v1", "classification": "unblinded",
                "basis": "fabricated fixture"}
    reveal = canonical_json({"schema": "synthetic-v2-reveal-v1", "heldout": heldout, "schedule": schedule,
        "heldout_root": roots[0].hex(), "site_nonce": roots[1].hex(), "prediction_nonce": roots[2].hex(),
        "prediction_record": json.loads(prediction), "exposure": {"schema": "synthetic-v2-exposure-reveal-v1",
        "classification": "unblinded", "private_record_sha256": _sha(canonical_json(exposure))}})
    private_material = canonical_json({"schema": "synthetic-v2-private-material-v1", "heldout": heldout,
        "schedule": schedule, "heldout_root": roots[0].hex(), "site_nonce": roots[1].hex(),
        "prediction_nonce": roots[2].hex()})
    return {"manifest": manifest, "freeze": freeze, "prediction": prediction,
            "commitment": public_commitment, "reveal": reveal, "exposure": canonical_json(exposure),
            "private_material": private_material, "results": results,
            "discovery_chunks": discovery_chunks, "heldout_chunks": heldout_chunks}


def save_fixture(root, fixture):
    """Persist only fabricated records using the actual two-registry layout."""
    root = Path(root)
    root.mkdir()
    pre = json.loads(fixture["manifest"])
    prep_fields = ("mode", "context", "context_sha256", "pilot_exclusions_sha256",
                   "development_exclusions_sha256", "historical_exclusions_sha256",
                   "historical_exclusions_count", "discovery_cases", "heldout_commitment",
                   "schedule_commitment")
    prep = canonical_json({key: pre[key] for key in prep_fields})
    identity = {"protocol": {"id": "synthetic-causal-v2", "pre_manifest_sha256": _sha(fixture["manifest"])},
                "inputs": {"public_preparation_sha256": _sha(prep)},
                "model": {"states": pre["context"]["model_state_hashes"]},
                "code": {"source_sha256": pre["source_sha256"]},
                "metadata": {"scope": "one-shot cooperative custodian"}}
    public = {"pre_manifest.json": fixture["manifest"], "prediction_freeze.json": fixture["freeze"],
              "prediction_commitment.json": fixture["commitment"], "results.json": fixture["results"],
              "reveal.json": fixture["reveal"]}
    for stage in ("before_discovery", "after_discovery", "after_heldout", "before_success"):
        public[f"source_inventory_{stage}.json"] = canonical_json({
            "schema": "synthetic-v2-source-inventory-v1", "stage": stage,
            "pre_manifest_sha256": _sha(fixture["manifest"]), "source_sha256": pre["source_sha256"]})
    private = {"private_material.json": fixture["private_material"], "prediction_record.json": fixture["prediction"],
               "exposure_record.json": fixture["exposure"]}
    for phase, suffix in (("discovery_chunks", "d"), ("heldout_chunks", "h")):
        private.update({f"cell-{index:02d}-{suffix}.json": raw
                        for index, raw in enumerate(fixture[phase])})

    def write_registry(name, artifacts, details):
        limits = {key: len(value) for key, value in artifacts.items()}
        local_identity = dict(identity)
        required = sorted(artifacts)
        if name == "private":
            from looking_glass.v2_evidence import _plan
            chunk_limits = {key: value for key, value in limits.items() if key.startswith("cell-")}
            extras = {key: value for key, value in limits.items() if not key.startswith("cell-")}
            local_identity["protocol"], limits, required = _plan(identity["protocol"], identity["metadata"],
                chunk_limits, extras, sorted(extras), 65536, 0, 1024 ** 3)
        else:
            limits["failure.json"] = 16384
        run = registry.create_run(root / name, **local_identity, artifact_limits=limits,
            required_artifacts=required, max_total_bytes=sum(limits.values()))
        for key, value in artifacts.items():
            run.write_artifact(key, value)
        run.finalize("succeeded", details)

    write_registry("private", private, {"scope": "private evidence", "submissions": 57408,
                                          "root_forwards": 57408})
    write_registry("public", public, {"scope": "complete study", "submissions": 57408,
        "root_forwards": 57408, "private_outcome_sha256": _sha((root / "private" / "outcome.json").read_bytes())})
    return root
