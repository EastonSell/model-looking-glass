"""Independent CPU-only audit of a saved exposed-pair owner and public result.

This checks the recorded evidence and does not attest native code, device
isolation, unrecorded full logits, or the physical GPU execution.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

GIB, MIB = 1024 ** 3, 1024 ** 2
MODEL = "distilbert/distilbert-base-uncased"
REV = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MANIFEST = "20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5"
ASSETS = {
    "config.json": (483, "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83"),
    "model.safetensors": (267954768, "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"),
    "tokenizer.json": (466062, "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98"),
    "tokenizer_config.json": (48, "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"),
    "vocab.txt": (231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
}
SOURCES = (
    "looking_glass/__init__.py", "looking_glass/bert_gpu_exposed_pair.py",
    "looking_glass/bert_gpu_evidence.py", "docs/BERT_GPU_DIAGNOSTIC_WRITER_V2.md",
    "experiments/bert_agreement_public_cases_v1.json", "looking_glass/gpu_oneshot.py",
    "looking_glass/device_identity.py", "looking_glass/device_lease.py",
    "looking_glass/supervisor.py", "looking_glass/memory_budget.py",
    "looking_glass/intervention_plan.py", "looking_glass/interventions.py",
    "looking_glass/model_ownership.py", "docs/BERT_GPU_EXPOSED_PAIR_V1.md",
    "scripts/bert_gpu_exposed_pair_bootstrap.py",
)
STAGES = (("before_model_load", 0), ("before_model_copy", None), ("after_model_copy", None),
          ("before_target", 0), ("after_target", 1), ("before_donor", 1), ("after_donor", 2),
          ("before_self", 2), ("after_self", 3), ("before_cross", 3), ("after_cross", 4),
          ("cleanup_before_workspace_clear", None), ("after_cleanup", None))
SCOPE = "cumulative_since_pre_model_load_reset"
MEMORY = ("device_free_bytes", "device_total_bytes", "allocated_bytes", "reserved_bytes",
          "peak_allocated_bytes", "peak_reserved_bytes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, limit=2 * MIB):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "evidence exceeds bounded size")
    def unique(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    require(type(value) is dict, "expected an evidence object")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(path):
    digest, count = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            count += len(block)
            digest.update(block)
    return count, digest.hexdigest()


def check_source(checkout, commit, saved):
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid source commit")
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT")
           if key in os.environ}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_COUNT": "0", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0"})
    def git(*args):
        return subprocess.check_output(["git", "--no-replace-objects", *args], cwd=checkout, env=env)
    require(git("rev-parse", "HEAD").decode().strip() == commit, "checkout commit changed")
    require(not git("status", "--porcelain", "--untracked-files=no"), "tracked checkout changed")
    require(type(saved) is dict and set(saved) == set(SOURCES), "source inventory differs")
    for relative in SOURCES:
        raw = (checkout / relative).read_bytes()
        require(raw == git("show", commit + ":" + relative), "source bytes differ: " + relative)
        require(saved[relative] == hashlib.sha256(raw).hexdigest(), "source digest differs: " + relative)
    require(saved["experiments/bert_agreement_public_cases_v1.json"] == MANIFEST,
            "public case manifest differs from the fixed digest")


def check_memory(record, total, cap, commitment, previous):
    require(type(record) is dict and set(record) == set(MEMORY), "memory fields differ")
    require(all(type(record[k]) is int and record[k] >= 0 for k in MEMORY), "memory values are not integers")
    free, observed, alloc, reserved, palloc, preserved = (record[k] for k in MEMORY)
    require(observed == total and free <= total and free + reserved <= total, "memory total/free differs")
    require(alloc <= reserved <= total and alloc <= palloc <= preserved <= total and reserved <= preserved,
            "allocator counters contradict")
    require(palloc >= previous[0] and preserved >= previous[1], "cumulative peaks decreased")
    reserve = max(GIB, (15 * total + 99) // 100)
    require(free >= reserve + commitment and reserved + reserve + commitment <= total,
            "recorded owner reserve was insufficient")
    require(free >= reserve + 256 * MIB and preserved <= cap - 256 * MIB,
            "recorded allocator headroom was insufficient")
    return palloc, preserved


def decoded_vector(value):
    require(type(value) is str, "missing activation vector")
    raw = base64.b64decode(value, validate=True)
    require(len(raw) == 3072 and base64.b64encode(raw).decode("ascii") == value,
            "activation vector encoding differs")
    require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)), "nonfinite activation")
    return raw


def selected_float(bits, saved):
    require(type(bits) is int and 0 <= bits <= 0xffffffff, "invalid selected bits")
    require(type(saved) is float and math.isfinite(saved), "invalid selected logit")
    decoded = struct.unpack("<f", struct.pack("<I", bits))[0]
    require(math.isfinite(decoded) and struct.pack("<d", saved) == struct.pack("<d", decoded),
            "selected logit differs from stored bits")
    return decoded


def check_public_partial(result, run_dir):
    projected = {}
    for name, count in STAGES:
        try:
            record = read_json(run_dir / (name + ".json"), 4096)
        except (OSError, ValueError, UnicodeError):
            continue
        mem = record.get("memory")
        if (type(record.get("observations_live")) is type(count) and record["observations_live"] == count
                and record.get("peak_scope") == SCOPE and type(mem) is dict
                and all(type(mem.get(key)) is int and mem[key] >= 0 for key in MEMORY)):
            projected[name] = {"memory": {key: mem[key] for key in MEMORY},
                               "observations_live": count, "peak_scope": SCOPE}
    try:
        weakrefs = read_json(run_dir / "cleanup_weakrefs.json", 4096)
    except (OSError, ValueError, UnicodeError):
        weakrefs = {}
    names = ("model", "target", "donor")
    if all(type(weakrefs.get(key)) is bool for key in names):
        projected["cleanup_weakrefs"] = {key: weakrefs[key] for key in names}
    require(canonical(result.get("partial_stage_observations")) == canonical(projected or None),
            "public partial observations differ from the typed raw projection")
    return len(projected)


def check_measurement(m, raw, stages, cache):
    require(type(m) is dict and type(raw) is dict, "missing measurement")
    expected_keys = {"target_row_id", "donor_row_id", "manifest_sha256", "target_ids", "donor_ids",
        "output_token_ids", "seed", "budget_active_before_candidate_write", "runtime_uuid_verified",
        "device_name", "device_total_bytes", "compute_capability", "allocator_fraction", "allocator_cap_bytes",
        "allocator_backend", "whole_run_commitment_bytes", "gpu_trace_claim_bytes", "host_trace_claim_bytes",
        "torch_version", "cuda_build", "execution_config", "peak_scope", "python_version", "model_metadata",
        "sequence", "pair_analysis", "cleanup", "cleanup_weakrefs", "workspace_cleanup"}
    require(set(m) == expected_keys, "public measurement has extra or missing fields")
    fixed = {"target_row_id": "evaluation-f01-A-s-s", "donor_row_id": "evaluation-f01-A-p-s",
        "manifest_sha256": MANIFEST, "target_ids": [101,1996,5276,3875,1996,3063,103,3201,1012,102],
        "donor_ids": [101,1996,12499,3875,1996,3063,103,3201,1012,102],
        "output_token_ids": {"is": 2003, "are": 2024}, "seed": 17,
        "budget_active_before_candidate_write": 1, "runtime_uuid_verified": True,
        "device_name": "NVIDIA GeForce RTX 2070 SUPER", "compute_capability": [7,5],
        "allocator_fraction": .25, "allocator_backend": "native", "gpu_trace_claim_bytes": 65536,
        "host_trace_claim_bytes": MIB, "torch_version": "2.8.0+cu128", "cuda_build": "12.8",
        "peak_scope": SCOPE,
        "workspace_cleanup": "PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces"}
    for key, expected in fixed.items():
        require(type(m.get(key)) is type(expected) and m[key] == expected, "fixed field differs: " + key)
        require(canonical(raw.get(key)) == canonical(m[key]), "raw and public field differ: " + key)
    total, cap, commitment = (m.get(k) for k in ("device_total_bytes", "allocator_cap_bytes", "whole_run_commitment_bytes"))
    require(all(type(v) is int for v in (total, cap, commitment)) and 7*GIB <= total <= 9*GIB
            and cap == total // 4 and commitment == cap + 256*MIB, "device commitment differs")
    require(all(type(raw.get(key)) is int and raw[key] == m[key]
                for key in ("device_total_bytes", "allocator_cap_bytes", "whole_run_commitment_bytes")),
            "raw/public device commitments differ")
    expected_config = {"device":"cuda:0", "threads":1, "interop_threads":1,
        "deterministic_algorithms":True, "matmul_tf32":False, "cudnn_tf32":False,
        "cudnn_benchmark":False, "cudnn_deterministic":True,
        "overwrite_module_params_on_conversion":False, "swap_module_params_on_conversion":False}
    require(canonical(m.get("execution_config")) == canonical(expected_config)
            and canonical(raw.get("execution_config")) == canonical(expected_config),
            "execution configuration differs")
    meta = m.get("model_metadata")
    meta_keys = {"model", "revision", "assets", "model_class", "tokenizer_class", "transformers_version",
                 "tokenizers_version", "safetensors_version", "attention_implementation", "parameter_count",
                 "cpu_tied_weight_verified"}
    require(type(meta) is dict and set(meta) == meta_keys and meta.get("model") == MODEL and meta.get("revision") == REV
            and meta.get("model_class") == "DistilBertForMaskedLM"
            and meta.get("tokenizer_class") == "DistilBertTokenizerFast"
            and meta.get("transformers_version") == "4.57.3" and meta.get("attention_implementation") == "eager"
            and meta.get("parameter_count") == 66985530 and meta.get("cpu_tied_weight_verified") is True,
            "model metadata differs")
    for key in ("python_version",):
        require(type(m.get(key)) is str and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", m[key]), "runtime version invalid")
    for key in ("tokenizers_version", "safetensors_version"):
        require(type(meta.get(key)) is str and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", meta[key]), "dependency version invalid")
    expected_assets = {name: {"size": size, "sha256": sha} for name, (size, sha) in ASSETS.items()}
    require(meta.get("assets") == expected_assets, "checkpoint asset record differs")
    snapshot = cache / "models--distilbert--distilbert-base-uncased" / "snapshots" / REV
    for name, expected in ASSETS.items():
        require(identity(snapshot / name) == expected, "checkpoint asset bytes differ: " + name)
    require(type(raw.get("model_metadata")) is dict
            and all(canonical(raw["model_metadata"].get(k)) == canonical(v) for k, v in meta.items()),
            "raw/public metadata differ")
    sequence, raw_sequence = m.get("sequence"), raw.get("sequence")
    require(type(sequence) is dict and type(raw_sequence) is dict
            and set(sequence) == {"forward_count", "self_mask_logits_bitwise_equal", "cross_source_landed",
                                  "source_differs_from_target", "observations"}, "missing or extra sequence fields")
    require(type(sequence.get("forward_count")) is int and sequence["forward_count"] == 4
            and all(sequence.get(k) is True for k in
                    ("self_mask_logits_bitwise_equal", "cross_source_landed", "source_differs_from_target")),
            "sequence result incomplete")
    require(all(canonical(raw_sequence.get(k)) == canonical(sequence[k]) for k in
                ("forward_count", "self_mask_logits_bitwise_equal", "cross_source_landed", "source_differs_from_target")),
            "raw/public sequence status differs")
    rows, raw_rows = sequence.get("observations"), raw_sequence.get("observations")
    require(type(rows) is list and len(rows) == 4 and type(raw_rows) is list and len(raw_rows) == 4,
            "observation count differs")
    expected_row_keys = {"stage", "is_bits", "are_bits", "is_logit", "are_logit", "mask_logits_sha256",
                         "before_float32_le_base64", "after_float32_le_base64", "events"}
    before, after, margins = [], [], {}
    path = "distilbert.transformer.layer.2"
    for index, name in enumerate(("target", "donor", "self", "cross")):
        row, raw_row = rows[index], raw_rows[index]
        require(type(row) is dict and set(row) == expected_row_keys and row.get("stage") == name,
                "public row shape or order differs")
        require(type(raw_row) is dict and all(canonical(raw_row.get(k)) == canonical(v) for k, v in row.items()),
                "raw/public row differs")
        a, b = selected_float(row["is_bits"], row["is_logit"]), selected_float(row["are_bits"], row["are_logit"])
        margins[name] = math.fsum((a, -b))
        require(type(row["mask_logits_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["mask_logits_sha256"]),
                "mask-logit digest invalid")
        before.append(decoded_vector(row["before_float32_le_base64"]))
        after.append(None if row["after_float32_le_base64"] is None else decoded_vector(row["after_float32_le_base64"]))
        names = ["before"] if index < 2 else ["before", "patch", "after"]
        ops = ["capture"] if index < 2 else ["capture", "patch", "capture"]
        events = [{"name": n, "operation": op, "order": j, "module_path": path, "module_aliases": [path],
            "call_site_known": False, "forward_index": 0, "invocation": 0, "batch_index": 0,
            "token_index": 2, "output_selector": [0], "output_shape": [1,10,768], "selected_shape": [768],
            "dtype": "torch.float32", "device": "cuda:0", "source_capture": "before" if index == 2 and j == 1 else None,
            "preceding_same_site_actions": names[:j]} for j, (n, op) in enumerate(zip(names, ops))]
        require(canonical(row["events"]) == canonical(events), "action coordinates differ")
    require(after[:2] == [None,None] and before[0] != before[1] and before[2] == before[0]
            and after[2] == before[0] and before[3] == before[0] and after[3] == before[1], "patch vector landing differs")
    require(rows[2]["mask_logits_sha256"] == rows[0]["mask_logits_sha256"]
            and all(rows[2][k] == rows[0][k] for k in ("is_bits", "are_bits")), "self control differs")
    analysis = {"margins": margins, "target_minus_cross": math.fsum((margins["target"], -margins["cross"])),
                "strict_sign_change": (margins["target"] > 0 > margins["cross"])
                                      or (margins["target"] < 0 < margins["cross"])}
    require(canonical(m.get("pair_analysis")) == canonical(analysis)
            and canonical(raw.get("pair_analysis")) == canonical(analysis), "margin analysis differs")
    previous = (0, 0)
    for name, count in STAGES:
        record = stages.get(name)
        require(type(record) is dict and set(record) == {"memory", "observations_live", "peak_scope"}
                and type(record["observations_live"]) is type(count) and record["observations_live"] == count
                and record["peak_scope"] == SCOPE, "missing or inconsistent stage: " + name)
        previous = check_memory(record["memory"], total, cap, commitment, previous)
    require(stages["before_model_load"]["memory"]["device_free_bytes"] >= 3*GIB, "initial free threshold failed")
    cleanup = stages["after_cleanup"]["memory"]
    require(cleanup == m.get("cleanup") == raw.get("cleanup") and cleanup["allocated_bytes"] == 0
            and cleanup["reserved_bytes"] == 0, "final allocator cleanup differs")
    weakrefs = {"model": False, "target": False, "donor": False}
    require(canonical(m.get("cleanup_weakrefs")) == canonical(weakrefs)
            and canonical(raw.get("cleanup_weakrefs")) == canonical(weakrefs)
            and canonical(stages.get("cleanup_weakrefs")) == canonical(weakrefs), "cleanup weakrefs differ")
    return analysis


def audit(outcome_path, run_dir, checkout, cache):
    result = read_json(outcome_path)
    public_keys = {"schema", "run_id", "source_commit", "source_sha256", "controller_fresh_pycache", "status",
        "supervisor", "supervisor_error_type", "supervision_evidence_error_type", "nvidia_smi_before",
        "nvidia_smi_after", "owner_presence_after_exit", "post_telemetry_error_type", "limitations",
        "partial_stage_observations", "measurement", "candidate_error_type", "controller_import_error_type",
        "owner_error_type"}
    require(set(result).issubset(public_keys), "public result has an unexpected field")
    for field in ("supervisor_error_type", "supervision_evidence_error_type", "post_telemetry_error_type",
                  "candidate_error_type", "controller_import_error_type", "owner_error_type"):
        value = result.get(field)
        require(value is None or (type(value) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", value)),
                "public error field is not a bounded error type")
    smi_keys = {"device_name", "total_bytes", "free_bytes", "used_bytes", "driver_version", "timestamp_utc"}
    for field in ("nvidia_smi_before", "nvidia_smi_after"):
        value = result.get(field)
        if value is None:
            continue
        require(type(value) is dict and set(value).issubset(smi_keys), "public device snapshot fields differ")
        for key in ("total_bytes", "free_bytes", "used_bytes"):
            require(key not in value or (type(value[key]) is int and value[key] >= 0), "invalid public device bytes")
        require("device_name" not in value or value["device_name"] == "NVIDIA GeForce RTX 2070 SUPER",
                "public device name differs")
        require("driver_version" not in value or (type(value["driver_version"]) is str
                and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", value["driver_version"])), "invalid driver version")
        require("timestamp_utc" not in value or (type(value["timestamp_utc"]) is str
                and re.fullmatch(r"[0-9TZ:+.\-]{10,40}", value["timestamp_utc"])), "invalid timestamp")
    require(result.get("schema") == "bert-gpu-exposed-pair-result-v1", "result schema differs")
    commit, run_id = result.get("source_commit"), result.get("run_id")
    require(type(commit) is str and type(run_id) is str and re.fullmatch(r"[0-9a-f]{32}", run_id), "identity invalid")
    check_source(checkout, commit, result.get("source_sha256"))
    require(result.get("status") in ("success", "incomplete", "quarantined"), "invalid outcome status")
    started = read_json(run_dir / "started.json")
    require(started.get("schema") == "bert-gpu-exposed-pair-start-v1" and started.get("run_id") == run_id
            and started.get("source_commit") == commit, "started evidence differs")
    partial_count = check_public_partial(result, run_dir)
    presence = result.get("owner_presence_after_exit")
    require(type(presence) is dict and set(presence) == {"query", "owner_pid_listed"}
            and presence.get("query") in ("ok", "unavailable", "inconclusive", "not_attempted"),
            "owner presence query invalid")
    require((presence.get("query") == "ok" and type(presence.get("owner_pid_listed")) is bool)
            or (presence.get("query") != "ok" and presence.get("owner_pid_listed") is None),
            "owner presence result contradicts its query status")
    report = {"source_commit": commit, "run_id": run_id, "status": result["status"],
              "complete_evidence_audited": False, "typed_partial_record_count": partial_count,
              "limitations": ["Recorded evidence only; no native-runtime or GPU isolation attestation.",
                              "Full mask logits are not retained and cannot be reconstructed from hashes."]}
    if "measurement" not in result:
        report["note"] = "No complete public measurement was emitted; partial records remain diagnostic."
        return report
    owner = read_json(run_dir / "owner.json")
    candidate, final = read_json(run_dir / "candidate.json"), read_json(run_dir / "final.json")
    supervision = read_json(run_dir / "supervision.json")
    require(owner.get("schema") == "bert-gpu-exposed-pair-owner-v1"
            and candidate.get("schema") == "bert-gpu-exposed-pair-candidate-v1"
            and final.get("schema") == "bert-gpu-exposed-pair-final-v1", "owner evidence schema differs")
    pid, supervisor_run_id = supervision.get("pid"), supervision.get("supervisor_run_id")
    require(type(pid) is int and pid > 0 and type(supervisor_run_id) is str
            and supervision.get("run_id") == run_id and type(supervision.get("exit_code")) is int
            and supervision["exit_code"] == 0 and supervision.get("clean") is True
            and supervision.get("timed_out") is False, "supervision evidence not clean")
    require(canonical(result.get("supervisor")) == canonical({"exit_code": 0, "clean": True, "timed_out": False}),
            "public supervision differs from retained evidence")
    for record in (owner, candidate, final):
        require(record.get("run_id") == run_id and record.get("supervisor_run_id") == supervisor_run_id
                and type(record.get("pid")) is int and record["pid"] == pid
                and record.get("source_commit") == commit, "owner identity differs")
    require(final.get("candidate_sha256") == identity(run_dir / "candidate.json")[1]
            and type(final.get("active_memory_leases")) is int and final["active_memory_leases"] == 0,
            "final candidate digest or lease count differs")
    stages = {name: read_json(run_dir / (name + ".json"), 4096) for name, _ in STAGES}
    stages["cleanup_weakrefs"] = read_json(run_dir / "cleanup_weakrefs.json", 4096)
    require(canonical(result.get("partial_stage_observations")) == canonical(stages),
            "public and private stage observations differ")
    analysis = check_measurement(result["measurement"], candidate.get("measurement"), stages, cache)
    if result["status"] == "success":
        require(presence.get("query") == "ok" and presence.get("owner_pid_listed") is False,
                "success lacks the required owner-presence observation")
        require(result.get("supervisor_error_type") is None and result.get("supervision_evidence_error_type") is None
                and result.get("post_telemetry_error_type") is None and result.get("controller_fresh_pycache") is True,
                "success conflicts with controller diagnostics")
    elif presence.get("query") != "ok":
        require(presence.get("owner_pid_listed") is None, "unavailable presence was treated as a boolean")
    report.update({"complete_evidence_audited": True, "pair_analysis": analysis,
                   "owner_presence_after_exit": presence,
                   "note": "A successful audit checks consistency of retained evidence, not unobserved device activity."})
    return report


def sanitized_report(audited):
    require(audited.get("complete_evidence_audited") is True, "a complete evidence audit is required")
    presence = audited["owner_presence_after_exit"]
    return {
        "schema": "bert-exposed-pair-independent-audit-v1",
        "audit_status": "pass",
        "recorded_run_status": audited["status"],
        "checks": {
            "source_and_public_manifest_rehashed": True,
            "all_five_checkpoint_assets_rehashed": True,
            "supervision_and_owner_evidence_consistent": True,
            "candidate_digest_and_recorded_zero_active_leases_checked": True,
            "complete_typed_stage_and_cleanup_records_verified": True,
            "recorded_memory_headroom_and_cumulative_peaks_checked": True,
            "float32_bits_margins_and_patch_vectors_recomputed": True,
            "public_projection_allowlists_checked": True,
            "owner_not_listed_in_post_exit_query": (presence["query"] == "ok"
                                                    and presence["owner_pid_listed"] is False),
        },
        "typed_stage_and_cleanup_record_count": audited["typed_partial_record_count"],
        "pair_analysis": audited["pair_analysis"],
        "scope": ["One previously exposed public evaluation pair; no new held-out or lexical-control claim.",
                  "Checks retained evidence consistency; native libraries and GPU isolation are not attested.",
                  "An owner not listed in the query is an observation, not proof of process isolation.",
                  "Full mask logits are not retained and cannot be reconstructed from their hashes."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="write a new path-free sanitized report")
    args = parser.parse_args()
    result = audit(args.outcome, args.run_dir, args.checkout, args.cache)
    if args.report is not None:
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(sanitized_report(result), stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
