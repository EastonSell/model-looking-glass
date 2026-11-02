"""Execute the fixed 192-row public copying baseline on one offline CPU."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import traceback

if __name__ == "__main__" and (
        not sys.dont_write_bytecode or not sys.pycache_prefix
        or not Path(sys.pycache_prefix).is_absolute() or not Path(sys.pycache_prefix).is_dir()
        or any(Path(sys.pycache_prefix).iterdir()) or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
        or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or any(k in os.environ for k in ("MKL_CBWR", "ATEN_CPU_CAPABILITY"))
        or not os.environ.get("COPY_BEHAVIOR_BOOTSTRAP_SHA256") or "torch" in sys.modules):
    raise RuntimeError("use the reviewed CPU/offline bootstrap before project imports")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network
GUARD_EVENTS = []
SAFE_QUERIES = {"cuda_is_available": 0, "cuda_device_count": 0}
if __name__ == "__main__":
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)
from scripts import copying_head_prompt_dose_v1 as old
from looking_glass.copy_behavior_screen_v1 import ROW_BYTES, score_screen
from looking_glass.pretrained import ASSETS
from looking_glass.copy_head_intervention import assert_no_forward_hooks
from looking_glass.copy_head_factorial import binary64_rounding_probe
from scripts.copy_behavior_screen_bootstrap_v1 import verify_manifest
prior, base = old.prior, old.base
base.GUARD_EVENTS = GUARD_EVENTS

PROTOCOL = "docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_V1.md"
AMENDMENT = "docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_RESOURCE_AMENDMENT_V1.md"
CASES = "experiments/distilgpt2_copy_behavior_screen_cases_v1.json"
MANIFEST = "experiments/distilgpt2_copy_behavior_screen_manifest_v1.json"
CASES_SHA = "96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734"
PROTOCOL_SHA = "78907b74bf3edb44a47c794508a9385da23b2bf2f26d07df4fefe74542b58875"
AMENDMENT_SHA = "832d115a81dc58359f5ba5189222253ea002036fa46a37f99b1813b5f51a8f01"
ROWS, RAW_LIMIT = 192, 192 * ROW_BYTES
STAGING_BYTES, METADATA_RESERVE, FAILURE_RESERVE = 355_679_200, 32 * 1024**2, 1024**2
WORK_ROOT = Path("E:/Codex/model-looking-glass/work")
MAX_MANIFEST_BYTES, MAX_JOURNAL_BYTES = 1024**2, 10 * 1024**2
MAX_RESULT_BYTES, MAX_FAILURE_BYTES = 5 * 1024**2, 1024**2


class BoundedJournal:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        if self.stream.tell() + len(data) > MAX_JOURNAL_BYTES:
            raise OSError("journal exceeds fixed metadata bound")
        count = self.stream.write(data)
        if count != len(data):
            raise OSError("short journal write")
        return count

    def flush(self):
        self.stream.flush()

    def fileno(self):
        return self.stream.fileno()


def bounded_json(path, value, limit):
    if len(base.canonical(value)) > limit:
        raise OSError("JSON exceeds fixed metadata bound")
    return base.durable_json(path, value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_paths(output, stage):
    root = WORK_ROOT.resolve(strict=True)
    parent, stage = output.parent.resolve(strict=True), stage.resolve(strict=True)
    if (not output.is_absolute() or not stage.is_absolute() or output.exists()
            or not parent.is_relative_to(root) or not stage.is_relative_to(root)
            or parent.is_relative_to(ROOT) or stage.is_relative_to(ROOT)):
        raise ValueError("output and existing stage must be under E work, outside checkout")
    needed = RAW_LIMIT + STAGING_BYTES + METADATA_RESERVE + FAILURE_RESERVE
    if shutil.disk_usage(parent).free < needed or shutil.disk_usage("C:/").free < 1024**3:
        raise OSError("insufficient E or C headroom at phase start")
    return parent, stage, needed


def verify_reservation(cases, manifest):
    version = manifest.get("reservation_version")
    if type(version) is not int or version < 40:
        raise ValueError("invalid reservation version")
    previous = importlib.import_module(f"looking_glass.pretrained_exposures_v{version-1}")
    reservation = importlib.import_module(f"looking_glass.pretrained_exposures_v{version}")
    before, before_sha = previous._checked_inventory()
    current, current_sha = reservation._checked_inventory()
    if (before_sha != manifest["predecessor_inventory_sha256"]
            or current_sha != manifest["reservation_inventory_sha256"]
            or current["records"][:len(before["records"])] != before["records"]
            or len(current["records"]) != len(before["records"]) + ROWS
            or len({r["record_id"] for r in current["records"]}) != len(current["records"])):
        raise RuntimeError("reservation predecessor, count or identities differ")
    for case, record in zip(cases, current["records"][len(before["records"]):]):
        if (record.get("record_id") != "planned-" + case["case_id"]
                or record.get("category") != "planned" or record.get("token_ids") != case["token_ids"]
                or record.get("target_token_id") != case["target_id"] or record.get("public_text") != case["text"]):
            raise RuntimeError("planned case identity or target differs")
    return {"predecessor_sha256": before_sha, "reservation_sha256": current_sha,
            "predecessor_records": len(before["records"]), "reservation_records": len(current["records"])}


def verify_tokenization(tokenizer, data):
    cases, groups = data["cases"], data["groups"]
    if (len(cases) != ROWS or len(groups) != 24 or len({c["case_id"] for c in cases}) != ROWS
            or any(c["group"] != i // 8 + 1 or c["variant"] != i % 8 for i, c in enumerate(cases))):
        raise ValueError("fixed case ordering differs")
    for case in cases:
        group = groups[case["group"] - 1]
        rotation = case["variant"] // 4
        word = group["words"][5 if rotation == 0 else 0]
        if (case["rotation"] != rotation or case["target_id"] != group["token_ids"][5 if rotation == 0 else 0]
                or tokenizer.encode(case["text"], add_special_tokens=False) != case["token_ids"]
                or tokenizer.encode(case["text"] + " " + word, add_special_tokens=False)
                   != case["token_ids"] + [case["target_id"]]):
            raise ValueError("fixed input or target tokenization differs")


def perform_action(case, index, *, model, runtime_before, path, log, counts, activity, completed,
                   run_forward=None, runtime=None):
    if run_forward is None:
        run_forward = prior.run_forward
    if runtime is None:
        runtime = prior.runtime
    observation = {}
    marker = {"recorded": False, "partial": False, "partial_save_error": None}
    activity.update({"phase": "forward", "case_id": case["case_id"], "index": index,
                     "observation": observation, "marker": marker})
    ids = torch.tensor([case["token_ids"]], dtype=torch.int64, device="cpu")
    base._append_event(log, {"event": "start", "case_id": case["case_id"], "index": index})
    if runtime(model) != runtime_before:
        raise RuntimeError("runtime changed before forward")
    start = path.stat().st_size if path.exists() else 0
    if start != index * ROW_BYTES:
        raise RuntimeError("raw matrix offset differs")
    prev_returns, prev_bound, prev_retained = counts["model_returns"], counts["returned"], counts["retained"]
    sentinel = object()
    payload = sentinel
    counts["submitted"] += 1
    try:
        payload = run_forward(model, ids, "clean", observation=observation)
        observation["helper_return_observed"] = True
        counts["model_returns"] = prev_returns + int(observation.get("model_call_return_observed", False))
        counts["returned"] = prev_bound + 1
        counts["retained"] = prev_retained + sum(map(len, payload.values()))
        if set(payload) != {"logits"} or len(payload["logits"]) != ROW_BYTES or counts["retained"] > RAW_LIMIT:
            raise RuntimeError("returned screen payload differs from its fixed bound")
        activity["phase"] = "persist_and_validate"
        with path.open("ab") as raw:
            offsets = base._write_payload(raw, payload)
        base._append_event(log, {"event": "raw_recorded", "case_id": case["case_id"],
                                 "index": index, "raw": offsets})
        marker["recorded"] = True
        if (ids.tolist() != [case["token_ids"]] or runtime(model) != runtime_before
                or any((x[0] & 0x7f800000) == 0x7f800000 for x in struct.iter_unpack("<I", payload["logits"]))):
            raise RuntimeError("input, runtime or finite logits changed")
        row = {"case_id": case["case_id"], "index": index, "group": case["group"],
               "variant": case["variant"], "target_id": case["target_id"], "raw": offsets}
        base._append_event(log, {"event": "complete", **row})
        completed.append(row)
    except BaseException:
        bound = payload is not sentinel
        observation["helper_return_observed"] = bound
        observation["helper_return_unbound_uncertain"] = bool(observation.get("model_call_return_observed", False) and not bound)
        counts["model_returns"] = prev_returns + int(observation.get("model_call_return_observed", False))
        counts["returned"] = prev_bound + int(bound)
        partial = observation.get("payload", {})
        size = sum(map(len, partial.values()))
        counts["retained"] = prev_retained + size if marker["recorded"] else prev_retained
        if partial and not marker["recorded"]:
            try:
                if prev_retained + size > RAW_LIMIT:
                    raise RuntimeError("partial payload exceeds fixed bound")
                if old.recover_payload(log, path, case["case_id"], "clean", partial, start,
                                       observation.get("model_call_return_observed", False)):
                    marker["partial"] = True
                    counts["retained"] = prev_retained + size
            except Exception as error:
                marker["partial_save_error"] = type(error).__name__
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if any(not p.is_absolute() for p in (args.cache_dir, args.stage_dir, args.output_dir, args.manifest)):
        raise ValueError("all paths must be absolute")
    digest = verify_manifest(args.manifest)
    if os.environ.get("COPY_BEHAVIOR_BOOTSTRAP_SHA256") != digest:
        raise RuntimeError("bootstrap and manifest differ")
    manifest = json.loads(args.manifest.read_bytes())
    raw_cases = (ROOT / CASES).read_bytes()
    if (sha(raw_cases) != CASES_SHA or sha((ROOT / PROTOCOL).read_bytes()) != PROTOCOL_SHA
            or sha((ROOT / AMENDMENT).read_bytes()) != AMENDMENT_SHA):
        raise RuntimeError("scientific rules or case bytes differ")
    data = json.loads(raw_cases)
    cases = data["cases"]
    exposure = verify_reservation(cases, manifest)
    if sum(size for size, _ in ASSETS.values()) != STAGING_BYTES:
        raise RuntimeError("pinned temporary asset size differs from resource amendment")
    _, stage, needed = checked_paths(args.output_dir, args.stage_dir)
    torch.set_num_interop_threads(1)
    initial = base.numeric_policy(initialize=True)
    binary64_rounding_probe()
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    output = args.output_dir
    output.mkdir()
    run_manifest = {"schema": "distilgpt2-copy-behavior-screen-run-v1", "source_revision": manifest["source_revision"],
                    "submitted_manifest_sha256": digest, "cases_sha256": CASES_SHA, "protocol_sha256": PROTOCOL_SHA,
                    "amendment_sha256": AMENDMENT_SHA, "exposure": exposure, "dependencies": base.dependency_identity(),
                    "model_id": base.MODEL_ID, "model_revision": base.MODEL_REVISION, "weights_sha256": base.WEIGHTS_SHA256,
                    "raw_limit": RAW_LIMIT, "staging_bytes": STAGING_BYTES, "required_e_free": needed,
                    "initial_numeric_probe": initial, "stage_directory": "E work", "scope": "public, nonblind"}
    run_sha = bounded_json(output / "manifest.json", run_manifest, MAX_MANIFEST_BYTES)
    counts = {"submitted": 0, "model_returns": 0, "returned": 0, "retained": 0}
    activity = {"phase": "model_load", "case_id": None, "index": None, "observation": {}, "marker": {}}
    completed = []
    try:
        bundle = base.load_distilgpt2(cache_dir=args.cache_dir, allow_download=False, stage_dir=stage)
        if bundle.metadata["weights_sha256"] != base.WEIGHTS_SHA256:
            raise RuntimeError("checkpoint differs")
        model = bundle.model
        verify_tokenization(bundle.tokenizer, data)
        torch.random.default_generator.manual_seed(base.CPU_RNG_SEED)
        state_before, buffers_before, runtime_before = base.state_hash(model), base.buffer_hash(model), prior.runtime(model)
        assert_no_forward_hooks(model)
        path = output / "screen.f32"
        with (output / "attempt.jsonl").open("xb") as stream:
            log = BoundedJournal(stream)
            base._append_event(log, {"event": "runtime_ready", "runtime": runtime_before})
            for index, case in enumerate(cases):
                perform_action(case, index, model=model, runtime_before=runtime_before, path=path, log=log,
                               counts=counts, activity=activity, completed=completed)
        activity["phase"] = "verification"
        if (counts != {"submitted": ROWS, "model_returns": ROWS, "returned": ROWS, "retained": RAW_LIMIT}
                or len(completed) != ROWS or path.stat().st_size != RAW_LIMIT
                or base.state_hash(model) != state_before or base.buffer_hash(model) != buffers_before
                or prior.runtime(model) != runtime_before or verify_manifest(args.manifest) != digest
                or GUARD_EVENTS or torch.cuda.is_initialized()):
            raise RuntimeError("final counts, state, runtime or source differ")
        assert_no_forward_hooks(model)
        rows = []
        with path.open("rb") as stream:
            for _ in range(ROWS):
                rows.append(stream.read(ROW_BYTES))
            if stream.read(1):
                raise RuntimeError("unexpected trailing raw bytes")
        summary = score_screen(cases, rows)
        for family in summary["families"]:
            for row in family["rows"]:
                representative = row["top_representative_id"]
                row["top_representative_text"] = bundle.tokenizer.decode([representative], skip_special_tokens=False,
                                                                          clean_up_tokenization_spaces=False)
                row["top_representative_raw_token"] = bundle.tokenizer.convert_ids_to_tokens(representative)
                row["top_unique_text"] = row["top_representative_text"] if row["top_unique_id"] is not None else None
        result = {"schema": "distilgpt2-copy-behavior-screen-result-v1", "status": "complete",
                  "manifest_sha256": run_sha, "checkpoint": bundle.metadata,
                  "state_sha256_before": state_before, "state_sha256_after": base.state_hash(model),
                  "buffers_sha256_before": buffers_before, "buffers_sha256_after": base.buffer_hash(model),
                  "runtime": runtime_before, "counts": counts, "records": completed,
                  "raw_sha256": sha(path.read_bytes()), "summary": summary}
        bounded_json(output / "result.json", result, MAX_RESULT_BYTES)
    except BaseException as error:
        path = output / "screen.f32"
        info = {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())} if path.is_file() else None
        try:
            bounded_json(output / "failure.json", {"schema": "distilgpt2-copy-behavior-screen-failure-v1",
                "status": "incomplete", "manifest_sha256": run_sha, "phase": activity["phase"],
                "case_id": activity["case_id"], "index": activity["index"], "marker": activity["marker"],
                "model_call_return_observed": activity["observation"].get("model_call_return_observed", False),
                "helper_return_observed": activity["observation"].get("helper_return_observed", False),
                "helper_return_unbound_uncertain": activity["observation"].get("helper_return_unbound_uncertain", False),
                "counts": counts, "completed_records": len(completed), "raw": info,
                "error_type": type(error).__name__, "traceback_frames": [{"file": Path(f.filename).name,
                    "line": f.lineno, "function": f.name} for f in traceback.extract_tb(error.__traceback__)]}, MAX_FAILURE_BYTES)
        finally:
            raise
    print(json.dumps({"status": "complete", "forwards": ROWS,
                      "supported_development": summary["supported_development"],
                      "supported_held_out": summary["supported_held_out"]}, allow_nan=False))


if __name__ == "__main__":
    main()