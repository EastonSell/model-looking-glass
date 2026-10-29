"""Check this public subset's integrity and the stated partial arithmetic.

This is not an attestation of the source process, device cleanup, or profile.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "evidence"
EXPECTED_FILES = {
    "stages.jsonl", "events.jsonl", "preload-admission.json", "terminal-projection.json"
}
EXPECTED_KINDS = (
    "warmup", "clean_capture", "layer1_attn_cross_patch", "layer1_mlp_cross_patch",
    "layer2_attn_cross_patch", "layer2_mlp_cross_patch", "layer3_attn_cross_patch",
    "layer3_mlp_cross_patch", "layer4_attn_cross_patch", "layer4_mlp_cross_patch",
)
PHASES = ("started", "baseline", "baseline_check", "after_observed", "postcheck", "completed")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_headroom(budget: dict, snap: dict) -> None:
    committed = budget["committed_incremental_bytes"]
    reserve = budget["reserve_bytes"]
    require(committed == 3_758_096_384 and reserve == 1_288_441_037, "claims/reserve")
    require(budget["active_leases"] == 2, "lease count")
    require(budget["gpu_trace_bytes"] == budget["host_trace_bytes"] == 2_097_152,
            "trace claims")
    require(committed + reserve <= snap["device_free_bytes"], "free-side headroom")
    require(snap["process_reserved_bytes"] + committed + reserve <= snap["device_total_bytes"],
            "total-side headroom")
    require(snap["device_free_bytes"] >= 3 * 1024 ** 3, "minimum free")


def main() -> None:
    inventory = json.loads((HERE / "inventory.json").read_bytes())
    require(inventory["schema"] == "agreement-gpu-resource-incomplete-public-subset-v1",
            "inventory schema")
    require(set(inventory["files"]) == EXPECTED_FILES, "subset file list")
    data = {}
    for name, item in inventory["files"].items():
        data[name] = (HERE / name).read_bytes()
        require(len(data[name]) == item["size"] and sha(data[name]) == item["sha256"],
                f"file integrity: {name}")
    entries = inventory["stage_files_in_concatenation_order"]
    require(len(entries) == 63, "stage count")
    stages = []
    offset = 0
    for index, item in enumerate(entries):
        require(item["name"] == f"stage-{index:03}.json" and item["offset"] == offset,
                "stage order/offset")
        chunk = data["stages.jsonl"][offset:offset + item["size"]]
        require(len(chunk) == item["size"] and sha(chunk) == item["sha256"], "stage hash")
        require(chunk.endswith(b"\n") and chunk.count(b"\n") == 1, "stage delimiter")
        stages.append(json.loads(chunk))
        offset += len(chunk)
    require(offset == len(data["stages.jsonl"]), "trailing stage bytes")
    completed = []
    snapshots = []
    checks = 0
    for index, kind in enumerate(EXPECTED_KINDS):
        group = stages[index * 6:index * 6 + 6]
        require([x["phase"] for x in group] == list(PHASES)
                and all(x["kind"] == kind for x in group), "sample schedule")
        for snap_index, budget_index in ((1, 2), (3, 4)):
            recorded = group[snap_index]["snapshot"]
            budget = group[budget_index]["budget"]
            check_headroom(budget, recorded)
            check_headroom(budget, budget["telemetry"])
            snapshots.extend((recorded, budget["telemetry"]))
            checks += 1
        completed.append(group[5]["sample"])
    prefix = stages[60:]
    require([x["phase"] for x in prefix] == list(PHASES[:3])
            and all(x["kind"] == "null" for x in prefix), "interrupted prefix")
    budget = prefix[2]["budget"]
    check_headroom(budget, prefix[1]["snapshot"])
    check_headroom(budget, budget["telemetry"])
    snapshots.extend((prefix[1]["snapshot"], budget["telemetry"]))
    checks += 1
    preload = json.loads(data["preload-admission.json"])
    check_headroom(preload, preload["telemetry"])
    events = [json.loads(line) for line in data["events.jsonl"].splitlines()]
    require(len(events) == 10 and [e["index"] for e in events] == list(range(10)), "journal")
    require([e["event"]["sample_id"] for e in events] == list(EXPECTED_KINDS),
            "journal/sample order")
    require(max(x["peak_allocated_bytes"] for x in completed) == 369_747_456, "allocated peak")
    require(max(x["peak_reserved_bytes"] for x in completed) == 406_847_488, "reserved peak")
    require(min(x["device_free_bytes"] for x in snapshots) == 7_018_119_168, "minimum free")
    terminal = json.loads(data["terminal-projection.json"])
    require(terminal["controller_status"] == "quarantined"
            and terminal["supervisor"] == {"clean": False, "exit_code": 1, "timed_out": True},
            "terminal projection")
    print(f"PASS: {len(stages)} stages, {len(completed)} completed samples, "
          f"{checks} stage headroom pairs, {len(events)} journal events; incomplete, not approved")


if __name__ == "__main__":
    main()
