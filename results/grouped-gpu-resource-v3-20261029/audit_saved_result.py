"""Audit the closed, public grouped-GPU result without model or CUDA imports.

This checks saved ordering, source binding and resource arithmetic. The owner
did not retain raw logits, so its numerical comparison cannot be replayed.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


GIB = 1024 ** 3
CHECKPOINT_SHA256 = "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"
COUNTERS = ("device_free_bytes", "device_total_bytes", "allocated_bytes", "reserved_bytes",
            "peak_allocated_bytes", "peak_reserved_bytes")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def same(left, right):
    return canonical(left) == canonical(right)


def row_digest(row):
    entries = [{"name": "attention_mask", "dtype": "int64", "shape": [1, 4],
                "values": [[1, 1, 1, 1]]},
               {"name": "input_ids", "dtype": "int64", "shape": [1, 4], "values": [row]}]
    return hashlib.sha256(canonical(entries)).hexdigest()


def audit(result, source):
    require(type(result) is dict and type(source) is dict and result.get("status") == "passed",
            "the public result is not a completed pass")
    require(source.get("status") == "passed" and source.get("gpu_boundary_enabled") is True,
            "source audit did not verify the enabled candidate")
    require(source.get("plan_in_source_manifest") is True
            and type(source.get("source_sha256")) is dict
            and source["source_sha256"].get("docs/GROUPED_GPU_DEV_PLAN.md") == source.get("plan_sha256"),
            "the committed plan is not bound in the source manifest")
    require(result.get("source_commit") == source.get("source_commit")
            and same(result.get("source_sha256"), source.get("source_sha256"))
            and result.get("protocol_sha256") == source.get("protocol_sha256")
            and same(result.get("verified_checkpoint_assets"), source.get("checkpoint_assets")),
            "source, protocol or recorded checkpoint binding differs from the independent source audit")
    require(same(result.get("supervisor"), {"exit_code": 0, "clean": True, "timed_out": False})
            and same(result.get("owner_presence_after_exit"), {"query": "ok", "owner_pid_listed": False}),
            "saved supervisor or point-in-time presence record does not pass")
    measurement = result.get("measurement")
    require(type(measurement) is dict and measurement.get("status") == "passed"
            and measurement.get("sample_status") == "passed", "measurement is incomplete")
    profile = measurement.get("resource_profile")
    require(type(profile) is dict and profile.get("schema") == "grouped-gpu-resource-profile-v1"
            and profile.get("status") == "provisional", "resource profile missing or nonprovisional")
    require(profile.get("source_sha256") == hashlib.sha256(canonical(source["source_sha256"])).hexdigest()
            and profile.get("protocol_sha256") == source["protocol_sha256"]
            and profile.get("runtime_sha256") == measurement.get("runtime_sha256")
            and profile.get("checkpoint_sha256") == CHECKPOINT_SHA256,
            "resource profile source, runtime or model binding differs")

    entries = []
    for prefix, group_prefix in (("serial", "mixed"), ("heldout-serial", "heldout-group")):
        for group in range(2):
            for case in range(2):
                entries.append({"request_id": f"{prefix}-{group}-{case}", "entry_shapes": [[1, 4]]})
            entries.append({"request_id": f"{group_prefix}-{group}", "entry_shapes": [[2, 4]]})
    require(same(measurement.get("model_entries"), entries), "model-entry order or shape differs")

    memory = measurement.get("memory")
    expected_stages = ["before_model_load", "admission", "admission", "after_model_load",
                       "admission", "admission"]
    for entry in entries:
        name = entry["request_id"]
        expected_stages.extend([f"before-{name}", "admission", "admission", f"after-{name}"])
    require(type(memory) is list and len(memory) == len(expected_stages)
            and [item.get("stage") for item in memory] == expected_stages,
            "memory stage count or ordering differs")
    total = memory[0].get("device_total_bytes")
    require(type(total) is int and 7 * GIB <= total <= 9 * GIB, "invalid device total")
    cap = int(0.25 * total)
    reserve = max(GIB, (total * 15 + 99) // 100, GIB // 2)
    require(measurement.get("runtime_redacted", {}).get("total_bytes") == total
            and same(profile.get("device_limits"), {"allocator_fraction": 0.25, "allocator_cap_bytes": cap,
                     "resident_commitment_bytes": 2 * GIB, "workspace_commitment_bytes": GIB,
                     "minimum_reserve_bytes": GIB // 2})
            and same(profile.get("host_ledger"), {"max_bytes": 64 * 1024**2, "max_items": 256,
                     "max_requests": 8, "max_operations": 16})
            and same(profile.get("control_host_ledger"), {"max_bytes": 32 * 1024**2, "max_items": 128,
                     "max_requests": 1, "max_operations": 1})
            and same(profile.get("snapshot_reservation"), {"nbytes": 256, "items": 8})
            and same(profile.get("worker_finite_scratch"), {"nbytes": 4 * 1024**2, "items": 8})
            and same(profile.get("validation_scratch"), {"nbytes": 4 * 1024**2, "items": 8})
            and same(profile.get("comparison_scratch"), {"chunk_elements": 4096, "declared_max_bytes": 1024**2})
            and same(profile.get("trace_limits"), {"device_bytes": 0, "host_bytes": 0}),
            "profile limits or redacted device total differ")
    for item in memory:
        require(type(item) is dict and all(type(item.get(key)) is int and item[key] >= 0
                                           for key in COUNTERS), "invalid saved memory counter")
        free, allocated, reserved, peak_allocated, peak_reserved = (
            item[key] for key in ("device_free_bytes", "allocated_bytes", "reserved_bytes",
                                  "peak_allocated_bytes", "peak_reserved_bytes"))
        require(item["device_total_bytes"] == total and free + reserved <= total
                and allocated <= reserved <= cap and allocated <= peak_allocated <= peak_reserved <= cap
                and reserved <= peak_reserved, "inconsistent saved memory counters")
    require(memory[0]["device_free_bytes"] >= 5 * GIB, "preload free-memory minimum failed")

    admissions = [("resident", 0, 2 * GIB), ("precopy", 2 * GIB, 0),
                  ("workspace", 2 * GIB, GIB), ("owner-verify", 3 * GIB, 0)]
    for entry in entries:
        admissions.extend([(f"forward-{entry['request_id']}", 3 * GIB, 0)] * 2)
    actual = [item for item in memory if item["stage"] == "admission"]
    require(len(actual) == len(admissions), "admission count differs")
    for item, (kind, committed, requested) in zip(actual, admissions):
        estimate = max(0, min(total - item["reserved_bytes"] - reserve - committed,
                              item["device_free_bytes"] - reserve - committed))
        keys = ("reserve_bytes", "committed_incremental_bytes", "requested_incremental_bytes",
                "admissible_estimate_bytes")
        require(all(type(item.get(key)) is int for key in keys)
                and item.get("admission_kind") == kind and item.get("reserve_bytes") == reserve
                and item.get("committed_incremental_bytes") == committed
                and item.get("requested_incremental_bytes") == requested and item.get("admissible_estimate_bytes") == estimate
                and estimate >= requested, "recorded admission arithmetic differs")

    def counters(item):
        return {key: item[key] for key in COUNTERS}

    def headroom(item):
        estimate = min(item["device_free_bytes"] - reserve - 3 * GIB,
                       total - item["reserved_bytes"] - reserve - 3 * GIB)
        require(estimate >= 0, "recorded point cannot cover commitments and reserve")
        return {"reserve_bytes": reserve, "committed_incremental_bytes": 3 * GIB,
                "admissible_estimate_bytes": estimate}

    headroom(memory[0])
    headroom(memory[3])
    stages = profile.get("stages")
    require(type(stages) is list and len(stages) == len(entries), "resource stage list differs")
    seen = set()
    for index, (entry, stage) in enumerate(zip(entries, stages)):
        first = 6 + 4 * index
        before, after = memory[first], memory[first + 3]
        require(before["peak_allocated_bytes"] == before["allocated_bytes"]
                and before["peak_reserved_bytes"] == before["reserved_bytes"],
                "pre-forward peak reset differs from current counters")
        for previous, current in zip(memory[first:first + 3], memory[first + 1:first + 4]):
            require(current["peak_allocated_bytes"] >= previous["peak_allocated_bytes"]
                    and current["peak_reserved_bytes"] >= previous["peak_reserved_bytes"],
                    "peak decreased within a forward interval")
            headroom(current)
            require(max(current["peak_allocated_bytes"] - before["allocated_bytes"],
                        current["peak_reserved_bytes"] - before["reserved_bytes"]) <= GIB,
                    "observed forward increment exceeds the workspace commitment")
        name = entry["request_id"]
        shape = entry["entry_shapes"][0]
        path = "broker" if name.startswith("heldout-group-") else "control"
        key = (path, tuple(shape))
        observation = "repeat" if key in seen else "first-for-path-and-shape"
        seen.add(key)
        expected_increment = {"allocated_increment_bytes": after["peak_allocated_bytes"] - before["allocated_bytes"],
                              "reserved_increment_bytes": after["peak_reserved_bytes"] - before["reserved_bytes"]}
        require(stage.get("request_id") == name and same(stage.get("shape"), shape)
                and stage.get("execution_path") == path and stage.get("observation") == observation
                and same(stage.get("baseline"), counters(before)) and same(stage.get("endpoint"), counters(after))
                and same(stage.get("observed_increment_bytes"), expected_increment)
                and same(stage.get("headroom_before"), headroom(before))
                and same(stage.get("headroom_after"), headroom(after)),
                "resource-profile stage differs from recorded counters")

    require(same(profile.get("model_load"), {"before": counters(memory[0]), "after": counters(memory[3])}),
            "model-load profile differs from memory observations")
    holdout = measurement.get("holdout")
    require(type(holdout) is list and len(holdout) == 2, "broker comparison groups missing")
    jobs = set()
    for group_index, group in enumerate(holdout):
        require(group.get("group_id") == f"holdout-{group_index}" and group.get("status") == "compared"
                and same(group.get("entry_shapes"), [[2, 4]]) and len(group.get("comparisons", [])) == 2
                and len(group.get("reference_separation", [])) == 2
                and all(item.get("status") == "mismatch" for item in group["reference_separation"]),
                "broker cohort or saved reference-separation status differs")
        for case_index, case in enumerate(group["comparisons"]):
            number = 107 + 2 * group_index + case_index
            row = [50256, number, number + 100, number + 200]
            job = case.get("job_slot")
            require(case.get("case_id") == f"holdout-{group_index}-{case_index}"
                    and case.get("owner") == ("team-a" if case_index == 0 else "team-b")
                    and case.get("request_id") == f"heldout-broker-{group_index}"
                    and case.get("input_row_sha256") == row_digest(row)
                    and case.get("status") == "matched"
                    and case.get("own_reference", {}).get("status") == "matched"
                    and case.get("other_reference", {}).get("status") == "mismatch"
                    and type(job) is int and job == group_index * 2 + case_index and job not in jobs,
                    "broker row, route or saved comparison status differs")
            jobs.add(job)
    cleanup = measurement.get("cleanup", {})
    before_cleanup = cleanup.get("before")
    require(type(before_cleanup) is dict and set(before_cleanup) == {"allocated_bytes", "reserved_bytes"}
            and all(type(before_cleanup[key]) is int and 0 <= before_cleanup[key] <= cap
                    for key in before_cleanup)
            and before_cleanup["allocated_bytes"] <= before_cleanup["reserved_bytes"]
            and same(cleanup.get("after"), {"allocated_bytes": 0, "reserved_bytes": 0})
            and type(cleanup.get("accounting_leases_after")) is int and cleanup["accounting_leases_after"] == 0
            and cleanup.get("api") == "PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces",
            "saved cleanup counters, accounting or API label differ")
    return {"status": "passed", "source_commit": source["source_commit"],
            "model_entries": len(entries), "singleton_entries": 8, "two_row_entries": 4,
            "memory_observations": len(memory), "admission_observations": len(actual),
            "minimum_sampled_free_bytes": min(item["device_free_bytes"] for item in memory),
            "maximum_recorded_peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in memory),
            "maximum_recorded_peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in memory),
            "limitations": ["saved-record consistency only; no execution or native-cleanup attestation",
                            "per-forward peaks were reset; memory counters are sampled observations",
                            "raw logits were not retained, so numerical comparisons cannot be independently replayed",
                            "broker rows are held out from this policy sample, with prior/outside exposure unknown",
                            "resource observations do not establish a worst-case or production capacity bound"]}


def check_copied_evidence(result, directory):
    inventory = result["evidence_inventory"]
    names = ["sample-evidence.json", "frozen-policy.json", "resource-profile.json", "protocol-source.json"]
    names += [f"memory-{index:03}.json" for index in range(54)]
    values = {}
    for name in names:
        path = directory / "memory" / name if name.startswith("memory-") else directory / name
        raw = path.read_bytes()
        item = inventory[name]
        require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
                "copied evidence differs from the controller inventory")
        values[name] = json.loads(raw)
    require([values[f"memory-{index:03}.json"] for index in range(54)] == result["measurement"]["memory"],
            "memory files differ from the projected observations")
    require(values["resource-profile.json"] == result["measurement"]["resource_profile"],
            "resource profile file differs from the projected profile")
    protocol = values["protocol-source.json"]
    require(protocol.get("sha256") == result["protocol_sha256"]
            and hashlib.sha256(protocol["protocol_utf8"].encode("utf-8")).hexdigest() == result["protocol_sha256"],
            "retained protocol bytes differ from the frozen digest")
    sample = values["sample-evidence.json"]
    policy = values["frozen-policy.json"]
    unsigned = {key: value for key, value in sample.items() if key != "evidence_sha256"}
    require(sample.get("status") == "passed" and sample.get("complete") is True
            and sample.get("error") is None and sample.get("calls") == {"serial": 4, "mixed": 2}
            and sample.get("tested_rows") == [2]
            and sample.get("evidence_sha256") == hashlib.sha256(canonical(unsigned)).hexdigest()
            and sample.get("evidence_sha256") == result["measurement"]["sample_evidence_sha256"]
            and sample.get("protocol_sha256") == policy.get("protocol_sha256")
            and len(sample.get("groups", [])) == 2, "sample evidence identity or completion differs")
    for group_index, group in enumerate(sample["groups"]):
        require(group.get("group_id") == f"policy-{group_index}" and len(group.get("cases", [])) == 2,
                "sample group differs")
        for case_index, case in enumerate(group["cases"]):
            number = 103 + 2 * group_index + case_index
            row = [50256, number, number + 100, number + 200]
            row_hash = row_digest(row)
            require(case.get("case_id") == f"policy-{group_index}-{case_index}"
                    and case.get("owner") == ("team-a" if case_index == 0 else "team-b")
                    and case.get("input_sha256") == row_hash and case.get("row_sha256") == [row_hash]
                    and case.get("status") == "matched", "sample row or reported comparison differs")
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closed_result")
    parser.add_argument("source_audit")
    args = parser.parse_args()
    projection_path = Path(args.closed_result)
    projection = json.loads(projection_path.read_bytes())
    require(projection.get("schema") == "grouped-gpu-public-projection-v1"
            and re.fullmatch(r"[0-9a-f]{64}", projection.get("original_result_sha256", "")) is not None,
            "public projection is missing its withheld-original commitment")
    result = projection["result"]
    source = json.loads(Path(args.source_audit).read_bytes())
    report = audit(result, source)
    report["copied_evidence_files_verified"] = check_copied_evidence(result, projection_path.parent)
    operation = json.loads((projection_path.parent / "operational-review.json").read_bytes())
    require(operation.get("result_sha256") == projection["original_result_sha256"],
            "operational report and projection refer to different original results")
    report["original_result_sha256_commitment"] = projection["original_result_sha256"]
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
