"""Check the sealed observer's supplied inventory; no model is loaded."""
from pathlib import Path
import hashlib
import json

base = Path("E:/Codex/model-looking-glass/work")
attempt = base / "bert-agreement-suffix-cpu-v1-20261029-01"
controller = base / "bert-agreement-suffix-controller-v1-20261029-01"
outer = base / "bert-agreement-suffix-outer-v1-20261029-01"
commit = "7dbf6e9ea498770e389cbf7fbc1eba5bc2303662"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_bytes())

outer_file = outer / "observer-exit.json"
assert digest(outer_file) == "5f68f07184d6d77b333fd8f42e771a9f720a746fab87994d97209951f4fbc815"
obs = load(outer_file)
outer_start = load(outer / "observer-spawn.json")
inner_start = load(controller / "spawn.json")
inner = load(controller / "exit-observation.json")
assert obs["schema"] == "bert-suffix-outer-observer-exit-v1"
assert type(obs["exit_code"]) is int and obs["exit_code"] == 0 and obs["interrupted"] is None
assert obs["source_commit"] == outer_start["source_commit"] == inner_start["source_commit"] == inner["source_commit"] == commit
assert type(obs["observer_pid"]) is int and obs["observer_pid"] == outer_start["pid"]
assert type(inner["child_pid"]) is int and inner["child_pid"] == inner_start["pid"]
assert obs["start_unix"] == outer_start["start_unix"] <= inner_start["start_unix"] == inner["start_unix"] <= inner["end_unix"] <= obs["end_unix"]
assert type(inner["exit_code"]) is int and inner["exit_code"] == 0 and inner["status"] == "complete"
assert obs["pending_marker_present"] is False and not (controller / "observer-pending.json").exists()
assert obs["receipt_sha256"] == digest(controller / "exit-observation.json")
for directory, name in ((attempt, "attempt_inventory"), (controller, "controller_inventory")):
    rows = obs[name]
    assert len({row["path"] for row in rows}) == len(rows)
    current = {p.name for p in directory.iterdir() if p.is_file()}
    assert current == {row["path"] for row in rows}
    for row in rows:
        p = directory / row["path"]
        assert not p.is_symlink() and p.is_file()
        assert type(row["bytes"]) is int and p.stat().st_size == row["bytes"] and digest(p) == row["sha256"]
    print(name, "verified_files", len(rows), "bytes", sum(r["bytes"] for r in rows))
print("outer_and_direct_observed_exits", obs["exit_code"], inner["exit_code"])
