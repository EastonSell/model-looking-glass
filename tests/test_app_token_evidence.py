"""Replay real CPU capture provenance through the production UI readout code.

Node supplies a minimal DOM/HTTP harness, not a browser or WebGL renderer.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from looking_glass.app_runtime import LocalModelRuntime
from looking_glass.app_storage import TraceStore


def test_recorded_token_evidence_uses_real_capture_identity_after_reopen(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the production JavaScript DOM contract")
    runtime = LocalModelRuntime()
    try:
        runtime.load_demo()
        store = TraceStore(tmp_path / "exports")
        report = store.publish(runtime.run("Hi", {"max_new_tokens": 2}))
        assert len(report["steps"]) == 2
        saved = store.save("token-evidence")
    finally:
        runtime.unload()
    reopened = TraceStore(tmp_path / "exports")
    reopened.open(saved["path"])
    observations = []
    for source in (store, reopened):
        for step in report["steps"]:
            event = next(row for row in source.events()["events"]
                         if row["metadata"]["app_forward_index"] == step["app_forward_index"])
            coordinate = source.tensor(event["index"], event["vector_count"] - 1)["coordinate"]
            assert coordinate["status"] == "known"
            assert coordinate["token_position"] == len(step["input_token_ids"]) - 1
            assert event["metadata"]["identity"]["run_id"] == step["run_id"]
            assert "run_id" not in event["metadata"]
            observations.append({"metadata": event["metadata"], "coordinate": coordinate})
    fixture = tmp_path / "capture-evidence.json"
    fixture.write_text(json.dumps({"family": report["model"]["family"],
                                  "steps": report["steps"], "observations": observations}))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / "tests/app_ui_contract.cjs")], cwd=root,
                            env={**os.environ, "MLG_CAPTURE_EVIDENCE": str(fixture)},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
