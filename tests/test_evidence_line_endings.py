"""Saved evidence must survive checkouts that otherwise convert LF to CRLF."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    "results/targeted-edit-cpu-v1-20261028/frozen.json",
    "results/targeted-edit-cpu-v1-20261028/reference/terminal.json",
    "results/targeted-edit-cpu-v1-20261028/reference/precommit.json",
    "results/targeted-edit-cpu-v1-20261028/reference/observations.jsonl",
    "results/distilgpt2-batch-cpu-v1-20261028/frozen.json",
    "results/distilgpt2-batch-cpu-v1-20261028/run/summary.json",
    "results/pretrained-trace-cpu-v1-20261028/reference/result.json",
    "results/pretrained-trace-cpu-v1-20261028/replay/identity.json",
    "results/verified-cpu-v1/raw_timing.csv",
    "results/bert-agreement-dev-v0-20261028/probe.py",
    "results/distilgpt2-agreement-preflight-v1-20261028/stdout.txt",
)


def git(*args):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    return subprocess.run(["git", *args], cwd=ROOT, env=env, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def test_saved_evidence_bytes_survive_autocrlf_checkout(tmp_path):
    try:
        git("rev-parse", "--git-dir")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("requires a Git checkout containing the saved evidence")
    prefix = str(tmp_path.resolve()) + os.sep
    git("-c", "core.autocrlf=true", "checkout-index", "--prefix=" + prefix, "--", *PATHS)
    for name in PATHS:
        assert (tmp_path / name).read_bytes() == git("show", ":" + name), name

    # Check two real saved-hash links, not just line-ending characteristics.
    reference = tmp_path / "results/targeted-edit-cpu-v1-20261028/reference"
    terminal = json.loads((reference / "terminal.json").read_bytes())
    for name in ("precommit.json", "observations.jsonl"):
        assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == terminal["artifacts"][name]
    trace = tmp_path / "results/pretrained-trace-cpu-v1-20261028"
    identity = json.loads((trace / "replay/identity.json").read_bytes())
    assert hashlib.sha256((trace / "reference/result.json").read_bytes()).hexdigest() == identity["reference_sha256"]
