"""Check fixed source, assets and tokenizer without importing Torch or forwarding."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

repo, snapshot, commit, git_exe, report = sys.argv[1:]
repo, snapshot, report = map(Path, (repo, snapshot, report))
os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1")


def blocked(*_args, **_kwargs):
    raise RuntimeError("network is disabled in this preflight")


socket.socket.connect = blocked
socket.create_connection = blocked
spec = importlib.util.spec_from_file_location("factorial_owner", repo / "scripts/bert_cue_factorial_v1.py")
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
prior = owner.prior_owner(repo)
source = prior.source_identity(repo, commit, Path(git_exe))
if source[owner.MANIFEST] != owner.MANIFEST_HASH:
    raise RuntimeError("manifest hash differs")
assets = prior.verify_assets(snapshot)
versions = {name: importlib.metadata.version(name) for name in prior.VERSIONS}
if versions != prior.VERSIONS:
    raise RuntimeError("dependency versions differ")
manifest = json.loads((repo / owner.MANIFEST).read_text(encoding="utf-8"))
tokenization = owner.verify_tokenizer(snapshot, manifest)
if "torch" in sys.modules or "transformers" in sys.modules:
    raise RuntimeError("preflight imported a model runtime")
prior.write_json_new(report, {"status": "pass", "source_commit": commit, "source_sha256": source,
                              "assets": assets, "versions": versions, "tokenizer": tokenization,
                              "torch_imported": False, "transformers_imported": False,
                              "model_loaded": False, "model_forward_run": False,
                              "selected_network_calls_blocked": True})
print(json.dumps({"status": "pass", "source_paths": len(source), "tokenized_rows": tokenization["rows"],
                  "torch_imported": False, "model_forward_run": False}))
