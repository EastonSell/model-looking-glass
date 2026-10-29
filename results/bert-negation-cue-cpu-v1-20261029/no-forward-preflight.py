"""Check a pinned negation owner and checkpoint without model forwards or GPU."""
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
                  HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
def blocked(*_args, **_kwargs):
    raise RuntimeError("CUDA, network and model forwards are disabled in this preflight")
socket.socket.connect = blocked
socket.create_connection = blocked
spec = importlib.util.spec_from_file_location("owner", repo / "scripts/bert_negation_cue_v1.py")
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
source = owner.source_identity(repo, commit, Path(git_exe))
assets = owner.verify_assets(snapshot)
versions = {name: importlib.metadata.version(name) for name in owner.VERSIONS}
assert versions == owner.VERSIONS
manifest = json.loads((repo / "experiments/bert_negation_public_cases_v1.json").read_text(encoding="utf-8"))
tokenization = owner.verify_tokenizer(snapshot, manifest)
import torch
torch.cuda._lazy_init = blocked
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
owner.enable_no_flush(torch)
torch.manual_seed(owner.SEED)
from transformers import DistilBertForMaskedLM
DistilBertForMaskedLM.forward = blocked
model = DistilBertForMaskedLM.from_pretrained(str(snapshot), local_files_only=True,
                                               attn_implementation="eager").eval()
assert model.config.dim == 768 and model.config.n_layers == 6 and model.config.vocab_size == 30522
assert model.config._attn_implementation == "eager"
assert all(type(layer.attention).__name__ == "MultiHeadSelfAttention" for layer in model.distilbert.transformer.layer)
assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in model.parameters())
assert not torch.cuda.is_initialized()
state = owner.state_hash(model)
owner.verify_no_flush(torch)
owner.write_json_new(report, {"status": "pass", "source_commit": commit, "source_sha256": source,
                             "assets": assets, "versions": versions, "tokenizer": tokenization,
                             "model_state_sha256": state, "model_class": type(model).__name__,
                             "model_forward_blocked": True, "selected_network_calls_blocked": True,
                             "cuda_initialization_blocked": True, "cuda_initialized": False})
print(json.dumps({"status": "pass", "tokenized_rows": tokenization["rows"],
                  "cuda_initialized": False, "model_forward_blocked": True}))
