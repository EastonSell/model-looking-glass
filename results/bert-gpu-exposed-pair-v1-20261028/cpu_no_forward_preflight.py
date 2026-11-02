"""Offline CPU-only loader preflight; no checkpoint forward or GPU use."""
import json
import socket
import sys
from pathlib import Path


def blocked(*args, **kwargs):
    raise AssertionError("guarded network, CUDA initialization, or model forward")


socket.socket.connect = blocked
socket.create_connection = blocked
import torch
torch.cuda._lazy_init = blocked
torch.set_num_threads(1)
import transformers
transformers.DistilBertForMaskedLM.forward = blocked
repo, cache = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo))
from looking_glass import bert_gpu_exposed_pair as pair
assert not torch.cuda.is_initialized()
model, metadata = pair._load_model(cache, torch)
assert not torch.cuda.is_initialized()
assert all(p.device.type == "cpu" for p in model.parameters())
print(json.dumps({"model_class": type(model).__name__, "model_id": metadata["model"],
                  "revision": metadata["revision"], "tokenizer_class": metadata["tokenizer_class"],
                  "target_row": pair.TARGET_ROW_ID, "donor_row": pair.DONOR_ROW_ID,
                  "parameter_count": metadata["parameter_count"],
                  "transformers_version": metadata["transformers_version"],
                  "tokenizers_version": metadata["tokenizers_version"],
                  "safetensors_version": metadata["safetensors_version"],
                  "cuda_initialized": torch.cuda.is_initialized(), "forward_blocked": True,
                  "selected_network_calls_blocked": True}, sort_keys=True))
