"""Model-state identity for the proposed CPU synthetic-v2 protocol.

The digest binds the supplied state bytes, not the execution environment or
the authenticity of a model. The caller must independently bind the expected
digest, source revision, runtime and CPU execution policy.
"""

import hashlib
import json

import torch


def state_sha256(model):
    """Hash sorted named CPU state tensors using the v2 length-prefixed format."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if (not isinstance(name, str) or not isinstance(value, torch.Tensor)
                or value.device.type != "cpu" or value.layout != torch.strided):
            raise ValueError("model state must contain named strided CPU tensors")
        header = (json.dumps({"name": name, "shape": list(value.shape),
                              "dtype": str(value.dtype)},
                             sort_keys=True, indent=2, allow_nan=False)
                  .encode("utf-8") + b"\n")
        data = bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()
