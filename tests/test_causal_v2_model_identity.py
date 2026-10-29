import hashlib
import json
import unittest

import torch

from looking_glass.causal_experiment import _state_sha256
from looking_glass.causal_v2_model_identity import state_sha256
from looking_glass.model import TinyResidual


class ModelIdentityTests(unittest.TestCase):
    def test_tiny_model_matches_existing_protocol_hash_and_detects_change(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(41977)
            model = TinyResidual(width=8, depth=4).eval()
        before = state_sha256(model)
        self.assertEqual(before, _state_sha256(model))
        with torch.no_grad():
            next(model.parameters()).reshape(-1)[0] += 0.125
        self.assertNotEqual(before, state_sha256(model))

    def test_length_prefixed_header_and_scalar_bytes(self):
        model = torch.nn.Module()
        model.register_buffer("z", torch.tensor(1.0, dtype=torch.float32))
        model.register_buffer("a", torch.tensor([1, -2], dtype=torch.int16))
        expected = hashlib.sha256()
        for name, tensor in (("a", model.a), ("z", model.z)):
            header = (json.dumps({"name": name, "shape": list(tensor.shape),
                                  "dtype": str(tensor.dtype)},
                                 sort_keys=True, indent=2, allow_nan=False)
                      .encode("utf-8") + b"\n")
            data = tensor.reshape(-1).numpy().tobytes()
            expected.update(len(header).to_bytes(8, "big") + header)
            expected.update(len(data).to_bytes(8, "big") + data)
        self.assertEqual(state_sha256(model), expected.hexdigest())

    def test_rejects_non_cpu_state_without_allocating_on_gpu(self):
        model = torch.nn.Module()
        model.register_buffer("not_cpu", torch.empty(1, device="meta"))
        with self.assertRaisesRegex(ValueError, "CPU"):
            state_sha256(model)


if __name__ == "__main__":
    unittest.main()
