"""Opt-in CPU integration test for a single verified pretrained model."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch


CACHE_ENV = "LOOKING_GLASS_TEST_CACHE"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
WEIGHT_SHA256 = "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"
SEED = 1729
ABSOLUTE_TOLERANCE = 1e-6

_CLIENT = r'''
import json
import os
import sys
from looking_glass.ipc import IPCClient

spec = json.loads(sys.stdin.readline())
client = IPCClient(tuple(spec["address"]), spec["token"], io_timeout=30)
handle = client.submit(team_id=spec["team_id"], request_id="same-request",
                       inputs={"input_ids": [[spec["token_id"]]]}, seed=1729)
print(json.dumps({"pid": os.getpid(), "team_id": spec["team_id"],
                  "broker_id": handle.broker_id, "job_id": handle.job_id}), flush=True)
outcome = handle.wait(timeout=60)
print(json.dumps(outcome), flush=True)
'''


@unittest.skipUnless(os.environ.get(CACHE_ENV),
                     "set LOOKING_GLASS_TEST_CACHE to opt in with a pinned local cache")
class PretrainedServiceSmokeTests(unittest.TestCase):
    def test_one_resident_checkpoint_serves_two_isolated_processes(self):
        import huggingface_hub
        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM
        import looking_glass.pretrained as pretrained
        from looking_glass.broker import Broker
        from looking_glass.ipc import IPCClient, IPCRejected, IPCServer
        from looking_glass.resident_worker import ResidentWorker

        entered = threading.Event()
        release = threading.Event()

        class GatedModel(nn.Module):
            """Hold the first forward so the second client can be admitted."""
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, **inputs):
                entered.set()
                if not release.wait(60):
                    raise TimeoutError("test gate was not released")
                return self.model(**inputs)

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        children = []
        broker = None
        server = None
        pool = None
        summary = None
        loader_patch = patch.object(pretrained, "load_distilgpt2", wraps=pretrained.load_distilgpt2)
        load = loader_patch.start()
        deserialize_patch = patch.object(AutoModelForCausalLM, "from_pretrained",
                                         wraps=AutoModelForCausalLM.from_pretrained)
        deserialize = deserialize_patch.start()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)
                with patch.object(huggingface_hub, "hf_hub_download", wraps=huggingface_hub.hf_hub_download) as hub:
                    bundle = pretrained.load_distilgpt2(cache_dir=os.environ[CACHE_ENV], allow_download=False)
                    self.assertEqual(load.call_count, 1)
                    self.assertEqual(deserialize.call_count, 1)
                    self.assertEqual(hub.call_count, 7)
                    self.assertTrue(all(call.kwargs["local_files_only"] is True
                                        for call in hub.call_args_list))

                self.assertEqual(bundle.metadata["model_revision"], MODEL_REVISION)
                self.assertEqual(bundle.metadata["tokenizer_revision"], MODEL_REVISION)
                self.assertEqual(bundle.metadata["weights_sha256"], WEIGHT_SHA256)
                self.assertEqual(type(bundle.model).__name__, "GPT2LMHeadModel")
                self.assertFalse(bundle.model.training)
                self.assertFalse(bundle.model.config.use_cache)
                self.assertEqual(bundle.model.config._attn_implementation, "eager")
                self.assertTrue(all(p.device.type == "cpu" for p in bundle.model.parameters()))

                # These are public, fixed GPT-2 token IDs, not user prompts.
                tokens = {"team-a": 262, "team-b": 257}
                expected_digests = {
                    "team-a": "44c56ff28f8f656b5baac23de670f27223141ab1df6afac93fe197b9cab44445",
                    "team-b": "8ca086bc587bcdca8619f3fc634f4c9d33446366115885bb41161a9da4e36f15",
                }
                for team, token in tokens.items():
                    encoded = json.dumps({"input_ids": [[token]]}, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")
                    self.assertEqual(hashlib.sha256(encoded).hexdigest(), expected_digests[team])
                with torch.inference_mode():
                    references = {
                        team: bundle.model(input_ids=torch.tensor([[token]], dtype=torch.long)).logits.clone()
                        for team, token in tokens.items()
                    }
                self.assertGreater(float((references["team-a"] - references["team-b"]).abs().max()), 1e-3)

                gated = GatedModel(bundle.model).eval()
                self.assertIs(gated.model, bundle.model)
                worker = ResidentWorker("distilgpt2-pinned", gated, device="cpu", max_batch_size=1,
                                        max_sequence_length=1, max_input_bytes=8,
                                        max_output_bytes=50257 * 4)
                broker = Broker(worker, max_pending=4, max_pending_per_team=2)
                server = IPCServer(broker, team_ids=list(tokens), max_batch=1, max_tokens=1,
                                   max_response_bytes=2_000_000, io_timeout=30, job_timeout=60).start()
                self.assertTrue(server.team_tokens["team-a"] != server.team_tokens["team-b"],
                                "team credentials must differ")

                # A team's token cannot submit as the other team.
                with self.assertRaises(IPCRejected) as rejection:
                    IPCClient(server.address, server.team_tokens["team-a"], io_timeout=30).submit(
                        team_id="team-b", request_id="unauthorized", inputs={"input_ids": [[262]]})
                self.assertEqual(rejection.exception.code, "unauthorized")

                for team, token in tokens.items():
                    child = subprocess.Popen([sys.executable, "-c", _CLIENT], stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    children.append(child)
                    child.stdin.write(json.dumps({"address": server.address, "team_id": team,
                                                   "token": server.team_tokens[team], "token_id": token}) + "\n")
                    child.stdin.close()
                    child.stdin = None
                pool = ThreadPoolExecutor(max_workers=2)
                lines = [pool.submit(child.stdout.readline) for child in children]
                accepted = [json.loads(line.result(timeout=30)) for line in lines]
                self.assertTrue(entered.wait(30), "no model forward entered")
                self.assertEqual({a["team_id"] for a in accepted}, set(tokens))
                self.assertEqual({a["broker_id"] for a in accepted}, {broker.broker_id})
                self.assertEqual(len({a["job_id"] for a in accepted}), 2)
                self.assertEqual(len({a["pid"] for a in accepted}), 2)
                self.assertNotIn(os.getpid(), {a["pid"] for a in accepted})
                release.set()
                completed = [pool.submit(child.communicate, timeout=60) for child in children]
                outputs = [future.result(timeout=65) for future in completed]

                max_errors = {}
                for child, ack, (stdout, stderr) in zip(children, accepted, outputs):
                    self.assertEqual(child.returncode, 0, stderr)
                    result = json.loads(stdout)
                    self.assertEqual(result["status"], "succeeded", result.get("error"))
                    self.assertEqual(result["team_id"], ack["team_id"])
                    self.assertEqual(result["request_id"], "same-request")
                    self.assertEqual(result["broker_id"], ack["broker_id"])
                    self.assertEqual(result["job_id"], ack["job_id"])
                    self.assertEqual(result["result"]["shape"], [1, 1, 50257])
                    self.assertEqual(result["result"]["dtype"], "torch.float32")
                    actual = torch.tensor(result["result"]["logits"], dtype=torch.float32).reshape(1, 1, 50257)
                    error = float((actual - references[ack["team_id"]]).abs().max())
                    self.assertLessEqual(error, ABSOLUTE_TOLERANCE, f"max absolute error: {error}")
                    max_errors[ack["team_id"]] = error
                self.assertEqual(load.call_count, 1)
                self.assertEqual(deserialize.call_count, 1)
                summary = {
                    "broker_id": broker.broker_id,
                    "jobs": {ack["team_id"]: ack["job_id"] for ack in accepted},
                    "max_absolute_error": max_errors,
                    "shape": [1, 1, 50257],
                    "model_deserializations": deserialize.call_count,
                }
        finally:
            release.set()
            for child in children:
                if child.poll() is None:
                    child.kill()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            for child in children:
                child.wait(timeout=10)
                for stream in (child.stdin, child.stdout, child.stderr):
                    if stream is not None:
                        stream.close()
            try:
                try:
                    if server is not None:
                        self.assertTrue(server.close(timeout=10), "IPC server did not shut down")
                finally:
                    if broker is not None:
                        self.assertTrue(broker.close(timeout=10), "broker did not shut down")
            finally:
                deserialize_patch.stop()
                loader_patch.stop()
                torch.set_num_threads(old_threads)
        if os.environ.get("LOOKING_GLASS_SERVICE_SUMMARY") == "1":
            print("PRETRAINED_SERVICE_SMOKE_V1 " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
