"""Opt-in CPU smoke of the real pretrained shared-model command-line owner."""

import gc
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch


CACHE_ENV = "LOOKING_GLASS_TEST_CACHE"
MODEL_ID = "distilbert/distilgpt2@2290a62682d06624634c1f46a6ad5be0f47f38aa"
WEIGHT_SHA256 = "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"
SEED = 1729
ABSOLUTE_TOLERANCE = 1e-6

_CLIENT = r'''
import json
import os
import sys
from looking_glass.ipc import IPCClient

spec = json.load(sys.stdin)
client = IPCClient(tuple(spec["address"]), spec["token"], io_timeout=30)
handle = client.submit(team_id=spec["team_id"], request_id="same-request",
                       inputs={"input_ids": [[spec["token_id"]]]}, seed=1729)
outcome = handle.wait(timeout=60)
print(json.dumps({"pid": os.getpid(), "outcome": outcome}), flush=True)
'''


@unittest.skipUnless(os.environ.get(CACHE_ENV),
                     "set LOOKING_GLASS_TEST_CACHE to opt in with a pinned local cache")
class PretrainedCliSmokeTests(unittest.TestCase):
    def test_two_process_clients_match_serial_reference_and_shutdown_cleanly(self):
        import torch
        import looking_glass
        import looking_glass.pretrained as pretrained
        from looking_glass.ipc import IPCClient, IPCRejected

        tokens = {"team-a": 262, "team-b": 257}
        expected_digests = {
            "team-a": "44c56ff28f8f656b5baac23de670f27223141ab1df6afac93fe197b9cab44445",
            "team-b": "8ca086bc587bcdca8619f3fc634f4c9d33446366115885bb41161a9da4e36f15",
        }
        for team, token in tokens.items():
            payload = json.dumps({"input_ids": [[token]]}, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_digests[team])

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        service = None
        reader = None
        summary = None
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)
                # The reference model is released before starting the owner.
                with patch.object(pretrained, "load_distilgpt2", wraps=pretrained.load_distilgpt2) as load:
                    bundle = pretrained.load_distilgpt2(cache_dir=os.environ[CACHE_ENV], allow_download=False)
                    self.assertEqual(load.call_count, 1)
                    reference_loads = load.call_count
                self.assertEqual(bundle.metadata["weights_sha256"], WEIGHT_SHA256)
                self.assertEqual(bundle.metadata["model_revision"], MODEL_ID.split("@", 1)[1])
                self.assertEqual(bundle.metadata["model_id"], MODEL_ID.split("@", 1)[0])
                self.assertEqual(bundle.metadata["tokenizer_id"], bundle.metadata["model_id"])
                self.assertEqual(bundle.metadata["tokenizer_revision"], bundle.metadata["model_revision"])
                self.assertEqual(bundle.metadata["asset_sha256"]["tokenizer.json"],
                                 "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6")
                self.assertFalse(bundle.model.training)
                self.assertFalse(bundle.model.config.use_cache)
                self.assertEqual(bundle.model.config._attn_implementation, "eager")
                self.assertTrue(all(p.device.type == "cpu" for p in bundle.model.parameters()))
                with torch.inference_mode():
                    references = {
                        team: bundle.model(input_ids=torch.tensor([[token]], dtype=torch.long,
                                                                  device="cpu")).logits.clone()
                        for team, token in tokens.items()
                    }
                self.assertGreater(float((references["team-a"] - references["team-b"]).abs().max()), 1e-3)
                reference_model = weakref.ref(bundle.model)
                del bundle
                gc.collect()
                self.assertIsNone(reference_model(), "serial reference model remained live")

                env = os.environ.copy()
                env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                            "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1",
                            "HF_HOME": str(Path(os.environ[CACHE_ENV]).resolve().parent)})
                repo = Path(looking_glass.__file__).resolve().parent.parent
                command = [sys.executable, "-m", "looking_glass.cpu_service", "--cache-dir",
                           os.environ[CACHE_ENV], "--team", "team-a", "--team", "team-b",
                           "--max-tokens", "1"]
                service = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                startup = queue.Queue(maxsize=1)
                reader = threading.Thread(target=lambda: startup.put(service.stdout.readline()), daemon=True)
                reader.start()
                try:
                    endpoint = json.loads(startup.get(timeout=60))
                except (queue.Empty, ValueError):
                    self.fail(f"CPU owner did not provide a valid private startup frame; "
                              f"exit code {service.poll()}")
                self.assertTrue(endpoint.get("device") == "cpu", "owner device mismatch")
                self.assertTrue(endpoint.get("model_id") == MODEL_ID, "owner model identity mismatch")
                self.assertTrue(set(endpoint.get("team_tokens", {})) == set(tokens), "owner teams mismatch")
                credentials = endpoint["team_tokens"]
                self.assertTrue(all(isinstance(credentials[t], str) and credentials[t] for t in tokens),
                                "missing team credentials")
                self.assertTrue(credentials["team-a"] != credentials["team-b"],
                                "team credentials must differ")
                address = tuple(endpoint["address"])
                with self.assertRaises(IPCRejected) as rejected:
                    IPCClient(address, credentials["team-a"], io_timeout=30).submit(
                        team_id="team-b", request_id="unauthorized", inputs={"input_ids": [[262]]})
                self.assertEqual(rejected.exception.code, "unauthorized")

                ready = threading.Barrier(2)

                def client_run(team, token):
                    ready.wait(timeout=30)
                    spec = {"address": address, "team_id": team, "token": credentials[team], "token_id": token}
                    return subprocess.run([sys.executable, "-c", _CLIENT], cwd=repo, env=env,
                                          input=json.dumps(spec), capture_output=True, text=True,
                                          timeout=90, check=False)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {team: pool.submit(client_run, team, token) for team, token in tokens.items()}
                    replies = {team: future.result(timeout=95) for team, future in futures.items()}
                parsed = {}
                for team, reply in replies.items():
                    self.assertEqual(reply.returncode, 0, "client process failed")
                    try:
                        parsed[team] = json.loads(reply.stdout)
                    except ValueError:
                        self.fail("client process returned invalid JSON")
                pids = {parsed[t]["pid"] for t in tokens}
                self.assertEqual(len(pids), 2)
                self.assertNotIn(os.getpid(), pids)
                self.assertNotIn(service.pid, pids)
                errors, jobs = {}, {}
                for team in tokens:
                    result = parsed[team]["outcome"]
                    self.assertTrue(result["status"] == "succeeded", "client request did not succeed")
                    self.assertTrue(result["team_id"] == team, "team identity mismatch")
                    self.assertTrue(result["request_id"] == "same-request", "request identity mismatch")
                    self.assertTrue(result["broker_id"] == endpoint["broker_id"], "broker identity mismatch")
                    jobs[team] = result["job_id"]
                    self.assertTrue(result["result"]["shape"] == [1, 1, 50257], "unexpected result shape")
                    self.assertTrue(result["result"]["dtype"] == "torch.float32", "unexpected result dtype")
                    actual = torch.tensor(result["result"]["logits"], dtype=torch.float32,
                                          device="cpu").reshape(1, 1, 50257)
                    errors[team] = float((actual - references[team]).abs().max())
                    self.assertLessEqual(errors[team], ABSOLUTE_TOLERANCE, "serial reference mismatch")
                self.assertNotEqual(jobs["team-a"], jobs["team-b"])

                service.stdin.write("shutdown\n")
                service.stdin.flush()
                service.stdin.close()
                service.stdin = None
                stdout, _stderr = service.communicate(timeout=30)
                self.assertEqual(service.returncode, 0, "CPU owner did not shut down cleanly")
                self.assertTrue(not stdout.strip(), "CPU owner emitted an unexpected second frame")
                summary = {"broker_id": endpoint["broker_id"], "jobs": jobs,
                           "max_absolute_error": errors, "shape": [1, 1, 50257],
                           "owner_processes": 1, "serial_reference_loads": reference_loads}
        finally:
            if service is not None:
                if service.poll() is None:
                    service.kill()
                service.wait(timeout=10)
                for stream in (service.stdin, service.stdout, service.stderr):
                    if stream is not None:
                        stream.close()
            if reader is not None:
                reader.join(timeout=10)
            torch.set_num_threads(previous_threads)
        if os.environ.get("LOOKING_GLASS_CLI_SUMMARY") == "1":
            print("PRETRAINED_CLI_SMOKE_V1 " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
