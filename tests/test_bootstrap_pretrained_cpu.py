"""Opt-in real-checkpoint CPU preflight for the fixed bootstrap-v1 case."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest


CACHE_ENV = "LOOKING_GLASS_TEST_CACHE"
SUMMARY_ENV = "LOOKING_GLASS_BOOTSTRAP_CPU_SUMMARY"


def _child(phase):
    """Run in a direct base interpreter; only emit a small aggregate summary."""
    import hashlib
    import json
    import shutil
    import socket
    import struct
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch

    phase[0] = "startup"
    repo = Path(os.environ["LOOKING_GLASS_BOOTSTRAP_REPO"]).resolve(strict=True)
    site = Path(os.environ["LOOKING_GLASS_BOOTSTRAP_SITE"]).resolve(strict=True)
    cache = Path(os.environ[CACHE_ENV]).resolve(strict=True)
    private = Path(os.environ["LOOKING_GLASS_BOOTSTRAP_PRIVATE"]).resolve(strict=True)
    pycache = private / "pycache"
    base_python = Path(os.environ["LOOKING_GLASS_BOOTSTRAP_BASE"]).resolve(strict=True)
    assert Path(sys.executable).resolve(strict=True) == base_python
    assert sys.flags.no_site == 1 and sys.flags.no_user_site == 1 and sys.flags.optimize == 0
    assert sys.dont_write_bytecode and Path(sys.pycache_prefix).resolve(strict=True) == pycache
    assert pycache.is_dir() and not any(pycache.iterdir())

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], timeout=30)

    phase[0] = "source_initial"
    assert not git("status", "--porcelain=v1", "--untracked-files=all")
    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    tree = git("show", "-s", "--format=%T", "HEAD").decode("ascii").strip()

    # The torch import itself precedes the guards. Hide CUDA from interpreter
    # startup, check no context is initialized afterward, and guard the loader
    # and forward from initialization, driver queries, and seeding helpers.
    phase[0] = "torch_import"
    import torch
    assert not torch.cuda._initialized
    calls = []

    def denied(name):
        def fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError("blocked CUDA or network entry point")
        return fail

    cuda_names = ("init", "_lazy_init", "device_count", "is_available",
                  "get_device_properties", "current_device", "set_device",
                  "synchronize", "manual_seed_all", "_is_in_bad_fork")
    with ExitStack() as stack:
        for name in cuda_names:
            stack.enter_context(patch.object(torch.cuda, name, denied("cuda." + name)))
        for name in ("_cuda_init", "_cuda_getDeviceCount"):
            if hasattr(torch._C, name):
                stack.enter_context(patch.object(torch._C, name, denied("torch._C." + name)))
        for name in ("connect", "connect_ex"):
            stack.enter_context(patch.object(socket.socket, name, denied("socket." + name)))
        for name in ("create_connection", "getaddrinfo"):
            stack.enter_context(patch.object(socket, name, denied("socket." + name)))

        phase[0] = "imports"
        import huggingface_hub
        import transformers
        from looking_glass import ipc, pretrained
        from looking_glass.resident_worker import ForwardResult

        phase[0] = "assets"
        expected_assets = {
            "config.json": (762, "4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a"),
            "generation_config.json": (124, "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805"),
            "tokenizer.json": (1355256, "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"),
            "tokenizer_config.json": (26, "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f"),
            "vocab.json": (1042301, "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
            "merges.txt": (456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
            "model.safetensors": (352824413, "e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b"),
        }
        model_id = "distilbert/distilgpt2"
        revision = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
        assert pretrained.MODEL_ID == model_id and pretrained.REVISION == revision
        assert pretrained.ASSETS == expected_assets
        paths = {}
        for name, (size, digest) in expected_assets.items():
            path = Path(huggingface_hub.hf_hub_download(
                repo_id=model_id, revision=revision, filename=name,
                cache_dir=str(cache), local_files_only=True, token=False)).resolve(strict=True)
            assert path.is_relative_to(cache) and path.is_file() and path.stat().st_size == size
            actual = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    actual.update(block)
            assert actual.hexdigest() == digest
            paths[name] = path

        # Real Hub cache lookup, with only config available; the missing second
        # asset must fail before Transformers deserializes a model.
        phase[0] = "negative_cache_checks"
        partial = private / "partial-cache"
        snapshot = partial / "models--distilbert--distilgpt2" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        shutil.copyfile(paths["config.json"], snapshot / "config.json")
        with patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as model_load:
            try:
                pretrained.load_distilgpt2(cache_dir=partial, allow_download=False)
            except pretrained.CheckpointUnavailableError:
                pass
            else:
                raise AssertionError("partial offline cache was accepted")
            assert model_load.call_count == 0
            altered = bytearray((snapshot / "config.json").read_bytes())
            altered[0] ^= 1
            (snapshot / "config.json").write_bytes(altered)
            try:
                pretrained.load_distilgpt2(cache_dir=partial, allow_download=False)
            except pretrained.CheckpointIntegrityError:
                pass
            else:
                raise AssertionError("tampered offline config was accepted")
            assert model_load.call_count == 0

        phase[0] = "model_load"
        torch.set_num_threads(1)
        torch.random.default_generator.manual_seed(17)
        bundle = pretrained.load_distilgpt2(cache_dir=cache, allow_download=False)
        model = bundle.model
        assert bundle.metadata["model_id"] == bundle.metadata["tokenizer_id"] == model_id
        assert bundle.metadata["model_revision"] == bundle.metadata["tokenizer_revision"] == revision
        assert bundle.metadata["asset_sha256"] == {name: item[1] for name, item in expected_assets.items()}
        assert type(model) is transformers.GPT2LMHeadModel and not model.training
        assert model.config.model_type == "gpt2" and model.config.vocab_size == 50257
        assert model.config.n_positions == 1024 and model.config._attn_implementation == "eager"
        assert model.config.use_cache is False
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight

        phase[0] = "forward"
        inputs = {"input_ids": torch.tensor([[15496, 995]], dtype=torch.int64, device="cpu")}
        ids = inputs["input_ids"]
        assert tuple(ids.shape) == (1, 2) and ids.dtype == torch.int64 and ids.device.type == "cpu"
        assert ids.numel() * ids.element_size() == 16 <= 64 * 1024
        case = {"schema": "resource-bootstrap-case-v1", "input_ids": ids.tolist()[0],
                "shape": list(ids.shape), "dtype": str(ids.dtype)}
        case_hash = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=True).encode()).hexdigest()
        assert case_hash == "a137759c4f2a40c4da15d1a4625086e160edc75f161748db91b07d83dae751e4"
        with torch.inference_mode():
            logits = model(**inputs).logits
        assert tuple(logits.shape) == (1, 2, 50257) and logits.dtype == torch.float32
        assert logits.device.type == "cpu" and logits.numel() * logits.element_size() == 402056
        assert 402056 <= 512 * 1024 and bool(torch.isfinite(logits).all())

        # Exercise the production result and frame encoders with synthetic
        # envelope metadata and an in-memory sink; no socket is opened.
        phase[0] = "response_frame"
        class Sink:
            def __init__(self):
                self.data = bytearray()
            def send(self, data):
                self.data.extend(data)
                return len(data)

        broker_id, request_id, team_id, job_id = "a" * 32, "r" * 64, "t" * 64, 2**63 - 1
        server = ipc.IPCServer(SimpleNamespace(broker_id=broker_id), max_batch=1,
                               max_tokens=2, max_response_bytes=4 * 1024 * 1024)
        outcome = SimpleNamespace(broker_id=broker_id, request_id=request_id, team_id=team_id,
                                  job_id=job_id, state="succeeded", submitted_ns=2**63 - 1,
                                  ready_ns=2**63 - 1, started_ns=2**63 - 1, finished_ns=2**63 - 1,
                                  result=ForwardResult(request_id, logits))
        sink = Sink()
        with patch.object(ipc.select, "select", return_value=([], [sink], [])):
            server._send_outcome(sink, outcome, SimpleNamespace(job_id=job_id), request_id, team_id)
        assert len(sink.data) <= 4 * 1024 * 1024
        assert struct.unpack("!I", sink.data[:4])[0] == len(sink.data) - 4
        framed = json.loads(sink.data[4:])
        assert framed["status"] == "succeeded" and framed["result"]["shape"] == [1, 2, 50257]
        assert framed["result"]["dtype"] == "torch.float32" and len(framed["result"]["logits"]) == 100514
        decoded = torch.tensor(framed["result"]["logits"], dtype=torch.float32, device="cpu")
        assert torch.equal(decoded.view(torch.int32), logits.reshape(-1).view(torch.int32))
        frame_bytes = len(sink.data)
        del decoded, framed, sink, logits, inputs, bundle, model
        assert not calls and not torch.cuda._initialized

        # Check loaded origins and bytecode policy. This checks the relevant
        # imported files, not every installed or native dependency.
        phase[0] = "provenance"
        roots = {"looking_glass": repo, "torch": site, "transformers": site,
                 "safetensors": site, "tokenizers": site, "huggingface_hub": site, "numpy": site}
        required = set(roots) | {"looking_glass.pretrained", "looking_glass.ipc",
                                 "looking_glass.resident_worker", "torch._ops", "torch._classes"}
        assert required.issubset(sys.modules)
        imported_repo_files = set()
        checked = 0
        for name, module in list(sys.modules.items()):
            top = name.split(".", 1)[0]
            if top not in roots or module is None:
                continue
            origin = getattr(module, "__file__", None)
            if origin is None:
                assert name not in required
                continue
            # These two PyTorch objects are synthetic namespace modules, not
            # files. Their defining torch._ops/_classes modules are checked
            # independently in this same origin walk.
            if name == "torch.ops":
                assert type(module) is torch._ops._Ops and origin == "_ops.py"
                continue
            if name == "torch.classes":
                assert type(module) is torch._classes._Classes and origin == "_classes.py"
                continue
            path = Path(origin).resolve(strict=True)
            assert path.is_relative_to(roots[top]) and path.suffix.lower() not in (".pyc", ".pyo")
            checked += 1
            if top == "looking_glass":
                imported_repo_files.add(path.relative_to(repo).as_posix())
        script = Path(__file__).resolve(strict=True)
        assert script == repo / "tests" / "test_bootstrap_pretrained_cpu.py"
        imported_repo_files.add(script.relative_to(repo).as_posix())
        for path in imported_repo_files:
            recorded = git("show", commit + ":" + path)
            assert hashlib.sha256(recorded).digest() == hashlib.sha256((repo / path).read_bytes()).digest()
        phase[0] = "source_final"
        assert not any(pycache.iterdir())
        assert not git("status", "--porcelain=v1", "--untracked-files=all")
        assert git("rev-parse", "HEAD").decode("ascii").strip() == commit
        assert git("show", "-s", "--format=%T", "HEAD").decode("ascii").strip() == tree
        assert not calls and not torch.cuda._initialized

    return {"status": "passed", "source_commit": commit, "source_tree": tree,
            "case_sha256": case_hash, "verified_assets": len(expected_assets),
            "input_bytes": 16, "output_bytes": 402056, "response_frame_bytes": frame_bytes,
            "response_frame_limit_bytes": 4 * 1024 * 1024,
            "imported_origins_checked": checked, "repo_files_checked": len(imported_repo_files),
            "guarded_cuda_and_network_calls": 0}


@unittest.skipUnless(os.environ.get(CACHE_ENV), "set LOOKING_GLASS_TEST_CACHE to opt in")
class BootstrapPretrainedCpuTests(unittest.TestCase):
    def test_direct_base_interpreter_offline_cpu_preflight(self):
        import json

        repo = Path(__file__).resolve().parent.parent
        site = Path(sysconfig.get_path("purelib")).resolve(strict=True)
        base = Path(getattr(sys, "_base_executable", None) or sys.executable).resolve(strict=True)
        scratch = os.environ.get("LOOKING_GLASS_TEST_SCRATCH") or None
        with tempfile.TemporaryDirectory(prefix="looking-glass-cpu-preflight-", dir=scratch) as temp:
            private = Path(temp).resolve(strict=True)
            (private / "pycache").mkdir()
            env = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("CUDA_", "PYTORCH_", "TORCH_", "CUBLAS_", "HF_",
                                                  "TRANSFORMERS_", "OMP_", "MKL_", "OPENBLAS_", "PYTHON",
                                                  "LOOKING_GLASS_"))}
            env.update({CACHE_ENV: str(Path(os.environ[CACHE_ENV]).absolute()),
                        "LOOKING_GLASS_BOOTSTRAP_REPO": str(repo),
                        "LOOKING_GLASS_BOOTSTRAP_SITE": str(site),
                        "LOOKING_GLASS_BOOTSTRAP_BASE": str(base),
                        "LOOKING_GLASS_BOOTSTRAP_PRIVATE": str(private),
                        "PYTHONPATH": os.pathsep.join((str(repo), str(site))),
                        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPYCACHEPREFIX": str(private / "pycache"), "PYTHONHASHSEED": "17",
                        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "HF_HOME": str(Path(os.environ[CACHE_ENV]).absolute().parent),
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1"})
            result = subprocess.run([str(base), "-S", str(Path(__file__).resolve()), "--preflight-child"],
                                    cwd=repo, env=env, capture_output=True, text=True, timeout=180)
        try:
            summary = json.loads(result.stdout)
        except ValueError:
            self.fail("private CPU preflight did not return a valid aggregate summary")
        if result.returncode != 0 or summary.get("status") != "passed":
            phases = {"startup", "source_initial", "torch_import", "imports", "assets",
                      "negative_cache_checks", "model_load", "forward", "response_frame",
                      "provenance", "source_final"}
            errors = {"AssertionError", "CheckpointUnavailableError", "CheckpointIntegrityError",
                      "OSError", "FileNotFoundError", "ImportError", "RuntimeError", "ValueError",
                      "TypeError", "TimeoutExpired", "KeyError", "CalledProcessError", "OtherError"}
            phase = summary.get("phase") if summary.get("phase") in phases else "unknown"
            error = summary.get("error_type") if summary.get("error_type") in errors else "OtherError"
            self.fail(f"private CPU preflight failed at {phase}: {error}")
        if os.environ.get(SUMMARY_ENV) == "1":
            print("BOOTSTRAP_PRETRAINED_CPU_V1 " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--preflight-child"]:
        import json
        phase = ["startup"]
        try:
            result = _child(phase)
        except BaseException as exc:
            allowed = {"AssertionError", "CheckpointUnavailableError", "CheckpointIntegrityError",
                       "OSError", "FileNotFoundError", "ImportError", "RuntimeError", "ValueError",
                       "TypeError", "TimeoutExpired", "KeyError", "CalledProcessError"}
            name = type(exc).__name__
            print(json.dumps({"status": "failed", "phase": phase[0],
                              "error_type": name if name in allowed else "OtherError"}), flush=True)
            raise SystemExit(1)
        print(json.dumps(result, sort_keys=True), flush=True)
    else:
        unittest.main()
