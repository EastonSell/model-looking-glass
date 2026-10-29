"""Standard-library/fake-device tests; never initialize CUDA."""

import importlib
import hashlib
import contextlib
import base64
import copy
import json
import subprocess
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from looking_glass.memory_budget import AdmissionError, CudaMemorySnapshot, TraceClaimsOutstanding


def valid_measurement(subject):
    path = "distilbert.transformer.layer.2"
    zero, donor = bytes(3072), bytes(3071) + b"\x01"
    def encoded(value): return base64.b64encode(value).decode("ascii")
    def events(stage):
        names = ["before"] if stage in ("target", "donor") else ["before", "patch", "after"]
        operations = ["capture"] if len(names) == 1 else ["capture", "patch", "capture"]
        return [{"name": name, "operation": operation, "order": index,
                 "module_path": path, "module_aliases": [path], "call_site_known": False,
                 "forward_index": 0, "invocation": 0, "batch_index": 0, "token_index": 2,
                 "output_selector": [0], "output_shape": [1, 10, 768], "selected_shape": [768],
                 "dtype": "torch.float32", "device": "cuda:0",
                 "source_capture": "before" if stage == "self" and index == 1 else None,
                 "preceding_same_site_actions": names[:index]}
                for index, (name, operation) in enumerate(zip(names, operations))]
    rows = []
    for stage in ("target", "donor", "self", "cross"):
        before = donor if stage == "donor" else zero
        after = {"target": None, "donor": None, "self": zero, "cross": donor}[stage]
        rows.append({"stage": stage, "is_bits": 0, "are_bits": 0, "is_logit": 0.0,
                     "are_logit": 0.0, "mask_logits_sha256": "0" * 64,
                     "before_float32_le_base64": encoded(before),
                     "after_float32_le_base64": None if after is None else encoded(after),
                     "events": events(stage)})
    total, cap = 8 * 1024 ** 3, 2 * 1024 ** 3
    return {"target_ids": list(subject.TARGET_IDS), "donor_ids": list(subject.DONOR_IDS),
        "output_token_ids": {"is": 2003, "are": 2024}, "seed": 17,
        "cleanup": {"allocated_bytes": 0, "reserved_bytes": 0},
        "cleanup_weakrefs": {"model": False, "target": False, "donor": False},
        "budget_active_before_candidate_write": 1, "runtime_uuid_verified": True,
        "device_name": "NVIDIA GeForce RTX 2070 SUPER", "device_total_bytes": total,
        "compute_capability": [7, 5], "allocator_fraction": .25, "allocator_cap_bytes": cap,
        "allocator_backend": "native",
        "whole_run_commitment_bytes": cap + 256 * 1024 ** 2,
        "gpu_trace_claim_bytes": 65536, "host_trace_claim_bytes": 1024 ** 2,
        "torch_version": "2.8.0+cu128", "cuda_build": "12.8",
        "python_version": "3.11.15",
        "execution_config": {"device": "cuda:0", "threads": 1, "interop_threads": 1,
            "deterministic_algorithms": True, "matmul_tf32": False, "cudnn_tf32": False,
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "overwrite_module_params_on_conversion": False, "swap_module_params_on_conversion": False},
        "peak_scope": "cumulative_since_pre_model_load_reset",
        "model_metadata": {"model": subject.MODEL_ID, "revision": subject.REVISION,
            "assets": {name: {"size": size, "sha256": digest}
                       for name, (size, digest) in subject.ASSETS.items()},
            "model_class": "DistilBertForMaskedLM", "tokenizer_class": "DistilBertTokenizerFast",
            "transformers_version": "4.57.3", "tokenizers_version": "0.22.2",
            "safetensors_version": "0.7.0", "attention_implementation": "eager",
            "parameter_count": 66985530, "cpu_tied_weight_verified": True},
        "sequence": {"forward_count": 4, "self_mask_logits_bitwise_equal": True,
                     "cross_source_landed": True, "source_differs_from_target": True,
                     "observations": rows}}


class OwnerAdmissionTests(unittest.TestCase):
    def test_parent_rejects_corrupt_identity_logits_and_patch_evidence(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        candidate = valid_measurement(subject)
        subject._validate_measurement(candidate)
        corruptions = [
            lambda data: data.update(runtime_uuid_verified=False),
            lambda data: data["model_metadata"].update(model="wrong"),
            lambda data: data["model_metadata"].update(tokenizers_version=""),
            lambda data: data.update(allocator_backend="cudaMallocAsync"),
            lambda data: data["sequence"]["observations"][0].update(is_bits=True),
            lambda data: data["sequence"]["observations"][0].update(is_bits=0x7f800000),
            lambda data: data["sequence"]["observations"][0].update(is_bits=0x80000000),
            lambda data: data["sequence"]["observations"][2].update(is_bits=0x3f800000, is_logit=1.0),
            lambda data: data["sequence"]["observations"][3].update(after_float32_le_base64="bad"),
            lambda data: data["sequence"]["observations"][3].update(
                after_float32_le_base64=data["sequence"]["observations"][0]["before_float32_le_base64"]),
            lambda data: data["sequence"]["observations"][3]["events"][1].update(token_index=3),
        ]
        for mutate in corruptions:
            altered = copy.deepcopy(candidate)
            mutate(altered)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                subject._validate_measurement(altered)

    def test_bootstrap_uses_verified_checkout_even_if_site_shadows_package(self):
        bootstrap = Path(__file__).resolve().parents[1] / "scripts" / "bert_gpu_bootstrap.py"
        self.assertTrue(bootstrap.is_file(), "the isolated source bootstrap is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, package, site = root / "scripts", root / "looking_glass", root / "site"
            scripts.mkdir()
            package.mkdir()
            (site / "looking_glass").mkdir(parents=True)
            (scripts / bootstrap.name).write_bytes(bootstrap.read_bytes())
            (package / "__init__.py").write_text("origin = 'checkout'\n", encoding="utf-8")
            (package / "bert_gpu_calibration.py").write_text(
                "import looking_glass, os; print(looking_glass.origin); "
                "print(os.environ['LOOKING_GLASS_BERT_BOOTSTRAP_SITE'])\n", encoding="utf-8")
            (site / "looking_glass" / "__init__.py").write_text(
                "raise RuntimeError('incorrect site package')\n", encoding="utf-8")
            prefix = root / "empty-cache"
            prefix.mkdir()
            completed = subprocess.run([sys.executable, "-I", "-S", "-B", "-X",
                "pycache_prefix=" + str(prefix), str(scripts / bootstrap.name),
                "--bootstrap-site", str(site)], capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.splitlines(), ["checkout", str(site.resolve())])

    def test_controller_accepts_only_matching_supervised_owner(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "run_calibration", None)),
                        "the supervised calibration controller is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "work").mkdir()
            cache, site = root / "cache", root / "site"
            cache.mkdir()
            site.mkdir()
            identity = SimpleNamespace(lock_path=root / "device.lock", marker_path=root / "unsafe")
            before = {"free_bytes": 6 * 1024 ** 3, "timestamp_utc": "2026-10-28T00:00:00Z"}
            invalid = None
            def fake_supervisor(command, **kwargs):
                self.assertEqual(Path(command[6]), Path(subject.__file__).resolve().parents[1]
                                 / "scripts" / "bert_gpu_bootstrap.py")
                self.assertEqual(Path(command[8]), site)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-fixed")
                run_dir, run_id = Path(command[-3]), command[-1]
                measurement = valid_measurement(subject)
                if invalid == "count": measurement["sequence"]["forward_count"] = 3
                candidate = {"schema": "bert-gpu-calibration-candidate-v1", "run_id": run_id,
                             "supervisor_run_id": "supervisor-fixed", "pid": 321 if invalid == "pid" else 123,
                             "measurement": measurement}
                raw = json.dumps(candidate).encode("utf-8")
                (run_dir / "candidate.json").write_bytes(raw)
                final = {"schema": "bert-gpu-calibration-final-v1", "run_id": run_id,
                         "supervisor_run_id": "supervisor-fixed", "pid": 123,
                         "candidate_sha256": hashlib.sha256(raw).hexdigest(), "active_memory_leases": 0}
                (run_dir / "final.json").write_text(json.dumps(final), encoding="utf-8")
                memory = {"device_free_bytes": 6 * 1024 ** 3, "device_total_bytes": 8 * 1024 ** 3,
                          "allocated_bytes": 0, "reserved_bytes": 0,
                          "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
                measurement["cleanup"].update(memory)
                candidate["measurement"] = measurement
                raw = json.dumps(candidate).encode("utf-8")
                (run_dir / "candidate.json").write_bytes(raw)
                final["candidate_sha256"] = hashlib.sha256(raw).hexdigest()
                (run_dir / "final.json").write_text(json.dumps(final), encoding="utf-8")
                stage_counts = (("before_model_load", 0), ("before_model_copy", None),
                    ("after_model_copy", None), ("before_target", 0), ("after_target", 1),
                    ("before_donor", 1), ("after_donor", 2), ("before_self", 2),
                    ("after_self", 3), ("before_cross", 3), ("after_cross", 4),
                    ("cleanup_before_workspace_clear", None), ("after_cleanup", None))
                for stage, count in stage_counts:
                    if invalid == "missing_stage" and stage == "after_cross":
                        continue
                    if invalid == "bad_stage" and stage == "after_cross":
                        count = 3
                    item = {"memory": memory, "observations_live": count,
                            "peak_scope": "cumulative_since_pre_model_load_reset"}
                    (run_dir / (stage + ".json")).write_text(json.dumps(item), encoding="utf-8")
                (run_dir / "cleanup_weakrefs.json").write_text(
                    json.dumps(measurement["cleanup_weakrefs"]), encoding="utf-8")
                return SimpleNamespace(run_id="supervisor-fixed", pid=123, exit_code=0,
                                       clean=True, timed_out=False, reason="completed")
            with patch.object(subject, "_verify_startup"), \
                 patch.object(subject, "_source_identity", return_value=("a" * 40, {"test.py": "b" * 64})), \
                 patch.object(subject, "_verify_offline_cache"), \
                 patch.object(subject, "_smi_snapshot", return_value=("GPU-fixed", before)), \
                 patch.object(subject, "resolve_device_identity", return_value=identity), \
                 patch.object(subject, "run_supervised", side_effect=fake_supervisor), \
                 patch.object(subject, "_owner_presence_after_exit",
                              return_value={"query": "ok", "owner_pid_listed": False}):
                outcome = subject.run_calibration(workspace_root=root, cache_dir=cache,
                                                  output=root / "result.json", site_packages=site,
                                                  smi="fake")
                invalid = "pid"
                rejected = subject.run_calibration(workspace_root=root, cache_dir=cache,
                                                   output=root / "rejected.json", site_packages=site,
                                                   smi="fake")
                invalid = "count"
                rejected_count = subject.run_calibration(workspace_root=root, cache_dir=cache,
                    output=root / "rejected-count.json", site_packages=site, smi="fake")
                invalid = "missing_stage"
                missing = subject.run_calibration(workspace_root=root, cache_dir=cache,
                    output=root / "missing.json", site_packages=site, smi="fake")
                invalid = "bad_stage"
                bad = subject.run_calibration(workspace_root=root, cache_dir=cache,
                    output=root / "bad.json", site_packages=site, smi="fake")
            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["measurement"]["sequence"]["forward_count"], 4)
            self.assertTrue((root / "result.json").exists())
            self.assertEqual(rejected["status"], "incomplete")
            self.assertNotIn("measurement", rejected)
            self.assertEqual(rejected_count["status"], "incomplete")
            self.assertEqual(missing["status"], "incomplete")
            self.assertEqual(bad["status"], "incomplete")

    def test_mismatched_device_totals_rejected_before_loading(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        gib = 1024 ** 3
        cuda = SimpleNamespace(is_initialized=lambda: False, device_count=lambda: 1,
            get_device_properties=lambda _: SimpleNamespace(uuid=object(), total_memory=8 * gib),
            get_allocator_backend=lambda: "native",
            set_per_process_memory_fraction=lambda *args: None,
            get_per_process_memory_fraction=lambda _: .25,
            reset_peak_memory_stats=lambda _: None)
        snapshot = {"device_free_bytes": 6 * gib, "device_total_bytes": 9 * gib,
                    "allocated_bytes": 0, "reserved_bytes": 0,
                    "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        session, lease = object.__new__(subject.OwnerSession), object.__new__(subject.DeviceLease)
        with patch.object(subject.OwnerSession, "check"), \
             patch.object(subject.OwnerSession, "verify_runtime_binding"), \
             patch.object(subject, "_runtime_uuid", return_value="GPU-fixed"), \
             patch.object(subject, "_guarded_snapshot", return_value=snapshot), \
             patch.object(subject, "_load_model") as load:
            with self.assertRaisesRegex(RuntimeError, "totals disagree"):
                subject.measure_owned(session, lease, SimpleNamespace(cuda=cuda), "unused",
                                      record_stage=lambda *args: None)
        load.assert_not_called()

    def test_guarded_snapshot_stops_queries_if_lease_is_revoked_during_sync(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "_guarded_snapshot", None)),
                        "guarded CUDA snapshot is unavailable")
        checks, queries = [], []
        def check(*args):
            checks.append(1)
            if len(checks) == 2:
                raise RuntimeError("lease revoked")
        cuda = SimpleNamespace(synchronize=lambda _: queries.append("synchronize"),
                               mem_get_info=lambda _: queries.append("memory"))
        with patch.object(subject.OwnerSession, "check", side_effect=check):
            with self.assertRaisesRegex(RuntimeError, "revoked"):
                subject._guarded_snapshot(object(), object(), SimpleNamespace(cuda=cuda))
        self.assertEqual(queries, ["synchronize"])

    def test_admission_rejects_total_changed_after_initial_snapshot(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        gib = 1024 ** 3
        cuda = SimpleNamespace(is_initialized=lambda: False, device_count=lambda: 1,
            get_device_properties=lambda _: SimpleNamespace(uuid=object(), total_memory=8 * gib),
            get_allocator_backend=lambda: "native",
            set_per_process_memory_fraction=lambda *args: None,
            get_per_process_memory_fraction=lambda _: .25, reset_peak_memory_stats=lambda _: None,
            mem_get_info=lambda _: (6 * gib, 9 * gib), memory_allocated=lambda _: 0,
            memory_reserved=lambda _: 0)
        snapshot = {"device_free_bytes": 6 * gib, "device_total_bytes": 8 * gib,
                    "allocated_bytes": 0, "reserved_bytes": 0,
                    "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        session, lease = object.__new__(subject.OwnerSession), object.__new__(subject.DeviceLease)
        with patch.object(subject.OwnerSession, "check"), \
             patch.object(subject.OwnerSession, "verify_runtime_binding"), \
             patch.object(subject, "_runtime_uuid", return_value="GPU-fixed"), \
             patch.object(subject, "_guarded_snapshot", return_value=snapshot), \
             patch.object(subject, "_load_model") as load:
            with self.assertRaisesRegex(RuntimeError, "total"):
                subject.measure_owned(session, lease, SimpleNamespace(cuda=cuda), "unused",
                                      record_stage=lambda *args: None)
        load.assert_not_called()

    def test_non_native_allocator_rejected_before_setting_cap_or_loading(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        gib = 1024 ** 3
        calls = []
        def set_cap(*args):
            calls.append("set-cap")
            raise RuntimeError("continued past backend check")
        cuda = SimpleNamespace(is_initialized=lambda: False, device_count=lambda: 1,
            get_device_properties=lambda _: SimpleNamespace(uuid=object(), total_memory=8 * gib),
            get_allocator_backend=lambda: "cudaMallocAsync",
            set_per_process_memory_fraction=set_cap)
        session, lease = object.__new__(subject.OwnerSession), object.__new__(subject.DeviceLease)
        with patch.object(subject.OwnerSession, "check"), \
             patch.object(subject.OwnerSession, "verify_runtime_binding"), \
             patch.object(subject, "_runtime_uuid", return_value="GPU-fixed"), \
             patch.object(subject, "_load_model") as load:
            with self.assertRaisesRegex(RuntimeError, "allocator"):
                subject.measure_owned(session, lease, SimpleNamespace(cuda=cuda), "unused",
                                      record_stage=lambda *args: None)
        load.assert_not_called()
        self.assertEqual(calls, [])

    def test_bounded_result_reader_rejects_oversized_and_non_object(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "_read_json_capped", None)),
                        "bounded result reader is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_bytes(b"{" + b" " * 100 + b"}")
            with self.assertRaises(ValueError):
                subject._read_json_capped(path, 16)
            path.write_bytes(b"[]")
            with self.assertRaises(ValueError):
                subject._read_json_capped(path, 16)

    def test_owner_holds_reservations_until_model_and_inputs_are_disposed(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        gib, mib = 1024 ** 3, 1024 ** 2

        class FakeCuda:
            def __init__(self):
                self.total, self.allocated, self.reserved, self.peak = 8 * gib, 0, 0, 0
            def is_initialized(self): return False
            def device_count(self): return 1
            def get_allocator_backend(self): return "native"
            def get_device_properties(self, index):
                return SimpleNamespace(uuid=object(), total_memory=self.total, name="Fake GPU", major=7, minor=5)
            def set_per_process_memory_fraction(self, fraction, index): self.fraction = fraction
            def get_per_process_memory_fraction(self, index): return self.fraction
            def reset_peak_memory_stats(self, index): self.peak = self.reserved
            def synchronize(self, index): pass
            def mem_get_info(self, index): return 6 * gib - self.reserved, self.total
            def memory_allocated(self, index): return self.allocated
            def memory_reserved(self, index): return self.reserved
            def max_memory_allocated(self, index): return self.peak
            def max_memory_reserved(self, index): return self.peak
            def manual_seed(self, seed): pass
            def empty_cache(self):
                if self.allocated == 0: self.reserved = 0
            def allocate(self, count):
                self.allocated += count
                self.reserved = max(self.reserved, self.allocated)
                self.peak = max(self.peak, self.reserved)
        cuda = FakeCuda()
        class FakeTensor:
            def __init__(self, values, gpu=False):
                self.values, self.bytes = values, 80 if gpu else 0
                self.device = SimpleNamespace(type="cuda" if gpu else "cpu", index=0 if gpu else None)
                if gpu: cuda.allocate(self.bytes)
            def to(self, device): return FakeTensor(self.values, gpu=True)
            def __del__(self): cuda.allocated -= self.bytes
        class FakeModel:
            def __init__(self):
                self.weight = SimpleNamespace(device=SimpleNamespace(type="cpu", index=None),
                                              dtype="float32", numel=lambda: 66985530)
                self.vocab_projector = SimpleNamespace(weight=self.weight)
                self.bytes, self.training = 0, False
            def to(self, device):
                self.bytes = 300 * mib
                cuda.allocate(self.bytes)
                self.weight.device = SimpleNamespace(type="cuda", index=0)
                return self
            def parameters(self): return iter((self.weight,))
            def buffers(self): return iter(())
            def get_input_embeddings(self): return SimpleNamespace(weight=self.weight)
            def __del__(self): cuda.allocated -= self.bytes
        torch = SimpleNamespace(
            cuda=cuda, float32="float32", long="int64", __version__="2.8.0+cu128",
            version=SimpleNamespace(cuda="fake"),
            random=SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda value: None)),
            __future__=SimpleNamespace(get_overwrite_module_params_on_conversion=lambda: False,
                                       get_swap_module_params_on_conversion=lambda: False),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                     cudnn=SimpleNamespace(allow_tf32=False, benchmark=False, deterministic=True)),
            set_num_threads=lambda value: None, set_num_interop_threads=lambda value: None,
            get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1,
            use_deterministic_algorithms=lambda value: None,
            are_deterministic_algorithms_enabled=lambda: True,
            tensor=lambda values, **kwargs: FakeTensor(values),
            _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: None))
        session = object.__new__(subject.OwnerSession)
        lease = object.__new__(subject.DeviceLease)
        stages = []
        def fake_sequence(model, target, donor, *, is_id, are_id, checkpoint, synchronize, check_owner):
            self.assertEqual(target.values, [list(subject.TARGET_IDS)])
            self.assertEqual(donor.values, [list(subject.DONOR_IDS)])
            for stage, count in (("before_target", 0), ("after_target", 1),
                                 ("before_donor", 1), ("after_donor", 2),
                                 ("before_self", 2), ("after_self", 3),
                                 ("before_cross", 3), ("after_cross", 4)):
                checkpoint(stage, count)
            return {"forward_count": 4, "observations": [], "self_mask_logits_bitwise_equal": True,
                    "cross_source_landed": True, "source_differs_from_target": True}
        def fake_loader(cache, runtime): return FakeModel(), {"parameter_count": 66985530}
        with patch.object(subject.OwnerSession, "check"), \
             patch.object(subject.OwnerSession, "verify_runtime_binding"), \
             patch.object(subject, "_runtime_uuid", return_value="GPU-fixed"), \
             patch.object(subject, "_load_model", new=fake_loader), \
             patch.object(subject, "run_sequence", new=fake_sequence):
            try:
                result = subject.measure_owned(session, lease, torch, "unused",
                                               record_stage=lambda name, data: stages.append((name, data)))
            except NotImplementedError:
                self.fail("the supervised owner workload is unavailable")
        self.assertEqual(result["sequence"]["forward_count"], 4)
        self.assertEqual(result["cleanup"]["allocated_bytes"], 0)
        self.assertEqual(result["cleanup"]["reserved_bytes"], 0)
        self.assertEqual(result["budget_active_before_candidate_write"], 1)
        self.assertEqual(result["execution_config"]["device"], "cuda:0")
        self.assertEqual(result["peak_scope"], "cumulative_since_pre_model_load_reset")
        self.assertEqual(result["allocator_backend"], "native")
        self.assertTrue(result["python_version"])
        self.assertEqual([name for name, _ in stages].count("after_cross"), 1)
        del result
        subject._release_owner_budget_after_candidate()
        self.assertEqual(subject._OWNER_CUSTODY["budget"].active_leases(), ())

    def test_tokenizer_must_encode_both_output_words_as_single_tokens(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "verify_tokenizer_inputs", None)),
                        "tokenizer verification is unavailable")
        class BadTokenizer:
            is_fast = True
            mask_token_id = 103
            def convert_tokens_to_ids(self, word): return {"is": 2003, "are": 2024}[word]
            def encode(self, word, add_special_tokens=False):
                return [2003, 9] if word == "is" else [2024]
        with self.assertRaisesRegex(RuntimeError, "token"):
            subject.verify_tokenizer_inputs(BadTokenizer())

    def test_loader_rejects_checkpoint_loading_diagnostics(self):
        # Incorrect/missing checkpoint weights must never reach model transfer.
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "_load_model", None)),
                        "the offline model loader is unavailable")
        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace(model_type="distilbert", n_layers=6, dim=768,
                                              vocab_size=30522, _attn_implementation="eager")
                self.training = True
            def eval(self): self.training = False
        fake_model = FakeModel()
        tokenizer = SimpleNamespace(mask_token_id=103,
                                    convert_tokens_to_ids=lambda word: {"is": 2003, "are": 2024}[word])
        missing = {"missing_keys": ["missing.weight"], "unexpected_keys": [],
                   "mismatched_keys": [], "error_msgs": []}
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            AutoModelForMaskedLM=SimpleNamespace(from_pretrained=lambda *a, **k: (fake_model, missing)),
            DistilBertForMaskedLM=FakeModel, __version__="fake")
        fake_torch = SimpleNamespace(device=lambda *_: contextlib.nullcontext(), float32="float32")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "hub"
            snapshot = cache / "models--distilbert--distilbert-base-uncased" / "snapshots" / subject.REVISION
            snapshot.mkdir(parents=True)
            data = b"pinned"
            (snapshot / "config.json").write_bytes(data)
            assets = {"config.json": (len(data), hashlib.sha256(data).hexdigest())}
            with patch.object(subject, "ASSETS", assets), patch.dict(sys.modules, {"transformers": fake_transformers}):
                with self.assertRaisesRegex(RuntimeError, "load|weight"):
                    subject._load_model(cache, fake_torch)

    def test_whole_owner_commitment_and_trace_claims_stay_charged(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "reserve_owner_budget", None)),
                        "owner-wide memory admission is unavailable")
        gib, mib = 1024 ** 3, 1024 ** 2
        snapshot = CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0)
        budget, lease = subject.reserve_owner_budget(lambda: snapshot, cap=2 * gib)
        held = budget.snapshot()
        self.assertEqual(held.committed_incremental_bytes, 2 * gib + 256 * mib)
        self.assertEqual((held.gpu_trace_bytes, held.host_trace_bytes), (64 * 1024, mib))
        with self.assertRaises(TraceClaimsOutstanding):
            lease.release()
        self.assertEqual(len(budget.active_leases()), 1)
        lease.release_trace("gpu", 64 * 1024)
        lease.release_trace("host", mib)
        lease.release()
        self.assertEqual(budget.active_leases(), ())

    def test_whole_owner_budget_rejects_insufficient_device_free_memory(self):
        subject = importlib.import_module("looking_glass.bert_gpu_calibration")
        self.assertTrue(callable(getattr(subject, "reserve_owner_budget", None)),
                        "owner-wide memory admission is unavailable")
        gib = 1024 ** 3
        snapshot = CudaMemorySnapshot(3 * gib, 8 * gib, 0, 0)
        with self.assertRaises(AdmissionError):
            subject.reserve_owner_budget(lambda: snapshot, cap=2 * gib)


if __name__ == "__main__":
    unittest.main()
