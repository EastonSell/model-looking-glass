"""Supervise a CPU-only, frozen ViT/COCO split in a fresh evidence directory.

The executing checkout/interpreter and dependencies are trusted. Git/blob and
file hashes are operational consistency checks, not execution attestation.
The reviewed CPU supervisor reaps the direct owner; this entry does not claim
operating-system process-tree or GPU custody.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "tools/run_vision_coco_study.py", "tools/prepare_vision_coco_cases.py",
    "looking_glass/__init__.py", "looking_glass/vision_coco_cases.py",
    "looking_glass/vision_coco_donors.py", "looking_glass/vision_coco_study.py",
    "looking_glass/intervention_plan.py", "looking_glass/interventions.py",
    "looking_glass/model_ownership.py", "looking_glass/cpu_study_supervision.py",
    "looking_glass/supervisor.py", "looking_glass/device_lease.py", "looking_glass/device_identity.py",
    "experiments/vision_coco_spatial_v1.json", "experiments/vision_coco_same_species_v1.json",
    "docs/VISION_COCO_SPATIAL_V1.md", "docs/CPU_STUDY_SUPERVISION.md",
)
CASES_SHA = "8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207"
DONORS_SHA = "5874dae623703d318d1b59bf30d427d053193932fd71e6ebca328a8e673439d0"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_project_origin():
    import importlib.util
    spec = importlib.util.find_spec("looking_glass")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != (ROOT / "looking_glass/__init__.py").resolve():
        raise ValueError("project package is shadowed by another installation")


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short record write")
        stream.flush()
    return sha(raw)


def git(args):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_GRAFT_FILE=os.devnull, GIT_NO_REPLACE_OBJECTS="1")
    return subprocess.check_output(["git", "--no-replace-objects", *args], cwd=ROOT, env=env)


def source_snapshot(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase commit ID")
    if git(["rev-parse", "HEAD"]).decode().strip() != revision:
        raise ValueError("checkout HEAD differs from frozen revision")
    items = {}
    for path in SOURCES:
        actual = (ROOT / path).read_bytes()
        committed = git(["show", f"{revision}:{path}"])
        if actual != committed:
            raise ValueError(f"checkout differs from committed source: {path}")
        items[path] = sha(actual)
    if (items["experiments/vision_coco_spatial_v1.json"] != CASES_SHA
            or items["experiments/vision_coco_same_species_v1.json"] != DONORS_SHA):
        raise ValueError("frozen case/donor bytes differ")
    return {"revision": revision, "sources": items,
            "scope": "trusted quiescent checkout; hashes do not attest already executing code or dependencies"}


def recheck_snapshot(snapshot):
    if set(snapshot["sources"]) != set(SOURCES):
        raise ValueError("source inventory differs")
    for name, expected in snapshot["sources"].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"source changed during attempt: {name}")


def completed_development(root, revision):
    supervisor = json.loads((root / "supervisor.json").read_bytes())
    raw = (root / "data" / "terminal.json").read_bytes()
    prior = json.loads(raw)
    rows = prior.get("pairs")
    count = prior.get("submissions")
    if (supervisor.get("status") != "complete" or supervisor.get("clean") is not True
            or type(supervisor.get("exit_code")) is not int or supervisor.get("exit_code") != 0
            or supervisor.get("timed_out") is not False or supervisor.get("terminal_sha256") != sha(raw)
            or prior.get("source_revision") != revision or prior.get("split") != "development"
            or prior.get("case_sha256") != CASES_SHA or prior.get("donor_sha256") != DONORS_SHA
            or prior.get("status") != "complete" or prior.get("checkpoint_preflight_only") is not False
            or prior.get("synthetic_rehearsal") is not False
            or type(count) is not int or not 8 <= count <= 120 or prior.get("saved_forwards") != count
            or type(rows) is not list or len(rows) != 4
            or [row.get("pair_id") for row in rows] != [f"pair-{i:02d}" for i in range(4)]
            or any(row.get("split") != "development" or row.get("status") not in
                   ("positive", "negative", "baseline-ineligible", "dose-ineligible", "dose-unmatchable")
                   for row in rows) or (root / "data" / "failure.json").exists()):
        raise ValueError("development evidence was not completed under this frozen source")
    data = root / "data"
    started = json.loads((data / "started.json").read_bytes())
    if (started.get("split") != "development" or started.get("case_sha256") != CASES_SHA
            or started.get("donor_sha256") != DONORS_SHA or started.get("checkpoint_preflight_only") is not False
            or started.get("source", {}).get("revision") != revision):
        raise ValueError("development start record differs")
    for row in rows:
        if json.loads((data / f"{row['pair_id']}.json").read_bytes()) != row:
            raise ValueError("development pair record differs")
    expected_records = {f"forward-{i:03d}.json" for i in range(1, count + 1)}
    expected_bins = {f"forward-{i:03d}.bin" for i in range(1, count + 1)}
    if ({p.name for p in data.glob("forward-*.json")} != expected_records
            or {p.name for p in data.glob("forward-*.bin")} != expected_bins):
        raise ValueError("development forward inventory differs")
    for i in range(1, count + 1):
        row = json.loads((data / f"forward-{i:03d}.json").read_bytes())
        name = f"forward-{i:03d}.bin"
        payload = (data / name).read_bytes()
        if (row.get("submission") != i or row.get("file") != name or not 0 < len(payload) <= 1_000_000
                or row.get("sha256") != sha(payload)):
            raise ValueError("development forward bytes differ")
    return sha(raw)


def worker(args):
    # Claim the CPU lease before torch, NumPy, Transformers or the study runner.
    from looking_glass.cpu_study_supervision import claim_cpu_study
    session, lease = claim_cpu_study(args.custody)
    model = evidence = None
    success = False
    try:
        snapshot = json.loads((args.output / "source.json").read_bytes())
        recheck_snapshot(snapshot)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import numpy as np
        import PIL
        from PIL import Image
        import torch
        import transformers
        from transformers import AutoImageProcessor, ViTForImageClassification
        from looking_glass.vision_coco_cases import verify_assets
        from looking_glass.vision_coco_donors import canonical_donors
        from looking_glass.vision_coco_study import Evidence, aggregate, run_split

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
            raise ValueError("CPU masking or pre-import CUDA contract failed")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise ValueError("denormal preservation unavailable")
        if torch.tensor([1.401298464324817e-45], dtype=torch.float32).item() == 0:
            raise ValueError("float32 denormal probe failed")
        if (torch.get_default_dtype() != torch.float32 or PIL.__version__ != "11.3.0"
                or transformers.__version__ != "4.57.3" or np.__version__ != "2.4.6"
                or torch.__version__ != "2.8.0+cu128"):
            raise ValueError("runtime versions or float32 default differ from the frozen contract")
        if args.synthetic_rehearsal:
            class Layer(torch.nn.Module):
                def forward(self, value):
                    return value + .01 * value.mean(dim=1, keepdim=True)
            class FakeVit(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.vit = torch.nn.Module()
                    self.vit.encoder = torch.nn.Module()
                    self.vit.encoder.layer = torch.nn.ModuleList([Layer() for _ in range(12)])
                def forward(self, pixel_values):
                    from types import SimpleNamespace
                    value = pixel_values
                    for layer in self.vit.encoder.layer:
                        value = layer(value)
                    scalar = value[:, 0, 0]
                    logits = torch.zeros((1, 1000), dtype=torch.float32)
                    logits[:, 151:269] = 2 + scalar[:, None]
                    logits[:, 281:286] = -scalar[:, None]
                    return SimpleNamespace(logits=logits)
            torch.manual_seed(4)
            model = FakeVit().eval()
            fabricated = {1: torch.randn(1, 197, 192), 2: torch.randn(1, 197, 192)}
            pair = {"pair_id": "pair-00", "split": "development", "cat": {"image_id": 1},
                    "dog": {"image_id": 2}, "box_tokens": [1, 2, 15, 16],
                    "control_tokens": [40, 41, 54, 55]}
            fake_donors = {"rows": [{"pair_id": "pair-00", "receiver_side": s, "donor": None}
                                     for s in ("cat", "dog")]}
            evidence = Evidence(args.output / "data", 30)
            rows = run_split(model, [pair], fake_donors, fabricated, evidence)
            if evidence.submissions != 28 or len(evidence.records) != 28:
                raise ValueError("synthetic rehearsal schedule differs")
            evidence.json("terminal.json", {"schema": "vision-coco-study-terminal-v1", "status": "complete",
                          "split": args.split, "source_revision": snapshot["revision"],
                          "case_sha256": CASES_SHA, "donor_sha256": DONORS_SHA,
                          "checkpoint_preflight_only": False, "synthetic_rehearsal": True,
                          "submissions": evidence.submissions, "pairs": rows,
                          "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]})
            model = None
            gc.collect()
            session.check(lease)
            session.attest_clean(lease)
            return 0
        case_bytes = (ROOT / "experiments/vision_coco_spatial_v1.json").read_bytes()
        donor_bytes = (ROOT / "experiments/vision_coco_same_species_v1.json").read_bytes()
        if sha(case_bytes) != CASES_SHA or sha(donor_bytes) != DONORS_SHA:
            raise ValueError("case or donor manifest changed")
        annotation_bytes = (args.assets / "instances_val2017.json").read_bytes()
        if canonical_donors(case_bytes, annotation_bytes) != donor_bytes:
            raise ValueError("same-species donor selection does not reproduce")
        manifest, donors = json.loads(case_bytes), json.loads(donor_bytes)
        verify_assets(args.assets)
        images = [pair[side] for pair in manifest["pairs"] for side in ("cat", "dog")]
        if (len(images) != 24 or len({row["image_id"] for row in images}) != 24
                or len({row["jpeg_sha256"] for row in images}) != 24
                or len({row["pixel_sha256"] for row in images}) != 24):
            raise ValueError("selected images are not disjoint")
        pairs = [pair for pair in manifest["pairs"] if pair["split"] == args.split]
        if [pair["pair_id"] for pair in pairs] != ([f"pair-{i:02d}" for i in
                                                    (range(4) if args.split == "development" else range(4, 12))]):
            raise ValueError("split identities differ")
        processor = AutoImageProcessor.from_pretrained(str(args.assets), local_files_only=True, use_fast=False)
        if (type(processor).__name__ != "ViTImageProcessor" or processor.size != {"height": 224, "width": 224}
                or int(processor.resample) != 2 or processor.image_mean != [.5] * 3
                or processor.image_std != [.5] * 3 or processor.rescale_factor != 1 / 255):
            raise ValueError("processor differs")
        pixels = {}
        for pair in pairs:
            for side in ("cat", "dog"):
                item = pair[side]
                path = args.assets / "images" / item["file_name"]
                raw = path.read_bytes()
                if len(raw) != item["jpeg_bytes"] or sha(raw) != item["jpeg_sha256"]:
                    raise ValueError("JPEG differs from manifest")
                with Image.open(path) as image:
                    if image.size != (item["width"], item["height"]) or image.getexif().get(274, 1) != 1:
                        raise ValueError("image dimensions or orientation differ")
                    value = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
                if (tuple(value.shape) != (1, 3, 224, 224) or value.dtype != torch.float32
                        or not bool(torch.isfinite(value).all()) or
                        sha(value.contiguous().numpy().astype("<f4", copy=False).tobytes()) != item["pixel_sha256"]):
                    raise ValueError("processed pixels differ from manifest")
                pixels[item["image_id"]] = value
        if args.preflight_only:
            def forbid_checkpoint_forward(*_args, **_kwargs):
                raise RuntimeError("checkpoint forward is forbidden in preflight")
            ViTForImageClassification.forward = forbid_checkpoint_forward
        model, load_info = ViTForImageClassification.from_pretrained(
            str(args.assets), local_files_only=True, use_safetensors=True,
            output_loading_info=True, attn_implementation="eager")
        if any(load_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("checkpoint loading was incomplete")
        model.eval().to("cpu")
        config = model.config
        if (config.hidden_size != 192 or config.num_hidden_layers != 12 or config.num_labels != 1000
                or config.patch_size != 16 or config.image_size != 224):
            raise ValueError("unexpected model geometry")
        for side, group in manifest["class_groups"].items():
            if any(config.id2label[item["id"]] != item["label"] for item in group):
                raise ValueError(f"{side} class labels differ")
        def hooks():
            return {name: (len(module._forward_pre_hooks), len(module._forward_hooks))
                    for name, module in model.named_modules()}
        initial_hooks = hooks()
        def state_digest():
            digest = hashlib.sha256()
            for name, value in sorted(model.state_dict().items()):
                cpu = value.detach().cpu().contiguous()
                digest.update(name.encode() + b"\0" + str(cpu.dtype).encode() + b"\0")
                digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode() + b"\0")
                digest.update(cpu.numpy().tobytes())
            return digest.hexdigest()
        before = state_digest()
        runtime = {"device": "cpu", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                   "cuda_initialized": torch.cuda.is_initialized(), "python": sys.version.split()[0],
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   "numpy": np.__version__, "pillow": PIL.__version__,
                   "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "mkldnn_enabled": torch.backends.mkldnn.enabled,
                   "processor": type(processor).__name__, "denormals_preserved": True}
        evidence = Evidence(args.output / "data", len(pairs) * 30, labels=config.id2label)
        evidence.json("started.json", {"schema": "vision-coco-study-start-v1", "split": args.split,
                                       "source": snapshot, "runtime": runtime, "model_state_sha256": before,
                                       "case_sha256": CASES_SHA, "donor_sha256": DONORS_SHA,
                                       "checkpoint_preflight_only": args.preflight_only})
        if args.preflight_only:
            rows, decision = [], None
        else:
            rows = run_split(model, pairs, donors, pixels, evidence)
            decision = aggregate(rows) if args.split == "heldout" else None
        after = state_digest()
        if after != before or hooks() != initial_hooks or torch.cuda.is_initialized():
            raise ValueError("model state, hooks or CPU-only runtime changed")
        recheck_snapshot(snapshot)
        terminal = {"schema": "vision-coco-study-terminal-v1", "status": "complete",
                    "split": args.split, "source_revision": snapshot["revision"],
                    "development_terminal_sha256": snapshot.get("development_terminal_sha256"),
                    "case_sha256": CASES_SHA, "donor_sha256": DONORS_SHA,
                    "submissions": evidence.submissions, "saved_forwards": len(evidence.records),
                    "model_state_sha256_before_after": before, "runtime": runtime,
                    "checkpoint_preflight_only": args.preflight_only, "pairs": rows, "aggregate": decision}
        terminal["synthetic_rehearsal"] = False
        if terminal["submissions"] != terminal["saved_forwards"]:
            raise ValueError("submission/evidence count differs")
        evidence.json("terminal.json", terminal)
        model = None
        gc.collect()
        session.check(lease)
        session.attest_clean(lease)
        success = True
    except BaseException as error:
        if evidence is not None:
            try:
                evidence.json("failure.json", {"status": "incomplete", "error_type": type(error).__name__,
                                                "active": evidence.active, "submissions": evidence.submissions,
                                                "saved_forwards": len(evidence.records)})
            except Exception:
                pass
        raise
    finally:
        model = None
        gc.collect()
        lease.release()
    return 0 if success else 1


def controller(args):
    from looking_glass.cpu_study_supervision import run_cpu_study
    if not args.output.is_absolute() or not args.custody.is_absolute() or not args.assets.is_absolute():
        raise ValueError("output, custody and assets paths must be absolute")
    if args.output.resolve().is_relative_to(ROOT) or args.custody.resolve().is_relative_to(ROOT):
        raise ValueError("evidence and custody roots must be outside the checkout")
    if args.synthetic_rehearsal and (args.split != "development" or args.preflight_only):
        raise ValueError("synthetic rehearsal requires development and no checkpoint preflight")
    snapshot = source_snapshot(args.revision)
    if args.split == "heldout" and not args.preflight_only:
        if args.development is None:
            raise ValueError("held-out phase requires the completed development receipt")
        snapshot["development_terminal_sha256"] = completed_development(args.development, args.revision)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "source.json", snapshot)
    bytecode_prefix = args.output / "empty-bytecode-cache"
    bytecode_prefix.mkdir()
    temporary = args.output / "temporary"
    temporary.mkdir()
    base = Path(sys._base_executable)
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    dependency = args.pillow.resolve(strict=True)
    bootstrap = ("import runpy,sys; sys.path.extend([sys.argv[2],sys.argv[3],sys.argv[1]]); "
                 "sys.argv=sys.argv[4:]; runpy.run_path(sys.argv[0],run_name='__main__')")
    command = [str(base), "-I", "-S", "-B", "-X", f"pycache_prefix={bytecode_prefix}",
               "-c", bootstrap, str(ROOT), str(site_packages),
               str(dependency), str(Path(__file__).resolve()), "--worker", "--split", args.split,
               "--revision", args.revision, "--assets", str(args.assets), "--output", str(args.output),
               "--custody", str(args.custody), "--pillow", str(dependency)]
    if args.preflight_only:
        command.append("--preflight-only")
    if args.synthetic_rehearsal:
        command.append("--synthetic-rehearsal")
    child_env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "TEMP": str(temporary), "TMP": str(temporary)}
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot")
        if not system_root:
            raise ValueError("Windows SystemRoot is unavailable")
        child_env.update(SystemRoot=system_root, WINDIR=system_root,
                         PATH=str(Path(system_root) / "System32"),
                         COMSPEC=str(Path(system_root) / "System32" / "cmd.exe"))
    result = run_cpu_study(command, study_root=args.custody, timeout=args.timeout, cwd=ROOT, env=child_env)
    data = dataclasses.asdict(result)
    data["scope"] = "cooperative CPU custody; direct child reaped; no process-tree or GPU attestation"
    data["status"] = "incomplete"
    data["failure_detail"] = ("owner setup failed before detailed evidence was created"
                              if not (args.output / "data").is_dir() else
                              "consult retained terminal, failure record and forward bytes")
    terminal_path = args.output / "data" / "terminal.json"
    if result.exit_code == 0 and result.clean and not result.timed_out and terminal_path.is_file():
        raw = terminal_path.read_bytes()
        terminal = json.loads(raw)
        if (terminal.get("status") == "complete" and terminal.get("source_revision") == args.revision
                and terminal.get("split") == args.split and terminal.get("case_sha256") == CASES_SHA
                and terminal.get("donor_sha256") == DONORS_SHA
                and terminal.get("checkpoint_preflight_only") == args.preflight_only
                and terminal.get("synthetic_rehearsal") == args.synthetic_rehearsal
                and not (args.output / "data" / "failure.json").exists()):
            recheck_snapshot(snapshot)
            data.update(status="complete", terminal_sha256=sha(raw))
            data.pop("failure_detail", None)
    write_json(args.output / "supervisor.json", data)
    print(json.dumps({"status": data["status"], "split": args.split, "owner_pid": result.pid,
                      "owner_exit": result.exit_code, "terminal_sha256": data.get("terminal_sha256")}))
    return 0 if data["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--custody", type=Path, required=True)
    parser.add_argument("--pillow", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--synthetic-rehearsal", action="store_true")
    args = parser.parse_args()
    return worker(args) if args.worker else controller(args)


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    verify_project_origin()
    raise SystemExit(main())
