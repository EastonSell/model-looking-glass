"""Supervise the frozen CPU-only ViT/COCO same-site study.

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
    "tools/run_vision_coco_same_site.py", "tools/prepare_vision_coco_same_site.py",
    "looking_glass/__init__.py", "looking_glass/vision_coco_cases.py",
    "looking_glass/vision_coco_same_site_cases.py", "looking_glass/vision_coco_small_dose_cases.py",
    "looking_glass/vision_coco_same_site_study.py", "looking_glass/vision_coco_study.py",
    "looking_glass/intervention_plan.py", "looking_glass/interventions.py",
    "looking_glass/model_ownership.py", "looking_glass/cpu_study_supervision.py",
    "looking_glass/supervisor.py", "looking_glass/device_lease.py", "looking_glass/device_identity.py",
    "experiments/vision_coco_spatial_v1.json", "experiments/vision_coco_same_site_v3.json",
    "docs/VISION_COCO_SPATIAL_V1.md", "docs/VISION_COCO_SMALL_DOSE_V2.md",
    "docs/VISION_COCO_SAME_SITE_SMALL_DOSE_V3.md", "docs/VISION_COCO_SAME_SITE_RUNNER_V3.md",
    "docs/CPU_STUDY_SUPERVISION.md",
)
CASES_SHA = "8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207"
MANIFEST_SHA = "6fc5159a57cf07a91e93c8bc38f27d3f5925877083532ce92fec828c6d777bfb"
SELECTION_REVISION = "20ba2249f8e385d1f81048ef27484c012afae93f"
MODEL_STATE_SHA = "d1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd"


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
    return subprocess.check_output(["git", "--no-replace-objects", "-c", "core.fsmonitor=false", *args], cwd=ROOT, env=env)


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
            or items["experiments/vision_coco_same_site_v3.json"] != MANIFEST_SHA):
        raise ValueError("frozen case bytes differ")
    git(["merge-base", "--is-ancestor", SELECTION_REVISION, revision])
    return {"revision": revision, "sources": items,
            "scope": "trusted quiescent checkout; hashes do not attest already executing code or dependencies"}


def recheck_snapshot(snapshot):
    if set(snapshot["sources"]) != set(SOURCES):
        raise ValueError("source inventory differs")
    for name, expected in snapshot["sources"].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"source changed during attempt: {name}")


def verify_saved_schedule(data_dir, terminal, pairs, source_snapshot, *, synthetic=False, preflight=False):
    """Check saved structure and schedule; numerical replay remains separate."""
    if (data_dir / "failure.json").exists():
        raise ValueError("failure evidence exists")
    started = json.loads((data_dir / "started.json").read_bytes())
    if (started.get("schema") != "vision-coco-same-site-start-v3"
            or json.dumps(started.get("source"), sort_keys=True, allow_nan=False) !=
               json.dumps(source_snapshot, sort_keys=True, allow_nan=False)
            or source_snapshot.get("revision") != terminal.get("source_revision")
            or started.get("manifest_sha256") != MANIFEST_SHA
            or started.get("checkpoint_preflight_only") is not preflight
            or started.get("synthetic_rehearsal") is not synthetic):
        raise ValueError("start record identity differs")
    if not synthetic and (started.get("split") != "heldout" or
                          started.get("model_state_sha256") != MODEL_STATE_SHA or
                          terminal.get("model_state_sha256_before_after") != MODEL_STATE_SHA):
        raise ValueError("checkpoint state binding differs")
    if not synthetic:
        runtime = started.get("runtime")
        fields = {"device", "cuda_visible_devices", "cuda_initialized", "python", "torch", "transformers",
                  "numpy", "pillow", "threads", "interop_threads", "deterministic_algorithms",
                  "float32_matmul_precision", "mkldnn_enabled", "processor", "denormals_preserved"}
        if (type(runtime) is not dict or set(runtime) != fields or
                json.dumps(runtime, sort_keys=True, allow_nan=False) !=
                json.dumps(terminal.get("runtime"), sort_keys=True, allow_nan=False) or
                type(runtime.get("threads")) is not int or runtime["threads"] != 1 or
                type(runtime.get("interop_threads")) is not int or runtime["interop_threads"] != 1 or
                runtime.get("device") != "cpu" or runtime.get("cuda_visible_devices") != "" or
                runtime.get("cuda_initialized") is not False or
                runtime.get("deterministic_algorithms") is not True or runtime.get("denormals_preserved") is not True or
                runtime.get("processor") != "ViTImageProcessor" or
                runtime.get("float32_matmul_precision") != "highest" or runtime.get("mkldnn_enabled") is not True or
                runtime.get("python") != "3.11.15" or runtime.get("torch") != "2.8.0+cu128" or
                runtime.get("transformers") != "4.57.3" or runtime.get("numpy") != "2.4.6" or
                runtime.get("pillow") != "11.3.0"):
            raise ValueError("runtime identity or settings differ")
    rows = terminal.get("pairs")
    if type(rows) is not list:
        raise ValueError("pair results missing")
    if preflight:
        if (rows or type(terminal.get("submissions")) is not int or terminal["submissions"] != 0 or
                type(terminal.get("saved_forwards")) is not int or terminal["saved_forwards"] != 0 or
                terminal.get("aggregate") is not None):
            raise ValueError("preflight performed a forward")
        if list(data_dir.glob("forward-*")):
            raise ValueError("preflight has forward evidence")
        return
    if len(rows) != len(pairs) or [r.get("pair_id") for r in rows] != [p["pair_id"] for p in pairs]:
        raise ValueError("pair result order differs")
    if not synthetic and (len(rows) != 8 or [r["pair_id"] for r in rows] != [f"pair-{i:02d}" for i in range(8)]):
        raise ValueError("the complete fixed evaluation set is required")
    if synthetic:
        if terminal.get("aggregate") is not None:
            raise ValueError("synthetic rehearsal cannot claim an evaluation aggregate")
    else:
        evaluable = [r for r in rows if r.get("status") in ("positive", "negative")]
        positive = sum(r.get("status") == "positive" for r in evaluable)
        decision = ("inconclusive" if len(evaluable) < 4 else
                    "positive" if positive * 4 >= len(evaluable) * 3 else "negative")
        expected_aggregate = {"status": decision, "selected": 8, "evaluable": len(evaluable), "positive": positive}
        if json.dumps(terminal.get("aggregate"), sort_keys=True, allow_nan=False) != json.dumps(expected_aggregate, sort_keys=True):
            raise ValueError("aggregate differs from pair statuses")
    labels, seen = [], set()
    for pair in pairs:
        for role in ("cat", "dog", "cat_donor", "dog_donor"):
            image_id = pair[role]["image_id"]
            if image_id not in seen:
                labels.append(f"{pair['pair_id']}:{role}:initial")
                seen.add(image_id)
    for pair, result in zip(pairs, rows):
        if result.get("split") != "heldout" or result.get("status") not in (
                "positive", "negative", "baseline-ineligible", "dose-ineligible", "dose-unmatchable"):
            raise ValueError("invalid pair status")
        if json.loads((data_dir / f"{pair['pair_id']}.json").read_bytes()) != result:
            raise ValueError("saved pair result differs")
        for side in ("cat", "dog"):
            info = result.get("sides", {}).get(side, {})
            if (type(info.get("image_id")) is not int or type(info.get("same_species_donor_id")) is not int or
                    info.get("image_id") != pair[side]["image_id"] or
                    info.get("same_species_donor_id") != pair[side + "_donor"]["image_id"]):
                raise ValueError("saved target or donor identity differs")
            dose = info.get("dose", {}).get("status")
            if dose not in ("matchable", "dose-ineligible", "dose-unmatchable"):
                raise ValueError("invalid saved dose status")
            names = ["repeat", "null_inside", "self_inside", "full_self", "full_swap", "natural_inside"]
            if dose == "matchable":
                names += ["matched_opposite", "matched_same_species", "matched_random"]
            names += ["final_inside", "post_clean"]
            if set(info.get("conditions", {})) != set(names):
                raise ValueError("saved conditions differ from dose status")
            labels.extend(f"{pair['pair_id']}:{side}:{name}" for name in names)
    count = len(labels)
    if (count > 256 or type(terminal.get("submissions")) is not int or terminal["submissions"] != count or
            type(terminal.get("saved_forwards")) is not int or terminal["saved_forwards"] != count):
        raise ValueError("forward count differs")
    expected_files = {f"forward-{i:03d}.{ext}" for i in range(1, count + 1) for ext in ("json", "bin")}
    if {p.name for p in data_dir.glob("forward-*")} != expected_files:
        raise ValueError("forward inventory differs")
    for index, label in enumerate(labels, 1):
        row = json.loads((data_dir / f"forward-{index:03d}.json").read_bytes())
        payload = (data_dir / f"forward-{index:03d}.bin").read_bytes()
        if (type(row.get("submission")) is not int or row["submission"] != index or row.get("label") != label or
                row.get("file") != f"forward-{index:03d}.bin" or row.get("sha256") != sha(payload) or
                not 0 < len(payload) <= 1_000_000 or row.get("dtype") != "little-endian-float32" or
                not isinstance(row.get("score"), dict) or
                not isinstance(row.get("metadata", {}).get("steps"), list) or not row["metadata"]["steps"]):
            raise ValueError("saved forward record differs")
        layout = row.get("layout")
        if type(layout) is not dict or "logits" not in layout:
            raise ValueError("saved forward layout is missing")
        if layout["logits"].get("shape") != [1, 1000] or layout["logits"].get("bytes") != 4000:
            raise ValueError("saved logit geometry differs")
        if label.endswith((":initial", ":repeat", ":post_clean")):
            if set(layout) != {"logits", "early", "late"}:
                raise ValueError("clean captures missing")
        else:
            if not {"logits", "before_all", "after_all"}.issubset(layout):
                raise ValueError("action witnesses missing")
            if not label.endswith(":null_inside") and "requested" not in layout:
                raise ValueError("requested action missing")
            if not label.endswith((":full_self", ":full_swap")):
                if not all(f"{prefix}_{i}" in layout for i in range(4) for prefix in ("before", "after")):
                    raise ValueError("token witnesses missing")
        cursor = 0
        for component in sorted(layout.values(), key=lambda value: value.get("offset", -1)):
            shape, nbytes = component.get("shape"), component.get("bytes")
            if (type(shape) is not list or not shape or
                    any(type(n) is not int or n <= 0 for n in shape) or
                    type(nbytes) is not int or nbytes != 4 * __import__("math").prod(shape) or
                    component.get("offset") != cursor):
                raise ValueError("invalid tensor byte layout")
            cursor += nbytes
        if cursor != len(payload):
            raise ValueError("tensor byte layout differs from payload")


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
        from looking_glass.vision_coco_same_site_cases import check_reused_images, select_same_site
        from looking_glass.vision_coco_study import Evidence
        from looking_glass.vision_coco_same_site_study import aggregate_same_site, run_same_site

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
                or torch.__version__ != "2.8.0+cu128" or
                torch.get_float32_matmul_precision() != "highest" or torch.backends.mkldnn.enabled is not True):
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
            fabricated = {i: torch.randn(1, 197, 192) for i in range(1, 5)}
            pair = {"pair_id": "pair-00", "split": "heldout", "cat": {"image_id": 1},
                    "dog": {"image_id": 2}, "cat_donor": {"image_id": 3}, "dog_donor": {"image_id": 4},
                    "box_tokens": [1, 2, 15, 16]}
            evidence = Evidence(args.output / "data", 256)
            evidence.json("started.json", {"schema": "vision-coco-same-site-start-v3",
                          "source": snapshot, "manifest_sha256": MANIFEST_SHA,
                          "checkpoint_preflight_only": False, "synthetic_rehearsal": True})
            rows = run_same_site(model, [pair], fabricated, evidence)
            evidence.json("terminal.json", {"schema": "vision-coco-same-site-terminal-v3",
                          "status": "complete", "source_revision": snapshot["revision"],
                          "manifest_sha256": MANIFEST_SHA, "case_sha256": MANIFEST_SHA, "checkpoint_preflight_only": False,
                          "synthetic_rehearsal": True, "submissions": evidence.submissions,
                          "saved_forwards": len(evidence.records), "pairs": rows, "aggregate": None})
            model = None
            gc.collect()
            session.check(lease)
            session.attest_clean(lease)
            return 0
        case_bytes = (ROOT / "experiments/vision_coco_same_site_v3.json").read_bytes()
        prior_bytes = (ROOT / "experiments/vision_coco_spatial_v1.json").read_bytes()
        if sha(case_bytes) != MANIFEST_SHA or sha(prior_bytes) != CASES_SHA:
            raise ValueError("case or prior manifest changed")
        manifest, prior = json.loads(case_bytes), json.loads(prior_bytes)
        if (manifest.get("schema") != "vision-coco-vit-same-site-cases-v3"
                or manifest.get("selection_source", {}).get("revision") != SELECTION_REVISION):
            raise ValueError("selection identity differs")
        for name, digest in manifest["selection_source"]["source_sha256"].items():
            if snapshot["sources"].get(name) != digest or sha((ROOT / name).read_bytes()) != digest:
                raise ValueError("selection source changed")
        annotation_bytes = (args.assets / "instances_val2017.json").read_bytes()
        expected_pairs, expected_diagnostics = select_same_site(annotation_bytes, prior_bytes)
        comparable = json.loads(json.dumps(manifest["pairs"], allow_nan=False))
        for pair in comparable:
            for role in ("cat", "dog", "cat_donor", "dog_donor"):
                for key in ("mirror_url", "jpeg_bytes", "jpeg_sha256", "pixel_sha256"):
                    pair[role].pop(key)
        if comparable != expected_pairs or manifest["selection_diagnostics"] != expected_diagnostics:
            raise ValueError("annotation-only selection does not reproduce")
        pairs = manifest["pairs"]
        if [p["pair_id"] for p in pairs] != [f"pair-{i:02d}" for i in range(8)] or any(
                p["split"] != "heldout" for p in pairs):
            raise ValueError("evaluation case identity differs")
        references = [p[role] for p in pairs for role in ("cat", "dog", "cat_donor", "dog_donor")]
        target_ids = [p[side]["image_id"] for p in pairs for side in ("cat", "dog")]
        donor_ids = [p[side + "_donor"]["image_id"] for p in pairs for side in ("cat", "dog")]
        prior_images = [p[side] for p in prior["pairs"] for side in ("cat", "dog")]
        check_reused_images(prior_images, references, target_ids, donor_ids)
        verify_assets(args.assets)
        processor = AutoImageProcessor.from_pretrained(str(args.assets), local_files_only=True, use_fast=False)
        if (type(processor).__name__ != "ViTImageProcessor" or processor.size != {"height": 224, "width": 224}
                or int(processor.resample) != 2 or processor.image_mean != [.5] * 3
                or processor.image_std != [.5] * 3 or processor.rescale_factor != 1 / 255):
            raise ValueError("processor differs")
        pixels = {}
        for item in references:
            if item["image_id"] in pixels:
                continue
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
        if before != MODEL_STATE_SHA:
            raise ValueError("loaded checkpoint state differs from the frozen model")
        runtime = {"device": "cpu", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                   "cuda_initialized": torch.cuda.is_initialized(), "python": sys.version.split()[0],
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   "numpy": np.__version__, "pillow": PIL.__version__,
                   "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "mkldnn_enabled": torch.backends.mkldnn.enabled,
                   "processor": type(processor).__name__, "denormals_preserved": True}
        evidence = Evidence(args.output / "data", 256, labels=config.id2label)
        evidence.json("started.json", {"schema": "vision-coco-same-site-start-v3", "split": "heldout",
                                       "source": snapshot, "runtime": runtime, "model_state_sha256": before,
                                       "manifest_sha256": MANIFEST_SHA,
                                       "checkpoint_preflight_only": args.preflight_only,
                                       "synthetic_rehearsal": False})
        if args.preflight_only:
            rows, decision = [], None
        else:
            rows = run_same_site(model, pairs, pixels, evidence)
            decision = aggregate_same_site(rows)
        after = state_digest()
        if (after != before or hooks() != initial_hooks or torch.cuda.is_initialized() or
                torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or
                not torch.are_deterministic_algorithms_enabled() or torch.get_default_dtype() != torch.float32 or
                torch.get_float32_matmul_precision() != runtime["float32_matmul_precision"] or
                torch.backends.mkldnn.enabled is not runtime["mkldnn_enabled"] or
                torch.tensor([1.401298464324817e-45], dtype=torch.float32).item() == 0):
            raise ValueError("model state, hooks or CPU runtime changed")
        recheck_snapshot(snapshot)
        terminal = {"schema": "vision-coco-same-site-terminal-v3", "status": "complete",
                    "split": "heldout", "source_revision": snapshot["revision"],
                    "manifest_sha256": MANIFEST_SHA, "case_sha256": MANIFEST_SHA,
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
    if (args.output.resolve().is_relative_to(args.assets.resolve()) or
            args.output.resolve().is_relative_to(args.custody.resolve()) or
            args.custody.resolve().is_relative_to(args.output.resolve())):
        raise ValueError("output must be separate from assets and custody")
    if args.synthetic_rehearsal and args.preflight_only:
        raise ValueError("synthetic rehearsal and checkpoint preflight are distinct")
    snapshot = source_snapshot(args.revision)
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
               str(dependency), str(Path(__file__).resolve()), "--worker",
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
    if (type(result.exit_code) is int and result.exit_code == 0 and result.clean is True and
            result.timed_out is False and terminal_path.is_file()):
        try:
            raw = terminal_path.read_bytes()
            terminal = json.loads(raw)
            if (terminal.get("schema") != "vision-coco-same-site-terminal-v3"
                    or terminal.get("status") != "complete" or terminal.get("source_revision") != args.revision
                    or terminal.get("manifest_sha256") != MANIFEST_SHA or terminal.get("case_sha256") != MANIFEST_SHA
                    or terminal.get("checkpoint_preflight_only") is not args.preflight_only
                    or terminal.get("synthetic_rehearsal") is not args.synthetic_rehearsal):
                raise ValueError("terminal identity differs")
            if args.synthetic_rehearsal:
                pairs = [{"pair_id": "pair-00", "cat": {"image_id": 1}, "dog": {"image_id": 2},
                          "cat_donor": {"image_id": 3}, "dog_donor": {"image_id": 4}}]
            else:
                pairs = json.loads((ROOT / "experiments/vision_coco_same_site_v3.json").read_bytes())["pairs"]
            verify_saved_schedule(args.output / "data", terminal, pairs, snapshot,
                                  synthetic=args.synthetic_rehearsal, preflight=args.preflight_only)
            recheck_snapshot(snapshot)
            data.update(status="complete", terminal_sha256=sha(raw))
            data.pop("failure_detail", None)
        except Exception as error:
            data["failure_detail"] = "saved evidence validation failed: " + type(error).__name__
    write_json(args.output / "supervisor.json", data)
    print(json.dumps({"status": data["status"], "split": "heldout", "owner_pid": result.pid,
                      "owner_exit": result.exit_code, "terminal_sha256": data.get("terminal_sha256")}))
    return 0 if data["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--custody", type=Path, required=True)
    parser.add_argument("--pillow", type=Path, required=True)
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
