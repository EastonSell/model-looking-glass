"""Stdlib-only gate for the fixed copying intervention before model imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "experiments/distilgpt2_copy_behavior_intervention_manifest_v1.json"
OWNER = "scripts/copy_behavior_intervention_v1.py"
BOOTSTRAP = "scripts/copy_behavior_intervention_bootstrap_v1.py"
REQUIRED = {BOOTSTRAP, OWNER, "scripts/copy_behavior_screen_v1.py",
            "scripts/copy_behavior_screen_bootstrap_v1.py", "scripts/copying_head_output_dose_v1.py",
            "looking_glass/copy_behavior_output_dose_v1.py", "looking_glass/copy_behavior_effects_v1.py", "looking_glass/copy_behavior_screen_v1.py",
            "docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_V1.md",
            "docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_RESOURCE_AMENDMENT_V1.md",
            "experiments/distilgpt2_copy_behavior_screen_cases_v1.json", "scripts/build_copy_output_dose_cases_v1.py",
            "scripts/build_copy_head_prompt_dose_cases_v1.py",
            "scripts/copying_head_prompt_dose_v1.py", "scripts/copying_head_factorial_v1.py",
            "scripts/copying_head_followup_v2.py", "scripts/copying_head_eval_v1.py",
            "scripts/copying_dev_v1.py", "scripts/distilgpt2_intervention_dev_bootstrap.py",
            "looking_glass/__init__.py", "looking_glass/pretrained.py",
            "looking_glass/copy_head_output_dose.py", "looking_glass/copy_head_prompt_dose.py",
            "looking_glass/copy_head_factorial.py", "looking_glass/copy_head_intervention.py",
            "looking_glass/copy_head_followup.py", "looking_glass/pretrained_exposures.py",
            *(f"looking_glass/pretrained_exposures_v{i}.py" for i in range(2, 42)),
            "docs/DISTILGPT2_COPY_OUTPUT_DOSE_V1.md", "experiments/distilgpt2_copy_output_dose_cases_v1.json",
            "docs/DISTILGPT2_COPY_HEAD_PROMPT_DOSE_V1.md", "experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json",
            "experiments/natural_agreement_original_public_support_v1.json",
            "experiments/natural_agreement_successor_inputs_v1.json"}


def git(*args):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"})
    return subprocess.check_output(["git", *args], cwd=ROOT, env=env, timeout=60)


def startup_gate():
    prefix = Path(sys.pycache_prefix) if sys.pycache_prefix else None
    if (not sys.dont_write_bytecode or prefix is None or not prefix.is_absolute()
            or not prefix.is_dir() or any(prefix.iterdir()) or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
            or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
            or any(name in os.environ for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY"))):
        raise RuntimeError("CPU/offline/fresh-bytecode startup conditions are not met")
    if any(name == "torch" or name.startswith("torch.") or name == "looking_glass"
           or name.startswith("looking_glass.") or name == "scripts" or name.startswith("scripts.")
           for name in sys.modules):
        raise RuntimeError("a project or torch module was already imported")


def verify_manifest(path):
    if not path.is_absolute() or path.resolve() != (ROOT / MANIFEST).resolve():
        raise RuntimeError("manifest must be the fixed published checkout file")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    sources = manifest.get("source_sha256")
    if (manifest.get("schema") != "distilgpt2-copy-behavior-intervention-manifest-v1"
            or type(sources) is not dict or not REQUIRED.issubset(sources)):
        raise RuntimeError("manifest schema or required pre-import source differs")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("checkout is not clean")
    revision = git("rev-parse", "HEAD").decode().strip()
    source_revision = manifest.get("source_revision")
    if type(source_revision) is not str or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise RuntimeError("invalid source revision")
    git("merge-base", "--is-ancestor", source_revision, revision)
    if (git("diff", "--name-only", source_revision, revision, "--").decode().splitlines() != [MANIFEST]
            or git("show", f"{revision}:{MANIFEST}") != raw
            or git("show", "-s", "--format=%T", source_revision).decode().strip() != manifest.get("source_tree")):
        raise RuntimeError("manifest or source tree binding differs")
    for name, expected in sources.items():
        if (type(name) is not str or not name or name.startswith("/") or "\\" in name
                or any(part in ("", ".", "..") for part in name.split("/"))
                or type(expected) is not str or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise RuntimeError("invalid source map entry")
        content = (ROOT / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected or git("show", f"{source_revision}:{name}") != content:
            raise RuntimeError(f"source differs before import: {name}")
    return hashlib.sha256(raw).hexdigest()


def main():
    startup_gate()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    digest = verify_manifest(args.manifest)
    if args.verify_only:
        print(json.dumps({"status": "source-manifest-verified", "manifest_sha256": digest,
                          "model_forward_asserted": False}))
        return
    os.environ["COPY_BEHAVIOR_INTERVENTION_BOOTSTRAP_SHA256"] = digest
    runpy.run_path(str(ROOT / OWNER), run_name="__main__")


if __name__ == "__main__":
    main()
