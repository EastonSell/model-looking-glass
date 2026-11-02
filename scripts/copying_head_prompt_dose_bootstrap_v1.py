"""Stdlib-only source gate before importing the public matched-dose owner.

This is a cooperative same-process check, not an authenticated interpreter,
operating-system or process-tree attestation.
"""
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
MANIFEST = "experiments/distilgpt2_copy_head_prompt_dose_manifest_v1.json"
OWNER = "scripts/copying_head_prompt_dose_v1.py"
BOOTSTRAP = "scripts/copying_head_prompt_dose_bootstrap_v1.py"
REQUIRED = {
    MANIFEST,  # deliberately excluded from the source commit and added in the child
    BOOTSTRAP, OWNER, "scripts/build_copy_head_prompt_dose_cases_v1.py",
    "scripts/copying_head_factorial_v1.py", "scripts/copying_head_followup_v2.py",
    "scripts/copying_head_eval_v1.py", "scripts/copying_dev_v1.py",
    "scripts/distilgpt2_intervention_dev_bootstrap.py", "looking_glass/__init__.py",
    "looking_glass/pretrained.py", "looking_glass/copy_head_prompt_dose.py",
    "looking_glass/copy_head_factorial.py", "looking_glass/copy_head_intervention.py",
    "looking_glass/copy_head_followup.py", "looking_glass/pretrained_exposures.py",
    *(f"looking_glass/pretrained_exposures_v{i}.py" for i in range(2, 32)),
    "docs/DISTILGPT2_COPY_HEAD_PROMPT_DOSE_V1.md",
    "experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json",
    "experiments/natural_agreement_original_public_support_v1.json",
    "experiments/natural_agreement_successor_inputs_v1.json",
}


def git(*args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"})
    return subprocess.check_output(["git", *args], cwd=ROOT, env=env, timeout=60)


def startup_gate():
    prefix = Path(sys.pycache_prefix) if sys.pycache_prefix else None
    if (not sys.dont_write_bytecode or prefix is None or not prefix.is_absolute()
            or not prefix.is_dir() or any(prefix.iterdir())
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
            or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
            or any(name in os.environ for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY"))):
        raise RuntimeError("the CPU/offline/fresh-bytecode startup conditions are not met")
    if any(name == "torch" or name.startswith("torch.") or name == "looking_glass"
           or name.startswith("looking_glass.") or name == "scripts" or name.startswith("scripts.")
           for name in sys.modules):
        raise RuntimeError("a project or torch module was already imported")


def verify_manifest(path):
    if not path.is_absolute() or path.resolve() != (ROOT / MANIFEST).resolve():
        raise RuntimeError("manifest must be the fixed published checkout file")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != "distilgpt2-copy-head-prompt-dose-manifest-v1":
        raise RuntimeError("unexpected manifest schema")
    sources = manifest.get("source_sha256")
    if type(sources) is not dict or not (REQUIRED - {MANIFEST}).issubset(sources):
        raise RuntimeError("manifest omits a required pre-import source")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("checkout is not clean")
    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    source_revision = manifest.get("source_revision")
    if type(source_revision) is not str or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise RuntimeError("invalid source revision")
    git("merge-base", "--is-ancestor", source_revision, revision)
    changes = git("diff", "--name-only", source_revision, revision, "--").decode().splitlines()
    if changes != [MANIFEST] or git("show", f"{revision}:{MANIFEST}") != raw:
        raise RuntimeError("manifest is not the sole committed source-to-HEAD change")
    if git("show", "-s", "--format=%T", source_revision).decode().strip() != manifest.get("source_tree"):
        raise RuntimeError("source tree differs")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true",
                        help="check the source/manifest without importing project or model code")
    args = parser.parse_args()
    manifest_hash = verify_manifest(args.manifest)
    if args.verify_only:
        print(json.dumps({"status": "source-manifest-verified", "manifest_sha256": manifest_hash,
                          "model_forward_asserted": False}))
        return
    os.environ["COPY_HEAD_PROMPT_BOOTSTRAP_SHA256"] = manifest_hash
    runpy.run_path(str(ROOT / OWNER), run_name="__main__")


if __name__ == "__main__":
    main()
