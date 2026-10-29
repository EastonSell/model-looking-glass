"""Freeze tokenizer rows or run the committed public agreement preflight."""

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if (not sys.dont_write_bytecode or not sys.pycache_prefix
        or not Path(sys.pycache_prefix).is_absolute() or os.path.lexists(sys.pycache_prefix)):
    raise RuntimeError("use -B and -X pycache_prefix=<absolute nonexistent path>")
sys.path.insert(0, str(ROOT))

from looking_glass import agreement_preflight_v1 as study  # noqa: E402


def _inventory(module_name, expected_digest):
    if not re.fullmatch(r"pretrained_exposures_v[0-9]+", module_name):
        raise ValueError("inventory module must be a versioned pretrained exposure module")
    relative = f"looking_glass/{module_name}.py"
    path = (ROOT / relative).resolve(strict=True)
    if ROOT not in path.parents:
        raise ValueError("inventory source escapes the checkout")
    raw = path.read_bytes()
    committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"], timeout=30)
    if raw != committed:
        raise ValueError("inventory source must match the committed checkout")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", relative],
                                       timeout=30).decode("ascii").strip()
    module = importlib.import_module(f"looking_glass.{module_name}")
    digest = module.inventory_digest()
    if digest != expected_digest:
        raise ValueError("inventory digest does not match the approved snapshot")
    return module.inventory_payload(), digest, revision, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--cache", required=True)
    freeze.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--cache", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--inventory-module", required=True)
    run.add_argument("--inventory-sha256", required=True)
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        _, digest = study.freeze(ROOT, args.cache, args.output)
        print(f"frozen_manifest_sha256={digest}")
    else:
        inventory = _inventory(args.inventory_module, args.inventory_sha256)
        result = study.run(ROOT, args.cache, args.manifest, *inventory, args.output)
        print(f"completed_rows={len(result['rows'])} informative={result['summary']['informative']}")


if __name__ == "__main__":
    main()
