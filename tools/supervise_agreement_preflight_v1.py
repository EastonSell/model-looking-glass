"""Record an independent process status for the public CPU preflight."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--pycache-prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prefix = Path(args.pycache_prefix)
    if not prefix.is_absolute() or os.path.lexists(prefix):
        raise ValueError("pycache prefix must be an absolute nonexistent path")
    out = Path(args.output).resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError("supervisor output must be outside the checkout")
    out.mkdir(parents=False, exist_ok=False)
    command = [sys.executable, "-B", "-X", f"pycache_prefix={prefix}",
               str(ROOT / "tools" / "run_agreement_preflight_v1.py"), "run",
               "--cache", args.cache, "--manifest", args.manifest,
               "--inventory-module", "pretrained_exposures_v2",
               "--inventory-sha256", args.inventory_sha256, "--output", str(out / "run")]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["OMP_NUM_THREADS"] = "1"
    started = datetime.now(timezone.utc).isoformat()
    returncode = None
    launch_error = None
    with (out / "stdout.txt").open("xb") as stdout, (out / "stderr.txt").open("xb") as stderr:
        try:
            returncode = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                        check=False).returncode
        except OSError as error:
            launch_error = type(error).__name__
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    record = {"status": "completed" if returncode == 0 else "failed",
              "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "cuda_visible_devices": "-1", "omp_num_threads": "1",
              "exit_code": returncode, "launch_error": launch_error,
              "stdout_sha256": hashlib.sha256((out / "stdout.txt").read_bytes()).hexdigest(),
              "stderr_sha256": hashlib.sha256((out / "stderr.txt").read_bytes()).hexdigest()}
    raw = json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with (out / "supervisor.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if returncode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
