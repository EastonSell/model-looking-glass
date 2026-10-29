"""Create a public projection of a closed grouped result from private bytes.

The original result is checked by SHA-256 and is not modified. Random run and
broker job identifiers are withheld; the four verified job suffixes become
ordered public slots. Private evidence referenced by hash remains withheld.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def reject_private(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"run_id", "job_id", "pid", "physical_gpu_uuid", "supervisor_run_id"}:
                raise ValueError("a private identity field remains in the projection")
            reject_private(item)
    elif isinstance(value, list):
        for item in value:
            reject_private(item)
    elif isinstance(value, str) and re.search(
        r"GPU-[0-9a-fA-F-]{36}|(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/[^/]+|/home/[^/]+", value
    ):
        raise ValueError("a private identity or absolute path remains in the projection")


def project(raw, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("closed result SHA-256 does not match")
    result = json.loads(raw)
    run_id = result.pop("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("unexpected private run identity")
    runtime = result["measurement"]["runtime_redacted"]
    if runtime.pop("physical_gpu_uuid") != "redacted":
        raise ValueError("device identity was not already redacted by the controller")
    cases = [case for group in result["measurement"]["holdout"] for case in group["comparisons"]]
    if len(cases) != 4:
        raise ValueError("the fixed protocol has four broker deliveries")
    prefixes = set()
    ids = set()
    for index, case in enumerate(cases):
        job = case.pop("job_id")
        match = re.fullmatch(r"([0-9a-f]{32})-j-([0-3])", job) if isinstance(job, str) else None
        if match is None or int(match.group(2)) != index or job in ids:
            raise ValueError("broker job identities do not match the fixed delivery order")
        prefixes.add(match.group(1))
        ids.add(job)
        case["job_slot"] = index
    if len(prefixes) != 1:
        raise ValueError("broker deliveries do not share the recorded job namespace")
    reject_private(result)
    return {"schema": "grouped-gpu-public-projection-v1", "original_result_sha256": expected,
            "transformations": ["removed private run identity", "replaced broker job identities with verified slots",
                                "removed the already-redacted physical-device identity field"],
            "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closed_result")
    parser.add_argument("original_sha256")
    args = parser.parse_args()
    result = project(Path(args.closed_result).read_bytes(), args.original_sha256)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
