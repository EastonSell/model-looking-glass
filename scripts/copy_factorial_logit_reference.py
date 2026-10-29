"""Reconstruct the public factorial's pre-forward additive-logit reference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looking_glass.copy_factorial_logit_reference import analyze_directory  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--source-revision", required=True, help="published runner revision observed before launch")
    args = parser.parse_args()
    expected = {
        "source_revision": args.source_revision,
        "case_sha256": "69f8463da5f919fb99aef05aa4401de439248100e99ca90899f8d9f98af90d24",
        "protocol_sha256": "467b566fd149bc7e489c0be265c01b63c3f005804bf30c7873560df930aa9d92",
        "clarification_sha256": "c5d2ba27ded504f2bc637f51110b48a2825fc1d2d4a3b2dae80bb596acc3037f",
        "clarification_revision": "7a5a7d6eaea4c82f02f7f4e05b6e5d41c85ecb49",
        "v11": "84b1a95a41670798ff55c9f08272890c00a939a74a4cad7ad9b95fbf4bcdaa25",
        "v12": "af374aaceb708b3b38e06b3955ca975467b048162ee105cbfcc9452a8a600e49",
    }
    report = analyze_directory(args.attempt, args.case_file, expected=expected,
                               vocabulary_size=50257, expected_total_bytes=61250432)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
