"""Audit one public pair/dose run against independently supplied pins."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looking_glass.copy_pairs_dose_audit import _json_object  # noqa: E402
from looking_glass.copy_pairs_dose_evidence import _read_regular, audit_directory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--pins", required=True, help="independently supplied source and evidence identities")
    parser.add_argument("--output", required=True, help="new report file outside the attempt")
    args = parser.parse_args()
    attempt, output = Path(args.attempt), Path(args.output).resolve(strict=False)
    resolved_attempt = attempt.resolve(strict=True)
    if output == resolved_attempt or output.is_relative_to(resolved_attempt):
        raise ValueError("report must be outside the saved attempt")
    pins = _json_object(_read_regular(args.pins, 2 * 1024 * 1024))
    report = audit_directory(attempt, args.cases, expected=pins)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("report exceeds its metadata bound")
    with output.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("report write was incomplete")
    print("Saved evidence is internally consistent; execution and provenance are not authenticated.")


if __name__ == "__main__":
    main()
