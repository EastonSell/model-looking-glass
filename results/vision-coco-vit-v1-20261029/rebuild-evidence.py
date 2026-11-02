"""Reassemble the published lossless vision archive after verifying each part."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new archive path; existing files are never overwritten")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "archive-parts.json").read_bytes())
    parts = manifest["parts"]
    if manifest["schema"] != "vision-coco-archive-parts-v1" or not 1 <= len(parts) <= 100:
        raise ValueError("unexpected archive parts manifest")
    total = hashlib.sha256()
    size = 0
    with args.output.open("xb") as out:
        for i, part in enumerate(parts):
            name = f"raw-evidence.tar.xz.part-{i:03d}"
            if part["name"] != name or type(part["bytes"]) is not int or not 0 < part["bytes"] <= 850 * 1024:
                raise ValueError("unexpected archive part identity or size")
            raw = (root / name).read_bytes()
            if len(raw) != part["bytes"] or hashlib.sha256(raw).hexdigest() != part["sha256"]:
                raise ValueError("archive part hash differs")
            if out.write(raw) != len(raw):
                raise OSError("short archive write")
            total.update(raw)
            size += len(raw)
    if size != manifest["archive_bytes"] or total.hexdigest() != manifest["archive_sha256"]:
        raise ValueError("reconstructed archive differs")
    print(json.dumps({"status": "verified", "bytes": size, "sha256": total.hexdigest()}))


if __name__ == "__main__":
    main()
