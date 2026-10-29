"""Verify and unpack the saved public ViT CUDA tensor evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="a new or empty directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    index = json.loads((root / "evidence-index.json").read_text(encoding="utf-8"))
    if index["schema"] != "vision-vit-cuda-public-evidence-v1":
        raise ValueError("unexpected index schema")
    parts = []
    for i, item in enumerate(index["parts"], 1):
        name = f"evidence.part{i:02d}"
        if item["name"] != name:
            raise ValueError("unexpected part name")
        data = (root / name).read_bytes()
        if len(data) != item["bytes"] or sha(data) != item["sha256"]:
            raise ValueError(f"part integrity failure: {name}")
        parts.append(data)
    archive = b"".join(parts)
    if len(archive) != index["archive_bytes"] or sha(archive) != index["archive_sha256"]:
        raise ValueError("archive integrity failure")
    destination = args.destination
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("destination must be a new or empty directory")
    members = index["members"]
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or set(names) != set(members):
            raise ValueError("archive member inventory mismatch")
        for name in names:
            path = PurePosixPath(name)
            if len(path.parts) != 2 or path.parts[0] not in ("data", "reference") or any(
                part in ("", ".", "..") for part in path.parts
            ):
                raise ValueError(f"unsafe member: {name}")
            item = members[name]
            info = zipped.getinfo(name)
            if info.is_dir() or info.file_size != item["bytes"]:
                raise ValueError(f"member size failure: {name}")
            data = zipped.read(name)
            if len(data) != item["bytes"] or sha(data) != item["sha256"]:
                raise ValueError(f"member digest failure: {name}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(zipped.read(name))
    print(f"Verified {len(names)} evidence files in {destination}")


if __name__ == "__main__":
    main()
