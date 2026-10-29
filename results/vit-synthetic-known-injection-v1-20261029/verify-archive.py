"""Verify the published split archive, optionally extracting it to a new directory."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--extract", type=Path, help="new output directory; omit for verification only")
args = parser.parse_args()
root = Path(__file__).resolve().parent
parts = json.loads((root / "parts-index.json").read_bytes())
archive = json.loads((root / "archive-index.json").read_bytes())
limit = 32 * 1024 * 1024
if (parts.get("schema") != "vit-synthetic-archive-parts-v1" or archive.get("schema") != "vit-synthetic-archive-v1"
        or type(parts.get("parts")) is not list or not 1 <= len(parts["parts"]) <= 64
        or type(parts.get("archive_bytes")) is not int or not 0 < parts["archive_bytes"] <= limit):
    raise ValueError("unsupported archive index")
pieces = []
offset = 0
for i, item in enumerate(parts["parts"]):
    expected = f"parts/attempt-{i:03d}.bin"
    if (item["path"] != expected or item["offset"] != offset or type(item["bytes"]) is not int
            or not 0 < item["bytes"] <= 960000 or offset + item["bytes"] > limit):
        raise ValueError("part order or offset differs")
    raw = (root / expected).read_bytes()
    git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"] or git_blob != item["git_blob_sha1"]:
        raise ValueError("part bytes differ")
    pieces.append(raw)
    offset += len(raw)
joined = b"".join(pieces)
if (len(joined) != parts["archive_bytes"] or hashlib.sha256(joined).hexdigest() != parts["archive_sha256"]
        or parts["archive_sha256"] != archive["archive_sha256"] or len(joined) != archive["archive_bytes"]):
    raise ValueError("assembled archive differs")
members = archive["members"]
if (type(members) is not list or not 1 <= len(members) <= 64
        or any(type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= limit for row in members)
        or sum(row["bytes"] for row in members) > limit):
    raise ValueError("unsupported member size or count")
with zipfile.ZipFile(io.BytesIO(joined)) as source:
    names = source.namelist()
    if len(names) != len(set(names)) or sorted(names) != sorted(row["path"] for row in members):
        raise ValueError("archive member inventory differs")
    for row in members:
        name = PurePosixPath(row["path"])
        if (name.is_absolute() or len(name.parts) != 2 or name.parts[0] != "attempt"
                or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", name.parts[1])
                or name.parts[1].endswith(".")
                or name.parts[1].split(".")[0].upper() in
                   {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}):
            raise ValueError("invalid archive member path")
        info = source.getinfo(row["path"])
        if info.is_dir() or info.file_size != row["bytes"] or info.compress_size > limit:
            raise ValueError("archive member declared size differs")
        with source.open(info) as member:
            raw = member.read(row["bytes"] + 1)
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("archive member bytes differ")
    if args.extract is not None:
        args.extract.mkdir(parents=False, exist_ok=False)
        for row in members:
            destination = args.extract / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                with source.open(row["path"]) as member:
                    raw = member.read(row["bytes"] + 1)
                if len(raw) != row["bytes"] or stream.write(raw) != len(raw):
                    raise OSError("short extracted member")
print(json.dumps({"status": "verified", "parts": len(parts["parts"]), "members": len(members),
                  "archive_sha256": parts["archive_sha256"]}))
