"""Recover matched donor-direction geometry from a closed saved journal; no model."""
import argparse
import array
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

MANIFEST_SHA = "917bad800f72a9613e69d8feae3faf90ffbae5cc446bbeb3e2c8df1d03678dc9"
CUES = ("never", "rarely", "always", "often")


def numeric_mode():
    tiny64 = struct.unpack("<d", bytes.fromhex("0100000000000000"))[0]
    neg64 = struct.unpack("<d", bytes.fromhex("0100000000000080"))[0]
    tiny32 = struct.unpack("<f", bytes.fromhex("01000000"))[0]
    neg32 = struct.unpack("<f", bytes.fromhex("01000080"))[0]
    if (struct.pack("<d", tiny64 * 2) != bytes.fromhex("0200000000000000") or
            struct.pack("<d", neg64 * 2) != bytes.fromhex("0200000000000080") or
            struct.pack("<d", tiny32) != struct.pack("<d", math.ldexp(1.0, -149)) or
            struct.pack("<d", neg32) != struct.pack("<d", -math.ldexp(1.0, -149)) or
            struct.pack("<f", 1 + 2 ** -24) != bytes.fromhex("0000803f") or
            struct.pack("<f", 1 + 3 * 2 ** -24) != bytes.fromhex("0200803f")):
        raise ValueError("floating-point mode differs")


def validate_edges(edges):
    expected = [(ci, cue, donor) for ci in range(2) for cue in CUES
                for donor in (CUES[2:] if cue in CUES[:2] else CUES[:2])]
    if type(edges) is not list or any(type(edge) is not dict or
            type(edge.get("target_context")) is not int for edge in edges):
        raise ValueError("edge identity type differs")
    actual = [(edge.get("target_context"), edge.get("target_cue"), edge.get("donor_cue")) for edge in edges]
    if actual != expected:
        raise ValueError("ordered matched edge identities differ")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(encoded, width):
    raw = base64.b64decode(encoded, validate=True)
    values = array.array("f")
    if values.itemsize != 4 or len(raw) != width * 4:
        raise ValueError("capture width differs")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite capture")
    return values


def distance(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("capture dimensions differ")
    return math.sqrt(math.fsum((float(y) - float(x)) ** 2 for x, y in zip(left, right)))


def matched_geometry(target, within, alternate):
    wd, ad = distance(target, within), distance(target, alternate)
    cosine = None if wd == 0 or ad == 0 else math.fsum(
        (float(w)-float(t)) * (float(a)-float(t)) for t, w, a in zip(target, within, alternate)) / (wd * ad)
    return wd, ad, cosine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "journal", "result", "output"):
        parser.add_argument(name, type=Path)
    for name in ("manifest", "journal", "result"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    parser.add_argument("--expected-source")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args()
    numeric_mode()
    paths = (args.manifest, args.journal, args.result)
    expected = (args.expected_manifest_sha256, args.expected_journal_sha256, args.expected_result_sha256)
    if any(not re.fullmatch("[0-9a-f]{64}", value) for value in expected) or tuple(sha(p) for p in paths) != expected:
        raise ValueError("fixed input hashes differ")
    manifest, result = json.loads(args.manifest.read_bytes()), json.loads(args.result.read_bytes())
    if not args.synthetic and (expected[0] != MANIFEST_SHA or args.width != 768 or
            not re.fullmatch("[0-9a-f]{40}", args.expected_source or "") or
            result.get("source_commit") != args.expected_source or result.get("events_sha256") != expected[1]):
        raise ValueError("source or journal identity differs")
    expected_ids = {(pair["id"], ci, cue): context["rows"][cue]["input_ids"]
                    for pair in manifest["pairs"] for ci, context in enumerate(pair["contexts"]) for cue in CUES}
    if len(expected_ids) != 32:
        raise ValueError("fixed clean input count differs")
    clean, transfers, events = {}, {}, 0
    with gzip.open(args.journal, "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            events += 1
            if event["kind"] == "clean":
                if type(event.get("context")) is not int or event["context"] not in (0, 1):
                    raise ValueError("clean context must be an integer")
                key = event["pair_id"], event["context"], event["cue"]
                if key in clean or event["input_ids"] != expected_ids[key]:
                    raise ValueError("clean identity differs")
                clean[key] = vector(event["captures_f32le"]["cue2"], args.width)
            elif event["kind"] == "transfer" and event["site"] == "cue2":
                pair, ci, cue = event["pair_id"], event["target_context"], event["target_cue"]
                donor_ci, donor_cue = event["donor_context"], event["donor_cue"]
                if type(ci) is not int or ci not in (0, 1) or type(donor_ci) is not int or donor_ci not in (0, 1):
                    raise ValueError("transfer contexts must be integers")
                if cue not in CUES or donor_cue not in CUES or pair not in {p["id"] for p in manifest["pairs"]}:
                    raise ValueError("transfer cue or pair differs")
                if (cue in CUES[:2]) == (donor_cue in CUES[:2]):
                    continue
                key = pair, ci, cue, donor_ci, donor_cue
                if key in transfers or event["input_ids"] != expected_ids[pair, ci, cue]:
                    raise ValueError("transfer identity differs")
                values = (event["requested_radius"], event["requested_fraction"],
                          event["actual_norm"], event["relative_error"])
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                    raise ValueError("nonfinite transfer geometry")
                transfers[key] = values
    if set(clean) != set(expected_ids):
        raise ValueError("missing clean records")
    results = {row["id"]: row for row in result["pairs"]}
    if len(result["pairs"]) != len(manifest["pairs"]) or set(results) != {pair["id"] for pair in manifest["pairs"]}:
        raise ValueError("result pair identities differ")
    rows, used = [], set()
    for pair in manifest["pairs"]:
        saved = results[pair["id"]]
        if type(saved.get("dose_eligible")) is not bool:
            raise ValueError("dose eligibility must be boolean")
        validate_edges(saved["retention"]["edges"])
        for edge in saved["retention"]["edges"]:
            ci, cue, donor = edge["target_context"], edge["target_cue"], edge["donor_cue"]
            if type(ci) is not int or ci not in (0, 1):
                raise ValueError("edge context must be an integer")
            target = clean[pair["id"], ci, cue]
            within = clean[pair["id"], ci, donor]
            alternate = clean[pair["id"], 1-ci, donor]
            wd, ad, cosine = matched_geometry(target, within, alternate)
            if wd != edge["within_distance"] or ad != edge["alternate_distance"]:
                raise ValueError("stored natural distance differs")
            matched = []
            for donor_ci in (ci, 1-ci):
                key = pair["id"], ci, cue, donor_ci, donor
                values = transfers.get(key)
                if saved["dose_eligible"]:
                    if values is None or values[0] != saved["q"]:
                        raise ValueError("missing matched transfer")
                    used.add(key)
                elif values is not None:
                    raise ValueError("dose-ineligible transfer exists")
                matched.append(values)
            rows.append({"pair_id": pair["id"], "split": pair["split"], "target_context": ci,
                         "target_cue": cue, "donor_cue": donor, "within_distance": wd,
                         "alternate_distance": ad, "direction_cosine": cosine,
                         "within_transfer": matched[0], "alternate_transfer": matched[1]})
    if len(rows) != 64 or set(transfers) != used:
        raise ValueError("matched edge count differs")
    if tuple(sha(p) for p in paths) != expected:
        raise ValueError("inputs changed during geometry replay")
    numeric_mode()
    report = {"status": "passed", "events": events, "matched_edges": len(rows), "rows": rows,
              "manifest_sha256": expected[0], "journal_sha256": expected[1], "result_sha256": expected[2],
              "source_commit": None if args.synthetic else args.expected_source,
              "scope": "Saved capture directions and recorded dose fields; full witness replay checks landing and execution is not attested."}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "events": events, "matched_edges": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
