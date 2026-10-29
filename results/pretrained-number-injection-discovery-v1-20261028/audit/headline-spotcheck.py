"""Independently recompute selected public injection-discovery quantities.

This stdlib-only check reads eight public cases and a separately pinned public
audit result. It never reads the concealed site, private witnesses, or any
held-out cases. It is a focused cross-check, not a full archive/custody audit.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct


def _load(path, maximum):
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not raw or len(raw) > maximum:
        raise ValueError("missing or oversized public input")
    return raw, json.loads(raw)


def _digest(value):
    if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA-256 pin")
    return value


def _decode(encoded, size):
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != 4 * size or base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError("public float32 payload has the wrong size or encoding")
    values = []
    for (bits,) in struct.iter_unpack("<I", raw):
        exponent, fraction = (bits >> 23) & 255, bits & 0x7FFFFF
        if exponent == 255:
            raise ValueError("public float32 payload contains a nonfinite value")
        magnitude = math.ldexp(float(fraction if exponent == 0 else (1 << 23) | fraction),
                               -149 if exponent == 0 else exponent - 150)
        values.append(-magnitude if bits >> 31 else magnitude)
    return values


def _margin(row):
    values = _decode(row["payload"]["logits"], 50257)
    return math.fsum((values[318], -values[389]))


def check(public, audit_path, *, audit_sha256, prediction_sha256, archive_sha256):
    audit_sha256, prediction_sha256, archive_sha256 = map(
        _digest, (audit_sha256, prediction_sha256, archive_sha256))
    audit_raw, audit = _load(audit_path, 128 * 1024)
    if (hashlib.sha256(audit_raw).hexdigest() != audit_sha256
            or audit.get("schema") != "mlg-independent-injection-public-audit-v1"
            or audit.get("status") != "passed" or audit.get("records") != 128
            or audit.get("prediction_sha256") != prediction_sha256
            or audit.get("archive_sha256") != archive_sha256):
        raise ValueError("public audit does not match the supplied pins")
    if not isinstance(audit.get("cases"), list) or len(audit["cases"]) != 8:
        raise ValueError("public audit does not contain eight cases")
    roles = ("clean", "repeat", "injected", *(f"forward_{i}" for i in range(1, 5)),
             *(f"reverse_{i}" for i in range(1, 5)), "null", "self", "upstream",
             "downstream", "verify_clean")
    file_digests = {}
    for index in range(8):
        for role in roles:
            path = Path(public) / f"observation-{index:02d}-{role}.json"
            with path.open("rb") as stream:
                raw = stream.read(512 * 1024 + 1)
            if not raw or len(raw) > 512 * 1024:
                raise ValueError("public observation is missing or oversized")
            file_digests[index, role] = hashlib.sha256(raw).digest()
    ordered = b"".join(file_digests[index, role] for index in range(8) for role in roles)
    if hashlib.sha256(b"MLG-number-injection-archive-v1\0" + ordered).hexdigest() != archive_sha256:
        raise ValueError("public observation files do not match the pinned archive")
    votes = {str(i): 0 for i in range(1, 5)}
    magnitudes = {str(i): [] for i in range(1, 5)}
    observations, rows, forward_four = {}, [], []
    for index in range(8):
        saved = {}
        for role in ("clean", "injected", "forward_4", "reverse_4"):
            raw, row = _load(Path(public) / f"observation-{index:02d}-{role}.json", 512 * 1024)
            if type(row) is not dict or type(row.get("case_index")) is not int or row["case_index"] != index or row.get("role") != role:
                raise ValueError("public observation identity differs")
            if hashlib.sha256(raw).digest() != file_digests[index, role]:
                raise ValueError("public observation changed during spot-check")
            observations[f"{index:02d}-{role}"] = hashlib.sha256(raw).hexdigest()
            saved[role] = row
        if len({row["input_sha256"] for row in saved.values()}) != 1:
            raise ValueError("same-case observation inputs differ")
        clean, injected = saved["clean"], saved["injected"]
        first, norms = None, []
        for site in range(1, 5):
            left = _decode(clean["payload"]["mlp"][site], 768)
            right = _decode(injected["payload"]["mlp"][site], 768)
            if first is None and any(abs(a - b) > 1e-5 + 1e-5 * max(abs(a), abs(b))
                                     for a, b in zip(left, right)):
                first = site
            norm = math.sqrt(math.fsum((a - b) * (a - b) for a, b in zip(left, right)))
            norms.append(norm)
            magnitudes[str(site)].append(norm)
        if first is not None:
            votes[str(first)] += 1
        clean_margin, injected_margin = _margin(clean), _margin(injected)
        forward = math.fsum((_margin(saved["forward_4"]), -injected_margin))
        reverse = math.fsum((_margin(saved["reverse_4"]), -clean_margin))
        forward_four.append(forward)
        expected = audit["cases"][index]
        if (expected["case_index"] != index or expected["first_changed"] != first
                or expected["l2"] != norms or expected["forward_effects"][3] != forward
                or expected["reverse_effects"][3] != reverse):
            raise ValueError("independent case calculation differs from public audit")
        rows.append({"case_index": index, "first_changed": first, "l2": norms,
                     "layer4_forward_effect": forward, "layer4_reverse_effect": reverse})
    ranking = sorted(range(1, 5), key=lambda site: (-votes[str(site)], site))
    means = {site: math.fsum(values) / 8 for site, values in magnitudes.items()}
    forward_mean = math.fsum(forward_four) / 8
    expected = audit["aggregate"]
    if (expected["votes"] != votes or expected["ranking"] != ranking
            or expected["first_changed"] != (ranking[0] if votes[str(ranking[0])] else None)
            or expected["mean_l2"] != means or expected["predictions"]["4"] != forward_mean):
        raise ValueError("independent aggregate differs from public audit")
    return {"schema": "mlg-injection-headline-spotcheck-v1", "status": "passed",
            "source_commit": audit["source_commit"], "audit_result_sha256": audit_sha256,
            "prediction_sha256": prediction_sha256, "archive_sha256": archive_sha256,
            "observation_sha256": observations, "cases": rows, "votes": votes,
            "ranking": ranking, "mean_l2": means, "layer4_forward_mean": forward_mean,
            "limitations": "Selected public saved records only; does not inspect the site, private witnesses or held-out cases, or attest execution and custody."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_directory")
    parser.add_argument("audit_result")
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--prediction-sha256", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).resolve().is_relative_to(Path(args.public_directory).resolve()):
        raise ValueError("output must be outside the public observation archive")
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = check(args.public_directory, args.audit_result, audit_sha256=args.audit_sha256,
                   prediction_sha256=args.prediction_sha256, archive_sha256=args.archive_sha256)
    if source_sha != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("spot-check source changed during the audit")
    result["spotcheck_source_sha256"] = source_sha
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")) + "\n").encode("utf-8")
    if len(raw) > 16 * 1024:
        raise ValueError("spot-check result exceeds the byte limit")
    with Path(args.output).open("xb") as output:
        if output.write(raw) != len(raw):
            raise OSError("short spot-check result write")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({"status": result["status"], "result_sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
