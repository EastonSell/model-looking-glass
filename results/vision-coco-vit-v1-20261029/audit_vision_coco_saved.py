"""Recompute saved ViT scores and check a closed, supervised vision run.

This reader loads no model. Use --synthetic only for the fabricated rehearsal.
It checks saved consistency, not the execution or physical process tree.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess

from audit_vision_coco_manifest import parse, require, typed_equal


CASES_SHA = "8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207"
DONORS_SHA = "5874dae623703d318d1b59bf30d427d053193932fd71e6ebca328a8e673439d0"
SOURCES = {
    "tools/run_vision_coco_study.py", "tools/prepare_vision_coco_cases.py",
    "looking_glass/__init__.py", "looking_glass/vision_coco_cases.py",
    "looking_glass/vision_coco_donors.py", "looking_glass/vision_coco_study.py",
    "looking_glass/intervention_plan.py", "looking_glass/interventions.py",
    "looking_glass/model_ownership.py", "looking_glass/cpu_study_supervision.py",
    "looking_glass/supervisor.py", "looking_glass/device_lease.py", "looking_glass/device_identity.py",
    "experiments/vision_coco_spatial_v1.json", "experiments/vision_coco_same_species_v1.json",
    "docs/VISION_COCO_SPATIAL_V1.md", "docs/CPU_STUDY_SUPERVISION.md",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def numeric(z):
    require(len(z) == 1000 and all(math.isfinite(x) for x in z), "invalid full logits")
    dog_ids, cat_ids = range(151, 269), range(281, 286)
    off_ids = [i for i in range(1000) if i not in dog_ids and i not in cat_ids]

    def logsum(ids):
        maximum = max(z[i] for i in ids)
        return maximum + math.log(math.fsum(math.exp(z[i] - maximum) for i in ids))

    all_log, dog_log, cat_log, off_log = (logsum(ids) for ids in (range(1000), dog_ids, cat_ids, off_ids))
    probs = [math.exp(z[i] - off_log) for i in off_ids]
    return {"margin": dog_log - cat_log, "dog_mass": math.exp(dog_log - all_log),
            "cat_mass": math.exp(cat_log - all_log), "off_mass": math.exp(off_log - all_log),
            "top1_id": max(range(1000), key=lambda i: z[i])}, probs


def checkpoint_state(assets):
    raw = (assets / "model.safetensors").read_bytes()
    require(sha(raw) == "a9146f3ee7cf6de399190132627e4aef1aeddd13692f5b333f24629dd36c37b0",
            "checkpoint file differs")
    require(len(raw) >= 8, "truncated safetensors")
    header_len = struct.unpack("<Q", raw[:8])[0]
    require(0 < header_len <= 1_000_000 and 8 + header_len <= len(raw), "invalid safetensors header")
    header = parse(raw[8:8 + header_len])
    base, tensors = 8 + header_len, {k: v for k, v in header.items() if k != "__metadata__"}
    require(len(tensors) == 200, "checkpoint tensor inventory differs")
    digest = hashlib.sha256()
    ranges = []
    for name, tensor in sorted(tensors.items()):
        shape, bounds = tensor["shape"], tensor["data_offsets"]
        require(tensor["dtype"] == "F32" and type(shape) is list and all(type(n) is int and n >= 0 for n in shape)
                and type(bounds) is list and len(bounds) == 2 and all(type(n) is int for n in bounds)
                and 0 <= bounds[0] <= bounds[1] <= len(raw) - base
                and bounds[1] - bounds[0] == 4 * math.prod(shape), "invalid checkpoint tensor")
        ranges.append(bounds)
        digest.update(name.encode() + b"\0torch.float32\0")
        digest.update(json.dumps(shape, separators=(",", ":")).encode() + b"\0")
        digest.update(raw[base + bounds[0]:base + bounds[1]])
    end = 0
    for start, stop in sorted(ranges):
        require(start == end, "checkpoint tensors overlap or omit bytes")
        end = stop
    require(base + end == len(raw), "extra checkpoint bytes")
    return digest.hexdigest()


def audit(root, repo, revision, synthetic, assets=None, development_root=None):
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "invalid revision")
    source = parse((root / "source.json").read_bytes())
    require(source["revision"] == revision and set(source["sources"]) == SOURCES, "source inventory differs")
    committed = {}
    for name, expected in source["sources"].items():
        p = subprocess.run(["git", "--no-replace-objects", "show", f"{revision}:{name}"], cwd=repo,
                           capture_output=True, check=True)
        require(sha(p.stdout) == expected, "source blob differs: " + name)
        committed[name] = p.stdout
    require(source["sources"]["experiments/vision_coco_spatial_v1.json"] == CASES_SHA
            and source["sources"]["experiments/vision_coco_same_species_v1.json"] == DONORS_SHA,
            "case/donor source differs")
    cases = parse(committed["experiments/vision_coco_spatial_v1.json"])
    donor_manifest = parse(committed["experiments/vision_coco_same_species_v1.json"])
    donor_rows = {(row["pair_id"], row["receiver_side"]): row["donor"] for row in donor_manifest["rows"]}
    case_rows = {row["pair_id"]: row for row in cases["pairs"]}
    labels = None
    if assets is not None:
        config_raw = (assets / "config.json").read_bytes()
        require(sha(config_raw) == "7e5aeb1add0a832153071a858bb18145cafb809e54e7fea4e5dc1183c82078f5",
                "model config differs")
        labels = parse(config_raw)["id2label"]
    data = root / "data"
    terminal_raw = (data / "terminal.json").read_bytes()
    terminal = parse(terminal_raw)
    supervisor_raw = (root / "supervisor.json").read_bytes()
    supervisor = parse(supervisor_raw)
    require(supervisor["status"] == "complete" and supervisor["clean"] is True
            and supervisor["timed_out"] is False and type(supervisor["exit_code"]) is int
            and supervisor["exit_code"] == 0 and supervisor["terminal_sha256"] == sha(terminal_raw),
            "supervisor/terminal receipt differs")
    require(terminal["status"] == "complete" and terminal["source_revision"] == revision
            and terminal["case_sha256"] == CASES_SHA and terminal["donor_sha256"] == DONORS_SHA
            and terminal["synthetic_rehearsal"] is synthetic and terminal["checkpoint_preflight_only"] is False,
            "terminal provenance differs")
    model_state_checked = False
    if not synthetic:
        started = parse((data / "started.json").read_bytes())
        require(started["split"] == terminal["split"] and started["source"] == source
                and started["case_sha256"] == CASES_SHA and started["donor_sha256"] == DONORS_SHA
                and started["checkpoint_preflight_only"] is False
                and started["model_state_sha256"] == terminal["model_state_sha256_before_after"]
                and typed_equal(started["runtime"], terminal["runtime"]), "start/terminal state differs")
        if assets is not None:
            require(checkpoint_state(assets) == started["model_state_sha256"], "loaded state differs from checkpoint")
            model_state_checked = True
    count, split = terminal["submissions"], terminal["split"]
    require(type(count) is int and 1 <= count <= 360 and split in ("development", "heldout"), "submission count")
    development_sha = None
    if not synthetic:
        require(type(terminal["saved_forwards"]) is int and terminal["saved_forwards"] == count,
                "terminal forward count differs")
        if split == "heldout":
            development_sha = terminal.get("development_terminal_sha256")
            require(type(development_sha) is str and re.fullmatch(r"[0-9a-f]{64}", development_sha)
                    and source.get("development_terminal_sha256") == development_sha,
                    "development receipt binding differs")
    names = {p.name for p in data.glob("forward-*.json")}
    bin_names = {p.name for p in data.glob("forward-*.bin")}
    require(names == {f"forward-{i:03d}.json" for i in range(1, count + 1)}
            and bin_names == {f"forward-{i:03d}.bin" for i in range(1, count + 1)}, "forward files incomplete")
    baselines, forward_scores = {}, {}
    for index in range(1, count + 1):
        record = parse((data / f"forward-{index:03d}.json").read_bytes())
        raw = (data / f"forward-{index:03d}.bin").read_bytes()
        label = record["label"]
        require(type(record["submission"]) is int and record["submission"] == index
                and record["file"] == f"forward-{index:03d}.bin" and record["sha256"] == sha(raw)
                and record["dtype"] == "little-endian-float32" and label not in forward_scores,
                "forward identity or bytes differ")
        offset = 0
        for name, layout in sorted(record["layout"].items(), key=lambda pair: pair[1]["offset"]):
            shape = layout["shape"]
            require(type(shape) is list and all(type(n) is int and n >= 0 for n in shape)
                    and type(layout["offset"]) is int and layout["offset"] == offset
                    and type(layout["bytes"]) is int and layout["bytes"] == 4 * math.prod(shape),
                    "tensor layout differs: " + name)
            offset += layout["bytes"]
        require(offset == len(raw) and len(raw) <= 1_000_000, "binary tensor inventory differs")
        layout = record["layout"]["logits"]
        require(typed_equal(layout["shape"], [1, 1000]), "logit shape differs")
        start = layout["offset"]
        z = struct.unpack("<1000f", raw[start:start + 4000])
        expected, off = numeric(z)
        actual = record["score"]
        for key, value in expected.items():
            if key == "top1_id":
                require(typed_equal(actual[key], value), "top-1 identity differs")
            else:
                require(type(actual[key]) is float and math.isclose(actual[key], value, rel_tol=1e-11, abs_tol=1e-12),
                        "score differs: " + key)
        if labels is not None and not synthetic:
            require(actual.get("top1_label") == labels[str(expected["top1_id"])], "top-1 label differs")
        fields = label.split(":")
        require(len(fields) == 3 and fields[1] in ("cat", "dog"), "forward label differs")
        key = (fields[0], fields[1])
        if fields[2] == "initial":
            require(key not in baselines, "duplicate baseline")
            baselines[key] = (actual, off)
        else:
            require(key in baselines, "forward lacks baseline")
            tv = .5 * math.fsum(abs(a - b) for a, b in zip(off, baselines[key][1]))
            require(type(actual["off_conditional_tv"]) is float
                    and math.isclose(actual["off_conditional_tv"], tv, rel_tol=1e-11, abs_tol=1e-12),
                    "off-group total variation differs")
        forward_scores[label] = actual
    rows = terminal["pairs"]
    pair_ids = ["pair-00"] if synthetic else [f"pair-{i:02d}" for i in (range(4) if split == "development" else range(4, 12))]
    require(type(rows) is list and [row["pair_id"] for row in rows] == pair_ids, "pair inventory differs")
    allowed = {"positive", "negative", "baseline-ineligible", "dose-ineligible", "dose-unmatchable"}
    status_counts = {k: 0 for k in sorted(allowed)}
    expected_labels = set()
    for row in rows:
        pid = row["pair_id"]
        require(row["split"] == split and row["status"] in allowed, "pair status differs")
        require(typed_equal(parse((data / f"{pid}.json").read_bytes()), row), "pair record differs")
        status_counts[row["status"]] += 1
        cat, dog = baselines[(pid, "cat")][0], baselines[(pid, "dog")][0]
        gap = dog["margin"] - cat["margin"]
        reasons = []
        if cat["margin"] >= 0:
            reasons.append("cat-relative-margin")
        if dog["margin"] <= 0:
            reasons.append("dog-relative-margin")
        if gap < 1:
            reasons.append("gap-below-one")
        if cat["cat_mass"] < .2:
            reasons.append("cat-mass-below-0.2")
        if dog["dog_mass"] < .2:
            reasons.append("dog-mass-below-0.2")
        require(typed_equal(row["baseline_eligibility"], {"eligible": not reasons, "reasons": reasons, "gap": gap}),
                "baseline eligibility differs")
        dose_statuses = {side: row["sides"][side]["dose"]["status"] for side in ("cat", "dog")}
        require(typed_equal(row["dose_matchability"], dose_statuses)
                and all(value in ("matchable", "dose-ineligible", "dose-unmatchable")
                        for value in dose_statuses.values()), "dose statuses differ")
        for side in ("cat", "dog"):
            require(typed_equal(row["sides"][side]["baseline"], baselines[(pid, side)][0]), "baseline record differs")
            if not synthetic:
                donor = donor_rows[(pid, side)]
                require(typed_equal(row["sides"][side]["image_id"], case_rows[pid][side]["image_id"])
                        and typed_equal(row["sides"][side]["same_species_donor"], donor),
                        "case or same-species donor binding differs")
                donor_baseline = None if donor is None else baselines[(donor["donor_pair_id"], side)][0]
                require(typed_equal(row["sides"][side]["same_species_baseline"], donor_baseline),
                        "same-species baseline differs")
            conditions = row["sides"][side]["conditions"]
            expected_conditions = {"repeat", "post_clean", "null_inside", "null_outside", "self_inside",
                                   "self_outside", "full_self", "full_swap", "natural_inside", "final_inside"}
            if dose_statuses[side] == "matchable":
                expected_conditions |= {"matched_inside", "matched_outside", "matched_random"}
                if row["sides"][side]["dose"]["same_species_status"] == "available":
                    expected_conditions.add("matched_same_species")
            require(set(conditions) == expected_conditions, "condition inventory differs")
            expected_labels.add(f"{pid}:{side}:initial")
            for condition, score in conditions.items():
                label = f"{pid}:{side}:{condition}"
                require(label in forward_scores and all(typed_equal(score[k], value)
                                                       for k, value in forward_scores[label].items()),
                        "condition score differs")
                expected_labels.add(label)
                if condition not in ("repeat", "post_clean"):
                    change = score["margin"] - row["sides"][side]["baseline"]["margin"]
                    require(typed_equal(score["raw_margin_change"], change)
                            and typed_equal(score["annotation_oriented_change"], (1 if side == "cat" else -1) * change),
                            "saved effect differs")
        if reasons:
            expected_status = "baseline-ineligible"
        elif any(value != "matchable" for value in dose_statuses.values()):
            expected_status = "dose-ineligible" if "dose-ineligible" in dose_statuses.values() else "dose-unmatchable"
        else:
            movements = {}
            for side in ("cat", "dog"):
                values = row["sides"][side]
                movements[side] = {site: (1 if side == "cat" else -1)
                                   * (values["conditions"]["matched_" + site]["margin"]
                                      - values["baseline"]["margin"])
                                   for site in ("inside", "outside")}
            require(typed_equal(row["movements"], movements), "primary movements differ")
            passed = all(movements[side]["inside"] >= .02 * gap
                         and movements[side]["inside"] >= abs(movements[side]["outside"]) + .01 * gap
                         for side in ("cat", "dog"))
            expected_status = "positive" if passed else "negative"
        require(row["status"] == expected_status, "pair classification differs")
    require(expected_labels == set(forward_scores), "unexpected or missing forward labels")
    if not synthetic and split == "heldout":
        evaluable = status_counts["positive"] + status_counts["negative"]
        positive = status_counts["positive"]
        decision = "inconclusive" if evaluable < 4 else "positive" if positive * 4 >= evaluable * 3 else "negative"
        expected = {"status": decision, "selected": 8, "evaluable": evaluable, "positive": positive}
        require(typed_equal(terminal["aggregate"], expected), "aggregate differs")
    development_verified = False
    if development_root is not None:
        require(not synthetic and split == "heldout", "development root only applies to held-out evidence")
        prior = audit(development_root, repo, revision, False, assets)
        require(prior["split"] == "development" and prior["terminal_sha256"] == development_sha,
                "held-out run does not bind the verified development receipt")
        development_verified = True
    return {"status": "saved_consistency_verified", "revision": revision, "split": split,
            "audit_reader_sha256": sha(Path(__file__).read_bytes()),
            "support_reader_sha256": sha(Path(__file__).with_name("audit_vision_coco_manifest.py").read_bytes()),
            "synthetic_rehearsal": synthetic, "source_files": len(SOURCES), "forwards": count,
            "model_labels_checked": labels is not None and not synthetic, "model_state_checked": model_state_checked,
            "pairs": len(rows), "pair_status_counts": status_counts, "terminal_sha256": sha(terminal_raw),
            "supervisor_sha256": sha(supervisor_raw),
            "aggregate": terminal.get("aggregate"), "development_terminal_sha256": development_sha,
            "development_receipt_verified": development_verified,
            "limitations": ["Consistency of saved files and logits does not prove actual execution or physical process-tree cleanup.",
                            "This reader does not independently verify hook landing or prospective publication timing."]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("revision")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--assets", type=Path, help="optional pinned model assets for top-1 label checks")
    parser.add_argument("--development-root", type=Path, help="verify the held-out run's saved development receipt")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.repository, args.revision, args.synthetic, args.assets,
                           args.development_root), indent=2))
