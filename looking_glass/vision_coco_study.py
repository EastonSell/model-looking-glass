"""Matched spatial interventions on a fixed public ViT/COCO split.

The runner retains every root-forward logit and same-invocation activation
witness. A successful local receipt is a cooperative record, not independent
attestation of the executing interpreter, dependencies, or native libraries.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .intervention_plan import ActivationPlan


EARLY = "vit.encoder.layer.2"
LATE = "vit.encoder.layer.11"
MAX_FORWARDS_PER_PAIR = 30


def check_denormals():
    # This study executes its numeric work on the same CPU calling thread.
    tiny = torch.tensor([1.401298464324817e-45], dtype=torch.float32)
    if float(tiny[0]) != 1.401298464324817e-45:
        raise ValueError("float32 denormal preservation changed")


def _bits(value):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.numel() > 3 * 224 * 224:
        raise ValueError("tensor cannot be retained as bounded float32")
    value = value.detach().to(device="cpu").contiguous()
    return value.numpy().astype("<f4", copy=False).tobytes()


def _same(a, b):
    return tuple(a.shape) == tuple(b.shape) and _bits(a) == _bits(b)


def _array(value):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError("expected float32 tensor")
    result = value.detach().to("cpu").contiguous().numpy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite activation")
    return result


def distance(a, b):
    x, y = _array(a).reshape(-1), _array(b).reshape(-1)
    if x.size != y.size:
        raise ValueError("distance shapes differ")
    result = math.sqrt(math.fsum((float(u) - float(v)) ** 2 for u, v in zip(x, y)))
    if not math.isfinite(result):
        raise ValueError("nonfinite activation distance")
    return result


def _tolerance(q):
    return min(max(1e-5, 1e-5 * q), 0.01 * q)


def _replacement(own, donor, q, d):
    # Separate CPU binary64 operations, followed by one float32 cast.
    x = _array(own).astype(np.float64)
    y = _array(donor).astype(np.float64)
    result = (x + (q / d) * (y - x)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite replacement")
    return torch.from_numpy(result.copy())


def _random_replacement(own, q, pair_id, side):
    x = _array(own).astype(np.float64)
    signs = [1.0 if hashlib.sha256(
        f"vision-coco-random-v1:{pair_id}:{side}:{i}".encode("ascii")).digest()[0] & 1
        else -1.0 for i in range(x.size)]
    result = (x + (q / math.sqrt(x.size)) * np.asarray(signs).reshape(x.shape)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite random replacement")
    return torch.from_numpy(result.copy())


def prepare_dose(own, other, inside, outside, pair_id, side, same=None):
    """Return exact planned CPU float32 deliveries and explicit availability."""
    check_denormals()
    oi, di = own[0, inside], other[0, inside]
    oo, do = own[0, outside], other[0, outside]
    inside_d, outside_d = distance(oi, di), distance(oo, do)
    status = {"inside_distance": inside_d, "outside_distance": outside_d,
              "status": "dose-ineligible", "same_species_status": "geometry-unavailable"}
    if min(inside_d, outside_d) <= 1e-6:
        return status, {}
    q = min(inside_d, outside_d)
    requested = {"inside": _replacement(oi, di, q, inside_d),
                 "outside": _replacement(oo, do, q, outside_d),
                 "random": _random_replacement(oi, q, pair_id, side)}
    norms = {key: distance(value, oo if key == "outside" else oi)
             for key, value in requested.items()}
    status.update(q=q, tolerance=_tolerance(q), planned_norms=norms)
    if (any(value <= 0 or abs(value - q) > _tolerance(q) for value in norms.values())
            or any(abs(norms[key] - norms["inside"]) > .01 * norms["inside"]
                   for key in ("outside", "random"))):
        status["status"] = "dose-unmatchable"
        return status, requested
    status["status"] = "matchable"
    if same is not None:
        same_d = distance(oi, same[0, inside])
        status["same_species_distance"] = same_d
        if same_d <= 1e-6 or same_d < q:
            status["same_species_status"] = "distance-too-small"
        else:
            same_value = _replacement(oi, same[0, inside], q, same_d)
            same_norm = distance(oi, same_value)
            status.update(same_species_scale=q / same_d, same_species_planned_norm=same_norm)
            if (same_norm <= 0 or abs(same_norm - q) > _tolerance(q)
                    or abs(same_norm - norms["inside"]) > .01 * norms["inside"]):
                status["same_species_status"] = "rounding-unmatchable"
            else:
                requested["same_species"] = same_value
                status["same_species_status"] = "available"
    return status, requested


def score(logits, reference=None):
    z = [float(x) for x in _array(logits).reshape(-1)]
    if len(z) != 1000:
        raise ValueError("expected 1000 logits")
    complement = [i for i in range(1000) if not (151 <= i <= 268 or 281 <= i <= 285)]
    def shifted(values):
        maximum = max(values)
        return maximum, math.fsum(math.exp(value - maximum) for value in values)
    dog_max, dog_sum = shifted(z[151:269])
    cat_max, cat_sum = shifted(z[281:286])
    off_max, off_sum = shifted([z[i] for i in complement])
    total_max, total_sum = shifted(z)
    conditional = [math.exp(z[i] - off_max) / off_sum for i in complement]
    result = {"margin": ((dog_max - cat_max) + math.log(dog_sum) - math.log(cat_sum)),
              "dog_mass": math.exp(dog_max - total_max) * dog_sum / total_sum,
              "cat_mass": math.exp(cat_max - total_max) * cat_sum / total_sum,
              "off_mass": math.exp(off_max - total_max) * off_sum / total_sum,
              "top1_id": max(range(1000), key=lambda i: z[i])}
    if reference is not None:
        result["off_conditional_tv"] = .5 * math.fsum(abs(a - b) for a, b in zip(conditional, reference))
    return result, conditional


class Evidence:
    """Create-only local evidence with a finite root-forward limit."""
    def __init__(self, root, limit, labels=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.limit = limit
        self.labels = labels
        self.submissions = 0
        self.records = []
        self.active = {"phase": "startup"}

    def write(self, name, data):
        with (self.root / name).open("xb") as stream:
            if stream.write(data) != len(data):
                raise OSError("short evidence write")
            stream.flush()
        return hashlib.sha256(data).hexdigest()

    def json(self, name, value):
        return self.write(name, (json.dumps(value, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False) + "\n").encode())

    def reserve(self, label):
        if self.submissions >= self.limit:
            raise ValueError("root-forward cap exceeded")
        self.submissions += 1
        self.active = {"phase": "forward", "submission": self.submissions, "label": label}

    def save_forward_bytes(self, label, tensors):
        pieces, layout, offset = [], {}, 0
        for key, tensor in tensors.items():
            data = _bits(tensor)
            layout[key] = {"shape": list(tensor.shape), "offset": offset, "bytes": len(data)}
            pieces.append(data)
            offset += len(data)
        packed = b"".join(pieces)
        if len(packed) > 1_000_000:
            raise ValueError("forward evidence exceeds byte bound")
        index = self.submissions
        name = f"forward-{index:03d}.bin"
        digest = self.write(name, packed)
        row = {"submission": index, "label": label, "file": name, "sha256": digest,
               "dtype": "little-endian-float32", "layout": layout}
        self.active.update(saved_file=name, saved_sha256=digest, tensor_layout=layout)
        return row

    def finish_forward(self, row, metadata, score_record):
        row.update(metadata=metadata, score=score_record)
        index = row["submission"]
        self.json(f"forward-{index:03d}.json", row)
        self.records.append(row)
        return row


def _check_metadata(metadata, steps):
    events = metadata.get("steps")
    if type(events) is not list or len(events) != len(steps):
        raise ValueError("missing ordered intervention metadata")
    for i, (event, step) in enumerate(zip(events, steps)):
        for key in ("name", "operation", "module_path"):
            if event.get(key) != step[key]:
                raise ValueError("intervention metadata mismatch")
        if (event.get("order") != i or event.get("invocation") != 0
                or event.get("batch_index") != step.get("batch_index")
                or event.get("token_index") != step.get("token_index")
                or event.get("output_shape") != [1, 197, 192]
                or event.get("dtype") != "torch.float32"):
            raise ValueError("intervention metadata geometry mismatch")


def forward(model, pixels, evidence, label, *, path=None, tokens=None, replacement=None,
            action="capture", reference=None, numeric_guard=None):
    """Reserve before input/hook preparation; save bytes before control checks."""
    evidence.reserve(label)
    check_denormals()
    if numeric_guard is not None:
        numeric_guard()
    pixels = pixels.detach().clone()
    original_pixels = _bits(pixels)
    if path is None:
        steps = [{"name": "early", "operation": "capture", "module_path": EARLY},
                 {"name": "late", "operation": "capture", "module_path": LATE}]
    else:
        site = {"module_path": path}
        steps = [dict(name="before_all", operation="capture", **site)]
        if tokens is None:
            steps.append(dict(name="action", operation=action, **site))
            if action == "patch":
                steps[-1]["value"] = replacement.to(pixels.device)
        else:
            cls_token = len(tokens) == 1 and type(tokens[0]) is int and tokens[0] == 0
            spatial_tokens = (len(tokens) == 4 and len(set(tokens)) == 4 and
                              all(type(t) is int and 1 <= t < 197 for t in tokens))
            if not (cls_token or spatial_tokens):
                raise ValueError("invalid CLS or four-token intervention")
            for i, token in enumerate(tokens):
                selected = dict(site, batch_index=0, token_index=token)
                steps.extend([dict(name=f"before_{i}", operation="capture", **selected),
                              dict(name=f"action_{i}", operation=action, **selected),
                              dict(name=f"after_{i}", operation="capture", **selected)])
                if action == "patch":
                    steps[-2]["value"] = replacement[i].to(pixels.device)
        steps.append(dict(name="after_all", operation="capture", **site))
    output = plan = None
    retained = False
    try:
        with torch.no_grad(), ActivationPlan(model, steps) as plan:
            output = model(pixel_values=pixels)
        captures = plan.captures
        logits = output.logits.detach().cpu().contiguous()
        tensors = {"logits": logits, **{key: value.detach().cpu() for key, value in captures.items()}}
        if replacement is not None:
            tensors["requested"] = replacement.detach().cpu()
        row = evidence.save_forward_bytes(label, tensors)
        retained = True
        evidence.active["phase"] = "validation"
        check_denormals()
        if numeric_guard is not None:
            numeric_guard()
        metadata = plan.metadata
        record_score, _ = score(logits, reference)
        if numeric_guard is not None:
            numeric_guard()
        if evidence.labels is not None:
            record_score["top1_label"] = evidence.labels[record_score["top1_id"]]
        row = evidence.finish_forward(row, metadata, record_score)
        if _bits(pixels) != original_pixels:
            raise ValueError("model modified input pixels")
        _check_metadata(metadata, steps)
        if tuple(logits.shape) != (1, 1000):
            raise ValueError("unexpected logits shape")
        for capture in captures.values():
            if capture.dtype != torch.float32 or not bool(torch.isfinite(capture).all()):
                raise ValueError("nonfinite or non-float32 capture")
        return logits, captures, row
    except BaseException as original:
        # Salvage individually available bounded output components even when
        # session cleanup, metadata, a getter or scoring fails after return.
        if not retained and output is not None:
            try:
                available = {}
                try:
                    value = output.logits
                    if isinstance(value, torch.Tensor) and value.dtype == torch.float32 and 0 < value.numel() <= 1000:
                        available["logits"] = value.detach().cpu()
                except Exception:
                    pass
                try:
                    for key, value in plan.captures.items():
                        if isinstance(value, torch.Tensor) and value.dtype == torch.float32 and 0 < value.numel() <= 197 * 192:
                            available[key] = value.detach().cpu()
                except Exception:
                    pass
                if available:
                    evidence.save_forward_bytes(label, available)
            except BaseException as salvage_error:
                if isinstance(original, Exception) and not isinstance(salvage_error, Exception):
                    raise
        evidence.active["return_observed"] = output is not None
        raise


def check_landing(captures, own, tokens, replacement, *, action="patch"):
    before, after = captures["before_all"], captures["after_all"]
    if not _same(before, own):
        raise ValueError("same-forward pre-action state differs from baseline")
    if tokens is None:
        target = own if action == "null" else replacement
        if not _same(after, target):
            raise ValueError("full-state landing differs")
        return
    expected = own.detach().clone()
    if action == "patch":
        expected[0, tokens] = replacement.to(expected.device)
    if not _same(after, expected):
        raise ValueError("patch landing or untouched tokens differ")
    for i, token in enumerate(tokens):
        if (not _same(captures[f"before_{i}"], own[0, token])
                or not _same(captures[f"after_{i}"], expected[0, token])):
            raise ValueError("per-token same-forward witness differs")


def baseline_status(cat_score, dog_score):
    gap = dog_score["margin"] - cat_score["margin"]
    reasons = []
    if cat_score["margin"] >= 0:
        reasons.append("cat-relative-margin")
    if dog_score["margin"] <= 0:
        reasons.append("dog-relative-margin")
    if gap < 1:
        reasons.append("gap-below-one")
    if cat_score["cat_mass"] < .2:
        reasons.append("cat-mass-below-0.2")
    if dog_score["dog_mass"] < .2:
        reasons.append("dog-mass-below-0.2")
    return {"eligible": not reasons, "reasons": reasons, "gap": gap}


def aggregate(rows):
    if (len(rows) != 8 or {row.get("pair_id") for row in rows} != {f"pair-{i:02d}" for i in range(4, 12)}
            or any(row.get("split") != "heldout" or row.get("status") not in
                   ("positive", "negative", "baseline-ineligible", "dose-ineligible", "dose-unmatchable")
                   for row in rows)):
        raise ValueError("aggregate requires all eight unique held-out pairs")
    eligible = [row for row in rows if row["status"] in ("positive", "negative")]
    positives = sum(row["status"] == "positive" for row in eligible)
    status = ("inconclusive" if len(eligible) < 4 else
              "positive" if positives * 4 >= 3 * len(eligible) else "negative")
    return {"status": status, "selected": len(rows), "evaluable": len(eligible), "positive": positives}


def run_split(model, pairs, donors, pixel_by_id, evidence):
    """Run all finite, matchable interventions, including unsupported baselines."""
    selected_ids = {p[side]["image_id"] for p in pairs for side in ("cat", "dog")}
    donor_rows = {(r["pair_id"], r["receiver_side"]): r for r in donors["rows"]}
    clean = {}
    for pair in pairs:
        for side in ("cat", "dog"):
            image_id = pair[side]["image_id"]
            logits, caps, row = forward(model, pixel_by_id[image_id], evidence,
                                        f"{pair['pair_id']}:{side}:initial")
            _, ref = score(logits)
            clean[image_id] = {"logits": logits, "early": caps["early"], "late": caps["late"],
                               "score": row["score"], "off_reference": ref}
    results = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        inside, outside = pair["box_tokens"], pair["control_tokens"]
        base = baseline_status(clean[pair["cat"]["image_id"]]["score"],
                               clean[pair["dog"]["image_id"]]["score"])
        side_results = {}
        for side, other_side in (("cat", "dog"), ("dog", "cat")):
            image_id, other_id = pair[side]["image_id"], pair[other_side]["image_id"]
            own, other, pixels = clean[image_id], clean[other_id], pixel_by_id[image_id]
            donor = donor_rows[(pair_id, side)]["donor"]
            donor_id = None if donor is None else donor["donor_image_id"]
            if donor_id is not None and (donor_id not in selected_ids or donor_id == image_id):
                raise ValueError("same-species donor leaves the split")
            same = None if donor_id is None else clean[donor_id]["early"]
            dose, requested = prepare_dose(own["early"], other["early"], inside, outside, pair_id, side, same)
            info = {"image_id": image_id, "baseline": own["score"], "dose": dose,
                    "same_species_donor": donor,
                    "same_species_baseline": None if donor_id is None else clean[donor_id]["score"],
                    "conditions": {}}
            for condition in ("repeat", "post_clean"):
                if condition == "post_clean":
                    continue
                logits, caps, row = forward(model, pixels, evidence, f"{pair_id}:{side}:{condition}",
                                            reference=own["off_reference"])
                if not all((_same(logits, own["logits"]), _same(caps["early"], own["early"]),
                            _same(caps["late"], own["late"]))):
                    raise ValueError("clean repeat failed")
                info["conditions"][condition] = row["score"]
            plans = [
                ("null_inside", EARLY, inside, None, "null", own["logits"]),
                ("null_outside", EARLY, outside, None, "null", own["logits"]),
                ("self_inside", EARLY, inside, own["early"][0, inside], "patch", own["logits"]),
                ("self_outside", EARLY, outside, own["early"][0, outside], "patch", own["logits"]),
                ("full_self", EARLY, None, own["early"], "patch", own["logits"]),
                ("full_swap", EARLY, None, other["early"], "patch", other["logits"]),
                ("natural_inside", EARLY, inside, other["early"][0, inside], "patch", None),
            ]
            if dose["status"] == "matchable":
                for key, indices in (("inside", inside), ("outside", outside), ("random", inside),
                                     ("same_species", inside)):
                    if key in requested:
                        plans.append(("matched_" + key, EARLY, indices, requested[key], "patch", None))
            plans.append(("final_inside", LATE, inside, other["late"][0, inside], "patch", own["logits"]))
            for name, path, tokens, replacement, action, expected_logits in plans:
                logits, caps, row = forward(model, pixels, evidence, f"{pair_id}:{side}:{name}",
                                            path=path, tokens=tokens, replacement=replacement,
                                            action=action, reference=own["off_reference"])
                own_state = own["early"] if path == EARLY else own["late"]
                check_landing(caps, own_state, tokens, replacement, action=action)
                if expected_logits is not None and not _same(logits, expected_logits):
                    raise ValueError(f"{name} logits control failed")
                saved = dict(row["score"])
                saved["raw_margin_change"] = saved["margin"] - own["score"]["margin"]
                saved["annotation_oriented_change"] = ((1 if side == "cat" else -1)
                                                       * saved["raw_margin_change"])
                if name.startswith("matched_"):
                    post = caps["after_all"][0, tokens]
                    pre = caps["before_all"][0, tokens]
                    norm = distance(post, pre)
                    if norm <= 0 or abs(norm - dose["q"]) > dose["tolerance"]:
                        raise ValueError("actual matched dose differs from q")
                    saved["actual_dose"] = norm
                    saved["token_doses"] = [distance(post[i], pre[i]) for i in range(4)]
                    saved["receiver_norm"] = distance(pre, torch.zeros_like(pre))
                    saved["q_over_receiver_norm"] = (dose["q"] / saved["receiver_norm"]
                                                     if saved["receiver_norm"] else None)
                    if name != "matched_inside":
                        inside_norm = info["conditions"]["matched_inside"]["actual_dose"]
                        if abs(norm - inside_norm) > .01 * inside_norm:
                            raise ValueError("actual matched doses differ")
                info["conditions"][name] = saved
            logits, caps, row = forward(model, pixels, evidence, f"{pair_id}:{side}:post_clean",
                                        reference=own["off_reference"])
            if not all((_same(logits, own["logits"]), _same(caps["early"], own["early"]),
                        _same(caps["late"], own["late"]))):
                raise ValueError("post-clean control failed")
            info["conditions"]["post_clean"] = row["score"]
            side_results[side] = info
        dose_statuses = {s: side_results[s]["dose"]["status"] for s in ("cat", "dog")}
        pair_row = {"pair_id": pair_id, "split": pair["split"], "baseline_eligibility": base,
                    "dose_matchability": dose_statuses, "sides": side_results}
        if not base["eligible"]:
            pair_row["status"] = "baseline-ineligible"
        elif any(value != "matchable" for value in dose_statuses.values()):
            pair_row["status"] = ("dose-ineligible" if "dose-ineligible" in dose_statuses.values()
                                  else "dose-unmatchable")
        else:
            movements = {}
            for side in ("cat", "dog"):
                values = side_results[side]
                sign = 1 if side == "cat" else -1
                movements[side] = {site: sign * (values["conditions"]["matched_" + site]["margin"]
                                                - values["baseline"]["margin"])
                                   for site in ("inside", "outside")}
            gap = base["gap"]
            passed = all(movements[s]["inside"] >= .02 * gap
                         and movements[s]["inside"] >= abs(movements[s]["outside"]) + .01 * gap
                         for s in ("cat", "dog"))
            pair_row.update(status="positive" if passed else "negative", movements=movements)
        evidence.json(f"{pair_id}.json", pair_row)
        results.append(pair_row)
    return results
