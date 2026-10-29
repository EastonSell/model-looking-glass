"""Independent stdlib audit of the closed ViT synthetic-injection saved records.

This reads the recorded bytes only. It does not import the study, PyTorch,
Transformers or a checkpoint, and does not authenticate execution or timing.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON number")
    return result


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_pairs,
                      parse_constant=reject_constant, parse_float=finite_float)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def floats(content):
    assert len(content) % 4 == 0
    values = struct.unpack("<" + "f" * (len(content) // 4), content)
    assert all(math.isfinite(x) for x in values)
    return values


def distance(left, right):
    assert len(left) == len(right)
    return math.sqrt(math.fsum((x-y)**2 for x, y in zip(floats(left), floats(right), strict=True)))


def equal_float(left, right):
    return type(left) in (float, int) and type(right) in (float, int) and math.isclose(
        left, right, rel_tol=1e-12, abs_tol=1e-14)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--source-observation", type=Path, required=True)
    args = parser.parse_args()
    root = args.attempt
    result = read_json(root / "completed.json")
    started = read_json(root / "started.json")
    loading = read_json(root / "load-observation.json")
    source_readback = read_json(args.source_observation)
    assert not (root / "incomplete.json").exists()
    assert result["status"] == "complete" and result["schema"] == "vit-synthetic-injection-results-v1"
    source = result["source"]
    assert started["source"] == source
    assert source["published_commit"] == source_readback["github_head"] == (
        "42a7ea04411f07210aaa9efc2e47956b16f4590c")
    assert source["files"] == {item["path"]: item["sha256"] for item in source_readback["source"]["files"]}
    assert source["model"] == "WinKawaks/vit-tiny-patch16-224"
    assert source["revision"] == "77d1485af66b34d4ed0fe95dbb0c60c7496f950b"
    assert source["assets"] == {
        "config.json": "7e5aeb1add0a832153071a858bb18145cafb809e54e7fea4e5dc1183c82078f5",
        "model.safetensors": "a9146f3ee7cf6de399190132627e4aef1aeddd13692f5b333f24629dd36c37b0",
        "preprocessor_config.json": "90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2"}
    assert loading == {name: [] for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    assert result["model_state_sha256_before"] == result["model_state_sha256_after"]
    raw = (root / "tensors.bin").read_bytes()
    assert len(raw) < 32 * 1024 * 1024
    index = read_json(root / "tensor-index.json")
    assert index["schema"] == "vit-synthetic-tensors-v1"
    entries, seen, cursor = {}, set(), 0
    for row in index["entries"]:
        assert row["name"] not in entries and type(row["offset"]) is int and row["offset"] == cursor
        assert row["dtype"] == "float32-le" and all(type(x) is int and x > 0 for x in row["shape"])
        assert type(row["length"]) is int and row["length"] == 4 * math.prod(row["shape"])
        chunk = raw[cursor:cursor + row["length"]]
        assert len(chunk) == row["length"] and digest(chunk) == row["sha256"]
        entries[row["name"]] = (row, chunk)
        cursor += row["length"]
    assert cursor == len(raw) == index["tensor_bytes"]

    def data(reference, shape):
        assert reference["shape"] == shape
        actual, content = entries[reference["name"]]
        assert actual == reference
        seen.add(reference["name"])
        return content

    width, sequence, labels = 192, 197, 1000
    vector_bytes = width * 4
    full_shape = [1, sequence, width]
    vec_shape = [width]
    logit_shape = [1, labels]
    tokens = (16, 52, 103, 166)
    seeds = (11, 12, 13, 14, 21, 22, 23, 24)
    targets = (16, 52, 103, 166, 103, 166, 16, 52)
    randoms = (166, 166, 52, 166)

    def vector(full, token):
        return full[token * vector_bytes:(token + 1) * vector_bytes]

    def plan(forward, injected_token, kind=None, patched_token=None):
        wanted = [("embedding_before", "capture", "vit.embeddings", None, None),
                  ("injection", "add", "vit.embeddings", injected_token, None),
                  ("embedding_after", "capture", "vit.embeddings", None, None),
                  ("block_before", "capture", "vit.encoder.layer.0", None, None)]
        if kind == "self":
            wanted.extend([("self_source", "capture", "vit.encoder.layer.0", patched_token, None),
                           ("patch", "patch", "vit.encoder.layer.0", patched_token, "self_source")])
        elif kind is not None:
            wanted.append(("patch", "patch", "vit.encoder.layer.0", patched_token, None))
        wanted.append(("block_after", "capture", "vit.encoder.layer.0", None, None))
        actual = forward["plan"]["steps"]
        assert len(actual) == len(wanted)
        prior = {}
        for order, (step, wanted_step) in enumerate(zip(actual, wanted, strict=True)):
            name, op, path, token, source_capture = wanted_step
            assert (step["name"], step["operation"], step["module_path"], step["token_index"],
                    step["source_capture"]) == wanted_step
            assert step["order"] == order and type(step["order"]) is int
            assert step["module_aliases"] == [path] and step["invocation"] == step["forward_index"] == 0
            assert type(step["invocation"]) is int and type(step["forward_index"]) is int
            assert step["call_site_known"] is False and step["output_selector"] == []
            assert step["output_shape"] == full_shape and step["dtype"] == "torch.float32" and step["device"] == "cpu"
            assert step["batch_index"] == (0 if token is not None else None)
            assert step["selected_shape"] == (vec_shape if token is not None else full_shape)
            assert step["preceding_same_site_actions"] == prior.get(path, [])
            prior.setdefault(path, []).append(name)
        assert forward["plan"]["source_bytes"] == vector_bytes * (1 + (kind is not None and kind != "self"))
        assert forward["plan"]["capture_bytes"] == 4 * sequence * vector_bytes + (vector_bytes if kind == "self" else 0)
        refs = forward["tensors"]
        before, after = data(refs["embedding_before"], full_shape), data(refs["embedding_after"], full_shape)
        pos = (injected_token * width + 7) * 4
        assert before[:pos] == after[:pos] and before[pos+4:] == after[pos+4:]
        assert after[pos:pos+4] == struct.pack("<f", floats(before[pos:pos+4])[0] + 4.0)
        assert before[pos:pos+4] != after[pos:pos+4]
        assert equal_float(forward["injection_l2"], distance(vector(before, injected_token), vector(after, injected_token)))
        return refs

    def pair(row, index, split):
        seed, target = seeds[index], targets[index]
        assert row["seed"] == seed and type(row["seed"]) is int
        assert row["injection_token"] == target and type(row["injection_token"]) is int
        ev = row["evidence"]
        pixels = b"".join(struct.pack("<f", (((17*x + 31*y + 47*c + 97*seed) % 257) - 128) / 128)
                          for c in range(3) for y in range(224) for x in range(224))
        assert digest(pixels) == ev["pixel_sha256"]
        geom = ev["geometry"]
        assert geom["pixel_shape"] == [1, 3, 224, 224] and geom["patch_shape"] == [16, 16]
        assert geom["grid_shape"] == [14, 14] and geom["sequence_length"] == sequence
        assert geom["patch_order"] == "row-major" and geom["cls_token_index"] == 0
        assert geom["box_coordinates"] == "processed-pixel-xyxy-exclusive"
        assert len(ev["clean_events"]) == len(ev["clean_vectors"]) == 5
        clean = {}
        for token, event, ref in zip((0, *tokens), ev["clean_events"], ev["clean_vectors"], strict=True):
            assert event["identity"]["run_id"] == f"{split}-{index if split == 'discovery' else index-4}/clean"
            assert event["token_positions"] == [token] and event["batch_positions"] == [0]
            assert event["module_path"] == "vit.encoder.layer.0" and event["module_kind"] == "vit_block"
            assert event["layer_index"] == event["invocation"] == event["forward_index"] == 0
            assert event["shape"] == full_shape and event["saved_shape"] == vec_shape
            assert event["dtype"] == event["saved_dtype"] == "torch.float32"
            assert event["device"] == event["saved_device"] == "cpu"
            clean[token] = data(ref, vec_shape)
        injected = ev["injected"]
        refs = plan(injected, target)
        block = data(refs["block_before"], full_shape)
        assert block == data(refs["block_after"], full_shape)
        clean_logits = data(ev["clean_logits"], logit_shape)
        injected_logits = data(refs["logits"], logit_shape)
        comparison = ev["comparison"]
        assert comparison["status"] == "compared" and comparison["causal_origin"] == "unknown"
        assert comparison["pairing_verified_by_caller"] is True and comparison["issues"] == []
        assert comparison["image_geometry"] == geom and len(comparison["comparisons"]) == 5
        for token, item in zip((0, *tokens), comparison["comparisons"], strict=True):
            a, b = clean[token], vector(block, token)
            diffs = [abs(x-y) for x,y in zip(floats(a),floats(b),strict=True)]
            assert item["token"] == token and item["feature_width"] == width
            assert item["module_path"] == "vit.encoder.layer.0" and item["module"] == "block"
            if token == 0:
                assert item["token_kind"] == "cls" and item["processed_pixel_box_xyxy"] is None
            else:
                row, col = divmod(token-1, 14)
                assert item["token_kind"] == "patch" and item["patch_row"] == row and item["patch_column"] == col
                assert item["processed_pixel_box_xyxy"] == [16*col, 16*row, 16*(col+1), 16*(row+1)]
            assert item["bitwise_changed_elements"] == sum(a[j:j+4] != b[j:j+4] for j in range(0, len(a), 4))
            assert item["numerically_changed_elements"] == sum(v != 0 for v in diffs)
            assert item["numeric_equality"] == ("different" if any(diffs) else "equal")
            assert equal_float(item["l2_difference"], math.sqrt(math.fsum(x*x for x in diffs)))
            assert equal_float(item["max_absolute_difference"], max(diffs))
            assert equal_float(item["mean_absolute_difference"], math.fsum(diffs)/width)
        delta = {token: tuple(x-y for x, y in zip(floats(vector(block, token)), floats(clean[token]), strict=True))
                 for token in tokens}
        return clean, block, delta, clean_logits, injected_logits

    fit_bytes = (root / "discovery-freeze.json").read_bytes()
    fit = read_json(root / "discovery-freeze.json")
    assert digest(fit_bytes) == result["discovery_freeze_sha256"]
    assert fit["source"] == source and fit["candidate_tokens"] == list(tokens)
    assert fit["random_choices"] == list(randoms) and fit["tie_break"] == "lowest token index"
    raw_direction = (root / "direction.f64le").read_bytes()
    assert len(raw_direction) == 8*width and digest(raw_direction) == fit["direction_sha256"]
    direction = struct.unpack("<" + "d" * width, raw_direction)
    assert all(math.isfinite(x) for x in direction) and any(x != 0 for x in direction)
    assert len(fit["discovery_cases"]) == 4 and fit["direction_length"] == width
    discovery, discovery_deltas = [], []
    for i, row in enumerate(fit["discovery_cases"]):
        assert row == read_json(root / f"discovery-{i}.json")
        deltas = pair(row, i, "discovery")[2]
        discovery_deltas.append(deltas)
        discovery.append(deltas[targets[i]])
    assert direction == tuple(math.fsum(row[j] for row in discovery)/4 for j in range(width))
    assert equal_float(fit["direction_l2"], math.sqrt(math.fsum(x*x for x in direction)))
    assert fit["state_sha256"] == result["model_state_sha256_before"]
    assert len(fit["discovery_scores"]) == 4
    for choice, delta in zip(fit["discovery_scores"], discovery_deltas, strict=True):
        scores = [(t, math.fsum(x*y for x,y in zip(direction,delta[t],strict=True))) for t in tokens]
        magnitudes = [(t, math.sqrt(math.fsum(x*x for x in delta[t]))) for t in tokens]
        assert choice["directional"] == min(scores, key=lambda item: (-item[1], item[0]))[0]
        assert choice["magnitude"] == min(magnitudes, key=lambda item: (-item[1], item[0]))[0]
        assert all(row[0] == expected[0] and equal_float(row[1], expected[1])
                   for row, expected in zip(choice["scores"], scores, strict=True))
        assert all(row[0] == expected[0] and equal_float(row[1], expected[1])
                   for row, expected in zip(choice["magnitudes"], magnitudes, strict=True))

    hits = magnitude_hits = random_hits = passes = eligible_count = 0
    summaries = []
    assert len(result["heldout"]) == 4
    for index, row in enumerate(result["heldout"]):
        i, target, random = index + 4, targets[index + 4], randoms[index]
        assert row == read_json(root / f"heldout-{index}.json")
        selection_file = read_json(root / f"heldout-{index}-selection.json")
        assert all(row[key] == value for key, value in selection_file.items())
        assert row["fitted_record_sha256"] == result["discovery_freeze_sha256"]
        clean, block, delta, clean_logits, injected_logits = pair(row, i, "heldout")
        scores = [(t, math.fsum(x*y for x,y in zip(direction,delta[t],strict=True))) for t in tokens]
        magnitudes = [(t, math.sqrt(math.fsum(x*x for x in delta[t]))) for t in tokens]
        selected = min(scores, key=lambda item: (-item[1], item[0]))[0]
        mag = min(magnitudes, key=lambda item: (-item[1], item[0]))[0]
        choice = row["selection"]
        assert choice["directional"] == selected and choice["magnitude"] == mag and row["random"] == random
        for actual, expected in zip(choice["scores"], scores, strict=True):
            assert actual[0] == expected[0] and equal_float(actual[1], expected[1])
        for actual, expected in zip(choice["magnitudes"], magnitudes, strict=True):
            assert actual[0] == expected[0] and equal_float(actual[1], expected[1])
        assert data(row["clean_repeat_logits"], logit_shape) == clean_logits
        assert [data(ref, vec_shape) for ref in row["clean_repeat_vectors"]] == [clean[t] for t in (0, *tokens)]
        assert len(row["clean_repeat_events"]) == 5
        for event, original in zip(row["clean_repeat_events"], row["evidence"]["clean_events"], strict=True):
            event = dict(event)
            identity = dict(event["identity"])
            assert identity["run_id"] == f"heldout-{index}/repeat"
            identity["run_id"] = original["identity"]["run_id"]
            event["identity"] = identity
            assert event == original
        hits += selected == target
        magnitude_hits += mag == target
        random_hits += random == target
        iv = row["interventions"]
        selected_dose = distance(vector(block, selected), clean[selected])
        initial_distance = distance(injected_logits, clean_logits)
        assert equal_float(selected_dose, iv["target_dose"])
        assert equal_float(initial_distance, iv["injected_logit_distance_from_clean"])
        off_target = min(t for t in tokens if t not in {selected, target})
        assert iv["off_target_token"] == off_target
        arm_distances = {}
        for name, arm in iv["arms"].items():
            assert arm == read_json(root / f"heldout-{index}-{name}.json")
            pt = off_target if name == "off_target" else selected
            refs = plan(arm["evidence"], target, name, pt)
            baseline = row["evidence"]["injected"]["tensors"]
            assert data(refs["embedding_before"], full_shape) == data(baseline["embedding_before"], full_shape)
            assert data(refs["embedding_after"], full_shape) == data(baseline["embedding_after"], full_shape)
            assert data(refs["block_before"], full_shape) == block
            after = data(refs["block_after"], full_shape)
            lo, hi = pt * vector_bytes, (pt + 1) * vector_bytes
            assert after[:lo] == block[:lo] and after[hi:] == block[hi:]
            if name == "target":
                assert after[lo:hi] == clean[selected]
            elif name == "self":
                assert data(refs["self_source"], vec_shape) == vector(block, selected)
                assert after == block and data(refs["logits"], logit_shape) == injected_logits
            elif name in ("reflected", "off_target"):
                before_vector, donor = vector(block, pt), clean[pt]
                natural = distance(before_vector, donor)
                assert natural > 0 and selected_dose > 0
                scale = -1.0 if name == "reflected" else selected_dose/natural
                expected = b"".join(struct.pack("<f", x+scale*(y-x)) for x,y in
                                    zip(floats(before_vector), floats(donor), strict=True))
                assert after[lo:hi] == expected
            else:
                raise AssertionError("unexpected intervention arm")
            actual_dose = distance(vector(block, pt), after[lo:hi])
            assert equal_float(actual_dose, arm["dose"])
            matched = selected_dose > 0 and actual_dose > 0 and abs(actual_dose/selected_dose - 1) <= 0.05
            assert arm["dose_matched"] == matched
            measured = distance(data(refs["logits"], logit_shape), clean_logits)
            assert equal_float(measured, arm["logit_distance_from_clean"])
            arm_distances[name] = measured
        eligible = (selected_dose > 0 and initial_distance > 0 and
                    all(name in iv["arms"] and iv["arms"][name]["dose_matched"]
                        for name in ("reflected", "off_target")))
        assert iv["eligible"] == eligible
        success = bool(eligible and arm_distances["target"] < initial_distance
                       and arm_distances["target"] < arm_distances["reflected"]
                       and arm_distances["target"] < arm_distances["off_target"])
        assert iv["prediction_pass"] == success
        eligible_count += eligible
        passes += success
        summaries.append({"seed": seeds[i], "injected_token": target, "selected_token": selected,
                          "magnitude_token": mag, "random_token": random, "eligible": eligible,
                          "initial_distance": initial_distance, "arm_distances": arm_distances,
                          "selected_dose": selected_dose, "passed": success})
    assert seen == set(entries)
    assert result["selection"]["directional_hits"] == hits
    assert result["selection"]["magnitude_hits"] == magnitude_hits
    assert result["selection"]["random_hits"] == random_hits
    assert result["selection"]["prediction"] == ("pass" if hits >= 3 and hits >= random_hits else "fail")
    assert result["restoration"]["eligible"] == eligible_count and result["restoration"]["passes"] == passes
    assert result["restoration"]["prediction"] == ("pass" if passes >= 3 else "inconclusive" if eligible_count < 4 else "fail")
    print(json.dumps({"status": "saved_record_arithmetic_consistent", "tensor_entries": len(entries),
                      "tensor_bytes": len(raw), "completed_sha256": digest((root / "completed.json").read_bytes()),
                      "directional_hits": hits, "magnitude_hits": magnitude_hits, "random_hits": random_hits,
                      "eligible_controls": eligible_count, "restoration_passes": passes, "cases": summaries,
                      "scope": "Saved bytes and declared records only; no execution, chronology, model identity or cleanup attestation."},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
