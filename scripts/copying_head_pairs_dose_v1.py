"""Public CPU pairwise copying study at fixed doses; not a blind test."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback

if __name__ == "__main__" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("set CUDA_VISIBLE_DEVICES=-1 before starting")
if __name__ == "__main__" and "torch" in sys.modules:
    raise RuntimeError("torch must not be imported before the CPU backend environment snapshot")
ENV_NAMES = ("MKL_CBWR", "ATEN_CPU_CAPABILITY")
ENV_AT_IMPORT = {name: {"present": name in os.environ, "value": os.environ.get(name)} for name in ENV_NAMES}
EXPECTED_ENV = {name: {"present": False, "value": None} for name in ENV_NAMES}
if __name__ == "__main__" and ENV_AT_IMPORT != EXPECTED_ENV:
    raise RuntimeError("CPU backend environment variables must be absent before importing torch")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.distilgpt2_intervention_dev_bootstrap import _guard_cuda_and_network  # noqa: E402

GUARD_EVENTS = []
SAFE_QUERIES = {"cuda_is_available": 0, "cuda_device_count": 0}
if __name__ == "__main__":
    _guard_cuda_and_network(torch, GUARD_EVENTS, SAFE_QUERIES)

from scripts import copying_head_factorial_v1 as prior  # noqa: E402
from looking_glass.copy_head_intervention import assert_no_forward_hooks  # noqa: E402
from looking_glass.copy_head_factorial import (  # noqa: E402
    EvidenceProjectionWitness, binary64_rounding_probe, head_slice, statistics,
)
from looking_glass.copy_head_pairs_dose_v1 import (  # noqa: E402
    ACTION_HEADS, ARMS, HEADS, PAIRS, PairProjectionWitness, action_order,
    joint_achieved, plan_family, projection_diagnostics, summarize,
)

base = prior.base
base.GUARD_EVENTS = GUARD_EVENTS
PROTOCOL_PATH = "docs/DISTILGPT2_COPY_HEAD_PAIRS_DOSE_V1.md"
PROTOCOL_REVISION = "8be210ce88a536dc9efcd769e40a8175b9beddfa"
PROTOCOL_SHA256 = "811c18baef640e08ccff7f6cade26d041672edda969ddbc7a6babdb60d70bddb"
CLARIFICATION_PATH = "docs/DISTILGPT2_COPY_PAIRS_DOSE_NUMERIC_CLARIFICATION_V1.md"
CLARIFICATION_REVISION = "f5fe8441ff34d4eb2913ac2e369db92c3dbe21f1"
CLARIFICATION_SHA256 = "ecd382e00656eb3c9245051b9aa4463a3223fb168987cb8e2bb05f643cd4dbc6"
CASE_PATH = "experiments/distilgpt2_copy_head_pairs_dose_v1.json"
CASE_SHA256 = "3e5d40fff832a6976f5e4f8d3bda916ab203ed93936463a1a0726e88dcab1106"
BASELINE_VERSION = "distilgpt2-development-exposures-v13"
BASELINE_SHA256 = "27ffe8d2ab312ccf2a62a07dab835d3cafce7364ca8abadbb1e73ec835e9aca8"
RESERVATION_VERSION = "distilgpt2-development-exposures-v15"
RESERVATION_SHA256 = "6c6ed34755dd9daebdde9a2de07f65c0c5d20301656416d873f321fa2c146d34"
RESERVATION_MODULE = "pretrained_exposures_v15"
PRIOR_ROOT = "results/distilgpt2-copy-head-factorial-v1-20261028"
PRIOR_RESULT_SHA256 = "7f62ed7a014978f71c09ed2e5973d1ce38f544eff0f2a1652cc9fd75e2fb125b"
PRIOR_SUPERVISOR_SHA256 = "35626526739498d468c9e91d1c4f39e3f79603cd9015fc7fbd386ba3917f086d"
PRIOR_SOURCE = "5e4a82baed62016950a6f0fd0f6487b3f7d3b698"
RAW_BYTES = 167_696_640
MAX_RAW_BYTES = 170_000_000
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_JOURNAL_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024
MAX_FAILURE_BYTES = 2 * 1024 * 1024
MAX_FORWARD_BYTES = base.LOGITS_BYTES + 3 * base.WITNESS_BYTES
SOURCES = tuple(dict.fromkeys((*prior.SOURCES,
    "scripts/copying_head_pairs_dose_v1.py", "looking_glass/copy_head_pairs_dose_v1.py",
    "tests/test_copy_head_pairs_dose_v1.py", "tests/test_copying_head_pairs_dose_v1.py",
    "looking_glass/pretrained_exposures_v13.py", "looking_glass/pretrained_exposures_v14.py",
    "looking_glass/pretrained_exposures_v15.py", PROTOCOL_PATH, CASE_PATH, CLARIFICATION_PATH,
    "experiments/distilgpt2_recipient_screen_v1.json",
    PRIOR_ROOT + "/run/result.json", PRIOR_ROOT + "/supervisor.json")))


def environment():
    current = {name: {"present": name in os.environ, "value": os.environ.get(name)} for name in ENV_NAMES}
    if ENV_AT_IMPORT != EXPECTED_ENV or current != ENV_AT_IMPORT:
        raise RuntimeError("CPU backend environment changed or was not frozen before torch import")
    return current


def runtime(model):
    binary64_rounding_probe()
    return {**base.runtime(model), "cpu_backend_environment": environment()}


def source_identity():
    clean_env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    clean_env.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_CONFIG_NOSYSTEM': '1',
                      'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_TERMINAL_PROMPT': '0'})
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, timeout=30, env=clean_env)
    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    tree = git("show", "-s", "--format=%T", revision).decode("ascii").strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("source checkout is not clean")
    sources = {}
    for path in SOURCES:
        data = (ROOT / path).read_bytes()
        if git("show", f"{revision}:{path}") != data:
            raise RuntimeError(f"source differs from the commit: {path}")
        sources[path] = base.sha(data)
    return {"revision": revision, "tree": tree, "sha256": sources}


def verify_previous():
    prior_info = prior.verify_previous()
    directory = ROOT / PRIOR_ROOT
    result_raw = (directory / "run/result.json").read_bytes()
    receipt_raw = (directory / "supervisor.json").read_bytes()
    result, receipt = json.loads(result_raw), json.loads(receipt_raw)
    if (base.sha(result_raw) != PRIOR_RESULT_SHA256 or base.sha(receipt_raw) != PRIOR_SUPERVISOR_SHA256
            or receipt.get("schema") != "copying-head-factorial-process-observation-v1"
            or receipt.get("source_revision") != PRIOR_SOURCE or receipt.get("interrupted") is not False
            or receipt.get("process_exit_code") != 0 or receipt.get("failure_file_present") is not False
            or receipt.get("run_sha256", {}).get("result.json") != PRIOR_RESULT_SHA256
            or result.get("status") != "complete" or result.get("submitted_forwards") != 224
            or result.get("model_call_returns_observed") != 224 or result.get("returned_forwards") != 224):
        raise RuntimeError("prior public factorial is not the fixed complete result")
    return {"earlier": prior_info, "factorial_result_sha256": PRIOR_RESULT_SHA256,
            "factorial_supervisor_sha256": PRIOR_SUPERVISOR_SHA256, "factorial_source": PRIOR_SOURCE,
            "prior_mean_contrast_by_arm": result["summary"]["mean_contrast_by_arm"]}


def inventory():
    if (CASE_SHA256 is None or RESERVATION_SHA256 is None or RESERVATION_VERSION is None
            or RESERVATION_MODULE is None or CLARIFICATION_SHA256 is None or CLARIFICATION_REVISION is None):
        raise RuntimeError("the cases, numeric clarification and reviewed reservation are not all frozen")
    protocol, data = (ROOT / PROTOCOL_PATH).read_bytes(), (ROOT / CASE_PATH).read_bytes()
    clarification = (ROOT / CLARIFICATION_PATH).read_bytes()
    if (base.sha(protocol) != PROTOCOL_SHA256 or base.sha(data) != CASE_SHA256
            or base.sha(clarification) != CLARIFICATION_SHA256):
        raise RuntimeError("the frozen pair protocol, cases or numeric clarification changed")
    return validate_cases(json.loads(data))


def prior_copy_tokens():
    """Read the four previously published copying groups without model work."""
    tokens = {token for _, values in base.GROUPS for token in values}
    for filename in ("distilgpt2_copying_head_eval_v1.json", "distilgpt2_copy_head_followup_v2.json",
                     "distilgpt2_copy_head_factorial_v1.json"):
        data = json.loads((ROOT / "experiments" / filename).read_bytes())
        tokens.update(token for group in data["groups"] for token in group["token_ids"])
    return tokens


def validate_cases(data):
    """Validate the published case schema and expand its groups exactly."""
    expected_arms = [{"layer": 3, "heads": list(ACTION_HEADS[name]), "position": 10,
                      "dose": name.rsplit('_', 1)[1]} for name in ARMS]
    recipient_check = [{"source_revision": "32164cd570dea1f4bc0022c5b7f81fe17f8c2797",
        "case_file": "experiments/distilgpt2_recipient_screen_v1.json",
        "sha256": "df9036ece115a42851a43f2d7833865fa1258ab542e322beff16d549df27d345",
        "input_rows_checked": 64,
        "comparison": "bidirectional complete-row and all candidate-prefix checks, including both appended public targets",
        "collisions_found": False}]
    if (type(data) is not dict or data.get("schema") != "distilgpt2-copy-head-pairs-dose-cases-v1"
            or data.get("model_id") != base.MODEL_ID or data.get("model_revision") != base.MODEL_REVISION
            or data.get("protocol_revision") != PROTOCOL_REVISION or data.get("protocol_sha256") != PROTOCOL_SHA256
            or data.get("known_exposure_version") != BASELINE_VERSION
            or data.get("known_exposure_sha256") != BASELINE_SHA256
            or data.get("tokenizer_sha256") != "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
            or data.get("selection_rule") != "First six eligible words in each ordered public pool: one spaced token, no earlier copying token or selected ID, and no one-token prefix in v13; tokenizer only, no model forward."
            or data.get("additional_public_exposure") != recipient_check
            or data.get("arms") != expected_arms or data.get("expected_forwards") != 576
            or data.get("raw_tensor_bytes") != RAW_BYTES
            or data.get("rotation") != "(4 * zero_based_family + variant) % 15, left"
            or data.get("baseline_order") != "four variants: clean then noop; after family dose calculation, each variant runs fifteen rotated arms then clean_after"
            or data.get("permutations") != [list(row) for row in base.PERMUTATIONS]):
        raise ValueError("case inventory does not match the fixed pair study")
    groups, cases = data.get("groups"), data.get("cases")
    if type(groups) is not list or len(groups) != 8 or type(cases) is not list or len(cases) != 32:
        raise ValueError("expected eight groups and 32 cases")
    seen = []
    for group_index, group in enumerate(groups):
        if type(group) is not dict or set(group) != {"words", "token_ids"}:
            raise ValueError("malformed group")
        words, tokens = group["words"], group["token_ids"]
        if (type(words) is not list or len(words) != 6 or any(type(w) is not str or not w or len(w) > 80 for w in words)
                or type(tokens) is not list or len(tokens) != 6
                or any(type(t) is not int or not 0 <= t < base.VOCAB for t in tokens)):
            raise ValueError("malformed group words or token IDs")
        seen.extend(tokens)
        for variant, permutation in enumerate(base.PERMUTATIONS):
            indices = permutation + (0, 1, 2, 3, 4)
            expected = {"case_id": f"copy-pairs-{group_index + 1:02d}-{variant}",
                        "group": group_index + 1, "variant": variant,
                        "text": "".join(" " + words[index] for index in indices),
                        "token_ids": [tokens[index] for index in indices], "target_id": tokens[5]}
            if cases[4 * group_index + variant] != expected:
                raise ValueError("case differs from its group")
    if len(set(seen)) != 48 or set(seen) & prior_copy_tokens():
        raise ValueError("case token IDs must be new and mutually distinct")
    proposed = [row["token_ids"] + [row["target_id"]] for row in cases]
    if base.sha(json.dumps(proposed, separators=(',', ':')).encode('ascii')) != data.get('ordered_input_plus_target_sha256'):
        raise ValueError('ordered input-plus-target hash differs')
    return cases


def check_exposure(cases):
    if RESERVATION_SHA256 is None or RESERVATION_VERSION is None or RESERVATION_MODULE is None:
        raise RuntimeError("the reviewed reservation is not bound")
    import importlib
    from looking_glass import pretrained_exposures_v13 as baseline
    reservation = importlib.import_module('looking_glass.' + RESERVATION_MODULE)
    if (baseline.INVENTORY_VERSION != BASELINE_VERSION or baseline.inventory_digest() != BASELINE_SHA256
            or reservation.INVENTORY_VERSION != RESERVATION_VERSION
            or reservation.inventory_digest() != RESERVATION_SHA256):
        raise RuntimeError("exposure inventory or reservation changed")
    inputs = tuple(tuple(row["token_ids"]) for row in cases)
    proposed = tuple(tuple((*row["token_ids"], row["target_id"])) for row in cases)
    records = reservation.inventory_payload()['records']
    own = [row for row in records if row['record_id'].startswith('planned-copy-pairs-dose-')]
    other = [row for row in records if not row['record_id'].startswith('planned-copy-pairs-dose-')]
    expected_own = {
        f"planned-copy-pairs-dose-{case['group']:02d}-{case['variant']}":
            (tuple(case['token_ids']), case['target_id'])
        for case in cases
    }
    if (len(own) != 32 or len(expected_own) != 32
            or any(row.get('category') != 'planned' for row in own)
            or {row['record_id']: (tuple(row['token_ids']), row.get('target_token_id')) for row in own}
               != expected_own):
        raise RuntimeError('reservation does not bind all and only the fixed rows and targets')
    old = [tuple(row['token_ids']) for row in other]
    old.extend(tuple((*row['token_ids'], row['target_token_id'])) for row in other
               if type(row.get('target_token_id')) is int)
    recipient_raw = (ROOT / 'experiments/distilgpt2_recipient_screen_v1.json').read_bytes()
    if base.sha(recipient_raw) != 'df9036ece115a42851a43f2d7833865fa1258ab542e322beff16d549df27d345':
        raise RuntimeError('public recipient cases changed')
    recipient = json.loads(recipient_raw)['cases']
    if type(recipient) is not list or len(recipient) != 64:
        raise RuntimeError('public recipient case count differs')
    old.extend(tuple(row['input_ids']) for row in recipient)
    old.extend(tuple((*row['input_ids'], row[key])) for row in recipient for key in ('giver_id', 'recipient_id'))
    prefixes = {row[:length] for row in old for length in range(1, len(row) + 1)}
    if (any(row[:length] in prefixes for row in proposed for length in range(1, len(row) + 1))
            or any(left[:len(right)] == right or right[:len(left)] == left
                   for index, left in enumerate(proposed) for right in proposed[index + 1:])
            or any(row[:len(known)] == known or known[:len(row)] == row for row in proposed for known in old)):
        raise RuntimeError("pair-study rows overlap a known or new prefix")
    if (inputs != reservation.PAIRS_DOSE_ROWS
            or tuple(row["target_id"] for row in cases)
            != tuple(reservation.PAIRS_DOSE_GROUPS[index // 4][5] for index in range(32))):
        raise RuntimeError("cases differ from the reviewed row reservation")
    return {"v13": BASELINE_SHA256, "reservation": RESERVATION_SHA256,
            "reservation_version": RESERVATION_VERSION}


def available_payload(result, witness):
    """Copy only safely bounded returned logits and available witness tensors."""
    payload = {}
    try:
        logits = result.logits if result is not None else None
        if (type(logits) is torch.Tensor and logits.device.type == "cpu" and logits.dtype == torch.float32
                and tuple(logits.shape) == (1, 11, base.VOCAB)):
            payload["logits"] = base._raw(logits[0, -1], base.LOGITS_BYTES)
    except Exception:
        pass
    for name, attribute in (("cproj_before", "before"), ("cproj_after", "after"), ("cproj_output", "output")):
        try:
            tensor = getattr(witness, attribute, None)
            if (type(tensor) is torch.Tensor and tensor.device.type == "cpu" and tensor.dtype == torch.float32
                    and tuple(tensor.shape) == (1, 11, 768)):
                payload[name] = base._raw(tensor, base.WITNESS_BYTES)
        except Exception:
            pass
    return payload


def run_forward(model, ids, action, *, expected_before=None, deliveries=None, observation=None):
    if observation is None:
        observation = {}
    observation.update({"model_call_return_observed": False, "payload": {}})
    assert_no_forward_hooks(model)
    if action not in ("clean", "noop", "clean_after", *ARMS):
        raise ValueError("unknown action")
    projection = model.transformer.h[3].attn.c_proj
    witness = None
    try:
        with torch.inference_mode():
            if action in ("clean", "clean_after"):
                result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                               output_attentions=False, output_hidden_states=False, return_dict=True)
                observation["model_call_return_observed"] = True
                observation["payload"] = available_payload(result, None)
            else:
                witness = (EvidenceProjectionWitness(projection, sequence_length=11, width=768, heads=12, mode="noop")
                           if action == "noop" else
                           PairProjectionWitness(projection, expected_before=expected_before, deliveries=deliveries))
                with witness:
                    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                   output_attentions=False, output_hidden_states=False, return_dict=True)
                    observation["model_call_return_observed"] = True
                    observation["payload"] = available_payload(result, witness)
    except BaseException:
        observation["payload"].update(available_payload(None, witness))
        raise
    assert_no_forward_hooks(model)
    logits = result.logits
    if (result.past_key_values is not None or result.attentions is not None or result.hidden_states is not None
            or type(logits) is not torch.Tensor or logits.device.type != "cpu" or logits.dtype != torch.float32
            or tuple(logits.shape) != (1, 11, base.VOCAB)):
        raise RuntimeError("model returned invalid logits")
    if not torch.isfinite(logits).all().item():
        raise RuntimeError('nonfinite full logits')
    payload = observation["payload"]
    expected_names = {"logits"} if witness is None else {"logits", "cproj_before", "cproj_after", "cproj_output"}
    if set(payload) != expected_names:
        raise RuntimeError("bounded model or witness payload is incomplete")
    if sum(map(len, payload.values())) > MAX_FORWARD_BYTES:
        raise RuntimeError("retained forward payload exceeds its limit")
    return payload


def dose_json(plan):
    return {"m": plan.m, "requested_l2": dict(plan.requested), "variants": [
        {dose: {str(head): {"source_l2": row[dose][head].norm,
             "requested_fraction": row[dose][head].fraction,
             "achieved_l2": row[dose][head].achieved,
             "changed_float32_elements": row[dose][head].changed_elements,
             "changed_binary_bits": row[dose][head].changed_bits}
            for head in HEADS} for dose in ('low', 'match', 'high')}
        for row in plan.variants]}


def append_event(stream, event):
    if stream.tell() + len(base.canonical(event)) + 1 > MAX_JOURNAL_BYTES:
        raise RuntimeError('attempt journal exceeds its metadata budget')
    base._append_event(stream, event)


def bounded_json(path, value, limit):
    if len(base.canonical(value)) > limit:
        raise RuntimeError('JSON evidence exceeds its metadata budget')
    return base.durable_json(path, value)


def persist_action(log, raw, case, action, payload, clean_logits, noop_before, postcheck, marker,
                   intervention=None):
    offsets = base._write_payload(raw, payload)
    append_event(log, {"event": "raw_recorded", "case_id": case["case_id"], "action": action, "raw": offsets})
    marker["recorded"] = True
    for name, data in payload.items():
        if any((value[0] & 0x7f800000) == 0x7f800000 for value in struct.iter_unpack("<I", data)):
            raise RuntimeError(f"nonfinite {name} in returned payload")
    postcheck()
    if action in ("noop", "clean_after") and payload["logits"] != clean_logits:
        raise RuntimeError("clean, no-op and clean-after logits differ")
    if action == "noop" and payload["cproj_before"] != payload["cproj_after"]:
        raise RuntimeError("no-op projection bits changed")
    if action in ARMS and payload["cproj_before"] != noop_before:
        raise RuntimeError("arm input differs from the no-op")
    record = {"case_id": case["case_id"], "group": case["group"], "variant": case["variant"],
              "target_id": case["target_id"], "action": action,
              "score": statistics(payload["logits"], case["target_id"]),
              "intervention": intervention, "raw": offsets}
    append_event(log, {"event": "complete", **record})
    return record


def persist_partial(log, path, case_id, action, observation, marker):
    """Retain available bounded bytes when a helper fails before returning."""
    partial = observation.get("payload", {})
    if not partial:
        return 0
    with path.open("xb" if action == "clean" else "ab") as raw:
        offsets = base._write_payload(raw, partial)
    append_event(log, {"event": "raw_recorded_partial", "case_id": case_id, "action": action,
                             "model_call_return_observed": observation.get("model_call_return_observed", False), "raw": offsets})
    marker["partial"] = True
    return sum(map(len, partial.values()))


def verify_tokenization(tokenizer, data, cases):
    """Check fixed inputs, target continuation, and outcome-independent pool selection."""
    from looking_glass import pretrained_exposures_v13 as baseline
    old_first = {row["token_ids"][0] for row in baseline.inventory_payload()["records"] if row["token_ids"]}
    prior_tokens = prior_copy_tokens()
    pools = data.get("candidate_word_pools")
    if type(pools) is not list or len(pools) != 8:
        raise ValueError("expected eight public candidate pools")
    used = set()
    for group, pool in zip(data["groups"], pools):
        if type(pool) is not list or any(type(word) is not str or not word or len(word) > 80 for word in pool):
            raise ValueError("malformed public candidate pool")
        words, ids = [], []
        for word in pool:
            tokens = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
            if len(tokens) != 1 or tokens[0] in old_first or tokens[0] in prior_tokens or tokens[0] in used:
                continue
            words.append(word); ids.append(tokens[0]); used.add(tokens[0])
            if len(words) == 6:
                break
        if group != {"words": words, "token_ids": ids} or len(ids) != 6:
            raise RuntimeError("candidate pool selection does not reproduce the frozen group")
    for case in cases:
        target_word = data["groups"][case["group"] - 1]["words"][5]
        if (tokenizer(case["text"], add_special_tokens=False)["input_ids"] != case["token_ids"]
                or tokenizer(case["text"] + " " + target_word, add_special_tokens=False)["input_ids"]
                != case["token_ids"] + [case["target_id"]]):
            raise RuntimeError("pinned tokenizer does not reproduce the input and target")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (not sys.dont_write_bytecode or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()
            or any(Path(sys.pycache_prefix).iterdir()) or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise RuntimeError("use -B, a fresh absolute -X pycache_prefix and offline flags")
    cases = inventory()
    source = source_identity()
    cpu_environment, dependencies = environment(), base.dependency_identity()
    previous, exposure = verify_previous(), check_exposure(cases)
    if (32 * (18 * base.LOGITS_BYTES + 48 * base.WITNESS_BYTES) != RAW_BYTES or RAW_BYTES > MAX_RAW_BYTES
            or MAX_MANIFEST_BYTES + MAX_JOURNAL_BYTES + MAX_RESULT_BYTES + MAX_FAILURE_BYTES > MAX_METADATA_BYTES):
        raise RuntimeError("declared raw payload limit is inconsistent")
    output = args.output_dir
    if not output.is_absolute() or not args.cache_dir.is_absolute():
        raise ValueError("cache and output paths must be absolute")
    parent = output.parent.resolve(strict=True)
    if ROOT == parent or ROOT in parent.parents or output.exists():
        raise ValueError("output must be a new directory outside the checkout")
    if shutil.disk_usage(parent).free < MAX_RAW_BYTES + MAX_METADATA_BYTES:
        raise OSError("insufficient space for the declared raw payload")
    initial_probe = base.numeric_policy(initialize=True)
    binary64_rounding_probe()
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    output.mkdir()
    manifest = {"schema": "distilgpt2-copy-head-pairs-dose-run-v1", "source": source,
                "dependencies": dependencies, "protocol_sha256": PROTOCOL_SHA256, "cases_sha256": CASE_SHA256,
                "numeric_clarification": {"revision": CLARIFICATION_REVISION, "sha256": CLARIFICATION_SHA256},
                "exposure_inventories": exposure, "previous": previous, "cases": cases,
                "arms": ARMS, "rotation": "(4 * zero_based_family + variant) % 15, left",
                "model_id": base.MODEL_ID, "model_revision": base.MODEL_REVISION,
                "weights_sha256": base.WEIGHTS_SHA256, "raw_payload_bytes": RAW_BYTES,
                "max_raw_bytes": MAX_RAW_BYTES, "max_retained_bytes_per_forward": MAX_FORWARD_BYTES,
                "metadata_limits": {"aggregate_bytes": MAX_METADATA_BYTES, "manifest_bytes": MAX_MANIFEST_BYTES,
                    "journal_bytes": MAX_JOURNAL_BYTES, "result_bytes": MAX_RESULT_BYTES,
                    "failure_bytes": MAX_FAILURE_BYTES},
                "retained_copy_accounting": {
                    "raw_stream_bytes_per_case": 18 * base.LOGITS_BYTES + 48 * base.WITNESS_BYTES,
                    "four_variant_cache_byte_arrays_upper": 4 * (base.LOGITS_BYTES + 32 * base.WITNESS_BYTES),
                    "planned_source_and_delivery_slice_bytes_upper": (12 + 36) * 64 * 4,
                    "one_returned_payload_byte_arrays_upper": MAX_FORWARD_BYTES,
                    "one_full_logit_tensor_bytes": 11 * base.LOGITS_BYTES,
                    "three_projection_witness_tensor_clones_bytes": 3 * base.WITNESS_BYTES,
                    "scope": "Enumerated logical payloads and selected live copies, not a process RSS limit; model state, activations, allocator pools, Python/JSON overhead, transient copies, stdout and later physical evidence copies are excluded. Free disk is checked for one raw run and the declared metadata budgets; later copies are accounted separately."},
                "initial_numeric_probe": initial_probe, "cpu_backend_environment_at_torch_import": cpu_environment,
                "intended_runtime": {"device": "cpu", "dtype": "float32", "model_eval": True,
                                     "attention_implementation": "eager", "use_cache": False,
                                     "intraop_threads": 1, "interop_threads": torch.get_num_interop_threads(),
                                     "deterministic_algorithms": False, "mkldnn_enabled": torch.backends.mkldnn.enabled,
                                     "float32_matmul_precision": "highest", "cpu_rng_seed": base.CPU_RNG_SEED},
                "counter_semantics": {"submitted_forwards": "attempts entering the helper; an error can precede a model call",
                                      "model_call_returns_observed": "model returns observed by the helper; asynchronous interruption can make this uncertain",
                                      "returned_forwards": "complete helper payloads; a model call may return before helper validation fails",
                                      "payload_recorded": "complete helper payload and raw_recorded event were durably written",
                                      "partial_payload_recorded": "available bounded evidence from a failed helper was durably written"},
                "evidence_status": "public disjoint exploratory; not a blind held-out evaluation"}
    manifest_hash = bounded_json(output / "manifest.json", manifest, MAX_MANIFEST_BYTES)
    completed, family_doses, case_diagnostics = [], [], []
    submitted = model_returns = returned = retained = 0
    phase, case_id, action, payload_returned, model_call_return_observed = "model_load", None, None, False, False
    marker = {"recorded": False, "partial": False, "partial_save_error": None}
    try:
        bundle = base.load_distilgpt2(cache_dir=args.cache_dir, allow_download=False)
        if bundle.metadata["weights_sha256"] != base.WEIGHTS_SHA256:
            raise RuntimeError("checkpoint weight hash changed")
        case_data = json.loads((ROOT / CASE_PATH).read_bytes())
        verify_tokenization(bundle.tokenizer, case_data, cases)
        model = bundle.model
        torch.random.default_generator.manual_seed(base.CPU_RNG_SEED)
        model_before, buffers_before = base.state_hash(model), base.buffer_hash(model)
        runtime_before = runtime(model)
        assert_no_forward_hooks(model)
        with (output / "attempt.jsonl").open("xb") as log:
            append_event(log, {"event": "runtime_ready", "cpu_rng_seed": base.CPU_RNG_SEED,
                                     "actual_runtime": runtime_before})

            def perform(case, kind, cache, deliveries=None, intervention=None):
                nonlocal phase, case_id, action, payload_returned, model_call_return_observed, marker
                nonlocal submitted, model_returns, returned, retained
                case_id, action = case["case_id"], kind
                phase, payload_returned, model_call_return_observed = "forward", False, False
                marker = {"recorded": False, "partial": False, "partial_save_error": None}
                ids = torch.tensor([case["token_ids"]], dtype=torch.int64, device="cpu")
                append_event(log, {"event": "start", "case_id": case_id, "action": action})
                if runtime(model) != runtime_before:
                    raise RuntimeError("runtime changed before a forward")
                submitted += 1
                observation = {}
                try:
                    payload = run_forward(model, ids, kind, expected_before=cache.get("before"),
                                          deliveries=deliveries, observation=observation)
                except BaseException:
                    model_call_return_observed = observation.get("model_call_return_observed", False)
                    model_returns += int(model_call_return_observed)
                    partial = observation.get("payload", {})
                    if partial:
                        try:
                            if retained + sum(map(len, partial.values())) > RAW_BYTES:
                                raise RuntimeError("partial evidence exceeds the raw payload limit")
                            retained += persist_partial(log, output / (case_id + ".f32"),
                                                        case_id, action, observation, marker)
                        except Exception as save_error:
                            marker["partial_save_error"] = type(save_error).__name__
                    raise
                model_call_return_observed = observation["model_call_return_observed"]
                model_returns += int(model_call_return_observed)
                returned += 1
                payload_returned = True
                retained += sum(map(len, payload.values()))
                if retained > RAW_BYTES:
                    raise RuntimeError("aggregate raw payload exceeds its limit")
                phase = "persist_and_validate"

                def postcheck():
                    if ids.tolist() != [case["token_ids"]] or runtime(model) != runtime_before:
                        raise RuntimeError("input, runtime or CPU RNG changed after a forward")

                path = output / (case_id + ".f32")
                with path.open("xb" if kind == "clean" else "ab") as raw:
                    record = persist_action(log, raw, case, kind, payload, cache.get("clean"),
                                            cache.get("before"), postcheck, marker, intervention)
                completed.append(record)
                if kind == "clean":
                    cache["clean"] = payload["logits"]
                elif kind == "noop":
                    cache["before"] = payload["cproj_before"]
                    cache["outputs"] = {"noop": payload["cproj_output"]}
                elif kind in ARMS:
                    cache["outputs"][kind] = payload["cproj_output"]
                    cache.setdefault("after", {})[kind] = payload["cproj_after"]

            for family in range(8):
                family_cases = cases[4 * family:4 * family + 4]
                caches = [{} for _ in range(4)]
                for variant, case in enumerate(family_cases):
                    perform(case, "clean", caches[variant])
                    perform(case, "noop", caches[variant])
                phase, case_id, action = "family_dose", None, None
                payload_returned, model_call_return_observed = False, False
                marker = {"recorded": False, "partial": False, "partial_save_error": None}
                plan = plan_family(tuple(cache["before"] for cache in caches))
                description = {"group": family + 1, **dose_json(plan)}
                family_doses.append(description)
                append_event(log, {"event": "family_dose", **description})
                for variant, case in enumerate(family_cases):
                    cache, row = caches[variant], plan.variants[variant]
                    for kind in action_order(family, variant):
                        heads = ACTION_HEADS[kind]
                        dose = kind.rsplit('_', 1)[1]
                        details = {"heads": list(heads), "position": 10, "dose": dose,
                                   "requested_l2_each": plan.requested[dose],
                                   "components": {str(head): description['variants'][variant][dose][str(head)]
                                                  for head in heads}}
                        if len(heads) == 2:
                            details["joint_combined_achieved_l2"] = joint_achieved(row[dose], heads)
                            details["joint_combined_requested_l2"] = (plan.requested['match'] if dose == 'low'
                                                                      else math.sqrt(2.0) * plan.requested['high'])
                            if dose == 'low':
                                details['joint_minus_matching_single_l2'] = {
                                    str(head): details['joint_combined_achieved_l2'] - row['match'][head].achieved
                                    for head in heads}
                        perform(case, kind, cache, {head: row[dose][head].delivered for head in heads}, details)
                    perform(case, "clean_after", cache)
                    phase, case_id, action = "case_diagnostics", case["case_id"], None
                    payload_returned, model_call_return_observed = False, False
                    marker = {"recorded": False, "partial": False, "partial_save_error": None}
                    for dose in ('low', 'high'):
                        for first, second in PAIRS:
                            for head in (first, second):
                                if head_slice(cache['after'][f'h{first}_h{second}_{dose}'], head) != head_slice(cache['after'][f'h{head}_{dose}'], head):
                                    raise RuntimeError('joint delivery differs from its single-arm bits')
                    diagnostic = {"case_id": case["case_id"], **projection_diagnostics(cache["outputs"])}
                    case_diagnostics.append(diagnostic)
                    append_event(log, {"event": "case_diagnostics", **diagnostic})
        phase = "verification"
        if (len(completed) != 576 or submitted != 576 or model_returns != 576 or returned != 576 or retained != RAW_BYTES
                or base.state_hash(model) != model_before or base.buffer_hash(model) != buffers_before
                or runtime(model) != runtime_before or source_identity() != source or GUARD_EVENTS
                or torch.cuda.is_initialized()):
            raise RuntimeError("final count, model, runtime or source verification failed")
        assert_no_forward_hooks(model)
        result = {"schema": "distilgpt2-copy-head-pairs-dose-result-v1", "status": "complete",
                  "manifest_sha256": manifest_hash, "checkpoint": bundle.metadata,
                  "model_state_sha256_before": model_before, "model_state_sha256_after": base.state_hash(model),
                  "model_buffers_sha256_before": buffers_before, "model_buffers_sha256_after": base.buffer_hash(model),
                  "runtime": runtime_before, "guard_safe_queries": SAFE_QUERIES,
                  "submitted_forwards": submitted, "model_call_returns_observed": model_returns,
                  "returned_forwards": returned, "raw_payload_bytes": retained,
                  "family_doses": family_doses, "case_diagnostics": case_diagnostics,
                  "cases": completed, "summary": summarize(completed, cases)}
        bounded_json(output / "result.json", result, MAX_RESULT_BYTES)
    except BaseException as error:
        attempt = output / "attempt.jsonl"
        if attempt.is_file():
            try:
                with attempt.open("ab") as log:
                    append_event(log, {"event": "failed", "case_id": case_id, "action": action,
                                             "phase": phase, "error_type": type(error).__name__,
                                             "model_call_return_observed": model_call_return_observed, "payload_returned": payload_returned,
                                             "payload_recorded": marker["recorded"], "partial_payload_recorded": marker["partial"],
                                             "partial_save_error": marker["partial_save_error"]})
            except OSError:
                pass
        bounded_json(output / "failure.json", {"schema": "distilgpt2-copy-head-pairs-dose-failure-v1",
                         "status": "incomplete", "manifest_sha256": manifest_hash, "phase": phase,
                         "case_id": case_id, "action": action, "payload_returned": payload_returned,
                         "model_call_return_observed": model_call_return_observed, "payload_recorded": marker["recorded"],
                         "partial_payload_recorded": marker["partial"], "partial_save_error": marker["partial_save_error"],
                         "submitted_forwards": submitted, "model_call_returns_observed": model_returns,
                         "returned_forwards": returned, "completed_records": len(completed),
                         "admitted_payload_bytes": retained, "raw_files": base.raw_file_inventory(output, cases),
                         "error_type": type(error).__name__, "traceback_frames": [
                             {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                             for frame in traceback.extract_tb(error.__traceback__)]}, MAX_FAILURE_BYTES)
        raise
    print(json.dumps({"status": "complete", "forwards": submitted, "raw_payload_bytes": retained,
                      "summary": result["summary"]}, allow_nan=False))


if __name__ == "__main__":
    main()
