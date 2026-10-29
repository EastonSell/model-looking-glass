"""Independent standard-library audit of one closed BERT family GPU profile.

Run only after the supervising process is confirmed reaped:
    python audit.py RESULT_JSON RUN_DIRECTORY SOURCE_CHECKOUT [OBSERVATION_JSON [ASSET_DIRECTORY]]

This checks saved records, not execution authenticity or full-logit values.
The printed JSON deliberately omits local paths, raw GPU IDs, PIDs and errors.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, maximum):
    with path.open("rb") as f:
        data = f.read(maximum + 1)
    need(0 < len(data) <= maximum, "missing or oversized evidence")
    return data


def parse(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def finite(number):
        value = float(number)
        need(math.isfinite(value), "nonfinite JSON number")
        return value
    def bad(_):
        raise ValueError("nonfinite JSON literal")
    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite, parse_constant=bad)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def same(left, right):
    """Compare JSON structure and exact scalar types, including nested values."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


ASSETS = {
    "config.json": (483, "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83"),
    "model.safetensors": (267954768, "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"),
    "tokenizer.json": (466062, "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98"),
    "tokenizer_config.json": (48, "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"),
    "vocab.txt": (231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
}


# Pinned by independently hashing the 25 blobs at the stated Git revision.
SOURCES = {
    "docs/BERT_AGREEMENT_V1.md": "cee94657997100f5a76418a858388e79fa80fb14846287ade0d68a85e89c3d50",
    "docs/BERT_GPU_FAMILY_PROFILE_V1.md": "722813b514ad5aa7542c63f5561c20764b500487cf4fd8893ebc146fba94b648",
    "docs/BERT_GPU_STUDY_ADAPTER_DRAFT.md": "1a5263da35a2ca31d91041183d74ef29d5631ce32504ee69d24f3f8011ae97c0",
    "experiments/bert_agreement_public_cases_v1.json": "20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5",
    "looking_glass/__init__.py": "9e937f02ec3e4ce0908849aa1741d8afafe378e64212362dbf67af73e11a4fe0",
    "looking_glass/bert_agreement_audit.py": "177378de15373664dd12090b6812a9efd13c3c4b586ca8d7bb19a2ee3b3e4183",
    "looking_glass/bert_agreement_cases.py": "8fe4b55b02505510a779aa8ac9c6c28123dd0b101528bec7f50c3fc6c94f4a61",
    "looking_glass/bert_agreement_runner.py": "e777a2eb647a1a5c024b369be0dc8941df664a7ee02958bcb3dec6f3f617739b",
    "looking_glass/bert_agreement_study.py": "38642f2f51eb74fba1d0156d530d1e6c88a75a923a64ba30d5a3a1d715d2abc4",
    "looking_glass/bert_gpu_calibration.py": "7ffe65e95c909438fe83ffff96f3f50a9e16b0e205176ad32711d572a8e175b9",
    "looking_glass/bert_gpu_evidence.py": "08a84c4eee1910f2322687f548501cc3eb0e39daa2714b36a26cf5f85ced3b5e",
    "looking_glass/bert_gpu_family_controller.py": "69e65c6fa81a153f3254017b2d089e4fa1f7e9fcf0e7a877ce302b2661d3aadf",
    "looking_glass/bert_gpu_family_owner.py": "2acbfc4fab93720a5ce8555a6061b09f820ed1c5d934a855443845dfcfba8dae",
    "looking_glass/bert_gpu_family_profile.py": "cfaf8e66fedefd15efbfa07755f0368075013b37c8c0f3c30fdc53a55928bd23",
    "looking_glass/bert_gpu_study_adapter.py": "26f3d227569aa5b3520eea03d5ec20b49639f5bc4f1212dee213866e3418851e",
    "looking_glass/device_identity.py": "ad08eff6edbb1c94af9abec0d07d8a10944624488b8d723b10c75e8c67e10c45",
    "looking_glass/device_lease.py": "7f52ad238a1c9e8800c3d1ebf3b6969edd810a02754d1462c34f06cf783b57bf",
    "looking_glass/gpu_oneshot.py": "bc994727276a9918741ee80bd1ad8913b01d474015bb8dfdadbbfed1de9e048f",
    "looking_glass/intervention_plan.py": "6470676938e73e48234128c30ba1d10a49104ee8513434871bc6e973c55d6120",
    "looking_glass/interventions.py": "71cbcf4776660d2dd883b82815f710b0c909228fd1762504f9e3f984f7f6c60f",
    "looking_glass/memory_budget.py": "27d7b1a352815b7a103e1d70a5f8f0a3ccffdd10dacb71a0744a47ad6f66e1b9",
    "looking_glass/model_ownership.py": "99fee108f56ff8b7b7361e39de982c242cff590cc06c5d7a640cc12123cfe434",
    "looking_glass/pretrained.py": "ef86be0af1fd548027d23c8de779ff8cc77a8e180a9cb05d0b10a919454752da",
    "looking_glass/supervisor.py": "87603aad90ff541e14e5a8f7c847f0237995be8e578fe1fb4f7e6c59aef94d2a",
    "scripts/bert_gpu_family_bootstrap.py": "8789a42617ff073580d8aad74dc14c0dc75f9a47c532720114ca92e4cd1324cd",
}


def check_assets(directory):
    for name, (expected_size, expected_hash) in ASSETS.items():
        hasher = hashlib.sha256()
        size = 0
        with (directory / name).open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                need(size <= expected_size, "oversized cached asset")
                hasher.update(chunk)
        need(size == expected_size and hasher.hexdigest() == expected_hash, "cached asset mismatch")


def number(bits):
    need(type(bits) is int and 0 <= bits < 2**32, "invalid float bits")
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    need(exponent != 255, "nonfinite float bits")
    value = math.ldexp(float(fraction), -149) if exponent == 0 else math.ldexp(float((1 << 23) | fraction), exponent - 150)
    return -value if bits >> 31 else value


def vector(encoded):
    need(type(encoded) is str, "missing vector")
    raw = base64.b64decode(encoded, validate=True)
    need(len(raw) == 3072 and base64.b64encode(raw).decode("ascii") == encoded, "invalid vector encoding")
    need(all((bits & 0x7f800000) != 0x7f800000 for (bits,) in struct.iter_unpack("<I", raw)), "nonfinite vector")
    return raw


def read_lines(path):
    raw = load(path, 4 * 1024 * 1024)
    need(raw.endswith(b"\n") and b"\r" not in raw and raw.count(b"\n") <= 328, "invalid journal framing")
    lines = raw[:-1].split(b"\n")
    need(all(0 < len(line) <= 12288 for line in lines), "invalid journal line")
    return [parse(line) for line in lines], digest(raw)


def audit(result_path, run_dir, source, observation_path=None, asset_directory=None):
    result = parse(load(result_path, 2 * 1024 * 1024))
    need(result.get("schema") == "bert-gpu-family-result-v1" and result.get("status") == "success", "profile not declared successful")
    inventory = result["evidence_inventory"]
    need(type(inventory) is list, "invalid inventory")
    required = {"started.json","supervision.json","owner.json","candidate.json","final.json","events.jsonl","memory.jsonl"}
    table = {}
    for row in inventory:
        name = row["name"]
        need(type(name) is str and name in required and name not in table, "invalid evidence name")
        data = load(run_dir / name, 4 * 1024 * 1024)
        need(type(row["size"]) is int and row["size"] == len(data) and row["sha256"] == digest(data), "inventory mismatch")
        table[name] = row["sha256"]
    need(required == set(table), "missing or contradictory artifacts")
    records = {name: parse(load(run_dir / name, 1024*1024)) for name in ("started.json","supervision.json","owner.json","candidate.json","final.json")}
    start, sup, owner, candidate, final = (records[name] for name in ("started.json","supervision.json","owner.json","candidate.json","final.json"))
    commit, run = result["source_commit"], result["run_id"]
    need(commit == "4a597641be89ee094145fbdacee4d3fcdd241e41", "unexpected source commit")
    if observation_path is not None:
        observation = parse(load(observation_path, 32768))
        need(observation["schema"] == "bert-gpu-family-controller-observation-v1"
             and observation["source_commit"] == commit and observation["run_id"] == run
             and type(observation["tool_exit_code"]) is int and observation["tool_exit_code"] == 0,
             "controller exit observation mismatch")
        stdout = observation["stdout"].splitlines()
        need(len(stdout) == 2 and parse(stdout[0]) == {"status": "success", "source_commit": commit, "run_id": run}
             and stdout[1] == "Controller exit code: 0", "controller output observation mismatch")
    need(start["run_id"] == run and start["source_commit"] == commit and sup["run_id"] == run, "start identity mismatch")
    need(sup["clean"] is True and sup["timed_out"] is False and type(sup["exit_code"]) is int and sup["exit_code"] == 0, "supervisor not clean")
    for row in (owner,candidate,final):
        need(row["run_id"] == run and row["source_commit"] == commit and row["supervisor_run_id"] == sup["supervisor_run_id"]
             and type(row["pid"]) is int and row["pid"] == sup["pid"], "owner identity mismatch")
    need(final["candidate_sha256"] == table["candidate.json"] and type(final["active_memory_leases"]) is int
         and final["active_memory_leases"] == 0, "terminal/candidate mismatch")
    measured = candidate["measurement"]
    for key, value in result["measurement"].items():
        need(same(value, measured[key]), "public projection differs")
    expected_runtime = {"python":"3.11.15","torch":"2.8.0+cu128","cuda_build":"12.8",
                        "transformers":"4.57.3","tokenizers":"0.22.2","safetensors":"0.8.0","numpy":"2.4.6"}
    expected_execution = {"cudnn_benchmark": False, "cudnn_deterministic": True, "cudnn_tf32": False,
                          "deterministic_algorithms": True, "device": "cuda:0", "interop_threads": 1,
                          "matmul_tf32": False, "overwrite_module_params_on_conversion": False,
                          "swap_module_params_on_conversion": False, "threads": 1}
    need(same(measured["execution_config"], expected_execution)
         and measured["device_name"] == "NVIDIA GeForce RTX 2070 SUPER"
         and measured["runtime_versions"] == expected_runtime and measured["torch_version"] == expected_runtime["torch"]
         and measured["cuda_build"] == expected_runtime["cuda_build"] and same(measured["seed"], 17)
         and measured["cublas_workspace_config"] == ":4096:8" and measured["runtime_uuid_verified"] is True
         and measured["model_weakref_alive"] is False and same(measured["budget_active_before_candidate_write"], 1),
         "runtime, device or custody attestation mismatch")
    meta = measured["model_metadata"]
    need(meta["model"] == "distilbert/distilbert-base-uncased"
         and meta["revision"] == "12040accade4e8a0f71eabdb258fecc2e7e948be"
         and meta["model_class"] == "DistilBertForMaskedLM" and meta["tokenizer_class"] == "DistilBertTokenizerFast"
         and same(meta["parameter_count"], 66985530) and meta["attention_implementation"] == "eager"
         and meta["cpu_tied_weight_verified"] is True and meta["transformers_version"] == expected_runtime["transformers"]
         and meta["tokenizers_version"] == expected_runtime["tokenizers"]
         and meta["safetensors_version"] == expected_runtime["safetensors"], "model identity mismatch")
    expected_assets = {name: {"size": size, "sha256": sha} for name, (size, sha) in ASSETS.items()}
    need(same(meta["assets"], expected_assets), "recorded model assets do not match pins")
    if asset_directory is not None:
        check_assets(asset_directory)
    source_map = measured["source_sha256"]
    need(type(source_map) is dict and source_map == result["source_sha256"] == SOURCES, "source map mismatch")
    for name, sha in source_map.items():
        relative = Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts, "invalid source path")
        need(digest(load(source / relative, 8 * 1024 * 1024)) == sha, "source hash mismatch")
    manifest_raw = load(source / "experiments/bert_agreement_public_cases_v1.json", 1024*1024)
    need(digest(manifest_raw) == measured["manifest_sha256"] == "20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5", "manifest hash mismatch")
    manifest = parse(manifest_raw)
    rows = [row for row in manifest["rows"] if row["split"] == "development" and row["family"] == 1]
    targets = [row for row in manifest["targets"] if row["split"] == "development" and row["family"] == 1]
    need(len(rows) == 8 and len(targets) == 4, "family inventory mismatch")
    by_id = {row["row_id"]:row for row in rows}
    need(len(by_id) == 8 and all(same(row["input_ids"][6], 103) and same(row["attention_mask"], [1]*10) for row in rows), "family input mismatch")
    events, event_sha = read_lines(run_dir / "events.jsonl")
    need(len(events) == 328 and event_sha == measured["profile"]["journal_sha256"] == table["events.jsonl"], "event inventory mismatch")
    cursor = count = 0
    controls = {}
    sites = (("candidate",2,2),("block2_distractor",2,5),("random_block3_distractor",3,5),("random_block4_distractor",4,5),("random_block0_subject",0,2))
    references, baseline, effects = {}, {}, {}
    def take():
        nonlocal cursor
        need(cursor < len(events), "event stream ended early")
        row = events[cursor]; cursor += 1
        return row
    def forward(kind, row_id, site, layer, token, donor=None, patch=False, self_patch=False):
        nonlocal count
        count += 1
        identity = {"split":"development","family":1,"forward_index":count,"kind":kind,"row_id":row_id,
                    "donor_row_id":donor,"site":site,"layer":layer,"token":token}
        started, completed = take(), take()
        for status, record in (("started",started),("completed",completed)):
            need(record.get("status") == status and all(type(record.get(k)) is type(v) and record.get(k) == v for k,v in identity.items()), "forward identity/order mismatch")
        need(same(completed["model_input_ids"], by_id[row_id]["input_ids"]) and same(completed["model_attention_mask"], by_id[row_id]["attention_mask"]), "forward input mismatch")
        before = vector(completed["before_float32_le_base64"])
        after = vector(completed["after_float32_le_base64"]) if patch else None
        if not patch: need(completed["after_float32_le_base64"] is None, "unexpected action vector")
        need(same(completed["capture_bytes"], (6144 if patch else 3072))
             and same(completed["source_bytes"], (3072 if patch and not self_patch else 0)), "capture accounting mismatch")
        is_bits, are_bits = completed["is_bits"], completed["are_bits"]
        need(number(is_bits) == completed["is_logit"] and number(are_bits) == completed["are_logit"], "logit bits mismatch")
        mask = completed["mask_logits_sha256"]
        need(type(mask) is str and len(mask) == 64 and all(c in "0123456789abcdef" for c in mask), "invalid mask digest")
        actions = completed["events"]
        names = ["before","patch","after"] if patch else ["before"]
        need(type(actions) is list and len(actions) == len(names), "action count mismatch")
        for index, action in enumerate(actions):
            expected = {"name":names[index], "operation":("patch" if patch and index == 1 else "capture"),
                        "order":index, "module_path":f"distilbert.transformer.layer.{layer}",
                        "module_aliases":[f"distilbert.transformer.layer.{layer}"], "call_site_known":False,
                        "forward_index":0,"invocation":0,"batch_index":0,"token_index":token,
                        "output_selector":[0],"output_shape":[1,10,768],"selected_shape":[768],
                        "dtype":"torch.float32","device":"cuda:0",
                        "source_capture":"before" if self_patch and index == 1 else None,
                        "preceding_same_site_actions":names[:index]}
            need(all(k in action and same(action[k], v) for k,v in expected.items()), "action metadata mismatch")
        return {"before":before,"after":after,"mask":mask,"bits":(is_bits,are_bits)}
    def control(name, case, site, value, reference):
        row = take()
        expected={"status":"control","split":"development","family":1,"name":name,"case_id":case,"site":site,"passed":True}
        need(all(type(row.get(k)) is type(v) and row.get(k) == v for k,v in expected.items()), "control mismatch")
        need(value["mask"] == reference["mask"] and value["bits"] == reference["bits"], "control output mismatch")
        controls[name] = controls.get(name,0)+1
    for site, layer, token in sites:
        donors = {}
        for row in rows:
            obs = forward("donor_capture",row["row_id"],site,layer,token)
            donors[row["row_id"]] = obs["before"]
            if row["subject_source"] == "A":
                if site == "candidate":
                    references[row["row_id"]] = obs
                    baseline[row["row_id"]] = obs["bits"]
                else:
                    control("clean_matches_candidate",row["row_id"],site,obs,references[row["row_id"]])
        for target in targets:
            target_id, case = target["target_row_id"], target["case_id"]
            margins = {}
            for cell in ("A_same","A_opposite","B_same","B_opposite"):
                donor = target["donor_row_ids"][cell]
                obs = forward("patch_"+cell,target_id,site,layer,token,donor,True,cell=="A_same")
                need(obs["before"] == donors[target_id] and obs["after"] == donors[donor], "patch vector did not land")
                if cell == "A_same": control("self_patch",case,site,obs,references[target_id])
                margins[cell] = math.fsum((number(obs["bits"][0]),-number(obs["bits"][1])))
            delta_a = math.fsum((margins["A_opposite"],-margins["A_same"]))
            delta_b = math.fsum((margins["B_opposite"],-margins["B_same"]))
            sign = 1 if by_id[target_id]["subject_number"] == "singular" else -1
            effects.setdefault(case,{})[site] = -sign * math.fsum((delta_a,delta_b)) / 2
    final_sources, final_before = {"final_subject":{},"final_mask":{}}, {"final_subject":{},"final_mask":{}}
    for target in targets:
        target_id,case = target["target_row_id"],target["case_id"]
        donor = target["donor_row_ids"]["A_opposite"]
        for name, token in (("final_subject",2),("final_mask",6)):
            source_obs = forward("delivery_donor",donor,name,5,token)
            patched = forward("delivery_patch",target_id,name,5,token,donor,True)
            need(patched["after"] == source_obs["before"],"final patch did not land")
            final_sources[name][donor] = source_obs["before"]
            final_before[name][target_id] = patched["before"]
            control(name,case,name,patched,references[target_id] if name == "final_subject" else source_obs)
        fresh = forward("final_clean",target_id,"candidate",2,2)
        control("final_clean",case,"candidate",fresh,references[target_id])
    need(final_sources == final_before and cursor == len(events) and count == 140, "final event inventory mismatch")
    cases=[{"case_id":t["case_id"],"target_row_id":t["target_row_id"],"baseline_is_bits":baseline[t["target_row_id"]][0],
            "baseline_are_bits":baseline[t["target_row_id"]][1],"effects":effects[t["case_id"]]} for t in targets]
    means={site:math.fsum(case["effects"][site] for case in cases)/4 for site,_,_ in sites}
    profile=measured["profile"]
    need(same(profile["forward_count"], count) and same(profile["record_count"], len(events))
         and profile["expected_device"] == "cuda:0"
         and same(profile["live_cases"], cases) and same(profile["audited_cases"], cases) and same(profile["site_means"], means)
         and same(profile["control_count"], sum(controls.values())),"effect/control summary mismatch")
    memories, memory_sha=read_lines(run_dir/"memory.jsonl")
    need(len(memories) == 287 and memory_sha == measured["memory_journal_sha256"] == table["memory.jsonl"],"memory inventory mismatch")
    stages=[(x,None) for x in ("before_model_load","before_model_copy","after_model_copy","after_initial_state")]
    for i in range(1,141):stages.extend((("before",i),("after",i)))
    stages.extend((x,None) for x in ("after_final_state","cleanup_before_workspace_clear","after_cleanup"))
    total,cap,commitment=measured["device_total_bytes"],measured["allocator_cap_bytes"],measured["whole_run_commitment_bytes"]
    need(type(total) is int and 7*1024**3 <= total <= 9*1024**3 and type(cap) is int and cap == total//4
         and type(commitment) is int and commitment == cap+256*1024**2
         and measured["allocator_backend"] == "native" and same(measured["allocator_fraction"], 0.25)
         and same(measured["gpu_trace_claim_bytes"], 65536) and same(measured["host_trace_claim_bytes"], 64*1024**2),
         "profile memory policy mismatch")
    reserve=max(1024**3,(15*total+99)//100)
    peak_a=peak_r=0
    free=[]
    for row,(stage,index) in zip(memories,stages):
        need(row["stage"] == stage and type(row["index"]) is type(index) and row["index"] == index
             and row["peak_scope"] == "cumulative_since_pre_model_load_reset","memory stage mismatch")
        m=row["memory"]
        keys=("device_free_bytes","device_total_bytes","allocated_bytes","reserved_bytes","peak_allocated_bytes","peak_reserved_bytes")
        need(all(type(m[k]) is int and m[k]>=0 for k in keys),"invalid memory counter")
        need(m["device_total_bytes"]==total and m["device_free_bytes"]<=total
             and m["allocated_bytes"]<=m["reserved_bytes"]<=total
             and m["allocated_bytes"]<=m["peak_allocated_bytes"]<=m["peak_reserved_bytes"]<=total
             and m["reserved_bytes"]<=m["peak_reserved_bytes"]
             and m["peak_allocated_bytes"]>=peak_a and m["peak_reserved_bytes"]>=peak_r,"inconsistent memory counters")
        need(m["device_free_bytes"]>=reserve+commitment and m["reserved_bytes"]+reserve+commitment<=total
             and m["device_free_bytes"]>=reserve+256*1024**2 and m["peak_reserved_bytes"]<=cap-256*1024**2,"resource policy failure")
        peak_a,peak_r=m["peak_allocated_bytes"],m["peak_reserved_bytes"]
        free.append(m["device_free_bytes"])
    cleanup=memories[-1]["memory"]
    need(free[0] >= 3*1024**3 and same(cleanup, measured["cleanup"])
         and cleanup["allocated_bytes"]==cleanup["reserved_bytes"]==0,"initial memory or allocator cleanup mismatch")
    return {"schema":"bert-gpu-family-independent-audit-v1","status":"saved_evidence_consistent",
            "source_commit":commit,"manifest_sha256":digest(manifest_raw),
            "event_sha256":event_sha,"memory_sha256":memory_sha,"artifact_sha256":table,
            "forward_count":count,"event_record_count":len(events),"memory_record_count":len(memories),
            "control_counts":controls,"cases":cases,"site_means":means,
            "device_total_bytes":total,"allocator_cap_bytes":cap,"peak_allocated_bytes":peak_a,
            "peak_reserved_bytes":peak_r,"minimum_device_free_bytes":min(free),
            "cleanup_allocated_bytes":cleanup["allocated_bytes"],"cleanup_reserved_bytes":cleanup["reserved_bytes"],
            "local_asset_files_checked": asset_directory is not None,
            "controller_exit_observation_checked": observation_path is not None,
            "limitations":["one public development family; no held-out or population claim",
              "record consistency cannot authenticate actual execution or physical isolation",
              "controller exit observation is operator-reported, not an independent operating-system attestation",
              "locally rehashed checkpoint assets do not prove which bytes were used during execution",
              "full mask logits were retained only as hashes; values and finiteness cannot be independently reconstructed",
              "synchronized endpoints and allocator peaks can miss external or transient memory pressure"]}


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5, 6):
        raise SystemExit("usage: audit.py RESULT_JSON RUN_DIRECTORY SOURCE_CHECKOUT [OBSERVATION_JSON [ASSET_DIRECTORY]]")
    checked = audit(*(Path(arg) for arg in sys.argv[1:]))
    print(json.dumps(checked,sort_keys=True,indent=2,allow_nan=False))

