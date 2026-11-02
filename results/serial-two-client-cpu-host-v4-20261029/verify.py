"""Check arithmetic and workload fields in the public numeric projection."""
import hashlib
import json
from pathlib import Path


if not __debug__:
    raise RuntimeError("Run without Python optimization so assertions are checked.")

root = Path(__file__).resolve().parent
data = (root / "observation.json").read_bytes()
assert hashlib.sha256(data).hexdigest() == "300769e4eda4693b0df77a2f651aee1085b125c5c472445942c2d5952ccb0cee"
result = json.loads(data)
memory = result["memory"]
samples = memory["samples"]
assert result["schema"] == "serial-two-client-synthetic-cpu-host-observation-public-v1"
assert len(samples) == memory["sample_count"] == 68
assert samples[0]["offset_ns"] == 0
gaps = [b["offset_ns"] - a["offset_ns"] for a, b in zip(samples, samples[1:])]
assert min(gaps) > 0 and max(gaps) == memory["max_observed_inter_sample_gap_ns"] == 125_000_000
for sample in samples:
    assert sample["sequential_working_set_sum_bytes"] == (
        sample["owner"]["working_set_bytes"] + sample["client"]["working_set_bytes"]
    )
assert max(s["sequential_working_set_sum_bytes"] for s in samples) == memory["max_sampled_sequential_working_set_sum_bytes"] == 866_799_616
assert samples[-1]["owner"]["peak_working_set_bytes"] <= memory["owner_process_lifetime_peak_working_set_bytes_at_final_sample"] == 450_191_360
assert samples[-1]["client"]["peak_working_set_bytes"] <= memory["client_process_lifetime_peak_working_set_bytes_at_final_sample"] == 437_305_344
assert memory["owner_private_commit_bytes_at_final_sample"] == 816_033_792
assert memory["client_private_commit_bytes_at_final_sample"] == 820_494_336
workload = result["workload"]
assert workload["client_statuses"] == ["succeeded", "succeeded"]
assert workload["fake_forwards"] == 2 and workload["maximum_simultaneous_forwards"] == 1
assert workload["both_admitted_before_release"] and workload["serial_execution_observed"]
assert workload["owner_pid_and_birth_match_retained_handle"] and workload["owner_exit_code"] == 0
assert result["runtime"]["cuda_initialized"] is False
assert result["operator_reported_background_context"] == "full-suite-active"
old = json.loads((root / "superseded-v3.json").read_bytes())
assert old["invalidated"] is True
print("Public projection arithmetic and stated synthetic workload checks passed.")
