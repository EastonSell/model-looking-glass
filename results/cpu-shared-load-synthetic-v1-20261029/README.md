# Synthetic CPU shared-load development checks

These are two short, unisolated CPU loopback checks of the shared-load event
path. They use the same frozen plan: three authenticated teams, four
closed-loop jobs per team, a synthetic identity model with a 20 ms sleep per
forward, and a global pending limit of three. Both windows observed 12
successful jobs, 87 events, no event-audit errors, and drained CPU work. The
model, device guard, leases, and memory telemetry were synthetic. These are
not GPU, production, isolated-latency, paired-comparison, stable-fairness, or
resource-profile results.

| Window | Release through drain | Successful jobs/s | Evidence |
| --- | ---: | ---: | --- |
| First | 266.9211 ms | 44.9571 | `report.json`; no separate saved watchdog receipt |
| Watched | 382.5928 ms | 31.3649 | `report_watchdog.json`, `watchdog_receipt.json` |

Both denominators include the entire common window, including IPC, recording,
hashing and shutdown; each team has four successes and uses that same
denominator. The two windows are not randomized or isolated. Their difference
does not establish an optimization, scheduling effect, or throughput change.
Four observations per team are too few to estimate a tail or an SLA.

The SHA-256 of the canonical nested plan (not the plan file's bytes) is
`18e23e94fd4263f2db5bc18794159063f34fcf34a2942134e5f83bc6ecdb2e2d`.
The first report SHA-256 is
`7adfaba3a9a66d31522f67ec5c03bb22487c4296c371f124feab0a292c0b8a38`;
the watched report SHA-256 is
`d978bd263f327126338534e762c9f4520b22dc94c8fbace1480e49e1ffe33fce`.
The separately saved watched-run receipt records an external Python subprocess
timeout and observed exit code zero; its SHA-256 is
`d47c11e53c53e72e55825d4aeb1c3a2b69a7a27a37530ce0c3c3acf777a34e3c`.
It is a local process observation, not independent hardware or process
attestation. The first run's shell reported exit zero, but no comparable
standalone receipt was saved.

The watched receipt records local source revision
`1d04f252860d9ed30d5e2d6b0b0b7c38e455cda1`. Its Git tree was
`beede9d84af2a772d77af3acaf387f3d189d4821`; the independently published
candidate commit `a851286f4bb40fbba488d5068fde658db0151303` has the same tree.
The plan and raw events are consistency evidence, not authenticated proof of
source, model execution, or physical device state. Output digests were checked
against the synthetic plan, but the original client output tensors were not
retained. The work-drained flag does not attest model or lease disposal.

An independent reviewer supplied `cpu_shared_load_independent_audit.py` to
recompute identities, timestamps, queue turns, counts, distributions and rates
from the raw files. Run without Python optimization, from the repository root:

```text
python results/cpu-shared-load-synthetic-v1-20261029/cpu_shared_load_independent_audit.py results/cpu-shared-load-synthetic-v1-20261029 report.json
python results/cpu-shared-load-synthetic-v1-20261029/cpu_shared_load_independent_audit.py results/cpu-shared-load-synthetic-v1-20261029 report_watchdog.json watchdog_receipt.json
```

The measurement boundaries and limits are described in
[`docs/CPU_SHARED_LOAD.md`](../../docs/CPU_SHARED_LOAD.md).
