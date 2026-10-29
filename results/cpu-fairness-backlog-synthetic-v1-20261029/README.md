# Synthetic CPU backlog and queue-rotation development run

These are two small, unisolated CPU development windows through the observed
fixed-workspace worker, serial broker and authenticated loopback IPC. The model,
device guard, leases and telemetry are synthetic; each forward requested a
30 ms sleep. They do not measure GPU performance, demonstrate fair network
admission or establish a general fairness or starvation-freedom guarantee.
Round-robin request turns do not imply equal GPU time.

Both windows used the same frozen twelve jobs and inputs, four each for `bulk`,
`peer-b` and `peer-c`. The baseline issued one request globally at a time in
that team order. The backlog window scheduled four bulk requests at 0, 3, 6
and 9 ms, and peer requests intermittently as recorded in the plan. The
scheduled times are not guarantees of actual submission or admission. Global
and per-team pending limits were 12 and 4; the global connection cap was 12.
The windows used separate warmed synthetic owners. External host activity was
not measured. The workload was developed with exploratory runs, so these are
not preregistered or held-out results.

| Observation | Baseline | Backlog |
| --- | ---: | ---: |
| Planned / attempted / observed accepted | 12 / 12 / 12 | 12 / 12 / 12 |
| Succeeded / failed / rejected / unresolved | 12 / 0 / 0 / 0 | 12 / 0 / 0 / 0 |
| Raw events; recorder loss | 87; 0 | 87; 0 |
| Window, release through work drain | 801.2880 ms | 450.0474 ms |
| Successes per common-window second | 14.9759 | 26.6639 |
| Peak pending bulk jobs | 1 | 3 |
| Selections with multiple eligible teams | 0 | 10 |
| Those with at least two bulk jobs pending | 0 | 4 |

There were four successes per team in each window, with no missing or ambiguous
samples or event-audit errors. The baseline had at most one pending job and no
competing experiment client. In the backlog window, source-stamped ready and
selection events reproduced the broker's per-team FIFO and round-robin turns
from an initially empty queue. Its observed selection teams were `bulk, bulk,
peer-b, peer-c, bulk, peer-b, peer-c, bulk, peer-b, peer-c, peer-b, peer-c`.
The first two bulk turns were separated by the actual arrival and eligibility
of peer work. Planned job numbers are not admission order: replay uses actual
source timestamps and fails closed on ties or incomplete work.

Median ready-to-selection waits, followed by the maximum in parentheses, were:

| Team | Baseline (ms) | Backlog (ms) |
| --- | ---: | ---: |
| bulk | 0.1159 (0.1268) | 78.5704 (219.8491) |
| peer-b | 0.0983 (0.1138) | 108.5554 (131.1715) |
| peer-c | 0.1104 (0.1539) | 136.4236 (156.1300) |

Each cell has only four observations; the reported nearest-rank p95 is the
maximum, not a tail or SLA estimate. The reports also retain client attempt,
post-decode acceptance, broker submission, ready, selection, model invocation
attempt, synchronized CPU finish, terminal outcome and post-decode/post-close
client receipt, with individual durations and failures. The model boundary
includes the final owner guard and CPU output handling. Median observed
entry-to-finish durations ranged from 30.34 to 30.72 ms across the six
team/window groups. Client receipt is not a broker cancellation decision; this
workload did not request cancellation.

The common-window rates include scheduled delays, recording, hashing, IPC,
polling, result handling and drain. Different concurrency and release patterns
make these windows unsuitable for a speedup comparison. Instrumentation and
host contention can affect both the waits and the realized queue. The IPC
connection cap is global and applies before authentication. Output digests
match the synthetic expected values, but original output tensors were not
retained; hashes do not authenticate execution. Successful work drain does not
attest model or lease disposal.

## Evidence and scope

The retained nested plan's canonical JSON SHA-256 is
`57e8b0a98f64e7bfb5ec3aa04a97bfb21c14a0af15772227264f033523d82049`;
the plan **file** byte hash appears below. The independent parent watchdog
observed each CPU-only child exit with code 0, reaped it, and saw a clean
on-disk source checkout at local commit
`9b6fb1a7d7140c9ef997159a7f1bf7360e877c71` (tree
`17addc9800ef0dd2c1a9d7a00139adc19c709f95`) before and after each run. The
receipts are not in-memory execution, dependency, hardware-activity or
isolation attestations. The captured [watcher](watch_windows.py) uses the
original local worktree layout. The exact local source commit and its
development history are preserved in [source.bundle](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/cpu-fairness-backlog-synthetic-v1-20261029/source.bundle), with
repository commits `d1382d2bdb23873a89a1a2234ecd29b14a82fbfa` and
`503a553f763d7f3a5f78864b5edc74926dbe7990` as prerequisites. In a clone
containing those commits, `git bundle verify source.bundle` verifies the
prerequisites; fetching its `refs/heads/research/cpu-fairness-backlog-source`
ref recovers the exact recorded commit. An equivalent GitHub source snapshot
at commit `2e0f649c300a36cf5a69154b602ac83fc776477d` has the same tree; it
does not change the commit recorded by the watchdog. Both runs report Python 3.11.15, Torch
2.8.0+cu128 with 12 CPU threads and `perf_counter_ns` backed by
`QueryPerformanceCounter()` with 100 ns nominal resolution; actual precision
and host isolation are not established.

| File | SHA-256 of captured bytes |
| --- | --- |
| [plan.json](plan.json) | `fa6f83f27cb981b807ea9efab99dfcb893410f9ad67b62ac2baad6fa2f68d128` |
| [baseline.json](baseline.json) | `07933415261946d10ec8bc6fa7c617768f900efdc4d01b3a48506b2c96e46580` |
| [backlog.json](backlog.json) | `d8de7f63fd812b984d4bbdd1a624b92ba629661d39508727ac0dd647967fb945` |
| [baseline watchdog receipt](baseline_watchdog.json) | `0460633b211d681ffdcb9a110951d3232e5c265be8cf439ada62d4377c29342f` |
| [backlog watchdog receipt](backlog_watchdog.json) | `8d9c8408359b8a5d2f3bb791fe940762620f15c04f287cd683525e4a434ae518` |
| [watch_windows.py](watch_windows.py) | `c2e10f55cff1c901834754fadbdcf84badf8bde2fc9dd70512abf9803a923071` |
| [independent_audit.py](independent_audit.py) | `06ab8c022dede10aa340bfbb21787de7a050dbdfae2cebd8fbdb63b3df0baa24` |
| [source.bundle](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/cpu-fairness-backlog-synthetic-v1-20261029/source.bundle) | `ae8fa6bd37865d619b8cab88daaf59d0f55d4295e602e766f36343f174e5942c` |

The four stdout/stderr captures are empty and match the hashes in their
receipts. The independent, standard-library-only audit recomputes the plan,
receipt bindings, inventory, timings, rates and rotation from the retained raw
events. Run it without Python's optimization flag, passing this directory as
its sole argument. Its checks are consistency checks on trusted local records,
not authentication of the experiment. Source timing and replay rules are documented in
[CPU_FAIRNESS_BACKLOG.md](../../docs/CPU_FAIRNESS_BACKLOG.md).
