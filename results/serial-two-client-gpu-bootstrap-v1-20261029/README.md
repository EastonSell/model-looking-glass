# Two-client serial GPU service bootstrap

On 2026-10-29, two clients each submitted one fixed two-token DistilGPT2
request to the same supervised service. Both requests completed successfully.
The saved broker timestamps show that the second request became ready before
the first finished, while their service intervals did not overlap. The
observed execution was serial; this is a small connectivity and queueing
bootstrap, not a throughput benchmark or a test of fairness under sustained
load.

The fixed source was commit `dd819b8731861794ae917d6126e1ae0c35c80103`.
Each request used a 1 × 2 input, with seeds 29 and 43 respectively, and no
activation trace or requested cancellation. The reviewed owner source
constructs one resident model and worker and performs a cold warmup before
serving; this run did not independently count model loads or forward calls.
The two returned float32 payload digests matched, which does not by itself
establish numerical correctness.

| Request | Ready to selected | Selected to terminal |
| --- | ---: | ---: |
| First | 0.0693 ms | 751.0538 ms |
| Second | 732.7922 ms | 751.2538 ms |

These are source-stamped broker intervals. Ready-to-selected is the observed
queue interval; selected-to-terminal includes the worker, synchronization,
CPU transfer, and broker completion. Neither is an isolated GPU latency.
The second request became ready 732.7259 ms before the first reached its
terminal timestamp. The first and second service intervals were separated by
0.0663 ms. The run did not include a matched sequential baseline.

Windows process lifetime peaks were 1,693,908,992 bytes of working set and
2,545,704,960 bytes of commit charge for the owner; the corresponding
controller peaks were 99,704,832 and 90,468,352 bytes. The observer saved
161 point samples, with a maximum recorded gap of 243.279 ms. Process peaks
need not occur at the same time, sampled values can miss transient changes,
and working sets can count shared pages in more than one process. The
bootstrap's 8 GiB logical host commitment and 2 GiB reserve were part of its
initial admission screen, not an operating-system reservation or an RSS
limit. Background activity and non-atomic sampling limit attribution.

**No serving-time GPU free-memory or allocator trace was saved in this run.**
Earlier source-specific collector samples were used for admission, but do
not measure memory during these client requests. The small sample is not a
future memory bound.

The saved supervisor reported a clean owner exit with no timeout, and the
parent's retained handle reported a waited direct-controller exit of zero.
The outer parent exit was observed as zero by the operator's execution tool;
there is no separate persisted outer-parent process receipt. A later unlocked
OS point reported the recorded owner and controller numeric PIDs absent and
the coordination marker and attestation absent. These records do not independently prove descendant
process drainage or native CUDA teardown. Two independent readers reconciled
the private source, supervisor, request, and host-observation records; the
public summary omits private paths, device UUID and request identifiers.
