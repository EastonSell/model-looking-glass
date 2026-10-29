# Synthetic v2 development resource pilot: 202 forwards

This is one CPU-only development run of the
[pilot protocol](CAUSAL_V2_PILOT_PROTOCOL.md), not a v2 discovery or held-out
study. It used one untrained model (seed 49351) and two development inputs;
no v2 split, private site schedule, or held-out input was generated. Its
[saved registry record](../results/synthetic-v2-resource-pilot-20261028/)
records a clean Git-status attestation and the published source revision
[`54c1d8a17d8ff041bc276d9373c19be3dc0be3da`](https://github.com/EastonSell/model-looking-glass/commit/54c1d8a17d8ff041bc276d9373c19be3dc0be3da).
The ten recorded project-source hashes match blobs at that revision. The
artifact alone cannot prove the complete checkout was clean or that the stated
bytes were actually executed.

The record reports successful execution of 202 planned root forwards: four
clean traces, six clean target captures, and 192 condition forwards. The
workload exercised the three specified synthetic injection types and the
in-process trace, intervention, and full-logit controls. The individual
control tensors are not included in the saved record, so the record alone
does not independently reconstruct those checks. The runner is designed to
write the plan and inputs before the first forward; a local record does not
prove an external timestamp or prove that an operation happened as claimed.

| Recorded quantity | Observation |
| --- | ---: |
| Workload-only wall time | 0.1824 s |
| Workload-only process CPU time | 0.1719 s |
| Function entry through workload end, wall time | 1.1705 s |
| Process-lifetime peak host working set | 455,159,808 bytes (434.1 MiB) |
| Proposed full v2 schedule | 57,408 forwards |
| Simple workload-only linear extrapolation | 51.84 s |

These timings exclude Python/PyTorch import and final evidence publication;
the workload-only span also excludes initialization, input publication and
registry setup. The process peak includes the interpreter and libraries from
process startup and is neither incremental tensor memory nor GPU memory.
The single-thread workload ran after a model initialization performed under
the ambient 12-thread setting; the proposed v2 study now requires one thread
before initialization. Both denormal-preservation probes passed in the pilot's
execution thread. Other processes and operating-system activity were not
controlled, and this is one short observation rather than an isolated benchmark.

The 51.84-second number is only 0.1824 seconds multiplied by 57,408/202; it
does not include the full study's six-model setup, persistence, or all fixed
costs, and it is not an upper bound. The arithmetic count fits the proposed
60,000-forward limit, but this observation alone does **not** establish that
the full schedule fits a wall-time or host-memory budget. A larger separately
labeled development pilot may be needed. No prediction or causal efficacy
claim follows from this resource pilot.

The saved canonical [`pilot_inputs.json`](../results/synthetic-v2-resource-pilot-20261028/artifacts/pilot_inputs.json)
has SHA-256 `024bdea3669683808302b953e7797afaa159d8d4719c574a3dc65e45bff74722`.
Both exact float32 tuples, with signed zeros treated as equivalent, must be
excluded from **both** future v2 splits before any split is generated. The
numeric policy is specified in the independently reviewed
[pre-seal clarification](CAUSAL_CONFIRMATION_V2_NUMERIC_CLARIFICATION.md).
