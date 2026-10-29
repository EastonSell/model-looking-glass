# Two-client serial service memory observation

On 2026-10-29, two clients each submitted one fixed two-token DistilGPT2 request to the same supervised GPU service at source commit `f2f1a18257fd8474f9ee460e243e3c3a3efa53a2`. Both requests completed. The requests were outstanding together, but the saved broker timestamps show nonoverlapping service intervals and a queued second request. This is one small serial-service observation, not a sustained throughput or fairness benchmark, and it has no matched sequential baseline.

The inputs were 1 Ã— 2 token IDs `[15496, 995]`, with seeds 29 and 43, no requested cancellation and no activation trace. The reviewed owner source constructs one resident model and performs a warmup before serving; the run did not independently count model loads or forwards. The two returned float32 digests matched, which does not independently establish numerical correctness. Only the payload digests were retained in the diagnostic; there was no independent raw-logit replay.

| Request | Ready to selected | Selected to terminal |
| --- | ---: | ---: |
| First, seed 43 | 0.1846 ms | 974.8769 ms |
| Second, seed 29 | 958.2438 ms | 949.1687 ms |

These are source-stamped broker intervals, not isolated GPU latency. Ready-to-selected describes the observed broker queue; selected-to-terminal includes the worker and its completion path. Guarded memory observation was active and may have affected these timings.

The opt-in memory observer covered 2.888 seconds from its post-warmup baseline to its endpoint after the worker drain and before clean attestation. It saved four intermediate device-memory points plus the baseline and endpoint, with no reported dropped points. The device-wide free-memory readings were 7,018,119,168 bytes at baseline and 6,951,010,304 bytes at the endpoint, which was also the smallest of the six saved points. These are discrete points and can miss changes between samples.

During that window, the owner's process-local PyTorch allocator peaks were 402,971,648 bytes allocated and 440,401,920 bytes reserved. They include allocations already resident at the baseline of 368,299,008 allocated and 406,847,488 reserved bytes. They are not device-wide or native-allocation peaks and do not establish a future memory bound.

The host observer recorded 122 samples with a maximum gap of 208.396 ms. Owner and controller lifetime private-commit peaks were 2,545,537,024 and 92,901,376 bytes respectively; those separate peaks need not be simultaneous and do not cover all processes. The retained 8 GiB logical host commitment and initial 10 GiB headroom screen were not operating-system reservations or RSS limits. The minimum later sampled system and caller commit availability was 7,573,504,000 bytes, illustrating that the initial screen was not a continuously maintained floor.

The saved supervisor reported a clean owner exit, and the retained parent handle recorded a waited direct-controller exit of zero. The outer parent exit was observed as zero by the operator's execution tool; there is no separate persisted outer-parent process receipt. At a later unlocked point, the coordinator observed the marker and attestation absent; that point was not saved as a separate receipt. The records do not prove descendant drainage or native CUDA teardown. Independent readers reconciled the private source and profile, host-memory, request-timing and telemetry evidence within their respective scopes; this projection omits local paths, process IDs and device identifiers.
