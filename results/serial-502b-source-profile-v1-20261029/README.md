# Fixed serial GPU profile for the 4 GiB host-bootstrap source

This one-shot calibration measured the fixed, no-trace DistilGPT2 service owner on one NVIDIA GeForce RTX 2070 SUPER. It collected one model-load sample, one cold forward and three warm forwards with a 1 x 2 input, one forward at a time. The input token IDs were `[15496, 995]`; the load and cold seed was 17, and warm seeds were 29, 43 and 71. This is a resource profile, not a throughput or numerical-parity benchmark.

The measured local source commit was `502b2d91e6854921e97dba1c63e6947281fb74fc`. The published [source snapshot](https://github.com/EastonSell/model-looking-glass/commit/6955941287ca22b649169bfe02e6410cd6ab6b2b) has the identical full Git tree `5e01969b5620f1671253e064800239a9d79bbf67`; its commit identity and ancestry differ. The measured profile also binds the original checkout and E-cache runtime paths: identical source bytes alone do not authorize use from another checkout.

| Stage | Samples | Estimated incremental device bytes, including the recorded margins |
| --- | ---: | ---: |
| Load | 1 | 1,024,458,752 |
| Cold forward | 1 | 355,598,356 |
| Warm forward | 3 | 270,488,596 |

Across the five samples, the largest process-local PyTorch allocator peaks were 369,417,216 bytes allocated and 406,847,488 bytes reserved; the minimum sampled device-wide free-memory point was 7,018,119,168 bytes. Peaks and point samples have different scopes, may miss other allocations or transient pressure, and do not bound future work. Four saved output metadata records report finite outputs; raw logits were not independently replayed for numerical correctness.

The source changes the *separate* experimental two-client bootstrap's logical host commitment from 8 GiB to 4 GiB, with a 2 GiB reserve and an existing 8 GiB minimum for each initial host headroom counter. The source change passed 72 CPU host/composition tests. This serial GPU calibration did not run the 4 GiB two-client bootstrap or measure its host-memory peak. Its own owner retained 2 GiB resident and 512 MiB workspace logical device commitments until exit; these are not OS reservations.

The saved terminal reports a clean supervised owner exit 0, a dead model weak reference and zero local PyTorch allocated/reserved counters after cleanup; the held direct-controller receipt reports a waited exit 0. A separate decision approved only these three recorded profiles for fixed 1 x 2, zero-trace development with fresh source/runtime matching, host/device admission and a coordinated slot. The original collector terminal remains unapproved. The controller receipt does not itself contain the candidate or terminal hashes; their association uses the retained operator context and matching saved fields and timing.

At a later unlocked OS-only point the coordinator observed the marker, attestation and recorded numeric process IDs absent; that point was not saved as a separate receipt. These observations do not prove descendant drainage or native CUDA teardown. Independent readers reconciled the saved source, artifacts, profile arithmetic and process records within their respective scopes. This projection omits local paths, process IDs and device identifiers.
