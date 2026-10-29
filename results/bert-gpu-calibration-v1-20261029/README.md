# DistilBERT GPU development calibration, 2026-10-29 UTC

This record covers one four-forward, ten-token development calibration on the
local NVIDIA GeForce RTX 2070 SUPER. The operator and reviewer observed the
source and protocol published before execution at commit
[`c2bff8183003ac25ec28475d23ddd20e7026f370`](https://github.com/EastonSell/model-looking-glass/commit/c2bff8183003ac25ec28475d23ddd20e7026f370),
using [protocol v1](https://github.com/EastonSell/model-looking-glass/blob/c2bff8183003ac25ec28475d23ddd20e7026f370/docs/BERT_GPU_CALIBRATION_V1.md).
No independently trusted publication timestamp was retained. Both farmer/farmers
rows were already exposed during the earlier public development probe. They
are not held-out cases.

The unchanged controller output is [result.json](result.json), SHA-256
`3ca46d8824a181e10836a701c5b6926dd865a86b171ba5cd3963a6db3477d7d2`.
The [evidence directory](evidence/) preserves the original start, owner,
candidate, final, supervisor and stage JSON records byte-for-byte. The final
record binds the candidate hash; run IDs and the ephemeral process ID agree
across the owner, candidate, final and supervisor records. The supervisor
reported a reaped child, exit code 0, no timeout and clean teardown. Its final
record reports zero active memory leases. The operator observed no canonical
unsafe marker after the run; no separate marker snapshot was retained.

## Observed values

The CUDA runtime reported 8,589,606,912 bytes total and a 25% native PyTorch
allocator cap of 2,147,401,728 bytes. The owner committed 2,415,837,184 bytes
and retained the prescribed trace claims. After model transfer, PyTorch
reported 269,079,040 bytes allocated and 306,184,192 reserved. With one peak
reset before model load, the **cumulative** maximum over this exact sequence
was 307,767,808 bytes allocated (293.510 MiB) and 339,738,624 reserved
(324 MiB). Before cuBLAS workspace clearing, allocated and reserved were each
33,554,432 bytes; after clearing, both were zero and model/input weakrefs
were dead. The stage files retain all 13 synchronized endpoint snapshots and
their observation counts. Device-wide `nvidia-smi` snapshots bracketed the run;
they are point samples, not exact lease-boundary timestamps.

The self-patch's full mask-position vocabulary logits matched the target
baseline by recorded raw-byte comparison. The donor and target selected
vectors differed; the recorded cross-patch post-action vector matched the
donor bits. The selected `is − are` margins were +1.5943 for the target,
−5.0622 for the donor, +1.5943 for the self-patch and −4.4023 for the
cross-patch. These are descriptive values for one previously exposed pair,
not a held-out result, a site-specific explanation or evidence about other
contexts.

Independent memory and provenance reviews checked the source/asset identities,
candidate/final/reaped-child linkage, all stage snapshots and cumulative
counter ordering, the stated admission inequalities, and the saved vector and
selected-logit bit relationships. The [review record](independent-review.json)
and [audit script](independent-review.py) are retained; the script accepts the
result, evidence directory, a clean checkout at the exact executed revision,
and optionally the offline cache. Current-cache hashes do not independently
attest the bytes loaded during an earlier run. The complete logits were checked for
finiteness by the owner but were not saved. Full mask-position logits are
represented only by digests plus the two selected values, so an offline reader
cannot reconstruct them or independently link those values to a digest. The
saved runtime UUID match is a controller/owner assertion: the raw preflight
and runtime UUIDs were not retained for a separate post-hoc comparison.

## Scope and persistence limits

This is a resource and intervention-delivery check for one model, shape,
sequence and allocator history. It is not a latency or throughput benchmark,
a worst-case device or host-memory bound, or an admission profile for the
full agreement experiment. The PyTorch allocator limit does not bound CUDA
context, driver, non-PyTorch or other-process allocations; point samples can
miss external pressure. Windows WDDM may not expose every GPU client. The
logical host-trace claim is not a cap on total process RAM.

The v1 writer creates final paths directly and does not check the return count
of each write. An interrupted or short write can therefore leave a truncated
stage, candidate, failure or result file. The saved records in this successful
run parsed and passed the controller and independent audits, but that does not
establish atomic or crash-durable publication. Future runs should use checked
write-all and flushed same-directory no-clobber publication; preserve any
partial or failed records instead of replacing them.
