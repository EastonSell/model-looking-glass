# Five-token GPU trace with retained-byte replay

The fixed ten-forward DistilGPT2 trace correctness successor completed on one
NVIDIA GeForce RTX 2070 SUPER. An independent reader compared the retained
float32 bytes after the owner and direct controller exited: all ten full logit
arrays agreed bit for bit, and all 18 direct and 18 auxiliary-gradient
captures matched the corresponding separately hooked reference captures.
The reader checked 54 ordered value-bearing segments and the 18 metadata-only
ID events. This is one fixed public five-token input, not a latency benchmark,
a causal result or a reusable GPU capacity profile.

Each full logit array has shape `[1, 5, 50257]`; the nine selected events per
forward have shape `[1, 2, 768]`, at attention, MLP and block outputs for layer
indices 0, 2 and 5, using token positions 0 and 1. There were two reference,
two direct and two auxiliary-gradient capture forwards. The independent reader
reconstructed nine full-logit comparisons against the first array and 36
capture comparisons against their matching reference coordinates. It checked
finite float32 values and exact bits; this method would distinguish signed
zeros and subnormals, although none occurred in these saved values.
Auxiliary-gradient mode differentiates a separate dot product to read the
activation; it is not a gradient of the model output or loss.

The 16 raw files total 10,383,176 bytes; a separate manifest binds their order,
offsets and hashes. The prospective protocol specified private E: retention and
a redacted, hash-based public result. The [independent byte audit](byte-audit.json)
and [public projection](public-projection.json) report the hashes and checks;
the former used the separately reviewed [reader](../../../tools/pretrained_trace_v2_independent.py).
This is a replay of quiescent saved bytes, not independent authentication of
the execution, source, process tree or chronology and not a rerun of the model.
The retained files were not leased against an external writer during replay.
The separate [envelope audit](envelope-audit.json) reconciles the source,
input, phase and exit records; neither audit proves arbitrary external behavior.

The table records synchronized device-free endpoints and the process's PyTorch
allocator peaks. Peaks were reset for each phase and include allocations
already live at that phase. Device-free readings are samples, not a device-wide
peak; allocator peaks exclude other processes and native CUDA allocations.

| Phase | Free before (B) | Free after (B) | Peak allocated (B) | Peak reserved (B) | Estimated increment (B) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Load | 7,477,395,456 | 7,104,102,400 | 334,744,576 | 373,293,056 | 444,176,794 |
| Cold baseline | 7,104,102,400 | 7,018,119,168 | 369,520,128 | 406,847,488 | 129,241,730 |
| Warm baseline | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,002,280 |
| Reference | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,146,459 |
| Direct | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,146,459 |
| Auxiliary-gradient | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,146,459 |
| ID | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,002,280 |

The highest recorded per-phase allocator peaks were 369,520,128 B allocated
and 406,847,488 B reserved; the minimum sampled device-free point was
7,018,119,168 B. The load estimate fit its 512 MiB logical resident commitment,
and the other phase estimates fit the 1 GiB job commitment. The 128 KiB GPU
and 4 MiB host trace/evidence charges are logical claims. Six retained full
logit tensors, up to 6,030,840 B, are outside the host trace charge. These
commitments do not bound process RSS, native workspaces, other processes or
future runs. The [memory audit](memory-audit.json) independently reconciles
the saved resource records.

The owner recorded zero local allocated and reserved bytes after disposal,
dead model/input/trace weakrefs, zero logical leases and a repeated runtime
device-binding check. The supervisor reported clean exit 0; the directly
observed controller exited 0 with its identity verified. The controller's
post-exit management point recorded the owner, marker and attestation absent.
These observations do not prove native-context teardown or exclude untracked
descendants. The recorded process peak working set was 1,087,614,976 B;
2,289,369,088 B of private bytes at the cleanup sample is not a measured
private-memory peak. Disk and host admission values are point observations,
not reservations.

The execution source was `8396611b2c9470594e895a3caebaa3cf52fe0749`, with
the checkpoint and input fixed in the [prospective protocol](../../../docs/PRETRAINED_GPU_TRACE_EVIDENCE_V2.md).
The owner used PyTorch 2.8.0+cu128, CUDA build 12.8, Transformers 4.57.3,
float32 eager evaluation, one CPU thread, TF32 disabled and no model cache.
The cold baseline was the first model forward after loading, CUDA setup and
input transfer, not a cold process or device start. Source verification assumes
a trusted quiescent checkout; the checks do not lease source or native libraries.

An unrelated one-thread CPU-only ViT screen was active during part of the
trace controller's recorded interval. Its observed file-creation and closure
points were 14:03:49 and 14:05:22 UTC; the trace direct-controller start and
exit records were 14:03:24 and 14:06:20 UTC on 2026-10-29. The CPU screen's
direct-observer receipt SHA-256 is
`516a2c8c53e3b94be2f534b35955d9c0470cc1a6fdba30a5bff52bb802bd782a`.
These are controller-level observations; per-forward overlap and causal
influence were not established. No timing conclusion is drawn from this
correctness run.
