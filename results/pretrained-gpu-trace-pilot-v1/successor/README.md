# Five-token pretrained GPU trace pilot

The separately admitted successor completed the fixed ten-forward DistilGPT2
trace correctness pilot on one NVIDIA GeForce RTX 2070 SUPER. The owner
reported bitwise-equal full float32 logits for the two uninstrumented calls,
two independent-hook calls, and two calls each in direct, auxiliary-gradient,
and ID modes. The 18 direct and 18 auxiliary-gradient snapshots each matched
the corresponding independent-hook snapshots bit for bit and matched each
other. ID mode recorded 18 events with no retained tensor values. Each set
covers two forwards at layers 0, 2 and 5, with attention, MLP and block
boundaries for token positions 0 and 1. Full logits have shape [1, 5, 50257].

This is one fixed public input on one device, not a latency benchmark, a
causal result, or a reusable GPU capacity profile. The auxiliary-gradient
readout returns selected activations by differentiating a separate dot product;
it is not a gradient of the model's output or loss. The public
[hash-only projection](public-projection.json) contains the recorded event
metadata, hashes, phase observations and provenance; it contains no raw logits
or activations. The owner performed the bit comparisons. An independent
saved-record check can reconcile the hashes and metadata but cannot recompute
the tensor comparisons from the public records.

The synchronized observations below show endpoint device-free memory and the
process's PyTorch allocator peaks for each phase. Peaks were reset before each
phase and include allocations already live at that phase; the free values are
device-wide samples, not a device-wide peak. Byte counts are exact saved
observations.

| Phase | Free before (B) | Free after (B) | Peak allocated (B) | Peak reserved (B) | Estimated increment (B) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Load | 7,477,395,456 | 7,104,102,400 | 334,744,576 | 373,293,056 | 444,176,794 |
| Cold baseline | 7,104,102,400 | 7,018,119,168 | 369,520,128 | 406,847,488 | 129,241,730 |
| Warm baseline | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,002,280 |
| Reference | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,146,459 |
| Direct | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,146,459 |
| Auxiliary-gradient | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,146,459 |
| ID | 7,018,119,168 | 7,018,119,168 | 369,520,128 | 406,847,488 | 36,002,280 |

The maximum recorded PyTorch allocated/reserved peaks were 369,520,128 and
406,847,488 B; the minimum sampled free value was 7,018,119,168 B. The load
estimate fit its proposed 512 MiB commitment, and every other phase fit the
1 GiB job commitment. These estimates use the prospectively specified margin;
they do not bound other processes, native CUDA allocations, unseen workspaces,
future runs or process RSS. The 128 KiB device and 384 KiB host trace charges
are logical custody claims. Each value-bearing set held 110,592 B; the three
sets together held 331,776 B, excluding full-logit arrays and temporary copies.

After disposal, the owner recorded local PyTorch allocated/reserved counters of
0/0, dead model/input/trace weakrefs, zero logical memory leases, and a repeated
runtime device binding check. Its supervisor reported clean exit 0, and the
separately retained direct controller exited 0 with its process identity
verified. The controller recorded the marker and attestation absent and the
owner absent from its post-exit management query. These observations do not
prove native-context teardown or exclude untracked descendants. The recorded
process peak working set was 1,087,512,576 B; private bytes at the cleanup
sample were 2,288,574,464 B and are not a measured private-memory peak.

The exact execution source is commit
`3499aa4ec97ce4625c0ac3b6bb9c44413dedb977`, with the pinned checkpoint and
five-token input specified in the prospective protocol. Runtime was PyTorch
2.8.0+cu128, CUDA build 12.8, Transformers 4.57.3, float32 eager evaluation,
one CPU thread, TF32 disabled and no cache. Deterministic and cuDNN settings
are recorded in the projection. The cold baseline is the first model forward
after load, CUDA setup and input transfer, not a cold process or latency test.

An earlier attempt stopped before a recorded model forward and was preserved as
[an incomplete startup](../failed-start/README.md). Its unsafe marker was
resolved through a separate qualified recovery before this successor; recovery
does not prove native teardown. The successor used a new directory and fresh
admission. Further numerical, source, memory and cleanup review is recorded
alongside the final publication.
