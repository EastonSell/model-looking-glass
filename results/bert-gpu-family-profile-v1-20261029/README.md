# DistilBERT GPU development-family profile, 2026-10-29 UTC

This is one supervised GPU resource and intervention-control run on the eight
already-exposed public development-family-1 rows. No evaluation row was
forwarded. The operator and reviewers observed the exact [source and protocol
published at commit 4a597641](https://github.com/EastonSell/model-looking-glass/commit/4a597641be89ee094145fbdacee4d3fcdd241e41)
before the launch; no independently trusted publication timestamp was
retained. The pinned checkpoint is `distilbert/distilbert-base-uncased` at
revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.

The unchanged [controller result](result.json) has SHA-256
`d7537ab322018794682d3e3ffd93eea83c6175a71d0ea20ecd156dac97cc2289`.
The [evidence directory](evidence/) preserves the seven original closed
start, owner, candidate, final, supervision, event and memory files. The
[controller observation](controller-observation.json) records the execution
tool's exit-code acknowledgement as a local operator observation, not an
independent process attestation. The supervisor recorded the reaped owner at
exit code 0, clean and without timeout; the controller exited 0. The operator
observed no unsafe marker after the run, and the post-exit NVIDIA query did
not list the owner PID. The separate [marker observation](marker-observation.json)
is a point-in-time operator record, not continuous device-state attestation.

The journal contains 140 started and completed forwards and 48 recorded
controls. Its SHA-256 is
`152e3f463a9bc7b8f8f0ef1e05e9304aa93c521b74ee19e5a2484af149246190`;
the 287-record memory journal's SHA-256 is
`c6f71ba07738a9c889e490122d841c1ec8fe5be74078b9857e5c0d4d9595cea9`.
Independent reviews reconstructed the schedule, stored donor and replacement
vectors, paired effects, controls, source identity and ordered memory checks.
The final mask-position patch is a delivery check expected to reproduce the
donor logits by construction.

The portable [independent auditor](independent-audit.py) and its [review
record](independent-audit.json) are included. The auditor accepts the result,
the evidence directory, a clean checkout of the exact source commit, the
controller observation, and optionally the pinned offline asset directory.
It checks consistency of retained evidence and does not authenticate execution.

The signed donor-number effect is the four-target family mean of the
prespecified same/opposite-number contrast, averaged across two donor lemmas.
Positive values indicate movement in the donor-number direction under that
contrast; selective erasure and surface-form or lexical changes remain
possible. The effects in this one development family were:

| Site | Mean signed effect, logits |
| --- | ---: |
| Block 2 subject, candidate | +5.76077 |
| Block 2 distractor | −0.22613 |
| Block 0 subject, fixed comparison | +8.51654 |
| Block 3 distractor, fixed comparison | −0.10013 |
| Block 4 distractor, fixed comparison | −0.01488 |

The earlier subject-position comparison has the larger effect. These data do
not uniquely locate a number computation or establish a causal mechanism.
Six of eight candidate opposite-number donor actions reversed the target's
`is` minus `are` margin; the two plural-subject/plural-distractor actions
weakened the margin from −5.0622 to −2.0642 and −1.1665, respectively, but
left it negative. The earlier [CPU development run](../bert-agreement-supervised-dev-v1-20261028/README.md)
on these same exposed cases reported the same site means to five decimal places;
the largest absolute difference between site means was approximately
`1.70e-6`. Across the 140 paired actions, none had both selected logit bits
equal, and every retained full-mask digest differed. Full vectors were not
saved. An independent read-only reviewer checked the paired action values and
digests. This is a descriptive comparison across runtimes, not independent
replication or bitwise CPU–GPU equivalence.

The runtime reported an 8,589,606,912-byte device total and a 25% native
PyTorch allocator cap of 2,147,401,728 bytes. The whole-run commitment was
2,415,837,184 bytes. With one peak reset before model loading, cumulative
PyTorch peaks were **307,758,080 bytes allocated** (293.501 MiB) and
**339,738,624 bytes reserved** (324 MiB). The minimum sampled device-free
value was 7,085,228,032 bytes. Before cuBLAS workspace clearing, allocated and
reserved were each 33,554,432 bytes; after cleanup both were zero and the
model weak reference was dead. The claim counts and checks are recorded by
the reviewed owner; they are not independently sampled at every checkpoint.

These observations apply only to this model, input shape, family, execution
path and allocator history. They are not latency or throughput measurements,
a worst-case GPU or host-memory bound, or a profile authorizing held-out GPU
evaluation. The 64 MiB host-evidence claim bounds selected logical payloads,
not process RSS; the allocator cap excludes CUDA context, driver, non-PyTorch
and other-process memory. Synchronized point samples can miss transient
external pressure, and the lease cannot control unrelated GPU applications.

The owner checked complete mask-position logits during the run, but the
journal retains full-vector digests and the two selected float32 logit bits,
not the full vectors. A reader cannot reconstruct those vectors or independently
repeat their bytewise comparisons from the saved record. The recorded UUID
match is a live owner/controller check; raw UUIDs were not retained for
post-hoc equality verification. The 25 selected project files and five model
assets are hash-bound; third-party packages are identified by version rather
than content-hashed or independently attested. Local evidence and path-pattern
scans do not prove faithful execution or complete privacy.
