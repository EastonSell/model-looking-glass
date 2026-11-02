# ViT CUDA exposed-pair observer v1

This diagnostic repeats one already exposed public pair, not a new held-out
experiment. Pair-04 contains cat image 520531 and dog image 377575 from the
fixed COCO mirror manifest `8b5cbfa3b2ab2786721b2c7f2e1975256aaf0864dd34784ed4bfb5e609706207`.
Their float32 processed-pixel hashes are respectively `e531519426eff5db71bac0fb3247417a894949db5dc785fc0a88b9f11081a8ed` and
`e63ad3e6f0a3a9b39f44e9828bc8be94e9eb3905bf5b4da11e19c91567c5c09a`.
The checkpoint is the exact ViT-tiny used by the published CPU study. Its
state digest is `d1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd`.
The reference manifest SHA-256 is `3df19e35e8550f456015f3af3a9ce5536d0387fbf19cca7c58f6dd25fa975fa0`;
it binds the saved CPU cat, dog, and cat-with-dog natural-inside records.
Reference preparation checks that the CPU requested donor bytes equal the four
tokens selected from the saved dog activation. No reference model forward is run.

The CUDA worker makes exactly six batch-one evaluation forwards in order:
cat and dog clean; cat with same-invocation self patches; cat with the CUDA
dog's captured tokens; cat with the saved CPU dog's exact token bytes; and a
cat clean repeat. Patches use tokens 161, 162, 175 and 176 after encoder block
index 2 (the third block), at the same receiver coordinates. A same-invocation
full pre/post witness and selected-token captures must establish exact landing
and unchanged unselected tokens. GPU self and repeat logits and early states
must match the first CUDA cat forward bit for bit. Full 1,000 logits and
activation witness bytes are retained for each completed forward, with bounded
sizes, metadata and digests. After a successful synchronization, a bounded
returned-logit receipt is saved before plan cleanup, shape, capture or score
validation; a completed forward must match that receipt. If synchronization
or copying fails, no salvage CUDA read is attempted. A failure remains
incomplete with the partial evidence available at that point.

Run float32 eager inference with one CPU thread, deterministic PyTorch options,
CUDA matmul and cuDNN TF32 disabled, and autocast disabled. The worker records
the actual runtime and device settings. Compare complete CPU and CUDA logits
and early activations by maximum and mean absolute difference and report the
dog-versus-cat group log-odds, without an arbitrary agreement pass/fail gate.
The CPU-donor contrast reuses donor bytes but also differs in the recipient's
state and downstream kernels. Comparing GPU and CPU donors on the same GPU
recipient describes donor-state differences, not a complete causal decomposition.

A supervised direct child takes the canonical physical-device lease before
importing model libraries. Under that lease it verifies the CUDA device identity,
native allocator, a 25% allocator cap and live memory admission, retaining a
whole-owner commitment and trace claims through synchronized execution and
cleanup. Stage snapshots are point measurements, not a worst-case bound. After
disposing model and tensors, the worker synchronizes, clears allocator cache and
the pinned PyTorch cuBLAS workspace, and checks its own allocator counters and
live-object references before clean attestation. An uncertain drain or failed
child leaves the supervisor's unsafe marker; no retry is silently started.
The supervisor reaps the direct child and assumes no CUDA-using descendants;
the lease is cooperative and does not control unrelated GPU applications.
Zero local allocator counters and WDDM process observations do not prove global
device idleness. The file and trace caps do not bound all host RAM, CUDA context
or non-PyTorch allocations.

The source checks and saved receipts assume a trusted, quiescent checkout,
interpreter, dependencies and native libraries. They are not execution
attestation. The case images were previously observed, their pretraining
exposure is unknown, and these six forwards establish no general accuracy,
latency, throughput or performance claim. Model outputs, differences and
control failures remain descriptive rather than a fresh localization result.
