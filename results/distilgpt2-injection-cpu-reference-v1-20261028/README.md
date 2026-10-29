# Disclosed-site CPU injection reference, 2026-10-28

This is an exploratory mechanics check on the pinned DistilGPT2 checkpoint,
one already-exposed two-token development input and disclosed MLP layer 2.
It is not a blind localization, held-out behavioral evaluation, capability
preservation claim, or performance benchmark. The protocol and rerun command
are in `docs/PRETRAINED_INJECTION_CPU_REFERENCE.md`.

The [public-source run](success-v2.json) passed at source commit
`b0a6788b3dea0074397a72719a7d09112877971c` / tree
`d35513e6eb37c1ef1dacf9f9d3a8ceaeea0069ab`. Four clean/injected/same-site
forward/reverse executor results matched separate plain-hook references on
all 13 selected boundaries and all 50,257 final-token logits; each condition's
maximum absolute difference was 0.0 in this run. Full-output action witness
bytes, patch restoration, cleanup and pinned-model checks also passed. The
selected CUDA/network helpers were called zero times after guard installation.
This does not measure behavior outside this single input and fixed site.

The [earlier run](success-v1.json) and [independent replay](independent-v1.json)
passed at local source commit `ca2a0f6d7bcac6d6ed44d9badf284e7c90bb2eaa`.
That local commit has the same tree as the public-source run; its exact commit
object is not asserted to be remotely available. The independent reviewer
used a separate clean detached checkout and observed the same four zero-error
aggregates. The records bind selected source-file bytes and Git-reported clean
state, not every installed/native dependency or a fully attested environment.

The [v1](missing-cache-v1.json) and [v2](missing-cache-v2.json) missing-cache
controls deliberately pointed at an absent local cache and record a nonzero
test outcome. Their redacted `test` failure phase does not by itself identify
the exact exception. Both are preserved as controlled failure records, not
negative model results. No raw stdout, stderr, activations or logits are stored.
