# DistilBERT exposed-pair GPU observation

On one previously exposed public pair, replacing the block-2 subject state
with the opposite-number donor state reduced the `is − are` margin by
**5.14678645 logits**. The patched margin remained slightly positive, so the
preference between `is` and `are` did not reverse. This is a descriptive
cross-runtime check of one nonblind pair; it is not an independent held-out
evaluation, a lexical-control study, or evidence that block 2 uniquely
localizes the behavior.

The fixed target was `The painter beside the artist [MASK] ready.` and the
donor changed only `painter` to `painters`. Both were already used in the
[published CPU evaluation](https://github.com/EastonSell/model-looking-glass/blob/9d0aa9c0999924dd36b12bbdb638a27476d86bc8/results/bert-agreement-cpu-evaluation-v1-20261029/README.md).
The frozen GPU sequence captured target and donor, self-patched the target,
then patched the target with the donor, all at block 2, batch 0, subject
token 2. Each margin below subtracts the saved float32 `are` logit from the
saved float32 `is` logit after exact decoding to binary64.

| Observation | Earlier CPU margin | GPU margin |
| --- | ---: | ---: |
| Target | +5.20401430 | +5.20401454 |
| Donor | −4.26625872 | −4.26626062 |
| Self-patched target | +5.20401430 | +5.20401454 |
| Donor-patched target | +0.05723381 | +0.05722809 |
| Target minus donor-patched target | +5.14678049 | +5.14678645 |

The positive movement means the margin decreased toward `are`; it does not by
itself establish a preference reversal or closeness to the donor. The
earlier CPU comparison is from the same exposed pair, not the previous
32-case A/B-averaged effect. Small numerical differences across runtimes are
expected; no CPU/GPU bit equality is claimed.

The retained target, donor, self-patch and cross-patch activation bits verify
that the repeated target states match and that the intended source landed at
the selected site. The owner checked equality of the complete mask-position
vocabulary bytes for the target and self-patch. The saved full-mask digests
and selected logit bits agree for that control, but full mask-logit vectors
were not saved and cannot be reconstructed from their hashes. There was no
additional post-cross clean forward, and there was no different-lemma control.

The supervised owner used the pinned
`distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, float32 eager attention, seed 17,
one CPU thread and `cuda:0` on an NVIDIA GeForce RTX 2070 SUPER. The earlier
CPU evaluation used seed 2026102810. The [result](result.json) records the
remaining runtime settings and exact input IDs. The controller's observed
exit was zero and its saved supervisor reports a clean reaped owner exit zero;
the post-exit device query did not list the owner. This query and Windows WDDM
visibility do not establish complete GPU isolation or native teardown.

Across the 13 synchronized memory stages, the cumulative PyTorch peak since
the pre-model-load reset was **307,769,344 bytes allocated** and
**339,738,624 bytes reserved** (293.512 and 324 MiB). The minimum sampled
device-free value was **7,085,228,032 bytes** (6,757 MiB). The owner reported
zero PyTorch allocated and reserved bytes after cleanup and no surviving
model/input weak references. These are point samples and PyTorch allocator
counters for this four-forward sequence, not total GPU or host-memory maxima,
per-forward peaks, or a worst-case resource bound. External memory pressure
can change between samples.

Before this attempt, a separately reviewed
[qualified recovery](../agreement-gpu-resource-v1-20261029/RECOVERY.md)
retired a marker left by an incomplete earlier GPU pilot. That recovery did
not verify the earlier owner's CUDA or native teardown. This owner performed
its own fresh identity, ownership and memory-admission checks.

The [fixed protocol and executable source](https://github.com/EastonSell/model-looking-glass/blob/a39945187c383c0bac9cfb0cce950ce590357261/docs/BERT_GPU_EXPOSED_PAIR_V1.md)
were published at commit `a39945187c383c0bac9cfb0cce950ce590357261`
([PR #357](https://github.com/EastonSell/model-looking-glass/pull/357)) before
the attempt according to the operator and GitHub records; no separate trusted
publication timestamp was retained. The [CPU preflight observation](cpu-preflight.json)
and its [source](cpu_no_forward_preflight.py) record an offline load and
tokenization check with the model forward, CUDA lazy
initialization and selected socket calls blocked; it does not cover every
native network API. The [controller-exit observation](controller-exit.json)
is an operator record. Source checks and these observations rely on a trusted,
quiescent checkout and dependency/bytecode paths; they do not attest every
interpreter or native-library byte, prove that files never changed during
execution, or independently prove the reported physical UUID equality.

The public [result](result.json), SHA-256
`8537dca3716c6ebcccec8e25d4bd3052300b73193edd92336cd53ce357e6b4a0`, contains
the projected measurement, saved vectors, selected bits, source hashes and
typed memory observations. The [stage records](stages/) are retained
separately. An independently written [numeric auditor](audit_saved_pair.py)
recomputes the saved vector controls, margins and memory checks without a
model or CUDA; its [report](independent-numeric-audit.json) states its limits.
From this directory it can be rerun with:

```text
python -B audit_saved_pair.py result.json 8537dca3716c6ebcccec8e25d4bd3052300b73193edd92336cd53ce357e6b4a0 a39945187c383c0bac9cfb0cce950ce590357261
```

Independent read-only reviews also checked the private
candidate/final/supervisor linkage and are summarized in the
[custody audit](independent-custody-audit.json) and
[operational review](operational-review.json). Hashes in the latter for withheld
inventories are commitments, not public copies or independent proof of their
contents. The UUID-bearing operator inventory and process-identifying private
records are not included here. These reviews verify saved-record consistency,
not physical execution or operating-system context teardown.
The [custody audit source](audit_private_chain.py) is included for inspection;
its complete replay requires the withheld private records and local cache.
The [evidence inventory](evidence-inventory.json) records the size and SHA-256
of each exported evidence file; it does not include this README or itself.
