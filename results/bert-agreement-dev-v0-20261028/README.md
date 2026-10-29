# DistilBERT masked agreement: exploratory development v0

These are four **exposed development prompts**, not held-out results. The
pilot motivated the later draft candidate; it does not establish an agreement
mechanism, generalization, or preservation of other capabilities. Both model
and tokenizer use `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`; the verified safetensors SHA-256 is
`5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`.
The original file lists all five asset hashes. Both runs were CPU-only with
CUDA hidden, PyTorch reporting CUDA uninitialized, one thread and seed
`2026102810`. The GPU was not measured.

Each input had ten tokens, subject at position 2 and mask at position 6.
The action replaced the subject vector at the output of **zero-based encoder
block 2**, the third block, with the same-lemma, opposite-number donor vector.
The full tokens, margins and donor IDs are in `result.json`; the independent
replay also records the exact input IDs and individual `is` and `are` logits.
The original baseline and patched margins were:

| Exposed prompt | Base is−are | Patched is−are | Change |
| --- | ---: | ---: | ---: |
| The farmer near the nurses [MASK] ready. | 1.5943 | -4.4023 | -5.9966 |
| The farmers near the nurses [MASK] ready. | -5.0622 | -2.0642 | +2.9980 |
| The pilot near the soldiers [MASK] ready. | 0.9104 | -5.1053 | -6.0158 |
| The pilots near the soldiers [MASK] ready. | -5.6391 | -1.7471 | +3.8920 |

All four changes point toward the donor's number in this margin, and both
singular targets change the preferred token. All distractors were plural, so
target number is confounded with subject–distractor congruence. The patch also
changes surface form; there were no different-lemma or rival-position controls
in v0. A donorward shift can arise through selective erasure or other features,
and does not prove a unique transferred number representation.

## Original record and separate replay

The original `result.json` is preserved unchanged (SHA-256
`2602232c7b0d5382f578b455b83496721ce6a68ff4a9fef398cdb033862387e0`). Its
`self_patch_bitwise_equal` field is **misnamed**: the original script used
`torch.equal`, which checks numeric tensor equality and does not prove raw-bit
identity. `final_mask_patch_matches_donor_logits` also uses `torch.equal`.
The original hook did not record a post-action landing check. The original
attention setting was requested as `eager`, without a recorded readback.

The separately recorded replay uses only those same four exposed prompts. It
matched all four original baseline, donor and patched margins and changes.
Its additional instrumentation checked replacement bits immediately after
the action, compared int32 views of contiguous copies of the complete
self-patched logits and baseline, compared int32 views of contiguous copies of
the final-mask patched and donor mask logits, and read back `eager` from the
instantiated config. Tensor strides and floating-point denormal mode were not
recorded; these are checks of the resulting copies.
Those checks passed in this replay; they are new observations, not proof that
the original record had raw-bit equality. Replacing the *final block's mask*
state is expected to reproduce the donor mask logits because the remaining
masked-LM head is positionwise; it is a delivery check, not localization.

The original and replay source files were local and uncommitted when executed.
Their recorded source SHA-256 values identify the files read by the scripts;
they do not independently authenticate which bytecode executed. These v0
records omit full Python/Transformers and numeric-backend state. Neither run
saved full raw output tensors, so a reader cannot reconstruct their complete
logits or independently verify the stated raw-bit checks from the JSON alone.
The replay scripts expect their cache/output paths beside the script; the
preserved copies are evidence, not a claim of path-independent replay.

The original process reported a 1,012,015,104-byte peak Windows working set
after forwards. This is an observed process high-water mark, not a guaranteed
host-memory limit and not a GPU-memory profile.

Original source SHA-256:
`f6604bb13246c2c14f12f87aea32b457c1b6e4ea850418e72246860a6b0eadaf`.
Replay source SHA-256:
`1847a4a9866c9de8b2c4c7b5ff07ae97ade1275f63ac59f3ddd51deee006550c`.
Replay result SHA-256:
`d0c4acfec26b80fd8609b3a005e3970c73eff76f6686ddc30823d1df1f89076a`.
