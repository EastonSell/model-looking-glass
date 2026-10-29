# Natural agreement patch executor

This is a low-level, single-owner executor for the natural track of
`PRETRAINED_NUMBER_PATCH_V1.md`. Its tests use synthetic models; this module
alone is not a pretrained discovery result and does not run or generate any
held-out cases. A separate controller must bind the published protocol, source
revision, tokenizer, checkpoint, reviewed public preflight and exact case labels
and tokenization before a real study. It must publish the reviewed runner and
runtime configuration before the first discovery forward.

`NaturalAgreementRunner(model, journal, device='cpu', expected_quartets=4)`
accepts `AgreementCase` objects for complete, aligned two-lemma by two-number
quartets. `AgreementJournal` requires a fresh absolute attempt directory. The
runner is single-use, uses one process-local model lease, and currently rejects
CUDA. It requires a dedicated trusted CPU process: it fixes the CPU RNG seed to
`20261028` and restores the prior CPU RNG state, but it cannot serialize
unrelated threads sharing global RNG, PyTorch settings, model state or hooks.
It records the seed, seeded RNG digest, thread/deterministic settings, cases,
all-one masks and sites. It checks eval mode, eager attention, no cache, CPU
float32 parameters and floating buffers, and disabled CPU autocast under the
lease before and after forwards. It records model bytes and compares parameter
and buffer identities, storage layouts, aliases, module tree and hook state
before releasing ownership. Changed or uninspectable cleanup is quarantined;
discard that worker. These checks rely on cooperative process ownership and do
not establish protection against transient hostile modifications.

Each family uses 152 root forwards: four clean captures, four clean repeats,
four nulls, 128 candidate target/donor patches, eight final-block position
controls and four final clean checks. Cross-input source vectors come from the
same attempt's clean captures; self-patches capture their source immediately
before patching in that same invocation. Each action records its selected
pre/post vector and full module output pre/post bytes, validates the selected
landing against the donor and verifies that all other positions are unchanged.
The null and self controls compare activation and last-token full-vocabulary
logits; the final-block last/subject controls use the same-lemma opposite-number
donor. Their meanings and limits are specified in the protocol. An independent
synthetic GPT-2 serial-hook oracle is maintained separately.

The executor stores all four continuation logits for each patch, and complete
last-token logits for clean and control forwards. Stored binary32 values are
decoded from integer bit fields into binary64 for the protocol's tolerance,
margin and L2 calculations, including subnormals. Present-tense factorial
results, ranking, positive-or-abstain selection, fixed random/earliest sites,
and equal-family magnitude baseline are recorded. The past-tense contrast is
stored separately and never selects a site. Each site's pair records include
endpoint margins and signs, strict sign reversals, and transitions involving
zero. These contrasts measure sensitivity to the specified replacements; they do
not by themselves establish an abstract number computation or donor transfer.

Journal payloads are immutable content-addressed float32 bytes with shape and
digest metadata. The default per-forward cap is 1 MiB, the per-attempt retained
tensor cap is 512 MiB, the cumulative event-log cap is 64 MiB, and event count
and individual metadata size are bounded separately. These caps do not measure
transient RAM or allocator overhead. The protocol's 512 MiB study-wide limit
spans discovery, held-out and participating tracks; a controller must enforce
that cumulative budget across attempts. Completed forwards are recorded before
honoring a cancellation observed during them. Failed records make the attempt
incomplete. When a failed plan does not expose its partial captures, the journal
states that they are unavailable; available rejected or pending forward data is
saved on a best-effort bounded basis. A prior primary error is retained over
secondary cleanup or evidence errors, including later interruptions.

`terminal.json` with `status='complete_candidate'` is an owner-side candidate,
not an authoritative experiment success. Its controller must confirm that the
actual owner exited successfully and was reaped, audit event order, counts,
shapes, referenced payload digests, total bytes, controls and the exact input,
source and runtime binding before publishing a result. A failed or interrupted
terminal write may leave a pending file or a candidate, so a candidate alone is
insufficient. The file and event writes are flushed and fsynced, but directory
entry persistence across sudden power loss is not established. Preserve failed
and uncertain attempts rather than overwriting them.
