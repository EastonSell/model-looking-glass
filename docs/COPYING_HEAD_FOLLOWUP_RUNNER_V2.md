# CPU runner for the expanded public copying follow-up

This implements the separately frozen public follow-up protocol. It compares
five fixed layer-3 heads at the final input position and head 10 at the earlier
target position. The two earlier development studies and the new disjoint
public families are exploratory. They are not a blind holdout, and this set of
controls is not a randomization distribution or an exhaustive search over
heads. Zeroing a head's projection input is a specific intervention; it does
not identify a unique circuit or remove the head's query/key computations.

For each of 32 cases, the action order is unhooked clean, a recorded no-op,
the six ablations, and unhooked clean-after. The six ablations rotate by
`(zero_based_family + 2 * variant) % 6`; each variant sees all six rotations
over the eight families, with two rotations occurring twice. The rotation is
deterministic rather than randomized or perfectly balanced, and family/case
order remains fixed. Earlier-position ablation can affect subsequent layers
through a different path from final-position ablation; its contrast is
reported separately rather than treated as an equivalent head control.

The runner verifies the frozen protocol and case hashes, the append-only
public exposure reservation, the earlier discovery attention ranking and
supervised first head-evaluation result, the source revision, offline
checkpoint and the fixed CPU runtime before any model forward. It checks the
full delivered projection input and untouched bits, records each selected
slice's pre-ablation L2 norm and changed bit-pattern count, compares all
pre-action inputs, and requires no-op and clean-after logits to match the
unhooked clean vector. Before/after model state, nonpersistent buffers, CPU
RNG state and execution settings are checked. The cooperative source, device
and network checks do not attest against native or external process mutation.
The launch must leave both `MKL_CBWR` and `ATEN_CPU_CAPABILITY` absent before
Torch import; their presence and values are recorded and checked throughout.
Those variables were not recorded for the earlier public studies, so this run
does not establish a byte-identical CPU backend environment across studies.

Each `copy-followup-NN-V.f32` file contains headerless little-endian float32
payloads. The 288 completed records contain offsets, lengths and SHA-256
digests for every 50,257-value final logit vector and all seven hooked
actions' before/after/projection-output `[1,11,768]` witnesses. Returned
payloads are durably recorded before control checks, so failed comparisons
retain their evidence. A failed or interrupted attempt is incomplete and is
never resumed in place; the failure record and any partial files are retained.
An external supervisor must record a zero process exit and absence of a
failure record, and an independent auditor must replay saved bytes and
statistics before the result is accepted. A result file alone is insufficient.

The fixed run retains 80,604,288 tensor bytes: 57,896,064 for final logits
and 22,708,224 for witnesses. The runner enforces an 81,000,000-byte raw
payload ceiling, 302,404 retained bytes per hooked forward, and initial disk
headroom. These are logical retained-payload bounds, not limits on the full
transient logits, model parameters, workspace, Python objects, filesystem
copies, or process RSS. This is a direct CPU experiment, outside the live
broker, IPC and GPU execution paths.
