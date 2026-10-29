# Public copying factorial runner

`scripts/copying_head_factorial_v1.py` implements the separately published
[matched-dose protocol](DISTILGPT2_COPY_HEAD_FACTORIAL_V1.md) on CPU. It is a
trusted, single-process research script. The fixed cases are public and
exploratory; the checks do not establish blindness, absence from pretraining,
causal mediation or a unique circuit.

For each family, four genuinely unhooked clean forwards and four recorded
no-ops precede all intervention forwards. The no-op inputs determine the
family's common dose: the minimum of the twelve H2/H10/H7 final-position
slice norms. Every single head removes that absolute L2 amount; the joint
arm uses the same delivered H2 and H10 bits as the singles. The smallest
slice is fully zeroed. This matches neither fractions nor directions, and
the joint arm has a larger combined perturbation. A dose at or below
`2**-20`, nonfinite data, or a realized-dose mismatch aborts the attempt.

Binary32 input values are decoded by their bits, the fixed formula is
evaluated in binary64, and the binary64 results are rounded to binary32 by
integer nearest-even logic. The original and delivered bits, achieved doses,
fractions and changed-bit counts are retained, including signed zeros.
Runtime probes check representative binary64 arithmetic and the CPU
denormal policy; these checks do not attest every native thread or physical
memory allocation. Joint and single slice bits must agree. No-op and
clean-after final logits must match the clean logits; other projection-input
coordinates must be unchanged.

The result includes all four case deltas, repeated and shuffle components,
family contrasts, per-prompt log-probability interactions and the
protocol-ordered family interaction. Projection-output deltas and the
joint-minus-singles output residual are descriptive diagnostics, without a
pass threshold. H7 is a single-head comparator, not a two-head interaction
control. Log-softmax or later nonlinear computation can create an output
interaction without direct head cooperation.

The separately published additive-logit clarification is bound by hash and
is reconstructed from the saved logits by an independent audit. It is a
hypothetical output-scale reference, not proof of a mechanism or an exact
decomposition of the observed interaction. A separately bound prelaunch
record must document the current natural and injection custodian responses,
without exposing private cases, and any later reviewed exposure inventory.

The script saves each bounded returned payload before post-return validation.
When a helper fails after a model return or after a bounded projection input
has been captured, it attempts to save the available bytes as a partial
record before failing. Wrong shapes or types, storage errors and asynchronous
interruptions may prevent a complete capture. Failure metadata distinguishes
observed model returns, successful helper payload returns and partial/full
raw writes; it is not independent evidence that a model call did or did not
occur. Any failed check or uncertain process exit is incomplete. Do not
automatically retry an uncertain attempt.

A successful run retains 61,250,432 raw tensor bytes in 32 per-case files,
plus metadata, model memory and transient copies. The fixed byte limit is a
retained-payload bound, not an RSS or machine-wide memory bound. Before any
forward, the exact protocol, case and exposure reservation, prior evidence,
source tree, dependencies and CPU backend environment must match their pins.
The runner requires `CUDA_VISIBLE_DEVICES=-1`, offline Hugging Face flags,
absent `MKL_CBWR` and `ATEN_CPU_CAPABILITY` before PyTorch's first import,
`-B`, and a fresh absolute `-X pycache_prefix`. No broker, GPU or IPC path is
used. Independently replay the retained bytes and observed process receipt
before reporting outcomes.
