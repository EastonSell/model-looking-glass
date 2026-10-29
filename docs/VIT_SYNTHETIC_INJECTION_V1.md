# ViT synthetic known-injection localization: fixed protocol

This is a small mechanics experiment on `WinKawaks/vit-tiny-patch16-224`,
revision `77d1485af66b34d4ed0fe95dbb0c60c7496f950b`, using the existing
verified local assets and eager CPU float32 inference. It uses no photographs
or image-study cases. Synthetic tensors are already processed model inputs,
not examples of natural visual reasoning.

## Inputs and injection

For case seed `s`, channel `c`, row `y` and column `x`, the input is exactly
`(((17*x + 31*y + 47*c + 97*s) % 257) - 128) / 128` in float32, with shape
`[1, 3, 224, 224]`. The four discovery seeds are 11, 12, 13 and 14; the four
locally held-out seeds are 21, 22, 23 and 24. These are fixed procedural cases,
not evidence that inputs or patterns are absent from pretraining. Discovery
and held-out tensors are distinct; the recipe and labels are public.

The four candidate processed-grid coordinates are `(1,1)`, `(3,9)`, `(7,4)`
and `(11,11)` in row-major order, giving token indices 16, 52, 103 and 166.
At the single invocation of `vit.embeddings`, add a length-192 vector whose
coordinate 7 is +4.0 and all other coordinates are zero to exactly one patch
token. Discovery injection indices are 16, 52, 103 and 166 respectively;
held-out indices are 103, 166, 16 and 52 respectively. A same-forward full
embedding capture before and after the addition must show the requested
float32 result at that position and identical bits everywhere else. Its actual
L2 displacement is recorded; a nonfinite or bitwise unchanged delivery stops
the attempt as an invalid treatment. CLS
is token zero and is never the injected token.

## Selection and comparison

Capture the clean and injected output of block 0 at the four patch tokens and
CLS, with complete layer, module, invocation, batch, grid and processed-pixel
coordinates. Use the same input and model in each pair. The paired trace
comparison reports observed differences; it cannot establish causal origin.
Let `delta` be injected minus clean, computed by decoding the saved float32
bits into binary64. Supervised discovery fits one direction by averaging the four discovery deltas
at their known injected patch coordinates. If that direction is zero or any
required value is unavailable, the selection is ineligible; retain the case
and reason without replacement. A delivered injection with zero observed
downstream change is retained as a valid negative, rather than excluded.

After the four discovery pairs and before the first held-out forward, write
the fitted direction's exact binary64 bytes and SHA-256, discovery scores,
selection rule, fixed random choices and source/protocol identities to a
retained freeze record. The held-out evaluation binds that record's digest.
This ordering is an operator observation, not independent timing proof.

On each held-out case a selector receives only that fixed direction and the
four ordered candidate deltas, never the injection-coordinate label. It chooses
the largest dot product; ties go to the smallest token index. Compare this
site-blind rule with the largest-delta-L2 rule (same tie break) and a random
control: Python `random.Random(218741).choices([16,52,103,166], k=4)` evaluated
once in held-out order, yielding `[166, 166, 52, 166]`. Record the full scores,
all choices and actual labels,
including misses. The planned prediction is that the directional rule chooses
the injected patch coordinate in at least three of four held-out cases and
does at least as well as the fixed random choices. Magnitude is a second
descriptive comparator; there is no statistical-significance claim. Each
candidate is a downstream block output, not the actual embedding boundary.
Coordinate recovery therefore does not locate a unique causal pathway.
The four procedural inputs are closely related and the block-0 residual is
structurally assisted by the implanted perturbation; the 3/4 threshold is a
small conditional mechanics test, not evidence of generalization.
The random sequence is one fixed realization; the uniform single-draw reference
is 1/4. If all candidate scores or all deltas are zero, the rule still makes
the declared lowest-index forced choice; that outcome is not evidence of
detection. The held-out selector is label-blind, but the published cases and
supervised discovery do not make the whole study blinded.

## Restoration and controls

For every eligible held-out selection, reapply the exact injection and replace
the selected block-0 patch vector with the clean vector at that same token.
The off-target token is the smallest candidate other than the selected token
and the known injected token; this evaluation-only choice uses the injection
label, while the selector does not. Its replacement follows the clean-donor
direction at that token, scaled to the selected replacement's realized L2
dose. A same-site reflected control follows the opposite of the selected
clean-donor displacement at the same intended dose. Each proposed control is
measured after float32 rounding; it is called dose-matched only when its
nonzero realized dose is within 5% of the selected dose. A zero direction,
nonfinite replacement, unmatched result or unchanged injected logits is
reported as unavailable or unmatched, not a success or a matched null.
When the off-target clean displacement is smaller than the target dose, this
construction extrapolates beyond the clean off-target state: it is a matched
directional intervention, not an off-target restoration.

For each arm, record full same-forward pre/post embedding and block outputs,
the realized displacement, all untouched-position checks and full logits.
Require the injection pre/post bits to match the baseline injection, and the
block pre-state to match the baseline injected state. A self-patch and a
post-session clean repeat must match their corresponding baseline logits
bit-for-bit. The selected clean replacement must land bit-for-bit. Compare each arm's raw
full-logit L2 distance from the clean logits; preserve zero and negative
improvements. The planned descriptive prediction is that the selected clean
patch reduces this distance more than both eligible dose-matched controls on
at least three held-out cases. Ineligible controls do not count as a pass.
The reflected control is not a semantic null; the off-target control can also
affect the output. One local patch can only establish a successful intervention
at that boundary, not necessity, uniqueness or natural behavior.

## Execution and reporting

Freeze this protocol, implementation and tests in a publicly readable immutable
commit, with independent source review, before any pretrained forward. Use one
CPU thread, offline assets, deterministic inference and no concurrent model
owners. Validate the fixed asset hashes and model-state digest before and
after, and check hooks at each phase boundary. Keep all eight cases,
all failures, per-case predictions and controls. Evidence is bounded to 32 MiB
and must fit a fresh minimum 2 GiB free-space check on both source and output
volumes before model load and each case; stop incomplete if either check fails.
No pretrained retry or replacement
case is implied by an incomplete attempt. Capture budgets cover retained
evidence, not transient model memory. The inherited ActivationPlan direct-call
contract requires discarding the model/worker after uncertain fatal interruption.
The run and its saved-byte review are cooperative observations, not independent
proof of execution timing or model/runtime authenticity.

Run the source-bound bootstrap with an isolated interpreter, bytecode disabled,
and a fresh absent absolute bytecode-prefix path, for example
`python -I -B -X pycache_prefix=<fresh-absent-path> examples/vit_synthetic_injection_bootstrap.py --assets <pinned-assets> --output <new-directory> --published-source <reviewed-commit>`.
The bootstrap keeps the repository root off the general import path and
checks the exact source files against the commit before project imports. Retain
a separate direct process-exit observation. Treat a completion file as provisional
until the process exited zero and no incomplete marker is present; a failed write
can leave a partial file. The runner rejects short writes and reads every saved
tensor slice back against its length and digest before completion.
