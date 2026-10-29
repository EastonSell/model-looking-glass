# DistilGPT2 copying: matched-dose head factorial, version 1

This is a prospective **public exploratory** experiment. It is fixed before
assembling its new case file or forwarding any of its inputs. It follows the
[six-arm result](../results/distilgpt2-copy-head-followup-v2-20261028/README.md):
on eight public families, final-position head 2 had a more negative
repeated-minus-shuffle contrast change than head 10 in seven families. This
study asks how same-dose interventions on those two heads behave separately
and together. It does not claim a unique circuit, a blind test, pretraining
independence, or population-level inference.

## Cases and fixed order

Use the pinned `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa` and its pinned tokenizer from the
previous study. After this protocol is public, separately construct and
publish exactly eight groups of six distinct one-token, space-prefixed words.
All 48 token IDs must be mutually distinct and disjoint from the token IDs in
all three previous copying case sets. Do not choose or alter them using model
outcomes. For each ordered group `(a,b,c,d,e,f)`, form four 11-token inputs
using the six-token orders `(a,b,c,d,e,f)`, `(e,c,a,d,b,f)`,
`(b,d,e,c,a,f)`, `(d,a,e,b,c,f)`, followed by `(a,b,c,d,e)`. The scored
next-token target is `f`. Verify each full text tokenizes to its saved IDs and
that appending the target preserves the prefix and adds exactly one token.
The first variant is repeated; the other three are matched shuffles. Publish
all words, texts, token IDs, target IDs and fixed group/variant order before
any of these forwards.

Before choosing these cases, bind the newest independently reviewed public
DistilGPT2 exposure inventory and its canonical digest, at least a successor
of v10. Reject any proposed complete input-plus-target row that is equal to,
or a complete-row prefix of, an inventoried row or another proposed row in
either direction. Also reject a proposed row if any of its nonempty prefixes
appears in the bound known-prefix set. The separately published case file
records the inventory version and digest. Reserve all 32 exact rows in a
reviewed append-only successor before the first forward, and recheck that it
contains all prior known prefixes. If a newer reviewed snapshot appears
before execution, bind it too and recheck the proposed rows before running.
Coordinate with the natural and injection custodians about any already drawn
private splits; record only the scope and pass/fail of a private overlap check,
or the custodians' statement that no split exists. This inventory guards
against known exposure; it cannot establish unknown exposure or blindness.
Resolve structural collisions before a forward; never adapt cases to outcomes.
The prefix rule is intentionally strong: a shared first token alone rejects a
candidate, even when the complete rows differ.

## Intervention and dose

Use evaluation mode, eager attention, no KV cache, float32, one CPU intra-op
thread, and the checked CPU/no-network policy of the preceding study. Freeze
the executor and runtime source and publish them before forwarding. Report
actual versions, checkpoint/source hashes, seed, backend environment, numeric
settings, persistent-state and nonpersistent-buffer hashes before and after.
The experiment remains on CPU unless a separately reviewed execution path is
published before use.

In zero-based layer 3 at zero-based final input position 10, the concatenated
`c_proj` input has 12 zero-based heads of width 64. The arms are H10 alone,
H2 alone, H7 alone, and H2 plus
H10 together. H7 is the fixed same-layer comparator from the previous study;
it is not a randomization distribution. Before any intervention in a family,
run an unhooked clean and a captured no-op for each of its four variants, in
variant order. Check the complete final-position clean and no-op logits match
bit for bit, and require the no-op's full pre- and post-action projection
inputs to be bit-identical. Retain that no-op input for each variant; it is
the source for **all** of that family's dose calculations.

For every variant `v` and head `h` in `{H2,H10,H7}`, let `x[v,h]` be its
64 saved finite float32 values. Compute, in binary64,

```
n[v,h] = sqrt(math.fsum(float(x)**2 for x in x[v,h]))
m = min(n[v,h] for all four variants and all three heads)
a[v,h] = m / n[v,h]
new[v,h,j] = round_to_float32_nearest_even((1.0 - a[v,h]) * float(x[v,h,j]))
```

Decode the saved binary32 bit pattern into binary64 by its sign, exponent and
significand, including subnormals and signed zero, rather than relying on an
ambient float32 cast. The subtraction, division and multiplication above are
binary64 operations with round-to-nearest, ties-to-even; the final conversion
is an exact integer-defined rounding
of the resulting binary64 value to IEEE binary32, nearest with ties to even,
independent of ambient float32 flush-to-zero mode. Require the runtime's
denormal-preservation probe and test the conversion on ties, subnormal
boundaries and signed zeros. Save the exact original and delivered float32
bits, including signed zeros.
The rule uses only pre-intervention activations, not logits or outcomes. It
sets the requested **absolute** perturbation L2 to the same `m` across the
12 single-head slices of a family. The minimum-norm slice is fully zeroed;
a small H7 slice can make every dose in that family small. Fractions removed,
directions and doses across different families are not matched, and equal
input-slice L2 does not equalize `c_proj` output or residual-stream effects.
The joint arm
uses exactly the same delivered H2 and H10 bit patterns as their single arms,
so its combined requested L2 is `sqrt(2) * m`, not `m`.

For each slice compute its achieved L2 from the saved binary32 before/after
values using binary64 `sqrt(math.fsum((after-before)**2))`. Require it to be
finite and within `1e-5 * m` of `m`. Record `m`, all 12 original norms, all
requested fractions, achieved norms and changed-bit counts, including both
joint components and the joint combined norm. If any source is nonfinite, any
norm is nonfinite, `m <= 2**-20`, a delivered bit pattern differs from the
specified conversion, or the achieved-dose check fails, preserve available
evidence and end the attempt **incomplete**. Do not tune, replace a case, or
call a failed control a negative scientific finding.

For each variant after its family's eight baseline forwards, run the four
intervention arms in the left-rotated order of `(H10,H2,H7,H2+H10)` by
`(zero_based_family + variant) % 4`, followed by an unhooked clean-after.
This deterministic schedule places each arm twice in each slot for each
variant across eight families; it is not random. Require every arm's full
pre-action projection input to match the corresponding no-op input bit for
bit, and require all non-target coordinates of its post-action input to be
bit-identical to the source. Require clean-after's full final-position logits
to equal the original clean logits bit for bit, and verify hook cleanup.
Any mismatch ends the attempt incomplete, retaining returned evidence.

## Outcomes and interpretation

For each prompt and arm, save the full final-position vocabulary logits,
target logit, target log probability, rank and lowest-ID top token. Compute
target log probability in binary64 from finite saved float32 logits, decoded
from their bits as above, as
`(target - maximum) - math.log(math.fsum(math.exp(x - maximum) for x in logits))`.
Rank is one plus the number of
strictly larger logits, so tied maxima all have rank one. For an arm, let
`delta[v] = logp_arm[v] - logp_clean[v]` and for each family define

```
D_arm = delta[repeated] - math.fsum(delta[the three shuffles]) / 3
J_family = (D_joint - D_H2) - D_H10
```

Evaluate these operations in the written binary64 order. Compute each
eight-family mean as `math.fsum(the eight family values) / 8`. Mathematically,
`J_family` is the repeated-minus-mean-shuffles contrast of the per-prompt
output interaction `logp_joint - logp_H2 - logp_H10 + logp_clean`; the two
floating-point evaluation orders need not be bit-identical.
Preserve all eight family values and components for each arm, H10-minus-H2
and H10/H2-minus-H7 comparisons, per-case doses and ranks, and top-one counts
for clean and every arm. Report means as descriptive summaries, including
zero, opposite-sign, weak or unchanged-top-one outcomes. Do not calculate a
significance test or treat heads, controls or the 32 prompts as independent
model samples. The interaction is defined on **log probability**: log-softmax
and nonlinear downstream computation can create it even without direct
head-to-head causation or a cooperative mechanism. Its sign or absence does
not establish mediation, additive natural computation, or a complete circuit.

As delivery diagnostics, use the saved `c_proj` output witnesses to record
the binary64 L2 norm of each arm's final-position output difference from its
no-op and of the final-position residual `((joint - H2) - H10) + no-op`.
Use bit-decoded float32 values and `sqrt(math.fsum(component**2))` in the
written order. These diagnostics do not select cases or define a success
threshold; numerical output residuals alone do not prove head interaction.

## Scope, failure, and retained evidence

The fixed schedule has 7 forwards per prompt, 224 total. Retain each
50,257-element float32 final-position logit vector. For the no-op and four
intervention forwards retain the complete 11-by-768 float32 `c_proj` input
before and after action and its full output. The successful raw-payload total
is `32 * (7 * 50,257 * 4 + 15 * 11 * 768 * 4) = 61,250,432` bytes. Stream
bounded per-case files and record offsets, lengths, hashes, start/return and
control results. Preserve the exact protocol, case and exposure digests,
source/runtime identity, the attempt log, partial available records on
failure, and an observed direct-process exit or an explicit incomplete/
uncertain outcome. Do not automatically retry an uncertain attempt. This is
not a claim of descendant supervision or independent execution attestation.

Before publication, independently reconstruct scores, delivered slices,
same-dose controls, family statistics and source/evidence hashes from the
saved bytes; a separate saved-data audit is not a model re-execution. Publish
all arms and family outcomes, including failures or null findings, with the
non-blind and dose-matching limits above.
