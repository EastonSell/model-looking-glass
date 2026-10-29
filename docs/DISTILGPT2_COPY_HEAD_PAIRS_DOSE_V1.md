# DistilGPT2 copying: pairwise head controls at two doses

This prospective **public exploratory** protocol is fixed before assembling
its new cases or forwarding them. The preceding [matched-dose factorial](../results/distilgpt2-copy-head-factorial-v1-20261028/README.md)
found a mean output-scale interaction of −0.024855 nats for H2 and H10,
with one positive and several near-zero family values. H7 was only a
single-head comparator. The new experiment adds H2+H7 and H7+H10 pair
comparators, a second pair dose, and a total-input-dose-matched single-head
control for the lower pair dose. It does not claim a blind evaluation,
pretraining independence, a randomization distribution, or a unique circuit.

## Fixed public cases

Use `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa` and its pinned tokenizer. After
this protocol is public, construct and separately publish eight ordered groups
of six distinct, space-prefixed, single-token words. The 48 token IDs must
be mutually distinct and disjoint from all four earlier copying case sets.
Selection and structural collision resolution must not use model outcomes.
For each group `(a,b,c,d,e,f)`, use the ordered six-token prefixes
`(a,b,c,d,e,f)`, `(e,c,a,d,b,f)`, `(b,d,e,c,a,f)`, and `(d,a,e,b,c,f)`,
each followed by `(a,b,c,d,e)`. The target is the next token `f`. Verify each
complete text tokenizes to the listed 11 IDs, and that appending the target
preserves the prefix and adds exactly one token. The first variant repeats a
prefix; the other three are fixed matched permutations, not random draws.
Publish all words, texts, IDs, targets and their fixed order before a forward.

Before choosing cases, bind the newest independently reviewed public
DistilGPT2 exposure inventory, at least the successor recording the preceding
factorial as observed, and its canonical digest. Reject any proposed complete
input-plus-target row that equals, or is a complete-row prefix of, an
inventoried row or another proposed row in either direction. Also reject a
row if any nonempty prefix appears in the bound known-prefix set; a shared
first token alone therefore rejects it. Record the version and digest in the
case file. Reserve all 32 exact rows in a separately reviewed append-only
successor before the first forward and verify all prior known prefixes are
preserved. If a newer reviewed inventory appears before execution, bind it
and recheck before running. Coordinate separately with the natural and
injection custodians about already drawn private splits: record only the scope
and pass/fail of a private row-overlap check, or the custodian's statement
that no split exists. This covers known exposure only. Do not disclose a
hidden site, nonce or private row, and never adapt public cases to outcomes.

## Runtime, intervention and matched doses

Use CPU, evaluation mode, eager attention, no KV cache, float32, one CPU
intra-op thread and the reviewed CPU/no-network guard used for the preceding
factorial. Publish the executor, the complete source inventory and its exact
runtime policy before forwarding. Record actual checkpoint and source hashes,
versions, seed, backend/environment and numeric settings, persistent-state
and nonpersistent-buffer hashes before and after. `MKL_CBWR` and
`ATEN_CPU_CAPABILITY` must be absent before importing Torch and remain absent.
Any alternative device requires a separately published, reviewed owner and
resource plan; the existing CPU result does not establish a GPU bound.

At zero-based layer 3, final input position 10, the concatenated attention
`c_proj` input has 12 zero-based heads of width 64. The single-head arms are
H2, H7 and H10; pair arms are H2+H7, H2+H10 and H7+H10. Before any
intervention for a family, run unhooked clean and captured no-op forwards for
each of its four variants, in that order. Full final-position clean and no-op
logits must match bit for bit, and the no-op's complete projection input
before and after its hook must match bit for bit. Those four retained no-op
inputs determine all doses and deliveries for that family.

For each variant `v` and head `h` in `{2,7,10}`, decode the 64 saved finite
binary32 source values into binary64 using their sign, exponent and
significand, including subnormals and signed zero. Compute, in binary64:

```
n[v,h] = sqrt(math.fsum(float(x)**2 for x in x[v,h]))
m = min(n[v,h] for all four variants and all three heads)
q_low = 0.5 * m
q_high = m
q_match = math.sqrt(2.0) * q_low
fraction[v,h,dose] = q_dose / n[v,h]
new[v,h,dose,j] = round_to_float32_nearest_even(
    (1.0 - fraction[v,h,dose]) * float(x[v,h,j]))
```

Evaluate the binary64 operations in the written order with nearest-even
rounding. Convert the resulting binary64 values to IEEE binary32 by an exact
integer-defined nearest-even conversion, independent of ambient float32
flush mode. Verify denormal preservation and test halfway ties, subnormal
boundaries and signed zeros. Recompute low-, match- and high-dose deliveries
from the original saved source, never from a previously rounded delivery.
Evaluate the square root, multiplication and later division for `q_match` in
the written binary64 order. Those singles match the requested whole-joint
input L2 of a low-dose pair; use low-dose singles, not the matching singles,
in the low-dose interaction.
Save original and delivered float32 bits. Require finite sources and norms,
`m > 2**-19`, finite deliveries, and for every head/variant/dose require the
achieved L2, computed from bit-decoded before/after values with binary64
`sqrt(math.fsum((after-before)**2))`, to be within `1e-5 * q_dose` of the
requested per-head dose. Record all source norms, requested fractions,
achieved norms, changed-bit counts and the joint combined norm. A nonfinite
value, invalid dose, bit mismatch or failed control ends the attempt
incomplete; preserve available evidence, do not replace a case or tune a
dose, and do not call a failed control a scientific negative result.

Each pair uses the exact delivered slice bits from the matching single arm at
the same prompt and dose; every non-target input bit must remain unchanged.
For the disjoint head slices, the requested whole-joint input L2 is
`sqrt(2) * q_dose`, not `q_dose`. Equal per-head absolute L2 does not match
fraction removed, direction, projection-output magnitude or residual-stream
effect. The `q_match` singles provide a requested total-input-L2 comparison
for low-dose pairs, not an equivalent intervention. Compute each low-dose
pair's achieved combined norm across both delivered slices with the same
binary64 sum-of-squares rule and require its difference from `q_match` to be
at most `1e-5 * q_match`. Retain that norm, each achieved matching-single
norm and their differences. High-dose pairs have no total-input-matched single
in this design: their requested total exceeds `m`, the available source norm
of at least one slice. The high-dose minimum-norm slice is fully zeroed; a
small H7 slice can reduce every dose. Because `m` uses all four variants, replacing a
shuffle can change the repeated prompt's dose. This is an intervention on
the fixed four-prompt bundle, not a context-independent per-prompt dose.

## Order and controls

The 15 intervention arms, before rotation, are:

```
H2-low, H7-low, H10-low, H2+H7-low, H2+H10-low, H7+H10-low,
H2-match, H7-match, H10-match,
H2-high, H7-high, H10-high, H2+H7-high, H2+H10-high, H7+H10-high
```

After the family's eight clean/no-op forwards, for each variant left-rotate
that list by `(4 * zero_based_family + variant) % 15`, run the 15 arms in
that order, and then run an unhooked clean-after. Across eight families each
variant uses eight distinct rotations; across all 32 prompts each rotation
occurs two or three times. The order is deterministic and not fully balanced,
and clean/no-op always precede intervention. Require every arm's complete
pre-action projection input to equal the corresponding no-op input bit for
bit, its post-action non-target bits to equal the source, and clean-after's
full final-position logits to equal clean bit for bit. Verify hook removal.
Any mismatch is incomplete, with returned partial evidence retained.

## Outcomes and additive-logit references

Save each full final-position vocabulary logit vector, target logit, target
log probability, rank and lowest-ID top token. Decode finite saved binary32
logits into binary64 as above and use
`(target - maximum) - math.log(math.fsum(math.exp(x - maximum) for x in logits))`.
Rank is one plus the number of logits strictly larger than the target, so
tied maxima have rank one. For each dose and arm define

```
delta[v] = logp_arm[v] - logp_clean[v]
D_arm = delta[repeated] - math.fsum(delta[three shuffles]) / 3
J_a_b = (D_a_plus_b - D_a) - D_b
```

Use the written binary64 order. Pairs are ordered `(2,7)`, `(2,10)`,
`(7,10)`, and `a,b` correspond to the first and second entries. Retain
every repeated delta and all three individual shuffle deltas, all eight
family values, ranks and clean/arm top-one counts, not only means. Report
`J_2_10 - J_2_7` and `J_2_10 - J_7_10` at each dose and each pair's
`J_high - J_low` by family; compute each eight-family mean with
`math.fsum(values) / 8`. Preserve weak, zero and opposite-sign comparisons.
Also report each low-dose pair's D and per-prompt deltas beside the `q_match`
single for each of its two heads, including `D_pair_low - D_single_match` in
that order. These comparisons match requested total input norm only; they do
not replace the same-component-dose singles in J or make the directions or
downstream effects equivalent.
If a comparator's mean J is equal to or more negative than H2+H10's, that
comparison does not support a more negative H2+H10-specific output contrast.
These fixed pairs are comparators, not assumed-zero nulls or a calibrated
randomization distribution.

For every prompt, dose and ordered pair, reconstruct one hypothetical
full-vocabulary additive-logit vector from saved single and clean logits:

```
z_add[j] = (z_a[j] - z_clean[j]) + z_b[j]
delta_add = logp(z_add) - logp_clean
D_add = delta_add[repeated] - math.fsum(delta_add[three shuffles]) / 3
J_add = (D_add - D_a) - D_b
joint_minus_additive = D_a_plus_b - D_add
```

Decode saved bits and evaluate subtraction, addition, log-sum-exp and the
family reductions in the written binary64 order, without rounding `z_add`
back to float32 or forwarding it through the model. Retain every prompt's
reference delta and every family/pair/dose `D_add`, `J_add` and
`joint_minus_additive`, plus their descriptive means. Check finiteness and
independently reconstruct the reference from saved bytes. If it fails,
preserve the measured results and mark the reference incomplete; do not
substitute another analysis. The algebraic equality of `J - J_add` and
`joint_minus_additive` need not be bit-identical under different operation
orders.

Separately repeat the per-prompt delta, D and pair interaction calculations
using the raw target logit rather than log probability. For each pair/dose
retain the raw-target-logit result for the hypothetical additive vector and
the difference from the observed pair, with the same reduction order and all
family values. These diagnostics distinguish the target component from its
log-softmax normalization; they do not eliminate other downstream effects or
establish direct head interaction.

For each pair/dose, also record the final-position `c_proj` output difference
norms from the no-op and the norm of `((pair - single_a) - single_b) + no-op`,
using saved binary32 bits and binary64 `sqrt(math.fsum(component**2))` in
that order. These are delivery diagnostics, not success thresholds. Output
log-softmax, downstream nonlinearities and rounding can generate J or a
difference from the additive-logit reference without direct head-to-head
causation. The reference is hypothetical, not the exact normalization
component of observed J. No dose response or comparator result alone proves
cooperation, mediation or a complete circuit. The cases are public, nonblind
and already shaped by earlier results; do not make a population or
pretraining-independence claim or treat the 32 prompts as independent model
samples.

## Evidence and failure scope

The schedule is 18 forwards per prompt, 576 total. Retain every 50,257-value
float32 final-position logit vector and, for each no-op or intervention,
the full 11-by-768 float32 projection input before and after action and its
full projection output. The successful raw-payload budget is
`32 * (18 * 50,257 * 4 + 16 * 3 * 11 * 768 * 4) = 167,696,640` bytes.
Bound metadata and failure/partial records separately, account for retained
physical copies separately from logical raw payload, and check free-disk
headroom before starting. Stream bounded per-case files and record lengths, offsets, hashes,
submission/return observations, controls, partial returned payloads on
failure, source/runtime identities and an observed direct-process exit or an
explicit incomplete/uncertain outcome. Do not automatically retry uncertain
attempts. Resource and storage budgets are not GPU VRAM guarantees.

Before reporting a complete result, independently reconstruct all scores,
delivered bits and doses, same-head/same-dose joint identity, controls,
pairwise and two-dose statistics, additive references and evidence hashes
from the saved bytes. Retain every arm, family, failure and null outcome.
Such a replay checks saved-data consistency; it is not an independent model
execution, proof of runtime authenticity or unique mechanism.
