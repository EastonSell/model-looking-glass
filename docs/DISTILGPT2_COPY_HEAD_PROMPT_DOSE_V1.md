# DistilGPT2 copying: per-prompt matched head dose

This is a prospective public exploratory follow-up, fixed before assembling
the new case bundle and before forwarding any of its inputs. The earlier
[zeroing result](../results/distilgpt2-copy-head-followup-v2-20261028/README.md)
found a larger repeated-versus-shuffled target-log-probability reduction for
layer 3 head 2 than head 10 in seven of eight families; those zeroed slices
had different norms. The later [matched-dose factorial](../results/distilgpt2-copy-head-factorial-v1-20261028/README.md)
found a smaller descriptive H2 advantage, but its dose was the minimum over
all four prompts and three heads in each family. Its prompts and outcomes are
discovery history, not held-out data for this follow-up.

Here the absolute removed activation L2 is matched **within each prompt** for
H2, H10 and the fixed same-layer H7 control. The dose can still differ between
the repeated and shuffled prompts, and H7 can set it. Equal pre-projection
input norms do not match the removed fraction, direction, projection-output
effect, residual-stream effect or downstream behavioral effect. The study
does not identify a unique copying circuit, causal mediation, a randomization
distribution, or a population effect. The earlier zeroing study remains intact.

## Fixed public family construction

Use `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa` and tokenizer SHA-256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
The following eight ordered public candidate pools are fixed; the categories
are only a way to assemble word lists, not a population sampling scheme:

1. `island desert mountain forest prairie marsh dune delta harbor lagoon reef volcano hillside plateau ravine tundra oasis beach jungle basin meadow canyon`
2. `apple pear peach plum lemon lime grape melon cherry berry lettuce olive potato bread muffin cereal bacon honey pasta yogurt apricot mango`
3. `lion tiger bear fox wolf deer moose raccoon badger otter seal shark trout robin parrot insect ant bee butterfly salmon lizard rabbit`
4. `shovel screwdriver pliers drill axe saw chisel hammer wrench rake spade scissor nail screw bolt nut wire pulley lever needle thread`
5. `drum harp banjo bass cymbal saxophone accordion opera orchestra festival chorus composer quartet music tempo harmony soprano trumpet cello whistle`
6. `airplane bicycle ship boat van sedan helicopter scooter carriage canoe kayak barge cruiser tram minivan sailboat spaceship hovercraft vessel cable`
7. `scarf mittens trousers sandal pajama robe bracelet tie necklace ring buckle zipper sleeve collar apron brooch pendant pocket ribbon cardigan`
8. `chair table desk sofa couch lamp clock frame cabinet window drawer stove oven kettle skillet fork blanket sheet mug rug vase`

The operator examined tokenization and public inventory eligibility for a
preliminary version of these pools before this protocol; no model forward or
model outcome selected a word. Prior published model results were already
known, so this is not a blind or representative sample.

Bind the latest independently reviewed public DistilGPT2 input inventory before
selection. The initial known version is v27, SHA-256
`ef42683a664dda450e82c69afedda34a8a34f2daf9f50f2db54f6c9d846f07e5`.
For each pool in order, retain each first-occurring word whose `" " + word`
tokenizes to exactly one token ID not used in the five preceding public copying
groups (development, original evaluation, head follow-up, factorial, and
pair-dose), nor in an earlier selected group. Scan six-word combinations of
the retained words in lexicographic order of their positions in the pool.
Select the first combination whose four complete inputs, constructed below,
have no equality or full-row-prefix relation in either direction to each
other, any earlier selected input, or a complete input in the bound public
inventory. If no combination qualifies, stop with a construction-ineligible
result; do not choose words based on model behavior. Publish the full pools,
selection rule, all selected words/IDs and all rejected structural candidates
needed to reproduce the first qualifying combination.

For each ordered group `(a,b,c,d,e,f)`, form 11-token inputs by concatenating
the following six-token prefixes with `(a,b,c,d,e)`:

```
repeat:     (a,b,c,d,e,f)
shuffle-1:  (e,c,a,d,b,f)
shuffle-2:  (b,d,e,c,a,f)
shuffle-3:  (d,a,e,b,c,f)
```

The scored continuation is `f`. Verify that each complete input tokenizes to
its exact saved IDs and appending `f` preserves the complete input and adds
exactly one token. The actual forward inputs are the 32 11-token prompts;
continuation encodings are comparison metadata, not extra forwards. All 48
selected word tokens are mutually distinct. Reserve the exact 32 public inputs
before the first forward, retain the previous inventory unchanged, and recheck
against a newer independently reviewed public inventory if one appears before
execution. Also compare the actual inputs against the available public 128-row
natural support and public 256-row successor pool; report those exact scopes.
An equality or full-row-prefix collision in either direction with either of
those two sets, or with the latest reviewed inventory, stops this attempt
before any forward; it is not a reason to replace cases after observation.
An available authorized custodian may report a scoped protected-row check
without disclosing rows. Otherwise identify that scope as unchecked; do not
open another study's protected inputs. None of these checks establishes absence
from pretraining, unknown exposures or related contexts.

## Intervention and numerical rules

Run CPU float32, evaluation mode, eager attention and no KV cache, with one
intra-op and inter-op thread. Publish the exact runner, dependencies, cases,
reservation and execution manifest before the first forward, with independent
review. Record checkpoint, tokenizer, model weights and buffers, source,
runtime, backend and denormal probes. Do not adjust the cases, thresholds or
arms after seeing outcomes.

At zero-based layer 3 and final input position 10, the `c_proj` input has 12
contiguous heads of 64 float32 coordinates. For each prompt, first run an
unhooked clean forward followed by a witnessed no-op. Their full final-position
logits must match bitwise; the no-op's complete before and after projection
inputs must be bitwise equal. For the no-op input of this *same prompt*, decode
each H2/H10/H7 float32 slice by its sign, exponent and significand into
binary64, and define

```
n_h = sqrt(math.fsum(float(x) ** 2 for x in slice_h))
q = min(n_H2, n_H10, n_H7)
```

Check that all three norms and `q` are finite before making a plan. If `q` is
at or below `2**-20`, do not divide or compute delivered slices. Otherwise
compute

```
fraction_h = q / n_h
delivered_h[j] = round_binary32_nearest_even((1.0 - fraction_h) * x_h[j])
```

Use the literal Python binary64 `** 2` and `math.fsum` reductions above, as in
the previous dose study; do not assume they are bit-identical to another
reduction implementation. Use the repository's integer-defined
binary64-to-binary32 conversion; save
exact original and delivered float32 bits. Measure each realized removal from
those bits as `sqrt(math.fsum((after-before)**2))` in binary64. For a valid
match, `q` must be finite and strictly greater than `2**-20`, each realized
norm must differ from `q` by at most `1e-5 * q`, and the spread between the
three realized norms must be at most `1e-5 * q`. Record `q` and source norms
for every valid no-op, and fractions, realized removals and changed-element/bit
counts whenever a plan exists. The planned dose uses activations, not clean
target or edited outputs.

If finite, otherwise valid no-op inputs yield a dose at or below the floor,
mark that prompt `dose-ineligible`; retain its full no-op witness, all three
head slices, source norms and `q`, and record `plan unavailable: dose floor`.
No fractions or planned deliveries exist for that row. If a positive-dose plan
cannot meet the fixed rounding/matching tolerance, mark it `dose-ineligible`
and retain its full no-op and planned-delivery evidence with that reason. In
either case skip its three interventions and run clean-after. Never replace it.
A family is evaluable only if all four prompts have valid matched doses. A
nonfinite tensor, wrong shape, inconsistent witness, failed forward or
uncertain cleanup is an invalid/incomplete attempt rather than a
dose-ineligible or negative result.

For an eligible prompt, run the three single-head arms `(H2,H10,H7)` in a
left-rotated order by `(4 * zero_based_family + zero_based_variant) % 3`, then
run unhooked clean-after. Each arm's full pre-action projection input must
equal that prompt's no-op input bitwise. The delivered slice must match its
planned bits; every other input coordinate must remain bit-identical. The
saved projection output is also retained. Clean-after logits must match the
original clean logits bitwise and hooks must be removed. Any failed check
invalidates the attempt and preserves available partial evidence.

## Outcomes

Save full final-position float32 vocabulary logits for every forward. From
finite bit-decoded logits calculate target logit, log probability, rank
(`1 + count(logit > target)`), unique top and tie count. For each arm compute
per-prompt `delta = logp_arm - logp_clean`; report the target-logit change,
target rank, top-token changes, `KL(P_clean || P_arm)` and total variation
over the full vocabulary. KL and TV include the target and are not pure
off-target measurements. Record the smallest token ID among tied maxima as the
representative and the number of tied maxima. A top-token comparison is
`changed` or `unchanged` only if both clean and intervention tops are unique;
otherwise it is `tied/indeterminate`, with both tie counts and representatives
retained. For each prompt also report the L2 change of the
768-value `c_proj` output, as a diagnostic of downstream dose differences.

For each evaluable four-prompt family and each arm, define

```
D_h = delta_h[repeat] - math.fsum(delta_h[shuffle] for shuffle in the three shuffles) / 3
C = D_H10 - D_H2
```

Positive `C` means the H2 intervention reduced this target contrast more than
H10 did. Also report each arm's repeated delta and three shuffle deltas, all
family values, the corresponding target-logit contrasts, H7 comparisons and
all full-distribution metrics. The three shuffles are fixed controls, not
independent samples. Different `q` values across the four prompts can weight
the family contrast differently; record them all. As a secondary descriptive
sensitivity check, compute the same `C` after dividing each prompt's
`delta_H10 - delta_H2` by that prompt's `q`; it is a finite-dose normalization,
not a derivative.

The fixed operational directional criterion is: at least six evaluable
families, strictly positive mean `C`, and strictly positive `C` in at least
three quarters of evaluable families. Ties count against the sign criterion.
If at least six families are evaluable and the criterion fails, report it as
not met. Fewer than six is ineligible, with all valid prompt and family data
still reported. No factual or target-top filter is applied; report clean
top/rank performance instead. Any invalid/incomplete execution takes
precedence over a numerical outcome. The criterion is descriptive, not a
significance test or evidence of effect equality, a general copying
mechanism, or complete generated behavior.

## Evidence and failure handling

The maximum is 192 forwards for 32 eligible prompts. For every no-op and arm,
retain the full 11×768 float32 projection input before and after action and
full projection output; retain every final 50,257-token logit vector. The
maximum tensor payload is 51,573,504 bytes, excluding metadata, model memory,
transient allocations and other workspace. Stream rows with case/action IDs,
offsets, lengths, hashes and start/return records. Save available partial
payloads and failure records on interruption; do not automatically retry an
uncertain attempt. Record the direct process exit and model/hook/state cleanup
without claiming an authenticated timestamp or process-tree attestation.
Independently reconstruct case binding, landed bits, dose, controls, all
outcomes and source/evidence hashes from the retained bytes before reporting;
saved-data replay is not independent model execution. Publish every result,
including underfill, negative criteria and failed attempts.
