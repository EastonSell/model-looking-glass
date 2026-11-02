# Public DistilBERT matched-dose taste-context follow-up

On six public sugar/lemon sentence pairs, the preregistered baseline rule
classified coffee, milk and soup as eligible and cake, pudding and cocoa as
ineligible. For eligible pairs, 29 of the 30 block-0–4 direction-specific
comparisons favored the ingredient-position transfer over the matched `the`
transfer. The exception was milk, block 0, lemon→sugar (-0.004739).
All block-5 comparisons and both same-position self-patch effects were zero.
These are repeated, dependent measurements on three convenience pairs, not
independent trials or evidence of a universal taste representation.

All three eligible pairs use the `after adding` template; all three ineligible
pairs use the `with some` template. Their ineligibility was not hidden or used
to stop forwards. This confounds eligibility and template, and the original
exploratory results were known before these public frames were assembled.
Pretraining and other unrecorded exposure are unknown; these are not held-out
confirmation cases.

The study used CPU DistilBERT revision
`12040accade4e8a0f71eabdb258fecc2e7e948be` and seed `20261030`. Its
[prospective protocol](../../docs/BERT_TASTE_MATCHED_DOSE_V1.md),
[12 reserved inputs](../../experiments/bert_taste_matched_dose_public_cases_v1.json),
and [observed exposure successor](../../experiments/bert_taste_matched_dose_public_exposures_v2.json)
were recorded separately. The operator observed public source commit
`74227e9909e4a485296065aced0368a1ccafba64` before launching the 516-forward
attempt; no independent trusted timestamp binds that exact head to the first
forward. Each block-output
comparison used the smaller of the ingredient and unchanged-token clean L2
distances as its requested radius; both same-site self-patches and full donor
replacements were retained. Equal norm matches displacement magnitude within
the documented float32 tolerance; it does not match direction, downstream
sensitivity, or activation naturalness. The `the` vector is contextual and is
not a promised no-effect null.

The 36 requested radii ranged from 0.457173 to 1.765979 (median
1.139973). All were set by the `the` distance; ingredient
fractions ranged 0.024398–0.257589, while the `the`
fraction was 1. The 144 realized intervention norms ranged 0.457172–
1.765979; their largest relative error from the requested radius was
7.03e-08. Delivered dose divided by recipient-vector norm ranged
2.52%–16.31% for ingredient and
3.47%–14.26% for `the`. Exact per-direction requested
fractions, realized norms, recipient-norm ratios, both self controls, full
donor effects and all comparisons are in [case-layer-effects.csv](case-layer-effects.csv).

## Baselines

Margins are `sweet` minus `sour` mask logits; probabilities are over the full
vocabulary. Eligibility requires a positive sugar margin, a negative lemon
margin, and a target word among each row's top five. Baseline full-logit bits
are retained in the journals, so the selected logits and probabilities can be
recomputed.

| Row | Pair eligible | Margin | P(sweet) | P(sour) | Top five |
|---|---|---:|---:|---:|---|
| coffee-sugar | yes | +0.564038 | 0.049752 | 0.028304 | good, better, delicious, sweet, pleasant |
| coffee-lemon | yes | -0.237923 | 0.036281 | 0.046027 | good, better, delicious, bitter, sour |
| milk-sugar | yes | +0.216306 | 0.067129 | 0.054072 | good, better, sweet, sour, bitter |
| milk-lemon | yes | -0.669939 | 0.045469 | 0.088852 | sour, better, good, bitter, lemon |
| soup-sugar | yes | +0.250435 | 0.069108 | 0.053798 | good, sweet, better, sour, delicious |
| soup-lemon | yes | -0.576127 | 0.039835 | 0.070871 | sour, lemon, good, better, bitter |
| cake-sugar | no | +1.538244 | 0.099236 | 0.021312 | sweet, good, delicious, rotten, vanilla |
| cake-lemon | no | +0.449010 | 0.076843 | 0.049046 | sweet, delicious, good, sour, bitter |
| pudding-sugar | no | +0.930407 | 0.075555 | 0.029798 | sweet, good, rotten, delicious, sour |
| pudding-lemon | no | +0.022707 | 0.053993 | 0.052781 | sweet, sour, delicious, bitter, good |
| cocoa-sugar | no | +1.119932 | 0.073727 | 0.024057 | sweet, good, delicious, salty, bitter |
| cocoa-lemon | no | +0.068521 | 0.042924 | 0.040081 | sweet, mixed, sour, bitter, pleasantly |

## Eligible-pair means

The signed advantage is ingredient matched-dose margin change minus `the`
matched-dose margin change for sugar→lemon, negated for lemon→sugar, so a
positive value favors the ingredient intervention in the donor's direction.
These are frame-order means over the three eligible frames for each direction;
the same frames occur across layers and are not independent replicates.

| Block output | Eligible pairs | Sugar→lemon | Lemon→sugar |
|---:|---:|---:|---:|
| 0 | 3 | +0.024315 | +0.002787 |
| 1 | 3 | +0.039077 | +0.015268 |
| 2 | 3 | +0.078217 | +0.042561 |
| 3 | 3 | +0.087631 | +0.065357 |
| 4 | 3 | +0.070309 | +0.039886 |
| 5 | 3 | +0.000000 | +0.000000 |

## Every case and direction

S→L means sugar donor to lemon recipient; L→S is the reverse. The three effect
columns are raw changes in the recipient's sweet-minus-sour margin. The final
column applies the direction sign above. Values are rounded for display; the
CSV and saved journals preserve full precision.

| Pair | Eligible | Block | Direction | Radius | Ingredient fraction | Full donor Δ | Matched ingredient Δ | Matched `the` Δ | Signed advantage |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| coffee | yes | 0 | S→L | 0.471653 | 0.025073 | +0.734702 | +0.024719 | -0.003405 | +0.028124 |
| coffee | yes | 0 | L→S | 0.471653 | 0.025073 | -0.742589 | -0.007425 | +0.003830 | +0.011255 |
| coffee | yes | 1 | S→L | 0.621576 | 0.038883 | +0.723059 | +0.032883 | -0.006426 | +0.039309 |
| coffee | yes | 1 | L→S | 0.621576 | 0.038883 | -0.698268 | -0.009508 | +0.007845 | +0.017353 |
| coffee | yes | 2 | S→L | 1.099151 | 0.073653 | +0.667105 | +0.057035 | -0.027208 | +0.084244 |
| coffee | yes | 2 | L→S | 1.099151 | 0.073653 | -0.618032 | -0.025817 | +0.024135 | +0.049952 |
| coffee | yes | 3 | S→L | 1.165112 | 0.097928 | +0.619430 | +0.064729 | -0.012583 | +0.077312 |
| coffee | yes | 3 | L→S | 1.165112 | 0.097928 | -0.622118 | -0.042945 | +0.010820 | +0.053765 |
| coffee | yes | 4 | S→L | 1.301293 | 0.121572 | +0.365689 | +0.058252 | -0.001614 | +0.059866 |
| coffee | yes | 4 | L→S | 1.301293 | 0.121572 | -0.395138 | -0.033682 | -0.001791 | +0.031891 |
| coffee | yes | 5 | S→L | 1.175415 | 0.206393 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| coffee | yes | 5 | L→S | 1.175415 | 0.206393 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| milk | yes | 0 | S→L | 0.462199 | 0.024563 | +0.782818 | +0.029829 | +0.006536 | +0.023293 |
| milk | yes | 0 | L→S | 0.462199 | 0.024563 | -0.817090 | -0.000891 | -0.005630 | -0.004739 |
| milk | yes | 1 | S→L | 0.659729 | 0.040076 | +0.771234 | +0.041007 | +0.001844 | +0.039163 |
| milk | yes | 1 | L→S | 0.659729 | 0.040076 | -0.780440 | -0.007195 | -0.000107 | +0.007088 |
| milk | yes | 2 | S→L | 1.116279 | 0.072580 | +0.713627 | +0.068079 | -0.009797 | +0.077876 |
| milk | yes | 2 | L→S | 1.116279 | 0.072580 | -0.695996 | -0.021781 | +0.007746 | +0.029527 |
| milk | yes | 3 | S→L | 1.199057 | 0.097570 | +0.669277 | +0.072290 | -0.021008 | +0.093299 |
| milk | yes | 3 | L→S | 1.199057 | 0.097570 | -0.724220 | -0.044526 | +0.015642 | +0.060168 |
| milk | yes | 4 | S→L | 1.377656 | 0.124750 | +0.421450 | +0.074960 | -0.003912 | +0.078872 |
| milk | yes | 4 | L→S | 1.377656 | 0.124750 | -0.484349 | -0.036872 | -0.001700 | +0.035172 |
| milk | yes | 5 | S→L | 1.311270 | 0.220709 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| milk | yes | 5 | L→S | 1.311270 | 0.220709 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| soup | yes | 0 | S→L | 0.457173 | 0.024398 | +0.812188 | +0.026137 | +0.004610 | +0.021527 |
| soup | yes | 0 | L→S | 0.457173 | 0.024398 | -0.811745 | -0.005868 | -0.004023 | +0.001845 |
| soup | yes | 1 | S→L | 0.574387 | 0.035864 | +0.795949 | +0.030513 | -0.008247 | +0.038760 |
| soup | yes | 1 | L→S | 0.574387 | 0.035864 | -0.768158 | -0.011103 | +0.010262 | +0.021364 |
| soup | yes | 2 | S→L | 1.017734 | 0.067804 | +0.743401 | +0.052328 | -0.020204 | +0.072532 |
| soup | yes | 2 | L→S | 1.017734 | 0.067804 | -0.680316 | -0.028650 | +0.019554 | +0.048204 |
| soup | yes | 3 | S→L | 1.102170 | 0.092465 | +0.700734 | +0.059829 | -0.032454 | +0.092282 |
| soup | yes | 3 | L→S | 1.102170 | 0.092465 | -0.705169 | -0.051935 | +0.030203 | +0.082138 |
| soup | yes | 4 | S→L | 1.256867 | 0.115352 | +0.421725 | +0.057865 | -0.014324 | +0.072189 |
| soup | yes | 4 | L→S | 1.256867 | 0.115352 | -0.473786 | -0.042790 | +0.009807 | +0.052597 |
| soup | yes | 5 | S→L | 1.242374 | 0.211144 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| soup | yes | 5 | L→S | 1.242374 | 0.211144 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| cake | no | 0 | S→L | 0.500463 | 0.026774 | +0.999851 | +0.026387 | +0.001093 | +0.025294 |
| cake | no | 0 | L→S | 0.500463 | 0.026774 | -0.994474 | -0.015698 | -0.000373 | +0.015325 |
| cake | no | 1 | S→L | 0.726108 | 0.045082 | +0.927790 | +0.040307 | -0.011525 | +0.051832 |
| cake | no | 1 | L→S | 0.726108 | 0.045082 | -0.914751 | -0.021791 | +0.015265 | +0.037055 |
| cake | no | 2 | S→L | 1.004071 | 0.068008 | +0.842783 | +0.060538 | +0.001519 | +0.059019 |
| cake | no | 2 | L→S | 1.004071 | 0.068008 | -0.829593 | -0.041231 | +0.002127 | +0.043358 |
| cake | no | 3 | S→L | 1.168592 | 0.096206 | +0.546981 | +0.061270 | -0.001752 | +0.063022 |
| cake | no | 3 | L→S | 1.168592 | 0.096206 | -0.593916 | -0.048199 | +0.002439 | +0.050638 |
| cake | no | 4 | S→L | 1.407391 | 0.130889 | +0.151223 | +0.026016 | +0.016901 | +0.009115 |
| cake | no | 4 | L→S | 1.407391 | 0.130889 | -0.188664 | -0.015919 | -0.019634 | -0.003715 |
| cake | no | 5 | S→L | 1.183191 | 0.206380 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| cake | no | 5 | L→S | 1.183191 | 0.206380 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| pudding | no | 0 | S→L | 0.494587 | 0.026263 | +0.919984 | +0.025669 | +0.000761 | +0.024908 |
| pudding | no | 0 | L→S | 0.494587 | 0.026263 | -0.900133 | -0.012886 | +0.001393 | +0.014279 |
| pudding | no | 1 | S→L | 0.637826 | 0.038491 | +0.933989 | +0.038359 | -0.012434 | +0.050793 |
| pudding | no | 1 | L→S | 0.637826 | 0.038491 | -0.880719 | -0.014866 | +0.014562 | +0.029428 |
| pudding | no | 2 | S→L | 1.085526 | 0.071921 | +0.829155 | +0.064827 | +0.001849 | +0.062978 |
| pudding | no | 2 | L→S | 1.085526 | 0.071921 | -0.814837 | -0.043655 | +0.000408 | +0.044064 |
| pudding | no | 3 | S→L | 1.264225 | 0.101733 | +0.546983 | +0.068686 | -0.007391 | +0.076077 |
| pudding | no | 3 | L→S | 1.264225 | 0.101733 | -0.575908 | -0.049533 | +0.007841 | +0.057374 |
| pudding | no | 4 | S→L | 1.625217 | 0.149228 | +0.131699 | +0.032120 | +0.011893 | +0.020226 |
| pudding | no | 4 | L→S | 1.625217 | 0.149228 | -0.171730 | -0.010079 | -0.016479 | -0.006399 |
| pudding | no | 5 | S→L | 1.276137 | 0.223167 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| pudding | no | 5 | L→S | 1.276137 | 0.223167 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| cocoa | no | 0 | S→L | 0.502847 | 0.026969 | +0.914005 | +0.022760 | +0.001548 | +0.021212 |
| cocoa | no | 0 | L→S | 0.502847 | 0.026969 | -0.909595 | -0.015914 | +0.001196 | +0.017110 |
| cocoa | no | 1 | S→L | 0.743466 | 0.045975 | +0.873745 | +0.039473 | -0.007463 | +0.046937 |
| cocoa | no | 1 | L→S | 0.743466 | 0.045975 | -0.860722 | -0.021106 | +0.009006 | +0.030112 |
| cocoa | no | 2 | S→L | 1.163667 | 0.079498 | +0.795659 | +0.066463 | -0.004442 | +0.070906 |
| cocoa | no | 2 | L→S | 1.163667 | 0.079498 | -0.801352 | -0.046747 | +0.006518 | +0.053265 |
| cocoa | no | 3 | S→L | 1.400966 | 0.115992 | +0.525343 | +0.069889 | +0.007332 | +0.062556 |
| cocoa | no | 3 | L→S | 1.400966 | 0.115992 | -0.584097 | -0.060581 | -0.001233 | +0.059348 |
| cocoa | no | 4 | S→L | 1.765979 | 0.165891 | +0.143394 | +0.032047 | +0.017730 | +0.014318 |
| cocoa | no | 4 | L→S | 1.765979 | 0.165891 | -0.177768 | -0.020032 | -0.018862 | +0.001170 |
| cocoa | no | 5 | S→L | 1.444631 | 0.257589 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| cocoa | no | 5 | L→S | 1.444631 | 0.257589 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

The negative block-0–4 signed advantages are milk L→S block 0, cake L→S
block 4, and pudding L→S block 4; the latter two pairs were ineligible.
At block 0, the eligible sugar→lemon mean full-donor effect was about +0.777,
while the matched ingredient mean was about +0.027; the matched intervention
used a much smaller shared radius, so these are different doses.
The final encoder block has no later attention step from the intervened token
to the separate mask position, making its zero a useful architectural check.

## Evidence and limits

The [six journals and terminal records](run/) retain the 516 planned
forward starts and completions, 36 dose plans, model-entry IDs/masks, baseline
full logits, selected float32 vectors and target bits, and action metadata.
The [observer receipt](receipts/observer.json) binds their sizes and hashes.
The separately launched process reported exit 0 in the retained
[operator exit projection](receipts/operator-exit-projection.json); its
[post-exit observation](receipts/post-exit-observation.json) reports the later
named-process check, not an authenticated whole-process-tree cleanup or exact
exit time. The [no-model preflight](receipts/preflight-observer.json) checked
source/bytecode flags and direct-child exit, not model or tokenizer behavior.
The public start/exit projections omit the PID and bind SHA-256 digests of
the exact original operator records retained in private custody. The independent
custody auditor checked those originals; its strict original-record check cannot
be reproduced from the projections alone.

The [independent registry auditor](independent-registry-auditor.py) and
[report](independent-registry-report.json) replay saved evidence without a
model. A separate [custody and selected-bit auditor](independent-custody-auditor.py)
and its [private-record report](independent-custody-report.json) independently
check the saved process records, source bindings, selected logits, effects and
comparisons. The [public-package replay](independent-custody-public-replay.json)
uses the redacted projections and does not authenticate the private originals.
The [second reader replay](independent-reader-replay.json) uses the
same independently authored reader already invoked by the observer, so those
two runs are not separate numerical methods. Both independently check saved
record consistency; neither proves actual execution, publication chronology,
unmodified dependencies, or whole-process-tree cleanup. Full-output untouched
coordinates and patched full-vocabulary probability/top-five summaries are
owner observations; the retained selected-vector bits establish landing and
the retained target bits reconstruct the reported signed margins. The full
patched logits are not retained, so their association with the saved target
bits or full-vocabulary summaries cannot be independently reconstructed.
