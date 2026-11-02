# Year versus label donor shifts: exposed-family continuation

The separately frozen three-family continuation completed its planned schedule,
but **0 of 3 families met the prespecified pattern**: none of the 12 targets
in any family passed all three strict control comparisons. All 36
year-donorward shifts were positive, from +0.012661 to +0.094070
in the fixed-band next-token contrast, but even the largest was below the
required +0.1 advantage over each control's absolute effect. This is a result
at the fixed matched input dose, not evidence that the effects are absent or
that year and label controls are equivalent.

The [v2 protocol](../../docs/DISTILGPT2_YEAR_LABEL_CONTROL_V2.md) retained the
three remaining fixed families from the [stopped v1 attempt](../distilgpt2-year-label-control-v1-20261029/).
All 36 prompts and their clean/no-op outputs were already exposed in v1. The
revised numerical tolerance and three-family denominator were chosen after
that stop; this is development evidence, not a held-out replication, and it
does **not** complete v1's four-family decision. The three families reuse a
template, years and labels and are not independent population samples.

Each target used the same pair-specific output bands for its matched prompts.
Let M be logsumexp of the three middle-band logits minus logsumexp of the three
above-both-band logits. A year-donorward effect is -ΔM for a low-anchor target
and +ΔM for a high-anchor target. The strict target rule required this effect
to exceed the absolute signed ΔM of **each** of the label-swap, position-zero,
and rotated-year controls by more than 0.1. A family required all 12 targets
to pass, and the v2 pattern required all three families. The controls are
active comparisons rather than assumed zero: without the +0.1 margin, the
year effect exceeded |label| in 34/36 targets, |position zero| in 36/36, and
|rotated year| in 34/36. These counts are descriptive and do not replace the
fixed rule.

| Original family index | Years and labels | Common dose q | Mean donorward | Targets passing |
| ---: | --- | ---: | ---: | ---: |
| 1 | 1709/1718, red/blue | 0.647854 | +0.022564 | 0/12 |
| 2 | 1754/1763, alpha/beta | 1.727107 | +0.065707 | 0/12 |
| 3 | 1754/1763, red/blue | 0.651152 | +0.026393 | 0/12 |

The dose was half the smallest of the family's 12 year/label donor-edge norms.
Across the three families it was 1.72%–5.86%
of the year-donor displacement norms, versus 42.53%–50.00%
of label-donor norms. Equal input L2 neither equalizes functional impact nor
matches the fraction of each donor swap. The small year fraction matters when
interpreting the failed fixed-margin prediction; another dose could behave
differently. No conclusion about numerical comparison, a particular circuit,
or generalization to other contexts follows.

The next table retains all 36 outcomes. Control columns are raw **signed** ΔM;
the rule compares each absolute value against the donorward year effect. All
36 targets fail each strict +0.1 inequality. Each self replacement had zero
effect, as required.

| Family | Wording | Label | Anchor | Year donorward | Label ΔM | Position-zero ΔM | Rotated-year ΔM |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | after | red | low | +0.013688 | +0.001984 | +0.000320 | +0.012388 |
| 1 | after | red | high | +0.029073 | -0.020531 | -0.000418 | -0.002948 |
| 1 | after | blue | low | +0.013458 | -0.001861 | +0.000328 | +0.011978 |
| 1 | after | blue | high | +0.029918 | +0.021270 | -0.000422 | -0.002947 |
| 1 | before | red | low | +0.013332 | +0.000948 | +0.000234 | +0.013710 |
| 1 | before | red | high | +0.034956 | -0.023183 | -0.000233 | -0.002173 |
| 1 | before | blue | low | +0.013026 | -0.000788 | +0.000224 | +0.013480 |
| 1 | before | blue | high | +0.035794 | +0.023822 | -0.000251 | -0.002159 |
| 1 | near | red | low | +0.012661 | +0.006101 | +0.000231 | +0.010095 |
| 1 | near | red | high | +0.030812 | -0.017653 | -0.000367 | -0.004531 |
| 1 | near | blue | low | +0.012680 | -0.006067 | +0.000206 | +0.009921 |
| 1 | near | blue | high | +0.031368 | +0.018441 | -0.000413 | -0.004651 |
| 2 | after | alpha | low | +0.082744 | -0.000235 | -0.000629 | +0.002165 |
| 2 | after | alpha | high | +0.040037 | +0.017012 | +0.000923 | +0.004143 |
| 2 | after | beta | low | +0.088262 | +0.002314 | -0.000804 | +0.002619 |
| 2 | after | beta | high | +0.043427 | -0.018659 | +0.001098 | +0.001719 |
| 2 | before | alpha | low | +0.088603 | -0.005749 | -0.000565 | -0.000335 |
| 2 | before | alpha | high | +0.038053 | +0.026990 | +0.000916 | +0.004900 |
| 2 | before | beta | low | +0.094070 | +0.008766 | -0.000712 | +0.000500 |
| 2 | before | beta | high | +0.040425 | -0.028830 | +0.001066 | +0.002498 |
| 2 | near | alpha | low | +0.083521 | -0.007023 | -0.000566 | +0.002822 |
| 2 | near | alpha | high | +0.049448 | +0.025806 | +0.000798 | +0.002531 |
| 2 | near | beta | low | +0.088696 | +0.009634 | -0.000685 | +0.004073 |
| 2 | near | beta | high | +0.051195 | -0.027667 | +0.000904 | -0.000477 |
| 3 | after | red | low | +0.034296 | +0.000921 | -0.000241 | +0.001786 |
| 3 | after | red | high | +0.017459 | -0.014727 | +0.000413 | +0.001370 |
| 3 | after | blue | low | +0.034274 | -0.001030 | -0.000195 | +0.001965 |
| 3 | after | blue | high | +0.017526 | +0.014627 | +0.000432 | +0.001498 |
| 3 | before | red | low | +0.036562 | +0.003392 | -0.000188 | +0.000848 |
| 3 | before | red | high | +0.016219 | -0.019072 | +0.000423 | +0.001849 |
| 3 | before | blue | low | +0.036725 | -0.003349 | -0.000174 | +0.001041 |
| 3 | before | blue | high | +0.016216 | +0.019111 | +0.000392 | +0.001859 |
| 3 | near | red | low | +0.033858 | +0.005026 | -0.000165 | +0.001667 |
| 3 | near | red | high | +0.019830 | -0.016664 | +0.000355 | +0.000374 |
| 3 | near | blue | low | +0.033889 | -0.005288 | -0.000118 | +0.001774 |
| 3 | near | blue | high | +0.019859 | +0.016771 | +0.000357 | +0.000412 |

## Evidence and numerical scope

Two separately written, model-free readers independently reconciled the saved
288 forwards, 77,802,624 raw bytes, all 36 targets and controls, the full
landing/untouched-coordinate witnesses, and all 144 non-self exact-rational
certificates. The v2 rule bounded both the delivered norm error and the full
displacement-vector error by 0.02% of q. The largest observed relative norm
error was 1.1327426132e-5; the largest vector error was 8.1446254107e-5, both
below 2e-4. The bound was informed by v1's exposed activations and its stop;
it certifies numerical fidelity at the input, not equal downstream effect.

Fresh v2 clean logits and no-op activations defined v2's plans. Their bytes
matched v1 for all 36 reused cases in a separate descriptive comparison; this
was not an eligibility rule. Every raw float32 stream, full-vocabulary logit
record, journal, manifest, result, process receipt, and both independent
reports/readers are retained here. See [replay.md](replay.md) for their pins
and scope. The original v1 direct receipt bound a private path-bearing stderr;
the v2 replay does not need or verify that private stderr.

The direct launcher reported an exit code of 0 with no interruption or failure,
an empty stderr, and a matching empty post-run cache prefix. That is a
cooperative direct-process observation, not independent authentication of
execution or of descendant cleanup. At the recorded launch observation the
latest reviewed public inventory was v27; it historically labeled these rows
planned even though v1 had observed them. V28 later reserved the separately
checked answer-format prompts. GitHub reports its merge at 10:31:37 UTC,
after the operator-observed v2 child exit at 10:29:58 UTC; these are reported
timestamps, not synchronized independent execution attestation. The old sealed
split was not queried for these inputs, and outside/pretraining exposure is
unknown. The public comparisons and scoped owner reports do not establish
complete exposure coverage.
