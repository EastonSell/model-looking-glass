# Public year-conditioned preference transfer

The fixed study did **not** support its prespecified forward-specific
intervention pattern: one of eight pair/noun observations met the criterion,
versus the required six. The separate clean comparator-pattern gate also failed
in all eight observations. The exploratory preference gate passed in six of
eight, so the protocol called for interventions on **all** 48 cases.

Partway moving the final-position layer-3 block output in the direction of an
opposite-year donor changed the fixed-band preference in the donor direction
in both swap directions for all eight observations under each wording. The
average donorward effect was +0.443 for forward, +0.423 for neutral and +0.361
for reverse wording. The forward-minus-neutral mean was +0.020; it exceeded
the prespecified +0.1 threshold for only one observation. The perturbation
therefore produced a year-conditioned preference shift in this setting, but
the requested forward-specific pattern was not established.

These are non-blind, public examples from one checkpoint, four year pairs,
two event nouns and three related templates. The band design and thresholds
were chosen after a [development screen](../distilgpt2-greater-than-screen-v1-20261028/README.md),
then the [protocol](../../docs/DISTILGPT2_YEAR_TRANSFER_V1.md) was published
before the fixed [cases](../../experiments/distilgpt2_year_transfer_v1.json)
and before this run. The operator reports assembling the cases after protocol
publication; Git history alone does not independently establish assembly time.
The interventions and controls do not
establish numerical comparison, identify a circuit, or rule out contextual
token associations and other lexical explanations. Equal input L2 dose does
not imply equal downstream effects.

## Clean fixed-band behavior

For each matched pair, `M` is the log-sum-exp of the three fixed middle-band
logits minus that of the three fixed above-both logits. `D = M(low year) -
M(high year)`. Within each matched pair the same output token bands are used
for both years and all three wordings; the bands differ between year pairs.
A positive reverse `D` fails the specified symmetric
comparator-pattern direction; it does not rule out a weaker or differently
expressed numerical capability.

| Start years | Event | Forward D | Reverse D | Neutral D | Forward − neutral |
| --- | --- | ---: | ---: | ---: | ---: |
| 1705 / 1714 | war | 3.466 | 2.162 | 2.445 | 1.021 |
| 1705 / 1714 | reign | 3.554 | 2.002 | 2.442 | 1.113 |
| 1727 / 1736 | war | 2.994 | 2.830 | 2.483 | 0.512 |
| 1727 / 1736 | reign | 2.942 | 2.556 | 2.374 | 0.568 |
| 1749 / 1758 | war | 3.158 | 2.960 | 2.537 | 0.621 |
| 1749 / 1758 | reign | 3.268 | 2.800 | 2.585 | 0.683 |
| 1771 / 1780 | war | 3.075 | 3.641 | 3.040 | 0.035 |
| 1771 / 1780 | reign | 3.065 | 3.335 | 3.013 | 0.052 |

The comparator gate required `D_forward > 0.5` and `D_reverse < -0.5`
in at least six observations spanning three year pairs; it passed 0/8. The
separate exploratory gate required `D_forward > 0.5`, `D_neutral > 0.5`
and `D_forward - D_neutral > 0.1` under the same count rule; it passed 6/8.
All eight reverse values were positive. The comparator intervention criterion
was ineligible and is not reported as a tested comparator conclusion.

## Intervention and controls

`S` averages the signed low-to-high and high-to-low shifts in `M`, with
positive sign defined toward the donor in the forward-style direction. The
main action uses the year-directed displacement at the final position. The
position control applies the same displacement at position zero; the
direction control cyclically rotates its channels at the final position.
Neither is assumed to be an inert perturbation. The common family dose is
half the smallest opposite-year displacement norm across the three wordings;
actual doses, fractions, both directional effects and every action are in
[result.json](run/result.json).

| Start years | Event | Main S forward | Main S neutral | Main S reverse | Main F − N | Position control F | Rotated control F | Pattern met |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1705 / 1714 | war | 0.527 | 0.430 | 0.277 | 0.097 | 0.002 | -0.031 | no |
| 1705 / 1714 | reign | 0.566 | 0.447 | 0.293 | 0.118 | 0.001 | -0.046 | yes |
| 1727 / 1736 | war | 0.450 | 0.438 | 0.340 | 0.012 | 0.002 | 0.061 | no |
| 1727 / 1736 | reign | 0.447 | 0.422 | 0.350 | 0.025 | 0.000 | 0.061 | no |
| 1749 / 1758 | war | 0.280 | 0.301 | 0.264 | -0.020 | 0.002 | 0.008 | no |
| 1749 / 1758 | reign | 0.324 | 0.321 | 0.270 | 0.002 | 0.004 | -0.001 | no |
| 1771 / 1780 | war | 0.474 | 0.498 | 0.550 | -0.024 | -0.004 | 0.106 | no |
| 1771 / 1780 | reign | 0.479 | 0.530 | 0.542 | -0.051 | -0.002 | 0.093 | no |

For all eight observations, the forward main `S` exceeded both forward
matched-dose controls. Both main-action directions were positive in all eight
observations for forward, neutral and reverse wordings. Three individual
forward-minus-neutral main effects were negative. The control means were
+0.0007 at position zero and +0.0313 for rotated channels in the forward
wording; controls can themselves affect the output. The specified joint
pattern additionally required positive directions in forward and neutral,
forward `S - S_neutral > 0.1`, and main forward `S` above both controls.
Only 1/8 met all conditions, below the six required. The eight observations
are related, not eight independent draws from a population.

## Evidence and replay

The one-shot direct CPU process exited 0 with no failure record and empty
stderr. It completed 336 forwards and retained 87,206,208 raw bytes: full
final-position float32 logits for every call and block before/after witnesses
for the hooked calls. The [journal](run/attempt.jsonl),
[manifest](run/manifest.json), [result](run/result.json), per-case `.f32`
files, [pre-run record](pre-run.json) and [process observation](supervisor.json)
are preserved. The process observation covers the directly observed process,
not an independent execution or descendant-process attestation.

The independent saved-byte replay checked the receipt and source pins, all
336 logits and score reductions, action and event order, gates, family doses,
planned and actual intervention bits, unchanged coordinates and all reported
decisions. It did not rerun the model. Its script is
[year_transfer_saved_replay.py](year_transfer_saved_replay.py) and its output
is [independent-audit.json](independent-audit.json). The exact source revision
is `d4e17d4a9db796dc48ab76dbe08500cdeff15007`; the pre-run and receipt SHA-256
values are `18d75ff2f833ccef272fa41abb303c7c9b18b860127abdd79c6bd2cbe6d252be`
and `1bd457a772a4985bf223624ea5248e8343771bde97495b240d9717748d997a42`.
The run binds the v16 reservation; the independently checked v17 status-only
successor and a separate retrospective exposure report are described in the
pre-run record. Outside and pretraining exposure remain unknown.

A second independent saved-byte audit also checked the stream, action order,
gates, effects, doses and intervention witnesses. Its portable
[script](year_transfer_second_audit.py) writes its report to standard output;
the original [report](independent-second-audit.json) is retained unchanged.
Neither audit is an independent attestation that the model was executed.

To replay, run the saved script with `--run` pointing at this package's `run`
directory, `--cases` at the frozen case JSON and `--source-root` at a clean
checkout of the exact source revision. Supply `--source-revision` as above,
`--reservation-sha256 275a16d49a386b1de247bff0f9031e92105d1b76f2493352090bf27c37d58843`,
`--current-exposure-sha256 e5734470ccd3b4e2655c91608caf8ffa158fa8650ff0bdd7f5aeaa1028046f3a`
and `--receipt-sha256` as above. This is a saved-data check, not a fresh
model run.
