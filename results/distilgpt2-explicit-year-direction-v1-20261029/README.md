# Explicit year wording and donorward preference shifts

The strict clean comparator gate failed on all eight pair-and-noun observations.
The separately predeclared generic-shift pattern passed on all eight: the
final-position year-directed displacement shifted the fixed-band score toward
the donor under **after**, **before**, and **near** wording, in both swap
directions, with the specified advantage over both matched-dose controls.
The comparator-like intervention decisions are **ineligible**, because their
clean gate failed. The generic pattern is a descriptive output effect; it
does not establish numerical comparison, a unique circuit, or the absence of
a comparator.

This is one public, non-blind CPU study of the pinned DistilGPT2 checkpoint,
using four related year pairs and two nouns. Its wording, thresholds and
question were informed by the earlier public
[year-transfer result](../distilgpt2-year-transfer-v1-20261029/). The fixed
[protocol](../../docs/DISTILGPT2_EXPLICIT_YEAR_DIRECTION_V1.md),
[cases](../../experiments/distilgpt2_explicit_year_direction_v1.json), and
[executor](../../scripts/explicit_year_direction_v1.py) were published before
the observed run. This is not evidence about outside or pretraining exposure.

## Clean behavior

For anchor year `y`, `W` is the log-sum-exp of three suffix logits just above
`y` minus the corresponding value for three just below it. Define `M` as
log-sum-exp of the middle-band logits minus log-sum-exp of the above-both-band
logits; `D` is `M(low anchor) - M(high anchor)`. Those two bands stay fixed
across both anchors and all wordings within a matched pair. The suffix IDs
differ between year pairs. `W` and `M` are next-token log-odds contrasts;
`D` is the difference between two `M` values. These describe next-token
scores, not generated complete-year outputs; each `W` entry is low/high.

| Suffix pair / noun | W after | W before | W near | D after | D before | D near |
|---|---:|---:|---:|---:|---:|---:|
| 07-16 war | +1.577 / +0.421 | +1.560 / +0.496 | +1.309 / +0.273 | +2.309 | +2.321 | +1.920 |
| 07-16 reign | +1.363 / +0.214 | +1.328 / +0.270 | +1.167 / +0.176 | +2.061 | +2.045 | +1.799 |
| 29-38 war | +0.426 / +1.668 | +0.445 / +1.691 | +0.307 / +1.484 | +2.173 | +2.251 | +1.989 |
| 29-38 reign | +0.426 / +1.572 | +0.464 / +1.641 | +0.348 / +1.454 | +2.091 | +2.252 | +2.045 |
| 51-60 war | +1.291 / +0.731 | +1.410 / +0.598 | +1.258 / +0.451 | +2.956 | +3.044 | +2.909 |
| 51-60 reign | +1.202 / +0.589 | +1.313 / +0.518 | +1.203 / +0.379 | +2.817 | +3.005 | +2.893 |
| 73-82 war | +1.703 / +1.874 | +1.767 / +1.913 | +1.404 / +1.771 | +3.453 | +3.512 | +3.162 |
| 73-82 reign | +1.646 / +1.915 | +1.690 / +1.984 | +1.351 / +1.832 | +3.279 | +3.338 | +3.051 |

The gate required at least six of eight observations, spanning three of four
year pairs, to satisfy all six strict checks: both `W_after > 0.5`, both
`W_before < -0.5`, `D_after > 0.5`, and `D_before < -0.5`. Individual
counts were 6/8 for each after-anchor `W`, 0/8 for each before-anchor `W`,
8/8 for `D_after`, and 0/8 for `D_before`; all-six count was **0/8**.
All 16 before-wording `W` values were actually positive, as were all eight
`D_before` values. The full-precision values and wording differences are in
[result.json](run/result.json).

The gate is deliberately strict and is neither necessary nor sufficient for
general numerical ability. For the low anchor under `before`, both fixed `D`
bands lie above the anchor; their relative score alone is not a necessary
condition for correct temporal comparison. Lexical, calendar and token
associations can also affect `W` and `D`.

## Intervention and matched controls

For each case, `E` is patched minus clean `M`; the donorward signs are
`s_low = -E_low` and `s_high = E_high`, and `S = (s_low+s_high)/2`. The main
action applies the signed opposite-year displacement at the final token of
the zero-based layer-3 block output. Its controls apply the same input L2
dose at position zero or a cyclic one-channel shift across all 768
displacement channels at the final position. A separate natural position-zero
equality check has zero dose.
The table lists final-site `S` for every pair/noun; `T` is after minus before.

| Suffix pair / noun | S after | S before | S near | T final | Generic criterion |
|---|---:|---:|---:|---:|---|
| 07-16 war | +0.550 | +0.558 | +0.518 | -0.008 | pass |
| 07-16 reign | +0.517 | +0.524 | +0.500 | -0.007 | pass |
| 29-38 war | +0.374 | +0.392 | +0.349 | -0.018 | pass |
| 29-38 reign | +0.374 | +0.403 | +0.374 | -0.029 | pass |
| 51-60 war | +0.684 | +0.685 | +0.697 | -0.002 | pass |
| 51-60 reign | +0.686 | +0.710 | +0.708 | -0.025 | pass |
| 73-82 war | +0.565 | +0.565 | +0.517 | +0.000 | pass |
| 73-82 reign | +0.559 | +0.554 | +0.503 | +0.005 | pass |

| Wording | Mean S final | Mean S position zero | Mean S rotated | Mean S self |
|---|---:|---:|---:|---:|
| after | +0.538599 | -0.008377 | +0.032201 | 0 |
| before | +0.548916 | -0.008629 | +0.036604 | 0 |
| near | +0.520833 | -0.009945 | +0.033251 | 0 |

The generic criterion required both donorward signs to be positive in each
wording and final-site `S` to exceed each control by more than 0.1 in each
wording; 8/8 met it, versus the fixed threshold of 6/8. Mean final-site
`T = S_after - S_before` was -0.010318. The two comparator-like direction
and control counts were each 0/8 descriptively, but their decisions are
ineligible, not tested-negative mechanism claims. Every per-direction `E`,
`s`, `S`, `T`, control and gate check is retained in the result file. The
related observations and fixed thresholds are not a statistical significance
test.

All 48 main signed effects were positive, ranging from +0.204389 to +1.006046.
Across the 24 pair/wording observations, the smallest final-minus-position
`S` difference was +0.353980 and the smallest final-minus-rotated difference
was +0.312084. The rotated control was not uniformly inert: its high-direction
signed effects were positive in all 24 observations, while its low-direction
signed effects were negative in 18/24. The final-action `T` values ranged
from -0.028629 to +0.004956. The criterion compares each signed control `S`
with the signed main `S`; it does not assume the rotated control is a null.
Self replacements had exact zero effect.

The common dose was half the smallest of the three wording-specific natural
displacement norms within each pair/noun. It ranged from 11.997 to 15.053,
or 0.442 to 0.500 of the corresponding natural norms. The largest relative
requested-versus-realized error across the 144 nonzero actions was
approximately `3.97e-7`. Equal input L2 does not imply equal downstream
effects; neither position nor rotated-direction control is guaranteed inert.
The pattern can reflect a broadly year-conditioned preference shift,
contextual associations or other effects of a replacement. It does not
identify what information was transferred.

Mean full-vocabulary probability mass on the context-aligned `01`-`99` next
tokens was 0.997893 after, 0.998101 before and 0.998049 near. The corresponding
separate `00` next-token masses were 0.000599, 0.000552 and 0.000644; all
other vocabulary mass was 0.001508, 0.001347 and 0.001307. The separate `00`
token need not equal the canonical full-string tokenization of `1700`.

## Evidence and scope

The direct CPU process exited zero, with no failure record and empty stderr.
It records 336 submitted, observed-return and completed actions, retaining
86,714,688 raw bytes across 48 files. The
[pre-run record](pre-run.json), [launch observations](launch-observation.json),
[updated observation](launch-observation-update.json),
[start observation](launch-observation-start.json),
[direct-process receipt](supervisor.json), [manifest](run/manifest.json),
[journal](run/attempt.jsonl), [result](run/result.json), stdout and stderr
are preserved. The receipt is an operator-observed direct-process exit, not
independent proof of execution or descendant-process custody. The cooperative
CPU/network guard is not a native security boundary.

The v21 inventory reserved all 48 inputs before the run. Live operator
coordination reported v22 merged after the first observed study action. V22
updated statuses and a prior-use annotation for previously reserved cases,
while preserving the public input/prefix set; its timing is not independently
timestamped execution proof. A separate status-only v23 was reported merged
after the process exited and also preserved that set. A voluntary trusted
comparison against the existing natural protected split reported no
complete-row-prefix collision; it does not establish absence of other hidden
or pretraining exposures.

Two independent, model-free saved-byte replays reconstructed all 336 logits,
scores, gates, action order, doses, planned/actual intervention bits and
untouched coordinates. The [source-and-receipt replay](audits/independent-saved-replay.py)
also checked the 38 pinned source paths, runtime, direct-process receipt and
three launch observations; its [report](audits/independent-saved-report.json)
has SHA-256 `691eb89c8eb51f9a273f6a2e7a913779ebcabef64e8dac2f23fd22f57ebd37bc`.
The separate [control audit](audits/independent-control-audit.py) and
[report](audits/independent-control-report.json) have report SHA-256
`b92f9c8cb5785dfc1e69a4ff13f14bdb102c99053799040d282f588479d83129`.
[Replay instructions](replay.md) list their required external pins and
checkout. These checks reconcile saved evidence; they do not independently
attest actual execution, complete process custody or all exposure history.
