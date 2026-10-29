# Year versus label control: stopped public attempt

The fixed four-family study stopped during planning of the second family. It
therefore has **no four-family decision**. The first family's intervention
records are complete and descriptive: 0 of 12 targets met all three strict
control margins, although all 12 year-donorward scores were positive. The
other three families have no intervention forwards. This does not establish
an absence of effects or an equivalence between year and label controls.

The [frozen protocol](../../docs/DISTILGPT2_YEAR_LABEL_CONTROL_V1.md) and
[cases](../../experiments/distilgpt2_year_label_control_v1.json) specified four
related, crossed families on a pinned DistilGPT-2 checkpoint. All 48 prompts
had clean and no-op forwards before the first intervention, so **all 48 inputs
and their baseline outputs were exposed**. The first family then completed
its five actions and a clean-after check for each target. Two independent
saved-byte replays reconcile 168 completed forwards and 43,725,984 raw bytes.
The planned full run was 384 forwards; no model call was submitted for the
failed second-family plan. The direct child process was observed to exit 1.

The strict fixed tolerance required the error in the realized float32
displacement norm, divided by the requested common dose, to be at most 1e-5.
From the retained no-op tensors,
the independent diagnostic reconstructs the first failing plan as the
position-zero year control for `near`, `blue`, low anchor in the second family
(zero-based index 1). The
family's minimum of 12 donor-edge norms was 1.2957076764311517 and the common
requested norm q was 0.6478538382155759. The planned float32-rounded vector
would have norm 0.6478465051609128: a shortfall of 0.00000733305466305012,
or 0.000011318995474732401 relative to q. This exceeded the frozen 0.00001
limit and correctly stopped the attempt before that vector was delivered.

The numerical reconstruction indicates a plausible cause: one position-zero
activation channel was +1750.384521484375, with float32 spacing 0.0001220703125.
Its intended change, -0.11686241712692279, became -0.1168212890625 after
rounding. The reconstructed rounding-error projection on the intended
direction was -0.0000073343498, close to the norm shortfall. This is an
inference from retained bits and arithmetic; the plan was not forwarded.

The first family crosses 1709/1718 with alpha/beta. The table reports its
signed year-donorward effect and signed
changes in the fixed-band next-token log-odds contrast M for each control.
The predeclared target rule required the year-donorward effect to be greater
than each control's absolute effect by more than 0.1; it is a per-target
comparison, not a comparison of signed means. All 12 failed the label and
rotated-control inequalities, and one passed the position-zero inequality.
The control effects were not assumed to be zero. Requested common dose for
this family was 1.8560272538338054; the fractions of the unscaled donor-edge
norms ranged from about 0.0483 to 0.5. Equal input L2 does not imply equal
functional strength.

| Wording | Label | Anchor | Year donorward | Label-control ΔM | Position zero ΔM | Rotated year ΔM |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| after | alpha | low | +0.046389 | -0.049890 | +0.000788 | +0.027745 |
| after | alpha | high | +0.069369 | -0.020979 | -0.000841 | -0.008664 |
| after | beta | low | +0.046559 | +0.052176 | +0.000793 | +0.031787 |
| after | beta | high | +0.079646 | +0.016854 | -0.000993 | -0.011257 |
| before | alpha | low | +0.045593 | -0.053657 | +0.000571 | +0.034059 |
| before | alpha | high | +0.086895 | -0.045397 | -0.000564 | -0.005129 |
| before | beta | low | +0.044528 | +0.056322 | +0.000514 | +0.037278 |
| before | beta | high | +0.101975 | +0.043782 | -0.000638 | -0.007814 |
| near | alpha | low | +0.039911 | -0.044078 | +0.000589 | +0.022673 |
| near | alpha | high | +0.080450 | -0.041559 | -0.000708 | -0.010715 |
| near | beta | low | +0.040369 | +0.044887 | +0.000543 | +0.024789 |
| near | beta | high | +0.091658 | +0.038222 | -0.000840 | -0.012122 |

The journal retains every recorded action and full-vocabulary float32 logits,
including the clean/no-op results for all four families; the audit reports
retain every target's individual control inequalities. The observed first
family and all baseline outputs are development evidence, not a held-out
evaluation. A separately frozen follow-up may use the remaining public
families but cannot turn this incomplete four-family attempt into a completed
preregistered result.

## Evidence and scope

- `run/manifest.json`, `run/attempt.jsonl`, `run/failure.json`, and all 48
  case `.f32` streams are the unchanged retained attempt. No final
  `result.json` was produced. The failure record does not retain final model
  state or checkpoint metadata; the manifest, runtime-ready event, and
  reviewed source/pre-run records have their separately stated scope.
- `supervisor.json` is the unchanged direct-process observation. Its hashes
  bind the privately retained original `stderr.txt`; `stderr-redacted.txt` is
  a separate path-free derivative with its own hash and redaction record.
  The original is not replaced or represented as public. Verification of
  the redaction and original stderr hash requires the private original.
- `pre-run.json` and both launch observations are the exact records that
  preceded the launch. Source commit `72ff52feca7c102e611023b54f031fbb8ff17bd4`
  and its tree, public inventory v25, runtime and source differences are
  described there. The old sealed natural split was not queried. Known-public
  exclusions and cooperative process observations do not attest every
  unrecorded exposure or independently authenticate execution.
- The independent reports and model-free replay scripts reconstruct saved
  evidence rather than perform a second model run. Their public projection
  modes and any required external source/receipt pins are explained in
  `replay.md`.
