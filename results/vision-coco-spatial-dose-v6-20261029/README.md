# Screened spatial ViT dose response: result

The fixed three-dose comparison was **negative: 0 of 4 evaluable pairs met
the preregistered composite**. All eight sides were dose-matchable, and no
side met the complete rule. This does not mean the interventions had no
effect: pair-00 dog moved +0.732727 toward the opposite species at the high
dose, while its high random control moved +0.981901. For pair-02 dog the high
opposite and same-species movements were -1.142481 and -1.136169. On every
side, the magnitude of a high-dose same-species or random control exceeded
the signed opposite-donor movement.

The [protocol](../../docs/VISION_COCO_SPATIAL_DOSE_V6.md), its
[runtime clarification](../../docs/VISION_COCO_SPATIAL_DOSE_V6_RUNTIME_NOTE.md),
and the [64-image manifest](../../experiments/vision_coco_spatial_dose_v6.json)
were public before screening. The annotation-only selector was public before
ranking. The fixed model-based screen recognized 31/64 images: 7/24 cat
targets, 1/8 cat donors, 17/24 dog targets and 6/8 dog donors. Four pairs
were selected using eight distinct targets and four distinct donors. Cat donor
398489 was reused in all four pairs; dog donors 176298, 121649 and 39913 were
used twice, once and once respectively. These observations are dependent.
The complete [screen result](screen-result.json),
[screen evidence](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/vision-coco-spatial-dose-v6-20261029/screen-evidence.zip), and
[screen audit](screen-independent-audit.json) were public before intervention.

The study patches one 2-by-2 spatial block after encoder block index 2 (the
third block). For receiver norm `n` and two natural donor distances it fixes
`q_high = min(0.396*n, d_opposite, d_same)`, `q_mid = q_high/2`, and
`q_low = q_high/4`. At each dose the opposite-species, same-species and one
fixed random direction receive the same absolute planned dose. The same
random signs are used at every level. On all eight sides `q_high/n` was 0.396;
none was dose-ineligible, unmatchable or limited-dose. The actual landed
fractions of receiver norm were 0.0989999911–0.0990000076 at low,
0.1979999925–0.1980000059 at mid and 0.3959999972–0.3960000063 at high;
the maximum relative discrepancy from the planned dose was 9.03e-8. All 72
landings passed the same-level matching and 40% receiver-norm cap checks.
Unscaled full-state donor swaps are descriptive controls and are not subject
to that cap.

Below, each `I/S/R` triple is the signed dog-versus-cat score movement toward
the opposite species for the opposite-donor, same-species and random arms.
`A = I - max(abs(S), abs(R))`; the adjusted triple is low/mid/high, and the
next two columns show the raw and adjusted
high-minus-low differences. The low and high criteria and pair/aggregate
denominators are fixed in the protocol. The middle dose is descriptive.

| Pair | Receiver | Low I/S/R | Mid I/S/R | High I/S/R | A low/mid/high | Raw high-low | Adjusted high-low | Side passes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| pair-00 | cat | +0.000952/+0.019954/+0.011386 | -0.017616/+0.028959/+0.011388 | -0.101265/+0.006389/-0.016404 | -0.019002/-0.046575/-0.117669 | -0.102217 | -0.098667 | no |
| pair-00 | dog | +0.180405/+0.056086/+0.228848 | +0.405569/+0.117392/+0.475616 | +0.732727/+0.290368/+0.981901 | -0.048443/-0.070047/-0.249174 | +0.552322 | -0.200731 | no |
| pair-01 | cat | -0.004316/-0.059833/+0.047689 | +0.009223/-0.003502/+0.091514 | +0.120791/+0.427567/+0.170469 | -0.064149/-0.082290/-0.306776 | +0.125107 | -0.242626 | no |
| pair-01 | dog | -0.061726/-0.057053/+0.075638 | -0.105075/-0.091175/+0.146106 | -0.054195/-0.120655/+0.227327 | -0.137364/-0.251181/-0.281522 | +0.007531 | -0.144158 | no |
| pair-02 | cat | +0.020229/-0.006888/-0.018073 | +0.045453/-0.016889/-0.033955 | -0.072717/-0.066654/-0.059630 | +0.002156/+0.011498/-0.139371 | -0.092946 | -0.141527 | no |
| pair-02 | dog | +0.078635/-0.222652/-0.060922 | -0.116088/-0.544119/-0.141572 | -1.142481/-1.136169/-0.399417 | -0.144017/-0.660207/-2.278649 | -1.221116 | -2.134632 | no |
| pair-03 | cat | +0.087461/+0.159490/-0.109044 | +0.217986/+0.398844/-0.196403 | +0.311616/+0.961126/-0.296123 | -0.072030/-0.180859/-0.649510 | +0.224155 | -0.577480 | no |
| pair-03 | dog | -0.012983/+0.062591/+0.054682 | +0.024181/+0.125930/+0.114380 | +0.152741/+0.435265/+0.241772 | -0.075575/-0.101750/-0.282524 | +0.165725 | -0.206949 | no |

Across the 72 matched conditions, whole-vocabulary TV from the clean output
ranged from 0.006642 to 0.511645, `KL(clean || patched)` from 0.000112 to
0.847997, and off-group conditional TV from 0.006505 to 0.232112. Nine
conditions changed top-1. Whole-vocabulary TV includes target-group changes.
All 1,000 logits, group masses, per-token doses and all control observations
are preserved in the archive and [descriptive extraction](intervention-descriptive.json).
The saved-byte replay checked the repeat, null, self, full-state self,
full-state donor-recovery, final-layer non-CLS and post-clean controls, as
well as exact selected and unselected token landing. The final-layer non-CLS
control is structurally unable to change the CLS-only classifier output;
it is an implementation check, not evidence of spatial localization.

The selected sample is conditioned on clean model recognition, geometry and
the high-dose threshold. The earlier v5 negative finding informed these
exploratory thresholds; this is not an independent replication or a
significance test. The low dose here (nominal 0.099 of receiver norm) does
not repeat v5's approximately 0.059–0.062 spatial fraction. The tokens already
mix global context through attention, class groups differ in size, and one
random vector per side is not a random-effect distribution. Generic smoothing
or other off-group changes can remain even with these controls. A bounded L2
change need not stay on-manifold. This result is conditional on this model,
site, doses and screened, donor-reusing sample; it does not establish absence
of information, class specificity, a population effect or localization.
Exact ID, JPEG and processed-pixel disjointness from 171 documented earlier
images does not establish absence of visual near-duplicates, unknown prior
exposures or pretraining overlap. Pinned mirror hashes establish byte
consistency, not authenticated upstream origin.

## Evidence and replay

The [intervention index](intervention-evidence-index.json) lists 51 parts and
286 files. Their concatenated ZIP is 43,038,169 bytes, SHA-256
`c796a6af6e0b01d7b491be729cab3072a485a4d81486965707bf43e15b24e216`.
It contains all 140 returned forward records and tensors, four pair records,
start and terminal. The [independent scientific replay](vision_spatial_v6_intervention_independent.py)
and its exact [screen-reader companion](vision_spatial_v6_screen_independent.py)
reconstructed nine requested/landed patches per side, untouched tokens,
controls, dose matching, distribution scores and decisions from saved bytes.
Its [report](intervention-independent-audit.json) has SHA-256
`bd8e4ef33ea58d80e419f28fac9138511b1ef80188323cae8631082b7d9540f0`.
The independently authored [Decimal extraction](vision_spatial_v6_descriptive_summary.py)
agreed on all 72 movements (maximum difference 1.78e-15) and gives every
continuous condition and diagnostic. Its [report](intervention-descriptive.json)
has SHA-256 `e3fad341b1da434a5870c8e79b52b17552ce9d16d0a58372b3c7390c55fbbab2`.
The separate [cross-phase audit](cross-phase-independent-audit.json) checks
that the 12 distinct selected images' clean logits match their screen bytes
before any patch. These replays establish saved-byte consistency under
quiescent inputs, not source execution or independent sampling. Top-one IDs
are checked; optional class-label strings are not.

The retained source snapshots, checkpoint-only preflight, child supervisor
and direct-controller observer receipts and [logs](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/vision-coco-spatial-dose-v6-20261029/controller-logs.zip)
record the CPU run and clean exits. Their scope is cooperative direct-process
observation, not proof of every executed instruction or descendant cleanup.
The original supervisor records retain numeric process IDs. As clarified before
screening, a nonfinite returned tensor is retained as an incomplete attempt,
not promoted as a dose-ineligible observation.

From the repository root, using an empty scratch directory outside the
repository, restore the two evidence sets and replay without a model:

```sh
R=results/vision-coco-spatial-dose-v6-20261029
OUT=/path/to/new/scratch
mkdir -p "$OUT/screen"
python -m zipfile -e "$R/screen-evidence.zip" "$OUT/screen"
python -B "$R/restore-intervention.py" --package "$R" --output "$OUT/intervention"
python -B "$R/vision_spatial_v6_intervention_independent.py" \
  --manifest experiments/vision_coco_spatial_dose_v6.json \
  --screen-data "$OUT/screen" --public-screen-result "$R/screen-result.json" \
  --public-screen-archive "$R/screen-evidence.zip" --public-screen-custody "$R/screen-custody.json" \
  --public-screen-observer "$R/screen-observer.json" --screen-supervisor "$R/screen-supervisor.json" \
  --intervention-data "$OUT/intervention/data" --report "$OUT/scientific-audit.json"
python -B "$R/vision_spatial_v6_descriptive_summary.py" \
  --data "$OUT/intervention/data" --output "$OUT/descriptive.json" \
  --terminal-sha256 f05117983f624ea7ec5305499d027e8b2448d65a230913d0102a23645f7016af
```

The output files must not already exist. Minor platform math-library differences
can change floating report bytes; assess the checks and decision rather than
assuming cross-platform report hashes must be identical.
