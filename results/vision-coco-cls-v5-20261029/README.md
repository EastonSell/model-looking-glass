# ViT CLS versus spatial comparison: negative on two screened pairs

Both selected pairs were evaluable and neither met the fixed composite:
**0/2 positive**, a negative result under the
[preregistered protocol](../../docs/VISION_COCO_CLS_COMPARE_V5.md). All four
sides failed the side rule. This does not mean that interventions had no
effect, or that CLS and spatial states are equivalent. For example, the
pair-01 cat's opposite-donor CLS change was −0.082342, and one matched
spatial intervention changed the top-1 class.

The 64-image [manifest](../../experiments/vision_coco_cls_compare_v5.json)
and selection rule were published before the clean CPU screen. Of 64 images,
31 passed recognition: 2/24 cat targets, 4/8 cat donors, 19/24 dog targets,
and 6/8 dog donors. The fixed selection yielded two pairs with four distinct
targets; one recognized cat donor and one recognized dog donor were each
reused in both pairs. All 64 scores and exclusions are retained in the
screen archive. The complete screen and its independent audit were published
at commit `1c2666cbd7f1ff343f5ff2d7b3ebebacde8fc8ad` before interventions.

The pairs are model-output-conditioned, and two paired observations sharing
both donors do not support population inference. They are fresh relative to
the documented local exposures; exact ID, JPEG and pixel disjointness does
not exclude near-duplicates, unrecorded exposure, or pretraining overlap.
The mirrored-image hashes establish consistency with those bytes, not
independently authenticated upstream origin. The inherited `heldout` field
denotes this conditional evaluation phase, not model-blind or
pretraining-disjoint selection.

At the output of encoder block index 2 (the third block), each receiver got
the same absolute dose at CLS and a four-token spatial site, for three
directions: opposite-species donor `I`, same-species donor `S`, and one fixed
random direction `R`. Values below are signed toward the opposite species.
`A = I - max(abs(S), abs(R))`. A side passed only if `I_CLS >= 0.05`,
`A_CLS >= 0.025`, and `A_CLS - A_spatial >= 0.025`; a pair required both
sides to pass. With two evaluable pairs, both had to pass for a positive
aggregate.

| Pair / receiver | CLS I / S / R | Spatial I / S / R | Raw I difference C−P | Adjusted C / P | Adjusted difference C−P |
| --- | --- | --- | ---: | --- | ---: |
| 00 cat | +0.032968 / +0.072347 / +0.024367 | +0.022810 / +0.036672 / +0.025436 | +0.010158 | −0.039379 / −0.013861 | −0.025517 |
| 00 dog | −0.005899 / +0.105519 / +0.144473 | −0.044281 / −0.093000 / +0.006562 | +0.038383 | −0.150372 / −0.137281 | −0.013091 |
| 01 cat | −0.082342 / +0.136108 / +0.103571 | +0.007718 / −0.001055 / −0.006768 | −0.090060 | −0.218450 / +0.000950 | −0.219400 |
| 01 dog | +0.003386 / +0.065835 / −0.058836 | −0.020587 / −0.056709 / +0.000619 | +0.023973 | −0.062449 / −0.077296 | +0.014847 |

All 24 matched landings passed the fixed actual-dose checks. Their delivered
norms were within `1.01e-7` relative of the same-side CLS-opposite delivery;
actual dose divided by receiver norm was 0.098999995–0.099000003 at CLS and
0.058963651–0.062134752 at the spatial site. Equal absolute L2 changes are not
equal per-coordinate changes: the sites have 192 and 768 coordinates. A
small L2 change does not establish an on-manifold activation.

Across the 24 matched conditions, full-vocabulary total variation ranged
from 0.001234 to 0.055749 and `KL(clean || patched)` from 0.0000083 to
0.008576. Whole-vocabulary TV includes changes in cat and dog classes; the
off-group conditional TV is reported separately in the complete
[descriptive summary](intervention-descriptive-summary.json). One matched
condition changed the top-1 class. Null and self checks at both sites, clean
repeats, full-state self and donor recovery, and the final-layer spatial
control all reconciled. Full-state donor swaps are descriptive and are not
subject to the matched 10% cap.

The adjusted site comparison does not imply that the raw opposite-donor
movement is larger at the winning site. One random vector per site is not a
random-control distribution. Both sites have already mixed global context
through attention; group-size effects, structured smoothing, other objects
and off-manifold perturbations remain alternatives. The result is conditional
on this model, layer, site, dose and selected sample; it cannot locate all
category information or establish that either site lacks it.

## Evidence and replay

The run used the pinned CPU checkpoint with 64 screen forwards and 66
intervention forwards. The forward-forbidden preflight, screen and
intervention each recorded a clean direct-child/controller exit. The raw
supervisor receipts retain numeric process IDs. These cooperative records
support saved-byte and direct-process consistency; they do not attest all
executed instructions, process-tree cleanup or publication timing.
The six original controller output/error logs are preserved in
`controller-logs.zip`; their bytes match the observer receipt hashes.

`screen-evidence.zip` contains the 64 screen records. The 24 indexed
`intervention-evidence.part-*` files reconstruct a 20,186,396-byte archive
with SHA-256
`7631d572053499c8dfa546126bbe8de07001426b776ff4709c9b3db97a7fedd1`;
its 136 members include every intervention forward and pair/terminal record.
`restore-intervention.py` verifies all part and member hashes before writing
the records. The [independent screen/cross-phase replay](vision_cls_v5_screen_independent.py)
checks selection and clean-logit identity; the [independent scientific replay](vision_cls_v5_intervention_independent.py)
checks the six matched landings, controls and decision. The Decimal replay
separately checks all saved logits and distribution summaries. These are
saved-data checks, not source, image, execution or training-history
attestations. Tiny floating-point summaries may differ across math libraries.

For example, from the repository root, with new empty paths for `SCREEN`,
`RESTORED`, and the output reports:

```sh
RESULT=results/vision-coco-cls-v5-20261029
python -m zipfile -e "$RESULT/screen-evidence.zip" "$SCREEN"
python "$RESULT/restore-intervention.py" --package "$RESULT" --output "$RESTORED"
python "$RESULT/vision_cls_v5_screen_independent.py" \
  --manifest experiments/vision_coco_cls_compare_v5.json --screen-data "$SCREEN" \
  --public-screen-result "$RESULT/screen-result.json" \
  --public-screen-archive "$RESULT/screen-evidence.zip" \
  --public-screen-custody "$RESULT/screen-custody.json" \
  --public-screen-observer "$RESULT/screen-observer.json" \
  --screen-supervisor "$RESULT/screen-supervisor.json" \
  --intervention-data "$RESTORED/data" --report "$CROSS_PHASE_REPORT"
python "$RESULT/vision_cls_v5_intervention_independent.py" \
  --manifest experiments/vision_coco_cls_compare_v5.json --screen-data "$SCREEN" \
  --public-screen-result "$RESULT/screen-result.json" \
  --public-screen-archive "$RESULT/screen-evidence.zip" \
  --public-screen-custody "$RESULT/screen-custody.json" \
  --public-screen-observer "$RESULT/screen-observer.json" \
  --screen-supervisor "$RESULT/screen-supervisor.json" \
  --intervention-data "$RESTORED/data" --report "$SCIENTIFIC_REPORT"
python "$RESULT/intervention-decimal-replay.py" --data "$RESTORED/data" --report "$DECIMAL_REPORT"
```
