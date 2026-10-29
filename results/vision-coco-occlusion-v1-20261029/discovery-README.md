# Occlusion and token transfer: discovery evidence

The fixed discovery phase used four previously exposed v6 screen images. All
four were dose-matchable and none met the joint rule; the saved aggregate is
**0/4, negative**. This is development evidence, not a fresh evaluation. The
unchanged [prospective protocol](../../docs/VISION_COCO_OCCLUSION_V1.md)
requires this evidence to be public before any fresh evaluation image is
forwarded. It fixes the image selection and thresholds independently of these
outcomes.

Each value below is `log P(annotated group | clean) - log P(group | changed)`;
positive values mean the changed input or state reduced the group probability.
`B` is the animal-mask drop minus the background-mask drop, and `T` is the
animal-transfer drop minus the background-transfer drop. The individual random
drops are retained because their difference can hide large changes.

| Image | Species | Animal mask | Background mask | Animal transfer | Background transfer | Animal random | Background random | B | T |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 328437 | Cat | -0.099957 | 0.319788 | 0.079590 | 0.031700 | 0.017161 | 0.015394 | -0.419745 | 0.047890 |
| 142719 | Cat | -0.002057 | -0.002689 | 0.000503 | -0.000348 | 0.000226 | 0.000067 | 0.000633 | 0.000851 |
| 476736 | Dog | 0.095441 | -0.004144 | -0.003200 | -0.000339 | -0.000923 | 0.001288 | 0.099585 | -0.002861 |
| 503893 | Dog | -0.005686 | -0.020656 | 0.014581 | -0.000902 | 0.011580 | 0.000222 | 0.014970 | 0.015483 |

The joint rule requires `B >= 0.05`, `T >= 0.025`, and `T` to exceed the
larger absolute random drop by at least 0.025. These failures do not mean that
occlusion or token transfers had no effects: for example, the background mask
of 328437 reduced its group probability more than the animal mask. Mid-gray
occlusion creates an artificial image, and the block-2 token state already
mixes context. The result does not establish mediation or localization.

The direct observer reports a clean, reaped exit and 72 saved forwards; this
is a cooperative process observation, not an attestation of every instruction
or descendant. The complete archive contains all clean and masked inputs,
logits, capture/landing witnesses and controls. Its 29 parts are listed in
[the index](discovery-evidence-index.json); their concatenation is 24,449,724
bytes with SHA-256 `1edc2ca93b47ff022d1ca6a7f0b2ecfe692b5fc7f949b66ce00e438a9c024499`.
The [freeze](discovery-freeze.json) binds the terminal, index and
[direct observer](discovery-observer.json).

To restore the ZIP without overwriting an existing output:

```sh
python restore-discovery.py --index discovery-evidence-index.json --output discovery-evidence.zip
```

The restored archive contains 162 indexed files. Reproduction and independent
saved-byte checks assess record consistency and arithmetic; they do not
authenticate the upstream image source, establish pretraining independence or
prove execution chronology.
