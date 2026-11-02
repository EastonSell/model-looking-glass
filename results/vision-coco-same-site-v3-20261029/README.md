# Fresh same-site small-dose ViT study

The preregistered aggregate is **inconclusive**: two of eight pairs met the clean baseline gate,
neither met the two-direction composite, and six were baseline-ineligible. At least four evaluable
pairs were required. This is not evidence of equivalence or absence of a visual representation.

The single CPU run retained 195/195 forwards. The independent saved-byte replay reconciled
all matched landings, unchanged-token witnesses, controls, dose calculations and the decision;
a separate provenance/score audit verified the input bytes, source and direct-child receipts.
All 16 sides were dose-matchable. The 48 matched deliveries were 0.0989999882–0.0990000082
of receiver-block norm; their maximum relative deviation from q was 1.20e-7.

Each entry below is an annotation-oriented change in the dog-minus-cat group log-odds:
positive moves toward the opposite-species label. I is the opposite-species patch,
S the annotated same-species donor, and R the fixed random direction. Ineligible rows
are descriptive and are retained in full.

| Pair | Status and baseline reason | Side | I | S | R |
|---|---|---|---:|---:|---:|
| pair-00 | negative; baseline eligible | cat | +0.03895 | +0.05094 | -0.01870 |
| pair-00 | negative; baseline eligible | dog | -0.01148 | -0.02510 | +0.02988 |
| pair-01 | baseline-ineligible; dog-mass-below-0.2 | cat | +0.08918 | +0.04336 | -0.00579 |
| pair-01 | baseline-ineligible; dog-mass-below-0.2 | dog | -0.02191 | -0.02327 | -0.01219 |
| pair-02 | baseline-ineligible; dog-mass-below-0.2 | cat | -0.02796 | +0.08122 | -0.01142 |
| pair-02 | baseline-ineligible; dog-mass-below-0.2 | dog | -0.00124 | -0.00257 | +0.00025 |
| pair-03 | negative; baseline eligible | cat | +0.03396 | -0.01660 | +0.02058 |
| pair-03 | negative; baseline eligible | dog | -0.05390 | +0.11020 | +0.01888 |
| pair-04 | baseline-ineligible; cat-relative-margin, cat-mass-below-0.2 | cat | +0.12875 | +0.00960 | -0.25743 |
| pair-04 | baseline-ineligible; cat-relative-margin, cat-mass-below-0.2 | dog | +0.04987 | +0.03377 | +0.06995 |
| pair-05 | baseline-ineligible; dog-mass-below-0.2 | cat | +0.01186 | +0.03873 | -0.03665 |
| pair-05 | baseline-ineligible; dog-mass-below-0.2 | dog | +0.02210 | +0.04498 | +0.00347 |
| pair-06 | baseline-ineligible; cat-mass-below-0.2 | cat | +0.11023 | +0.08521 | -0.04562 |
| pair-06 | baseline-ineligible; cat-mass-below-0.2 | dog | -0.02860 | -0.05847 | -0.00417 |
| pair-07 | baseline-ineligible; cat-mass-below-0.2 | cat | -0.07752 | +0.13228 | +0.06660 |
| pair-07 | baseline-ineligible; cat-mass-below-0.2 | dog | +0.19576 | +0.69339 | -0.33771 |

All eight cat sides use the same annotated cat donor. The model classified it top-1 as
a schipperke, with cat-group probability 0.0306 and dog-group probability 0.5204.
The two dog donors were reused five and three times. The comparator is therefore
annotation-matched, and these pair outcomes are dependent.

The result is a same-site comparison, not a spatial-localization claim. The random
control is one fixed direction; a small activation norm does not establish an on-manifold
state or rule out structured disruption and smoothing. Prior attention mixes context,
selected blocks can contain other objects, and the dataset mirrors and hashes do not
prove official origin or independence from pretraining. The observed direct-child exit
does not attest the whole process tree or public timing.

The exact pre-forward source was `c260321cb955ae31490da639fabc23a290c3a1ce`;
the fixed manifest SHA-256 is `6fc5159a57cf07a91e93c8bc38f27d3f5925877083532ce92fec828c6d777bfb`.
The terminal SHA-256 is `1114e86ccc95cc5ae48daef40fcc3c1cfd7ffadd6bc102fa6c9350bd135d59b4`
and independent scientific report SHA-256 is
`e2d32a029311622285f661fa8ee8d47fb24ec0d2dc2d2bdabbe436cd3efd09ed`.
The full, byte-preserving evidence ZIP is split into ordered `evidence.zip.part-*` files;
concatenate them in numerical order and verify the archive and per-member SHA-256 values
in `evidence-index.json` before use. It includes the original supervisor receipt, including
its numeric process ID. The reviewed scientific replay source is preserved byte-for-byte in
`independent-scientific-replay.zip`; the Decimal check can be rerun with
`python -B independent-decimal-check.py --data <extracted-data-directory>`. Its published
copy differs from the reviewer’s original only in accepting this data path. The earlier failed
v2 selector and its receipt are retained separately.
