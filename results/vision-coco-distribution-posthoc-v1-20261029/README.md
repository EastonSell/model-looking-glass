# Post-hoc distribution check for the ViT COCO interventions

The saved results contain a concrete measurement warning: for the cat receiver
in public pair 04, the inside and random-direction patches produced probability
displacements with cosine **0.938**, but their annotation-oriented dog/cat
log-odds changes had opposite signs: **+0.254** and **−0.513**, respectively.
The full-class total variations were **0.251** and **0.087**, and the shares
of absolute probability movement outside the cat/dog groups were **0.742** and
**0.814**. Thus similar global displacement direction alone does not identify
the direction of the particular group score. The clean probabilities were
0.288 cat, 0.0145 dog and 0.698 off-group; the clean top class was ID 453.
All four probability vectors for this side and the other 15 sides are saved in
the report, so the example can be inspected without relying on the cosine alone.
For a future prospective check, report group-specific contrasts alongside
whole-distribution movement, use multiple random directions and test smaller
doses and position-matched controls. These are design suggestions, not results
of this replay.

This analysis was designed **after** the original results. It does not change
the original prespecified outcome: zero of four evaluable public held-out
pairs met the two-direction criterion; the other four pairs failed the baseline
gate. Across the eight receiver sides of the four evaluable pairs, the
descriptive medians and counts are:

| Matched condition | Median full-class total variation | Median off-group share of absolute movement | Median annotation-oriented log-odds change |
|---|---:|---:|---:|
| Inside | 0.139 | 0.552 | +0.387 |
| Outside | 0.053 | 0.540 | −0.010 |
| Random | 0.109 | 0.458 | +0.176 |

Inside total variation exceeded outside on 7/8 sides and random on 6/8;
random's oriented score change was at least inside's on 3/8. These are paired
descriptions of only four image pairs, not eight independent seeds or a
population estimate. The report keeps the eight baseline-ineligible sides
separate. The single random direction per side is not a distribution of random
controls. Off-group movement cannot by itself establish a cause of the group
score change; if only off-group logits change, the dog/cat log odds are
unchanged.

The patches were large relative to receiver norms. Tokens at the patched layer
were contextualized, inside and outside positions differed, and the 118 dog and
five cat classes have unequal group sizes; nonspecific smoothing can move their
score. Same-species controls were available for only four of 16 sides and are
not used as paired controls here. The public held-out images are not known to be
independent of pretraining. These results do not localize animal information,
establish a donor-specific mechanism, or generalize to other models or images.
The [original study](../vision-coco-vit-v1-20261029/README.md) gives the full
control and selection caveats.

## Reproduction and evidence

The [analysis and equations](../../docs/VISION_COCO_DISTRIBUTION_POSTHOC_V1.md)
describe the model-free reader and reproduction command. It reads the original
public archive with pinned SHA-256
`519d30de52b85194f97559a7164031b83f6465e39fb659b399965efedb3373fd`, checks
the index, member and binary hashes, and verifies the saved baseline scores,
gate and recorded matched doses. It did not modify the original evidence or
run a model. These are saved-byte consistency checks, not independent
attestation of execution.

[result.json.gz](result.json.gz) retains the machine-readable result with all
64 full probability vectors. The gzip file SHA-256 is
`e01a52a49b791bb65d97f1a92a93b5b7d4e0608f9d3474477cad5d58578e6c76`; the
uncompressed JSON SHA-256 is
`1da867dddba96c54795120541577f62b1009103f467f544059d2130189450413`. The
executed analysis source SHA-256 is
`c9c2be21ef9e027f0cee8be553c195d6f7f9b08b96f27d9dd234c7272c92df9c`.
The saved run used Python 3.11.15 with CUDA hidden; it records a Python-traced
heap peak of 11,858,578 bytes through report construction. That figure omits
output serialization, native and decoder allocations, and the interpreter
baseline, and is not a process RSS bound. The decoder and logical input/output
caps are documented separately. Runtime metadata and last-bit numerical values
can differ on a replay.
