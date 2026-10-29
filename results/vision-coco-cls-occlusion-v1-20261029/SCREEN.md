# Fresh response screen

The fixed 64-image inventory, exactly disjoint from 299 recorded earlier cases, was screened on the pinned CPU ViT. All 64 clean predictions were saved before the 36 eligible animal/background mask forwards. Four cats and six dogs met recognition, paired geometry and the prespecified response contrast; all four eligible cats and the first four eligible dogs were selected in fixed rank order. No token patch was run in this phase. These are response-conditioned cases, not a population estimate or an intervention result.

| Species | Candidates | Paired geometry | Recognized | Both | Response eligible | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cat | 32 | 17 | 8 | 4 | 4 | 4 |
| Dog | 32 | 16 | 24 | 14 | 6 | 4 |

The signed drop is `log P(correct group | clean) - log P(correct group | masked)`; B is animal-mask drop minus background-mask drop. A positive B can arise partly from a background-mask increase in correct-group probability, as the individual columns show.

| Image | Species | Animal drop | Background drop | B |
| --- | --- | ---: | ---: | ---: |
| 63970 | Cat | +0.008397 | -0.062451 | +0.070848 |
| 560923 | Cat | +0.134701 | -0.069857 | +0.204558 |
| 378832 | Cat | +0.362974 | -0.004697 | +0.367671 |
| 549410 | Cat | +0.014963 | -0.055251 | +0.070214 |
| 292135 | Dog | +0.504066 | -0.390149 | +0.894216 |
| 261664 | Dog | +1.920315 | -0.201210 | +2.121525 |
| 374832 | Dog | +2.504407 | +0.071262 | +2.433145 |
| 300322 | Dog | +0.055358 | -0.005350 | +0.060708 |

[The complete screen result](screen-result.json) retains all 64 recognition and geometry outcomes, all eligible mask results, and all selection reasons. The [archive index](screen-evidence-index.json) binds 23 parts and all 257 retained files, including full raw logits and exact processed pixel masks. Reassemble it without overwriting an existing file:

```sh
python -B restore-screen.py --index screen-evidence-index.json --output screen-evidence.zip
```

The restored ZIP is 19,268,696 bytes with SHA-256 `699eeb5f401bd78189263e0a4e77def091b440cc8843631bcf4bbd5ea967382e`. The independent [screen audit](screen-independent-audit.json) replays all 100 saved forwards, selection and the [direct observer](screen-observer.json); its separately published [reader](screen-replay-sources.zip) was reviewed on fabricated data before the screen. The forward-forbidden [preflight terminal](screen-preflight-terminal.json) and [observer](screen-preflight-observer.json) record zero submissions/saved forwards. The actual local source revision was `2b555f1ee1dacbb3dc2a880302f5fa4faea6631d`; all relevant source blobs were checked equal to public commit `c921d87d977796d390e73eb030be68da259a09ae`, followed by the reader-only commit. A separate GPU trace owner was active during the observed CPU-controller interval 14:03:49–14:05:22 UTC; this is context rather than a per-forward timing claim.

The normalized zero-fill is mid-gray in the processed input and may create artificial edges; equal area does not imply equal semantic content, and box-disjoint background can contain unlabeled objects. Geometry maps processed 16-pixel grid patches, not precise native-image causal sites. Saved-byte and direct-child checks are cooperative observations, not attestation of every executed instruction or full process cleanup. Exact exclusions do not establish near-duplicate or pretraining independence.
