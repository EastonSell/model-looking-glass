# DistilGPT-2 natural agreement: public discovery

The CPU discovery completed 608 prescribed forwards on 16 intentionally exposed prompts in four families. The independent saved-tensor audit passed the donor landing, untouched-output, null, self, repeat, delivery, subject-position and final-clean checks. Both present and past clean baselines were 16/16 correct.

The selected last-token site was **layer 3 attention**, with target-oriented number contrast D = 3.095347. The fixed earliest, random and greatest-L2 baselines were layer 1 attention, layer 3 MLP and layer 4 MLP respectively. The fixed random site is one prespecified draw, not a randomization distribution; the greatest-L2 site is an association baseline.

For each donor cell, the target-oriented shift is the clean correct-number margin minus the patched margin, with the target's number defining the sign. D averages the opposite-number-minus-same-number cell contrasts for the same-lemma and different-lemma comparisons, then averages the four families equally. The same-lemma/same-number cell is a self patch.

| Last-token site | Present D | Past D | Present opposite-number strict reversals |
| --- | ---: | ---: | ---: |
| 1:attn:last | 0.075660 | 0.064484 | 0/32 |
| 1:mlp:last | -0.016216 | -0.009237 | 0/32 |
| 2:attn:last | 0.095207 | 0.088573 | 0/32 |
| 2:mlp:last | -0.041031 | -0.037637 | 0/32 |
| 3:attn:last | 3.095347 | 2.684149 | 10/32 |
| 3:mlp:last | -0.156761 | -0.175753 | 0/32 |
| 4:attn:last | 1.693050 | 1.517922 | 0/32 |
| 4:mlp:last | 1.823285 | 1.437677 | 0/32 |

For the selected site, the present contrast by public family and the two lemma comparisons were:

| Family | Same lemma: opposite minus same | Different lemma: opposite minus same | D |
| --- | ---: | ---: | ---: |
| q1 | 2.890914 | 2.885392 | 2.888153 |
| q2 | 4.119334 | 4.123967 | 4.121651 |
| q3 | 2.081141 | 2.072004 | 2.076573 |
| q4 | 3.256648 | 3.333376 | 3.295012 |

The selected site produced ten strict sign reversals across 32 opposite-number donor pairs, involving six distinct target prompts; these overlapping pairs are not independent samples. The other 22 opposite-number pairs retained the correct sign but had smaller margins. A positive D can also reflect weakening a correct margin without reversing it; sensitivity to cross-prompt replacement alone does not prove that number information transferred. The four families were used to select the site and cannot establish held-out performance or a general agreement circuit.

## Reproducibility and evidence

- [Machine-readable discovery record](discovery-summary.json) retains every site, family and donor-cell effect.
- [Independent audit acknowledgment](independent-audit.json) is the exact saved audit output.
- [Audited evidence archive](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/distilgpt2-natural-agreement-v1-20261028/discovery-evidence.zip) contains the exact journal start, event and terminal files and all 1,120 content-addressed tensor payloads. The archive SHA-256 is `9ea7ebf83f0580335cbf35b65b2277d8990090adff5b3eb8d0b3b68409cd3a69`.
- [Published runtime binding](binding-v2.json) identifies the source, checkpoint, exposed cases and budgets. The executed source is commit `def07cbb168408a12c76ba30f196af45149e111d`.
- [Frozen predictions](natural-prediction-v2.json), [publication provenance](natural-prediction-provenance.json) and the later [prediction freeze](natural-prediction-freeze.json) were published before held-out allocation.
- The exact [prediction publication receipt](natural-prediction-track-receipt.json) and [aggregate publication receipt](natural-prediction-aggregate-receipt.json) record the observed official commits.
- The owner exited zero, was reaped, and its job drained and closed; the private original supervisor and launcher terminal are bound by the hashes in the discovery record. They retain local paths and are not reproduced in this public archive.

The earlier validation-only rehearsal remains preserved as incomplete: it loaded no model and ran no study forward. The corrected no-model rehearsal passed before this public discovery. No held-out examples had been generated when this discovery record was prepared.

The discovery owner inherited `MKL_CBWR` and `ATEN_CPU_CAPABILITY`, but their values were not recorded, so their historical values are unknown. Separate synthetic CPU checks found that `MKL_CBWR=AUTO` versus `COMPATIBLE` can change matrix-multiplication bytes and `ATEN_CPU_CAPABILITY=default` versus `avx2` can change softmax bytes on this host. These omissions limit exact replay of the original runtime; they do not by themselves establish an error in the saved discovery.
