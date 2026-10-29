# Matched-dose DistilBERT agreement study

The preregistered early-site persistence criterion **was not met**. The mean
matched-dose effect was 2.413225 at block 0 and 2.179126 at block 2, with a
mean within-target difference of +0.234099 logits. The first two thresholds
were met, but only **4 of 8** family differences were positive; the rule
required at least 6. This failed directional rule does not establish that the
two sites are equivalent or identify a unique location for the behavior.

The study used the pinned DistilBERT masked-language model, 32 target prompts
in eight public, nonblind families, and the fixed matched-magnitude protocol.
All 32 unpatched targets preferred the correct `is`/`are` token. At each site,
the primary effect contrasts the different-lemma opposite-number donor with
the different-lemma same-number donor after matching their target-specific
absolute displacement norm. The values below are family averages of four
targets; the final row is the mean of those eight family averages.

| Family | Block 0 | Block 2 | Paired difference |
| ---: | ---: | ---: | ---: |
| 1 | 1.558145 | 1.864908 | -0.306764 |
| 2 | 3.410244 | 2.118207 | +1.292037 |
| 3 | 1.207598 | 1.396465 | -0.188867 |
| 4 | 3.657467 | 2.931286 | +0.726180 |
| 5 | 1.928210 | 2.158137 | -0.229927 |
| 6 | 2.670507 | 2.296593 | +0.373913 |
| 7 | 3.104022 | 2.071709 | +1.032312 |
| 8 | 1.769606 | 2.595701 | -0.826095 |
| Mean | 2.413225 | 2.179126 | +0.234099 |

The four subject/distractor-number cells each contain eight targets. The
unpatched margin is `logit(is) - logit(are)`; all cells had 8/8 correct
baseline predictions.

| Subject / distractor | Mean clean margin | Block 0 effect | Block 2 effect | Paired difference |
| --- | ---: | ---: | ---: | ---: |
| Singular / singular | +4.907717 | 2.108933 | 1.767116 | +0.341817 |
| Singular / plural | +1.659609 | 3.250792 | 2.936020 | +0.314773 |
| Plural / singular | -3.405063 | 3.191938 | 3.076593 | +0.115345 |
| Plural / plural | -4.890803 | 1.101236 | 0.936776 | +0.164461 |

The fixed rule required a block-0 mean greater than 0.1, a mean paired
difference greater than 0.1, and at least six positive family differences.
These are descriptive cutoffs, not significance tests. Families share a cyclic
donor design and are not independent population samples; the public cases are
a convenience sample, and prior exposure outside the cited history is unknown.

For each target, the common radius was the minimum of its six donor distances
across the two sites. The 32 radii ranged from **7.939570 to 10.603014**, with
mean 8.942285. Across the three natural donor directions, realized displacement
was **45.29%–61.66%** of the base-vector norm at block 0 and **42.12%–56.56%**
at block 2. The corresponding interpolation fractions ranged from 0.389342 to
0.912125 and from 0.487209 to 1.000000, respectively. The maximum observed
absolute norm errors in those natural directions were below 4.4e-8. Equal
absolute norms do not imply equal relative or semantic changes across layers;
these substantial interpolated states may be unnatural. The separately fixed
artificial-direction control had mean oriented shifts of 0.199714 at block 0
and 0.138424 at block 2, with a maximum absolute norm error of 2.57e-7; it is
descriptive, not a randomization null.
The different-lemma same-number donor was also not an identically zero
control: its mean oriented change from the clean margin was 0.082107 at block
0 and 0.058579 at block 2. Each of those controls had a nonzero recorded shift
for all 32 targets at both sites. Here the oriented change is `-r * (L -
L_clean)`, where `r` is +1 for singular and -1 for plural targets.

From the 32 initially correct targets, the A/opposite donor reversed the
`is`/`are` preference in 30 cases at block 0 and 21 at block 2. The B/opposite
donor reversed it in 13 and 11 cases, respectively; B/same reversed none.
The A/same self-patch has zero dose, so the A/opposite versus self contrast is
a secondary comparison rather than a matched-dose donor contrast.

The closed journal contains 1,152 ordered records for 480 completed forward
observations: 16 source captures, 24 scaled donor patches, eight self-patches,
eight artificial-direction patches, and four final clean forwards per family.
The saved-record auditor checked the ordered schedule, recorded inputs and
donors, matched-dose calculations, delivered vector bits, clean/self/final
controls and aggregation. The live runner and the later closed-journal reads
use the same auditor implementation; their agreement is a saved-byte
consistency check. Full mask-logit vectors were compared during execution, but
only their digests and the selected `is`/`are` bits are retained, so the full
vectors cannot be reconstructed from the journal. All 160 prescribed equality
controls passed: 64 cross-site clean comparisons, 64 self-patches, and 32 final
clean comparisons.

A separately written [numeric replay](independent-numeric-replay.json) decoded
the saved float32 patterns and independently reconstructed all 480 recorded
input/site identities, 32 common radii and their six donor distances, 256
rounded requested and landed vectors, the equality controls, and the
per-target, family and aggregate primary contrasts. Its
[source](../../tools/audit_bert_matched_results.py) uses a separate
`Fraction`-based rounding implementation; its report SHA-256 is
`5cf0aaf24debb4071e5b64bc0e6be623880f2cfe047e56a46e80d95d62f4e5ca`.
The replay checks saved public data; it does not prove execution provenance or
recover the full mask-logit values.

The public controller exited 0, and the separately observed owner exited 0 and
was reaped; no failure or CPU unsafe marker was present. The owner recorded the
same CPU model-state hash before and after the run, and checked hook cleanup,
tokenization, the calling-thread denormal and binary64 probes, and the fixed
runtime settings. The frozen source revision is
[`4ac00abf6938b843abc5015fa83e7ad60d993262`](https://github.com/EastonSell/model-looking-glass/commit/4ac00abf6938b843abc5015fa83e7ad60d993262),
published in [PR #331](https://github.com/EastonSell/model-looking-glass/pull/331).
The [protocol](../../docs/BERT_MATCHED_MAGNITUDE_V1.md) and
[public case manifest](../../experiments/bert_matched_public_cases_v1.json)
have SHA-256 `91683a8f58ec182428f54c29e00d2bace692de3288014f3bf5f16bcbf665d35c`
and `5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb`.
The model-state SHA-256 was
`6bc3ce8e2f3d6518d6cc69e5f983af0208eab3957fa216be26006d502e18e9c0`.

[Evidence inventory](evidence-inventory.json) pins the exported evidence and
replay files and the auditor source; it does not hash this README or itself.
[events.jsonl.gz](events.jsonl.gz) decompresses exactly to the closed journal,
SHA-256 `9f482a369742960e24273d595d33d2a5fdc8ba2dfc82f1d65f6046266301a3ca`;
the gzip SHA-256 is
`1a405510648a649b7561a88fa2a770532fcafb2e6e87d587e038eada99033495`.
The original reservation, owner candidate, controller candidate, start/exit
observations, stdout and empty stderr are retained separately. The controller
candidate is promoted here only with the separately observed zero exit and no
failure marker. These are cooperative local process and file observations,
not independent attestation of the executed interpreter, native libraries, or
unknown previous exposure. Source locks rely on a trusted, quiescent checkout;
forced controller death can release them before its child has fully exited.
