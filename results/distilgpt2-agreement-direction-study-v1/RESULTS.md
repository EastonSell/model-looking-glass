# DistilGPT-2 agreement direction: public follow-up

Both predeclared exploratory response criteria passed on the fixed public cases: the signed direction comparison passed in 8/8 families, and the projection-centering comparison passed in 16/16 pairs. The same direction also moved the `is − are` margin substantially in fragments without an explicit subject. These results support sensitivity to this projection in these contexts; they do not establish transfer of an abstract grammatical-number feature or a unique circuit.

The direction and L3 attention site were developed using an earlier, exposed discovery set. The case bundle was published after the [protocol](../../docs/AGREEMENT_DIRECTION_STUDY_V1.md) and [direction artifacts](../distilgpt2-agreement-direction-training-v1/); the operator reports that case assembly also began after both publications. They are public, nonblind evaluations, not a result from the separately sealed natural held-out split. The independent saved-byte replay reports every target, context, arm, dose, vocabulary diagnostic, and prediction in [science-replay.json](science-replay.json).

The fixed study used 32 target sentences in eight families and eight additional context fragments. Each family has four target variants; the eight families share four base frames. Each context fragment omits an explicit subject but retains a number-marked attractor. The protocol fixes 784 forward calls: clean/no-op and donor controls, signed direction and site/position interventions, pair centering, and clean-after checks. The frozen directions are `u`, estimated from 16 previously saved L3-final attention outputs, `v1`, a number-by-lemma modulation direction orthogonalized against `u`, and `v2`, a fixed pseudorandom orthogonal control. Neither control is a randomization distribution.

For the `is − are` logit margin `m`, the signed response is `O_a = (m(+q a) − m(−q a))/2`. `q` is half the minimum of six within-family donor distances for a same-lemma singular/plural pair, so the two targets share the same requested input-space L2. A direction response for a class is compared with the **mean of the absolute individual responses** of each control, avoiding cancellation. The fixed class threshold is 0.05 logit, with the same requirement in at least six of eight families. This is an exploratory cutoff, not a significance test or a validated practically meaningful effect size.

| Target class | n | Mean O_u | Mean absolute O_v1 | Mean absolute O_v2 |
| --- | ---: | ---: | ---: | ---: |
| singular | 16 | +0.832977 | 0.140962 | 0.026819 |
| plural | 16 | +0.334464 | 0.062909 | 0.018632 |

The corresponding eight family means retain their full four-target denominators:

| Family | Mean O_u | Mean absolute O_v1 | Mean absolute O_v2 | Criterion |
| ---: | ---: | ---: | ---: | --- |
| 1 | +0.460911 | 0.102890 | 0.048005 | pass |
| 2 | +0.540812 | 0.069381 | 0.015450 | pass |
| 3 | +0.511313 | 0.114975 | 0.031820 | pass |
| 4 | +0.587954 | 0.107231 | 0.012808 | pass |
| 5 | +0.456353 | 0.093547 | 0.016531 | pass |
| 6 | +0.551614 | 0.089078 | 0.015079 | pass |
| 7 | +0.661677 | 0.114913 | 0.016696 | pass |
| 8 | +0.899132 | 0.123470 | 0.025415 | pass |

The separate projection test moves the two members of each same-lemma pair toward the midpoint of their `u` coordinates. For `G = m_singular − m_plural`, it reports `loss_a = G_clean − G_after_a`; both controls use the same nominal input L2. The predeclared rule requires mean `loss_u >= 0.10` and greater than each mean absolute control loss, and at least 12/16 pairs with `loss_u >= 0.05` and greater than each pair's absolute control losses. The mean loss was 2.017209 for `u`, versus 0.229095 and 0.029738 for mean absolute `v1` and `v2` losses. This supports a contribution to this matched-pair contrast under these interventions, subject to the alternatives below.

| Family | Subject morphology | loss_u | loss_v1 | loss_v2 | Criterion |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | regular | +2.357338 | -0.315811 | +0.022224 | pass |
| 1 | irregular | +1.749802 | -0.205734 | +0.057983 | pass |
| 2 | regular | +2.464645 | -0.230659 | +0.061272 | pass |
| 2 | irregular | +2.011040 | -0.157860 | +0.065491 | pass |
| 3 | regular | +1.795452 | -0.315918 | -0.010986 | pass |
| 3 | irregular | +1.856060 | -0.246548 | +0.009724 | pass |
| 4 | regular | +1.735531 | -0.298817 | -0.008522 | pass |
| 4 | irregular | +1.901711 | -0.262489 | -0.005974 | pass |
| 5 | regular | +2.059406 | -0.232758 | +0.037796 | pass |
| 5 | irregular | +2.350883 | -0.202572 | +0.032681 | pass |
| 6 | regular | +2.406124 | -0.291332 | +0.033478 | pass |
| 6 | irregular | +2.784466 | -0.207626 | +0.019844 | pass |
| 7 | regular | +1.661022 | -0.190151 | +0.036354 | pass |
| 7 | irregular | +1.392498 | -0.148022 | +0.000523 | pass |
| 8 | regular | +1.845234 | -0.184090 | -0.028610 | pass |
| 8 | irregular | +1.904133 | -0.175140 | -0.044350 | pass |

There were no zero doses. The target-pair `q` values ranged from 2.543762 to 4.402681 (mean 3.461579); the independently replayed 560 intervention deliveries all satisfied the protocol's relative input-L2 tolerance, with maximum relative error 1.81e−8. The largest absolute residual `u` coordinate gap after `u` centering was 7.40e−8. The full replay retains each donor distance, requested and achieved norm, source norm, changed elements and bits, and pair's achieved gaps. Equal input L2 does not match direction, relative activation change, or downstream functional effect across sites.

The same-number/different-lemma and opposite-number donor arms are useful controls, but opposite-same versus same-different changes lemma relation as well as number. Opposite-different versus same-different preserves the donor-lemma relation. Mean weakening of the target-oriented present margin was:

| Target class | Same number, different lemma | Opposite number, same lemma | Opposite number, different lemma |
| --- | ---: | ---: | ---: |
| singular | -0.009324 | +0.956543 | +0.762317 |
| plural | -0.016833 | +0.378951 | +0.307231 |

All 32 clean target margins favored the grammatical choice in the `is`/`are` pair, and no donor arm reversed that pairwise margin. Donor weakening is not by itself evidence of number transfer: generic attenuation can produce it. Full per-target effects and family donor breakdowns are retained in the replay.

As descriptive site/position comparators, mean `O_u` for singular/plural targets was 0.255384/0.092337 at L1-final and 0.257207/0.112966 at L3-subject, compared with 0.832977/0.334464 at L3-final. The local activation distributions and functional effects differ, so the common nominal L2 is not a matched behavioral dose.

The context fragments show a substantial signed `u` response despite lacking an explicit subject:

| Context | Requested q | O_u (is − are) |
| --- | ---: | ---: |
| agreement-direction-context-01 | 3.596091 | +1.058495 |
| agreement-direction-context-02 | 3.664544 | +0.814159 |
| agreement-direction-context-03 | 2.543762 | +0.532719 |
| agreement-direction-context-04 | 2.830316 | +0.527657 |
| agreement-direction-context-05 | 3.421174 | +0.889399 |
| agreement-direction-context-06 | 3.089501 | +0.974962 |
| agreement-direction-context-07 | 4.402681 | +0.956306 |
| agreement-direction-context-08 | 4.144563 | +1.261559 |

Their mean response was 0.876907 (range 0.527657–1.261559). These are not neutral contexts: each retains a number-marked attractor, and two `Beside` fragments have five tokens while the other six have three. Their planned dose is the average of the family's two pair doses. In this run the two pair doses happened to match within each family, so each fragment and its corresponding targets had the same nominal q; q varied between families. The fragments are not a controlled subject-removal comparison, and equal input L2 does not imply equal functional effect. A shared output/token-bias or attractor-related explanation remains viable; the data do not prove a context-independent bias.

On the 32 targets, the full-vocabulary top token changed in 4 cases for `u+` and 6 for `u−`. Median `KL(clean || intervention)` was 0.002253 for `u+`, 0.002396 for `u−`, and 0.003529 for `v1+`. Thus the agreement-margin comparison and overall distribution change are different measurements. The saved replay contains all four recorded margins, raw plus/minus and even responses, target-token ranks, top-20 logits and full-vocabulary KL for every case/action; the 50,257-dimensional last-position vectors are in the evidence archive.

The results remain compatible with a global decoder bias, surface-morphology or lexical effects, attractor/context effects, and nonlinear or asymmetric disruption. The regular and irregular forms improve descriptive coverage but morphology is still entangled with lexical identity. A synthetic output-bias example in the independently reviewed reader passes the arithmetic criteria without establishing a grammatical feature. Families share frames, the threshold decisions are not independent statistical tests, and neither the direction nor site was discovered blindly.

## Evidence and reproducibility

The single supervised CPU attempt recorded 784 forward events and 809 journal events; its direct child exited 0 and the supervising parent was observed to exit 0. The owner recorded an unchanged before/after model-state hash and successful cleanup. The independent replay verified the fixed schedule, 1,872 content-addressed tensor payloads (132,032,352 bytes), saved final-position logits, clean/no-op/clean-after controls, exact intervention landing and untouched bits, geometry, all margins, vocabulary diagnostics, and prediction denominators. A separate provenance review checked the recorded file inventory, source blobs and cooperative process receipts. These checks are saved-data and observation consistency, not authenticated proof of execution, publication chronology, or descendant drain. The owner checked finiteness of full-sequence logits during the run; only last-position logits were saved, so the replay cannot independently check the unsaved positions.

The 13 `evidence.tar.gz.part-*` files contain the exact retained run, prelaunch, cooperative receipts, boundary/exit observations, and a path-free projection of the one-use claim. They rejoin to 108,205,215 bytes with SHA-256 `dba1da139648d8dd841bd9cc8cca41f8bfe64a2bdb73939e1f76857e97bcc352`. [archive-index.json](archive-index.json) gives the exact ordered part sizes and hashes. Run `python -B rebuild_evidence.py --directory <this-directory> --output <new-path>/evidence.tar.gz` to verify each part and the complete archive before inspecting/extracting it. The original claim contained a local output path and is retained locally; the published archive contains only its digest and path-free projection. The archive includes ordinary process IDs in cooperative receipts. The independent archive audit compared all 1,889 archive members to retained originals; its selected Windows-path scan is not a comprehensive privacy audit.

The model-free reader [source and instructions](https://github.com/EastonSell/model-looking-glass/tree/950ec5160ec691d7877bf3b14a4969117835ac13) are independently reviewed. Reconstruct the archive into a fresh directory, use its `run/attempt` as `--journal-directory`, and supply the published case file, direction directory, source revision `acb280820ea4fa3aaa4d41e6fc5b680c5955440c`, prelaunch SHA `c367112faf54cf8321b15059fbc8bb65460f3f83c0b52c7e9eb58a81dad49e33`, reservation SHA `3df696303a06b2ec6ddf4113df1969897dbae89566375b7d3a9cecb85b3d59cd`, and child-recorded v23 SHA `2549f1430ada47da088d2a909f8aa57ad5c08bf7c6d09f187bd34bb76b99f658`. The reader compares supplied pins to records; it does not authenticate them. Independently assembled [science-pins.json](science-pins.json), [primary-oracle.json](primary-oracle.json), [margins-review.json](margins-review.json), [custody-review.json](custody-review.json), and [archive-review.json](archive-review.json) document the separate checks. The archive checker imports the custody checker from the same directory. Repeating the original-to-copy custody audit needs the private local original claim and original attempt, which are not supplied by the public archive alone.

The v20 inventory reserved the exact 40 input rows. The child checked the v23 inventory; an independently reviewed v24 observation at launch documented that all relevant input/candidate rows, prefixes and agreement reservations were unchanged. The natural custodian reported that the existing sealed split was clear in both complete-row-prefix directions for this fixed case bundle; no natural successor split existed at launch. The prior natural held-out attempt had stopped without a saved scientific result after a conservative possible-forward announcement. The injection custodian reported no protected held-out rows and a scoped no-direct-target-conflict assessment for these fixed attention-output sites. These are scoped custody reports, not independently authenticated proof of all exposure or private disjointness.

Key saved SHA-256s: case file `a68aa63ff06aa97c5c68d941fe27160c4c4698cdf070325b3cefa2b29ec58b13`; scientific replay `8b816905a3009e212ab138c62cbb24f1dd06d53079e228c283bb5d8f6ae1842f`; journal `7c8be25d480d986b087052685720ba7d2813c53d27509dcc81e67f9e2049eca6`; supervisor receipt `7e5fcae4b415afdf06b3221e67692707e43449ee20e236d8ec580d9c57f843af`; current v24 canonical inventory `877fa0376925be34ac3de14ff066e5465a67953660bce81011e9c8fd87810bd3`. The 13 retained archive parts total 108,205,215 bytes; the original attempt used 135,057,009 bytes before the outer receipt. Archive reconstruction and local/Git copies require additional physical storage.

