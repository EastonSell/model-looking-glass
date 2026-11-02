# Agreement noun-substitution study

The preregistered descriptive comparison passed on these fixed public cases. Across 12 cells, the mean difference between opposite-number and same-number different-noun response distances was **0.219906** logit for the frozen agreement direction, versus **0.034990** and **0.050526** for the means of the individual absolute active-control differences. Eleven cells were strictly positive and one was exactly zero. All 48 target clean baselines had the expected strict is/are sign.

This is prospective, public and nonblind evidence of differential response on a small fixed design. It does not identify a unique abstract grammatical representation: surface inflection, lexical differences and prompt-dependent or nonlinear gain remain viable explanations. The same nouns recur across frames and contexts, so the 12 cells are not independent replications and the regular/irregular comparison is lexically confounded.

## Design and fixed decision rule

The [protocol](../../docs/AGREEMENT_NOUN_SUBSTITUTION_V1.md) was published before the [54 fixed cases](../../experiments/distilgpt2_agreement_noun_substitution_cases_v1.json). The 48 targets cross two frames, three contexts (no attractor, singular attractor, plural attractor), two morphology groups, two lemmas and two subject forms; six subjectless fragments are separate descriptive controls. The regular pairs are hand/hands and arm/arms; the irregular pairs are foot/feet and tooth/teeth. These shared lexical forms are not a lexical holdout. The frozen layer-3 final-token attention-output direction u and active controls v1/v2 each used signed input dose ±2.

For the is−are logit margin m, each response is `O_a(x) = (m(x,+2a) − m(x,−2a))/2`. Within a cell let A and B be different lemmas and s/p their singular/plural forms. `D_opp = (|O(A_s)−O(B_p)| + |O(B_s)−O(A_p)|)/2`, `D_same = (|O(A_s)−O(B_s)| + |O(A_p)−O(B_p)|)/2`, and `Δ = D_opp − D_same`. The signed contrasts are `N = ((A_s−A_p)+(B_s−B_p))/2`, `L = ((A_s−B_s)+(A_p−B_p))/2`, and `I = (A_s−A_p)−(B_s−B_p)`, using the corresponding O values.

The fixed rule required adequate clean baselines, overall mean Δ_u ≥0.05 and above each mean individual |Δ_control|; each morphology mean ≥0.02 and above its controls; at least 8/12 positive cells and 4/6 in each morphology; and positive means in both frames. All conditions passed. These thresholds were descriptive choices motivated by earlier exposed data, not validated practical thresholds or a statistical test.

## All 12 primary cells

Values are is−are logits. Control columns are signed cell Δ, whereas the decision compares the means of their individual absolute values. Frame 0 is “Before sunrise,” and frame 1 is “After sunset,”.

| Frame | Context | Morphology | D_opp u | D_same u | Δ u | Δ v1 | Δ v2 | N u | L u | I u |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | none | regular | 0.414049 | 0.037376 | 0.376673 | 0.000000 | 0.108929 | 0.414049 | -0.037376 | 0.062267 |
| 0 | none | irregular | 0.282120 | 0.029177 | 0.252943 | 0.001627 | 0.065756 | 0.282120 | 0.029177 | -0.008595 |
| 0 | attractor_singular | regular | 0.569252 | 0.090824 | 0.478428 | 0.134136 | 0.083080 | 0.569252 | -0.090824 | 0.010651 |
| 0 | attractor_singular | irregular | 0.479259 | 0.082588 | 0.396671 | 0.112217 | 0.080112 | 0.479259 | 0.029350 | -0.165176 |
| 0 | attractor_plural | regular | 0.221288 | 0.099093 | 0.122194 | -0.004856 | 0.003933 | 0.221288 | -0.099093 | -0.049166 |
| 0 | attractor_plural | irregular | 0.148212 | 0.033108 | 0.115105 | 0.000669 | 0.007700 | 0.148212 | -0.011023 | -0.066216 |
| 1 | none | regular | 0.321067 | 0.071069 | 0.249998 | 0.041882 | 0.086197 | 0.321067 | -0.064754 | 0.142138 |
| 1 | none | irregular | 0.206285 | 0.033024 | 0.173262 | 0.017879 | 0.053413 | 0.206285 | 0.033024 | -0.053261 |
| 1 | attractor_singular | regular | 0.109648 | 0.109648 | 0.000000 | 0.007154 | 0.000452 | 0.046196 | -0.109648 | 0.118271 |
| 1 | attractor_singular | irregular | 0.084254 | 0.053889 | 0.030365 | 0.030067 | 0.009842 | -0.084254 | 0.053889 | -0.106527 |
| 1 | attractor_plural | regular | 0.308219 | 0.092633 | 0.215586 | 0.031193 | 0.047094 | 0.308219 | -0.092633 | 0.012308 |
| 1 | attractor_plural | irregular | 0.254210 | 0.026558 | 0.227652 | 0.038193 | 0.059797 | 0.254210 | 0.026558 | -0.047497 |

The zero Δ is frame 1 / singular attractor / regular morphology: both distances are 0.109648, while N is 0.046196. The irregular cell in that same frame/context has N = −0.084254 despite positive Δ. This matters because the absolute-distance statistic can be zero even when a number-form term is nonzero, and a positive value can reflect surface or lexical gain.

| Group | Mean Δ u | Mean abs(Δ v1) | Mean abs(Δ v2) |
|---|---:|---:|---:|
| all 12 | 0.219906 | 0.034990 | 0.050526 |
| by_morphology: irregular | 0.199333 | 0.033442 | 0.046103 |
| by_morphology: regular | 0.240480 | 0.036537 | 0.054948 |
| by_frame: 0 | 0.290336 | 0.042251 | 0.058252 |
| by_frame: 1 | 0.149477 | 0.027728 | 0.042799 |
| by_condition: attractor_plural | 0.170134 | 0.018728 | 0.029631 |
| by_condition: attractor_singular | 0.226366 | 0.070894 | 0.043372 |
| by_condition: none | 0.263219 | 0.015347 | 0.078574 |

Every baseline subgroup was fully correct: 24/24 in each frame and morphology, 16/16 in each context, 12/12 for each lemma, and all 12 cells had four correct forms. No case was filtered.

## Subjectless and output controls

The six subjectless fragments had positive u odd responses in the is−are margin, mean **0.269007** and range **0.234711–0.315760**. Their syntax, final token and position differ from the target sentences; they are not a controlled subject deletion or a neutral null. Their responses leave shared output, attractor and context effects plausible.

| Fragment | O u | O v1 | O v2 |
|---|---:|---:|---:|
| noun-sub-f1-c0-context | 0.237177 | -0.031483 | 0.004910 |
| noun-sub-f1-c1-context | 0.244991 | -0.022953 | 0.023975 |
| noun-sub-f1-c2-context | 0.300611 | -0.034933 | 0.015188 |
| noun-sub-f2-c0-context | 0.234711 | -0.032005 | 0.006685 |
| noun-sub-f2-c1-context | 0.280794 | -0.048462 | 0.010088 |
| noun-sub-f2-c2-context | 0.315760 | -0.006021 | -0.006001 |

| Margin | Mean Δ u | Mean abs(Δ v1) | Mean abs(Δ v2) |
|---|---:|---:|---:|
| is−are | 0.219906 | 0.034990 | 0.050526 |
| was−were | 0.164316 | 0.031258 | 0.028521 |
| the−a | 0.009076 | 0.000745 | 0.001167 |
| in−on | 0.004776 | 0.001213 | 0.000744 |

For the 48 targets, median full-vocabulary KL(clean‖arm) and top-one changes are shown below. Equal input L2 is not equal functional strength. The [full replay](science-replay.json) retains every case, four margins, odd/even values, signed factorial differences, top-20 lists, target ranks, KL values and delivery diagnostics. The raw witness tensors are in the archive.

| Arm | Median KL | Top-one changes |
|---|---:|---:|
| u_plus | 0.002068 | 3/48 |
| u_minus | 0.001893 | 4/48 |
| v1_plus | 0.000947 | 2/48 |
| v1_minus | 0.000854 | 2/48 |
| v2_plus | 0.001048 | 3/48 |
| v2_minus | 0.001047 | 2/48 |

All 324 nonzero deliveries passed the fixed norm and signed-projection checks. Realized norms ranged from 1.999999947 to 2.000000045; the maximum relative norm error was 2.64e-08. These checks concern input displacement, not matched output strength.

## Evidence and scope

The single supervised CPU attempt recorded 486 forwards and 487 journal events. The direct observer exited zero and reported a complete candidate; its cooperative observation does not authenticate execution or prove descendant drain. The [independent source-first replay](science-replay.json) checked the 1,134 retained tensor payloads, schedule, full saved last-position logits, no-op equality, landing bits and doses. A [separate selected-logit replay](selected-replay.json) agrees on all primary margins and factorial values. The owner recorded full-sequence finiteness live; unsaved sequence positions cannot be independently replayed from these records.

The exact [owner source](https://github.com/EastonSell/model-looking-glass/tree/ca6c67f30b90f5d79df68e379cbb7cafe21acb08) was public before the run; the reviewed owner was later merged at `1d9b97cd31ebe13effbe844e9668567c58ce7283`. The child reconstructed the v35 reservation; a separate operator observation bound the independently reviewed v36 successor. The [prelaunch](prelaunch.json), [boundary](launch-boundary.json), [parent exit](parent-exit.json), [custodian reports](retrospective-custody.json), and [redacted replay observation](replay-observation-public.json) are cooperative observations, not proof of complete exposure history or private membership. The original natural-split support implication is conditional on its published preparation; no private comparison was performed.

After the direct exit, Windows reported 91,825,856 allocated bytes for ordinary files in the attempt; directory metadata and alternate streams were outside that observation. The inherited supervisor field `physical_bytes_before_receipt` sums apparent file lengths, not allocated bytes. Neither observation is an OS quota or descendant quiescence guarantee.

The [parts manifest](parts-manifest.json) binds nine archive parts (70,545,740 bytes total, SHA-256 `2db2b35d1ed61aad125ea089d22429f92c635fa5b657c654b1a5e21ae453540b`). The [file manifest](archive-manifest.json) lists all 1,158 files and their hashes; the original case-bound claim stays outside the archive. Run `python -I -S -B join_evidence.py --parts-dir parts --manifest parts-manifest.json --output study-attempt-v1.tar.gz` to verify and join them. A failed write or final hash check can leave a partial output file. No full archive copy was created during publication staging.
