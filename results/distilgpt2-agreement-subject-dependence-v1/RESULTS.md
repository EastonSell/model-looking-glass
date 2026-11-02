# Agreement subject-form sensitivity: saved-run results

The fixed L3-final attention-output direction had a larger signed effect on
the `is − are` logit margin in the singular member of 31 of 32 matched
subject-form pairs. The mean difference was **+0.333422 logits**, compared
with mean absolute pair differences of **0.075509** and **0.064553** for the
two fixed equal-input-norm controls. All four frame-level criteria and the
prespecified overall sensitivity criterion passed. All 32 clean pairs had
the correct sign of the binary `is − are` margin. These are descriptive
results on fixed, public, nonblind cases, not an independent-sample test.

The same intervention also shifted the margin in all 16 fragments without an
explicit subject: their mean signed response was **+0.306221**, including
**+0.206851** for the four no-attractor fragments. The fragments differ in
syntax, token identity and sometimes length from the target sentences; they
are not controlled subject deletion. Together, these results support a
subject-form-dependent response alongside substantial subjectless-context
sensitivity. They do not identify a unique grammatical-number representation:
surface morphology, lexical identity, prompt-dependent gain, downstream
nonlinearity and shared output bias remain possible contributors. A genuine
number direction with a linear readout could also produce a zero pair
difference, so the criterion is not a complete test for number encoding.

## Study and fixed comparisons

The [published protocol](../../docs/AGREEMENT_SUBJECT_DEPENDENCE_V1.md)
specified 64 targets and 16 shared fragments across four frames, two crossed
regular/irregular subject pairs per frame and four conditions: no attractor,
mass noun, singular attractor and plural attractor. Each subject pair appears
in two frames; the 32 pair cells and four frame aggregates are correlated.
The eight fixed subjects and four attractors were selected by tokenizer rules
after protocol/reference publication, without model-score selection. The
direction and L3 site were selected from earlier exposed discovery, and the
prior public evaluation helped motivate this follow-up. These cases are
prospective but neither secret nor blind.

For direction `a`, let `m = z(is) − z(are)` and
`O_a(x) = (m(x,+2a) − m(x,−2a))/2`. The main pair contrast is
`K_a = O_a(singular) − O_a(plural)` at the same nominal dose. `G_a` is the
dot product of `a` with the clean singular-minus-plural L3 output vector.
`G` describes association with the clean projection and does not show that
the model uses an isolated number feature. `v1` is a fixed number-by-lemma
modulation control; `v2` is a fixed pseudorandom orthogonal control. Neither
is a randomization distribution or a guaranteed non-number direction.

| Condition | Pairs | Mean K_u | Mean abs(K_v1) | Mean abs(K_v2) | Mean G_u | Fragment mean O_u |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No attractor | 8 | 0.270595 | 0.047799 | 0.032838 | 2.921499 | 0.206851 |
| Mass noun | 8 | 0.534387 | 0.109916 | 0.143598 | 5.313162 | 0.348616 |
| Singular attractor | 8 | 0.251168 | 0.096954 | 0.058932 | 7.061149 | 0.312996 |
| Plural attractor | 8 | 0.277540 | 0.047367 | 0.022846 | 7.741938 | 0.356422 |

Each fragment mean has four fixed frames. The no-attractor condition changes
the final token and position compared with the locatives; `water` also differs
semantically from the count nouns. These rows therefore do not isolate an
attractor-number effect. [Pair values](pair-contrasts.csv), all
[case responses](case-responses.csv) and the prespecified
[within-pair condition contrasts](within-pair-condition-contrasts.csv) retain
the individual outcomes. The four frame means of K_u were 0.238631, 0.347657,
0.335451 and 0.411950. One K_u cell was negative (−0.019634); none was zero.

The mean `O_u` was 0.610858 across singular targets, 0.277435 across plural
targets and 0.306221 across fragments. Mean paired `O_u` differences for the
other fixed margins were +0.287683 for `was − were`, +0.023560 for
`the − a`, and +0.002746 for `in − on`; these are descriptive output
controls, not proof of selectivity.

The predeclared exploratory geometry criterion passed: mean G_u was 5.759437,
versus mean absolute control values 0.368979 and 0.088394, with all four
frames passing. The general constant-additive K tolerance did not pass:
mean |K_u| was 0.334649 and 2/32 cells were within 0.10, versus the fixed
requirements of at most 0.10 mean and at least 24/32 cells. This challenges
that narrow prompt-independent constant-shift description, not all output
bias or nonlinear alternatives. The separate earlier-fragment fixed-slope
reference also did not pass: mean absolute O error was 0.180995 (within the
0.20 mean limit), but only 39/80 cases met the 0.20 per-case limit rather
than the required 60. The reference scales earlier finite-dose logits to
q=2; poor fit may reflect context or nonlinear dose response and does not
uniquely identify an activation mechanism.

## Delivery and vocabulary checks

All 480 signed interventions met the prespecified norm and signed-projection
tolerances. For requested input L2 2.0, realized L2 ranged
1.99999996196–2.00000004516; the largest relative norm error was 2.26e−8.
The clean source norms ranged 7.004207–18.322874. Equal input L2 does not
imply equal functional effect. Successful full-sequence logits were checked
for finiteness by the owner, but only final-position logits were retained, so
the saved-byte replay cannot independently establish finiteness at other
positions.

Across the 80 inputs, `u_plus` and `u_minus` changed the full-vocabulary
top token in 5 and 7 cases respectively. Their median KL divergence from the
clean distribution was 0.001468 and 0.001424; `v1_plus` and `v2_plus` had
median KL 0.001271 and 0.001151 and changed the top token in 4 cases each.
These broad-output diagnostics and the subjectless responses constrain a
simple specificity claim. The full replay records every top-20 list, target
rank, four margins, reference comparison, realized dose and centered
full-vocabulary alignment.

## Execution, evidence and reproducibility

The reviewed source reported 720 forwards in a single gated CPU attempt; its
closed journal contains 721 events covering all 80 cases. The operator
observed a zero parent exit and a matching clean cooperative direct-child
receipt. An independent source/receipt reconciliation and a separate
source-first saved-byte scientific replay passed; a second selected-logit and
clean-capture replay agreed on every case margin, pair K/G and threshold
decision. These checks establish consistency of retained records, not
authenticated execution or descendant drain. The full report is
[science-replay.json](science-replay.json); the independent selected replay,
comparison and custody reports are retained beside it.

The child reconstructed and bound v27. A later v28 snapshot merged after the
observer started; the separate [post-start check](post-start-v28-observation.json)
found that it preserved all v27 records and the 80 reservations, with zero
bidirectional relations across 1,872,000 comparisons against other known
inputs/continuations. This later observation does not change the child's v27
binding. The natural custodian retrospectively reported no successor claim or
split during the attempt and the injection custodian reported no protected
injection rows or claim; these are scoped reports, not a continuous monitor
or proof of complete exposure history. The original natural held-out attempt
had stopped without a scientific result; the public-support coverage remains
conditional on its recorded source-bound preparation.

The exact 1,704-file closed attempt is retained as 13 ordered archive parts.
Its reassembled gzip tar is 104,294,509 bytes, SHA-256
`14accf1935da546d76214c06436ec7a1d94896fa22201782587d37a20490d688`;
[parts-manifest.json](parts-manifest.json) lists each part's size and digest,
and [archive-manifest.json](archive-manifest.json) lists the original file
digests. Its three 48-MiB part entries document an earlier scratch split;
the separate parts manifest describes the 13 parts published here. The separate
CPU-root claim, which includes a local output path, is not in the archive. A
bounded text-pattern scan found no local-path or
credential matches; it is not a comprehensive privacy guarantee.

From this directory, verify and join the parts to a new file with
`python -I -S -B join_subject_archive.py --parts-dir . --output NEW_ARCHIVE`,
then extract the archive to a new directory. A failed join may leave an
incomplete output file. The source-first
[independent reader](../../docs/AGREEMENT_SUBJECT_SAVED_REPLAY_V1.md) at merged
revision `534eaea3bebb96e4ba4f337eef4ffe7cac2c205a` gives its CLI and the
required externally verified source and binding pins. The recorded owner was
`cc4937690db8e0ea8532d3f89f14b6e95465ac65`; the prelaunch SHA-256 was
`da8cd8e61200837a5e602ae24cee3effe791c68a8358529784be688e18557bce`, and both
reservation and child-bound exposure SHA-256 were
`ef42683a664dda450e82c69afedda34a8a34f2daf9f50f2db54f6c9d846f07e5`.
Use a checkout with the reviewed reader files and its referenced Git objects.
The separate custody and selected-logit audit scripts accept explicit paths;
the supplemental boundary and v28 review helpers use workspace-relative paths
and need adaptation in another checkout.
