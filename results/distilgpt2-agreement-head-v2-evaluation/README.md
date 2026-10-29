# DistilGPT-2 agreement-head held-out evaluation

The frozen layer 3 head 4 passed the specified comparison on 16 new, public
evaluation prompts. The clean `is` versus `are` margin had the correct sign in
16/16 prompts. The head-4 mean signed contrast was **0.1722009182 logits**,
versus **0.0110042095** for the mean absolute contrast of the frozen head-8
comparator. All 16 head-4 contrasts were positive, and all four frame-level
criteria passed. The fixed rule required a correct baseline in at least 12/16
and 3/4 per frame, mean head-4 contrast at least 0.05 and greater than the
mean absolute comparator contrast, at least 10 positive head-4 contrasts,
and at least three passing frame comparisons.

For each receiver, let the margin be the final-position `is` logit minus the
`are` logit. The contrast D is the receiver's grammatical-number sign
(+1 for singular, -1 for plural) times the margin under the same-number donor
patch minus the margin under the opposite-number donor patch. Both patches
replace the last-token head slice at the input to `transformer.h.3.attn.c_proj`.
Each is interpolated by the same requested L2 dose, **5% of that receiver
head-slice's L2 norm**, separately for each head and receiver. Both donor
distances had to reach that dose without extrapolation; realized doses had to
be within 1e-5 relative error of the request and agree within a ratio of
1.00001. All 32 fixed head/receiver cells met the rule. The 32 self patches,
16 clean repeats, direct landings and untouched-state bit checks passed.
Head 8 was the adaptively selected development runner-up, not a random-head
null. Equal relative input doses do not match absolute dose or downstream
output effects between heads.

| Frame | Subject lemmas | Head 4 mean D | Head 8 mean absolute D | Correct baseline |
| --- | --- | ---: | ---: | ---: |
| During the storm … beside the fences | pilot, guard | 0.1906414 | 0.0153255 | 4/4 |
| After the concert … near the tower | nurse, clerk | 0.1581974 | 0.0053406 | 4/4 |
| Before the meeting … by the gardens | pilot, nurse | 0.1094904 | 0.0134506 | 4/4 |
| Throughout the afternoon … behind the barn | guard, clerk | 0.2304745 | 0.0099001 | 4/4 |

Within the four observations per lemma, head-4 mean D was 0.1257524 for
pilot, 0.2550364 for guard, 0.1266918 for nurse and 0.1813231 for clerk.
These reused nouns and templates are not 16 independent samples from a
population. More specifically, the opposite-number donor has the same noun
in the other number, while the same-number donor has a different noun in the
same frame. The donor contrast therefore also changes lexical identity; it is
not a clean isolation of grammatical number from noun identity.

An exploratory check used the saved 50,257-element final-position logits.
Neither head changed the top-ranked token in any of the 64 patched-versus-clean
comparisons. Mean total variation from the clean next-token distribution was
0.006188 and 0.008568 for head-4 opposite- and same-number patches, versus
0.001012 and 0.001213 for head 8. Mean RMS logit difference between the two
donor patches was 0.113829 for head 4 and 0.010715 for head 8. These are
post-hoc descriptions, not frozen endpoints or generated-text results; the
larger overall head-4 output shift is not itself evidence of an
agreement-specific effect.

The development scan and its 5% redesign were informed by already exposed
activations. The heads were frozen before constructing these 16 cases, which
were published and reserved before forwarding; the cases are public and not
blind. Exact prefix checks cover recorded public inventories and the stated
public natural-support pools, not unknown exposure or pretraining data. This
result supports this fixed, small intervention comparison; it does not
identify a unique agreement circuit or establish generalization to other
frames, words, doses, models or generated text.

`casewise.csv` lists the fixed-input margins, contrasts and requested and
realized doses. `summary.json` records the outcome and exploratory metrics.
`saved-records.zip` contains the public case file, event and terminal records,
candidate, child-exit receipt, both saved-byte audits, and post-hoc result
tables. It does **not** include the 20,530,496 bytes of content-addressed
tensor witnesses, so that public archive alone cannot replay tensor-level
checks. A complete local evidence archive with those witnesses, the case file,
auditor and standalone replay has SHA-256
`0f862e6973eb5978dc836948db17b42d4acc6e88a110816c5b28976961fcbdbb`;
it was independently decompressed and replayed. The direct child exited 0;
the outer launcher's observed 0 exit was manually transcribed. Neither record
is an authenticated process-tree or execution attestation.
