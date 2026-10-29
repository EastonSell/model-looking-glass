# Exploratory fixed-donor agreement contrast

On the same 16 already-exposed prompts used in the v2 evaluation, the frozen
layer-3 head 4 had a mean signed contrast of **0.1575095654 logits**; all 16
contrasts were positive. The adaptively selected head-8 comparator's mean
absolute contrast was **0.0081913471 logits**. All 16 clean `is`/`are` margins
had the correct sign. This is an exploratory repeat, chosen after the prior
outcomes and feasibility inspection; it is not a new held-out test or a
pass/fail claim.

The earlier comparison used a same-lemma opposite-number donor versus a
different-lemma same-number donor. Here, both donor rows use the **same other
lemma**, one singular and one plural, within the receiver's four-case frame.
For each head and receiver, both donor directions are interpolated to the
same L2 dose q = 5% of that receiver head slice's L2 norm at the final-token
input to `transformer.h.3.attn.c_proj`. Neither displacement is amplified;
the realized dose must be within 1e-5 relative error of q and the donor-pair
dose ratio at most 1.00001. D is receiver-number sign (+1 singular, -1 plural)
times the `is`-minus-`are` margin under the same-number donor minus that
under the opposite-number donor. Head 8 was selected as the development
runner-up; it is an adaptive active comparator, not a random-head null.

| Frame | Subject lemmas | Head 4 mean D | Head 8 mean absolute D | Correct baseline |
| --- | --- | ---: | ---: | ---: |
| During the storm … beside the fences | pilot, guard | 0.1705322 | 0.0101547 | 4/4 |
| After the concert … near the tower | nurse, clerk | 0.1545277 | 0.0037346 | 4/4 |
| Before the meeting … by the gardens | pilot, nurse | 0.0999479 | 0.0112514 | 4/4 |
| Throughout the afternoon … behind the barn | guard, clerk | 0.2050304 | 0.0076246 | 4/4 |

Head-4 mean D over the four observations per lemma was 0.1114235 for pilot,
0.2242880 for guard, 0.1219511 for nurse and 0.1723757 for clerk. The 16
observations reuse four frames and four nouns and are not independent samples
from a population. The fixed other lemma removes the prior between-lemma donor
difference, but its singular and plural forms are different lexical tokens;
this alone does not identify an abstract number feature or unique circuit.
Equal relative input doses do not match the heads' absolute or downstream
output changes.

All 16 new clean projection inputs and 50,257-element last-logit vectors
matched the saved v2 witnesses bit for bit before any patch. All 32 shared
same-number donor patches also matched their v2 prestate, replacement,
landing, projection output and last logits. The 32 self patches, 16 clean
repeats, 64 direct donor landings and untouched-state checks passed. The
recorded runtime fields and model-state digest matched v2; that does not prove
every unrecorded environmental detail was identical.

In a post-hoc full-vocabulary analysis, no top-ranked token changed in any of
the 64 patch-versus-clean comparisons or the 32 same-versus-opposite donor
comparisons. Mean RMS difference over all logits between donor patches was
0.0864793 for head 4 and 0.00594258 for head 8. Mean total variation from the
clean next-token distribution was 0.0065451 and 0.0085678 for head-4
opposite- and same-number patches, versus 0.0011573 and 0.0012132 for head 8.
These are descriptive post-hoc next-token statistics, not generated text or
evidence that the larger overall head-4 shift is agreement-specific.

The source cases were public before this run and already forwarded in v2;
head choice and this follow-up were data-informed. The result does not
establish pretraining independence, new held-out generalization, a unique
mechanism or generated behavior. `casewise.csv` records the per-case margins,
contrasts and doses, and `summary.json` records the descriptive aggregates.
`saved-records.zip` includes the public case, v2 and v3 event/terminal records,
two saved-byte audits and full-vocabulary summaries, but **omits all tensor
witnesses**; it cannot alone replay the tensor checks. The complete local
archive includes the v3 tensor witnesses and the 16 prior clean witness pairs,
with a standalone replay; its SHA-256 is
`63472172a375da118befcdcb45c08766e90044ba1c48130e74ba2411db293bd3`.
The direct child exited 0; the outer launcher's observed 0 exit was manually
transcribed. Neither is an authenticated process-tree or execution
attestation.
