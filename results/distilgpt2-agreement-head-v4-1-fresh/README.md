# Fresh-context fixed-donor agreement check

On 16 fixed public DistilGPT-2 prompts, layer-3 head 4's mean signed donor-number contrast was **0.1351812 logits**, positive in all 16 cases. The adaptively selected head-8 comparator's mean absolute contrast was **0.0166411 logits**. All 16 clean `is`/`are` margins had the correct sign, and all four frames met their prespecified comparison. The result is `passes_fixed_rule`; it is a descriptive result for these cases, not evidence of a unique grammatical circuit or a population-wide effect.

## Design and fixed result

At the last-token input to `transformer.h.3.attn.c_proj`, the study replaces one head's receiver slice with an interpolated displacement toward a donor slice. Within each four-case frame, the same-number and opposite-number donors use the **same other subject lemma**, respectively in the same or opposite number as the receiver. For each head and receiver, both displacements have the same requested L2 dose **q = 5% of that receiver head slice's L2 norm**. Donor distances must each be at least q, delivered doses must be within 1e-5 relative error of q, and the within-pair ratio must be at most 1.00001. The realized dose divided by the receiver norm ranged from 0.0499999862 to 0.0500000078; q divided by the donor distance ranged from 0.0493905 to 0.3432416. Equal receiver-relative input doses do not imply equal absolute doses, equal donor interpolation fractions or matched downstream output changes.

For each receiver, `margin = logit(" is") - logit(" are")` at the final position, and `D = sign * (margin_same - margin_opposite)`, with sign +1 for singular and -1 for plural. The clean baseline judges the sign of this pairwise margin; it does **not** require `is` or `are` to be the top token. The frozen rule required an adequate baseline (12/16 overall and 3/4 in each frame), head-4 mean D at least 0.05 and greater than head 8's mean absolute D, at least 10 positive head-4 contrasts, and at least three frame means positive and greater than their head-8 comparators. All requirements passed.

| Fixed frame | Subject lemmas | Head 4 mean D | Head 8 mean absolute D | Correct clean signs | Frame criterion |
| --- | --- | ---: | ---: | ---: | --- |
| Before the ceremony, the {subject} near the lanterns | singer, dancer | 0.1557713 | 0.0217648 | 4/4 | Pass |
| After the performance, the {subject} beside the museum | lawyer, actor | 0.1458817 | 0.0103951 | 4/4 | Pass |
| During the lesson, the {subject} behind the curtains | singer, lawyer | 0.1232538 | 0.0154610 | 4/4 | Pass |
| Throughout the morning, the {subject} by the orchard | dancer, actor | 0.1158180 | 0.0189438 | 4/4 | Pass |

The descriptive head-4 mean D across each lemma's four observations was 0.1353445 for singer, 0.1192627 for dancer, 0.1362734 for lawyer and 0.1498442 for actor. The observations reuse four frames and four lemmas and are not independent draws from a population. Intervening-noun number varies with frame and subject pairing, so these data do not isolate an attractor-number effect.

## Whole-vocabulary observations

The following are post-hoc next-token distribution summaries over all 50,257 saved logits, not a preregistered endpoint or generated text. In the 16 head-4 cases, the same-number patch changed the clean top token 0 times and the opposite-number patch changed it twice. For head 8, both kinds of patch changed the top token once, on the same case; its same and opposite patches never differed in top token. For example, in `After the performance, the lawyer beside the museum`, the head-4 opposite patch changed the clean top token from `,` to `'s`. In `During the lesson, the singers behind the curtains`, it changed ` were` to ` of`; both head-8 patches also changed that case's top token to ` of`.

| Head | Mean TV, clean to same | Mean TV, clean to opposite | Mean full-logit RMS, same versus opposite |
| --- | ---: | ---: | ---: |
| 4 | 0.0053694 | 0.0059681 | 0.0792034 |
| 8 | 0.0020777 | 0.0020072 | 0.0125037 |

TV is total variation between softmax distributions. A larger overall distribution change does not establish that the change is agreement-specific, and no continuation was generated.

## Evidence and scope

The 128-forward run comprises 16 clean observations, 32 self-patch controls, 64 donor patches and 16 clean repeats. An independent saved-byte replay verified the dose plan, requested and landed head-slice bits, unchanged other coordinates, self controls, repeats, fixed outcome and direct-child receipt. The direct supervisor recorded child exit 0; the outer launcher's exit 0 is an operator transcription, not authenticated execution or process-tree attestation.

`casewise.csv` contains each fixed case, margins, contrasts, donor distances, q/d, requested and delivered doses and top-token IDs. `summary.json` contains the frame, lemma and vocabulary summaries. `saved-records.zip` contains the fixed cases, owner/event/terminal records, two saved-byte reviews and post-hoc vocabulary report, but **omits all 20,776,256 bytes of tensor witnesses** and cannot by itself replay tensor checks. A complete local archive includes the witnesses and standalone replay; its compressed size is 17,611,859 bytes and SHA-256 is `6643d4f6815cfae656f0e37f32e5ee0cf533ab5ca5fd896e736422204d984714`. It was separately extracted, CRC-checked and replayed to the independently reviewed result.

The corrected fixed cases were published before forwarding (SHA-256 `bfc93dbcf33f06c8266b0fabf351dfa9f17ab98d6501ca7380e2cd171bddc8b4`); the earlier v4 proposal was superseded without a forward. The reviewed worker source was commit `97aba62c998ebe48c8f902468f712109b06a07b3`. Public inventory v44 recorded the cases as planned before this run; that historical label does not mean they remain unexposed. Public-only prefix checks covered the named known inputs, but unknown exposure and separately protected history were not queried or established.

The frames were public and nonblind. The four lemmas were absent from the earlier head-study development and evaluation cases, which does not establish absence from other model inputs or pretraining. Both heads were selected using exposed development; head 8 is an adaptive active comparator rather than a random-head or downstream-dose-matched null. Singular and plural donor forms are different lexical tokens, and the four frames and lemmas do not establish broader generalization or unique localization.
