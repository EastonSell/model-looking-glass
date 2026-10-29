# FLAN-T5 targeted input-row edit: held-out targets and locality

The fixed primary target criterion was **positive**: all four baseline-eligible target wordings changed from strict `positive` to strict `negative`. This is a narrow result for the same `wonderful`/`awful` word pair and the fixed prompts. It is not evidence of a general sentiment concept edit or broad capability preservation.

The primary edit copied the exact `awful` shared encoder/decoder input row into the `wonderful` row. For every target, its comparable same-prefix logits and generated sequence matched the clean donor-word input. That correspondence is a mechanical consequence of the aligned token substitution while the compared decoder prefix omits the edited token.

The broader negative-label output-head control met the development calibration rule at dose 2.0 but **failed the held-out balance rule**: on target 2 its margin gain differed from the primary gain by 1.15287, above the allowed 1.06858. The random and opposite input-row controls were unmatched on calibration. Thus every efficacy-matched locality comparison is ineligible; their outcomes are descriptive and do not establish a comparative locality advantage.

| Locality | Input contains edited token | Clean strict result | Primary edit | Broader head control |
| --- | --- | --- | --- | --- |
| 1 | no | correct | correct | correct |
| 2 | no | correct | correct | wrong |
| 3 | no | correct | correct | wrong |
| 4 | no | correct | correct | wrong |
| 5: repeat `wonderful` | yes | correct | wrong | correct |
| 6: sides of a wonderful triangle | yes | wrong | wrong | wrong |
| 7: sides of a square | no | correct | correct | correct |
| 8: opposite of hot | no | wrong | wrong | wrong |

The six token-absent inputs have structurally unchanged logits under the four shared-row arms whenever the actual paired decoder prefixes agree and omit token 1627. Five of these six were baseline-correct. They check that structural identity, not selective preservation. The broad head arm changes a different parameter and is not covered by that identity.

The two token-present inputs provide the relevant collateral probes. For `Repeat exactly this word: wonderful`, clean output was `wonderful` with EOS, whereas the primary edit decoded `awful` at the four-token cap **without EOS** and failed the fixed strict criterion. The broader control returned `wonderful` with EOS, but its efficacy comparison was ineligible. All four shared-row arms failed this repetition case. Its teacher-forced candidate prefix `[0, 1627]` reads the edited decoder input row, so that score includes a direct decoder effect. The primary greedy sequence never used 1627 in its decoder prefix; its failure should not be attributed to that decoder-token path. The triangle question was already baseline-wrong, so it cannot establish preservation or loss of a previously correct answer. Locality 8 was also baseline-wrong.

For these 40 actual first-step locality/arm pairs, an [independent Decimal replay](independent-science.json) agrees with the saved values to within 1.0e-15 nats for KL and 9.0e-17 for TV; the 24 structural shared-row pairs are byte-identical zeros. On the token-present repetition case under the primary edit, first-step KL was approximately 3.043 nats and TV 0.734. These are single-decision distribution measurements, not full-sequence distributions or matched-control evidence.

The frozen scorer has a reproducible near-identity cancellation limitation: on fabricated logits `[0, -100, -1000, ...]` versus `[0, -80, -1000, ...]` it returns a slightly negative KL and understates TV. The actual-row Decimal agreement does not remove that general numerical weakness; the original values and raw logits remain available.

The [independent saved-logit reader](independent-reader.json) retains every target, arm, prefix, strict output and fixed decision. [Process and restoration review](independent-process.json) binds the two phase records, published development handoff and scoped custody. The evaluation recorded 419 calls and 838 passing before/after resource observations; five restoration-probe rows matched both phase reference probes. State/buffer digests and process observations are cooperative records, not independent measurements of all live memory or whole-process-tree custody. The original outer-exit observation and separately created retrospective phase binding are retained distinctly.

The [events](events.jsonl), [results](result.json), [resources](resources.jsonl), [phase](phase.json), logs and [ordered logit parts](logit-parts.json) preserve the full evaluation evidence. Concatenate the 64 parts in index order to recover `logits.f32`, 66,697,728 bytes, SHA-256 `e5165d0d8f2ce4cfd20b86b01cf8a281e24eeb0c015d088f22be7764e97ed4ef`. The index records each part's length, SHA-256, Git blob and path. The independent audit scripts use this host's E: attempt and Git paths and C: tokenizer cache. Replaying elsewhere requires adapting those paths, reconstructing the indexed logits for both phases, and supplying the pinned tokenizer, published development audit and required Git objects.

The [development record](../development/README.md), [fixed protocol](../../../docs/FLAN_T5_TARGETED_WEIGHT_EDIT_V1.md) and [cases](../../../docs/FLAN_T5_WEIGHT_EDIT_CASES_V1.json) precede these forwards. The held-out targets use new wordings of the same lexical substitution; they are not held-out lexical content or pretraining-blind. Public and custodian checks cover named inputs and reported pools, not unknown exposure.
