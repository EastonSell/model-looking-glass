# Screened projected-edit follow-on: discovery and baseline screen

The supervised discovery-only stage is complete. It used the earlier public discovery prompt and four calibration prompts; none of the 32 new target-screen prompts or 12 locality prompts was forwarded in that stage. The subsequent baseline screen is now complete; edited evaluation remains a separate stage. The discovery measurements are not results on the new cases.

The fixed discovery grid selected the unprojected dose **0.00075**, or 0.075% of the original edited-weight norm. The projected dose is 0.001, or 0.1%. The original discovery London-minus-Paris margin was −1.268082; the projected edit increased it by 1.175011 and the matched unprojected edit increased it by 1.165703. Their gain difference, 0.009308, is below the prespecified tolerance 0.117501. Both edited margins remain negative. The realized matched unprojected update norm was 0.230065 versus 0.306753 for projected. The match is on this one previously exposed discovery prompt and does not establish matched efficacy on the new targets.

| Unprojected planned norm fraction | Discovery margin gain |
| --- | ---: |
| 0.000500 | 0.777115 |
| 0.000625 | 0.971413 |
| **0.000750** | **1.165703** |
| 0.000875 | 1.359993 |
| 0.001000 | 1.554276 |
| 0.001125 | 1.748520 |
| 0.001250 | 1.942780 |

The projection uses all 18 calibration-token activations; its retained rank was 17. The maximum observed calibration-site output shift was approximately 0.0000229 for projected and 0.252833 for the discovery-matched unprojected dose. These same activations define the projection, so this is a construction check, not an independent locality outcome. The equal-norm unprojected, projected-random, opposite, and nuisance controls and all seven dose candidates are retained in the evidence. The projected-random and nuisance controls are single fixed draws.

The [candidate](development/candidate/pre-evaluation-candidate.json) and [phase record](development/phase-record.json) inventory all raw logits, calibration outputs, algebraic effects, factors, gradients, states and attestations. An independent reviewer reconstructed the saved math from the pinned checkpoint bytes and artifacts; the [math audit](audits/development-math.json) and [saved custody audit](audits/development-custody.json) describe their scope. The source and runtime are bound by the [published manifest](../../experiments/targeted_edit_screened_run_v1.json); the [protocol](../../docs/TARGETED_EDIT_SCREENED_MATCHED_V1.md) and [case-scope note](../../docs/TARGETED_EDIT_SCREENED_CASES.md) define the remaining gates and limitations. The recorded child and supervisor identities agree, and the reports state clean exit, reaping, job drain, restoration and model disposal. These are cooperative execution and cleanup attestations, not authenticated proof against concurrent mutation.

The saved v18 public-inventory attestation records a local timestamp of 07:57:32.6912981 UTC. Its named reviewer did not independently attest that exact fractional instant; they report that the check completed no later than their 07:58:06 UTC manifest-review observation, before the recorded development start. The original receipt is retained unchanged. Those reported clock values and Git ancestry do not authenticate real-world chronology. The public comparison and custodian checks cover only known histories and the named protected split; unknown exposure and pretraining remain unknown.

## Baseline screen

The original model was run on all 32 fixed public target prompts. **Twenty-five had a strictly negative London-minus-Paris margin**, and the frozen round-robin rule selected eight: offsets 0, 8, 16, 24, 1, 11, 17 and 26, with two from each stratum. This satisfies the target-count eligibility condition; it is not an edited-evaluation outcome. All 32 margins, eligibility flags, selected IDs and per-row hashes are in the [selection record](screen/screen/screen-selection.json), with the [full-vocabulary baseline rows](screen/screen/screen.f32), [ordered observations](screen/screen/observations.jsonl) and [phase record](screen/phase-record.json). The selected targets are public and baseline-conditioned, not blind or representative of all paraphrases.

On all 32 baselines, neither London nor Paris was the full-vocabulary top token. A later reversal of the London-versus-Paris margin would therefore show a change in their pairwise preference; it would not by itself show that the model's top prediction or a generated answer became London. The 12 locality prompts and the selected targets have not yet been evaluated under edits.

Two independent model-free reviews checked the saved screen: the [selection/math report](audits/screen-math.json), [process-record review](audits/screen-custody.json), and [scalar per-row audit](audits/screen-scalar.json) (with its [post-run source](audits/screen-scalar.py)). These verify saved bytes and reported consistency, not authenticated execution or cleanup. The screen's [manifest](../../experiments/targeted_edit_screened_screen_run_v1.json) binds the published discovery record and case pool. The saved v20 inventory and named custodian reports were checked before the recorded start; v21 added only separately checked planned inputs. GitHub reports that v21 merged at 08:20:56 UTC, after the screen's recorded 08:20:36 UTC start, but these clocks are not independently authenticated or synchronized. The inventory and private-split checks do not establish disjointness from unknown pretraining or exposure.
