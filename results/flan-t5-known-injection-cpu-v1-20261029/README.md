# FLAN-T5 known-injection localization

The fixed public study completed 80 teacher-forced CPU forwards on eight prompts, with separate encoder and decoder injections for each prompt. Across the 16 trials, the first changed observed coordinate was the injected site every time. This is an instrumentation check: the injected site is explicitly observed, and the first-change rule is expected to find it. A constant guess given the declared site also scored 16/16 by construction.

| Baseline or check | Encoder | Decoder | Total |
| --- | ---: | ---: | ---: |
| First changed observed coordinate | 8/8 | 8/8 | 16/16 |
| Largest observed L2 displacement | 0/8 | 7/8 | 7/16 |
| Fixed random guess across both selected stacks | 0/8 | 1/8 | 1/16 |
| Fixed random guess within the known stack | 2/8 | 1/8 | 3/16 |
| Known-site constant | 8/8 | 8/8 | 16/16 |

The same-site self controls reproduced the injected traces and logits bitwise, and same-site clean replacements restored the clean traces and full logits bitwise in all 16 trials. The latter immediately undo the intervention by construction. All 16 later-block alternate replacements had nonzero landing displacement. Their output-logit L2 distance from clean was 0.521–1.006 for encoder trials and zero for all eight decoder trials; in the corresponding injected trials it was 1.204–9.304 and 0.301–0.840, respectively. These alternate-site outcomes illustrate that restoring the output need not uniquely identify the original intervention site.

A post-run descriptive replay found no change in the top-vocabulary token across the 64 injection/alternate comparisons at the two teacher-forced decoder positions. The largest clean-to-injected total-variation distance was 0.00371 for encoder injections and 0.0000821 for decoder injections. These are saved-distribution summaries, not generated answers or evidence that the interventions had no effect.

The model has eight blocks in each stack, but this protocol observes only blocks 0–5; its first-change and magnitude rankings are limited to those twelve selected coordinates, with an unobserved encoder-to-decoder gap. The discovery and held-out labels distinguish fixed groups withheld from this study's local development; they do not establish independence from pretraining or other unknown exposures. Encoder token 1 is the last letter of `Archive` in discovery inputs and the ` label` token in held-out inputs. These results do not localize a naturally occurring computation or identify a unique mechanism.

The first source version stopped before forwarding any case because its guard incorrectly required six total blocks per stack. Its direct-process observation and path-free failure record are preserved in the [correction](../../docs/FLAN_T5_KNOWN_INJECTION_ADMISSION_FAILURE_V1.json). The corrected run used the same cases, sites, predictions, and controls. The study protocol and cases are published in [the protocol](../../docs/FLAN_T5_KNOWN_INJECTION_V1.md) and [case inventory](../../experiments/flan_t5_known_injection_cases_v1.json).

`process-observation.json` records a directly waited base interpreter with exit 0, source pins checked before and after, and file hashes read after the wait. The child checked the pinned assets before and after the study. This is cooperative evidence for that direct process, not an independent attestation of all descendant lifetimes, closed handles, or synchronized clocks. `public-currentness.json` states the scoped public-input and sentiment-custody checks. A separate model-free reader replayed the retained journal and all 368 content-addressed tensors; it checks saved-byte consistency, not independent execution. The 9,510,912 bytes of raw tensor evidence include both full teacher-forced vocabulary logits and patch landing witnesses.

To repeat the saved-byte audit, extract `tensors.zip` into this directory, then run `python audit_saved.py --data . --manifest cases.json --output audit-repeated.json`. On the recorded runtime, the independent report reproduced SHA-256 `3e22a3f20c8a79ea0a91bc24dd4aa81e57a8c0fc1d7d818c507458ea3155c428`; a different runtime may change final floating-point bits and the report digest while preserving the validated decisions. The script's SHA-256 is `e5c60b84b20c0e7063ad0db075cc5d3a7ee557a2a899915abd7205a17eff3cd3`. The separate `describe_logits.py` pins the packaged original `independent-audit.json` rather than a locally regenerated report.
