# Per-prompt matched-dose copying-head follow-up

**The fixed criterion was not met.** On eight evaluable public word families,
the head-2 advantage over head 10 had the prespecified positive sign in four
families, below the required six of eight (75%). The mean contrast was
`+0.0346295`. This is a negative result for the fixed descriptive rule, not a
test of statistical significance or proof that either head has no effect.

The [protocol](../../docs/DISTILGPT2_COPY_HEAD_PROMPT_DOSE_V1.md) froze eight
families of four prompts each before this run: a repeated sequence and three
shuffles. For each prompt, the actual removed pre-projection activation L2 was
matched within the fixed relative tolerance of `1e-5` across layer-3 heads 2,
10 and the fixed head-7 active control. Define
`D_h` as the change in target-token log probability for the repeat minus the
mean change for the shuffles. `D_h10 - D_h2 > 0` favors a more negative
repeat-specific change for head 2. At least six evaluable families, a positive
mean contrast and at least 75% positive signs were required.

| Family | D head 2 | D head 10 | D head 7 | Head 10 − head 2 |
| --- | ---: | ---: | ---: | ---: |
| 1 | -0.827708 | -0.366239 | +0.050739 | +0.461469 |
| 2 | -0.281543 | -0.602156 | +0.248903 | -0.320614 |
| 3 | -0.317499 | -0.239231 | -0.059789 | +0.078268 |
| 4 | -0.583578 | -0.902343 | -0.130863 | -0.318765 |
| 5 | -0.448268 | -0.456521 | +0.016017 | -0.008253 |
| 6 | -0.627313 | -0.449704 | +0.064093 | +0.177609 |
| 7 | -0.416424 | -0.688401 | -0.141702 | -0.271977 |
| 8 | -1.197779 | -0.718481 | -0.008964 | +0.479298 |

Across the eight families, mean D was -0.587514 for head 2, -0.552885 for
head 10, and +0.004804 for head 7. Comparisons with head 7 are descriptive;
the primary criterion compares heads 2 and 10. The [family table](family-effects.csv)
retains full precision and target-logit contrasts. The
[prompt/arm table](prompt-effects.csv) has all 96 edited observations, their
common and realized doses, target ranks and logit/log-probability changes,
top-token changes, full-vocabulary KL and total variation, and the norm of
the change at the projection output.

All 32 prompts were dose-eligible. The common dose `q` ranged from 0.620670
to 6.056641; head 2, head 10 and head 7 set the minimum in 14, 14 and 4
prompts respectively. The largest relative error from `q` was `2.79e-8`, and
the largest within-prompt relative spread was `3.20e-8`. Matching these
pre-projection norms does not match the removed directions, fractions, or
downstream effect. Mean projection-output L2 changes were 13.8036, 11.1075
and 13.7202 for heads 2, 10 and 7 respectively. Mean full-vocabulary
`KL(clean || edited)` values were 0.05280, 0.03936 and 0.02542; the unique
top token changed in 6, 5 and 4 of 32 prompts. Head 7 is one fixed active
control, not a distribution of random controls or a match on output effect.

The earlier [zeroing evaluation](../distilgpt2-copy-head-eval-v1-20261028/README.md),
[head-2 versus head-10 zeroing follow-up](../distilgpt2-copy-head-followup-v2-20261028/README.md)
and [family-dose follow-up](../distilgpt2-copy-head-factorial-v1-20261028/README.md)
remain unchanged. Their results informed this design, so this is not a blind
confirmation. The public families were selected without new model outputs and
excluded known public input-prefix conflicts; this does not establish that the
words or patterns were absent from pretraining, nor that the cases represent a
population. Norm matching alone does not establish a copying mechanism or
identify why the head effects differ.

The CPU run recorded 192 returned actions and 51,573,504 tensor bytes. Its
[manifest](run/manifest.json), [journal](run/attempt.jsonl),
[result](run/result.json), [independent replay](audit/independent.json), and
[redacted process projection](process-projection.json) are included. The
independent reader checked every saved tensor segment, delivered bits, doses,
controls, full-vocabulary statistics and fixed decision. Recorded clean,
no-op and clean-after checks, and before/after state and buffer digests,
matched. Direct-process exit, source checks and clock are cooperative local
records, not an independent execution attestation; the projection cannot
authenticate the withheld original path-containing process records.

The 32 raw tensor files and original process records remain in the local
workspace; they are not included in this compact publication. Their relative
names, lengths and SHA-256 digests are in the process projection. Therefore
the independent saved-byte reader can be rerun only where those raw files are
available. The [reader and prospective launch record](../../audits/copy_head_prompt_dose/prelaunch.md)
were published before the run. The original incorrect boundary observation
and its [public correction](../../audits/copy_head_prompt_dose/launch-boundary-correction.md)
are both retained: the matching v34 candidate was checked before launch, while
the official v34 bytes and complete public inventory were verified after it.
None of these checks establishes unknown pretraining or protected-pool scope.

To regenerate the two tables from the published small result, cases and replay:

```text
python -I -B build_tables.py --result run/result.json --cases ../../experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json --audit audit/independent.json --out <new-empty-directory>
```
