# Agreement patching: factorial arithmetic

`looking_glass.agreement_metrics.score_factorial` computes a bounded, complete
two-lemma by two-number analysis from supplied clean and patched logits. It
does not run a model, verify tokenization or checkpoint identity, prove that a
patch landed, or authenticate its inputs. The caller must perform the study's
independent capture, landing, cleanup, source and runtime checks and retain
incomplete attempts separately.

For each target, `m = logit(" is") - logit(" are")`, `w = +1` for a
singular target and `-1` for a plural target, and the signed shift is
`u = w * (m(clean) - m(patched))`. It returns every pair's endpoints and shift,
all four cell means per quartet and site, and the equally weighted mean of
quartet contrasts. For `E[lemma relation, number relation]`, the prespecified
number main contrast is

```
D = 0.5 * ((E[same, opposite] - E[same, same])
         + (E[different, opposite] - E[different, same]))
```

The separately reported factorial interaction is the first bracket minus the
second. A positive D can reflect erasing a target preference; it alone does not
show that a donor-number representation transferred. The caller supplies
finite exact binary32 values decoded into Python binary64; the helper checks
that these values are representable and uses binary64 arithmetic and
`math.fsum` for aggregation. Each contrast uses a single four-term `fsum`,
rather than separately rounded within-lemma differences. Signed-zero bit identity and the full-vocabulary
and post-action controls must be checked separately from the numeric scores.

The default requires four complete quartets. Each contains two lemmas, each
once in singular and plural; each target must have every donor from its own
quartet at every declared site. Duplicate, missing, nonfinite, out-of-range,
non-binary32 or cross-quartet entries raise `StudyInvalid`; no denominator is
silently reduced. Incorrect and tied clean preferences stay in the results and
are counted. Selected `is` and `are` logits for a same-target self patch must
also agree with their clean counterparts within the fixed `1e-5` absolute plus
`1e-5` relative tolerance. This partial numeric guard does not replace the
required same-invocation self-patch and full-vocabulary checks.

The synthetic tests use fabricated scores; no model or held-out cases are
executed. The natural-language protocol and its execution runner must be
versioned and reviewed before using these calculations for a study.
