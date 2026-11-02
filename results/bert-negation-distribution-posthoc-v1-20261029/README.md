# Post-hoc full-vocabulary replay of negation-cue transfers

This exploratory, model-free replay asks whether the saved cue interventions
change only the original `any`/`some` contrast or also redistribute probability
elsewhere. It was designed after the results of the [original study](../bert-negation-cue-cpu-v1-20261029/README.md)
were known; it is not a preregistered result or an independent execution.

The original study used one fixed template, two development contexts and six
evaluation contexts. Each evaluation context supplies two negative cues
(`not`, `never`), so the 12 cue/context comparisons below are not 12
independent contexts. All eight contexts, all four saved sites, both transfer
directions and all eight directed actions are retained in `analysis.json`.
For the `cue2` summary below, `q -> just` and `also -> just` share the same
clean `just` recipient. The one `also -> just` action in each context is
reused for the two negative cues.

| Evaluation `cue2` measurement | `q -> just` | `also -> just` |
| --- | ---: | ---: |
| Median change in `logit(any) - logit(some)` | +0.4523 | -0.0411 |
| Median full-vocabulary total variation | 0.01746 | 0.01902 |
| Median conditional TV after excluding `any` and `some` | 0.01582 | 0.01640 |

The negative-to-`just` margin change is positive and exceeds the absolute
active-control change in 12/12 comparisons. Its full-vocabulary TV exceeds the
control in 5/12, while its conditional TV exceeds the control in 7/12. These
paired counts and separately computed medians describe the fixed saved cases;
they are not significance tests. Conditional TV uses independently normalized
probabilities over the vocabulary after excluding the two scored tokens, so a
positive value establishes a relative distribution change among the remaining
vocabulary tokens, not merely a normalization shift. It does not identify a cause.

For `q -> just`, the median fraction of absolute probability movement on
`any` and `some` is 0.122 (range 0.011–0.303). The remainder is distributed
across the other tokens. This is a descriptive accounting of absolute changes,
not the proportion of a causal effect. The cosine between the full probability
displacements for the primary and active-control transfers has median -0.545
and ranges from -0.713 to +0.274; its sign is not uniform and does not establish
a shared or distinct mechanism.

The object and mask sites are active comparators. For evaluation `q -> just`,
their median full TVs are 0.01075 and 0.02062 respectively, versus 0.01188 and
0.02843 for the corresponding `also` controls. All final `cue5` transfer
logits equal the corresponding clean logits, yielding zero TV there.

In the reverse same-recipient comparison, `just -> q` and `also -> q` have
median full TV 0.01098 and 0.01157 respectively; the absolute control margin
change is at least the primary's in 11/12 comparisons. Both `just` and `also`
are affirmative donors into a negative recipient. This is evidence about
nonuniqueness to `just`, not a polarity-preserving null or a test that rules
out a polarity-class effect.

One concrete example is `negation-08` with a `just` recipient. Its clean
top token is `some` (probability 0.0801), with `all` second (0.0725). Both
`not -> just` and `never -> just` change the top token to `all`; for
`not -> just`, `all` is 0.0850 and `some` is 0.0766. The `also -> just`
control leaves `some` first (0.0919). The `any` rank changes from 55 to 38
under either negative donor and to 54 under `also`. These are two correlated
variants of one context and a near top-rank swap, not two independent examples
or a grammaticality judgment.

The cue words differ in lexical meaning; contexts and sites vary across the
study, although each paired comparison uses the same target and site.
`any` also has free-choice uses. Equal input L2 does not establish equal
functional dose. The fixed public examples are not known to be absent from
pretraining or other exposure. The findings do not establish a unique circuit
or general negation capability. A useful prospective follow-up would test
within-class donor controls and fixed full-vocabulary outcomes across new
frames and doses; its design and results must be reported separately.

## Data and verification

`analysis.json` is the full per-case, per-site report and index (SHA-256
`dec7f4871b41b115f8e3e3d65d395b1875e25d56c29a55add8e58e3ca391704c`).
The six binary parts in `parts.json` reconstruct a lossless gzip of the
selected little-endian float32 mask-logit vectors. It has 288 logical event
references and 224 case-local physically stored vectors, 24,470,665 compressed
bytes, and SHA-256 `4308884eefd1bd4f69da55d4e5f6e58f4df56381401b2a39cd7e02ea302fdc61`.
Each reference has its original event index and byte hash in `analysis.json`.

The original complete 456-record gzip journal remains locally retained; it is
not in this public package. Its SHA-256 is
`a67b5ea0ab7d4f7078bab57ad315f9bcf90302be674e8723c05c0785a011ccd1`.
An independent reviewer streamed it and compared all 288 selected vectors
byte for byte, as well as independently recomputing the reported metrics; the
review script and digest-only result are included here. That is a saved-byte
and numerical consistency check. Subset hashes by themselves do not authenticate
the unshared journal, and neither the subset nor process records independently
attest execution. The separate original full-witness audit covers the saved
activation/landing checks; this analysis does not repeat them.

The analysis source is `looking_glass/bert_negation_distribution_posthoc.py`
(SHA-256 `c510defe6da8b6a733c227f4dc90039f1864833a5b535dc46f0fd5540d575493`).
The public manifest SHA-256 is `4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff`,
the original compact result SHA-256 is `20bcd1777e343a268e73042e664bfeb0603a01791ae0b16d908f5d5ea6d08f9b`,
and the pinned `distilbert-base-uncased` vocabulary SHA-256 is
`07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`.
The source reads one case at a time, limits line and record sizes, and requires
binary64 subnormal handling; these logical limits are not a measured RSS bound.
No model, tokenizer or GPU is loaded. Details and formulas are in
`docs/BERT_NEGATION_DISTRIBUTION_POSTHOC_V1.md`.

From the repository root, rebuild and replay the public subset:

```text
python results/bert-negation-distribution-posthoc-v1-20261029/rebuild_subset.py --output <new-subset.f32.gz>
python -m looking_glass.bert_negation_distribution_posthoc subset --subset <new-subset.f32.gz> --index results/bert-negation-distribution-posthoc-v1-20261029/analysis.json --manifest experiments/bert_negation_public_cases_v1.json --result results/bert-negation-cue-cpu-v1-20261029/result.json --vocab <pinned-vocab.txt>
```

The rebuild command checks each part and the closed output digest and refuses
to overwrite an existing output. If rebuilding fails, the newly created output
may be incomplete. If the original journal is also available, the
independent `review_negation_distribution_independent.py` can separately
recheck source-to-subset bytes and all metrics; its `--help` lists arguments.
The independent test fixture is alongside it. All analyses are CPU-only.
