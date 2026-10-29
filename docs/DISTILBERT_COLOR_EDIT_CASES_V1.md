# DistilBERT masked-color cases: planned public inputs

These 37 inputs are a **planned public reservation**, fixed before this study's
baseline screen. They are not evidence that any of these rows were forwarded.
The [prospective protocol](DISTILBERT_COLOR_EDIT_TOP_V1.md) was public and
independently reviewed before these tokenizer-only cases were constructed.
It fixes candidate order, eligibility, edit and controls, stopping rules and
interpretation. The original case file has SHA-256
`4f00000cf205137347d551fc2837e082ac632a515e92c1e5893b3312f1f71623`.

The three usual-color relations are conventional readings, not assertions about
every cultivar or ripeness. The four evaluation phrasings per object are public
within-object rewordings and share vocabulary and structure with the screen;
they are not independent objects or a private test set. Locality accepted sets
are fixed probes, not exhaustive lists of plausible answers. The banana
plant locality accepts `plant` only. An independent tokenizer-only review
checked all 37 rows and 80 complete single-token fills without a model forward.

The saved public-input comparison has SHA-256
`20cfcbe25d911a3392ab4fdba04228fd41eaac0efe624332003096ddc0cbbf65`.
It compares each proposed row against 396 references across nine named
public inventories, and against the other proposed rows, both as full inputs
and with exactly one final `[SEP]` removed. There were no exact or
bidirectional complete-row-prefix relations among 14,652 cross-inventory pairs
or 666 within-proposal pairs. Some historical references recur across source
inventories. The separate cue-frame proposal is public, but its presence here
does not assert that it has been forwarded.

This is a point-in-time check of the cited public inputs, not evidence about
unknown exposures, model pretraining or a protected split. The taste owner
reported holding no protected DistilBERT pool, and the negation/cue-class owner
reported holding no protected inputs in those lanes. These are scoped owner
reports, not a complete model-wide history. Recheck newly published same-model
inputs and applicable custody immediately before any forward. No case is
replaced or reselected based on a model outcome.

The builder uses `transformers==4.57.3` and `tokenizers==0.22.2` with the
five exact offline files and hashes recorded in the case JSON. From the
repository root, supply the pinned snapshot directory and a new output path:

```text
python scripts/build_distilbert_color_edit_cases_v1.py --snapshot <snapshot> --protocol docs/DISTILBERT_COLOR_EDIT_TOP_V1.md --output <new-case-file>
python scripts/check_distilbert_color_edit_exposures_v1.py --repository . --cases experiments/bert_color_edit_public_cases_v1.json --from-git --output <new-check-file>
```

The comparator reads the eight inventories at commit
`2090fe8d1158dc7c9bb309563d76a5d842bc8d8b` and the cue-frame file at public
PR #541 head `b4b2d1d01c923cd37bce537973047261c585cbe6` directly from Git
objects. Fetch those commits first if the checkout lacks them. SHA checks fail
closed, and both output paths must not already exist. Byte-pinned case and
comparison files are marked `-text` to preserve their hashes across checkout
line-ending settings. Neither command loads or runs a pretrained model.
