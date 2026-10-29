# France full-top v2: fixed-input and numerical preparation

This branch preserves the [original protocol](TARGETED_EDIT_FRANCE_FULL_TOP_V2.md)
and tokenizer-only inventory, which are superseded by the pre-forward
[v4 protocol](TARGETED_EDIT_FRANCE_FULL_TOP_V4.md). Its L10 was replaced after
an exact public-reservation collision; the replacement's tokenizer inventory
and reviews are being prepared. This branch does not authorize a pretrained
forward: a source-bound owner, public/pending exposure checks, a bounded
schedule, independent review, capacity checks and an observed exit are still
required. No model outcome is reported here.

The [pre-forward scope amendment](TARGETED_EDIT_FRANCE_FULL_TOP_SCOPE_AMENDMENT_V2.md)
withdraws the earlier protected-check requirement and explicitly leaves
private and unknown exposure disjointness unverified. No protected or
flagged split will be queried, accessed, or checked indirectly.

The superseded original [manifest](../experiments/france_full_top_cases_v2.json)
contains 27 study rows and four previously exposed projection-basis rows. Its
L10 must not be used for the revised attempt. The three
development rows carry their earlier case IDs. The exposed locality sentinels
L01, L02 and L05 correspond respectively to
`full-top-japan-evaluation-04`, `full-top-uk-evaluation-04` and
`full-top-italy-evaluation-04` in the earlier full-top screen. The tokenizer
[operator observation](../experiments/france_full_top_tokenizer_observation_v2.json)
is a cooperative process record, not an authenticated timestamp or proof
of private-split disjointness. The tokenizer assembler is
[`scripts/assemble_france_full_top_v2.py`](../scripts/assemble_france_full_top_v2.py).

The model-free [scorer](../looking_glass/targeted_edit_france_top_v2.py)
calculates the fixed baselines, full-vocabulary paired measurements and
provisional decision. The CPU [edit helper](../looking_glass/targeted_edit_france_top_edits_v2.py)
uses the inherited screened SVD/seeded-control construction and the new dose
grids. When `prepare()` is called, it also constructs and validates all five
inherited 0.001-dose controls once as a no-model cross-check, regardless of
the later chosen dose. A failure in this cross-check must be retained as an
incomplete attempt. This extra construction does not select a dose or run a
model. The inherited numerical dependencies are pinned by the future owner;
neither helper proves application, state restoration or process cleanup.

The CPU-only synthetic tests can be run from the repository root with
`python -I -B tests/test_france_top_v2.py` and
`python -I -B tests/test_france_top_edits_v2.py` in an environment containing
the declared Torch and NumPy dependencies. They use small fabricated tensors
and never load a checkpoint. The original tokenizer observations, scientific
reviews and executable source must be bound independently before a model run.
