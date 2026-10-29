# France full-top v2: fixed-input and numerical preparation

This branch preserves the [original protocol](TARGETED_EDIT_FRANCE_FULL_TOP_V2.md)
and tokenizer-only inventory, which are superseded by the pre-forward
[v4 protocol](TARGETED_EDIT_FRANCE_FULL_TOP_V4.md). Its L10 was replaced after
an exact public-reservation collision. The revised tokenizer-only
[manifest](../experiments/france_full_top_cases_v3.json) and
[operator observation](../experiments/france_full_top_tokenizer_observation_v3.json)
are recorded separately. This branch does not authorize a pretrained
forward: a source-bound owner, public/pending exposure checks, a bounded
schedule, independent review, capacity checks and an observed exit are still
required. No model outcome is reported here.

The [pre-forward scope amendment](TARGETED_EDIT_FRANCE_FULL_TOP_SCOPE_AMENDMENT_V2.md)
withdraws the earlier protected-check requirement and explicitly leaves
private and unknown exposure disjointness unverified. No protected or
flagged split will be queried, accessed, or checked indirectly.

Each manifest contains 27 study rows and four previously exposed
projection-basis rows. The superseded original
[manifest](../experiments/france_full_top_cases_v2.json) must not be used for
the revised attempt. The three
development rows carry their earlier case IDs. The exposed locality sentinels
L01, L02 and L05 correspond respectively to
`full-top-japan-evaluation-04`, `full-top-uk-evaluation-04` and
`full-top-italy-evaluation-04` in the earlier full-top screen. Both tokenizer
operator observations are cooperative process records, not authenticated
timestamps or proof of private-split disjointness. The tokenizer assembler is
[`scripts/assemble_france_full_top_v3.py`](../scripts/assemble_france_full_top_v3.py);
the earlier assembler is retained beside it.

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

The baseline [journal core](../looking_glass/targeted_edit_france_baseline_v2.py)
records each start, observed model entry and validated full-vocabulary row,
and preserves a bounded returned-invalid or failed record on later errors.
The [single-use call boundary](../looking_glass/targeted_edit_france_baseline_once_v2.py)
fences the model even if the Python return is interrupted. The separate
[stdlib saved-byte auditor](../looking_glass/targeted_edit_france_baseline_audit_v2.py)
reconstructs all baseline tops and gates from the retained binary32 words.
A separate supervised CPU owner and outer observer now bind these components to a recorded source inventory; an independent public-blob check remains necessary. A saved
`completed.json` or `owner-return.json` remains provisional until a direct
child exit and all required source, public-input and cleanup checks have been
independently assessed. An uncertain call is never retried.

The CPU-only synthetic tests can be run from the repository root with
`python -I -B tests/test_france_top_v2.py` and
`python -I -B tests/test_france_top_edits_v2.py` in an environment containing
the declared Torch and NumPy dependencies. The three `test_france_baseline_*`
files exercise the journal, independent auditor and return boundary. They use
small fabricated tensors and never load a checkpoint. The tokenizer observations, scientific
reviews and executable source must be bound independently before a model run.

The [CPU parent and child](../scripts/france_baseline_cpu_owner_v2.py) use a fixed, dedicated study root, a durable one-shot claim, and an independently observed direct-process [start gate and exit](../scripts/observe_france_baseline_v2.py). This root coordinates this study, not unrelated CPU work. The source revision is an operator-reported label until the checked source bytes are compared separately with public Git blobs. A no-model rehearsal and small synthetic failure tests do not establish a pretrained run.

The [public inventory comparison](../experiments/france_full_top_public_exposure_v4.json), [v44 additions](../experiments/france_full_top_public_v44_addendum_v1.json), [active-mask supplement](../experiments/france_full_top_public_mask_addendum_v1.json), [public support comparison](../experiments/france_full_top_public_support_v1.json), and [pending copying comparison](../experiments/france_full_top_public_copy_addendum_v1.json) cover their named public inputs only. Later additions require fresh checks. Unknown, protected and pretraining history remains unverified; no protected or flagged split was queried.

The [31-file source inventory](../experiments/france_full_top_source_inventory_v1.json) and its [observed freeze](../experiments/france_full_top_source_freeze_v1.json) bind the intended files to the preceding public revision. This is a saved source comparison, not proof of execution or a trusted timestamp.

A subsequent [public pending-propagation comparison](../experiments/france_full_top_public_propagation_addendum_v1.json) found no prefix relation in its named bundle. The outer observer now waits for an atomically published operator-start record before it can launch the owner; all such receipts remain cooperative observations.
