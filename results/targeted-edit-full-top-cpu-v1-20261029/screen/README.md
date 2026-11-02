# Full-vocabulary targeted-edit screen: ineligible

The fixed baseline screen did not select a fact. None of the four facts met the
preregistered requirement that the factual answer be the **unique top token**
on its discovery prompt and both efficacy-calibration prompts, with at least
four of six evaluation phrasings also eligible. The screen is **ineligible**;
this is a feasibility result for these prompts, not a negative result for an
edit. The protocol stops here: no edit, edited evaluation or locality prompt was
forwarded.

| Fact | Discovery factual top | Calibration factual tops | Evaluation factual tops |
|---|---:|---:|---:|
| France | No | 1 / 2 | 2 / 6 |
| Japan | No | 1 / 2 | 2 / 6 |
| Italy | No | 0 / 2 | 0 / 6 |
| United Kingdom | No | 1 / 2 | 2 / 6 |

All 36 original rows had a unique full-vocabulary top; nine had the factual
token as that top. Each of the four discovery prompts had ` Madrid` as its top
and contains a Spain–Madrid example. This association does not establish that
the example caused the prediction. The result concerns one next token under
the fixed public prompts, not a generated answer or a general conclusion about
the model's factual knowledge. The screen was public and baseline-conditioned;
the candidate phrasings are not independent facts or a representative sample.

[All 36 rows](screen-rows.csv) list the prompts, top tokens, candidate log
probabilities, eligibility and exclusions. The [fixed cases](../../../experiments/targeted_edit_full_top_cases_v1.json)
include the 12 locality prompts, which remain unforwarded. The raw
[screen logits](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/targeted-edit-full-top-cpu-v1-20261029/screen/attempt/screen/screen.f32) are 36 ordered little-endian float32 rows of
50,257 vocabulary logits each; the [journal](attempt/screen/observations.jsonl),
[selection](attempt/screen/screen-selection.json), [decoded top tokens](attempt/screen/screen-top-tokens.json),
and [phase record](attempt/phase-record.json) bind the saved rows. The independent
[saved-logit audit](independent-screen-audit.json),
[owner comparison](independent-owner-comparison.json), and
[source/process record audit](independent-record-audit.json) agree with the
ineligible decision. They check saved evidence and cooperative attestations;
they do not authenticate model execution, cleanup or clock ordering.

The [protocol](../../../docs/TARGETED_EDIT_FULL_TOP_CPU_V1.md) and
[scope clarification](../../../docs/TARGETED_EDIT_FULL_TOP_SCOPE_V1.md) were
published before the screen. The [stage-source clarification](../../../docs/TARGETED_EDIT_FULL_TOP_STAGE_SOURCE_V1.md)
explicitly records a prospective deviation: the screening and scientific helpers
were frozen before the screen, while later executors could have been authored
after earlier results. No later executor or edited run was used here. Source
`9d10c8e221a39589173fba9ad1726022e9e19950` and the
[screen manifest](../../../experiments/targeted_edit_full_top_screen_manifest_v1.json)
SHA-256 `458c934bbe74ad01fd0d6abe6d4686aa1e47cbb993f4b76e04d451d07b3e56bb`
were independently read back before launch. The v26 public inventory reserved
all 48 inputs; the currentness check and named custodians reported no new
relevant conflict. Natural-split exclusion relies on the public 128-row support
and the custodian's reported original preparation; no new private split query
was performed. These checks do not establish absence of unknown exposure or
pretraining overlap.

The model-free [reader](audit-screen.py) needs Python, NumPy, `tokenizers`, the
published cases and the pinned `tokenizer.json`; it does not load a model. It
accepts paths via `--help` and writes a new report. The separate
[owner comparator](compare-screen-owner.py) checks that report against the
saved screen files. The [record auditor](audit-screen-record.py) checks the
saved process and binding records with the pinned manifest, cases and tokenizer;
its path arguments and prerequisites are described by `--help`.

For a record-auditor replay, copy the published `attempt/` directory to a
separate temporary directory and create an empty `pycache/` inside that copy;
Git does not retain empty directories. Recreating it cannot independently
verify its original execution-time state. The saved-logit reader runs directly
against the published raw files.
