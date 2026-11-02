# DistilBERT agreement: continuation and subject-state interventions

**Observed, fixed-case CPU result, 2026-10-29.** In 32 paired sentences,
adding the number-neutral continuation `ready.` after the mask increased
`P(is) + P(are)` in every case: the median rose from **0.4802% to 21.1532%**
and the mean paired increase was **24.05 percentage points**. The binary
`is`/`are` preference agreed with the subject in 28/32 terminal-mask cases
and 29/32 continued cases. This is a within-pair context comparison on one
suffix, two surface frames and four repeatedly used subject lemmas, not 32
independent trials or evidence of general grammatical improvement.

Across each context, replacing the subject representation with a bounded
opposite-number donor direction shifted the binary margin toward that donor
more than both a same-number donor and an equal-input-L2 random control in
32/32 cases. **The opposite-donor intervention reversed the binary preference
in zero cases in either context.** It changed the unique full-vocabulary top
token in one case per context; the controls also produced top-token changes.
These are directional sensitivity results, not evidence that the intervention
reliably changed predicted agreement or located a general mechanism.

## Design and outcome

The [protocol](../../docs/BERT_AGREEMENT_SUFFIX_V1.md) and
[cases](../../experiments/bert_agreement_suffix_cases_v1.json) were frozen
before any forward. The pinned checkpoint is
`distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`. One example is `At the station,
the scout near the visitor [MASK]` versus its exact prefix followed by
`ready.` before `[SEP]`. The suffix includes punctuation and the shifted
separator. BERT can incorporate this right context into the subject state and
all other earlier states. These new pairs use different nouns from the
preceding study, so comparisons with that study cannot isolate the effect of
the suffix.

The zero-indexed layer-3 block-output vector at the subject token was
replaced using same-context donors. For each pair, a common dose was fixed as
`min(2,` all four natural donor distances `)`; it was 2.0 in all 32 cases.
The same-number control used a different lemma in the same number; the random
control used a fixed direction shared across contexts. A separate continued
arm used the terminal-derived opposite-donor direction. Self and clean-after
controls were also retained. This run completed **416 forwards**: 64 clean
captures and 352 actual intervention/control forwards. Seed was 20261029.

| Measure | Terminal mask | With `ready.` |
| --- | ---: | ---: |
| Median full-vocabulary probability of `is` or `are` | 0.4802% | 21.1532% |
| Mean probability of `is` or `are` | 1.3124% | 25.3618% |
| Strict binary subject-correct sign | 28/32 | 29/32 |
| `is` or `are` is unique full-vocabulary top | 0/32 | 10/32 |
| Opposite donor beats both controls on donorward margin | 32/32 | 32/32 |
| Opposite donor flips the binary sign | 0/32 | 0/32 |
| Opposite donor changes unique full-vocabulary top | 1/32 | 1/32 |
| Median total variation, opposite donor vs clean | 0.01964 | 0.03043 |

The clean full-vocabulary top changed between contexts in all 32 pairs;
terminal `.` was top in 21/32 and continued `are` was top in 10/32. The
accuracy transition was 28 correct-to-correct, one wrong-to-correct, three
wrong-to-wrong and zero correct-to-wrong. All four terminal errors and all
three continued errors occurred in the first frame with singular subjects and
plural attractors. The mean probability gain was 40.73 percentage points in
the first frame and 7.37 in the second. The reused frames and lemmas make
these cases dependent; the small change in correct count should not be
interpreted as a population estimate.

For a singular target the donorward effect is minus the change in
`logit(is)-logit(are)`; for a plural target it is plus that change. The mean
opposite-minus-same / opposite-minus-random effects were 0.3066 / 0.3084
logit units for terminal and 0.4727 / 0.4693 for continued. The prespecified
mass and both donor criteria passed; all individual outcomes are in the
replay files. A same-number control changed one continued-context binary sign
and a unique top token in each context; the random control changed two
terminal and one continued top tokens, without binary sign flips.

The descriptive fixed-terminal-direction context effect averaged +0.1524
logit units (25 positive, 7 negative); its difference from the shared-random
context delta averaged +0.1347 (24 positive, 8 negative). Although the
intended vector is shared, the bases and context change, rounding can differ,
and the continued-context vector can be off-manifold. These contrasts do not
isolate downstream gain. The context-specific donor directions can themselves
rotate, and equal vector norms do not imply equal causal potency.

## Evidence, review and limits

The [owner replay](owner-replay.json) and separately authored
[independent replay](independent-replay.json) report every fixed case and
summary. The independent checker verified all 416 saved starts, selected
full-vocabulary logits, donor mapping, subject witnesses, landing doses,
controls, recorded source/asset identities and direct-child receipt; its
common per-case and summary fields agreed with the owner reader. It did not
run the model. The separately recorded
[outer observation](outer-observer-exit.json) and
[direct-child observation](direct-child-exit.json) record exit code zero and
the original closed-directory inventories. Their scope is cooperative direct
process and saved-byte observation, not process-tree attestation or proof of
unsaved positions. The source checks record the intended checkpoint and runtime
identities; they are not an independent attestation of the executed process.
The attempt retained 53,689,696 bytes; the model weak reference was reported
dead on cleanup. CPU was used; these data are not a GPU benchmark.

Before launch, the independently reviewed public comparison covered 1,191
row references from 25 named public or then-pending sources: 76,224
cross-bundle comparisons had no complete or terminal-SEP-stripped token-prefix
collision, and the only within-proposal relations were the 32 intended pairs.
The [published audit](../../audits/bert_agreement_suffix_v1/README.md) pins
those inputs. An additional then-pending signed cue-context bundle with SHA
`843649c3f105e26185d96f31d769b0585d219bb7a45ba445cf2c2548598927df`
had zero collisions against these 64 variants in 2,048 comparisons
([receipt](signed-context-comparison.json)). These are named public-input
checks only; they cannot establish absence from unknown, private or pretraining
exposures. The syntactic core was reused. The result is conditional on this
single continuation and these dependent fixed families.

The frozen run code capped the ordinary evidence journal at 128 MiB, and the
successful attempt stayed below it. An independent synthetic failure test
found that its traceback-writing failure path could exceed the stated total
attempt cap for an exceptionally large exception. This failure did not occur
in the completed attempt. The cap is a disk-retention bound, not a VRAM limit;
successor runs require a bounded failure-message path.

## Reproduce the saved-byte analysis

The raw archive is retained by the operator, not stored in this repository:
`bert-agreement-suffix-cpu-v1-20261029.zip`, **46,172,678 bytes**, SHA-256
`57d60d5f18b72a305d12b08cd6137f76be62cdc2580bf74f4bfe0dcd14b0cf91`.
It contains the original attempt and controller directories, the outer
observation, the source bytes and both replays. Its `manifest.json` hashes
all 31 other members. Verify the ZIP hash and each member before use; retain
the original directory names. Absence of a pending sentinel in a copied
archive alone cannot prove absence in the original directory: the outer
observation records that original-directory check at closure.

With Python 3.11, a NumPy site directory, a Git executable and a checkout
containing commit `7dbf6e9ea498770e389cbf7fbc1eba5bc2303662`, extract the ZIP
and run the following with `ROOT`, `SITE`, `REPO` and `GIT` replaced by
absolute paths. The command is illustrative POSIX shell syntax; use the
platform's equivalent on Windows. It uses no checkpoint or model forward.

```sh
python -I -S -B "$ROOT/review/independent-replay.py" \
  --site "$SITE" --attempt "$ROOT/bert-agreement-suffix-cpu-v1-20261029-01" \
  --cases "$ROOT/source/experiments/bert_agreement_suffix_cases_v1.json" \
  --receipt "$ROOT/bert-agreement-suffix-controller-v1-20261029-01/exit-observation.json" \
  --receipt-sha256 aba62af990b2ff1752de77ae5d6a76779383c331931df50bebd043152921c7fa \
  --complete-sha256 de0d88d3f28463bdbf778cc83f00c3da91eaf1dc7996554f39de7f514c5d6214 \
  --source-commit 7dbf6e9ea498770e389cbf7fbc1eba5bc2303662 \
  --repo "$REPO" --git "$GIT" --output "$ROOT/new-independent-replay.json"
```

A fresh extraction and invocation reproduced independent report SHA-256
`03ead9680e1dc3f9054ee51f81f14aa598cde09e645aa4e1e7cddbae9fdfebaf`.
The independent outer-audit script is retained verbatim for its original
operator paths; the inventory hashes are independently visible in the outer
observation and archive manifest.
