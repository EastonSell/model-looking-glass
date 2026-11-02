# Two-dose cue study: separate continuation

The four-cell CPU continuation completed, and each cell remained below the
fixed matched-control specificity cutoff at both requested patch norms. A
post-stop calculation combining these cells with the two completed cells from
the earlier attempt finds **0 of 3** passing triples per cue set at either
dose. The earlier one-process attempt remains **incomplete**: this combined
calculation uses two processes and was fixed after its partial outcomes were
observed. It is descriptive, not the originally planned one-shot result or a
blinded confirmation.

The fixed criterion uses `L = logit(any) - logit(some)`. For each recipient,
it subtracts the absolute effect of a same-class donor from the weaker of two
signed opposite-class donor effects at the block-2 cue site. `M` is the
smallest of the four recipient margins, and a cell passes only if `M > 0.05`
and its baseline and dose screens are eligible. All six combined evaluation
cells passed the baseline and both dose screens. All requested patches were
non-amplifying. The complete per-recipient/donor effects and margins are in
[result.json](result.json), [continuation-full-audit.json](continuation-full-audit.json)
and [original-prefix-audit.json](original-prefix-audit.json).

| Triple | Cue set | Evidence origin | M at 0.5 | M at 0.7 | Limiting target |
| --- | --- | --- | ---: | ---: | --- |
| chef / prepared / meals | prior | original completed case | 0.03306103 | 0.04924583 | `never` |
| chef / prepared / meals | new | original completed case | 0.01302242 | 0.01843452 | `not` |
| farmer / harvested / crops | prior | continuation; previously partially exposed | 0.02177715 | 0.03286648 | `rarely` |
| farmer / harvested / crops | new | continuation; not recorded in original prefix | 0.00896454 | 0.01333809 | `not` |
| pilot / inspected / instruments | prior | continuation; not recorded in original prefix | 0.02702141 | 0.03958416 | `rarely` |
| pilot / inspected / instruments | new | continuation; not recorded in original prefix | 0.01719666 | 0.02474308 | `not` |

All six pairs are fail→fail under the strict cutoff. The mean `M(0.7)-M(0.5)`
is +0.01327896 for the prior cues and +0.00577736 for the new cues, each over
three eligible pairs. Excluding the partially exposed farmer/prior cell leaves
two eligible prior-cue pairs, 0 of 2 passing at either dose, with mean change
+0.01437378. These are changes in the *minimum* margin; the limiting target or
donor can change, and same-class controls can change as well. A positive
change in `M` is not automatically growth of a particular opposite-class
transfer. The machine-readable reports retain full precision and all
eligibility, signed control and per-target values.

The prior cues are `never`, `rarely`, `always`, `often`; the new cues are
`not`, `hardly`, `really`, `certainly`. The cues differ in lexical,
frequency, temporal and intensity meaning, and the same-class donors are
active comparators rather than inert semantic nulls. The 0.7 norm is below
the earlier studies' smallest observed radius of about 0.7678. Two doses and
three public, nonblind lexical triples do not establish a monotonic or general
dose-response curve, isolate syntax or a negation mechanism, or establish
absence from pretraining. The `any`−`some` score is not an acceptability
judgment or proof of the model's full-vocabulary top prediction.

The [original attempt](https://github.com/EastonSell/model-looking-glass/blob/9c24d512ac1e0ff42c40f93c3f20a95b040bcdde/results/bert-cue-dose-response-partial-cpu-v1-20261029/README.md)
stopped at its disk reserve after 499 saved forward witnesses. Its farmer
prior cell had saved 19 forward records plus baseline and distance records;
the operator reports that the next transfer returned but was not retained.
The continuation repeated that
partially exposed cell, then ran farmer/new and both pilot cells in the fixed
order. Those latter three cells were not recorded in the original prefix;
that statement is not independent proof of all prior exposure. The exact
21-record comparison passed byte-for-byte, including full logits, all fields
and witness bits. This allows the separately labeled pooling calculation;
it does not turn the interrupted attempt into a completed one.

The continuation used 480 forwards and saved 488 events: 16 clean, four
baseline, four radius, 64 self, 384 transfer and 16 repeat. Independent
full-witness and selected-score readers agree on all four cells. Recorded
actual norms ranged 0.49999995297–0.50000004905 at the low dose and
0.69999995511–0.70000004577 at the high dose; the largest relative error was
below `9.811e-8`. Object and mask controls were active: across both doses their
raw `L` shifts ranged −0.053181 to +0.054409 and −0.119316 to +0.183979,
respectively. All final-block cue controls preserved the full mask logits.
Clean and repeat runs used passive capture hooks. Equal input norms do not
equalize functional impact.

The operator waited for the direct CPU process and observed exit 0. The
saved continuation result records matching before/after model-state digests,
the fixed runtime and CUDA uninitialized at its final check. The no-forward
preflight checked the original evidence hashes, 20 bound source paths, cached
assets, recorded dependency/CPU settings, all 32 tokenizer rows and both
volume reserves without importing Torch. These are observations and
saved-byte consistency checks under trusted, quiescent local source and
dependencies, not an independent execution attestation.

Execution is bound to public source
[`cdc9630e24542fed84b0c9556bb7f37c60d722d1`](https://github.com/EastonSell/model-looking-glass/tree/cdc9630e24542fed84b0c9556bb7f37c60d722d1),
not the later [merge](https://github.com/EastonSell/model-looking-glass/commit/a18f6a6503b14f2df40c4614f2cc674a882c6bf0),
which retained a different shared intervention helper. The operator and
reviewer report that the full-witness reader was published and independently
reviewed before the continuation's first forward. The reviewer reports that
the pooling calculator was published after process exit and before they
inspected continuation outcomes. The operator reports that the
[continuation protocol](BERT_CUE_DOSE_CONTINUATION_V1.md), which fixed the
pooling rule, was public before the run.

The full 94,535,539-byte compressed continuation journal is retained on the
local E: scratch volume; it is not in this repository copy and has no public
download location. A compact, operator-derived target-logit subset supports a
selected-score replay, but cannot authenticate its derivation or verify full
logits, dose, landing or execution. The [inventory](inventory.json) binds the
files in this compact package; hashes of unavailable evidence are commitments,
not public copies of the withheld records.

From the compact package, the standard-library selected-score replay writes a
new file whose bytes match [subset-audit.json](subset-audit.json):

```text
python -I -B audit_subset.py manifest.json score-subset.jsonl.gz --expected-subset-sha256 342f3750e10509d7baebc314d4351dbd56ee1302a82caf125fd7e8a62380c3f9 --output replay.json
```

The descriptive cross-process calculation can be reproduced from its bundled
reports and receipt:

```text
python -I -B summarize.py --prefix-report original-prefix-audit.json --continuation-report continuation-full-audit.json --continuation-report-sha256 eb22a7db644c10664915de1d9c5493c7b90e4532fdcebb8a787aee9034f40abf --comparison prefix-comparison.json --comparison-sha256 7b19778100d02eadba9ba225504e8575fa0842df471ca53af323320ab70a9db7 --receipt controller-exit.json --receipt-sha256 a932efaf2be7b8a8cf72456c73e9bbcb8dd4623e1fa370ef6fe56384dd38f042 --continuation-events-sha256 67c022d36d7427840334390602706e286aab60168c50c42dad72b7e711b2c87f --continuation-result-sha256 cce115669ca1a957428d000950aa36e04c8087a0a96a87dc78492cdff59662d5
```

With the locally retained full journal, replace `PATH_TO_JOURNAL` below to
rerun the full-witness reader and compare with
[continuation-full-audit.json](continuation-full-audit.json):

```text
python -I -B audit_cont_full.py manifest.json PATH_TO_JOURNAL result.json --started started.json --expected-started-sha256 c7d35e9c1113eef722a7a6654965cfcda0fcde801314513b22270a5011e22e2c --expected-result-sha256 cce115669ca1a957428d000950aa36e04c8087a0a96a87dc78492cdff59662d5 --expected-source cdc9630e24542fed84b0c9556bb7f37c60d722d1
```

[derive_subset.py](derive_subset.py) requires the retained continuation
journal and its manifest, started, result and exit records.
[compare_prefix.py](compare_prefix.py) requires both full
journals; neither operation can be reconstructed from the compact subset
alone.
