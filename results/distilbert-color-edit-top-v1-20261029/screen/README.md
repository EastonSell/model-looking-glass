# DistilBERT color-edit baseline screen: ineligible

The fixed nine-row CPU screen completed, and none of the three candidate
objects met the predeclared requirement that the usual color be the unique
full-vocabulary top token on discovery and both calibration rows. Every one
of the nine individual rows failed that requirement. The study therefore
stopped at **baseline-ineligible**. No weight edit, bias control, held-out
rephrasing or locality evaluation appears in the completed recorded attempt;
the operator reports that none was run. This is not a negative result for an
attempted edit.

The columns below are single masked-token predictions from the pinned
`distilbert/distilbert-base-uncased` checkpoint, not generated answers. Rank is
one plus the number of vocabulary logits strictly greater than the factual
color logit; all nine top tokens were unique. Probabilities are rounded to
six decimal places.

| Object | Fixed screen row | Factual color | Color rank | Color probability | Top token | Top probability |
| --- | --- | --- | ---: | ---: | --- | ---: |
| banana | discovery | yellow | 2 | 0.095918 | orange | 0.130597 |
| banana | calibration 1 | yellow | 180 | 0.000259 | eaten | 0.365028 |
| banana | calibration 2 | yellow | 17 | 0.012210 | smooth | 0.195045 |
| strawberry | discovery | red | 13 | 0.020299 | orange | 0.091537 |
| strawberry | calibration 1 | red | 133 | 0.000711 | eaten | 0.125575 |
| strawberry | calibration 2 | red | 23 | 0.007938 | smooth | 0.240639 |
| lemon | discovery | yellow | 2 | 0.098252 | orange | 0.122028 |
| lemon | calibration 1 | yellow | 110 | 0.000686 | eaten | 0.219835 |
| lemon | calibration 2 | yellow | 11 | 0.019967 | smooth | 0.225171 |

The same top recurs across the three objects within each frame. The two
calibration frames, “A ripe [object] is typically [MASK]” and “The surface
of a ripe [object] is [MASK],” also admit non-color completions. This is a
descriptive prompt-format limitation; it does not establish that the model
ignores the objects or lacks color knowledge. The candidate rows were public
and the results are conditioned on this fixed small screen; no pretraining
novelty or broad capability conclusion follows.

The [frozen protocol](../../../docs/DISTILBERT_COLOR_EDIT_TOP_V1.md),
[case manifest](../../../experiments/bert_color_edit_public_cases_v1.json),
and [owner](../../../docs/DISTILBERT_COLOR_EDIT_OWNER_V1.md) were published
before the screen. The reviewed owner source commit is
`707e5806ef9a77137a8b03b80a5b7ce716579ca0`; its merge is
`036fe2e1fe916631b0ef7b923c6ba758f8de6de0`. The screen contains nine starts
and nine completed full 30,522-logit f32 rows. The other 28 fixed input rows
do not appear as forwards in this completed attempt; the operator reports
that they were not forwarded.

## Saved evidence and scope

- [Archive index](index.json), [compressed stage evidence](part-0000.gzchunk),
  and the indexed [run record](run.json) preserve the exact cooperative stage
  files. The journal SHA-256 is
  `3e55c8eebf990e930f73f902a635d7866f0a7f314b30dc62f41e43277b97fb83`;
  the result SHA-256 is
  `1852609621972c1434c7d5ccf4e8935828e3e55149ba2ef5f0185fc28b0b9742`.
- The independent [full-logit replay](independent/color-screen-independent-review-707e5806.json),
  [owner-summary comparison](independent/color-screen-summary-review-707e5806.json),
  [receipt check](independent/color-screen-receipt-review-707e5806.json), and
  [token readout](independent/color-screen-readout-707e5806.json) agree with
  the nine-row ineligible result. A separate reviewer also reproduced the
  ranks and top IDs from all nine saved vectors. These checks establish
  consistency of saved bytes and reported process observations; they are not
  independent attestation of model execution or of unrecorded exposures.
- The first [prelaunch observation](preclaim/prelaunch.json) and its
  [supervisor failure record](preclaim/observer-failure.json) are retained.
  That supervisor exited 1; its original stderr was not retained. The
  operator observed the run directory absent before the successful launch.
  A separate re-execution of the same preclaim command with the then-stale
  main observation failed with “official main moved after the prelaunch
  observation” and left the run directory absent. This reproduction is
  consistent with a preclaim currentness failure, not proof of the original
  exit's cause or an independently authenticated absence of execution.
  The next, fresh prelaunch and the closed successful screen are in the
  archive. The command files containing local paths remain private; their
  digests are in the cooperative observations.

The successful run's retained original files and duplicated prelaunch/observer
metadata total 1,489,972 bytes. The public index and run sidecar add 1,926
bytes, the earlier failed preclaim's original prelaunch and observer records
add 3,812, and the five independent JSON reports add 17,718: 1,513,428 bytes
for those items, before this note and the exposure inventory. This remains
well below the 128 MiB limit; compressed public copies are not counted twice.

The independent reader is included for a no-model replay. From the repository
root, the following reads only the named journal member of the archive, then
recomputes full-vocabulary ranks and the fixed selection:

```text
python -c "import pathlib,tarfile; p=pathlib.Path('results/distilbert-color-edit-top-v1-20261029/screen/part-0000.gzchunk'); pathlib.Path('work').mkdir(exist_ok=True); pathlib.Path('work/color-screen-events.jsonl').open('xb').write(tarfile.open(p,'r:gz').extractfile('events.jsonl').read())"
python results/distilbert-color-edit-top-v1-20261029/screen/independent/review_distilbert_color_screen.py --manifest experiments/bert_color_edit_public_cases_v1.json --journal work/color-screen-events.jsonl --output work/color-screen-replay.json
```

The reader checks the fixed manifest hash, each full f32 row hash and all
nine ordered input bindings. Compare its journal SHA-256 with the pin above;
the archive index separately pins the compressed part and stage files.
