# Synthetic causal confirmation v2: 2026-10-29 UTC

The one-shot CPU study completed on source revision
`4e657aeb92c2b69edf8e6e5c70a9b6c7e4a6507f`. The owner verified the saved
terminal counts and artifact inventories: 57,408 submissions and 57,408
root forwards, with the child job drained. This is a synthetic instrumentation
experiment, not evidence about a pretrained model or an explanation of its
behavior. The saved report labels exposure **unknown** and its interpretation
**unconfirmed**.

Six fixed `TinyResidual(width=8, depth=4)` initializations were crossed with
nine injected block/operation configurations apiece. Each cell used 16
discovery and 16 held-out inputs at two strengths. The discovery policy saw
four complete block-output traces and three candidate block-output patches;
it was not supplied the hidden injection coordinate. A first changed block is
an observed boundary, not recovery of the exact within-block operation. This
feedforward architecture structurally favors first-difference localization
when the injection survives to the next block boundary.

The following values are from the saved scorer output. A separate,
independently reviewed saved-data auditor returned `saved_data_consistent`
after checking all 54 discovery and 54 held-out chunks, reconstructing the
cell selections and paired metrics, and checking the retained intervention
and negative controls. It did not rerun the model or prove execution,
chronology, secrecy, case generation or exclusion enforcement.

| Model seed | First-changed block hits | Magnitude block hits | Polarity predictor MAE | Constant predictor MAE | Zero predictor MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 31201 | 9/9 | 3/9 | 0.007069 | 0.098856 | 0.098856 |
| 31202 | 9/9 | 8/9 | 0.002037 | 0.060627 | 0.060627 |
| 31203 | 9/9 | 4/9 | 0.007361 | 0.103190 | 0.103190 |
| 31204 | 9/9 | 5/9 | 0.003950 | 0.061793 | 0.061793 |
| 31205 | 9/9 | 8/9 | 0.003329 | 0.068885 | 0.068885 |
| 31206 | 9/9 | 5/9 | 0.004008 | 0.066244 | 0.066244 |
| Totals / equal-model mean | 54/54 | 33/54 | 0.004625 | 0.076599 | 0.076599 |

Both policies selected in all 54 cells. A uniform distribution across the
three candidate sites has a 1/3 expected block-hit rate; this is an analytic
budget-matched comparison, not a sampled count of random successes. The MAEs
compare the polarity-conditioned and constant predictors on the **same
selected sites**, with the zero predictor as an additional same-site baseline.
The two selection policies have identical saved predictor MAEs despite their
different hit rates: magnitude never chose a block upstream of the injected
block, and the same-input patch at or after that block has the same rescue
effect in this architecture. Polarity was a known input property used by the
injection, not a feature discovered by the model. Comparing errors across
different selected sites would also change the effects being predicted.

The analytically averaged random-site baseline has a **lower**
polarity-conditioned MAE, 0.003693, than the 0.004625 at the selected sites.
Its own same-site constant and zero baseline MAEs both round to 0.053054. The
random-site figure averages outcomes at all three sites, including sites with
different effect magnitudes; it is therefore not a same-outcome comparison
showing that random selection predicts the selected sites better. No result
here identifies a learned model feature: both the feedforward step map and
the injected input polarity give the experiment substantial structure.

The saved report records zero observation misses in 1,728 discovery and
1,728 held-out injected comparisons, and zero false detections among the 96
unique clean/repeat comparisons in each split. The aggregate no-injection
first-changed and magnitude policies both abstained. The 3,456 held-out
patches at or after the injection block had zero scalar repair residual, while
the 1,728 upstream site effects were zero. The first-changed site's
polarity-conditioned prediction had the correct sign in all 1,728 eligible
held-out conditions. These controls follow the known step map and the
injected input polarity. A later block can have the same rescue effect, so
rescue does not uniquely locate an injected operation or establish a learned
mechanism. The fixed seeds and synthetic inputs do not define a sampled model
population; no significance or population confidence interval is claimed.

## Evidence and chronology

[`public-evidence.zip`](https://github.com/EastonSell/model-looking-glass/blob/3cbdec78134855532811a58a64e8dd5d4d457185/results/synthetic-causal-v2-20261029/public-evidence.zip) contains the nine public artifacts,
public registry manifest and outcome, and registry metadata as their exact
saved bytes, with `public/` as the extraction root. Its SHA-256 is
`f80b01e1d3e0ffea13e454c1202814e1015ea2d4d6c21a3a6b3eb7736ba7ef04`.
The full 111-artifact private registry remains on the originating machine at
`work/study-results/synthetic-v2-20261029-4e657aeb/private/`; it is not
included in this public archive. The public outcome binds its private outcome
hash. The saved public results and reveal are both included in the archive.
The public pre-manifest and inventory artifacts record source, runtime,
settings and model-state hashes; these records do not independently prove the
executed interpreter or native-library bytes. The owner and rehearsal JSON
receipts were emitted to the Codex command transcript and transcribed into the
accompanying receipt, rather than independently captured as files.

The source-bound custodian is designed to save and re-read the discovery
freeze and concrete prediction commitment before it calls any held-out
forward. The independent filesystem polling in this attempt saw 46 discovery
chunks, no held-out chunks and no freeze/commitment, then saw both files with
three held-out chunks present. It did **not** witness the transition or capture
pre-held-out file hashes. The current hashes and registry bindings check the
saved bytes; they are not an independently authenticated timestamp or proof
of pre-held-out contents. The attempt has no external trusted timestamp, and
the exposure classifier did not establish unexposed custody. The accompanying
[`observer-receipt.json`](observer-receipt.json) records the scope and the
post-run pins, without claiming a stronger chronology.

The separately produced
[`independent-semantic-audit.json`](independent-semantic-audit.json) has
SHA-256 `ce2c7e2a5d147281c0ad0b64ff86183db3dcb65eebf8c86eb7aee31bff33137f`
and was generated by reviewed auditor source revision
`95af78d6430858578baf2e2ad7112bab0d43712b` in a local checkout. The
byte-identical auditor source was published at
`5d3fce399d9a2c224b83b9cfdddfc673eddc0635` with SHA-256
`b94b5430173720c3741f950ee07fbc8330bc9159834c0ccd1453d298aa58e63a`.
Its positive result is a check of saved-data consistency, not an independent
observation of the experiment's execution or the timing of its commitments.

A second reviewer independently ran the **same** auditor on the completed
registry and obtained the same semantic report. The reviewer also compared
the 187 recorded study source hashes with the frozen checkout and Git tree.
[`independent-replay-check.json`](independent-replay-check.json) preserves
that review's scoped record, SHA-256
`c03ecdba2fa9048d79a8a580ce40e9b2aaf5653b008f8a44649bfd64e3d4d9f1`.
This is a repeated saved-data check, not a second implementation or an
independent operating-system record of the owner's exit or job drain.

The published archive can be extracted with `python -m zipfile -e
public-evidence.zip <directory>`. The saved reveal contains the fixed cases
and schedule after completion; creating a fresh attempt would draw new private
roots and is not a replay of this one. To independently check the complete
study, retain both the public archive and the original private registry, then
use the separately reviewed saved-data semantic auditor. The source owner
receipt verifies terminals, counts and inventory; the separate auditor checks
the saved scientific evidence without rerunning model forwards.
