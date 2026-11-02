# Subject-state agreement interventions in two pretrained models

The fixed opposite-number donor moved the `is`/`are` logit preference toward
the donor more than both active controls in **32/32 cases for each model**.
This is a within-design directional comparison, not 32 independent trials or
a demonstration that either model changed its top prediction. The cases are
public, nonblind and reuse four subject lemmas across two frames.

The output limitation is substantial. In DistilBERT, the unique
full-vocabulary top prediction was **`.` in all 32 clean cases and all 32
opposite-donor cases**. The two candidate verbs together had median clean
probability **0.141%** (range 0.0256%–0.3375%). In DistilGPT-2, `is` or
`are` was the clean full-vocabulary top in 9/32 cases and their median combined
probability was 8.40% (range 2.86%–21.35%). The opposite patch reversed the
strict binary preference in **0/32 GPT cases** and **2/26 initially correct
BERT cases**. No active arm changed the unique full-vocabulary top token in
either model. These observations do not establish preserved capabilities.

## Fixed comparison

For each of 32 cases, the intervention replaced the subject token's
zero-indexed layer-3 block-output vector with a displacement toward the same
lemma in the opposite number. Controls used an equal-L2 displacement toward a
different lemma of the same number and one fixed equal-L2 Gaussian direction;
clean, self-patch and clean-after were also recorded. The dose was 2.0 in all
cases. Equal input L2 does not imply equal functional effect, and the random
control is one direction, not a randomization distribution.

Let `m = logit(is) − logit(are)`. A donorward change is `−Δm` for singular
targets and `+Δm` for plural targets. The prespecified descriptive rule
required positive mean opposite-minus-control effects against both controls
and at least 24/32 cases positive against both. A correct clean binary sign in
at least 24/32 cases was required to avoid the baseline-limited label. Ties
count as incorrect; all cases remain in the denominator.

| Model and score | Correct clean binary sign | Mean opposite minus same-number control | Mean opposite minus random control | Cases positive against both |
| --- | ---: | ---: | ---: | ---: |
| DistilGPT-2, next token | 32/32 | +0.114993 logits | +0.102311 logits | 32/32 |
| DistilBERT, terminal mask | 26/32 | +0.506194 logits | +0.487784 logits | 32/32 |

Both meet the fixed descriptive rule. These model-specific logits are not on a
common scale. GPT sees only the prefix and predicts its next token; BERT sees
the mask and terminal separator and uses bidirectional attention. Similar
signs do not establish a shared internal mechanism. Token identity,
inflection, different lexical donors, context-dependent gain and nonlinear
readout remain possible explanations. Null or positive effects at one site do
not establish the absence or uniqueness of a number representation.

All six BERT clean-sign errors were singular subjects with a plural attractor:
that cell was correct in 2/8 cases, versus 8/8 in each of the other three
subject/attractor-number cells. This is a fixed-set observation, not a
population estimate or proof that the model implements a nearest-noun rule.
The opposite arm changed BERT from 26/32 to 24/32 subject-correct binary signs;
the same-number and random controls changed none. The full-vocabulary median
total variation under the opposite arm was 0.00481 for GPT and 0.01130 for
BERT; corresponding same-number/random medians were 0.00208/0.00262 and
0.01087/0.00856. Full per-case scores and diagnostics are in
[GPT replay](gpt-replay.json) and [BERT replay](bert-replay.json).

## Post-hoc context and dependence check

In each of 16 matched frame/lemma/subject-number pairs, only the **later**
attractor token changes number. GPT's clean subject-site vector was bitwise
identical in 16/16 pairs, as were its opposite- and same-donor landed vectors,
but the clean `is−are` margin changed in all 16 (plural-attractor minus
singular-attractor range −1.81619 to +0.29466; 13 decreased). BERT's subject
vector differed in all 16 pairs (L2 range 1.584–2.531), and its margin
decreased in all 16 (range −5.72156 to −0.79200), with six strict binary
sign changes. The [witness replay](attractor-pair-posthoc.json) reports aggregate bit-identity
counts; an independent [context audit](attractor-context-audit.json) provides
the per-pair data.

GPT's causal subject state cannot incorporate a later attractor, while later
positions and downstream layers can. BERT's bidirectional subject state can
incorporate later context. These post-hoc observations do not locate the
downstream route, establish an abstract number feature, or make the 32 rows
independent. They motivate a separately frozen BERT continuation-context
follow-up rather than treating the terminal-mask comparison as a complete
behavioral explanation.

## Execution, exposure and evidence

The published [protocol](../../docs/CROSS_MODEL_AGREEMENT_V1.md) and fixed
[case manifest](../../experiments/cross_model_agreement_cases_v1.json) were
available before model forwarding. The manifest SHA-256 is
`b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503`.
The GPT rows and four answer continuations were reserved in inventory v36;
before launch the operator recorded v37 as the latest reviewed inventory and
checked a then-pending 16-case distractor proposal separately. Its
[public comparison](pending-distractor-comparison.json) found zero relations
among 10,240 bidirectional sequence pairs. The named BERT public inventories,
including the 37 color cases, were independently checked before launch; the
subsequent color-v2 status ledger kept the same rows. These are scoped
point-in-time checks, not proof of complete past or future exposure history,
private-split disjointness, or absence from pretraining.

Each CPU child made 32 clean captures and five additional forwards per case:
192 forwards/model, 384 total. The direct-child observers recorded starts at
2026-10-29 12:10:13 UTC and successful exits at 12:10:47 UTC (BERT) and
12:10:53 UTC (GPT). Both owners reported dead model weak references after
cleanup. The [GPT exit receipt](gpt-exit-observation.json) SHA-256 is
`dabbee83479c7cdb647900c2b6d45d620866f23077bc7b6114d242d8f15eadcd`; the
[BERT exit receipt](bert-exit-observation.json) is
`a4a4cb09da30754ec6c814f9d042ca86115369830b6e2f5fc9eac62cddc72b00`.
These cooperative direct-child observations do not prove model execution,
loaded-asset identity, OS-wide descendant drain or driver cleanup.

The one-shot source was commit `763402e8311e2402c49eb865c1812c5a8b5380a2`
(merged in #637), Python 3.11.15, PyTorch 2.8.0+cu128, Transformers 4.57.3,
NumPy 2.4.6, seed 20261029, CPU, float32 and eager attention. Pinned checkpoint
revisions and asset digests are in the protocol and saved metadata. The
owner recorded full-sequence finiteness, and the hook source assigns only the
selected position; the retained selected-position logits and vectors cannot
independently attest unsaved positions.

Two independent post-closure audits checked the retained raw bytes, ordered
events, donor identities, bitwise self/clean-after controls, doses and signed
outcomes without model execution. Their reports are
[GPT audit](gpt-independent-audit.json), [BERT audit](bert-independent-audit.json),
and [methods audit](methods-audit.json). Maximum absolute realized L2 error
was 2.54e−7 for GPT and 1.29e−7 for BERT; maximum relative direction error was
1.74e−6 and 3.90e−7 respectively. The independent auditor's
[portable script](audit_cross_model_saved_independent.py) requires the exact
external receipt and completion digests. The methods/context scripts here
adapt only the original local paths to arguments; their reruns reproduce all
scientific fields, with only the script-hash field changed.

The exact full raw evidence is retained in the operator's 52,880,294-byte
`cross-model-agreement-cpu-v1-20261029.zip`, SHA-256
`6b8d68b824a8724bf3c1644f4ca5a701753588ae933e76d5f178570432b280d6`.
It includes full selected-position vocabulary logits, selected-site witnesses,
start/event journals, both exit observations, the manifest and source reader;
it is not included in this repository. ZIP member hashes were verified and
both included readers were rerun successfully after extraction. Raw child
logs are excluded because they may contain local paths. These CPU runs do not
measure GPU VRAM or throughput.

### Replaying the saved evidence

Obtain the separately retained archive and verify the SHA-256 above, then
extract it without renaming the two attempt directories. A checkout containing
the Git object `763402e8311e2402c49eb865c1812c5a8b5380a2`, Python and NumPy is
needed for the independent script. The following example uses a POSIX shell;
set `EVIDENCE` to the extracted directory and `REPO` to that checkout. The
archive itself contains the exact case manifest. These commands do not load a
model.

```sh
RESULTS="$REPO/results/cross-model-agreement-cpu-v1-20261029"
CASES="$EVIDENCE/source/experiments/cross_model_agreement_cases_v1.json"
git -C "$REPO" cat-file -e 763402e8311e2402c49eb865c1812c5a8b5380a2^{commit}

# Stage copies under the original observer/replay names expected by the
# methods and context readers; source archive bytes remain unchanged.
python - "$EVIDENCE" <<'PY'
from pathlib import Path
from shutil import copyfile
import sys
root = Path(sys.argv[1])
for model in ("gpt", "bert"):
    controller = root / f"cross-model-{model}-controller-v1-20261029-01"
    controller.mkdir(exist_ok=True)
    copyfile(root / f"{model}-exit-observation.json", controller / "exit-observation.json")
    copyfile(root / f"{model}-replay.json", root / f"cross-model-{model}-cpu-replay-v1.json")
PY
python -B "$RESULTS/audit_cross_model_methods.py" "$EVIDENCE" "$CASES" "$EVIDENCE/methods-replay.json"
python -B "$RESULTS/audit_attractor_context.py" "$EVIDENCE" "$CASES" "$EVIDENCE/context-replay.json"

for model in gpt bert; do
  if [ "$model" = gpt ]; then
    receipt=dabbee83479c7cdb647900c2b6d45d620866f23077bc7b6114d242d8f15eadcd
    complete=611f57ea1d1fb937c79580c18bc25e5949e4d0b7774f120b2745104d42c97034
  else
    receipt=a4a4cb09da30754ec6c814f9d042ca86115369830b6e2f5fc9eac62cddc72b00
    complete=8ca452816ac95be6ccaa3f1e04587440322d54acc3c6bf67cacf72fd507ca15a
  fi
  python -B "$RESULTS/audit_cross_model_saved_independent.py" \
    --attempt "$EVIDENCE/cross-model-$model-cpu-v1-20261029-01" \
    --cases "$CASES" --receipt "$EVIDENCE/$model-exit-observation.json" \
    --receipt-sha256 "$receipt" --complete-sha256 "$complete" \
    --source-commit 763402e8311e2402c49eb865c1812c5a8b5380a2 \
    --repo "$REPO" --output "$EVIDENCE/$model-independent-replay.json"
done
```

The methods/context report's source-hash field refers to the running copied
script; compare the scientific fields when checking against the published
original reports. These checks use the retained outputs and cooperative
receipts, with the scope and limitations stated above.
