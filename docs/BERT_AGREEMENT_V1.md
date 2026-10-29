# DistilBERT masked agreement: protocol v1

Status: **frozen before evaluation.**
The public development probe used four exposed prompts and motivated the
candidate location below. It is exploratory and does not establish a general
mechanism. No evaluation attempt on these cases is known from the cited records
at this source freeze; exposure outside those records remains unverified.

## Question and model

Can a specified hidden-state replacement at the subject token move a masked
language model's `is`/`are` preference in the donor's number direction, including
when a noun of either number intervenes? This asks about a concrete intervention
on these inputs; it does not claim that activation similarity explains the
model's reasoning, that subject number is represented uniquely at one site,
or that changed states remain natural.

Use `distilbert/distilbert-base-uncased` at revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, with the pinned safetensors SHA-256
`5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063` and the
verified tokenizer/config assets:

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 483 | `69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83` |
| `tokenizer_config.json` | 48 | `a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10` |
| `vocab.txt` | 231508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |
| `tokenizer.json` | 466062 | `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98` |
| `model.safetensors` | 267954768 | `5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063` |

Freeze the
complete source, checkpoint, tokenizer, exact input IDs, runtime, attention
backend, float settings, seeds, selected locations, output and trace limits
before held-out runs. Record actual model entry, hook invocation and replacement
landing at each named block/token; preserve all failures and incomplete runs.
Evaluation mode, disabled dropout, float32, eager attention, deterministic
algorithms and one CPU intra-op/inter-op thread are the CPU reference. The
CPU study hides CUDA and records PyTorch CUDA initialization state. A GPU
execution has a distinct runtime identity and must use the common supervisor,
UUID-bound device lease, a separately reviewed workload-specific memory
profile, synchronized measurements and a clean drain; the two-token
DistilGPT2 service profile does not authorize it. A limited first GPU profile
is development/calibration evidence rather than normal workload admission.

The CPU implementation launches a separate owner through a source-only
bootstrap, checks the actual checkout against the named Git ancestor before
and after execution, verifies the five staged model/tokenizer files, and
re-tokenizes the entire public inventory before any forward. It requires
PyTorch 2.8.0+cu128, Transformers 4.57.3 and NumPy 2.4.6, with an isolated
Python interpreter, a fresh empty bytecode prefix, and CUDA hidden. Python hash
randomization uses an OS-provided seed that is not fixed. The calling-thread
denormal check does not attest every native thread. An unavailable OS machine
string is recorded as `unknown` rather than inferred from the host. The interpreter, installed
dependency contents, OS and absence of concurrent filesystem/model mutation
are trusted; source hashes do not independently prove faithful execution.
Evaluation requires this unique frozen status line in a committed, published
source revision, the source-isolated launcher, two separate supervised CPU
custody roots, and the matching append-only attempt ledger. An interruption
while starting a legacy unsupervised development owner may leave a child active
without a recoverable handle; that attempt is incomplete and any possible child
must be resolved before retrying.

The optional supervised CPU development path uses the cooperative CPU-study
contract and one explicitly supplied persistent custody directory, separate
from the checkout and each attempt. Its child claims the matching launch grant
and lease before importing NumPy, PyTorch or Transformers. The controller
records the contract and a SHA-256 of the canonical custody path in preflight;
this hash is an identifier, not secrecy against guesses. The child retains the
lease through exit and attests only after the CPU workload and cleanup. Any
uncertain or unclean owner state leaves the custody marker in place and blocks
later cooperative use of that root. This contract does not attest GPU state or
isolate a hostile process. The supervisor's raw reason is retained, and its
result is explicitly scoped to cooperative CPU-study custody; a generic
`clean device teardown` reason does not attest GPU teardown. The child uses an attempt-local working directory and
TEMP/TMP, and omits `SYSTEMDRIVE` from the sanitized environment to contain the
observed unexpanded relative cache path; absolute writes are not sandboxed.
Use one agreed root for all attempts. The unsupervised development path remains
legacy-only; it does not authorize evaluation.

## Inputs and exposure

Each template produces a single masked token and single-WordPiece subject and
distractor in the pinned tokenizer. Reject a case rather than silently changing
positions if these checks fail. The target is `logit(is) - logit(are)` at the
mask; the two target words must each be one token. Inputs include standard
special tokens, with subject position 2, distractor position 5, mask position
6, and a ten-token sequence in the pinned tokenizer. Verify and record the
actual token IDs and attention mask for every forward.

Development families use `The {subject} near the {distractor} [MASK] ready.`
and the ordered singular lemma pairs `(farmer, nurse)`, `(pilot, soldier)`,
`(driver, singer)`, `(doctor, dancer)`. The original four-prompt pilot exposed
singular and plural `farmer` with `nurses`, and singular and plural `pilot`
with `soldiers`; the candidate site was selected from that pilot. Two later
family-1 development attempts exposed all eight A/B subject-number and
distractor-number rows, as recorded in
`results/bert-agreement-dev-v1-20261028/`. Their complete and incomplete
statuses are distinct. The original v1 manifest's exposure labels are an
immutable earlier snapshot; the append-only
`experiments/bert_agreement_exposures_v2.json` records those attempts and the
GPU calibration. Its pinned SHA-256 is
`224d683291b5b4f3a617697c8f9811351cf63bef19958fff9870c943e88fcdbc`.
The owner checks that snapshot's exact bytes, its source binding and the full
token-row disjointness of the cited development and evaluation cases before
model loading. This snapshot covers the cited records, not every possible
historical exposure; later attempts must be separately recorded before an
evaluation freeze. All development rows are reserved for development, not
held-out evidence; only the disclosed rows are known to have been forwarded. The
candidate, comparisons, thresholds and held-out families below remain fixed
after examining those development results. Unknown prior exposure remains possible. Freeze a
separate DistilBERT-specific inventory with exact text, input IDs, attention
masks, tokenizer and model asset hashes; another model's inventory cannot
certify these inputs. The runner records the values of the tensors supplied at
each model call, and the closed-journal auditor checks their integer types,
order and values against the corresponding manifest row. These records remain
self-reported execution evidence rather than independent execution proof.
The separate four-forward GPU calibration in
`results/bert-gpu-calibration-v1-20261029/` used only the previously exposed
`farmer`/`farmers` rows with plural `nurses`; it added no unique input rows and
is device/resource evidence, not a held-out behavioral result.
The subsequent supervised CPU retry is preserved in
`results/bert-agreement-supervised-dev-v1-20261028/`. It reused the same eight
development-family-1 rows. The append-only v3 snapshot in
`experiments/bert_agreement_exposures_v3.json`, SHA-256
`6945a478e3af37083af055bf20bf65628fa979e0ee8b253ea8bacb475cb9e9f2`,
records that retry. The later supervised 140-forward GPU development-family
profile is preserved in `results/bert-gpu-family-profile-v1-20261029/` and
reused the same eight already exposed rows. The append-only v4 snapshot in
`experiments/bert_agreement_exposures_v4.json`, SHA-256
`1f9ae67040646341097414f5ef6da1f66d0ccc5f0be47deeee36cfe66550e32e`,
records that attempt and is the cited pre-evaluation snapshot. The candidate,
controls and thresholds below were not changed in response to that profile.
The snapshot does not certify all prior exposure or replace later attempt records.

Held-out families use the distinct template `The {subject} beside the
{distractor} [MASK] ready.` and ordered pairs `(painter, artist)`, `(actor,
guard)`, `(teacher, worker)`, `(chef, clerk)`, `(student, player)`, `(lawyer,
coach)`, `(writer, sailor)`, `(leader, cook)`. For each pair, cross singular
and plural subject with singular and plural distractor, giving 32 target cases
in eight distinct noun-pair families. The subject and distractor lemmas do not
overlap development lemmas, and the preposition also changes. These are
held-out from **our** development and site selection, not proven unseen in
the checkpoint's training data. They are public fixtures, not a blind test.

Before an evaluation owner can run, publish and independently review the
complete source, protocol, exact manifest and launch-custody binding; record
their hashes in a separate preflight. A started evaluation row is considered
exposed even if its forward or the attempt later fails. If a partial journal
or uncertain owner state cannot establish which rows were reached, conservatively
treat every row in the selected workload as potentially exposed. Preserve each
incomplete attempt and its observed or uncertain owner state. A retry is a new
attempt under the same frozen protocol and source, never a pristine first
evaluation; do not alter sites, examples, controls or thresholds in response
to partial results. The first complete evaluation attempt is the primary
result; any later complete runs under the same protocol are separately labeled
replays. If any prior attempt may have started, the first complete result is
also a repeat and must not be described as a clean first-exposure result. Retain
all attempts and report which were inspected and the retry order.

## Interventions and comparisons

The candidate is tuple element 0 of `distilbert.transformer.layer.2`, the
**output of zero-based encoder block 2**, invocation 0, batch 0, token 2.
The full tensor shape is `[1,10,768]` and the selected vector shape is
`[768]`, float32. The same path/selector/invocation convention applies to
each alternative block and token. It was chosen from the exploratory development probe, which only used
plural distractors. For each target and site, capture the donor's vector from
the same location on a separate forward, then replace exactly one target vector
at that location. Hooks must be removed even on error; model training mode and
registered parameter/buffer identities and values must be unchanged afterward.
Fresh unpatched model outputs are compared before and after the session. These
checks address cleanup, not preservation of unrelated capabilities during a
patch. Captures and results are bounded; logical bytes are not peak memory.

For each target family, let A be its subject lemma and B the next subject lemma
in the same ordered split, cyclically. At fixed template and distractor, use
four donors: A with the target's number, A with the opposite number, B with the
target's number, and B with the opposite number. A/same is a self-patch.
Every donor and base identity, `is` and `are` logit for each intervention and
control, actual site/invocation, and post-action landing evidence is retained.
The self-patch must reproduce unpatched `[MASK]` logits at raw float32 bits;
the patch hook must separately verify donor bits at the replaced position.
Same-number/different-lemma patches are lexical controls, not assumed to have
zero effect. Opposite-number same-lemma replacements also change surface form;
the crossed controls reduce but do not remove lexical confounding.

Use the same donor contrasts at the distractor token of block 2 as a rival
position. Since this is a bidirectional encoder, that token may already carry
subject information: its effect is a comparison, not automatically a null.
Random comparison sites are a fixed seed-2026102811 sample without replacement
from blocks `{0,1,3,4}` crossed with subject/distractor positions: in generated
order they are `(3,distractor)`, `(4,distractor)`, `(0,subject)`. They are
three fixed draws, not an estimate over all possible sites.

For delivery controls, the same-lemma A/opposite-number donor patches the final block output
at the subject position should leave `[MASK]` logits unchanged: no cross-token
block remains. A final-block `[MASK]` patch should reproduce the donor's mask
logits, because the remaining masked-LM head is positionwise. Check both at
raw float32 bits. The latter is a positive delivery check by construction, not
evidence locating a grammatical computation. Fail the run as incomplete if a
required identity, landing, position, hook-cleanup or delivery check fails.

## Prespecified outcome

Let `L = logit(is) - logit(are)` and `r = +1` for a singular target and `-1`
for a plural target. For a site, the target-level signed donor-number effect is
`-r/2 * [(L_A,opposite - L_A,same) + (L_B,opposite - L_B,same)]`.
Positive values mean that switching donor number shifts this signed contrast
in the donor's number direction, averaged across the two donor lemmas. They can
arise from selectively erasing the target preference or from other lexical and
surface changes; they do not establish that a unique number feature was
transferred. Report the underlying four margins and both logits for each, not
just the derived effect.

Average the four subject-number × distractor-number effects within each of the
eight held-out noun-pair families, then average those eight family means. The
two float32 logit bit patterns are independently decoded into binary64 before
each margin subtraction and both patterns are saved. Compute each margin as
`math.fsum((is_logit, -are_logit))`; compute each A and B donor delta as
`math.fsum((L_opposite, -L_same))`; compute each effect as
`-r * (math.fsum((delta_A, delta_B)) / 2.0)`. Within each family the target
order is `(singular,singular)`, `(singular,plural)`, `(plural,singular)`,
`(plural,plural)`, where the two entries are subject and distractor number.
Use `math.fsum(effects) / 4.0` in that order, then `math.fsum(family_means) /
8.0` in the listed family order.
Reject nonfinite logits, decoded margins, contrasts or aggregates. Ties in
the baseline preference count as incorrect. This specifies arithmetic rather
than a tolerance or a post-hoc exclusion rule. The
predeclared directional hypothesis passes if the candidate's family-mean
average is greater than `0.1` logit and at least six of the eight family means
are positive; otherwise report a negative result. The 0.1 threshold is a
family-averaged signed log-odds contrast: its exponential, about 1.105, is the
corresponding aligned geometric-mean odds multiplier, not a per-case effect.
For singular targets the raw `is`/`are` odds move in the opposite direction
from the aligned contrast. The separate location-specificity prediction is that the candidate's
family-mean average exceeds the matched block-2 distractor effect by more than
0.1; it can fail independently of the directional hypothesis. Report each
random site's family means and candidate-minus-site contrasts as descriptive
baselines. For each family compute `math.fsum((candidate_family_mean,
-comparison_family_mean))`; for the aggregate contrast compute
`math.fsum((candidate_site_mean, -comparison_site_mean))`. Also report
`math.fsum(family_contrasts) / 8.0` in the fixed family order. This arithmetic
mean may differ slightly from the difference of site means by rounding; the
prespecified specificity decision uses the latter. Do not infer significance
from these descriptive values.

All 32 target cases, including model mistakes, remain in both the data and
effect summary. Report baseline agreement accuracy and mean margins for all
four subject/distractor-number cells, plus per-family and per-case effects.
Do not use post-hoc eligibility filtering, drop difficult cases, retune the
threshold or select a new layer after observing held-out outcomes. A missing
case or invalid control makes the run incomplete. The sample is small and
families/templates are not a population sample; cyclic donor reuse also makes
families dependent. Treat these as descriptive units and avoid significance or
cross-model generalization claims. The held-out nouns and preposition both
change from development, so a failure cannot distinguish lexical from template
generalization. Save the frozen protocol, source revision,
case and donor IDs, runtime and model hashes, complete scalar outputs, actual
hook metadata, controls, resource observations, terminal status and failures.

## Forward and storage plan

Each family has four target cases and eight unique A/B subject-number donor
inputs across its two distractor numbers. For each of the five candidate,
rival and random sites, capture the eight donors once and run 16 target/donor
patches: 120 forwards. Capture the opposite-number donor at both final-block
control sites and run both control patches for each target: 16 forwards. Run
four final clean checks. Thus each family has at most 140 root forwards; eight
families have at most 1,120, excluding recorded development pilots. The four
target baselines are among the site captures, and the A/same donor patches are
the self controls, so neither is counted twice. Every invocation counts even
if it fails, and a retry is a new, separately recorded attempt.

For each target, the candidate-site clean capture is its authoritative
baseline. Retain its complete mask logits in memory for the family; compare
each other site's clean capture, each self-patch, and the final clean check to
that reference at float32 raw-bit equality. Record each check and full-mask
digest; if any fails, preserve the attempt and mark it incomplete.

Process one family and one site at a time, keeping at most eight source
vectors and four target reference mask logits for that site, plus the current
observation. Stream complete scalar results and selected vector bit evidence to
a bounded result file as forwards complete; store full mask-logit hashes and
the identity/delivery comparisons, rather than retaining all full logits in
memory. A stored hash permits digest comparison but cannot independently
reconstruct or verify the original full logits. A separately run replay can
provide additional evidence, not proof of the original execution. Reserve at
most 16 MiB for serialized output and trace evidence for the eight-family study;
stop and mark the attempt incomplete instead of truncating. The capture payload
limit is separate from process peak: eight retained 768-float source vectors
are 24,576 bytes, but the live forward also includes the source copy held by
the plan, pre/post captures, a `[1,10,768]` hidden clone (30,720 bytes), the
full `[1,10,30522]` logits (1,220,880 bytes), finite-check scratch, model,
activations, allocator and host serialization. The exact final runner's
lifetimes, device and host budgets must be profiled; finite measured peaks do
not prove an upper bound for arbitrary workloads. Hold the common device lease
for the entire supervised GPU owner lifetime, with a per-forward admission
reservation through synchronized disposal of all referenced device tensors.

The CPU controller creates a new attempt directory, journals each started and
completed forward, and rechecks the closed journal against the declared
schedule before publishing a terminal record. It hashes and sizes the start,
preflight, owner-start, owner-candidate and journal files into the terminal.
JSON records are flushed to a same-directory temporary file and published by a
no-clobber hard link; this is not a claim of crash-proof directory durability.
Accept a completed terminal only with the controller's zero process exit,
zero owner exit, a reaped owner, matching evidence hashes, and neither
`controller-failure.json` nor `controller-cancelled.json`. In particular, interruption can arrive after the
complete record's link: the file may remain present while the controller
reports cancellation and exits nonzero. A nonzero, missing or uncertain process
exit, partial journal, failed audit, missing evidence, or a possibly active
owner is incomplete. The failure record identifies a bounded controller stage
and distinguishes the original failure,
reap/cleanup failure and any validated owner failure where available. A later
retry uses a fresh directory and is a separate attempt.

The CPU evaluation uses the fixed, machine-specific root identities in
`experiments/bert_agreement_cpu_evaluation_v1.json`: distinct cooperative CPU
custody roots for the controller and model owner, and a persistent attempt
ledger. Their path hashes identify the
chosen roots and are not secrets. The controller acquires the ledger's
cross-process lock, validates the cited v4 snapshot and exact 64 target/donor
rows, and publishes a no-clobber reservation before granting a child. Each
reservation binds the source, protocol, checkpoint, manifest, roots, controller
PID and supervisor birth token, exact rows, and hashes of all prior attempt
records. At most 32 attempts are allowed. A possibly started attempt makes all
64 rows potentially exposed; completed-forward records describe a narrower
observed subset and must not erase that conservative accounting.

Under the ledger lock the controller first refuses an already visible inner
owner unsafe marker or attestation without reserving an attempt. The check is
not atomic with other users of the inner root: its own supervisor remains the
admission authority, and an uncertain later launch stays blocked for recovery.

The source-isolated launcher uses the separate controller root and the reviewed
private launch grant before the controller can claim its lease. The controller
claims before source preflight, reservation, or model-capable imports. The
launcher supervises the exact direct child, whose parent watchdog exits if the
launcher dies; a lost spawn, cancellation or uncertain teardown retains the
controller root's unsafe marker. The controller retains that lease through
exit and attests only after the inner owner is reaped or known never started
and its ledger session is released. The trusted controller holds its ledger session through owner launch, reaping,
journal audit and terminal publication. The owner separately claims the CPU
supervisor session before model dependencies, verifies its direct parent and
birth token, and checks that the ledger lock is contended. Contention alone
does not identify the lock holder; ownership depends on the trusted controller
retaining its session. The supervisor's parent watchdog and the sticky CPU
custody marker are separate controls. A missing reservation, uncertain owner,
partial record, absent local controller-exit observation or contradictory
failure sidecar blocks an automatic retry. The launcher records its observed
reaped controller exit only after clean controller custody; the observation
must match the controller PID and supervisor run ID in the reservation.
The inner owner-start and candidate records bind the inner supervisor run ID
to the controller's reaped-owner observation. These are cooperative checks.
The launcher publishes its local exit observation separately;
it is cooperative evidence, not independent process attestation. The launcher
uses a new local work directory for the controller and its TEMP/TMP, and
omits `SYSTEMDRIVE` from their sanitized environment to contain the observed
unexpanded relative cache path. Absolute filesystem writes are not sandboxed.

File contents are flushed before atomic no-clobber publication, but directory
entries are not proven durable across OS crashes or power loss. After such an
event, reconcile the externally preserved attempt history and custody state
before any retry; this ledger does not establish a complete history after a
filesystem failure. Its locks and records require cooperating processes and a
trusted filesystem, not a hostile-process boundary.

Before a later retry, completed prior attempts are re-audited against the
canonical manifest and their full recorded schedule; source, runtime, model,
custody, artifact and exit records must agree. That read-only controller audit
may import the pinned Torch runtime with CUDA masked, checks PyTorch CUDA state
and caller-thread denormal mode, and performs no model forward. Its runtime is
recorded in the next reservation. These checks do not attest system-wide GPU
inactivity. The 16 MiB journal limit bounds stored bytes, not the larger host
memory needed for decoded JSON, Python objects and Torch during re-audit.

For a development-only CPU run on Windows, use the base interpreter behind
the installed virtual environment, the virtual environment's `site-packages`,
the exact published source commit, and a new output directory. The parent
bytecode-prefix directory must already exist and be empty; the output directory
must not exist yet. For example, after setting `$repo`, `$cache`, `$basePython`,
`$sitePackages`, `$parentPrefix`, `$output`, and `$sourceCommit` to absolute
paths or the full commit SHA as applicable:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
& $basePython -I -S -B -X "pycache_prefix=$parentPrefix" `
  "$repo/scripts/bert_agreement_bootstrap.py" --bootstrap-site $sitePackages `
  --repo $repo --cache $cache --output $output --commit $sourceCommit `
  --split development --family 1
```

`$cache` names the local Hugging Face `hub` directory containing the pinned
snapshot. The controller sanitizes the owner environment, keeps CUDA hidden,
and records an incomplete attempt if preflight, owner execution, reaping or
the post-close audit fails. Preserve the terminal plus its inventoried files
and the actual controller exit code; do not reinterpret partial attempts as
successful results.

## Motivation

Prior BERT subject-verb agreement studies show the need to distinguish
representation from causal use and to control lexical frequency; these motivate
the present tests but do not establish a result for this checkpoint:

- [Hao and Linzen, 2023](https://aclanthology.org/2023.findings-emnlp.300/)
- [Hanna et al., 2023](https://aclanthology.org/2023.eacl-main.58/)
- [Wei et al., 2021](https://aclanthology.org/2021.emnlp-main.72/)
