# DistilBERT cue transfer for an `any`/`some` masked prediction

## Question and prior work

Does an equal-norm replacement of a contextual cue state move DistilBERT's
masked-word preference between `any` and `some` in the direction associated
with `not` or `never`? This is a causal intervention on a model state and a
fixed two-token prediction, not a general test of grammatical knowledge or
negation scope. [Warstadt et al. (2019)](https://aclanthology.org/D19-1286/)
found that estimates of BERT's negative-polarity knowledge varied by method;
[Ettinger (2020)](https://aclanthology.org/2020.tacl-1.3/) and
[Kassner and Schütze (2020)](https://aclanthology.org/2020.acl-main.698/)
reported limited sensitivity to negation in other probes. [Kletz et al.
(2023)](https://aclanthology.org/2023.nodalida-1.54/) found a clause-level
lexical confound in representation probes. These are motivations, not
predictions that this DistilBERT checkpoint must behave in the same way.
The active controls below also follow the caution that patching conclusions
depend on choices of intervention and metric ([Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042)).

## Cases and chronology

Before any pretrained study forward, fix the eight rows in
`experiments/bert_negation_public_cases_v1.json`: two development contexts,
then six evaluation contexts. Each uses the template `The {role} has {cue}
{verb} [MASK] of the {object}.` and all four cues `not`, `never`, `just`,
`also`. These are hand-constructed public convenience cases, not blinded or
independent samples. The tokenizer-only check found all four variants of all
eight cases had exactly 12 tokens, equal except at cue index 4; the mask is at
index 6 and the object at index 9. No tokenizer candidate was rejected. Check
the manifest again with the pinned tokenizer before model loading. `any` and
`some` are single tokens 2151 and 2070. Do not select cases, change the split,
sites, threshold or contrast after inspecting any study model output. The
evaluation cases are held out from this experiment's development selection;
the model's pretraining and other prior exposure are unknown.

This template contains no question, modal or second explicit negator. It is
still not a minimal pure-negation contrast: `never` changes temporal meaning,
`just` changes focus and time, `also` has an additive presupposition, and
`any` has free-choice uses. `any` versus `some` is a masked-logit preference,
not an acceptability or entailment judgment. Retain baseline failures and all
unexpected signs rather than filtering for a desired clean preference.

## Runtime and capture

Use cached `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, float32, CPU, evaluation mode,
eager attention, one intra-op and one inter-op thread, deterministic algorithms,
no autocast and no network. Set `CUDA_VISIBLE_DEVICES=-1` before importing
Torch; never initialize or use CUDA. Fix the Torch seed to `2026102901` and
disable float32 denormal flushing. Record whether `MKL_CBWR` and
`ATEN_CPU_CAPABILITY` were present and their values before Torch import. The
owner records and checks the installed
versions, the five pinned asset hashes, source and case-manifest hashes, and
model-state digest before and after the run. The digest includes the state
dictionary and all named buffers, including nonpersistent buffers. Verify the
no-flush setting with a float32 subnormal probe before and after the forwards.
The reference environment is
Torch `2.8.0+cu128`, Transformers `4.57.3`, tokenizers `0.22.2`, safetensors
`0.8.0` and NumPy `2.4.6`.

Each clean forward captures the full 768-float state at four sites, with block
indices zero-based: cue token after block 2 (`cue2`), unchanged object token
after block 2 (`object2`), mask token after block 2 (`mask2`), and cue token
after final block 5 (`cue5`). In this bidirectional model the object and mask
states already contain context and are active transfer comparators, not nulls.
The final cue patch has no later cross-token attention: because the MLM head
is positionwise, its entire mask-logit vector must match the clean target bits.

## Matched displacement and fixed schedule

For each context, take the minimum of the 20 Euclidean distances between
clean states for the five unordered cue pairs `(not, just)`, `(never, just)`,
`(also, just)`, `(also, not)` and `(also, never)` at all four sites. Denote it
`r`. Record every distance, the binding pair/site and the requested fraction
`r / distance` for every patch. All must be finite, and `r >= 0.001`; if not,
retain the failure and declare the attempt incomplete. A small binding edge
may make every patch weak and a negative result then cannot establish absence
of a larger-dose effect.

For a target vector `x` and donor vector `y`, compute each coordinate term in
index order as Python binary64 `(float(y_i) - float(x_i)) ** 2`; apply
`math.fsum` and then `math.sqrt`. Compute binary64 `alpha = r / distance`,
requiring `0 < alpha <= 1`. Form each requested value using separate binary64
subtraction, multiplication by `alpha`, and addition to `x`, in that order;
round once to float32, with ordinary nearest-even runtime behavior checked
before the study. Measure the actual
float32 displacement from the same-forward pre-action witness in binary64,
report the norm and relative error, and
require `abs(actual - r) / r <= 0.001`. Record the full target-before, clean
donor, submitted and landed float32 vectors. The same-forward pre-action
target vector must match that target/site's clean capture bit-for-bit before
its requested dose is accepted. A failed control, nonfinite value
or radius/rounding condition makes the run incomplete; never silently discard
a case or substitute a different dose.

For each context run four clean cue variants, then at each of the four sites
run four same-forward self patches and the following eight directed transfers:
`not -> just`, `just -> not`, `never -> just`, `just -> never`, `also -> just`,
`just -> also`, `also -> not`, `also -> never` (donor -> target). Thus the
patch section has 48 forwards. Finish each context with four unpatched repeats,
one for each cue, and require their full mask-logit vectors and selected site
captures to match the corresponding clean observations bit-for-bit. The fixed
complete schedule has 56 forwards per context, 448 total. Capture before
and after every patch, verify the selected replacement bits and all other
token coordinates, and require self-patched full mask logits to equal the
corresponding clean logits. Save the full float32 mask-vocabulary vector for
every forward, including all self, active-control and final-block actions.

## Preregistered descriptive evaluation

For any action, let `L = logit(any) - logit(some)` and let
`delta(donor -> target, site) = L(patched target) - L(clean target)`.
For `q` in `not`, `never`, define the bidirectional transfer at the cue site
as `(delta(q -> just, cue2) - delta(just -> q, cue2)) / 2`. Average the two
cue values for each context. The fixed directional prediction is that the
mean over all six evaluation contexts is strictly greater than `0.02` logit
and at least four of the six context means are strictly positive. Both parts
must hold. No result is declared for this criterion unless all six evaluation
contexts and required controls are complete; a tie at either bound fails.
The criterion measures directional transfer, not specificity to negation.

Report separately each negative cue, each direction and every context; clean
`L(q) - L(just)` gaps and their signs; the `also`-to-`just` same-recipient lexical comparator
and `also`-to-negative comparators; all corresponding object2, mask2 and cue5
effects; actual radius and displacement fractions; and the full clean
vocabulary baselines, including top-five IDs/logits and `any`/`some`
probabilities. Positive intervention transfer when a clean baseline gap is
zero or points the other way does not establish transfer of the model's
naturally expressed cue preference. The lexical control can be as large as the nominal effect, and
equal input norms do not equate functional impact. No p-value, unique site,
independent population generalization or broad negation-capability claim is
specified.

## Retention

Freeze and independently review this protocol, cases and executable source
before the first pretrained study forward. Use a fresh create-only result
location, retain ordered actions and their float32 vectors, full mask logits,
model/runtime and source identities, controls, and any incomplete attempt.
The owner saves each completed bounded witness before testing subsequent
controls; if a forward completes but its landing or hook/input control fails,
it saves the available bounded witness and failure phase. A failure inside the
model or before a witness is assigned can leave no completed witness for that
forward. Incomplete attempts must never be reported as completed evaluations.
Each emitted JSONL record is gzip-sync-flushed and file-flushed before the
next action. If interruption leaves the gzip footer absent, a streaming gzip
decoder may recover complete records from the flushed prefix; the file is
still incomplete and is not a completed evaluation.
An independent saved-data audit must check the complete action schedule,
landing, common radius, both threshold terms and all results without another
model or GPU forward. Hashes and local chronology are evidence of saved bytes,
not an attestation of execution or hidden pretraining exposure. Source and
asset checks assume a trusted, quiescent checkout, interpreter and installed
dependencies; they are not continuous execution or dependency attestations.
