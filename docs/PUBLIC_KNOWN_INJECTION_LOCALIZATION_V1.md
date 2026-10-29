# Public known-injection localization, version 1

This is a small, public instrumentation study on a pinned pretrained DistilGPT-2. We deliberately alter a known activation and ask whether a site-blind selection procedure can locate it and whether its selected patch predicts or restores held-out next-token logits. It is a test of this tracing and intervention setup, not evidence for a naturally occurring circuit, agreement ability, or general localization performance. We do not access, query, or retry the old protected one-use split. The public cases below are held out only from this study's discovery procedure; exposure during pretraining or outside the recorded public inventory is unknown.

## Fixed inputs and partition

The complete, exact UTF-8 prompt texts and their order are fixed here before any forward. Each line is one prompt, with no added trailing whitespace. No prompt or split will be changed in response to model outcomes. Tokenization, row identities, contextual target continuations, and public exposure comparisons will be checked and frozen separately before execution. A failed pre-forward check stops this version; any revision is explicitly versioned and reviewed.

Discovery, in order:

1. `The silver compass beside the harbor`
2. `The quiet kettle beneath the walnut tree`
3. `The linen satchel near the stone bridge`
4. `The yellow lantern beside the mill`
5. `The bronze pitcher beyond the orchard`
6. `The faded journal under the window`
7. `The woven basket behind the barn`
8. `The small sundial above the meadow`

Held out, in order:

1. `The copper anchor near the lighthouse`
2. `The painted violin inside the cabin`
3. `The narrow ladder beside the garden`
4. `The paper envelope beneath the table`
5. `The marble statue beyond the fountain`
6. `The blue teapot behind the curtain`
7. `The glass bottle above the hearth`
8. `The scarlet flag outside the station`

Use the pinned tokenizer without added special tokens and a nonempty sequence of at most 32 tokens per prompt. Require exact text/token round trips and exact, distinct full input rows. Reject any equality or full-row prefix relation in either direction among the 16 inputs, between them and all other known public inventory inputs or recorded continuations, and against pending public input bundles at the actual reservation/launch boundary. Check each input plus each fixed score token against those known rows and extensions in both directions as well. Record the inventory revision, pending-bundle identities, comparison counts, and the limits of that public check; reserve the 16 inputs before any forward. Do not query a protected split. The score token IDs 318 and 389 must each be a single continuation token at every fixed prompt boundary, with exact appended-token agreement. They are a fixed logit contrast here, not a claim that these ordinary texts form an agreement task.

## Model, sites, and fixed injections

Use the offline, verified `distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`, with model safetensors SHA-256 `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b` and tokenizer JSON SHA-256 `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`. Execute serially on CPU in float32, evaluation/eager mode, with no key/value cache. Use the repository's `GPT2TraceSession` and ordered `ActivationPlan` on the 768-channel MLP output of each zero-based transformer block, at the final input token. The four eligible candidate sites are blocks 1, 2, 3, and 4, in that order.

There are two separate, predeclared injection cells: A adds the fixed vector at block 1; B adds it at block 4. The vector is exactly +4.0 in channel 0 and +0.0 in every other channel. Both the coordinate and strength are fixed before any model outcome. The two injected sites are two site trials, not 16 independent site trials. Each injection is applied once, immediately after that block's MLP output, before a later patch in the same ordered plan if the candidate site is the injection site. No injection strength, target channel, site, prompt, or selection rule may be adapted after observing outcomes.

For each injection save same-invocation full MLP output bytes before and after the add. Check that channel 0 at the final token equals the correctly rounded float32 result of adding 4.0 to its saved float32 before value, and that every other channel and every other token position has identical float32 bits. A failed landing, nonfinite tensor, missing observation, unexpected model return, or hook cleanup failure stops the attempt and is reported as incomplete. Preserve the observed bytes and the original error; never silently retry an uncertain or failed action.

## Fixed discovery and baselines

For each discovery case and injection cell save clean, injected, and each of the four candidate-patched full-vocabulary last-token logits. A candidate patch replaces its final-token MLP vector with the corresponding same-prompt clean vector captured before the injected run. The other intervention is unchanged. The candidate may be upstream, at, or downstream of the true injection. Preserve before/after full-block snapshots and check the target vector and all untouched coordinates on the same invocation. Candidate order is 1, 2, 3, 4 for every case and both cells.

The injected-only path's four MLP vectors are captured in model execution order; at the injection site, capture its vector **after** the add, not before it. A fabricated ordered-hook test must verify this sequence and the first-change result before any pretrained forward.

Let `m(z) = float64(z[318]) - float64(z[389])`, with each saved float32 logit decoded exactly to float64. For discovery case `i` and candidate `j`, define

`R[i,j] = abs(m(injected[i]) - m(clean[i])) - abs(m(patched[i,j]) - m(clean[i]))`.

The primary selector takes only the eight numerical clean/injected/patched margins indexed by candidate site and case order; it receives no injection-site, role, case-text, or held-out metadata. Compute the arithmetic mean of each candidate's eight `R` values in fixed case order, choose the largest strictly positive mean, and break exact ties toward the lower block index. If no candidate has a strictly positive mean, abstain. Record all scores, ranking, ties and abstention. The code boundary enforces this restricted input, but injection labels are public and the people preparing the study know them; this is procedural blindness, not strong human blinding.

Compare the primary selector with these predeclared controls, all using the same four candidates and discovery cases:

* **First changed:** for each discovery case, find the lowest candidate index whose injected and clean final-token MLP vectors differ in any float32 bit. Select the site with the most such first-change votes; break ties toward the lower index and abstain if no vector changed. Directly altering the observed MLP output makes this control structurally easy; success is principally an instrumentation check.
* **Magnitude:** for each site and case compute the Euclidean norm of the clean-to-injected final-token MLP vector difference using decoded float32 values and `math.fsum` of squared float64 differences. Select the largest mean across cases if strictly positive, with lower-index ties; otherwise abstain.
* **Constant earliest site:** always select block 1. On the two declared cells this hits A and misses B regardless of model observations, explicitly exposing any early-site advantage.
* **Fixed pseudorandom:** for each public cell label A or B, take the first byte of SHA-256 of ASCII `public-known-injection-v1-random`, a zero byte, and the cell label, and select block `1 + (byte modulo 4)`. This is one fixed realization; report its selected site and the analytic uniform-site reference of 1/4 separately, without a significance claim from two deliberately chosen sites.

For each candidate also freeze the signed prediction `P[j]`, the arithmetic mean over the eight discovery cases of `m(patched[i,j]) - m(injected[i])`. Use `math.fsum` in fixed discovery-case order divided by eight for each score and signed prediction. Publish the immutable discovery record with the full ranking, all four signed predictions and all five methods' choices, together with hashes and the exact selector source, **before the first held-out forward or inspection**. Verify this freeze independently. All-zero or very small true effects remain outcomes: use the positive-only selection rule exactly, retain continuous values, and do not exclude or replace cases based on effect size.

## Held-out evaluation and controls

Run every one of the four candidate patches on each held-out case for each cell, regardless of the chosen sites. For each method's frozen candidate and case, compare its frozen `P[j]` with the realized signed change `m(patched[j]) - m(injected)`. Report per-case errors and mean absolute error, and the corresponding zero-prediction mean absolute error on exactly the same cases. A strictly smaller mean absolute error than zero is the fixed descriptive prediction-success condition. If the selector abstained, report it and leave its selected-site metrics undefined; all four candidate results remain available.

Independently compute, from all 50,257 saved last-token float32 logits, Euclidean distances `d_injected = ||injected - clean||_2` and `d_patch = ||patched - clean||_2`, accumulating squared float64 differences with `math.fsum` in vocabulary order. Report both distances for every candidate and case. Report `d_patch / d_injected` only when `d_injected > 1e-12`; otherwise report the ratio as undefined and retain the absolute distances and count of undefined cases. If the mean `d_injected` over the eight held-out cases is at most `1e-12`, the descriptive restoration-success label is undefined. Otherwise it is true exactly when mean `d_patch` is strictly below mean `d_injected`, and false otherwise. Retain worsened outcomes and all continuous values. These are logit-space metrics, not measures of complete generated outputs or preservation of general model behavior.

Every prompt includes an unmodified clean trace, a null plus same-invocation self-patch control, both injected-only runs, the four candidate patches and an injected self-patch for each cell, and an unmodified clean-repeat trace. Thus the fixed schedule is 15 forwards per prompt: 120 discovery and, after the frozen selection, 120 held-out forwards, for 240 in total. Null/self control logits must equal the corresponding clean or injected-only logits bit-for-bit and their full before/after vectors must be identical; clean-repeat logits and four traced vectors must equal their clean captures bit-for-bit. Record first-change false positives on the no-injection repeat. A failed control is retained and explicitly disqualifies the affected result rather than being hidden. Same-site, upstream and downstream patches are reported separately, with their actual landing and unchanged-coordinate witnesses; the full-vocabulary same-site comparison is a check, not a guaranteed scientific success.

For every candidate strictly upstream of the injection, check that its pre-patch target vector equals the same-input clean donor bit-for-bit and its full logits equal injected-only logits bit-for-bit. At the true site, check that the clean vector lands bit-for-bit after the add and the full logits equal clean-path logits bit-for-bit. Either mismatch is a recorded instrumentation/control failure, with all bytes retained, rather than a scientific success. Downstream patches have no prescribed restoration effect. These are deterministic CPU bitwise checks; no tolerance is tuned after observing results.

## Evidence, limits, and one-shot execution

Freeze case bytes, source revision and tree, the complete imported source inventory, runtime/weight/tokenizer hashes, two phase commands, maximum action count, byte caps, and the public exposure comparison before the first forward. Use a fresh empty bytecode-cache prefix checked before project imports and afterward; interpreter startup and process custody remain trusted/cooperative. Create a fresh output directory. Save each full-vocabulary last-token logit vector, relevant site vectors, full before/after intervention tensors, and start/recorded/completed event with case/cell/action order, finite checks, shapes, offsets, hashes and actual hook events. Bound the saved evidence to 150 MiB and metadata to 16 MiB, excluding the already cached model and temporary verified checkpoint staging. Recheck available disk space before each phase; stop before a forward if its declared worst-case evidence cannot fit. Do not delete unrelated files to create room.

The direct-process supervisor records phase start and exit, explicit interruption or uncertainty, stdout/stderr and a SHA-256 inventory of the closed attempt. A missing receipt or lost child ownership does not prove the child stopped and prohibits a retry until independently reconciled. Preserve a private original of any path-bearing traceback; a public path-redacted projection is separately identified and hashed and does not establish the private bytes or redaction fidelity. Save all positive, negative, zero-effect, control-failure, abstention and incomplete outcomes, and make an independent saved-byte audit and scientific interpretation available. No result will be presented as independent confirmation from a private split, independent execution attestation, or a population-level estimate.
