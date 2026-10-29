# Public future-distractor and downstream-repair protocol v1

## Question and prior evidence

A [previous public known-injection study](../results/public-known-injection-v1-20261029/README.md) found that a wrong-site downstream patch sometimes moved the full-vocabulary output closer to clean. This prospective follow-up asks whether a distinct, immediately downstream clean-vector patch can reduce output error after an explicitly known artificial injection, in the presence of a separately disclosed future-token distractor. The target layers, 25% threshold and aggregate rule below are informed by that earlier block-1 result; this is an exploratory follow-up, not a blind replication or a claim that an amount of causal pathway has been restored. Smaller full-vocabulary distance does not imply every token improved.

A secondary selector tests whether the largest observed contrast effect identifies the known injected coordinate when an upstream, downstream and distractor candidate are also available. Its function receives only anonymous candidate-effect arrays; injection positions and roles are public, so this is a blinded interface rather than human blinding. The first-change and causal-mask controls have outcomes implied by this construction. Neither the primary experiment nor this selector identifies a natural model circuit or general language capability. The old protected one-use split is not accessed.

Use `distilbert/distilgpt2`, revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`, tokenizer JSON SHA-256 `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`, and safetensors SHA-256 `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`. Use CPU, float32, eager attention, deterministic evaluation, no cache and the reviewed GPT-2 trace/intervention API. Fix and publish the protocol, exact tokenizer-checked cases, exposure reservation and source before forwarding; independently review each. The held-out prompts are held out from this selection procedure, not from pretraining or unknown outside exposure.

## Fixed text and scored position

The first eight texts are discovery, the last eight held out, in listed order:

1. The violinist beside the mossy courtyard gate.
2. The sailors beyond the narrow harbor bridge.
3. A lantern hanging above the library staircase.
4. The farmers along the quiet mountain road.
5. This delicate instrument beneath the table.
6. Those young artists outside the crowded gallery.
7. The committee near the old riverside hall.
8. Several clouds behind the distant ridge.
9. A traveler beside the orchard's wooden fence.
10. The carpenters beyond the restored brick tower.
11. Our patient neighbor behind the blue curtain.
12. Those small boats near the eastern jetty.
13. The scholar beneath the garden's ancient oak.
14. Two apprentices outside the workshop door.
15. A silver vessel on the shaded terrace.
16. Many visitors near the northern meadow.

Require each full text to round-trip under the pinned tokenizer, contain 3–32 tokens and end with a separate period token 13. Tokenizing the text without that period must give exactly the full row without its last token. Let `t = length - 2` be the scored position and `f = length - 1` the future position. Verify contextual token 318 (` is`) and 389 (` are`) after the text without the period. The secondary scalar is their next-token logit difference at `t`; these ordinary texts are not claimed as an agreement benchmark. Preserve and compare each full input, its scored prefix and the two scored-prefix continuations bidirectionally against the latest reviewed public inventory, relevant pending public cases and public support inputs. These comparisons do not establish private-split or unknown pretraining disjointness.

## Fixed corruption, candidates and causal controls

In three fixed cells, add +4.0 to channel 0 of the MLP output at block `L` and scored token `t`, for `L` = 1, 2, 3. Independently add +4.0 at block 0 and the future token `f` as the openly labeled distractor. Layer numbers are zero-based. The distractor is irrelevant to the scored position under causal masking, not to the model's behavior at `f`; save both positions' full 50,257-logit outputs. A target is observed to affect a case only if its target-only versus clean scored-logit L2 distance exceeds `1e-12`; report no-effect cases rather than assuming all targets are behaviorally relevant.

For every case/cell retain clean, distractor-only, target-only and combined evidence. Require scored-position distractor-only logits equal clean bit-for-bit, and combined logits equal target-only bit-for-bit. Require the clean patch at the target coordinate to reproduce clean scored logits, and clean patches at the upstream and future-distractor coordinates to reproduce combined scored logits. Measure the final-position effects separately; they need not vanish. The upstream patch is at `(L−1,t)`, the source patch at `(L,t)`, the distinct downstream patch at `(L+1,t)`, and the distractor patch at `(0,f)`. A source-site patch is an expected instrumentation control; the downstream patch may improve, worsen or leave unchanged the scored output.

Every add and patch has saved full-block before/after witnesses, exact float32 landed values and untouched-bit checks. Correctly rounded +4 must actually change its target coordinate. Nonfinite or undelivered injections, failed ordering/landing, failed causal equality or null/self/repeat check stops that phase incomplete with all available evidence retained; do not retry or silently drop a case. The existing model ownership/trace API must be used with capture explicitly after the relevant add.

## Prespecified downstream control and primary outcome

For each combined run, the downstream clean patch replaces the full MLP vector at `(L+1,t)` with its same-input clean donor. Save its realized input displacement vector and L2 norm. For the single active direction control at the same site, let `B` be the saved combined-run vector at `(L+1,t)` and `C` the same-input clean donor. Decode both float32 vectors exactly into binary64, form `d_i = binary64(C_i) − binary64(B_i)` in that order, and cyclically shift by one channel so `r_i = d_(i−1 mod 768)`. Form `z_i = round_binary32_RNE(binary64(B_i) + r_i)`; use these fixed binary64 operations and one final round-to-nearest-even float32 conversion. In the control forward require its pre-patch vector to equal the saved `B` bit-for-bit, patch exactly to `z`, and retain planned and landed vectors, untouched bits, the clean-patch and control realized L2 norms and their cosine alignment. Do not redraw, choose a favorable direction or use multiple samples. Let `q` be the realized clean-patch norm and `q_rot` the realized control norm; call the control norm-matched only when both are finite and nonzero and `abs(q_rot − q)/q ≤ 0.001`. Otherwise retain the action and report it as unmatched or undefined. Report its own `1 − mean(D_control)/mean(D_combined)` with the same undefined-denominator rule as the primary outcome, along with per-case values and match flags. Equal input L2 does not imply equal functional dose, and one rotated direction is not a null distribution.

For each held-out cell, let `D_combined` and `D_downstream` be the full-vocabulary L2 distances at `t` from clean. Define `R = 1 − mean(D_downstream)/mean(D_combined)` over all eight held-out cases, with the aggregate undefined if the denominator is at or below `1e-12`. Report every per-case distance, ratio, undefined denominator, target effect and rotated-control effect. The prospective descriptive pattern is supported if all three cell denominators are defined and strictly more than 25% reduction (`R > 0.25`) occurs in at least two of the three cells. If any cell denominator is undefined the overall rule is ineligible; otherwise fewer than two passing cells is unsupported, not evidence of equivalence or no effect. These three fixed layers share prompts and are not independent population samples.

## Anonymous secondary selector and baselines

For each cell, the selector receives only numerically ordered arrays of the eight discovery cases' combined and candidate-patch `M = logit(318) − logit(389)` values. It gets no role, coordinate, cell label or true-site metadata. Candidate IDs and their publicly known mappings are fixed before outcomes:

| Target layer L | Q1 | Q2 | Q3 | Q4 |
|---:|---|---|---|---|
| 1 | downstream | distractor | source | upstream |
| 2 | source | upstream | distractor | downstream |
| 3 | upstream | downstream | source | distractor |

For each candidate compute the mean absolute patch effect `mean(abs(M_patch − M_combined))`. Select the largest strictly positive mean, breaking an exact tie by Q1–Q4 order; abstain if all are zero. This is patch-impact ranking: an overshooting downstream patch may beat the source. Freeze and publicly commit all three rankings, tie/abstention outcomes and the mean signed `M_patch − M_combined` prediction for every candidate before any held-out forward. Report whether each selected candidate is the known source and compare every candidate's held-out prediction MAE with the zero-prediction MAE on identical cases; strict improvement is required and ties do not count.

For the same cases, compare first changed layer/position in a clean-versus-combined post-add trace, greatest mean clean-versus-combined activation L2 at the candidate coordinate, a constant Q3 (which selects the known source in two of the three publicly mapped cells), a fixed random Q chosen by SHA-256 of `public-distractor-repair-v1|L` (first eight bytes big-endian modulo four, giving Q1/Q1/Q4 for L=1/2/3), and an openly position-aware causal-mask reference that returns the set of the three scored-position candidates and excludes the future coordinate. That set retains the source but is not scored as a unique-site selection. Ties use Q order; no observed change is abstention. The fixed random choices are one realization and the analytic uniform reference is 1/4 per cell, not a significance test. First-change points to the earlier distractor in this design; the causal mask explains why it is irrelevant to `t` but does not identify a unique later source.

A deliberately restricted activation-only record containing just the two unlabeled changed/not-changed booleans for the adds has two true entries when delivery succeeds; its fixed tie rule abstains. Without any additional role metadata or prior, those booleans are unchanged by swapping the two role labels and cannot assign target versus distractor. This is not an impossibility statement about full activation traces or a representation that includes token positions and causal structure; the public Q mapping itself supplies a prior.

## Fixed schedule, bounded evidence and publication

Per prompt run one clean, one null/self, one distractor-only, then for each L in order: target-only, combined, Q1–Q4 patches, the single downstream permutation control, and a downstream self-patch, then one clean-repeat. This is 28 forwards per case, 224 per phase and 448 total. The order and candidate mapping are fixed, with no retries or outcome-driven case changes. Save both full logits, candidate/trace values, full landing and unchanged-coordinate witnesses, ordered action/journal records and failures. Before each forward, enforce its declared worst-case evidence reserve; if the fixed 32-token bound and exact schedule cannot fit, stop before model use. Each phase is capped at 256 MiB raw and 16 MiB metadata. Place new checkpoint staging, outputs, temporary files and bytecode prefixes on E:; require at least 2 GiB there before loading, 1 GiB plus next-action reserve before each forward, and 256 MiB on each source/cache volume. Keep a direct-child receipt with file hashes. An absent receipt or uncertain child does not establish that it stopped and blocks retry until reconciled.

Independently review this protocol and the later fixed cases, executor, exposure check and saved evidence. Publish discovery and its selector/prediction freeze before held-out; a local hash is not proof of public chronology. Publish all positive, negative, undefined and incomplete outcomes. Public input comparisons and cooperative direct-process receipts do not attest all execution or unknown exposure.
