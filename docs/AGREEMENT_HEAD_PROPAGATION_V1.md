# DistilGPT-2 agreement-head propagation pilot

This prospective CPU pilot asks whether a previously selected attention-head
intervention has an observable, intervention-dependent contribution through the
next block's MLP output. Layer 3 head 4 was selected after earlier development
results; layer 3 head 8 is the earlier adaptively selected active comparator.
Neither is a random or blinded control. The experiment uses fresh public cases,
not a private held-out set; private-history and pretraining overlap are unknown.
No protected or flagged one-use split is accessed or indirectly compared.

## Fixed public cases

Before any model forward, tokenize these exact four frames, with the specified
two lemmas and their regular plurals. In each frame the four rows are ordered
first-lemma singular, first-lemma plural, second-lemma singular, second-lemma
plural. Each quartet must differ at one subject-token position only, consist of
four distinct subject-token IDs, have equal lengths at most 12, and have ` is`,
` are`, ` was`, and ` were` append exactly token IDs 318, 389, 373, and 547.
If tokenization fails, stop; do not change cases based on model outcomes.

1. `Before the festival, the {subject} near the river`: captain, artist.
2. `After the rehearsal, the {subject} beside the violins`: merchant, poet.
3. `At the workshop, the {subject} behind the lanterns`: captain, merchant.
4. `Throughout the evening, the {subject} near the garden`: artist, poet.

Publish the fixed manifest and a planned-only public exposure reservation
before a pretrained forward. Compare bases and the four answer extensions in
both prefix directions against the current public inventory and known public
pending bundles. Refresh the public check at launch. This is not a complete
exposure certificate, a query of private data, or a pretraining guarantee.

## Intervention sites and doses

Use eager, uncached, float32 CPU DistilGPT-2, one unpadded row per forward and
its final-token position. A head intervention replaces exactly the relevant
64-element slice at the *input* to `transformer.h.3.attn.c_proj`; this is not
the projected attention output. For receiver quartet-local index `i`, use the
other-lemma same-number donor `i XOR 2` and the other-lemma opposite-number
donor `i XOR 3`. The two donor words differ lexically as well as in number.

For one receiver, define `d[h,k]` as donor minus clean receiver head input,
and `W[h]` as the corresponding 64 rows of the attention projection weight.
Its bias cancels when comparing two projection outputs. For the four fixed
head/donor combinations, cap the input displacement at both the full donor
distance and 5% of the receiver's head-input L2 norm. Set one common requested
projected-output L2 dose to the minimum of the four corresponding capped
`||d @ W[h]||` values. Interpolate each donor direction to that common dose,
without extrapolation, and round its replacement to float32. Thus the 5% bound
is on each receiver head-input norm, whereas the matched dose is in projected
output units. Require all finite nonzero distances, a common dose of at least
`2**-20`, and exact source/landing witnesses. At every delivered head edit,
measure the final-token projected-output displacement against that receiver's
clean projection output and require relative error at most `1e-4` from the
common requested dose; retain its actual input dose and verify the cap within
the same tolerance. Equal output norms do not match vector directions or
downstream changes.

Specifically, for each head and donor set `c = min(1, 0.05 * ||r|| / ||d||)`
for receiver head input `r`, and `p = ||d @ W[h]||`. With `p > 0` for all
four combinations, `q_head = min(c * p)` and the replacement is
`round_float32(r + (q_head / p) * d)`. The cap and common dose are computed
from clean observations, before any edit forward. All norm calculations use
the recorded float32 values interpreted in binary64; the real module output
provides the independently checked delivered projected dose.
Promote the saved float32 vectors and projection weights to binary64 before
forming donor differences, the matrix products, caps, norms and interpolation
scalars. Form each replacement and MLP-vector component in binary64, then
round it once to float32. The MLP addition itself is the actual float32 model
operation. Disable denormal flushing and record the runtime numeric setting.

The second component acts at the final-token output of
`transformer.h.4.mlp`, after its projection/dropout and before its residual
addition. Once per receiver, form a fixed additive float32 vector along the
clean MLP-output displacement toward its opposite-number other-lemma donor,
with requested norm 5% of the receiver's clean MLP-output L2 norm. Require a
finite request of at least `2**-20`, donor distance at least the request, no
extrapolation, and actual delivered displacement within relative `1e-4` of the
request. The exact same vector is used for that receiver in both head
backgrounds and all relevant cells. The observed float32 displacement can
differ slightly by background; record it rather than assuming equality.
For clean receiver `u` and donor `v`, the fixed vector is
`round_float32((0.05 * ||u|| / ||v-u||) * (v-u))`; the edited current MLP
output is the float32 sum of its actual pre-intervention vector and this
fixed vector. In the clamp arms it is replaced with the clean receiver `u`.

Capture the final-token outputs of layer-3 attention, layer-3 block, layer-4
MLP and block, and layer-5 block, as well as head-projection witnesses. The MLP
intervention has full-sequence before/after witnesses; unselected positions
must be bit-identical. Each forward retains the 50,257 final-position logits.

## Fixed schedule and prediction

First observe all 16 clean rows and precompute every dose and replacement.
If any head or MLP plan is ineligible, preserve the clean records and report
`dose_ineligible`; do not choose replacement inputs, sites, or doses. For each
row after planning, run head-4 self-patch, head-8 self-patch, clean MLP
self-clamp, MLP-addition alone, then for head 4 and head 8 in that order, and
same-number then opposite-number donor in that order, run native MLP, fixed
addition, and clean-receiver MLP clamp. Finish with a clean repeat. Together
with the initial observations this is exactly 18 forwards per row, 288 total.
Self patches, self-clamp, and clean repeat must reproduce clean logits exactly;
every intervention must retain exact source, landing and untouched witnesses.
Missing, nonfinite, invalid, or failed delivery makes the prediction
incomplete, not a smaller denominator.

Let `m = sign * (logit_is - logit_are)`, where the receiver-number sign is
+1 for singular and -1 for plural. For each head and receiver,
`D_native = m(same donor, native MLP) - m(opposite donor, native MLP)`, and
`D_clamped` is the corresponding contrast when the MLP output is replaced
with the receiver's own clean MLP output. Define the *signed decrement*
`G = D_native - D_clamped`; a positive G does not necessarily mean the
absolute contrast moved closer to zero.

The prespecified pilot pattern is: the clean margin has the correct strict
sign in at least 12/16 rows and at least 3/4 within every frame; among all 16
valid rows, head 4 has G > 0 in at least 10, its mean G is positive, and its
mean G exceeds the mean absolute G for head 8. If baseline is inadequate,
report `baseline_limited`; if the baseline and all deliveries are valid but
the pattern fails, report `does_not_support_fixed_pattern`. The selected
thresholds are descriptive pilot criteria, not a statistical test or proof of
mediation. Report all rows, frame/lemma dependencies, the raw four margins,
the D and G values, head-8 values, and whether absolute contrast increased or
decreased separately. Preserve failures rather than silently excluding rows.

For a true 2×2, A is the opposite-number donor head patch off/on and B is the
fixed MLP addition off/on. On the same receiver, report all four signed margins
`y00`, `y10`, `y01`, `y11` and interaction
`I = y11 - y10 - y01 + y00`, for each head. The unedited clean row is y00 and
the same B-only row is y01 for both heads. The same-number donor and its
addition are additional context controls; the clean clamp is a separate
intervention, not one of the factorial cells. No sign is predicted for I.

Only one dose is fixed for this bounded pilot. Matching at the projection
output does not match later doses; the additive vector and clamp change the
same downstream computation and can interact nonlinearly. An effect of the
clamp or a nonzero interaction is conditional on these interventions and
the remaining residual stream. It does not identify a unique route, prove a
unique agreement circuit, quantify a causal fraction, or imply generation or
population-wide behavior. Retain full-logit context and expose both positive
and negative or inconclusive outcomes. Independent saved-byte review and
observed direct-child closure are required before publishing a result.

This design is motivated by [Vaidyanathan et al., 2026](https://arxiv.org/abs/2606.27510)
on patching interactions and [Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042)
on sensitivity to metrics and corruption methods; neither source establishes
the outcome for this experiment.

## Runtime and evidence bounds

Use the pinned `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, with checkpoint SHA-256
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b` and
the exact tokenizer assets bound in the case manifest. Run offline with
network/model downloads disabled, CUDA hidden, one CPU intra-op and inter-op
thread, evaluation/inference mode and deterministic algorithms. The intended
runtime is Python 3.11.15, Torch 2.8.0+cu128, Transformers 4.57.3 and
tokenizers 0.22.2. Tokenizer-only preparation and fabricated model tests are
not pretrained forwards.

Save raw float32 witnesses and full final-position logit vectors, with
content hashes and event order, so independent readers can reconstruct the
landing, untouched-position, norm, control and score checks. The 288 logit
vectors alone total 57,896,064 bytes. The attempt journal caps retained tensor
payloads at 256 MiB and each forward record at 1 MiB; preserve any partial
attempt and treat a failed or missing terminal as incomplete. Use E:-backed
output and temporary staging and check at least 1 GiB free on that volume
before starting. This point check includes the approximately 356 MB staged
checkpoint copy and the bounded evidence but is not a disk reservation or a
host-RSS bound. Record runtime, source/input identities and a separately
observed reaped direct-child exit; these are cooperative evidence rather than
execution or process-tree attestation.
