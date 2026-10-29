# Public DistilGPT-2 final-position head ablation

The public [development screen](../results/distilgpt2-copying-dev-v1-20261028/README.md)
ranked layer 3, head 10 (zero-based) first by the difference in attention from
the final input position to the earlier target between repeated and shuffled
examples. This follow-up tests whether zeroing that head's final-position
output changes the repeated-versus-shuffled next-token contrast on **new
public examples**. Attention alone did not establish a causal contribution.
The development source was `55d11a53eb34a13eb6b14e19a5bd0c47172edda5`;
the exact development result bytes have SHA-256
`8914c014bb57e37bd662f73f9f87f1c2431ad6f04a084a2920e44b68b89f0e5d`.

The entire fixed [case and action inventory](../experiments/distilgpt2_copying_head_eval_v1.json)
is published before the first follow-up forward. It lists eight disjoint
six-token word groups, their exact pinned tokenizer IDs, 32 ordered prompts,
and the next-token target for each. Each family has a repeated input
`a b c d e f a b c d e` and three first-block shuffles preserving the token
multiset, suffix, target and earlier target position. The 48 evaluation token
IDs are distinct from one another and from the 48 screen IDs. Before running,
verify each full text against the pinned tokenizer and verify all 32 full rows
against the current reviewed known-exposure inventory; publish an append-only
reservation. These are public examples, not a blinded holdout; known-history
checks cannot establish absence from unknown pretraining or other exposure.

Use the same pinned offline `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, eval mode, float32 and no KV cache.
This version uses CPU, one intra-op thread, and the prior cooperative
CPU/network guard. Publish and independently review the exact executor source
before any follow-up forward, and record its committed identity and the
intended runtime configuration in a pre-run manifest. Record and check the actual model,
weights, tokenizer, source, device, runtime, backend, numeric policy and
assets, and recheck state and relevant execution settings throughout. No
cross-device equality is assumed.

For every case, in order, perform exactly these five forwards: clean, a no-op
hook, candidate ablation, control ablation, and clean after hook removal. This
is 160 planned root forwards. The action is applied to the input of
`transformer.h[3].attn.c_proj`, where the `[1,11,768]` input concatenates 12
heads of width 64. At the last input position only, set channels `[640:704]`
to zero for head 10, or `[448:512]` to zero for head 7. No-op returns an
identical input. The fixed layer-matched control head 7 was chosen from the
ordered heads excluding 10 using the big-endian SHA-256 integer of ASCII
`MLG-copy-head-control-v1`, a NUL byte, and the unsigned 64-bit big-endian
seed `20261029`, modulo 11. The digest is
`7a864639743da70e88dcd2c5f3b6bf79cf2942c70a27d8889395fedc57c309fc`.
This control matches the layer, not activation magnitude or attention.

Record the complete final logit vector per forward and the full c_proj input
before and after each hook action. Check action count, exact shape/dtype/device,
the requested 64-element zero slice, and bit-identical untouched components
and all other positions. Retain each selected slice's pre-ablation norm and
number of changed elements to expose weak or zero interventions. Check no-op
is bit-identical, and require its entire final-position float32 logit vector
and the post-removal clean vector to match the same-runtime clean vector
bit-for-bit. Verify
hook absence/ownership and cleanup, and mark the attempt incomplete on failed
checks or interruption while retaining completed records and available partial
evidence. A post-hook witness proves the delivered tensor at that hook under
these cooperative assumptions; it does not establish behavior against hidden
independent hooks or native mutation.

Calculate target log probability from each full saved float32 logit vector
using binary64 log-sum-exp with a maximum subtraction; retain target logit,
log probability, rank `1 + count(logit > target)`, and top-token ID with the
lowest ID breaking ties. For each family and each ablation, let `delta` be
ablated minus clean target log probability. Its selective statistic is
`delta(repeated) - mean(delta(three shuffled controls))`. A negative value
indicates the operation reduced the repeated-versus-shuffle contrast. Report
the eight candidate and control statistics, their means and the paired
candidate-minus-control differences, plus every raw per-case result and
negative or weak outcome. A negative candidate-minus-control difference alone
does not establish a candidate decrease: report the candidate statistic
itself and the paired difference separately. A failed control makes the
attempt incomplete, not a negative scientific outcome. No significance or
population-generalization claim is prespecified for these eight public families.

Zeroing a head output may be out of distribution and can change downstream
processing. A difference between the chosen head and this one layer-matched
control supports only sensitivity to these specific interventions on these
examples; it does not establish an induction circuit, a unique mechanism, or
absence of compensating pathways. Retain all failed attempts, controls, and
the selected head even if results are null or opposite to expectation.
