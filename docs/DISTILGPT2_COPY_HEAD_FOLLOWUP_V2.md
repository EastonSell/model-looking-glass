# Public copying-head expansion: protocol before case assembly

This prospective CPU follow-up asks whether the final-position effect seen for
one attention head is specific to that head and position on another small set
of public sequences. The [first public ablation](../results/distilgpt2-copy-head-eval-v1-20261028/README.md)
found a mean −0.6381-nat change in the repeated-versus-shuffle contrast for
layer 3 head 10, versus +0.1298 for head 7; the repeated target stayed top-1
in all eight families. Those observations informed this design. They do not
establish a complete circuit or unique mechanism.

This document fixes the arms, selection rules, case-construction constraints,
action schedule and outcomes **before the next rows are assembled**. It does
not contain or generate those rows. Before model execution, publish and
independently review a separate canonical case inventory, an updated
known-exposure reservation, the executor source and its fixed runtime. The new
cases will be public, so this is an exploratory follow-up, not a blind or
independently sampled benchmark. Unkeyed hashes and local records do not prove
an external timestamp or absence from pretraining.

Use eight new families, each with six distinct one-token words
`a,b,c,d,e,f`. The fixed repeated input is `a b c d e f a b c d e`, with
target `f`. The three controls use first-block index orders
`(4,2,0,3,1,5)`, `(1,3,4,2,0,5)`, `(3,0,4,1,2,5)`, followed by suffix
`(0,1,2,3,4)` in each case. A separately published case inventory must contain
the exact texts, token IDs, targets, group IDs and 32 rows. IDs are
`copy-followup-01-0` through `copy-followup-08-3`; variant 0 is repeated and
variants 1–3 are the listed shuffles. All 48 new token IDs must be unique and
distinct from both prior copying datasets' 96 token IDs. Verify actual pinned
tokenization, all full-row relationships against the current reviewed
known-exposure inventory, and reserve the rows before any forward. These
checks do not establish that unknown exposures never occurred.

At minimum, check every proposed full token row against every row in v7,
whose canonical inventory digest is
`e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344`.
Reject equality or a complete-row prefix in either direction, both against
known rows and among the 32 proposals. If a later reviewed inventory exists
when cases are published, check it too and bind its exact version and digest
in the case record; retain v7 as a baseline. Resolve a structural collision
before any new model forward, or abort the attempt. Never revise cases in
response to these follow-up model outcomes.

All sites are in layer 3 (zero-based), at the input of `attn.c_proj` on the
pinned 12-head, width-768 DistilGPT-2. In canonical arm order, zero only the
following 64-value head/position slices:

1. Head 10, final position 10 (the prior candidate).
2. Head 2, final position 10 (second in the original public attention ranking).
3. Head 7, final position 10 (the prior layer-matched comparator).
4. Head 3, final position 10 (first additional seed-selected comparator).
5. Head 1, final position 10 (second additional seed-selected comparator).
6. Head 10, earlier target position 5 (a positional comparison).

All indices are zero-based. For each additional comparator compute
`SHA256(b'MLG-copy-expanded-control-v1\0' + seed.to_bytes(8, 'big'))`,
interpret the digest as a big-endian integer, reduce modulo the number of
ordered remaining heads 0–11 and select that index. For seed `20261030`,
exclude heads `{2,7,10}`: digest
`c8c677ecc975438c531a4924bde114be5ce725033bb74223bf4c3d8b4a94e89e`
selects head 3. For seed `20261031`, exclude `{2,3,7,10}`: digest
`1a0b56b4d8e52e240475c432b3cdb636daf3690f34a0187b38a9c963441cf329`
selects head 1. These are two fixed layer-matched comparators, not an
activation-matched control set or a randomization distribution.

For every case run genuine unhooked clean, no-op, all six zero arms, and
genuine unhooked clean-after: **288 root forwards**. Rotate the six-arm list
left by `(zero_based_family + 2 * variant) % 6` before execution. Every
variant spans all six rotations across eight families, with two rotations
repeated. Family and case order remain fixed; the rotation is deterministic,
not random, and does not eliminate order effects.

Use `distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`
offline, CPU float32, eager eval, no cache, one intra-op thread and a reviewed
cooperative CUDA/network guard. Fix and record the exact interpreter,
dependency/source hashes, backend/numeric policy, model state and buffers,
CPU seed and runtime before forwarding; recheck relevant settings and hook
ownership throughout. Verify tokenization and the intended 12×64 head layout.

Retain the complete 50,257-element final-logit vector for every forward and
full `[1,11,768]` projection input before/after and projection output for each
of the seven hooked actions. Verify invocation, expected position and exact
slice, zero delivery, all other bits unchanged, and the slice's pre-ablation
L2 norm and changed-element count. Require no-op and clean-after final logits
to equal unhooked clean bits, require the hook to be removed, and retain
completed/available partial records if any check fails. A failed control or
uncertain process exit is incomplete, not a negative scientific result. One
complete attempt contains 57,896,064 logit bytes plus 22,708,224 witness
bytes, **80,604,288 raw tensor bytes**; host-disk copies, transient memory and
model memory are separate and need actual headroom. No automatic retry is
implied after an uncertain attempt.

For each arm and family, compute each case's ablated-minus-clean target log
probability from its saved logits, then `D = delta(repeated) −
mean(delta(three shuffles))`. Report all case deltas, all eight family D
values, the mean D, repeated and shuffle components, target ranks/top-1, and
head-slice norms and changes. Report paired family differences between head
10 at position 10 and each other arm, including its position-5 comparison.
All signs, weak/null/opposite effects and failures remain visible; no
significance or population-generalization claim is prespecified.

For each comparison, if a comparator or the earlier-position arm has an
equal or more-negative mean D than head 10 at the final position, that
comparison does not support a selective final-position head-10 effect.
Preserve family disagreements and near-zero paired differences, rather than
classifying them as decisive wins. Earlier and final slices may have different
norms, so the position comparison is not magnitude-matched. Even if all fixed
comparisons favor the candidate, they do not imply significance or a unique
head mechanism.

An earlier-position ablation can affect the final prediction through later
layers and is not equivalent to zeroing a final-position value. Zeroing is
potentially out of distribution, multiple heads may compensate, and fixed
public examples and deterministic order constrain interpretation. Even a
replicated difference supports only the tested interventions and does not
demonstrate a complete induction circuit.
