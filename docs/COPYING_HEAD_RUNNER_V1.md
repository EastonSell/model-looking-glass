# CPU runner for the public head-ablation follow-up

This runner implements the fixed [head-ablation protocol](DISTILGPT2_COPY_HEAD_EVAL_V1.md).
Its 32 disjoint public cases are exploratory, not a blind or independently
preregistered holdout. The selected and control heads were chosen before the
rows were assembled, but that ordering was not committed at the time. The
candidate, control, scoring rule and cases were published before this run.
The single deterministic same-layer control is not a randomization distribution.

Before forwarding, the runner verifies the published case bytes, the v7
exposure inventory, the pinned screen result and its raw attention ranking,
checkpoint assets, the clean source revision, GPT-2 implementation source
hashes, and the fixed CPU runtime. It requires the previously reviewed
public-case reservation; the inventory records planned public exposure and
does not prove absence from pretraining or other unknown exposure. The
cooperative CUDA/network guard and source checks do not attest against native
code bypass or independent mutation of a running process.

Each case runs five forwards in the fixed order: clean, no-op, head-10 zero,
head-7 zero and clean-after. Clean and clean-after are genuinely unhooked.
The three hooked forwards save the exact input to layer 3's `attn.c_proj`
before and after the action and the projection output. The input to all
three actions must match, untouched components must match bit for bit, the
requested final-position 64-channel slice must be zero, and clean/no-op/
clean-after full final-position logits must match bit for bit. The runner
records each zeroed slice's original L2 norm and count of changed float32 bit
patterns. This zeros the head's projection input; it leaves the projection
bias and all other heads in place and does not remove the head's Q/K work.
The actions and cases run in a fixed order rather than a randomized or
counterbalanced order. The controls test for several changes during the run,
but pairing alone does not remove all possible order effects.

The attempt directory is new and outside the checkout. `manifest.json` binds
the source, cases and runtime; `attempt.jsonl` records each start and completed
forward. Each `copy-eval-NN-V.f32` is a headerless little-endian float32 stream.
The corresponding completed records provide offsets, byte lengths and SHA-256
hashes for the 50,257-value final logit vector and, for hooked forwards, the
three `[1,11,768]` witnesses. `result.json` retains every score and the eight
candidate, control and paired family contrasts, including negative results.
An incomplete or interrupted attempt retains its prefix and `failure.json`
when the process is able to write it; it must never be resumed in place.
An external supervisor must record a zero process exit and no failure file,
and independently replay the saved bytes and scores before treating a result
as complete. A result file alone cannot establish successful process exit.
In failure records, `submitted_forwards` counts attempts entering the forward
helper and may include an early failure before the model is called;
`returned_forwards` counts complete payloads returned by that helper and may
exclude a model call that returned before helper validation failed.
`payload_returned` and `payload_recorded` use those same helper/persistence
boundaries and do not independently count model-root returns.

The fixed 160 forwards retain exactly 41,896,576 raw tensor bytes: 32,164,480
for final logits and 9,732,096 for the hooked witnesses. The runner enforces a
42,000,000-byte aggregate raw payload ceiling and at most 302,404 retained
tensor bytes per hooked forward, plus initial filesystem headroom. These are
logical payload bounds, not a bound on model parameters, the full transient
`[1,11,50257]` logit tensor, intermediate activations, Python objects, filesystem
copies or process RSS. A full forward may therefore need substantially more
memory than its retained payload. The output has no live broker, IPC, GPU or
remote-custody integration.
