# FLAN-T5 sentiment: CPU development screen

The fixed development screen produced the assigned strict greedy label on
**64/64 inputs** and **32/32 positive/negative pairs**. Its prespecified
single-token, exact-character-span alignment diagnostic was **0/32 pairs**:
the tokenizer offsets included a leading space. Both are part of the result.
An explicitly post-screen check that also accepts exactly one leading ASCII
space found **32/32 pairs**, with three clean same-polarity donor candidates
per case. That secondary diagnostic does not change the failed fixed
alignment rule or establish that future held-out cases will be eligible.
No activation intervention was run in this screen.

## What was tested

The operator reported that the [screen protocol and runner](https://github.com/EastonSell/model-looking-glass/tree/6f7493f0db5114ead1336bef2383dd3b20d5e582)
were published before this recorded attempt; no separately trusted timestamp
of the exact source head was retained. The fixed examples cross two
instructions, four subjects (`book`, `movie`, `meal`, `service`) and four
positive/negative adjective contrasts. The 32 pairs are repeated contexts
of four lexical contrasts, not 32 independent lexical families. One
instruction is also interpretable as a yes/no question, so the exact bare-label
score includes output-format compliance. These public development cases are
exposed for later work; model pretraining exposure is unknown.

The pinned model was `google/flan-t5-small` revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`; its seven-file asset manifest
has SHA-256 `d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083`.
The saved run records CPU float32, deterministic eager attention, one
intra/inter-op thread, offline loading and no decoder cache. The explicit
greedy loop saved the full vocabulary logits at every decoding step and did
not force EOS. All positive outputs were IDs `[1465, 1]` and all negative
outputs were `[2841, 1]`; the final `1` was an emitted EOS. Both fixed clean
repeats agreed bit-for-bit on generation and both teacher-forced target logits.

Separate teacher-forced scores summed the log probabilities of each complete
`positive` or `negative` token sequence **including EOS**. The sign of
`S(positive) - S(negative)` agreed with the assigned label on all 64 rows;
its sign-adjusted range was 3.8502–6.4742 nats. This is a contrast between
two exact sequence log scores, not a calibrated class probability or the
observed greedy output. Raw full-vocabulary logits are retained for both
generation and teacher forcing. The behavioral feasibility rule passed on
its fixed counts: 16/16 pairs per instruction and 8/8 per subject and per
adjective contrast. A successful behavioral screen does not itself show that
the proposed single-token intervention is feasible.

## Descriptive post-screen geometry

Using only saved encoder bytes, the [geometry report](reviews/encoder-geometry-post-screen.json)
groups the exposed examples into 16 four-case groups. The minimum over the
specified opposite-word, same-polarity word, instruction and subject donor
distances ranged from 0.309744 to 0.616159; its ratio to the smallest natural
opposite-word distance in the same group ranged from 0.113441 to 0.198993.
The minimum occurred at the instruction site in six groups and the subject
site in ten. An independent saved-byte reconstruction checked these numbers.
They are not an intervention effect, a power estimate, or a prediction of
the radius on fresh held-out inputs. The controls can carry relevant context;
equal input-space L2 would not guarantee equal functional disruption.

## Evidence and replay

The local operator recorded one direct child with exit code 0, successful
reaping, no timeout and an empty child bytecode prefix. A separate outer
record reports operator exit 0 and an empty parent prefix. The operator start,
exit and logs are in [receipts](receipts/). The spawn receipt contains a
process ID and remains private; the public [redacted projection](receipts/operator-spawn-redacted.json)
records its original SHA-256. A projection and local receipts do not
authenticate the original process, establish trusted chronology, or prove
whole-process-tree cleanup. The outer exit record does not hash-bind the
inner receipt or outer stdout. The separate [custody review](reviews/custody-audit.json)
checks the saved source, asset and receipt consistency within those limits.
The copied custody-audit script requires the private original PID-bearing
spawn receipt and local assets; the public projection alone cannot rerun it.

The closed child directory comprises 330 raw tensors, `events.jsonl` and
`terminal.json`, totaling 53,340,549 bytes. The 18 ZIP parts contain
45,774,592 compressed bytes. [The manifest](parts/manifest.json) records
each part and tensor's byte length and SHA-256, plus both journal records;
the independent [numerical review](reviews/numerical-audit.json) replayed all
330 saved forwards, both fixed repeats and the 64-case, 32-pair rule. Its
float32 log-softmax calculations use the local CPU PyTorch kernel; its
separate binary64 diagnostic has maximum per-token difference 2.34e-6 from
the recorded float32 values. Saved-byte consistency does not independently
prove model execution or private-input separation.

With this result directory checked out, the following reconstructs the exact
original child layout into a **new** directory under a trusted parent. The
extractor checks each part, member and record against the manifest, and refuses
to overwrite the destination. The independent replay requires the pinned
tokenizer assets and matching Python dependencies; it loads no model checkpoint.
Supply fresh output paths and hide CUDA as shown for a POSIX shell:

```sh
python -B unpack_evidence.py --output NEW_CHILD_DIR
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -I -B replay/audit_flan_sentiment_saved.py \
    --run-dir NEW_CHILD_DIR --asset-dir PINNED_ASSET_DIR \
    --source-commit 6f7493f0db5114ead1336bef2383dd3b20d5e582 \
    --output NEW_REPLAY.json
python -B replay/flan_sentiment_offset_diagnostic.py \
  --attempt NEW_CHILD_DIR --output NEW_POST_SCREEN_OFFSETS.json
python -B replay/encoder_geometry.py \
  --attempt NEW_CHILD_DIR --output NEW_POST_SCREEN_GEOMETRY.json
```

The exact original `events.jsonl` SHA-256 is
`eb17ca8482e0c28ed4d6146285aaf33fa63db3437f0fb308c5becfa5e50d3d6b`, and
`terminal.json` is
`843a7140cdd7914fb2e5d6a0c382b05f4eb721b561f71f70d04fcdb4ff1d1e8f`.
The independent review and the post-screen reports retain per-case outputs and
failed fixed alignment counts. The original PID-bearing receipt, local
checkpoint cache and process are not included in this public package.

The earlier [held-out FLAN translation study](../flan-t5-heldout-response-cpu-v1-20261029/)
had an inconclusive fixed primary result and used a teacher-forced translation
metric rather than this greedy sentiment behavior. Neither its outcome nor
this small screen establishes a unique model mechanism or general performance
outside these fixed cases.
