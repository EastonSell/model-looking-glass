# Independent FLAN weight-edit saved-evidence reader

`scripts/audit_flan_weight_edit_saved.py` replays the fixed prospective
FLAN-T5 single-token weight-change study from closed files. It does not load
or run a model. It requires Python 3.11 and the pinned `tokenizers` package.
The reader and its cases, protocol, and execution manifest must be frozen
before looking at study outcomes.
This v2 revision binds source `b3c7e19a5349cd2fb76ef4908407342233c7e240` and
manifest commit `60691c687cc737c081a6f2f27df3b51bcc4d5c87` (SHA-256
`09623ddf8716a1467cc35ef0469f7bb6d2119fdf574fd35aafe13f55e9fa669b`).
Any source or protocol successor needs a newly reviewed reader binding.

For development, supply absolute paths to the repository, published execution
manifest, fixed case JSON, pinned asset directory, closed development attempt,
the exact scope record, and a fresh output path outside the repository, assets
and attempt directories. `--manifest-revision` names the manifest's Git commit.
The attempt and scope must be under the fixed
`E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v2` study root. The
one-shot claim must be at `claims/development.json` there, as in the fixed
owner layout. The v1 attempt and claim are distinct and remain preserved.
The reader verifies that commit's sole parent and every declared source blob,
the protocol/case hashes, all seven asset hashes, and case tokenization. It
reads the five checkpoint rows directly from safetensors bytes and reconstructs
every prescribed float32 candidate and the seeded random direction.

```text
python -B scripts/audit_flan_weight_edit_saved.py \
  --repo ABS_REPO --manifest ABS_MANIFEST --manifest-revision COMMIT \
  --cases ABS_CASES --assets ABS_ASSETS --development ABS_DEV_ATTEMPT \
  --development-scope ABS_DEV_SCOPE \
  --output ABS_NEW_REPORT
```

After a reviewed development handoff has been published, evaluation is
checked with the same arguments plus `--evaluation ABS_EVAL_ATTEMPT`,
`--evaluation-scope ABS_EVAL_SCOPE` and
`--handoff ABS_PUBLISHED_HANDOFF`. The handoff's development files and review
must match their published Git bytes, including ordered split blobs if used.
The output path is never placed inside either closed attempt. Keep the saved raw logits and process
records private when their paths or identities are sensitive; a public report
must be projected and reviewed separately.

The reader validates call/event ordering and contiguous, hashed, finite
full-vocabulary rows; reconstructs greedy IDs, ties, EOS/truncation, decoded
text, label scores and fixed grid selection; and replays every target and
locality. It checks each target's efficacy against each control, separates
token-present locality from token-absent identity checks, and recomputes
first-step full-vocabulary KL and total variation. It verifies candidate,
original and recorded restored-row hashes, state/buffer records and exact
restoration probe bytes for every arm. An unknown exit, failed restoration,
missing row, unexplained substitution mismatch, or inconsistent evidence
produces `incomplete`, not a favorable result. A valid but underfilled or
otherwise scientifically ineligible study is reported separately.
The reader checks the tokenizer package version against the manifest and
checks float32 subnormal decoding/encoding and rounding before checkpoint
decoding, after the tokenizer import, and after replay. Its mode observations
do not prove the absence of transient changes between those points.
It checks the resource log's before/after observation pair for every recorded
model call, both free-space floors and the saved minima, plus recorded parent
and child cache paths. These free-space readings are point observations, not a
disk reservation. The resource and model journals are separate streams, so
their cross-stream ordering relies on the source and owner records.

These are checks of saved observations. They do not prove that a process ran
the claimed code, that the physical model state was restored, or that a
published handoff preceded a model call. The recorded operator, source,
restoration and custody claims need their own review. No unseen pretraining
history, population-wide generalization, GPU profile, or throughput claim is
established by this reader.
`operator-exit.json` is the operator's saved observation of the model child;
the outer operator's own exit is a separate cooperative observation and is not
verified by this reader.
