# Two-client CPU trace smoke

This bounded development smoke exercised the brokered trace IPC added in
[PR #599](https://github.com/EastonSell/model-looking-glass/pull/599) on CPU with
the pinned `distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`.
The two already observed inputs were `[50256, 103, 203, 303]` and
`[50256, 104, 204, 304]`. The harness calls the loader once and supplies one
resident model object to a serial broker; the retained records are not an
independently instrumented count of all loads.

Two distinct clients each received two full 768-element float32 traces at
layer 0, token position 3: one attention output and one MLP output. Each of
the four 3,072-byte captures matched the corresponding vector from a separate
direct-hook reference forward byte for byte. The two reference forwards also
retained their complete `[1, 4, 50257]` logits, which were finite on readback.
The brokered trace API in #599 does not deliver logits from its traced forward,
so those saved reference logits are not a same-forward comparison.

Both clients have distinct team, request, job and transfer identities. The
harness observed the second client queued behind the first on the same broker
owner thread; both clients exited zero and the reported service and host
claims drained. A separately retained direct process handle recorded the owner
exit as zero. `observer-public.json` projects the local observation without
absolute paths or operating-system process identifiers and binds the original
record hashes. The independent saved-byte audit is in `independent-review.json`;
`audit_public.py` checks the public projection and payloads with Python 3.11:

```
python -B audit_public.py .
```

An earlier attempt saved the same two separate reference outputs but its
first trace plan was rejected: the harness had set a one-layer limit for a
six-layer model. Its observed exit code was null, so it remains incomplete;
the corrected run is a separate attempt. `incomplete-attempt.json` retains
the relevant non-path metadata and hashes of the private original observation
and diagnostic files. `first-attempt-smoke.py` was reconstructed and matched
to the original recorded source hash; its old label says "untraced" even though
the separate reference calls installed direct capture hooks.

The recorded source identities are before/after file readbacks under a trusted,
quiescent checkout, and the process and queue facts are cooperative local
observations. These artifacts do not independently attest executed bytes or
establish a GPU, batching, throughput, or general capacity result.


The complete public projection is stored as base64 text parts for transport.
Run `python verify_archive.py` to verify the parts and archive, or
`python verify_archive.py --extract <new-directory>` to reconstruct them.
Then run `python <new-directory>/audit_public.py <new-directory>` to check the
public trace and reference bytes. The index binds the parts and every member.
