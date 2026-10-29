# FLAN-T5 supervised GPU development smoke, draft v1

This is a separately source-frozen development check for the device owner. It
uses only the already exposed public feasibility pair 1, not the locally
withheld pool or its cases. It does not establish a held-out result, general
memory bound, latency benchmark, or service readiness. No run is authorized by
this draft alone; it requires source and independent review, a recovered and
coordinated device, and an external supervisor. Preserve every started, failed,
timed-out or incomplete attempt and its exposure record.

Use the pinned seven-file FLAN-T5-small checkpoint and its manifest from the
CPU feasibility study. Verify and stage exact asset bytes, source and runtime,
one supervised physical GPU identity, native allocator and the numerical
settings in `FLAN_T5_DEVICE_OBSERVATION_V1.md`. The fixed input is public pair
1 from `FLAN_T5_PUBLIC_CASES_V1.json`: source IDs
`[30355,15,1566,12,2968,10,484,1]` and
`[30355,15,1566,12,2968,10,629,1]`, each with an all-one mask, and target IDs
`[4675,1]`. No tokenizer or model forward selects or replaces these rows.

Execute six one-item forwards in this order: capture source A, capture source
B, repeat capture A, patch A with its own captured token 6, patch A with B's
captured token 6, and null token 6 in A. All use the final encoder layer norm,
invocation zero, output selector `[]`, float32, no cache and explicit masks.
Keep both original captures and one current observation under separate active
trace claims. Save complete CPU float32 bit encodings for logits, encoder state,
and available before/after captures, along with exact input IDs, masks, shapes,
metadata, digests and every synchronized memory record.

Require bitwise equality to capture A for repeat, self-patch and null logits
and state, and require the donor vector to differ from A and the donor patch to
change at least one saved logit bit. Exact landing and unchanged other token
positions are enforced by the observer. These are within-device development
checks, not a CPU/GPU equality requirement or evidence of generalization.
Each completed model call is bracketed by a bounded registered-state identity
and bit fingerprint check. A failure remains an incomplete result; it must not
be reclassified as a successful or negative scientific study.

The whole allocator cap plus 256 MiB is held as a conservative logical model
and forward commitment, with separate one-use per-observation claims and a
three-observation ceiling. Check free memory and allocator peaks at synchronized
endpoints. These checks do not guarantee that every allocation fits, prevent
other processes using the device, or bound host RSS. Save incomplete and partial
memory records; absence of a record does not establish cleanup. Release claims
only after known host and device roots are disposed. An owner may attest only
after model disposal, synchronized workspace and allocator cleanup, final
source/asset checks and durable result recording. The supervisor must observe
the exact child exit and complete its own recovery protocol. Process-local
allocator counters do not by themselves prove device-wide or post-exit memory
recovery. Uncertain ownership, cancellation, native-return gaps or failed
cleanup require quarantine and separately reviewed process recovery.

The 16 MiB observation-record limit excludes the separately staged checkpoint,
the manifest and any controller or operator logs. The owner rechecks the
staged assets and records their manifest and payload byte counts separately.
The future controller must check available disk before launch and reconcile
complete or partial staged files and logs; a free-space check is a point-in-time
observation, not a reservation against other writers. A diagnostic-write
interruption leaves both exceptions rooted, and a cancellation is reflected in
the owner exit even when it arrives while recording an earlier failure.
