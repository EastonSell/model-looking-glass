# Targeted-edit v2 evaluation writer

These private helpers write a **synthetic-only** evaluation attempt while the
live evaluation entry remains disabled. No pretrained checkpoint, evaluation
forward, complete terminal, or experimental outcome is claimed here.

Before writing a row, the writer checks the supplied selection snapshot's
manifest/source, selection commit and terminal identities, distinct run ID,
model/weight and supplemental buffer digests, and recorded runtime against the
exclusive model guard. It checks every selected, random, opposite and 1% v1
control against both its frozen intended-delta digest and its realized-weight
digest before opening the original held-out condition, then repeats each check
before applying that control. The sequential
temporary tensors used for these checks are not a general host-memory cap.
The caller must keep the model, snapshots and update tensors quiescent and must
obtain the snapshot through the committed-selection gate. The writer does not
prove that a constructible snapshot came from that gate, or verify the remote
witness; the future controller must independently establish both before any
held-out forward.

The fixed condition order is original, selected, random, opposite, then v1's
1% reference. Each row has a durable identity/probe journal entry. The model
guard must verify restoration before the condition-state record is flushed and
fsynced; only then does the writer acknowledge the condition and reopen the
guard. A failed condition, write or interrupted return stops the attempt and
retains its partial files. If the acknowledgement's return is lost, a state
record may exist and the guard may have reopened; the controller must stop and
reconcile, never blindly rerun the condition.
Once any evaluation row has been forwarded, those inputs are exposed. A
scientifically fresh retry that changes the plan requires a new, separately
frozen evaluation set and protocol; preserving the partial attempt does not
make the original inputs held out again.

The report is recomputed from the saved rows and selected snapshot with
denormal and runtime checks. It is provisional, including favorable numbers,
and this component never writes a successful terminal. A future supervising
parent must verify exact child identity and reaping, witness and source binding,
model disposal and all cleanup, then independently audit the complete saved
artifacts before publishing. A reader pass is an attestation about saved bytes,
not proof that the model or native process state was disposed.
