# DistilBERT agreement exposure status v6

This append-only snapshot extends v5, whose canonical SHA-256 is
`34db577a211d6036e7803b8cdb6aaab35687b56974de8cf1ac6bccbf3272fb3c`.
The v5 snapshot and its 96 rows and earlier evidence entries remain unchanged.
The v6 canonical payload SHA-256 is
`0d0b37539c30c8ff25ea5b1cfa95ecd4587f048434348a1c91ec010418553afb`.

The matched-dose protocol was frozen at source commit
`168939c651221e52bcc3a07bbe37c267ced49022` before the public cases were
assembled. The unchanged pre-run case manifest has SHA-256
`5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb`.
The manifest's original status is a historical pre-run statement: its 64
public evaluation rows now appear, with exact input IDs and masks, in the
completed-forward records of the cited CPU attempt. The v6 overlay marks all
64 **observed in cited evidence**. They are not unexposed inputs for a later
study. It preserves the manifest itself and adds these rows to the v5 inventory.

The study used source commit `4ac00abf6938b843abc5015fa83e7ad60d993262`
and attempt `6ce9531f5be90f46574ba6771851817c`. Its published evidence commit
is `48b490ec1dc5fe146fa8f752a7cc4a6be18e9aa7`, under
`results/bert-matched-dose-v1-20261029/`. The
controller's provisional complete-candidate record has SHA-256
`c5045afb6b7dbe518517a8156e52469055d9bd578b33524c7c8beb1a3e7e0eb2`;
the separately recorded local controller exit observation has SHA-256
`ce014315ebb7bb6e54c784a0241f28f76dc2dd751cd07e220b98df76261bc996`.
The compressed journal SHA-256 is
`1a405510648a649b7561a88fa2a770532fcafb2e6e87d587e038eada99033495`,
and its uncompressed 6,369,065 bytes have SHA-256
`9f482a369742960e24273d595d33d2a5fdc8ba2dfc82f1d65f6046266301a3ca`.
It records 480 started and 480 completed forwards, 160 passing controls,
and 32 dose plans. The owner is recorded reaped with exit 0; the controller
PID and exit 0 were recorded separately by a local wrapper. These saved
records and observations do not independently attest that execution occurred
as described.

All cases were public and nonblind. The cited prior records and pre-run
manifest do not certify complete local or external exposure history or
pretraining. This overlay accounts for cited records; it does not authorize a
rerun, establish statistical independence, or prove generalization. No model
is loaded or forwarded to create or verify this snapshot.
