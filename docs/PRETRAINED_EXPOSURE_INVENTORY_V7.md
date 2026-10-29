# Known DistilGPT2 development exposures: inventory v7

This immutable successor preserves the v6 predecessor snapshot with canonical
SHA-256 `76803775e6929b4a20463a50bcae839a727b1a875a0d433f7c27c9d1d69f0681`.
The v7 payload SHA-256 is
`e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344`.
It retains all 106 v6 input rows and adds
32 planned public evaluation rows, for 138 records and 962 known or reserved
nonempty prefixes. Of the records, 72 are saved observations, three are
operator-reported observations, 51 are planned, four are reference-only and
eight are synthetic-only.

The 32 public copying-screen rows reserved in v6 are now recorded as observed
in the saved CPU development screen. The source is
`55d11a53eb34a13eb6b14e19a5bd0c47172edda5`. Its manifest SHA-256 is
`ab403154ed12f1776b993cc5d8ed45525e0406e1835af14d041bf596497f8843`,
its result SHA-256 is
`8914c014bb57e37bd662f73f9f87f1c2431ad6f04a084a2920e44b68b89f0e5d`,
and its attempt-log SHA-256 is
`d96e2595041702878e2bcde77d07054bbf93e698e6859a8682fdeb4123cf8bf2`.
The log records 33 started and completed forwards: one of the 32 distinct
inputs was repeated as an uncaptured comparison.
The audited evidence was published at commit
`357579d87c74a515b60f53fa92819ab010e16dc1` under
`results/distilgpt2-copying-dev-v1-20261028/run/`.
The earlier v6 snapshot remains unchanged and still accurately describes its
pre-forward planned status.

The additional 32 public, non-blind copying evaluation proposals use eight
new six-token groups and the same four fixed matched permutations. The 48 new
token IDs are distinct from one another and from the 48 screen IDs. Their
protocol and case inventory were published at commit
`947332a52aa9db197cd8624f839ed8c50c68154a`; the exact case JSON SHA-256 is
`34bbb68a5acb8615e856a20b77bc3ec93787cc82afa11c9dabb83d7c209baca9`.
No model forward is asserted for these new rows. All 610 v6 prefixes are retained, and 352 new
prefixes are added. The snapshot rejects a full-row prefix relationship in
either direction between a new evaluation row and a prior complete row or
another new row.

Strict 8+8/16+16 and exposed-discovery 16+16 checks reject a held-out full row
when it equals a known or reserved nonempty prefix. They do not automatically
reject a longer candidate extending a shorter known row, or rule out unknown
history, semantic overlap or pretraining. These records and hashes do not
independently prove faithful execution, and known-row checking does not
authorize a study seal or a model forward.

The trusted custodian must bind the actual checkpoint/tokenizer and current
reviewed snapshot before use, retain immutable bytes for reveal, and keep
candidate inputs quiescent during copying. A caught Python exception's
traceback may retain private inputs; keep exceptions and tracebacks private and
publish only sanitized statuses. No private or hidden held-out case was
generated, and no model forward was run, while preparing this snapshot.
