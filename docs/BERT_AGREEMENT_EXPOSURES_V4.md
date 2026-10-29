# DistilBERT agreement exposure status v4

This append-only snapshot preserves v3, SHA-256
`6945a478e3af37083af055bf20bf65628fa979e0ee8b253ea8bacb475cb9e9f2`.
Its own canonical payload SHA-256 is `1f9ae67040646341097414f5ef6da1f66d0ccc5f0be47deeee36cfe66550e32e`.
It adds the completed GPU development-family profile to the eight family-1
rows already observed in v3. All 96 row identities and status categories are
unchanged: ten cited observed development rows, 22 reserved development rows,
and 64 public evaluation rows not observed in the cited evidence.

The GPU attempt used source `4a597641be89ee094145fbdacee4d3fcdd241e41` and
run ID `1f67a55c190391f693c3f3739de6ff12`. Published evidence is at commit
`bca5a886303e070e90f92c7e23ef301f0ded836f` under `results/bert-gpu-family-profile-v1-20261029/`. The result
SHA-256 is `d7537ab322018794682d3e3ffd93eea83c6175a71d0ea20ecd156dac97cc2289`;
the event journal SHA-256 is
`152e3f463a9bc7b8f8f0ef1e05e9304aa93c521b74ee19e5a2484af149246190`.
The journal records 140 started and 140 completed forwards, plus 48 controls,
all on the eight public family-1 development rows. The reported terminal is
successful with clean supervised teardown. No evaluation row appears in that
journal.

This is a cited-evidence overlay, not an independent process attestation or a
complete exposure history. Exact token-row comparison does not establish
semantic independence, unknown earlier use or absence from pretraining. The
evaluation fixtures are public and nonblind; this snapshot does not authorize
their execution. No model was loaded or forwarded to create this snapshot.
