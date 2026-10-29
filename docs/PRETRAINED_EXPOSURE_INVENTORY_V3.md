# Known DistilGPT2 development exposures: inventory v3

This version preserves the exact v2 predecessor payload SHA-256
`e64bc0b875fb295f03d7ad4d5180f6978d1d632487a98ea2824df01844ea4fb1`.
Its own canonical payload SHA-256 is
`76baf0c8996bf2872a3a1b8fd0fe2734e15525ce0f7014e54811145cb98ce912`.
It changes observed status and provenance without changing any token row:
the same 57 records yield the same 204 distinct nonempty forbidden prefixes.
The categories are 40 saved observations, three operator-reported
observations, two planned reservations, four reference-only rows and eight
synthetic-only rows. Some saved observations also have later GPU provenance.
The inventory is not an independently certified complete access history.

The 16 public agreement-discovery rows reserved in v2 were subsequently
forwarded on the pinned model on CPU. The frozen public manifest was commit
`0e5e3ae68d8dc4fba51e4aadf378e4c3dac27e96`, SHA-256
`1b1adf4d72163bfa3e0399ed8ee92eba6b5c0eb82a47edf28cecde5ffb1deafe`.
The reviewed raw evidence is in
`results/distilgpt2-agreement-preflight-v1-20261028/` at public commit
`abc1b3500928246152b113bb2821bd1028a596b9`, including the attempt log,
terminal result and full last-token logits. These are public preflight inputs,
not hidden held-out cases; the research protocol and its result contain the
actual criterion and outcome.

The 13 public targeted-edit inputs reserved in v2 were also forwarded on CPU
under original, gradient-edit and random-edit conditions. Their frozen
manifest is commit `ce025e994d28969384bd7fbd07b9e9e8e220c8f7`, SHA-256
`93d953494cca7f2fcf0e7cd8baec35fa9ca1b68945e1e44f183e75e10ed03d1d`.
The saved reference and replay evidence is in
`results/targeted-edit-cpu-v1-20261028/` at public commit
`4beaaf17eadc9715ae355d802fea3194b234aa98`; both terminal records report a
negative decision. Their evaluation prompts were deliberately public and
non-blind. The exposure record does not reinterpret the experiment or its
negative result.

The already-observed `[15496, 995]` bootstrap row was also used in two saved
GPU correctness pilots in commit `082b1c350df8ce82352326dd1490cdd3a5aa7747`.
The v1 pilot recorded failed cleanup and quarantine; v2 recorded a clean
supervised exit. The two previously observed public throughput rows were also
used in the saved quick GPU shared-load run whose privacy-reviewed evidence is
in `results/throughput-gpu-quick-dev-v1-20261029/` at commit
`271d6339fd19fe3c1ffaab2bdee801a0bac78765`. This inventory records input
provenance, not a standalone latency, throughput or memory conclusion.

`looking_glass/pretrained_exposures_v3.py` verifies the immutable predecessor
before producing its own pinned payload. Its strict 8+8/16+16 policy and its
16+16 public-discovery-reuse policy retain v2's exact rules: reject duplicates,
cross-split whole-row prefix relationships, and applicable held-out rows
equal to a known row prefix. Common shorter prefixes are permitted, and a
longer candidate extending an unrelated previously exposed row is not
automatically rejected. The trusted caller must keep the entire input snapshot
quiescent during checking. Exception text is sanitized, but tracebacks and
locals may retain inputs and must remain inside custody.

The scoped disclosure reports the inventory digest and discovery policy; it
does not prove publication, a seal, independent review, or authorization.
Unknown historical access, semantic overlap and pretraining remain unresolved.
Future evidence, new input rows or corrections belong in a new version;
preserve v1, v2 and this snapshot. No hidden case or model forward was created
to produce this inventory.
