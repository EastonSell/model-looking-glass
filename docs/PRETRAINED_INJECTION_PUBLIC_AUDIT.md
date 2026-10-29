# Independent public injection-discovery audit

`tools/review_injection_discovery.py` independently recomputes the public
CPU injection-discovery analysis from the saved observation bytes. It uses
only the Python standard library and does not load a model, private injection
witnesses, the concealed site, or held-out cases. It is an additional check
of an archive, not the study's selector or a substitute for its custodian.

After an authorized discovery attempt has produced its complete public
archive, obtain the expected request-plan, prediction, and archive SHA-256
digests from the separately verified published artifacts. Run from a trusted,
unchanged source snapshot, for example:

```text
python -B tools/review_injection_discovery.py <public-directory> \
  --request-plan <public-plan.json> --source-commit <full-source-commit> \
  --plan-sha256 <plan-sha256> --prediction-sha256 <prediction-sha256> \
  --archive-sha256 <archive-sha256> --output <new-audit-record.json>
```

The output path must be new and outside the input archive directory, whose
inventory is checked exactly. The audit checks the planned family-final
request schedule and site-independent request identities; all 128 canonical
public records, capture coordinates, donor metadata and tensor digests; public
controls; and the frozen case effects, norms, votes, ranking and baselines.
It verifies candidate forward/reverse and upstream/downstream post-capture
bits against their donors. The self control is compared with the injected
reference under the fixed tolerance; its same-invocation pre-action witness,
the addition and untouched coordinates require the private audit. Finite
IEEE float32 bits are decoded directly into exact binary64 values before the
protocol's tolerance and arithmetic are applied. The saved record contains
the computed aggregate, case summaries and artifact hashes, rather than raw
logits or activations.

The CLI requires all three digests. The Python `audit()` function also permits
them to be omitted for fabricated tests; programmatic callers must supply
independently verified pins when claiming a bound archive audit. The operator
must independently verify those digests, the source, actual forward order,
publication chronology and independence of the public entropy from the site
nonce. The public archive alone does not establish those facts.
The recorded audit-source hash is an end-of-run file observation, not proof
of which bytes executed. Verify the committed source and its working bytes
before and after running it. A pass does not attest the hidden-site addition,
private witnesses, model execution, publication chronology, held-out results,
or an explanation of natural model behavior.

The regression suite uses only a fabricated site and a synthetic CPU model:

```text
python -m unittest tests.test_review_injection_discovery
```

It checks a complete fabricated archive and rejects altered donors, logits,
votes, request identities, malformed typed metadata and omitted CLI digests.
Those tests do not run the pinned checkpoint or allocate study held-out inputs.
