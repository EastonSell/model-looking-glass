# Known DistilGPT2 development exposures: inventory v2

This immutable snapshot preserves
[`PRETRAINED_EXPOSURE_INVENTORY_V1.md`](PRETRAINED_EXPOSURE_INVENTORY_V1.md)
and binds its canonical SHA-256
`6f21bffeacbe2323c21e53378161443eae8ba28b7210f30e70fd353497e87c8f`.
The v2 canonical payload SHA-256 is
`e64bc0b875fb295f03d7ad4d5180f6978d1d632487a98ea2824df01844ea4fb1`.
The snapshot contains 57 records: 11 saved observations, three
operator-reported observations, 31 planned reservations, four reference-only
rows, and eight synthetic-only rows. They yield 204 distinct nonempty token
prefixes. These categories distinguish evidence of use from conservative
reservations; neither the list nor its hashes certify a complete history.

The public `[50256, 50256, 50256, 50256]` row was planned in v1 and later
executed on CPU in a pinned-model intervention smoke and an independent
same-host replay. The source revision was
`3f8de5d7fb299e4878dbf75f86bb58f2249d6a37`; the privacy-reviewed results and
publication map are in `results/distilgpt2-intervention-dev-cpu-v1-20261028/`
at commit `c6f5bf6bd17c440432af14655cc0ff505b45337e`. The publication map
links the redacted evidence to raw attempt and result hashes. The hashes
identify records, not a proof that an execution occurred or an authentication
of the process. This is a software-control input, not a causal-study held-out
prompt, GPU measurement, or generalization result.

This snapshot also reserves the 16 public agreement-discovery rows announced
by their study owner before a model forward. Token IDs were checked offline
against the pinned tokenizer. After a tokenizer-only alignment check, the
original baker/bakers proposals were superseded by writer/writers; those two
older proposals remain reference-only reservations. Thirteen proposed public
parameter-edit discovery, evaluation, and locality rows are also reserved.
The parameter-edit evaluations are public and non-blind, held out only from
the edit-fitting procedure. Planned status is the status recorded by this
snapshot; later model forwards or additional provenance require a new version
rather than rewriting this payload. No hidden study split was generated or
examined for this inventory.

The model and tokenizer identity are pinned to `distilbert/distilgpt2` at
revision `2290a62682d06624634c1f46a6ad5be0f47f38aa` and the asset hashes in
the payload. The caller must independently bind the actual loader and
tokenization to that identity. Other checkpoints or tokenizers need their own
inventory; equal integer IDs alone do not establish equal inputs.

`looking_glass/pretrained_exposures_v2.py` builds a fresh canonical payload,
checks the v1 predecessor, and pins its own digest. It offers two explicit
policies:

- `check_known_splits` accepts exactly 8+8 or 16+16 candidate rows and rejects
  known or reserved prefixes from both splits.
- `check_exposed_discovery` accepts exactly 16+16 rows, permits deliberately
  public discovery reuse, and reports its aggregate known-or-reserved overlap
  count: the number of discovery rows equal to a recorded prefix, not every
  row sharing or extending a known context. It rejects known or reserved
  prefixes from the held-out rows. It does not attest that discovery rows are
  the ones specified by a protocol.

Both policies reject duplicate rows anywhere and full-row prefix relationships
between the two splits, even if a relationship is otherwise unknown. They
allow ordinary common starting tokens. The known-prefix check rejects a
candidate equal to any nonempty prefix of an inventoried row. It does not
reject every longer candidate that extends an inventoried row, and it cannot
detect semantic similarity, unseen historical exposures, or pretraining data.
Thus it supports a bounded disclosure, not a guarantee of independence.

`prepare_preseal_disclosure` requires a matching check, an exact protocol hash,
and explicit acknowledgment of historical uncertainty. Its frozen output
records the checking policy, candidate counts and inventory digest while
stating that the seal, independent review and execution authorization are
unverified. This is a trusted-caller report, not an authenticated proof: the
caller must preserve a properly ordered, reviewed commitment and its own case
and source binding. The legacy `require_preseal_readiness` remains fail-closed;
it does not prohibit a separately reviewed study with a scoped disclosure.

The result and exception messages do not contain candidate rows, but Python
exception objects and tracebacks can retain candidate locals. Keep inputs,
exceptions and tracebacks inside the trusted custodian and serialize only a
sanitized status. The trusted caller must keep both candidate splits quiescent
throughout the check: copying Python lists in chunks is not an atomic snapshot
against concurrent mutation. This cooperative boundary does not isolate
processes or accounts. Later development, execution, newly discovered
history, or revised evidence should produce a versioned successor while
retaining v1 and v2.

