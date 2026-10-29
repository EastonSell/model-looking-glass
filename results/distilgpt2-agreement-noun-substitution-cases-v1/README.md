# Agreement noun-substitution: fixed public cases

This tokenizer-only bundle contains 48 target inputs and six subjectless
fragments across two frames and three contexts. No follow-up model forward
had occurred at case publication. The cases are public and nonblind; they
were selected without model scores after the [fixed protocol](../../docs/AGREEMENT_NOUN_SUBSTITUTION_V1.md)
was merged at `d4376c19caae67ff8c946f415d74cdedf7332bc5`. The listed word
forms were inspected with the tokenizer during protocol review.

The regular noun pairs are `hand/hands` and `arm/arms`; the irregular pairs
are `foot/feet` and `tooth/teeth`. Each frame/context contains all eight
forms at the same token position, with no other token changing across the
target variants. The two frame-specific attractor pairs are `stone/stones`
and `wall/walls`; no-attractor, singular-attractor and plural-attractor
conditions each have one shared subjectless fragment. Body-part vocabulary
reduces a broad semantic category difference but does not separate lexical
identity, frequency, surface morphology or context from grammatical number.
Fragments are descriptive controls, not controlled subject deletion.

The [cases](../../experiments/distilgpt2_agreement_noun_substitution_cases_v1.json)
have SHA-256 `94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe`;
their ordered token rows have SHA-256
`c34dd209b8037d0a024654c260377be691a9d87446e0b2ad873d9ace474e41f3`.
They use the pinned tokenizer and eight one-token candidate continuations
listed in the protocol. Input lengths are two 4-token, twenty 6-token and
thirty-two 8-token rows. The [preparer](prepare_cases.py) binds the three
earlier public source datasets, excludes previously used subject/attractor
tokens, checks exact target and continuation boundaries, and records its own
source hash. Its [synthetic tests](test_prepare_cases.py) load no model.

Independent reconstruction retokenized all 54 rows and 432 continuations,
verified the selected forms, metadata, exact one-token subject substitution
and locative alignment, and checked all distinct-case prefix relationships.
Against the canonical v32 inventory at merge
`9d9d82b3a53a4977e55250c4202ca207971ada7e`, SHA-256
`4a4ed0127d18a7c5a8f6472af619b82895cb4efa6c07314a48a463510b1878b2`,
the 486 proposed inputs/extensions had zero relations in either full-row
prefix direction with 4,089 known base/target/grouped-candidate sequences.
A separate inventory reviewer checked the 33 public source modules against
their Git blobs and reproduced those canonical bytes. These
are point-in-time checks against recorded public data.

The independent [natural public-support receipt](natural-support-comparison.json)
checks all 486 sequences against the original 128-row conservative public
support and the successor's 256 public inputs. Its
[review script](audit_natural_support.py) uses workspace-relative paths and
must be adapted elsewhere. No private membership was queried. The implication
for the old sealed split is conditional on the recorded sampler/tokenizer
preparation and is not a complete exposure certificate.

Additional named public checks found zero bidirectional prefix relations:

* [Two-attractor patch inputs](patch-comparison.json): 486 × 160 base and
  declared continuation sequences. The [portable comparator](compare_patch.py)
  uses the separately published patch bundle.
* [Short-service inputs](service-comparison.json): 486 × six distinct active
  or padded rows. The [comparator](compare_service_cases.py) preserves the
  service bundle's intentional internal active-to-padded prefix relations.
* [Known-injection proposal](known-injection-comparison.json): 486 × 48
  input/continuation sequences. The [comparator](compare_known_injection.py)
  binds the separately [published proposal](../../experiments/public_known_injection_cases_v1.json)
  by digest; it does not attest execution or address protected rows.

These checks do not establish currentness after publication or exclude
unknown earlier inputs. A reviewed planned exposure reservation, a fresh
latest-inventory comparison, current natural and injection custodian reports,
and an independently reviewed source-bound owner are required before the
single attempt. A failed or uncertain check stops it without silently
reselecting or retrying. The later result must preserve positive, negative,
baseline-limited and incomplete outcomes and include independent saved-byte
replay.
