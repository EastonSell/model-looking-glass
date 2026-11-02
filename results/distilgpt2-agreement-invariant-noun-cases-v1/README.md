# Fixed invariant-noun study cases

The [fixed public cases](../../experiments/distilgpt2_agreement_invariant_noun_cases_v1.json)
implement the preregistered [protocol](../../docs/AGREEMENT_INVARIANT_NOUN_V1.md)
and have SHA-256 `50ef3aea53a21a76ecf1bc4910882f78939a4d2fcca12528013ea56eaa586827`.
The ordered token rows hash to
`54eb514e8afbb82ca3e341893b452c7ea926a8bd40489851b3e5bc0d07332161`.
This package was constructed with the tokenizer only; no new model forward
had occurred at publication. The two partitions are public and nonblind.

The ordered selection produced `sheep` and `deer` for discovery and `squid`
and `shrimp` for evaluation. The first evaluation candidate, `bison`, was
multi-token. Four distinct attractor pairs were selected: gate/gates,
lamp/lamps, rock/rocks and box/boxes. The prior-token and multi-token reasons
for skipping tree/trees and arch/arches are saved in the case file. Each
partition contains 24 targets, six subjectless fragments and four noun-omitted
demonstrative comparators. Token lengths are 4 (4 cases), 5 (8), 6 (24), and
8 (32). The same noun form and token are held fixed within each this/these
pair, while the determiner changes; this cannot isolate an abstract number
representation from determiner or context effects.

The [preparer](prepare_invariant_cases_v1.py) reconstructs all 68 texts and
544 single-token answer extensions from the pinned tokenizer and four prior
public bundles; its fabricated [tests](test_prepare_invariant_cases_v1.py)
do not use a model. The case file records the v38 snapshot used during its
construction. The later public [v39 comparison](v39-comparison.json) checked
all 612 base/answer sequences against 4,979 distinct inventory sequences:
3,047,148 bidirectional sequence pairs, zero prefix relations. The comparator
is [provided](compare_invariant_inventory_v1.py); it verifies supplied bytes,
not whether an operator's inventory revision is current.

A separate [public support and service observation](public-support-comparison.json)
also reports zero relations against the 128 original public-support rows, the
256 successor-pool inputs and six active/padded service rows. Coverage of the
old sealed split remains conditional on the original source-bound preparation;
no private membership was queried. These named public comparisons do not prove
that unknown exposures or later protected pools do not exist. Before a model
forward, publish a reviewed planned reservation and refresh current inventory
and the scoped custody observations. All failed or incomplete outcomes remain
reportable; no case may be changed in response to model output.
