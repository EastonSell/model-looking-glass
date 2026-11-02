# Answer-format feasibility: prospective exemplar amendment

This note records a pre-forward amendment to
[ANSWER_FORMAT_FEASIBILITY_CPU_V1.md](ANSWER_FORMAT_FEASIBILITY_CPU_V1.md).
After that protocol was published and before any model forward, the first
tokenizer-only case construction found that every Portugal–Lisbon prompt
extended an already reserved public input: record
planned-parameter-edit-followon-locality-08 has the full token row for
"The capital of Portugal is". The builder failed closed and did not publish a
case bundle. No baseline logits, model outputs or private-split query informed
the replacement.

To keep the complete-row-prefix exclusion, replace the fixed portugal prefix
with this single correct alternative:

- sweden: "The capital of Sweden is Stockholm.\n"

Use Sweden–Stockholm wherever the original protocol refers to Portugal–Lisbon:
the fourth prefix label is sweden, the example-answer token is " Stockholm",
the six ordered reference/comparison pairs end with spain→sweden, and the
example-specific contrast becomes

D = [log P(Madrid | spain) - log P(Madrid | sweden)]
  - [log P(Stockholm | spain) - log P(Stockholm | sweden)].

The original Spain–Madrid and neutral prefixes are unchanged. Each nonempty
prefix remains eight tokens under the pinned tokenizer and each example answer
is one token. Stockholm is neither a target factual answer nor one of the
fixed target counterfactuals. The length-matched comparisons remain scoped to
these two fixed examples, not all demonstrations.

The operator checked the tokenizer and public v27 input rows for fourteen
candidate country–capital prefixes before choosing Sweden–Stockholm:
Canada–Ottawa, Germany–Berlin, Greece–Athens, Austria–Vienna, Norway–Oslo,
Sweden–Stockholm, Finland–Helsinki, Ireland–Dublin, Belgium–Brussels,
Denmark–Copenhagen, Egypt–Cairo, Kenya–Nairobi, China–Beijing and
Mexico–Mexico City. Sweden was the first listed candidate with an eight-token
prefix, a one-token city continuation and no relation to a complete public
input row. This selection used only tokenization and public collision status,
not model behavior. The eventual complete 64 prompt rows must still pass the
protocol's full inventory, natural-support, successor-pool and current custody
checks; a prefix-only candidate check is not a substitute.

Keep the original protocol, this amendment, its corrected exact cases and both
phase source/manifest reviews in public ancestry before the first forward.
All remaining selection order, seeds, outcomes, underfill and failure rules
are unchanged. The comparison is exploratory and the documented amendment
means the alternate exemplar was chosen after the initial case construction,
though before any model observation.

