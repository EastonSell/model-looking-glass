# Prospective naturally phrased masked-antonym screen, v1

The preceding [antonym screen](../results/distilbert-antonym-preference-v1-20261029/README.md)
was baseline-ineligible: the factual answer was not the unique vocabulary top
on any of four discovery rows. Its calibration prompts put `short` on top for
both rope rows and `dark` on top for one room row. Those rows and their results
are exposed development evidence, not confirmation. This distinct, prospective
follow-up fixes more natural comparison phrases before any new forward. It
reuses those two objects and antonym pairs, so neither the objects nor the pair
choice is held out; the new phrasings are same-pair paraphrases, not independent
facts or evidence of general antonym understanding.

The question is whether a fixed final-block change can reverse a masked-token
preference on those paraphrases, and how it compares on fixed localities with
one deliberately global, efficacy-matched decoder-bias control. The desired
counterfactual copies the cue word. A reversal can reflect copying or lexical
preference and is not a generated answer, abstract antonym computation, broad
capability-preservation result or evidence of causal knowledge localization.
Lower disturbance than the global bias does not establish superiority over
other editing methods.

## Fixed inputs and chronology

Use the same pinned offline `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, CPU float32, eager evaluation and
one-thread deterministic runtime as the [preceding protocol](DISTILBERT_ANTONYM_PREFERENCE_V1.md)
at `4104b5ac8eb83904839fc60f7b16489a228eb481` (document SHA-256
`a0735f00bb3f9e91f494456d5a1e060b88ea5dc181cf58dcd08c1593540800b1`).
Save the full 30,522-token mask logits for every completed action. Publish and
independently review this protocol, the complete tokenizer-only case inventory,
the named-public overlap comparison and the exact bounded owner before any new
pretrained forward. Validate every fixed factual, cue, wrong-target and
accepted fill as a whole non-special token in the complete sentence. An invalid
fixed row or semantic label requires a disclosed pre-forward protocol revision,
not an outcome-dependent replacement.

Use this outcome-informed candidate order:

| Order | Object | Dimension | Cue / counterfactual | Preferred factual | Other accepted factual |
| --- | --- | --- | --- | --- | --- |
| 1 | rope | length | long | short | none |
| 2 | room | brightness | bright | dark | dim |

The preferred factual token alone determines the discovery-top and pairwise
gates. `red` is the fixed wrong-target adjective. The accepted sets are
operational, non-exhaustive lists of plausible fills; an alternative does not
rescue a failed preferred-token gate. All fixed object/cue/factual words take
`a` in the target templates. Substitute exactly:

| Role | Text |
| --- | --- |
| discovery | `The opposite of a {cue} {object} is a [MASK] {object}.` |
| calibration 1 | `A {cue} {object} is the opposite of a [MASK] {object}.` |
| calibration 2 | `In {dimension}, a {cue} {object} is the opposite of a [MASK] {object}.` |
| held-out 1 | `The opposite of a {cue} {object} in {dimension} is a [MASK] {object}.` |
| held-out 2 | `A {cue} {object} is the opposite in {dimension} of a [MASK] {object}.` |
| held-out 3 | `A {cue} {object} and a [MASK] {object} are opposite in {dimension}.` |
| held-out 4 | `In {dimension}, a [MASK] {object} is the opposite of a {cue} {object}.` |

For each candidate, use these two same-object locality phrasings and fixed
accepted set. They are two phrasings of one relation, not two independent facts.

| Object | Locality 1 | Locality 2 | Accepted |
| --- | --- | --- | --- |
| rope | `A rope is a [MASK] used for tying.` | `For tying, a rope is a kind of [MASK].` | `{cord}` |
| room | `A room is an enclosed [MASK] inside a building.` | `Inside a building, a room is an enclosed [MASK].` | `{space}` |

The selected candidate also has two reverse-relation localities,
`The opposite of a {factual} {object} is a [MASK] {object}.` and
`In {dimension}, a {factual} {object} is the opposite of a [MASK] {object}.`,
both accepting the cue token. The other alternate-use locality is
`The novel was quite [MASK], taking weeks to finish.` with `{long}` for rope,
or `The [MASK] engineer solved the difficult puzzle.` with `{bright}` for room.
These are operational contextual labels; alternate uses need not be independent
lexical senses. A global cue bias can improve these rows and reverse localities.

The seven common localities are fixed as follows:

| Stratum | Text | Accepted |
| --- | --- | --- |
| wrong-target correct | `A stop sign is typically [MASK] in color.` | `{red}` |
| unrelated antonym | `The opposite of up is [MASK].` | `{down}` |
| unrelated antonym | `The opposite of dead is [MASK].` | `{alive}` |
| unrelated antonym | `The opposite of closed is [MASK].` | `{open}` |
| unrelated relation | `A piano is a musical [MASK].` | `{instrument}` |
| unrelated relation | `A week contains seven [MASK].` | `{days}` |
| unrelated relation | `A bird is a kind of [MASK].` | `{animal}` |

There are 31 fixed input rows: two times three screen, four held-out and five
candidate-specific locality rows, plus seven common localities. Compare every
row against named current public DistilBERT inventories in both prefix
directions, with complete IDs and after removing one terminal SEP; report exact
scope and counts, and coordinate current custodians. This is not a claim about
unknown, private or pretraining history.

## Screen, change and comparison

Run all six discovery/calibration rows in candidate and role order. A candidate
qualifies only when its preferred factual token is the strict unique full
vocabulary top on discovery and its exact saved-f32 factual-minus-cue margin is
strictly greater than exact rational 1/10 on all three rows. Preserve ties and
all failures; choose the first qualifying candidate only after all six rows.
Report full-vocabulary tops, factual/cue/red ranks and probabilities, accepted
alternatives and combined factual-plus-cue mass. Publish all screen logits and
the selection before constructing an edit. If none qualifies, publish
`baseline-ineligible` and stop without replacing candidates or interpreting an
edit outcome.

For a selected candidate, use the preceding protocol's exact development,
evaluation, numerical, restoration, failure and publication contracts, with
this document's fixed inputs and candidate order. In particular, the discovery
gradient targets cue-minus-factual on final-block FFN `lin2.weight`; choose the
first successful fixed one of the six prescribed normalized doses using only
discovery. Keep the fixed opposite, seeded Gaussian rank-one and red-target
controls, their realized-norm and unavailable-control rules, and the
discovery-only two-entry global-bias solve. Publish the entire development
schedule before held-out actions; do not tune on held-outs or locality.

On both calibration and all four held-out rows, the primary and bias arms must each gain at least
0.10 in cue-minus-factual margin and 0.025 in cue full-vocabulary
log-probability relative to the same clean row. For each gain separately, the
arm difference must be at most 10% of the larger absolute gain; their edited
full-vocabulary tops must be unique and identical. A failed match is retained
descriptively and cannot support an efficacy-matched locality interpretation.
The primary-versus-bias locality comparison additionally requires discovery
reversal and both calibration matches, factual-minus-cue greater than exact
1/10 on each clean held-out, all four held-out efficacy matches, and both arms
reversing by more than 1/10 on at least two of the same four held-out rows.
The relative locality criterion requires primary mean KL(clean || arm) below
0.8 times the bias mean and at least 1e-6 lower, with no more changed unique
tops. Also require primary mean KL at most 0.02 and maximum KL at most 0.10,
mean accepted-set NLL deterioration at most 0.05 and maximum at most 0.20,
and retention of every initially accepted unique top. A tie in any original,
primary or bias locality top, a missing required arm, or zero initially
accepted unique tops makes this comparison not evaluable. Report each arm and
each locality, including baseline errors, ties, unavailable controls and
per-stratum numerators and denominators. A missing auxiliary arm does not
itself invalidate the primary-versus-bias comparison.
Uncertifiable numerical boundaries are indeterminate; disagreement with the
independent adaptive-precision saved-logit audit is never promoted to success.

The maximum schedule is six screen, 23 development and 96 evaluation starts,
125 total under the shared 256-start ceiling. The 128 MiB retained-original
evidence ceiling includes logits, gradients, update and process/failure
metadata. A failed action, invalid installed weights, uncertain restoration,
partial stage or uncertain process cleanup stays incomplete and is not silently
retried. Full distributions and accepted-set outcomes are reported rather than
equating a pairwise shift with a predicted-answer switch or broad preservation.
