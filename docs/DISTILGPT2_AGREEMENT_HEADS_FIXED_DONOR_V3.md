# Exploratory fixed-donor agreement contrast

The v2 held-out comparison used a same-noun, opposite-number donor against a
different-noun, same-number donor. The resulting contrast changes donor noun
identity as well as number. This follow-up asks a narrower question: **when
both donors are variants of the same other noun, does the head-4 patch still
produce a number-consistent difference in the receiver's `is`/`are` margin?**

This design was selected after the v2 outcomes and after checking feasibility
on their saved activations. All 16 prompts have already been forwarded and
are now exposed. This is exploratory development on those same public cases,
not a new held-out confirmation, random sample or preregistered significance
test. No threshold or replacement-head selection is introduced.

Use exactly the v2 case file (SHA-256
`51d14d87f7a318ff94e613d3ff88dfd338a6906675a92af41899b3a027eead7b`),
the frozen layer-3 head 4 and its adaptively selected head-8 comparator, and
the same model/runtime. For each receiver's ordered four-case frame, the
same-number donor is local index `receiver XOR 2`; the opposite-number donor
is `receiver XOR 3`. Both donor rows therefore use the other subject lemma,
one singular and one plural. As before, the patches replace one last-token
head input slice immediately before `attn.c_proj`, with q = 5% of the
receiver slice's L2 norm, independently per head and receiver. Both donors
must be at least q away; both rounded doses must be within 1e-5 relative
error of q and their ratio at most 1.00001. No donor displacement is
amplified. Calculate the complete dose plan before any patch; if either
fixed head fails on any receiver, stop as `dose_ineligible` after the clean
observations without substituting cases or heads.

Bind the saved v2 runtime record and require the same recorded dependency
versions, Python/platform, CPU threading and relevant CPU environment,
seed, denormal probes and model-state digest. Hash-check the v2 clean
projection-input and full last-logit witnesses. Each of the 16 new clean
observations must match both corresponding saved witnesses bit for bit before
any patch; a difference stops the attempt without a donor comparison. The
saved-byte reader independently checks this match against those prior
witnesses. The runtime checks and exact baseline do not prove every hidden
machine detail is identical.

The signed contrast is `number_sign * (margin_same - margin_opposite)`,
where margin is the last-position `is` logit minus the `are` logit and the
sign is +1 for a singular receiver and -1 for a plural receiver. Report
all 16 contrasts for both heads, frame and noun groupings, clean baseline,
requested and realized doses, and full-logit or top-token changes if saved.
Use the existing 32 self patches, 16 clean repeats, direct landing and
bitwise untouched-state checks. Exactly 128 forwards are planned when all
cells qualify; no continuation strings are forwarded. A saved-byte replay
and observed process exit are required before reporting a result.

The fixed other lemma removes the earlier *between-lemma* donor confound,
but its singular and plural forms are still different lexical tokens and
can differ in meaning or context. Head 8 is an adaptive active comparator,
not a random-head null, and relative input doses do not match downstream
changes. This comparison alone cannot identify an abstract number feature,
a unique circuit, generated behavior, or generalization beyond the exposed
small set of reused frames and nouns. Public-row checks and known exposure
records do not establish pretraining independence.
