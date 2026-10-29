# Answer-format feasibility: fixed public cases

The [case file](../experiments/answer_format_feasibility_cases_v1.json)
freezes 64 tokenizer-only inputs (32 development, 32 held-out) under the
[protocol](ANSWER_FORMAT_FEASIBILITY_CPU_V1.md) and its pre-forward
[exemplar amendment](ANSWER_FORMAT_FEASIBILITY_SCOPE_V1.md). Its SHA-256 is
cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a.
The operator reports no model forward during case construction. Independent
review verified the answer-slot semantics, meaningful development/held-out
wording differences, matched suffixes, eight-token prefixes, all 256
single-token continuation boundaries and both seeded permutations. The facts
are reused across roles; held-out means new templates, not unseen facts.

Before any forward, independent numerical review identified cancellation in
a draft scorer when an unrelated finite logit is extremely large. The frozen
implementation evaluates D by summing the four individually promoted city
logits with math.fsum, because the two log-softmax normalizers cancel. It
computes each two-template fact mean from all eight terms at once. Paired label
log-probability changes similarly use the two logits, shifted maxima and
log-normalizers in one math.fsum. These are the same prespecified mathematical
quantities and avoid cancellation from subtracting or averaging separately
rounded displayed values. Paired log-probability changes still depend on
rounded exponentials and log-normalizers, so signs arbitrarily near zero can
remain numerically uncertain.

The builder compared all 2,016 within-bundle input pairs and all 46,272
bidirectional pairs against the 723 v27 public input records; none has a
complete-row-prefix relation. Its public_appended_metadata count of 56,832
covers only candidate_token_ids metadata, not every prior continuation field.
A separate independent check compared all 2,277 prior target, candidate and
group metadata extensions (145,728 pairs) and found no relation. Every such
extension begins with its corresponding already-checked complete input.
Neither continuation metadata nor these comparisons asserts a model forward.

The [public natural comparison](../experiments/answer_format_feasibility_natural_comparison_v1.json)
reports zero relations to all 128 original sampler-support rows and 256
successor-pool rows. Its conclusion about the still-current original sealed
split is conditional on the recorded sampler/tokenizer preparation and the
custodian's report; no private query was performed. This and the public
inventory checks do not rule out unknown exposure or pretraining overlap.
Fresh relevant custody and the latest reviewed inventory remain required before
each phase.

The [independent case review](../experiments/answer_format_feasibility_case_review_v1.json)
and its [reader](../tools/audit_answer_format_cases_v1.py) cover exact case and
tokenizer identities, role order, the prior 36 screen rows and boundaries;
semantic judgment is separately reported by the named reviewer. The
[public-support checker](../tools/audit_answer_format_natural_public_v1.py)
is the exact reviewed script and uses its original workspace-relative paths,
so running it elsewhere requires adapting those paths and retaining the pinned
input hashes. Both checks are tokenizer-only and do not authenticate source
chronology or execution.

