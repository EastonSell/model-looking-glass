# DistilGPT-2 agreement: relative-dose head comparison v2

**Status: prospective.** This is a new development-informed protocol, distinct
from [v1](DISTILGPT2_AGREEMENT_HEADS_V1.md). The v1 development attempt at
source commit `416ed7c60d030518e5d45aa9a774169b632c4013` stopped after 29
completed forwards: its common dose failed the fixed float32 delivery check
before the second donor patch. The failed prefix, not a head ranking, remains
part of the record. Its terminal and event-log SHA-256s are respectively
`a198617a3feaa3b45549a979e82f7ff0c24f339e23bc2ff147409f33f79435e9` and
`5c4ea15e2b1ff766dad39a379f699c6ddf4be080419449d3ce81171e37eb1382`.
Its 16 clean activation witnesses informed this revision. All 16 public clean
rows, 12 self-patches and one opposite-number head-0 patch were already
forwarded in v1; v2 repeats them. Development is exposed and not blind.

The old common absolute dose was only 0.0094%–0.2531% of the receiver-slice
norms. The new estimand is response to a **5% relative L2 change in each
receiver head slice**. On the retained v1 clean witnesses, ten of twelve heads
would pass the eligibility rule below and heads 2 and 10 would not. These
observations informed the design; v2 recomputes and saves all eligibilities.
The excluded heads may still matter to the model: selection is conditional on
eligibility and cannot rank all twelve or establish their relative importance.

## Inputs, model and controls

Use the pinned DistilGPT2 checkpoint, eager float32 CPU model, one thread,
uncached batch-one forward, token IDs, four discovery quartets, target and
donor mapping, intervention point, no-op/self/repeat controls, answer-token
IDs, source/checkpoint checks, and full saved witnesses specified in v1. The
intervention is the 64-coordinate head slice at the last token before
`transformer.h.3.attn.c_proj`. Do not forward appended answer tokens.

For each head and target, decode the clean receiver slice into binary64 and
compute `norm = sqrt(fsum(x*x))`, then `q = 0.05 * norm`. Compute the two
distances to the opposite-number same-lemma donor and the same-number
different-lemma donor. Require a finite q of at least `2**-20`, and **each**
distance to be at least q. Thus neither donor displacement is amplified.
For each direction, compute `receiver + (q / distance) * (donor - receiver)`
in binary64 and round each coordinate directly to float32 nearest-even. Each
realized L2 must differ from q by at most `1e-5 * q`, and the larger realized
L2 divided by the smaller must be at most `1.00001` within the pair. Both
landings and all untouched projection-input bits are checked exactly.

Calculate and retain the entire plan, including every head, target, receiver
norm, requested absolute q, donor, distance, realized absolute dose and reason
for ineligibility, **before any patch forward**. A head is eligible only when
every one of its target pairs passes the fixed rule. If fewer than two heads
are eligible, record an explicit `DoseIneligible` terminal and stop without
patching; the saved-byte reader checks the clean-prefix plan before reporting
`dose_ineligible` with no new ranking. An ineligible development has no selected
heads; an ineligible evaluation retains the already frozen candidate and
comparator identities without an evaluation decision. No head is
substituted, no dose is changed, and no target is dropped based on outcome.
Apply one self control per eligible head; execute all opposite and
same donor patches for each eligible head and all eight fixed development
targets, followed by all 16 repeats. With ten eligible heads this is 202
forwards; the actual count and all eligible/ineligible heads are recorded.

Rank eligible heads by mean `D = s * (m_same - m_opposite)` exactly as in v1,
freeze the winner and runner-up, and retain all signed values. This is a
comparison at equal **relative** input dose: absolute doses vary with head and
target. It is neither equal absolute L2 nor equal projected or behavioral dose.
Donor directions and lexical identities differ. The runner-up is selected on
the same development data, not a random-head null. Selection and eligibility
may favor heads with larger donor distances and exclude important heads.

## Held-out evaluation

After a reviewed saved-evidence selection is frozen, construct and reserve
the distinct 16 public evaluation rows exactly as v1 specifies, checking all
current public exposure inventories, pending studies and the natural and
injection custodian scopes before launch. The rows are public, not secret or
known absent from pretraining. No new evaluation input has yet been forwarded.

For the fixed candidate and runner-up, compute the same 5%-of-receiver rule
for every one of 16 targets and both donor directions, before any patch.
If any cell is ineligible, report dose-ineligible without selecting another
head, changing a case, relaxing the rule or proceeding to patches. Otherwise
run both heads' self and two donor controls on every target plus clean and
repeat forwards (128 total). Report absolute q, realized absolute and relative
doses, clean and patched margins, per-target and per-frame contrasts, all
negative and zero values, and the secondary `was - were` contrast. Retain the
v1 fixed baseline and exploratory decision rule; it now compares two
equal-relative-dose responses rather than equal-absolute-dose responses.
The reuse of nouns and frames precludes an independence claim. A negative,
baseline-limited or dose-ineligible outcome is retained, not proof of no head
effect at other doses.

## Execution and evidence

Save source and checkpoint identities, model-state checks, full last-token
logits, projection-input and projection-output witnesses, exact submitted
replacement bits, call starts, completions and numerical postchecks. The
independent reader reconstructs the complete eligibility plan and every saved
patch, checks both dose criteria and unselected bits, and recomputes ranking
and the fixed rule. Require the pinned 768-wide, 12-head, 50,257-vocabulary
layout for real-run replay. The journal retains at most 128 MiB of unique
tensor payloads plus a bounded event log; this is not a peak-RAM or process
tree bound. A saved journal and observed child exit are consistency evidence,
not authenticated execution or descendant-drain proof.

The v2 launcher uses an explicit output root, a distinct one-shot claim and
new run directory. It checks at least 1 GiB free on the source volume and 2
GiB on the actual output volume before claim. The v1 attempt and claim are
never overwritten. Preserve failures and do not automatically retry uncertain
forwards. The claim coordinates cooperating launches in that output root; it
is not a machine-global lock. Hook or cleanup interruption requires discarding
the worker/model. There is no GPU use or timing claim in this protocol.
