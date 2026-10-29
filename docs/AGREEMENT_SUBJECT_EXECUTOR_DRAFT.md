# Agreement subject-dependence CPU worker: development checkpoint

This is a prospective, internal worker for the separately published subject
follow-up. It does not create cases, load a checkpoint, establish exposure or
custodian clearance, bind a source tree, supervise a process, or authorize a
forward. No real case or model run is authorized by this file. The final entry
and its parent must bind the published protocol, reference, directions, cases,
reservation, current exposure, custodian observations and exact source before
using it. They must record their own observed child exit, preserve incomplete
attempts, and prohibit retries after uncertain work. Independent scientific
replay is required before a candidate is treated as a result.

The worker checks canonical case coordinates and matched-token relationships,
then executes the fixed 80-case, 720-forward schedule using the existing CPU
direction-study intervention machinery. Its clean/no-op/clean-after comparison
is bitwise. Every hooked result retains whole and selected before/after
activations; each arm uses a directly rounded binary64 direction displacement
and must pass both the L2 and signed-projection tolerances. It saves last-row
logits for each successful forward and checks every sequence logit for
finiteness live. The saved successful last row cannot independently establish
that all other positions were finite.

The journal's default per-forward cap remains 1 MiB, while the subject worker
explicitly requires enough room for its largest full failed return. The
optional upper ceiling is 4 MiB. For the fixed 12-token, 768-wide,
50,257-vocabulary workload the conservative logical success bound is
189,468,480 bytes; the additional failed-return reserve is 2,492,208 bytes,
and the invalid-candidate reserve is 6,144 bytes. The journal separately caps
retained tensor payloads at 512 MiB and the event log at 64 MiB. These are
logical payload limits, not RAM, filesystem allocation, copy, or process-tree
limits. The final caller must enforce and report its own physical-attempt and
copy accounting and check the prescribed free disk before the one-shot claim.

Tests in this checkpoint use fabricated token IDs, tiny CPU models, synthetic
vectors and fault injection. They do not establish the behavior of the pinned
checkpoint, actual published cases, source-bound entry, process exit or the
future scientific analysis.

## Supervised child draft

`tools/agreement_subject_cpu_v1.py` takes the following required arguments:
`--prelaunch`, `--prelaunch-sha256`, `--latest-exposure`,
`--operator-exposure`, `--operator-observation`, `--natural-observation`,
`--injection-observation`, `--case-file`, `--direction-dir`,
`--reference-file`, `--reference-manifest`, `--cache-dir`, `--output-dir`,
`--study-root`, `--venv`, and `--child-custody`. The existing reviewed gated
CPU observer launches a direct base interpreter with `-I -S -B` and the empty
pre-claim CUDA mask. `--validate-only` verifies the same source, published
assets, inventory and observation bindings without loading the model or
creating the one-shot case claim. This draft still requires source, entry and
prelaunch review before any real study run.

The prelaunch schema is `distilgpt2-agreement-subject-prelaunch-v1`; the
exact required keys and pinned values are in `validate_prelaunch`. Its
`exposure.latest_*` fields name the inventory independently reconstructed
from an ancestor source module. The separate `operator_*` fields name the
current inventory payload and refresh observation supplied by the operator;
the child checks that payload's bytes, known sequences and collisions, but
does not reconstruct or authenticate its claimed current revision. The parent
must refresh the official inventory and both custodian observations just
before launch and retain the ordering as an operator observation.

Natural and injection refresh files are operator records of custodian
reports, not authenticated attestations. The natural record distinguishes an
undrawn successor from a drawn split covered by the published 256-row support;
unavailable coverage stops admission. The published natural receipt checks
public support and makes the old sealed-split coverage conditional on its
source-bound preparation; it is not a private membership query. The injection
record covers only the specified L3-final site and reported protected rows.

The entry retains the same `agreement-direction-supervised-child-v1` custody
frame and `attempt/terminal.json` and `entry-result.json` candidate paths as
the prior direction owner. The supervisor observes its direct child only; Git
verification may use descendants and no descendant-drain or authenticated
execution claim follows from a zero exit. A case-bound exclusive claim in the
CPU custody root prevents another cooperating attempt under that root, even
if the output path or prelaunch record changes. Different or aliased roots are
not coordinated. The caller must preserve that root and never retry after an
uncertain forward.

The entry checks at least 2 GiB free on the output volume before the claim,
caps journal payload/log lengths, and retains the input/reference/receipt
copies with the candidate. Its 1 GiB physical-attempt limit is based on summed
file lengths plus a 16 MiB external-receipt allowance; it is not a bound on
filesystem allocation, RAM, process-tree use, or uncounted external copies.
Any such copy must be separately accounted for by the parent.
