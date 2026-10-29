# Bounded color-preference baseline screen

This owner implements only the nine-row baseline screen in the
[v2 protocol](DISTILBERT_COLOR_PREFERENCE_V2.md). It does not implement a
parameter change or a held-out/locality evaluation. A separately published
and reviewed owner is required before any subsequent stage.

The owner reuses the earlier reviewed CPU action engine and cooperative
direct-child supervisor/observer. It verifies the exact case, protocol,
source, pinned checkpoint and tokenizer, uses eager evaluation-mode CPU
float32 with one thread, and retains all 30,522 f32 mask logits for each
completed action. It starts at most nine forwards and stops incomplete on a
failed action or uncertain restoration. A completed ineligible screen is a
result and is not retried with substituted cases.

Eligibility is recomputed from the saved f32 logits. The exact rational
difference `factual - purple` must be strictly greater than the exact decimal
value `1/10` on all three fixed rows for a candidate. The displayed decimal
margin may round to 0.10 and is not the decision value; each row also records
the exact numerator and denominator. All nine rows are retained before the
first eligible candidate in the fixed order is selected. A non-factual full
vocabulary top is reported but does not itself fail this pairwise screen.
The summary includes both token probabilities and their combined mass;
combined mass is calculated in log space before exponentiation.

The prelaunch record reconstructs the earlier frozen comparison and scans the
current official main tree for named BERT and cross-model case/exposure JSON
inputs. It validates the known input schemas and both complete and one-SEP-
removed prefix directions, and accepts separately pinned public proposals.
If main advances, the supervisor checks the newly observed tree instead of
rejecting an unrelated commit; the claim binds that exact scan. The worker
reconstructs the launch scan from its pinned Git commit before model loading.
This is a point-in-time named-public input check, not proof of unknown or
private history, pretraining novelty, or an authenticated process execution.

The screen is a one-shot attempt. Before a completed result is validated,
its saved runtime must identify the pinned CPU model/revision, full registered
state digest, deterministic one-thread settings, tied decoder and the claimed
dependency-site and scratch-root hashes. The retained start/completion journal,
failure witnesses, cooperative child-exit record, post-validation marker and
separate supervisor-exit observation have the same scope and incomplete
precedence as the earlier owner. A lost spawn handle or uncertain child status
is not a clean failure or permission to retry. Scratch and run evidence must
be disjoint from the source, cache and dependency site; free-space checks
precede the claim. This note and source do not report a real v2 outcome.
