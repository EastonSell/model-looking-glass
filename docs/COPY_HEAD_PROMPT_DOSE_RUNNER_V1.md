# Matched-dose copying-head CPU owner

The owner implements the fixed
[per-prompt protocol](DISTILGPT2_COPY_HEAD_PROMPT_DOSE_V1.md) using the
previous copying-head witness. It accepts the already published 32 cases and
v31 reservation, and it does not select cases from model outputs. The
bootstrap verifies the committed manifest and its source map with only Python's
standard library before importing project code or PyTorch. The source commit
must be an ancestor of the launch checkout, with the committed manifest as
the only tree difference. The caller must use a clean checkout, an empty
absolute bytecode-prefix directory, `-B`, offline flags and hidden CUDA. These
are cooperative local checks; they do not authenticate the interpreter,
operating system, process tree or clock.
The operator launches the bootstrap explicitly; its environment marker is not
an authentication mechanism and does not prevent a caller from bypassing it.

The operator must keep the source checkout exclusively assigned and refrain
from changing its files from the bootstrap's first source check through the
owner's final source recheck and process exit. An outside actor can still
change a file between checks; without this cooperative condition the saved
checks cannot establish which bytes were imported.

For each input, the owner runs clean, witnessed no-op, three left-rotated
single-head actions if the planned dose is eligible, and clean-after. It
computes `q` from the three slices of that prompt's no-op, never from another
prompt. A zero or floor-level `q` creates no fractions or planned delivery;
a rounding mismatch retains the plan but skips the three actions. Both cases
still run clean-after. An invalid tensor, failed forward, bit mismatch or
uncertain hook cleanup invalidates the attempt. The prespecified decision
requires at least six evaluable families; it is descriptive, not a
significance test.

The journal records starts, raw offsets and hashes, dose plans, complete
observations and failure markers. Each case's raw file holds all returned
final-position vocabulary logits and all no-op/arm projection-input and
projection-output witnesses. If an exception interrupts the post-return
path, the owner attempts to complete any matching partial payload without
duplicating bytes. Counters are derived from pre-call values and the observed
return; an interruption before the payload binds is recorded as uncertain.
The actual partial-file inventory is authoritative if recovery fails. A hard
process termination can still prevent further recording.

The 192-forward and 51,573,504-byte maxima apply when every prompt is eligible;
ineligible prompts require three rather than six forwards. The tensor-byte
bound excludes the model, metadata, temporary copies and other process memory.
The result retains each prompt's requested/realized dose, all target scores,
directional full-vocabulary KL and total variation, projection-output dose,
family contrasts and the fixed decision. KL and total variation include the
target token. Equal pre-projection L2 does not match downstream effect.
