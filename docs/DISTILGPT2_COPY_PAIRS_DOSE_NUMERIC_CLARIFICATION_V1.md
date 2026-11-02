# Pairwise copying doses: pre-forward numerical clarification

This clarification is published after the public case file was assembled and
before any forward on those cases. It preserves the original [protocol](DISTILGPT2_COPY_HEAD_PAIRS_DOSE_V1.md),
its cases, controls, statistical comparisons, dose formula and evaluation
order. The original protocol SHA-256 is
`811c18baef640e08ccff7f6cade26d041672edda969ddbc7a6babdb60d70bddb`;
the fixed case-file SHA-256 is
`3e5d40fff832a6976f5e4f8d3bda916ab203ed93936463a1a0726e88dcab1106`.

The protocol's norm formulas mean the literal Python operations as written:
`math.sqrt(math.fsum(v ** 2 for v in values))`, including for each decoded
source value and for each binary64 before/after difference. They do **not**
authorize silently replacing `v ** 2` by `v * v`, even though both express
the same mathematical square. The earlier blanket description of binary64
operations as nearest-even does not guarantee that Python's exponentiation
or underlying library computes a correctly rounded square at every tie.
Basic-operation and denormal probes, the integer-defined nearest-even final
binary32 conversion, input-bit decoding, the written reductions and all
achieved-dose tolerances remain unchanged.

On the intended Windows CPython 3.11.15 runtime, the independently found
value `x = float.fromhex('-0x1.8198ccc000000p+0')` demonstrates the issue:
`(x ** 2).hex()` is `0x1.22667986a3e15p+1`, while `(x * x).hex()` is
`0x1.22667986a3e14p+1`. Exact rational arithmetic places the square halfway
between those adjacent binary64 values; the lower value is the even result.
This is a specific observed midpoint, not a claimed error bound for all
inputs. Add a regression for the midpoint and for the literal norm reduction.
Record the actual Python/platform/runtime used. A saved-data reader should
follow the same literal expression and documented runtime; a replay on a
different implementation may differ in low bits and must not be described as
bit-identical solely because the mathematical formulas agree.

No new model outcomes informed this clarification. A failure of the numeric
probe, dose, bit-delivery or control checks still makes the attempt
incomplete. The output-scale, matched-input-dose and public/nonblind limits
of the original protocol remain in force.
