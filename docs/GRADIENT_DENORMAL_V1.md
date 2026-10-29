# Gradient readout and denormal control v1

This deterministic CPU control compares direct capture and the separate
gradient readout on eight fixed float32 bit patterns: positive and negative
minimum and maximum subnormals, minimum normals, and signed zeros. It is not a
model study, a held-out evaluation, a performance measurement, or a claim
about every processor or PyTorch version. No random input or seed is involved.

In a fresh process with one PyTorch intra-op thread, create the inputs by
reinterpreting integer bits, then run the same direct clone and
`gradient_readout` with denormal flushing disabled and enabled. For each mode
record the raw input, direct and gradient bits, the result of `torch.equal`,
and bitwise equality. Compare integer views rather than floating-point
comparisons when testing the bit-level claim. The script checks the two source
files against the recorded Git commit, creates a new output directory, and
does not overwrite prior evidence. The denormal setting is thread-local and is
changed only inside this short-lived process. Run on CPU; do not treat the
installed PyTorch CUDA build as evidence of GPU execution.

The real-arithmetic derivative of `sum(stopgrad(a) * q)` with respect to `q`
is `a`. That does not guarantee that floating-point autodiff preserves every
bit. On the inspected Windows x86-64 host with PyTorch 2.8.0+cu128, flushing
changed the separate gradient readout's positive and negative subnormals into
signed zeros while the direct clone retained their bits. `torch.equal`
reported equality in both modes. Signed zeros also compare equal numerically.
The [saved result](../results/gradient-denormal-cpu-v1/result.json) records the
actual patterns and is a single-platform observation, not a universal bound.

Run from a clean committed checkout in a fresh CPU-only process, using an empty
private Python bytecode-cache prefix. For example, on Windows, set the
following paths to the checkout, its verified virtual environment's
`Lib/site-packages`, a new empty cache directory, and a new result directory:

```powershell
$launcher = @'
import pathlib, runpy, sys
checkout, packages, output = sys.argv[1:]
sys.path[:0] = [checkout, packages]
script = pathlib.Path(checkout) / 'experiments' / 'gradient_denormal_v1.py'
sys.argv = [str(script), '--output', output]
runpy.run_path(str(script), run_name='__main__')
'@
python -I -S -B -X pycache_prefix=<empty-cache> -c $launcher <checkout> <site-packages> <new-output>
```

The saved source hashes provide consistency evidence, not proof of executed
machine code or isolation from a malicious process. Do not compare timings
from this control.

The earlier [verified CPU result](../RESULTS.md) used `torch.equal` and did
not record the denormal mode or retain activation bits. Its recorded equal
flags remain unchanged, but they do not establish bitwise identity or rule out
an FTZ-sensitive difference. The source code for the earlier initial run is
not fully available, so its comparator cannot be independently established.
