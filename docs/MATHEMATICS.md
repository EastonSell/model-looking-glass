# What a transparent observation can tell us

Let a be a finite activation tensor. A forward hook that returns `None` observes the tensor but does not replace the module's output. Saving `a.detach().clone()` retains its current values without keeping the model graph alive. Detaching only the saved copy does not detach the main model computation.

## Separate gradient channel
Create an ephemeral tensor q of zeros, with the same shape and dtype as a, requiring gradients. Define

    t(a, q) = sum(stop_gradient(a) * q).

For finite a at q=0, the reported scalar is zero and the mathematical derivative with respect to q is exactly a. The implementation computes only this separate gradient. q is not a model parameter and is never passed to an optimizer. This is an ordinary autodiff identity, not a new theorem and not an application of the Trojan cancellation gate. Finite precision and execution contexts still require empirical checks.

That identity is not a bitwise guarantee for floating-point execution. A
[CPU denormal control](GRADIENT_DENORMAL_V1.md) observed subnormal bits being
lost by gradient readout under flushing, while `torch.equal` still reported
equality. Compare raw representations when bitwise fidelity is required and
record the execution mode.

The channel observes a before constructing t; it therefore cannot reveal information that direct capture lacked. The useful empirical comparison is reliability and overhead, not a presumed information advantage. The full tensor must still be stored or processed somewhere.

## Provenance is additional information
Suppose branch A and branch B both emit the same vector v. Any function receiving only v has identical input in both cases. It cannot identify which branch was used without additional information. Node labels and occurrence counters provide that information explicitly; no branch lineage is recovered from the vector itself.

At a residual merge y=x+f(x), there are at least two contributors. A single 'last neuron' is generally not a meaningful description. Instrumentation should name its observation boundary, and route labels should be recorded before branches merge. Execution labels are not a causal explanation of the model's output. Gradient sensitivities are also different measurements from execution provenance.

## Sources
- PyTorch module forward hook behavior: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- PyTorch autograd.grad: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
- PyTorch gradient modes and inference tensors: https://docs.pytorch.org/docs/stable/notes/autograd.html#grad-modes
- Related prior audit project: https://github.com/EastonSell/turing-complete-malware-pytorch-backward-pass-audit

No code or model payload from the Trojan audit project is executed here.
