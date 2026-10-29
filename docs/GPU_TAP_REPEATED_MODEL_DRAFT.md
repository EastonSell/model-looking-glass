# Repeated tap fixed-model setup, disabled draft

`tap_repeated_model._FixedModelSetup` remains disabled and is not a supervised
GPU entrypoint. Its guarded offline loader checks the pinned DistilGPT2 asset
digests and revision, eager evaluation configuration, tied weights, 81,912,576
FP32 parameters and their actual device. It then retains the model through
transfer, creates and checks the exact int64 input `[15496, 995]`, and keeps
both model and input rooted for the preflight and samples. It checks memory
before and after transfer and after input creation. The source checkpoint
bundle is dropped only after synchronization; an unexpected surviving bundle
causes a fail-stop result rather than a retry or cleanup claim.

The 25% allocator cap must already be set and read back by the guarded CUDA
backend, after physical-device binding and admission, before this setup is
enabled. The tested CPU copy is not evidence of CUDA transfer or GPU timing.
This setup does not cap host RSS, authenticate the machine, reserve physical
GPU memory, or cover native temporaries interrupted before Python assignment.
The actual supervised bootstrap, one-use start grant, durable journal,
cleanup barrier and direct hard-exit policy remain separate requirements.
