# GPU and machine inventory, v1

This is a point-in-time resource inventory, **not a throughput, latency,
memory-peak, isolation, or model-capability benchmark**. The saved record is
[gpu-inventory-v1-20261029T003034Z.json](../results/gpu-inventory-v1-20261029T003034Z.json).
It contains only selected hardware and software fields; usernames, process
paths, process lists, secrets, and the physical GPU UUID are not recorded.

At the `nvidia-smi` sample on **October 28, 2026, 17:30:34.453 Pacific time**
(between 00:30:34.378 and 00:30:34.466 UTC on October 29), the single
reported NVIDIA GPU was an **NVIDIA GeForce RTX 2070 SUPER**. The reported
framebuffer values were 8,192 MiB total, 203 MiB reserved, 1,424 MiB used,
and 6,566 MiB free. These individually reported, rounded values sum to
8,193 MiB; do not treat the one-MiB discrepancy as exact allocation accounting.
Free and used memory can change immediately with other activity, and the
reported free amount is neither an allocation guarantee nor a peak-memory
measurement.

The installed NVIDIA driver reported version 616.92. A separate header query
at 17:30:47 Pacific reported `KMD Version: 616.92` and `CUDA UMD Version:
13.4`. NVIDIA defines the latter as the newest CUDA version supported by the
driver; it is not proof that a CUDA 13.4 toolkit is installed. The shared
Python environment reported CPython 3.11.15, PyTorch 2.8.0+cu128, and
`torch.version.cuda == "12.8"`, the CUDA version against which that PyTorch
build was compiled. These are distinct fields, not contradictory claims.
[NVIDIA's field descriptions](https://docs.nvidia.com/deploy/nvidia-smi/index.html#gpu-attributes)
explain the driver and framebuffer reporting semantics.

Selected Windows CIM fields reported Windows 11 Pro, OS version 10.0.26200,
64-bit architecture, an AMD Ryzen 9 5900X processor with 12 physical cores
and 24 logical processors, and 67,032,384 KiB of OS-visible memory. The
machine fields were collected in the same session; their exact query time was
not recorded. OS-visible memory is not a measurement of free RAM.

## Collection protocol

The GPU snapshot was obtained with the following CPU-side query, bracketed by
UTC timestamps from the system clock. The query prints only the selected
fields. The local timezone was supplied by the session context; clock
synchronization was not independently verified.

```powershell
$queryStart = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.reserved,memory.used,memory.free,driver_version --format=csv,noheader,nounits
$queryEnd = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
```

The header was sampled separately with `nvidia-smi | Select-Object -First 3`.
The selected operating-system and processor fields came from
`Get-CimInstance Win32_OperatingSystem` and `Get-CimInstance Win32_Processor`.
The shared interpreter imported `platform` and `torch`, then read
`platform.python_version()`, `platform.python_implementation()`,
`platform.machine()`, `torch.__version__`, and `torch.version.cuda`. No CUDA
availability, device, or memory API was called for this fresh inventory.
Future snapshots should be saved under a new timestamped filename rather than
overwriting this record.

## Earlier CUDA smoke observation

Earlier on October 28, a coordinated setup check used the same PyTorch build
to query the GPU, multiply the two-element CUDA tensor `[1.0, 2.0]` by two,
call `torch.cuda.synchronize()`, and verify `[2.0, 4.0]`. It passed and
reported compute capability 7.5, 8,589,606,912 bytes of CUDA-reported total
memory, and 7,477,395,456 bytes free at its query. The exact wall-clock time
and raw result file were not recorded; these values are transcribed from the
session's ad hoc tool output, not independently reproducible timing evidence.
Project GPU workloads were asked to pause for the smoke, but unrelated desktop
or external GPU activity was not controlled. No latency, throughput, peak
memory, or trace overhead was measured, and the smoke was not repeated during
this inventory.
