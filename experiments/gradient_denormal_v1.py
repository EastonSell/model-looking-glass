"""Record a CPU floating-point control for the gradient readout.

Run in a fresh process: changing PyTorch's denormal mode affects that thread.
This is a small deterministic control, not a model or performance benchmark.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import torch

import looking_glass.probes as probes


PATTERNS = (
    "00000001", "007fffff", "00800000",  # positive subnormals and minimum normal
    "80000001", "807fffff", "80800000",  # their negative counterparts
    "00000000", "80000000",              # positive and negative zero
)


def _hex_bits(tensor):
    return [f"{value & 0xffffffff:08x}" for value in tensor.view(torch.int32).tolist()]


def _source_manifest(root):
    paths = ("experiments/gradient_denormal_v1.py", "looking_glass/probes.py")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    result = {}
    for name in paths:
        data = (root / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
        if data != committed:
            raise RuntimeError(f"source differs from committed revision: {name}")
        result[name] = hashlib.sha256(data).hexdigest()
    return revision, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if Path(probes.__file__).resolve() != (root / "looking_glass" / "probes.py").resolve():
        raise RuntimeError("loaded probes module is not from this checkout")
    revision, sources = _source_manifest(root)
    torch.set_num_threads(1)
    values = [int(bits, 16) if int(bits, 16) < 2**31 else int(bits, 16) - 2**32
              for bits in PATTERNS]
    activation = torch.tensor(values, dtype=torch.int32, device="cpu").view(torch.float32)
    observations = []
    for flush in (False, True):
        if not torch.set_flush_denormal(flush):
            raise RuntimeError("CPU denormal-mode control is not supported on this host")
        direct = activation.detach().clone()
        gradient = probes.gradient_readout(activation)
        direct_bits, gradient_bits = _hex_bits(direct), _hex_bits(gradient)
        observations.append({
            "flush_denormal": flush,
            "direct_bits_hex": direct_bits,
            "gradient_bits_hex": gradient_bits,
            "torch_equal": bool(torch.equal(direct, gradient)),
            "bitwise_equal": direct_bits == gradient_bits,
        })
    if _source_manifest(root) != (revision, sources):
        raise RuntimeError("committed source changed during the control")
    report = {
        "protocol": "gradient-denormal-cpu-v1",
        "source_commit": revision,
        "source_sha256": sources,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
        "dtype": "float32",
        "torch_intraop_threads": torch.get_num_threads(),
        "input_bits_hex": list(PATTERNS),
        "observations": observations,
        "limitations": [
            "fixed hand-selected inputs; no statistical or model-level claim",
            "source hashes are consistency evidence, not proof of executed machine code",
            "torch.equal is a floating-point comparison and does not distinguish signed zero",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"protocol": report["protocol"], "observations": observations}, sort_keys=True))


if __name__ == "__main__":
    main()
