"""Fit and check a Jacobian lens from the terminal.

    python -m looking_glass.jlens_cli --checkpoint ./models/gpt2

Loads the checkpoint the same way the app does, fits a lens on the built-in
passages (or ``--text-file``), saves it where the app looks for it, and prints
a short report: timings, a batch-size consistency check, and how the Jacobian
lens compares with the logit lens on a few prompts.  Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time

EVALUATION_PROMPTS = (
    "The Eiffel Tower is located in the city of",
    "The capital of Japan is the city of",
    "Two plus two equals",
    "The opposite of hot is",
    "She opened the door and walked into the",
    "The first month of the year is",
    "Water is made of hydrogen and",
    "He picked up the phone and called his",
)


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def compare_methods(state, prompts=EVALUATION_PROMPTS):
    """Rank of the model's own answer at the final position, per layer, under each lens.

    Lower is better: rank 1 means that layer's read-out already names the token
    the model goes on to produce.  Returns ``{layer: {"logit": r, "jacobian": r}}``
    with geometric-mean ranks over the prompts.
    """
    ranks = {}
    for prompt in prompts:
        try:
            run = state.post("run", {"prompt": prompt, "settings": {"max_new_tokens": 1}})
        except ValueError:
            continue        # for example a prompt longer than a tiny model's context
        for method in ("logit", "jacobian"):
            lens = state.post("lens", {"capture_id": run["capture_id"], "method": method, "top_k": 1})
            for layer in lens["layers"]:
                ranks.setdefault(layer["layer_index"], {"logit": [], "jacobian": []})[method].append(layer["cells"][-1]["answer_rank"])
    return {layer: {method: geometric_mean(values) for method, values in by_method.items()}
            for layer, by_method in sorted(ranks.items())}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit and check a Jacobian lens for a local GPT-2 or Llama checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local checkpoint folder, as loaded in the app")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:N (default auto prefers CUDA)")
    parser.add_argument("--prompts", type=int, default=40, help="Number of fitting prompts (default 40)")
    parser.add_argument("--text-file", type=Path, help="Plain-text fitting corpus; passages separated by blank lines")
    parser.add_argument("--dim-batch", type=int, default=8, help="Output dimensions per backward pass (lower uses less memory)")
    parser.add_argument("--lens-dir", type=Path, default=Path.cwd() / "looking-glass-lenses", help="Where the app keeps fitted lenses")
    parser.add_argument("--allow-cpu", action="store_true", help="Fit on CPU even for a wide model (slow)")
    parser.add_argument("--skip-compare", action="store_true", help="Do not run the logit-vs-Jacobian comparison")
    parser.add_argument("--import-lens", type=Path, help="Use a lens file fitted elsewhere with Anthropic's jlens library instead of fitting")
    parser.add_argument("--report", type=Path, help="Also write the report as JSON to this file")
    args = parser.parse_args(argv)

    import torch
    from . import app_jlens, jacobian_lens
    from .app import AppState

    state = AppState(export_root=tempfile.mkdtemp(prefix="mlg-jlens-"), lens_root=args.lens_dir)
    loaded = state.post("load", {"path": str(args.checkpoint), "device": args.device, "mode": "capture"})
    identity = state.runtime.jacobian_identity()
    device = identity["device"]
    report = {"torch": torch.__version__, "device": loaded.get("device"), "family": identity["family"],
              "layers": identity["layer_count"], "width": identity["width"],
              "backward_passes_per_prompt": jacobian_lens.backward_passes(identity["width"], args.dim_batch)}
    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(state.runtime._device)
    print(json.dumps(report), flush=True)
    if args.import_lens:
        status = state.post("jlens-import", {"path": str(args.import_lens)})
        print(json.dumps({"imported": status["lens"]}), flush=True)
        if not args.skip_compare:
            for layer, row in compare_methods(state).items():
                print(f"layer {layer:>3}  logit {row['logit']:>10.1f}  Jacobian {row['jacobian']:>10.1f}")
        return 0
    if device != "cuda" and identity["width"] > app_jlens.CPU_FIT_MAX_WIDTH and not args.allow_cpu:
        print("This model is on CPU. Fitting needs a GPU; pass --allow-cpu to fit anyway (slow).", file=sys.stderr)
        return 2

    text = args.text_file.read_text(encoding="utf-8") if args.text_file else app_jlens.builtin_corpus()
    prompts = state.runtime.jacobian_prompts(text, args.prompts)
    if not prompts:
        print("No passage is long enough to fit on.", file=sys.stderr)
        return 2

    # Consistency: the same prompt with two batch sizes must give the same matrices.
    first, _ = state.runtime.jacobian_fit_prompt(prompts[0], args.dim_batch)
    other_batch = max(1, args.dim_batch // 2) if args.dim_batch > 1 else 2
    second, _ = state.runtime.jacobian_fit_prompt(prompts[0], other_batch)
    difference = max(float((first[layer] - second[layer]).abs().max()) for layer in first)
    scale = max(float(first[layer].abs().max()) for layer in first)
    report["batch_consistency"] = {"dim_batch": [args.dim_batch, other_batch], "max_abs_difference": difference,
                                   "largest_entry": scale, "ok": difference <= 1e-3 * max(1.0, scale)}
    print(json.dumps({"batch_consistency": report["batch_consistency"]}), flush=True)

    fit = jacobian_lens.LensFit(identity["fingerprint"], identity["family"], identity["layer_count"], identity["width"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(state.runtime._device)
    seconds = []
    for index, prompt in enumerate(prompts):
        began = time.perf_counter()
        try:
            jacobians, valid = state.runtime.jacobian_fit_prompt(prompt, args.dim_batch)
        except ValueError as error:
            fit.skipped += 1
            print(f"prompt {index + 1}/{len(prompts)} skipped: {error}", flush=True)
            continue
        fit.add(jacobians, valid)
        seconds.append(time.perf_counter() - began)
        change = "first" if fit.last_change is None else f"{fit.last_change:.2e}"
        print(f"prompt {index + 1}/{len(prompts)}: {len(prompt)} tokens, {seconds[-1]:.1f}s, mean moved {change}", flush=True)
    lens = fit.lens()
    path = jacobian_lens.lens_path(args.lens_dir, identity["fingerprint"])
    lens.save(path)
    report["fit"] = {**lens.summary(), "skipped": fit.skipped, "seconds_total": sum(seconds),
                     "seconds_per_prompt": sum(seconds) / len(seconds), "last_mean_change": fit.last_change,
                     "saved_to": str(path)}
    if device == "cuda":
        report["fit"]["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated(state.runtime._device))
    print(json.dumps({"fit": report["fit"]}), flush=True)

    if not args.skip_compare:
        table = compare_methods(state)
        report["answer_rank_by_layer"] = table
        print("Rank of the model's own answer at the last position (geometric mean over up to "
              f"{len(EVALUATION_PROMPTS)} prompts; lower is better):")
        print(f"{'layer':>6} {'logit lens':>12} {'Jacobian lens':>14}")
        for layer, row in table.items():
            print(f"{layer:>6} {row['logit']:>12.1f} {row['jacobian']:>14.1f}")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Done. Start the app with the same --lens-dir (the default matches) and choose Read-out method: Jacobian lens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
