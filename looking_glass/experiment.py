"""Run the prespecified, bounded CPU experiment; never overwrite evidence."""
import argparse
from contextlib import nullcontext
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from .model import TinyResidual
from .probes import TraceSession


PACKAGE = Path(__file__).resolve().parent


def summarize_times(ns):
    if not ns or any(n < 0 for n in ns):
        raise ValueError("timing samples must be nonempty and nonnegative")
    values = sorted(ns)
    return {"median_us": statistics.median(values)/1000,
            "p90_us": values[min(len(values)-1, int(.9*(len(values)-1)))]/1000,
            "samples": len(values)}


def first_difference(a, b):
    if len(a) != len(b):
        raise ValueError("trace lengths differ")
    previous = None
    for x, y in zip(a, b):
        if (x.node_id, x.occurrence) != (y.node_id, y.occurrence):
            raise ValueError("trace metadata differs")
        if x.value is None or y.value is None:
            raise ValueError("value comparison requires value-capture traces")
        if not torch.equal(x.value, y.value):
            return {"last_equal": previous, "first_unequal": x.node_id}
        previous = x.node_id
    return None


def _snapshot(model, x, mode=None, placement="dense"):
    model.zero_grad(set_to_none=True)
    xx = x.detach().clone().requires_grad_(True)
    session = TraceSession(model, mode=mode, placement=placement) if mode else nullcontext()
    with session:
        output = model(xx, route=1)
        loss = output.square().mean()
        loss.backward()
        gradients = {name: p.grad.detach().clone() if p.grad is not None else None
                     for name, p in model.named_parameters()}
        input_gradient = xx.grad.detach().clone()
        torch.optim.SGD(model.parameters(), lr=.01).step()
    return {"output": output.detach(), "loss": loss.detach(),
            "input_gradient": input_gradient, "gradients": gradients,
            "state": copy.deepcopy(model.state_dict())}


def _same_snapshot(a, b):
    checks = {key: torch.equal(a[key], b[key]) for key in ("output", "loss", "input_gradient")}
    for key in ("gradients", "state"):
        checks[key] = all((value is None and b[key][name] is None) or
                         (value is not None and b[key][name] is not None
                          and torch.equal(value,b[key][name])) for name,value in a[key].items())
    return checks


def _capture(model, x, mode, placement, route=0, perturb=None):
    with torch.no_grad(), TraceSession(model, mode=mode, placement=placement) as session:
        output = model(x, route=route, perturb=perturb)
    return output, session


def _time_cell(model, x, mode, placement, warmup, repeats):
    def once():
        with torch.no_grad():
            if mode == 'none':
                model(x)
                return 0, 0
            with TraceSession(model, mode=mode, placement=placement) as session:
                model(x)
            return len(session.events), session.tensor_bytes
    for _ in range(warmup):
        once()
    raw = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        events, tensor_bytes = once()
        raw.append(time.perf_counter_ns()-start)
    return {**summarize_times(raw), "raw_ns":raw, "events":events,
            "recorded_tensor_bytes":tensor_bytes}


def _hash_sources():
    files = sorted(PACKAGE.glob('*.py')) + [PACKAGE/'PROTOCOL.md']
    return {str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _evaluate(quick):
    if torch.is_autocast_enabled('cpu'):
        raise RuntimeError('CPU autocast must be disabled for the float32 protocol')
    torch.set_num_threads(1)
    widths, seeds = ([4],[0]) if quick else ([16,64],[0,1,2,3])
    warmup,repeats = (1,3) if quick else (25,100)
    timing, transparency, reconstruction, localization = [],[],[],[]
    for width in widths:
        for seed in seeds:
            torch.manual_seed(seed)
            model = TinyResidual(width,depth=4).eval()
            x = torch.randn(8,width, device="cpu", dtype=torch.float32)
            baseline = _snapshot(copy.deepcopy(model),x)
            cells = [('none','none')] + [(m,p) for m in ('direct','gradient','id') for p in ('sparse','dense')]
            random.Random(seed+width).shuffle(cells)
            for mode,placement in cells:
                if mode != 'none':
                    checks = _same_snapshot(baseline,_snapshot(copy.deepcopy(model),x,mode,placement))
                    transparency.append({"width":width,"seed":seed,"mode":mode,"placement":placement,**checks})
                    if not all(checks.values()):
                        raise AssertionError(f"transparency mismatch: {transparency[-1]}")
                timing.append({"width":width,"seed":seed,"mode":mode,"placement":placement,
                               **_time_cell(model,x,mode,placement,warmup,repeats)})
            for placement in ('sparse','dense'):
                _,direct = _capture(model,x,'direct',placement)
                _,gradient = _capture(model,x,'gradient',placement)
                equal = first_difference(direct.events,gradient.events) is None
                reconstruction.append({"width":width,"seed":seed,"placement":placement,"equal":equal,
                                       "tensor_elements":sum(e.value.numel() for e in direct.events)})
                if not equal:
                    raise AssertionError("capture channels differ")
                for node in [n for n in model.active_nodes(0) if not n.endswith('.output')]:
                    _,modified = _capture(model,x,'direct',placement,perturb=(node,.125))
                    difference = first_difference(direct.events,modified.events)
                    localization.append({"width":width,"seed":seed,"placement":placement,
                                         "injected_at":node,"detected":difference is not None,
                                         "exact_first_point":difference is not None and difference['first_unequal']==node,
                                         "observed_interval":difference})
    torch.manual_seed(111)
    ambiguity_model = TinyResidual(4,depth=2,identical_branches=True).eval()
    xx = torch.randn(2,4, device="cpu", dtype=torch.float32)
    ya,a = _capture(ambiguity_model,xx,'direct','dense',route=0)
    yb,b = _capture(ambiguity_model,xx,'direct','dense',route=1)
    value_ambiguous = torch.equal(ya,yb) and all(torch.equal(e.value,f.value) for e,f in zip(a.events,b.events))
    metadata_distinct = [e.node_id for e in a.events] != [e.node_id for e in b.events]
    if not value_ambiguous or not metadata_distinct:
        raise AssertionError("identical-output branch control failed")
    return {"protocol_mode":"smoke, not full protocol" if quick else "full protocol v1",
            "environment":{"python":platform.python_version(),"torch":torch.__version__,
                           "platform":platform.platform(),"device":"cpu","dtype":"float32","threads":1},
            "source_sha256":_hash_sources(),"config":{"widths":widths,"seeds":seeds,"depth":4,
                    "batch":8,"warmup":warmup,"repeats":repeats},
            "all_transparent":True,"channels_equal":True,
            "identical_branches_value_ambiguous":value_ambiguous,
            "identical_branches_metadata_distinct":metadata_distinct,
            "transparency":transparency,"reconstruction":reconstruction,
            "localization":localization,"timing":timing,
            "limits":["Synthetic residual network, CPU eager float32 only",
                      "recorded_tensor_bytes is not peak memory",
                      "explicit execution labels are not causal explanations",
                      "timing includes session setup, capture, metadata and cleanup"]}


def run_experiment(output_dir, quick=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    try:
        result = _evaluate(quick)
        (output_dir/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        with (output_dir/'timing.csv').open('x',newline='') as f:
            fields = ['width','seed','mode','placement','median_us','p90_us','samples','events','recorded_tensor_bytes']
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows({k:row[k] for k in fields} for row in result['timing'])
        with (output_dir/'raw_timing.csv').open('x',newline='') as f:
            fields = ['width','seed','mode','placement','sample','nanoseconds']
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for row in result['timing']:
                for index, ns in enumerate(row['raw_ns']):
                    writer.writerow({**{k:row[k] for k in fields[:4]},'sample':index,'nanoseconds':ns})
        return result
    except BaseException as exc:
        try:
            (output_dir/'failure.json').write_text(json.dumps({"type":type(exc).__name__,"message":str(exc)})+'\n')
        except Exception as recording_error:
            raise exc from recording_error
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--quick',action='store_true',help='smoke check, not the full protocol')
    args = parser.parse_args()
    result = run_experiment(args.output,args.quick)
    print(json.dumps({"output":str(args.output),"mode":result['protocol_mode'],
                      "transparency_cases":len(result['transparency']),
                      "timing_cells":len(result['timing']),"all_transparent":result['all_transparent']}))


if __name__ == '__main__':
    main()
