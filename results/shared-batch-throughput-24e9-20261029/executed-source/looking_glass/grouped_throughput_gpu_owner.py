"""Disabled fixed-workload extension for a supervised grouped owner.

This is not a launcher or an admission decision. The parent must source-bind
this module and its dependencies before CUDA work, and keep the existing owner,
device lease and supervision active throughout. An escaping failure retains
the known host objects and leaves recovery to that supervisor.
"""

from pathlib import Path
import weakref

from . import grouped_gpu_dev as _dev
from .grouped_throughput_driver import run_pair
from .grouped_throughput_memory import WindowMemory
from .grouped_throughput_private_evidence import PrivateRows
from .host_budget import HostBudget
from .resident_worker import ForwardRequest
from .throughput_benchmark import canonical_digest


_OWNER_ENABLED = True
_HOST_BYTES = 64 * 1024 * 1024
_ROOTS = [None] * 12
_STARTED = False
_ROWS = (
    ((50256, 103, 203, 303), (50256, 104, 204, 304)),
    ((50256, 105, 205, 305), (50256, 106, 206, 306)),
    ((50256, 107, 207, 307), (50256, 108, 208, 308)),
    ((50256, 109, 209, 309), (50256, 110, 210, 310)),
)


def run_extension(*, session, lease, torch, protocol, run_dir, worker, broker, model,
                  expected_total, runtime_sha256, source_hashes, run_id):
    """Run eight serial and eight grouped rows through the same resident.

    The 64 MiB host claim accounts for the two retained copies of each output
    and bounded scratch; it does not cap process RSS or Python allocations.
    The caller supplies a fresh, independently checked disk/host envelope.
    """
    global _STARTED
    if _OWNER_ENABLED is not True:
        raise RuntimeError('the fixed throughput owner is disabled')
    if _STARTED:
        raise RuntimeError('the fixed throughput extension is single use')
    _STARTED = True
    _ROOTS[0] = {'session': session, 'lease': lease, 'worker': worker,
                 'broker': broker, 'model': model}
    custody = {'windows': {}}
    _ROOTS[1] = custody
    try:
        if (type(protocol) is not dict or protocol.get('seed') != 2026102808
                or protocol.get('model_output_width') != 50257
                or protocol.get('absolute_tolerance') != 1e-6
                or protocol.get('relative_tolerance') != 1e-5
                or type(expected_total) is not int or expected_total <= 0
                or type(runtime_sha256) is not str or len(runtime_sha256) != 64
                or type(source_hashes) is not dict or type(run_id) is not str or not run_id):
            raise RuntimeError('the fixed throughput owner context differs')
        run_dir = Path(run_dir)
        if not run_dir.is_absolute() or run_dir.resolve(strict=True) != run_dir:
            raise ValueError('the private run directory must be canonical')
        row_dir = run_dir / 'throughput-rows'
        row_dir.mkdir()

        host = HostBudget(max_bytes=_HOST_BYTES, max_items=128, max_requests=1, max_operations=1)
        _ROOTS[2] = host
        request_guard = host.request_guard('throughput-fixed')
        _ROOTS[3] = request_guard
        request_guard.open()
        host_request = request_guard.adopt()
        claim_guard = host_request.claim_guard('throughput-fixed', 'inputs-and-outputs',
                                                _HOST_BYTES, items=128)
        _ROOTS[4] = claim_guard
        claim_guard.reserve()
        claim = claim_guard.adopt()

        requests, jobs = {}, []
        _ROOTS[5] = requests
        for ordinal, pair in enumerate(_ROWS):
            for index, tokens in enumerate(pair):
                team = ('team-a', 'team-b')[index]
                job_id = f'{team}-{ordinal}'
                inputs = {'input_ids': torch.tensor([tokens], dtype=torch.int64),
                          'attention_mask': torch.ones((1, 4), dtype=torch.int64)}
                request = ForwardRequest(job_id, inputs, protocol['seed'])
                input_sha256 = canonical_digest(inputs)
                request_sha256 = canonical_digest({'job_id': job_id, 'team_id': team,
                    'seed': protocol['seed'], 'input_sha256': input_sha256})
                jobs.append({'job_id': job_id, 'team_id': team, 'seed': protocol['seed'],
                             'input_sha256': input_sha256, 'request_sha256': request_sha256})
                requests[job_id] = request
        inputs = request = None

        writer = PrivateRows(row_dir)
        memory = WindowMemory(run_dir, session, lease, torch, expected_total, protocol)
        _ROOTS[6], _ROOTS[7] = writer, memory

        def guarded(operation, /, *args, **kwargs):
            return _dev._cuda(session, lease, operation, *args, **kwargs)

        def record(name, value):
            if name not in ('serial-window', 'grouped-window', 'pair', 'failure'):
                raise ValueError('unexpected fixed throughput record')
            _dev._write_json(run_dir / ('throughput-' + name + '.json'), value, max_bytes=256 * 1024)

        result = run_pair(owner=worker, broker=broker, model=model, jobs=jobs, requests=requests,
            guarded_call=guarded, synchronize=lambda: torch.cuda.synchronize(0), memory=memory,
            writer=writer, custody=custody, record=record)
        if result.get('status') != 'complete':
            raise RuntimeError('the paired throughput observation is incomplete')

        # Capture weak references before dropping the two CPU copies of each
        # completed row. No device operation is issued after any failed check.
        references = []
        for window in custody['windows'].values():
            references.extend(weakref.ref(row) for row in window['outputs'].values())
            references.extend(weakref.ref(item.output) for item in window['items'].values())
        window = None
        if len(references) != 32:
            raise RuntimeError('the retained output inventory differs')
        custody['windows'].clear()
        requests.clear()
        if any(reference() is not None for reference in references):
            raise RuntimeError('a completed throughput output remains retained')
        if guarded(claim.release) is not True or guarded(host_request.close) is not True:
            raise RuntimeError('throughput host custody did not close')
        state = host.snapshot()
        if state.used_bytes or state.used_items or state.requests or state.claims or state.operations:
            raise RuntimeError('throughput host custody remains')
        _ROOTS[:] = [None] * len(_ROOTS)
        return result
    except BaseException as exc:
        _ROOTS[8] = exc
        raise
