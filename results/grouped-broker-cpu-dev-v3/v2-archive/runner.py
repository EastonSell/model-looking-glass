"""Reproducible CPU-only integration review; not a benchmark or profile."""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback


repo = Path(sys.argv[1]).resolve(strict=True)
out_path = Path(sys.argv[2]).resolve()
workspace = Path(__file__).resolve().parent
protocol_path = workspace / 'grouped_broker_cpu_dev_protocol_v1.json'
protocol_bytes = protocol_path.read_bytes()
protocol = json.loads(protocol_bytes)
if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
    raise RuntimeError('this review requires CUDA_VISIBLE_DEVICES=-1')
sys.path.insert(0, str(repo))

with out_path.open('x', encoding='utf-8') as stream:
    def write(kind, **fields):
        stream.write(json.dumps({'kind': kind, **fields}, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()

    def source_digest(path):
        return hashlib.sha256((repo / path).read_bytes()).hexdigest()

    write('start', protocol=protocol, protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
          script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          repo_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
          source_sha256={path: source_digest(path) for path in (
              'looking_glass/grouped_broker.py', 'looking_glass/grouped_worker.py',
              'looking_glass/grouped_broker_state.py', 'looking_glass/budgeted_worker.py',
              'looking_glass/resident_worker.py', 'looking_glass/pretrained.py',
              'looking_glass/batch_validation.py', 'looking_glass/certified_batch_policy.py')})
    try:
        import torch
        import transformers
        from looking_glass import batch_validation
        from looking_glass.budgeted_worker import BudgetedWorker, DeviceAdmission, WorkloadProfile
        from looking_glass.certified_batch_policy import CertifiedBatchPolicy, runtime_fingerprint
        from looking_glass.grouped_broker import GroupedBroker, SnapshotReservation, ValidationReservation
        from looking_glass.grouped_worker import GroupedWorker
        from looking_glass.host_budget import HostBudget
        from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
        from looking_glass.microbatch import BatchProfile, ForwardContract
        from looking_glass.pretrained import ASSETS, MODEL_ID, REVISION, load_distilgpt2
        from looking_glass.resident_worker import ForwardRequest, ResidentWorker

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False) or torch.cuda.is_initialized():
            raise RuntimeError('expected a denormal-preserving CPU process without initialized CUDA')
        seed = protocol['seed']
        torch.random.default_generator.manual_seed(seed)
        loaded = load_distilgpt2(cache_dir=str(workspace / 'hf-cache' / 'hub'), allow_download=False)
        model = loaded.model
        if model.training or model.config.use_cache is not False:
            raise RuntimeError('expected the evaluated model with cache disabled')
        runtime = {'scope': 'local-cpu-development-only-not-an-attested-runtime',
                   'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                   'transformers': transformers.__version__, 'device': 'cpu',
                   'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                   'deterministic': torch.are_deterministic_algorithms_enabled(),
                   'cuda_initialized': torch.cuda.is_initialized()}
        model_id = hashlib.sha256(json.dumps(loaded.metadata['asset_sha256'],
                                            sort_keys=True).encode()).hexdigest()
        input_schema = (('attention_mask', torch.int64), ('input_ids', torch.int64))
        resident = ResidentWorker(model_id, model, device='cpu', max_batch_size=2,
                                  max_sequence_length=4, max_input_bytes=128,
                                  max_output_bytes=2 * 1024 * 1024)
        device_budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                                     max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        gate = DeviceAdmission(device_budget, device='cpu', max_estimated_incremental_bytes=4 * 1024 * 1024)
        signatures = (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64'))
        workloads = tuple(WorkloadProfile('cpu-development-' + str(n), model_id, 'cpu', (n, 4),
                                          signatures, 2 * 1024 * 1024, 3 * 1024 * 1024)
                          for n in (1, 2))
        worker = BudgetedWorker(resident, gate, workloads)
        contract = ForwardContract(model_id, REVISION, 'cpu-development-review', 'cpu', torch.float32,
                                   'sha256:' + runtime_fingerprint(runtime), seed, True, True,
                                   'eval', 'none', 'disabled', 'none', False, False)
        output_bytes = 2 * 4 * 50257 * 4
        profile = BatchProfile('cpu-review-profile', 'cpu-review-sample', contract, input_schema,
                               (1, 4), torch.float32, 50257, 1e-6, 1e-5, (2,),
                               2, 1, 2, 8, 128, output_bytes, 128 + output_bytes)
        identity = {'model_fingerprint': model_id,
                    'checkpoint_sha256': ASSETS['model.safetensors'][1],
                    'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'backend': 'cpu-eager-development'}
        binding = batch_validation.binding_from_profile(profile, **identity)

        def inputs(row):
            return {'input_ids': torch.tensor([row], dtype=torch.int64),
                    'attention_mask': torch.ones((1, 4), dtype=torch.int64)}

        groups = [{'group_id': 'group-' + str(i + 1), 'cases': [
            {'case_id': f'case-{i + 1}-{j + 1}', 'owner': ('team-a', 'team-b')[j],
             'inputs': inputs(row)} for j, row in enumerate(rows)]}
                  for i, rows in enumerate(protocol['inputs']['groups'])]
        discovery = [digest for row in protocol['inputs']['discovery_rows_not_forwarded']
                     for digest in batch_validation.input_row_digests(binding, inputs(row))]
        frozen = batch_validation.freeze_protocol(binding, groups, discovery_row_hashes=discovery)
        def forward(values, *, request_id, seed):
            return worker.execute(ForwardRequest(request_id, values, seed)).logits
        report = batch_validation.run_validation(frozen, binding, groups, forward,
                                                 expected_protocol_sha256=frozen['protocol_sha256'])
        policy = CertifiedBatchPolicy(profile, frozen, report,
                                      expected_protocol_sha256=frozen['protocol_sha256'],
                                      evidence_sha256=report['evidence_sha256'], minimum_groups=2,
                                      max_atol=1e-6, max_rtol=1e-5, runtime_manifest=runtime, **identity)
        write('sampled_policy', runtime=runtime, model_metadata=loaded.metadata,
              frozen=frozen, report=report)
        host = HostBudget(max_bytes=64 * 1024 * 1024, max_items=256,
                          max_requests=8, max_operations=16)
        grouped = GroupedWorker(worker, (profile,), host_budget=host,
                                finite_scratch_bytes={profile.profile_id: 4 * 1024 * 1024},
                                finite_scratch_items={profile.profile_id: 8})
        snapshot = SnapshotReservation(policy.compatibility_key, 'e' * 64, 256, 8)
        validation = ValidationReservation(policy.compatibility_key, 2, 'd' * 64, 4 * 1024 * 1024, 8)
        broker = GroupedBroker(grouped, profile, policy, host_budget=host, identity=identity,
                               collect_runtime=lambda: dict(runtime), snapshot_reservation=snapshot,
                               validation_reservations=(validation,), max_pending=4,
                               max_pending_per_team=2, max_coalesce_wait=0.1, max_lifetime_jobs=8)
        model_code = type(model).forward.__code__
        observed = []
        def observe(frame, event, arg):
            if event == 'call' and frame.f_code is model_code:
                values = frame.f_locals.get('input_ids')
                observed.append(tuple(values.shape) if values is not None else None)

        for index, group in enumerate(groups):
            cases = group['cases']
            request_id = f'group-{index + 1}-request'
            sys.setprofile(observe)
            try:
                serial = [worker.execute(ForwardRequest('serial-' + str(index) + '-' + str(i),
                                                        case['inputs'], seed)).logits
                          for i, case in enumerate(cases)]
                handles = [broker.submit(case['owner'], ForwardRequest(request_id, case['inputs'], seed))
                           for case in cases]
                if not broker.run_once(timeout=3):
                    raise RuntimeError('cohort did not dispatch')
            finally:
                sys.setprofile(None)
            tickets = [handle.delivery() for handle in handles]
            results = [ticket.wait(3) for ticket in tickets]
            comparisons = []
            for i, (case, result, reference) in enumerate(zip(cases, results, serial)):
                if (result.status != 'ok' or result.team_id != case['owner']
                        or result.request_id != request_id or result.logits is None):
                    raise RuntimeError('wrong or missing routed group output')
                delta = (result.logits.double() - reference.double()).abs()
                tolerance = 1e-6 + 1e-5 * reference.double().abs()
                ratio = delta / tolerance
                comparisons.append({'case_id': case['case_id'], 'owner': case['owner'],
                                    'max_absolute': delta.max().item(),
                                    'max_tolerance_ratio': ratio.max().item(),
                                    'violations': int((delta > tolerance).sum().item())})
                if comparisons[-1]['violations']:
                    raise RuntimeError('group output exceeded the frozen tolerance')
            if observed[-3:] != [(1, 4), (1, 4), (2, 4)]:
                raise RuntimeError('unexpected actual model entry shapes: ' + repr(observed[-3:]))
            write('group_completed', group=group['group_id'], observed_shapes=observed[-3:],
                  comparisons=comparisons)
            for ticket, result in zip(tickets, results):
                ticket.confirm(result)
            del result, reference, delta, tolerance, ratio, results, serial
            for ticket in tickets:
                ticket.release()
            for handle in handles:
                handle.dispose()
            if host.snapshot().used_bytes != 0 or device_budget.snapshot().active_leases != 0:
                raise RuntimeError('cooperative claims did not drain after disposal')
        if torch.cuda.is_initialized():
            raise RuntimeError('unexpected CUDA initialization')
        write('complete', host_used_bytes=host.snapshot().used_bytes,
              device_active_leases=device_budget.snapshot().active_leases,
              note='Same public cases used for sample policy and broker smoke; not independent held-out or GPU evidence.')
    except BaseException as exc:
        write('incomplete', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
