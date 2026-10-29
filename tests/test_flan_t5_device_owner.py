"""CPU-only custody tests for the future FLAN-T5 device owner."""

import gc
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from looking_glass.flan_t5_device_observation import (GPU_TRACE_CLAIM, HOST_TRACE_CLAIM,
                                                     Observation, ObservationCustody,
                                                     _consume_observation_claim)
from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget, MemoryLease


def subject():
    try:
        return importlib.import_module('looking_glass.flan_t5_device_owner')
    except ModuleNotFoundError as error:
        if error.name != 'looking_glass.flan_t5_device_owner':
            raise
        pytest.fail('the FLAN-T5 device evidence owner is unavailable')


def make_lease():
    gib = 1024 ** 3
    budget = MemoryBudget(lambda: CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0),
                          max_gpu_trace_bytes=GPU_TRACE_CLAIM, max_host_trace_bytes=HOST_TRACE_CLAIM)
    lease = budget.acquire(GPU_TRACE_CLAIM, max_gpu_trace_bytes=GPU_TRACE_CLAIM,
                           max_host_trace_bytes=HOST_TRACE_CLAIM)
    lease.reserve_trace('gpu', GPU_TRACE_CLAIM)
    lease.reserve_trace('host', HOST_TRACE_CLAIM)
    return budget, lease


def make_observation():
    return Observation(torch.zeros((1, 2, 40)), torch.zeros((1, 3, 16)),
                       torch.zeros((1, 3, 16)), torch.ones((1, 3, 16)), {},
                       (8, 9, 1), (1, 1, 1), (0, 10), (1, 1))


def make_custody(observation, lease):
    custody = ObservationCustody()
    _consume_observation_claim(lease, custody)
    custody.entered = True
    custody.host_result = observation
    custody.host_logits, custody.host_state = observation.logits, observation.state
    custody.host_before, custody.host_after = observation.before, observation.after
    custody.host_logits_bits, custody.host_state_bits = observation.logits.view(torch.int32), observation.state.view(torch.int32)
    custody.host_before_bits = None if observation.before is None else observation.before.view(torch.int32)
    custody.host_after_bits = None if observation.after is None else observation.after.view(torch.int32)
    return custody


def test_observation_claim_stays_active_until_returned_storage_alias_is_gone():
    api = subject()
    budget, lease = make_lease()
    observation = make_observation()
    alias = observation.logits.view(-1)
    retained = api.RetainedObservation(observation, make_custody(observation, lease), lease, check_owner=lambda: None)
    retained.drop()
    del observation
    gc.collect()
    with pytest.raises(RuntimeError, match='live'):
        retained.retire()
    assert budget.active_leases() == (lease,)
    assert lease.trace_bytes('gpu') == GPU_TRACE_CLAIM
    assert lease.trace_bytes('host') == HOST_TRACE_CLAIM
    del alias
    gc.collect()
    retained.retire()
    assert budget.active_leases() == ()


def test_ambiguous_claim_release_cannot_be_retried(monkeypatch):
    api = subject()
    budget, lease = make_lease()
    observation = make_observation()
    retained = api.RetainedObservation(observation, make_custody(observation, lease), lease, check_owner=lambda: None)
    retained.drop()
    del observation
    gc.collect()
    original = MemoryLease.release_trace
    def interrupted(self, location, amount):
        original(self, location, amount)
        if location == 'gpu':
            raise RuntimeError('interrupted after mutation')
    monkeypatch.setattr(MemoryLease, 'release_trace', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        retained.retire()
    with pytest.raises(RuntimeError, match='uncertain'):
        retained.retire()
    assert lease in budget.active_leases() and lease.trace_bytes('host') == HOST_TRACE_CLAIM


def test_numpy_alias_also_keeps_host_storage_charged():
    api = subject()
    budget, lease = make_lease()
    observation = make_observation()
    alias = observation.state.numpy()
    retained = api.RetainedObservation(observation, make_custody(observation, lease), lease, check_owner=lambda: None)
    retained.drop()
    del observation
    gc.collect()
    with pytest.raises(RuntimeError, match='live'):
        retained.retire()
    assert lease in budget.active_leases()
    del alias
    gc.collect()
    retained.retire()
    assert budget.active_leases() == ()


def test_foreign_thread_cannot_drop_or_retire_owner_claim():
    api = subject()
    budget, lease = make_lease()
    observation = make_observation()
    retained = api.RetainedObservation(observation, make_custody(observation, lease), lease, check_owner=lambda: None)
    def foreign():
        with pytest.raises(RuntimeError, match='owner thread'):
            retained.drop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(foreign).result(timeout=5)
    assert retained.value is observation and budget.active_leases() == (lease,)
    retained.drop()
    del observation
    gc.collect()
    retained.retire()


def test_unrelated_host_bit_root_is_not_silently_dropped():
    api = subject()
    budget, lease = make_lease()
    observation = make_observation()
    custody = make_custody(observation, lease)
    custody.host_logits_bits = torch.ones_like(custody.host_logits_bits)
    with pytest.raises(ValueError, match='bit'):
        api.RetainedObservation(observation, custody, lease, check_owner=lambda: None)
    assert lease in budget.active_leases()
    lease.release_trace('gpu', GPU_TRACE_CLAIM)
    lease.release_trace('host', HOST_TRACE_CLAIM)
    lease.release()


def test_cross_wired_observation_lease_is_rejected_without_releasing_either_claim():
    api = subject()
    budget_a, lease_a = make_lease()
    budget_b, lease_b = make_lease()
    observation = make_observation()
    custody = make_custody(observation, lease_a)
    other = make_observation()
    other_custody = make_custody(other, lease_b)
    with pytest.raises(ValueError, match='lease'):
        api.RetainedObservation(observation, custody, lease_b, check_owner=lambda: None)
    # A mutable custody attribute alone cannot authorize a different consumed lease.
    custody.consumed_lease = lease_b
    with pytest.raises(ValueError, match='lease'):
        api.RetainedObservation(observation, custody, lease_b, check_owner=lambda: None)
    assert budget_a.active_leases() == (lease_a,)
    assert budget_b.active_leases() == (lease_b,)
    assert other_custody.host_result is other
    for lease in (lease_a, lease_b):
        assert lease.trace_bytes('gpu') == GPU_TRACE_CLAIM
        assert lease.trace_bytes('host') == HOST_TRACE_CLAIM
        lease.release_trace('gpu', GPU_TRACE_CLAIM)
        lease.release_trace('host', HOST_TRACE_CLAIM)
        lease.release()


def test_interrupted_garbage_collection_keeps_claim_and_blocks_retry(monkeypatch):
    api = subject()
    budget, lease = make_lease()
    observation = make_observation()
    retained = api.RetainedObservation(observation, make_custody(observation, lease), lease, check_owner=lambda: None)
    retained.drop()
    del observation
    def interrupted():
        raise RuntimeError('interrupted collection')
    monkeypatch.setattr(api.gc, 'collect', interrupted)
    with pytest.raises(RuntimeError, match='interrupted collection'):
        retained.retire()
    with pytest.raises(RuntimeError, match='uncertain'):
        retained.retire()
    assert lease in budget.active_leases()
    assert lease.trace_bytes('gpu') == GPU_TRACE_CLAIM and lease.trace_bytes('host') == HOST_TRACE_CLAIM


def test_model_budget_and_three_observation_claims_are_separately_rooted():
    api = subject()
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    assert memory.model_lease.estimated_incremental_bytes == 2 * gib + 256 * 1024 ** 2
    assert memory.model_lease.trace_bytes('gpu') == memory.model_lease.trace_bytes('host') == 0
    first, second, third = (memory.reserve_observation() for _ in range(3))
    assert len({first.lease, second.lease, third.lease}) == 3
    assert all(slot.custody.consumed_lease is None for slot in (first, second, third))
    assert len(memory.budget.active_leases()) == 4
    with pytest.raises(RuntimeError, match='three'):
        memory.reserve_observation()
    assert len(memory.budget.active_leases()) == 4 and not memory.failed
    # No observation was attempted; this is just synthetic accounting cleanup.
    for slot in (first, second, third):
        slot.lease.release_trace('gpu', GPU_TRACE_CLAIM)
        slot.lease.release_trace('host', HOST_TRACE_CLAIM)
        slot.lease.release()
    memory.model_lease.release()


def test_interrupted_observation_reservation_retains_all_possible_claims(monkeypatch):
    api = subject()
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    original = MemoryLease.reserve_trace
    def interrupted(self, location, amount):
        original(self, location, amount)
        if location == 'host':
            raise KeyboardInterrupt('after host claim')
    monkeypatch.setattr(MemoryLease, 'reserve_trace', interrupted)
    with pytest.raises(KeyboardInterrupt, match='host claim'):
        memory.reserve_observation()
    assert memory.failed and isinstance(memory.failure, KeyboardInterrupt)
    assert len(memory.slots) == 1 and memory.slots[0].lease in memory.budget.active_leases()
    assert memory.slots[0].lease.trace_bytes('host') == HOST_TRACE_CLAIM
    with pytest.raises(RuntimeError, match='uncertain'):
        memory.reserve_observation()


def test_endpoint_headroom_counts_all_active_observation_commitments(tmp_path, monkeypatch):
    api = subject()
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    slots = [memory.reserve_observation(), memory.reserve_observation()]
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    owner.allocator_cap, owner.memory = 2 * gib, memory
    monkeypatch.setattr(owner, '_guard', lambda: None)
    assert owner._policy_commitment() == 2 * gib + 256 * 1024 ** 2 + 2 * GPU_TRACE_CLAIM
    for slot in slots:
        slot.lease.release_trace('gpu', GPU_TRACE_CLAIM)
        slot.lease.release_trace('host', HOST_TRACE_CLAIM)
        slot.lease.release()
    memory.slots.clear()  # Synthetic teardown; real slots retire through their owner.
    assert owner._policy_commitment() == 2 * gib + 256 * 1024 ** 2
    memory.model_lease.release()


def test_interrupted_model_lease_return_keeps_the_budget_rooted(monkeypatch):
    api = subject()
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    original = MemoryBudget.acquire
    def interrupted(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise KeyboardInterrupt('after model lease publication')
    monkeypatch.setattr(MemoryBudget, 'acquire', interrupted)
    with pytest.raises(KeyboardInterrupt, match='publication'):
        memory.start()
    assert memory.failed and memory.model_lease is None
    assert len(memory.budget.active_leases()) == 1
    with pytest.raises(RuntimeError, match='uncertain'):
        memory.start()


def test_asset_stage_copies_only_fixed_verified_bytes(tmp_path, monkeypatch):
    api = subject()
    from looking_glass import flan_t5_feasibility as feasibility
    source, stage = tmp_path / 'assets', tmp_path / 'stage'
    source.mkdir()
    payloads = {'config.json': b'config', 'model.safetensors': b'weights'}
    manifest_raw = b'fixed test manifest'
    (source / 'asset-manifest.json').write_bytes(manifest_raw)
    for name, payload in payloads.items():
        (source / name).write_bytes(payload)
    manifest = {'files': {name: {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
                          for name, payload in payloads.items()}}
    def verify(directory):
        assert (directory / 'asset-manifest.json').read_bytes() == manifest_raw
        for name, payload in payloads.items():
            assert (directory / name).read_bytes() == payload
        return manifest
    monkeypatch.setattr(feasibility, 'verify_assets', verify)
    monkeypatch.setattr(feasibility, 'MANIFEST_HASH', hashlib.sha256(manifest_raw).hexdigest())
    assert api._stage_assets(source, stage) == manifest
    assert set(path.name for path in stage.iterdir()) == set(payloads) | {'asset-manifest.json'}
    with pytest.raises(FileExistsError):
        api._stage_assets(source, stage)


def test_asset_stage_rejects_mid_copy_change_without_loading_model(tmp_path, monkeypatch):
    api = subject()
    from looking_glass import flan_t5_feasibility as feasibility
    source, stage = tmp_path / 'assets', tmp_path / 'stage'
    source.mkdir()
    (source / 'asset-manifest.json').write_bytes(b'manifest')
    (source / 'config.json').write_bytes(b'changed')
    manifest = {'files': {'config.json': {'bytes': 6, 'sha256': hashlib.sha256(b'config').hexdigest()}}}
    monkeypatch.setattr(feasibility, 'verify_assets', lambda _: manifest)
    monkeypatch.setattr(feasibility, 'MANIFEST_HASH', hashlib.sha256(b'manifest').hexdigest())
    with pytest.raises((ValueError, RuntimeError), match='asset'):
        api._stage_assets(source, stage)


def test_source_binding_uses_exact_committed_bytes_even_if_git_hides_edit(tmp_path):
    api = subject()
    executable = shutil.which('git')
    if executable is None:
        pytest.skip('Git executable is unavailable')
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.run([executable, '-C', str(repo), *args], check=True,
                              capture_output=True).stdout
    git('init', '-q')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    (repo / 'owner.py').write_bytes(b'unchanged\n')
    git('add', 'owner.py')
    git('commit', '-qm', 'initial')
    commit = git('rev-parse', 'HEAD').decode().strip()
    binding = api._verify_source_tree(repo, Path(executable), commit, ('owner.py',))
    assert binding == {'owner.py': {'size': 10, 'sha256': hashlib.sha256(b'unchanged\n').hexdigest()}}
    git('update-index', '--assume-unchanged', 'owner.py')
    (repo / 'owner.py').write_bytes(b'modified!\n')
    with pytest.raises(RuntimeError, match='source'):
        api._verify_source_tree(repo, Path(executable), commit, ('owner.py',))


def test_dependency_origin_is_bounded_and_must_belong_to_selected_site(tmp_path, monkeypatch):
    api = subject()
    site = tmp_path / 'site'
    site.mkdir()
    source = site / 'dependency.py'
    source.write_bytes(b'entry file\n')
    module = SimpleNamespace(__file__=str(source), __spec__=SimpleNamespace(origin=str(source)))
    monkeypatch.setitem(sys.modules, 'synthetic_dependency', module)
    result = api._dependency_origin('synthetic_dependency', site)
    assert result == {'path': 'dependency.py', 'size': 11,
                      'sha256': hashlib.sha256(b'entry file\n').hexdigest()}
    outside = tmp_path / 'outside.py'
    outside.write_bytes(b'outside')
    module.__file__ = module.__spec__.origin = str(outside)
    with pytest.raises(ValueError, match='site'):
        api._dependency_origin('synthetic_dependency', site)


def test_owner_module_does_not_import_model_dependencies():
    root = Path(__file__).resolve().parents[1]
    code = ("import importlib.util, sys\n"
            "root = sys.argv[1]\n"
            "spec = importlib.util.spec_from_file_location('looking_glass', root + '/looking_glass/__init__.py', "
            "submodule_search_locations=[root + '/looking_glass'])\n"
            "package = importlib.util.module_from_spec(spec); sys.modules['looking_glass'] = package; "
            "spec.loader.exec_module(package)\n"
            "import looking_glass.flan_t5_device_owner\n"
            "assert not any(x in sys.modules for x in ('torch', 'transformers', 'numpy'))\n")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('PYTHON')}
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(root)],
                            env=env, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_device_owner_rejects_missing_supervision_before_source_or_cuda(monkeypatch, tmp_path):
    api = subject()
    monkeypatch.setattr(api, '_verify_source_tree', lambda *args: pytest.fail('source read before supervision'))
    monkeypatch.setattr(api, '_import_torch', lambda: pytest.fail('torch imported before supervision'))
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    with pytest.raises((ValueError, RuntimeError), match='supervis'):
        owner.start()


@pytest.mark.parametrize('mismatch', [False, True])
def test_device_owner_checks_uuid_before_allocator_and_roots_model_budget(monkeypatch, tmp_path, mismatch):
    api = subject()
    from looking_glass import flan_t5_device_observation as observation_api
    from looking_glass import gpu_oneshot
    gib = 1024 ** 3
    events, stages = [], []
    class Session:
        def check(self, lease):
            assert type(lease) is Lease
            events.append('owner-check')
        def verify_runtime_binding(self, lease, *, runtime_uuid, device_count):
            assert runtime_uuid == 'GPU-test' and device_count == 1
            if mismatch:
                raise RuntimeError('physical UUID mismatch')
            events.append('verified-uuid')
    class Lease:
        pass
    class Props:
        total_memory = 8 * gib
        name = 'synthetic device'
        uuid = object()
    class Cuda:
        def is_initialized(self): return False
        def device_count(self): return 1
        def get_device_properties(self, device): return Props()
        def get_allocator_backend(self): return 'native'
        def set_per_process_memory_fraction(self, fraction, device):
            assert 'verified-uuid' in events and fraction == .25 and device == 0
            events.append('allocator-cap')
        def get_per_process_memory_fraction(self, device): return .25
        def manual_seed(self, seed): assert seed == 20261028
        def reset_peak_memory_stats(self, device): events.append('reset-peak')
        def synchronize(self, device): events.append('sync')
        def mem_get_info(self, device): return 7 * gib, 8 * gib
        def memory_allocated(self, device): return 0
        def memory_reserved(self, device): return 0
        def max_memory_allocated(self, device): return 0
        def max_memory_reserved(self, device): return 0
    class FakeTorch:
        cuda = Cuda()
    monkeypatch.setattr(api, 'OwnerSession', Session)
    monkeypatch.setattr(api, 'DeviceLease', Lease)
    monkeypatch.setattr(api, '_verify_source_tree', lambda *args: {'source.py': {'size': 1, 'sha256': 'a' * 64}})
    monkeypatch.setattr(api, '_check_bootstrap', lambda *args: None)
    monkeypatch.setattr(api, '_check_workspace', lambda *args: None)
    monkeypatch.setattr(api, '_require_pristine_dependencies', lambda: None)
    monkeypatch.setattr(api, '_import_torch', lambda: FakeTorch())
    monkeypatch.setattr(api, '_configure_numerics', lambda torch: None)
    monkeypatch.setattr(observation_api, '_cuda_numerics', lambda: {'runtime': 'test'})
    monkeypatch.setattr(observation_api, '_cuda_numeric_state', lambda: None)
    monkeypatch.setattr(gpu_oneshot, '_runtime_uuid', lambda torch, value: 'GPU-test')
    session, lease = Session(), Lease()
    owner = api.FlanT5CudaOwner(session, lease, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda stage, data: stages.append((stage, data)))
    if mismatch:
        with pytest.raises(RuntimeError, match='UUID mismatch'):
            owner.start()
        assert 'allocator-cap' not in events
        assert owner.failed and owner.memory is None
        assert isinstance(owner.failure, RuntimeError)
        return
    owner.start()
    assert owner.memory.model_lease in owner.memory.budget.active_leases()
    assert owner.memory.model_lease.estimated_incremental_bytes == 2 * gib + 256 * 1024 ** 2
    assert events.index('verified-uuid') < events.index('allocator-cap')
    assert [stage for stage, _ in stages] == ['before_model_load', 'after_admission']
    assert not owner.failed


def test_model_copy_failure_retains_cpu_model_and_marks_owner_uncertain(monkeypatch, tmp_path):
    api = subject()
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    owner.phase, owner.torch = 'admitted', torch
    monkeypatch.setattr(owner, '_guard', lambda: None)
    monkeypatch.setattr(owner, '_model_guard', lambda: None)
    monkeypatch.setattr(owner, '_stage', lambda stage: None)
    monkeypatch.setattr(api, '_stage_assets', lambda *args: {'fixed': True})
    class Model:
        def to(self, device):
            assert device == 'cuda:0' and owner.model is self
            raise MemoryError('copy failed')
    model = Model()
    monkeypatch.setattr(api, '_load_staged_model', lambda *args: model)
    with pytest.raises(MemoryError, match='copy failed'):
        owner.load_model(tmp_path / 'assets', tmp_path / 'stage')
    assert owner.failed and isinstance(owner.failure, MemoryError)
    assert owner.model is model and owner.stage_directory == tmp_path / 'stage'
    with pytest.raises(RuntimeError, match='uncertain'):
        owner.load_model(tmp_path / 'assets', tmp_path / 'other-stage')


def test_pinned_loader_rejects_a_mismatched_synthetic_t5(monkeypatch, tmp_path):
    api = subject()
    from transformers import T5Config, T5ForConditionalGeneration
    model = T5ForConditionalGeneration(T5Config(vocab_size=40, d_model=16, d_ff=32,
             num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
             pad_token_id=0, eos_token_id=1))
    diagnostics = {name: [] for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
    monkeypatch.setattr(T5ForConditionalGeneration, 'from_pretrained',
                        lambda *args, **kwargs: (model, diagnostics))
    with pytest.raises(ValueError, match='pinned'):
        api._load_staged_model(tmp_path, torch)


def test_pinned_loader_accepts_the_eight_layer_untied_meta_architecture(monkeypatch, tmp_path):
    api = subject()
    from looking_glass import flan_t5_device_observation as observation_api
    from transformers import T5Config, T5ForConditionalGeneration
    config = T5Config(vocab_size=32128, d_model=512, d_ff=1024, d_kv=64,
        num_layers=8, num_decoder_layers=8, num_heads=6, feed_forward_proj='gated-gelu',
        tie_word_embeddings=False, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    with torch.device('meta'):
        model = T5ForConditionalGeneration(config)
    diagnostics = {name: [] for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
    monkeypatch.setattr(T5ForConditionalGeneration, 'from_pretrained',
                        lambda *args, **kwargs: (model, diagnostics))
    # This checks architecture only; a real staged load still requires CPU tensors.
    monkeypatch.setattr(observation_api, '_model_contract', lambda *args: None)
    assert api._load_staged_model(tmp_path, torch) is model


@pytest.mark.parametrize('diagnostics', [
    {},
    {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': [], 'error_msgs': None},
    {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': (), 'error_msgs': []},
    {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': [], 'error_msgs': [], 'other': []},
])
def test_pinned_loader_rejects_incomplete_or_malformed_loading_diagnostics(monkeypatch, tmp_path, diagnostics):
    api = subject()
    from transformers import T5ForConditionalGeneration
    monkeypatch.setattr(T5ForConditionalGeneration, 'from_pretrained',
                        lambda *args, **kwargs: (object(), diagnostics))
    with pytest.raises(ValueError, match='diagnostics'):
        api._load_staged_model(tmp_path, torch)


def test_post_forward_model_drift_keeps_result_and_claim_rooted(monkeypatch, tmp_path):
    api = subject()
    from looking_glass import bert_gpu_family_owner as state_api
    from looking_glass import flan_t5_device_observation as observation_api
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    owner.phase, owner.memory, owner.allocator_cap = 'model-ready', memory, 2 * gib
    owner.model, owner.witness, owner.torch = object(), object(), torch
    monkeypatch.setattr(owner, '_guard', lambda: None)
    monkeypatch.setattr(owner, '_model_guard', lambda: None)
    monkeypatch.setattr(owner, '_stage', lambda stage: None)
    calls = []
    def verify(*args, **kwargs):
        calls.append('state')
        if len(calls) == 2:
            raise RuntimeError('model drift')
    monkeypatch.setattr(state_api, '_verify_state', verify)
    observation = make_observation()
    def observe(*args, custody, memory_lease, **kwargs):
        _consume_observation_claim(memory_lease, custody)
        custody.entered = True
        custody.host_result = observation
        custody.host_logits, custody.host_state = observation.logits, observation.state
        custody.host_before, custody.host_after = observation.before, observation.after
        custody.host_logits_bits, custody.host_state_bits = observation.logits.view(torch.int32), observation.state.view(torch.int32)
        custody.host_before_bits, custody.host_after_bits = observation.before.view(torch.int32), observation.after.view(torch.int32)
        return observation
    monkeypatch.setattr(observation_api, 'observe_cuda_owned', observe)
    with pytest.raises(RuntimeError, match='model drift'):
        owner.observe({}, {}, action='capture')
    assert owner.failed and isinstance(owner.failure, RuntimeError)
    assert len(memory.slots) == 1 and memory.slots[0].custody.host_result is observation
    assert memory.slots[0].lease in memory.budget.active_leases()
    assert memory.model_lease in memory.budget.active_leases()


def test_owner_observation_retains_the_exact_claim_until_host_alias_dies(monkeypatch, tmp_path):
    api = subject()
    from looking_glass import bert_gpu_family_owner as state_api
    from looking_glass import flan_t5_device_observation as observation_api
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    owner.phase, owner.memory, owner.allocator_cap = 'model-ready', memory, 2 * gib
    owner.model, owner.witness, owner.torch = object(), object(), torch
    monkeypatch.setattr(owner, '_guard', lambda: None)
    monkeypatch.setattr(owner, '_model_guard', lambda: None)
    monkeypatch.setattr(owner, '_stage', lambda stage: None)
    monkeypatch.setattr(state_api, '_verify_state', lambda *args, **kwargs: True)
    pending = [make_observation()]
    alias = pending[0].logits.view(-1)
    def observe(*args, custody, memory_lease, **kwargs):
        result = pending.pop()
        _consume_observation_claim(memory_lease, custody)
        custody.entered = True
        custody.host_result = result
        custody.host_logits, custody.host_state = result.logits, result.state
        custody.host_before, custody.host_after = result.before, result.after
        custody.host_logits_bits, custody.host_state_bits = result.logits.view(torch.int32), result.state.view(torch.int32)
        custody.host_before_bits, custody.host_after_bits = result.before.view(torch.int32), result.after.view(torch.int32)
        return result
    monkeypatch.setattr(observation_api, 'observe_cuda_owned', observe)
    retained = owner.observe({}, {}, action='capture')
    slot = memory.slots[0]
    assert retained is slot.retained and slot.custody.consumed_lease is slot.lease
    retained.drop()
    with pytest.raises(RuntimeError, match='live'):
        retained.retire()
    assert slot.lease in memory.budget.active_leases()
    del alias
    gc.collect()
    retained.retire()
    assert memory.budget.active_leases() == (memory.model_lease,)


def test_later_forward_failure_blocks_retiring_an_earlier_observation(monkeypatch, tmp_path):
    api = subject()
    from looking_glass import bert_gpu_family_owner as state_api
    from looking_glass import flan_t5_device_observation as observation_api
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    owner.phase, owner.memory, owner.allocator_cap = 'model-ready', memory, 2 * gib
    owner.model, owner.witness, owner.torch = object(), object(), torch
    monkeypatch.setattr(owner, '_guard', lambda: None)
    monkeypatch.setattr(owner, '_model_guard', lambda: None)
    monkeypatch.setattr(owner, '_stage', lambda stage: None)
    monkeypatch.setattr(state_api, '_verify_state', lambda *args, **kwargs: True)
    observation = make_observation()
    calls = []
    def observe(*args, custody, memory_lease, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError('second forward failed')
        _consume_observation_claim(memory_lease, custody)
        custody.entered = True
        custody.host_result = observation
        custody.host_logits, custody.host_state = observation.logits, observation.state
        custody.host_before, custody.host_after = observation.before, observation.after
        custody.host_logits_bits, custody.host_state_bits = observation.logits.view(torch.int32), observation.state.view(torch.int32)
        custody.host_before_bits, custody.host_after_bits = observation.before.view(torch.int32), observation.after.view(torch.int32)
        return observation
    monkeypatch.setattr(observation_api, 'observe_cuda_owned', observe)
    retained = owner.observe({}, {}, action='capture')
    first_lease = memory.slots[0].lease
    with pytest.raises(MemoryError, match='second forward'):
        owner.observe({}, {}, action='capture')
    with pytest.raises(RuntimeError, match='uncertain'):
        retained.drop()
    assert first_lease in memory.budget.active_leases() and retained._observation is observation


def test_model_guard_allows_ordered_retirement_and_rejects_direct_claim_release(monkeypatch, tmp_path):
    api = subject()
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    owner.phase, owner.memory, owner.allocator_cap = 'model-ready', memory, 2 * gib
    monkeypatch.setattr(owner, '_guard', lambda: None)
    slot = memory.reserve_observation()
    observation = make_observation()
    # The real producer associates the reserved custody before returning a result.
    _consume_observation_claim(slot.lease, slot.custody)
    slot.custody.entered = True
    slot.custody.host_result = observation
    slot.custody.host_logits, slot.custody.host_state = observation.logits, observation.state
    slot.custody.host_before, slot.custody.host_after = observation.before, observation.after
    slot.custody.host_logits_bits = observation.logits.view(torch.int32)
    slot.custody.host_state_bits = observation.state.view(torch.int32)
    slot.custody.host_before_bits = observation.before.view(torch.int32)
    slot.custody.host_after_bits = observation.after.view(torch.int32)
    retained = api.RetainedObservation(observation, slot.custody, slot.lease,
                                       check_owner=owner._model_guard, access_check=owner._retained_guard)
    slot.retained = retained
    retained.drop()
    del observation
    retained.retire()
    owner._model_guard()
    second = memory.reserve_observation()
    second.lease.release_trace('gpu', GPU_TRACE_CLAIM)
    with pytest.raises(RuntimeError, match='trace claims'):
        owner._model_guard()


def test_cleanup_releases_model_claim_only_after_zero_allocator(monkeypatch, tmp_path):
    api = subject()
    from looking_glass import bert_gpu_family_owner as state_api
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    events = []
    class Model:
        pass
    class Cuda:
        def synchronize(self, device): events.append('sync')
        def empty_cache(self): events.append('empty')
    class Native:
        def _cuda_clearCublasWorkspaces(self): events.append('clear')
    class FakeTorch:
        __version__ = '2.8.0+cu128'
        cuda = Cuda()
        _C = Native()
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    owner.phase, owner.memory, owner.allocator_cap = 'model-ready', memory, 2 * gib
    owner.model = owner.converted = Model()
    owner.witness, owner.torch = object(), FakeTorch()
    monkeypatch.setattr(owner, '_guard', lambda: None)
    monkeypatch.setattr(owner, '_model_guard', lambda: None)
    monkeypatch.setattr(owner, '_stage', lambda stage: {'allocated_bytes': 0, 'reserved_bytes': 0})
    monkeypatch.setattr(owner, '_verify_inputs_still_bound', lambda: None)
    monkeypatch.setattr(state_api, '_verify_state', lambda *args, **kwargs: True)
    cleanup = owner.finish()
    assert cleanup['allocated_bytes'] == cleanup['reserved_bytes'] == 0
    assert events == ['sync', 'empty', 'clear', 'sync', 'empty']
    assert memory.budget.active_leases() == () and owner.phase == 'disposed' and not owner.failed


def test_cleanup_refuses_a_live_observation_before_disposal(monkeypatch, tmp_path):
    api = subject()
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    slot = memory.reserve_observation()
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    owner.phase, owner.memory, owner.allocator_cap = 'model-ready', memory, 2 * gib
    owner.model = object()
    monkeypatch.setattr(owner, '_model_guard', lambda: None)
    with pytest.raises(RuntimeError, match='retired'):
        owner.finish()
    assert owner.model is not None
    assert slot.lease in memory.budget.active_leases()
    assert memory.model_lease in memory.budget.active_leases()


def test_cleanup_failure_keeps_model_commitment_and_disables_retry(monkeypatch, tmp_path):
    api = subject()
    from looking_glass import bert_gpu_family_owner as state_api
    gib = 1024 ** 3
    memory = api.DeviceMemoryCustody(lambda: CudaMemorySnapshot(7 * gib, 8 * gib, 0, 0),
                                     allocator_cap=2 * gib, check_owner=lambda: None)
    memory.start()
    class Model:
        pass
    class Cuda:
        def synchronize(self, device): raise RuntimeError('synchronization failed')
    class FakeTorch:
        cuda = Cuda()
    owner = api.FlanT5CudaOwner(None, None, repo=tmp_path, git_executable=tmp_path / 'git',
                               source_commit='0' * 40, record_stage=lambda *args: None)
    owner.phase, owner.memory, owner.allocator_cap = 'model-ready', memory, 2 * gib
    owner.model = owner.converted = Model()
    owner.witness, owner.torch = object(), FakeTorch()
    monkeypatch.setattr(owner, '_guard', lambda: None)
    monkeypatch.setattr(owner, '_model_guard', lambda: None)
    monkeypatch.setattr(owner, '_verify_inputs_still_bound', lambda: None)
    monkeypatch.setattr(state_api, '_verify_state', lambda *args, **kwargs: True)
    with pytest.raises(RuntimeError, match='synchronization failed'):
        owner.finish()
    assert owner.failed and memory.model_lease in memory.budget.active_leases()
    with pytest.raises(RuntimeError, match='uncertain'):
        owner.finish()
