"""Direct, no-site source bootstrap for the CPU-only intervention dev smoke."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import stat
import subprocess
import sys
import traceback


PROTOCOL = 'distilgpt2-intervention-dev-v1'
MODEL_ID = 'distilbert/distilgpt2'
REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
INPUT_IDS = [50256, 50256, 50256, 50256]
SEED = 1729
SOURCE_FILES = (
    'scripts/distilgpt2_intervention_dev_bootstrap.py',
    'docs/DISTILGPT2_INTERVENTION_DEV_V1.md',
    'tests/test_distilgpt2_intervention_dev.py',
    'looking_glass/__init__.py',
    'looking_glass/distilgpt2_intervention_dev.py',
    'looking_glass/pretrained.py',
    'looking_glass/intervention_plan.py',
    'looking_glass/interventions.py',
    'looking_glass/model_ownership.py',
)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write(path, payload):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _within(path, root):
    return path == root or root in path.parents


def _bootstrap(site):
    flags = sys.flags
    if not (flags.ignore_environment and flags.no_site and flags.no_user_site
            and flags.safe_path and sys.dont_write_bytecode and flags.optimize == 0):
        raise RuntimeError('requires isolated interpreter flags: -B -E -s -S -P')
    if Path(sys.executable).resolve(strict=True) != Path(sys._base_executable).resolve(strict=True):
        raise RuntimeError('requires the direct base interpreter')
    if not sys.pycache_prefix:
        raise RuntimeError('requires an explicit fresh -X pycache_prefix')
    prefix_path = Path(sys.pycache_prefix)
    if not prefix_path.is_absolute():
        raise RuntimeError('private bytecode prefix must be absolute')
    prefix_stat = prefix_path.lstat()
    if (stat.S_ISLNK(prefix_stat.st_mode)
            or getattr(prefix_stat, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
        raise RuntimeError('private bytecode prefix cannot be an alias')
    prefix = prefix_path.resolve(strict=True)
    if not prefix.is_dir() or any(prefix.iterdir()):
        raise RuntimeError('private bytecode prefix must be empty')
    site = Path(site).resolve(strict=True)
    if not site.is_dir():
        raise RuntimeError('package directory must exist')
    repo = Path(__file__).resolve(strict=True).parents[1]
    base = Path(sys.base_prefix).resolve(strict=True)
    initial = []
    for entry in sys.path:
        if not entry:
            raise RuntimeError('empty startup import path is unsupported')
        path = Path(entry).resolve()
        if not _within(path, base):
            raise RuntimeError('unexpected startup import path')
        initial.append(str(path))
    if _within(prefix, repo) or _within(prefix, site):
        raise RuntimeError('private bytecode prefix must be outside project and packages')
    # Do not place the checkout root on sys.path: a top-level torch.py, numpy.py,
    # or similar file there must not run before its dependency origin is checked.
    sys.path[:] = initial + [str(site)]
    return repo, site, prefix, {
        'ignore_environment': bool(flags.ignore_environment),
        'no_site': bool(flags.no_site), 'no_user_site': bool(flags.no_user_site),
        'safe_path': bool(flags.safe_path), 'dont_write_bytecode': bool(sys.dont_write_bytecode),
        'optimize': flags.optimize, 'hash_randomization': flags.hash_randomization,
        'base_executable': str(Path(sys.executable).resolve()),
        'initial_sys_path': initial, 'effective_sys_path': list(sys.path),
        'project_root': str(repo), 'site_packages': str(site),
        'private_pycache_prefix': str(prefix), 'prefix_initially_empty': True,
    }


def _git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], timeout=30).decode('utf-8').strip()


def _sources(repo, revision, files):
    if _git(repo, 'rev-parse', 'HEAD') != revision:
        raise RuntimeError('source revision changed')
    if _git(repo, 'status', '--porcelain=v1', '--untracked-files=all'):
        raise RuntimeError('source checkout must remain clean')
    result = {}
    for relative in sorted(set(files)):
        path = (repo / relative).resolve(strict=True)
        if not _within(path, repo) or not path.is_file():
            raise RuntimeError('source path is outside the project')
        git_blob = _git(repo, 'rev-parse', f'{revision}:{relative}')
        committed = subprocess.check_output(['git', '-C', str(repo), 'show',
                                             f'{revision}:{relative}'], timeout=30)
        working = path.read_bytes()
        if hashlib.sha256(committed).digest() != hashlib.sha256(working).digest():
            raise RuntimeError(f'raw source bytes differ from Git: {relative}')
        result[relative] = {'git_blob': git_blob,
                            'sha256': hashlib.sha256(working).hexdigest()}
    return result


def _load_project_package(repo):
    """Load the package from the verified checkout without exposing its root."""
    if 'looking_glass' in sys.modules:
        raise RuntimeError('project package was imported before source checks')
    package = (repo / 'looking_glass').resolve(strict=True)
    initializer = (package / '__init__.py').resolve(strict=True)
    if not _within(package, repo) or not _within(initializer, package):
        raise RuntimeError('project package escapes the checkout')
    spec = importlib.util.spec_from_file_location('looking_glass', initializer,
                                                  submodule_search_locations=[str(package)])
    if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
        raise RuntimeError('project package requires a source loader')
    module = importlib.util.module_from_spec(spec)
    sys.modules['looking_glass'] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop('looking_glass', None)
        raise


def _project_imports(repo):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name != 'looking_glass' and not name.startswith('looking_glass.'):
            continue
        spec = getattr(module, '__spec__', None)
        origin = getattr(spec, 'origin', None)
        if not origin or not isinstance(getattr(spec, 'loader', None), importlib.machinery.SourceFileLoader):
            raise RuntimeError(f'project module did not load from Python source: {name}')
        path = Path(origin).resolve(strict=True)
        if not _within(path, repo) or path.suffix != '.py':
            raise RuntimeError(f'project module origin is outside the project: {name}')
        result[name] = path.relative_to(repo).as_posix()
    return result


def _deny(entries, name):
    def fail(*args, **kwargs):
        entries.append(name)
        raise RuntimeError(f'blocked device or network entry: {name}')
    return fail


def _guard_cuda_and_network(torch, events, safe_queries):
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA context was initialized before guards')
    for name in ('init', '_lazy_init', 'get_device_properties', 'current_device', 'set_device',
                 'synchronize', 'manual_seed_all', '_is_in_bad_fork', 'mem_get_info'):
        if hasattr(torch.cuda, name):
            setattr(torch.cuda, name, _deny(events, 'torch.cuda.' + name))
    for name in ('_cuda_init', '_cuda_getDeviceCount'):
        if hasattr(torch._C, name):
            setattr(torch._C, name, _deny(events, 'torch._C.' + name))

    def unavailable():
        safe_queries['cuda_is_available'] += 1
        return False
    def no_devices():
        safe_queries['cuda_device_count'] += 1
        return 0
    torch.cuda.is_available = unavailable
    torch.cuda.device_count = no_devices
    for owner, names in ((socket.socket, ('connect', 'connect_ex')),
                         (socket, ('create_connection', 'getaddrinfo'))):
        for name in names:
            setattr(owner, name, _deny(events, 'socket.' + name))


def _origin(module, site):
    path = Path(module.__file__).resolve(strict=True)
    if not _within(path, site):
        raise RuntimeError(f'unexpected package origin: {module.__name__}')
    return str(path)


def _runtime_status(torch):
    status = {'intraop_threads': torch.get_num_threads(),
              'interop_threads': torch.get_num_interop_threads(),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'matmul_precision': torch.get_float32_matmul_precision()}
    if (status['intraop_threads'] != 1 or status['interop_threads'] != 1
            or not status['deterministic_algorithms'] or status['matmul_precision'] != 'highest'):
        raise RuntimeError('CPU runtime settings differ from the protocol')
    probe = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    status['ftz_min_subnormal_times_one_bits'] = int((probe * 1.0).view(torch.int32).item())
    if status['ftz_min_subnormal_times_one_bits'] != 0:
        raise RuntimeError('flush-to-zero probe failed')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', required=True)
    parser.add_argument('--cache')
    parser.add_argument('--output')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    repo, site, prefix, bootstrap = _bootstrap(args.site)
    if args.check_only:
        print(json.dumps(bootstrap, sort_keys=True))
        return
    if not args.cache or not args.output:
        parser.error('--cache and --output are required')
    cache = Path(args.cache)
    output = Path(args.output)
    if not cache.is_absolute() or not output.is_absolute():
        raise ValueError('cache and output must be absolute paths')
    cache = cache.resolve(strict=True)
    output = output.resolve()
    if not cache.is_dir() or _within(output, repo) or not output.parent.is_dir():
        raise ValueError('cache and output locations are invalid')
    output.mkdir()
    phase = 'attempt_manifest'
    partial = {}
    source = None
    started = {'schema': 'distilgpt2-intervention-dev-attempt-v1', 'protocol': PROTOCOL,
               'started_at_utc': _utc(), 'input_ids': [INPUT_IDS],
               'attention_mask': [[1, 1, 1, 1]], 'seed': SEED,
               'model_id': MODEL_ID, 'model_revision': REVISION, 'bootstrap': bootstrap}
    try:
        phase = 'source_preflight'
        revision = _git(repo, 'rev-parse', 'HEAD')
        started['source_commit'] = revision
        _write(output / 'attempt.json', started)
        source = {'commit': revision, 'tree': _git(repo, 'show', '-s', '--format=%T', revision),
                  'files': _sources(repo, revision, SOURCE_FILES)}
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
            raise RuntimeError('CUDA_VISIBLE_DEVICES must be -1 for this CPU smoke')
        if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
            raise RuntimeError('offline environment flags are required')

        phase = 'torch_import_and_guards'
        import torch
        torch_origin = _origin(torch, site)
        events = []
        safe_queries = {'cuda_is_available': 0, 'cuda_device_count': 0}
        _guard_cuda_and_network(torch, events, safe_queries)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.random.default_generator.manual_seed(SEED)
        if not torch.set_flush_denormal(True):
            raise RuntimeError('CPU flush-to-zero is unsupported')
        initial_runtime = _runtime_status(torch)

        phase = 'package_and_project_imports'
        import huggingface_hub
        import numpy
        import safetensors
        import tokenizers
        import transformers
        _load_project_package(repo)
        from looking_glass import distilgpt2_intervention_dev as smoke, pretrained
        packages = {module.__name__: {'version': module.__version__, 'origin': _origin(module, site)}
                    for module in (torch, numpy, transformers, huggingface_hub, safetensors, tokenizers)}
        if packages['torch']['origin'] != torch_origin:
            raise RuntimeError('torch import origin changed')
        if pretrained.MODEL_ID != MODEL_ID or pretrained.REVISION != REVISION:
            raise RuntimeError('loader checkpoint identity changed')
        if smoke.PROTOCOL != PROTOCOL or smoke.INPUT_IDS != tuple(INPUT_IDS) or smoke.SEED != SEED:
            raise RuntimeError('runner protocol constants changed')
        project_imports = _project_imports(repo)
        source['files'] = _sources(repo, revision, (*SOURCE_FILES, *project_imports.values()))

        phase = 'offline_model_load'
        bundle = pretrained.load_distilgpt2(cache_dir=str(cache), allow_download=False)
        model = bundle.model
        if (bundle.tokenizer.eos_token_id != 50256 or model.config.vocab_size != 50257
                or model.config.n_embd != 768 or len(model.transformer.h) != 6):
            raise RuntimeError('pinned checkpoint dimensions or EOS identity differ')
        if (bundle.metadata['weights_sha256'] !=
                'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'):
            raise RuntimeError('verified checkpoint hash differs')
        phase = 'intervention_controls'
        ids = torch.tensor([INPUT_IDS], dtype=torch.int64, device='cpu')
        smoke.evaluate(model, ids, report=partial)

        phase = 'final_verification'
        project_imports_after = _project_imports(repo)
        final_sources = _sources(repo, revision, (*SOURCE_FILES, *project_imports_after.values()))
        if source['files'] != final_sources or project_imports != project_imports_after:
            raise RuntimeError('project imports or source changed during the smoke')
        final_runtime = _runtime_status(torch)
        if final_runtime != initial_runtime:
            raise RuntimeError('CPU runtime settings changed during the smoke')
        if events or torch.cuda.is_initialized() or any(prefix.iterdir()):
            raise RuntimeError('device, network or bytecode-cache guard failed')
        result = {
            'schema': 'distilgpt2-intervention-dev-result-v1', 'protocol': PROTOCOL,
            'status': 'passed', 'evidence_status': 'development-only',
            'started_at_utc': started['started_at_utc'], 'finished_at_utc': _utc(),
            'input_ids': [INPUT_IDS], 'attention_mask': [[1, 1, 1, 1]],
            'checkpoint': bundle.metadata,
            'checkpoint_source': {'cache_dir': str(cache), 'allow_download': False},
            'source': source, 'project_imports': project_imports,
            'runtime': {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
                        'device': 'cpu', 'seed': SEED, 'intraop_threads': torch.get_num_threads(),
                        'interop_threads': torch.get_num_interop_threads(),
                        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                        'matmul_precision': torch.get_float32_matmul_precision(),
                        'flush_to_zero_enabled': True,
                        'runtime_settings_before': initial_runtime,
                        'runtime_settings_after': final_runtime,
                        'cuda_initialized': bool(torch.cuda.is_initialized()),
                        'blocked_cuda_or_network_calls': events, 'safe_cuda_query_counts': safe_queries,
                        'bootstrap': bootstrap, 'prefix_empty_at_completion': True},
            'observations': partial,
            'limitations': ['single public development input', 'not held-out', 'not a causal localization result',
                            'no GPU measurement', 'hash-only outputs require replay to recompute tensor errors'],
        }
        _write(output / 'result.json', result)
    except BaseException as error:
        failure = {'schema': 'distilgpt2-intervention-dev-failure-v1', 'protocol': PROTOCOL,
                   'status': 'failed', 'finished_at_utc': _utc(), 'phase': phase,
                   'error_type': type(error).__name__, 'message': str(error),
                   'traceback': traceback.format_exc(), 'source': source, 'partial': partial}
        _write(output / 'failure.json', failure)
        raise
    print(str(output / 'result.json'))


if __name__ == '__main__':
    main()
