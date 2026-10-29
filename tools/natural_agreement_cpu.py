"""Freeze and supervise only the published natural-agreement CPU discovery.

Preparation uses the fixed public development rows and makes no model forward.
Publish the resulting binding and the source revision before invoking run.
The owner entry point accepts no held-out cases or arbitrary model inputs.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
import urllib.parse
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]


class _ProjectSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve project imports only to source files, without adding ROOT to sys.path."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scripts':
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        if fullname.startswith('scripts.'):
            if fullname != 'scripts.v2_source_guards':
                raise ModuleNotFoundError('unapproved project script import')
            return importlib.util.spec_from_file_location(fullname, ROOT / 'scripts/v2_source_guards.py')
        if fullname != 'looking_glass' and not fullname.startswith('looking_glass.'):
            return None
        parts = fullname.split('.')
        if any(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) is None for part in parts):
            raise ModuleNotFoundError('invalid project module name')
        base = ROOT.joinpath(*parts)
        package = base / '__init__.py'
        source = base.with_suffix('.py')
        if package.is_file():
            return importlib.util.spec_from_file_location(fullname, package, submodule_search_locations=[])
        if source.is_file():
            return importlib.util.spec_from_file_location(fullname, source)
        raise ModuleNotFoundError('project module has no source file')


if __name__ == '__main__':
    if any(name == 'looking_glass' or name.startswith('looking_glass.')
           or name == 'scripts' or name.startswith('scripts.') for name in sys.modules):
        raise RuntimeError('project modules were imported before the source bootstrap')
    sys.meta_path.insert(0, _ProjectSourceFinder())
from looking_glass import natural_agreement_controller as control  # noqa: E402

PROTOCOL = '95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee'
CLARIFICATION_PATH = 'docs/PRETRAINED_NUMBER_NATURAL_PREDICTION_CLARIFICATION.md'
CLARIFICATION = 'e9cd1e0d1b1144ffdc7b3c45be12393dd3c285e28b310b3334f4926fdb2b8d06'
AUDITOR_PATH = 'tools/audit_natural_agreement_discovery.py'
AUDITOR = '7b82b476788ccec49c8f48bd30dfd9a21cdedd2c5532057c11b087e08b641a6d'
EXPOSURE = 'e5feaccfeaed288bd53509ee4d68675d6b00ddb1a876279d5c42465010c63344'
MANIFEST_PATH = 'results/distilgpt2-agreement-preflight-v1-20261028/frozen.json'
PREFLIGHT_RESULT = '96bcb687dbdd9bb4017a8506b6acaffeb5a3da43478cf2da551a4a5cb0c723c8'
PREFLIGHT_ARCHIVE = '7c262f3fe81236888a8f5fdb6d31f7835666674be9ffbf36f8ce2ca6a375eb6c'
CHECKPOINT = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
PREFLIGHT_BYTES = 3216448
STUDY_LIMIT_BYTES = 512 * 1024 * 1024
RESERVATIONS = {'natural_discovery': 128 * 1024 * 1024, 'natural_heldout': 160 * 1024 * 1024,
                'injection_discovery': 48 * 1024 * 1024, 'injection_heldout': 72 * 1024 * 1024}
HOST_DISK_RESERVATIONS = {'natural_discovery': 224 * 1024 * 1024, 'natural_heldout': 224 * 1024 * 1024,
                          'injection_discovery': 128 * 1024 * 1024, 'injection_heldout': 192 * 1024 * 1024}
BOOTSTRAP = ("import os,runpy,sys; sys.path.append(os.environ['MLG_NATURAL_SITE']); "
             "script=sys.argv.pop(1); sys.argv[0]=script; runpy.run_path(script,run_name='__main__')")
REQUIRED_STUDY_SOURCES = (
    'tools/natural_agreement_cpu.py', AUDITOR_PATH, 'scripts/v2_source_guards.py',
    'looking_glass/natural_agreement_controller.py', 'looking_glass/agreement_patch_runner.py',
    'looking_glass/agreement_journal.py', 'looking_glass/agreement_heldout.py',
    'looking_glass/pretrained_number_commitments.py', 'looking_glass/natural_prediction_freeze.py',
    'looking_glass/agreement_metrics.py', 'looking_glass/agreement_numeric.py',
    'looking_glass/distilgpt2_intervention_dev.py', 'looking_glass/intervention_plan.py',
    'looking_glass/interventions.py', 'looking_glass/model_ownership.py',
    'looking_glass/pretrained_exposures.py', 'looking_glass/pretrained_exposures_v2.py',
    'looking_glass/pretrained_exposures_v3.py', 'looking_glass/pretrained_exposures_v4.py',
    'looking_glass/pretrained_exposures_v5.py', 'looking_glass/pretrained_exposures_v6.py',
    'looking_glass/pretrained_exposures_v7.py',
    'looking_glass/agreement_preflight_v1.py',
    'looking_glass/pretrained.py',
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n').encode('utf-8')


def _write(path, data):
    with Path(path).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short artifact write')
        stream.flush()
        os.fsync(stream.fileno())


def _claim_budget(binding_path, raw, binding, output):
    """Make one durable local claim after all gates and before starting the owner."""
    free = shutil.disk_usage(output).free
    if free < binding['study_budget']['required_free_disk_bytes']:
        raise RuntimeError('insufficient free disk for the bounded discovery attempt')
    claim = {'schema': 'mlg-natural-study-claim-v1',
             'binding_sha256': hashlib.sha256(raw).hexdigest(), 'output': str(output),
             'claimed_utc': datetime.now(timezone.utc).isoformat(), 'free_disk_bytes': free,
             'budget': binding['study_budget']}
    _write(binding_path.with_name(binding_path.name + '.natural-claim.json'), _bytes(claim))


def _dependency_configuration():
    site = Path(os.environ.get('MLG_NATURAL_SITE', '')).resolve(strict=True)
    if (not site.is_dir() or site.name.lower() != 'site-packages' or site.parent.name.lower() != 'lib'
            or sys.path.count(str(site)) != 1 or ROOT == site or ROOT in site.parents):
        raise RuntimeError('requires the explicitly selected virtualenv site-packages directory')
    config = site.parent.parent / 'pyvenv.cfg'
    data = config.read_bytes()
    if len(data) > 16384:
        raise RuntimeError('virtualenv configuration is too large')
    entries = {}
    for line in data.decode('utf-8').splitlines():
        if '=' in line:
            name, value = (part.strip() for part in line.split('=', 1))
            if name in entries:
                raise RuntimeError('duplicate virtualenv configuration entry')
            entries[name] = value
    home = entries.get('home', '')
    if (not home or not Path(home).is_absolute() or entries.get('implementation') != 'CPython'
            or entries.get('version_info') != '.'.join(map(str, sys.version_info[:3]))
            or entries.get('include-system-site-packages', '').lower() != 'false'
            or Path(home).resolve() != Path(sys.base_prefix).resolve()
            or Path(sys.executable).resolve() != (Path(sys.base_prefix) / 'python.exe').resolve()):
        raise RuntimeError('requires the matching direct base interpreter and virtualenv')
    return {'site_packages_sha256': hashlib.sha256(str(site).encode('utf-8')).hexdigest(),
            'pyvenv_config_sha256': hashlib.sha256(data).hexdigest()}


def _startup():
    prefix = sys.pycache_prefix
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or not prefix
            or not Path(prefix).is_absolute() or os.path.lexists(prefix)
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '-1'
            or os.environ.get('OMP_NUM_THREADS') != '1' or sys.byteorder != 'little'):
        raise RuntimeError('requires isolated -I -S -B, nonexistent absolute pycache prefix, CPU mask and one OMP thread')
    _dependency_configuration()


def _configure_runtime():
    """Set the preregistered CPU state once, before model work."""
    # Windows stdlib platform.win32_ver otherwise starts `cmd /c ver`, including
    # when Torch asks platform.machine() while importing its native libraries.
    # Supply the same process-observed Win32 version without a descendant.
    import platform
    windows = sys.getwindowsversion()
    version = f'{windows.major}.{windows.minor}.{windows.build}'
    platform._syscmd_ver = lambda *args, **kwargs: ('Windows', str(windows.major), version)
    import torch
    if torch.cuda.is_initialized() or torch.is_autocast_enabled('cpu'):
        raise RuntimeError('CUDA initialization or CPU autocast is unsupported')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise RuntimeError('cannot disable denormal flushing')


def _runtime():
    """Observe the current state without resetting it, including after the run."""
    import huggingface_hub
    import safetensors
    import tokenizers
    import torch
    import transformers
    if torch.cuda.is_initialized() or torch.is_autocast_enabled('cpu'):
        raise RuntimeError('CUDA initialization or CPU autocast is unsupported')
    subnormal = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    normal = torch.tensor([0x00800000], dtype=torch.int32).view(torch.float32)
    probe = [int((subnormal * 1.0).view(torch.int32).item()),
             int((normal * 0.5).view(torch.int32).item())]
    if probe != [1, 0x00400000]:
        raise RuntimeError('CPU subnormal probe failed')
    windows = sys.getwindowsversion()
    return {'python': sys.version, 'python_executable_sha256': _sha(sys.executable),
            'dependency_configuration': _dependency_configuration(),
            'platform': 'Windows',
            'windows_version': {'major': windows.major, 'minor': windows.minor, 'build': windows.build,
                                'platform': windows.platform, 'service_pack': windows.service_pack,
                                'platform_version': list(windows.platform_version)},
            'processor': os.environ.get('PROCESSOR_IDENTIFIER'),
            'cpu_count': os.cpu_count(), 'byteorder': sys.byteorder,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'tokenizers': tokenizers.__version__, 'safetensors': safetensors.__version__,
            'huggingface_hub': huggingface_hub.__version__,
            'torch_build_sha256': hashlib.sha256(torch.__config__.show().encode('utf-8')).hexdigest(),
            'device': 'cpu', 'driver': 'not applicable', 'dtype': 'float32',
            'attention': 'eager', 'evaluation': True, 'cache': False,
            'batch_size': 1, 'cpu_threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'default_dtype': str(torch.get_default_dtype()),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'mkldnn_available': torch.backends.mkldnn.is_available(),
            'mkldnn_enabled': torch.backends.mkldnn.enabled,
            'mkl_available': torch.backends.mkl.is_available(),
            'openmp_available': torch.backends.openmp.is_available(),
            'denormal_probe': probe, 'cpu_rng_seed': 20261028,
            'cuda_visible_devices': '-1', 'cuda_initialized': torch.cuda.is_initialized(),
            'autocast_cpu': torch.is_autocast_enabled('cpu'), 'omp_num_threads': '1',
            'thread_environment': {name: os.environ.get(name) for name in
                                   ('MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def build_binding(cache, *, expected_source=None):
    """Validate public inputs and runtime without loading or running a model."""
    from looking_glass import agreement_preflight_v1 as preflight
    from looking_glass import pretrained_exposures_v7
    from looking_glass.pretrained import ASSETS, MODEL_ID, REVISION
    from huggingface_hub import hf_hub_download
    revision, source = (control.source_inventory(ROOT) if expected_source is None else
                        control.verify_source_inventory(ROOT, *expected_source))
    for required in REQUIRED_STUDY_SOURCES:
        if required not in source:
            raise RuntimeError('required committed study source is absent')
    if source[AUDITOR_PATH] != AUDITOR:
        raise RuntimeError('independent audit source differs from the reviewed bytes')
    if pretrained_exposures_v7.inventory_digest() != EXPOSURE:
        raise RuntimeError('public development exposure inventory differs from the reviewed snapshot')
    if _sha(ROOT / 'docs/PRETRAINED_NUMBER_PATCH_V1.md') != PROTOCOL:
        raise RuntimeError('published protocol bytes changed')
    if _sha(ROOT / CLARIFICATION_PATH) != CLARIFICATION:
        raise RuntimeError('published natural prediction clarification bytes changed')
    raw = (ROOT / MANIFEST_PATH).read_bytes()
    manifest = json.loads(raw)
    if (MODEL_ID != 'distilbert/distilgpt2' or REVISION != CHECKPOINT
            or {name: digest for name, (_, digest) in ASSETS.items()} != manifest['asset_sha256']):
        raise RuntimeError('checkpoint asset identity differs from the frozen public manifest')
    cache = Path(cache).resolve(strict=True)
    for name, (size, digest) in ASSETS.items():
        path = Path(hf_hub_download(repo_id=MODEL_ID, filename=name, revision=REVISION,
                                    cache_dir=str(cache), local_files_only=True, token=False))
        if path.stat().st_size != size or _sha(path) != digest:
            raise RuntimeError('offline checkpoint asset differs from its pinned digest: ' + name)
    runtime = _runtime()
    tokenizer = preflight._tokenizer(cache)
    rows, groups = preflight.tokenize_inputs(tokenizer)
    cases = control.bind_public_cases(raw, rows, groups)
    result_path = ROOT / 'results/distilgpt2-agreement-preflight-v1-20261028/run/result.json'
    archive_path = result_path.parent / 'last_logits.f32'
    if (_sha(result_path) != PREFLIGHT_RESULT or _sha(archive_path) != PREFLIGHT_ARCHIVE
            or archive_path.stat().st_size != PREFLIGHT_BYTES):
        raise RuntimeError('reviewed public preflight evidence changed')
    summary = json.loads(result_path.read_bytes())['summary']
    if (summary.get('informative') is not True or summary['counts']['overall'] != {'correct': 16, 'total': 16}
            or any(summary['counts'][name] != {'correct': 8, 'total': 8}
                   for name in ('singular', 'plural', 'congruent', 'incongruent'))):
        raise RuntimeError('the published preflight did not meet its fixed criterion')
    budget = {'study_limit_bytes': STUDY_LIMIT_BYTES,
              'prior_unique_tensor_bytes': PREFLIGHT_BYTES,
              'prior_payloads': [{'sha256': PREFLIGHT_ARCHIVE, 'bytes': PREFLIGHT_BYTES,
                                  'description': 'public preflight full-vocabulary logits'}],
              'phase_reservations_bytes': dict(RESERVATIONS),
              'unallocated_bytes': STUDY_LIMIT_BYTES - PREFLIGHT_BYTES - sum(RESERVATIONS.values()),
              'future_reservations_status': 'prospective; other track owners must enforce and reconcile',
              'coordination': 'external single study owner; source leases are not a study-wide lock',
              'host_disk_allowance_bytes': 1024 * 1024 * 1024,
              'host_disk_phase_reservations_bytes': dict(HOST_DISK_RESERVATIONS),
              'host_disk_unallocated_bytes': 1024 * 1024 * 1024 - sum(HOST_DISK_RESERVATIONS.values()),
              'host_disk_scope': ('prospective physical evidence including serialized copies and metadata; '
                                  'custodian reconciles preflight copies, failed attempts and overhead; not VRAM'),
              'required_free_disk_bytes': 1024 * 1024 * 1024}
    return {'schema': 'mlg-natural-cpu-binding-v1', 'source_commit': revision,
            'source_sha256': source, 'protocol_sha256': PROTOCOL,
            'natural_prediction_clarification_sha256': CLARIFICATION,
            'independent_natural_audit_source_sha256': AUDITOR,
            'development_exposure_inventory_sha256': EXPOSURE,
            'development_exposure_inventory_version': 'distilgpt2-development-exposures-v7',
            'development_exposure_inventory_scope': 'snapshot at source freeze; recheck latest before held-out draw',
            'preflight_manifest_sha256': control._PUBLIC_MANIFEST_SHA256,
            'preflight_result_sha256': PREFLIGHT_RESULT, 'preflight_archive_sha256': PREFLIGHT_ARCHIVE,
            'preflight_status': 'passed; public exposed development',
            'checkpoint_revision': CHECKPOINT, 'model_id': MODEL_ID,
            'assets_sha256': manifest['asset_sha256'], 'cases': cases,
            'runtime': runtime, 'study_budget': budget, 'execution_timeout_seconds': 7200,
            'expected_forwards': 608, 'track': 'natural discovery only'}


def _published(binding, url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.netloc != 'raw.githubusercontent.com'
            or parsed.query or parsed.fragment
            or not re.fullmatch(r'/EastonSell/model-looking-glass/[0-9a-f]{40}/results/[A-Za-z0-9_./-]+\.json', parsed.path)
            or '..' in parsed.path.split('/')):
        raise RuntimeError('binding publication must be a commit-pinned raw GitHub result URL')
    with urllib.request.urlopen(url, timeout=20) as response:
        public = response.read(2 * 1024 * 1024 + 1)
    if public != binding:
        raise RuntimeError('public binding bytes differ from the local binding')
    data = json.loads(binding)
    for name in ('tools/natural_agreement_cpu.py', AUDITOR_PATH, 'looking_glass/natural_agreement_controller.py',
                 'looking_glass/agreement_patch_runner.py', 'looking_glass/agreement_journal.py',
                 'looking_glass/agreement_heldout.py', 'looking_glass/pretrained_number_commitments.py',
                 'looking_glass/natural_prediction_freeze.py', 'scripts/v2_source_guards.py'):
        source_url = f'https://raw.githubusercontent.com/EastonSell/model-looking-glass/{data["source_commit"]}/{name}'
        with urllib.request.urlopen(source_url, timeout=20) as response:
            remote = response.read(8 * 1024 * 1024 + 1)
        if hashlib.sha256(remote).hexdigest() != data['source_sha256'][name]:
            raise RuntimeError('bound study source is not publicly retrievable at its source commit')
    if data['source_sha256'][AUDITOR_PATH] != data['independent_natural_audit_source_sha256']:
        raise RuntimeError('independent audit digest differs from the bound source')
    clarification_url = f'https://raw.githubusercontent.com/EastonSell/model-looking-glass/{data["source_commit"]}/{CLARIFICATION_PATH}'
    with urllib.request.urlopen(clarification_url, timeout=20) as response:
        clarification = response.read(1024 * 1024 + 1)
    if hashlib.sha256(clarification).hexdigest() != data['natural_prediction_clarification_sha256']:
        raise RuntimeError('bound natural prediction clarification is not publicly retrievable')


def _start_record(raw, binding):
    return {'schema': 'mlg-natural-cpu-attempt-v1',
            'binding_sha256': hashlib.sha256(raw).hexdigest(), 'binding': binding}


class _OwnerAuditHookCheck(RuntimeError):
    pass


def _owner_audit_hook(event, args):
    """Forbid Python-level descendant creation in the cooperating CPU owner."""
    if event == 'mlg.natural_owner.audit_hook_check':
        raise _OwnerAuditHookCheck('owner audit hook is active')
    if event in ('subprocess.Popen', '_winapi.CreateProcess', 'os.system', 'os.spawn', 'os.posix_spawn', 'os.exec',
                 'os.fork', 'os.forkpty', 'os.startfile', 'os.startfile/2'):
        raise RuntimeError('CPU owner subprocess creation is prohibited')


def _install_owner_audit_hook():
    sys.addaudithook(_owner_audit_hook)
    try:
        sys.audit('mlg.natural_owner.audit_hook_check')
    except _OwnerAuditHookCheck:
        return
    raise RuntimeError('CPU owner audit hook could not be verified')


def _child(cache, binding_path, output, publication_url, *, validation_only=False):
    control.await_start()
    _startup()
    _install_owner_audit_hook()
    _configure_runtime()
    from scripts.v2_source_guards import SourceFileLease
    binding = json.loads(Path(binding_path).read_bytes())
    initial = control.verify_source_inventory(ROOT, binding['source_commit'], binding['source_sha256'])
    with SourceFileLease(ROOT, [ROOT / name for name in initial[1]]):
        if control.verify_source_inventory(ROOT, *initial) != initial:
            raise RuntimeError('owner source changed before protection was acquired')
        _child_guarded(cache, binding_path, output, publication_url, expected_source=initial,
                       validation_only=validation_only)


def _child_guarded(cache, binding_path, output, publication_url, *, expected_source=None, validation_only=False):
    raw = Path(binding_path).read_bytes()
    binding = json.loads(raw)
    _published(raw, publication_url)
    if _bytes(build_binding(cache, expected_source=expected_source)) != raw:
        raise RuntimeError('child runtime/source/assets differ from the published binding')
    if validation_only:
        print(json.dumps({'status': 'validated_no_model_forward', 'binding_sha256': hashlib.sha256(raw).hexdigest()}))
        return
    import torch
    from looking_glass.agreement_journal import AgreementJournal
    from looking_glass.agreement_patch_runner import AgreementCase, NaturalAgreementRunner
    from looking_glass.pretrained import load_distilgpt2
    bundle = load_distilgpt2(cache_dir=str(cache), allow_download=False)
    if (bundle.metadata['model_revision'] != CHECKPOINT or bundle.metadata['asset_sha256'] != binding['assets_sha256']
            or bundle.metadata['device'] != 'cpu' or bundle.metadata['dtype'] != 'float32'
            or bundle.metadata['attention_implementation'] != 'eager' or bundle.metadata['use_cache'] is not False):
        raise RuntimeError('loaded model does not match the published binding')
    rows, groups = __import__('looking_glass.agreement_preflight_v1', fromlist=['tokenize_inputs']).tokenize_inputs(bundle.tokenizer)
    if control.bind_public_cases((ROOT / MANIFEST_PATH).read_bytes(), rows, groups) != binding['cases']:
        raise RuntimeError('loaded model tokenizer differs from the frozen public cases')
    cases = [AgreementCase(case_id=row['case_id'], quartet=row['quartet'], lemma=row['lemma'],
                           number=row['number'], input_ids=tuple(row['input_ids']),
                           subject_index=row['subject_index']) for row in binding['cases']]
    journal = AgreementJournal(Path(output) / 'attempt', _start_record(raw, binding),
                               max_total_bytes=binding['study_budget']['phase_reservations_bytes']['natural_discovery'])
    NaturalAgreementRunner(bundle.model, journal, device='cpu', expected_quartets=4).run(cases)
    if torch.cuda.is_initialized() or _bytes(build_binding(cache, expected_source=expected_source)) != raw:
        raise RuntimeError('post-run source, assets or runtime verification failed')


def _guarded_main(args, binding_path, state):
    if args.mode == 'prepare':
        if ROOT == binding_path or ROOT in binding_path.parents:
            raise ValueError('binding output must be outside the checkout')
        raw = _bytes(build_binding(args.cache))
        _write(binding_path, raw)
        print('binding_sha256=' + hashlib.sha256(raw).hexdigest())
        return
    if args.mode in ('child', 'validate-child'):
        if not args.output or not args.published_url:
            raise ValueError('child output and publication URL are required')
        _child(args.cache, binding_path, Path(args.output).resolve(), args.published_url,
               validation_only=args.mode == 'validate-child')
        return
    if not args.output or not args.published_url:
        raise ValueError('run requires a new output directory and published binding URL')
    out = Path(args.output).resolve()
    state['phase'] = 'binding_validation'
    raw = binding_path.read_bytes()
    binding = json.loads(raw)
    if _bytes(build_binding(args.cache)) != raw:
        raise RuntimeError('launcher runtime/source/assets differ from the published binding')
    state['phase'] = 'publication_validation'
    _published(raw, args.published_url)
    state['phase'] = 'budget_claim'
    _claim_budget(binding_path, raw, binding, out)
    prefix = out.parent / ('natural-pycache-' + uuid.uuid4().hex)
    owner_out = out / 'owner'
    command = [sys.executable, '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}', '-c', BOOTSTRAP,
               str(ROOT / 'tools/natural_agreement_cpu.py'),
               'child', '--cache', str(Path(args.cache).resolve()), '--binding', str(binding_path), '--output', str(owner_out),
               '--published-url', args.published_url]
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1'})
    state['phase'] = 'owner_supervision'
    verdict = control.supervise(command, owner_out, binding['cases'], env,
                                expected_start=_start_record(raw, binding), timeout=binding['execution_timeout_seconds'])
    print(json.dumps({'status': verdict['status'], 'output': str(out),
                      'scientific_validation': verdict.get('scientific_validation')}))
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check-runtime', 'prepare', 'run', 'child', 'validate-child'))
    parser.add_argument('--cache', required=True)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--output')
    parser.add_argument('--published-url')
    args = parser.parse_args()
    binding_path = Path(args.binding).resolve()
    if args.mode == 'check-runtime':
        _startup()
        print(json.dumps({'status': 'ok', 'interpreter': sys.executable,
                          'dependencies': _dependency_configuration()}, sort_keys=True))
        return
    if args.mode == 'run':
        if not args.output:
            raise ValueError('run requires an output directory')
        output = Path(args.output).resolve()
        if ROOT == output or ROOT in output.parents or not output.parent.is_dir():
            raise ValueError('run output must be new, outside the checkout, with an existing parent')
        output.mkdir()
        state = {'phase': 'startup'}
        _write(output / 'launcher-started.json', _bytes({'schema': 'mlg-natural-launch-v1',
                                                         'started_utc': datetime.now(timezone.utc).isoformat()}))
        try:
            _startup()
            _configure_runtime()
            from scripts.v2_source_guards import SourceFileLease
            state['phase'] = 'source_acquisition'
            initial = control.source_inventory(ROOT)
            with SourceFileLease(ROOT, [ROOT / name for name in initial[1]]):
                if control.source_inventory(ROOT) != initial:
                    raise RuntimeError('source changed before protection was acquired')
                verdict = _guarded_main(args, binding_path, state)
        except BaseException as error:
            trace = traceback.format_exc().encode('utf-8', errors='replace')
            log_error = None
            try:
                _write(output / 'launcher-error.txt', trace[:1024 * 1024])
            except BaseException as secondary:
                log_error = secondary
            terminal = {'status': 'preflight_failed' if state['phase'] != 'owner_supervision' else 'owner_uncertain',
                        'phase': state['phase'], 'error_type': type(error).__name__,
                        'child_started': False if state['phase'] != 'owner_supervision' else None,
                        'error_log_truncated': len(trace) > 1024 * 1024,
                        'error_log_status': 'unavailable' if log_error else 'written'}
            if log_error:
                terminal['error_log_error_type'] = type(log_error).__name__
                note = getattr(error, 'add_note', None)
                if callable(note):
                    note('The bounded error log could not be written: ' + type(log_error).__name__)
            try:
                _write(output / 'launcher-terminal.json', _bytes(terminal))
            except BaseException as secondary:
                if isinstance(secondary, (KeyboardInterrupt, SystemExit)):
                    raise secondary from error
                if isinstance(log_error, (KeyboardInterrupt, SystemExit)):
                    secondary.__cause__ = error
                    raise log_error from secondary
                note = getattr(error, 'add_note', None)
                if callable(note):
                    note('The terminal record could not be written: ' + type(secondary).__name__)
                raise error from secondary
            if isinstance(log_error, (KeyboardInterrupt, SystemExit)):
                raise log_error from error
            raise
        terminal = {'status': verdict['status'], 'phase': 'owner_supervision',
                    'child_started': None if not verdict['child_reaped'] else True,
                    'cleanup_verified': verdict['cleanup_verified'], 'retry_safe': verdict['retry_safe'],
                    'owner_record': 'owner/supervisor.json',
                    'owner_supervisor_sha256': _sha(output / 'owner/supervisor.json')}
        _write(output / 'launcher-terminal.json', _bytes(terminal))
        if verdict['status'] != 'evidence_complete':
            raise SystemExit(130 if verdict['status'] == 'cancelled' else 1)
        return
    _startup()
    if args.mode in ('child', 'validate-child'):
        _guarded_main(args, binding_path, {})
        return
    _configure_runtime()
    from scripts.v2_source_guards import SourceFileLease
    initial = control.source_inventory(ROOT)
    files = [ROOT / name for name in initial[1]]
    with SourceFileLease(ROOT, files):
        if control.source_inventory(ROOT) != initial:
            raise RuntimeError('source changed before protection was acquired')
        _guarded_main(args, binding_path, {})


if __name__ == '__main__':
    main()
