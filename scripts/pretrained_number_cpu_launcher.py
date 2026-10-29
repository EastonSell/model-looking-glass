"""Prepare and launch the pinned CPU-only number-injection discovery.

The custodian draws one private site nonce during ``seal``. This launcher does
not allocate held-out cases or prove publication chronology or isolation from
other processes running as the same user.
"""
import argparse
import hashlib
import hmac
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import secrets
import stat
import subprocess
import sys
import types
import urllib.request


_PUBLIC_URL = re.compile(
    r'https://raw\.githubusercontent\.com/EastonSell/model-looking-glass/'
    r'[0-9a-f]{40}/[A-Za-z0-9._/-]+\Z'
)
_CONTEXT = None
_SEED = 20261028
_LOGICAL_LIMIT = 48 * 1024 ** 2
_DISK_LIMIT = 128 * 1024 ** 2


def verify_published_bytes(url, expected):
    """Compare bounded bytes to an immutable public GitHub raw-file URL."""
    if (type(url) is not str or not _PUBLIC_URL.fullmatch(url)
            or any(part in ('', '.', '..') for part in url.split('/')[6:])
            or type(expected) is not bytes or not 0 < len(expected) <= 1024 * 1024):
        raise ValueError('a bounded published artifact at an immutable URL is required')
    request = urllib.request.Request(url, headers={'User-Agent': 'looking-glass-cpu-study'})
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read(1024 * 1024 + 1)
        if response.geturl() != url or data != expected:
            raise ValueError('published artifact differs from its expected bytes')


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _runtime_record(raw, source_commit):
    from looking_glass.pretrained_number_commitments import CHECKPOINT_REVISION, PROTOCOL_SHA256
    if (type(raw) is not bytes or not 0 < len(raw) <= 1024 * 1024
            or type(source_commit) is not str or re.fullmatch(r'[0-9a-f]{40}', source_commit) is None):
        raise ValueError('bounded runtime bytes and an exact source revision are required')
    try:
        record = json.loads(raw)
        if (type(record) is not dict or _canonical(record) != raw
                or record.get('schema') != 'mlg-number-injection-cpu-runtime-v1'
                or record.get('source_commit') != source_commit
                or record.get('protocol_sha256') != PROTOCOL_SHA256
                or record.get('checkpoint_revision') != CHECKPOINT_REVISION):
            raise ValueError('runtime manifest does not match this study source')
    except (UnicodeError, TypeError, RecursionError) as error:
        raise ValueError('invalid runtime manifest') from error
    return record


def _write_new(path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            if type(count) is not int or not 0 < count <= len(view):
                raise OSError('incomplete evidence write')
            view = view[count:]
        os.fsync(fd)
    except BaseException as error:
        try:
            os.close(fd)
        except BaseException as close_error:
            if (isinstance(close_error, (KeyboardInterrupt, SystemExit))
                    and not isinstance(error, (KeyboardInterrupt, SystemExit))):
                raise close_error from error
        raise
    else:
        os.close(fd)


def _read_bounded(path, limit):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('an absolute file path is required')
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
            or not 0 < info.st_size <= limit):
        raise ValueError('a bounded ordinary file is required')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) != info.st_size:
        raise ValueError('file changed while reading')
    return data


def _verify_unloaded_source(repo, revision):
    """Check raw tracked project blobs before executing any project module."""
    root = Path(repo).resolve(strict=True)
    if type(revision) is not str or re.fullmatch(r'[0-9a-f]{40}', revision) is None:
        raise ValueError('a full source commit is required')
    git_env = {name: value for name, value in os.environ.items() if not name.upper().startswith('GIT_')}
    git_env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                    'GIT_ATTR_NOSYSTEM': '1', 'GIT_NO_REPLACE_OBJECTS': '1',
                    'GIT_TERMINAL_PROMPT': '0', 'GIT_OPTIONAL_LOCKS': '0'})
    def git(*args):
        return subprocess.run(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false',
                               '-c', 'core.attributesFile=' + os.devnull, '-C', str(root), *args],
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=30, env=git_env).stdout
    def identity():
        return (git('rev-parse', 'HEAD').strip(), git('rev-parse', '--abbrev-ref', 'HEAD').strip(),
                git('show', '-s', '--format=%T', 'HEAD').strip(), git('ls-files', '-v', '-z'),
                git('ls-files', '--stage', '-z'), git('ls-files', '--others', '--exclude-standard', '-z'))
    initial = identity()
    if (initial[0] != revision.encode('ascii') or initial[1] != b'HEAD' or initial[5]
            or re.fullmatch(rb'[0-9a-f]{40}', initial[2]) is None
            or any(item and not item.startswith(b'H ') for item in initial[3].split(b'\0'))):
        raise ValueError('source checkout must be clean, detached and at the published revision')
    extras = ('scripts/pretrained_number_cpu_launcher.py',
              'scripts/distilgpt2_intervention_dev_bootstrap.py',
              'docs/PRETRAINED_NUMBER_PATCH_V1.md', 'docs/PRETRAINED_NUMBER_INJECTION_REQUESTS.md')
    output = git('ls-tree', '-r', '-z', '--full-tree', revision)
    entries = [item for item in output.split(b'\0') if item]
    if not 0 < len(entries) <= 4096:
        raise ValueError('source inventory is outside its bounds')
    records, tree_index, total_bytes = {}, {}, 0
    for item in entries:
        try:
            metadata, raw_name = item.split(b'\t', 1)
            mode, kind, blob = metadata.split(b' ')
            name = raw_name.decode('utf-8')
        except (ValueError, UnicodeError) as error:
            raise ValueError('malformed source inventory') from error
        relative = Path(name)
        if (mode not in (b'100644', b'100755') or kind != b'blob'
                or re.fullmatch(rb'[0-9a-f]{40}', blob) is None or relative.is_absolute()
                or '..' in relative.parts or '\\' in name or name in records):
            raise ValueError('unsupported source file')
        path = root / relative
        tree_index[name] = (mode, blob)
        cursor = root
        for part in relative.parts:
            cursor /= part
            info = cursor.lstat()
            if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0)
                    & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
                raise ValueError('source path contains an alias')
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 ** 2:
            raise ValueError('source file is not bounded and ordinary')
        total_bytes += info.st_size
        if total_bytes > 512 * 1024 ** 2:
            raise ValueError('source snapshot exceeds its aggregate bound')
        git_digest = hashlib.sha1()
        git_digest.update(b'blob ' + str(info.st_size).encode('ascii') + b'\0')
        content_digest = hashlib.sha256()
        observed = 0
        with path.open('rb') as stream:
            while part := stream.read(1024 * 1024):
                observed += len(part)
                if observed > info.st_size:
                    raise ValueError('source grew while reading')
                git_digest.update(part)
                content_digest.update(part)
        if observed != info.st_size:
            raise ValueError('source changed while reading')
        if git_digest.hexdigest().encode('ascii') != blob:
            raise ValueError('source bytes differ from the committed blob')
        records[name] = content_digest.hexdigest()
    index = {}
    for item in initial[4].split(b'\0'):
        if not item:
            continue
        try:
            metadata, raw_name = item.split(b'\t', 1)
            mode, blob, stage = metadata.split(b' ')
            name = raw_name.decode('utf-8')
        except (ValueError, UnicodeError) as error:
            raise ValueError('malformed source index') from error
        if stage != b'0' or name in index:
            raise ValueError('source index is conflicted or duplicated')
        index[name] = (mode, blob)
    if index != tree_index:
        raise ValueError('source index differs from the committed tree')
    if not set(extras).issubset(records) or 'looking_glass/__init__.py' not in records:
        raise ValueError('required bootstrap source is missing')
    package = root / 'looking_glass'
    for base, dirs, files in os.walk(package, followlinks=False):
        dirs[:] = [name for name in dirs if name != '__pycache__']
        for name in dirs:
            info = (Path(base) / name).lstat()
            if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0)
                    & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
                raise ValueError('package directory is an alias')
        for name in files:
            if (Path(base) / name).relative_to(root).as_posix() not in records:
                raise ValueError('untracked package file is present')
    if identity() != initial:
        raise ValueError('source changed while checking')
    return {'git_commit': revision, 'git_tree': initial[2].decode('ascii'), 'files': records}


def create_public_plan(*, source_commit, runtime_bytes, runtime_url, plan_dir):
    """Draw one public request-plan entropy value after runtime publication."""
    _environment_policy()
    verify_published_bytes(runtime_url, runtime_bytes)
    _runtime_record(runtime_bytes, source_commit)
    from looking_glass.pretrained_number_requests import create_family_request_plan

    directory = Path(plan_dir)
    if not directory.is_absolute() or not directory.parent.is_dir() or directory.exists():
        raise ValueError('a fresh absolute public plan directory is required')
    root_marker = _canonical({'schema': 'mlg-number-plan-root-v1', 'source_commit': source_commit,
        'runtime_sha256': hashlib.sha256(runtime_bytes).hexdigest(), 'plan_directory': directory.name})
    try:
        _write_new(directory.parent / 'number-injection-plan-attempt.json', root_marker)
    except FileExistsError as error:
        raise ValueError('this study root already attempted a public plan') from error
    directory.mkdir(mode=0o700)
    _write_new(directory / 'started.json', _canonical({
        'schema': 'mlg-number-plan-start-v1', 'source_commit': source_commit,
        'runtime_sha256': hashlib.sha256(runtime_bytes).hexdigest(),
    }))
    entropy = secrets.token_bytes(32)
    plan = create_family_request_plan(entropy, source_commit)
    _write_new(directory / 'plan.json', plan)
    return hashlib.sha256(plan).hexdigest()


def _plan_attempt_marker(plan_attempt_dir, source_commit, runtime_bytes, plan_bytes):
    directory = Path(plan_attempt_dir)
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError('the original absolute plan-attempt directory is required')
    start = _canonical({'schema': 'mlg-number-plan-start-v1', 'source_commit': source_commit,
                        'runtime_sha256': hashlib.sha256(runtime_bytes).hexdigest()})
    root_marker = _canonical({'schema': 'mlg-number-plan-root-v1', 'source_commit': source_commit,
        'runtime_sha256': hashlib.sha256(runtime_bytes).hexdigest(), 'plan_directory': directory.name})
    if (_read_bounded(directory.parent / 'number-injection-plan-attempt.json', 4096) != root_marker
            or _read_bounded(directory / 'started.json', 4096) != start
            or _read_bounded(directory / 'plan.json', 128 * 1024) != plan_bytes):
        raise ValueError('plan attempt does not match published source and runtime')
    return directory, _canonical({'schema': 'mlg-number-site-draw-v1',
        'source_commit': source_commit, 'runtime_sha256': hashlib.sha256(runtime_bytes).hexdigest(),
        'plan_sha256': hashlib.sha256(plan_bytes).hexdigest()})


def create_site_custody(*, source_commit, runtime_bytes, runtime_url, plan_bytes, plan_url,
                        plan_attempt_dir, custody_dir):
    """Verify public prerequisites before drawing one private site nonce.

    This check does not prove earlier publication time or same-account secrecy.
    A created directory is an attempt marker and must not be reused on failure.
    """
    _environment_policy()
    verify_published_bytes(runtime_url, runtime_bytes)
    verify_published_bytes(plan_url, plan_bytes)
    _runtime_record(runtime_bytes, source_commit)
    from looking_glass.pretrained_number_commitments import seal_site
    from looking_glass.pretrained_number_requests import load_family_request_plan

    plan_digest = hashlib.sha256(plan_bytes).hexdigest()
    plan = load_family_request_plan(plan_bytes, expected_source_commit=source_commit,
                                    expected_sha256=plan_digest)
    plan_directory, draw_marker = _plan_attempt_marker(plan_attempt_dir, source_commit, runtime_bytes, plan_bytes)
    directory = Path(custody_dir)
    if not directory.is_absolute() or not directory.parent.is_dir() or directory.exists():
        raise ValueError('a fresh absolute private custody directory is required')
    if directory.resolve(strict=False).is_relative_to(plan_directory.resolve(strict=True)):
        raise ValueError('private custody must be outside the public plan directory')
    try:
        _write_new(plan_directory / 'site-draw-started.json', draw_marker)
    except FileExistsError as error:
        raise ValueError('the plan site draw was already attempted') from error
    directory.mkdir(mode=0o700)
    _write_new(directory / 'started.json', _canonical({
        'schema': 'mlg-number-site-custody-start-v1', 'source_commit': source_commit,
        'runtime_sha256': hashlib.sha256(runtime_bytes).hexdigest(), 'plan_sha256': plan_digest,
    }))
    nonce = secrets.token_bytes(32)
    _write_new(directory / 'nonce.bin', nonce)
    public_entropy = bytes.fromhex(json.loads(plan.record_bytes)['public_entropy'])
    if hmac.compare_digest(nonce, public_entropy):
        raise ValueError('private nonce equals public request entropy')
    sealed = seal_site(nonce, source_commit)
    _write_new(directory / 'site-record.json', sealed.record_bytes)
    _write_new(directory / 'commitment.txt', (sealed.commitment_sha256 + '\n').encode('ascii'))
    return sealed.commitment_sha256


def run_published_discovery(*, source_commit, runtime_bytes, runtime_url, plan_bytes, plan_url,
                            commitment_bytes, commitment_url, plan_attempt_dir, custody_dir, attempt_dir,
                            cache_dir, site_packages,
                            phase_ledger=None):
    """Verify all public artifacts before reading private custody or loading a model."""
    _environment_policy()
    verify_published_bytes(runtime_url, runtime_bytes)
    verify_published_bytes(plan_url, plan_bytes)
    verify_published_bytes(commitment_url, commitment_bytes)
    _runtime_record(runtime_bytes, source_commit)
    from looking_glass.pretrained_number_commitments import verify_site_reveal
    from looking_glass.pretrained_number_requests import load_family_request_plan

    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = load_family_request_plan(plan_bytes, expected_source_commit=source_commit,
                                    expected_sha256=plan_sha256)
    if type(commitment_bytes) is not bytes or re.fullmatch(rb'[0-9a-f]{64}\n', commitment_bytes) is None:
        raise ValueError('a canonical published site commitment is required')
    commitment = commitment_bytes[:-1].decode('ascii')
    plan_directory, draw_marker = _plan_attempt_marker(plan_attempt_dir, source_commit, runtime_bytes, plan_bytes)
    if _read_bounded(plan_directory / 'site-draw-started.json', 4096) != draw_marker:
        raise ValueError('plan site draw marker does not match published inputs')
    custody = Path(custody_dir)
    if not custody.is_absolute() or not custody.is_dir():
        raise ValueError('an absolute private custody directory is required')
    expected_start = _canonical({'schema': 'mlg-number-site-custody-start-v1',
        'source_commit': source_commit, 'runtime_sha256': hashlib.sha256(runtime_bytes).hexdigest(),
        'plan_sha256': plan_sha256})
    if (_read_bounded(custody / 'started.json', 4096) != expected_start
            or _read_bounded(custody / 'commitment.txt', 65) != commitment_bytes):
        raise ValueError('private custody does not match published inputs')
    nonce = _read_bounded(custody / 'nonce.bin', 32)
    record = _read_bounded(custody / 'site-record.json', 4096)
    if len(nonce) != 32 or hmac.compare_digest(nonce, bytes.fromhex(json.loads(plan.record_bytes)['public_entropy'])):
        raise ValueError('private site nonce is invalid or equals public entropy')
    verify_site_reveal(nonce, record, commitment, expected_source_commit=source_commit)
    return _execute_verified(source_commit=source_commit, runtime_bytes=runtime_bytes, plan_bytes=plan_bytes,
                             plan_sha256=plan_sha256, site_nonce=nonce, site_record_bytes=record,
                             site_commitment=commitment, attempt_dir=attempt_dir,
                             cache_dir=cache_dir, site_packages=site_packages, phase_ledger=phase_ledger)


def _execute_bootstrap_source(path, expected_sha256):
    """Compile the verified raw bytes, bypassing timestamp-valid bytecode."""
    source = _read_bounded(path, 16 * 1024 ** 2)
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('bootstrap changed after its source check')
    module = types.ModuleType('_number_cpu_bootstrap')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def _boot(site_packages, source_commit):
    """Establish the published source and isolated import path before model work."""
    _environment_policy()
    if type(source_commit) is not str or re.fullmatch(r'[0-9a-f]{40}', source_commit) is None:
        raise ValueError('a full source commit is required')
    repo = Path(__file__).resolve(strict=True).parents[1]
    script = repo / 'scripts' / 'pretrained_number_cpu_launcher.py'
    source_url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
                  + source_commit + '/scripts/pretrained_number_cpu_launcher.py')
    verify_published_bytes(source_url, _read_bounded(script, 1024 * 1024))
    verified = _verify_unloaded_source(repo, source_commit)
    bootstrap = repo / 'scripts' / 'distilgpt2_intervention_dev_bootstrap.py'
    module = _execute_bootstrap_source(bootstrap, verified['files']['scripts/distilgpt2_intervention_dev_bootstrap.py'])
    actual_repo, site, prefix, _ = module._bootstrap(site_packages)
    if actual_repo != repo:
        raise ValueError('checkout does not match the published source revision')
    extras = ('scripts/pretrained_number_cpu_launcher.py',
              'scripts/distilgpt2_intervention_dev_bootstrap.py',
              'docs/PRETRAINED_NUMBER_PATCH_V1.md',
              'docs/PRETRAINED_NUMBER_INJECTION_REQUESTS.md')
    module._load_project_package(repo)
    return {'repo': repo, 'site': site, 'prefix': prefix, 'bootstrap': module,
            'extra_files': extras, 'source_record': verified, 'source_commit': source_commit}


def _numeric_environment():
    fixed = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
             'OPENBLAS_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE'}
    prefixes = ('MKL_', 'OMP_', 'OPENBLAS_', 'ATEN_', 'TORCH_', 'PYTORCH_', 'CUDA_', 'CUBLAS_',
                'VECLIB_', 'NUMEXPR_', 'KMP_', 'DNNL_', 'ONEDNN_', 'GOTO_', 'BLIS_')
    if any(os.environ.get(name) != value for name, value in fixed.items()):
        raise ValueError('the CPU numerical environment differs from the fixed policy')
    if any(name != name.upper() or name not in fixed for name in os.environ
           if name.upper().startswith(prefixes)):
        raise ValueError('unrecorded numerical backend controls are unsupported')
    return fixed


def _environment_policy():
    numeric = _numeric_environment()
    controls = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
    cache_names = {'MLG_STUDY_WORKSPACE', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'}
    system_names = {'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'COMSPEC', 'PATHEXT',
                    'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH'}
    allowed = set(numeric) | set(controls) | cache_names | system_names
    if (any(os.environ.get(name) != value for name, value in controls.items())
            or any(not os.environ.get(name) for name in cache_names | {'PATH'})
            or any(name != name.upper() or name not in allowed for name in os.environ)):
        raise ValueError('child environment contains missing or unreviewed controls')
    system = {name: os.environ[name] for name in sorted(system_names) if name in os.environ}
    return hashlib.sha256(_canonical(system)).hexdigest()


def _numeric_state(torch):
    from looking_glass.pretrained_number_injection import verify_cpu_float32_addition
    if (torch.cuda.is_initialized() or torch.is_autocast_enabled('cpu')
            or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.get_default_dtype() is not torch.float32
            or str(torch.get_default_device()) != 'cpu'
            or torch.backends.mkldnn.enabled is not True
            or torch.backends.mkldnn.deterministic is not True
            or torch.get_float32_matmul_precision() != 'highest'):
        raise ValueError('CPU numerical runtime differs from the recorded policy')
    verify_cpu_float32_addition()
    return {'device': 'cpu', 'dtype': 'float32', 'seed': _SEED,
            'intraop_threads': 1, 'interop_threads': 1, 'deterministic_algorithms': True,
            'deterministic_warn_only': False, 'default_dtype': 'float32',
            'default_device': 'cpu', 'mkldnn_enabled': True, 'mkldnn_deterministic': True,
            'cpu_capability': torch.backends.cpu.get_cpu_capability(),
            'torch_config_sha256': hashlib.sha256(torch.__config__.show().encode('utf-8')).hexdigest(),
            'matmul_precision': 'highest', 'cpu_autocast': False, 'flush_denormal': False,
            'float32_addition_probe': 'passed'}


def _cache_policy(cache_dir):
    names = ('MLG_STUDY_WORKSPACE', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE')
    values = [os.environ.get(name) for name in names]
    if any(type(value) is not str or not value or not Path(value).is_absolute() for value in values):
        raise ValueError('explicit absolute workspace and model-cache paths are required')
    try:
        workspace, home, hub, transformers = (Path(value).resolve(strict=True) for value in values)
        cache = Path(cache_dir).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError('model cache paths are unavailable') from error
    if (not workspace.is_dir() or not home.is_dir() or not cache.is_dir()
            or home == workspace or not home.is_relative_to(workspace)
            or cache == home or not cache.is_relative_to(home) or hub != cache or transformers != cache):
        raise ValueError('offline model cache must be contained in the declared workspace HF_HOME')
    conflicts = ('HF_ASSETS_CACHE', 'HF_TOKEN_PATH', 'HF_XET_CACHE', 'HUGGINGFACE_HUB_CACHE',
                 'PYTORCH_PRETRAINED_BERT_CACHE', 'PYTORCH_TRANSFORMERS_CACHE')
    if any(name in os.environ for name in conflicts):
        raise ValueError('conflicting model cache overrides are unsupported')
    return home, cache


def _verify_effective_cache(home, cache):
    constants = importlib.import_module('huggingface_hub.constants')
    hub = importlib.import_module('transformers.utils.hub')
    if (Path(constants.HF_HOME).resolve() != home or Path(constants.HF_HUB_CACHE).resolve() != cache
            or Path(hub.TRANSFORMERS_CACHE).resolve() != cache
            or constants.HF_HUB_OFFLINE is not True or constants.HF_HUB_DISABLE_TELEMETRY is not True
            or hub.is_offline_mode() is not True):
        raise ValueError('effective model cache or offline mode differs from the requested runtime')


def _initialize_runtime(*, source_commit, cache_dir, site_packages):
    context = _CONTEXT
    if type(context) is not dict or context.get('source_commit') != source_commit:
        raise ValueError('the published source bootstrap is required')
    cache = Path(cache_dir)
    if not cache.is_absolute() or not cache.is_dir() or Path(site_packages).resolve(strict=True) != context['site']:
        raise ValueError('absolute offline cache and the verified package directory are required')
    system_environment_sha256 = _environment_policy()
    expected_env = {**_numeric_environment(), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                    'HF_HUB_DISABLE_TELEMETRY': '1'}
    if sys.byteorder != 'little' or any(os.environ.get(k) != v for k, v in expected_env.items()):
        raise ValueError('the CPU-only offline environment is not configured')
    home, cache = _cache_policy(cache)
    torch = importlib.import_module('torch')
    events, safe_queries = [], {'cuda_is_available': 0, 'cuda_device_count': 0}
    context['bootstrap']._guard_cuda_and_network(torch, events, safe_queries)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision('highest')
    torch.set_default_dtype(torch.float32)
    torch.backends.mkldnn.enabled = True
    torch.backends.mkldnn.deterministic = True
    torch.random.default_generator.manual_seed(_SEED)
    if torch.set_flush_denormal(False) is not True:
        raise ValueError('CPU denormal preservation is unavailable')
    modules = [torch] + [importlib.import_module(name) for name in
                        ('numpy', 'transformers', 'tokenizers', 'safetensors', 'huggingface_hub')]
    _verify_effective_cache(home, cache)
    versions = {}
    for module in modules:
        context['bootstrap']._origin(module, context['site'])
        version = getattr(module, '__version__', None)
        if not isinstance(version, str) or not version:
            raise ValueError('runtime dependency version is unavailable')
        versions[module.__name__] = str(version)
    from looking_glass.pretrained_number_commitments import CHECKPOINT_REVISION, PROTOCOL_SHA256
    from looking_glass.trusted_service_profile import prepare_trusted_source
    repository = _verify_unloaded_source(context['repo'], source_commit)
    if repository != context['source_record']:
        raise ValueError('verified source revision changed')
    preflight = prepare_trusted_source(context['repo'], cache.resolve(strict=True))
    if hashlib.sha256((context['repo'] / 'docs/PRETRAINED_NUMBER_PATCH_V1.md').read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('number-study protocol bytes differ from the commitment')
    state = _numeric_state(torch)
    if events or any(context['prefix'].iterdir()):
        raise ValueError('CPU device, network or bytecode guard was triggered')
    manifest = {'schema': 'mlg-number-injection-cpu-runtime-v1', 'source_commit': source_commit,
                'protocol_sha256': PROTOCOL_SHA256, 'checkpoint_revision': CHECKPOINT_REVISION,
                'source_tree': repository['git_tree'],
                'repository_source_sha256': hashlib.sha256(_canonical(repository)).hexdigest(),
                'trusted_source_sha256': hashlib.sha256(preflight.source_base).hexdigest(),
                'extra_sources': {name: repository['files'][name] for name in context['extra_files']},
                'assets': json.loads(preflight.asset_manifest),
                'python_version': platform.python_version(), 'system': platform.system(),
                'machine': platform.machine(), 'package_versions': versions,
                'environment': expected_env, 'numerics': state,
                'system_environment_sha256': system_environment_sha256,
                'cache_policy': {'hf_home': 'workspace-contained', 'hub_cache': 'inside-hf-home',
                                 'transformers_cache_matches_hub': True, 'offline_readback': True},
                'model_policy': {'device': 'cpu', 'dtype': 'float32', 'eval': True,
                                 'attention': 'eager', 'use_cache': False}}
    return {'runtime_bytes': _canonical(manifest), 'preflight': preflight, 'torch': torch,
            'events': events, 'repository': repository, 'home': home, 'cache': cache.resolve(strict=True),
            'system_environment_sha256': system_environment_sha256}


def _verify_public_tokenizer(tokenizer):
    from looking_glass.agreement_preflight_v1 import tokenize_inputs
    from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY
    rows, _ = tokenize_inputs(tokenizer)
    if (type(rows) is not list or len(rows) != len(PUBLIC_DISCOVERY)
            or any(type(row) is not dict or type(row.get('input_ids')) is not list
                   or any(type(value) is not int for value in row['input_ids'])
                   or row.get('text') != expected[0] or tuple(row['input_ids']) != expected[1]
                   for row, expected in zip(rows, PUBLIC_DISCOVERY))):
        raise ValueError('public discovery tokenizer alignment differs from the fixed rows')


def _load_trusted_model(state):
    from looking_glass import pretrained
    from looking_glass.trusted_service_profile import verify_trusted_cpu_model
    bundle = pretrained.load_distilgpt2(cache_dir=str(state['cache']), allow_download=False)
    if verify_trusted_cpu_model(state['preflight'], bundle.model) is not True:
        raise ValueError('offline model verification did not succeed')
    _verify_public_tokenizer(bundle.tokenizer)
    if _cache_policy(state['cache']) != (state['home'], state['cache']):
        raise ValueError('model cache configuration changed while loading')
    _verify_effective_cache(state['home'], state['cache'])
    if _environment_policy() != state['system_environment_sha256']:
        raise ValueError('child environment changed while loading')
    if state['events']:
        raise ValueError('a blocked CPU device or network entry was attempted')
    _numeric_state(state['torch'])
    return bundle


def _execute_verified(*, source_commit, runtime_bytes, plan_bytes, plan_sha256, site_nonce,
                      site_record_bytes, site_commitment, attempt_dir, cache_dir, site_packages, phase_ledger=None):
    state = _initialize_runtime(source_commit=source_commit, cache_dir=cache_dir, site_packages=site_packages)
    if state['runtime_bytes'] != runtime_bytes:
        raise ValueError('actual runtime differs from the published runtime manifest')
    attempt = Path(attempt_dir)
    if not attempt.is_absolute() or not attempt.parent.is_dir() or attempt.exists():
        raise ValueError('a fresh absolute discovery attempt directory is required')
    marker = attempt.with_name(attempt.name + '.launcher-started.json')
    terminal = attempt.with_name(attempt.name + '.launcher-terminal.json')
    if marker.exists() or terminal.exists():
        raise ValueError('launch attempt has already been used')
    start_bytes = _canonical({'schema': 'mlg-number-injection-launch-start-v1',
        'source_commit': source_commit, 'runtime_sha256': hashlib.sha256(runtime_bytes).hexdigest(),
        'plan_sha256': plan_sha256, 'site_commitment': site_commitment})
    from looking_glass.agreement_phase_ledger import claim_phase
    grant = claim_phase(phase_ledger, 'injection_discovery', output_path=attempt)
    if (type(grant.logical_limit_bytes) is not int or grant.logical_limit_bytes != _LOGICAL_LIMIT
            or type(grant.disk_limit_bytes) is not int or grant.disk_limit_bytes != _DISK_LIMIT):
        raise ValueError('phase grant does not match the approved limits')
    # Reserve the exact start marker and a bounded terminal-marker allowance
    # inside this phase's physical share, including incomplete attempts.
    supervisor_disk_limit = grant.disk_limit_bytes - len(start_bytes) - 4096
    try:
        _write_new(marker, start_bytes)
        bundle = _load_trusted_model(state)
        from looking_glass.pretrained_number_supervisor import run_discovery
        result = run_discovery(model=bundle.model, attempt_dir=attempt, plan_bytes=plan_bytes,
            source_commit=source_commit, plan_sha256=plan_sha256,
            runtime_sha256=hashlib.sha256(runtime_bytes).hexdigest(), site_nonce=site_nonce,
            site_record_bytes=site_record_bytes, site_commitment=site_commitment,
            logical_limit_bytes=grant.logical_limit_bytes, disk_limit_bytes=supervisor_disk_limit)
        from looking_glass.trusted_service_profile import prepare_trusted_source, verify_trusted_cpu_model
        if verify_trusted_cpu_model(state['preflight'], bundle.model) is not True:
            raise ValueError('post-discovery model verification failed')
        if _cache_policy(state['cache']) != (state['home'], state['cache']):
            raise ValueError('model cache configuration changed during discovery')
        _verify_effective_cache(state['home'], state['cache'])
        if _environment_policy() != state['system_environment_sha256']:
            raise ValueError('child environment changed during discovery')
        if any(os.environ.get(k) != v for k, v in json.loads(runtime_bytes)['environment'].items()):
            raise ValueError('runtime environment changed during discovery')
        if (result.status != 'complete-local' or result.record_count != 128
                or state['events'] or _numeric_state(state['torch']) != json.loads(runtime_bytes)['numerics']
                or _verify_unloaded_source(_CONTEXT['repo'], source_commit) != state['repository']
                or prepare_trusted_source(_CONTEXT['repo'], state['cache']) != state['preflight']
                or any(_CONTEXT['prefix'].iterdir())):
            raise ValueError('post-discovery source, runtime or controls are incomplete')
        _write_new(terminal, _canonical({'schema': 'mlg-number-injection-launch-terminal-v1',
            'status': 'complete-local', 'prediction_sha256': result.prediction_sha256,
            'archive_sha256': result.archive_sha256, 'record_count': result.record_count}))
        return result
    except BaseException as error:
        if marker.exists() and not terminal.exists():
            try:
                _write_new(terminal, _canonical({'schema': 'mlg-number-injection-launch-terminal-v1',
                                                 'status': 'incomplete'}))
            except BaseException as cleanup_error:
                if (isinstance(cleanup_error, (KeyboardInterrupt, SystemExit))
                        and not isinstance(error, (KeyboardInterrupt, SystemExit))):
                    raise cleanup_error from error
        raise


def _outside_checkout(path, repo):
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError('an absolute artifact path is required')
    resolved = candidate.resolve(strict=False)
    root = Path(repo).resolve(strict=False)
    if resolved == root or root in resolved.parents:
        raise ValueError('study artifacts must be outside the source checkout')
    return candidate


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, 'CPU discovery command incomplete; invalid command line.\n')


def main(argv=None):
    parser = _PrivateArgumentParser(description='Prepare and run CPU-only injection discovery')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--site-packages', required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--cache-dir', required=True)
    prepare.add_argument('--runtime-output', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--runtime', required=True)
    plan.add_argument('--runtime-url', required=True)
    plan.add_argument('--plan-dir', required=True)
    seal = commands.add_parser('seal')
    seal.add_argument('--runtime', required=True)
    seal.add_argument('--runtime-url', required=True)
    seal.add_argument('--plan', required=True)
    seal.add_argument('--plan-url', required=True)
    seal.add_argument('--custody-dir', required=True)
    run = commands.add_parser('run')
    for name in ('cache-dir', 'runtime', 'runtime-url', 'plan', 'plan-url',
                 'commitment', 'commitment-url', 'custody-dir', 'attempt-dir', 'phase-ledger'):
        run.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    global _CONTEXT
    try:
        _environment_policy()
        _CONTEXT = _boot(args.site_packages, args.source_commit)
        repo = _CONTEXT['repo']
        if args.command == 'prepare':
            dest = _outside_checkout(args.runtime_output, repo)
            state = _initialize_runtime(source_commit=args.source_commit, cache_dir=args.cache_dir,
                                        site_packages=args.site_packages)
            _write_new(dest, state['runtime_bytes'])
            report = {'status': 'prepared',
                      'runtime_sha256': hashlib.sha256(state['runtime_bytes']).hexdigest()}
        elif args.command == 'plan':
            dest = _outside_checkout(args.plan_dir, repo)
            digest = create_public_plan(source_commit=args.source_commit,
                runtime_bytes=_read_bounded(args.runtime, 1024 * 1024),
                runtime_url=args.runtime_url, plan_dir=dest)
            report = {'status': 'plan-created', 'plan_sha256': digest}
        elif args.command == 'seal':
            dest = _outside_checkout(args.custody_dir, repo)
            digest = create_site_custody(source_commit=args.source_commit,
                runtime_bytes=_read_bounded(args.runtime, 1024 * 1024), runtime_url=args.runtime_url,
                plan_bytes=_read_bounded(args.plan, 128 * 1024), plan_url=args.plan_url,
                plan_attempt_dir=Path(args.plan).parent, custody_dir=dest)
            report = {'status': 'commitment-created', 'site_commitment': digest}
        else:
            custody = _outside_checkout(args.custody_dir, repo)
            attempt = _outside_checkout(args.attempt_dir, repo)
            ledger = _outside_checkout(args.phase_ledger, repo)
            result = run_published_discovery(source_commit=args.source_commit,
                runtime_bytes=_read_bounded(args.runtime, 1024 * 1024), runtime_url=args.runtime_url,
                plan_bytes=_read_bounded(args.plan, 128 * 1024), plan_url=args.plan_url,
                commitment_bytes=_read_bounded(args.commitment, 65), commitment_url=args.commitment_url,
                plan_attempt_dir=Path(args.plan).parent, custody_dir=custody,
                attempt_dir=attempt, cache_dir=args.cache_dir,
                site_packages=args.site_packages, phase_ledger=ledger)
            report = {'status': result.status, 'prediction_sha256': result.prediction_sha256,
                      'archive_sha256': result.archive_sha256}
        print(_canonical(report).decode('ascii'))
        return 0
    except KeyboardInterrupt:
        print('CPU discovery command cancelled; retain its private attempt records.', file=sys.stderr)
        return 130
    except BaseException:
        print('CPU discovery command incomplete; retain its private attempt records.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
