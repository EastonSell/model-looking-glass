"""Source-bound natural held-out child bootstrap; it does not draw a split.

The custodian supplies an exclusive, already committed attempt and a private
configuration containing paths and immutable public URLs only. The child must
run under the reviewed Windows supervisor and protected frozen source leases.
"""

import ast
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
import urllib.parse
import urllib.request


SOURCE_COMMIT = 'def07cbb168408a12c76ba30f196af45149e111d'
BINDING_SHA256 = '23575704f3e55d741123d3c1024accfb15aa0a048a84ce415bc765573d913ba0'
MODEL_STATE_SHA256 = '94b22301fa1a03550f8b75a255d6c7bb2dda25ee6d9079b70321a81ac884c1f1'


class BootstrapError(RuntimeError):
    """The frozen child cannot establish a required precondition."""


def _read(path, maximum):
    path = Path(path)
    if not path.is_absolute():
        raise BootstrapError('input path must be absolute')
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or info.st_size > maximum):
        raise BootstrapError('input is nonregular, linked or oversized')
    with path.open('rb') as stream:
        data = stream.read(maximum + 1)
    if len(data) != info.st_size or len(data) > maximum or path.stat().st_size != info.st_size:
        raise BootstrapError('input changed while reading')
    return data


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapError('duplicate configuration key')
        result[key] = value
    return result


def _check_environment():
    # These must be absent before Torch, tokenizers, or the model are imported.
    _check_overrides()
    if any(name == 'torch' or name.startswith('torch.') or name == 'transformers'
           or name.startswith('transformers.') for name in sys.modules):
        raise BootstrapError('model dependencies were imported before the CPU gate')


def _check_overrides():
    if any(name in os.environ for name in ('MKL_CBWR', 'ATEN_CPU_CAPABILITY')):
        raise BootstrapError('CPU numerical override variables must be unset')


def _record_overrides(phase):
    _check_overrides()
    path = Path(phase) / 'owner' / 'runtime-overrides.json'
    raw = (json.dumps({'schema': 'mlg-natural-heldout-runtime-overrides-v1',
                       'MKL_CBWR': None, 'ATEN_CPU_CAPABILITY': None},
                      sort_keys=True, ensure_ascii=True, separators=(',', ':')) + '\n').encode('ascii')
    with path.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short runtime observation write')
        stream.flush()
        os.fsync(stream.fileno())


def _write_observation(path, value):
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')) + '\n').encode('utf-8')
    if len(raw) > 512 * 1024:
        raise BootstrapError('staging observation exceeds its bound')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short staging observation write')
        stream.flush()
        os.fsync(stream.fileno())


def _staging_snapshot(cache):
    entries = []
    with os.scandir(cache) as listing:
        for entry in listing:
            if not entry.name.startswith(('looking-glass-stage-', 'agreement-tokenizer-')):
                continue
            info = entry.stat(follow_symlinks=False)
            entries.append({'name': entry.name, 'directory': stat.S_ISDIR(info.st_mode),
                            'reparse': bool(getattr(info, 'st_file_attributes', 0) & 0x400),
                            'mtime_ns': info.st_mtime_ns})
            if len(entries) > 4096:
                raise BootstrapError('too many existing cache staging entries')
    return sorted(entries, key=lambda item: item['name'])


@contextmanager
def _observe_staging(cache, phase):
    cache, phase = Path(cache), Path(phase)
    required = 3 * 1024 * 1024 * 1024
    cache_free, destination_free = shutil.disk_usage(cache).free, shutil.disk_usage(phase).free
    if cache_free < required or destination_free < required:
        raise BootstrapError('cache or destination has less than 3 GiB free')
    before = _staging_snapshot(cache)
    _write_observation(phase / 'owner/checkpoint-staging-before.json',
                       {'cache_free_bytes': cache_free, 'destination_free_bytes': destination_free,
                        'required_free_bytes': required, 'entries': before})
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
    try:
        after = _staging_snapshot(cache)
        new_names = sorted({item['name'] for item in after} - {item['name'] for item in before})
        _write_observation(phase / 'owner/checkpoint-staging-after.json',
                           {'entries': after, 'new_unresolved_names': new_names})
    except BaseException as error:
        if primary is not None:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error from primary
            raise primary from error
        raise
    if primary is not None:
        raise primary
    if new_names:
        raise BootstrapError('new cache staging entries remain; preserve and reconcile them')


class _BoundSourceLoader(importlib.abc.Loader):
    """Hash and compile exact source bytes; never read bytecode."""
    def __init__(self, path, digest):
        self.path, self.digest = Path(path), digest

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raw = _read(self.path, 8 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != self.digest:
            raise BootstrapError('project source differs before execution')
        exec(compile(raw, str(self.path), 'exec', dont_inherit=True), module.__dict__)


class _FrozenSourceFinder(importlib.abc.MetaPathFinder):
    """Load project code only from the specified source checkout, not sys.path."""
    def __init__(self, root, inventory):
        self.root = Path(root)
        self.inventory = inventory

    def _spec(self, fullname, source, *, package=False):
        name = source.relative_to(self.root).as_posix()
        digest = self.inventory.get(name)
        if type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise BootstrapError('project source is absent from the pinned binding')
        loader = _BoundSourceLoader(source, digest)
        return importlib.util.spec_from_file_location(fullname, source, loader=loader,
                                                     submodule_search_locations=[] if package else None)

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scripts':
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        if fullname.startswith('scripts.'):
            if fullname != 'scripts.v2_source_guards':
                raise ModuleNotFoundError('unapproved project script import')
            return self._spec(fullname, self.root / 'scripts/v2_source_guards.py')
        if fullname != 'looking_glass' and not fullname.startswith('looking_glass.'):
            return None
        parts = fullname.split('.')
        if any(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) is None for part in parts):
            raise ModuleNotFoundError('invalid project module name')
        base = self.root.joinpath(*parts)
        package, source = base / '__init__.py', base.with_suffix('.py')
        if package.is_file():
            return self._spec(fullname, package, package=True)
        if source.is_file():
            return self._spec(fullname, source)
        raise ModuleNotFoundError('project module has no source file')


def _install_frozen_finder(root, inventory):
    if any(name == 'looking_glass' or name.startswith('looking_glass.')
           or name == 'scripts' or name.startswith('scripts.') for name in sys.modules):
        raise BootstrapError('project modules were imported before the source bootstrap')
    root = Path(root)
    if not root.is_absolute() or not root.is_dir():
        raise BootstrapError('frozen source root must be an absolute directory')
    if type(inventory) is not dict or not inventory:
        raise BootstrapError('pinned project source inventory is absent')
    finder = _FrozenSourceFinder(root, inventory)
    sys.meta_path.insert(0, finder)
    return finder


def _frozen_launcher(root, inventory):
    relative = 'tools/natural_agreement_cpu.py'
    path = Path(root) / relative
    raw = _read(path, 8 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != inventory.get(relative):
        raise BootstrapError('frozen launcher differs before execution')
    namespace = {'__name__': 'mlg_frozen_bootstrap', '__file__': str(path), '__package__': None}
    exec(compile(raw, str(path), 'exec', dont_inherit=True), namespace)
    return namespace


def _await_frozen_start(root, inventory):
    """Run only the pinned stdlib gate before importing any project module."""
    relative = 'looking_glass/natural_agreement_controller.py'
    path = Path(root) / relative
    raw = _read(path, 8 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != inventory.get(relative):
        raise BootstrapError('frozen start gate differs before execution')
    tree = ast.parse(raw, filename=str(path))
    candidates = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'await_start']
    if len(candidates) != 1 or candidates[0].decorator_list:
        raise BootstrapError('frozen start gate is not uniquely defined')
    gate_module = ast.fix_missing_locations(ast.Module(body=candidates, type_ignores=[]))
    namespace = {'os': os, 'sys': sys, 'time': time}
    exec(compile(gate_module, str(path), 'exec', dont_inherit=True), namespace)
    namespace['await_start']()


_FILE_KEYS = frozenset(('binding', 'prediction', 'aggregate', 'inventory', 'v7_inventory',
                        'inventory_source', 'operational_binding', 'deviation_bytes'))
_URL_KEYS = frozenset(('binding_url', 'prediction_url', 'aggregate_url', 'inventory_source_url',
                       'operational_binding_url', 'wrapper_url', 'deviation_url', 'commitment_url'))


def _load_config(path):
    raw = _read(path, 32768)
    try:
        config = json.loads(raw, object_pairs_hook=_object)
        if (type(config) is not dict or set(config) != {'schema', 'frozen_root', 'cache_dir', 'phase_dir',
                                                         'preparation_supervisor', 'public_files', 'public_urls'}
                or config['schema'] != 'mlg-natural-heldout-child-config-v1'
                or type(config['public_files']) is not dict or set(config['public_files']) != _FILE_KEYS
                or type(config['public_urls']) is not dict or set(config['public_urls']) != _URL_KEYS):
            raise BootstrapError('configuration fields differ from the approved path-only schema')
        for value in [config['frozen_root'], config['cache_dir'], config['phase_dir'], config['preparation_supervisor'],
                      *config['public_files'].values()]:
            if type(value) is not str or not value or len(value) > 4096 or not Path(value).is_absolute():
                raise BootstrapError('configuration paths must be bounded absolute paths')
        for value in config['public_urls'].values():
            if type(value) is not str or len(value) > 4096:
                raise BootstrapError('publication URL is invalid')
            url = urllib.parse.urlsplit(value)
            if (url.scheme != 'https' or url.netloc != 'raw.githubusercontent.com'
                    or url.query or url.fragment or '..' in url.path.split('/')
                    or re.fullmatch(r'/EastonSell/model-looking-glass/[0-9a-f]{40}/[A-Za-z0-9_./-]+',
                                    url.path) is None):
                raise BootstrapError('publication URL must identify an immutable official source')
    except (ValueError, TypeError, RecursionError) as error:
        raise BootstrapError('invalid configuration JSON') from error
    return raw, config


def _public_inputs(config, custody):
    files = {name: _read(path, 2 * 1024 * 1024) for name, path in config['public_files'].items()}
    urls = config['public_urls']
    public = custody.PublicInputs(**files, inventory_sha256=hashlib.sha256(files['inventory']).hexdigest(),
                                  **{name: value for name, value in urls.items() if name != 'commitment_url'})
    return files, public


def _load_custody(path, raw):
    if _read(path, 2 * 1024 * 1024) != raw:
        raise BootstrapError('custody source changed under its lease')
    name = 'mlg_natural_heldout_custody'
    if name in sys.modules:
        raise BootstrapError('custody module was already loaded')
    loader = _BoundSourceLoader(path, hashlib.sha256(raw).hexdigest())
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _check_operational_sources(raw, child_bytes, custody_bytes):
    try:
        operation = json.loads(raw, object_pairs_hook=_object)
        if (operation['sources']['child']['sha256'] != hashlib.sha256(child_bytes).hexdigest()
                or operation['sources']['custody']['sha256'] != hashlib.sha256(custody_bytes).hexdigest()):
            raise BootstrapError('executed operational source differs from the public binding')
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise BootstrapError('operational source binding is invalid') from error


def _check_published(url, suffix, expected):
    parsed = urllib.parse.urlsplit(url)
    if not parsed.path.endswith('/' + suffix):
        raise BootstrapError('publication path differs from its required artifact')
    with urllib.request.urlopen(url, timeout=20) as response:
        if response.geturl() != url:
            raise BootstrapError('publication redirected')
        remote = response.read(2 * 1024 * 1024 + 1)
    if remote != expected:
        raise BootstrapError('published bytes differ from the retained artifact')


def _check_public_operation(config, raw, *, require_commitment=True):
    _check_published(config['public_urls']['operational_binding_url'],
                     'results/distilgpt2-natural-agreement-v1-20261028/heldout-operational-binding-v1.json', raw)
    if not require_commitment:
        return
    commitment_raw = _read(Path(config['phase_dir']) / 'commitment.json', 8192)
    _check_published(config['public_urls']['commitment_url'],
                     'results/distilgpt2-natural-agreement-v1-20261028/heldout-commitment-v1.json', commitment_raw)
    try:
        commitment = json.loads(commitment_raw, object_pairs_hook=_object)
        if commitment['operational_binding_sha256'] != hashlib.sha256(raw).hexdigest():
            raise BootstrapError('public commitment does not bind this operational source')
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise BootstrapError('public operational commitment is invalid') from error


def _run_pipeline(frozen, config, binding, public, custody, wrapper_bytes, dependencies, state):
    """Use only validated committed cases and the pre-frozen scoring source."""
    cache, phase = Path(config['cache_dir']), Path(config['phase_dir'])
    commitment_url = config['public_urls']['commitment_url']
    preflight, commitments, heldout = dependencies['preflight'], dependencies['commitments'], dependencies['heldout']

    def verify(tokenizer):
        return custody.verify_prepared(phase, public, commitment_url, commitments=commitments, heldout=heldout,
                                       tokenizer=tokenizer, wrapper_bytes=wrapper_bytes,
                                       observe_inventory=custody.observe_current_inventory,
                                       preparation_supervisor=config['preparation_supervisor'])

    state['phase'] = 'commitment_verification'
    tokenizer = preflight._tokenizer(cache)
    freeze, snapshot, nonce, record, commitment = verify(tokenizer)
    custody.start_owner_once(phase, commitment)
    state['phase'] = 'model_loading'
    bundle = dependencies['load_model'](cache_dir=str(cache), allow_download=False)
    if (bundle.metadata['model_revision'] != binding['checkpoint_revision']
            or bundle.metadata['asset_sha256'] != binding['assets_sha256']
            or bundle.metadata['device'] != 'cpu' or bundle.metadata['dtype'] != 'float32'
            or bundle.metadata['attention_implementation'] != 'eager' or bundle.metadata['use_cache'] is not False):
        raise BootstrapError('loaded model differs from the frozen CPU binding')
    if dependencies['state_hash'](bundle.model) != MODEL_STATE_SHA256:
        raise BootstrapError('loaded model state differs from the discovery checkpoint')
    rows, groups = preflight.tokenize_inputs(bundle.tokenizer)
    if dependencies['control'].bind_public_cases(dependencies['manifest'], rows, groups) != binding['cases']:
        raise BootstrapError('loaded tokenizer differs from the public binding')
    allocation = commitments.sample_families(freeze, nonce, inventory_snapshot=snapshot,
                                              current_inventory_sha256=snapshot.sha256)
    cases = custody.tokenize_assignment(allocation, bundle.tokenizer)
    resealed = commitments.seal_heldout(freeze, nonce, cases, allocation=allocation,
                                         inventory_snapshot=snapshot, current_inventory_sha256=snapshot.sha256)
    if (resealed.record_bytes != record
            or resealed.commitment_sha256 != commitment['heldout_commitment']):
        raise BootstrapError('loaded tokenizer differs from the committed held-out cases')
    journal = dependencies['journal'](phase / 'owner' / 'attempt',
                                       frozen['_start_record'](public.binding, binding),
                                       max_total_bytes=160 * 1024 * 1024,
                                       max_event_log_bytes=48 * 1024 * 1024)
    runner = dependencies['runner'](bundle.model, journal, device='cpu', expected_quartets=4)
    state['phase'] = 'heldout_execution'
    heldout.run_committed_heldout(runner, freeze=freeze, prediction_bytes=public.prediction,
                                 split_nonce=nonce, record_bytes=record,
                                 heldout_commitment=commitment['heldout_commitment'],
                                 inventory_snapshot=snapshot, current_inventory_sha256=snapshot.sha256)
    if dependencies['state_hash'](bundle.model) != MODEL_STATE_SHA256:
        raise BootstrapError('model state changed during held-out execution')


def _execute(config_path, state, *, validation_only=False):
    _check_environment()
    config_raw, config = _load_config(config_path)
    if validation_only and any(os.path.lexists(Path(config['phase_dir']) / name) for name in
                               ('claim.json', 'nonce.bin', 'heldout-record.json', 'commitment.json',
                                'owner-started.json')):
        raise BootstrapError('validation requires a separate unclaimed rehearsal directory')
    root = Path(config['frozen_root']).resolve(strict=True)
    binding_raw = _read(config['public_files']['binding'], 2 * 1024 * 1024)
    if hashlib.sha256(binding_raw).hexdigest() != BINDING_SHA256:
        raise BootstrapError('discovery runtime binding differs')
    binding = json.loads(binding_raw, object_pairs_hook=_object)
    if binding['source_commit'] != SOURCE_COMMIT:
        raise BootstrapError('discovery scientific source revision differs')
    state['phase'] = 'supervisor_gate'
    _await_frozen_start(root, binding['source_sha256'])
    _install_frozen_finder(root, binding['source_sha256'])
    from looking_glass import natural_agreement_controller as control
    # The parent has already leased both the operational source and this
    # frozen source inventory. The child acquires independent leases below.
    frozen = _frozen_launcher(root, binding['source_sha256'])
    frozen['_startup']()
    frozen['_install_owner_audit_hook']()
    initial = control.verify_source_inventory(root, binding['source_commit'], binding['source_sha256'])
    from scripts.v2_source_guards import SourceFileLease

    operational_root = Path(__file__).resolve(strict=True).parents[1]
    child_source = operational_root / 'tools/natural_agreement_heldout_child.py'
    custody_source = operational_root / 'tools/natural_heldout_custody.py'
    state['phase'] = 'source_verification'
    with SourceFileLease(root, [root / name for name in initial[1]]), \
            SourceFileLease(operational_root, [child_source, custody_source]):
        if control.verify_source_inventory(root, *initial) != initial or _read(config_path, 32768) != config_raw:
            raise BootstrapError('source or configuration changed before protection')
        child_bytes = _read(child_source, 2 * 1024 * 1024)
        wrapper_bytes = _read(custody_source, 2 * 1024 * 1024)
        operation_bytes = _read(config['public_files']['operational_binding'], 2 * 1024 * 1024)
        _check_public_operation(config, operation_bytes, require_commitment=not validation_only)
        _check_operational_sources(operation_bytes, child_bytes, wrapper_bytes)
        custody = _load_custody(custody_source, wrapper_bytes)
        files, public = _public_inputs(config, custody)
        if files['binding'] != binding_raw or files['operational_binding'] != operation_bytes:
            raise BootstrapError('public inputs changed during source verification')
        _record_overrides(config['phase_dir'])
        with _observe_staging(config['cache_dir'], config['phase_dir']):
            frozen['_configure_runtime']()
            state['phase'] = 'runtime_verification'
            frozen['_published'](binding_raw, config['public_urls']['binding_url'])
            if frozen['_bytes'](frozen['build_binding'](config['cache_dir'], expected_source=initial)) != binding_raw:
                raise BootstrapError('runtime, tokenizer, assets or source differ from the published binding')
            from looking_glass import agreement_preflight_v1 as preflight
            from looking_glass import pretrained_number_commitments as commitments
            from looking_glass import agreement_heldout as heldout
            if validation_only:
                _, _, snapshot = custody.check_public_inputs(public, commitments=commitments, heldout=heldout,
                                                             wrapper_bytes=wrapper_bytes)
                observation = custody.observe_current_inventory()
                if observation.get('inventory_sha256') != snapshot.sha256:
                    raise BootstrapError('latest exposure inventory differs from the public snapshot')
                _check_overrides()
                if (frozen['_bytes'](frozen['build_binding'](config['cache_dir'], expected_source=initial)) != binding_raw
                        or control.verify_source_inventory(root, *initial) != initial
                        or _read(child_source, 2 * 1024 * 1024) != child_bytes
                        or _read(custody_source, 2 * 1024 * 1024) != wrapper_bytes
                        or _read(config_path, 32768) != config_raw):
                    raise BootstrapError('post-validation runtime, source or configuration differs')
                state['phase'] = 'validated_no_model_forward'
                return
            from looking_glass.agreement_journal import AgreementJournal
            from looking_glass.agreement_patch_runner import NaturalAgreementRunner, _state_hash
            from looking_glass.pretrained import load_distilgpt2
            dependencies = {'preflight': preflight, 'commitments': commitments, 'heldout': heldout,
                            'journal': AgreementJournal, 'runner': NaturalAgreementRunner,
                            'load_model': load_distilgpt2, 'state_hash': _state_hash, 'control': control,
                            'manifest': _read(root / frozen['MANIFEST_PATH'], 2 * 1024 * 1024)}
            _run_pipeline(frozen, config, binding, public, custody, wrapper_bytes, dependencies, state)
            state['phase'] = 'postrun_verification'
            _check_overrides()
            if (frozen['_bytes'](frozen['build_binding'](config['cache_dir'], expected_source=initial)) != binding_raw
                    or control.verify_source_inventory(root, *initial) != initial
                    or _read(child_source, 2 * 1024 * 1024) != child_bytes
                    or _read(custody_source, 2 * 1024 * 1024) != wrapper_bytes
                    or _read(config_path, 32768) != config_raw):
                raise BootstrapError('post-run runtime, source or configuration differs')
    state['phase'] = 'complete_candidate'


def main():
    state = {'phase': 'configuration'}
    try:
        if len(sys.argv) == 2:
            config_path, validation_only = sys.argv[1], False
        elif len(sys.argv) == 3 and sys.argv[1] == '--validate-only':
            config_path, validation_only = sys.argv[2], True
        else:
            raise BootstrapError('one private path-only configuration is required')
        _execute(config_path, state, validation_only=validation_only)
    except KeyboardInterrupt:
        print(json.dumps({'status': 'cancelled', 'phase': state['phase'],
                          'error_type': 'KeyboardInterrupt'}), file=sys.stderr, flush=True)
        raise SystemExit(130) from None
    except BaseException as error:
        # Owner logs are private, but do not emit the sealed nonce or cases.
        print(json.dumps({'status': 'incomplete', 'phase': state['phase'],
                          'error_type': type(error).__name__}), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'status': state['phase']}), flush=True)


if __name__ == '__main__':
    main()
