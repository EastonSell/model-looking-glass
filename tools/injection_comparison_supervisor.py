"""Independently observe the source-bound split-comparison operator.

The source-first execution owner must verify this module and the entire
comparison manifest before use. A clean comparison is not a forward gate.
"""
import hashlib
import json
from pathlib import Path
import sys


_OPERATOR_SHA = 'cd40e556d68a05a0a430a682163eb457edfc5f29c49b012050f748df71bb42c1'
_OPERATOR_URL = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
    'bb81320ac0f0072db63bf1cd1627bca7e9dc0ff8/tools/injection_comparison_operator.py')
_START = b'MLG-natural-discovery-start-v1\n'
_BOOTSTRAP = ("import hashlib,pathlib,sys\n"
              "p=pathlib.Path(sys.argv[1])\n"
              "with p.open('rb') as f: raw=f.read(1048577)\n"
              "if not 0<len(raw)<=1048576 or hashlib.sha256(raw).hexdigest()!=sys.argv[2]: raise SystemExit(1)\n"
              "sys.argv=[str(p),'--run']+sys.argv[3:]\n"
              "exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),"
              "{'__name__':'__main__','__file__':str(p)})")


class SupervisorError(ValueError):
    """The operator's process or its bound evidence is incomplete."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def operator_command(operator, config, source, config_path, config_sha256, manifest_sha256, manifest_url):
    """Build a direct-base isolated command with hash-before-execution."""
    if type(config) is not dict:
        raise SupervisorError('a bound operator configuration is required')
    try:
        executable, prefix, source, config_path = map(Path, (
            config['base_python'], config['pycache_prefix'], source, config_path))
        if (any(not path.is_absolute() or '..' in path.parts
                for path in (executable, prefix, source, config_path))
                or executable.resolve(strict=True) != Path(sys._base_executable).resolve(strict=True)):
            raise SupervisorError('the direct-base interpreter and absolute paths are required')
        operator._digest(config_sha256)
        operator._digest(manifest_sha256)
        operator._url(manifest_url)
        operator._read_source(source, _OPERATOR_SHA)
        return [str(executable), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix), '-c', _BOOTSTRAP,
                str(source), _OPERATOR_SHA, str(config_path), config_sha256, manifest_sha256, manifest_url]
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, SupervisorError):
            raise
        raise SupervisorError('operator command is invalid') from error


def observe_protected_operator(*, operator, modules, captured, operator_source, operator_config_path,
                               operator_config_sha256, manifest_sha256, manifest_url,
                               output, environment, timeout, reader):
    """Retain and reap the operator, then return its verified clear/collision.

    The calling execution owner must itself be source-bound and protect its
    own lifecycle. A caller that saves this result for another process must
    independently observe its own clean exit. Neither status announces a
    first forward; the later shared gate must explicitly require clear.
    """
    required = {'comparison', 'custody', 'source_guards', 'replay_owner',
                'natural_query', 'adapter', 'exit', 'observer'}
    if (type(modules) is not dict or not required <= set(modules) or not callable(reader)
            or type(environment) is not dict or environment.get('CUDA_VISIBLE_DEVICES') != '-1'
            or type(timeout) is not int or not 1 <= timeout <= 7200):
        raise SupervisorError('bound operator observation inputs are required')
    comparison, custody = modules['comparison'], modules['custody']
    guard, owner, adapter, exit_check = (modules[key] for key in
                                         ('source_guards', 'replay_owner', 'adapter', 'exit'))
    operator_source, operator_config_path, output = map(Path, (operator_source, operator_config_path, output))
    try:
        outer_raw = operator._read_source(operator_config_path, operator_config_sha256)
        outer = operator.parse_operator_config(outer_raw, manifest_sha256, manifest_url)
        manifest_path = Path(outer['manifest_path'])
        manifest_raw = operator._read_source(manifest_path, manifest_sha256)
        manifest = operator.validate_source_manifest(manifest_raw, manifest_sha256, manifest_url, reader)
        if manifest['sources'].get('operator') != {'sha256': _OPERATOR_SHA, 'url': _OPERATOR_URL}:
            raise SupervisorError('the published operator identity differs')
        own = manifest['sources'].get('supervisor')
        if type(own) is not dict:
            raise SupervisorError('the supervisor is absent from the manifest')
        own_source = Path(__file__)
        own_raw = operator._read_source(own_source, own['sha256'])
        operator_raw = operator._read_source(operator_source, _OPERATOR_SHA)
        if reader(own['url']) != own_raw or reader(_OPERATOR_URL) != operator_raw:
            raise SupervisorError('published operator or supervisor differs')
        operator.verify_operator_manifest(manifest_raw, manifest_sha256, manifest_url, operator_source, reader)
        if (type(captured) is not tuple or len(captured) != len(operator._PINNED_SOURCES)
                or [item[0] for item in captured] != [item[0] for item in operator._PINNED_SOURCES]
                or any(Path(outer['source_paths'][entry[0]]) != entry[3] for entry in captured)):
            raise SupervisorError('captured source paths differ')
        config_path = Path(outer['comparison_config_path'])
        operation_path = Path(outer['operation_path'])
        config_raw = comparison._read_file(config_path, 65536)
        operation_raw = comparison._read_file(operation_path, 2 * 1024 * 1024)
        if _sha(config_raw) != outer['comparison_config_sha256']:
            raise SupervisorError('comparison configuration differs')
        config = comparison.parse_config(config_raw)
        private = operator._private_paths(comparison, config, outer['natural_parent'],
                                          outer['injection_parent'], outer['site_parent'])
        phase_parent, phase = private['injection_parent'], private['injection_phase']
        inner = Path(outer['inner_output'])
        terminal, receipt = Path(outer['terminal_path']), Path(config['output'])
        if (not operator_config_path.is_relative_to(phase_parent) or operator_config_path.is_relative_to(phase)
                or not output.is_absolute() or '..' in output.parts or not output.is_relative_to(phase_parent)
                or output.is_relative_to(phase) or output.is_relative_to(inner) or inner.is_relative_to(output)
                or terminal.is_relative_to(output) or receipt.is_relative_to(output)):
            raise SupervisorError('operator evidence paths differ from the phase parent')
        freeze = comparison._read_file(config['injection']['freeze'], 65536)
        commitment = comparison._read_file(config['injection']['commitment'], 8192)
        def public_check():
            operator.validate_public_binding(custody=custody, comparison=comparison,
                operation=operation_raw, operation_sha256=outer['operation_sha256'],
                operation_url=outer['operation_url'], freeze=freeze, commitment=commitment,
                commitment_sha256=config['injection_commitment_sha256'],
                commitment_url=config['injection_commitment_url'], manifest=manifest_raw,
                manifest_sha256=manifest_sha256, manifest_url=manifest_url, reader=reader)
            return True
        public_check()
        public_paths = {operator_config_path:(outer_raw,65536), config_path:(config_raw,65536),
            operation_path:(operation_raw,2*1024*1024), manifest_path:(manifest_raw,65536),
            Path(config['injection']['freeze']):(freeze,65536),
            Path(config['injection']['commitment']):(commitment,8192)}
        for key, limit in (('freeze',65536),('commitment',8192),('inventory',16*1024*1024)):
            path = Path(config['natural'][key])
            public_paths[path] = (comparison._read_file(path, limit), limit)
        prefixes = (Path(outer['pycache_prefix']), Path(config['pycache_prefix']))
        if prefixes[0] == prefixes[1] or any(not path.is_dir() or any(path.iterdir()) for path in prefixes):
            raise SupervisorError('distinct fresh bytecode prefixes are required')
        private_files = [path for name,path in private.items() if name not in
            ('natural_parent','natural_phase','injection_parent','injection_phase','site_parent')]
        files = {operator_source, own_source, *public_paths, *(entry[3] for entry in captured), *private_files}
        groups = {}
        for path in files: groups.setdefault(path.parent, []).append(path)
        leases = [guard.SourceFileLease(parent, sorted(paths,key=str))
                  for parent,paths in sorted(groups.items(), key=lambda item:str(item[0]))]
        command = operator_command(operator, outer, operator_source, operator_config_path,
                                   operator_config_sha256, manifest_sha256, manifest_url)
        source_paths = {entry[0]:entry[3] for entry in captured}
        comparator_sha = dict((name,digest) for name,digest,_ in operator._PINNED_SOURCES)['comparison']
        comparator_command = [outer['base_python'],'-I','-S','-B','-X',
            'pycache_prefix='+config['pycache_prefix'],'-c',adapter._BOOTSTRAP,
            str(source_paths['comparison']),comparator_sha,str(config_path)]
        comparator_command_sha = _sha(_canonical(comparator_command))

        def source_check():
            try:
                if (operator._read_source(operator_source,_OPERATOR_SHA) != operator_raw
                        or operator._read_source(own_source,own['sha256']) != own_raw
                        or any(comparison._read_file(path,limit) != raw
                               for path,(raw,limit) in public_paths.items())
                        or any(operator._read_source(path,expected) != raw
                               for (_,_,_,path,raw),(_,expected,_) in zip(captured,operator._PINNED_SOURCES))):
                    return False
                return all(path.is_dir() and not any(path.iterdir()) for path in prefixes)
            except (OSError, ValueError, TypeError):
                return False

        def private_check():
            return operator.verify_private_inputs(comparison,config,acl=owner,
                natural_query=modules['natural_query'],natural_parent=outer['natural_parent'],
                injection_parent=outer['injection_parent'],site_parent=outer['site_parent'],
                identities=outer['identities'])

        def precheck(_leases):
            if source_check() is not True or public_check() is not True or private_check() is not True:
                raise SupervisorError('the protected comparison inputs differ')
            return True

        def launch(acquired):
            handles = tuple(handle for lease in acquired for handle in lease.handles())
            def protect_process(process):
                return len(guard.duplicate_to_process(handles,process)) == len(handles)
            observed = exit_check.observe_operator(comparison=comparison, command=command, output=output,
                environment=environment, config_path=config_path, operation_path=operation_path,
                terminal_path=Path(outer['terminal_path']), inner_path=Path(outer['inner_output'])/'observation.json',
                receipt_path=Path(config['output']), start_token=_START, timeout=timeout,
                source_check=source_check, protect_process=protect_process,
                receipt_check=lambda raw:exit_check.verify_public_receipt(comparison=comparison,
                    custody=custody,config=config,receipt=raw,reader=reader),
                observer=modules['observer'].observe_supervisor, config_sha256=outer['comparison_config_sha256'],
                operation_sha256=outer['operation_sha256'], comparator_command_sha256=comparator_command_sha,
                operator_source_sha256=_OPERATOR_SHA)
            if type(observed) is not dict or observed.get('status') not in ('clear','collision'):
                raise SupervisorError('the observed operator result is incomplete')
            return {'cleanup_verified':True,'result':observed}

        def postcheck(result):
            if source_check() is not True or private_check() is not True:
                raise SupervisorError('protected inputs changed during observation')
            return result['result']
        return owner.run_guarded(leases,precheck,launch,postcheck)
    except (KeyError, OSError, TypeError, ValueError) as error:
        if isinstance(error, SupervisorError):
            raise
        raise SupervisorError('the operator observation could not be established') from error
