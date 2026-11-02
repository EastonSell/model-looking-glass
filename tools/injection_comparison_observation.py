"""Bind a comparison receipt to the reviewed process observer.

This adapter reads public commitments and a path-only configuration. It never
opens private split or site records. A trusted source-first operator protects
the paths and source leases, and independently observes this process's exit.
The result is not a first-forward authorization.
"""
import hashlib
import json
from pathlib import Path


_CALLER_SHA = '3e61675a962a205aebf336b84699ee7a4fc6d500b01cb25ab7ebea0fda094ab4'
_CALLER_URL = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
               'b83cc2cf7b2534318f66145f07873c890feb5f1e/tools/injection_split_comparison.py')
_START = b'MLG-natural-discovery-start-v1\n'
_BOOTSTRAP = ("import hashlib,pathlib,sys\n"
              "p=pathlib.Path(sys.argv[1])\n"
              "with p.open('rb') as f: raw=f.read(1048577)\n"
              "if not 0<len(raw)<=1048576 or hashlib.sha256(raw).hexdigest()!=sys.argv[2]: raise SystemExit(1)\n"
              "sys.argv=[str(p),'--compare',sys.argv[3]]\n"
              "exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),"
              "{'__name__':'__main__','__file__':str(p)})")


class ObservationError(ValueError):
    """A clean and bound comparison process was not established."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def observe_comparison(*, comparison, custody, caller_source, config_path, config_sha256,
                       operation_path, operation_sha256, output, base_python, environment,
                       timeout, source_check, protect_process, reader, observer):
    """Observe one direct comparator process and verify its public receipt.

    ``observer`` is the reviewed outer observer; the trusted operator must
    verify its source and retain an independent observation of this adapter's
    own clean exit. ``source_check`` and ``protect_process`` must protect the
    source, configuration and evidence identities for the whole launch.
    """
    if any(not callable(value) for value in (source_check, protect_process, reader, observer)):
        raise ObservationError('trusted process and source observers are required')
    config_raw = comparison._read_file(config_path, 65536)
    operation_raw = comparison._read_file(operation_path, 2 * 1024 * 1024)
    if _sha(config_raw) != config_sha256 or _sha(operation_raw) != operation_sha256:
        raise ObservationError('bound configuration or operation differs')
    config = comparison.parse_config(config_raw)
    caller = comparison._read_file(caller_source, 1024 * 1024)
    if (_sha(caller) != _CALLER_SHA or reader(_CALLER_URL) != caller
            or source_check() is not True):
        raise ObservationError('comparison source was not verified')

    public = {}
    for track, expected, url in (
        ('natural', comparison._NATURAL_COMMITMENT, comparison._PUBLIC_URLS['natural_commitment']),
        ('injection', config['injection_commitment_sha256'], config['injection_commitment_url']),
    ):
        raw = comparison._read_file(config[track]['commitment'], 1024 * 1024)
        if _sha(raw) != expected or reader(url) != raw:
            raise ObservationError('published commitment differs')
        freeze = comparison._NATURAL_FREEZE if track == 'natural' else comparison._INJECTION_FREEZE
        public[track] = (raw, comparison._public_commitment(track, raw, freeze))

    base = Path(base_python)
    if not base.is_absolute() or not Path(caller_source).is_absolute() or not Path(config_path).is_absolute():
        raise ObservationError('absolute executable and source paths are required')
    command = [str(base), '-I', '-S', '-B', '-X', 'pycache_prefix=' + config['pycache_prefix'],
               '-c', _BOOTSTRAP, str(caller_source), _CALLER_SHA, str(config_path)]
    command_sha = _sha(_canonical(command))
    verified = {}

    def source_still_bound():
        return (comparison._read_file(caller_source, 1024 * 1024) == caller
                and comparison._read_file(config_path, 65536) == config_raw
                and comparison._read_file(config['natural']['commitment'], 1024 * 1024) == public['natural'][0]
                and comparison._read_file(config['injection']['commitment'], 1024 * 1024) == public['injection'][0]
                and source_check() is True)

    def check_terminal(raw, purpose, pid, birth):
        try:
            if (purpose != 'execution' or type(pid) is not int or pid <= 0
                    or type(birth) is not int or birth <= 0 or type(raw) is not bytes
                    or not 0 < len(raw) <= 8192):
                return False
            value = json.loads(raw)
            if (type(value) is not dict or _canonical(value) != raw
                    or value.get('status') not in ('clear', 'collision')):
                return False
            status = value['status']
            # Verify the same exact identities for a negative result, while
            # retaining its actual collision status and bytes.
            clear_value = dict(value, status='clear')
            if custody.verify_clear_receipt(_canonical(clear_value),
                    natural_freeze_sha256=comparison._NATURAL_FREEZE,
                    injection_freeze_sha256=comparison._INJECTION_FREEZE,
                    natural_inventory_sha256=public['natural'][1]['exposure_inventory_sha256'],
                    injection_inventory_sha256=public['injection'][1]['exposure_inventory_sha256'],
                    natural_commitment_bytes=public['natural'][0],
                    injection_commitment_bytes=public['injection'][0],
                    comparator_source_sha256=comparison._SOURCE_SPECS[-1][1]) is not True:
                return False
            verified.update(status=status, receipt_sha256=_sha(raw), pid=pid, birth=birth)
            return True
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
            return False

    observation = observer(purpose='execution', command=command, output=Path(output),
        environment=environment, config_path=Path(config_path), config_sha256=config_sha256,
        operation_path=Path(operation_path), operation_sha256=operation_sha256,
        command_sha256=command_sha, supervisor_source_sha256=_CALLER_SHA,
        terminal_path=Path(config['output']), start_token=_START, timeout=timeout,
        source_check=source_still_bound, protect_process=protect_process, terminal_check=check_terminal)
    if (type(observation) is not dict or observation.get('schema') != 'mlg-injection-outer-process-observation-v1'
            or observation.get('purpose') != 'execution' or observation.get('status') != 'completed'
            or type(observation.get('exit_code')) is not int or observation['exit_code'] != 0
            or any(observation.get(key) is not True for key in
                   ('parent_reaped', 'parent_handle_closed', 'outer_job_drained', 'outer_job_closed'))
            or type(observation.get('active_job_members')) is not int or observation['active_job_members'] != 0
            or observation.get('parent_pid') != verified.get('pid')
            or observation.get('parent_birth_100ns') != verified.get('birth')
            or observation.get('supervisor_terminal_sha256') != verified.get('receipt_sha256')
            or observation.get('config_sha256') != config_sha256
            or observation.get('operation_sha256') != operation_sha256
            or observation.get('command_sha256') != command_sha
            or observation.get('supervisor_source_sha256') != _CALLER_SHA
            or verified.get('status') not in ('clear', 'collision')):
        raise ObservationError('comparison process observation is incomplete')
    return {'comparison_status': verified['status'], 'receipt_sha256': verified['receipt_sha256'],
            'observer_observation_sha256': _sha(_canonical(observation))}
