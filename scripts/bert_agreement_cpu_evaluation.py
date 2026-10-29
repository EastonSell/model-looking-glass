"""Launch a frozen CPU evaluation and record the controller's local exit.

Run with an isolated base interpreter and an empty pycache prefix. A missing
exit record leaves the cooperative ledger blocked; these observations are not
an independent attestation of process execution.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import sys


def _require_isolation():
    prefix = sys.pycache_prefix
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or type(prefix) is not str or not Path(prefix).is_absolute()
            or sys._xoptions.get('pycache_prefix') != prefix or not Path(prefix).is_dir()
            or Path(prefix).is_symlink() or any(Path(prefix).iterdir())):
        raise RuntimeError('use an isolated interpreter and a fresh empty pycache prefix')
    if (os.environ.get('LOOKING_GLASS_BERT_BOOTSTRAP_PID') != str(os.getpid())
            or os.environ.get('LOOKING_GLASS_BERT_ROOT') != str(Path(__file__).resolve().parent.parent)):
        raise RuntimeError('use the source-isolated BERT bootstrap launcher entry')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'cache', 'site', 'commit', 'custody-root', 'controller-custody-root',
                 'ledger-root', 'work-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    _require_isolation()
    from looking_glass import bert_agreement_study as study
    if not study.EVALUATION_ENABLED:
        raise RuntimeError('evaluation remains disabled pending the frozen reviewed protocol')
    repo = Path(args.repo).resolve(strict=True)
    if repo != Path(__file__).resolve(strict=True).parent.parent:
        raise ValueError('launcher and requested checkout differ')
    cache, site = Path(args.cache).resolve(strict=True), Path(args.site).resolve(strict=True)
    bootstrap_site = os.environ.get('LOOKING_GLASS_BERT_SITE')
    if not bootstrap_site or Path(bootstrap_site).resolve(strict=True) != site:
        raise ValueError('launcher and bootstrap dependency directories differ')
    custody, ledger = Path(args.custody_root).resolve(strict=True), Path(args.ledger_root).resolve(strict=True)
    controller_custody = Path(args.controller_custody_root).resolve(strict=True)
    work = Path(args.work_dir)
    if (not work.is_absolute() or work.exists() or not work.parent.is_dir()
            or not cache.is_dir() or not site.is_dir() or not custody.is_dir() or not ledger.is_dir()
            or not controller_custody.is_dir()):
        raise ValueError('launcher requires existing roots and a new absolute work directory')
    parent = work.parent.resolve(strict=True)
    prospective = parent / work.name
    if any(prospective == other or prospective.is_relative_to(other) or other.is_relative_to(prospective)
           for other in (repo, cache, site, custody, controller_custody, ledger)):
        raise ValueError('launcher work directory must be separate from checkout and study roots')
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for a in (custody, controller_custody, ledger) for b in (repo, cache, site)
           ) or any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                    for a, b in ((custody, controller_custody), (custody, ledger), (controller_custody, ledger))):
        raise ValueError('controller, owner, ledger, checkout, cache and dependency roots must be separate')
    work.mkdir()
    controller_prefix, temporary = work / 'controller-pycache', work / 'tmp'
    controller_prefix.mkdir()
    temporary.mkdir()
    run_id = secrets.token_hex(16)
    destination = ledger / run_id
    base = Path(sys._base_executable).resolve(strict=True)
    bootstrap = repo / 'scripts/bert_agreement_bootstrap.py'
    common = ['--bootstrap-site', str(site), '--repo', str(repo), '--cache', str(cache),
              '--output', str(destination), '--commit', args.commit, '--split', 'evaluation',
              '--custody-root', str(custody), '--ledger-root', str(ledger),
              '--controller-custody-root', str(controller_custody)]
    git_exe = shutil.which('git')
    if not git_exe:
        raise RuntimeError('Git is required for evaluation preflight')
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT') if key in os.environ}
    system_root = Path(env.get('SYSTEMROOT', 'C:/Windows'))
    search_paths = [str(Path(git_exe).resolve(strict=True).parent)]
    search_paths.append(str(system_root / 'System32') if os.name == 'nt' else '/usr/bin:/bin')
    env.update({'PATH': os.pathsep.join(search_paths), 'TEMP': str(temporary), 'TMP': str(temporary),
                'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    print(json.dumps({'schema': 'bert-evaluation-launch-v1', 'run_id': run_id}), flush=True)
    command = [str(base), '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(controller_prefix),
               str(bootstrap), *common]
    from looking_glass.cpu_study_supervision import run_cpu_study
    from looking_glass.supervisor import SupervisionResult
    observed = run_cpu_study(command, study_root=controller_custody, timeout=1300,
                             cwd=work, env=env, inherit_environment=False)
    if (type(observed) is not SupervisionResult or type(observed.pid) is not int or observed.pid <= 0
            or type(observed.exit_code) is not int or observed.clean is not True
            or observed.timed_out is not False):
        # In particular, do not fabricate an exit after a lost/uncertain
        # supervisor return or after controller teardown failed.
        return 1
    if not destination.exists():
        # A clean no-start controller can reject a known inner quarantine
        # before making any reservation; there is no attempt to record.
        return 1
    supervision = {'scope': 'cooperative-cpu-controller-custody-only', 'run_id': observed.run_id,
                   'pid': observed.pid, 'exit_code': observed.exit_code, 'clean': observed.clean,
                   'timed_out': observed.timed_out, 'reason': observed.reason}
    study.record_cpu_evaluation_exit(repo=repo, commit=args.commit, output_dir=destination, run_id=run_id,
                                     supervision=supervision, custody_root=custody, ledger_root=ledger,
                                     controller_custody_root=controller_custody)
    return 0 if observed.exit_code == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
