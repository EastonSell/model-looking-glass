"""Supervised, source-bound CPU owner for the public matched-dose study.

Run this committed script under ``-I -S -B`` with a fresh cache prefix.  The
ledger and process controls are cooperative, not execution attestation; the
returned receipt requires an independently observed controller exit as well.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import secrets
import signal
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
EXTRA = ("scripts/bert_matched_cpu.py", "docs/BERT_MATCHED_MAGNITUDE_V1.md",
         "experiments/bert_matched_public_cases_v1.json", "docs/BERT_AGREEMENT_V1.md",
         "docs/CPU_STUDY_SUPERVISION.md", "docs/SUPERVISOR.md", "docs/BERT_MATCHED_CPU_OWNER.md")
# Preallocated before any child/resource exists: no allocation is needed to
# retain a ledger or source lease after an exceptional supervisor return.
_ACTIVE = [None, None, False]
_INTERRUPTED = [False]



def _isolated():
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()
            or not Path(sys.pycache_prefix).is_dir() or any(Path(sys.pycache_prefix).iterdir())):
        raise RuntimeError('fresh isolated, no-site, no-bytecode startup is required')
    if any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules):
        raise RuntimeError('project imports preceded source verification')


def _project_loader(source_hashes, site):
    site = Path(site).resolve(strict=True)
    if (not site.is_dir() or site == ROOT or site.is_relative_to(ROOT) or ROOT.is_relative_to(site)):
        raise ValueError('dependency directory must be separate from checkout')
    bootstrap = ROOT / 'scripts/v2_source_bootstrap.py'
    if hashlib.sha256(bootstrap.read_bytes()).hexdigest() != source_hashes['scripts/v2_source_bootstrap.py']:
        raise RuntimeError('source bootstrap changed')
    tools = runpy.run_path(str(bootstrap))
    project = {key: value for key, value in source_hashes.items()
               if key.startswith('looking_glass/') and key.endswith('.py')}
    finder = tools['_ProjectFinder'](tools['_source_map'](ROOT, project))
    sys.meta_path.insert(0, finder)
    sys.path.append(str(site))
    return finder


def _source_check(hashes):
    for relative, digest in hashes.items():
        path = (ROOT / relative).resolve(strict=True)
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError('source escaped the frozen checkout')
        with path.open('rb') as stream:
            raw = stream.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError('source bytes changed')


def _source_inventory(revision):
    # Verify the helper against Git before executing it. Git is still a trusted
    # local executable. The already-running owner relies on the documented
    # trusted-checkout assumption; these checks are not execution attestation.
    git = shutil.which('git')
    if not git:
        raise RuntimeError('Git is unavailable')
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    environment.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_GRAFT_FILE': os.devnull,
                        'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                        'GIT_TERMINAL_PROMPT': '0'})
    result = subprocess.run([git, '--no-replace-objects', '-C', str(ROOT), 'show',
                             revision + ':scripts/v2_source_launcher.py'],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, env=environment)
    launcher_path = ROOT / 'scripts/v2_source_launcher.py'
    if result.returncode or len(result.stdout) > 8 * 1024 * 1024 or launcher_path.read_bytes() != result.stdout:
        raise RuntimeError('source launcher differs from the requested commit')
    launcher_globals = {'__name__': '_matched_verified_launcher', '__file__': str(launcher_path)}
    exec(compile(result.stdout, str(launcher_path), 'exec'), launcher_globals)
    launcher = launcher_globals
    actual = launcher['_revision'](ROOT)
    if actual != revision:
        raise RuntimeError('checkout revision differs from the requested freeze')
    blobs = launcher['_committed_sources'](ROOT, actual, extra=EXTRA)
    return launcher, blobs, launcher['_verify_sources'](ROOT, blobs)


def _binding(revision, hashes, cache):
    from looking_glass import bert_matched_cases as cases
    from looking_glass.bert_agreement_study import ASSETS, MODEL_ID, REVISION
    manifest_path = ROOT / 'experiments/bert_matched_public_cases_v1.json'
    raw = manifest_path.read_bytes()
    checked = cases.verify_manifest_bytes(raw)
    if (checked['manifest_sha256'] != hashes[EXTRA[2]]
            or hashes[EXTRA[1]] != cases.PROTOCOL_SHA256):
        raise RuntimeError('the committed protocol or manifest differs from the frozen study')
    snapshot = Path(cache).resolve(strict=True) / 'models--distilbert--distilbert-base-uncased' / 'snapshots' / REVISION
    assets = {}
    for name, (size, expected) in ASSETS.items():
        path = snapshot / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            total = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                total += len(chunk)
                if total > size:
                    raise RuntimeError('checkpoint asset is oversized')
                digest.update(chunk)
        if total != size or digest.hexdigest() != expected:
            raise RuntimeError('checkpoint asset differs from its frozen digest')
        assets[name] = {'size': size, 'sha256': expected}
    manifest = cases.build_manifest()
    return {'source_commit': revision, 'source_sha256': hashes,
            'protocol_sha256': cases.PROTOCOL_SHA256, 'manifest_sha256': checked['manifest_sha256'],
            'prior_exposure_sha256': cases.PRIOR_EXPOSURE_SHA256,
            'model': {'model': MODEL_ID, 'revision': REVISION, 'assets': assets},
            'rows': [{key: row[key] for key in ('row_id', 'input_ids', 'attention_mask')}
                     for row in manifest['rows']]}, manifest, snapshot


class _AuditSentinel(Exception):
    pass


def _deny_spawn(event, _args):
    if event == 'mlg.bert_matched.audit_check':
        raise _AuditSentinel()
    if event in ('subprocess.Popen', '_winapi.CreateProcess', 'os.system', 'os.spawn',
                 'os.posix_spawn', 'os.exec', 'os.fork', 'os.forkpty', 'os.startfile', 'os.startfile/2'):
        raise RuntimeError('the CPU owner cannot create descendant processes')


def _install_spawn_guard():
    sys.addaudithook(_deny_spawn)
    try:
        sys.audit('mlg.bert_matched.audit_check')
    except _AuditSentinel:
        return
    raise RuntimeError('process creation audit hook could not be established')


def _runtime(torch, numpy, transformers):
    import platform
    if (torch.cuda.is_initialized() or torch.is_autocast_enabled('cpu')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise RuntimeError('CUDA or CPU autocast is unsupported')
    return {'python': platform.python_version(), 'platform': platform.system(),
            'machine': platform.machine() or 'unknown', 'torch': str(torch.__version__),
            'transformers': str(transformers.__version__), 'numpy': str(numpy.__version__),
            'intraop_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'denormals_preserved_calling_thread': True, 'pytorch_cuda_initialized': False,
            'cuda_visible_devices': '',
            'python_hash_seed': 'isolated interpreter; OS-randomized', 'seed': 2026102810}


def _validate_candidate(candidate, observed, reservation_sha, args, binding):
    from looking_glass.bert_agreement_study import validate_cpu_runtime_and_model
    base = {'schema', 'pid', 'reservation_sha256', 'source_commit', 'manifest_sha256', 'forward_count'}
    expected = (base | {'cuda_visible_devices'} if args.rehearsal else
                base | {'journal_sha256', 'model_state_sha256_before_after', 'model', 'runtime', 'audit'})
    schema = 'bert-matched-owner-rehearsal-v1' if args.rehearsal else 'bert-matched-owner-candidate-v1'
    if (type(candidate) is not dict or set(candidate) != expected or candidate['schema'] != schema
            or type(candidate['pid']) is not int or candidate['pid'] != observed.pid
            or candidate['reservation_sha256'] != reservation_sha
            or candidate['source_commit'] != args.revision
            or candidate['manifest_sha256'] != binding['manifest_sha256']
            or type(candidate['forward_count']) is not int
            or candidate['forward_count'] != (0 if args.rehearsal or args.checkpoint_preflight else 480)):
        raise RuntimeError('owner candidate does not match the reaped process and reservation')
    if args.rehearsal:
        if type(candidate['cuda_visible_devices']) is not str or candidate['cuda_visible_devices'] != '':
            raise RuntimeError('CPU visibility mask was not recorded')
        return
    runtime = candidate['runtime']
    if (type(runtime) is not dict or type(runtime.get('cuda_visible_devices')) is not str
            or runtime['cuda_visible_devices'] != '' or runtime.get('python') != '3.11.15'):
        raise RuntimeError('CPU runtime identity or visibility mask differs')
    validate_cpu_runtime_and_model(runtime, candidate['model'], candidate['model_state_sha256_before_after'])
    if args.checkpoint_preflight:
        if candidate['journal_sha256'] is not None or candidate['audit'] is not None:
            raise RuntimeError('checkpoint preflight unexpectedly reported a journal')
    elif (type(candidate['journal_sha256']) is not str or len(candidate['journal_sha256']) != 64
          or any(c not in '0123456789abcdef' for c in candidate['journal_sha256'])
          or type(candidate['audit']) is not dict):
        raise RuntimeError('owner journal identity is invalid')


def _worker(args, hashes, finder):
    from looking_glass.cpu_study_supervision import claim_cpu_study
    from looking_glass.bert_evaluation_ledger import read_record, write_record
    from looking_glass.bert_matched_owner import verify_owner_claim
    session, lease = claim_cpu_study(Path(args.custody))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('the CPU visibility mask changed after the owner claim')
    attempt = Path(args.ledger) / args.attempt
    reservation, reservation_sha = read_record(attempt / 'reservation.json')
    if reservation_sha != args.reservation_sha:
        raise RuntimeError('reservation digest changed')
    binding, manifest, snapshot = _binding(args.revision, hashes, args.cache)
    verify_owner_claim(Path(args.ledger), Path(args.custody), args.attempt,
                       reservation_sha, binding, session=session, lease=lease)
    _source_check(hashes)
    finder.verify_inventory()
    if args.rehearsal:
        if any(name.split('.')[0] in ('torch', 'numpy', 'transformers') for name in sys.modules):
            raise RuntimeError('model dependencies appeared in the no-checkpoint rehearsal')
        write_record(attempt / 'owner-rehearsal.json', {
            'schema': 'bert-matched-owner-rehearsal-v1', 'pid': os.getpid(),
            'reservation_sha256': reservation_sha, 'source_commit': args.revision,
            'manifest_sha256': binding['manifest_sha256'], 'forward_count': 0,
            'cuda_visible_devices': ''})
        session.attest_clean(lease)
        return

    _install_spawn_guard()
    if os.name == 'nt':
        import platform
        windows = sys.getwindowsversion()
        version = f'{windows.major}.{windows.minor}.{windows.build}'
        platform._syscmd_ver = lambda *unused, **kwargs: ('Windows', str(windows.major), version)
    import gc
    import numpy
    import torch
    import transformers
    from looking_glass.bert_agreement_study import (EventJournal, _require_denormals,
        capture_model_state, load_pinned_distilbert, validate_cpu_runtime_and_model,
        verify_model_state, verify_tokenizer_rows)
    from looking_glass.bert_agreement_runner import _has_hooks
    from looking_glass.bert_matched_cases import verify_offline_tokenizer
    from looking_glass.bert_matched_evidence import audit_event_journal
    from looking_glass.bert_matched_numeric import binary64_rounding_probe
    from looking_glass.bert_matched_runner import run_family
    if (str(torch.__version__) != '2.8.0+cu128' or str(transformers.__version__) != '4.57.3'
            or str(numpy.__version__) != '2.4.6' or sys.version_info[:3] != (3, 11, 15)
            or torch.cuda.is_initialized() or torch.is_autocast_enabled('cpu')):
        raise RuntimeError('CPU dependency or runtime identity differs from the protocol')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise RuntimeError('denormal preservation could not be requested')
    _require_denormals(torch)
    binary64_rounding_probe()
    verify_offline_tokenizer(snapshot)
    torch.random.default_generator.manual_seed(2026102810)
    if args.checkpoint_preflight:
        def forbidden_forward(*_args, **_kwargs):
            raise RuntimeError('a model forward is forbidden in checkpoint preflight')
        transformers.DistilBertForMaskedLM.forward = forbidden_forward
    loaded = load_pinned_distilbert(args.cache)
    if verify_tokenizer_rows(loaded.tokenizer, manifest) != 64 or _has_hooks(loaded.model):
        raise RuntimeError('tokenizer or initial hook controls failed')
    witness = capture_model_state(loaded.model)
    state_sha, model_metadata = witness.sha256, loaded.metadata
    runtime = _runtime(torch, numpy, transformers)
    validate_cpu_runtime_and_model(runtime, model_metadata, state_sha)
    if args.checkpoint_preflight:
        verify_model_state(loaded.model, witness)
        _require_denormals(torch)
        binary64_rounding_probe()
        validate_cpu_runtime_and_model(_runtime(torch, numpy, transformers), model_metadata, state_sha)
        _source_check(hashes)
        finder.verify_inventory()
        del witness, loaded
        gc.collect()
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA was initialized during checkpoint preflight')
        write_record(attempt / 'owner-candidate.json', {
            'schema': 'bert-matched-owner-candidate-v1', 'pid': os.getpid(),
            'reservation_sha256': reservation_sha, 'source_commit': args.revision,
            'manifest_sha256': binding['manifest_sha256'], 'journal_sha256': None,
            'model_state_sha256_before_after': state_sha, 'model': model_metadata,
            'runtime': runtime, 'forward_count': 0, 'audit': None})
        session.attest_clean(lease)
        return
    live = []
    with EventJournal(attempt / 'events.jsonl') as journal:
        for family in range(1, 9):
            session.check(lease)
            _require_denormals(torch)
            binary64_rounding_probe()
            live.append(run_family(loaded.model, manifest, family=family, emit=journal.emit))
            _require_denormals(torch)
            binary64_rounding_probe()
    journal_sha = journal.sha256
    if (torch.cuda.is_initialized() or _has_hooks(loaded.model)
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or torch.get_default_dtype() != torch.float32
            or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError('CUDA initialization or residual hooks invalidate the attempt')
    verify_model_state(loaded.model, witness)
    _require_denormals(torch)
    binary64_rounding_probe()
    final_runtime = _runtime(torch, numpy, transformers)
    validate_cpu_runtime_and_model(final_runtime, model_metadata, state_sha)
    if final_runtime != runtime:
        raise RuntimeError('CPU runtime settings changed during the attempt')
    audited = audit_event_journal(manifest, attempt / 'events.jsonl', expected_sha256=journal_sha)
    if json.dumps(live, sort_keys=True, allow_nan=False) != json.dumps(audited['families'], sort_keys=True, allow_nan=False):
        raise RuntimeError('closed journal differs from the live family results')
    _source_check(hashes)
    finder.verify_inventory()
    del witness, loaded
    gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized during teardown')
    write_record(attempt / 'owner-candidate.json', {
        'schema': 'bert-matched-owner-candidate-v1', 'pid': os.getpid(),
        'reservation_sha256': reservation_sha, 'source_commit': args.revision,
        'manifest_sha256': binding['manifest_sha256'], 'journal_sha256': journal_sha,
        'model_state_sha256_before_after': state_sha, 'model': model_metadata,
        'runtime': runtime,
        'forward_count': audited['forward_count'], 'audit': audited})
    session.attest_clean(lease)


def _controller(args, launcher, blobs, hashes, finder):
    from looking_glass.bert_evaluation_ledger import read_record, write_record
    from looking_glass.bert_matched_evidence import audit_event_journal
    from looking_glass.bert_matched_owner import MatchedLedger
    from looking_glass.cpu_study_supervision import run_cpu_study
    from looking_glass.supervisor import pending_supervised_children
    binding, manifest, _ = _binding(args.revision, hashes, args.cache)
    for value in (args.ledger, args.custody):
        location = Path(value).resolve(strict=True)
        if location == ROOT or location.is_relative_to(ROOT) or ROOT.is_relative_to(location):
            raise ValueError('ledger and custody must be separate from the checkout')
    if _ACTIVE[0] is not None:
        raise RuntimeError('a matched owner is already active in this process')
    ledger = MatchedLedger.acquire(Path(args.ledger), Path(args.custody), binding)
    _ACTIVE[0] = ledger
    try:
        guards = launcher['_load_guards'](ROOT, blobs)
        source_lease = guards.SourceFileLease(ROOT, [ROOT / relative for relative in blobs])
        _ACTIVE[1] = source_lease
        source_lease.__enter__()
    except BaseException:
        _ACTIVE[2] = True
        raise
    attempt = None
    observed = None
    try:
        if launcher['_verify_sources'](ROOT, blobs) != hashes:
            raise RuntimeError('source changed before reservation')
        attempt = ledger.reserve(secrets.token_hex(16))
        _, reservation_sha = read_record(attempt / 'reservation.json')
        work, prefix = attempt / 'working', attempt / 'pycache'
        work.mkdir()
        prefix.mkdir()
        temporary = work / 'tmp'
        temporary.mkdir()
        env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT',
                'PROCESSOR_ARCHITECTURE', 'PROCESSOR_ARCHITEW6432', 'PROCESSOR_IDENTIFIER') if key in os.environ}
        env.update({'PATH': str(Path(env.get('SYSTEMROOT', 'C:/Windows')) / 'System32'),
                    'TEMP': str(temporary), 'TMP': str(temporary), 'OMP_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1',
                    'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                    'CUDA_VISIBLE_DEVICES': ''})
        command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                   str(ROOT / 'scripts/bert_matched_cpu.py'), '--worker', '--revision', args.revision,
                   '--site', args.site, '--cache', args.cache, '--ledger', args.ledger,
                   '--custody', args.custody, '--attempt', attempt.name, '--reservation-sha', reservation_sha]
        if args.rehearsal:
            command.append('--rehearsal')
        elif args.checkpoint_preflight:
            command.append('--checkpoint-preflight')
        else:
            command.append('--execute')
        job = guards.WindowsJob()
        if pending_supervised_children():
            raise RuntimeError('another supervised child is already pending in this process')
        _ACTIVE[2] = True
        observed = run_cpu_study(command, study_root=Path(args.custody), timeout=args.timeout,
                                 cwd=work, env=env, process_job=job)
        _ACTIVE[2] = bool(pending_supervised_children())
        if _ACTIVE[2]:
            raise RuntimeError('supervised child custody remains pending')
        if (type(observed.pid) is not int or observed.pid <= 0 or type(observed.exit_code) is not int
                or observed.exit_code != 0 or observed.clean is not True or observed.timed_out is not False):
            ledger.record_incomplete(observed)
            raise RuntimeError('owner did not exit cleanly')
        candidate_name = 'owner-rehearsal.json' if args.rehearsal else 'owner-candidate.json'
        candidate, candidate_sha = read_record(attempt / candidate_name)
        _validate_candidate(candidate, observed, reservation_sha, args, binding)
        if args.rehearsal or args.checkpoint_preflight:
            audited = None
        else:
            audited = audit_event_journal(manifest, attempt / 'events.jsonl',
                                          expected_sha256=candidate['journal_sha256'])
            if (audited['forward_count'] != 480 or audited['record_count'] != 1152
                    or json.dumps(audited, sort_keys=True, allow_nan=False) !=
                    json.dumps(candidate.get('audit'), sort_keys=True, allow_nan=False)):
                raise RuntimeError('closed journal failed independent readback')
        if launcher['_verify_sources'](ROOT, blobs) != hashes:
            raise RuntimeError('source changed during the attempt')
        finder.verify_inventory()
        receipt = {'schema': 'bert-matched-controller-v1', 'status': 'rehearsal_candidate' if (args.rehearsal or args.checkpoint_preflight) else 'complete_candidate',
                   'attempt_id': attempt.name, 'reservation_sha256': reservation_sha,
                   'owner_candidate_sha256': candidate_sha, 'owner_pid': observed.pid,
                   'owner_exit_code': observed.exit_code, 'owner_reaped': True,
                   'scope': 'cooperative CPU custody; observed controller exit is separately required',
                   'source_commit': args.revision, 'manifest_sha256': binding['manifest_sha256'],
                   'forward_count': 0 if audited is None else audited['forward_count'],
                   'journal_sha256': None if audited is None else audited['sha256'],
                   'audit': audited}
        write_record(attempt / 'controller-result.json', receipt)
        return receipt
    except BaseException as primary:
        # The supervisor retains unconfirmed job/child capabilities. Do not
        # release source or ledger custody while those capabilities remain.
        _ACTIVE[2] = True
        try:
            pending = pending_supervised_children()
        except BaseException as query_error:
            if isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise primary.with_traceback(primary.__traceback__) from query_error
            if isinstance(query_error, (KeyboardInterrupt, SystemExit)):
                raise query_error from primary
            raise primary.with_traceback(primary.__traceback__) from query_error
        if not pending:
            _ACTIVE[2] = False
        if attempt is not None:
            try:
                if observed is not None and not (attempt / 'result.json').exists():
                    ledger.record_incomplete(observed)
                write_record(attempt / 'controller-failure.json', {'schema': 'bert-matched-controller-failure-v1',
                    'status': 'incomplete', 'attempt_id': attempt.name, 'possibly_exposed': True,
                    'pending_child_custody': bool(pending), 'error_type': type(primary).__name__})
            except Exception:
                pass
        raise
    finally:
        if not _ACTIVE[2]:
            primary = sys.exc_info()[1]
            _ACTIVE[2] = True
            try:
                source_lease.__exit__(None, None, None)
            except BaseException as cleanup:
                if attempt is not None:
                    try:
                        write_record(attempt / 'controller-failure.json', {
                            'schema': 'bert-matched-controller-failure-v1', 'status': 'incomplete',
                            'attempt_id': attempt.name, 'possibly_exposed': True,
                            'pending_child_custody': False, 'error_type': type(cleanup).__name__})
                    except Exception:
                        pass
                if primary is not None:
                    if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                        raise cleanup from primary
                    raise primary.with_traceback(primary.__traceback__) from cleanup
                raise
            ledger.release()
            _ACTIVE[0] = None
            _ACTIVE[1] = None
            _ACTIVE[2] = False


def _recover_active():
    """Keep rooted custody while retrying an uncertain child or lease cleanup."""
    previous = []
    masked = False
    while True:
        try:
            if not _ACTIVE[2]:
                break
            if not masked:
                for name in ('SIGINT', 'SIGBREAK'):
                    signum = getattr(signal, name, None)
                    if signum is not None:
                        old = signal.signal(signum, signal.SIG_IGN)
                        previous.append((signum, old))
                masked = True
            ledger, source_lease = _ACTIVE[0], _ACTIVE[1]
            from looking_glass.supervisor import pending_supervised_children
            pending = pending_supervised_children()
            for record in pending:
                record.recover(terminate_grace=5)
            if pending_supervised_children():
                continue
            if source_lease is not None and source_lease._kernel is not None:
                source_lease.close()
            if ledger is not None:
                ledger.release()
            _ACTIVE[0] = None
            _ACTIVE[1] = None
            _ACTIVE[2] = False
        except BaseException:
            try:
                time.sleep(1)
            except BaseException:
                pass
    for signum, old in reversed(previous):
        try:
            signal.signal(signum, old)
        except BaseException:
            pass


def _console_interrupt(_signum, _frame):
    # Once a cancellation is already in flight, do not let repeated console
    # signals cut short recovery of rooted child/source custody.
    if _ACTIVE[2] and (_INTERRUPTED[0] or sys.exc_info()[0] is not None):
        _INTERRUPTED[0] = True
        return
    _INTERRUPTED[0] = True
    raise KeyboardInterrupt()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', required=True)
    parser.add_argument('--site', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--custody', required=True)
    parser.add_argument('--timeout', type=int, default=3600)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--rehearsal', action='store_true')
    mode.add_argument('--checkpoint-preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--attempt', help=argparse.SUPPRESS)
    parser.add_argument('--reservation-sha', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    _isolated()
    if args.worker:
        # The reservation was created by the parent before the gated spawn.
        path = Path(args.ledger) / args.attempt / 'reservation.json'
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != args.reservation_sha:
            raise RuntimeError('invalid worker reservation')
        reservation = json.loads(raw)
        hashes = reservation['binding']['source_sha256']
        _source_check(hashes)
        finder = _project_loader(hashes, args.site)
        _worker(args, hashes, finder)
    else:
        launcher, blobs, hashes = _source_inventory(args.revision)
        finder = _project_loader(hashes, args.site)
        receipt = _controller(args, launcher, blobs, hashes, finder)
        print(json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    for _signal_name in ('SIGINT', 'SIGBREAK'):
        _signum = getattr(signal, _signal_name, None)
        if _signum is not None:
            signal.signal(_signum, _console_interrupt)
    try:
        main()
    finally:
        # A failed drain/close keeps the process, source handles and ledger
        # lease alive until recovery actually succeeds. An external forced
        # process termination remains outside this cooperative guarantee.
        _recover_active()
