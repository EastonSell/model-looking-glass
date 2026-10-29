"""A no-checkpoint rehearsal of the matched CPU supervision boundary.

This development primitive does not authorize a study or authenticate the
caller-supplied child. The real owner must bind its fixed source, manifest,
checkpoint and independently audited journal before using a study entry point.
"""

import hashlib
import os
from pathlib import Path
import re
import stat

from .bert_evaluation_ledger import read_record, write_record
from .bert_matched_owner import MatchedLedger
from .cpu_study_supervision import run_cpu_study
from .supervisor import SupervisionResult


_FAKE_JOURNAL = b'{"kind":"fabricated-no-model"}\n'


def _closed_journal(path, expected_sha):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400 or info.st_size > 1024):
        raise ValueError("rehearsal journal is not a bounded regular file")
    with path.open("rb") as stream:
        raw = stream.read(1025)
    if raw != _FAKE_JOURNAL or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("rehearsal journal differs from the fabricated fixture")
    return hashlib.sha256(raw).hexdigest()


def run_rehearsal(ledger, attempt_id, *, command, cwd, env, timeout, process_job=None):
    """Exercise a trusted fabricated child, never a real matched-study run.

    The sentinel source and manifest identity constrain this helper to a
    fabricated test binding. It records a zero-forward *claim* from the child;
    neither this helper nor lock contention attests what arbitrary child code
    executed. The completed rehearsal is not a study result or a retry grant.
    """
    if type(ledger) is not MatchedLedger or not callable(command):
        raise ValueError("a matched ledger and trusted fixture command are required")
    ledger._check()
    binding = ledger.binding
    if binding["source_commit"] != "a" * 40 or binding["manifest_sha256"] != "c" * 64:
        raise ValueError("rehearsal only accepts the fabricated fixture binding")
    work = Path(cwd)
    if not work.is_absolute() or not work.is_dir():
        raise ValueError("rehearsal working directory must already exist")
    work = work.resolve(strict=True)
    if (work == ledger.root or work == ledger.custody or work.is_relative_to(ledger.root)
            or work.is_relative_to(ledger.custody)):
        raise ValueError("rehearsal work must be separate from custody roots")
    if type(env) is not dict or any(type(k) is not str or type(v) is not str for k, v in env.items()):
        raise ValueError("rehearsal environment must contain only plain strings")
    path = ledger.reserve(attempt_id)
    _, reservation_sha = read_record(path / "reservation.json")
    try:
        args = command(path, reservation_sha)
        if (type(args) is not list or not args or len(args) > 128
                or any(type(arg) is not str or not arg or len(arg) > 8192 for arg in args)):
            raise ValueError("fixture command is invalid")
        observed = run_cpu_study(args, study_root=ledger.custody, timeout=timeout,
                                 cwd=work, env=dict(env), process_job=process_job)
        ledger._check()
        if (type(observed) is not SupervisionResult or type(observed.pid) is not int or observed.pid <= 0
                or type(observed.exit_code) is not int or observed.exit_code != 0
                or observed.clean is not True or observed.timed_out is not False
                or os.path.lexists(ledger.custody / "cpu-study.unsafe.json")
                or os.path.lexists(str(ledger.custody / "cpu-study.unsafe.json") + ".attestation")):
            raise RuntimeError("fixture owner did not exit cleanly")
        candidate, _ = read_record(path / "owner-candidate.json")
        if (set(candidate) != {"schema", "attempt_id", "pid", "reservation_sha256", "journal_sha256", "forward_count"}
                or candidate["schema"] != "bert-matched-rehearsal-candidate-v1"
                or candidate["attempt_id"] != attempt_id or type(candidate["pid"]) is not int
                or candidate["pid"] != observed.pid or candidate["reservation_sha256"] != reservation_sha
                or type(candidate["journal_sha256"]) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", candidate["journal_sha256"])
                or type(candidate["forward_count"]) is not int or candidate["forward_count"] != 0):
            raise ValueError("fixture owner candidate does not match the attempt")
        digest = _closed_journal(path / "events.jsonl", candidate["journal_sha256"])
        if read_record(path / "reservation.json")[1] != reservation_sha:
            raise RuntimeError("reservation changed before rehearsal readback")
        result = {"schema": "bert-matched-rehearsal-result-v1", "status": "rehearsal_complete",
                  "attempt_id": attempt_id, "reservation_sha256": reservation_sha,
                  "journal_sha256": digest, "forward_count": 0, "child_reaped": True,
                  "scope": "cooperative fabricated fixture; child execution is not independently attested"}
        write_record(path / "rehearsal-result.json", result)
        return result
    except BaseException as primary:
        try:
            write_record(path / "rehearsal-failure.json", {
                "schema": "bert-matched-rehearsal-failure-v1", "attempt_id": attempt_id,
                "status": "incomplete", "reservation_sha256": reservation_sha,
                "possibly_exposed": True,
            })
        except BaseException as cleanup:
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary.with_traceback(primary.__traceback__) from cleanup
        raise
