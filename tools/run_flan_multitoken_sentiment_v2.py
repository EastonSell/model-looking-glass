"""Observe the fixed three-subject FLAN CPU study using direct children.

Run a verified base interpreter with -I -S -B and a fresh empty bytecode
prefix. Each child waits for a one-shot grant before loading project code,
tokenizers or private inputs. The receipts are cooperative direct-child and
saved-byte observations, not execution authentication or descendant proof.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


PATH_OPTIONS = ("repo", "venv", "asset-dir", "attempt", "binding", "discovery", "painting", "selection-review")
SUBJECTS = ("story", "concert", "painting")
LOG_LIMIT = 64 * 1024


def _clear_git_environment():
    for name in tuple(os.environ):
        if name.upper().startswith("GIT_"):
            os.environ.pop(name, None)


def _raise_with_priority(primary, cleanup_error):
    if primary is None:
        raise cleanup_error
    if not isinstance(primary, Exception) or isinstance(cleanup_error, Exception):
        raise primary from cleanup_error
    raise cleanup_error from primary


def _empty_prefix():
    prefix = sys.pycache_prefix
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or not prefix
            or Path(sys.executable).resolve() != Path(sys._base_executable).resolve()):
        raise ValueError("run a direct base interpreter with -I -S -B and a fresh bytecode prefix")
    path = Path(prefix)
    if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
        raise ValueError("bytecode prefix is not fresh and empty")


def _receive_grant(expected):
    if type(expected) is not str or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("invalid child grant digest")
    done = threading.Event()
    def expire():
        if not done.wait(30):
            os._exit(70)
    threading.Thread(target=expire, name="multisubtoken-grant-deadline", daemon=True).start()
    try:
        grant = sys.stdin.buffer.read(33)
        if len(grant) != 32 or hashlib.sha256(grant).hexdigest() != expected:
            raise ValueError("child grant is absent or invalid")
    finally:
        done.set()


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")
    with Path(path).open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short operator record write")
        stream.flush()
        os.fsync(stream.fileno())


def _drain_log(pipe, path, state):
    stream = None
    try:
        try:
            stream = Path(path).open("xb")
        except BaseException as error:
            state["error_type"] = type(error).__name__
        while True:
            block = pipe.read(16384)
            if not block:
                break
            keep = block[:max(0, LOG_LIMIT - state["retained_bytes"])]
            if len(keep) != len(block):
                state["truncated"] = True
            if keep and stream is not None:
                try:
                    if stream.write(keep) != len(keep):
                        raise OSError("short log write")
                    state["retained_bytes"] += len(keep)
                except BaseException as error:
                    state["error_type"] = type(error).__name__
                    stream.close()
                    stream = None
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as error:
        state["error_type"] = type(error).__name__
    finally:
        if stream is not None:
            stream.close()
        pipe.close()
        state["finished"] = True


def _load_owner(root, revision):
    """Verify bootstrap Git blobs without adding the checkout to sys.path."""
    git = ["git", "--no-replace-objects"]
    if (type(revision) is not str or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
            or subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=root).strip().decode("ascii") != revision
            or subprocess.check_output(git + ["cat-file", "-t", revision], cwd=root).strip() != b"commit"):
        raise ValueError("operator checkout revision differs")
    for name in ("tools/run_flan_multitoken_sentiment_v2.py", "looking_glass/__init__.py",
                 "looking_glass/flan_multitoken_owner_v2.py"):
        if (root / name).read_bytes() != subprocess.check_output(git + ["show", revision + ":" + name], cwd=root):
            raise ValueError("operator bootstrap source differs")
    package = root / "looking_glass"
    spec = importlib.util.spec_from_file_location("looking_glass", package / "__init__.py",
                                                 submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ValueError("cannot resolve source-bound package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["looking_glass"] = module
    spec.loader.exec_module(module)
    from looking_glass import flan_multitoken_owner_v2 as owner
    return owner


def _load_runtime_helper(root):
    path = root / "tools/run_flan_sentiment_selector_stage_v1.py"
    spec = importlib.util.spec_from_file_location("_flan_multitoken_direct_runtime", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot resolve verified direct-runtime helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PATH_OPTIONS:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--subject", choices=SUBJECTS, help=argparse.SUPPRESS)
    parser.add_argument("--grant-sha256", help=argparse.SUPPRESS)
    return parser


def _supervise(command, *, args, root, owner, runtime, source, subject, directory, prefix):
    """Run one granted direct child and durably record its observed outcome."""
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "ATEN_CPU_CAPABILITY"):
        environment.pop(name, None)
    environment.update({"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE",
                        "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
    grant = os.urandom(32)
    command = [*command, "--grant-sha256", hashlib.sha256(grant).hexdigest()]
    child, code, timed_out, failure, primary = None, None, False, None, None
    drainers = []
    logs = {name: {"retained_bytes": 0, "truncated": False, "error_type": None, "finished": False}
            for name in ("stdout", "stderr")}
    try:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        factory = subprocess.Popen
        child_options = {"cwd": root, "env": environment, "stdin": subprocess.PIPE,
                         "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, **options}
        if isinstance(factory, type):
            child = factory.__new__(factory)
            factory.__init__(child, command, **child_options)
        else:
            child = factory(command, **child_options)
        for name in ("stdout", "stderr"):
            thread = threading.Thread(target=_drain_log, args=(getattr(child, name), directory / (name + ".log"),
                                                               logs[name]), daemon=True)
            drainers.append(thread)
            thread.start()
        _write_new(directory / "operator-spawn.json", {"pid": child.pid, "parent_pid": os.getpid()})
        if child.stdin.write(grant) != len(grant):
            raise OSError("short child grant write")
        child.stdin.flush()
        child.stdin.close()
        try:
            code = child.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait(timeout=15)
    except BaseException as error:
        primary, failure = error, type(error).__name__
        if child is not None:
            try:
                if child.poll() is None:
                    child.kill()
                    code = child.wait(timeout=15)
            except BaseException as cleanup_error:
                if isinstance(error, Exception) and not isinstance(cleanup_error, Exception):
                    primary, failure = cleanup_error, type(cleanup_error).__name__
                    raise cleanup_error from error
        raise
    finally:
        try:
            for thread in drainers:
                try:
                    thread.join(timeout=15)
                except RuntimeError:
                    pass
            logs_complete = (len(drainers) == 2 and all(not thread.is_alive() for thread in drainers)
                             and all(item["finished"] and not item["truncated"] and item["error_type"] is None
                                     for item in logs.values()))
            terminal, events = directory / "child/terminal.json", directory / "child/events.jsonl"
            owner_receipt = directory / "owner-receipt.json"
            terminal_sha = _hash_file(terminal) if terminal.is_file() else None
            events_sha = _hash_file(events) if events.is_file() else None
            owner_sha = _hash_file(owner_receipt) if owner_receipt.is_file() else None
            consistent, status = False, None
            try:
                record = json.loads(owner_receipt.read_bytes())
                status = record["status"]
                common = (record["subject"] == subject and record["source_revision"] == args.revision
                          and record["binding_sha256"] == args.binding_sha256 and logs_complete
                          and failure is None and owner.source_snapshot(root, args.revision) == source
                          and runtime._check_child_identity(directory / "stdout.log", args.venv, child.pid))
                if status == "complete_candidate":
                    saved_terminal = json.loads(terminal.read_bytes())
                    consistent = (common and record["model_loaded"] is True
                                  and record["terminal_sha256"] == terminal_sha and record["events_sha256"] == events_sha
                                  and saved_terminal["status"] == "complete_candidate")
                elif status == "no-selected-groups":
                    consistent = common and record["model_loaded"] is False and not (directory / "child").exists()
            except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.CalledProcessError):
                pass
            _write_new(directory / "operator-exit.json", {
                "schema": "flan-multisubtoken-operator-exit-v1", "subject": subject,
                "direct_child_exit_code": code, "child_reaped": code is not None, "timed_out": timed_out,
                "operator_exception_type": failure, "owner_status": status, "terminal_consistent": consistent,
                "terminal_sha256": terminal_sha, "events_sha256": events_sha, "owner_receipt_sha256": owner_sha,
                "child_prefix_empty_after": not any(prefix.iterdir()), "logs": logs,
                "log_drain_complete": logs_complete,
                "stdout_sha256": _hash_file(directory / "stdout.log") if (directory / "stdout.log").is_file() else None,
                "stderr_sha256": _hash_file(directory / "stderr.log") if (directory / "stderr.log").is_file() else None})
        except BaseException as cleanup_error:
            _raise_with_priority(primary, cleanup_error)
    if code != 0 or timed_out or not consistent or any(prefix.iterdir()):
        raise RuntimeError("subject pass is incomplete or uncertain; retain its evidence")
    return {"subject": subject, "status": status, "operator_exit_sha256": _hash_file(directory / "operator-exit.json")}


def main(argv=None):
    _empty_prefix()
    _clear_git_environment()
    args = _parser().parse_args(argv)
    root = args.repo.resolve(strict=True)
    if Path(__file__).resolve() != (root / "tools/run_flan_multitoken_sentiment_v2.py").resolve():
        raise ValueError("operator must execute from its bound checkout")
    if not all(getattr(args, name.replace("-", "_")).is_absolute() for name in PATH_OPTIONS):
        raise ValueError("all operator paths must be absolute")
    if not 1 <= args.timeout_seconds <= 86400:
        raise ValueError("invalid timeout")
    if args.child:
        if args.subject is None:
            raise ValueError("child subject is required")
        _receive_grant(args.grant_sha256)
    elif args.grant_sha256 is not None or args.subject is not None:
        raise ValueError("child grant and subject are reserved")
    owner = _load_owner(root, args.revision)
    source = owner.source_snapshot(root, args.revision)
    runtime = _load_runtime_helper(root)
    site, _ = runtime._dependency_site(args.venv)
    if args.child:
        primary = None
        try:
            if str(site) in sys.path:
                raise ValueError("venv site-packages was already on the search path")
            sys.path.append(str(site))
            print(json.dumps(runtime._process_identity(args.venv, os.getpid(), os.getppid()), sort_keys=True), flush=True)
            owner.run(args)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                _empty_prefix()
                if owner.source_snapshot(root, args.revision) != source:
                    raise ValueError("source changed during child execution")
            except BaseException as cleanup_error:
                _raise_with_priority(primary, cleanup_error)
        return
    attempt = args.attempt
    if not attempt.parent.is_dir() or attempt.exists():
        raise ValueError("attempt must be a new directory under an existing parent")
    owner.ensure_output_directory(attempt.parent)
    attempt.mkdir()
    owner.ensure_output_directory(attempt)
    _write_new(attempt / "operator-start.json", {"schema": "flan-multisubtoken-operator-v1",
               "source": source, "binding_sha256": args.binding_sha256, "subjects": list(SUBJECTS),
               "scope": "Local direct-child observation; interpreter and dependencies remain trusted."})
    completed = []
    for subject in SUBJECTS:
        directory = attempt / subject
        directory.mkdir()
        prefix = directory / "child-pycache"
        prefix.mkdir()
        command = [sys.executable, "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
                   str(root / "tools/run_flan_multitoken_sentiment_v2.py"), "--child", "--subject", subject,
                   "--revision", args.revision, "--binding-sha256", args.binding_sha256,
                   "--timeout-seconds", str(args.timeout_seconds)]
        for name in PATH_OPTIONS:
            value = directory if name == "attempt" else getattr(args, name.replace("-", "_"))
            command.extend(["--" + name, str(value)])
        item = _supervise(command, args=args, root=root, owner=owner, runtime=runtime,
                          source=source, subject=subject, directory=directory, prefix=prefix)
        completed.append(item)
        if subject == "story" and item["status"] == "no-selected-groups":
            break
        if subject == "concert" and item["status"] == "no-selected-groups":
            raise RuntimeError("discovery pass selections differ; retain evidence")
    status = "no-selected-groups" if completed[0]["status"] == "no-selected-groups" else "complete_candidate"
    _write_new(attempt / "operator-summary.json", {"schema": "flan-multisubtoken-operator-summary-v1",
               "status": status, "source_revision": args.revision, "binding_sha256": args.binding_sha256,
               "passes": completed, "scope": "Direct-child observations only; scientific replay remains separate."})
    print(json.dumps({"status": status, "passes": completed}, sort_keys=True))


if __name__ == "__main__":
    main()
