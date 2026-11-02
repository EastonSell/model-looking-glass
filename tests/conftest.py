"""Explicit boundary for CPython's undefined opcode-exception experiments.

These tests are evidence-producing experiments, not supported-runtime guarantees.
https://docs.python.org/3.12/reference/datamodel.html#frame.f_trace_opcodes
"""
import ast
import json
import os
import math
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest
from _pytest.reports import TestReport

_MANIFEST = Path(__file__).with_name("experimental_opcode_faults.json")
_CHILD_NODE = "MLG_EXPERIMENTAL_OPCODE_CHILD_NODE"
_CHILD_REPORT = "MLG_EXPERIMENTAL_OPCODE_CHILD_REPORT"
_REASON = "experimental opcode-exception probe: CPython documents undefined behavior; opt in with --run-experimental-opcode-faults"


def pytest_addoption(parser):
    parser.addoption("--run-experimental-opcode-faults", action="store_true", default=False,
                     help="Run exact manifest-listed undefined opcode-exception probes in fresh bounded children")
    parser.addoption("--experimental-opcode-timeout", type=float, default=60.0,
                     help="Per-experiment child timeout in seconds (default: 60)")


def _load_manifest(root, path=_MANIFEST):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("nodes"), list):
        raise ValueError("invalid experimental opcode manifest schema")
    nodes = data["nodes"]
    if any(not isinstance(node, str) for node in nodes) or len(set(nodes)) != len(nodes):
        raise ValueError("experimental opcode manifest contains non-string or duplicate nodes")
    for node in nodes:
        parts = node.split("::")
        source = (root / parts[0]).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file() or len(parts) not in (2, 3):
            raise ValueError(f"unknown experimental node: {node}")
        body = ast.parse(source.read_text(encoding="utf-8")).body
        for name in parts[1:]:
            name = name.split("[", 1)[0]
            matches = [entry for entry in body if isinstance(entry, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and entry.name == name]
            if len(matches) != 1:
                raise ValueError(f"unknown experimental node: {node}")
            body = matches[0].body
    return frozenset(nodes)


def pytest_configure(config):
    config.addinivalue_line("markers", "experimental_opcode_fault: undefined CPython opcode-exception experiment")
    try:
        config._mlg_opcode_nodes = _load_manifest(Path(config.rootpath))
    except (ValueError, OSError, SyntaxError) as error:
        raise pytest.UsageError(str(error)) from error
    timeout = config.getoption("--experimental-opcode-timeout")
    if not 0 < timeout < float("inf"):
        raise pytest.UsageError("experimental opcode timeout must be finite and positive")
    child = os.environ.get(_CHILD_NODE)
    if child and (not config.getoption("--run-experimental-opcode-faults") or child not in config._mlg_opcode_nodes or not os.environ.get(_CHILD_REPORT)):
        raise pytest.UsageError("invalid experimental child selection")
    config._mlg_opcode_selected = 0


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    nodes = config._mlg_opcode_nodes
    collected = {item.nodeid for item in items}
    files = {node.split("::")[0] for node in collected}
    # Validate exact parameter IDs whenever their module is collected, before
    # pytest's -k/-m selection. Other modules are checked structurally above.
    full_files = set()
    for argument in config.args:
        if "::" in argument:
            continue
        selected_path = (Path(config.rootpath) / argument).resolve()
        for filename in files:
            file_path = (Path(config.rootpath) / filename).resolve()
            if selected_path == file_path or (selected_path.is_dir() and file_path.is_relative_to(selected_path)):
                full_files.add(filename)
    stale = [node for node in nodes if node.split("::")[0] in full_files and node not in collected]
    if stale:
        raise pytest.UsageError("unknown experimental node IDs: " + ", ".join(sorted(stale)))
    child = os.environ.get(_CHILD_NODE)
    if child and collected != {child}:
        raise pytest.UsageError("experimental child must collect exactly its assigned node")
    for item in items:
        if item.nodeid in nodes:
            item.add_marker("experimental_opcode_fault")
            if not config.getoption("--run-experimental-opcode-faults") and not child:
                item.add_marker(pytest.mark.skip(reason=_REASON))


def pytest_collection_finish(session):
    session.config._mlg_opcode_selected = sum(item.nodeid in session.config._mlg_opcode_nodes for item in session.items)


def _child_result(item):
    with tempfile.TemporaryDirectory(prefix="mlg-opcode-child-") as temporary:
        report = Path(temporary) / "reports.jsonl"
        env = os.environ.copy()
        env[_CHILD_NODE] = item.nodeid
        env[_CHILD_REPORT] = str(report)
        # Parent selection options must not alter the exact child selection.
        env.pop("PYTEST_ADDOPTS", None)
        command = [sys.executable, "-B", "-X", "faulthandler", "-m", "pytest", item.nodeid,
                   "--run-experimental-opcode-faults", "-o", "addopts=", "-p", "no:cacheprovider", "-q"]
        # Own only this fresh process group, never unrelated processes. POSIX
        # timeout cleanup includes descendants retaining our output pipes.
        # Other platforms guarantee direct-child cleanup only.
        process = subprocess.Popen(command, cwd=item.config.rootpath, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=(os.name == "posix"))
        try:
            stdout, stderr = process.communicate(timeout=item.config.getoption("--experimental-opcode-timeout"))
        except subprocess.TimeoutExpired as error:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=5)
            # Do not wait indefinitely for descendants outside the owned group.
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            return "failed", f"Experimental child timed out: {error}", 0.0
        output = stdout + stderr
        if process.returncode != 0:
            return "failed", f"Experimental child exited {process.returncode}\n{output}", 0.0
        try:
            records = [json.loads(line) for line in report.read_text().splitlines()]
        except (OSError, ValueError) as error:
            return "failed", f"Experimental child report missing/invalid: {error}\n{output}", 0.0
        valid = all(
            isinstance(row, dict)
            and set(row) == {"when", "outcome", "reason", "duration"}
            and row["when"] in ("setup", "call", "teardown")
            and row["outcome"] in ("passed", "failed", "skipped")
            and isinstance(row["reason"], str)
            and type(row["duration"]) in (int, float)
            and math.isfinite(row["duration"]) and row["duration"] >= 0
            for row in records
        )
        if not valid:
            return "failed", f"Malformed experimental child lifecycle: {records}\n{output}", 0.0
        phases = [(row["when"], row["outcome"]) for row in records]
        complete = (
            [phase for phase, _ in phases] == ["setup", "call", "teardown"]
            and phases[0][1] == "passed"
            and all(outcome in ("passed", "skipped") for _, outcome in phases)
        ) or (
            [phase for phase, _ in phases] == ["setup", "teardown"]
            and phases[0][1] == "skipped"
            and phases[1][1] in ("passed", "skipped")
        )
        if not complete:
            return "failed", f"Incomplete or failed experimental child lifecycle: {records}\n{output}", 0.0
        skipped = [row for row in records if row["outcome"] == "skipped"]
        if skipped:
            return "skipped", skipped[0]["reason"], sum(row["duration"] for row in records)
        return "passed", None, sum(row["duration"] for row in records)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if item.nodeid not in item.config._mlg_opcode_nodes or not item.config.getoption("--run-experimental-opcode-faults") or os.environ.get(_CHILD_NODE):
        return None
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    started = time.time()
    outcome, detail, duration = _child_result(item)
    if outcome == "skipped":
        detail = (str(item.path), item.location[1], str(detail))
    for when in ("setup", "call", "teardown"):
        report = TestReport(nodeid=item.nodeid, location=item.location, keywords=dict(item.keywords),
                            outcome=outcome if when == "call" else "passed",
                            longrepr=detail if when == "call" else None, when=when,
                            duration=duration if when == "call" else 0.0,
                            start=started, stop=time.time())
        item.ihook.pytest_runtest_logreport(report=report)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def pytest_runtest_logreport(report):
    if os.environ.get(_CHILD_NODE) == report.nodeid:
        with open(os.environ[_CHILD_REPORT], "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"when": report.when, "outcome": report.outcome,
                                     "reason": str(report.longrepr), "duration": report.duration}) + "\n")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    selected = config._mlg_opcode_selected
    mode = "fresh-child opt-in" if config.getoption("--run-experimental-opcode-faults") else "excluded by default"
    terminalreporter.write_line(f"Opcode-exception experiments: {selected} selected; {mode}. Undefined interpreter behavior is not a supported safety guarantee.")
