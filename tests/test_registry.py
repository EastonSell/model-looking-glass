import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


def race_terminal(path, start, messages, outcome):
    from looking_glass.registry import open_run

    run = open_run(path)
    start.wait()
    try:
        run.finalize(outcome, {"writer": outcome})
        messages.put("won")
    except RuntimeError:
        messages.put("lost")


def race_artifact(path, start, messages, index):
    from looking_glass.registry import open_run

    run = open_run(path)
    start.wait()
    data = bytes([index]) * 16
    try:
        run.write_artifact("shared", data)
        messages.put(("won", index))
    except FileExistsError:
        messages.put(("lost", index))


class RegistryTests(unittest.TestCase):
    def make_run(self, path, **changes):
        from looking_glass.registry import create_run

        arguments = dict(
            protocol={"id": "example-v1"}, inputs={"sha256": "inputs"},
            model={"sha256": "model"}, code={"sha256": "code"}, metadata={"seed": 9},
            artifact_limits={"predictions.json": 256, "results.json": 512},
            required_artifacts=["predictions.json", "results.json"], max_total_bytes=768,
        )
        arguments.update(changes)
        return create_run(path, **arguments)

    def test_short_pending_writes_never_publish_truncated_bytes(self):
        original_open = Path.open
        class ShortWriter:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                self.stream.__enter__()
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def write(self, data):
                return self.stream.write(data[:2])
            def flush(self):
                return self.stream.flush()
            def fileno(self):
                return self.stream.fileno()
        def open_short(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return ShortWriter(stream) if path.name == ".pending" and args[0] == "xb" else stream
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run")
            payload = b"complete predictions"
            with patch.object(Path, "open", new=open_short):
                run.write_artifact("predictions.json", payload)
                run.write_artifact("results.json", b"complete results")
                run.finalize("succeeded", {"reason": "complete"})
            self.assertEqual((run.path / "artifacts" / "predictions.json").read_bytes(), payload)
            self.assertEqual(run.status()["state"], "succeeded")

    def test_no_progress_in_pending_write_does_not_publish(self):
        original_open = Path.open
        class NoProgress:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                self.stream.__enter__()
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def write(self, data):
                return 0
        def open_stalled(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return NoProgress(stream) if path.name == ".pending" and args[0] == "xb" else stream
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run")
            with patch.object(Path, "open", new=open_stalled):
                with self.assertRaises(OSError):
                    run.write_artifact("predictions.json", b"must not truncate")
            self.assertFalse((run.path / "artifacts" / "predictions.json").exists())
            self.assertEqual(run.status()["state"], "incomplete")

    def test_manifest_records_canonical_identities_and_never_reuses_output(self):
        from looking_glass.registry import create_run

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run = create_run(
                output,
                protocol={"version": 1, "name": "protocol-v1"},
                inputs={"sha256": "input-hash"},
                model={"sha256": "model-hash"},
                code={"sha256": "code-hash"},
                metadata={"seed": 17},
                artifact_limits={"predictions.json": 1024},
                required_artifacts=["predictions.json"],
                max_total_bytes=1024,
            )
            manifest = json.loads((output / "manifest.json").read_bytes())
            expected = hashlib.sha256(b'{"name":"protocol-v1","version":1}').hexdigest()
            self.assertEqual(manifest["identities"]["protocol"]["sha256"], expected)
            self.assertEqual(manifest["identities"]["protocol"]["value"]["version"], 1)
            self.assertEqual(manifest["metadata"]["seed"], 17)
            self.assertEqual(run.status()["state"], "incomplete")
            with self.assertRaises(FileExistsError):
                create_run(
                    output, protocol={}, inputs={}, model={}, code={}, metadata={},
                    artifact_limits={}, required_artifacts=[], max_total_bytes=0,
                )
            self.assertEqual(json.loads((output / "manifest.json").read_bytes()), manifest)

    def test_required_artifacts_and_terminal_outcome_are_immutable(self):
        from looking_glass.registry import verify_run

        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run")
            predictions = b'{"predicted_effect":-0.5}'
            results = b'{"observed_effect":-0.4}'
            run.write_artifact("predictions.json", predictions)
            with self.assertRaises(ValueError):
                run.finalize("succeeded", {})
            self.assertEqual(run.status()["state"], "incomplete")
            with self.assertRaises(FileExistsError):
                run.write_artifact("predictions.json", b"replacement")
            run.write_artifact("results.json", results)
            run.finalize("succeeded", {"reason": "complete"})
            outcome = verify_run(run.path)["outcome"]
            self.assertEqual(outcome["state"], "succeeded")
            self.assertEqual(outcome["artifacts"]["predictions.json"]["sha256"],
                             hashlib.sha256(predictions).hexdigest())
            self.assertEqual(outcome["artifacts"]["results.json"]["bytes"], len(results))
            with self.assertRaises(RuntimeError):
                run.finalize("failed", {})
            with self.assertRaises(RuntimeError):
                run.write_artifact("results.json", b"replacement")
            self.assertEqual((run.path / "artifacts" / "predictions.json").read_bytes(), predictions)

    def test_failure_and_cancellation_retain_partial_artifacts(self):
        from looking_glass.registry import verify_run

        for terminal in ("failed", "cancelled"):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as directory:
                run = self.make_run(Path(directory) / "run")
                run.write_artifact("predictions.json", b"frozen")
                run.finalize(terminal, {"reason": "injected"})
                outcome = verify_run(run.path)["outcome"]
                self.assertEqual(outcome["state"], terminal)
                self.assertEqual(outcome["details"]["reason"], "injected")
                self.assertEqual(set(outcome["artifacts"]), {"predictions.json"})

    def test_budgets_and_windows_unsafe_names_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ("../escape", "sub/file", "sub\\file", "CON", "nul.txt",
                         "report.", "trailing ", "", ".", "a:b"):
                with self.subTest(name=name):
                    path = base / "invalid"
                    with self.assertRaises(ValueError):
                        self.make_run(path, artifact_limits={name: 1}, required_artifacts=[],
                                      max_total_bytes=1)
                    self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                self.make_run(base / "alias", artifact_limits={"A.json": 1, "a.json": 1},
                              required_artifacts=[], max_total_bytes=2)
            with self.assertRaises(ValueError):
                self.make_run(base / "budget", artifact_limits={"a": 6, "b": 5},
                              required_artifacts=[], max_total_bytes=10)
            run = self.make_run(base / "valid", artifact_limits={"one.bin": 3},
                                required_artifacts=["one.bin"], max_total_bytes=3)
            with self.assertRaises(ValueError):
                run.write_artifact("one.bin", b"four")
            self.assertFalse((run.path / "artifacts" / "one.bin").exists())
            with self.assertRaises(KeyError):
                run.write_artifact("other.bin", b"x")
            run.write_artifact("one.bin", b"123")

    def test_failed_atomic_publication_is_incomplete_and_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run")
            with patch("looking_glass.registry.os.link", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    run.write_artifact("predictions.json", b"frozen")
            self.assertFalse((run.path / "artifacts" / "predictions.json").exists())
            self.assertEqual(run.status()["state"], "incomplete")
            run.write_artifact("predictions.json", b"frozen")

    def test_published_terminal_outcome_survives_pending_cleanup_error(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run", required_artifacts=[])
            original_unlink = Path.unlink

            def fail_after_outcome(path, *args, **kwargs):
                if path.name == ".pending" and (run.path / "outcome.json").exists():
                    raise PermissionError("injected cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", new=fail_after_outcome):
                run.finalize("succeeded", {"completed": True})
                result = run.status()
                self.assertEqual(result["state"], "succeeded")
                self.assertTrue(result["pending_write"])
                self.assertEqual(result["outcome"]["details"], {"completed": True})
            with self.assertRaisesRegex(RuntimeError, "terminal"):
                run.finalize("failed", {})
            self.assertFalse(run.status()["pending_write"])
            self.assertEqual(run.status()["state"], "succeeded")

    def test_manifest_and_artifact_publication_survive_pending_cleanup_error(self):
        for kind in ("manifest", "artifact"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run"
                target = (path / "manifest.json" if kind == "manifest" else
                          path / "artifacts" / "predictions.json")
                run = None if kind == "manifest" else self.make_run(path)
                original_unlink = Path.unlink

                def fail_after_publication(candidate, *args, **kwargs):
                    if candidate.name == ".pending" and target.exists():
                        raise PermissionError("injected cleanup failure")
                    return original_unlink(candidate, *args, **kwargs)

                with patch.object(Path, "unlink", new=fail_after_publication):
                    if kind == "manifest":
                        run = self.make_run(path)
                    else:
                        run.write_artifact("predictions.json", b"frozen")
                    self.assertTrue(run.status()["pending_write"])
                    self.assertEqual(run.status()["state"], "incomplete")
                if kind == "manifest":
                    run.write_artifact("predictions.json", b"frozen")
                else:
                    with self.assertRaises(FileExistsError):
                        run.write_artifact("predictions.json", b"replacement")
                self.assertFalse(run.status()["pending_write"])
                self.assertEqual((path / "artifacts" / "predictions.json").read_bytes(), b"frozen")

    def test_cleanup_failure_does_not_mask_failed_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run")
            original_unlink = Path.unlink
            publication_error = OSError("injected link failure")
            cleanup_error = PermissionError("injected cleanup failure")

            def fail_if_pending_exists(path, *args, **kwargs):
                if path.name == ".pending" and path.exists():
                    raise cleanup_error
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", new=fail_if_pending_exists), \
                    patch("looking_glass.registry.os.link", side_effect=publication_error):
                with self.assertRaises(OSError) as caught:
                    run.write_artifact("predictions.json", b"frozen")
            self.assertIs(caught.exception, publication_error)
            self.assertIs(caught.exception.__cause__, cleanup_error)
            self.assertEqual(run.status()["state"], "incomplete")
            self.assertTrue(run.status()["pending_write"])
            run.write_artifact("predictions.json", b"frozen")
            self.assertFalse(run.status()["pending_write"])

    def test_tampered_artifact_or_manifest_fails_verification(self):
        from looking_glass.registry import RegistryIntegrityError, verify_run

        for file_name in ("artifacts/predictions.json", "manifest.json"):
            with self.subTest(file=file_name), tempfile.TemporaryDirectory() as directory:
                run = self.make_run(Path(directory) / "run")
                run.write_artifact("predictions.json", b"first")
                run.finalize("failed", {"reason": "test"})
                (run.path / file_name).write_bytes(b"modified")
                with self.assertRaises(RegistryIntegrityError):
                    verify_run(run.path)

    def test_terminal_integer_fields_reject_boolean_and_float_aliases(self):
        from looking_glass.registry import RegistryIntegrityError, verify_run

        for field in ("schema_version", "bytes"):
            for invalid in (True, 1.0):
                with self.subTest(field=field, invalid=invalid), tempfile.TemporaryDirectory() as directory:
                    run = self.make_run(Path(directory) / "run", artifact_limits={"a": 1},
                                        required_artifacts=["a"], max_total_bytes=1)
                    run.write_artifact("a", b"x")
                    run.finalize("succeeded", {})
                    outcome_path = run.path / "outcome.json"
                    outcome = json.loads(outcome_path.read_bytes())
                    if field == "schema_version":
                        outcome[field] = invalid
                    else:
                        outcome["artifacts"]["a"][field] = invalid
                    outcome_path.write_bytes(json.dumps(outcome, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8"))
                    with self.assertRaises(RegistryIntegrityError):
                        verify_run(run.path)

    def test_deleted_manifest_of_terminal_run_is_corruption_not_uninitialized(self):
        from looking_glass.registry import RegistryIntegrityError, verify_run

        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run")
            run.finalize("cancelled", {"reason": "test"})
            (run.path / "manifest.json").unlink()
            with self.assertRaises(RegistryIntegrityError):
                verify_run(run.path)

    def test_uninitialized_directory_cannot_be_opened_as_an_experiment(self):
        from looking_glass.registry import open_run, verify_run

        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(verify_run(directory)["state"], "uninitialized")
            with self.assertRaises(ValueError):
                open_run(directory)

    def test_verification_serializes_with_in_progress_initialization(self):
        import looking_glass.registry as registry

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.make_run(base / "source")
            raw_manifest = (source.path / "manifest.json").read_bytes()
            path = base / "initializing"
            path.mkdir()
            (path / "artifacts").mkdir()
            (path / ".lock").write_bytes(b"\0")
            checked, resume, finished = threading.Event(), threading.Event(), threading.Event()
            statuses, errors = [], []
            original_exists = Path.exists

            def paused_exists(candidate):
                result = original_exists(candidate)
                if (threading.current_thread().name == "registry-reader" and
                        candidate == path / "manifest.json" and not result):
                    checked.set()
                    if not resume.wait(5):
                        raise TimeoutError("reader was not resumed")
                return result

            def inspect():
                try:
                    statuses.append(registry.verify_run(path)["state"])
                except Exception as exc:
                    errors.append(exc)

            def publish():
                try:
                    with registry._lock(path, 5):
                        registry._publish(path, path / "manifest.json", raw_manifest)
                        registry._publish(path, path / "artifacts" / "predictions.json", b"frozen")
                except Exception as exc:
                    errors.append(exc)
                finally:
                    finished.set()

            with patch.object(Path, "exists", new=paused_exists):
                reader = threading.Thread(target=inspect, name="registry-reader")
                reader.start()
                self.assertTrue(checked.wait(5))
                writer = threading.Thread(target=publish)
                writer.start()
                finished.wait(0.4)
                resume.set()
                reader.join(5)
                writer.join(5)
            self.assertFalse(reader.is_alive())
            self.assertFalse(writer.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(statuses, ["uninitialized"])
            self.assertEqual(registry.verify_run(path)["state"], "incomplete")

    @unittest.skipUnless(os.name == "nt", "Windows byte-range lock semantics")
    def test_initializing_lock_file_does_not_write_through_another_process_lock(self):
        import looking_glass.registry as registry

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            visible, resume_creator, locked, release_reader = (threading.Event() for _ in range(4))
            creator_done = threading.Event()
            errors = []
            original_open = Path.open

            def paused_open(candidate, *args, **kwargs):
                stream = original_open(candidate, *args, **kwargs)
                mode = args[0] if args else kwargs.get("mode", "r")
                if candidate == path / ".lock" and mode == "xb":
                    visible.set()
                    if not resume_creator.wait(5):
                        stream.close()
                        raise TimeoutError("creator was not resumed")
                return stream

            def create():
                try:
                    self.make_run(path)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    creator_done.set()

            def hold():
                try:
                    with registry._lock(path, 5):
                        locked.set()
                        if not release_reader.wait(5):
                            raise TimeoutError("reader was not released")
                except Exception as exc:
                    errors.append(exc)

            with patch.object(Path, "open", new=paused_open):
                creator = threading.Thread(target=create)
                creator.start()
                self.assertTrue(visible.wait(5))
                reader = threading.Thread(target=hold)
                reader.start()
                self.assertTrue(locked.wait(5))
                resume_creator.set()
                creator_done.wait(0.4)
                release_reader.set()
                creator.join(5)
                reader.join(5)
            self.assertFalse(creator.is_alive())
            self.assertFalse(reader.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(registry.verify_run(path)["state"], "incomplete")

    def test_relative_run_path_stays_bound_after_working_directory_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                run = self.make_run("run")
                elsewhere = Path(directory) / "elsewhere"
                elsewhere.mkdir()
                os.chdir(elsewhere)
                run.write_artifact("predictions.json", b"frozen")
                self.assertEqual((Path(directory) / "run" / "artifacts" / "predictions.json").read_bytes(),
                                 b"frozen")
            finally:
                os.chdir(previous)

    def test_stale_pending_file_is_removed_before_the_next_write(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run")
            (run.path / ".pending").write_bytes(b"interrupted")
            self.assertTrue(run.status()["pending_write"])
            run.write_artifact("predictions.json", b"correct")
            self.assertFalse((run.path / ".pending").exists())
            self.assertEqual((run.path / "artifacts" / "predictions.json").read_bytes(), b"correct")

    def test_killed_writer_before_and_after_publication_preserves_outcome(self):
        from looking_glass.registry import verify_run

        script = """
import os
import sys
import looking_glass.registry as registry
run = registry.open_run(sys.argv[1])
link = registry.os.link
def crash(src, dst):
    if sys.argv[2] == 'after':
        link(src, dst)
    os._exit(19)
registry.os.link = crash
run.write_artifact('shared', b'12345678')
"""
        for phase in ("before", "after"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                run = self.make_run(Path(directory) / "run", artifact_limits={"shared": 8},
                                    required_artifacts=["shared"], max_total_bytes=8)
                completed = subprocess.run([sys.executable, "-c", script, str(run.path), phase],
                                           timeout=15, check=False)
                self.assertEqual(completed.returncode, 19)
                self.assertTrue(verify_run(run.path)["pending_write"])
                if phase == "after":
                    with self.assertRaises(FileExistsError):
                        run.write_artifact("shared", b"other")
                else:
                    run.write_artifact("shared", b"12345678")
                run.finalize("succeeded", {})
                result = verify_run(run.path)
                self.assertFalse(result["pending_write"])
                self.assertEqual((run.path / "artifacts" / "shared").read_bytes(), b"12345678")

    def test_competing_processes_publish_same_artifact_only_once(self):
        from looking_glass.registry import verify_run

        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run", artifact_limits={"shared": 16},
                                required_artifacts=["shared"], max_total_bytes=16)
            context = multiprocessing.get_context("spawn")
            start, messages = context.Event(), context.Queue()
            processes = [context.Process(target=race_artifact,
                         args=(run.path, start, messages, index)) for index in range(6)]
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(15)
                self.assertEqual(process.exitcode, 0)
            results = [messages.get(timeout=3) for _ in processes]
            winners = [index for state, index in results if state == "won"]
            self.assertEqual(len(winners), 1)
            self.assertEqual(sum(state == "lost" for state, _ in results), 5)
            payload = bytes([winners[0]]) * 16
            self.assertEqual((run.path / "artifacts" / "shared").read_bytes(), payload)
            run.finalize("succeeded", {})
            self.assertEqual(verify_run(run.path)["outcome"]["artifacts"]["shared"]["sha256"],
                             hashlib.sha256(payload).hexdigest())

    def test_competing_processes_commit_exactly_one_terminal_state(self):
        from looking_glass.registry import verify_run

        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run", artifact_limits={},
                                required_artifacts=[], max_total_bytes=0)
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            messages = context.Queue()
            processes = [context.Process(target=race_terminal,
                         args=(run.path, start, messages, state))
                         for state in ("failed", "cancelled")]
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(15)
                self.assertEqual(process.exitcode, 0)
            self.assertCountEqual([messages.get(timeout=3), messages.get(timeout=3)],
                                  ["won", "lost"])
            self.assertIn(verify_run(run.path)["state"], ("failed", "cancelled"))

    def test_writer_and_terminal_race_cannot_publish_an_unrecorded_artifact(self):
        from looking_glass.registry import verify_run

        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory) / "run", required_artifacts=[])
            gate = threading.Barrier(2)
            errors = []

            def writer():
                gate.wait()
                try:
                    run.write_artifact("predictions.json", b"frozen")
                except RuntimeError:
                    pass
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=writer)
            thread.start()
            gate.wait()
            run.finalize("cancelled", {})
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            verified = verify_run(run.path)
            files = {p.name for p in (run.path / "artifacts").iterdir()}
            self.assertEqual(files, set(verified["outcome"]["artifacts"]))


if __name__ == "__main__":
    unittest.main()
