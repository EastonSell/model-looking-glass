"""Offline archive integrity tests; no torch or model execution required."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_results.py"
SPEC = importlib.util.spec_from_file_location("archive_results", SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class ArchiveResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.payloads = {"results/a/data.bin": b"\x00\xffpayload\x00",
                         "results/b/empty.f32": b"",
                         "results/a2/data.bin": b"another"}
        for name, contents in self.payloads.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        self.git("add", "results")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "evidence")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.data = {"base_commit": self.commit, "files": [
            {"path": name, "blob_sha": self.git("rev-parse", f"HEAD:{name}").strip(),
             "size": len(contents)} for name, contents in self.payloads.items()]}
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(self.data))
        self.output = self.root / "restored"

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()

    def test_roundtrip_pinned_tree_not_working_tree(self):
        (self.repo / "results/a/data.bin").write_bytes(b"modified")
        data = archive.load_manifest(self.manifest)
        rows = archive.select_files(data, all_files=True)
        result = archive.restore(data, rows, self.repo, self.output)
        self.assertEqual(result["status"], "complete")
        for name, payload in self.payloads.items():
            self.assertEqual((self.output / name).read_bytes(), payload)
        self.assertEqual(json.loads((self.output / "archive-restore-status.json").read_text()), result)

    def test_selection_is_explicit_and_directory_bounded(self):
        with self.assertRaises(ValueError):
            archive.select_files(self.data)
        rows = archive.select_files(self.data, prefixes=["results/a/"])
        self.assertEqual([r["path"] for r in rows], ["results/a/data.bin"])
        self.assertEqual(len(archive.select_files(self.data, paths=["results/b/empty.f32"])), 1)
        for kwargs in ({"paths": ["results/missing"]}, {"prefixes": ["results/absent"]},
                       {"paths": ["results/a/data.bin"], "all_files": True}):
            with self.assertRaises(ValueError):
                archive.select_files(self.data, **kwargs)

    def test_nonoverwrite_existing_directory(self):
        self.output.mkdir()
        (self.output / "keep").write_text("untouched")
        with self.assertRaises(FileExistsError):
            archive.restore(self.data, self.data["files"], self.repo, self.output)
        self.assertEqual((self.output / "keep").read_text(), "untouched")

    def test_output_symlink_and_symlink_ancestor_rejected(self):
        real = self.root / "real"
        real.mkdir()
        link = self.root / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation unavailable")
        for output in (link, link / "new"):
            with self.assertRaises(ValueError):
                archive.restore(self.data, self.data["files"], self.repo, output)
        self.assertEqual(list(real.iterdir()), [])

    def test_manifest_paths_rejected(self):
        for path in ("/results/a", "results/../x", "results//x", "results/./x",
                     "results\\x", "C:/results/x", "outside/x", "results/NUL.bin",
                     "results/x.", "results/x\n", "results"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                archive.safe_path(path)

    def test_manifest_schema_duplicates_and_counts(self):
        invalid = [[], {}, {**self.data, "base_commit": "main"},
                   {**self.data, "count": True}, {**self.data, "total_bytes": -1},
                   {**self.data, "files": self.data["files"] * 2},
                   {**self.data, "files": [{**self.data["files"][0], "size": True}]},
                   {**self.data, "files": [{**self.data["files"][0], "blob_sha": "bad"}]},
                   {**self.data, "files": [{**self.data["files"][0], "extra": 1}]}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.manifest.write_text(json.dumps(value))
                archive.load_manifest(self.manifest)
        self.manifest.write_text('{"base_commit":"a","base_commit":"b"}')
        with self.assertRaises(ValueError):
            archive.load_manifest(self.manifest)

    def test_file_directory_collision_rejected(self):
        rows = [dict(self.data["files"][0]), dict(self.data["files"][1])]
        rows[1]["path"] = rows[0]["path"] + "/child"
        self.manifest.write_text(json.dumps({**self.data, "files": rows}))
        with self.assertRaises(ValueError):
            archive.load_manifest(self.manifest)

    def test_tree_sha_mismatch_fails_before_output_created(self):
        row = {**self.data["files"][0], "blob_sha": self.data["files"][1]["blob_sha"]}
        with self.assertRaises(ValueError):
            archive.restore(self.data, [row], self.repo, self.output)
        self.assertFalse(self.output.exists())

    def test_tree_object_is_not_accepted_as_base_commit(self):
        data = {**self.data, "base_commit": self.git("rev-parse", "HEAD^{tree}").strip()}
        with self.assertRaises(subprocess.CalledProcessError):
            archive.restore(data, data["files"], self.repo, self.output)
        self.assertFalse(self.output.exists())

    def test_symlink_in_git_tree_is_rejected(self):
        link = self.repo / "results/link.bin"
        try:
            link.symlink_to("a/data.bin")
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation unavailable")
        self.git("add", "results/link.bin")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "link")
        data = {"base_commit": self.git("rev-parse", "HEAD").strip(), "files": [
            {"path": "results/link.bin", "blob_sha": self.git("rev-parse", "HEAD:results/link.bin").strip(),
             "size": len("a/data.bin")}]}
        with self.assertRaises(ValueError):
            archive.restore(data, data["files"], self.repo, self.output)
        self.assertFalse(self.output.exists())

    def test_corruption_is_not_exposed_and_status_is_incomplete(self):
        def corrupted(repo, sha, path, size):
            path.write_bytes(b"X" * size)
        with mock.patch.object(archive, "_write_git_blob", side_effect=corrupted):
            with self.assertRaises(ValueError):
                archive.restore(self.data, [self.data["files"][0]], self.repo, self.output)
        self.assertFalse((self.output / self.data["files"][0]["path"]).exists())
        status = json.loads((self.output / "archive-restore-status.json").read_text())
        self.assertEqual(status["status"], "incomplete")
        self.assertEqual(status["verified_paths"], [])
        self.assertIn("integrity", status["error"])
        self.assertFalse(list(self.output.rglob(".archive-*")))

    def test_wrong_size_preserves_already_verified_files(self):
        rows = [self.data["files"][1], {**self.data["files"][0], "size": 1}]
        with self.assertRaises(ValueError):
            archive.restore(self.data, rows, self.repo, self.output)
        self.assertEqual((self.output / rows[0]["path"]).read_bytes(), b"")
        status = json.loads((self.output / "archive-restore-status.json").read_text())
        self.assertEqual(status["verified_paths"], [rows[0]["path"]])
        self.assertEqual(status["status"], "incomplete")

    def test_truncated_blob_rejected(self):
        def truncated(repo, sha, path, size):
            path.write_bytes(b"")
        with mock.patch.object(archive, "_write_git_blob", side_effect=truncated):
            with self.assertRaises(ValueError):
                archive.restore(self.data, [self.data["files"][0]], self.repo, self.output)

    def test_local_verification_no_git_and_detects_changed_bytes(self):
        rows = [self.data["files"][0]]
        archive.restore(self.data, rows, self.repo, self.output)
        with mock.patch.object(archive.subprocess, "run", side_effect=AssertionError("no Git")), \
             mock.patch.object(archive.subprocess, "Popen", side_effect=AssertionError("no Git")):
            archive.verify_local(rows, self.output)
            (self.output / rows[0]["path"]).write_bytes(b"bad")
            with self.assertRaises(ValueError):
                archive.verify_local(rows, self.output)

    def test_local_verification_rejects_missing_file(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            archive.verify_local([self.data["files"][0]], self.output)

    def test_atomic_publication_does_not_overwrite_competing_file(self):
        original_link = os.link
        def racing_link(source, destination):
            Path(destination).write_bytes(b"other writer")
            return original_link(source, destination)
        row = self.data["files"][0]
        with mock.patch.object(archive.os, "link", side_effect=racing_link):
            with self.assertRaises(FileExistsError):
                archive.restore(self.data, [row], self.repo, self.output)
        self.assertEqual((self.output / row["path"]).read_bytes(), b"other writer")
        status = json.loads((self.output / "archive-restore-status.json").read_text())
        self.assertEqual(status["status"], "incomplete")
        self.assertEqual(status["verified_paths"], [])

    def test_status_write_failure_preserves_original_restore_error(self):
        original_write = Path.write_text
        writes = []
        def status_write(path, *args, **kwargs):
            if path.name == "archive-restore-status.json":
                writes.append(path)
                if len(writes) > 1:
                    raise OSError("status disk failure")
            return original_write(path, *args, **kwargs)
        with mock.patch.object(Path, "write_text", status_write), \
             mock.patch.object(archive, "copy_blob", side_effect=ValueError("original blob error")):
            with self.assertRaisesRegex(ValueError, "original blob error") as caught:
                archive.restore(self.data, [self.data["files"][0]], self.repo, self.output)
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_cli_list_and_restore(self):
        base = ["--manifest", str(self.manifest), "--repo", str(self.repo)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(archive.main(["list", *base]), 0)
            self.assertEqual(archive.main(["restore", *base, "--output", str(self.output)]), 1)
            self.assertEqual(archive.main(["restore", *base, "--all", "--output", str(self.output)]), 0)
            self.assertEqual(archive.main(["verify", *base, "--all", "--output", str(self.output)]), 0)
            self.assertEqual(archive.main(["restore", *base, "--all", "--output", str(self.output)]), 1)


if __name__ == "__main__":
    unittest.main()
