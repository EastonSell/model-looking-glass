"""CPU-only tests for canonical GPU lease and quarantine identities."""

import importlib
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from looking_glass.device_lease import DeviceLease


GPU_A = "GPU-12345678-1234-5678-9abc-1234567890ab"
GPU_B = "GPU-abcdef12-3456-789a-bcde-f01234567890"


def _resolve_in_process(root, start, outcomes):
    try:
        api = importlib.import_module("looking_glass.device_identity")
        if not start.wait(10):
            outcomes.put(("error", "start timed out"))
            return
        identity = api.resolve_device_identity(root, GPU_A, physical_uuids=(GPU_A,), environ={})
        outcomes.put(("ok", str(identity.lock_path), str(identity.marker_path), str(identity.controller_path)))
    except BaseException as exc:
        outcomes.put(("error", repr(exc)))


class DeviceIdentityTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("looking_glass.device_identity")
        except ModuleNotFoundError as exc:
            if exc.name == "looking_glass.device_identity":
                self.fail("the device identity module is not implemented")
            raise
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def resolve(self, uuid=GPU_A, *, inventory=(GPU_A,), environ=None, root=None, index=0):
        return self.api.resolve_device_identity(
            self.root if root is None else root,
            uuid,
            physical_uuids=inventory,
            cuda_device_index=index,
            environ={} if environ is None else environ,
        )

    def lease_for(self, identity):
        lease = DeviceLease.acquire(identity.lock_path, timeout=1)
        self.addCleanup(lease.release)
        return lease

    def test_uuid_case_and_root_spelling_resolve_same_bundle(self):
        a = self.resolve()
        b = self.resolve(uuid=GPU_A.upper(), inventory=(GPU_A.upper(),), root=self.root / ".")
        self.assertEqual(a.gpu_uuid, GPU_A)
        self.assertEqual(a.workspace_root, self.root.resolve())
        self.assertEqual(a.lock_path, b.lock_path)
        self.assertEqual(a.marker_path, b.marker_path)
        self.assertEqual(a.controller_path, Path(str(a.marker_path) + ".controller.lock"))
        self.assertEqual(a.attestation_path, Path(str(a.marker_path) + ".attestation"))
        self.assertNotEqual(a.lock_path, a.marker_path)

    @unittest.skipUnless(os.name == "nt", "Windows filesystem case test")
    def test_windows_root_case_alias_resolves_to_same_bundle(self):
        a = self.resolve()
        alias = Path(str(self.root).swapcase())
        b = self.resolve(root=alias)
        self.assertEqual(a.lock_path, b.lock_path)
        self.assertEqual(a.marker_path, b.marker_path)

    @unittest.skipUnless(os.name == "nt", "Windows junction test")
    def test_windows_junction_root_alias_and_state_directory_rejection(self):
        target = self.root / "target"
        target.mkdir()
        alias = self.root / "root-alias"
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            self.skipTest("directory junction creation unavailable")
        try:
            a = self.resolve(root=target)
            b = self.resolve(root=alias)
            self.assertEqual(a.lock_path, b.lock_path)
            self.assertEqual(a.marker_path, b.marker_path)
        finally:
            os.rmdir(alias)

        hostile_root = self.root / "hostile-root"
        hostile_root.mkdir()
        redirect = hostile_root / ".looking-glass-devices"
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(redirect), str(target)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            self.skipTest("directory junction creation unavailable")
        try:
            with self.assertRaises(self.api.DeviceIdentityError):
                self.resolve(root=hostile_root)
        finally:
            os.rmdir(redirect)

    def test_existing_root_symlink_alias_resolves_to_same_bundle(self):
        target = self.root / "real"
        target.mkdir()
        alias = self.root / "alias"
        try:
            os.symlink(target, alias, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        a = self.resolve(root=target)
        b = self.resolve(root=alias)
        self.assertEqual(a.lock_path, b.lock_path)
        self.assertEqual(a.marker_path, b.marker_path)

    def test_explicit_full_uuid_visibility_can_select_from_inventory(self):
        item = self.resolve(inventory=(GPU_A, GPU_B), environ={"CUDA_VISIBLE_DEVICES": GPU_A.upper()})
        self.assertEqual(item.gpu_uuid, GPU_A)
        self.assertEqual(item.cuda_device_index, 0)
        lease = self.lease_for(item)
        item.verify_runtime_binding(lease=lease, runtime_uuid=GPU_A, device_count=1, environ={"CUDA_VISIBLE_DEVICES": GPU_A.upper()})

    def test_ambiguous_visibility_and_remapping_are_rejected(self):
        invalid = ("", "-1", "0", "1", "0,1", "GPU-12345678", "MIG-12345678-1234-5678-9abc-1234567890ab", GPU_A + "," + GPU_B)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(self.api.DeviceIdentityError):
                self.resolve(inventory=(GPU_A, GPU_B), environ={"CUDA_VISIBLE_DEVICES": value})
        with self.assertRaises(self.api.DeviceIdentityError):
            self.resolve(inventory=(GPU_A, GPU_B))
        with self.assertRaises(self.api.DeviceIdentityError):
            self.resolve(inventory=(GPU_A, GPU_B), environ={"CUDA_VISIBLE_DEVICES": GPU_B})
        for index in (-1, 1, True, 0.0):
            with self.subTest(index=index), self.assertRaises(self.api.DeviceIdentityError):
                self.resolve(index=index)

    def test_noncanonical_environment_key_case_cannot_hide_visibility(self):
        for env in (
            {"cuda_visible_devices": "0"},
            {"CUDA_VISIBLE_DEVICES": GPU_A, "cuda_visible_devices": "0"},
            {"cuda_device_order": "PCI_BUS_ID"},
        ):
            with self.subTest(env=env), self.assertRaises(self.api.DeviceIdentityError):
                self.resolve(environ=env)

    def test_distinct_physical_uuids_get_distinct_bundles(self):
        a = self.resolve(inventory=(GPU_A, GPU_B), environ={"CUDA_VISIBLE_DEVICES": GPU_A})
        b = self.resolve(uuid=GPU_B, inventory=(GPU_A, GPU_B), environ={"CUDA_VISIBLE_DEVICES": GPU_B})
        self.assertNotEqual(a.lock_path, b.lock_path)
        self.assertNotEqual(a.marker_path, b.marker_path)

    def test_concurrent_processes_derive_the_same_bundle_during_first_creation(self):
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        outcomes = context.Queue()
        processes = [context.Process(target=_resolve_in_process, args=(str(self.root), start, outcomes)) for _ in range(4)]
        for process in processes:
            process.start()
        try:
            start.set()
            values = [outcomes.get(timeout=15) for _ in processes]
            self.assertTrue(all(value[0] == "ok" for value in values), values)
            self.assertEqual(len(set(values)), 1)
        finally:
            for process in processes:
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(10)
            outcomes.close()

    def test_uuid_inventory_rejects_malformed_duplicates_and_wrong_target(self):
        bad = ("../" + GPU_A, "GPU-1234", GPU_A + "\n", "MIG-" + GPU_A, "GPU-12345678-1234-5678-9abc-1234567890a/", "")
        for uuid in bad:
            with self.subTest(uuid=uuid), self.assertRaises(self.api.DeviceIdentityError):
                self.resolve(uuid=uuid)
        with self.assertRaises(self.api.DeviceIdentityError):
            self.resolve(inventory=(GPU_A, GPU_A.upper()))
        with self.assertRaises(self.api.DeviceIdentityError):
            self.resolve(inventory=(GPU_B,))
        with self.assertRaises(self.api.DeviceIdentityError):
            self.resolve(inventory=GPU_A)

    def test_runtime_binding_fails_closed_on_mismatch_or_environment_change(self):
        item = self.resolve(environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        lease = self.lease_for(item)
        item.verify_runtime_binding(lease=lease, runtime_uuid=GPU_A.upper(), device_count=1, environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        for uuid, count, index, env in (
            (GPU_B, 1, 0, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
            (None, 1, 0, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
            (GPU_A, 2, 0, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
            (GPU_A, True, 0, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
            (GPU_A, 1, 1, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
            (GPU_A, 1, 0, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}),
            (GPU_A, 1, 0, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPU_A}),
        ):
            with self.subTest(uuid=uuid, count=count, index=index, env=env), self.assertRaises(self.api.DeviceIdentityError):
                item.verify_runtime_binding(lease=lease, runtime_uuid=uuid, device_count=count, cuda_device_index=index, environ=env)

    def test_runtime_binding_requires_matching_live_lease(self):
        item = self.resolve()
        other = DeviceLease.acquire(self.root / "different.lock", timeout=1)
        self.addCleanup(other.release)
        with self.assertRaises(self.api.DeviceIdentityError):
            item.verify_runtime_binding(lease=None, runtime_uuid=GPU_A, device_count=1, environ={})
        with self.assertRaises(self.api.DeviceIdentityError):
            item.verify_runtime_binding(lease=other, runtime_uuid=GPU_A, device_count=1, environ={})
        lease = self.lease_for(item)
        item.verify_runtime_binding(lease=lease, runtime_uuid=GPU_A, device_count=1, environ={})
        lease.release()
        with self.assertRaises(self.api.DeviceIdentityError):
            item.verify_runtime_binding(lease=lease, runtime_uuid=GPU_A, device_count=1, environ={})

    def test_counterfeit_lease_subclass_cannot_bypass_live_lock_check(self):
        item = self.resolve()

        class CounterfeitLease(DeviceLease):
            def __init__(self, path):
                self.path = path

            def check(self):
                return None

        with self.assertRaises(self.api.DeviceIdentityError):
            item.verify_runtime_binding(
                lease=CounterfeitLease(item.lock_path),
                runtime_uuid=GPU_A,
                device_count=1,
                environ={},
            )

    def test_root_must_be_an_existing_absolute_directory(self):
        for root in ("relative-workspace", self.root / "missing"):
            with self.subTest(root=root), self.assertRaises(self.api.DeviceIdentityError):
                self.resolve(root=root)
        file = self.root / "file"
        file.write_text("x")
        with self.assertRaises(self.api.DeviceIdentityError):
            self.resolve(root=file)

    def test_hardlinked_lock_or_marker_siblings_are_rejected(self):
        item = self.resolve()
        for path in (item.lock_path, item.marker_path, item.controller_path, item.attestation_path):
            with self.subTest(path=path):
                path.write_text("x")
                alias = self.root / (path.name + ".alias")
                try:
                    os.link(path, alias)
                except OSError as exc:
                    path.unlink()
                    self.skipTest(f"hard links unavailable: {exc}")
                try:
                    with self.assertRaises(self.api.DeviceIdentityError):
                        item.validate_paths()
                finally:
                    alias.unlink()
                    path.unlink()
        item.validate_paths()

    def test_replaced_state_directory_is_rejected(self):
        item = self.resolve()
        original = item.lock_path.parent
        moved = self.root / "old-device-state"
        original.rename(moved)
        original.mkdir()
        with self.assertRaises(self.api.DeviceIdentityError):
            item.validate_paths()

    def test_replaced_bundle_fields_cannot_split_marker_from_lock(self):
        item = self.resolve()
        false_marker = item.marker_path.with_name("different.unsafe.json")
        forged = replace(
            item,
            marker_path=false_marker,
            controller_path=Path(str(false_marker) + ".controller.lock"),
            attestation_path=Path(str(false_marker) + ".attestation"),
        )
        with self.assertRaises(self.api.DeviceIdentityError):
            forged.validate_paths()
        lease = self.lease_for(item)
        with self.assertRaises(self.api.DeviceIdentityError):
            forged.verify_runtime_binding(lease=lease, runtime_uuid=GPU_A, device_count=1, environ={})
        for field in ("lock_path", "controller_path", "attestation_path"):
            forged = replace(item, **{field: self.root / ("different-" + field)})
            with self.subTest(field=field), self.assertRaises(self.api.DeviceIdentityError):
                forged.validate_paths()

    def test_strict_nvidia_smi_inventory_parser(self):
        self.assertEqual(self.api.parse_nvidia_smi_inventory(f" {GPU_A.upper()}\n{GPU_B}\n"), (GPU_A, GPU_B))
        for text in ("", f"{GPU_A}\n{GPU_A.upper()}\n", "GPU 0: name", "MIG-123", "x" * 20000):
            with self.subTest(text=text[:20]), self.assertRaises(self.api.DeviceIdentityError):
                self.api.parse_nvidia_smi_inventory(text)


if __name__ == "__main__":
    unittest.main()
