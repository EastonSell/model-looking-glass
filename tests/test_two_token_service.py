import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from looking_glass import gpu_service_process as child
from looking_glass import two_token_service as service
from looking_glass.gpu_service import ProfileFiles
from looking_glass.trusted_service_profile import TrustedServiceSource


UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


class ServicePreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root / "repo"
        self.cache = self.root / "cache"
        self.env = self.root / "env"
        self.site = self.env / "site-packages"
        self.private = self.root / "private"
        for path in (self.repo, self.cache, self.site, self.private):
            path.mkdir(parents=True)
        self.package = self.repo / "looking_glass"
        self.package.mkdir()
        (self.package / "gpu_service.py").write_text("# fixed owner fixture\n")
        (self.package / "supervisor.py").write_text("# reaping supervisor fixture\n")
        (self.package / "trusted_profile_collector.py").write_text(
            "TRUSTED_SERVICE_COLLECTOR_CONTRACT = 'two-token-worker-profile-v1'\n")
        self.source = TrustedServiceSource(str(self.repo), str(self.cache), b"source", b"assets", ())
        self.profiles = tuple(ProfileFiles(self.root / f"profile-{index}",
                                           self.root / f"evidence-{index}",
                                           self.root / f"manifest-{index}",
                                           str(index) * 64) for index in range(1, 4))
        self.source_mock = patch.object(service, "prepare_trusted_source", return_value=self.source).start()
        self.registry_mock = patch.object(service, "_load_profile_registry", return_value=(None, None)).start()
        patch.object(service._owner_module, "FIXED_SERVICE_OWNER_CONTRACT",
                     service._REQUIRED_OWNER_CONTRACT, create=True).start()
        patch.object(service._owner_module, "__file__", str(self.package / "gpu_service.py")).start()
        patch.object(service._supervisor_module, "REAPABLE_CHILD_SPAWN_CONTRACT",
                     service._REQUIRED_SPAWN_CONTRACT, create=True).start()
        patch.object(service._supervisor_module, "__file__", str(self.package / "supervisor.py")).start()
        self.addCleanup(patch.stopall)

    def prepare(self, **changes):
        arguments = dict(repo_root=self.repo, cache_root=self.cache, workspace_root=self.root,
                         private_parent=self.private, site_packages=self.site,
                         environment_root=self.env, gpu_uuid=UUID, physical_uuids=(UUID,),
                         profiles=self.profiles, teams=("alpha", "beta"),
                         trusted_path=str(self.root))
        arguments.update(changes)
        return service.prepare_two_token_service(**arguments)

    def test_configuration_and_launch_are_bound_before_process_start(self):
        prepared = self.prepare()
        launch = prepared._launch
        config = child._parse_config(launch.config_path.read_bytes())
        self.assertEqual(config.teams, ("alpha", "beta"))
        self.assertEqual(config.profiles, tuple((p.document, p.evidence, p.source_manifest,
                                                p.approved_profile_id) for p in self.profiles))
        self.assertEqual(config.source_base_sha256, hashlib.sha256(b"source").hexdigest())
        self.assertEqual(config.asset_manifest_sha256, hashlib.sha256(b"assets").hexdigest())
        self.assertEqual(launch.environment["CUDA_VISIBLE_DEVICES"], UUID)
        self.assertEqual(launch.environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertNotIn("alpha", launch.environment.values())
        self.assertIn("looking_glass.gpu_service_process", launch.command)
        self.assertNotIn(UUID, repr(prepared))
        self.assertFalse(prepared._controller._claimed)

        with patch.object(service, "_start_owner_controller") as start:
            prepared.start(owner_timeout=123)
        args, kwargs = start.call_args
        self.assertIs(args[0], prepared._controller)
        self.assertEqual(args[1], launch.command)
        self.assertIs(kwargs["identity"], prepared._identity)
        self.assertEqual(kwargs["device_lease_path"], prepared._identity.lock_path)
        self.assertEqual(kwargs["marker_path"], prepared._identity.marker_path)
        self.assertEqual(kwargs["env"], launch.environment)
        self.assertEqual(kwargs["cwd"], str(self.repo))
        self.assertEqual(kwargs["teams"], ("alpha", "beta"))
        self.assertEqual(kwargs["owner_timeout"], 123)
        self.assertEqual(self.registry_mock.call_count, 2)

    def test_changed_config_source_or_profiles_reject_before_start(self):
        prepared = self.prepare()
        with patch.object(service, "_start_owner_controller") as start:
            prepared._launch.config_path.write_bytes(b"{}")
            with self.assertRaises(ValueError):
                prepared.start()
            start.assert_not_called()

        prepared = self.prepare()
        self.source_mock.return_value = TrustedServiceSource(str(self.repo), str(self.cache), b"changed", b"assets", ())
        with patch.object(service, "_start_owner_controller") as start:
            with self.assertRaisesRegex(RuntimeError, "changed"):
                prepared.start()
            start.assert_not_called()

        self.source_mock.return_value = self.source
        prepared = self.prepare()
        self.registry_mock.side_effect = ValueError("changed profile")
        with patch.object(service, "_start_owner_controller") as start:
            with self.assertRaisesRegex(ValueError, "profile"):
                prepared.start()
            start.assert_not_called()

    def test_missing_owner_supervisor_or_collector_contract_rejects_before_start(self):
        prepared = self.prepare()
        with patch.object(service, "_start_owner_controller") as start:
            with patch.object(service._owner_module, "FIXED_SERVICE_OWNER_CONTRACT", "old"):
                with self.assertRaisesRegex(RuntimeError, "reaping supervisor"):
                    prepared.start()
            with patch.object(service._supervisor_module, "REAPABLE_CHILD_SPAWN_CONTRACT", "old"):
                with self.assertRaisesRegex(RuntimeError, "reaping supervisor"):
                    prepared.start()
            (self.package / "trusted_profile_collector.py").write_text(
                "TRUSTED_SERVICE_COLLECTOR_CONTRACT = 'old'\n")
            with self.assertRaisesRegex(RuntimeError, "reaping supervisor"):
                prepared.start()
            (self.package / "trusted_profile_collector.py").unlink()
            with self.assertRaisesRegex(RuntimeError, "reaping supervisor"):
                prepared.start()
            start.assert_not_called()

    def test_rejects_noncanonical_device_and_profile_arguments(self):
        with self.assertRaises(ValueError):
            self.prepare(physical_uuids=[UUID])
        with self.assertRaises(ValueError):
            self.prepare(profiles=self.profiles[:2])
        with self.assertRaises(ValueError):
            self.prepare(physical_uuids=("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",))
        with patch.object(service, "resolve_device_identity", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "canonical"):
                self.prepare()

    def test_client_and_recovery_operations_use_caller_held_controller(self):
        prepared = self.prepare()
        prepared._controller._claimed = True
        with patch.object(prepared._controller, "client", return_value="team client") as client, \
             patch.object(prepared._controller, "shutdown", return_value="shutdown result") as shutdown, \
             patch.object(prepared._controller, "wait", return_value="wait result") as wait:
            self.assertEqual(prepared.client("alpha"), "team client")
            self.assertEqual(prepared.shutdown(timeout=12), "shutdown result")
            self.assertEqual(prepared.wait(timeout=13), "wait result")
            client.assert_called_once_with("alpha")
            shutdown.assert_called_once_with(timeout=12)
            wait.assert_called_once_with(timeout=13)

    def test_shutdown_during_preflight_closes_start_gate(self):
        prepared = self.prepare()
        entered, resume = threading.Event(), threading.Event()
        failures = []

        def preflight(*_):
            entered.set()
            self.assertTrue(resume.wait(5))
            return self.source

        def start():
            try:
                prepared.start()
            except BaseException as exc:
                failures.append(exc)

        self.source_mock.side_effect = preflight
        thread = threading.Thread(target=start)
        thread.start()
        self.assertTrue(entered.wait(5))
        self.assertIsNone(prepared.shutdown(timeout=2))
        resume.set()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RuntimeError)
        self.assertIn("already used", str(failures[0]))
        self.assertFalse(prepared._controller._claimed)
        self.assertFalse(prepared.unsafe_marker_path.exists())
        with self.assertRaisesRegex(RuntimeError, "StartupCancelled"):
            prepared.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
