import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from looking_glass.resource_profile import ProfileRegistry, ProfileUnavailable
if __package__:
    from .test_trusted_service_profile import TrustedServiceProfileTests, canonical
else:
    from test_trusted_service_profile import TrustedServiceProfileTests, canonical


class CustodyServiceProfileTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('looking_glass.custody_service_profile')
        self.trusted = importlib.import_module('looking_glass.trusted_service_profile')
        self.fixtures = TrustedServiceProfileTests()
        self.direct = self.fixtures.make_identity()

    def custody_identity(self):
        with patch.object(self.module, 'build_trusted_service_identity', return_value=self.direct), \
             patch.object(self.module, '_guard_owner') as guard, \
             patch.object(self.module, '_verify_protocol_document'):
            result = self.module.build_custody_service_identity(object(), object(), object(), object())
        self.assertGreaterEqual(guard.call_count, 1)
        return result

    def registry_for(self, identity):
        profiles = self.fixtures.make_profiles(identity)
        pins = tuple(profile.profile_id for profile in profiles)
        return ProfileRegistry(profiles, approved_profile_ids=pins), pins

    def test_distinct_custody_identity_cannot_resolve_direct_profiles(self):
        identity = self.custody_identity()
        runtime = json.loads(identity.runtime_manifest)
        self.assertEqual(runtime['schema'], 'trusted-custody-service-runtime-v1')
        self.assertEqual(runtime['execution_path'],
                         'looking_glass.owner_completion.LifetimeWorkspaceWorker.execute')
        self.assertEqual(runtime['custody_execution']['workspace_commitment_bytes'], 512 * 1024 ** 2)
        self.assertEqual(runtime['custody_execution']['resident_commitment_bytes'], 2 * 1024 ** 3)
        self.assertTrue(all(a.runtime_settings_sha256 != b.runtime_settings_sha256
                            for a, b in zip(self.direct.keys, identity.keys)))
        registry, pins = self.registry_for(identity)
        self.module.resolve_custody_service_profiles(registry, identity, pins)
        with self.assertRaises(ValueError):
            self.trusted.resolve_trusted_service_profiles(registry, identity, pins)
        direct_registry, direct_pins = self.registry_for(self.direct)
        with self.assertRaises(ValueError):
            self.module.resolve_custody_service_profiles(direct_registry, self.direct, direct_pins)
        with self.assertRaises(ProfileUnavailable):
            self.module.resolve_custody_service_profiles(direct_registry, identity, direct_pins)
        with self.assertRaises(ProfileUnavailable):
            self.trusted.resolve_trusted_service_profiles(registry, self.direct, pins)

    def test_repinned_wrong_custody_protocol_is_rejected(self):
        identity = self.custody_identity()
        source_hash = hashlib.sha256(identity.source_manifest).hexdigest()
        for field, value in (('workspace_commitment_bytes', 1), ('resident_commitment_bytes', 1),
                             ('guarded_rng', False), ('max_concurrent_forwards', 2),
                             ('protocol_document_sha256', 'f' * 64),
                             ('call_chain', ['looking_glass.resident_worker.ResidentWorker.execute'])):
            runtime = json.loads(identity.runtime_manifest)
            runtime['custody_execution'][field] = value
            changed = dataclasses.replace(identity, runtime_manifest=canonical(runtime),
                keys=self.module._make_workload_keys(source_hash, runtime))
            registry, pins = self.registry_for(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.module.resolve_custody_service_profiles(registry, changed, pins)

    def test_direct_schema_cannot_advertise_a_different_execution_path(self):
        runtime = json.loads(self.direct.runtime_manifest)
        runtime['execution_path'] = 'looking_glass.owner_completion.LifetimeWorkspaceWorker.execute'
        source_hash = hashlib.sha256(self.direct.source_manifest).hexdigest()
        changed = dataclasses.replace(self.direct, runtime_manifest=canonical(runtime),
            keys=self.module._make_workload_keys(source_hash, runtime))
        registry, pins = self.registry_for(changed)
        with self.assertRaises(ValueError):
            self.trusted.resolve_trusted_service_profiles(registry, changed, pins)

    def test_builder_rejects_inconsistent_direct_identity(self):
        malformed = dataclasses.replace(self.direct, runtime_manifest=b'{}')
        with patch.object(self.module, 'build_trusted_service_identity', return_value=malformed), \
             patch.object(self.module, '_guard_owner'), patch.object(self.module, '_verify_protocol_document'):
            with self.assertRaises(ValueError):
                self.module.build_custody_service_identity(object(), object(), object(), object())

    def test_exact_versioned_protocol_document_is_required(self):
        source = Path(__file__).resolve().parents[1] / 'docs' / 'CUSTODY_SERVICE_PROFILE_V1.md'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'docs').mkdir()
            target = root / 'docs' / source.name
            target.write_bytes(source.read_bytes())
            self.module._verify_protocol_document(SimpleNamespace(repo_root=str(root)))
            target.write_bytes(target.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                self.module._verify_protocol_document(SimpleNamespace(repo_root=str(root)))


if __name__ == '__main__':
    unittest.main()
