"""CPU-only launch-setting checks; these never initialize CUDA."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from looking_glass.trusted_service_launch import prepare_trusted_launch, read_trusted_config


UUID = "GPU-00000000-0000-0000-0000-000000000001"


class TrustedServiceLaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root / 'repo'
        self.site = self.root / 'env' / 'site-packages'
        self.private = self.root / 'private'
        self.hf_home = self.root / 'hf-home'
        self.repo.mkdir()
        self.site.mkdir(parents=True)
        self.private.mkdir()
        self.hf_home.mkdir()
        package = self.repo / 'looking_glass'
        package.mkdir()
        (package / '__init__.py').write_text('', encoding='utf-8')
        source = ("import json,os,sys\n"
                  "print(json.dumps({'path':sys.path,'argv':sys.argv[1:],"
                  "'visibility':os.environ.get('CUDA_VISIBLE_DEVICES'),"
                  "'hf_home':os.environ.get('HF_HOME'),"
                  "'programfiles':os.environ.get('PROGRAMFILES'),"
                  "'rogue':os.environ.get('CUDA_LAUNCH_BLOCKING'),"
                  "'pythonpath':os.environ.get('PYTHONPATH')}))\n")
        for name in ('gpu_service_process', 'trusted_profile_collector'):
            (package / (name + '.py')).write_text(source, encoding='utf-8')
        self.config = self.private / 'config.json'
        self.config.write_text('{"test_only":true}', encoding='utf-8')

    def build(self, module='looking_glass.gpu_service_process', **overrides):
        arguments = dict(repo_root=self.repo, site_packages=self.site,
            environment_root=self.site.parent, private_parent=self.private, hf_home=self.hf_home,
            gpu_uuid=UUID, module=module, config_path=self.config,
            trusted_path=str(self.root), inherited_environment={**os.environ,
                'CUDA_LAUNCH_BLOCKING': '1', 'PYTHONPATH': 'untrusted', 'HF_HOME': 'untrusted',
                'EXTRA_SECRET': 'hidden'})
        arguments.update(overrides)
        return prepare_trusted_launch(**arguments)

    def test_service_and_collector_have_same_interpreter_paths_and_fixed_environment(self):
        spec = [self.build(), self.build('looking_glass.trusted_profile_collector')]
        outputs = []
        for launch in spec:
            launch.verify_ready()
            result = subprocess.run(launch.command, cwd=self.root, env=dict(launch.environment),
                                    capture_output=True, text=True, check=True, timeout=10)
            outputs.append(json.loads(result.stdout))
            launch.verify_ready()
            self.assertNotIn('EXTRA_SECRET', launch.environment)
            self.assertNotIn('CUDA_LAUNCH_BLOCKING', launch.environment)
            self.assertNotIn('PYTHONPATH', launch.environment)
            self.assertEqual(outputs[-1]['programfiles'], launch.environment.get('PROGRAMFILES'))
        self.assertEqual(outputs[0]['path'], outputs[1]['path'])
        self.assertEqual(outputs[0]['path'][-2:], [str(self.repo), str(self.site)])
        info = self.config.stat()
        digest = hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.assertEqual(outputs[0]['argv'], ['--config', str(self.config),
            '--config-id', f'{info.st_dev}:{info.st_ino}', '--config-sha256', digest])
        self.assertEqual(outputs[0]['visibility'], UUID)
        self.assertEqual([output['hf_home'] for output in outputs], [str(self.hf_home)] * 2)
        self.assertIsNone(outputs[0]['rogue'])
        self.assertIsNone(outputs[0]['pythonpath'])

    def test_preserves_only_validated_programfiles_for_driver_discovery(self):
        program_files = self.root / 'program files'
        program_files.mkdir()
        environment = {'PROGRAMFILES': str(program_files), 'EXTRA_SECRET': 'not retained'}
        for module in ('looking_glass.gpu_service_process', 'looking_glass.trusted_profile_collector'):
            launch = self.build(module=module, inherited_environment=environment)
            self.assertEqual(launch.environment['PROGRAMFILES'], str(program_files.resolve()))
            self.assertNotIn('EXTRA_SECRET', launch.environment)
        for invalid in ('relative-program-files', str(self.root / 'missing')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.build(inherited_environment={'PROGRAMFILES': invalid})

    def test_changed_configuration_and_nonempty_prefix_are_rejected(self):
        launch = self.build()
        self.config.write_text('{"changed":true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            launch.verify_ready()
        launch = self.build()
        (launch.pycache_prefix / 'stale.pyc').write_bytes(b'stale')
        with self.assertRaisesRegex(ValueError, 'bytecode prefix'):
            launch.verify_ready()

    def test_child_read_rejects_configuration_replaced_after_prelaunch_check(self):
        launch = self.build()
        launch.verify_ready()
        info = self.config.stat()
        identity = (info.st_dev, info.st_ino)
        digest = hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.assertEqual(read_trusted_config(self.config, expected_identity=identity,
                                            expected_sha256=digest), b'{"test_only":true}')
        self.config.write_text('{"test_only":false}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'differs from the launch pin'):
            read_trusted_config(self.config, expected_identity=identity, expected_sha256=digest)

    def test_real_child_rejects_same_size_mutation_and_replacement_after_check(self):
        program = ("import json,sys;sys.path.insert(0,sys.argv[1]);"
                   "from looking_glass.trusted_service_launch import read_trusted_config;"
                   "identity=tuple(map(int,sys.argv[3].split(':')));"
                   "json.loads(read_trusted_config(sys.argv[2],expected_identity=identity,"
                   "expected_sha256=sys.argv[4]))")
        actual_repo = Path(__file__).resolve().parents[1]
        for change in ('mutate', 'replace'):
            self.config.write_bytes(b'{"test_only":true}')
            launch = self.build()
            launch.verify_ready()
            info = self.config.stat()
            identity = f'{info.st_dev}:{info.st_ino}'
            digest = hashlib.sha256(self.config.read_bytes()).hexdigest()
            if change == 'mutate':
                self.config.write_bytes(b'{"test_only":null}')
            else:
                replacement = self.private / 'replacement.json'
                replacement.write_bytes(self.config.read_bytes())
                os.replace(replacement, self.config)
            outcome = subprocess.run([sys._base_executable, '-c', program, str(actual_repo),
                                      str(self.config), identity, digest], cwd=self.root,
                                     capture_output=True, text=True, timeout=10)
            with self.subTest(change=change):
                self.assertNotEqual(outcome.returncode, 0)
                self.assertIn('private configuration differs', outcome.stderr)

    def test_rejects_untrusted_module_config_and_environment_paths(self):
        with self.assertRaisesRegex(ValueError, 'unsupported trusted owner'):
            self.build(module='arbitrary.module')
        with self.assertRaisesRegex(ValueError, 'outside the selected environment'):
            self.build(environment_root=self.repo)
        external = self.root / 'external.json'
        external.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'inside the private launch'):
            self.build(config_path=external)
        with self.assertRaisesRegex(ValueError, 'absolute'):
            self.build(trusted_path='relative-directory')
        with self.assertRaisesRegex(ValueError, 'absolute'):
            self.build(hf_home='relative-directory')


if __name__ == '__main__':
    unittest.main()
