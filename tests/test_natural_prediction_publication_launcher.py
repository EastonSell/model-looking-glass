"""Isolated source-bootstrap checks; no model or held-out inputs."""

import hashlib
import base64
import importlib.util
import os
from pathlib import Path
import py_compile
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

from test_natural_prediction_freeze import canonical, replace_binding
from test_natural_prediction_publication import fake_inputs
from looking_glass import natural_prediction_publication as publication


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'publish_natural_prediction_freeze.py'


def run_probe(root, package_hash, module_hash):
    code = '''import importlib.util, pathlib, sys
script, root, package_hash, module_hash = sys.argv[1:]
spec = importlib.util.spec_from_file_location("publication_wrapper", script)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
mapping = {"looking_glass/__init__.py": package_hash, "looking_glass/fake.py": module_hash}
sys.meta_path.insert(0, wrapper._VerifiedFinder(pathlib.Path(root), mapping, "0" * 64))
from looking_glass.fake import VALUE
print(VALUE)
'''
    return subprocess.run([sys.executable, '-I', '-B', '-c', code, str(SCRIPT), str(root),
                           package_hash, module_hash], capture_output=True, text=True, timeout=30)


class PublicationLauncherTests(unittest.TestCase):
    def test_output_path_rejects_parent_alias_into_protected_directory(self):
        spec = importlib.util.spec_from_file_location('publication_wrapper_path_test', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protected = root / 'owner' / 'attempt' / 'tensors'
            protected.mkdir(parents=True)
            (root / 'elsewhere').mkdir()
            alias = root / 'elsewhere' / '..' / 'owner' / 'attempt' / 'tensors' / 'stage'
            with self.assertRaises(RuntimeError):
                module._safe_output(alias, [protected])
            self.assertEqual(module._safe_output(root / 'safe', [protected]), root / 'safe')

    @unittest.skipUnless(os.name == 'nt', 'Windows junction test')
    def test_output_path_rejects_junction_alias_into_protected_directory(self):
        spec = importlib.util.spec_from_file_location('publication_wrapper_junction_test', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protected, alias = root / 'protected', root / 'alias'
            protected.mkdir()
            link = subprocess.run(['cmd', '/d', '/c', 'mklink', '/J', str(alias), str(protected)],
                                  capture_output=True, text=True, timeout=30)
            if link.returncode:
                self.skipTest('Windows junction creation unavailable')
            try:
                with self.assertRaises(RuntimeError):
                    module._safe_output(alias / 'receipt.json', [protected])
                with self.assertRaises(RuntimeError):
                    module._existing_path(alias)
            finally:
                os.rmdir(alias)

    def test_source_finder_bypasses_timestamp_valid_stale_bytecode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / 'looking_glass'
            package.mkdir()
            initializer = package / '__init__.py'
            initializer.write_bytes(b'')
            module = package / 'fake.py'
            module.write_bytes(b'VALUE = 1\n')
            original_mtime = module.stat().st_mtime_ns
            py_compile.compile(str(module), doraise=True,
                               invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            module.write_bytes(b'VALUE = 2\n')
            os.utime(module, ns=(original_mtime, original_mtime))
            result = run_probe(root, hashlib.sha256(b'').hexdigest(), hashlib.sha256(b'VALUE = 2\n').hexdigest())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), '2')

    def test_source_finder_rejects_changed_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / 'looking_glass'
            package.mkdir()
            (package / '__init__.py').write_bytes(b'')
            (package / 'fake.py').write_bytes(b'VALUE = 2\n')
            result = run_probe(root, hashlib.sha256(b'').hexdigest(), hashlib.sha256(b'VALUE = 1\n').hexdigest())
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('source changed before import', result.stderr)

    def test_command_rejects_nonisolated_or_bytecode_enabled_execution(self):
        result = subprocess.run([sys.executable, str(SCRIPT), 'track', '--wrapper-revision', 'f' * 40],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertIn('RuntimeError', result.stderr)

    def test_staging_requires_gpu_hidden_before_any_checkout_is_read(self):
        names = ('source-root', 'source-revision', 'binding', 'binding-url', 'owner', 'launcher-terminal',
                 'audit', 'preflight-manifest', 'auditor-sha256', 'clarification-sha256', 'output')
        arguments = [item for name in names for item in ('--' + name, 'invalid')]
        env = dict(os.environ)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        process = subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), 'stage',
                                  '--wrapper-revision', 'f' * 40, *arguments], env=env,
                                 capture_output=True, text=True, timeout=30)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(process.stdout, '')
        self.assertIn('GpuVisibilityError', process.stderr)

    @unittest.skipUnless(os.name == 'nt', 'Windows read-share leases are required')
    def test_fabricated_stage_runs_in_isolated_leased_source_checkout(self):
        project = SCRIPT.parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, wrapper, private = root / 'source', root / 'wrapper', root / 'private'
            source.mkdir()
            wrapper.mkdir()
            private.mkdir()
            for path in (project / 'looking_glass').rglob('*.py'):
                relative = path.relative_to(project)
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            for relative in ('scripts/v2_source_guards.py', 'tools/audit_natural_agreement_discovery.py'):
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project / relative, target)
            for relative in ('scripts/v2_source_guards.py', 'tools/publish_natural_prediction_freeze.py',
                             'looking_glass/natural_prediction_publication.py'):
                target = wrapper / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project / relative, target)

            def commit_checkout(directory):
                def git(*args):
                    return subprocess.run(['git', '-C', str(directory), *args], check=True,
                                          capture_output=True, text=True, timeout=30).stdout.strip()
                git('init', '-q')
                git('add', '.')
                git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
                return git('rev-parse', 'HEAD')

            source_revision, wrapper_revision = commit_checkout(source), commit_checkout(wrapper)
            inventory = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (source / 'looking_glass').rglob('*.py')}
            for relative in ('scripts/v2_source_guards.py', 'tools/audit_natural_agreement_discovery.py'):
                inventory[relative] = hashlib.sha256((source / relative).read_bytes()).hexdigest()

            owner = private / 'owner'
            attempt = owner / 'attempt'
            tensors = attempt / 'tensors'
            tensors.mkdir(parents=True)
            inputs = fake_inputs(tensors)
            binding = json.loads(inputs['binding'])
            auditor = inventory['tools/audit_natural_agreement_discovery.py']
            binding.update(source_commit=source_revision, source_sha256=inventory,
                           independent_natural_audit_source_sha256=auditor)
            replace_binding(inputs, binding)
            audit = json.loads(inputs['audit'])
            audit['audit_source_sha256'] = auditor
            inputs['audit'] = canonical(audit)
            for name in ('events', 'started', 'terminal'):
                (attempt / {'events': 'events.jsonl', 'started': 'started.json', 'terminal': 'terminal.json'}[name]).write_bytes(inputs[name])
            (owner / 'supervisor.json').write_bytes(inputs['supervisor'])
            for name in ('binding', 'launcher_terminal', 'audit', 'preflight_manifest'):
                (private / (name + '.json')).write_bytes(inputs[name])

            tool = wrapper / 'tools/publish_natural_prediction_freeze.py'
            output = root / 'stage'
            url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + '1' * 40 +
                   '/results/distilgpt2-natural-agreement-v1-20261028/binding.json')
            args = ['stage', '--wrapper-revision', wrapper_revision, '--source-root', str(source),
                    '--source-revision', source_revision, '--binding', str(private / 'binding.json'),
                    '--binding-url', url, '--owner', str(owner),
                    '--launcher-terminal', str(private / 'launcher_terminal.json'),
                    '--audit', str(private / 'audit.json'), '--preflight-manifest', str(private / 'preflight_manifest.json'),
                    '--auditor-sha256', auditor, '--clarification-sha256', '3' * 64, '--output', str(output)]
            code = '''import importlib.util, pathlib, sys, urllib.request
tool, binding = sys.argv[1:3]
class Response:
    status = 200
    def __init__(self, url): self.url = url
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def geturl(self): return self.url
    def read(self, limit): return pathlib.Path(binding).read_bytes()[:limit]
class Opener:
    def open(self, req, timeout): return Response(req.full_url)
urllib.request.build_opener = lambda *args: Opener()
spec = importlib.util.spec_from_file_location("publication_wrapper", tool)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.argv = [tool] + sys.argv[3:]
module.main()
'''
            process = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(tool),
                                      str(private / 'binding.json'), *args], capture_output=True, text=True,
                                     timeout=120, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'})
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result['status'], 'stage_complete')
            self.assertEqual(result['stage_sha256'], hashlib.sha256((output / 'stage.json').read_bytes()).hexdigest())
            self.assertEqual(json.loads((output / 'stage.json').read_bytes())['source_commit'], source_revision)

            prediction_commit, observed_head, aggregate_commit, final_head = ('2' * 40, '4' * 40, '5' * 40, '6' * 40)
            stage = json.loads((output / 'stage.json').read_bytes())
            prediction = (output / 'natural-prediction-v2.json').read_bytes()
            aggregate = (output / 'natural-prediction-freeze.json').read_bytes()
            provenance = (output / 'stage.json').read_bytes()
            remote = {url: inputs['binding'],
                      publication.raw_url(prediction_commit, stage['binding_path']): inputs['binding'],
                      publication.raw_url(prediction_commit, stage['prediction_path']): prediction,
                      publication.raw_url(prediction_commit, stage['provenance_path']): provenance,
                      publication.raw_url(observed_head, stage['binding_path']): inputs['binding'],
                      publication.raw_url(observed_head, stage['prediction_path']): prediction,
                      publication.raw_url(observed_head, stage['provenance_path']): provenance,
                      publication.raw_url(aggregate_commit, stage['binding_path']): inputs['binding'],
                      publication.raw_url(aggregate_commit, stage['prediction_path']): prediction,
                      publication.raw_url(aggregate_commit, stage['provenance_path']): provenance,
                      publication.raw_url(aggregate_commit, stage['aggregate_path']): aggregate,
                      publication.raw_url(final_head, stage['binding_path']): inputs['binding'],
                      publication.raw_url(final_head, stage['prediction_path']): prediction,
                      publication.raw_url(final_head, stage['provenance_path']): provenance,
                      publication.raw_url(final_head, stage['aggregate_path']): aggregate}
            for before, after in ((source_revision, stage['binding_commit']), (stage['binding_commit'], prediction_commit),
                                  (wrapper_revision, prediction_commit),
                                  (prediction_commit, observed_head), (observed_head, aggregate_commit),
                                  (prediction_commit, aggregate_commit), (aggregate_commit, final_head)):
                remote[publication._compare_url(before, after)] = canonical(
                    {'status': 'ahead', 'base_commit': {'sha': before}, 'merge_base_commit': {'sha': before}})

            def verify(mode, arguments, head):
                remote[publication._HEAD_URL] = canonical({'ref': 'refs/heads/main',
                                                            'object': {'type': 'commit', 'sha': head}})
                network = root / (mode + '-network.json')
                network.write_text(json.dumps({url: base64.b64encode(data).decode('ascii')
                                               for url, data in remote.items()}), encoding='utf-8')
                verify_code = '''import base64, importlib.util, json, pathlib, sys, urllib.error, urllib.request
tool, network = sys.argv[1:3]
mapping = json.loads(pathlib.Path(network).read_text(encoding="utf-8"))
class Response:
    status = 200
    def __init__(self, url): self.url = url
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def geturl(self): return self.url
    def read(self, limit): return base64.b64decode(mapping[self.url])[:limit]
class Opener:
    def open(self, req, timeout):
        if req.full_url not in mapping: raise urllib.error.HTTPError(req.full_url, 404, "missing", None, None)
        return Response(req.full_url)
urllib.request.build_opener = lambda *args: Opener()
spec = importlib.util.spec_from_file_location("publication_wrapper", tool)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.argv = [tool] + sys.argv[3:]
module.main()
'''
                return subprocess.run([sys.executable, '-I', '-B', '-c', verify_code, str(tool), str(network),
                                       mode, '--wrapper-revision', wrapper_revision, '--stage', str(output),
                                       '--stage-sha256', result['stage_sha256'], *arguments],
                                      capture_output=True, text=True, timeout=120)

            track_dir, aggregate_dir = root / 'track-receipt', root / 'aggregate-receipt'
            track_dir.mkdir()
            aggregate_dir.mkdir()
            track = track_dir / 'receipt.json'
            aggregate_receipt = aggregate_dir / 'receipt.json'
            tracked = verify('track', ['--prediction-commit', prediction_commit, '--receipt', str(track)], observed_head)
            self.assertEqual(tracked.returncode, 0, tracked.stderr)
            frozen = verify('aggregate', ['--aggregate-commit', aggregate_commit, '--track-receipt', str(track),
                                          '--receipt', str(aggregate_receipt)], final_head)
            self.assertEqual(frozen.returncode, 0, frozen.stderr)
            self.assertEqual(json.loads(frozen.stdout)['status'], 'aggregate_observed')
            self.assertEqual(json.loads(aggregate_receipt.read_bytes())['aggregate_commit'], aggregate_commit)


if __name__ == '__main__':
    unittest.main()
