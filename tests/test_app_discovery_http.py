"""Real loopback discovery routing with synthetic storage/owner, no ML imports.

The production AppState, HTTP handler, validators and discovery module run here.
Only unrelated tensor stores are replaced under an isolated test package name.
This does not validate tensors, inference, Windows filesystem or GPU behavior.
"""
from __future__ import annotations

import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "_mlg_discovery_http_fixture"


class SyntheticStore:
    def __init__(self, *args, **kwargs):
        self.record = None

    def summary(self):
        return None if self.record is None else dict(self.record)

    def clear(self):
        raise AssertionError("Discovery must not clear retained data")


class SyntheticReference(SyntheticStore):
    max_bytes = 1024 * 1024


class SyntheticOwner:
    def __init__(self):
        self.binding = object()
        self.record = {"name": "retained synthetic owner", "family": "vit", "state": "ready"}

    def info(self):
        return dict(self.record)

    def capture_binding(self):
        return self.binding

    def capabilities(self):
        return {"devices": [{"id": "cpu", "available": True}], "cuda": {"available": False}}

    def __getattr__(self, name):
        raise AssertionError(f"Discovery must not execute owner operation: {name}")


class DiscoveryHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert not any(name == PREFIX or name.startswith(PREFIX + ".") for name in sys.modules)
        cls.torch_before = "torch" in sys.modules
        package = types.ModuleType(PREFIX)
        package.__path__ = [str(ROOT / "looking_glass")]
        storage = types.ModuleType(PREFIX + ".app_storage")
        storage.TraceStore = SyntheticStore
        reference = types.ModuleType(PREFIX + ".app_reference")
        reference.ReferenceVector = SyntheticReference
        sys.modules.update({PREFIX: package, storage.__name__: storage, reference.__name__: reference})
        try:
            spec = importlib.util.spec_from_file_location(PREFIX + ".app", ROOT / "looking_glass/app.py")
            cls.app = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.app
            spec.loader.exec_module(cls.app)
        except BaseException:
            cls._drop_modules()
            raise

    @classmethod
    def _drop_modules(cls):
        for name in list(sys.modules):
            if name == PREFIX or name.startswith(PREFIX + "."):
                del sys.modules[name]

    @classmethod
    def tearDownClass(cls):
        cls._drop_modules()
        if not cls.torch_before:
            assert "torch" not in sys.modules, "Discovery unexpectedly imported PyTorch"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mlg discovery HTTP ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "models"
        self.root.mkdir()
        self.owner = SyntheticOwner()
        self.state = self.app.AppState(runtime=self.owner, export_root=Path(self.tmp.name) / "not-created")
        self.state.store.record = {"run_id": "kept-run", "total_bytes": 16}
        self.state.experiment_store.record = {"run_id": "kept-comparison", "total_bytes": 16}
        self.state.reference.record = {"width": 4}
        self.state._live_binding = self.owner.binding
        self.state._reference_binding = self.owner.binding
        self.state._capture_id = "kept-capture"
        self.state._experiment_id = "kept-experiment"
        self.state._reference_id = "kept-reference"
        self.before = self.snapshot()
        self.server = self.app.create_server(self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def snapshot(self):
        s = self.state
        return (s.runtime, s.store, s.experiment_store, s.reference, s._live_binding,
                s._reference_binding, s._capture_id, s._experiment_id, s._reference_id,
                s.store.summary(), s.experiment_store.summary(), s.reference.summary(), s.runtime.info())

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(5)
        self.server.server_close()
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.before, self.snapshot())
        self.assertFalse((Path(self.tmp.name) / "not-created").exists())

    def call(self, data=None, *, token=True, host=None, origin=None, route="/api/browse-models"):
        port = self.server.server_port
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        headers = {}
        if token:
            headers["X-MLG-Token"] = self.state.token
        if host is not None:
            headers["Host"] = host
        if origin is not None:
            headers["Origin"] = origin
        body = None
        if data is not None:
            body = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        try:
            connection.request("GET" if data is None else "POST", route, body, headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def checkpoint(self, path, *, family="llama", head="LlamaForCausalLM"):
        path.mkdir(exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": family, "architectures": [head]}))
        # Presence-only hints; deliberately not valid weights/tokenizer: no loader may run.
        (path / "model.safetensors").write_bytes(b"not parsed by discovery")
        (path / "tokenizer.json").write_text("not parsed by discovery")

    def test_empty_explicit_browse_offers_locations_without_loading_owner(self):
        status, result = self.call({"path": ""})
        self.assertEqual(status, 200, result)
        self.assertEqual(result.get("mode"), "locations")
        self.assertEqual(result.get("folders"), [])
        self.assertIsNone(result.get("current"))
        self.assertTrue(result.get("locations"))

    def test_current_checkpoint_and_child_classification_use_real_files(self):
        self.checkpoint(self.root)
        child = self.root / "unsupported"
        self.checkpoint(child, family="qwen2", head="Qwen2ForCausalLM")
        status, result = self.call({"path": str(self.root)})
        self.assertEqual(status, 200, result)
        self.assertEqual(result.get("mode"), "directory")
        self.assertEqual(result["current"]["candidate"]["family"], "llama")
        self.assertEqual(result["current"]["candidate"]["status"], "candidate")
        self.assertEqual(result["folders"][0]["candidate"]["status"], "unsupported")
        self.assertEqual(result["root"], str(self.root))

    def test_rejected_http_requests_never_enter_filesystem_browser(self):
        with mock.patch.object(self.state, "browse_models", side_effect=AssertionError("scanner reached")) as scanner:
            for kwargs in ({"token": False}, {"host": "evil.example"}, {"origin": "https://evil.example"}):
                status, _ = self.call({"path": str(self.root)}, **kwargs)
                self.assertEqual(status, 403)
            scanner.assert_not_called()

    def test_root_override_and_bad_input_leave_owner_and_capture_untouched(self):
        self.state.models_dir = self.root
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        for data in ({"path": str(outside), "root": str(outside)}, {"path": []}, {"root": 17}):
            status, _ = self.call(data)
            self.assertEqual(status, 400)
        status, result = self.call({"path": ""})
        self.assertEqual(status, 200, result)
        self.assertEqual(result["root"], str(self.root))

    def test_optional_status_discovery_failure_keeps_ready_model(self):
        self.state.models_dir = self.root
        for error in (PermissionError("inaccessible"), RuntimeError("symlink loop")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(self.state, "browse_models", side_effect=error):
                    try:
                        status, result = self.call(route="/api/status")
                    except http.client.RemoteDisconnected:
                        self.fail("Optional discovery error closed the status connection")
                self.assertEqual(status, 200, result)
                self.assertEqual(result["model"], self.owner.info())
                self.assertEqual(result["run"]["capture_id"], "kept-capture")
                self.assertTrue(result["model_discovery_warning"])


if __name__ == "__main__":
    unittest.main()
