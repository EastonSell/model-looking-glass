"""Authenticated download routes with actual manager and synthetic bytes/owner.

No registry, model weights, ML packages or GPU are used by these tests.
"""
from __future__ import annotations

import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "_mlg_download_http_fixture"


class Store:
    def __init__(self, *args, **kwargs): self.record = None
    def summary(self): return None if self.record is None else dict(self.record)
    def clear(self): raise AssertionError("Download must not clear retained captures")


class Reference(Store):
    max_bytes = 1024 * 1024


class Owner:
    def info(self): return {"family": "vit", "state": "ready", "name": "retained synthetic owner"}
    def capabilities(self): return {"devices": [{"id": "cpu", "available": True}], "cuda": {"available": False}}
    def capture_binding(self): return self
    def __getattr__(self, name): raise AssertionError("Download must not execute model operation: " + name)


class DownloadHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert not any(n == PREFIX or n.startswith(PREFIX + ".") for n in sys.modules)
        package = types.ModuleType(PREFIX); package.__path__ = [str(ROOT / "looking_glass")]
        storage = types.ModuleType(PREFIX + ".app_storage"); storage.TraceStore = Store
        reference = types.ModuleType(PREFIX + ".app_reference"); reference.ReferenceVector = Reference
        sys.modules.update({PREFIX: package, storage.__name__: storage, reference.__name__: reference})
        try:
            spec = importlib.util.spec_from_file_location(PREFIX + ".app", ROOT / "looking_glass/app.py")
            cls.app = importlib.util.module_from_spec(spec); sys.modules[spec.name] = cls.app
            spec.loader.exec_module(cls.app)
            cls.download = __import__(PREFIX + ".app_downloads", fromlist=["DownloadManager"])
        except BaseException:
            cls.tearDownClass(); raise

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == PREFIX or name.startswith(PREFIX + "."): del sys.modules[name]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mlg download HTTP "); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.owner = Owner()
        self.state = self.app.AppState(runtime=self.owner, export_root=self.root / "exports")
        self.state.store.record = {"run_id": "retained", "total_bytes": 16}
        self.state.experiment_store.record = {"run_id": "comparison", "total_bytes": 16}
        self.state._capture_id = "retained-id"; self.state._live_binding = self.owner
        self.before = self.snapshot(); self.transport_calls = []; self.release = threading.Event(); self.block = False
        self.spec = self.download.ModelSpec(id="test-model", repo="public/test-model", revision="a" * 40,
            title="Synthetic test checkpoint", license="apache-2.0", family="llama", float32_element_count=64,
            hidden_size=8, layers=2, attention_heads=2, intermediate_size=16, vocab_size=32, max_context=256,
            files=(self.download.FileSpec("config.json", 2, hashlib.sha256(b"{}").hexdigest()),))
        def transport(url):
            self.transport_calls.append(url)
            if self.block and not self.release.wait(5): raise TimeoutError("test release not received")
            return io.BytesIO(b"{}")
        self.state._download_specs = (self.spec,)
        self.state.downloads = self.download.DownloadManager(self.root / "models", self.state._download_specs, transport=transport)
        self.server = self.app.create_server(self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.addCleanup(self.stop)

    def snapshot(self):
        return (self.state.runtime, self.state.store, self.state.experiment_store, self.state.reference,
                self.state._capture_id, self.state._live_binding, self.state.store.summary(), self.state.experiment_store.summary())

    def stop(self):
        self.release.set()
        current = self.state.downloads.status()
        if current["job_id"]: self.state.downloads.cancel(current["job_id"])
        if self.state.downloads._thread: self.state.downloads._thread.join(5)
        self.server.shutdown(); self.thread.join(5); self.server.server_close()
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.before, self.snapshot())
        self.assertFalse((self.root / "exports").exists())

    def call(self, route, body=None, *, token=True, origin=None, server=None, state=None):
        port = (server or self.server).server_port; connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        headers = {"X-MLG-Token": (state or self.state).token} if token else {}
        if origin: headers["Origin"] = origin
        payload = None if body is None else json.dumps(body)
        if body is not None: headers["Content-Type"] = "application/json"
        try:
            connection.request("GET" if body is None else "POST", route, payload, headers)
            response = connection.getresponse(); return response.status, json.loads(response.read())
        finally: connection.close()

    def terminal(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status, result = self.call("/api/download")
            self.assertEqual(status, 200)
            if result["state"] not in {"running", "cancelling"}: return result
            time.sleep(.01)
        self.fail("Synthetic download did not reach a terminal state")

    def raw_get(self, path, *, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request("GET", path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_scene_and_fonts_are_exact_local_assets_with_security_headers(self):
        assets = {
            "/app_scene.js": "application/javascript; charset=utf-8",
            "/vendor/three-0.128.0.min.js": "application/javascript; charset=utf-8",
            "/vendor/OrbitControls-0.128.0.js": "application/javascript; charset=utf-8",
            **{f"/vendor/{font}-latin-{weight}-normal.woff2": "font/woff2"
               for font in ("inter", "jetbrains-mono") for weight in (400, 600)},
        }
        for path, mime in assets.items():
            with self.subTest(path=path):
                status, headers, data = self.raw_get(path)
                self.assertEqual(status, 200)
                self.assertEqual(data, (self.app.STATIC / path.lstrip("/")).read_bytes())
                self.assertEqual(headers["Content-Type"], mime)
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
                self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
        self.assertFalse(self.transport_calls)

    def test_scene_static_allowlist_does_not_expand_filesystem_or_origin_access(self):
        for path in ("/vendor/README.md", "/vendor/provenance.json", "/vendor/../app.py",
                     "/vendor/%2e%2e/app.py", "/app_scene.js/extra", "/vendor/missing.woff2"):
            with self.subTest(path=path):
                self.assertEqual(self.raw_get(path)[0], 404)
        self.assertEqual(self.raw_get("/app_scene.js", headers={"Host": "evil.invalid"})[0], 403)
        self.assertEqual(self.raw_get("/app_scene.js", headers={"Origin": "https://evil.invalid"})[0], 403)
        with mock.patch.object(self.app, "STATIC", self.root / "no-assets"):
            status, _, body = self.raw_get("/app_scene.js")
            self.assertEqual(status, 500)
            self.assertIn("reinstall the package", json.loads(body)["error"])
        self.assertFalse(self.transport_calls)

    def test_catalog_is_local_read_only_and_cpu_is_not_gpu_fit(self):
        code, data = self.call("/api/model-catalog?device=auto&context_tokens=128&show_all=0")
        self.assertEqual(code, 200); self.assertEqual(data["device"], "cpu")
        self.assertEqual(data["models"][0]["fit"]["status"], "cpu")
        self.assertEqual(data["models"][0]["revision"], self.spec.revision)
        self.assertEqual(data["download_root"], str(self.root / "models"))
        self.assertEqual(self.call("/api/browse-models", {"path": data["download_root"], "root": data["download_root"]})[0], 400)
        self.assertEqual(self.transport_calls, []); self.assertFalse((self.root / "models").exists())

    def test_download_requires_auth_explicit_curated_id_and_exact_revision(self):
        body = {"model_id": self.spec.id, "revision": self.spec.revision}
        self.assertEqual(self.call("/api/download", body, token=False)[0], 403)
        self.assertEqual(self.call("/api/download", body, origin="https://evil.invalid")[0], 403)
        for invalid in ({"model_id": self.spec.id}, {**body, "revision": "b" * 40},
                        {**body, "model_id": "other/repo"}, {**body, "url": "https://evil.invalid"},
                        {**body, "root": "/somewhere"}):
            self.assertEqual(self.call("/api/download", invalid)[0], 400)
        self.assertEqual(self.transport_calls, []); self.assertFalse((self.root / "models").exists())

    def test_explicit_start_verifies_bytes_without_selecting_or_loading(self):
        code, result = self.call("/api/download", {"model_id": self.spec.id, "revision": self.spec.revision})
        self.assertEqual(code, 200); self.assertTrue(result["job_id"])
        result = self.terminal(); self.assertEqual(result["state"], "complete")
        self.assertEqual(result["received_bytes"], 2); self.assertEqual(Path(result["path"], "config.json").read_bytes(), b"{}")
        self.assertEqual(len(self.transport_calls), 1)
        self.assertEqual(self.before, self.snapshot())

    def test_cancel_uses_current_job_identity_and_never_publishes_partial(self):
        self.block = True
        code, result = self.call("/api/download", {"model_id": self.spec.id, "revision": self.spec.revision})
        self.assertEqual(code, 200)
        self.assertEqual(self.call("/api/cancel-download", {"job_id": "old-job"})[0], 400)
        self.assertEqual(self.call("/api/download", {"model_id": self.spec.id, "revision": self.spec.revision})[0], 400)
        self.assertEqual(self.call("/api/cancel-download", {"job_id": result["job_id"]})[0], 200)
        self.release.set(); result = self.terminal()
        self.assertEqual(result["state"], "cancelled"); self.assertIsNone(result["path"])

    def test_live_cuda_snapshot_filters_only_known_unfit_with_override(self):
        caps = {"devices": [{"id": "cpu", "available": True}, {"id": "cuda:0", "available": True, "free_bytes": 1, "total_bytes": 1024**3}], "cuda": {"available": True}}
        with mock.patch.object(self.owner, "capabilities", return_value=caps):
            code, data = self.call("/api/model-catalog?device=auto")
            self.assertEqual(code, 200); self.assertEqual(data["device"], "cuda:0")
            self.assertEqual(data["models"], []); self.assertEqual(data["hidden_too_large"], 1)
            code, data = self.call("/api/model-catalog?device=cuda:0&show_all=1")
            self.assertEqual(code, 200); self.assertEqual(data["models"][0]["fit"]["status"], "too_large")
            caps["devices"][1].pop("free_bytes")
            code, data = self.call("/api/model-catalog?device=cuda:0")
            self.assertEqual(code, 200); self.assertEqual(data["models"][0]["fit"]["status"], "unknown")
        self.assertEqual(self.transport_calls, [])

    def test_catalog_and_status_require_auth_before_device_probe(self):
        with mock.patch.object(self.owner, "capabilities") as probe:
            self.assertEqual(self.call("/api/model-catalog", token=False)[0], 403)
            self.assertEqual(self.call("/api/model-catalog", origin="https://evil.invalid")[0], 403)
            self.assertEqual(self.call("/api/download", token=False)[0], 403)
            probe.assert_not_called()
        self.assertEqual(self.transport_calls, [])
        self.assertFalse((self.root / "models").exists())

    def test_stop_download_requests_cooperative_cancellation_only(self):
        self.block = True
        code, result = self.call("/api/download", {"model_id": self.spec.id, "revision": self.spec.revision})
        self.assertEqual(code, 200)
        self.state.stop_download()
        self.release.set()
        self.assertEqual(self.terminal()["state"], "cancelled")
        self.assertEqual(self.before, self.snapshot())
        self.state.stop_download()  # Terminal jobs need no second cancel request.

    def test_fresh_app_can_explicitly_browse_retained_download_without_redownloading(self):
        self.call("/api/download", {"model_id": self.spec.id, "revision": self.spec.revision})
        result = self.terminal(); self.assertEqual(result["state"], "complete")
        self.state.downloads._thread.join(5)
        model = Path(result["path"]); root = self.root / "models"
        original = (model / "config.json").read_bytes()
        fresh = self.app.AppState(runtime=Owner(), export_root=self.root / "fresh-exports", download_root=root)
        fresh._download_specs = (self.spec,)
        def no_download(url): raise AssertionError("Reuse must not download")
        fresh.downloads = self.download.DownloadManager(root, fresh._download_specs, transport=no_download)
        with self.app.create_server(fresh) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                call = lambda path, body=None: self.call(path, body, server=server, state=fresh)
                code, status = call("/api/download")
                self.assertEqual(code, 200); self.assertEqual(status["state"], "idle")
                code, catalog = call("/api/model-catalog?device=cpu")
                self.assertEqual(code, 200); self.assertEqual(catalog["download_root"], str(root))
                code, listing = call("/api/browse-models", {"path": str(root), "root": str(root)})
                self.assertEqual(code, 200)
                self.assertEqual([folder["path"] for folder in listing["folders"]], [str(model.parent)])
                code, listing = call("/api/browse-models", {"path": str(model.parent), "root": str(root)})
                self.assertEqual(code, 200)
                self.assertEqual([folder["path"] for folder in listing["folders"]], [str(model)])
                # Restart never turns a folder listing into verified compatibility or a model load.
                self.assertIsNone(fresh.store.summary()); self.assertEqual((model / "config.json").read_bytes(), original)
                restricted = self.root / "restricted"; restricted.mkdir(); fresh.models_dir = restricted
                self.assertEqual(call("/api/browse-models", {"path": str(root), "root": str(root)})[0], 400)
            finally:
                server.shutdown(); thread.join(5)
                self.assertFalse(thread.is_alive())
        self.assertEqual(len(self.transport_calls), 1)
        self.assertFalse((self.root / "fresh-exports").exists())

    def test_invalid_catalog_options_fail_without_fallback_or_files(self):
        for query in ("device=cuda:99", "device=cpu&device=auto", "show_all=maybe", "context_tokens=-1", "root=/tmp", "context_tokens=not-an-int"):
            self.assertEqual(self.call("/api/model-catalog?" + query)[0], 400, query)
        self.assertEqual(self.transport_calls, []); self.assertFalse((self.root / "models").exists())


if __name__ == "__main__": unittest.main()
