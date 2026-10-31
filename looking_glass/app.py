"""Loopback-only Model Looking Glass workbench (no external web dependencies)."""
from __future__ import annotations

import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .app_storage import TraceStore

MAX_BODY = 8 * 1024 * 1024
MAX_IMAGE = 4 * 1024 * 1024
STATIC = Path(__file__).parent / "app_static"


class AppState:
    def __init__(self, *, runtime=None, export_root="looking-glass-runs", models_dir=None,
                 max_model_bytes=2 * 1024**3, max_trace_bytes=512 * 1024**2,
                 max_vram_bytes=64 * 1024**3):
        if runtime is None:
            # Application mode is explicitly local-only, even if user shell settings differ.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from .app_runtime import LocalModelRuntime
            runtime = LocalModelRuntime(max_model_bytes=max_model_bytes)
        self.runtime = runtime
        self.store = TraceStore(export_root, max_bytes=max_trace_bytes)
        self.experiment_store = TraceStore(export_root, max_bytes=max_trace_bytes)
        self._live_binding = None  # Process-local capability; never accepted from JSON/export.
        self._capture_id = None
        self._experiment_id = None
        self.lock = threading.Lock()
        self.models_dir = None if models_dir is None else Path(models_dir).expanduser().resolve()
        self.max_trace_bytes = max_trace_bytes
        if type(max_vram_bytes) is not int or not 1 <= max_vram_bytes <= 64 * 1024**3:
            raise ValueError("max_vram_bytes must be a positive integer <= 64 GiB")
        self.max_vram_bytes = max_vram_bytes
        self.token = secrets.token_hex(32)

    def status(self):
        models, discovery_warning = [], None
        if self.models_dir is not None:
            try:
                for folder in self.browse_models({"path": str(self.models_dir)})["folders"]:
                    if folder["is_checkpoint"]:
                        models.append({"name": folder["name"], "path": folder["path"]})
            except (OSError, ValueError):
                discovery_warning = "Configured model folder is unavailable. Runtime and captures are unchanged; use an accessible model folder."
        return {"model": self.runtime.info(), "run": self._run_summary(), "models": models,
                "experiment": self._experiment_summary(),
                "model_discovery_warning": discovery_warning,
                **self.runtime.capabilities(),
                "limits": {"max_trace_bytes": self.max_trace_bytes, "max_image_bytes": MAX_IMAGE,
                           "max_vram_bytes": self.max_vram_bytes},
                "privacy": "Local RAM only until Save; exports contain inputs, outputs and activations."}

    def _clear_experiment(self):
        self.experiment_store.clear()
        self._experiment_id = None

    def _experiment_summary(self):
        summary = self.experiment_store.summary()
        if summary is not None: summary["experiment_id"] = self._experiment_id
        return summary

    def _intervention_status(self):
        if self.store.summary() is None:
            return False, "Run a fresh GPT-2 capture first."
        model = self.runtime.info()
        if model is None or model.get("state") == "quarantined":
            return False, "A ready live model is required; unloading or quarantine disables experiments."
        if model.get("family") != "gpt2":
            return False, "Fixed-forward experiments currently support GPT-2 only."
        binding = getattr(self.runtime, "capture_binding", lambda: None)()
        if self._live_binding is None or binding is not self._live_binding:
            return False, "Run a fresh live capture from this model; imported or reloaded captures are read-only."
        return True, "Replay one recorded input prefix; runtime activations only, never checkpoint edits."

    def _run_summary(self):
        summary = self.store.summary()
        if summary is not None:
            summary["intervention_available"], summary["intervention_reason"] = self._intervention_status()
            summary["capture_id"] = self._capture_id
        return summary

    def browse_models(self, data):
        requested = _text(data, "path", 4096, empty=True).strip()
        root_text = _text(data, "root", 4096, empty=True).strip()
        if self.models_dir is not None:
            root = self.models_dir.resolve(strict=True)
        else:
            if not (root_text or requested):
                raise ValueError("Choose an explicit model folder, or start with --models-dir; no home/filesystem scan is automatic")
            root = Path(root_text or requested).expanduser().resolve(strict=True)
        path = Path(requested).expanduser().resolve(strict=True) if requested else root
        if not root.is_dir() or not path.is_dir():
            raise ValueError("Model browser needs a directory")
        if not path.is_relative_to(root):
            raise ValueError("Requested folder is outside the chosen model root")
        folders, truncated = [], False
        # Bounded discovery only: no recursive scan or model/weight parsing.
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index == 500:
                    truncated = True; break
                if entry.name.startswith(".") or entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    continue
                try: child = Path(entry.path).resolve(strict=True)
                except FileNotFoundError: continue
                if not child.is_relative_to(root):
                    continue
                config = child / "config.json"
                folders.append({"name": entry.name, "path": str(child),
                                "is_checkpoint": not config.is_symlink() and config.is_file()})
        folders.sort(key=lambda row: (row["name"].casefold(), row["name"]))
        if len(folders) > 200:
            folders = folders[:200]; truncated = True
        return {"root": str(root), "path": str(path),
                "parent": None if path == root else str(path.parent),
                "folders": folders, "truncated": truncated}

    def post(self, action, data):
        if not isinstance(data, dict): raise ValueError("Expected a JSON object")
        if action == "browse-models": return self.browse_models(data)
        if action == "unload":
            try: self.runtime.unload()
            finally: self._live_binding = None
            return self.status()
        if action == "inspect": return self.runtime.inspect(_text(data, "path", 4096))
        if action == "demo":
            if data:
                raise ValueError("The tiny demo takes no configuration; it is a fixed untrained CPU model")
            answer = self.runtime.load_demo()
            # Only a successful replacement retires the retained source/comparison.
            self.store.clear(); self._clear_experiment(); self._live_binding = None; self._capture_id = None
            return {"model": answer, "run": None}
        if action == "load":
            budget = data.get("max_vram_bytes", min(2 * 1024**3, self.max_vram_bytes))
            if type(budget) is not int or not 1 <= budget <= self.max_vram_bytes:
                raise ValueError("GPU allowance exceeds the server limit or is invalid")
            device = data.get("device", "auto")
            if type(device) is not str or not 1 <= len(device) <= 32:
                raise ValueError("device must be auto, cpu or cuda:N")
            answer = self.runtime.load(_text(data, "path", 4096), device=device, max_vram_bytes=budget)
            self.store.clear(); self._clear_experiment(); self._live_binding = None; self._capture_id = None
            return answer
        if action == "run":
            settings = data.get("settings", {})
            if not isinstance(settings, dict): raise ValueError("settings must be an object")
            budget = settings.get("max_trace_bytes", 128 * 1024**2)
            if type(budget) is not int or not 1 <= budget <= self.max_trace_bytes:
                raise ValueError("Trace allowance exceeds the server limit")
            image = None
            if data.get("image_base64"):
                encoded = data["image_base64"]
                if type(encoded) is not str: raise ValueError("image_base64 must be a base64 string")
                try: image = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError) as exc: raise ValueError("Invalid image encoding") from exc
                if len(image) > MAX_IMAGE: raise ValueError("Image exceeds 4 MiB")
            # A failed query leaves no previous trace masquerading as the new answer.
            self.store.clear(); self._clear_experiment(); self._live_binding = None; self._capture_id = None
            result = self.runtime.run(_text(data, "prompt", 100000, empty=True), settings, image_bytes=image)
            self.store.publish(result)
            self._capture_id = secrets.token_hex(16)
            self._live_binding = getattr(self.runtime, "capture_binding", lambda: None)()
            return self._run_summary()
        if action == "intervene":
            self._clear_experiment()  # Never present an earlier experiment as this attempt.
            available, reason = self._intervention_status()
            if not available: raise ValueError(reason)
            if type(data.get("capture_id")) is not str or data["capture_id"] != self._capture_id:
                raise ValueError("Capture selection is stale; refresh and select the current live capture")
            operation = data.get("operation")
            if operation not in ("zero", "add", "replace"):
                raise ValueError("Choose zero, add one feature, or replace one feature")
            remaining = min(128 * 1024**2, self.max_trace_bytes - self.store.summary()["total_bytes"])
            if remaining <= 0: raise ValueError("No remaining trace budget for a separate experiment")
            source = self.store.intervention_source(data.get("event"), data.get("vector"))
            result = self.runtime.intervene_fixed_forward(self._live_binding, source, operation,
                feature_index=data.get("feature_index"), value=data.get("value"), max_pair_bytes=remaining)
            try:
                summary = self.experiment_store.publish(result)
                if summary["total_bytes"] > remaining:
                    raise ValueError("Comparison witnesses exceed remaining trace allowance")
                self._experiment_id = secrets.token_hex(16)
                return self._experiment_summary()
            except BaseException:
                self._clear_experiment()
                raise
        if action == "save":
            if "capture_id" in data and (type(data["capture_id"]) is not str or data["capture_id"] != self._capture_id):
                raise ValueError("Capture selection is stale; refresh before saving")
            return self.store.save(_text(data, "name", 64))
        if action == "save-experiment":
            if type(data.get("experiment_id")) is not str or data["experiment_id"] != self._experiment_id:
                raise ValueError("Comparison selection is stale; refresh before saving")
            return self.experiment_store.save(_text(data, "name", 64))
        if action == "open":
            self.store.open(_text(data, "path", 4096))
            self._clear_experiment(); self._live_binding = None
            self._capture_id = secrets.token_hex(16)
            return self._run_summary()
        if action == "clear":
            self.store.clear(); self._clear_experiment(); self._live_binding = None; self._capture_id = None
            return self.status()
        raise ValueError("Unknown action")


def _text(data, key, limit, empty=False):
    value = data.get(key, "" if empty else None)
    if type(value) is not str or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(f"{key} must be {'a nonempty' if not empty else 'a'} string of at most {limit} characters")
    return value


def create_server(state, port=0):
    """Create, but do not start, an authenticated loopback server."""
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        server_version = "ModelLookingGlass"

        def log_message(self, *_):
            pass  # Never log model paths, prompts, tokens or request contents.

        def setup(self):
            super().setup(); self.connection.settimeout(30)

        def _send(self, status, body, mime="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass

        def _allowed(self, api=False):
            port = self.server.server_address[1]
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            hosts = self.headers.get_all("Host", [])
            if len(hosts) != 1 or hosts[0] not in allowed_hosts:
                self._send(403, {"error": "Invalid loopback Host"}); return False
            origins = self.headers.get_all("Origin", [])
            if len(origins) > 1 or (origins and origins[0] not in {f"http://{h}" for h in allowed_hosts}):
                self._send(403, {"error": "Cross-origin requests are not allowed"}); return False
            tokens = self.headers.get_all("X-MLG-Token", [])
            if api and (len(tokens) != 1 or not hmac.compare_digest(tokens[0].encode(), state.token.encode())):
                self._send(403, {"error": "Invalid local session token; reopen this app's page"}); return False
            return True

        def do_GET(self):
            parsed = urlsplit(self.path); path = parsed.path
            if not self._allowed(api=path.startswith("/api/")): return
            if path in ("/", "/app.js", "/app.css"):
                name, mime = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "application/javascript; charset=utf-8"), "/app.css": ("app.css", "text/css; charset=utf-8")}[path]
                try:
                    body = (STATIC / name).read_bytes()
                    if path == "/": body = body.replace(b"__MLG_TOKEN__", state.token.encode())
                    self._send(200, body, mime)
                except OSError: self._send(500, {"error": "Application assets are missing; reinstall the package"})
                return
            if not path.startswith("/api/"):
                self._send(404, {"error": "Not found"}); return
            try:
                query = parse_qs(parsed.query, strict_parsing=True)
                if any(len(values) != 1 for values in query.values()): raise ValueError("Duplicate query parameter")
                def number(key, default): return int(query.get(key, [str(default)])[0])
                def optional_number(key): return number(key, 0) if key in query else None
                with state.lock:
                    if path in {"/api/events", "/api/vectors", "/api/tensor"} and "capture_id" in query:
                        if query["capture_id"][0] != state._capture_id:
                            raise ValueError("Capture selection is stale; refresh before reading values")
                    if path in {"/api/experiment-events", "/api/experiment-tensor"}:
                        if query.get("experiment_id", [None])[0] is None or query["experiment_id"][0] != state._experiment_id:
                            raise ValueError("Comparison selection is stale; refresh before reading witnesses")
                    if path == "/api/status": answer = state.status()
                    elif path == "/api/saved": answer = state.store.saved_runs()
                    elif path == "/api/experiment": answer = state._experiment_summary()
                    elif path == "/api/experiment-events": answer = state.experiment_store.events(number("offset", 0), number("limit", 100))
                    elif path == "/api/experiment-tensor": answer = state.experiment_store.tensor(number("event", 0), number("vector", 0), number("offset", 0), number("limit", 256), "raw")
                    elif path == "/api/events": answer = state.store.events(
                        number("offset", 0), number("limit", 100), query.get("q", [""])[0],
                        token_position=optional_number("token_position"), forward=optional_number("forward"),
                        stack=query.get("stack", [None])[0])
                    elif path == "/api/vectors": answer = state.store.vectors(number("event", 0), number("offset", 0), number("limit", 100))
                    elif path == "/api/tensor": answer = state.store.tensor(number("event", 0), number("vector", 0), number("offset", 0), number("limit", 256), query.get("view", ["raw"])[0])
                    else:
                        self._send(404, {"error": "Not found"}); return
                self._send(200, answer)
            except (ValueError, OSError, TypeError, KeyError) as exc:
                self._send(400, {"error": str(exc)})

        def do_POST(self):
            if not self._allowed(api=True): return
            path = urlsplit(self.path).path
            if path not in {"/api/inspect", "/api/load", "/api/demo", "/api/unload", "/api/browse-models", "/api/run", "/api/intervene", "/api/save", "/api/save-experiment", "/api/open", "/api/clear"}:
                self._send(404, {"error": "Not found"}); return
            if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
                self._send(400, {"error": "A single bounded Content-Length is required"}); return
            if self.headers.get_content_type() != "application/json":
                self._send(415, {"error": "Use application/json"}); return
            try:
                length = int(self.headers["Content-Length"])
                if not 0 < length <= MAX_BODY:
                    self._send(413, {"error": "Request body exceeds the limit"}); return
                body = self.rfile.read(length)
                if len(body) != length: raise ValueError("Incomplete request body")
                data = json.loads(body)
                # One model owner and capture lifecycle at a time, including browse/save.
                with state.lock: answer = state.post(path.rsplit("/", 1)[1], data)
                self._send(200, answer)
            except (ValueError, OSError, TypeError, KeyError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"Run failed ({type(exc).__name__}): {exc}. No partial trace was published."})

        def do_OPTIONS(self): self._send(403, {"error": "Cross-origin access is disabled"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local CUDA/CPU model query and full supported-boundary activation inspector")
    parser.add_argument("--port", type=int, default=0, help="Loopback port; zero chooses a free port")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--models-dir", type=Path, help="Optional explicit directory whose immediate checkpoint subfolders appear in settings")
    parser.add_argument("--export-dir", type=Path, default=Path.cwd() / "looking-glass-runs")
    parser.add_argument("--max-model-mib", type=int, default=2048, help="Parameter-storage preflight cap, not peak RAM")
    parser.add_argument("--max-vram-mib", type=int, default=65536, help="Maximum selectable CUDA allocator allowance; not all GPU/driver memory")
    parser.add_argument("--max-trace-mib", type=int, default=512, help="Saved tensor payload cap, not peak RAM")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535 or not 1 <= args.max_model_mib <= 16384 or not 1 <= args.max_trace_mib <= 512 or not 1 <= args.max_vram_mib <= 65536:
        parser.error("Invalid port or memory allowance")
    state = AppState(export_root=args.export_dir, models_dir=args.models_dir,
                     max_model_bytes=args.max_model_mib * 1024**2, max_trace_bytes=args.max_trace_mib * 1024**2,
                     max_vram_bytes=args.max_vram_mib * 1024**2)
    server = create_server(state, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Model Looking Glass: {url}\nCUDA/CPU selection in Settings; local checkpoints; captures stay in CPU RAM until Save.\nPress Ctrl+C here to stop.", flush=True)
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        with state.lock: state.store.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
