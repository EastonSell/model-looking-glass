"""Malformed saved captures are input errors, not failed model executions."""
import json
from pathlib import Path

import pytest

from looking_glass.app_storage import TraceStore
from test_app_http import call, server
from test_app_storage import result


def damaged_manifest(path, location, value, entry=0):
    manifest_path = Path(path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if location == "manifest":
        manifest = value
    elif location == "entry":
        manifest["events"][entry] = value
    elif location == "metadata":
        manifest["events"][entry]["metadata"] = value
    else:
        manifest["run"] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("location", ["manifest", "entry", "metadata", "run"])
@pytest.mark.parametrize("value", [None, [], "broken", 1, True])
def test_malformed_export_objects_are_validation_errors_without_replacing_trace(tmp_path, location, value):
    store = TraceStore(tmp_path)
    store.publish(result())
    saved = store.save("damaged")
    before, vector = store.summary(), store.tensor(0)
    damaged_manifest(saved["path"], location, value)

    with pytest.raises(ValueError, match="object"):
        store.open(saved["path"])

    assert store.summary() == before
    assert store.tensor(0) == vector


@pytest.mark.parametrize("location,entry", [("entry", 0), ("entry", 1), ("metadata", 0), ("metadata", 1), ("run", 0)])
def test_invalid_metadata_is_rejected_before_reading_tensor_payload(tmp_path, monkeypatch, location, entry):
    import looking_glass.app_storage as storage

    store = TraceStore(tmp_path)
    captures = result()
    captures["events"].extend(result()["events"])
    store.publish(captures)
    saved = store.save("damaged")
    damaged_manifest(saved["path"], location, None, entry=entry)

    def unexpected_payload_read(*args, **kwargs):
        pytest.fail("Invalid JSON structure must be rejected before hashing/loading tensor payload")

    monkeypatch.setattr(storage, "_digest", unexpected_payload_read)
    with pytest.raises(ValueError, match="object"):
        store.open(saved["path"])


@pytest.mark.parametrize("location", ["manifest", "entry", "metadata", "run"])
def test_open_bad_metadata_returns_actionable_400_and_keeps_capture_readable(server, location):
    code, body, _ = call(server, "/api/run", "POST", {"prompt": "synthetic retained capture", "settings": {}})
    assert code == 200
    original = json.loads(body)
    capture = original["capture_id"]
    tensor_url = f"/api/tensor?event=0&vector=0&capture_id={capture}"
    original_tensor = json.loads(call(server, tensor_url)[1])
    code, body, _ = call(server, "/api/save", "POST", {"name": "damaged", "capture_id": capture})
    assert code == 200
    saved = json.loads(body)
    damaged_manifest(saved["path"], location, None)

    code, body, _ = call(server, "/api/open", "POST", {"path": saved["path"]})

    assert code == 400
    assert "object" in json.loads(body)["error"]
    assert "Run failed" not in json.loads(body)["error"]
    assert json.loads(call(server)[1])["run"] == original
    assert json.loads(call(server, tensor_url)[1]) == original_tensor
