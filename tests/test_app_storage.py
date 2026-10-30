"""Lossless trace persistence and bounded, explicitly approximate presentation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from looking_glass.app_storage import TraceStore


class Capture:
    def __init__(self, value): self.value = value
    def metadata(self): return {"module_path": "layer.0", "forward_index": 0, "token_positions": [0, 1]}


def result(value=None):
    if value is None: value = torch.tensor([[[0., -0., 1., -2.], [3., 4., 5., 6.]]])
    return {"answer": "test", "task": "next_token", "events": [Capture(value)], "steps": [{"input_ids": [2, 3]}], "tokens": ["a", "b"], "coverage": "all supported points"}


def test_publish_raw_pagination_and_bits(tmp_path):
    store = TraceStore(tmp_path)
    report = store.publish(result())
    assert report["event_count"] == 1 and report["total_bytes"] == 32
    view = store.tensor(0, 0, 0, 2)
    assert view["values"] == [0., -0.]
    assert view["raw_bits"] == ["0x00000000", "0x80000000"]
    assert store.tensor(0, 1, 1, 2)["values"] == [4., 5.]
    assert list(tmp_path.iterdir()) == []  # RAM-only until explicit save.


def test_reduced_view_preserves_raw_and_labels_approximation(tmp_path):
    store = TraceStore(tmp_path); store.publish(result())
    before = store.tensor(0, 1, 0, 4)
    reduced = store.tensor(0, 1, 0, 2, "reduced")
    assert reduced["approximate"] is True
    assert reduced["bins"] == [{"start": 0, "end": 2, "mean": 3.5, "min": 3., "max": 4.}, {"start": 2, "end": 4, "mean": 5.5, "min": 5., "max": 6.}]
    assert store.tensor(0, 1, 0, 4) == before


def test_save_open_bitwise_and_never_overwrite(tmp_path):
    store = TraceStore(tmp_path); store.publish(result())
    out = store.save("example")
    reopened = TraceStore(tmp_path / "other"); reopened.open(out["path"])
    assert reopened.tensor(0, 0, 0, 4) == store.tensor(0, 0, 0, 4)
    assert reopened.summary()["answer"] == "test"
    with pytest.raises(FileExistsError): store.save("example")
    with pytest.raises(ValueError): store.save("../escape")
    assert not (tmp_path.parent / "escape").exists()


def test_corrupt_export_rejected_without_replacing_current_run(tmp_path):
    store = TraceStore(tmp_path); store.publish(result()); saved = store.save("one")
    p = Path(saved["path"]) / "tensors.safetensors"
    data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
    with pytest.raises(ValueError, match="digest"): store.open(saved["path"])
    assert store.summary()["answer"] == "test"


def test_invalid_capture_and_limits_do_not_publish(tmp_path):
    store = TraceStore(tmp_path, max_bytes=16)
    with pytest.raises(ValueError): store.publish(result())
    assert store.summary() is None
    with pytest.raises(ValueError): store.publish(result(torch.tensor([float("nan")])) )
    with pytest.raises(ValueError): store.publish(result(torch.ones(2, dtype=torch.float64)))


@pytest.mark.parametrize("args", [(-1, 0, 0, 2), (0, -1, 0, 2), (0, 2, 0, 2), (0, 0, -1, 2), (0, 0, 0, 0), (0, 0, 0, 4097)])
def test_presentation_bounds(tmp_path, args):
    store = TraceStore(tmp_path); store.publish(result())
    with pytest.raises(ValueError): store.tensor(*args)


def test_open_rejects_wrong_shapes_and_pickle_paths(tmp_path):
    store = TraceStore(tmp_path); store.publish(result()); saved = store.save("one")
    p = Path(saved["path"]) / "manifest.json"; manifest = json.loads(p.read_text())
    manifest["events"][0]["shape"] = [999]
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shape"): TraceStore(tmp_path).open(saved["path"])
    manifest["tensor_file"] = "../bad.pkl"; p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): TraceStore(tmp_path).open(saved["path"])


def test_clear_drops_run_without_writing(tmp_path):
    store = TraceStore(tmp_path); store.publish(result()); store.clear()
    assert store.summary() is None
    with pytest.raises(ValueError): store.events()
    assert list(tmp_path.iterdir()) == []
