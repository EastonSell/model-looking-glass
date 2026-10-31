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


def test_global_search_preserves_capture_indices_and_ignores_private_data(tmp_path):
    store = TraceStore(tmp_path)
    captures = []
    for i in range(305):
        metadata = dict(module_path=f"layers.{i}", module_kind="Attention Block" if i in (202, 304) else "MLP", layer_index=i, app_forward_index=7, forward_index=9, prompt="private needle")
        captures.append(SimpleNamespace(value=torch.ones(1), metadata=lambda m=metadata: m))
    store.publish(dict(events=captures, prompt="private needle"))
    page = store.events(1, 1, "  ATTENTION\t Block ")
    assert [event["index"] for event in page["events"]] == [304]
    assert page["total"] == 2 and page["total_all"] == 305
    assert store.events(q="private needle")["total"] == 0
    assert store.events(q="304")["events"][0]["index"] == 304
    assert store.events(q="7")["total"] == 305
    assert store.events(q="9")["total"] == 305
    assert store.events(limit=200, q=" \n ")["total"] == 305
    assert len(store.events(limit=200)["events"]) == 200


@pytest.mark.parametrize("kwargs", [dict(q=None), dict(q=3), dict(q="x" * 257), dict(offset=-1), dict(limit=201), dict(limit=0)])
def test_event_search_bounds(tmp_path, kwargs):
    store = TraceStore(tmp_path); store.publish(result())
    with pytest.raises(ValueError): store.events(**kwargs)


def make_catalog_run(root, name, status='{"status":"complete"}'):
    run = root / name; run.mkdir(parents=True)
    (run / "status.json").write_text(status)
    (run / "manifest.json").write_text("not read or validated by catalog")
    (run / "tensors.safetensors").write_bytes(b"not hashed")
    return run


def test_saved_runs_read_only_nonexistent_and_unverified(tmp_path, monkeypatch):
    root = tmp_path / "exports"; store = TraceStore(root)
    assert store.saved_runs() == {"runs": [], "truncated": False}
    assert not root.exists()
    run = make_catalog_run(root, "one")
    before = {p: p.read_bytes() for p in run.iterdir()}
    import looking_glass.app_storage as storage
    monkeypatch.setattr(storage, "_digest", lambda _: pytest.fail("catalog must not hash tensors"))
    assert store.saved_runs() == {"runs": [dict(name="one", path=str(run), status="complete_unverified", tensor_file_bytes=10)], "truncated": False}
    assert {p: p.read_bytes() for p in run.iterdir()} == before


def test_catalog_symlinks_partial_invalid_and_no_recursion(tmp_path):
    root = tmp_path / "exports"; root.mkdir()
    outside = make_catalog_run(tmp_path, "private")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    for name, status in (("bad", "invalid"), ("partial", '{"status":"incomplete"}'), ("large", ' ' * 1025)):
        make_catalog_run(root, name, status)
    run = make_catalog_run(root, "symlink_file")
    (run / "tensors.safetensors").unlink()
    (run / "tensors.safetensors").symlink_to(outside / "tensors.safetensors")
    make_catalog_run(root / "partial", "nested")
    rows = TraceStore(root).saved_runs()["runs"]
    assert [r["name"] for r in rows] == ["bad", "large", "partial", "symlink_file"]
    assert all(r["status"] == "incomplete" for r in rows)
    assert rows[-1]["tensor_file_bytes"] is None
    assert all(Path(r["path"]).parent == root for r in rows)


def test_catalog_bounded_deterministic_and_invalid_root(tmp_path):
    for i in reversed(range(205)): (tmp_path / f"run_{i:03}").mkdir()
    catalog = TraceStore(tmp_path).saved_runs()
    assert catalog["truncated"] is True
    assert [r["name"] for r in catalog["runs"]] == [f"run_{i:03}" for i in range(200)]
    invalid = tmp_path / "not_directory"; invalid.write_text("x")
    with pytest.raises(OSError): TraceStore(invalid).saved_runs()


def test_catalog_refuses_replaced_root_symlink(tmp_path):
    root = tmp_path / "exports"; root.mkdir(); store = TraceStore(root)
    root.rmdir(); root.symlink_to(make_catalog_run(tmp_path, "outside"), target_is_directory=True)
    with pytest.raises(OSError): store.saved_runs()


def test_catalog_permission_error_is_not_empty_success(tmp_path, monkeypatch):
    import looking_glass.app_storage as storage
    store = TraceStore(tmp_path)
    original = storage.os.open
    def denied(path, flags, **kwargs):
        if path == store.export_root: raise PermissionError("Export root is not readable")
        return original(path, flags, **kwargs)
    monkeypatch.setattr(storage.os, "open", denied)
    with pytest.raises(PermissionError, match="not readable"): store.saved_runs()


def test_catalog_never_reads_manifest_or_tensor_and_does_not_verify_integrity(tmp_path, monkeypatch):
    import looking_glass.app_storage as storage
    run = make_catalog_run(tmp_path, "one")
    with (run / "manifest.json").open("wb") as stream: stream.truncate(64 * 1024 * 1024)
    original = storage.os.open
    def guarded(path, flags, **kwargs):
        assert str(path) not in ("manifest.json", "tensors.safetensors")
        return original(path, flags, **kwargs)
    monkeypatch.setattr(storage.os, "open", guarded)
    def no_path_read(*args, **kwargs): pytest.fail("catalog must not use unbounded Path reads")
    monkeypatch.setattr(Path, "read_text", no_path_read)
    monkeypatch.setattr(Path, "read_bytes", no_path_read)
    assert TraceStore(tmp_path).saved_runs()["runs"][0]["status"] == "complete_unverified"


def test_catalog_bounds_enumeration_even_when_entries_are_not_directories(tmp_path, monkeypatch):
    import looking_glass.app_storage as storage
    for i in range(510): (tmp_path / f"file_{i}").touch()
    original = storage.os.scandir
    seen = []
    class CountingScan:
        def __init__(self, fd): self.iterator = original(fd)
        def __enter__(self): return self
        def __exit__(self, *args): self.iterator.close()
        def __iter__(self): return self
        def __next__(self):
            item = next(self.iterator); seen.append(item.name); return item
    monkeypatch.setattr(storage.os, "scandir", CountingScan)
    assert TraceStore(tmp_path).saved_runs() == {"runs": [], "truncated": True}
    assert len(seen) == 501


def test_portable_catalog_same_bounded_read_only_contract(tmp_path, monkeypatch):
    import looking_glass.app_storage as storage
    root = tmp_path / "exports"; store = TraceStore(root)
    monkeypatch.delattr(storage.os, "O_NOFOLLOW")
    assert store.saved_runs() == {"runs": [], "truncated": False}
    assert not root.exists()
    make_catalog_run(root, "complete")
    make_catalog_run(root, "invalid", "bad json")
    outside = make_catalog_run(tmp_path, "outside")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    linked_file = make_catalog_run(root, "linked_file")
    (linked_file / "status.json").unlink()
    (linked_file / "status.json").symlink_to(outside / "status.json")
    rows = store.saved_runs()["runs"]
    assert [(r["name"], r["status"]) for r in rows] == [("complete", "complete_unverified"), ("invalid", "incomplete"), ("linked_file", "incomplete")]
    for i in range(510): (root / f"noise_{i}").touch()
    assert store.saved_runs()["truncated"] is True


def test_portable_catalog_rejects_replaced_root(tmp_path, monkeypatch):
    import looking_glass.app_storage as storage
    root = tmp_path / "root"; root.mkdir(); store = TraceStore(root)
    root.rmdir(); root.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.delattr(storage.os, "O_NOFOLLOW")
    with pytest.raises(ValueError, match="ordinary directory"): store.saved_runs()


def coordinate_result(family="gpt2", *, batches=None, positions=None, path=None, forward=0):
    batches = [0] if batches is None else batches
    positions = [0, 1] if positions is None else positions
    shape = [len(batches), len(positions), 4] if family in ("gpt2", "distilbert") else [4]
    metadata = dict(identity={"run_id": "step-a"}, app_forward_index=forward,
                    module_path=path or {"t5": "encoder.block.0", "gpt2": "transformer.h.0", "distilbert": "distilbert.transformer.layer.0", "vit": "vit.encoder.layer.0"}[family],
                    layer_index=0, batch_axis=0, token_axis=1, shape=[max(batches) + 1, 3 if family == "t5" and path and path.startswith("decoder.") else 5, 4], saved_shape=shape,
                    batch_positions=batches, token_positions=positions)
    return dict(model={"family": family}, tokens=["initial", "second", "third", "fourth", "fifth"],
                token_labels={str(i): {"piece": f"piece{i}", "text": f"text{i}", "truncated": False} for i in range(10)},
                steps=[dict(app_forward_index=forward, run_id="step-a", input_token_ids=[0, 1, 2, 3, 4], decoder_token_ids=[0, 8, 9], selected_token_id=9)],
                events=[SimpleNamespace(value=torch.arange(4 * (len(batches) * len(positions) if len(shape) == 3 else 1), dtype=torch.float32).reshape(shape), metadata=lambda: metadata)])


def test_coordinate_selected_reordered_vectors_and_roundtrip(tmp_path):
    store = TraceStore(tmp_path); store.publish(coordinate_result(positions=[4, 1, 3]))
    options = store.vectors(0, 1, 2)
    assert options["total"] == 3
    assert [(v["index"], v["coordinate"]["token_position"], v["coordinate"]["token_id"]) for v in options["vectors"]] == [(1, 1, 1), (2, 3, 3)]
    assert all(v["coordinate"]["status"] == "known" for v in options["vectors"])
    assert store.tensor(0, 2)["coordinate"] == options["vectors"][1]["coordinate"]
    saved = store.save("coordinates"); reopened = TraceStore(tmp_path); reopened.open(saved["path"])
    assert reopened.vectors(0) == store.vectors(0)
    assert reopened.tensor(0, 2) == store.tensor(0, 2)


@pytest.mark.parametrize("family,path,expected", [("distilbert", "distilbert.transformer.layer.0", 2), ("t5", "encoder.block.0", 2), ("t5", "decoder.block.0", 9)])
def test_coordinate_adapter_layouts(family, path, expected, tmp_path):
    store = TraceStore(tmp_path); store.publish(coordinate_result(family, positions=[2], path=path))
    coordinate = store.tensor(0)["coordinate"]
    assert coordinate["status"] == "known" and coordinate["token_id"] == expected
    assert coordinate["token_piece"] == f"piece{expected}"


@pytest.mark.parametrize("key,value", [("batch_axis", False), ("token_axis", True), ("app_forward_index", False), ("layer_index", True), ("saved_shape", [1, 2, 99]), ("shape", [1, 1, 4]), ("token_positions", [1, 1]), ("batch_positions", [True]), ("identity", {"run_id": "wrong"})])
def test_coordinate_malformed_metadata_is_unknown(tmp_path, key, value):
    data = coordinate_result(); data["events"][0].metadata()[key] = value
    store = TraceStore(tmp_path); store.publish(data)
    coordinate = store.tensor(0)["coordinate"]
    assert coordinate["status"] == "unknown" and coordinate["reason"]


def test_coordinate_duplicate_forward_and_nonzero_batch_unknown(tmp_path):
    data = coordinate_result(); data["steps"].append(dict(data["steps"][0]))
    store = TraceStore(tmp_path); store.publish(data)
    assert store.tensor(0)["coordinate"]["status"] == "unknown"
    store.publish(coordinate_result(batches=[1, 0]))
    assert store.tensor(0, 0)["coordinate"]["status"] == "unknown"
    assert store.tensor(0, 2)["coordinate"]["status"] == "unknown"
    assert store.tensor(0, 2)["raw_bits"] == ["0x41000000", "0x41100000", "0x41200000", "0x41300000"]


def test_coordinate_generation_input_not_selected_output_and_decoder_start(tmp_path):
    data = coordinate_result(positions=[2], forward=1)
    data["steps"][0]["input_token_ids"] = [0, 1, 7]
    data["events"][0].metadata()["shape"][1] = 3
    store = TraceStore(tmp_path); store.publish(data)
    assert store.tensor(0)["coordinate"]["token_id"] == 7  # Not selected output 9.
    store.publish(coordinate_result("t5", positions=[0], path="decoder.block.0"))
    assert store.tensor(0)["coordinate"]["token_id"] == 0


def test_coordinate_missing_labels_remain_grounded_without_decoding(tmp_path):
    data = coordinate_result(); del data["token_labels"]
    store = TraceStore(tmp_path); store.publish(data)
    coordinate = store.tensor(0)["coordinate"]
    assert coordinate["status"] == "known" and coordinate["token_id"] == 0
    assert coordinate["token_text"] is None
    assert coordinate["token_piece"] in (None, "initial")
    assert coordinate["reason"]
    data["token_labels"] = {"0": {"piece": "x" * 513, "text": "x", "truncated": False}}
    store.publish(data)
    assert store.tensor(0)["coordinate"]["token_piece"] is None


@pytest.mark.parametrize("position,image", [(0, dict(token_kind="cls", patch_row=None, patch_column=None, processed_pixel_box_xyxy=None)), (2, dict(token_kind="patch", patch_row=0, patch_column=1, processed_pixel_box_xyxy=[4, 0, 8, 4]))])
def test_coordinate_vit_image_not_tokens(tmp_path, position, image):
    data = coordinate_result("vit", positions=[position]); data["events"][0].metadata()["image_coordinate"] = image
    data["image_metadata"] = dict(grid_shape=[2, 2], patch_shape=[4, 4], sequence_length=5, cls_token_index=0)
    store = TraceStore(tmp_path); store.publish(data)
    coordinate = store.tensor(0)["coordinate"]
    assert coordinate["status"] == "known" and coordinate["stack"] == "image"
    assert coordinate["image_coordinate"] == image
    assert coordinate["token_id"] is None and coordinate["token_piece"] is None
    image["patch_row"] = True
    store.publish(data)
    assert store.tensor(0)["coordinate"]["status"] == "unknown"


@pytest.mark.parametrize("args", [(-1, 0, 1), (0, -1, 1), (0, 0, 201), (0, 0, 0), (True, 0, 1)])
def test_vector_option_bounds(tmp_path, args):
    store = TraceStore(tmp_path); store.publish(coordinate_result())
    with pytest.raises(ValueError): store.vectors(*args)


def test_coordinate_requires_input_sequence_shape_consistency(tmp_path):
    data = coordinate_result(); data["steps"][0]["input_token_ids"] = [0, 1]
    store = TraceStore(tmp_path); store.publish(data)
    assert store.tensor(0)["coordinate"]["status"] == "unknown"


def test_coordinate_vit_cross_checks_processed_geometry(tmp_path):
    data = coordinate_result("vit", positions=[2])
    data["image_metadata"] = dict(grid_shape=[2, 2], patch_shape=[4, 4], sequence_length=5, cls_token_index=0)
    data["events"][0].metadata()["image_coordinate"] = dict(token_kind="patch", patch_row=0, patch_column=1, processed_pixel_box_xyxy=[4, 0, 8, 4])
    store = TraceStore(tmp_path); store.publish(data)
    assert store.tensor(0)["coordinate"]["status"] == "known"
    data["events"][0].metadata()["image_coordinate"]["patch_column"] = 0
    store.publish(data)
    assert store.tensor(0)["coordinate"]["status"] == "unknown"


def test_coordinate_boolean_step_and_truncated_label(tmp_path):
    data = coordinate_result(); data["steps"][0]["app_forward_index"] = False
    store = TraceStore(tmp_path); store.publish(data)
    assert store.tensor(0)["coordinate"]["status"] == "unknown"
    data = coordinate_result(); data["token_labels"]["0"] = {"piece": "x" * 512, "text": "partial", "truncated": True}
    store.publish(data); coordinate = store.tensor(0)["coordinate"]
    assert coordinate["status"] == "known" and coordinate["token_label_truncated"] is True
    assert coordinate["reason"] == "Token display label is truncated"


def test_coordinate_rejects_module_layer_contradiction(tmp_path):
    data = coordinate_result(); data["events"][0].metadata()["module_path"] = "transformer.h.5.attn"
    store = TraceStore(tmp_path); store.publish(data)
    assert store.tensor(0)["coordinate"]["status"] == "unknown"


def test_coordinate_missing_vit_geometry_is_unknown(tmp_path):
    data = coordinate_result("vit", positions=[0])
    data["events"][0].metadata()["image_coordinate"] = dict(token_kind="cls", patch_row=None, patch_column=None, processed_pixel_box_xyxy=None)
    store = TraceStore(tmp_path); store.publish(data)
    assert store.tensor(0)["coordinate"]["status"] == "unknown"


def intervention_result(**kwargs):
    data = coordinate_result(**kwargs)
    data["steps"][0].update(input_attention_mask=[1] * len(data["steps"][0]["input_token_ids"]), use_cache=False)
    return data


def test_intervention_source_exact_selected_vector_clone_and_deepcopies(tmp_path):
    store = TraceStore(tmp_path); store.publish(intervention_result(positions=[4, 1, 3]))
    before = store.tensor(0, 1)
    source = store.intervention_source(0, 1)
    assert source["coordinate"] == before["coordinate"]
    assert source["coordinate"]["token_position"] == 1
    assert source["step"]["input_token_ids"] == [0, 1, 2, 3, 4]
    assert source["reference_value"].tolist() == before["values"]
    assert source["reference_value"].dtype == torch.float32
    assert source["reference_value"].device.type == "cpu"
    assert source["reference_value"].is_contiguous() and not source["reference_value"].requires_grad
    source["reference_value"].zero_()
    source["coordinate"]["token_position"] = 99
    source["step"]["input_token_ids"][0] = 99
    source["model"]["family"] = "changed"
    assert store.tensor(0, 1) == before
    again = store.intervention_source(0, 1)
    assert again["step"]["input_token_ids"][0] == 0 and again["model"]["family"] == "gpt2"
    assert again["reference_value"].tolist() == before["values"]


@pytest.mark.parametrize("mask", [None, [], [1], [1, 1, 1, 1, 2], [1, 1, True, 1, 1], [1., 1, 1, 1, 1], "11111"])
def test_intervention_source_invalid_masks_preserve_store(tmp_path, mask):
    data = intervention_result(); data["steps"][0]["input_attention_mask"] = mask
    store = TraceStore(tmp_path); store.publish(data); before = store.tensor(0)
    with pytest.raises(ValueError, match="mask"): store.intervention_source(0, 0)
    assert store.tensor(0) == before


@pytest.mark.parametrize("mutation", ["missing_mask", "cache_true", "cache_zero", "cache_missing", "unknown", "nongpt2", "ambiguous"])
def test_intervention_source_requires_uncached_known_gpt2_provenance(tmp_path, mutation):
    data = intervention_result()
    if mutation == "missing_mask": del data["steps"][0]["input_attention_mask"]
    elif mutation == "cache_true": data["steps"][0]["use_cache"] = True
    elif mutation == "cache_zero": data["steps"][0]["use_cache"] = 0
    elif mutation == "cache_missing": del data["steps"][0]["use_cache"]
    elif mutation == "unknown": data["events"][0].metadata()["token_positions"] = [1, 1]
    elif mutation == "nongpt2": data = intervention_result(family="distilbert", positions=[1])
    elif mutation == "ambiguous": data["steps"].append(dict(data["steps"][0]))
    store = TraceStore(tmp_path); store.publish(data); before = store.tensor(0)
    with pytest.raises(ValueError): store.intervention_source(0, 0)
    assert store.tensor(0) == before


@pytest.mark.parametrize("event,vector", [(-1, 0), (True, 0), (1, 0), (0, -1), (0, True), (0, 2)])
def test_intervention_source_bounds(tmp_path, event, vector):
    store = TraceStore(tmp_path); store.publish(intervention_result())
    with pytest.raises(ValueError): store.intervention_source(event, vector)


def test_intervention_source_requires_loaded_finite_values(tmp_path):
    store = TraceStore(tmp_path)
    with pytest.raises(ValueError): store.intervention_source(0, 0)
    data = intervention_result(); store.publish(data)
    data["events"][0].value[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"): store.intervention_source(0, 0)


def test_event_source_summary_distinguishes_single_and_multiple_vectors(tmp_path):
    store = TraceStore(tmp_path)
    store.publish(coordinate_result("distilbert", positions=[2]))
    row = store.events()["events"][0]
    assert row["matched_vector_index"] is None
    summary = row["source_summary"]
    assert summary["status"] == "known" and summary["vector_count"] == 1
    assert summary["coordinate"]["token_position"] == 2 and summary["coordinate"]["token_piece"] == "piece2"
    store.publish(coordinate_result(positions=[4, 1, 3]))
    row = store.events()["events"][0]
    assert row["source_summary"]["status"] == "multiple"
    assert row["source_summary"]["vector_count"] == 3 and row["source_summary"]["coordinate"] is None


def test_event_exact_filters_resolve_reordered_vector_and_survive_export(tmp_path):
    store = TraceStore(tmp_path); store.publish(coordinate_result(positions=[4, 1, 3], forward=2))
    page = store.events(token_position=1, forward=2, stack="input")
    assert page["total"] == 1 and page["total_all"] == 1
    row = page["events"][0]
    assert row["index"] == 0 and row["matched_vector_index"] == 1
    assert row["source_summary"]["coordinate"] == store.tensor(0, 1)["coordinate"]
    assert row["source_summary"]["vector_count"] == 3
    for filters in ({"token_position": 2}, {"forward": 1}, {"stack": "decoder"}):
        assert store.events(**filters)["total"] == 0
    first = store.events(forward=2)["events"][0]
    assert first["matched_vector_index"] == 0 and first["source_summary"]["coordinate"]["token_position"] == 4
    saved = store.save("source-filter"); store.clear(); store.open(saved["path"])
    assert store.events(token_position=1, forward=2, stack="input") == page
    manifest = json.loads((Path(saved["path"]) / "manifest.json").read_text())
    assert "source_summary" not in manifest["events"][0]  # View-only, not a schema migration.


def test_event_source_filter_paginates_original_event_indices(tmp_path):
    data = coordinate_result("distilbert", positions=[2])
    data["events"] = [coordinate_result("distilbert", positions=[position])["events"][0] for position in (1, 2, 1, 2, 3)]
    store = TraceStore(tmp_path); store.publish(data)
    page = store.events(offset=1, limit=1, q="distilbert", token_position=2)
    assert page["total"] == 2 and page["total_all"] == 5
    assert [row["index"] for row in page["events"]] == [3]
    assert page["events"][0]["matched_vector_index"] == 0


@pytest.mark.parametrize("kwargs", [{"token_position": True}, {"token_position": -1}, {"token_position": "1"},
    {"forward": False}, {"forward": -1}, {"forward": 1.0}, {"stack": ""}, {"stack": "vision"}, {"stack": []}])
def test_event_source_filters_reject_invalid_types_and_values(tmp_path, kwargs):
    store = TraceStore(tmp_path); store.publish(coordinate_result())
    with pytest.raises(ValueError): store.events(**kwargs)


@pytest.mark.parametrize("mutation", ["duplicates", "identity", "shape", "position_bool"])
def test_unknown_event_provenance_never_matches_source_filters(tmp_path, mutation):
    data = coordinate_result(positions=[1, 3])
    metadata = data["events"][0].metadata()
    if mutation == "duplicates": metadata["token_positions"] = [1, 1]
    elif mutation == "identity": metadata["identity"] = {"run_id": "wrong"}
    elif mutation == "shape": metadata["saved_shape"] = [1, 1, 4]
    else: metadata["token_positions"] = [True, 3]
    store = TraceStore(tmp_path); store.publish(data)
    row = store.events(q="transformer")["events"][0]
    assert row["source_summary"]["status"] == "unknown" and row["source_summary"]["reason"]
    assert store.events(token_position=1)["total"] == 0
    assert store.events(forward=0)["total"] == 0


@pytest.mark.parametrize("position,image", [(0, dict(token_kind="cls", patch_row=None, patch_column=None, processed_pixel_box_xyxy=None)),
    (2, dict(token_kind="patch", patch_row=0, patch_column=1, processed_pixel_box_xyxy=[4, 0, 8, 4]))])
def test_event_image_filters_keep_cls_and_processed_patch_coordinates(tmp_path, position, image):
    data = coordinate_result("vit", positions=[position]); data["events"][0].metadata()["image_coordinate"] = image
    data["image_metadata"] = dict(grid_shape=[2, 2], patch_shape=[4, 4], sequence_length=5, cls_token_index=0)
    store = TraceStore(tmp_path); store.publish(data)
    row = store.events(token_position=position, stack="image")["events"][0]
    assert row["source_summary"]["coordinate"]["image_coordinate"] == image
    assert row["matched_vector_index"] == 0
    assert store.events(token_position=position, stack="input")["total"] == 0


def test_unfiltered_source_summaries_resolve_only_the_returned_page(tmp_path, monkeypatch):
    data = coordinate_result("distilbert", positions=[2]); data["events"] *= 305
    store = TraceStore(tmp_path); store.publish(data)
    observed = []; original = store._coordinate
    def tracked(event, vector, **kwargs): observed.append((event, vector)); return original(event, vector, **kwargs)
    monkeypatch.setattr(store, "_coordinate", tracked)
    monkeypatch.setattr(store, "tensor", lambda *a, **k: pytest.fail("event metadata must not read tensor values"))
    assert len(store.events(offset=200, limit=3)["events"]) == 3
    assert observed == [(200, 0), (201, 0), (202, 0)]


@pytest.mark.parametrize("path,stack,token_id", [("encoder.block.0", "encoder", 2), ("decoder.block.0", "decoder", 9)])
def test_event_t5_stack_filters_do_not_conflate_input_and_decoder_ids(tmp_path, path, stack, token_id):
    store = TraceStore(tmp_path); store.publish(coordinate_result("t5", positions=[2], path=path))
    row = store.events(token_position=2, stack=stack)["events"][0]
    assert row["source_summary"]["coordinate"]["token_id"] == token_id
    assert store.events(token_position=2, stack="decoder" if stack == "encoder" else "encoder")["total"] == 0
    row["source_summary"]["coordinate"]["token_id"] = 999
    assert store.events(token_position=2, stack=stack)["events"][0]["source_summary"]["coordinate"]["token_id"] == token_id


def test_event_image_source_keeps_explicit_nonzero_batch_coordinate(tmp_path):
    data = coordinate_result("vit", positions=[0], batches=[1])
    data["events"][0].metadata()["image_coordinate"] = dict(token_kind="cls", patch_row=None, patch_column=None, processed_pixel_box_xyxy=None)
    data["image_metadata"] = dict(grid_shape=[2, 2], patch_shape=[4, 4], sequence_length=5, cls_token_index=0)
    store = TraceStore(tmp_path); store.publish(data)
    row = store.events(token_position=0, stack="image")["events"][0]
    assert row["source_summary"]["coordinate"]["batch_position"] == 1
    assert row["matched_vector_index"] == 0


@pytest.mark.parametrize("mutation", [None, "bad_ids", "duplicate_forward", "bad_step", "bad_forward", "wrong_length", "wrong_identity"])
def test_request_coordinate_context_matches_uncached_validation(tmp_path, mutation):
    data = coordinate_result(); data["events"] *= 3
    if mutation == "bad_ids": data["steps"][0]["input_token_ids"][-1] = True
    if mutation == "duplicate_forward": data["steps"] *= 2
    if mutation == "bad_step": data["steps"].append(None)
    if mutation == "bad_forward": data["steps"][0]["app_forward_index"] = True
    if mutation == "wrong_length": data["steps"][0]["input_token_ids"].append(5)
    if mutation == "wrong_identity": data["steps"][0]["run_id"] = "wrong"
    store = TraceStore(tmp_path); store.publish(data)
    context = {}
    for event in range(3):
        for vector in range(2):
            expected = store._coordinate(event, vector)
            assert store._coordinate(event, vector, _context=context) == expected
            assert expected["status"] == ("known" if mutation is None else "unknown")


def test_browse_context_is_shared_only_within_each_request(tmp_path, monkeypatch):
    data = coordinate_result(); data["events"] *= 4
    store = TraceStore(tmp_path); store.publish(data)
    original = store._coordinate; seen = []
    def tracked(event, vector, **kwargs):
        seen.append(kwargs.get("_context"))
        return original(event, vector, **kwargs)
    monkeypatch.setattr(store, "_coordinate", tracked)
    store.events(stack="input")
    assert len(seen) == 4 and seen[0] is not None and all(x is seen[0] for x in seen)
    first = seen[0]; seen.clear()
    store.events(stack="input")
    assert all(x is seen[0] for x in seen) and seen[0] is not first
    second = seen[0]; seen.clear()
    store.vectors(0)
    assert len(seen) == 2 and all(x is seen[0] for x in seen) and seen[0] is not second


def test_browse_revalidates_mutations_and_replaced_runs(tmp_path):
    store = TraceStore(tmp_path); store.publish(coordinate_result())
    saved = store.save("valid")
    assert store.events(stack="input")["total"] == 1
    store._run["steps"][0]["input_token_ids"][-1] = True
    assert store.events(stack="input")["total"] == 0
    assert store.vectors(0)["vectors"][0]["coordinate"]["status"] == "unknown"
    store.clear(); store.open(saved["path"])
    assert store.events(stack="input")["total"] == 1
    store._run["steps"].append(dict(store._run["steps"][0]))
    assert store.events(stack="input")["total"] == 0
    store.publish(coordinate_result())
    rows = store.events(stack="input")
    rows["events"][0]["source_summary"]["coordinate"]["token_id"] = 999
    assert store.tensor(0)["coordinate"]["token_id"] == 0


@pytest.mark.parametrize("family,path", [("distilbert", None), ("t5", "encoder.block.0"), ("t5", "decoder.block.0"), ("vit", None)])
def test_request_context_other_adapter_coordinates_match(tmp_path, family, path):
    data = coordinate_result(family, positions=[2], path=path)
    if family == "vit":
        data["events"][0].metadata()["image_coordinate"] = dict(token_kind="patch", patch_row=0, patch_column=1, processed_pixel_box_xyxy=[4, 0, 8, 4])
        data["image_metadata"] = dict(grid_shape=[2, 2], patch_shape=[4, 4], sequence_length=5, cls_token_index=0)
    store = TraceStore(tmp_path); store.publish(data)
    context = {}
    for _ in range(3):
        assert store._coordinate(0, 0, _context=context) == store._coordinate(0, 0)
    assert store._coordinate(0, 0, _context=context)["status"] == "known"


@pytest.mark.parametrize("mutation", ["source_length", "identity", "module", "axis", "positions"])
def test_shared_input_validation_never_skips_event_checks(tmp_path, mutation):
    data = coordinate_result(); data["events"] *= 2
    store = TraceStore(tmp_path); store.publish(data)
    second = store._events[1]["metadata"]
    if mutation == "source_length": second["shape"][1] += 1
    if mutation == "identity": second["identity"]["run_id"] = "different"
    if mutation == "module": second["module_path"] = "transformer.h.9"
    if mutation == "axis": second["batch_axis"] = True
    if mutation == "positions": second["token_positions"] = [0, 0]
    context = {}
    assert store._coordinate(0, 0, _context=context)["status"] == "known"
    assert store._coordinate(1, 0, _context=context) == store._coordinate(1, 0)
    assert store._coordinate(1, 0, _context=context)["status"] == "unknown"


def test_request_input_context_is_bounded_by_steps_not_events(tmp_path):
    data = coordinate_result("t5", positions=[2], path="encoder.block.0")
    decoder = coordinate_result("t5", positions=[2], path="decoder.block.0")["events"][0]
    data["events"] = (data["events"] + [decoder]) * 150
    store = TraceStore(tmp_path); store.publish(data)
    context = {}
    for index in range(300):
        coordinate = store._coordinate(index, 0, _context=context)
        assert coordinate["token_id"] == (2 if index % 2 == 0 else 9)
    assert len(context["steps_lookup"]) == 1
    assert len(context["token_ids"]) == 2
