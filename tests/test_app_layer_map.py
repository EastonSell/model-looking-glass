"""Layer maps summarize only grounded capture metadata, never inferred topology."""
from types import SimpleNamespace

import pytest
import torch

from looking_glass.app_storage import TraceStore


def fixture(family="gpt2", layers=(0, 2), positions=(0, 1)):
    records = []
    stacks = ("encoder", "decoder") if family == "t5" else ("input",)
    for stack in stacks:
        for layer in layers:
            if family == "gpt2":
                sites = [(f"transformer.h.{layer}", "block", [0]), (f"transformer.h.{layer}.attn", "attn", [0]), (f"transformer.h.{layer}.mlp", "mlp", [])]
            elif family == "distilbert":
                sites = [(f"distilbert.transformer.layer.{layer}", "block", [0]), (f"distilbert.transformer.layer.{layer}.attention", "attention", [0]), (f"distilbert.transformer.layer.{layer}.ffn", "ffn", [])]
            elif family == "llama":
                sites = [(f"model.layers.{layer}", "llama_block", []), (f"model.layers.{layer}.self_attn", "llama_attention", [0]), (f"model.layers.{layer}.mlp", "llama_mlp", [])]
            elif family == "vit":
                sites = [(f"vit.encoder.layer.{layer}", "vit_block", []), (f"vit.encoder.layer.{layer}.attention", "vit_attention", []), (f"vit.encoder.layer.{layer}.intermediate", "vit_intermediate", [])]
            else:
                sites = [(f"{stack}.block.{layer}", f"t5_{stack}_block_output", [0]), (f"{stack}.block.{layer}.layer.0", f"t5_{stack}_self_attention_residual", [0]), (f"{stack}.block.{layer}.layer.{1 if stack == 'encoder' else 2}", f"t5_{stack}_feed_forward_residual", [])]
            for path, kind, selector in sites:
                for selected in (list(positions),) if family in ("gpt2", "llama") else ([p] for p in positions):
                    shape = [1, len(selected), 4] if family in ("gpt2", "distilbert", "llama") else [4]
                    value = torch.arange(4 * len(selected), dtype=torch.float32).reshape(shape)
                    metadata = dict(module_path=path, module_kind=kind, output_path=selector, layer_index=layer,
                                    identity={"run_id": "step0"}, app_forward_index=0, shape=[1, 3, 4], saved_shape=shape,
                                    batch_axis=0, token_axis=1, batch_positions=[0], token_positions=selected)
                    if family == "vit":
                        p = selected[0]
                        metadata["image_coordinate"] = dict(token_kind="cls" if p == 0 else "patch", patch_row=None if p == 0 else 0, patch_column=None if p == 0 else p-1, processed_pixel_box_xyxy=None if p == 0 else [(p-1)*4, 0, p*4, 4])
                    records.append(SimpleNamespace(value=value, metadata=lambda m=metadata: m))
    return dict(model={"family": family}, events=records, steps=[dict(app_forward_index=0, run_id="step0", input_token_ids=[4, 5, 6], decoder_token_ids=[0, 7, 8])], image_metadata=dict(grid_shape=[1, 2], patch_shape=[4, 4], sequence_length=3, cls_token_index=0))


@pytest.mark.parametrize("family", ["gpt2", "distilbert", "t5", "vit", "llama"])
def test_map_groups_actual_sites_and_counts_without_inventing_missing_layer(tmp_path, family):
    data = fixture(family); store = TraceStore(tmp_path); store.publish(data)
    view = store.layer_map()
    assert view["total_layers"] == (4 if family == "t5" else 2)
    assert view["known_events"] == view["selected_forward_events"] == view["total_all_events"] == len(data["events"])
    assert view["unmapped_events"] == view["other_forward_events"] == view["missing_forward_events"] == 0
    assert {layer["layer_index"] for layer in view["layers"]} == {0, 2}
    assert all(len(layer["nodes"]) == 3 for layer in view["layers"])
    nodes = [node for layer in view["layers"] for node in layer["nodes"]]
    assert sum(n["event_count"] for n in nodes) == len(data["events"])
    assert sum(n["tensor_bytes"] for n in nodes) == store.summary()["total_bytes"]
    assert sum(n["vector_count"] for n in nodes) == sum(e["vector_count"] for e in store._events)
    for node in nodes:
        row = node["representative_event"]
        assert row["matched_vector_index"] == 0
        if row["vector_count"] > 1:
            assert row["source_summary"]["status"] == "multiple" and row["source_summary"]["coordinate"] is None
        else:
            assert row["source_summary"]["coordinate"] == store.tensor(row["index"], 0)["coordinate"]
    assert "not" in view["scope_note"]


def test_map_filters_forward_counts_invalid_and_unknown_separately(tmp_path):
    data = fixture(layers=(0,)); records = data["events"]
    records[0].metadata()["app_forward_index"] = 1
    records[1].metadata()["app_forward_index"] = True
    records[2].metadata()["output_path"] = [False]
    store = TraceStore(tmp_path); store.publish(data)
    view = store.layer_map()
    assert view["total_all_events"] == 3 and view["selected_forward_events"] == 1
    assert view["other_forward_events"] == view["missing_forward_events"] == view["unmapped_events"] == 1
    assert view["known_events"] == view["total_layers"] == 0
    assert store.events()["total"] == 3


@pytest.mark.parametrize("key,value", [("module_kind", "wrong"), ("module_path", "transformer.h.9"), ("output_path", [1]), ("output_path", None), ("shape", [1, 5, 4]), ("token_positions", [0, 0])])
def test_map_rejects_conflicting_metadata_but_preserves_raw_table(tmp_path, key, value):
    data = fixture(layers=(0,)); data["events"][0].metadata()[key] = value
    store = TraceStore(tmp_path); store.publish(data)
    view = store.layer_map()
    assert view["unmapped_events"] == 1 and view["known_events"] == 2
    assert store.tensor(0)["values"] == [0., 1., 2., 3.]


def test_map_metadata_only_pagination_roundtrip_and_response_independence(tmp_path, monkeypatch):
    store = TraceStore(tmp_path); store.publish(fixture(layers=(0, 2, 7)))
    saved = store.save("map")
    page = store.layer_map(offset=1, limit=1)
    assert page["total_layers"] == 3 and page["layers"][0]["layer_index"] == 2
    assert {n["representative_event"]["index"] for n in page["layers"][0]["nodes"]} == {3, 4, 5}
    reopened = TraceStore(tmp_path); reopened.open(saved["path"])
    assert reopened.layer_map(offset=1, limit=1) == page
    monkeypatch.setattr(store, "_values", None)
    monkeypatch.setattr(store, "tensor", lambda *a, **k: pytest.fail("map must not read tensor values"))
    assert store.layer_map(offset=1, limit=1) == page
    page["layers"][0]["nodes"][0]["representative_event"]["metadata"]["layer_index"] = 99
    assert store.layer_map(offset=1, limit=1)["layers"][0]["nodes"][0]["representative_event"]["metadata"]["layer_index"] == 2
    assert store.layer_map(offset=3)["layers"] == []


@pytest.mark.parametrize("kwargs", [{"forward": True}, {"forward": -1}, {"forward": "0"}, {"offset": True}, {"offset": -1}, {"offset": 99}, {"limit": True}, {"limit": 0}, {"limit": 51}])
def test_map_argument_bounds(tmp_path, kwargs):
    store = TraceStore(tmp_path); store.publish(fixture())
    with pytest.raises(ValueError): store.layer_map(**kwargs)


def test_gpt2_recorded_selector_variants_are_separate_bounded_nodes(tmp_path):
    data = fixture(layers=(0,))
    variants = fixture(layers=(0,))["events"]
    for capture in variants:
        metadata = capture.metadata()
        metadata["output_path"] = [] if metadata["output_path"] else [0]
    data["events"] += variants
    store = TraceStore(tmp_path); store.publish(data)
    view = store.layer_map()
    nodes = view["layers"][0]["nodes"]
    assert len(nodes) == 6 and view["known_events"] == 6
    assert len({(n["module_path"], tuple(n["output_path"])) for n in nodes}) == 6
    assert {n["representative_event"]["index"] for n in nodes} == set(range(6))


@pytest.mark.parametrize("family", ["distilbert", "t5", "vit", "llama"])
def test_non_gpt2_selector_variants_are_unmapped_not_coalesced(tmp_path, family):
    data = fixture(family, layers=(0,))
    for capture in data["events"]:
        metadata = capture.metadata()
        metadata["output_path"] = [1]  # Neither current nor historical boundary output.
    store = TraceStore(tmp_path); store.publish(data)
    view = store.layer_map()
    assert view["known_events"] == view["total_layers"] == 0
    assert view["unmapped_events"] == len(data["events"])


def test_map_page_size_bound_and_no_synthetic_gaps(tmp_path):
    store = TraceStore(tmp_path); store.publish(fixture(layers=tuple(range(0, 104, 2))))
    view = store.layer_map(limit=50)
    assert view["total_layers"] == 52 and len(view["layers"]) == 50
    assert [layer["layer_index"] for layer in view["layers"]] == list(range(0, 100, 2))
    assert [layer["layer_index"] for layer in store.layer_map(offset=50)["layers"]] == [100, 102]
    assert store.layer_map(forward=9)["other_forward_events"] == 156
    assert store.layer_map(forward=9)["layers"] == []


def test_map_ambiguous_steps_remain_unmapped_and_fresh_requests_revalidate(tmp_path):
    store = TraceStore(tmp_path); store.publish(fixture(layers=(0,)))
    assert store.layer_map()["known_events"] == 3
    store._run["steps"] *= 2
    assert store.layer_map()["unmapped_events"] == 3
    store.clear()
    with pytest.raises(ValueError, match="No completed trace"): store.layer_map()
    store.publish(fixture(layers=(7,)))
    assert store.layer_map()["layers"][0]["layer_index"] == 7


def test_map_representatives_are_first_capture_not_cross_node_token_alignment(tmp_path):
    data = fixture("distilbert", layers=(0,), positions=(2, 0))
    # Different recorded ordering per site must remain visible, not normalized.
    data["events"][2:4] = list(reversed(data["events"][2:4]))
    store = TraceStore(tmp_path); store.publish(data)
    nodes = store.layer_map()["layers"][0]["nodes"]
    positions = {node["module_kind"]: node["representative_event"]["source_summary"]["coordinate"]["token_position"] for node in nodes}
    assert positions == {"block": 2, "attention": 0, "ffn": 2}
    assert all(node["event_count"] == 2 for node in nodes)
