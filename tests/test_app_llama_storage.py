"""Recorded Llama coordinates stay exact without a loaded model or tokenizer."""
from types import SimpleNamespace

import pytest
import torch

from looking_glass.app_storage import TraceStore


def llama_result(*, suffix="", positions=None):
    positions = [3, 0, 2] if positions is None else positions
    # Include signed zero and a subnormal: persistence must preserve all bits.
    bits = torch.tensor([0, -2147483648, 1, 1065353216] * len(positions), dtype=torch.int32)
    value = bits.view(torch.float32).reshape(1, len(positions), 4)
    metadata = dict(identity={"run_id": "llama-forward-7"}, app_forward_index=7,
                    module_path="model.layers.2" + suffix, layer_index=2,
                    batch_axis=0, token_axis=1, shape=[1, 4, 4], saved_shape=list(value.shape),
                    batch_positions=[0], token_positions=positions,
                    output_path=[0] if suffix == ".self_attn" else [])
    return dict(model={"family": "llama"},
                steps=[dict(app_forward_index=7, run_id="llama-forward-7", input_token_ids=[10, 11, 12, 13],
                            input_attention_mask=[1, 1, 1, 1], use_cache=False, selected_token_id=99)],
                token_labels={str(i): dict(piece=f"piece{i}", text=f"text{i}", truncated=False) for i in range(10, 14)},
                events=[SimpleNamespace(value=value, metadata=lambda: metadata)])


@pytest.mark.parametrize("suffix", ["", ".self_attn", ".mlp"])
def test_llama_recorded_coordinates_filter_reordered_vectors_and_roundtrip(tmp_path, suffix):
    store = TraceStore(tmp_path); store.publish(llama_result(suffix=suffix))
    assert store.events()["events"][0]["source_summary"]["status"] == "multiple"
    vectors = store.vectors(0)
    assert [v["coordinate"]["token_position"] for v in vectors["vectors"]] == [3, 0, 2]
    assert [v["coordinate"]["token_id"] for v in vectors["vectors"]] == [13, 10, 12]
    for vector in range(3):
        coordinate = store.tensor(0, vector)["coordinate"]
        assert coordinate["status"] == "known" and coordinate["stack"] == "input"
        assert coordinate["app_forward_index"] == 7 and coordinate["layer_index"] == 2
        assert coordinate["token_piece"] == f"piece{coordinate['token_id']}"
    filtered = store.events(token_position=0, forward=7, stack="input")
    assert filtered["total"] == 1 and filtered["events"][0]["index"] == 0
    assert filtered["events"][0]["matched_vector_index"] == 1
    assert store.events(token_position=1)["total"] == 0
    assert store.events(stack="decoder")["total"] == 0
    expected = [store.tensor(0, vector) for vector in range(3)]
    saved = store.save("llama"); reopened = TraceStore(tmp_path); reopened.open(saved["path"])
    assert reopened.events(token_position=0, forward=7, stack="input") == filtered
    assert reopened.vectors(0) == vectors
    assert [reopened.tensor(0, vector) for vector in range(3)] == expected
    assert expected[0]["raw_bits"] == ["0x00000000", "0x80000000", "0x00000001", "0x3f800000"]
    source = reopened.intervention_source(0, 0)
    assert source["coordinate"] == expected[0]["coordinate"]
    assert source["step"]["input_token_ids"] == [10, 11, 12, 13]
    assert torch.equal(source["reference_value"].view(torch.int32), torch.tensor([0, -2147483648, 1, 1065353216], dtype=torch.int32))
    # Extraction validates recorded data only; server private binding separately
    # rejects imported captures as live intervention authority.
    source["reference_value"].zero_()
    assert reopened.tensor(0, 0) == expected[0]


@pytest.mark.parametrize("mutation", ["foreign_path", "wrong_layer", "extra_suffix", "wrong_layout", "wrong_axes", "wrong_forward", "duplicate_forward", "wrong_run_id", "bad_ids", "short_ids", "duplicate_positions", "boolean_position", "batch", "unknown_family"])
def test_llama_conflicting_recorded_provenance_stays_unknown(tmp_path, mutation):
    data = llama_result(); metadata = data["events"][0].metadata()
    if mutation == "foreign_path": metadata["module_path"] = "transformer.h.2"
    if mutation == "wrong_layer": metadata["layer_index"] = 1
    if mutation == "extra_suffix": metadata["module_path"] += ".self_attn.q_proj"
    if mutation == "wrong_layout": metadata["saved_shape"] = [3, 4]
    if mutation == "wrong_axes": metadata["token_axis"] = 0
    if mutation == "wrong_forward": metadata["app_forward_index"] = 8
    if mutation == "duplicate_forward": data["steps"] *= 2
    if mutation == "wrong_run_id": metadata["identity"]["run_id"] = "different"
    if mutation == "bad_ids": data["steps"][0]["input_token_ids"][-1] = True
    if mutation == "short_ids": data["steps"][0]["input_token_ids"].pop()
    if mutation == "duplicate_positions": metadata["token_positions"] = [3, 0, 3]
    if mutation == "boolean_position": metadata["token_positions"][1] = False
    if mutation == "batch": metadata["shape"][0] = 2; metadata["batch_positions"] = [1]
    if mutation == "unknown_family": data["model"]["family"] = "custom_llama"
    store = TraceStore(tmp_path); store.publish(data)
    assert store.tensor(0)["coordinate"]["status"] == "unknown"
    assert store.events()["events"][0]["source_summary"]["status"] == "unknown"
    assert store.events(token_position=3, stack="input")["total"] == 0
    assert store.events(q="2")["total"] == 1
    with pytest.raises(ValueError, match="known GPT2 or Llama"):
        store.intervention_source(0, 0)


@pytest.mark.parametrize("suffix,output_path", [(".self_attn", []), (".self_attn", [False]), (".self_attn", [1]), (".self_attn", None), ("", [0]), (".mlp", [0]), (".mlp", None)])
def test_llama_output_selector_must_match_exact_supported_boundary(tmp_path, suffix, output_path):
    data = llama_result(suffix=suffix)
    data["events"][0].metadata()["output_path"] = output_path
    store = TraceStore(tmp_path); store.publish(data)
    assert store.tensor(0)["raw_bits"] == ["0x00000000", "0x80000000", "0x00000001", "0x3f800000"]
    assert store.tensor(0)["coordinate"]["status"] == "unknown"
    assert store.events(stack="input")["total"] == 0
