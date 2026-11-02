"""Known historical study coordinates resolve to real current ViT modules."""
import pytest
from transformers import ViTConfig, ViTForImageClassification
from looking_glass.vision_coco_study import EARLY, LATE, _runtime_site


def test_current_vit_study_coordinates_preserve_block_indices():
    config = ViTConfig(image_size=16, patch_size=8, hidden_size=8,
                       intermediate_size=12, num_attention_heads=2, num_hidden_layers=12)
    config._attn_implementation = "eager"
    model = ViTForImageClassification(config).eval()
    assert _runtime_site(model, EARLY) == "vit.layers.2"
    assert _runtime_site(model, LATE) == "vit.layers.11"
    assert model.get_submodule(_runtime_site(model, EARLY)) is model.vit.layers[2]
    with pytest.raises(AttributeError):
        _runtime_site(model, "vit.encoder.layer.7")
