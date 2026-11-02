"""A real local untied checkpoint keeps its tensor values and logits."""
import torch
import pytest
from safetensors.torch import save_file
from transformers import T5Config, T5ForConditionalGeneration
from looking_glass.t5_compat import validate_t5_checkpoint_contract


def test_untied_local_checkpoint_roundtrip_preserves_logits_and_shared_inputs(tmp_path):
    config = T5Config(vocab_size=40, d_model=16, d_ff=32, d_kv=8, num_layers=1,
                      num_decoder_layers=1, num_heads=2, tie_word_embeddings=False,
                      decoder_start_token_id=0, dropout_rate=0)
    config.tie_word_embeddings = False
    config._attn_implementation = "eager"
    model = T5ForConditionalGeneration(config).eval()
    model.set_input_embeddings(model.shared)
    config.to_json_file(tmp_path / "config.json")
    excluded = {"encoder.embed_tokens.weight", "decoder.embed_tokens.weight"}
    save_file({k: v.clone() for k, v in model.state_dict().items() if k not in excluded},
              str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    loaded, info = T5ForConditionalGeneration.from_pretrained(
        tmp_path, local_files_only=True, use_safetensors=True, trust_remote_code=False,
        output_loading_info=True, attn_implementation="eager")
    assert not any(info.values())
    validate_t5_checkpoint_contract(loaded, config.to_dict())
    assert loaded.config.tie_word_embeddings is False
    assert loaded.encoder.embed_tokens.weight is loaded.shared.weight
    assert loaded.decoder.embed_tokens.weight is loaded.shared.weight
    assert loaded.lm_head.weight is not loaded.shared.weight
    ids, dec = torch.tensor([[2, 3]]), torch.tensor([[0, 4]])
    with torch.inference_mode():
        assert torch.equal(model(input_ids=ids, decoder_input_ids=dec).logits,
                           loaded(input_ids=ids, decoder_input_ids=dec).logits)
    with pytest.raises(ValueError, match="differ"):
        validate_t5_checkpoint_contract(loaded, {"tie_word_embeddings": True})
    with pytest.raises(ValueError, match="booleans"):
        validate_t5_checkpoint_contract(loaded, {"tie_word_embeddings": 0})
